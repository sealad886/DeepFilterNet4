# Migration Guide: DeepFilterNet3 → DeepFilterNet4

This guide helps you migrate from DeepFilterNet3 to DeepFilterNet4.

## Table of Contents

- [Overview](#overview)
- [Key Differences](#key-differences)
- [Configuration Migration](#configuration-migration)
- [Model Loading](#model-loading)
- [Training Migration](#training-migration)
- [Inference Migration](#inference-migration)
- [Common Issues](#common-issues)
- [API Reference Changes](#api-reference-changes)

---

## Overview

DeepFilterNet4 introduces several architectural improvements while maintaining API compatibility where possible. The main changes are:

1. **Mamba backbone** replaces GRU for temporal modeling
2. **Python dataclass configuration** replaces INI files
3. **New optional features**: hybrid encoder, multi-resolution DF, adaptive order
4. **Enhanced training**: GAN, QAT, knowledge distillation

### Compatibility Matrix

| Feature | DFNet3 | DFNet4 | Notes |
|---------|--------|--------|-------|
| Python | 3.8+ | 3.9+ | Minimum version increased |
| PyTorch | 1.9+ | 2.0+ | Required for Mamba SSM |
| Sample rate | 48kHz | 48kHz | Same |
| FFT size | 960 | 960 | Same |
| Hop size | 480 | 480 | Same |
| Checkpoint format | state_dict | state_dict | Compatible |
| ONNX export | ✓ | ✓ | New export script |

---

## Key Differences

### Architecture

| Component | DFNet3 | DFNet4 |
|-----------|--------|--------|
| Encoder | GRU-based | Mamba SSM |
| Temporal modeling | Bidirectional GRU | Selective State Space |
| Complexity | O(n²) attention-like | O(n) linear |
| Feature fusion | Single domain | Multi-domain (optional) |
| Deep filtering | Fixed order | Adaptive order (optional) |
| DF resolution | Single | Multi-resolution (optional) |

### Parameters

| Model | Parameters | RTF (CPU) |
|-------|-----------|-----------|
| DFNet3 | ~1.6M | 0.30 |
| DFNet4 | ~2.6M | 0.40 |
| DFNet4Lite | ~1.3M | 0.22 |

---

## Configuration Migration

### DFNet3 Configuration (INI)

```ini
[deepfilternet]
fft_size = 960
hop_size = 480
nb_erb = 32
nb_df = 96
df_order = 5
df_lookahead = 2
conv_lookahead = 2
conv_ch = 16
conv_kernel = [1, 3]
df_hidden_dim = 256
nb_layers = 3
```

### DFNet4 Configuration (Python dataclass)

```python
from df.deepfilternet4 import DfNet4Config

config = DfNet4Config(
    # Same audio parameters
    fft_size=960,
    hop_size=480,
    nb_erb=32,
    nb_df=96,
    df_order=5,
    df_lookahead=2,
    
    # New Mamba parameters (replaces GRU)
    mamba_d_model=256,      # Similar to df_hidden_dim
    mamba_d_state=64,
    mamba_d_conv=4,
    mamba_expand=2,
    num_mamba_layers=4,     # Similar to nb_layers
    
    # Optional new features
    hybrid_encoder=False,
    multi_res_df=False,
    adaptive_df_order=False,
)
```

### Automatic Migration Helper

```python
from df.deepfilternet4 import DfNet4Config

def migrate_config_from_dfnet3(ini_path: str) -> DfNet4Config:
    """Convert DFNet3 INI config to DFNet4 dataclass."""
    import configparser
    
    parser = configparser.ConfigParser()
    parser.read(ini_path)
    
    dfnet3 = parser["deepfilternet"]
    
    return DfNet4Config(
        fft_size=int(dfnet3.get("fft_size", 960)),
        hop_size=int(dfnet3.get("hop_size", 480)),
        nb_erb=int(dfnet3.get("nb_erb", 32)),
        nb_df=int(dfnet3.get("nb_df", 96)),
        df_order=int(dfnet3.get("df_order", 5)),
        df_lookahead=int(dfnet3.get("df_lookahead", 2)),
        mamba_d_model=int(dfnet3.get("df_hidden_dim", 256)),
        num_mamba_layers=int(dfnet3.get("nb_layers", 3)) + 1,
    )
```

---

## Model Loading

### DFNet3 Loading

```python
from df.deepfilternet3 import ModelParams, init_model

p = ModelParams()
model, df_state, _ = init_model(p)
model.load_state_dict(torch.load("dfnet3_checkpoint.pt"))
```

### DFNet4 Loading

```python
from df.deepfilternet4 import DfNet4, DfNet4Config

config = DfNet4Config()
model = DfNet4(config)

# Load DFNet4 checkpoint
checkpoint = torch.load("dfnet4_checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
```

### Loading DFNet3 Weights into DFNet4

DFNet3 and DFNet4 have different architectures, so weights cannot be directly transferred. You must:

1. Train DFNet4 from scratch, OR
2. Use knowledge distillation with DFNet3 as teacher

```python
# Knowledge distillation approach
from df.deepfilternet3 import init_model as init_dfnet3
from df.deepfilternet4 import DfNet4, DfNet4Config, DistillationLoss

# Load teacher (DFNet3)
teacher, _, _ = init_dfnet3(ModelParams())
teacher.load_state_dict(torch.load("dfnet3_checkpoint.pt"))
teacher.eval()

# Create student (DFNet4Lite for efficiency)
config = DfNet4Config()
student = DfNet4(config)

# Train with distillation
loss_fn = DistillationLoss(teacher, alpha_kd=0.7, temperature=4.0)
```

---

## Training Migration

### DFNet3 Training

```bash
python df/train.py \
    --config deepfilternet3 \
    path/to/dataset.cfg \
    path/to/data/ \
    path/to/output/
```

### DFNet4 Training

```bash
python df/train.py \
    --model-type dfnet4 \
    path/to/dataset.cfg \
    path/to/data/ \
    path/to/output/
```

### Training Script Changes

```python
# DFNet3
from df.deepfilternet3 import ModelParams, DfNet
from df.loss import Loss

p = ModelParams()
model = DfNet(p)
loss_fn = Loss(p)

# DFNet4
from df.deepfilternet4 import DfNet4Config, DfNet4
from df.deepfilternet4 import SpectralLoss, MaskLoss

config = DfNet4Config()
model = DfNet4(config)

# New modular loss functions
spec_loss = SpectralLoss(factor_magnitude=1.0, factor_complex=0.1)
mask_loss = MaskLoss(factor_erb=0.1, factor_df=0.1)

def combined_loss(pred, target, masks):
    return spec_loss(pred, target) + mask_loss(masks, target_masks)
```

### GAN Training (New in DFNet4)

```python
from df.deepfilternet4 import DfNet4, DfNet4Discriminator

# Generator
model = DfNet4(config)

# Discriminator (new)
discriminator = DfNet4Discriminator(
    in_channels=1,
    hidden_channels=[64, 128, 256, 512],
    num_scales=3,
)

# Training loop
for batch in dataloader:
    # Generator step
    enhanced = model(noisy)
    g_loss = adversarial_loss(discriminator(enhanced))
    
    # Discriminator step
    d_real = discriminator(clean)
    d_fake = discriminator(enhanced.detach())
    d_loss = discriminator_loss(d_real, d_fake)
```

---

## Inference Migration

### DFNet3 Inference

```python
from df import enhance, init_df

model, df_state, _ = init_df(model_base_dir="DeepFilterNet3")
enhanced = enhance(model, df_state, noisy_audio)
```

### DFNet4 Inference

```python
from df import enhance, init_df

# Same API, just specify DFNet4 model directory
model, df_state, _ = init_df(model_base_dir="DeepFilterNet4")
enhanced = enhance(model, df_state, noisy_audio)
```

### Direct Model Inference

```python
# DFNet3
from df.deepfilternet3 import forward

spec_out, gains, df_coefs = forward(model, df_state, spec, erb)

# DFNet4
from df.deepfilternet4 import DfNet4

model = DfNet4(config)
erb_out, df_out = model(erb_features, spec_features)
# Apply gains and DF filtering separately
```

---

## Common Issues

### Issue 1: PyTorch Version Error

```
ImportError: cannot import name 'MambaBlock' from 'mamba_ssm'
```

**Solution**: DFNet4 includes a pure-PyTorch Mamba implementation. No external `mamba_ssm` package needed.

### Issue 2: Checkpoint Loading Fails

```
RuntimeError: Error(s) in loading state_dict for DfNet4
```

**Cause**: DFNet3 checkpoints are incompatible with DFNet4.

**Solution**: 
- Train DFNet4 from scratch
- Use knowledge distillation
- Don't attempt to load DFNet3 weights into DFNet4

### Issue 3: Memory Usage Increased

**Cause**: DFNet4 has more parameters by default.

**Solutions**:
1. Use `DfNet4Lite` for memory-constrained environments
2. Enable mixed precision: `config = DfNet4Config(use_fp16=True)`
3. Reduce batch size

### Issue 4: Configuration Key Errors

```
KeyError: 'conv_ch' not found in DfNet4Config
```

**Cause**: DFNet4 uses different configuration parameters.

**Solution**: Use the [migration helper](#automatic-migration-helper) or manually map parameters.

### Issue 5: Training Slower Than Expected

**Cause**: Mamba requires specific optimizations.

**Solutions**:
1. Use CUDA if available (CPU training is slower)
2. Enable mixed precision training
3. Adjust batch size for your GPU memory

---

## API Reference Changes

### Removed APIs

| DFNet3 | DFNet4 Replacement |
|--------|-------------------|
| `ModelParams` | `DfNet4Config` |
| `init_model()` | `DfNet4(config)` |
| `p.conv_ch` | Removed (internal) |
| `p.conv_kernel` | Removed (internal) |

### New APIs

| API | Description |
|-----|-------------|
| `DfNet4Config` | Dataclass configuration |
| `DfNet4` | Main model class |
| `DfNet4Lite` | Lightweight variant |
| `DfNet4Discriminator` | GAN discriminator |
| `MambaBlock` | Mamba SSM block |
| `HybridEncoder` | Multi-domain encoder |
| `prepare_qat()` | QAT preparation |
| `DistillationLoss` | Knowledge distillation |
| `SpectralLoss` | Modular spectral loss |
| `MaskLoss` | Mask prediction loss |
| `LSNRLoss` | Local SNR loss |

### Changed APIs

| API | DFNet3 | DFNet4 |
|-----|--------|--------|
| Model output | `(spec, gains, df_coefs)` | `(erb_out, df_out)` |
| Config format | INI file | Python dataclass |
| Hidden size | `df_hidden_dim` | `mamba_d_model` |
| Num layers | `nb_layers` | `num_mamba_layers` |

---

## Migration Checklist

- [ ] Update Python to 3.9+
- [ ] Update PyTorch to 2.0+
- [ ] Convert INI config to `DfNet4Config` dataclass
- [ ] Update model instantiation code
- [ ] Update loss function imports
- [ ] Train new DFNet4 model (or use distillation)
- [ ] Update inference code for new output format
- [ ] Test enhanced audio quality
- [ ] Update deployment scripts for ONNX export
- [ ] Benchmark RTF on target hardware

---

## Getting Help

If you encounter issues not covered here:

1. Check the [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component documentation
2. Review the test files in `DeepFilterNet/tests/test_deepfilternet4.py`
3. Open an issue on GitHub with:
   - Python/PyTorch versions
   - Error message and traceback
   - Minimal reproduction code

---

## Whisper Backend Abstraction

### Overview

The whisper integration for ASR-based loss has been refactored to support multiple backends:
- **PyTorch backend** (openai-whisper): Works on all platforms (CUDA, CPU, MPS)
- **MLX backend** (mlx-whisper): Optimized for Apple Silicon (M1/M2/M3/M4)

This provides **5-10x speedup** on Apple Silicon for ASR-based loss computation.

### Breaking Changes

**None.** The `ASRLoss` class maintains full backward compatibility. Existing code will continue to work without modification.

### New Features

- `ASRLoss` now accepts a `backend` parameter: `"auto"`, `"pytorch"`, or `"mlx"`
- New module `df.whisper_adapter` provides direct backend access
- Automatic platform detection selects optimal backend
- Performance improvement on Apple Silicon when using MLX backend

### Migration Guide

**No changes required** for existing code. The adapter automatically selects the best backend.

To explicitly use a backend:

```python
# Before (still works)
from df.loss import ASRLoss
loss_fn = ASRLoss(model="base")

# After (optional explicit backend)
loss_fn = ASRLoss(model="base", backend="mlx")  # Apple Silicon
loss_fn = ASRLoss(model="base", backend="pytorch")  # CUDA/CPU
loss_fn = ASRLoss(model="base", backend="auto")  # Auto-detect (default)
```

### Direct Backend Access

For advanced use cases, you can access the backend directly:

```python
from df.whisper_adapter import get_whisper_backend, is_apple_silicon

# Check platform
if is_apple_silicon():
    print("Running on Apple Silicon")

# Get backend with auto-detection
backend = get_whisper_backend("base")
print(f"Using {backend.backend_name} backend")

# Extract embeddings
mel = backend.log_mel_spectrogram(audio)
features = backend.embed_audio(mel)

# Transcribe
result = backend.decode(mel)
print(result.text)
```

### Dependencies

| Backend | Package | Platform |
|---------|---------|----------|
| PyTorch | `openai-whisper>=20240930` | All |
| MLX | `mlx>=0.0.6`, `mlx-whisper>=0.4.0` | Apple Silicon only |

Install with:

```bash
# PyTorch only (all platforms)
pip install deepfilternet[asr]

# MLX support (Apple Silicon)
pip install deepfilternet[asr-mlx]

# Or install MLX manually
pip install mlx mlx-whisper
```
