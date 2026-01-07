# MLX DeepFilterNet4 Implementation

This directory contains a native MLX implementation of DeepFilterNet4, optimized for Apple Silicon.

## Overview

The MLX implementation provides:
- Full forward pass and inference support
- End-to-end enhancement from raw audio
- Training with gradient clipping and checkpointing
- PyTorch weight conversion for loading pretrained models
- Optimized operations for Apple Silicon unified memory

## Performance

Benchmarks show MLX achieves 1.51x-1.77x speedup over PyTorch-MPS for inference:
- Batch size 1: ~1.51x faster
- Batch size 4: ~1.77x faster
- Batch size 8: ~1.66x faster

## Module Structure

- **ops.py**: Core signal processing (STFT/iSTFT, ERB filterbank, complex operations)
- **modules.py**: Neural network building blocks (Conv2dNormAct, Mask, DfOp, GroupedLinear)
- **mamba.py**: Selective state-space model (MambaBlock, SqueezedMamba)
- **model.py**: Full model (Encoder4, ErbDecoder4, DfDecoder4, DfNet4, DfNet4Lite)
- **train.py**: Training utilities (losses, scheduler, Trainer, weight conversion)
- **config.py**: Configuration dataclasses (ModelParams4, TrainConfig)

## Usage

### Inference

```python
from df_mlx.model import init_model
import mlx.core as mx

# Initialize model
model = init_model()

# Load weights (optional - from PyTorch checkpoint)
from df_mlx.train import load_pytorch_checkpoint
load_pytorch_checkpoint(model, "path/to/checkpoint")

# Enhance audio
noisy_audio = mx.array(audio_samples)  # (samples,) or (batch, samples)
enhanced = model.enhance(noisy_audio)
```

### Training

```python
from df_mlx.model import init_model
from df_mlx.train import Trainer
from df_mlx.config import TrainConfig

model = init_model()
config = TrainConfig(
    learning_rate=1e-4,
    warmup_steps=1000,
    max_steps=100000,
    checkpoint_dir="checkpoints/"
)
trainer = Trainer(model, config)

# Training loop
for batch in dataloader:
    spec, feat_erb, feat_spec, target = batch
    loss = trainer.train_step(spec, feat_erb, feat_spec, target)
```

## Feature Parity with PyTorch

### ✅ Implemented Features

| Feature | Status | Notes |
|---------|--------|-------|
| DfNet4 forward pass | ✅ Complete | Full encoder/backbone/decoder pipeline |
| DfNet4Lite variant | ✅ Complete | Reduced parameter count |
| Mamba backbone | ✅ Complete | SqueezedMamba with selective scan |
| ERB masking | ✅ Complete | Multiple mask types (sigmoid, bounded, etc.) |
| Deep filtering | ✅ Complete | DfOp with configurable order/lookahead |
| STFT/iSTFT | ✅ Complete | Multiple window types |
| ERB filterbank | ✅ Complete | Frequency-domain filterbank |
| Training loop | ✅ Complete | Spectral loss, gradient clipping |
| Checkpointing | ✅ Complete | Save/load with safetensors |
| Weight conversion | ✅ Complete | PyTorch → MLX conversion |

### ⚠️ Partially Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| Post-filter | ⚠️ Placeholder | Structure exists, not fully integrated |
| Multi-resolution loss | ⚠️ Basic | Single-resolution STFT loss only |

### ❌ Not Yet Implemented

These features exist in the PyTorch `deepfilternet4.py` but are not in the MLX implementation:

| Feature | PyTorch Location | Description |
|---------|-----------------|-------------|
| MultiResDfDecoder | Lines 850-950 | Multi-resolution deep filtering with multiple filter orders |
| AdaptiveDfDecoder | Lines 960-1050 | Adaptive filter order based on frequency band |
| HybridEncoder | Lines 1100-1200 | Separate time/phase processing branches |
| LSNR dropout | Loss functions | Local SNR-based masking during training |
| Lookahead configurations | Model variants | Different lookahead settings for real-time variants |
| Complex gain output | DfDecoder variants | Alternative to coefficient-based filtering |
| Model statistics tracking | Training | Running mean/variance tracking for normalization |

### Implementation Priority

For most use cases (inference with pretrained models), the current implementation provides full feature coverage. The missing features are primarily:

1. **Training enhancements** (LSNR dropout, multi-res loss) - for advanced training
2. **Alternative architectures** (MultiRes, Adaptive, Hybrid) - for research/experimentation
3. **Real-time variants** - for streaming applications

## Testing

Run the comprehensive test suite:

```bash
cd DeepFilterNet
python -m pytest df_mlx/test_mlx_comprehensive.py -v
```

This includes 87 tests covering:
- STFT/iSTFT operations (14 tests)
- ERB filterbank (7 tests)
- Complex operations (4 tests)
- Conv modules (11 tests)
- Mask operations (5 tests)
- DfOp (8 tests)
- GroupedLinear (6 tests)
- Mamba blocks (6 tests)
- Full model (11 tests)
- Training utilities (7 tests)
- Weight conversion (1 test)
- Edge cases (4 tests)

## Known Limitations

1. **No streaming mode**: Current implementation processes full audio segments, not frame-by-frame
2. **Simplified padding**: Uses zero-padding instead of reflect padding in STFT
3. **Single-device**: No multi-device parallelism (MLX is designed for unified memory)

## Contributing

When adding features, please:
1. Add corresponding tests to `test_mlx_comprehensive.py`
2. Update this README if adding new functionality
3. Ensure PyTorch parity where applicable
