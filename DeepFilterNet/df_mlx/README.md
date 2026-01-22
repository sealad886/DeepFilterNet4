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

There are two training approaches:

#### Option 1: Pre-computed Datastore (Faster startup, limited diversity)

Pre-compute spectral features once, then train:

```bash
# Build datastore (one-time)
./scripts/datasets/build_mlx_datastore.sh

# Train
python -m df_mlx.train_with_data \
    --datastore ./mlx_datastore \
    --epochs 100 \
    --batch-size 8
```

#### Option 2: Dynamic On-the-Fly Mixing (Full diversity, matches original training)

This approach mirrors the original Rust DataLoader:
- Dynamic speech + noise + RIR mixing each epoch
- Same speech file sees different noise/SNR/RIR each epoch
- Full dataset diversity (all files available)
- Configurable augmentations (reverb, clipping, EQ)

```bash
# Generate file lists (from directories)
python -m df_mlx.generate_file_lists \
    --speech-dirs /path/to/speech \
    --noise-dirs /path/to/noise \
    --rir-dirs /path/to/rirs \
    --output-dir ./file_lists \
    --generate-config

# Train with dynamic mixing
python -m df_mlx.train_dynamic \
    --config ./file_lists/config.json \
    --epochs 100 \
    --batch-size 8 \
    --p-reverb 0.5
```

#### Awesome dynamic loss (speech-preserving)

Enable the speech-preserving contrastive loss and cheap VAD proxy gating:

```bash
python -m df_mlx.train_dynamic \
    --config ./file_lists/config.json \
    --epochs 100 \
    --batch-size 8 \
    --dynamic-loss awesome \
    --awesome-loss-weight 0.4 \
    --awesome-mask-sharpness 6.0 \
    --awesome-warmup-steps 2000
```

Optional VAD controls (all optional; defaults are safe):

```bash
# Periodic VAD eval metrics (proxy-based, lightweight)
--vad-eval-mode auto --vad-eval-every 1 --vad-eval-batches 8

# Sparse training-time VAD regularizer (disabled by default)
--vad-train-prob 0.01  # or --vad-train-every-steps 500

# Disable proxy gating if needed
--no-vad-proxy
```

Or specify file lists directly:

```bash
python -m df_mlx.train_dynamic \
    --speech-list speech_files.txt \
    --noise-list noise_files.txt \
    --rir-list rir_files.txt \
    --epochs 100 \
    --batch-size 8
```

The dynamic approach provides better model generalization due to the vastly
larger effective training set (each epoch sees different combinations).

### High-Throughput Data Loading with mlx-data

The `train_dynamic` script supports Apple's `mlx-data` library for optimized
data loading with parallel prefetching and checkpoint/resume capability:

```bash
# Install mlx-data (Apple Silicon only)
pip install mlx-data

# Train with mlx-data for 4.5x faster data loading
python -m df_mlx.train_dynamic \
    --config ./file_lists/config.json \
    --use-mlx-data \
    --prefetch-size 4 \
    --num-workers 4 \
    --checkpoint-batches 100  # Save every 100 batches for resume
```

Key features:
- **4.5x throughput improvement** over sequential loading (416 vs 93 samples/s)
- **Checkpoint/resume**: Saves progress (epoch, batch, samples) for interruption recovery
- **Auto-resume**: Automatically resumes from last checkpoint on restart
- **Parallel prefetching**: Multi-threaded sample loading with configurable depth

Resume from interruption:
```bash
# Resume from specific checkpoint
python -m df_mlx.train_dynamic \
    --config ./file_lists/config.json \
    --resume-data-from checkpoints/data_checkpoint.json

# Or auto-resume (default if checkpoint exists)
python -m df_mlx.train_dynamic \
    --config ./file_lists/config.json
```

### Basic Training API

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
| LSNR estimation | ✅ Complete | Encoder outputs per-frame LSNR |
| LSNR dropout | ✅ Complete | Training-mode dropout based on LSNR threshold |
| MultiResDfDecoder | ✅ Complete | Multi-resolution DF with shared Mamba backbone |
| AdaptiveOrderPredictor | ✅ Complete | Predicts optimal filter order per frame |
| LSNR loss | ✅ Complete | L1 loss for LSNR prediction |
| WaveformEncoder | ✅ Complete | Time-domain waveform encoder with strided conv layers |
| PhaseEncoder | ✅ Complete | Phase spectrum encoder using cos/sin representation |
| CrossDomainAttention | ✅ Complete | Multi-head cross-attention for time-mag and mag-phase fusion |
| HybridEncoder | ✅ Complete | Full multi-domain encoder with Mamba backbone |
| MLXDataStream | ✅ Complete | High-throughput data loading with mlx-data (4.5x speedup) |
| Checkpoint/Resume | ✅ Complete | Save/load data progress for interruption recovery |

### ⚠️ Partially Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| Post-filter | ⚠️ Placeholder | Structure exists, not fully integrated |
| Multi-resolution loss | ⚠️ Basic | Single-resolution STFT loss only |

### ❌ Not Yet Implemented

These features exist in the PyTorch `deepfilternet4.py` but are not in the MLX implementation:

| Feature | PyTorch Location | Description |
|---------|-----------------|-------------|
| Lookahead configurations | Model variants | Different lookahead settings for real-time variants |
| Complex gain output | DfDecoder variants | Alternative to coefficient-based filtering |
| Model statistics tracking | Training | Running mean/variance tracking for normalization |

### Implementation Priority

For most use cases (inference with pretrained models), the current implementation provides full feature coverage. The missing features are primarily:

1. **Real-time variants** - for streaming applications with specific lookahead requirements
2. **Alternative outputs** (Complex gain) - for different filtering strategies

## Testing

Run the comprehensive test suite:

```bash
cd DeepFilterNet
python -m pytest df_mlx/test_mlx_comprehensive.py -v
```

This includes 144 tests covering:
- STFT/iSTFT operations (15 tests)
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
- Numerical properties (6 tests)
- LSNR features (7 tests)
- Multi-resolution decoder (7 tests)
- Adaptive order predictor (12 tests)
- LSNR config (3 tests)
- WaveformEncoder (6 tests)
- PhaseEncoder (6 tests)
- CrossDomainAttention (3 tests)
- HybridEncoder (7 tests)

## Known Limitations

1. **No streaming mode**: Current implementation processes full audio segments, not frame-by-frame
2. **Simplified padding**: Uses zero-padding instead of reflect padding in STFT
3. **Single-device**: No multi-device parallelism (MLX is designed for unified memory)

## Contributing

When adding features, please:
1. Add corresponding tests to `test_mlx_comprehensive.py`
2. Update this README if adding new functionality
3. Ensure PyTorch parity where applicable
