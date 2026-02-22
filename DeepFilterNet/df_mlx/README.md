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

#### Benchmark Dynamic Data Pipeline

Use the benchmark harness to compare loader throughput and tail latency.
All tunable benchmark flags accept comma-separated lists, so one command
can execute a full benchmark matrix across `PrefetchDataLoader` and
`MLXDataStream`.

```bash
# Cache-backed benchmark (recommended)
python -m df_mlx.benchmark_pipeline \
    --cache-dir /path/to/audio_cache \
    --split train,valid \
    --epoch 0,1 \
    --batch-size 8,12 \
    --batches 150,300 \
    --warmup-batches 5,10 \
    --repeats 2 \
    --workers 1,2,4,8 \
    --prefetch-factor 2,4 \
    --prefetch-size 8,16 \
    --backends prefetch,mlx_stream \
    --sync-arrays true,false \
    --json-out ./benchmarks/df_mlx_pipeline.json

# Raw file-list benchmark (without cache)
python -m df_mlx.benchmark_pipeline \
    --speech-list ./file_lists/speech.txt \
    --noise-list ./file_lists/noise.txt \
    --rir-list ./file_lists/rir.txt \
    --batch-size 8,12 \
    --batches 100 \
    --workers 1,2,4,8 \
    --sample-rate 48000 \
    --segment-length 4.0,5.0 \
    --fft-size 960,1024 \
    --hop-size 480,512 \
    --nb-erb 32 \
    --nb-df 96 \
    --seed 42,1337
```

The script reports mean/p50/p95/p99 batch fetch latency, throughput
(`batches/s`, `samples/s`), and measured batch/sample counts.

#### Run-config (CLI/runtime settings)

`--config` remains the dataset/mixer JSON. Use `--run-config` for all CLI/runtime
settings (TOML). Precedence: defaults < run-config < explicit CLI flags.

```bash
# Generate a fully-commented example config
python -m df_mlx.train_dynamic --print-run-config > run.toml

# Use both configs together
python -m df_mlx.train_dynamic \
    --run-config run.toml \
    --config ./file_lists/config.json
```

Single-file mode (no separate `--train-config` INI): inline train.py-compatible
INI sections inside the run-config TOML under `train_ini.*`.

```toml
[train_ini.df]
sr = 48000
fft_size = 960
hop_size = 480
nb_erb = 32
nb_df = 96

[train_ini.train]
max_epochs = 100
batch_size = 12
num_workers = 6
num_prefetch_batches = 18

[train_ini.deepfilternet4]
backbone = "attention"
model_variant = "full"
conv_ch = 64
conv_kernel = [1, 3]
conv_stride = [1, 2]

[train_ini.MultiResSpecLoss]
factor = 0.6
gamma = 0.5
factor_complex = 0.25
fft_sizes = [512, 1024, 2048]
hop_sizes = [128, 256, 512]
```

#### Train-config (train.py-compatible INI)

`train_dynamic` also accepts a train.py-style `config.ini` and maps supported
sections to MLX training + model settings. Precedence: defaults < train-config
< run-config < explicit CLI flags.

Example template:
`DeepFilterNet/df_mlx/configs/train.ini`

```bash
python -m df_mlx.train_dynamic \
    --config ./file_lists/config.json \
    --train-config /path/to/config.ini
```

Example using the repo template:

```bash
python -m df_mlx.train_dynamic \
    --config /path/to/dataset/config.json \
    --train-config DeepFilterNet/df_mlx/configs/train.ini \
    --checkpoint-dir /path/to/checkpoints \
    --dynamic-loss pipeline_awesome \
    --epochs 100
```

Supported sections:
- `[df]`, `[train]`, `[optim]`, `[distortion]`, `[deepfilternet4]`
- `[loss]` (multi_res_stft_* keys)
- `[MultiResSpecLoss]`
- `[GANLoss]`, `[FeatureMatchingLoss]` (adversarial training)

Unsupported sections (e.g. `[ASRLoss]`, `[MaskLoss]`, `[SpectralLoss]`) are ignored with warnings.
Use `df/train.py` for ASR loss; GAN training is supported directly in `train_dynamic`.

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

# Silero VAD eval (requires optional deps; no network at runtime)
# pip install silero-vad onnxruntime torch
--vad-eval-mode silero --vad-eval-batches 8 --vad-eval-max-seconds 10 \
    --vad-silero-sample-rate 16000
# Optionally point to a custom ONNX model file:
--vad-silero-model-path /path/to/silero_vad.onnx

# Sparse training-time VAD regularizer (disabled by default)
--vad-train-prob 0.01  # or --vad-train-every-steps 500

# Disable proxy gating if needed
--no-vad-proxy
```

#### Multi-res STFT loss (speech clarity)

Enable the time-domain multi-resolution STFT loss (analogous to PyTorch
`MultiResSpecLoss`) to improve speech detail:

```bash
python -m df_mlx.train_dynamic \
    --config ./file_lists/config.json \
    --train-config /path/to/config.ini \
    --mrstft-factor 1.0 \
    --mrstft-gamma 1.0 \
    --mrstft-f-complex 0.5 \
    --mrstft-fft-sizes 512 1024 2048
```

Note: When FP16 mixed precision is enabled, the MRSTFT path is computed in
FP32 internally to avoid overflow in magnitude squaring and power compression.

#### GAN adversarial loss (optional)

Enable GAN-based perceptual cleanup using a discriminator + feature matching.
When GAN is enabled, the compiled training step is disabled automatically.
Set `[GANLoss] factor` (and optionally `[FeatureMatchingLoss] factor`) to
non-zero values in your train.ini.

```bash
python -m df_mlx.train_dynamic \
    --config ./file_lists/config.json \
    --train-config /path/to/config.ini \
    --gan-enabled \
    --gan-start-epoch 0 \
    --gan-ramp-epochs 5 \
    --gan-adv-weight 0.1 \
    --gan-fm-weight 2.0 \
    --gan-discriminator combined
```

GAN checkpoints include discriminator weights alongside generator checkpoints
using the same stem: `epoch_060.safetensors` + `epoch_060.disc.safetensors`.

#### Numeric debug mode (NaN/inf diagnosis)

Use the built-in numeric debugger to find the first non-finite tensor and
dump a compact snapshot for analysis. Debug mode runs a short, deterministic
job and disables the compiled training step for better visibility.

```bash
python -m df_mlx.train_dynamic \
    --config ./file_lists/config.json \
    --dynamic-loss awesome \
    --debug-numerics \
    --debug-numerics-dump-arrays \
    --no-fp16
```

Optional controls:
- `--debug-numerics-every 1` (check every step)
- `--nan-skip-batch` (skip optimizer update on non-finite)
- `--seed 123` (deterministic sampling)

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
