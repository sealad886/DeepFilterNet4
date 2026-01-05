# DeepFilterNet Architecture Guide

This document provides a comprehensive overview of the DeepFilterNet architecture, focusing on DeepFilterNet4 (DFNet4) while covering the evolution from earlier versions.

## Table of Contents

- [Overview](#overview)
- [Architecture Evolution](#architecture-evolution)
- [DeepFilterNet4 Architecture](#deepfilternet4-architecture)
  - [Mamba Backbone](#mamba-backbone)
  - [Encoder Architecture](#encoder-architecture)
  - [Decoder Architecture](#decoder-architecture)
  - [Deep Filtering](#deep-filtering)
- [Feature Extraction](#feature-extraction)
- [Training Components](#training-components)
- [Configuration Reference](#configuration-reference)

---

## Overview

DeepFilterNet is a low-complexity speech enhancement framework designed for real-time full-band audio processing (48kHz). The core innovation is **Deep Filtering** - a learnable short-time convolution that directly estimates complex-valued filter coefficients for noise suppression.

### Key Design Principles

1. **Efficiency First**: Real-time processing on embedded devices
2. **Full-Band Audio**: Native 48kHz processing without downsampling
3. **Perceptual Quality**: Preserve speech naturalness and intelligibility
4. **Flexible Deployment**: CPU, GPU, mobile, and edge platforms

---

## Architecture Evolution

### DeepFilterNet (v1)

```
Input → STFT → ERB Analysis → GRU Encoder → ERB Decoder + DF Decoder → iSTFT → Output
```

- First to introduce Deep Filtering for speech enhancement
- GRU-based temporal modeling
- ERB-scale spectral representation

### DeepFilterNet2

- Optimized for embedded deployment
- Reduced model size and complexity
- Improved real-time factor (RTF)

### DeepFilterNet3

- Perceptual loss functions
- Improved GRU encoder
- Better noise suppression for non-stationary noise

### DeepFilterNet4 (Current)

```
Input → STFT → [Hybrid Encoder] → Mamba Backbone → [Multi-Res DF] → iSTFT → Output
```

Major innovations:
- **Mamba SSM**: State-space model for O(n) temporal modeling
- **Hybrid Encoder**: Multi-domain feature fusion
- **Multi-Resolution DF**: Frequency-dependent filter resolution
- **Adaptive Filter Order**: Signal-complexity-based order selection
- **GAN Training**: Adversarial training for perceptual quality
- **Quantization-Aware**: INT8 deployment support

---

## DeepFilterNet4 Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DeepFilterNet4                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Audio Input (48kHz)                                               │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────┐                                                       │
│   │  STFT   │  FFT size: 960, Hop: 480 (10ms frames)               │
│   └────┬────┘                                                       │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────────────────────────────────────────────┐          │
│   │              Feature Extraction                      │          │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │          │
│   │  │   ERB-32    │  │   DF-96     │  │   Phase     │  │          │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  │          │
│   └────────────────────────┬────────────────────────────┘          │
│                            │                                        │
│                            ▼                                        │
│   ┌─────────────────────────────────────────────────────┐          │
│   │              Hybrid Encoder (Optional)               │          │
│   │  ┌───────────────┐  ┌───────────────┐               │          │
│   │  │ Time-Domain   │  │    Phase      │               │          │
│   │  │   Branch      │  │   Branch      │               │          │
│   │  └───────┬───────┘  └───────┬───────┘               │          │
│   │          └──────────┬───────┘                       │          │
│   │                     ▼                               │          │
│   │              Feature Fusion                         │          │
│   └────────────────────────┬────────────────────────────┘          │
│                            │                                        │
│                            ▼                                        │
│   ┌─────────────────────────────────────────────────────┐          │
│   │               Mamba Backbone                         │          │
│   │  ┌──────────────────────────────────────────────┐   │          │
│   │  │  Mamba Block 1                                │   │          │
│   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐       │   │          │
│   │  │  │  In Proj │→│   SSM   │→│ Out Proj │       │   │          │
│   │  │  └─────────┘  └─────────┘  └─────────┘       │   │          │
│   │  └──────────────────────────────────────────────┘   │          │
│   │                         │                            │          │
│   │                         ▼                            │          │
│   │  ┌──────────────────────────────────────────────┐   │          │
│   │  │  Mamba Block 2 ... N                          │   │          │
│   │  └──────────────────────────────────────────────┘   │          │
│   └────────────────────────┬────────────────────────────┘          │
│                            │                                        │
│              ┌─────────────┴─────────────┐                         │
│              ▼                           ▼                         │
│   ┌──────────────────┐        ┌──────────────────┐                 │
│   │   ERB Decoder    │        │    DF Decoder    │                 │
│   │  (Gain Masks)    │        │(Filter Coeffs)   │                 │
│   └────────┬─────────┘        └────────┬─────────┘                 │
│            │                           │                           │
│            ▼                           ▼                           │
│   ┌──────────────────────────────────────────────────┐             │
│   │              Apply Deep Filtering                 │             │
│   │   enhanced = gains ⊙ input + DF(input, coeffs)   │             │
│   └────────────────────────┬─────────────────────────┘             │
│                            │                                        │
│                            ▼                                        │
│                    ┌─────────────┐                                  │
│                    │   iSTFT     │                                  │
│                    └─────────────┘                                  │
│                            │                                        │
│                            ▼                                        │
│                    Enhanced Audio                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Mamba Backbone

The Mamba backbone replaces traditional RNN/LSTM/GRU with a state-space model (SSM) that provides:

- **Linear complexity**: O(n) vs O(n²) for attention
- **Long-range modeling**: Theoretically infinite context
- **Parallelizable training**: Hardware-efficient implementation

#### State-Space Model Equation

```
h'(t) = A h(t) + B x(t)
y(t)  = C h(t) + D x(t)
```

Where:
- `h(t)`: Hidden state (dimension: `d_state`)
- `A, B, C, D`: Learned parameters
- `x(t)`: Input at time t
- `y(t)`: Output at time t

#### Mamba Block Structure

```python
class MambaBlock:
    def forward(self, x):
        # Input projection: d_model → d_inner
        x_proj = self.in_proj(x)
        
        # Split into main and gate paths
        x_main, x_gate = x_proj.chunk(2, dim=-1)
        
        # Causal convolution for local context
        x_conv = self.conv1d(x_main)
        
        # SSM computation
        x_ssm = self.ssm(x_conv)
        
        # Gated output
        return self.out_proj(x_ssm * F.silu(x_gate))
```

#### Configuration

```python
DfNet4Config(
    mamba_d_model=256,     # Model dimension
    mamba_d_state=64,      # State space dimension  
    mamba_d_conv=4,        # Convolution width
    mamba_expand=2,        # Expansion factor (d_inner = expand * d_model)
    num_mamba_layers=4,    # Number of Mamba blocks
)
```

### Encoder Architecture

#### Standard Encoder

The base encoder converts ERB features to a latent representation:

```python
ERB Input (B, 1, nb_erb, T)
    │
    ▼
Conv2d + GroupNorm + ELU  (channels: 64)
    │
    ▼
Conv2d + GroupNorm + ELU  (channels: 128)
    │
    ▼  
Conv2d + GroupNorm + ELU  (channels: 256)
    │
    ▼
Linear (256 → mamba_d_model)
    │
    ▼
Encoder Output (B, T, mamba_d_model)
```

#### Hybrid Encoder (Optional)

The hybrid encoder fuses multiple domains:

```
                    ┌──────────────────────────┐
                    │      Input Features       │
                    └────────────┬─────────────┘
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
      │   ERB Branch   │  │ Time-Domain   │  │    Phase      │
      │   (Standard)   │  │    Branch     │  │    Branch     │
      └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │    Feature Fusion        │
                    │  (Concat + Linear)       │
                    └──────────────────────────┘
```

##### Time-Domain Branch

Processes raw waveform with 1D convolutions:

```python
TimeDomainEncoder(
    in_channels=1,
    hidden_channels=[64, 128, 256],
    kernel_size=7,
    stride=2,
)
```

##### Phase Branch

Extracts phase information:

```python
PhaseEncoder(
    nb_df=96,
    hidden_dim=64,
)
```

### Decoder Architecture

#### ERB Decoder (Gain Estimation)

Estimates per-band gains for coarse spectral shaping:

```python
Mamba Output (B, T, d_model)
    │
    ▼
Linear (d_model → 256)
    │
    ▼
GRU (hidden: 256) × 2 layers
    │
    ▼
Linear (256 → nb_erb)
    │
    ▼
Sigmoid → Gains (B, T, nb_erb) ∈ [0, 1]
```

#### DF Decoder (Filter Coefficient Estimation)

Estimates deep filtering coefficients for fine spectral detail:

```python
Mamba Output (B, T, d_model)
    │
    ▼
Linear (d_model → 256)
    │
    ▼
GRU (hidden: 256) × 2 layers
    │
    ▼
Linear (256 → nb_df × df_order × 2)  # Real + Imag
    │
    ▼
Reshape → Coefficients (B, T, nb_df, df_order, 2)
```

### Deep Filtering

Deep Filtering applies a learnable short-time convolution to enhance speech:

#### Standard DF

```python
def deep_filter(input_spec, coefficients, order, lookahead):
    """
    input_spec: (B, C, F, T, 2)  - Complex spectrum
    coefficients: (B, T, F, order, 2)  - Filter coeffs
    """
    output = torch.zeros_like(input_spec[..., :1, :])
    
    for k in range(order):
        t_idx = t - lookahead + k
        # Complex multiplication
        output += input_spec[..., t_idx, :] * coefficients[..., k, :]
    
    return output
```

#### Multi-Resolution DF

Different frequency bands use different filter resolutions:

```
┌─────────────────────────────────────────────────────┐
│   Frequency                                          │
│      ↑                                               │
│      │   Low bands:  order=8, resolution=1          │
│      │   Mid bands:  order=5, resolution=2          │
│      │   High bands: order=3, resolution=4          │
│      └───────────────────────────────────────→ Time │
└─────────────────────────────────────────────────────┘
```

Configuration:
```python
DfNet4Config(
    multi_res_df=True,
    df_resolutions=[2, 4],      # Mid and high band resolutions
    df_order_per_band=[8, 5, 3], # Order per band group
)
```

#### Adaptive Filter Order

Dynamically selects filter order based on signal complexity:

```python
def adaptive_order_selection(features, order_range=(3, 8)):
    complexity = estimate_complexity(features)  # 0.0 - 1.0
    order = order_range[0] + complexity * (order_range[1] - order_range[0])
    return int(order)
```

---

## Feature Extraction

### ERB (Equivalent Rectangular Bandwidth)

ERB bands model human auditory perception:

```python
# ERB scale conversion
erb_freq = lambda f: 9.265 * np.log(1 + f / (24.7 * 9.265))
hz_to_erb = lambda f: 24.7 * 9.265 * (np.exp(f / 9.265) - 1)
```

Default configuration:
- `nb_erb=32` bands
- Frequency range: 0 Hz - 24 kHz
- Logarithmic spacing following ERB scale

### DF (Deep Filtering) Features

DF features focus on speech-dominant frequencies:

```python
# DF frequency range (default)
nb_df = 96  # Bins
df_bins = spectrum[:, :nb_df]  # 0 - ~4kHz for 48kHz audio
```

---

## Training Components

### Loss Functions

#### Spectral Loss

```python
SpectralLoss(
    factor_magnitude=1.0,
    factor_complex=1.0,
    factor_stft=1.0,
)
```

Components:
- **Magnitude Loss**: L1/L2 on magnitude spectrum
- **Complex Loss**: L1/L2 on real and imaginary parts
- **Multi-Resolution STFT**: Multiple FFT sizes

#### Mask Loss

```python
MaskLoss(
    factor_erb=1.0,   # ERB gain loss
    factor_df=1.0,    # DF coefficient loss
)
```

#### LSNR (Local SNR) Loss

Perceptual loss based on local SNR improvement:

```python
LSNRLoss(
    factor=1.0,
    eps=1e-7,
)
```

#### GAN Losses

```python
# Generator loss
g_loss = adversarial_loss + λ_fm * feature_matching_loss

# Discriminator loss  
d_loss = real_loss + fake_loss
```

### Discriminator Architecture

Multi-scale spectrogram discriminator:

```python
DfNet4Discriminator(
    in_channels=1,
    hidden_channels=[64, 128, 256, 512],
    num_scales=3,        # Multiple spectrogram resolutions
    num_layers_per_scale=3,
)
```

### Knowledge Distillation

Teacher-student training for lightweight deployment:

```python
DistillationLoss(
    teacher_model=teacher,
    alpha_kd=0.5,         # Distillation weight
    temperature=4.0,      # Softmax temperature
    layer_matching=True,  # Intermediate layer matching
)
```

### Quantization-Aware Training

```python
# Prepare model for QAT
qat_model = prepare_qat(model)

# Train with quantization simulation
for batch in dataloader:
    output = qat_model(batch)
    loss = criterion(output, target)
    ...

# Convert to quantized model
quantized = torch.quantization.convert(qat_model)
```

---

## Configuration Reference

### DfNet4Config

```python
@dataclass
class DfNet4Config:
    # Audio parameters
    sr: int = 48000                # Sample rate
    fft_size: int = 960            # FFT size
    hop_size: int = 480            # Hop size (10ms)
    
    # Feature dimensions
    nb_erb: int = 32               # ERB bands
    nb_df: int = 96                # DF frequency bins
    df_order: int = 5              # Default filter order
    df_lookahead: int = 2          # Lookahead frames
    
    # Mamba backbone
    mamba_d_model: int = 256       # Model dimension
    mamba_d_state: int = 64        # State dimension
    mamba_d_conv: int = 4          # Convolution width
    mamba_expand: int = 2          # Expansion factor
    num_mamba_layers: int = 4      # Number of layers
    mamba_dropout: float = 0.0     # Dropout rate
    
    # Optional features
    hybrid_encoder: bool = False   # Multi-domain encoder
    use_time_domain_enc: bool = False  # Time-domain branch
    use_phase_enc: bool = False    # Phase branch
    multi_res_df: bool = False     # Multi-resolution DF
    adaptive_df_order: bool = False # Adaptive order
    
    # Adaptive DF parameters
    df_order_range: Tuple[int, int] = (3, 8)
    
    # Multi-res DF parameters
    df_resolutions: List[int] = field(default_factory=lambda: [2, 4])
    
    # Decoder parameters
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    
    # Training
    use_fp16: bool = False         # Mixed precision
```

### Model Size Comparison

| Model | Parameters | RTF (CPU) | RTF (GPU) |
|-------|-----------|-----------|-----------|
| DFNet1 | ~1.8M | 0.35 | 0.05 |
| DFNet2 | ~1.4M | 0.25 | 0.04 |
| DFNet3 | ~1.6M | 0.30 | 0.04 |
| **DFNet4** | ~2.6M | 0.40 | 0.05 |
| DFNet4Lite | ~1.3M | 0.22 | 0.03 |

*RTF < 1.0 means real-time capable. Lower is better.*

---

## File Structure

```
DeepFilterNet/
├── df/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration utilities
│   ├── model.py              # Base model interface
│   ├── modules.py            # Shared neural network modules
│   ├── deepfilternet.py      # DFNet1 implementation
│   ├── deepfilternet2.py     # DFNet2 implementation
│   ├── deepfilternet3.py     # DFNet3 implementation
│   ├── deepfilternet4.py     # DFNet4 implementation ← NEW
│   ├── loss.py               # Loss functions
│   ├── train.py              # Training script
│   ├── enhance.py            # Inference script
│   └── scripts/
│       ├── export_onnx.py    # ONNX export ← NEW
│       └── benchmark_dfnet4.py # Benchmarking ← NEW
└── tests/
    └── test_deepfilternet4.py # DFNet4 tests ← NEW
```

---

## References

1. Schröter et al., "DeepFilterNet: A Low Complexity Speech Enhancement Framework", ICASSP 2022
2. Schröter et al., "DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices", IWAENC 2022
3. Schröter et al., "DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement", INTERSPEECH 2023
4. Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
