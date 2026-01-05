<!-- markdownlint-disable-file -->
# DeepFilterNet4 Architecture Research

## Research Date: 2026-01-05

## 1. Current Architecture Analysis

### 1.1 DeepFilterNet3 Architecture Overview

**Source Files:**
- `DeepFilterNet/df/deepfilternet3.py` - Main model implementation
- `DeepFilterNet/df/deepfilternet2.py` - Previous generation reference
- `DeepFilterNet/df/modules.py` - Core building blocks
- `DeepFilterNet/df/multiframe.py` - Multi-frame filtering operations
- `DeepFilterNet/df/loss.py` - Loss functions

**Key Components:**

1. **Encoder** (Lines 97-163 in deepfilternet3.py):
   - ERB pathway: `erb_conv0` → `erb_conv1` → `erb_conv2` → `erb_conv3`
   - DF pathway: `df_conv0` → `df_conv1` → `df_fc_emb`
   - Embedding GRU: `SqueezedGRU_S` with configurable skip connections
   - LSNR estimation: Linear + Sigmoid

2. **ERB Decoder** (Lines 188-247):
   - Embedding GRU processing
   - Transposed convolutions: `convt3` → `convt2` → `convt1`
   - Pathway convolutions: `conv3p`, `conv2p`, `conv1p`, `conv0p`
   - Output: ERB mask with Sigmoid activation

3. **DF Decoder** (Lines 268-310):
   - GRU-based coefficient prediction
   - Output: Deep filter coefficients [B, T, F, O*2]
   - Optional skip connection (identity/groupedlinear)

4. **DfNet Main Model** (Lines 312-457):
   - ERB + DF pathway fusion
   - Multi-frame deep filtering via `MF.DF`
   - Optional post-filter for artifact reduction
   - LSNR-based dropout for training

**Current Hyperparameters (ModelParams):**
```python
conv_ch: int = 16                    # Convolutional channels
conv_kernel: List[int] = (1, 3)      # Kernel size
emb_hidden_dim: int = 256            # Embedding GRU hidden dim
emb_num_layers: int = 2              # Number of embedding GRU layers
df_hidden_dim: int = 256             # DF GRU hidden dim
df_num_layers: int = 3               # DF GRU layers
df_order: int = 5                    # Deep filter order (frame size)
nb_df: int = 96                      # DF frequency bins
nb_erb: int = 32                     # ERB bands
lin_groups: int = 1                  # Linear layer groups
```

### 1.2 GRU Usage Analysis

**Current GRU Modules:**
1. `SqueezedGRU_S` (modules.py Lines 650-750):
   - Input squeeze → GRU → Output expand
   - Optional skip connections
   - Used in: Encoder embedding, ERB decoder, DF decoder

2. `GroupedGRU` (modules.py):
   - Grouped hidden states for parallelism
   - Used in: DFNet2 variants

**GRU Limitations Identified:**
- Sequential processing prevents parallelization
- O(T) memory for hidden states
- Limited long-range dependency modeling
- Gradient vanishing on very long sequences

### 1.3 Multi-Frame Filtering Analysis

**Source:** `DeepFilterNet/df/multiframe.py` (Lines 1-120)

**Core Classes:**
1. `MultiFrameModule` - Base class for MF operations
   - `spec_unfold()` - Unfolds spectrogram for frame-based processing
   - `solve()` - Solves Wiener-like filtering equations
   - `apply_coefs()` - Applies filter coefficients

2. `DF` class (MF.DF):
   - Implements deep filtering with learned coefficients
   - Input: spec [B, 1, T, F, 2], coefs [B, O, T, F, 2]
   - Output: Filtered spectrum

**Current Limitations:**
- Fixed filter order (df_order=5)
- Single resolution processing
- No adaptive order selection

### 1.4 Loss Function Analysis

**Source:** `DeepFilterNet/df/loss.py` (Lines 1-300+)

**Current Losses:**
1. `MultiResSpecLoss` - Multi-resolution spectral magnitude/complex loss
2. `SpectralLoss` - Single-resolution with magnitude/complex/underestimation weighting
3. `MaskLoss` - ERB mask prediction loss (WG/IRM/IAM targets)
4. `DfAlphaLoss` - Penalty for DF usage in very noisy segments

**Missing Perceptual Losses:**
- No adversarial/discriminator losses
- No differentiable DNSMOS proxy
- No speaker preservation losses

---

## 2. State-of-the-Art Research (2024-2026)

### 2.1 Mamba/State Space Models for Speech Enhancement

**Key Papers & Models:**

| Model | Domain | PESQ | STOI | Params | FLOPs |
|-------|--------|------|------|--------|-------|
| SE-Mamba | Mag + Phase | 3.55 | 0.96 | 2.25M | 65.46G |
| Mamba-SEUNet | Mag + Phase | 3.57 | 0.96 | 3.78M | 10.28G |
| MH-SENet | Time + Mag + Phase | 3.62 | 0.96 | 0.99M | 16.63G |

**Mamba Advantages:**
- Linear O(n) complexity vs O(n²) for attention
- Selective state space modeling for long-range dependencies
- Hardware-efficient parallel scan implementation
- Better memory efficiency for streaming

**Key Insight from MH-SENet:**
- Hybrid time-frequency processing achieves best results
- Simultaneous processing of time-domain and frequency-domain signals
- Lower parameter count with better performance

### 2.2 Conformer Architecture

**Structure:**
```
Input → FFN → Multi-Head Self-Attention → Conv Module → FFN → Output
              ↓                            ↓
         LayerNorm                    Depthwise Conv
```

**Benefits for Speech:**
- Local patterns via depthwise convolutions
- Global context via self-attention
- Better gradient flow than pure RNN

### 2.3 Hybrid Time-Frequency Processing

**MH-SENet Architecture (ISCA Interspeech 2025):**
1. Time-domain branch: Raw waveform encoding
2. Magnitude branch: STFT magnitude processing
3. Phase branch: Phase spectrum processing
4. Cross-domain fusion: Attention-based feature fusion

**Why It Works:**
- Time domain captures fine temporal details
- Magnitude captures spectral envelope
- Phase critical for speech naturalness
- Fusion leverages complementary information

---

## 3. Proposed DeepFilterNet4 Architecture

### 3.1 High-Level Design

```
                    ┌─────────────────────────────────────┐
                    │         DeepFilterNet4              │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌──────────┐   ┌──────────┐   ┌──────────┐
              │  Time    │   │Magnitude │   │  Phase   │
              │ Encoder  │   │ Encoder  │   │ Encoder  │
              └────┬─────┘   └────┬─────┘   └────┬─────┘
                   │              │              │
                   └──────────────┼──────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │   Cross-Domain Fusion       │
                    │   (Mamba + Attention)       │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────┼───────────────┐
                    ▼             ▼               ▼
              ┌──────────┐ ┌──────────┐   ┌──────────┐
              │   ERB    │ │Multi-Res │   │Adaptive  │
              │ Decoder  │ │   DF     │   │ Order    │
              └────┬─────┘ └────┬─────┘   └────┬─────┘
                   │            │              │
                   └────────────┼──────────────┘
                                ▼
                    ┌─────────────────────────────┐
                    │      Output Synthesis       │
                    └─────────────────────────────┘
```

### 3.2 Module Specifications

#### 3.2.1 Mamba Block (Core Building Block)

**File Location:** `DeepFilterNet/df/mamba.py` (new file)

```python
class MambaBlock(nn.Module):
    """Selective State Space Model block for sequence modeling.
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension (default: 16)
        d_conv: Local convolution width (default: 4)
        expand: Expansion factor for inner dimension (default: 2)
        dt_rank: Rank for delta projection (default: "auto")
        bias: Whether to use bias (default: False)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        bias: bool = False,
    ):
        # Inner projection
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        
        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # SSM parameters
        self.A_log = nn.Parameter(...)  # Log of A matrix
        self.D = nn.Parameter(...)       # Skip connection
        
        # Selective scan
        self.x_proj = nn.Linear(...)     # (dt, B, C) projection
        self.dt_proj = nn.Linear(...)    # Delta projection
        
        # Output
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
```

**Key Operations:**
1. Input projection → (x, z) split
2. Causal 1D convolution on x
3. Selective scan with input-dependent (Δ, B, C)
4. Gated output: y * silu(z)
5. Output projection

#### 3.2.2 Hybrid Encoder

**File Location:** `DeepFilterNet/df/deepfilternet4.py`

```python
class HybridEncoder(nn.Module):
    """Parallel time/magnitude/phase encoding with fusion."""
    
    def __init__(self, params: ModelParams):
        # Time-domain branch
        self.time_conv = nn.Sequential(
            nn.Conv1d(1, params.conv_ch, 7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv1d(params.conv_ch, params.conv_ch * 2, 5, stride=2, padding=2),
        )
        
        # Magnitude branch (similar to current ERB encoder)
        self.mag_encoder = Conv2dNormAct(...)
        
        # Phase branch
        self.phase_encoder = Conv2dNormAct(...)
        
        # Cross-domain fusion
        self.fusion = CrossDomainAttention(
            time_dim=params.conv_ch * 2,
            freq_dim=params.conv_ch * 4,
            out_dim=params.emb_hidden_dim,
        )
        
        # Mamba sequence modeling (replaces GRU)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(params.emb_hidden_dim)
            for _ in range(params.emb_num_layers)
        ])
```

#### 3.2.3 Multi-Resolution Deep Filter

**File Location:** `DeepFilterNet/df/multiframe.py` (extend)

```python
class MultiResolutionDF(nn.Module):
    """Apply deep filtering at multiple frequency resolutions."""
    
    def __init__(
        self,
        resolutions: List[Tuple[int, int]],  # [(num_freqs, frame_size), ...]
        lookahead: int = 0,
    ):
        self.df_ops = nn.ModuleList([
            DF(num_freqs=nf, frame_size=fs, lookahead=lookahead)
            for nf, fs in resolutions
        ])
        
        # Learnable resolution weighting
        self.resolution_weights = nn.Parameter(
            torch.ones(len(resolutions)) / len(resolutions)
        )
    
    def forward(self, spec: Tensor, coefs_list: List[Tensor]) -> Tensor:
        outputs = []
        for df_op, coefs in zip(self.df_ops, coefs_list):
            outputs.append(df_op(spec, coefs))
        
        # Weighted combination
        weights = F.softmax(self.resolution_weights, dim=0)
        return sum(w * out for w, out in zip(weights, outputs))
```

#### 3.2.4 Adaptive Filter Order

```python
class AdaptiveOrderPredictor(nn.Module):
    """Predict optimal filter order based on input characteristics."""
    
    def __init__(self, emb_dim: int, max_order: int = 7):
        self.max_order = max_order
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_order),
        )
    
    def forward(self, emb: Tensor, temperature: float = 1.0) -> Tensor:
        logits = self.predictor(emb)
        # Soft selection during training, hard during inference
        if self.training:
            return F.softmax(logits / temperature, dim=-1)
        else:
            return F.one_hot(logits.argmax(-1), self.max_order).float()
```

### 3.3 Training Enhancements

#### 3.3.1 Multi-Period Discriminator

**File Location:** `DeepFilterNet/df/discriminator.py` (new file)

```python
class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator for adversarial training."""
    
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])
    
    def forward(self, y: Tensor) -> Tuple[List[Tensor], List[List[Tensor]]]:
        """Returns (scores, feature_maps) for each period."""
        ...

class PeriodDiscriminator(nn.Module):
    """Single-period sub-discriminator."""
    
    def __init__(self, period: int):
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
```

#### 3.3.2 Differentiable DNSMOS Proxy

**File Location:** `DeepFilterNet/df/dnsmos_proxy.py` (new file)

```python
class DNSMOSProxy(nn.Module):
    """Differentiable proxy model for DNSMOS prediction.
    
    Pre-trained on (audio, DNSMOS_score) pairs to approximate
    the non-differentiable DNSMOS evaluation.
    """
    
    def __init__(self, n_fft: int = 512, hop_length: int = 256):
        self.feature_extractor = nn.Sequential(
            # Mel spectrogram features
            ...
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # SIG, BAK, OVL scores
        )
    
    def forward(self, audio: Tensor) -> Dict[str, Tensor]:
        features = self.feature_extractor(audio)
        scores = self.regressor(features)
        return {
            "sig": scores[..., 0],
            "bak": scores[..., 1],
            "ovl": scores[..., 2],
        }
```

#### 3.3.3 Speaker Preservation Loss

```python
class SpeakerContrastiveLoss(nn.Module):
    """Ensure enhanced speech preserves speaker characteristics."""
    
    def __init__(self, speaker_model: str = "ecapa_tdnn"):
        self.speaker_encoder = load_pretrained_speaker_model(speaker_model)
        self.speaker_encoder.eval()
        for p in self.speaker_encoder.parameters():
            p.requires_grad = False
    
    def forward(
        self,
        clean: Tensor,
        enhanced: Tensor,
        noisy: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            clean_emb = self.speaker_encoder(clean)
            noisy_emb = self.speaker_encoder(noisy)
        enhanced_emb = self.speaker_encoder(enhanced)
        
        # Pull enhanced toward clean speaker embedding
        pos_sim = F.cosine_similarity(enhanced_emb, clean_emb)
        # Push away from noise characteristics (optional)
        neg_sim = F.cosine_similarity(enhanced_emb, noisy_emb)
        
        return -pos_sim.mean() + 0.1 * neg_sim.mean()
```

---

## 4. Implementation Phases

### Phase 1: Foundation (Core Mamba Integration)
1. Implement `MambaBlock` module
2. Create `SqueezedMamba` as drop-in GRU replacement
3. Unit tests for Mamba modules
4. Benchmark Mamba vs GRU performance

### Phase 2: Hybrid Encoder
1. Implement time-domain encoder branch
2. Implement phase encoder branch
3. Create `CrossDomainAttention` fusion module
4. Integrate with existing magnitude/ERB pathway
5. Update `HybridEncoder` to use Mamba layers

### Phase 3: Multi-Resolution Deep Filtering
1. Implement `MultiResolutionDF` module
2. Add `AdaptiveOrderPredictor`
3. Update DF decoder for multi-resolution outputs
4. Integration tests

### Phase 4: Training Enhancements
1. Implement `MultiPeriodDiscriminator`
2. Train `DNSMOSProxy` model
3. Implement `SpeakerContrastiveLoss`
4. Update `Loss` class to support new losses
5. GAN training loop modifications

### Phase 5: Model Variants & Optimization
1. Create `DFNet4Lite` variant (50% params)
2. Quantization-aware training setup
3. Knowledge distillation training script
4. ONNX/TensorRT export support

### Phase 6: Integration & Testing
1. Full model integration
2. Comprehensive test suite
3. Benchmark against DFNet3
4. Documentation and examples

---

## 5. File Structure

```
DeepFilterNet/df/
├── __init__.py                    # Update exports
├── mamba.py                       # NEW: Mamba/S4 modules
├── deepfilternet4.py              # NEW: DFNet4 model
├── discriminator.py               # NEW: Adversarial training
├── dnsmos_proxy.py                # NEW: Differentiable DNSMOS
├── hybrid_encoder.py              # NEW: Time+Freq encoder
├── multiframe.py                  # MODIFY: Add MultiResolutionDF
├── modules.py                     # MODIFY: Add SqueezedMamba
├── loss.py                        # MODIFY: Add new losses
├── train.py                       # MODIFY: GAN training support
└── config.py                      # MODIFY: DFNet4 params
```

---

## 6. Configuration Parameters (DFNet4)

```ini
[deepfilternet4]
# Architecture
BACKBONE = "mamba"                  # "mamba" or "gru" (fallback)
MAMBA_D_STATE = 16                  # SSM state dimension
MAMBA_D_CONV = 4                    # Local conv width
MAMBA_EXPAND = 2                    # Expansion factor

# Hybrid Encoder
USE_TIME_BRANCH = true              # Enable time-domain branch
USE_PHASE_BRANCH = true             # Enable phase branch
FUSION_TYPE = "attention"           # "attention" or "concat"

# Multi-Resolution DF
DF_RESOLUTIONS = "96,5;48,3;24,2"   # num_freqs,frame_size pairs
ADAPTIVE_ORDER = true               # Enable adaptive order
MAX_DF_ORDER = 7                    # Maximum filter order

# Training
USE_DISCRIMINATOR = true            # Enable adversarial training
USE_DNSMOS_LOSS = true              # Enable DNSMOS proxy loss
USE_SPEAKER_LOSS = true             # Enable speaker preservation
DISCRIMINATOR_WEIGHT = 0.1          # Adversarial loss weight
DNSMOS_WEIGHT = 0.05                # DNSMOS loss weight
SPEAKER_WEIGHT = 0.02               # Speaker loss weight
```

---

## 7. Expected Performance Targets

| Metric | DFNet3 | DFNet4 (Target) | Improvement |
|--------|--------|-----------------|-------------|
| PESQ | ~3.10 | 3.50-3.60 | +13-16% |
| STOI | ~0.95 | 0.96-0.97 | +1-2% |
| DNSMOS-OVL | ~3.20 | 3.50-3.60 | +9-12% |
| Parameters | ~2.0M | 1.5-2.5M | 0-25% reduction |
| RTF (CPU) | ~0.30 | <0.25 | +17% faster |
| RTF (MPS) | ~0.15 | <0.10 | +33% faster |

---

## 8. Dependencies

### New Python Dependencies
```
mamba-ssm>=1.2.0          # Mamba implementation
einops>=0.7.0             # Tensor operations
resemblyzer>=0.1.3        # Speaker embeddings (optional)
```

### Hardware Requirements
- **Training:** GPU with 16GB+ VRAM (A100/H100 recommended for GAN training)
- **Inference:** CPU, CUDA, or MPS (Apple Silicon)

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Mamba complexity | Medium | High | Fallback to GRU, gradual integration |
| GAN training instability | High | Medium | Careful hyperparameter tuning, progressive training |
| DNSMOS proxy accuracy | Medium | Low | Ensemble of proxy models, validation checks |
| Increased model size | Low | Medium | Knowledge distillation, pruning |
| MPS compatibility | Medium | Medium | Already addressed in recent work |

---

## 10. References

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
2. Kim et al. (2025). Mamba-based Hybrid Model for Speech Enhancement. ISCA Interspeech.
3. Kong, J., et al. (2020). HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.
4. Gulati, A., et al. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition.
5. Schröter, H., et al. (2022). DeepFilterNet: A Low Complexity Speech Enhancement Framework.

