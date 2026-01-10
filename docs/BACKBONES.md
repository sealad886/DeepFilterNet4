# DeepFilterNet4 Training Backbones

This document explains the three sequence modeling backbone architectures available for training DeepFilterNet4 on Apple Silicon using MLX.

## Overview

The **backbone** is the temporal sequence modeling component that processes audio features across time. It sits in the middle of the network architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DfNet4 Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Audio Input                                                   │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────────┐                                          │
│   │  ERB Encoder    │  Frequency → Feature extraction           │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │    BACKBONE     │  ◄── Temporal sequence modeling           │
│   │  (Mamba/GRU/    │      This is what we're comparing!        │
│   │   Attention)    │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │  ERB Decoder +  │  Feature → Enhanced audio                 │
│   │  Deep Filtering │                                          │
│   └─────────────────┘                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Comparison

All benchmarks on Apple M3 Pro (18 GPU cores), batch=8, sequence=500 frames:

| Backbone | Forward | Backward | **Total** | vs Mamba | Memory |
|----------|---------|----------|-----------|----------|--------|
| **Mamba** | 575ms | 2,732ms | 3,307ms | 1.0× (baseline) | Low |
| **GRU** | 77ms | 857ms | 934ms | **3.5× faster** | Low |
| **Attention** | 66ms | 637ms | 703ms | **4.7× faster** | Higher |

### Recommendation

| Use Case | Recommended Backbone |
|----------|---------------------|
| 🚀 **Fastest training** | `attention` |
| ⚖️ **Balance speed/memory** | `gru` |
| 📚 **Research/compatibility** | `mamba` |
| 🎯 **Production inference** | `gru` (streaming-friendly) |

---

## 1. Mamba (State Space Model)

```
--backbone-type mamba
```

### Architecture

Mamba is a **Selective State Space Model (S6)** that provides linear-time sequence modeling with input-dependent state transitions.

```
                    Mamba Block
    ┌──────────────────────────────────────────────┐
    │                                              │
    │  Input x ──────┬───────────────────┐         │
    │                │                   │         │
    │                ▼                   ▼         │
    │         ┌───────────┐       ┌───────────┐   │
    │         │  In Proj  │       │  In Proj  │   │
    │         │  (Linear) │       │  (Linear) │   │
    │         └─────┬─────┘       └─────┬─────┘   │
    │               │                   │         │
    │               ▼                   │         │
    │         ┌───────────┐             │         │
    │         │   Conv1D  │             │         │
    │         │ (causal)  │             │         │
    │         └─────┬─────┘             │         │
    │               │                   │         │
    │               ▼                   │         │
    │         ┌───────────┐             │         │
    │         │   SiLU    │             │         │
    │         └─────┬─────┘             │         │
    │               │                   │         │
    │               ▼                   │         │
    │    ┌──────────────────────┐       │         │
    │    │   Selective Scan     │       │         │
    │    │   ┌───────────────┐  │       │         │
    │    │   │ h_t = Ā·h_{t-1}  │       │         │
    │    │   │     + B̄·x_t   │  │       │         │
    │    │   │ y_t = C·h_t    │  │       │         │
    │    │   └───────────────┘  │       │         │
    │    └──────────┬───────────┘       │         │
    │               │                   │         │
    │               ▼                   ▼         │
    │         ┌─────────────────────────────┐     │
    │         │     Element-wise Multiply    │     │
    │         │        y × SiLU(z)           │     │
    │         └─────────────┬───────────────┘     │
    │                       │                     │
    │                       ▼                     │
    │                 ┌───────────┐               │
    │                 │  Out Proj │               │
    │                 └─────┬─────┘               │
    │                       │                     │
    │  Output ◄─────────────┘                     │
    │                                              │
    └──────────────────────────────────────────────┘
```

### Key Concepts

**State Space Model (SSM):**
```
Continuous:          Discretized (for input x):
  h'(t) = A·h(t) + B·x(t)    h_t = Ā·h_{t-1} + B̄·x_t
  y(t)  = C·h(t) + D·x(t)    y_t = C·h_t + D·x_t
```

**Selective Mechanism:** Unlike traditional SSMs with fixed parameters, Mamba makes A, B, C **input-dependent**:
- Δ (delta): Input-dependent discretization step
- B: Input-dependent input matrix  
- C: Input-dependent output matrix

This allows the model to **selectively remember or forget** information based on content.

**Parallel Scan:** The recurrence is computed efficiently using associative scan:
```
(a₁, b₁) ⊗ (a₂, b₂) = (a₂·a₁, a₂·b₁ + b₂)
```
This reduces complexity from O(L) sequential to O(log L) parallel.

### Pros & Cons

✅ **Pros:**
- Linear O(L) complexity in sequence length
- Constant memory during inference (state-based)
- Good at modeling long-range dependencies
- Theoretically elegant

❌ **Cons:**
- Slow backward pass in MLX (2,732ms)
- Complex implementation
- Parallel scan not fully optimized in MLX

### Code

```python
# SqueezedMamba in df_mlx/mamba.py
self.backbone = SqueezedMamba(
    input_size=256,      # Feature dimension
    hidden_size=256,     # State dimension
    output_size=256,
    num_layers=2,        # Stacked layers
    d_state=16,          # SSM state dimension
    d_conv=4,            # Local conv kernel
    expand_factor=2,     # Inner expansion
)
```

---

## 2. GRU (Gated Recurrent Unit)

```
--backbone-type gru
```

### Architecture

GRU is a **recurrent neural network** that processes sequences step-by-step, maintaining a hidden state.

```
                    GRU Cell (single timestep)
    ┌──────────────────────────────────────────────┐
    │                                              │
    │  x_t ─────────────┬───────────┬─────────┐   │
    │                   │           │         │   │
    │  h_{t-1} ─────┬───┼───────┬───┼─────┐   │   │
    │               │   │       │   │     │   │   │
    │               ▼   ▼       ▼   ▼     │   │   │
    │            ┌────────┐  ┌────────┐   │   │   │
    │            │   σ    │  │   σ    │   │   │   │
    │            │ (reset)│  │(update)│   │   │   │
    │            └───┬────┘  └───┬────┘   │   │   │
    │                │           │        │   │   │
    │           r_t  │      z_t  │        │   │   │
    │                │           │        │   │   │
    │                ▼           │        │   │   │
    │         ┌──────────┐       │        │   │   │
    │         │ r_t ⊙ h  │       │        │   │   │
    │         └────┬─────┘       │        │   │   │
    │              │             │        │   │   │
    │              ▼             │        ▼   ▼   │
    │         ┌─────────┐        │   ┌─────────┐ │
    │         │  tanh   │        │   │  Linear │ │
    │         │(new mem)│        │   │ concat  │ │
    │         └────┬────┘        │   └────┬────┘ │
    │              │             │        │      │
    │         h̃_t  │             │        │      │
    │              │             │        │      │
    │              ▼             ▼        │      │
    │    ┌─────────────────────────────┐  │      │
    │    │ h_t = z_t⊙h_{t-1} + (1-z_t)⊙h̃_t │      │
    │    └─────────────┬───────────────┘  │      │
    │                  │                         │
    │  h_t ◄───────────┘                         │
    │                                              │
    └──────────────────────────────────────────────┘

                SqueezedGRU_S Wrapper
    ┌────────────────────────────────────────────┐
    │                                            │
    │  Input ──► Linear_in ──► GRU ──► Linear_out ──► Output
    │    │       (group)              (group)         │
    │    │                                            │
    │    └────────────── Skip Connection ─────────────┘
    │                                            │
    └────────────────────────────────────────────┘
```

### Key Equations

```
Reset gate:    r_t = σ(W_r·[h_{t-1}, x_t])
Update gate:   z_t = σ(W_z·[h_{t-1}, x_t])  
New memory:    h̃_t = tanh(W·[r_t ⊙ h_{t-1}, x_t])
Hidden state:  h_t = z_t ⊙ h_{t-1} + (1-z_t) ⊙ h̃_t
```

The **reset gate** (r) controls how much past information to forget.
The **update gate** (z) controls the balance between old and new information.

### Pros & Cons

✅ **Pros:**
- Simple and well-understood
- MLX has native `nn.GRU` implementation
- Good for streaming inference (constant state size)
- 3.5× faster than Mamba overall

❌ **Cons:**
- Sequential computation (can't parallelize across time)
- Slow backward pass due to backprop-through-time (857ms)
- O(L) sequential operations in both directions

### Code

```python
# SqueezedGRU_S in df_mlx/modules.py
self.backbone = SqueezedGRU_S(
    input_size=256,
    hidden_size=256,
    output_size=256,
    num_layers=1,
    linear_groups=8,    # Grouped linear for efficiency
    gru_skip=True,      # Residual connection
)
```

---

## 3. Attention (Causal Self-Attention)

```
--backbone-type attention
```

### Architecture

Attention uses **causal self-attention** (like GPT) to model temporal dependencies with fully parallelizable operations.

```
              Multi-Head Causal Self-Attention
    ┌──────────────────────────────────────────────┐
    │                                              │
    │  Input X ─────┬─────────┬─────────┐         │
    │               │         │         │         │
    │               ▼         ▼         ▼         │
    │            ┌─────┐   ┌─────┐   ┌─────┐      │
    │            │ W_Q │   │ W_K │   │ W_V │      │
    │            └──┬──┘   └──┬──┘   └──┬──┘      │
    │               │         │         │         │
    │               ▼         ▼         ▼         │
    │              Q         K         V          │
    │               │         │         │         │
    │               ▼         ▼         │         │
    │         ┌─────────────────┐       │         │
    │         │   Q × Kᵀ        │       │         │
    │         │  ──────────     │       │         │
    │         │    √d_k         │       │         │
    │         └────────┬────────┘       │         │
    │                  │                │         │
    │                  ▼                │         │
    │         ┌─────────────────┐       │         │
    │         │  Causal Mask    │       │         │
    │         │  ┌───────────┐  │       │         │
    │         │  │ 0 -∞ -∞ -∞│  │       │         │
    │         │  │ 0  0 -∞ -∞│  │       │         │
    │         │  │ 0  0  0 -∞│  │       │         │
    │         │  │ 0  0  0  0│  │       │         │
    │         │  └───────────┘  │       │         │
    │         └────────┬────────┘       │         │
    │                  │                │         │
    │                  ▼                │         │
    │            ┌──────────┐           │         │
    │            │ Softmax  │           │         │
    │            └────┬─────┘           │         │
    │                 │                 │         │
    │            Attention              │         │
    │            Weights                │         │
    │                 │                 │         │
    │                 ▼                 ▼         │
    │           ┌───────────────────────────┐     │
    │           │   Attention × V           │     │
    │           └─────────────┬─────────────┘     │
    │                         │                   │
    │                         ▼                   │
    │                   ┌──────────┐              │
    │                   │  W_out   │              │
    │                   └────┬─────┘              │
    │                        │                    │
    │  Output ◄──────────────┘                    │
    │                                              │
    └──────────────────────────────────────────────┘

            SqueezedAttention (Pre-Norm Transformer)
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │  Input ──► Linear_in ──┬─────────────────────┐   │
    │                        │                     │   │
    │           ┌────────────▼────────────┐        │   │
    │           │      × num_layers       │        │   │
    │           │  ┌───────────────────┐  │        │   │
    │           │  │     LayerNorm     │  │        │   │
    │           │  │         │         │  │        │   │
    │           │  │    Attention      │  │        │   │
    │           │  │         │         │  │        │   │
    │           │  │    + Residual ◄───┼──┼───┐    │   │
    │           │  │         │         │  │   │    │   │
    │           │  │     LayerNorm     │  │   │    │   │
    │           │  │         │         │  │   │    │   │
    │           │  │       FFN         │  │   │    │   │
    │           │  │   (expand×2)      │  │   │    │   │
    │           │  │         │         │  │   │    │   │
    │           │  │    + Residual ◄───┼──┼───┘    │   │
    │           │  └───────────────────┘  │        │   │
    │           └─────────────┬───────────┘        │   │
    │                         │                    │   │
    │                         ▼                    │   │
    │                    Linear_out                │   │
    │                         │                    │   │
    │                         ▼                    │   │
    │                    + Skip ◄──────────────────┘   │
    │                         │                        │
    │  Output ◄───────────────┘                        │
    │                                                  │
    └──────────────────────────────────────────────────┘
```

### Key Equations

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QKᵀ/√d_k + M) × V

where M is the causal mask:
      ┌ 0    if i ≥ j  (can attend)
M_ij =│
      └ -∞   if i < j  (cannot attend to future)
```

**Why Causal?** In audio processing, we can only use past and present information, not future frames (for real-time/streaming).

### Pros & Cons

✅ **Pros:**
- **Fully parallelizable** across time dimension
- MLX's attention is highly optimized (Metal kernels)
- **18× faster backward** than GRU for the backbone alone
- **4.7× faster** total training step than Mamba
- Excellent gradient flow (no vanishing gradient through time)

❌ **Cons:**
- O(L²) memory and compute in sequence length
- No persistent state (must recompute for each window)
- Not ideal for streaming inference

### Code

```python
# SqueezedAttention in df_mlx/modules.py
self.backbone = SqueezedAttention(
    input_size=256,
    hidden_size=256,
    output_size=256,
    num_layers=2,        # Pre-norm transformer layers
    num_heads=4,         # Multi-head attention
    linear_groups=8,
    gru_skip=True,       # Skip connection
)
```

---

## Training Speed Analysis

### Why is Attention Fastest?

The key insight is in the **backward pass**:

```
                    Backward Pass Comparison

    GRU (Sequential - must backprop through each timestep):
    ┌─────────────────────────────────────────────────────┐
    │  t=500   t=499   t=498   ...   t=2    t=1    t=0   │
    │    │       │       │             │      │      │    │
    │    ├──────►├──────►├──────► ... ├─────►├─────►│    │
    │    │       │       │             │      │      │    │
    │   ∂L      ∂L      ∂L           ∂L     ∂L     ∂L    │
    │   ──      ──      ──           ──     ──     ──    │
    │   ∂h     ∂h      ∂h           ∂h     ∂h     ∂h    │
    │                                                     │
    │   Sequential: Must compute one at a time!           │
    │   Time: O(L) serial operations                      │
    └─────────────────────────────────────────────────────┘

    Attention (Parallel - all gradients computed together):
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │    ∂L/∂Q    ∂L/∂K    ∂L/∂V                         │
    │      │        │        │                            │
    │      ▼        ▼        ▼                            │
    │  ┌──────────────────────────┐                      │
    │  │   Matrix multiplication   │ ◄── GPU parallel!   │
    │  │   (all timesteps at once) │                      │
    │  └──────────────────────────┘                      │
    │                                                     │
    │   Parallel: Compute all gradients simultaneously!   │
    │   Time: O(1) with enough parallelism               │
    └─────────────────────────────────────────────────────┘
```

### Detailed Timing Breakdown

```
Component Analysis (batch=8, seq=500):

MAMBA:
├── Forward:  575ms  │████████████████████████████████████████│
│   └── Selective scan (sequential associative)
└── Backward: 2732ms │████████████████████████████████████████████████████████████████████████████████████████████████████████│
    └── Backprop through scan + input-dependent params

GRU:
├── Forward:   77ms  │█████│
│   └── MLX native GRU (optimized)
└── Backward: 857ms  │█████████████████████████████████████████████│
    └── Backprop-through-time (500 sequential steps)

ATTENTION:
├── Forward:   66ms  │████│
│   └── Parallel QKV projection + attention
└── Backward: 637ms  │█████████████████████████████████│
    └── Parallel gradient computation (Metal optimized)
```

---

## Usage Examples

### Training with Each Backbone

```bash
# Fastest training (attention)
python -m df_mlx.train_dynamic \
    --cache-dir /path/to/cache \
    --backbone-type attention \
    --batch-size 32 \
    --epochs 100

# Balanced (GRU)
python -m df_mlx.train_dynamic \
    --cache-dir /path/to/cache \
    --backbone-type gru \
    --batch-size 32 \
    --epochs 100

# Original architecture (Mamba)
python -m df_mlx.train_dynamic \
    --cache-dir /path/to/cache \
    --backbone-type mamba \
    --batch-size 16 \
    --epochs 100
```

### Benchmarking

```bash
# Compare all backbones
cd DeepFilterNet
python profile_training.py
```

---

## Implementation Details

### Common Interface

All backbones share the same interface for drop-in replacement:

```python
class BackboneInterface:
    def __call__(
        self,
        x: mx.array,           # (batch, time, features)
        h: mx.array | None     # optional hidden state
    ) -> tuple[mx.array, mx.array]:
        """
        Returns:
            output: (batch, time, output_size)
            hidden: (batch, hidden_size) - last timestep
        """
```

### Memory Considerations

| Backbone | Training Memory | Inference Memory | Streaming? |
|----------|-----------------|------------------|------------|
| Mamba | O(B×L×D×N) | O(B×D×N) | ✅ Yes |
| GRU | O(B×L×D) | O(B×D) | ✅ Yes |
| Attention | O(B×L²×H) | O(B×L²×H) | ⚠️ Windowed |

Where:
- B = batch size
- L = sequence length
- D = hidden dimension
- N = state dimension (Mamba)
- H = number of heads (Attention)

---

## References

1. **Mamba**: Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
2. **GRU**: Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder"
3. **Attention**: Vaswani, A., et al. (2017). "Attention Is All You Need"
4. **DeepFilterNet**: Schröter, H., et al. (2022). "DeepFilterNet: A Low Complexity Speech Enhancement Framework"
