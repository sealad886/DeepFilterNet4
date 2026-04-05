# df_mlx Training Optimization Design

**Date**: 2026-04-05
**Status**: Draft
**Priority**: High

## Executive Summary

Optimize memory and speed performance of the `df_mlx` training pathways. The strategy prioritizes high-ROI custom Metal kernels for tensor math hotspots, pure-MLX vectorization for control-plane bottlenecks, and data path optimizations. All changes maintain numerical parity with existing implementations.

## Current State Analysis

### Training Pipeline Overview

```
Data Loading (Python) → Model Forward (MLX + Metal) → Loss (MLX) → Backward (MLX + Metal)
     ↓                        ↓                         ↓               ↓
PrefetchDataLoader      DfOp, STFT/iSTFT        SpectralLoss      DfOp VJP
                         Metal Kernels          (multi-res)
```

### Identified Bottlenecks

| ID | Component | Current Issue | Impact |
|----|-----------|---------------|--------|
| P1 | SpectralLoss multi-res STFT | 3 separate STFTs per call, kernel overhead | High |
| P2 | DfOp backward (VJP) | Python loop over df_order taps for gradient accumulation | High |
| P3 | Mel frontend (DNSMOS) | Nested Python loops for filterbank | Medium |
| P4 | Batch assembly | np.stack + Python list growth | Medium |

### Existing Custom Kernels

| Kernel | Status | Used in Training |
|--------|--------|------------------|
| df_op_kernel | ✓ Differentiable | ✓ via mx.custom_function |
| istft_overlap_add_kernel | ✓ Differentiable | ✓ via mx.custom_function |
| mel_power_log_kernel | ✓ Differentiable | Partial |
| post_filter_kernel | ✓ Differentiable | ✓ |

---

## Optimization Plan

### Epic 1: Fused Multi-Resolution Spectral Loss Frontend

**Problem**: `FusedSpectralLoss` calls STFT 6 times (3 FFT sizes × 2 signals) with per-call framing, windowing, and FFT kernel launches.

**Solution**: Create a fused Metal kernel that:
1. Accepts raw audio (pred, target)
2. Applies all 3 FFT resolutions in a single dispatch
3. Returns magnitude + complex losses directly

**Files**: `df_mlx/kernels.py`, `df_mlx/loss.py`

**Kernel Design**:
```metal
// Single kernel processes all resolutions
// Threadgroup: (batch * n_frames, 1, 1)
// Each thread computes one output bin from one FFT resolution
kernel void multi_res_spectral_loss(
    device float* pred_audio,  // (B, samples)
    device float* target_audio,
    device float* windows[],    // 3 window arrays
    device uint* config,       // [n_fft_0, n_fft_1, n_fft_2, hop_0, hop_1, hop_2]
    device float* out_loss,    // (B, 3) output losses per resolution
    ...
)
```

**Expected Impact**: 30-50% reduction in loss computation time

### Epic 2: DfOp VJP Optimization

**Problem**: The backward pass (`_dfop_vjp` in `kernels.py`) uses a Python `for k in range(df_order)` loop to accumulate gradients to `d_spec_real_pad`. This creates O(df_order) sequential scatter operations.

**Solution**: Two-phase optimization:
1. **Phase A**: Vectorize the gradient accumulation using MLX's scatter operations
2. **Phase B**: Create a fused Metal kernel for the entire VJP

**Files**: `df_mlx/kernels.py`

**Current VJP Code** (lines 195-201):
```python
for k in range(df_order):
    cr_k = coef_real[:, :, :, k]
    ci_k = coef_imag[:, :, :, k]
    grad_r = cr_k * d_out_real + ci_k * d_out_imag
    grad_i = cr_k * d_out_imag - ci_k * d_out_real
    d_spec_real_pad = d_spec_real_pad.at[:, k : k + output_time, :].add(grad_r)
    d_spec_imag_pad = d_spec_imag_pad.at[:, k : k + output_time, :].add(grad_i)
```

**Optimized Approach**:
```python
# Expand gradients for all taps at once
# d_grad[b, t+k, f] = conj(coef[b,t,f,k]) * d_out[b,t,f]
# Use mx.pad + reshape to avoid Python loop

# Step 1: Compute all gradient tensors at once
# grad_r[b, k, t, f] and grad_i[b, k, t, f]
# Step 2: Use vectorized scatter with cumulative indices
```

**Expected Impact**: 20-40% faster backward pass for DfOp

### Epic 3: Mel Frontend Vectorization

**Problem**: `dnsmos_proxy.py` uses nested Python loops for mel filterbank computation.

**Solution**: Replace with fused MLX operations:
1. Pre-compute filter indices
2. Use MLX's matmul for filterbank application
3. Ensure compatibility with existing `mel_power_log_kernel`

**Files**: `df_mlx/dnsmos_proxy.py`, `df_mlx/ops.py`

**Current Pattern**:
```python
for m in range(n_mels):
    for j in range(n_freqs):
        mel_spec[b, t, m] += mel_fb[m, j] * power[b, t, j]
```

**Optimized Pattern**:
```python
# mel_fb: (n_mels, n_freqs), power: (B, T, n_freqs)
# Result: (B, T, n_mels) via matmul
mel_spec = mx.matmul(power, mx.transpose(mel_fb))
```

**Expected Impact**: 2-3x faster mel feature extraction

### Epic 4: Batch Assembly Optimization

**Problem**: `_assemble_batch` in `dynamic_dataset.py` uses Python loops with indexed numpy assignment.

**Solution**: Pre-allocate buffers and use vectorized operations where possible.

**Current** (lines 799-808):
```python
for i, s in enumerate(samples):
    noisy_real[i] = s.noisy_spec.real
    ...
```

**Optimized**:
```python
# Stack complex arrays directly if they're contiguous
noisy_spec = np.stack([s.noisy_spec for s in samples], axis=0)
# Use mx.array directly with correct dtypes
```

**Expected Impact**: 10-20% reduction in data loading overhead

---

## Implementation Order

1. **Spectral Loss Kernel** (highest ROI, clean VJP)
2. **DfOp VJP Vectorization** (medium complexity, high impact)
3. **Mel Frontend** (straightforward vectorization)
4. **Batch Assembly** (incremental improvement)

---

## Validation Strategy

### Benchmark Infrastructure

Existing benchmarks in `df_mlx/benchmark_*.py`:
- `benchmark_hotspots.py` - Microbench for DfOp, STFT, iSTFT, Mel, SpectralLoss
- `benchmark_train_step.py` - Full training step profiling
- `benchmark_pipeline.py` - Data pipeline benchmarks

### Acceptance Criteria

| Optimization | Metric | Target |
|--------------|--------|--------|
| Spectral Loss | p95 latency | < 15% regression |
| DfOp VJP | backward pass time | > 20% speedup |
| Mel Frontend | feature extraction time | > 50% speedup |
| Batch Assembly | samples/s | > 10% improvement |

### Tests Required

1. Numerical parity tests for all modified operations
2. Training convergence tests (loss curves should match)
3. Model quality tests (SI-SDR improvement on validation set)

---

## Technical Constraints

1. **MLX Version**: Must work with current MLX API
2. **Metal Compatibility**: Kernels must compile for Apple Silicon
3. **Differentiable**: All custom kernels must have VJP for training
4. **Fallback**: Pure-MLX fallback when Metal kernels unavailable

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Metal kernel compilation issues | Maintain pure-MLX fallback paths |
| Numerical instability in fused ops | Extensive testing with tolerance checks |
| Training convergence changes | Validate on benchmark dataset |

---

## Success Metrics

- **Training throughput**: > 20% improvement in samples/second
- **Memory efficiency**: < 10% increase in peak memory
- **p95 latency jitter**: Reduced variance in step times
- **Numerical parity**: All changes within 1e-6 tolerance

---

## Commit Strategy

Commits follow Conventional Commits:
- `feat(mlx-kernel): fused multi-res spectral loss kernel`
- `perf(dfop): vectorized VJP gradient accumulation`
- `perf(mel): vectorized mel frontend computation`
- `perf(data): optimized batch assembly`
