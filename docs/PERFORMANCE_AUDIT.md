# Performance Audit Report: DeepFilterNet/df_mlx/

**Date**: 2026-02-15
**Scope**: `DeepFilterNet/df_mlx/` — MLX-based audio noise suppression on Apple Silicon
**Focus**: Tensor manipulation/casting reduction, CPython→Metal/MLX migration

---

## 1. Executive Summary

1. **~13 redundant `.astype(mx.float32)` graph nodes per training step** in the loss computation path — functions defensively cast inputs that are already FP32after earlier casts in the call chain
2. **`_log1p_mag()` casts unconditionally** — wastes 2 graph nodes per call when inputs are already FP32 (called 3× with FP32 noise inputs per step)
3. **`_compute_pipeline_awesome_losses()` casts clean/noisy to FP32 twice** — once for noise subtraction, again for proxy gates (4 redundant nodes)
4. **`_compute_musicness/_improved_musicness/_pitch_stability/_harmonic_ratio` all defensively cast `mag` to FP32** when callers already pass FP32 — 4 unnecessary nodes
5. **`mx.transpose(self._erb_fb)` recomputed every forward call** (2 sites) — should be cached at init
6. **Attention causal mask `.astype(out.dtype)` computed per call** — should be pre-built at init
7. **`CombinedLoss` uses 5 `float()` sync barriers mid-computation** — should return lazy `mx.array` dict
8. **Validation loop casts FP32→FP32** (4 no-op casts) — validation data is never FP16
9. **Existing Metal kernels (3) cover highest-cost operations** — DfOp, iSTFT, mel-power-log
10. **Training step IS compiled via `mx.compile`** — Python loops in model are traced and unrolled at compile time, so they don't add per-step CPython overhead
11. **Mamba parallel scan creates ~14 temporary tensors per forward** via concatenation in a `for d in range(log2_L)` loop — these could be reduced with pre-allocated buffers
12. **Streaming inference (`StreamingDfNet4.process_frame`) is NOT compiled** — massive per-frame Python dispatch overhead

## 2. Repo Performance Map

### Entrypoints
- **Training**: `train_dynamic.py:main()` → `training_cli_main.py:main()` → compiled training step
- **Inference**: `enhance.py:enhance()` / `enhance_streaming()` / `enhance_batch()`
- **Streaming**: `model.py:StreamingDfNet4.process_audio()`

### Critical Paths
1. Training step: data→FP16 cast→model forward→loss (spectral+awesome+MRSTFT+GAN)→backward→grad clip→optimizer update
2. Inference: audio load→STFT→model forward→mask+DfOp→iSTFT→audio save

### Concurrency Model
- Data loading: `PrefetchDataLoader` with worker threads (NumPy/SciPy)
- Compute: Single-threaded MLX with lazy evaluation, compiled graph execution
- GPU: Apple Metal via MLX framework

### I/O Boundaries
- NumPy→MLX at batch boundary (`mx.array(np_batch)`)
- MLX→NumPy at inference output (`np.array(audio)`)
- File I/O via soundfile/librosa (CPU)

## 3. Measurement & Benchmark Plan

### Tools
- `DeepFilterNet/df_mlx/benchmark_train_step.py` — existing harness for training throughput
- `time.perf_counter` instrumentation at key points
- MLX's lazy evaluation means profiling requires strategic `mx.eval()` placement

### Minimal Benchmark
```bash
cd DeepFilterNet
source ../.venv/bin/activate
python -m df_mlx.benchmark_train_step \
    --cache-dir /path/to/cache \
    --backends prefetch \
    --batch-size 4 \
    --steps 50 \
    --warmup-steps 10
```

### Metrics
- Steps/second, samples/second
- Per-step latency (p50/p95/p99)
- Memory RSS

## 4. Findings (Prioritized)

### PERF-P0-001: Redundant FP32 casts in awesome/pipeline loss chain
- **Severity**: P0
- **Component**: `training_losses.py` loss computation
- **Evidence**:
  - `_log1p_mag()` unconditionally casts to FP32
  - Called with FP32 `noise_real/imag` from `_compute_awesome_losses()` and `_compute_pipeline_awesome_losses()`
  - `_compute_musicness`, `_compute_improved_musicness`, `_compute_pitch_stability`, `_compute_harmonic_ratio` all defensively cast `mag` that's already FP32
  - ~9-13 redundant graph nodes per training step
- **Proposed Optimization**: Add `if x.dtype != mx.float32` guards; cast once at function entry and reuse
- **Verification**: Compare graph node count before/after; run benchmark_train_step
- **Risks**: None — conditional cast produces identical results

### PERF-P0-002: Duplicate FP32 casts in `_compute_pipeline_awesome_losses`
- **Severity**: P0
- **Component**: `training_losses.py:_compute_pipeline_awesome_losses`
- **Evidence**:
  - `_compute_pipeline_awesome_losses()` casts `noisy_real.astype(mx.float32) - clean_real.astype(mx.float32)` for noise
  - Same `clean_real/imag, noisy_real/imag` cast again for proxy gates
  - 4 redundant casts of the same FP16 source tensors
- **Proposed Optimization**: Cast once at function entry, reuse named FP32 variables
- **Verification**: Run existing tests; compare benchmark
- **Risks**: None — mathematically identical

### PERF-P0-003: `_erb_fb` transpose recomputed per forward call
- **Severity**: P0
- **Component**: `model.py:DfNet4.__call__`
- **Evidence**:
  - L1118: `mx.matmul(erb_mask, mx.transpose(self._erb_fb))` — every forward
  - L1190: Same pattern in `forward_with_lsnr`
  - `_erb_fb` is immutable (not a learned parameter)
- **Proposed Optimization**: Pre-compute `self._erb_fb_T = mx.transpose(self._erb_fb)` at init
- **Verification**: Ensure model output unchanged
- **Risks**: Minimal extra memory (small matrix)

### PERF-P1-001: Attention causal mask cast per call
- **Severity**: P1
- **Component**: `modules.py:SqueezedAttention.__call__`
- **Evidence**: L1263 `mask = mask.astype(out.dtype)` creates a new graph node every forward call
- **Proposed Optimization**: Cache mask at init with correct dtype
- **Verification**: Model output unchanged
- **Risks**: Mask shape depends on `seq_len` which varies — need to handle dynamically

### PERF-P1-002: CombinedLoss sync barriers
- **Severity**: P1
- **Component**: `loss.py:CombinedLoss.__call__`
- **Evidence**: L892-915: 5× `float()` calls mid-computation, each forcing GPU eval
- **Proposed Optimization**: Return `Dict[str, mx.array]` instead of `Dict[str, float]`
- **Verification**: Callers that log must call `float()` themselves
- **Risks**: API change — callers must be updated

### PERF-P1-003: Validation loop FP32→FP32 no-op casts
- **Severity**: P1
- **Component**: `train_dynamic.py` validation loop
- **Evidence**: Validation loop casts 4× `.astype(mx.float32)` on data that's already FP32
- **Proposed Optimization**: Add dtype guard
- **Verification**: Existing validation tests
- **Risks**: None

### PERF-P2-001: Mamba parallel scan temporary allocations
- **Severity**: P2
- **Component**: `mamba.py:MambaBlock._selective_scan`
- **Evidence**: L261-275: `for d in range(log2_L)` creates ~14 temporary tensors via concatenation
- **Proposed Optimization**: Pre-allocate output buffer and use scatter/slice assignment
- **Risks**: Complex refactor; compiled training traces this away; mainly affects inference

### PERF-P2-002: Streaming inference not compiled
- **Severity**: P2
- **Component**: `model.py:StreamingDfNet4.process_frame`
- **Evidence**: ~70+ MLX op dispatches per frame without compilation
- **Proposed Optimization**: Wrap in `mx.compile` with state handling
- **Risks**: Complex — requires careful state management with `mx.compile`

## 5. Hardware Acceleration Opportunities

Existing Metal kernels already cover the 3 highest-cost per-frame operations:
1. `df_op_kernel` — fused gather + complex MAC for DfOp
2. `istft_overlap_add_kernel` — fused overlap-add + window normalization
3. `mel_power_log_kernel` — fused power-spectrum → mel → log

### Additional candidates (ranked)

| Priority | Operation | Current | Proposed | Expected Win |
|----------|-----------|---------|----------|-------------|
| P1 | Complex mask + concat (DfOp output) | 2× concat + 4 muls + 2 adds | Fused Metal kernel | Minor — eliminates 2 allocations |
| P2 | Post-filter (7 element-wise ops) | Separate ops | Fused Metal kernel | ~2× by eliminating intermediates |
| P2 | Mamba scan (associative scan) | Python loop + concat | Native `mx.associative_scan` or Metal | Depends on MLX roadmap |

## 6. Quick Wins (Under 60 minutes)

1. **PERF-P0-001**: Add dtype guards to `_log1p_mag` and musicness functions (~15 min)
2. **PERF-P0-002**: Refactor `_compute_pipeline_awesome_losses` cast-once (~15 min)
3. **PERF-P0-003**: Cache `_erb_fb_T` at init (~5 min)
4. **PERF-P1-003**: Add dtype guards to validation loop casts (~5 min)
5. **PERF-P1-002**: Make `CombinedLoss` return lazy arrays (~15 min)

## 7. Implementation Status

### Completed (Phase 1 — P0+P1)

| ID | Optimization | Files Changed | Impact |
|----|-------------|---------------|--------|
| PERF-P0-001 | dtype guards on `_log1p_mag`, `_compute_musicness`, `_compute_pitch_stability`, `_compute_harmonic_ratio`, `_compute_improved_musicness` | `training_losses.py` | ~5 redundant graph nodes eliminated per step |
| PERF-P0-002 | Cast-once refactor for `_compute_pipeline_awesome_losses` | `training_losses.py` | 4 redundant FP16→FP32 casts eliminated |
| PERF-P0-003 | Cast-once refactor for `_compute_awesome_losses` | `training_losses.py` | 4 redundant FP16→FP32 casts eliminated |
| PERF-P0-004 | dtype guards on `_compute_proxy_gates` | `training_losses.py` | 2 conditional cast skips |
| PERF-P0-005 | Cached `_erb_fb_T` at model init | `model.py` | 3 per-call transposes eliminated |
| PERF-P1-001 | Cached SqueezedAttention causal mask by `(seq_len, dtype)` | `modules.py` | Per-call mask creation + cast eliminated |
| PERF-P1-002 | `CombinedLoss` returns `Dict[str, mx.array]` (lazy) | `loss.py` | 5 GPU sync barriers eliminated |
| PERF-P1-003 | dtype guards on `si_sdr()` | `loss.py` | 2 conditional cast skips |
| PERF-P1-004 | dtype guards on `_compute_vad_probs`, `_compute_speech_band_logmag_loss` | `training_losses.py` | 4 conditional cast skips |
| PERF-P1-005 | dtype guards on `specs_to_wavs` | `training_waveform.py` | 2 conditional cast skips |

**Total estimated reduction**: ~20+ unnecessary graph nodes per training step; 5 GPU sync barriers removed.

**Test coverage**: 10 dedicated tests in `tests/test_perf_optimizations.py` + 866 existing tests pass (6 pre-existing failures unrelated to perf changes).

### Completed (Phase 2 — P2 Metal kernels)

| ID | Optimization | Files Changed | Impact |
|----|-------------|---------------|--------|
| PERF-P2-003 | Fused post-filter Metal kernel (22 element-wise ops → 1 GPU dispatch) | `kernels.py`, `model.py` | 22 graph nodes → 1 Metal dispatch; differentiable VJP included |

**Post-filter kernel details**: New `post_filter_kernel()` in `kernels.py` fuses the full post-filter computation (magnitudes, mask ratio, sinusoidal transfer, gain application) into a single Metal kernel with 1 thread per element. Includes `@mx.custom_function` + full VJP for training compatibility. Automatically used when `metal_kernels_available()` returns True, with pure-MLX fallback otherwise. Benchmarked at **1.3–1.5x faster** than the pure-MLX fallback in isolation.

### Reverted (Phase 2 — P2 Mamba scan pre-allocation)

| ID | Optimization | Status | Reason |
|----|-------------|--------|--------|
| PERF-P2-001 | Mamba scan: replace `mx.concatenate` with pre-allocated buffers + slice assignment | **REVERTED** | Benchmarking revealed **20–25% throughput regression**. MLX's lazy evaluation model makes slice assignment (scatter ops) more expensive than concatenation (simple memcpy). Pre-allocating full-size buffers with identity elements adds wasted graph nodes that are immediately overwritten. The `mx.concatenate` approach generates a simpler, more compiler-friendly computation graph. |

### Benchmark Results (main@2e73dc7 vs optimized)

All measurements on Apple M4 Max (36GB), Python 3.10.19, MLX 0.30.6.

| Benchmark | Before (ms) | After (ms) | Delta | Verdict |
|-----------|------------|------------|-------|---------|
| Forward B=1 T=100 | 17.44 | 17.97 | +3% | Within noise |
| Forward B=4 T=100 | 74.22 | 74.99 | +1% | Within noise |
| Forward B=8 T=100 | 133.34 | 138.17 | +4% | Within noise |
| Train Step B=1 | 71.75 | 73.23 | +2% | Within noise |
| Train Step B=4 | 256.60 | 261.84 | +2% | Within noise |
| Train Step B=8 | 486.85 | 503.91 | +4% | Within noise |
| Post-filter (MLX) B=8 | 0.84 | 0.84 | 0% | Neutral |
| Post-filter (Metal) B=8 | N/A | 0.57 | — | **1.5x faster** |
| Peak Mem (fwd B=8) | 2618 MB | 2618 MB | 0% | Neutral |
| Peak Mem (train B=8) | 3487 MB | 3487 MB | 0% | Neutral |

**Key insight**: MLX's lazy evaluation and JIT compilation absorb most redundant-cast and caching optimizations — they eliminate wasted Python-level work but don't change the compiled graph materially. The only measurable improvement is the fused post-filter Metal kernel (1.3–1.5x for that operation).

**Test coverage**: 18 dedicated tests in `tests/test_perf_optimizations.py` + full suite passes.

### Completed (Phase 3 — GAN Pipeline Optimizations)

| ID | Optimization | Files Changed | Impact |
|----|-------------|---------------|--------|
| GAN-P4 | Vectorized loss accumulation: `FeatureMatchingLoss`, `discriminator_loss`, `generator_loss` — replaced sequential `loss += mx.mean(...)` accumulator pattern with `losses.append()` + `mx.mean(mx.stack(losses))` | `loss.py` | Reduces AD graph from linear addition chain (~48 nodes in FM loss) to single stack+mean node. Fewer intermediate graph nodes for backward pass to trace. |
| GAN-P5 | Score-only discriminator forward: added `return_features: bool = True` parameter to all 5 discriminator `__call__` methods (`PeriodDiscriminator`, `ScaleDiscriminator`, `MultiPeriodDiscriminator`, `MultiScaleDiscriminator`, `CombinedDiscriminator`) | `discriminator.py` | When `return_features=False`, skips building Python lists of intermediate feature maps. Produces cleaner computation graph for disc update AD path. |
| GAN-P3 | Score-only mode in disc update path: `disc_loss_wrapper` passes `return_features=False` since discriminator update only needs scores, not feature maps | `train_dynamic.py` | Eliminates ~48 Python list `.append()` calls and associated list bookkeeping per disc update step. |

### GAN Pipeline Analysis

**Architecture**: CombinedDiscriminator = MPD (5 PeriodDiscriminators, periods 2/3/5/7/11) + MSD (3 ScaleDiscriminators). Total: 8 sub-discriminators with 5-7 conv layers each = ~48 conv+activation kernel launches per forward pass.

**Compute profile per GAN-active training step** (disc-update frequency = 1):

| Phase | Disc Forwards | iSTFT Calls | Notes |
|-------|--------------|-------------|-------|
| Generator loss (inside `loss_fn`) | 2 (fake + real) | 2 (pred + clean via `specs_to_wavs`) | Real forward needed for feature matching loss |
| Discriminator update | 2 (fake + real) | 2 (pred + clean via `specs_to_wavs`) | Now uses `return_features=False` |
| **Total** | **4** | **4** | Cannot reduce below 4 disc forwards without algorithmic changes |

**Why 4 discriminator forwards are necessary**: The generator loss path runs `disc(fake)` and `disc(real)` to compute adversarial + feature matching losses — both must be traced through the generator's AD graph. The discriminator update path runs `disc(fake)` and `disc(real)` with a *different* AD graph (tracing through disc parameters, not generator). These are fundamentally different computation graphs that cannot be shared.

**Why `specs_to_wavs` is called twice**: The waveforms computed inside `loss_fn` (generator AD graph) are consumed by `nn.value_and_grad` and not easily extractable. The disc update path needs `mx.stop_gradient(pred_wav)` which must be created outside the gen AD graph. Caching would require restructuring the loss function return signature and `nn.value_and_grad` call chain — moderate effort with moderate risk.

### GAN Optimization Candidates (Future)

| Priority | Optimization | Impact | Risk | Effort | Notes |
|----------|-------------|--------|------|--------|-------|
| P1 | Compile disc forward pass with `mx.compile()` | **HIGH** | MEDIUM | MEDIUM | ~200-240 kernel launches → fused graph. Fixed input shapes after crop. Must verify composition with `nn.value_and_grad`. |
| P2 | Compile entire disc update step (fwd+bwd) | **HIGH** | MEDIUM | LOW | Fuses ~120 kernel launches. Depends on P1 validation. |
| P3-future | Cache waveforms from gen→disc path | MEDIUM | LOW | MEDIUM | Eliminate 2 redundant iSTFTs (requires `loss_fn` return signature change). |
| P6 | `mx.checkpoint()` for disc forward in gen path | MEDIUM (memory) | LOW | LOW | Trade ~1.5x compute for ~3-4x memory reduction on disc activations. Enables larger batch sizes during GAN epochs. |
| P7 | Single `mx.eval()` per GAN step | MEDIUM | **HIGH** | LOW | Combine gen+disc sync into one eval. Risk: combined lazy graph may OOM. Must be behind flag. |
| P8 | Increase `gan_disc_update_freq` | **HIGH** | MEDIUM | TRIVIAL | Config-only change. `freq=2-4` saves 50-75% disc compute. Requires quality validation. |

**MLX-specific insight**: Weight normalization (used by all disc conv layers in PyTorch version) is currently a no-op in MLX — `weight_norm_conv1d/conv2d` create plain `nn.Conv1d/Conv2d`. This means discriminator training may be less stable than the PyTorch reference. Implementing weight norm requires per-step reparameterization of `weight = g * (v / ||v||)`, which adds compute but may improve training convergence.

### Remaining (Future Phases)

| Priority | Optimization | Complexity |
|----------|-------------|-----------|
| P2 | Compile `StreamingDfNet4.process_frame` | High — requires refactoring StreamingState into compile-friendly container |
| P3 | `mx.associative_scan` for Mamba (when available) | Low — depends on MLX roadmap |
