# Performance Audit Pass 6: DeepFilterNet/df_mlx/

**Date**: 2026-02-19
**Scope**: `DeepFilterNet/df_mlx/` — all MLX training and inference hot paths
**Methodology**: Full manual review of 15 core source files (~15,600 lines), cross-function data-flow analysis for redundant computation in compiled graphs
**Verdict**: **Four optimizations implemented.** Three eliminate redundant computation in compiled training graphs; one adds defensive dtype guards for tracing consistency.

---

## 1. Executive Summary

Pass 6 focuses on **cross-function data-flow analysis** — a dimension prior passes did not systematically cover. Prior passes optimized individual functions (caching, dtype guards, scalar allocation), but redundant computation across function call boundaries remained.

**Result**: 2 P1 and 2 P2 findings implemented. All correctness-preserving (977 tests pass, 11 skipped).

### Impact Summary

| ID | Priority | Component | Optimization | Tensors Saved |
|----|----------|-----------|-------------|---------------|
| PERF-6.1 | P1 | `_compute_pipeline_awesome_losses` | Eliminate duplicate noise subtraction | 2 × (B,T,F) subtractions |
| PERF-6.2 | P1 | `_compute_awesome_losses` → `_compute_proxy_gates` | Pass pre-computed noise arrays across function boundary | 2 × (B,T,F) subtractions |
| PERF-6.3 | P2 | `spectral_loss` | Add conditional dtype guards | 4 tracing-level cast avoidances |
| PERF-6.4 | P2 | Module constants | Hoist `_MIN_VARIANCE` to module level | 2 local variable eliminations |

---

## 2. Findings Detail

### PERF-6.1 (P1): Duplicate noise subtraction in `_compute_pipeline_awesome_losses`

- **File**: `training_losses.py`
- **Bottleneck evidence**: Lines ~705-706 compute `noise_real = noisy_real_f32 - clean_real_f32` and `noise_imag = noisy_imag_f32 - clean_imag_f32` for log-magnitude computation. Lines ~733-734 recompute the identical values as `noise_real_f32` and `noise_imag_f32` for power computation. Both operate on full (B, T, F) tensors (e.g., 4×150×481 = ~289K floats).
- **Change**: Reuse `noise_real`/`noise_imag` for `noise_power` computation. Removed 2 redundant subtraction ops.
- **Before**: 4 subtraction ops on (B,T,F) tensors in compiled graph
- **After**: 2 subtraction ops on (B,T,F) tensors in compiled graph
- **Risk**: None — pure variable reuse within the same function scope
- **Correctness**: Identical numerical output (same operands, same order)

### PERF-6.2 (P1): Cross-boundary noise duplication in awesome loss → proxy gates

- **File**: `training_losses.py`
- **Bottleneck evidence**: `_compute_awesome_losses` computes `noise_real = noisy_real_f32 - clean_real_f32` (line ~445) then calls `_compute_proxy_gates`, which independently recomputes the same subtraction (line ~330). Inside mx.compile, this creates duplicate graph nodes that MLX's compiler may not eliminate via CSE.
- **Change**: Added optional `noise_real`/`noise_imag` parameters to `_compute_proxy_gates` (keyword-only, default `None`). When provided, skip the subtraction. `_compute_awesome_losses` now passes its pre-computed values.
- **Before**: `_compute_proxy_gates` always recomputes noise = noisy - clean
- **After**: Caller can pass pre-computed noise; proxy_gates skips the subtraction
- **Risk**: Low — backward-compatible (keyword args default to `None`; existing callers unaffected)
- **Correctness**: Identical numerical output (same operands passed explicitly)

### PERF-6.3 (P2): Unconditional FP32 casts in `spectral_loss`

- **File**: `train.py` (lines ~370-373)
- **Bottleneck evidence**: 4 unconditional `.astype(mx.float32)` calls without dtype check. All other loss functions in the codebase use the pattern `if x.dtype != mx.float32: x = x.astype(mx.float32)`. While MLX may optimize away same-dtype casts, the conditional guard is more explicit for tracing and consistent with codebase conventions.
- **Change**: Wrapped each cast in `if x.dtype != mx.float32` guard
- **Before**: 4 unconditional `.astype(mx.float32)` calls
- **After**: 4 conditional casts (no-op when dtype already matches)
- **Risk**: None — guards are pure defensive checks
- **Correctness**: Identical — same cast is applied when dtype differs

### PERF-6.4 (P2): Module-level `_MIN_VARIANCE` constant

- **File**: `training_losses.py`
- **Bottleneck evidence**: `_MIN_VARIANCE = 1e-4` defined as a local variable inside `_compute_vad_probs` and `_compute_pipeline_awesome_losses`. Redefined on every call during tracing. Module-level constants are standard practice throughout the file (see `_EPS`, `_VAD_LOGIT_CLAMP`, etc.).
- **Change**: Moved to module-level constant alongside `_EPS`
- **Before**: 2 local variable definitions per trace
- **After**: Single module-level constant
- **Risk**: None
- **Correctness**: Identical value

---

## 3. Considered but Not Implemented

### PERF-6.C1 (Deferred): `_compute_vad_reg_loss` z-scoring deduplication

- **Pattern**: `_compute_vad_reg_loss` calls both `_compute_vad_probs` and `_compute_proxy_gates`, each independently computing the full z-scored log-energy pipeline (clean_power → clean_band → log_clean → mu → variance → sigma → z_ref → p_ref). This duplicates ~8 operations on (B, T) tensors.
- **Why deferred**: Fixing requires either (a) extracting a shared z-scoring helper and refactoring both callers, or (b) changing `_compute_proxy_gates` return signature to expose z-scoring intermediates. Both approaches touch 3+ functions across call boundaries. The (B, T) tensors are small (4×150 = 600 elements) compared to the (B, T, F) tensors in PERF-6.1/6.2 (4×150×481 = 289K elements), making the impact ~480× smaller.
- **Estimated savings**: ~8 ops on 600-element tensors per step when `use_vad_train_reg` is active
- **Recommendation**: Implement in a dedicated refactoring pass if profiling identifies vad_reg_loss as a bottleneck

### PERF-6.C2 (Deferred): Return individual loss scalars from compiled `loss_fn`

- **Pattern**: The compiled loss function returns only `total_loss` + auxiliary outputs. At each sync point (every `eval_frequency` batches), individual loss components are recomputed eagerly for metrics logging. This duplicates the full forward pass of awesome/pipeline/VAD losses.
- **Why deferred**: Changing the `loss_fn` return signature affects the compiled graph structure, the `nn.value_and_grad` unpacking, the GAN variant, and all downstream metric code. High refactoring risk. The overhead is amortized (1/eval_frequency of steps) and the eager forward recomputation is cheap relative to the compiled forward+backward.
- **Estimated savings**: ~20% of sync-step wall-clock time, which is 1/eval_frequency of total training time
- **Recommendation**: Consider only if profiling shows sync-step metrics dominate wall-clock

### PERF-6.C3 (Not actionable): `_log1p_mag` redundant dtype guards

- **Pattern**: Called from `_compute_awesome_losses` and `_compute_pipeline_awesome_losses` where inputs are already FP32-cast. The dtype guards are Python-level checks that execute only during tracing (once per compilation).
- **Why not actionable**: The guards cost ~0 GPU time. Removing them would save microseconds during compilation but would make the function unsafe for standalone use. Not worth the API fragility.

### PERF-6.C4 (Not actionable): Data pipeline `compute_stft` optimization

- **Pattern**: `compute_stft` in `feature_ops.py` uses `np.fft.rfft`. Could potentially use `scipy.fft.rfft` or `pyfftw` for multi-threaded FFT.
- **Why not actionable**: The STFTs run in CPU worker threads via `PrefetchDataLoader`. Each worker processes one sample independently. Adding per-FFT parallelism would cause thread contention with the worker pool. NumPy 1.17+ uses pocketfft which handles batch FFTs efficiently. No evidence that data loading is the bottleneck.

---

## 4. Prior Optimization Status (Verified)

All optimizations from Passes 1-5 remain correctly in place (spot-checked during code review).

| Pass | Key Optimizations | Status |
|------|------------------|--------|
| 1-3 | dtype guards, cast-once, `_erb_fb_T` cache, mask cache | ✅ Active |
| 4 | Conv1d dilation/groups, O(1) scheduler resume, cached weight scalars, running-sum GAN losses | ✅ Active |
| 5 | Verified all clean; no actionable findings | ✅ Confirmed |

---

## 5. Architecture Observations

The training pipeline has reached a mature optimization level:

1. **Compiled graph coverage**: Generator forward+backward, discriminator update, and discriminator inference are all compiled. Only metric-logging code runs eagerly.
2. **Cross-boundary redundancy**: This pass found the last remaining cross-boundary redundancies in the loss computation functions. The compiled graph now has minimal duplicate operations.
3. **Data pipeline**: Pre-allocated numpy buffers (`_assemble_batch`), mmap-backed sharded cache, threadpool prefetching — no further CPU-side optimizations without moving to Rust extensions.
4. **Metal kernel coverage**: DfOp, iSTFT, mel-power-log, and post-filter all have custom kernels with VJPs.

---

## 6. Remaining Optimization Backlog (unchanged from Pass 5)

| Priority | Item | Notes |
|----------|------|-------|
| P2 | Compile streaming inference | Complex — requires state management with `mx.compile` |
| P3 | `mx.associative_scan` for Mamba | Depends on MLX framework roadmap |
| P3-future | Cache waveforms from gen→disc path | Moderate effort, moderate risk |
| Config | Increase `gan_disc_update_freq` | Training hyperparameter, not code change |
| Deferred | PERF-6.C1 (vad_reg z-scoring dedup) | Low impact, high refactoring risk |
| Deferred | PERF-6.C2 (scalar returns from loss_fn) | Moderate impact, high refactoring risk |

---

## 7. Verification

- **Tests**: 977 passed, 11 skipped (identical to pre-change baseline)
- **Files modified**: `training_losses.py`, `train.py`
- **Formatting**: black (line-length=100) + isort applied
- **No semantic changes**: All optimizations are pure redundancy elimination or defensive guard additions

---

## 8. Recommendation

Pass 6 addresses the last remaining cross-function redundancies in the compiled loss computation graph. Further performance gains require either:

1. **Profiling-driven optimization** on real training runs to identify actual wall-clock bottlenecks
2. **MLX framework improvements** (e.g., better CSE in `mx.compile`, `mx.associative_scan`)
3. **Algorithmic changes** to the training procedure (hyperparameter tuning, not code optimization)
4. **Rust/C extensions** for the CPU data pipeline (if data loading becomes the bottleneck)

The code-level audit series (Passes 1-6) has systematically covered all hot-path functions, cross-boundary data flow, compilation boundaries, sync barriers, memory allocation patterns, and Metal kernel opportunities. The remaining items in the optimization backlog are either framework-dependent or require profiling evidence to justify their complexity.
