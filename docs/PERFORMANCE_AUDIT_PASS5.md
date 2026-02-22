# Performance Audit Pass 5: DeepFilterNet/df_mlx/

**Date**: 2026-02-18
**Scope**: `DeepFilterNet/df_mlx/` — all MLX training and inference hot paths
**Methodology**: Full manual review of 15 core source files (~15,600 lines)
**Verdict**: **No actionable new findings.** The codebase is release-ready from a performance standpoint.

---

## 1. Executive Summary

Pass 5 is a diminishing-returns sweep. All files were re-read with fresh eyes looking
for patterns that prior passes might have missed: duplicate computation across call
boundaries, hidden sync barriers, uncached per-call allocations, and opportunities for
algorithmic improvements.

**Result**: Zero new P0 or P1 findings. Three P3 (informational) observations documented
for completeness. All prior optimizations (Passes 1–4) remain correctly in place.

## 2. Files Reviewed

| File | Lines | Hot-Path? | Verdict |
|------|-------|-----------|---------|
| `loss.py` | 1041 | Yes (compiled) | Clean — `_ZERO` caching, lazy returns, running-sum |
| `training_losses.py` | 977 | Yes (compiled) | Clean — dtype guards, cast-once pattern |
| `train_dynamic.py` | 4443 | Yes (compiled + eager) | Clean — cached weight scalars, strategic sync points |
| `ops.py` | 627 | Yes (compiled) | Clean — `lru_cache` windows, vectorized OLA |
| `model.py` | 1992 | Yes (compiled) | Clean — `_erb_fb_T` cached, no per-call allocs |
| `modules.py` | 1294 | Yes (compiled) | Clean — mask cached by `(seq_len, dtype)` |
| `discriminator.py` | 686 | Yes (compiled GAN) | Clean — `return_features=False` for disc update |
| `mamba.py` | 606 | Yes (compiled) | Clean — parallel scan is O(log L) Python ops, traced once |
| `kernels.py` | 797 | Yes (Metal) | Clean — 4 custom kernels with VJPs |
| `grad_utils.py` | 60 | Yes (eager) | Clean — `_ZERO`/`_ONE` cached |
| `training_ops.py` | 227 | Yes (eager) | Clean — batched finite-check, single sync |
| `training_waveform.py` | 136 | Yes (compiled) | Clean — `_ZERO` caching |
| `dynamic_dataset.py` | 1728 | CPU worker threads | Clean — pre-allocated numpy buffers |
| `dnsmos_proxy.py` | 476 | Not used in training | N/A |
| `enhance.py` | 1064 | Inference only | Clean — appropriate `mx.eval` placement |

## 3. Prior Optimization Status (Verified In Place)

| ID | Optimization | Status |
|----|-------------|--------|
| PERF-P0-001–005 | dtype guards, cast-once, `_erb_fb_T` cache | ✅ Active |
| PERF-P1-001–005 | Mask cache, lazy CombinedLoss, si_sdr guards | ✅ Active |
| PERF-P2-003 | Fused post-filter Metal kernel | ✅ Active |
| GAN-P1 | Compiled disc inference (`_compiled_disc_infer`) | ✅ Active |
| GAN-P2 | Compiled disc update (`_compiled_disc_update_step`) | ✅ Active |
| GAN-P3 | `return_features=False` in disc update | ✅ Active |
| GAN-P4 | Running-sum loss accumulation | ✅ Active |
| GAN-P5 | Score-only disc forward API | ✅ Active |
| Pass 4 | Conv1d dilation/groups, O(1) scheduler resume, cached weight scalars, running-sum GAN losses | ✅ Active |

## 4. New Findings (Informational Only)

### PERF-5.1 (P3): Edge-case `mx.array(0.0)` in training_losses.py

- **File**: `training_losses.py` lines 487, 819
- **Pattern**: `smooth_loss = mx.array(0.0)` in `else` branch when `out_log.shape[1] <= 1`
- **Impact**: None — branch only taken for single-frame inputs (extremely rare during training). Within `mx.compile`, this is traced once regardless.
- **Action**: None required. Could trivially use a module-level `_SMOOTH_ZERO = mx.array(0.0)` but benefit is unmeasurable.

### PERF-5.2 (P3): Uncached mel filterbank transpose in dnsmos_proxy.py

- **File**: `dnsmos_proxy.py` line 183
- **Pattern**: `mx.transpose(self._mel_fb)` in MelSpectrogram non-Metal fallback path
- **Impact**: Zero — DNSMOS is not imported by `train_dynamic.py` and the Metal kernel path bypasses this code entirely on Apple Silicon.
- **Action**: None required.

### PERF-5.3 (P3): Utility disc/gen loss functions use `float()` sync barriers

- **File**: `discriminator.py` lines 642–680 (`compute_discriminator_loss`, `compute_generator_loss`)
- **Pattern**: Return dicts contain `float(loss)` values
- **Impact**: Zero — these utility functions are NOT used by the main training loop. `train_dynamic.py` has its own optimized disc/gen loss computation.
- **Action**: None required for training. Could be cleaned up if these utilities are used elsewhere.

## 5. Architecture Observations

The training pipeline achieves near-optimal MLX utilization:

1. **`mx.compile` coverage**: Generator forward+backward, discriminator update, and discriminator inference are all compiled. The only eager-path code runs during gradient accumulation windows.
2. **Sync point discipline**: A single `mx.eval` per `eval_frequency` batches, with `mx.clear_cache()` between generator and discriminator graphs to bound peak memory.
3. **Metal kernel coverage**: DfOp, iSTFT overlap-add, mel-power-log, and post-filter all have custom Metal kernels with proper VJPs for training.
4. **Data pipeline isolation**: All NumPy/SciPy work runs in CPU worker threads via `PrefetchDataLoader`. The only CPU→GPU transfer is the `mx.array()` conversion at batch boundary.

## 6. Remaining Optimization Backlog (from prior passes, unchanged)

| Priority | Item | Notes |
|----------|------|-------|
| P2 | Compile streaming inference | Complex — requires state management with `mx.compile` |
| P3 | `mx.associative_scan` for Mamba | Depends on MLX framework roadmap |
| P3-future | Cache waveforms from gen→disc path | Moderate effort, moderate risk |
| Config | Increase `gan_disc_update_freq` | Training hyperparameter, not code change |

## 7. Recommendation

**Pass 5 concludes the performance audit series.** The hot-path code in `df_mlx/` has been
systematically reviewed across 5 passes covering:
- Scalar allocation caching
- dtype cast elimination
- Sync barrier removal
- Metal kernel fusion
- GAN pipeline optimization
- Compile boundary analysis
- Data pipeline efficiency

Further performance gains require either:
1. MLX framework improvements (e.g., `mx.associative_scan`)
2. Algorithmic changes to the training procedure (e.g., disc update frequency)
3. Profiling-driven optimization on real training runs to identify actual bottlenecks vs. theoretical ones

The benchmark data from Pass 1–4 confirms that MLX's lazy evaluation and JIT compilation
absorb most micro-optimizations — the measurable wins came exclusively from Metal kernels
and compile boundary expansion.
