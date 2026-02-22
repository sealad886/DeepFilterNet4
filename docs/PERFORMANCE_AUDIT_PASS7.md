# Performance Audit — Pass 7

**Date:** 2025-01-28
**Branch:** `feat/df_mlx-mlx-mega-optimization`
**Scope:** `DeepFilterNet/df_mlx/` — hot-path loss computation, gradient utilities, GAN losses
**Baseline:** 977 passed, 11 skipped (pytest)
**Prior passes:** Pass 4 (cached weight scalars, O(1) scheduler, mx.stack→running sum in GAN), Pass 6 (redundant compiled graph ops, GAN accumulator inflation fix)

## Summary

5 optimizations implemented across 3 files. All target the compiled computation graph or gradient utility hot path. Zero test regressions.

## Findings

### PERF-7.1 — Eliminate redundant FP32 dtype casts (P1)

| Field | Value |
|-------|-------|
| **Component** | `training_losses.py` — inner loss functions |
| **Evidence** | 30 `.astype(mx.float32)` calls; 20+ occur inside functions already called with pre-cast FP32 inputs from `_compute_awesome_losses` / `_compute_pipeline_awesome_losses` |
| **Change** | Added `_assume_float32: bool = False` parameter to 8 inner functions. Callers that cast at entry pass `_assume_float32=True`, skipping 20+ conditional dtype checks |
| **Affected functions** | `_log1p_mag`, `_compute_musicness`, `_compute_proxy_gates`, `_compute_vad_probs`, `_compute_speech_band_logmag_loss`, `_compute_pitch_stability`, `_compute_harmonic_ratio`, `_compute_improved_musicness` |
| **Risk** | Low — parameter defaults preserve existing behavior for standalone callers |
| **Verification** | 977 passed, 11 skipped |
| **Commit** | `9262d8d` |

### PERF-7.2 — Deduplicate z-scored energy computation (P1)

| Field | Value |
|-------|-------|
| **Component** | `training_losses.py` — `_compute_vad_reg_loss` path |
| **Evidence** | `_compute_vad_reg_loss` calls both `_compute_vad_probs` and `_compute_proxy_gates`, each independently computing: `clean_power → clean_band → log_clean → mu → variance → sigma → z_ref → p_ref` (~12 MLX ops duplicated) |
| **Change** | Extracted `_z_score_clean_energy()` helper. Both `_compute_vad_probs` and `_compute_proxy_gates` accept `_precomputed_z` tuple. `_compute_vad_reg_loss` calls the helper once and shares the result |
| **Risk** | Low — optional parameter, defaults to full computation |
| **Verification** | 977 passed, 11 skipped |
| **Commit** | `22d0313` |

### PERF-7.3 — SKIPPED (code duplication, not runtime duplication)

`_compute_pipeline_awesome_losses` reimplements ~80 lines of proxy gate logic from `_compute_proxy_gates`. However, these two functions are **mutually exclusive at runtime** (training config selects one). No performance impact — purely a maintenance concern (covered by duplication audits).

### PERF-7.4 — Vectorize FeatureMatchingLoss accumulation (P2)

| Field | Value |
|-------|-------|
| **Component** | `loss.py` — `FeatureMatchingLoss.__call__` |
| **Evidence** | Python loop with `total = total + mx.mean(mx.abs(...))` creates sequential scalar chain dependency in compiled graph, preventing parallel scheduling of independent `mx.mean(mx.abs(...))` operations |
| **Change** | Collect per-layer means into list, then `mx.mean(mx.stack(means))`. Semantically identical (`mean of means` = `sum / count`) |
| **Risk** | None — mathematically equivalent |
| **Verification** | 977 passed, 11 skipped |
| **Commit** | `1485f74` |

### PERF-7.5 — Vectorize discriminator/generator loss accumulation (P2)

| Field | Value |
|-------|-------|
| **Component** | `loss.py` — `discriminator_loss`, `generator_loss` |
| **Evidence** | Same sequential scalar accumulation pattern as PERF-7.4 |
| **Change** | Collect per-discriminator hinge losses into list, then `mx.mean(mx.stack(...))`. Early-return for empty input lists |
| **Risk** | None — mathematically equivalent |
| **Verification** | 977 passed, 11 skipped |
| **Commit** | `1485f74` |

### PERF-7.6 — Replace mx.stack with mx.concatenate in _tree_all_finite (P2)

| Field | Value |
|-------|-------|
| **Component** | `training_ops.py` — `_tree_all_finite` |
| **Evidence** | `mx.stack(checks)` on a list of scalar boolean arrays incurs shape-inference overhead. `mx.concatenate` on pre-reshaped 1-element arrays avoids this |
| **Change** | `mx.all(mx.isfinite(v)).reshape(1)` per leaf, then `mx.all(mx.concatenate(checks))` |
| **Risk** | None — semantically identical reduction |
| **Verification** | 977 passed, 11 skipped |
| **Commit** | `1485f74` |

## Files Changed

| File | Lines changed | Optimizations |
|------|--------------|---------------|
| `training_losses.py` | +139 / -68 | PERF-7.1, PERF-7.2 |
| `loss.py` | +23 / -21 | PERF-7.4, PERF-7.5 |
| `training_ops.py` | +4 / -3 | PERF-7.6 |

## Not Flagged (already optimized in prior passes)

- Cached weight scalars in loss calculations (Pass 4)
- Running-sum instead of list+mx.stack in GAN loss accumulators (Pass 4)
- O(1) cosine scheduler resume (Pass 4)
- Redundant ops in compiled training graph (Pass 6)
- GAN loss accumulator inflation fix (Pass 6)

## Remaining Backlog

- `_compute_pipeline_awesome_losses` inline proxy gate code duplication (~80 lines) — code quality not performance (PERF-7.3 SKIPPED)
- No remaining hot-path performance issues identified in model forward pass, STFT ops, or training loop structure
