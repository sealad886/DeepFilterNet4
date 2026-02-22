# Pass 7 Error Audit Report

**Branch:** `feat/df_mlx-mlx-mega-optimization`  
**Baseline:** 977 passed, 11 skipped  
**Scope:** `DeepFilterNet/df_mlx/` — training loop, checkpoint system, loss computation, gradient ops, signal handling, CLI validation  
**Commit:** `d7b85d9`

## Findings Summary

| ID | Sev | Component | Status |
|----|-----|-----------|--------|
| ERR-7.1 | P1 | training_checkpoints.py | **FIXED** |
| ERR-7.2 | P1 | train_dynamic.py (validation) | **FIXED** |
| ERR-7.3 | P1 | train_dynamic.py (disc update) | **FIXED** |

## Detailed Findings

### ERR-7.1 — Checkpoint state file loaded without UTF-8 encoding

- **Severity:** P1
- **Component:** `training_checkpoints.py:706`
- **Evidence:** `save_checkpoint` writes state JSON with `encoding="utf-8"` (line 606), but `load_checkpoint` reads with `with open(state_path) as f:` — no encoding specified.
- **Why it fails:** On systems with non-UTF-8 default locale (some Linux containers, CI environments), reading a UTF-8–encoded file without explicit encoding causes `UnicodeDecodeError`, preventing checkpoint resume.
- **Fix:** Added `encoding="utf-8"` to the `open()` call in `load_checkpoint`.
- **Regression risk:** None — adds explicit encoding matches the save path.

### ERR-7.2 — Unprotected ablation metrics write crashes validation

- **Severity:** P1
- **Component:** `train_dynamic.py:2583–2584` (inside `run_validation`)
- **Evidence:** `with open(ablation_path, "a", encoding="utf-8") as f:` writes ablation metrics. No exception handling.
- **Why it fails:** If this write fails (disk full, permissions, I/O error), the unhandled `OSError` propagates out of `run_validation`, crashing the training loop. Ablation metrics are non-critical diagnostics — their failure should never abort training.
- **Fix:** Wrapped in `try/except OSError` with a `tqdm.write` warning.
- **Regression risk:** None — failure now degrades gracefully to a warning.

### ERR-7.3 — Discriminator eager update lacks NaN gradient guard

- **Severity:** P1
- **Component:** `train_dynamic.py:3355–3362` (disc update in eager training loop)
- **Evidence:** Generator eager update (line 3249) checks `if _tree_all_finite(final_grads)` before calling `optimizer.update()`. Discriminator eager update does NOT have this check.
- **Why it fails:** When `gan_disc_grad_clip=0` (clipping disabled), NaN gradients pass directly to `disc_optimizer.update()`, corrupting discriminator parameters. Once disc weights are NaN, all subsequent disc outputs are NaN, cascading into generator GAN loss. Default `gan_disc_grad_clip=1.0` masks this via `clip_grad_norm_tree` (which zeroes non-finite grads), but the guard should be explicit.
- **Fix:** Added `_tree_all_finite(disc_grads)` check with skip-and-warn pattern, matching the generator's guard.
- **Regression risk:** None — skipping a disc update on NaN grads is strictly safer than applying them.

## Areas Audited (No Issues Found)

| Area | Files | Verdict |
|------|-------|---------|
| Checkpoint save atomicity | training_checkpoints.py | SAFE — tmp files + rename + fsync + cleanup on error |
| Checkpoint validation | training_checkpoints.py | SAFE — monotonicity checks, size checks, marker validation |
| Signal handling (SIGINT) | training_signals.py | SAFE — double-SIGINT exits, single-SIGINT saves then raises |
| Gradient accumulation | training_ops.py | SAFE — proper tree addition, scale caching, finite checks |
| `clip_grad_norm_tree` | grad_utils.py | SAFE — zeros all grads when norm is non-finite |
| LR scheduler resume | lr.py | SAFE — analytical fast-forward, max() guards on divisors |
| VAD loss computation | training_losses.py | SAFE — `_MIN_VARIANCE` floor, logit clamping, eps on all divisions |
| Awesome/pipeline awesome losses | training_losses.py | SAFE — FP32 casts at entry, stop_gradient on gates, eps, clamps |
| Proxy gate computation | training_losses.py | SAFE — single-frame edge cases handled, clip bounds |
| Spectral losses | loss.py | SAFE — complex_norm uses eps, gamma compression avoids arctan2 |
| SI-SDR | loss.py | SAFE — eps on denominator, mean removal |
| Discriminator model | discriminator.py | SAFE — standard Conv1d/Conv2d architecture |
| CLI argument validation | training_cli.py | SAFE — type coercion, mutual exclusion checks |
| INI config adapter | train_dynamic_config.py | SAFE — unused-key warnings, getint/getfloat with section checks |
| Compiled step boundaries | train_dynamic.py | SAFE — shape assertions before compiled entry |
| Waveform processing | training_waveform.py | SAFE — FP32 stabilization, crop alignment |
| Training session API | training_session.py | SAFE — kwarg validation against `_TRAIN_KWARGS` |

## Residual Risks

1. **Compiled disc update path** — `_compiled_disc_update_step` also lacks a finite guard, but inside `mx.compile` we cannot use `_tree_all_finite` (requires sync barrier). Relies on `clip_grad_norm_tree` zeroing non-finite grads when `max_disc_grad_norm > 0` (default 1.0). This path is explicitly labeled "experimental".

2. **TODOs in non-training code** — 5 TODOs found in `enhance.py` (2), `convert.py` (2), `quantization.py` (1). All are in inference/utility paths, not training. No correctness impact.

## Test Results

```
977 passed, 11 skipped, 20 warnings in 37.73s
```

Baseline maintained — no regressions.
