# Errors Audit — Pass 10 (Final)

**Branch:** `feat/major_refactor_train_dynamic`  
**Commit (before fixes):** `9649760`  
**Commit (after fixes):** `9b2e822`  
**Date:** 2027-02-28  
**Scope:** All 14 `training_*.py` modules + `train_dynamic.py` in `DeepFilterNet/df_mlx/`  
**Test baseline:** 1007 passed, 11 skipped  
**Test after fixes:** 1007 passed, 11 skipped  

---

## Summary

| Severity | Count |
|----------|-------|
| P0 (blocker) | 0 |
| P1 (high) | 0 |
| P2 (medium) | 2 — **both fixed** |
| Observation | 5 |
| Test coverage gap | 2 |

**Verdict: RELEASE-READY.** No blockers or high-severity issues. Two medium-severity bugs fixed and verified. The 14-module decomposition is behaviorally correct and well-guarded against non-finite states.

---

## BLOCKERS

None.

---

## WARNINGS (P2) — Fixed

### W-1: `_nonfinite_loss_count` persists across `train()` invocations

- **ID:** ERR-10.1
- **Severity:** P2
- **Component:** `train_dynamic.py` (line ~428)
- **Evidence:** `train._nonfinite_loss_count` is stored as an attribute on the `train` function object. Python function objects persist across calls, so the counter carries over from a previous `train()` invocation in the same process.
- **Impact:** If a first training run accumulated, say, 45 non-finite losses, a second `train()` call (e.g., via `TrainingSession`) would abort after only 5 more non-finite losses—well before the 50-loss threshold intended for that run.
- **Fix:** Reset `train._nonfinite_loss_count = 0` at the top of `train()`.
- **Regression risk:** None. Counter behavior is now per-invocation as intended.

### W-2: `grad_norm` clobbered in compiled + grad-accumulation path

- **ID:** ERR-10.2
- **Severity:** P2
- **Component:** `train_dynamic.py` (line ~2163)
- **Evidence:** `grad_norm = float("nan")` was placed at the same indentation level as the `if grad_accumulation_steps > 1:` / `else:` branches, meaning it executed unconditionally after either branch. When `grad_accumulation_steps > 1`, a valid `grad_norm` was extracted from `float(grad_norm_arr)` but immediately overwritten with NaN.
- **Impact:** Progress bar and debug/profile logs show `grad_norm=nan` even when the grad-accumulation path successfully computes it. Display-only bug; model weights unaffected.
- **Fix:** Moved `grad_norm = float("nan")` inside the fully-compiled (non-accumulation) `else:` branch.
- **Regression risk:** None. Only changes which code path writes the sentinel NaN.

---

## OBSERVATIONS

### O-1: `loss_finite_arr` checks only the last micro-batch in accumulation window (eager path)

- **Component:** `train_dynamic.py` (~line 2193)
- **Detail:** In the eager path with `grad_accumulation_steps > 1`, `loss_finite_arr = mx.all(mx.isfinite(loss))` only reflects the current micro-batch's loss. A previous micro-batch in the accumulation window could have had a non-finite loss that goes unreported in the diagnostic message.
- **Mitigation:** `_tree_all_finite(final_grads)` catches accumulated non-finite gradients and skips the optimizer update, so model weights are safe. The gap is diagnostic only.
- **Action:** No fix needed. Could be improved in a future pass by tracking `any_nonfinite_in_window` across accumulation micro-batches.

### O-2: `_disc_crop_waveform` uses Python `random.randint` (non-deterministic under MLX seed)

- **Component:** `training_waveform.py` (line ~128)
- **Detail:** Discriminator waveform cropping uses Python stdlib `random.randint`, which is not controlled by `mx.random.seed` or NumPy seed.
- **Mitigation:** Discriminator cropping offset is intentionally random and does not need to be training-reproducible. The model's forward pass uses `mx.random` which is properly seeded.
- **Action:** No fix needed. Document if exact training reproducibility is ever required.

### O-3: Duplicate cached zero in `training_waveform.py`

- **Component:** `training_waveform.py` (line 28)
- **Detail:** `_ZERO = mx.array(0.0)` duplicates `SCALAR_ZERO` from `training_helpers.py`. Per established convention (documented in memory and `AUDIT_ERRORS_PASS9.md`), shared scalar constants should be imported from `training_helpers`.
- **Action:** Low priority. Can be consolidated in a future duplication audit pass.

### O-4: `_TQDM_KWARGS` uses `sys.stderr` reference captured at import time

- **Component:** `train_dynamic.py` (lines 214–222)
- **Detail:** `_TQDM_KWARGS["file"] = sys.stderr` captures the stderr reference at module import time. If stderr is redirected after import but before `train()` is called, tqdm output goes to the original stderr.
- **Mitigation:** This is standard Python behavior and is the correct pattern for tqdm. The `DFNET_TQDM` env var provides explicit override.
- **Action:** No fix needed.

### O-5: `best.safetensors` writes `last_completed_epoch=epoch` before epoch-end checkpoint

- **Component:** `train_dynamic.py` (line ~2634)
- **Detail:** When saving `best.safetensors` (after validation), `last_completed_epoch=epoch` is written to checkpoint metadata even though the epoch-end checkpoint hasn't been saved yet. A resume from `best.safetensors` after a crash between best-save and epoch-end-save would set `last_completed_epoch=epoch`.
- **Mitigation:** The `reconcile_resume` function in `training_checkpoints.py` validates checkpoint integrity during resume. The best checkpoint contains the full model state at the point of best validation, which is a valid resume point. The `epoch_completed = epoch_saved or best_saved` logic at the end of the epoch correctly handles this case.
- **Action:** No fix needed. The behavior is intentional and safe.

---

## TEST COVERAGE GAPS

### T-1: No dedicated unit tests for 11 of 14 extracted modules

- **Detail:** Only `training_helpers.py`, `training_session.py`, and `training_session_poc.py` have dedicated `test_training_*.py` files. The other 11 modules are covered indirectly:
  - `training_checkpoints`: covered by `test_checkpoint_*.py`, `test_train_control_semantics.py`
  - `training_losses`: covered by `test_loss_correctness.py`, `test_loss_audit_fixes.py`, `test_dynamic_loss_numerics.py`, `test_perf_optimizations.py`
  - `training_ops`: covered by `test_grad_utils.py`, `test_grad_nonfinite.py`
  - `training_cli`/`training_cli_main`: covered by `test_train_dynamic_cli_toml_parity.py`, `test_train_dynamic_help_defaults.py`
  - `training_signals`, `training_diagnostics`, `training_metrics`, `training_setup`, `training_waveform`, `training_validation`: indirect coverage only through integration tests
- **Action:** Consider adding focused unit tests for `training_metrics.py` (complex accumulator/sync logic) and `training_validation.py` (complex validation loop) in future passes.

### T-2: No test for `_nonfinite_loss_count` reset behavior

- **Detail:** The newly fixed ERR-10.1 has no regression test. A test would call `train()` twice with controlled non-finite injections to verify the counter resets between calls.
- **Action:** Low priority since the fix is trivial (single assignment). Would require significant test infrastructure (mock dataset, model) to test in isolation.

---

## MODULE-BY-MODULE AUDIT SUMMARY

| Module | Lines | Findings | Status |
|--------|-------|----------|--------|
| `training_signals.py` | 204 | Clean. Signal handler is safe. | ✅ |
| `training_helpers.py` | 211 | Clean. SCALAR_ZERO pattern correct. | ✅ |
| `training_waveform.py` | 143 | O-2 (random.randint), O-3 (dup zero) | ✅ |
| `training_ops.py` | 245 | Clean. clip_grad_norm zeros non-finite. | ✅ |
| `training_diagnostics.py` | 261 | Clean. | ✅ |
| `training_session.py` | 315 | Clean. | ✅ |
| `training_cli.py` | 355 | Clean. | ✅ |
| `training_metrics.py` | 779 | Clean. Batched sync barriers correct. | ✅ |
| `training_cli_main.py` | 965 | Clean. Config precedence correct. | ✅ |
| `training_setup.py` | 1073 | Clean. | ✅ |
| `training_checkpoints.py` | 1098 | Clean. Atomic save pattern correct. | ✅ |
| `training_losses.py` | 1099 | Clean. FP32 casting consistent. | ✅ |
| `training_validation.py` | 780 | Clean. | ✅ |
| `train_dynamic.py` | 2845 | **ERR-10.1** (fixed), **ERR-10.2** (fixed), O-1, O-4, O-5 | ✅ |

**Total:** 10,372 lines audited across 14 modules.

---

## RELEASE-READINESS

The `feat/major_refactor_train_dynamic` branch is **release-ready** with the two P2 fixes applied:

1. **Correctness:** No P0/P1 issues. Model weights are protected against non-finite states by `clip_grad_norm_tree` (zeros non-finite grads) and `_tree_all_finite` (skips optimizer update). Checkpoint save/load uses atomic rename pattern with validation.

2. **Safety:** Signal handler saves both model and data checkpoints before exit. Pipeline stage progression is monotonically forward. Train mode transitions are one-way (COMPILED→EAGER) to prevent determinism drift.

3. **Behavioral equivalence:** The `test_train_dynamic_reexports.py` test verifies all public symbols from 8 training_* modules are re-exported from `train_dynamic.py` and are the same objects (not copies).

4. **Test confidence:** 1007/1007 tests pass, 11 skipped (hardware-specific markers).
