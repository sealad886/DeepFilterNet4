# Incident Report: MLX Training Control Counter Mismatch

**Date:** 2026-02-13  
**Severity:** HIGH  
**Status:** RESOLVED

## Summary

`df_mlx/train_dynamic.py` mixed two different units in epoch control:

- progress-bar total derived from optimizer steps (`steps_per_epoch`)
- loop iteration performed over micro-batches

This made epoch progress appear to run past expected totals and created ambiguity in resume behavior. In addition, in-progress checkpoint `batch_idx` semantics were inconsistent (index vs count), and model/data checkpoint mismatch handling could silently continue.

## Impact

- Misleading tqdm progress during training epochs.
- Off-by-one risk for interrupted resume.
- Potential silent divergence when model checkpoint and data checkpoint were out of sync.

## Root cause

1. **Unit mismatch**: epoch total used optimizer-step units while loop consumed micro-batches.
2. **Checkpoint ambiguity**: `batch_idx` interpreted inconsistently across save/load/resume paths.
3. **Weak reconciliation**: model/data checkpoint mismatches were sometimes logged and ignored instead of failing loudly.

## Fixes implemented

1. **Canonical counters**
   - Epoch progress now uses micro-batches.
   - `global_step` remains optimizer-step based.
2. **Deterministic loop boundary**
   - Training iterates over `enumerate(islice(data_iterator, train_total))` with `train_total` in micro-batches.
3. **Checkpoint metadata normalization**
   - Added explicit fields: `counter_semantics_version`, `micro_batches_completed`, `optimizer_steps_completed`.
   - Kept compatibility with legacy checkpoints via conversion.
4. **Strict resume reconciliation**
   - In-progress resume now requires exact `(epoch, micro_batch)` match between model and data checkpoints.
   - Mismatch raises a hard error with remediation guidance.
5. **Data stream resume API**
   - Added `MLXDataStream.set_resume_position(...)` to align stream position to model checkpoint deterministically.

## Verification

- Added regression tests in `DeepFilterNet/tests/test_train_control_semantics.py` covering:
  - legacy checkpoint conversion
  - v2 counter semantics
  - checkpoint metadata persistence
  - resume skip behavior
  - bounded iterator usage in training loop

## Prevention

- Added repository convention entry in `docs/CONVENTIONS.md`:
  - micro-batch vs optimizer-step semantics
  - checkpoint counter definition
  - strict model/data resume invariants
