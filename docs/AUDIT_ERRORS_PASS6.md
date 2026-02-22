# Correctness/Errors Audit — Pass 6

**Scope:** `DeepFilterNet/df_mlx/` (critical training, loss, model, inference, and dataset paths)  
**Date:** 2026-02-22  
**Auditor:** Checker agent (Pass 6)  
**Status:** P1 FIXED, P2 DOCUMENTED

---

## Methodology

Systematic line-by-line reading of all high-criticality files:
- `train_dynamic.py` (4245 lines — complete)
- `training_losses.py` (977 lines — complete)
- `training_checkpoints.py` (869 lines — complete)
- `training_waveform.py` (136 lines — complete)
- `loss.py` (941 lines — complete)
- `discriminator.py` (551 lines — complete)
- `dynamic_dataset.py` (1728 lines — complete)
- `run_config.py` (1363 lines — complete)
- `enhance.py` (1064 lines — complete)
- `model.py` (1992 lines — complete)

**Total lines audited:** ~13,866

Focus: loss/metric accumulation correctness, numerical precision, state machine bugs in
training loop, checkpoint resume fidelity, inference correctness, and any issues NOT
previously reported in Passes 1–5.

---

## Previously Fixed (NOT re-reported)

| Pass | Finding | Status |
|------|---------|--------|
| P2 | Atomic checkpoint writes | FIXED |
| P2 | float16 → float32 for loss computation | FIXED |
| P3 | ZeroDivisionError in metric averaging | FIXED |
| P3 | Overwrite guard in `save_audio` | FIXED |
| P3 | Temp file cleanup in checkpoints | FIXED |
| Audit-2025-01 | Mask saturation penalty inverted | FIXED |
| Audit-2025-01 | Sigma variance floor missing | FIXED |
| Audit-2025-01 | Single-frame edge cases | FIXED |
| Audit-2025-01 | Empty array mean | FIXED |
| P4 (859759c) | CosineScheduler O(N) replay on resume | FIXED |
| P4 (859759c) | Conv1d dilation/groups silently dropped | FIXED |
| P4 (4ac917e) | Cache weight scalars, eliminate mx.stack in GAN losses | FIXED |

---

## Still Open from Prior Passes (NOT re-flagged)

| Pass | Finding | Status | Notes |
|------|---------|--------|-------|
| P4-P1-001 | Interfering speaker same-file guard compares wrong indices | OPEN | `dynamic_dataset.py:1064` |
| P4-P1-002 | Signal handler performs non-async-signal-safe I/O | OPEN | `training_signals.py` |
| P4-P2-001 | Worker errors propagated only after queued batches consumed | OPEN | `dynamic_dataset.py:1239` |
| P4-P2-002 | Compiled grad accumulation unbounded lazy graph | OPEN | `train_dynamic.py:3205` |
| P4-P3-001 | Shard double-load TOCTOU race | OPEN | `dynamic_dataset.py` |
| P4-P3-002 | Dead code in ASRLoss | OPEN | `loss.py` |

---

## Investigated and Eliminated (NOT bugs)

### MLX `.at[].add()` in StreamingDfNet4 (model.py)

**Suspected issue:** `state_real = state_real.at[..., :-1, :].add(...)` was suspected to
be a PyTorch-style in-place op that silently fails in MLX's functional model.

**Verification:** Tested in terminal — MLX's `.at[].add()` correctly returns a *new array*
with the addition applied (like JAX), consistent with MLX's non-mutating semantics.
Not a bug.

---

## New Findings

### P6-P1-001: Loss accumulation multiplier uses wrong frequency during GAN epochs — FIXED

**Severity:** P1 (correctness — inflated loss metrics corrupt training decisions)  
**File:** [train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py)  
**Lines:** 3388–3390, 3487, 3501, 3518, 3521, 3566–3567, 3665–3668, 3790–3795, 3829
(19 accumulator lines total)

**Problem:**

During GAN epochs, `epoch_eval_frequency` (line 2757) caps the sync cadence to
`min(training.eval_frequency, gan.eval_frequency)`. With defaults
(`training.eval_frequency=10`, `gan.eval_frequency=2`), syncing happens every 2 batches
instead of every 10.

However, **all 19 loss accumulators** used `eval_frequency` (the uncapped value) as the
multiplier:

```python
# BEFORE (wrong during GAN epochs):
train_loss += loss_val * eval_frequency       # line 3388
train_gan_d_loss += gan_d_loss_val * eval_frequency  # line 3390
train_spec_loss += spec_loss_val * eval_frequency    # line 3487
# ... 16 more identical patterns
```

The accumulation pattern works by: `train_loss += loss_at_sync * steps_between_syncs`,
then averaging with `avg = train_loss / num_train_batches`. When the sync cadence is
`epoch_eval_frequency` but the multiplier is `eval_frequency`, the average is inflated by
`eval_frequency / epoch_eval_frequency`.

**Impact:**

With default settings during GAN epochs:
- All reported average losses (train, spec, MRSTFT, GAN-G, GAN-FM, VAD, speech,
  awesome, music suppression, mask saturation, VAD regularization) are inflated by **5x**
- The inflated `avg_train_loss` is stored in checkpoints (line 3897) and used for
  best-model selection (indirectly via validation, but the displayed training loss
  misleads debugging)
- The discriminator loss (`train_gan_d_loss`) was also inflated, with an additional
  mismatch: it divides by `train_gan_d_updates` (sync count) not `num_train_batches`,
  so the multiplier should have been 1, not `eval_frequency`

**Root cause:** The loss accumulation code was written for non-GAN training where
`epoch_eval_frequency == eval_frequency`. When GAN support added the eval_frequency
capping (via `gan.eval_frequency`), the accumulation multipliers were not updated.

**Fix applied:**

```python
# AFTER (correct):
train_loss += loss_val * epoch_eval_frequency          # Uses actual sync interval
train_gan_d_loss += gan_d_loss_val                      # No multiplier (divides by sync count)
train_spec_loss += spec_loss_val * epoch_eval_frequency
# ... all other accumulators likewise
```

All 18 accumulators that divide by `num_train_batches` changed from `* eval_frequency`
to `* epoch_eval_frequency`. The discriminator loss accumulator (which divides by
`train_gan_d_updates`) had its `* eval_frequency` multiplier removed entirely.

**Verification:**
- 977 tests pass, 11 skipped (unchanged from baseline)
- Black + isort formatting applied
- Non-GAN epochs are unaffected (`epoch_eval_frequency == eval_frequency`)

---

### P6-P2-001: `enhance_frame_compiled` is a misleading no-op placeholder

**Severity:** P2 (dead code — misleading but no runtime impact)  
**File:** [enhance.py](../DeepFilterNet/df_mlx/enhance.py#L390-L404)  
**Lines:** 390–404

**Problem:**

```python
@mx.compile
def enhance_frame_compiled(
    spec: Tuple[mx.array, mx.array],
    feat_erb: mx.array,
    feat_spec: mx.array,
    erb_mask: mx.array,
    df_out: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Compiled enhancement kernel (stateless)."""
    # This is a placeholder for future optimization
    return spec  # Returns input unchanged!
```

This function:
1. Is decorated with `@mx.compile` but performs no computation
2. Takes `erb_mask` and `df_out` parameters that are never used
3. Returns the input `spec` unchanged — a semantic no-op
4. Is not called anywhere in the codebase
5. Could mislead a developer into calling it for "compiled enhancement"

**Impact:** No runtime impact (dead code). Risk is developer confusion.

**Recommended fix:** Delete the function, or implement it if compiled inference is
intended. Not fixed in this pass.

---

## Audit Summary

| ID | Severity | Component | Status | Description |
|----|----------|-----------|--------|-------------|
| P6-P1-001 | P1 | train_dynamic.py | **FIXED** | Loss accumulators used wrong multiplier during GAN epochs (5x inflation with defaults) |
| P6-P2-001 | P2 | enhance.py | OPEN | `enhance_frame_compiled` is an unused no-op placeholder |

### Risk Assessment

- **P6-P1-001** was the highest-impact finding: it silently corrupts all displayed and
  checkpoint-stored loss metrics during GAN training. While it doesn't corrupt the
  actual model weights (the optimizer sees the real gradient), it makes training
  monitoring unreliable during GAN epochs and poisons checkpoint metadata.
- No P0 (crash/data-loss) findings in this pass.

### Areas Verified Clean

- Gradient accumulation logic (both compiled and eager paths)
- Validation loop loss accumulation (no multiplier — correct)
- DfOp complex convolution math
- Checkpoint save/load state consistency
- MLX `.at[].add()` functional semantics in StreamingDfNet4
- run_config.py normalization and validation
- Dataset fallback error handling (noise/RIR loading)
