# Duplication Audit — Pass 7

**Scope:** `DeepFilterNet/df_mlx/` module  
**Branch:** `feat/df_mlx-mlx-mega-optimization`  
**Baseline:** 977 tests pass, 11 skipped  
**Date:** 2025-07-14

## Findings Summary

| ID | Sev | Component | Classification | Status |
|----|-----|-----------|---------------|--------|
| DUP-7.1 | P0 | loss.py | Unnecessary | **FIXED** |
| DUP-7.2 | P1 | train.py / loss.py | Unnecessary | **FIXED** |
| DUP-7.3 | P1 | train_with_data.py | Justified | Documented |
| DUP-7.4 | P2 | train.py / lr.py | Justified | Documented |
| DUP-7.5 | P2 | loss.py / evaluation.py | Justified | Documented |
| DUP-7.6 | P2 | train.py (Trainer) | Justified | Documented |

---

## DUP-7.1 — Dead MaskSpecLoss class (P0, FIXED)

**Files:** `loss.py:409-477`  
**Evidence:** `MaskSpecLoss` was an internal helper class only called by itself (via `MaskLoss`) and had zero external callers. Pass 6 identified it for removal but it persisted.  
**Action:** Removed 68 lines. No exports, no callers affected.  
**Commit:** `ecdd80a refactor(loss): remove dead MaskSpecLoss class (DUP-7.1)`  
**Verification:** 977 passed, 11 skipped ✓

## DUP-7.2 — MultiResolutionSTFTLoss ≈ SpectralLoss (P1, FIXED)

**Files:** `train.py:385-520` (MultiResolutionSTFTLoss), `loss.py:100-205` (SpectralLoss)  
**Evidence:** Both classes implement identical multi-resolution spectral loss with gamma compression and optional complex loss. Same algorithm (STFT → magnitude → gamma compress → MSE → average over resolutions). Differences were superficial:
- Parameter naming: `f_complex` vs `factor_complex`
- eps defaults: `1e-12` vs `1e-10`
- `mx.maximum` guard in gamma compression (MRSTFT had it, SpectralLoss didn't — negligible since `pred_mag >= sqrt(eps)`)
- MRSTFT had extra methods: `compute_per_resolution`, `from_config`

**Action:** Made `MultiResolutionSTFTLoss.__call__` delegate to an internal `SpectralLoss` instance. Preserved all external API (parameter names, extra methods, type annotations). Removed ~40 lines of duplicated loop logic.  
**Callers migrated:** None needed — external API unchanged. `train_dynamic.py`, `train_gan.py`, `test_mlx_mrstft_fp32.py`, `test_loss_audit_fixes.py` all use the same interface.  
**Commit:** `b44dad1 refactor(train): delegate MultiResolutionSTFTLoss to SpectralLoss (DUP-7.2)`  
**Verification:** 977 passed, 11 skipped ✓

## DUP-7.3 — train_with_data.py checkpoint functions (P1, JUSTIFIED)

**Files:** `train_with_data.py:47-92` (save_checkpoint), `train_with_data.py:97-132` (load_checkpoint) vs `train.py:521-585` (save_checkpoint), `train.py:588-625` (load_checkpoint)  
**Evidence:** Similar structure but semantically distinct:
- Different state file extensions: `.state.json` vs `.json`
- Different parameter signatures (keyword-only vs positional)
- Different weight-loading strategies (manual tree_flatten vs `model.load_weights()`)
- `train_with_data.py` is a self-contained training script

**Rationale:** Migrating would break backward compatibility with existing checkpoints (file extension mismatch). Pass 6 reached the same conclusion ("de minimis risk"). Self-contained script pattern is intentional.

## DUP-7.4 — WarmupCosineSchedule vs CosineScheduler (P2, JUSTIFIED)

**Files:** `train.py:630-670` (WarmupCosineSchedule), `lr.py` (CosineScheduler + WarmupScheduler)  
**Evidence:** `lr.py` provides a full LR scheduler library with composable warmup/decay. `WarmupCosineSchedule` in `train.py` is a simple standalone scheduler for the legacy Trainer class. Different API complexity levels and different consumers.

## DUP-7.5 — loss.py si_sdr vs evaluation.py si_sdr (P2, JUSTIFIED)

**Files:** `loss.py` (si_sdr returns mean scalar), `evaluation.py` (si_sdr returns per-batch values)  
**Evidence:** Different return semantics serve different purposes — training loss (needs scalar) vs evaluation metrics (needs per-sample breakdown).

## DUP-7.6 — Trainer.save_checkpoint vs standalone (P2, JUSTIFIED)

**Files:** `train.py` Trainer class method vs standalone `save_checkpoint` function  
**Evidence:** Internal class method wrapping the standalone function. Normal OOP pattern, no drift risk.

---

## Metrics

| Metric | Value |
|--------|-------|
| Lines removed | ~108 (68 MaskSpecLoss + 40 MRSTFT loop) |
| Lines added | ~10 (delegation + import) |
| Net reduction | ~98 lines |
| Commits | 2 |
| Test regression | None (977/977 pass) |
| Justified duplications | 4 (documented above) |

## Residual Risks

None identified. All remaining duplications are justified with documented rationale.
