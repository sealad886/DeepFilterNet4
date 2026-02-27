# Pass 9 Error Audit Report — train_dynamic Module Decomposition

**Branch:** `feat/major_refactor_train_dynamic`  
**Baseline (df_mlx module subset):** 88 passed, 0 failed  
**Post-audit (df_mlx module subset):** 88 passed, 0 failed (no code fixes — no blockers found)  
**Scope:** `DeepFilterNet/df_mlx/` — 14-module decomposition of `train_dynamic.py` (4580 → 2846 lines); full CI reports **1007 passed, 11 skipped**  
**Audit focus:** Behavioral preservation, error handling, state corruption risks, import-time side effects, test regression risks  
**Commit:** `e8d721a`

---

## Findings Summary

| ID | Sev | Component | Status |
|----|-----|-----------|--------|
| REF-1 | P3 | training_metrics.py / train_dynamic.py | NOTED (cosmetic) |
| REF-2 | P3 | training_ops.py | NOTED (acceptable) |
| REF-3 | INFO | training_signals.py | NOTED (by design) |
| REF-4 | INFO | all modules | VERIFIED CLEAN |
| REF-5 | INFO | test suite | VERIFIED — coverage preserved |
| REF-6 | INFO | re-exports | VERIFIED — completeness test passes |

**Verdict: NO BLOCKERS. Release-ready.**

---

## Detailed Findings

### REF-1 — Duplicated constants across modules (P3)

- **Severity:** P3 (cosmetic — no behavioral impact today, divergence risk over time)
- **Component:** `_SCALAR_ZERO` and `_GAN_SCORE_ABS_CLIP`
- **Evidence:**
  - `_SCALAR_ZERO = mx.array(0.0)` appears in 4 files:
    - `train_dynamic.py:202`
    - `training_metrics.py:46`
    - `training_validation.py:49`
    - `training_waveform.py:30` (as `_ZERO`)
  - `_GAN_SCORE_ABS_CLIP = 30.0` appears in 2 files:
    - `train_dynamic.py:206`
    - `training_metrics.py:47`
- **Why it's acceptable now:** `mx.array(0.0)` is value-immutable — each module gets its own constant instance, and the value `0.0` cannot drift. `_GAN_SCORE_ABS_CLIP` is a plain float — if someone changes one copy, the other may diverge silently.
- **Recommendation:** Consolidate `_GAN_SCORE_ABS_CLIP` into `training_helpers.py` and import from there. `_SCALAR_ZERO` can stay duplicated (per-module avoids cross-imports for a trivial constant), but could optionally be centralized too.

### REF-2 — Module-level mutable cache in training_ops.py (P3)

- **Severity:** P3 (acceptable — bounded, single-threaded)
- **Component:** `training_ops.py:201` — `_scale_cache: dict[float, mx.array] = {}`
- **Evidence:** `_get_scale_array()` caches `mx.array` instances keyed by float values. Cache is bounded to 8 entries with FIFO eviction. Only called from `scale_grads()` within the single-threaded training loop.
- **Why it's acceptable:** Training is single-threaded. Cache is bounded. Used with typically 1 distinct scale value per training run (`1/grad_accumulation_steps`).
- **Risk:** If the module is ever imported in a multi-process context (e.g., data workers), the cache is per-process and harmless. Not a concern.

### REF-3 — Signal handler module-level state (INFO)

- **Severity:** INFO (by design)
- **Component:** `training_signals.py:31-48` — `_interrupt_state` dict
- **Evidence:** Module-level mutable dict shared between `_handle_sigint` (signal handler) and the training loop (via `_update_interrupt_state`). The handler captures model/optimizer references and writes checkpoints asynchronously on SIGINT.
- **Why it's correct:**
  1. Signal handlers in CPython run in the main thread, same as the training loop.
  2. `_update_interrupt_state` is called every batch, keeping references fresh.
  3. Double-SIGINT correctly triggers `sys.exit(1)` (hard exit).
  4. Exception handling in the handler is proper: inner try for data checkpoint, outer try for model checkpoint, with informative error messages.
  5. The `raise KeyboardInterrupt()` at the end propagates correctly to the batch loop's `except KeyboardInterrupt` handler.
- **One note:** After extraction, `training_signals.py` imports `save_checkpoint` at module level. This means importing `training_signals` triggers an import of `training_checkpoints`, which imports `mlx.core`, `safetensors`, etc. This is fine because `training_signals` is only imported by `train_dynamic.py`, which already imports everything anyway.

### REF-4 — Exception handling audit (INFO — verified clean)

- **Severity:** INFO
- **Scope:** All 14 extracted modules
- **Findings:**
  - **No bare `except:` clauses** anywhere in the codebase.
  - **`except Exception as e:` usage** (all proper):
    - `training_checkpoints.py`: 11 occurrences — all handle checkpoint I/O failures (file read/write, JSON parse, safetensors load) with descriptive error messages and graceful fallback behavior (e.g., return `None` or `False` for failed saves).
    - `training_setup.py:setup_auxiliary_losses`: 1 occurrence — wraps optional MRSTFT loss import with fallback to `None`.
    - `training_signals.py:_handle_sigint`: 2 occurrences — inner (data checkpoint failure) and outer (model checkpoint failure) with error messages.
  - **No swallowed exceptions** — every `except` block either logs/prints the error, re-raises, or returns a sentinel indicating failure.
  - **`raise` usage in `training_cli.py`**: 10 `ValueError` raises with descriptive messages for invalid CLI arguments (JSON parse errors, missing keys, negative values, conflicting flags). All correct.
  - **`training_cli_main.py`**: No exception handling — delegates to `argparse` (which handles its own errors) and `train()` (which handles its own errors). Correct for a thin CLI wrapper.
  - **`training_losses.py`**: No exception handling — pure computation functions with no I/O. Correct.
  - **`training_ops.py`**: No `except` — `NumericDebugger.check()` raises `FloatingPointError` when `fail_fast=True` and a non-finite value is detected. `check_tree()` explicitly never raises (documents this in docstring). Correct.

### REF-5 — Test coverage preservation (INFO — verified)

- **Severity:** INFO
- **Test suite:** 88 df_mlx-focused tests, 0 failures, 0 skips (1007 total project tests pass)
- **Test diff analysis:**
  - `test_gan_memory_path.py`: 1 line change (`global_step` → `loop_state.global_step`) — adapts to `TrainingLoopState` extraction.
  - `test_loss_audit_fixes.py`: 16+/-5 lines — source grep assertions now check `training_metrics.py` and `training_diagnostics.py` instead of just `train_dynamic.py`. All assertions are equivalent or stricter.
  - `test_sync_cadence_integration.py`: 8+/-3 lines — added `training_metrics.py` to source file set for sync-cadence pattern verification.
  - `test_train_control_semantics.py`: 16+/-6 lines — updated source assertions for `loop_state.*` fields and `training_helpers.py`.
  - `test_train_logging_integrity.py`: 3+/-1 lines — added `TRAINING_METRICS_PATH` to source file reading.
  - `test_training_helpers.py`: 112 lines NEW — 5 tests for `build_setup_panel_line`, `curriculum_schedule`, `clip_gan_scores`, `is_vad_train_reg_enabled`, and wrapper equivalence.
  - `test_train_dynamic_reexports.py`: 3 tests for re-export completeness, symbol identity, and hardware diagnostics — ALL PASS.
- **No test cases dropped or weakened.** Test changes are purely mechanical: adjusting file paths in source-grep assertions to reflect new module locations.

### REF-6 — Re-export completeness (INFO — verified)

- **Severity:** INFO
- **Evidence:** `test_train_dynamic_reexports.py` verifies:
  1. Every public symbol from all 8 re-exported modules (`training_ops`, `training_losses`, `training_waveform`, `training_checkpoints`, `training_signals`, `training_session`, `training_diagnostics`, `training_helpers`) is accessible from `train_dynamic`.
  2. Re-exported symbols are the **same objects** (identity check via `is`), not copies.
  3. Hardware diagnostics functions are specifically verified.
- **Import chain test:** `import df_mlx.train_dynamic` succeeds with no circular import errors.

---

## State Corruption Risk Analysis

### TrainingLoopState Dataclass

`training_helpers.py:31-64` defines `TrainingLoopState` as a mutable `@dataclass` with 11 fields tracking loop progress (global_step, epoch tracking, best loss, stage info, weights).

**Initialization correctness** (train_dynamic.py:1593-1606):
- `global_step`: Correctly set from `resume_global_step` when resuming, or `start_epoch * optimizer_steps_per_epoch` for fresh starts.
- `last_completed_epoch`: Correctly set from `max(last_completed_epoch, start_epoch - 1)`.
- `best_valid_loss`: Correctly plumbed from `reconcile_resume → ResumeResult → loop_state`.
- `active_stage_name/index`: Correctly computed from `max(resume_stage_index, scheduled_start_stage_index)` with clamping and normalization logging.

**Thread safety:** Not a concern — single-threaded training loop. The `_interrupt_state` dict (module-level in `training_signals.py`) reads `loop_state` fields via `_update_interrupt_state` calls, but signal handlers run in the main thread in CPython.

**Mutation discipline:** `loop_state` is mutated directly by the batch loop code in `train_dynamic.py`. The `gan_active` field has a special separate variable for closure capture (`gan_active = loop_state.gan_active` at line 1607), which is correctly kept in sync when GAN activates later in the loop.

---

## Import-Time Side Effect Analysis

| Module | Side Effects | Risk |
|--------|-------------|------|
| `training_signals.py` | Imports `save_checkpoint` at top level | Low — only triggers `safetensors` import chain, which is needed anyway |
| `training_diagnostics.py` | Imports `tqdm.auto` at top level | Low — tqdm is always needed for training |
| `training_ops.py` | Creates `_scale_cache = {}` dict | None — empty dict, lazy-filled |
| `training_metrics.py` | Imports from `training_losses`, `training_ops`, `training_waveform` | None — these are all in-package imports |
| `training_checkpoints.py` | Imports `safetensors`, `json`, `shutil` at top level | None — standard for checkpoint module |
| All others | Pure function/class definitions | None |

**No module executes expensive initialization (model loading, file I/O, network calls) at import time.**

---

## Coverage Gaps

While all 88 df_mlx-focused tests pass, the following areas have limited direct test coverage:

1. **`training_checkpoints.py`** (1098 lines): `reconcile_resume` (220 lines of complex resume logic) is tested only indirectly via integration tests. A dedicated unit test with mock checkpoints would improve confidence.
2. **`training_setup.py`** (1063 lines): `setup_auxiliary_losses`, `setup_dataset`, `setup_data_pipeline` — tested indirectly via training integration tests. Could benefit from unit tests with mock configs.
3. **`training_metrics.py`** (748 lines): `collect_sync_metrics` is a large function (~500 lines) with many conditional branches. Only tested indirectly through source-grep pattern tests.
4. **`training_validation.py`** (781 lines): `run_validation` — tested indirectly. Would benefit from a mock-model validation test.

These gaps **predate the refactoring** — the extraction did not reduce coverage. The existing integration tests exercise these paths through the full `train()` function.

---

## Release Readiness Assessment

| Criterion | Status |
|-----------|--------|
| All tests pass | ✅ 88/88 (df_mlx subset) |
| No bare except clauses | ✅ |
| No swallowed exceptions | ✅ |
| Re-exports complete | ✅ (3 tests) |
| No circular imports | ✅ |
| No import-time side effects | ✅ |
| State initialization correct | ✅ |
| Signal handler correctness | ✅ |
| No behavioral changes in loss computation | ✅ |
| Test coverage preserved | ✅ |
| Constants consistent | ⚠️ P3 (cosmetic duplication) |

**VERDICT: RELEASE-READY. No blockers. One P3 cosmetic finding (constant duplication) for future cleanup.**
