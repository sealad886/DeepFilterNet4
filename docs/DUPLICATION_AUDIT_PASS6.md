# Duplication Audit — Pass 6

**Date:** 2025-07-09  
**Scope:** `DeepFilterNet/df_mlx/`, `DeepFilterNet/examples/`, `DeepFilterNet/tests/`  
**Method:** Full function/class inventory, cross-file import tracing, dead-code analysis  
**Predecessor:** Pass 4 (DUP-4.1–4.6), Pass 5 (DUP-5.1–5.3)

---

## Summary

| ID | Severity | Classification | Description | Status |
|---|---|---|---|---|
| DUP-6.1 | **P2** | UNNECESSARY | `combined_loss` standalone function in `train.py` | **REMOVED** |
| DUP-6.2 | **P2** | UNNECESSARY | `snr_loss` standalone function in `train.py` | **REMOVED** |
| DUP-6.3 | **P2** | UNNECESSARY | `lsnr_loss` standalone function in `train.py` | **REMOVED** |
| DUP-6.4 | **P2** | UNNECESSARY | `multi_resolution_stft_loss` standalone function in `train.py` | **REMOVED** |
| DUP-6.5 | **P1** | JUSTIFIED | `loss_fn` vs `loss_fn_gan` in `train_dynamic.py` (~166 lines each) | DOCUMENTED |
| DUP-6.6 | **P1** | UNNECESSARY — deferred | `train_gan.py` entire module (718 lines) — effectively dead code | DEFERRED |
| DUP-6.7 | **P2** | UNNECESSARY — deferred | `checkpoint.py` only consumed by dead `train_gan.py` + `__init__.py` exports | DEFERRED |
| DUP-6.8 | **P2** | JUSTIFIED | `WarmupCosineSchedule` (train.py) vs `CosineScheduler` (lr.py) | DOCUMENTED |
| DUP-6.9 | **P2** | UNNECESSARY — deferred | 5× `save_checkpoint`/`load_checkpoint` implementations | DEFERRED |
| DUP-6.10 | **P3** | JUSTIFIED | 5× `count_params` helper functions (script-local) | DOCUMENTED |

**Net result:** ~131 lines removed from production code. 4 dead functions eliminated. 0 test regressions (977 passed, 11 skipped).

### Carry-forward from prior passes (OPEN)

| Prior ID | Severity | Description | Notes |
|---|---|---|---|
| DUP-4.1 | **P0** | ERB filterbank formula divergence (`feature_ops.py` vs `ops.py`) | Correctness risk — not yet resolved |
| DUP-4.2 | **P1** | `whisper_adapter.py` duplicated across `df/` and `df_mlx/` | ~800 shared lines |
| DUP-4.3 | **P1** | Spectral loss overlap (`train.py:spectral_loss` vs `loss.py:SpectralLoss`) | Partially addressed: removed `multi_resolution_stft_loss` standalone |
| DUP-4.4 | **P2** | Dual `load_pytorch_checkpoint` in `train.py` + `convert.py` | Different model types |
| DUP-5.1 | **P1** | `sepm.py` 100% copy between `df/` and `df_mlx/` | ~500 LOC pure NumPy |

---

## DUP-6.1 — `combined_loss` Standalone Function (REMOVED)

**Severity:** P2  
**Classification:** UNNECESSARY  
**File:** `df_mlx/train.py` (formerly lines 652–693, ~42 lines)

### What was duplicated

`combined_loss()` was a standalone function composing spectral + STFT losses. It overlapped with:
- `spectral_loss()` in the same file (used by production `train_dynamic.py`)
- `CombinedLoss` class in `loss.py` (used by `train_gan.py`)
- The inline loss computation in `loss_fn` / `loss_fn_gan` in `train_dynamic.py`

### Usage before removal

| Caller | Type |
|---|---|
| `examples/mlx_training.py` | Example script (only caller) |

### Action taken

- **Removed** `combined_loss()` from `train.py`
- **Updated** `examples/mlx_training.py` to use `spectral_loss()` directly (the production-canonical loss function)
- No `__init__.py` change needed — `combined_loss` was never exported

---

## DUP-6.2 — `snr_loss` Standalone Function (REMOVED)

**Severity:** P2  
**Classification:** UNNECESSARY  
**File:** `df_mlx/train.py` (formerly lines 598–620, ~23 lines)

### What was duplicated

`snr_loss()` computed signal-to-noise ratio loss. Production training (`train_dynamic.py`) computes SNR inline via `training_losses.py`. This function existed solely for test consumption.

### Usage before removal

| Caller | Type |
|---|---|
| `df_mlx/test_mlx_comprehensive.py::test_snr_loss` | Test |

### Action taken

- **Removed** `snr_loss()` from `train.py`
- **Updated** test to use inline SNR calculation (tests the math, not the removed wrapper)

---

## DUP-6.3 — `lsnr_loss` Standalone Function (REMOVED)

**Severity:** P2  
**Classification:** UNNECESSARY  
**File:** `df_mlx/train.py` (formerly lines 621–649, ~29 lines)

### What was duplicated

`lsnr_loss()` computed log-SNR loss with clipping. Production training computes LSNR inline. This function existed solely for test consumption.

### Usage before removal

| Caller | Type |
|---|---|
| `df_mlx/test_mlx_comprehensive.py::test_lsnr_loss_function` | Test |
| `df_mlx/test_mlx_comprehensive.py::test_lsnr_loss_clipping` | Test |

### Action taken

- **Removed** `lsnr_loss()` from `train.py`
- **Updated** tests to use inline LSNR clipping tests

---

## DUP-6.4 — `multi_resolution_stft_loss` Standalone Function (REMOVED)

**Severity:** P2  
**Classification:** UNNECESSARY  
**File:** `df_mlx/train.py` (formerly lines 380–414, ~35 lines)

### What was duplicated

`multi_resolution_stft_loss()` was a standalone function that duplicated the `MultiResolutionSTFTLoss` *class* in the same file. The class is used by production training (`train_dynamic.py`). The standalone function was only used by one test.

### Usage before removal

| Caller | Type |
|---|---|
| `df_mlx/test_mlx_comprehensive.py::test_multi_resolution_stft_loss` | Test |

### Action taken

- **Removed** `multi_resolution_stft_loss()` standalone function from `train.py`
- **Removed** from `__init__.py` imports and `__all__`
- **Updated** test to use `MultiResolutionSTFTLoss` class directly (the canonical implementation)
- **Kept** `MultiResolutionSTFTLoss` class (actively used by `train_dynamic.py` and `train_gan.py`)

---

## DUP-6.5 — `loss_fn` vs `loss_fn_gan` in train_dynamic.py (JUSTIFIED)

**Severity:** P1  
**Classification:** JUSTIFIED  
**File:** `df_mlx/train_dynamic.py` (`loss_fn` ~L1201–1365, `loss_fn_gan` ~L1379–1545)

### What's duplicated

Two near-identical ~166-line loss computation functions that share >90% of their logic. The `loss_fn_gan` variant adds GAN discriminator/generator loss after the base spectral + alpha losses.

### Why justified

These functions are compiled via `mx.compile()`. MLX's compilation captures Python boolean values at trace time — the `gan_active` flag cannot be changed after compilation. Creating a single function with a runtime `gan_active` branch would require:
- Recompilation on every mode switch (expensive), or
- Always computing GAN losses even when GAN is inactive (wasteful)

The duplication is an intentional design response to `mx.compile` constraints.

### Recommendation

No action. Document the constraint in code comments if not already present. If MLX adds runtime-conditional compilation in the future, this can be unified.

---

## DUP-6.6 — `train_gan.py` Entire Module — Effectively Dead Code (DEFERRED)

**Severity:** P1  
**Classification:** UNNECESSARY — deferred  
**File:** `df_mlx/train_gan.py` (718 lines)

### Evidence

`train_gan.py` exports `GANConfig`, `GANTrainer`, and `train_gan()`. Import trace:

| Consumer | Import | Status |
|---|---|---|
| `__init__.py` L198 | `from .train_gan import GANConfig, GANTrainer, train_gan` | Re-export only |
| All tests | — | No test imports `GANTrainer` or `train_gan` |
| All examples | — | No example imports `GANTrainer` or `train_gan` |
| `train_dynamic.py` | — | Uses its own GAN training loop (inline `loss_fn_gan`) |

Additionally, `GANTrainer.__init__` instantiates `self.combined_loss = CombinedLoss(self.loss_config)` at line 146, but `self.combined_loss` is **never called** anywhere in the class — a dead instantiation.

### Impact

`train_gan.py` is a legacy standalone GAN trainer that was superseded by `train_dynamic.py`'s integrated GAN training. Removing it would also make `checkpoint.py` mostly dead (see DUP-6.7) and reduce the `CosineScheduler` usage footprint.

### Why deferred

- 718 lines is a large removal that needs team buy-in
- The `__init__.py` public API exports `GANTrainer` and `train_gan` — downstream consumers may exist outside this repo
- Requires deprecation notice before removal

### Consolidation plan

1. Add deprecation warning to `GANTrainer.__init__` and `train_gan()`
2. Remove `self.combined_loss` dead instantiation (zero-risk)
3. After deprecation period, remove `train_gan.py` and update `__init__.py` exports
4. Cascade: evaluate `checkpoint.py` removal (DUP-6.7)

---

## DUP-6.7 — `checkpoint.py` Consumed Primarily by Dead `train_gan.py` (DEFERRED)

**Severity:** P2  
**Classification:** UNNECESSARY — deferred (gated on DUP-6.6)  
**File:** `df_mlx/checkpoint.py` (633 lines)

### Usage trace

| Consumer | Symbols imported |
|---|---|
| `train_gan.py` L26 | `CheckpointManager`, `PatienceState`, `check_patience` |
| `__init__.py` L62–70 | `CheckpointManager`, `CheckpointState`, `PatienceState`, `check_patience`, `load_checkpoint`, `read_patience`, `save_checkpoint`, `write_patience` |
| `tests/test_audit_fixes.py` L147 | `CheckpointState`, `save_checkpoint` |

### Analysis

- `train_gan.py` is the **only production consumer** of `checkpoint.py` — and `train_gan.py` is effectively dead (DUP-6.6)
- `__init__.py` re-exports symbols but no code outside this repo has been confirmed to call them
- `test_audit_fixes.py` imports `CheckpointState` and `save_checkpoint` for testing
- Production training (`train_dynamic.py`) uses `training_checkpoints.py` — a completely separate implementation

### Near-duplicate with `training_checkpoints.py`

Both have `save_checkpoint()` and `load_checkpoint()` with different signatures:

| checkpoint.py | training_checkpoints.py |
|---|---|
| `save_checkpoint(model, optimizer, path, epoch, loss, scheduler)` | `save_checkpoint(model, optimizer, ...)` with `CheckpointManifest`, `CheckpointRecord` |
| Dict-based state | Structured `CheckpointManifest` with manifest JSON |
| No discriminator support | Full discriminator optimizer/scheduler save |

### Why deferred

Gated on DUP-6.6 decision. Cannot remove checkpoint.py while train_gan.py exists.

---

## DUP-6.8 — `WarmupCosineSchedule` vs `CosineScheduler` (JUSTIFIED)

**Severity:** P2  
**Classification:** JUSTIFIED  
**Files:**
- `df_mlx/train.py:671` — `WarmupCosineSchedule` (~35 lines)
- `df_mlx/lr.py:131` — `CosineScheduler` (~100+ lines)

### What's duplicated

Two cosine learning rate schedulers with overlapping purpose:

| Feature | `WarmupCosineSchedule` | `CosineScheduler` |
|---|---|---|
| Interface | `__call__(step) -> float` | `.step() -> float` (stateful) |
| Warmup | Linear by step count | By epoch count + step count |
| Cycle support | No | Yes (decay, multiplier) |
| State tracking | Stateless | Tracks epoch/step/lr |
| Complexity | ~35 lines | ~100+ lines |

### Usage

| Implementation | Callers |
|---|---|
| `WarmupCosineSchedule` | `train_dynamic.py` (production), `Trainer` (train.py), tests, examples |
| `CosineScheduler` | `train_gan.py` (dead code — DUP-6.6), `checkpoint.py` (DUP-6.7) |

### Why justified

Different APIs, different feature sets, different consumers. `WarmupCosineSchedule` is simple and stateless (ideal for compiled training loops). `CosineScheduler` has full state management needed by the legacy checkpoint system. If DUP-6.6 is resolved (train_gan.py removed), `CosineScheduler` usage drops to only `checkpoint.py`, which itself may become removable.

---

## DUP-6.9 — 5× `save_checkpoint`/`load_checkpoint` Implementations (DEFERRED)

**Severity:** P2  
**Classification:** UNNECESSARY — deferred  

### Inventory

| Location | Scope | Callers |
|---|---|---|
| `training_checkpoints.py:493,655` | Production (train_dynamic.py) | Production training, training_signals.py, training_cli_main.py |
| `checkpoint.py:206,289` | Legacy (train\_gan.py) | train\_gan.py, \_\_init\_\_.py exports |
| `train.py:562,629` | Original (Trainer class) | examples, Trainer.save_checkpoint/load_checkpoint |
| `train_with_data.py:47,97` | Standalone script | Self-contained |
| `train_gan.py:594,641` | GANTrainer methods | Self-contained (dead code) |

(`dynamic_dataset.py:1705` has a data-specific checkpoint — excluded as justifiably different)

### Analysis

Five separate save/load checkpoint implementations exist. Only `training_checkpoints.py` is used by production training. The others serve legacy, example, or standalone contexts.

### Why deferred

- `training_checkpoints.py` is the canonical production implementation
- `train.py` functions are used by examples — would need example updates
- `train_with_data.py` is a self-contained script — duplication is low-risk
- `checkpoint.py` removal gated on DUP-6.6
- `train_gan.py` methods dead (DUP-6.6)

### Consolidation plan (future)

1. Resolve DUP-6.6 (remove train_gan.py) — cascades to checkpoint.py and train_gan.py methods
2. Update examples to use `training_checkpoints.py` or a shared utility — removes train.py functions
3. Leave `train_with_data.py` as self-contained (de minimis risk)

---

## DUP-6.10 — 5× `count_params` Helper Functions (JUSTIFIED)

**Severity:** P3  
**Classification:** JUSTIFIED  
**Files:**
- `deepfilternet.py:513` — DFNet1 model file
- `deepfilternet2.py:587` — DFNet2 model file
- `deepfilternet3.py:533` — DFNet3 model file  
- `quantization.py:102` — Quantization utility (different signature)
- `quantization.py:280` — Second variant in same file

### Why justified

Each `count_params` is a trivial 3–5 line helper defined inline within a script's `__main__` block or diagnostic section. They're script-local utilities, not shared library functions. The overhead of extracting them to a shared module exceeds the duplication cost.

---

## Changes Made This Pass

### Files modified

| File | Change |
|---|---|
| `df_mlx/train.py` | Removed `combined_loss` (~42 lines), `snr_loss` (~23 lines), `lsnr_loss` (~29 lines), `multi_resolution_stft_loss` (~35 lines) |
| `df_mlx/__init__.py` | Removed `multi_resolution_stft_loss` from imports and `__all__` |
| `df_mlx/test_mlx_comprehensive.py` | Updated 4 tests to not import removed functions |
| `examples/mlx_training.py` | Changed from `combined_loss` to `spectral_loss` |

### Verification

- **Tests:** 977 passed, 11 skipped, 0 failures (full pytest suite)
- **Formatting:** black + isort applied to all modified files
- **Errors:** No compilation/type errors in any modified file
- **Net lines removed:** ~131 lines from production code

---

## Priority Queue for Future Passes

| Priority | ID | Action | Impact |
|---|---|---|---|
| 1 | DUP-4.1 | Fix ERB filterbank formula divergence | **P0** correctness risk |
| 2 | DUP-6.6 | Deprecate/remove `train_gan.py` | Unlocks DUP-6.7, DUP-6.9 cascade |
| 3 | DUP-4.2 | Consolidate `whisper_adapter.py` | ~800 lines |
| 4 | DUP-5.1 | Consolidate `sepm.py` | ~500 lines |
| 5 | DUP-6.7 | Remove `checkpoint.py` (after DUP-6.6) | ~633 lines |
| 6 | DUP-4.4 | Consolidate `load_pytorch_checkpoint` | ~100 lines |
| 7 | DUP-6.9 | Consolidate checkpoint save/load to canonical impl | Architecture cleanup |
