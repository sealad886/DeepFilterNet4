# Duplication Audit — Pass 4

**Date:** 2026-02-19  
**Scope:** `DeepFilterNet/df_mlx/`, `DeepFilterNet/df/`, `DeepFilterNet/tests/`, root-level scripts  
**Method:** Full function/class inventory, cross-file signature matching, formula comparison  
**Exclusions:** Pass 2 fixes (committed), Pass 3 findings (TinyModel×3, checkpoint save/load×5, WarmupCosine/CosineScheduler, count_params×3, LossConfig/GANConfig, TrainConfig/TrainingConfig, load_audio variants)

---

## Summary

| ID | Severity | Classification | Description |
|---|---|---|---|
| DUP-4.1 | **P0** | UNNECESSARY — correctness risk | ERB filterbank: two incompatible formulas in data prep vs. model inference |
| DUP-4.2 | **P1** | UNNECESSARY | `whisper_adapter.py` duplicated across `df/` and `df_mlx/` (~800 shared lines) |
| DUP-4.3 | **P1** | UNNECESSARY | Spectral loss functions in `train.py` vs. `loss.py` (semantic overlap) |
| DUP-4.4 | **P2** | UNNECESSARY | Dual `convert_pytorch_weights` / `load_pytorch_checkpoint` in `train.py` + `convert.py` |
| DUP-4.5 | **P3** | JUSTIFIED | `clip_grad_norm` reimplemented in `benchmark_sync_barriers.py` |
| DUP-4.6 | **P3** | JUSTIFIED | Signal handlers in `training_signals.py` vs. `datastore.py` |

---

## DUP-4.1 — ERB Filterbank Formula Divergence

**Severity:** P0 (correctness risk)  
**Classification:** UNNECESSARY  
**Files:**

- `df_mlx/feature_ops.py:38` — `create_erb_filterbank()`
- `df_mlx/ops.py:329-340` — `erb_frequency()`, `erb_inv()`, `erb_fb()`
- `df_mlx/modules.py` — `ErbFilterbank` (delegates to `ops.erb_fb`)

### What's duplicated

Three ERB filterbank implementations exist within `df_mlx/`. Two use **mathematically different ERB-scale formulas**:

| Location | Formula | Scale |
|---|---|---|
| `feature_ops.py` | `9.265 × ln(1 + f / (24.7 × 9.265))` | Glasberg & Moore (1990) |
| `ops.py` | `21.4 × log₁₀(1 + f / 229)` | Moore & Glasberg (1983) |

These are **not** equivalent — they produce different frequency-to-ERB mappings.

Additionally:
- `feature_ops` generates `nb_erb` center frequencies directly; `ops.erb_fb` generates `nb_bands + 2` points (mel-filterbank-style triangular overlap).
- `feature_ops` applies ERB bandwidth via `24.7 × (4.37 × f/1000 + 1)` for the triangle width; `ops.erb_fb` uses adjacent center spacing.
- `feature_ops` returns `np.ndarray` (no normalization); `ops.erb_fb` supports normalization and min-width enforcement, returns `mx.array` by default.

### Usage paths

| Implementation | Callers | Pipeline stage |
|---|---|---|
| `feature_ops.create_erb_filterbank` | `dynamic_dataset.py`, `prepare_data.py`, `benchmark_workers.py` | **Data preprocessing** |
| `ops.erb_fb` → `modules.ErbFilterbank` | Model forward pass, `train_dynamic.py`, all inference | **Model inference** |

### Impact

Training data features are computed with one ERB filterbank (`feature_ops`), but the model's forward pass uses a different one (`ops.erb_fb`). This means the ERB representation seen during data loading may not match the ERB representation computed inside the model — a train/inference mismatch.

### Consolidation plan

1. **Audit correctness first:** Compare the two filterbanks numerically for the default config (sr=48000, fft=960, nb_erb=32). Quantify the divergence.
2. **Determine canonical formula:** Decide which ERB scale is correct for this model (likely `ops.erb_fb`, since it's used in inference and matches libDF's Rust implementation).
3. **Unify:** Replace `feature_ops.create_erb_filterbank` with a call to `ops.erb_fb(as_numpy=True)`, or extract a shared private function.
4. **Re-validate:** If the filterbanks diverge significantly, cached/preprocessed data may need regeneration.

---

## DUP-4.2 — Whisper Adapter Cross-Backend Duplication

**Severity:** P1  
**Classification:** UNNECESSARY  
**Files:**

- `df/whisper_adapter.py` (1023 lines)
- `df_mlx/whisper_adapter.py` (1286 lines)

### What's duplicated

Both files contain near-identical implementations of:

| Symbol | In both files |
|---|---|
| `is_apple_silicon()` | ✓ |
| `WhisperDecodingResult` (dataclass) | ✓ |
| `WhisperBackend` (Protocol) | ✓ |
| `to_numpy()` | ✓ |
| `mx_to_torch()` | ✓ |
| `torch_to_mx()` | ✓ |
| `_resolve_mlx_model_name()` | ✓ |
| `MLXWhisperBackend` | ✓ |
| `PyTorchWhisperBackend` | ✓ |
| `get_whisper_backend()` | ✓ |
| `load_whisper_model()` | ✓ |

`df_mlx/whisper_adapter.py` additionally contains: `_ensure_torch()`, `_get_torch()`, `mx_to_numpy()`, `numpy_to_mx()`, `compute_asr_features()`, `compute_whisper_loss()`, `compute_word_accuracy()`, `evaluate_transcription_batch()`.

The shared portion is approximately 800+ lines. The `df/` version imports torch eagerly; the `df_mlx/` version lazy-imports both.

### Consolidation plan

1. Extract shared code (dataclass, protocol, conversion utils, backend routing) into a new `df/whisper_common.py` (or `whisper_core.py`).
2. Both `df/whisper_adapter.py` and `df_mlx/whisper_adapter.py` import from the shared module.
3. `df_mlx/whisper_adapter.py` keeps its MLX-specific training helpers (`compute_whisper_loss`, etc.).
4. ~600-800 lines eliminated.

---

## DUP-4.3 — Spectral Loss Function Overlap (train.py vs. loss.py)

**Severity:** P1  
**Classification:** UNNECESSARY  
**Files:**

- `df_mlx/train.py:346-595` — `spectral_loss()`, `multi_resolution_stft_loss()`, `MultiResolutionSTFTLoss`
- `df_mlx/loss.py:100-210` — `SpectralLoss`, `FusedSpectralLoss`
- `df_mlx/train.py:598-693` — `snr_loss()`, `lsnr_loss()`, `combined_loss()`
- `df_mlx/loss.py:848+` — `CombinedLoss`

### What's duplicated

`train.py` and `loss.py` both provide spectral loss implementations:

| `train.py` | `loss.py` | Relationship |
|---|---|---|
| `spectral_loss()` (L1 mag + L1 complex) | n/a | Standalone function, simplest form |
| `multi_resolution_stft_loss()` | n/a | Wrapper calling `spectral_loss` at multiple FFT sizes |
| `MultiResolutionSTFTLoss` class | `SpectralLoss` class | Both do multi-resolution STFT → magnitude → loss, but `SpectralLoss` adds gamma compression, MSE (vs L1), and compressed-complex loss |
| `combined_loss()` function | `CombinedLoss` class | Both combine spectral + SNR losses with configurable weights |

`train.py`'s functions predate `loss.py` and are simpler (L1, no compression). `loss.py` is the more mature, configurable implementation. Both are actively imported:

- `train.py:spectral_loss` is used by: `train_dynamic.py`, `benchmark_gan_sync.py`, `benchmark_gan_profile.py`, `benchmark_gan_throughput.py`, `benchmark_train_step.py`, `train_with_data.py`, `test_mlx.py`, `test_mlx_comprehensive.py`
- `loss.py:SpectralLoss` is used by: `benchmark_hotspots.py`, `train_dynamic.py` (for GAN path)

### Consolidation plan

1. Migrate callers of `train.py:spectral_loss` to use `loss.py:SpectralLoss(gamma=1.0, factor_complex=alpha)` for equivalent behavior.
2. `multi_resolution_stft_loss()` becomes `SpectralLoss(fft_sizes=(...))` — already equivalent.
3. Deprecate `train.py:MultiResolutionSTFTLoss` in favor of `loss.py:SpectralLoss`.
4. Deprecate `train.py:combined_loss` in favor of `loss.py:CombinedLoss`.
5. Keep re-exports in `train.py` for backward compatibility with a deprecation warning.
6. ~350 lines removed from `train.py`.

---

## DUP-4.4 — Dual PyTorch Weight Conversion Functions

**Severity:** P2  
**Classification:** UNNECESSARY  
**Files:**

- `df_mlx/train.py:1150` — `convert_pytorch_weights()` (generic: transposes 4D convs, direct name map)
- `df_mlx/train.py:1189` — `load_pytorch_checkpoint()` (loads torch file, calls `convert_pytorch_weights`)
- `df_mlx/convert.py:296` — `load_pytorch_checkpoint()` (full: model-type dispatch for dfnet1/2/3/4, `weights_only=True`, returns metadata)
- `df_mlx/__init__.py:80` — exports `convert.load_pytorch_checkpoint` as `load_pytorch_ckpt`
- `df_mlx/__init__.py:196` — exports `train.load_pytorch_checkpoint` as `load_pytorch_checkpoint`

### Impact

Two functions with the **same name** (`load_pytorch_checkpoint`) are exported from `__init__.py` under different aliases. The `train.py` version is simpler and less safe (`weights_only` not enforced). This creates confusion for callers.

### Consolidation plan

1. Deprecate `train.py:convert_pytorch_weights` and `train.py:load_pytorch_checkpoint`.
2. All callers should use `convert.py:load_pytorch_checkpoint` (the more complete version).
3. Update `__init__.py` to export only one `load_pytorch_checkpoint` (from `convert.py`).
4. Keep a thin re-export in `train.py` with deprecation warning for backward compatibility.

---

## DUP-4.5 — clip_grad_norm in Benchmark (JUSTIFIED)

**Severity:** P3  
**Classification:** JUSTIFIED  
**Files:**

- `df_mlx/grad_utils.py` — `clip_grad_norm_tree()` (canonical, with NaN-zeroing)
- `df_mlx/training_ops.py` — `clip_grad_norm()` (thin wrapper → `grad_utils`)
- `df_mlx/benchmark_sync_barriers.py` — `clip_grad_norm()` (independent reimplementation, simplified)

### Justification

`benchmark_sync_barriers.py` is a standalone benchmark script that intentionally uses a minimal implementation to isolate performance characteristics. It does not participate in training and should not depend on the training module graph. The simplification (no NaN-zeroing) is intentional for benchmarking.

**Recommendation:** Add a comment in `benchmark_sync_barriers.py` noting this is intentionally independent. No code change needed.

---

## DUP-4.6 — Signal Handlers (JUSTIFIED)

**Severity:** P3  
**Classification:** JUSTIFIED  
**Files:**

- `df_mlx/training_signals.py` — `_handle_sigint()` (saves checkpoints, handles double-CTRL-C)
- `df_mlx/datastore.py` — `_signal_handler()` (sets `_shutdown_requested` flag for graceful I/O)

### Justification

These serve fundamentally different purposes:
- **Training signals:** Complex checkpoint-save logic, double-SIGINT escalation to abort
- **Datastore signals:** Simple boolean flag for graceful loop termination

The only "duplication" is that both register a SIGINT handler. The logic is entirely different. No consolidation needed.

---

## Items Investigated and Cleared

These patterns were investigated during the audit and determined to be **not duplicated**:

| Pattern | Finding |
|---|---|
| `discriminator.py:compute_discriminator_loss` vs `loss.py:discriminator_loss` | Proper wrapper — adds `stop_gradient`, forward passes, returns loss dict |
| `build_audio_cache.py:read_file_list` | Thin wrapper around `file_lists.read_file_list()` with preset args |
| `test_feature_ops_equivalence.py:_legacy_create_erb_filterbank` | Intentional test fixture for regression testing |
| `hardware.py` in `df/` vs `df_mlx/` | Cross-backend port (CUDA/MPS vs MLX) — different target platforms |

---

## Prioritized Action Items

1. **DUP-4.1 (ERB filterbank):** Audit immediately — potential correctness impact on training. Quantify formula divergence, determine canonical implementation, unify.
2. **DUP-4.3 (loss functions):** Consolidate `train.py` loss functions into `loss.py` to eliminate semantic overlap and reduce maintenance surface.
3. **DUP-4.2 (whisper adapter):** Extract shared whisper code into common module. Largest LOC savings (~800 lines).
4. **DUP-4.4 (pytorch conversion):** Deprecate `train.py` version, canonicalize `convert.py` version.
