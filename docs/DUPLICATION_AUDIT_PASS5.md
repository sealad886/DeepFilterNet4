# Duplication Audit — Pass 5

**Date:** 2026-02-19  
**Scope:** `DeepFilterNet/df_mlx/`, `DeepFilterNet/df/`  
**Method:** Full function/class inventory, cross-file signature matching, semantic comparison  
**Predecessor:** Pass 4 (DUP-4.1–4.6); Pass 3 (checkpoint×5, WarmupCosine, count_params×3, LossConfig/GANConfig, TrainConfig/TrainingConfig, load_audio variants)

---

## Summary

| ID | Severity | Classification | Description |
|---|---|---|---|
| DUP-5.1 | **P1** | UNNECESSARY — deferred | `sepm.py` 100% copy between `df/` and `df_mlx/` (~500 LOC pure NumPy) |
| DUP-5.2 | **P2** | UNNECESSARY — deferred | `prepare_data.py:mix_audio` semantic duplicate of `augment_ext._mix_audio_python` |
| DUP-5.3 | **P2** | UNNECESSARY — deferred | Critical band constant arrays duplicated within `sepm.py` (internal) |

**Result:** No fixes implemented. All findings deferred due to architectural constraints or insufficient risk/reward for this pass.

---

## DUP-5.1 — sepm.py Exact Copy Between df/ and df_mlx/

**Severity:** P1  
**Type:** Exact  
**Classification:** UNNECESSARY — deferred (architectural constraint)  
**Files:**

- `df/sepm.py` (501 lines)
- `df_mlx/sepm.py` (507 lines — 6-line docstring added, function bodies identical)

### What's duplicated

Both files are **100% pure NumPy** — no PyTorch, no MLX, no backend-specific code. They implement identical speech enhancement metrics: `SNRseg`, `fwSNRseg`, `lpcoeff`, `llr`, `findLocPeaks`, `wss`, `composite`, `extractOverlappedWindows`. Both import only `numpy`, `pesq`, `scipy.linalg.toeplitz`, and `scipy.signal.stft`.

`df_mlx/sepm.py` explicitly states: *"intentionally mirrored from /df/sepm.py so df_mlx does not depend on df internals while preserving equivalent metric behavior."*

### Usage paths

| Copy | Callers |
|---|---|
| `df/sepm.py` | `df/evaluation_utils.py:27` — `from df.sepm import composite as composite_py` |
| `df_mlx/sepm.py` | `df_mlx/evaluation.py:424,467` — `from df_mlx.sepm import composite` |

### Drift risk

**Moderate.** Any bugfix or enhancement applied to one copy will not automatically propagate to the other. The functions are mathematically complex (LPC, critical band filtering) and subtle bugs could diverge silently. Currently both copies are identical.

### Why deferred

`df_mlx/` has **zero imports** from `df/` — this is a deliberate architectural boundary. Consolidation requires either:

1. Creating a shared `df_common/` package (new package infrastructure)
2. Breaking the boundary by importing `df.sepm` from `df_mlx/`
3. Moving `sepm.py` to a neutral location (e.g., project root)

All options require architectural discussion beyond a duplication audit pass.

### Recommendation

Add a CI check or test that verifies byte-for-byte equivalence of the two files (excluding the docstring) to prevent silent drift until a shared package is established.

---

## DUP-5.2 — prepare_data.py:mix_audio vs augment_ext._mix_audio_python

**Severity:** P2  
**Type:** Semantic  
**Classification:** UNNECESSARY — deferred (behavioral differences)  
**Files:**

- `df_mlx/prepare_data.py:78-120` — `mix_audio()` (43 lines)
- `df_mlx/augment_ext.py:157-193` — `_mix_audio_python()` (37 lines)

### What's duplicated

Both implement the same core algorithm: power-based SNR scaling, noise repeat/trim to match clean signal length, and anti-clipping normalization. The core computation is identical:

```python
clean_power = np.mean(clean**2) + 1e-10
noise_power = np.mean(noise**2) + 1e-10
target_noise_power = clean_power / (10 ** (snr_db / 10))
noise_scaled = noise * np.sqrt(target_noise_power / noise_power)
noisy = clean + noise_scaled
```

### Behavioral differences

| Aspect | `prepare_data.py` | `augment_ext.py` |
|---|---|---|
| `gain_db` parameter | Not supported | Supported (default 0.0) |
| Return type | 2-tuple `(noisy, clean_out)` | 3-tuple `(clean_out, noise_scaled, noisy)` |
| Clipping threshold | 0.99 (leaves headroom) | 1.0 (full range) |
| What gets rescaled | Only `noisy` and `clean` | All three signals |

### Usage

- `prepare_data.py:mix_audio` — called by `benchmark_workers.py`, tested in `test_datastore.py`
- `augment_ext.py:mix_audio` (delegates to Rust or `_mix_audio_python`) — called by `dynamic_dataset.py`

### Why deferred

The behavioral differences (return tuple shape, normalization threshold, gain handling) mean callers expect different interfaces. Consolidation requires:
1. Deciding canonical normalization behavior
2. Updating callers for different return shape
3. Verifying `prepare_data.py` callers don't depend on the 0.99 headroom

Risk/reward ratio insufficient for Pass 5. **No silent drift risk** — these are intentionally different implementations optimized for their respective use cases.

---

## DUP-5.3 — Critical Band Constants Duplicated Within sepm.py

**Severity:** P2  
**Type:** Exact  
**Classification:** UNNECESSARY — deferred  
**Files:**

- `df_mlx/sepm.py:70-96` and `df_mlx/sepm.py:313-339` (same file, two functions)
- `df/sepm.py:66-92` and `df/sepm.py:289-315` (same file, two functions)

### What's duplicated

A 25-element `cent_freq` array and 25-element `bandwidth` array (critical band center frequencies and bandwidths) are defined identically inside both `fwSNRseg()` and `wss()` within each sepm.py file. This is ~50 LOC of inline constant duplication per file.

### Why deferred

1. These constant arrays are part of a vendored reference implementation. Modifying the structure of vendored code is generally avoided.
2. Since DUP-5.1 means the file itself is duplicated, fixing internal duplication in only one copy would break the mirror property.
3. Impact is purely cosmetic — no functional or drift risk.

---

## Items Investigated and Cleared

These patterns were investigated and determined to be **not duplication** or already **previously identified**:

| Pattern | Finding |
|---|---|
| `read_file_list` in 5 files | All are thin 1-line wrappers delegating to canonical `file_lists.read_file_list()` with preset kwargs — proper delegation |
| `init_model` in deepfilternet.py/2.py/model.py | Different models, different parameters and return types — not duplication |
| `compute_stft` (feature_ops.py) vs `stft` (ops.py) | NumPy (data prep) vs MLX (inference) — justified backend separation |
| `dnsmos_proxy.py` in df/ vs df_mlx/ | PyTorch vs MLX port — justified backend port |
| `stoi.py` in df/ vs df_mlx/ | PyTorch vs MLX port — justified backend port |
| `training_*` modules (post-decomposition) | Clean extraction, no new duplication introduced by decomposition |
| `compute_mrstft_loss` (training_waveform.py) | Delegates to external loss_fn callable, not reimplementing spectral loss |
| `count_params` in deepfilternet*.py `__main__` blocks | Local test/demo helpers — previously identified in Pass 3 |
| `load_audio` / `load_audio_file` variants | Previously identified as "load_audio variants" in Pass 3 |
| Checkpoint save/load ×6+ | Previously identified in Pass 3 (count has grown but finding is not new) |
| `LossConfig` / `TrainConfig` / `GANConfig` config class proliferation | Previously identified in Pass 3 |
| `WarmupCosineSchedule` vs `CosineScheduler` | Previously identified in Pass 3 |

---

## Previously Deferred Items (Carried Forward)

These items were identified in prior passes and remain unresolved. They are **not re-reported** — see the respective pass documents for full details.

| ID | Severity | Status | Summary |
|---|---|---|---|
| DUP-4.1 | P0 | OPEN | ERB filterbank formula divergence (Glasberg & Moore 1990 vs 1983) |
| DUP-4.2 | P1 | OPEN | `whisper_adapter.py` duplicated across `df/` and `df_mlx/` (~800 LOC) |
| DUP-4.3 | P1 | OPEN | Spectral loss overlap between `train.py` and `loss.py` |
| DUP-4.4 | P2 | OPEN | Dual `load_pytorch_checkpoint` with different safety levels |
| DUP-4.5 | P3 | JUSTIFIED | `clip_grad_norm` in benchmark — intentional independence |
| DUP-4.6 | P3 | JUSTIFIED | Signal handlers — fundamentally different purposes |
| Pass 3 | Various | OPEN | checkpoint save/load, WarmupCosine/CosineScheduler, count_params, config classes, load_audio variants |

---

## Fixes Applied

**None.** No P0 or safely-fixable P1 findings in this pass.

- DUP-5.1 requires architectural change (shared package or boundary-breaking import)
- DUP-5.2 has behavioral differences that make mechanical consolidation risky
- DUP-5.3 modifies vendored reference code and would break mirror property given DUP-5.1

---

## Residual Risks

1. **sepm.py drift** (DUP-5.1): If either `df/sepm.py` or `df_mlx/sepm.py` is modified without updating the other, metric evaluation could silently diverge between backends. **Mitigation:** Add a CI sync-check test.
2. **mix_audio divergence** (DUP-5.2): If the SNR mixing algorithm is improved in `augment_ext.py` (the actively-maintained training path), `prepare_data.py` will not benefit. **Impact:** Low — `prepare_data.py` is used for offline preprocessing, not live training.

---

## Completion Status

Pass 5 shows expected diminishing returns. Three genuinely new findings were identified (DUP-5.1, 5.2, 5.3), all at P1–P2 severity. None meet the criteria for safe in-pass implementation. The most impactful remaining duplication in the codebase is carried forward from prior passes (DUP-4.1 ERB formula divergence at P0).
