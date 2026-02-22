# Loss Functions, Weight Updates & Checkpoints

Technical reference for the MLX DeepFilterNet4 training pipeline. Covers every loss component, weight update path, and checkpoint save location used by `train_dynamic.py` and its extracted modules.

---

## 1. Overview

The MLX training pipeline (`df_mlx/train_dynamic.py`) computes a composite loss $L_{total}$ from multiple components, runs backpropagation, clips gradients, and applies optimizer updates. Checkpoints are saved at configurable intervals, on interrupts, and at the end of training.

The total generator loss is:

$$L_{total} = L_{spec} + L_{mrstft} + w_{gan} \cdot L_{gen} + w_{fm} \cdot L_{fm} + w_{awesome} \cdot L_{awesome} + w_{vad} \cdot L_{vad} + w_{speech} \cdot L_{speech} + w_{vad\_reg} \cdot L_{vad\_reg}$$

Each component is optionally enabled by config flags. The discriminator has a separate loss and optimizer (§2.5, §3.6).

---

## 2. Loss Components

### 2.1 Spectral Loss

| | |
|---|---|
| **Source** | [df_mlx/train.py](../DeepFilterNet/df_mlx/train.py#L346-L377) — `spectral_loss()` |
| **Measures** | Combined magnitude L1 and complex (real + imaginary) L1 between predicted and target spectrograms. |
| **Always active** | Yes — this is the base loss, not optional. |

**Formula:**

$$L_{spec} = (1 - \alpha) \cdot \text{mean}\!\bigl(\lvert\hat{M} - M\rvert\bigr) + \alpha \cdot \text{mean}\!\bigl(\lvert\hat{R} - R\rvert + \lvert\hat{I} - I\rvert\bigr)$$

where $\alpha = 0.5$ by default, $\hat{M}$ = predicted magnitude, $M$ = target magnitude, $R, I$ = real/imaginary components.

**Details:**
- Casts all inputs to FP32 internally for numerical stability.
- $\varepsilon = 10^{-8}$ inside `mx.sqrt` prevents zero gradients on silence.
- Valid for all data including silence.

**Effect of weight change:** Increasing emphasises spectral fidelity; decreasing makes room for perceptual losses (GAN, awesome).

---

### 2.2 Multi-Resolution STFT Loss

| | |
|---|---|
| **Source** | [df_mlx/train.py](../DeepFilterNet/df_mlx/train.py#L416-L575) — `MultiResolutionSTFTLoss` class |
| **Measures** | Spectral fidelity at multiple frequency resolutions simultaneously. |
| **Operates on** | Waveforms (requires iSTFT conversion from spectrogram domain first; see §2.11). |

**Formula:**

$$L_{mrstft} = \frac{1}{N_{res}} \sum_{n} \Bigl[ f_{mag} \cdot \lVert M_n^\gamma - \hat{M}_n^\gamma \rVert_2^2 + f_{complex} \cdot \bigl(\lVert R_n - \hat{R}_n \rVert_2^2 + \lVert I_n - \hat{I}_n \rVert_2^2\bigr) \Bigr]$$

**Config options:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `use_mrstft_loss` | `bool` | Enable/disable |
| `mrsl_fft_sizes` | `tuple[int, ...]` | FFT sizes (default: 512, 1024, 2048) |
| `mrsl_gamma` | `float` | Magnitude compression exponent ($\gamma < 1$ compresses dynamic range; 0.3 = heavy compression for quiet details) |
| `mrsl_factor` | `float` | Weight for magnitude loss |
| `mrsl_f_complex` | `float \| None` | Weight for complex loss (`None` disables) |

**Details:**
- Complex loss uses compressed-complex formulation: `mag_comp / mag * (real, imag)`. This avoids `arctan2` whose gradient explodes as $O(1/\varepsilon)$ for near-silent bins; direct division has stable gradients of $O(1/\sqrt{\varepsilon})$.
- `eps = 1e-12` guards silence.
- Valid for all data.

**Effect of weight change:** Higher weight improves multi-scale spectral accuracy; too high can overshadow perceptual losses.

---

### 2.3 Generator Loss

| | |
|---|---|
| **Source** | [df_mlx/loss.py](../DeepFilterNet/df_mlx/loss.py#L830-L846) — `generator_loss()` |
| **Measures** | How well the generator fools the discriminator (adversarial signal). |

**Formula:**

$$L_{gen} = \frac{1}{N_d} \sum_d \text{mean}(-f_d)$$

where $f_d$ = discriminator output on fake (generated) samples.

**Config options:**

| Parameter | Description |
|-----------|-------------|
| `gan_weight` | Multiplier applied to $L_{gen}$ in the total loss |
| `gan_enabled` | Master switch for GAN training |
| `gan_start_epoch` | Delayed activation — GAN losses are zero before this epoch |

**Details:**
- Hinge loss variant: generator wants discriminator outputs to be large (close to real).
- Only active when `gan_active = True` and discriminator is not `None`.

**Effect of weight change:** Increasing drives more perceptually pleasing output; too high causes training instability. Decreasing favours reconstruction fidelity over perceptual quality.

---

### 2.4 Feature Matching Loss

| | |
|---|---|
| **Source** | [df_mlx/loss.py](../DeepFilterNet/df_mlx/loss.py#L750-L794) — `FeatureMatchingLoss` class |
| **Measures** | L1 distance between discriminator intermediate feature maps on real vs. fake samples. |

**Formula:**

$$L_{fm} = \frac{f_{fm}}{N \cdot L} \sum_{n,l} \text{mean}\!\bigl(\lvert F_{n,l}^{real} - F_{n,l}^{fake}\rvert\bigr)$$

where $f_{fm}$ is the class `factor` (default 2.0), $N$ = number of sub-discriminators, $L$ = number of layers per discriminator.

**Config options:**

| Parameter | Description |
|-----------|-------------|
| `fm_weight` | Multiplier in the total loss |

**Details:**
- Only active when GAN is active **and** `fm_weight > 0`.
- Provides stable gradients to the generator — less prone to mode collapse than pure adversarial loss.
- Helps the generator match internal discriminator representations, not just fool the output layer.

---

### 2.5 Discriminator Loss

| | |
|---|---|
| **Source** | [df_mlx/loss.py](../DeepFilterNet/df_mlx/loss.py#L801-L827) — `discriminator_loss()` |
| **Measures** | Discriminator's ability to distinguish real from generated samples. |

**Formula:**

$$L_{disc} = \frac{1}{N_d} \sum_d \Bigl[ \text{mean}\!\bigl(\max(1 - r_d,\; 0)\bigr) + \text{mean}\!\bigl(\max(1 + f_d,\; 0)\bigr) \Bigr]$$

Hinge loss: real outputs should be $> 1$, fake outputs should be $< -1$.

**Config options:**

| Parameter | Description |
|-----------|-------------|
| `gan_disc_update_freq` | Update discriminator every N generator steps |
| `gan_disc_grad_clip` | Gradient clipping norm for discriminator |

**Important:** This loss is used **only** for discriminator weight updates via a separate optimizer (`disc_optimizer`). It is **not** part of the generator's `total_loss`.

---

### 2.6 Awesome Loss

| | |
|---|---|
| **Source** | [df_mlx/training_losses.py](../DeepFilterNet/df_mlx/training_losses.py#L398-L499) — `_compute_awesome_losses()` |
| **Measures** | Speech-preserving contrastive quality via three sub-components. |

**Sub-components:**

| Sub-loss | Formula | Purpose |
|----------|---------|---------|
| Speech preservation | $L_{speech} = \text{mean}\!\bigl(\lvert\log(1{+}\lvert\hat{S}\rvert) - \log(1{+}\lvert S\rvert)\rvert \cdot m \cdot w_{proxy}\bigr)$ | Preserve speech content |
| Noise suppression | $L_{noise} = \text{mean}\!\bigl(\lvert\log(1{+}\lvert\hat{S}\rvert)\rvert \cdot (1 - m)\bigr)$ | Suppress noise-dominant bins |
| Temporal smoothness | $L_{smooth} = \text{mean}\!\bigl(\lvert\hat{S}_{t+1} - \hat{S}_t\rvert \cdot (1 - m_t)\bigr)$ | Reduce temporal artefacts |

**Total:**

$$L_{awesome} = L_{speech} + L_{noise} + 0.2 \cdot L_{smooth}$$

**Mask computation:** $m = \sigma\!\bigl(\text{sharpness} \cdot (\text{clean\_log} - \text{noise\_log})\bigr)$, clamped to $\pm 30$ logits, then `stop_gradient`.

**Proxy frame:** Per-frame speech presence weighting derived from z-scored log-energy, modulation energy, and musicness gate. Applied with `stop_gradient` so it acts as a non-differentiable weight.

**Config options:**

| Parameter | Description |
|-----------|-------------|
| `use_awesome_loss` | `bool` — enable/disable |
| `awesome_weight` | Multiplier in total loss |
| `awesome_mask_sharpness` | Controls sigmoid steepness of the speech/noise mask |

**Edge cases:**
- Single-frame input: `smooth_loss = 0` (no temporal derivative).
- Silence: $\varepsilon = 10^{-8}$ in `_log1p_mag` prevents log-of-zero.
- Music: musicness gate (spectral flatness + temporal flux) downweights music-like content.

**Effect of weight change:** Increasing preserves speech better and suppresses noise harder; too high can over-constrain the model against spectral loss.

---

### 2.7 Pipeline Awesome Loss

| | |
|---|---|
| **Source** | [df_mlx/training_losses.py](../DeepFilterNet/df_mlx/training_losses.py#L640-L860) — `_compute_pipeline_awesome_losses()` |
| **Measures** | Improved version of awesome loss with better speech preservation and music suppression. |

**Key improvements over basic awesome (§2.6):**

1. **Minimum mask floor** (`_PIPELINE_MIN_MASK_FLOOR = 0.08`) — prevents complete speech suppression.
2. **Additive boosts** for low-energy (`+0.25`) and low-SNR (`+0.25`) speech, rather than multiplicative.
3. **Improved musicness detection** with vocal/instrument separation via `_compute_improved_musicness()`.
4. **Explicit music suppression loss** for instrumental content.
5. **Mask saturation metric** ($4 \cdot \text{mean}(m(1-m))$) — diagnostic only, excluded from backprop (zero gradient w.r.t. model params since `raw_mask` depends only on ground truth).

**Sub-components:**

| Sub-loss | Weight | Notes |
|----------|--------|-------|
| Speech preservation | 1.0 | Same formula as awesome but with mask floor |
| Noise suppression | 1.0 | Same as awesome |
| Temporal smoothness | `_PIPELINE_ARTIFACT_SMOOTH_WEIGHT = 0.3` | Stronger than awesome's 0.2 |
| Music suppression | `_PIPELINE_MUSIC_SUPPRESSION_WEIGHT = 1.5` | Penalises output energy where instrumental music is detected |
| Mask saturation | (excluded) | Diagnostic metric only |

**Total:**

$$L_{pipeline} = L_{speech} + L_{noise} + 0.3 \cdot L_{smooth} + 1.5 \cdot L_{music}$$

**Music suppression formula:**

$$L_{music} = \text{mean}\!\bigl(\lvert\log(1{+}\lvert\hat{S}\rvert)\rvert \cdot w_{instrument} \cdot (1 - m)\bigr)$$

**Proxy frame:** Higher floor (0.15), additive boosts, vocal gate restores music gate for vocal-like content.

**Config:** `use_pipeline_awesome_loss` (bool). Shares `awesome_weight` multiplier. Mutually exclusive with basic awesome in practice.

---

### 2.8 VAD Loss

| | |
|---|---|
| **Source** | [df_mlx/training_losses.py](../DeepFilterNet/df_mlx/training_losses.py#L152-L199) — `_compute_vad_loss()` |
| **Measures** | Decrease in Voice Activity Detection probability after enhancement. |

**Formula:**

$$L_{vad} = \text{mean}\!\bigl(\max(p_{ref} - p_{out} - \text{margin},\; 0) \cdot \text{gate}\bigr)$$

where:

$$\text{gate} = \text{speech\_gate} \times \text{snr\_gate}$$

- $\text{speech\_gate} = \text{clip}\!\bigl(\frac{p_{ref} - \text{threshold}}{1 - \text{threshold}},\; 0, 1\bigr)$ — high when reference has speech.
- $\text{snr\_gate} = \sigma\!\bigl(\frac{\text{SNR} - \text{snr\_gate\_db}}{\text{snr\_gate\_width}}\bigr)$ — sigmoid on SNR.
- $p_{ref}, p_{out}$ computed from z-scored speech-band log-energy.

**Config options:**

| Parameter | Description |
|-----------|-------------|
| `use_vad_loss` | `bool` — enable/disable |
| `vad_weight` | Multiplier in total loss |
| `vad_threshold` | Speech gate activation threshold |
| `vad_margin` | Allowed drop in VAD probability before penalty |
| `vad_snr_gate_db` | SNR centre for sigmoid gate |
| `vad_snr_gate_width` | SNR sigmoid width |
| `vad_z_threshold` | Z-score threshold for VAD probability |
| `vad_z_slope` | Z-score sigmoid slope |

**Details:**
- Only penalises VAD **decrease** — no penalty if output has more voice activity.
- Gate is `stop_gradient` — non-differentiable weight.

**Effect of weight change:** Increasing makes the model more cautious about suppressing speech; decreasing allows more aggressive noise reduction.

---

### 2.9 Speech Band Log-Magnitude Loss

| | |
|---|---|
| **Source** | [df_mlx/training_losses.py](../DeepFilterNet/df_mlx/training_losses.py#L201-L230) — `_compute_speech_band_logmag_loss()` |
| **Measures** | L1 error in log-magnitude within the speech frequency band, weighted by VAD gate. |

**Formula:**

$$L_{speech} = \text{mean}\!\bigl(\lvert\bar{O} - \bar{C}\rvert \cdot \text{gate}\bigr)$$

where $\bar{O}, \bar{C}$ are speech-band averaged log magnitudes (via band mask summation divided by band bins).

**Config:** Active when `vad_speech_loss_weight > 0` within the VAD loss block. `speech_weight` is applied as a multiplier in `loss_fn`.

**Details:**
- Focuses model attention on preserving speech-band energy specifically.
- Casts to FP32 for stability.

---

### 2.10 VAD Regulariser

| | |
|---|---|
| **Source** | [df_mlx/training_losses.py](../DeepFilterNet/df_mlx/training_losses.py#L870-L947) — `_compute_vad_reg_loss()` |
| **Measures** | Sparse regularisation using speech ratio and musicness proxy gates. |

**Details:**
- Computes VAD decrease ($\max(p_{ref} - p_{out} - \text{margin}, 0)$), then gates it by `speech_ratio_gate × music_gate`.
- Internally calls `_compute_speech_band_logmag_loss` with the computed gate.
- Uses `_compute_proxy_gates` with `proxy_enabled=True` for speech ratio and musicness scoring.

**Config:**

| Parameter | Description |
|-----------|-------------|
| `use_vad_train_reg` | `bool` — enable/disable |
| `vad_reg_weight` | Multiplier in total loss |

**Effect of weight change:** Complements VAD loss (§2.8) with finer-grained speech-ratio–aware control. Increasing adds stronger regularisation pressure on speech-present frames.

---

### 2.11 MRSTFT Helper (Waveform Wrapper)

| | |
|---|---|
| **Source** | [df_mlx/training_waveform.py](../DeepFilterNet/df_mlx/training_waveform.py#L65-L100) — `compute_mrstft_loss()` |
| **Purpose** | Orchestrates the spectral → waveform → MRSTFT flow. |

**Details:**
- Converts complex spectrograms to waveforms via `specs_to_wavs()` (using iSTFT).
- Calls the `MultiResolutionSTFTLoss` instance (`loss_fn`).
- Handles optional FP32 stabilisation for the waveform domain computation.
- Not a standalone loss — it is the bridge between the spectrogram-domain model output and the waveform-domain MRSTFT loss.

---

## 3. Weight Update Paths

Six production weight update paths exist. Paths 1–4 use `mx.compile` for throughput; paths 5–6 are eager-mode fallbacks.

### 3.1 Compiled Step (no GAN, no grad accumulation)

| | |
|---|---|
| **Location** | [train_dynamic.py ~L1740](../DeepFilterNet/df_mlx/train_dynamic.py#L1748) — `compiled_step()` |
| **State bindings** | `[model.state, optimizer.state]` |

**Flow:**

```
loss_and_grad → clip_grad_norm → optimizer.update(model, grads)
```

All operations in a single `mx.compile` graph for maximum throughput.

### 3.2 Compiled Loss+Grad Step (with grad accumulation)

| | |
|---|---|
| **Location** | [train_dynamic.py ~L1790](../DeepFilterNet/df_mlx/train_dynamic.py#L1783) — `compiled_loss_and_grad_step()` |
| **State bindings** | `[model.state]` (no optimizer) |

**Flow:**

```
loss_and_grad → return grads (no optimizer update)
```

Gradients are accumulated across N microbatches in eager code. After N batches:

```
scale_grads(1/N) → clip_grad_norm → _tree_all_finite check → optimizer.update
```

Non-finite grad check: `_tree_all_finite` skips the update if any NaN/Inf is detected.

### 3.3 Compiled GAN Step (experimental)

| | |
|---|---|
| **Location** | [train_dynamic.py ~L1830](../DeepFilterNet/df_mlx/train_dynamic.py#L1822) — `_compiled_gan_step()` |

Same as §3.1 but uses `loss_fn_gan` with GAN paths hardcoded as always-active. Required because `mx.compile` traces Python booleans at trace time — `if gan_active` would be captured as `False` during pre-GAN compilation and never re-traced. One-time correctness verification compares compiled vs. eager loss.

### 3.4 Compiled GAN Loss+Grad Step (experimental + grad accum)

| | |
|---|---|
| **Location** | [train_dynamic.py ~L1860](../DeepFilterNet/df_mlx/train_dynamic.py#L1860) — `_compiled_gan_loss_and_grad_step()` |

Same as §3.2 but uses `loss_fn_gan`.

### 3.5 Eager Mode Generator Update

| | |
|---|---|
| **Location** | [train_dynamic.py ~L3200–3260](../DeepFilterNet/df_mlx/train_dynamic.py#L3220) |

Used when compiled mode is disabled or batch size does not match the canonical shape.

**Flow:**

```
loss_and_grad → accumulate_grads → (after N batches) scale_grads(1/N) → clip_grad_norm → optimizer.update
```

- `clip_grad_norm` zeros non-finite grads (safe no-op update).
- Cached waveforms detached with `mx.stop_gradient` before discriminator update.

### 3.6 Eager Mode Discriminator Update

| | |
|---|---|
| **Location** | [train_dynamic.py ~L3350–3420](../DeepFilterNet/df_mlx/train_dynamic.py#L3307) |

**Flow:**

```
nn.value_and_grad(discriminator, disc_loss_wrapper) → clip_grad_norm → disc_optimizer.update
```

- Only runs when `did_optimizer_update` **and** `global_step % gan_disc_update_freq == 0`.
- Uses a separate optimizer (`disc_optimizer`) — never shares with the generator.
- Can use the compiled path (`compiled_disc_update_step`) when `experimental_compiled_gan` is active.

---

## 4. Checkpoint Save Locations

Seven save sites in the training pipeline (`train_dynamic.py` and `training_signals.py`):

| # | Trigger | Filename Pattern | Kind | Purpose |
|---|---------|------------------|------|---------|
| 1 | Step-based (`save_strategy == "steps"` and `global_step % save_steps == 0`) | `step_{global_step:06d}.safetensors` | `step` | Periodic intermediate saves |
| 2 | Best validation loss improvement | `best.safetensors` | `best` | Best model for deployment |
| 3 | End of epoch | `epoch_{epoch+1:03d}.safetensors` | `epoch_end` | Authoritative epoch completion |
| 4 | Final best (if final validation beats best) | `best.safetensors` | `best_final` | Updated best after final validation |
| 5 | Training complete (always) | `final.safetensors` | `final` | Final weights regardless of quality |
| 6 | Interrupt handler (SIGINT / SIGTERM) | `interrupted_epoch_{epoch+1:03d}.safetensors` | `interrupted` | Crash-safe resume point |
| 7 | Data stream checkpoint | `data_checkpoint_path` | N/A | MLX data stream position for resume |

**Checkpoint mechanics:**
- All saves use atomic temp → rename pattern (`save_checkpoint` in `training_checkpoints.py`).
- Generator and discriminator weights are saved in separate files.
- Optimizer state is saved in JSON alongside the weights.
- Validation runs after saves that trigger it.

---

## 5. Dead Code Inventory

The following loss functions and helpers are **not** used by the production training pipeline (`train_dynamic.py`):

### df_mlx/loss.py
- `SpectralLoss` — class-based spectral loss (production uses standalone `spectral_loss()` in train.py)
- `FusedSpectralLoss`
- `MaskLoss`, `MaskSpecLoss`
- `SiSdrLoss`, `SegmentalSiSdrLoss`, `SdrLoss`
- `DfAlphaLoss`
- `CombinedLoss`
- ~~`ASRLoss`~~ — **removed** (placeholder that raised NotImplementedError)
- ~~`create_loss_fn`~~ — **removed** (factory with zero callers)

### df_mlx/train.py
- `snr_loss`, `lsnr_loss`, `combined_loss`
- `multi_resolution_stft_loss` (standalone function; production uses the `MultiResolutionSTFTLoss` class)

### df_mlx/discriminator.py
- ~~`compute_discriminator_loss`~~, ~~`compute_generator_loss`~~, ~~`create_discriminator`~~ — **removed** (zero callers; train_dynamic uses loss.py functions directly)

### df_mlx/whisper_adapter.py
- `compute_whisper_loss` — exported but never called from training

### df_mlx/checkpoint.py
- Entire file — used only by the legacy `Trainer` class

**Not dead:**
- `si_sdr` from `loss.py` **is** used in validation.
- Some dead-code items are used by benchmark scripts or `test_mlx_comprehensive.py`.

---

## 6. Duplicate Code: `loss_fn` vs `loss_fn_gan`

`train_dynamic.py` contains two nearly identical loss functions (~200 lines each):

| Function | GAN paths | Used by |
|----------|-----------|---------|
| `loss_fn` (L1201) | Gated by `gan_active` closure variable | `compiled_step`, `compiled_loss_and_grad_step`, eager mode |
| `loss_fn_gan` (L1382) | Hardcoded as always-active | `_compiled_gan_step`, `_compiled_gan_loss_and_grad_step` |

**Why they exist:** `mx.compile` captures Python booleans at trace time. If `loss_fn` is compiled when `gan_active == False`, the GAN branch is permanently dead in the compiled graph — even after `gan_active` flips to `True` at `gan_start_epoch`. The duplicate `loss_fn_gan` hardcodes `True` for GAN paths so the compiled graph always includes generator adversarial loss computation.

A one-time correctness check compares compiled-GAN vs. eager-GAN loss on the first GAN step to verify equivalence.
