# DeepFilterNet Enhancement Error Taxonomy

## Purpose

Catalog known failure modes in speech enhancement to guide model evaluation, loss design, and debugging. Each failure mode is grounded in the DeepFilterNet codebase with citations to specific files, functions, loss components, and incident reports.

---

## Failure Mode Categories

### 1. Speech Distortion

#### 1.1 Quiet Speech Attenuation

- **Description**: Low-energy speech segments are over-suppressed, producing unintelligible output. The model treats quiet speech as noise.
- **Root Cause**: Aggressive noise estimation when SNR is low; VAD misclassifying quiet speech as noise due to z-scored energy falling below threshold. When log-energy variance is near zero (silence-adjacent segments), z-scores become unstable and VAD probabilities unreliable.
- **Diagnostic Signals**:
  - SI-SDR degradation on low-SNR segments
  - VAD false-negative rate on quiet speech (`p_ref` high but `p_out` low in [`_compute_vad_probs()`](DeepFilterNet/df_mlx/train_dynamic.py#L556))
  - Mask values near 0 in speech-active regions
  - `energy_boost` saturated at 1.0 (indicates model detected low energy but couldn't compensate)
- **Relevant Metrics**: SI-SDR per SNR bucket, VAD precision/recall, `energy_boost` distribution
- **Mitigation**:
  - Minimum mask floor (`_PIPELINE_MIN_MASK_FLOOR`) prevents complete suppression — see [`_compute_pipeline_awesome_losses()`](DeepFilterNet/df_mlx/train_dynamic.py#L1081)
  - Additive energy boost for low-energy speech (`_PIPELINE_LOW_ENERGY_ADDITIVE`) — see [line ~1224](DeepFilterNet/df_mlx/train_dynamic.py#L1224)
  - Minimum variance floor (`_MIN_VARIANCE = 1e-4`) for z-score stability — see [INCIDENT_LOSS_VAD_AUDIT_2025_01.md](docs/INCIDENT_LOSS_VAD_AUDIT_2025_01.md), Bug 2
  - `f_under` weighting in PyTorch `MaskLoss` penalizes under-prediction more than over-prediction — see [`MaskLoss.forward()`](DeepFilterNet/df/loss.py#L244)

#### 1.2 Speech Over-Processing (Metallic/Robotic Artifacts)

- **Description**: Excessive spectral modification creates robotic or metallic quality. The enhanced speech sounds "processed" even when the original was clean.
- **Root Cause**: Complex spectral loss with high gamma compression can over-smooth fine spectral structure. Overly aggressive DF alpha scheduling at moderate SNR levels.
- **Diagnostic Signals**:
  - High spectral loss but low perceptual quality (PESQ)
  - DF alpha near 1.0 even at moderate SNR levels
  - Gamma-compressed magnitude spectra diverge from natural speech distribution
- **Relevant Metrics**: PESQ, spectral distortion per frequency band, DF alpha distribution
- **Codebase References**:
  - Gamma compression in [`SpectralLoss`](DeepFilterNet/df_mlx/loss.py#L96) controls trade-off between dynamic range and spectral fidelity
  - [`DfAlphaLoss`](DeepFilterNet/df_mlx/loss.py#L574) LSNR-based scheduling: below `lsnr_lo` suppress DF, above `lsnr_hi` enable fully
  - PyTorch [`SpectralLoss.forward()`](DeepFilterNet/df/loss.py#L141) with `factor_under` asymmetric weighting

#### 1.3 Speaker Identity Degradation

- **Description**: Enhanced speech loses speaker-specific characteristics (timbre, pitch contour).
- **Root Cause**: Over-aggressive spectral modifications that distort formant structure.
- **Diagnostic Signals**: Cosine similarity drop between clean and enhanced speaker embeddings.
- **Relevant Metrics**: Speaker embedding cosine similarity
- **Codebase References**: [`SpeakerContrastiveLoss`](DeepFilterNet/df/loss.py#L744) uses resemblyzer embeddings to preserve speaker identity during training.

---

### 2. Noise Residuals

#### 2.1 Musical Noise (Tonal Artifacts)

- **Description**: Random spectral peaks survive noise reduction, creating "bird chirp" or twinkling artifacts. Individual T-F bins are inconsistently masked across frames.
- **Root Cause**: Frame-by-frame independent mask estimation without temporal regularization. Mask values oscillate between high suppression and pass-through for noise-dominant bins.
- **Diagnostic Signals**:
  - Isolated high-energy bins in residual spectrogram
  - High temporal variance in mask values for noise-dominated regions
  - `smooth_loss` elevated in [`_compute_pipeline_awesome_losses()`](DeepFilterNet/df_mlx/train_dynamic.py#L1256)
- **Relevant Metrics**: Segmental spectral flatness deviation, temporal smoothness loss
- **Mitigation**:
  - Temporal smoothing loss (`_PIPELINE_ARTIFACT_SMOOTH_WEIGHT`) penalizes frame-to-frame output variation in noise regions — see [line ~1256](DeepFilterNet/df_mlx/train_dynamic.py#L1256)
  - `MaskLoss` optional `f_temporal` for temporal smoothing of mask predictions — see [`MaskLoss.__call__()`](DeepFilterNet/df_mlx/loss.py#L268)
  - Literature: Xu et al. 2015, Park & Lee 2017 (temporal smoothness regularization)

#### 2.2 Wind/Hiss Residual

- **Description**: Broadband noise components partially survive enhancement, leaving an audible noise floor.
- **Root Cause**: Insufficient low-frequency masking (wind: 0–500 Hz) or high-frequency masking bandwidth (hiss: 4–8 kHz). ERB bands that aggregate multiple frequency bins may under-suppress distributed noise.
- **Diagnostic Signals**:
  - Elevated noise floor in specific frequency ranges
  - Noise suppression ratio below target per band
  - Mask values insufficiently low in noise-dominated ERB bands
- **Relevant Metrics**: Noise reduction per frequency band, segmental SNR
- **Codebase References**:
  - ERB filterbank mask in [`MaskLoss`](DeepFilterNet/df/loss.py#L186) maps between ERB and frequency domains via `erb_fb` / `erb_inv_fb`
  - `f_max_idx` parameter limits loss computation to a maximum frequency bin — see [`MaskLoss.__init__()`](DeepFilterNet/df/loss.py#L196)

---

### 3. Non-Speech Signal Handling

#### 3.1 Music Leakage/Suppression

- **Description**: Music incorrectly treated as noise (suppressed) or music leaking through as speech. Instrumental music and vocals are handled differently.
- **Root Cause**: Music shares spectral characteristics with both speech (vocals) and noise (instruments). The musicness detector uses tonal analysis and spectral flux to distinguish, but edge cases remain.
- **Diagnostic Signals**:
  - `music_suppression` metric elevated
  - `musicness` score not matching ground truth (instrumental vs. vocal content)
  - Mask values during music-only segments near 0 (false suppression) or near 1 (leakage)
- **Relevant Metrics**: `music_suppression` loss, `musicness` score, `vocal_gate`, `instrument_gate`
- **Codebase References**:
  - [`_compute_musicness()`](DeepFilterNet/df_mlx/train_dynamic.py#L699): Basic musicness from tonal analysis + spectral flux gating
  - [`_compute_improved_musicness()`](DeepFilterNet/df_mlx/train_dynamic.py#L1012): Enhanced version with vocal/instrument separation
  - Music suppression loss in [`_compute_pipeline_awesome_losses()`](DeepFilterNet/df_mlx/train_dynamic.py#L1261): Penalizes output energy where instrumental music is detected
  - `_PIPELINE_MUSIC_SUPPRESSION_WEIGHT = 1.5` — see [line 137](DeepFilterNet/df_mlx/train_dynamic.py#L137)
  - `_AWESOME_MUSICNESS_THR`, `_AWESOME_MUSICNESS_WIDTH` control gate sensitivity — see [lines 122–123](DeepFilterNet/df_mlx/train_dynamic.py#L122)
- **Reference**: [INCIDENT_LOSS_VAD_AUDIT_2025_01.md](docs/INCIDENT_LOSS_VAD_AUDIT_2025_01.md)

#### 3.2 Background Speech Leakage

- **Description**: Interfering speakers' speech not fully suppressed, creating intelligible crosstalk.
- **Root Cause**: The model and VAD proxy treat all speech-like signals as target speech. Background speakers produce high `p_ref` values, suppressing the noise gate.
- **Diagnostic Signals**:
  - SI-SDR degradation in multi-speaker scenarios
  - `speech_ratio` elevated during non-target speaker segments
  - `proxy_frame` weighting incorrectly preserves interfering speech
- **Relevant Metrics**: Speaker separation SI-SDR, target speaker preservation ratio

---

### 4. Temporal Artifacts

#### 4.1 Speech Onset/Offset Clipping

- **Description**: Beginning or end of speech utterances clipped by delayed mask activation or premature deactivation.
- **Root Cause**: VAD probability transitions have finite slope (`vad_z_slope`); the sigmoid gate introduces lag at speech boundaries. Frame-level processing cannot capture sub-frame onsets.
- **Diagnostic Signals**:
  - VAD transition timing lag: `p_ref` rises before `p_out`
  - Mask rise/fall time exceeds one analysis frame
  - First/last few frames of utterances show excessive attenuation
- **Relevant Metrics**: Onset detection delay, offset trailing duration
- **Codebase References**:
  - VAD slope parameter controls transition speed: `vad_z_slope` in [`_compute_vad_probs()`](DeepFilterNet/df_mlx/train_dynamic.py#L556)
  - Modulation energy gate (`mod_energy`, `mod_gate`) may suppress steady-state onsets — see [line ~1216](DeepFilterNet/df_mlx/train_dynamic.py#L1216)

#### 4.2 Processing Discontinuities

- **Description**: Audible clicks or gaps at frame boundaries from overlap-add reconstruction.
- **Root Cause**: Inadequate overlap-add windowing, frame-level processing inconsistencies, or DF alpha changing abruptly between frames.
- **Diagnostic Signals**:
  - Waveform discontinuities at frame boundaries
  - Zero-crossing rate anomalies at hop boundaries
  - DF alpha changes > 0.5 between adjacent frames
- **Relevant Metrics**: Zero-crossing rate anomalies, waveform discontinuity magnitude
- **Codebase References**:
  - ISTFT reconstruction: [`Istft.forward()`](DeepFilterNet/df/loss.py#L80) with proper window and hop handling
  - Temporal smoothing loss reduces frame-boundary artifacts — see [line ~1256](DeepFilterNet/df_mlx/train_dynamic.py#L1256)

---

### 5. Model Convergence Failures

#### 5.1 GAN Mode Collapse

- **Description**: Generator produces limited output diversity; discriminator saturates. Enhanced outputs converge to a narrow spectral template regardless of input.
- **Root Cause**: Discriminator learns too quickly, providing uninformative gradients. Feature matching loss weight insufficient to stabilize generator training.
- **Diagnostic Signals**:
  - `disc_accuracy` near 0 or 1 (discriminator trivially solves task)
  - Generator loss plateaus while discriminator loss approaches 0
  - Feature matching loss diverges from discriminator loss trend
- **Relevant Metrics**: Discriminator accuracy balance, generator/discriminator loss ratio, feature matching loss
- **Codebase References**:
  - GAN hinge loss: [`discriminator_loss()`](DeepFilterNet/df_mlx/loss.py#L685), [`generator_loss()`](DeepFilterNet/df_mlx/loss.py#L718)
  - [`FeatureMatchingLoss`](DeepFilterNet/df_mlx/loss.py#L648) provides stable gradients via L1 distance on discriminator feature maps
  - PyTorch GAN losses: [`GeneratorLoss`](DeepFilterNet/df/loss.py#L858) supports lsgan/vanilla/hinge; [`DiscriminatorLoss`](DeepFilterNet/df/loss.py#L898)
  - Training mode convention: compiled before GAN activation, eager once GAN-active; never switch back mid-run — see [CONVENTIONS.md](docs/CONVENTIONS.md)

#### 5.2 Loss Divergence

- **Description**: Training losses explode, producing NaN/Inf values.
- **Root Cause**: Numerical instability in loss computation — division by near-zero in z-score normalization, log of zero, or unbounded gradients. Known triggers: near-silence inputs, extreme magnitude inputs, empty-array mean operations.
- **Diagnostic Signals**:
  - NaN/Inf in loss values
  - `grad_norm` exceeding `max_grad_norm` threshold consistently
  - Loss curve shows sudden spike before NaN
- **Relevant Metrics**: Loss curve, gradient norm trend, frequency of gradient clipping events
- **Codebase References**:
  - Gradient clipping: [`clip_grad_norm()`](DeepFilterNet/df_mlx/train_dynamic.py#L1668) with configurable `max_grad_norm`
  - `_MIN_VARIANCE = 1e-4` floor prevents z-score explosion — see [Bug 2 in INCIDENT_LOSS_VAD_AUDIT_2025_01.md](docs/INCIDENT_LOSS_VAD_AUDIT_2025_01.md)
  - Empty-array mean guard prevents NaN from `mx.mean([])` — see [Bug 3 in INCIDENT_LOSS_VAD_AUDIT_2025_01.md](docs/INCIDENT_LOSS_VAD_AUDIT_2025_01.md)
  - EPS constants (`1e-10` in MLX, `1e-12` in PyTorch) used throughout loss computations
  - Logit clamping: `_AWESOME_MASK_LOGIT_CLAMP`, `_VAD_LOGIT_CLAMP` prevent extreme sigmoid inputs

#### 5.3 Mask Saturation

- **Description**: Model learns trivially saturated masks (all 0 or all 1), losing nuanced suppression ability. Alternatively, masks remain near 0.5 (uncertain), providing no meaningful enhancement.
- **Root Cause**: Without regularization, the model can exploit mask extremes. The mask saturation penalty steers predictions toward confident values while MaskLoss ensures correctness.
- **Diagnostic Signals**:
  - `mask_saturation` metric elevated (high entropy = uncertain masks)
  - Mean mask value distribution bimodal at 0 and 1 (saturated) or peaked at 0.5 (uncertain)
  - Enhanced output sounds unprocessed (masks ≈ 1) or silent (masks ≈ 0)
- **Relevant Metrics**: `mask_saturation` loss, mean mask value, mask value histogram
- **Codebase References**:
  - Mask saturation penalty: `mask_entropy = mx.mean(raw_mask * (1.0 - raw_mask))` — see [`_compute_pipeline_awesome_losses()`](DeepFilterNet/df_mlx/train_dynamic.py#L1270)
  - `_PIPELINE_MASK_SATURATION_PENALTY = 0.1` — see [line 141](DeepFilterNet/df_mlx/train_dynamic.py#L141)
  - **Historical Bug**: Previous implementation _inverted_ the penalty, rewarding uncertainty instead of penalizing it — see [Bug 1 in INCIDENT_LOSS_VAD_AUDIT_2025_01.md](docs/INCIDENT_LOSS_VAD_AUDIT_2025_01.md) (SEVERITY: HIGH, now RESOLVED)

#### 5.4 Checkpoint/Resume Counter Mismatch

- **Description**: Training resumes from a checkpoint with inconsistent epoch/batch counters, causing silent divergence or off-by-one errors in epoch progress.
- **Root Cause**: Mixed units (optimizer steps vs. microbatches) in epoch control; inconsistent `batch_idx` semantics across save/load/resume paths. Model and data checkpoint mismatch may silently continue.
- **Diagnostic Signals**:
  - tqdm progress bar exceeds expected epoch total
  - Metrics at resume differ unexpectedly from pre-interruption values
  - Model/data checkpoint epoch mismatch warning in logs
- **Relevant Metrics**: `counter_semantics_version`, `micro_batches_completed`, `optimizer_steps_completed`
- **Codebase References**: [INCIDENT_TRAIN_CONTROL_RESUME_2026_02.md](docs/INCIDENT_TRAIN_CONTROL_RESUME_2026_02.md) — resolved with strict resume reconciliation and canonical counter semantics (v2)

---

### 6. Data Pipeline Failures

#### 6.1 Single-Frame Edge Cases

- **Description**: Audio segments with a single analysis frame cause NaN from frame-differencing operations (spectral flux, modulation energy, temporal smoothing).
- **Root Cause**: Operations like `x[:, 1:] - x[:, :-1]` produce empty arrays when `n_frames == 1`, and `mx.mean([])` returns NaN.
- **Diagnostic Signals**: NaN loss values on very short audio segments
- **Relevant Metrics**: Minimum segment length in dataset, NaN frequency per epoch
- **Codebase References**: [Bug 3 in INCIDENT_LOSS_VAD_AUDIT_2025_01.md](docs/INCIDENT_LOSS_VAD_AUDIT_2025_01.md) — fixed with `shape[1] > 1` guards in 5 locations including `_compute_musicness()`, `_compute_improved_musicness()`, `_compute_proxy_gates()`, `_compute_pipeline_awesome_losses()`

#### 6.2 Near-Silence Input Instability

- **Description**: Near-silence inputs cause z-score instability in VAD computation.
- **Root Cause**: When input is near-silence, variance of log-energy approaches zero, making `z = (x - mu) / sigma` produce extreme values.
- **Diagnostic Signals**: Extreme VAD probabilities (0 or 1) on silence, unstable training metrics during quiet passages
- **Relevant Metrics**: Sigma (standard deviation) distribution, z-score range
- **Codebase References**: [Bug 2 in INCIDENT_LOSS_VAD_AUDIT_2025_01.md](docs/INCIDENT_LOSS_VAD_AUDIT_2025_01.md) — fixed with `_MIN_VARIANCE = 1e-4` floor; see [`_compute_vad_probs()`](DeepFilterNet/df_mlx/train_dynamic.py#L586)

---

## Severity Classification

| Severity | Impact | Example |
|----------|--------|---------|
| **Critical** | Completely unintelligible output or training crash | Speech fully suppressed; NaN loss divergence; checkpoint resume mismatch |
| **Major** | Noticeable quality degradation | Metallic speech; music fully removed; GAN mode collapse |
| **Minor** | Subtle quality reduction | Slight hiss residual; minor onset clip; elevated mask uncertainty |
| **Cosmetic** | Only detectable in A/B comparison | Marginal spectral coloring; sub-frame onset lag |

---

## Metrics Reference

| Metric | Source | Use Case |
|--------|--------|----------|
| SI-SDR | [`SiSdrLoss`](DeepFilterNet/df_mlx/loss.py#L400), [`SiSdr`](DeepFilterNet/df/loss.py#L331) | Overall enhancement quality |
| Segmental SI-SDR | [`SegmentalSiSdrLoss`](DeepFilterNet/df_mlx/loss.py#L420), [`SegSdrLoss`](DeepFilterNet/df/loss.py#L376) | Local quality variation |
| PESQ | External | Perceptual quality |
| Spectral loss | [`SpectralLoss`](DeepFilterNet/df_mlx/loss.py#L96), [`SpectralLoss`](DeepFilterNet/df/loss.py#L141) | Spectral fidelity |
| Multi-res spectral loss | [`MultiResSpecLoss`](DeepFilterNet/df/loss.py#L103) | Multi-scale spectral fidelity |
| Mask loss | [`MaskLoss`](DeepFilterNet/df_mlx/loss.py#L210), [`MaskLoss`](DeepFilterNet/df/loss.py#L186) | Mask prediction accuracy |
| DF Alpha loss | [`DfAlphaLoss`](DeepFilterNet/df_mlx/loss.py#L574), [`DfAlphaLoss`](DeepFilterNet/df/loss.py#L286) | Deep filtering schedule correctness |
| ASR loss | [`ASRLoss`](DeepFilterNet/df/loss.py#L417) | Speech intelligibility preservation |
| Feature matching | [`FeatureMatchingLoss`](DeepFilterNet/df_mlx/loss.py#L648), [`FeatureMatchingLoss`](DeepFilterNet/df/loss.py#L729) | GAN training stability |
| Speaker similarity | [`SpeakerContrastiveLoss`](DeepFilterNet/df/loss.py#L744) | Speaker identity preservation |
| `music_suppression` | [`_compute_pipeline_awesome_losses()`](DeepFilterNet/df_mlx/train_dynamic.py#L1261) | Music handling quality |
| `mask_saturation` | [`_compute_pipeline_awesome_losses()`](DeepFilterNet/df_mlx/train_dynamic.py#L1270) | Mask prediction confidence |
| `musicness` | [`_compute_musicness()`](DeepFilterNet/df_mlx/train_dynamic.py#L699) | Music content detection |
| VAD precision/recall | [`_compute_vad_probs()`](DeepFilterNet/df_mlx/train_dynamic.py#L556) | Speech detection accuracy |
| Gradient norm | [`clip_grad_norm()`](DeepFilterNet/df_mlx/train_dynamic.py#L1668) | Training stability |
| Temporal smoothness | [`_compute_pipeline_awesome_losses()`](DeepFilterNet/df_mlx/train_dynamic.py#L1256) | Temporal artifact control |
| `energy_boost` / `snr_boost` | [`_compute_pipeline_awesome_losses()`](DeepFilterNet/df_mlx/train_dynamic.py#L1224) | Low-signal speech compensation |

---

## References

- [INCIDENT_LOSS_VAD_AUDIT_2025_01.md](docs/INCIDENT_LOSS_VAD_AUDIT_2025_01.md) — Loss/VAD correctness audit (4 bugs fixed)
- [INCIDENT_TRAIN_CONTROL_RESUME_2026_02.md](docs/INCIDENT_TRAIN_CONTROL_RESUME_2026_02.md) — Checkpoint counter mismatch incident
- [BENCHMARK_CONTRACT.md](docs/BENCHMARK_CONTRACT.md) — Baseline metric definitions and regression gates
- [CONVENTIONS.md](docs/CONVENTIONS.md) — Repository conventions including training mode semantics
