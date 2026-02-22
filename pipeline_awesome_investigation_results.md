# Pipeline Awesome: Completed Investigations and Recommended Path

Date: 2026-02-20

## What was completed

I completed both investigations proposed in `pipeline_awesome_investigation.md`:

1. **Mask vs VAD behavior during inference** (real checkpoint, real audio files)
2. **Loss component contribution analysis during training** using the requested datastore and MLX loader:
   - cache: `/Users/andrew/DataDump/datasets/mlx_datastore`
   - loader path: `MLXDataStream`

---

## Investigation 1 — Learned mask vs VAD proxy behavior

### Method

A probe loaded checkpoint `gated_gan_from_75/best.safetensors`, ran inference-like forward passes, and compared:

- frame-mean ERB-expanded mask values,
- VAD-like speech probability proxy (z-scored log-band energy sigmoid),
- speech-band attenuation (dB) in speech vs non-speech frames.

Artifacts:
- Probe script: `.copilot-tracking/research/20260220-pipeline-awesome-mask-vad-probe.py`
- Probe results: `/tmp/investigation1_mask_vad_results.json`

### Key quantitative result

Aggregate across 5 files:

- `mask_mean_speech = 0.0403`
- `mask_mean_nonspeech = 0.0419`
- ratio `mask_nonspeech_over_speech = 1.061`
- `mask_vad_corr = 0.227`
- attenuation: `att_db_mean_speech = -2.49 dB`, `att_db_mean_nonspeech = -2.70 dB`

Interpretation: the learned mask is **not strongly speech-selective** in this checkpoint; non-speech is not being dramatically suppressed relative to speech in a VAD-like way.

Citations:
- `/tmp/investigation1_mask_vad_results.json`
- `DeepFilterNet/df_mlx/model.py:1139` (`DfNet4.__call__` inference graph)
- `DeepFilterNet/df_mlx/model.py:1286` (`DfNet4.enhance` path)
- `scripts/compare_models.py:503` (uses `self.model.enhance(audio_mx)`)

---

## Investigation 2 — Loss component contribution analysis

### Method

Ran a short training probe with `pipeline_awesome`:

- `--cache-dir /Users/andrew/DataDump/datasets/mlx_datastore`
- default MLX data path (no `--no-mlx-data`)
- short run limits for observability (`--max-train-batches 20`, `--max-valid-batches 6`)

Observed runtime confirms:

- `Loaded config from cache: /Users/andrew/DataDump/datasets/mlx_datastore`
- `Using MLXDataStream (workers=2, prefetch=4)`

From validation metrics in run log:

- `spec = 0.2795`
- `awesome = 0.0348`
- `vad = 0.0064`
- configured weights shown at startup: `awesome_loss_weight=0.4`, `vad_loss_weight=0.05`

Approx weighted contribution to total objective:

$$
L \approx L_{spec} + 0.4\,L_{awesome} + 0.05\,L_{vad}
$$

So approximately:

- spec: $0.2795$
- awesome: $0.4\times0.0348=0.01392$
- vad: $0.05\times0.0064=0.00032$
- estimated total: $0.29374$ (matches logged valid loss $\approx 0.2937$)

This implies rough share:

- **spec ~95.1%**
- **awesome ~4.7%**
- **vad ~0.1%**

Interpretation: the objective is currently dominated by spectral reconstruction; VAD consistency is too weak to enforce hard non-speech suppression behavior.

Citations:
- `/tmp/investigation2_loss_contrib.log`
- `DeepFilterNet/df_mlx/training_losses.py:877` (`speech_loss` term)
- `DeepFilterNet/df_mlx/training_losses.py:880` (`noise_loss` term)
- `DeepFilterNet/df_mlx/training_losses.py:891` (`music_suppression_loss`)
- `DeepFilterNet/df_mlx/training_losses.py:902` (`total_loss` composition)
- `DeepFilterNet/df_mlx/train_dynamic.py:2264` (awesome/pipeline term added)
- `DeepFilterNet/df_mlx/train_dynamic.py:2267` (VAD term weighting)

---

## Additional finding (important)

There is a diagnostics mismatch for `pipeline_awesome` mask stats in validation:

- detailed mask stats accumulation is guarded by `if use_awesome_loss and emit_detailed_metrics:`
- this excludes `pipeline_awesome`, so reported validation `mask/proxy/...` can remain zero-like even when pipeline loss is active.

Citations:
- `DeepFilterNet/df_mlx/train_dynamic.py:2362`
- `DeepFilterNet/df_mlx/train_dynamic.py:2532`
- `DeepFilterNet/df_mlx/train_dynamic.py:2548`

---

## Option set (ranked)

### Option A (Best immediate): Explicit inference-time speech gate (hard suppression)

Add/enable explicit VAD-conditioned attenuation in inference output path so non-speech regions are actively suppressed rather than implicitly learned.

- Pros: directly matches your stated goal (speech focus, stronger background suppression)
- Cons: threshold/transition tuning needed to avoid speech chopping

Why ranked #1: investigation #1 shows mask is not sufficiently speech-selective; investigation #2 shows training VAD signal is too weak to guarantee this behavior.

Relevant existing building blocks:
- `df_mlx/enhance.py` already has Silero-based speech-segment post processing (`SpeechBoostConfig`, `apply_speech_gain`) that could be adapted toward suppression policy.

Citations:
- `DeepFilterNet/df_mlx/enhance.py:60`
- `DeepFilterNet/df_mlx/enhance.py:74`
- `DeepFilterNet/df_mlx/enhance.py:609`
- `DeepFilterNet/df_mlx/enhance.py:757`

### Option B (Best training-only change): Rebalance objective weights and scheduling

Increase effective non-speech control signal (awesome/VAD terms), possibly with stage scheduling, then retrain and re-evaluate.

- Pros: no new inference branch; behavior remains model-internal
- Cons: likely slower iteration to achieve hard suppression target; may trade off speech fidelity

Why ranked #2: low engineering risk, but evidence indicates current weighting underpowers VAD behavior.

### Option C (Most robust long-term): Multi-task VAD head + gating policy

Add explicit VAD prediction head and use its output to condition suppression at inference.

- Pros: strongest architectural alignment between training objective and runtime behavior
- Cons: larger implementation/test/training cost

Why ranked #3: likely best final architecture, but more expensive than Option A/B.

### Option D (New option distinct from previous 3): Ship a fast mitigation using existing `enhance.py` speech-segment controls in evaluation pipeline

For immediate practical improvement, route `compare_models` MLX path through `df_mlx.enhance` speech-aware post stage (or equivalent wrapper), then measure objective/subjective gains while larger training changes are developed.

- Pros: fastest path to user-visible behavior change using existing code
- Cons: not a pure model fix; adds post-processing dependency

Why included: this was not one of the original three options and is directly supported by existing code.

---

## Recommended sequence

1. **Implement Option A first** (explicit inference gating/suppression), evaluate quickly on your 17-file set.
2. In parallel, **apply Option B** (objective rebalance) and retrain a short pilot.
3. If A+B still underperform or create artifacts, invest in **Option C**.
4. Also fix the diagnostics guard issue so future tuning uses trustworthy mask/proxy validation stats.

This sequence maximizes short-term outcome while preserving a path to a cleaner long-term architecture.
