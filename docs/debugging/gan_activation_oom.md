# Debugging GAN-Activation OOM in `df_mlx.train_dynamic`

This note covers OOM failures that appear **right when GAN activates** (for example, at `gan.start_epoch`).

## Typical Symptoms

- Training is stable for pre-GAN epochs.
- At the first GAN-active epoch, the process is killed immediately (often before meaningful progress-bar advance).
- This is more likely with large `batch_size`, high `grad_accumulation_steps`, and enabled MRSTFT + FM GAN loss.

## Why It Happens

GAN activation increases peak memory sharply because, in addition to generator loss, the loop now performs:

- discriminator forwards for generator adversarial + feature-matching terms,
- discriminator updates (backward through discriminator),
- and optional detailed metric recomputation paths.

In this repository, the primary mitigations are:

1. discriminator updates are aligned to **optimizer-step cadence** under grad accumulation,
2. GAN discriminator waveform tensors use model precision (FP16 when enabled),
3. MRSTFT can remain FP32 for numerical stability.

## Practical Mitigations

If you still hit memory limits for a specific run profile:

- Reduce `training.batch_size` first (largest memory lever).
- Reduce `training.grad_accumulation_steps` if discriminator pressure remains high.
- Reduce GAN complexity (`gan.discriminator`, `gan.fm_weight`, `gan.disc_update_freq`).
- Use `debug.sync_mode = "fast"` to avoid extra detailed metric overhead during triage.

## Quick Triage Checklist

1. Confirm GAN starts at the intended epoch (`gan.start_epoch`).
2. Confirm effective batch = `batch_size * grad_accumulation_steps` is realistic for hardware.
3. Check whether MRSTFT + GAN + FM are enabled together.
4. Retry with smaller `batch_size` and compare startup behavior at first GAN-active epoch.
