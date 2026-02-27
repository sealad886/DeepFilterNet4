"""Training setup helpers extracted from train_dynamic.train().

Provides console config printing and train-config dict construction, keeping
train() focused on the training loop itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def print_training_config(
    config: Any,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    min_lr: float,
    weight_decay: float,
    checkpoint_dir: str,
    dynamic_loss: str,
    mrstft_cfg: Any | None = None,
    awesome_loss_weight: float = 0.0,
    awesome_mask_sharpness: float = 1.0,
    awesome_warmup_steps: int = 0,
    vad_proxy_enabled: bool = False,
    gan_enabled: bool = False,
    gan_adv_weight: float = 0.0,
    gan_fm_weight: float = 0.0,
    gan_start_epoch: int = 0,
    gan_ramp_epochs: int = 0,
    gan_disc_type: str = "mpd",
    gan_mpd_periods: tuple | list | None = None,
    gan_msd_scales: int = 3,
    gan_disc_update_freq: int = 1,
    gan_disc_max_samples: int | None = None,
    gan_mpd_channels: int = 32,
    gan_msd_channels: int = 16,
    vad_loss_weight: float = 0.0,
    vad_speech_loss_weight: float = 0.0,
    vad_threshold: float = 0.5,
    vad_margin: float = 0.1,
    vad_warmup_epochs: int = 0,
    vad_snr_gate_db: float = 40.0,
    vad_snr_gate_width: float = 5.0,
    vad_band_low_hz: float = 0.0,
    vad_band_high_hz: float = 8000.0,
    vad_eval_mode: str = "off",
    vad_eval_every: int = 1,
    vad_eval_batches: int = 10,
    vad_eval_max_seconds: float = 0.0,
    vad_silero_sample_rate: int = 16000,
    vad_silero_model_path: str | None = None,
    use_vad_train_reg: bool = False,
    vad_train_prob: float = 0.0,
    vad_train_every_steps: int = 1,
    pipeline_stage_defs: list | None = None,
) -> bool:
    """Print a summary of the training configuration to stdout.

    Derives several convenience booleans internally and returns ``vad_enabled``
    so the caller can reuse it without recomputing.
    """
    use_mrstft_loss = mrstft_cfg is not None and mrstft_cfg.factor > 0
    use_awesome_loss = dynamic_loss == "awesome"
    use_pipeline_awesome_loss = dynamic_loss == "pipeline_awesome"
    vad_eval_enabled = vad_eval_mode != "off"

    # Print file counts after dataset init (so cache files are included)
    print(f"Speech files:   {len(config.speech_files):,}")
    print(f"Noise files:    {len(config.noise_files):,}")
    print(f"RIR files:      {len(config.rir_files):,}")
    print(f"Epochs:         {epochs}")
    print(f"Batch size:     {batch_size}")
    print(f"Learning rate:  {learning_rate} (min {min_lr})")
    print(f"Weight decay:   {weight_decay}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"P(reverb):      {config.p_reverb}")
    print(f"P(clipping):    {config.p_clipping}")
    print(f"SNR range:      {config.snr_range} dB")
    print(f"SNR extreme:    {config.snr_range_extreme} dB (p={config.p_extreme_snr})")
    print(f"Speech gain:    {config.speech_gain_range} dB")
    print(f"Noise gain:     {config.noise_gain_range} dB")
    print(f"Dynamic loss:   {dynamic_loss}")
    if use_mrstft_loss and mrstft_cfg is not None:
        hop_sizes_display = mrstft_cfg.hop_sizes if mrstft_cfg.hop_sizes is not None else "auto"
        print(
            "MRSTFT loss:   "
            f"factor={mrstft_cfg.factor}, gamma={mrstft_cfg.gamma}, "
            f"f_complex={mrstft_cfg.f_complex}, fft_sizes={mrstft_cfg.fft_sizes}, "
            f"hop_sizes={hop_sizes_display}"
        )
    if use_awesome_loss or use_pipeline_awesome_loss:
        print(
            f"  Awesome loss: weight={awesome_loss_weight}, mask_sharpness={awesome_mask_sharpness}, "
            f"warmup_steps={awesome_warmup_steps}, proxy={'on' if vad_proxy_enabled else 'off'}"
        )
    if gan_enabled:
        print(
            "GAN loss:       on "
            f"(adv={gan_adv_weight}, fm={gan_fm_weight}, start={gan_start_epoch}, ramp={gan_ramp_epochs})"
        )
        print(
            "  Discriminator: "
            f"type={gan_disc_type}, mpd_periods={gan_mpd_periods or [2, 3, 5, 7, 11]}, "
            f"msd_scales={gan_msd_scales}, update_freq={gan_disc_update_freq}"
        )
        print(
            "  Disc memory:  "
            f"max_samples={gan_disc_max_samples or 'full'}, "
            f"mpd_ch={gan_mpd_channels}, msd_ch={gan_msd_channels}"
        )
    vad_enabled = vad_loss_weight > 0 or vad_speech_loss_weight > 0
    print(
        f"VAD loss:       {'on' if vad_enabled else 'off'} "
        f"(w_vad={vad_loss_weight}, w_speech={vad_speech_loss_weight})"
    )
    if vad_enabled:
        print(f"  VAD threshold: {vad_threshold} | margin: {vad_margin}")
        print(f"  VAD warmup:    {vad_warmup_epochs} epochs")
        print(f"  VAD SNR gate:  {vad_snr_gate_db} dB (width {vad_snr_gate_width} dB)")
        print(f"  VAD band:      {vad_band_low_hz:.0f}-{vad_band_high_hz:.0f} Hz")
    if vad_eval_enabled:
        print(
            f"  VAD eval:      mode={vad_eval_mode} every={vad_eval_every} epochs batches={vad_eval_batches}"
        )
        if vad_eval_mode == "silero":
            max_sec = vad_eval_max_seconds if vad_eval_max_seconds > 0 else "full"
            print(
                "  Silero VAD:    "
                f"sr={vad_silero_sample_rate}Hz, max_sec={max_sec}, "
                f"model={vad_silero_model_path or 'package'}"
            )
    if use_vad_train_reg:
        print(
            "  VAD train:     "
            f"prob={vad_train_prob} every_steps={vad_train_every_steps} (weight={vad_loss_weight})"
        )
    if pipeline_stage_defs:
        print("  Pipeline stages:")
        for idx, stage in enumerate(pipeline_stage_defs):
            stage_name = stage.get("name", f"stage_{idx}")
            stage_parts = [f"start={stage['start_epoch']}", f"name={stage_name}"]
            if stage.get("awesome_loss_weight") is not None:
                stage_parts.append(f"awesome_w={stage['awesome_loss_weight']}")
            if stage.get("vad_loss_weight") is not None:
                stage_parts.append(f"vad_w={stage['vad_loss_weight']}")
            if stage.get("vad_speech_loss_weight") is not None:
                stage_parts.append(f"speech_w={stage['vad_speech_loss_weight']}")
            print("    - " + ", ".join(stage_parts))
    print("=" * 60)

    return vad_enabled


def build_train_config(
    config: Any,
    *,
    mrstft_cfg: Any | None = None,
    gan_mpd_periods: tuple | list | None = None,
    pipeline_stage_defs: list | None = None,
    **params: Any,
) -> dict[str, Any]:
    """Build the serialisable training-config dict.

    ``config.__dict__`` forms the base.  MRSTFT fields are unpacked from
    *mrstft_cfg*, ``gan_mpd_periods`` is normalised to a list (defaulting to
    ``[2, 3, 5, 7, 11]``), and ``pipeline_stage_defs`` is stored under the key
    ``"pipeline_stages"``.  All remaining *params* are merged as-is.
    """
    train_config: dict[str, Any] = {
        **config.__dict__,
        "pipeline_stages": pipeline_stage_defs,
        "mrstft_factor": mrstft_cfg.factor if mrstft_cfg is not None else 0.0,
        "mrstft_gamma": mrstft_cfg.gamma if mrstft_cfg is not None else 1.0,
        "mrstft_f_complex": mrstft_cfg.f_complex if mrstft_cfg is not None else None,
        "mrstft_fft_sizes": list(mrstft_cfg.fft_sizes) if mrstft_cfg is not None else None,
        "mrstft_hop_sizes": (
            list(mrstft_cfg.hop_sizes) if (mrstft_cfg and mrstft_cfg.hop_sizes) else None
        ),
        "gan_mpd_periods": list(gan_mpd_periods) if gan_mpd_periods else [2, 3, 5, 7, 11],
        **params,
    }
    return train_config
