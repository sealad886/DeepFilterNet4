"""Training setup helpers extracted from train_dynamic.train().

Provides console config printing, train-config dict construction, dataset
configuration setup, and GAN initialisation — keeping train() focused on the
training loop itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class DatasetSetupResult:
    """Result of dataset configuration setup."""

    config: Any  # DatasetConfig
    seed: int | None = None
    max_train_batches: int | None = None
    max_valid_batches: int | None = None
    eval_frequency: int = 1
    num_workers: int = 0
    prefetch_size: int = 2
    use_mlx_data: bool = True


def setup_dataset(
    *,
    # Data source params
    cache_dir: str | None = None,
    config_path: str | None = None,
    speech_list: str | None = None,
    noise_list: str | None = None,
    rir_list: str | None = None,
    p_reverb: float = 0.0,
    p_clipping: float = 0.0,
    num_workers: int = 0,
    # Override params
    dataset_overrides: dict[str, Any] | None = None,
    snr_range: tuple[float, float] | None = None,
    snr_range_extreme: tuple[float, float] | None = None,
    snr_range_very_low: tuple[float, float] | None = None,
    p_extreme_snr: float | None = None,
    p_very_low_snr: float | None = None,
    p_interfer_speech: float | None = None,
    speech_gain_range: tuple[float, float] | None = None,
    noise_gain_range: tuple[float, float] | None = None,
    # Debug mode params
    debug_numerics: bool = False,
    max_train_batches: int | None = None,
    max_valid_batches: int | None = None,
    eval_frequency: int = 1,
    prefetch_size: int = 2,
    use_mlx_data: bool = True,
    seed: int | None = None,
) -> DatasetSetupResult:
    """Load or create a DatasetConfig, apply CLI/debug overrides, seed RNG."""
    import random
    from pathlib import Path

    import mlx.core as mx
    import numpy as np

    from df_mlx.dynamic_dataset import DatasetConfig, read_file_list

    # Load or create config
    if cache_dir:
        if str(cache_dir).startswith("hf://"):
            import json

            from huggingface_hub import HfFileSystem

            from df_mlx.hf_paths import hf_dataset_fsspec_path, normalize_hf_dataset_cache_dir

            fs = HfFileSystem()
            normalized_cache_dir = normalize_hf_dataset_cache_dir(str(cache_dir))
            hf_path = hf_dataset_fsspec_path(normalized_cache_dir)
            config_file = f"{hf_path}/config.json"
            if fs.exists(config_file):
                with fs.open(config_file, "r") as f:
                    data = json.load(f)
                if "cache_dir" in data:
                    data["cache_dir"] = data["cache_dir"]
                config = DatasetConfig(
                    **{k: v for k, v in data.items() if hasattr(DatasetConfig, k) or k == "cache_dir"}
                )
                config.cache_dir = normalized_cache_dir
                print(f"Loaded config from HF cache: {normalized_cache_dir}")
            else:
                raise ValueError(f"Cache config not found in HF repo: {config_file}")
        else:
            # Load config from pre-built audio cache
            cache_path = Path(cache_dir).expanduser().resolve()
            config_file = cache_path / "config.json"
            if config_file.exists():
                config = DatasetConfig.from_json(str(config_file))
                config.cache_dir = cache_dir
                print(f"Loaded config from cache: {cache_dir}")
            else:
                raise ValueError(f"Cache config not found: {config_file}")
    elif config_path:
        config = DatasetConfig.from_json(config_path)
        print(f"Loaded config from: {config_path}")
    else:
        if not speech_list:
            raise ValueError("Either --cache-dir, --config, or --speech-list required")

        speech_files = read_file_list(speech_list)
        noise_files = read_file_list(noise_list) if noise_list else []
        rir_files = read_file_list(rir_list) if rir_list else []

        config = DatasetConfig(
            speech_files=speech_files,
            noise_files=noise_files,
            rir_files=rir_files,
            p_reverb=p_reverb,
            p_clipping=p_clipping,
            num_workers=num_workers,
        )

    # Apply train-config dataset overrides before CLI/runtime overrides
    if dataset_overrides:
        for key, value in dataset_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: train-config dataset override ignored: {key}")

    if snr_range is not None:
        config.snr_range = snr_range
    if snr_range_extreme is not None:
        config.snr_range_extreme = snr_range_extreme
    if snr_range_very_low is not None:
        config.snr_range_very_low = snr_range_very_low
    if p_extreme_snr is not None:
        config.p_extreme_snr = p_extreme_snr
    if p_very_low_snr is not None:
        config.p_very_low_snr = p_very_low_snr
    if p_interfer_speech is not None:
        config.p_interfer_speech = p_interfer_speech
    if speech_gain_range is not None:
        config.speech_gain_range = speech_gain_range
    if noise_gain_range is not None:
        config.noise_gain_range = noise_gain_range

    # Numeric debug mode overrides (deterministic, short runs)
    if debug_numerics:
        # NOTE: do NOT override epochs here.  The max_train_batches cap
        # already limits per-epoch work, and forcing epochs=1 breaks
        # checkpoint resume when start_epoch > 0.
        if max_train_batches is None:
            max_train_batches = 50
        if max_valid_batches is None:
            max_valid_batches = 10
        if eval_frequency != 1:
            print(f"  Debug numerics: overriding eval_frequency {eval_frequency} -> 1")
            eval_frequency = 1
        if num_workers != 0:
            print(f"  Debug numerics: overriding num_workers {num_workers} -> 0")
            num_workers = 0
        if prefetch_size != 1:
            print(f"  Debug numerics: overriding prefetch_size {prefetch_size} -> 1")
            prefetch_size = 1
        if use_mlx_data:
            print("  Debug numerics: disabling mlx-data for deterministic loading")
            use_mlx_data = False

    # RNG seeding (optional, default only in debug mode)
    if seed is None and debug_numerics:
        seed = getattr(config, "seed", 42)
    if seed is not None:
        config.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        mx.random.seed(seed)
        print(f"  RNG seed set to {seed}")

    # Keep dataset config aligned with CLI worker setting
    config.num_workers = num_workers

    return DatasetSetupResult(
        config=config,
        seed=seed,
        max_train_batches=max_train_batches,
        max_valid_batches=max_valid_batches,
        eval_frequency=eval_frequency,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
        use_mlx_data=use_mlx_data,
    )


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


def print_epoch_summary(
    epoch_avgs: dict[str, float],
    *,
    epoch: int,
    epochs: int,
    avg_valid_loss: float,
    best_valid_loss: float,
    samples_processed: int,
    epoch_time: float,
    use_vad_loss: bool,
    use_awesome_loss: bool,
    use_pipeline_awesome_loss: bool,
    use_mrstft_loss: bool,
    use_vad_train_reg: bool,
    gan_enabled: bool,
    gan_fm_weight: float,
    verbose: bool,
    debug_numerics: bool,
    num_debug_logs: int = 0,
    train_mask_clip_rate: float = 0.0,
    train_eps_clean_rate: float = 0.0,
    train_eps_noise_rate: float = 0.0,
    train_mask_logit_min: float = 0.0,
    train_mask_logit_max: float = 0.0,
    num_vad_logs: int = 0,
    train_vad_clip_ref: float = 0.0,
    train_vad_clip_out: float = 0.0,
) -> None:
    """Print a formatted epoch summary line plus optional verbose details.

    *epoch_avgs* is a ``dict[str, float]`` keyed by short metric names
    (``"loss"``, ``"spec_loss"``, ``"mrstft_loss"``, etc.).  Config flags
    control which loss components appear.  Debug-numerics stats are only
    printed when *debug_numerics* is ``True`` and the relevant counters
    are positive.
    """
    epoch_throughput = samples_processed / epoch_time if epoch_time > 0 else 0

    improvement_marker = "★" if avg_valid_loss <= best_valid_loss else ""
    loss_summary = ""
    if (
        use_vad_loss
        or use_awesome_loss
        or use_pipeline_awesome_loss
        or use_vad_train_reg
        or use_mrstft_loss
        or gan_enabled
    ):
        loss_parts = [f"Spec: {epoch_avgs['spec_loss']:.4f}"]
        if use_mrstft_loss:
            loss_parts.append(f"MRSTFT: {epoch_avgs['mrstft_loss']:.4f}")
        if gan_enabled:
            loss_parts.append(f"GAN_G: {epoch_avgs['gan_g_loss']:.4f}")
            loss_parts.append(f"GAN_D: {epoch_avgs['gan_d_loss']:.4f}")
            if gan_fm_weight > 0:
                loss_parts.append(f"FM: {epoch_avgs['gan_fm_loss']:.4f}")
        if use_vad_loss:
            loss_parts.extend(
                [
                    f"VAD: {epoch_avgs['vad_loss']:.4f}",
                    f"Speech: {epoch_avgs['speech_loss']:.4f}",
                ]
            )
        if use_awesome_loss or use_pipeline_awesome_loss:
            loss_parts.extend(
                [
                    f"Awesome: {epoch_avgs['awesome_loss']:.4f}",
                    f"AwS: {epoch_avgs['awesome_speech']:.4f}",
                    f"AwN: {epoch_avgs['awesome_noise']:.4f}",
                    f"AwSm: {epoch_avgs['awesome_smooth']:.4f}",
                ]
            )
        if use_pipeline_awesome_loss:
            loss_parts.extend(
                [
                    f"MusSup: {epoch_avgs['music_supp']:.4f}",
                    f"MaskSat: {epoch_avgs['mask_sat']:.4f}",
                ]
            )
        if use_vad_train_reg:
            loss_parts.append(f"VADreg: {epoch_avgs['vad_reg_loss']:.4f}")
        loss_summary = " | " + " | ".join(loss_parts)

    print(
        f"✓ Epoch {epoch + 1}/{epochs} complete | "
        f"Train: {epoch_avgs['loss']:.4f}{loss_summary} | "
        f"Valid: {avg_valid_loss:.4f} {improvement_marker}| "
        f"Best: {best_valid_loss:.4f} | "
        f"{samples_processed:,} samples @ {epoch_throughput:.0f}/s | "
        f"{epoch_time:.1f}s"
    )

    if use_vad_loss and verbose:
        print(
            f"  VAD stats: p_ref={epoch_avgs['p_ref']:.2f} | "
            f"p_out={epoch_avgs['p_out']:.2f} | gate={epoch_avgs['gate']:.0f}%"
        )
    if (use_awesome_loss or use_pipeline_awesome_loss) and verbose:
        print(
            "  Awesome stats: "
            f"mask={epoch_avgs['mask_mean']:.2f} "
            f"(hi {epoch_avgs['mask_high']:.0f}%, lo {epoch_avgs['mask_low']:.0f}%) | "
            f"proxy={epoch_avgs['proxy']:.2f} ratio={epoch_avgs['speech_ratio']:.2f} | "
            f"music_gate={epoch_avgs['music_gate']:.2f} "
            f"music={epoch_avgs['musicness']:.2f} | "
            f"mod={epoch_avgs['mod']:.2f} "
            f"e_boost={epoch_avgs['energy_boost']:.2f} "
            f"snr_boost={epoch_avgs['snr_boost']:.2f}"
        )
    if debug_numerics:
        parts: list[str] = []
        if (use_awesome_loss or use_pipeline_awesome_loss) and num_debug_logs > 0:
            avg_mask_clip = train_mask_clip_rate / num_debug_logs
            avg_eps_clean = train_eps_clean_rate / num_debug_logs
            avg_eps_noise = train_eps_noise_rate / num_debug_logs
            parts.append(
                f"mask_logit=[{train_mask_logit_min:.1f},{train_mask_logit_max:.1f}] "
                f"clip={avg_mask_clip:.1f}% eps_clean={avg_eps_clean:.1f}% "
                f"eps_noise={avg_eps_noise:.1f}%"
            )
        if use_vad_loss and num_vad_logs > 0:
            avg_vad_clip_ref = train_vad_clip_ref / num_vad_logs
            avg_vad_clip_out = train_vad_clip_out / num_vad_logs
            parts.append(
                f"vad_clip_ref={avg_vad_clip_ref:.1f}% "
                f"vad_clip_out={avg_vad_clip_out:.1f}%"
            )
        if parts:
            print("  Debug numerics: " + " | ".join(parts))


def setup_gan(
    *,
    gan_enabled: bool,
    gan_disc_type: str,
    gan_mpd_periods: tuple | list | None,
    gan_mpd_channels: int,
    gan_msd_scales: int,
    gan_msd_channels: int,
    gan_disc_lr: float,
    gan_disc_weight_decay: float,
) -> tuple[Any, Any, Any, tuple | None]:
    """Create GAN discriminator, optimizer, and loss functions.

    Returns ``(discriminator, disc_optimizer, feature_match_loss, gan_loss_fns)``.
    All elements are ``None`` when *gan_enabled* is ``False``.
    """
    discriminator = None
    disc_optimizer = None
    feature_match_loss = None
    gan_loss_fns = None

    if not gan_enabled:
        return discriminator, disc_optimizer, feature_match_loss, gan_loss_fns

    import mlx.optimizers as optim

    from df_mlx.discriminator import (
        CombinedDiscriminator,
        MultiPeriodDiscriminator,
        MultiScaleDiscriminator,
    )
    from df_mlx.loss import FeatureMatchingLoss, discriminator_loss, generator_loss

    mpd_periods = tuple(gan_mpd_periods) if gan_mpd_periods else (2, 3, 5, 7, 11)
    if gan_disc_type == "mpd":
        discriminator = MultiPeriodDiscriminator(periods=mpd_periods, channels=gan_mpd_channels)
    elif gan_disc_type == "msd":
        discriminator = MultiScaleDiscriminator(
            num_scales=gan_msd_scales, channels=gan_msd_channels
        )
    else:
        discriminator = CombinedDiscriminator(
            mpd_periods=mpd_periods,
            mpd_channels=gan_mpd_channels,
            msd_scales=gan_msd_scales,
            msd_channels=gan_msd_channels,
        )

    disc_optimizer = optim.AdamW(
        learning_rate=gan_disc_lr,
        weight_decay=gan_disc_weight_decay,
    )
    feature_match_loss = FeatureMatchingLoss(factor=1.0)
    gan_loss_fns = (generator_loss, discriminator_loss)

    return discriminator, disc_optimizer, feature_match_loss, gan_loss_fns
