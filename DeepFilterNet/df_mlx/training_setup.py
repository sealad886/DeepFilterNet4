"""Training setup, teardown, and auxiliary-loss configuration.

Consolidates everything that happens *before* and *after* the epoch loop:
dataset/data-pipeline construction, GAN discriminator initialisation,
auxiliary-loss wiring, console config printing, train-config dict assembly,
epoch-summary logging, and post-training finalisation.

Key exports:
    - _sync_model_config_with_dataset: Align model config with dataset params.
    - DatasetSetupResult / setup_dataset: Build and validate the DatasetConfig.
    - DataPipelineResult / setup_data_pipeline: Construct train/valid iterators.
    - AuxLossSetupResult / setup_auxiliary_losses: Wire VAD, awesome, pipeline losses.
    - print_training_config: Pretty-print the full training configuration panel.
    - build_train_config: Assemble the flat config dict saved alongside checkpoints.
    - print_epoch_summary: Log end-of-epoch stats.
    - finalize_training: Post-training cleanup (final checkpoint, summary, etc.).

Relationship to train_dynamic:
    Called during the setup phase of train() (before the epoch loop) and during
    teardown (finalize_training).  Not included in the backward-compat re-export
    block; imported directly by train_dynamic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _sync_model_config_with_dataset(model_cfg: Any, dataset_cfg: Any) -> None:
    """Align MLX model config with dataset audio parameters."""
    model_cfg.audio.sr = dataset_cfg.sample_rate
    model_cfg.audio.fft_size = dataset_cfg.fft_size
    model_cfg.audio.hop_size = dataset_cfg.hop_size
    n_freqs = dataset_cfg.fft_size // 2 + 1
    model_cfg.audio.nb_freqs = n_freqs
    model_cfg.audio.n_freqs = n_freqs
    model_cfg.erb.nb_erb = dataset_cfg.nb_erb
    model_cfg.df.nb_df = dataset_cfg.nb_df


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


def _candidate_cache_dirs(cache_dir: str) -> list[Path]:
    requested = Path(cache_dir).expanduser().resolve()
    candidates = [requested]
    if requested.name.endswith("_cleaned"):
        candidates.append(requested.with_name(requested.name.removesuffix("_cleaned")))
    return candidates


def _resolve_cache_config_path(cache_dir: str) -> tuple[Path, Path]:
    candidates = _candidate_cache_dirs(cache_dir)
    for candidate in candidates:
        config_file = candidate / "config.json"
        if config_file.exists():
            return candidate, config_file

    searched = ", ".join(str(candidate / "config.json") for candidate in candidates)
    raise ValueError(f"Cache config not found. Checked: {searched}")


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
            requested_cache_path = Path(cache_dir).expanduser().resolve()
            cache_path, config_file = _resolve_cache_config_path(cache_dir)
            if cache_path != requested_cache_path:
                print(
                    "Warning: cache config missing at "
                    f"{requested_cache_path / 'config.json'}; using {config_file} instead"
                )
            config = DatasetConfig.from_json(str(config_file))
            config.cache_dir = str(cache_path)
            print(f"Loaded config from cache: {cache_path}")
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
        print(f"  VAD eval:      mode={vad_eval_mode} every={vad_eval_every} epochs batches={vad_eval_batches}")
        if vad_eval_mode == "silero":
            max_sec = vad_eval_max_seconds if vad_eval_max_seconds > 0 else "full"
            print(
                "  Silero VAD:    "
                f"sr={vad_silero_sample_rate}Hz, max_sec={max_sec}, "
                f"model={vad_silero_model_path or 'package'}"
            )
    if use_vad_train_reg:
        print(
            "  VAD train:     " f"prob={vad_train_prob} every_steps={vad_train_every_steps} (weight={vad_loss_weight})"
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
        "mrstft_hop_sizes": (list(mrstft_cfg.hop_sizes) if (mrstft_cfg and mrstft_cfg.hop_sizes) else None),
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
            parts.append(f"vad_clip_ref={avg_vad_clip_ref:.1f}% " f"vad_clip_out={avg_vad_clip_out:.1f}%")
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
        discriminator = MultiScaleDiscriminator(num_scales=gan_msd_scales, channels=gan_msd_channels)
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


@dataclass
class AuxLossSetupResult:
    """Result from auxiliary loss configuration."""

    use_awesome_loss: bool
    use_pipeline_awesome_loss: bool
    use_vad_loss: bool
    use_vad_train_reg: bool
    use_mrstft_loss: bool
    pipeline_stage_defs: list[dict[str, Any]]
    base_awesome_loss_weight: float
    base_vad_loss_weight: float
    base_vad_speech_loss_weight: float
    stage_max_vad_weight: float
    stage_max_vad_speech_weight: float
    mrstft_cfg: Any | None
    mrstft_loss_fn: Any | None
    mrstft_istft: Any | None
    mrstft_target_len: int | None
    mrstft_hop_sizes: tuple | None
    gan_enabled: bool
    gan_target_len: int
    gan_istft: Any | None
    gan_disc_type: str
    gan_disc_update_freq: int
    discriminator: Any | None
    disc_optimizer: Any | None
    feature_match_loss: Any | None
    gan_loss_fns: tuple | None
    vad_eval_enabled: bool
    vad_eval_mode: str
    silero_vad: Any | None
    vad_band_mask: Any  # mx.array
    vad_band_bins: float


def setup_auxiliary_losses(
    *,
    config: Any,
    dynamic_loss: str,
    pipeline_stages: list[dict[str, Any]] | None,
    awesome_loss_weight: float,
    vad_loss_weight: float,
    vad_speech_loss_weight: float,
    mrstft_config: Any | None,
    # GAN params
    gan_enabled: bool,
    gan_adv_weight: float,
    gan_fm_weight: float,
    gan_disc_type: str,
    gan_mpd_periods: tuple | None,
    gan_mpd_channels: int,
    gan_msd_scales: int,
    gan_msd_channels: int,
    gan_disc_lr: float,
    gan_disc_weight_decay: float,
    gan_disc_update_freq: int,
    # VAD eval params
    vad_eval_mode: str,
    vad_silero_model_path: str | None,
    vad_silero_sample_rate: int,
    vad_eval_max_seconds: float,
    vad_band_low_hz: float,
    vad_band_high_hz: float,
    vad_train_prob: float,
    vad_train_every_steps: int,
) -> AuxLossSetupResult:
    """Configure auxiliary losses: MRSTFT, GAN, VAD eval, and band mask.

    Returns an :class:`AuxLossSetupResult` bundling all computed artefacts so
    the caller can unpack them without dozens of local variables.
    """
    import mlx.core as mx

    from df_mlx.training_helpers import is_vad_train_reg_enabled as _is_vad_train_reg_enabled
    from df_mlx.training_losses import _build_speech_band_mask

    use_awesome_loss = dynamic_loss == "awesome"
    use_pipeline_awesome_loss = dynamic_loss == "pipeline_awesome"
    pipeline_stage_defs = sorted((pipeline_stages or []), key=lambda s: int(s.get("start_epoch", 0)))

    base_awesome_loss_weight = awesome_loss_weight
    base_vad_loss_weight = vad_loss_weight
    base_vad_speech_loss_weight = vad_speech_loss_weight
    stage_max_vad_weight = max(
        [
            base_vad_loss_weight,
            *[
                float(s.get("vad_loss_weight", 0.0))
                for s in pipeline_stage_defs
                if s.get("vad_loss_weight") is not None
            ],
        ]
    )
    stage_max_vad_speech_weight = max(
        [
            base_vad_speech_loss_weight,
            *[
                float(s.get("vad_speech_loss_weight", 0.0))
                for s in pipeline_stage_defs
                if s.get("vad_speech_loss_weight") is not None
            ],
        ]
    )

    mrstft_cfg = mrstft_config
    use_mrstft_loss = mrstft_cfg is not None and mrstft_cfg.factor > 0
    mrstft_loss_fn = None
    mrstft_hop_sizes = None
    mrstft_istft = None
    mrstft_target_len = None
    if use_mrstft_loss:
        if not mrstft_cfg or not mrstft_cfg.fft_sizes:
            print("Warning: mrstft enabled but fft_sizes is empty; disabling MRSTFT loss.")
            use_mrstft_loss = False
        else:
            from functools import partial

            from df_mlx.ops import istft
            from df_mlx.train import MultiResolutionSTFTLoss

            mrstft_istft = partial(istft)
            mrstft_hop_sizes = tuple(mrstft_cfg.hop_sizes) if mrstft_cfg.hop_sizes is not None else None
            mrstft_loss_fn = MultiResolutionSTFTLoss(
                fft_sizes=tuple(mrstft_cfg.fft_sizes),
                hop_sizes=mrstft_hop_sizes,
                gamma=mrstft_cfg.gamma,
                factor=mrstft_cfg.factor,
                f_complex=mrstft_cfg.f_complex,
            )
            mrstft_target_len = int(round(config.segment_length * config.sample_rate))

    # GAN configuration (adversarial + feature matching)
    gan_enabled = bool(gan_enabled or gan_adv_weight > 0 or gan_fm_weight > 0)
    gan_disc_type = gan_disc_type.lower()
    if gan_disc_type not in {"combined", "mpd", "msd"}:
        print(f"Warning: unsupported gan_disc_type={gan_disc_type}; using combined.")
        gan_disc_type = "combined"
    gan_disc_update_freq = max(int(gan_disc_update_freq), 1)
    gan_target_len = int(round(config.segment_length * config.sample_rate))
    gan_istft = mrstft_istft

    discriminator, disc_optimizer, feature_match_loss, gan_loss_fns = setup_gan(
        gan_enabled=gan_enabled,
        gan_disc_type=gan_disc_type,
        gan_mpd_periods=gan_mpd_periods,
        gan_mpd_channels=gan_mpd_channels,
        gan_msd_scales=gan_msd_scales,
        gan_msd_channels=gan_msd_channels,
        gan_disc_lr=gan_disc_lr,
        gan_disc_weight_decay=gan_disc_weight_decay,
    )
    if gan_enabled and gan_istft is None:
        from functools import partial

        from df_mlx.ops import istft

        gan_istft = partial(istft)

    if vad_eval_mode == "auto":
        vad_eval_mode = "proxy" if (use_awesome_loss or use_pipeline_awesome_loss) else "off"
    vad_eval_enabled = vad_eval_mode != "off"
    silero_vad = None
    if vad_eval_mode == "silero":
        from df_mlx.vad_silero import SileroVAD, SileroVADConfig

        silero_vad = SileroVAD(
            SileroVADConfig(
                sample_rate=vad_silero_sample_rate,
                model_path=vad_silero_model_path,
                max_seconds=vad_eval_max_seconds if vad_eval_max_seconds > 0 else None,
                force_cpu=True,
            )
        )

    use_vad_loss = stage_max_vad_weight > 0 or stage_max_vad_speech_weight > 0
    use_vad_train_reg = _is_vad_train_reg_enabled(
        vad_train_prob=vad_train_prob,
        vad_train_every_steps=vad_train_every_steps,
        max_stage_vad_weight=stage_max_vad_weight,
    )

    scalar_zero = mx.array(0.0)
    need_band_mask = (
        use_vad_loss or use_awesome_loss or use_pipeline_awesome_loss or vad_eval_enabled or use_vad_train_reg
    )
    if need_band_mask:
        n_freqs = config.fft_size // 2 + 1
        vad_band_mask, vad_band_bins = _build_speech_band_mask(
            n_freqs,
            config.sample_rate,
            vad_band_low_hz,
            vad_band_high_hz,
        )
    else:
        vad_band_mask = scalar_zero
        vad_band_bins = 1.0

    return AuxLossSetupResult(
        use_awesome_loss=use_awesome_loss,
        use_pipeline_awesome_loss=use_pipeline_awesome_loss,
        use_vad_loss=use_vad_loss,
        use_vad_train_reg=use_vad_train_reg,
        use_mrstft_loss=use_mrstft_loss,
        pipeline_stage_defs=pipeline_stage_defs,
        base_awesome_loss_weight=base_awesome_loss_weight,
        base_vad_loss_weight=base_vad_loss_weight,
        base_vad_speech_loss_weight=base_vad_speech_loss_weight,
        stage_max_vad_weight=stage_max_vad_weight,
        stage_max_vad_speech_weight=stage_max_vad_speech_weight,
        mrstft_cfg=mrstft_cfg,
        mrstft_loss_fn=mrstft_loss_fn,
        mrstft_istft=mrstft_istft,
        mrstft_target_len=mrstft_target_len,
        mrstft_hop_sizes=mrstft_hop_sizes,
        gan_enabled=gan_enabled,
        gan_target_len=gan_target_len,
        gan_istft=gan_istft,
        gan_disc_type=gan_disc_type,
        gan_disc_update_freq=gan_disc_update_freq,
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
        feature_match_loss=feature_match_loss,
        gan_loss_fns=gan_loss_fns,
        vad_eval_enabled=vad_eval_enabled,
        vad_eval_mode=vad_eval_mode,
        silero_vad=silero_vad,
        vad_band_mask=vad_band_mask,
        vad_band_bins=vad_band_bins,
    )


def finalize_training(
    *,
    final_epoch: int,
    global_step: int,
    avg_train_loss: float,
    best_valid_loss: float,
    last_completed_epoch: int,
    last_valid_epoch: int | None,
    last_valid_loss: float | None,
    model: Any,
    optimizer: Any,
    state: list,
    discriminator: Any | None,
    disc_optimizer: Any | None,
    ckpt_dir: Any,  # Path
    train_config: dict[str, Any],
    active_stage_index: int,
    active_stage_name: str,
    tqdm_setup_panel: Any | None,
    run_validation_fn: Any,  # Callable[[], float]
) -> None:
    """Run final validation, save final/best checkpoints, and print summary."""
    import mlx.core as mx

    from df_mlx.training_checkpoints import save_checkpoint

    ckpt_dir_path = ckpt_dir if hasattr(ckpt_dir, "name") else __import__("pathlib").Path(ckpt_dir)

    # Final validation to compare against best checkpoint.
    if last_valid_epoch == final_epoch and last_valid_loss is not None:
        final_valid_loss = last_valid_loss
    else:
        final_valid_loss = run_validation_fn()

    if final_valid_loss < best_valid_loss:
        best_valid_loss = final_valid_loss
        best_path = ckpt_dir_path / "best.safetensors"
        best_final_saved = save_checkpoint(
            model,
            best_path,
            epoch=final_epoch,
            batch_idx=None,
            global_step=global_step,
            loss=avg_train_loss,
            best_valid_loss=best_valid_loss,
            config=train_config,
            optimizer=optimizer,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
            last_completed_epoch=max(last_completed_epoch, final_epoch),
            pipeline_stage_index=active_stage_index,
            pipeline_stage_name=active_stage_name,
            kind="best_final",
        )
        if best_final_saved:
            print(f"  ✅ Final weights set new best: {best_valid_loss:.4f}")
        else:
            print("  ⚠️  Failed to save final best checkpoint.")

    # Save final weights (even if not aligned to checkpoint interval).
    mx.eval(state)
    final_path = ckpt_dir_path / "final.safetensors"
    final_saved = save_checkpoint(
        model,
        final_path,
        epoch=final_epoch,
        batch_idx=None,
        global_step=global_step,
        loss=avg_train_loss,
        best_valid_loss=best_valid_loss,
        config=train_config,
        optimizer=optimizer,
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
        last_completed_epoch=max(last_completed_epoch, final_epoch),
        pipeline_stage_index=active_stage_index,
        pipeline_stage_name=active_stage_name,
        kind="final",
    )
    if final_saved:
        print(f"  📦 Final checkpoint saved: {final_path.name}")
    else:
        print("  ⚠️  Final checkpoint save failed.")

    # ====== Final Summary ======
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final epoch:     {final_epoch + 1}")
    print(f"Best valid loss: {best_valid_loss:.4f}")
    if final_valid_loss != float("inf"):
        print(f"Final valid loss: {final_valid_loss:.4f}")
    else:
        print("Final valid loss: N/A")
    print(f"Final checkpoint: {final_path}")
    print(f"Best checkpoint: {ckpt_dir_path / 'best.safetensors'}")
    print(f"Checkpoints:     {ckpt_dir_path}")

    if tqdm_setup_panel is not None:
        tqdm_setup_panel.close()


@dataclass
class DataPipelineResult:
    """Result of data pipeline setup."""

    ckpt_dir: Any  # Path
    debugger: Any | None  # NumericDebugger | None
    debug_dump_dir: Any | None  # Path | None
    validation_report: dict[str, Any] | None
    use_mlx_stream: bool
    train_stream: Any | None  # MLXDataStream | None
    data_checkpoint_path: Any  # Path
    data_resume_progress: dict[str, Any] | None
    data_resume_source: str | None
    resume_from: str | None  # May be updated by check_chkpts


def setup_data_pipeline(
    *,
    dataset: Any,  # DynamicDataset
    checkpoint_dir: str,
    batch_size: int,
    num_workers: int,
    prefetch_size: int,
    use_mlx_data: bool,
    resume_from: str | None = None,
    resume_data_from: str | None = None,
    # Debug params
    debug_numerics: bool = False,
    debug_numerics_fail_fast: bool = False,
    debug_numerics_every: int = 1,
    debug_numerics_dump_dir: str | None = None,
    debug_numerics_dump_arrays: bool = False,
    debug_numerics_max_dumps: int = 10,
    nan_skip_batch: bool = False,
    # Checkpoint validation
    check_chkpts: bool = False,
) -> DataPipelineResult:
    """Set up checkpoint directory, debug numerics, checkpoint validation, and data stream.

    Returns a :class:`DataPipelineResult` containing all pipeline artefacts.
    The caller is responsible for wiring ``_interrupt_state`` from the result.
    """
    from pathlib import Path

    from df_mlx.dynamic_dataset import HAS_MLX_DATA, MLXDataStream
    from df_mlx.training_checkpoints import validate_checkpoint_dir
    from df_mlx.training_ops import NumericDebugConfig, NumericDebugger

    # Create checkpoint directory early (needed for data checkpoint path)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    debug_dump_dir = None
    if debug_numerics:
        debug_dump_dir = Path(debug_numerics_dump_dir) if debug_numerics_dump_dir else ckpt_dir / "debug_numerics"
        debug_cfg = NumericDebugConfig(
            enabled=True,
            fail_fast=debug_numerics_fail_fast and not nan_skip_batch,
            skip_batch=nan_skip_batch,
            every=max(debug_numerics_every, 1),
            dump_dir=debug_dump_dir,
            dump_arrays=debug_numerics_dump_arrays,
            max_dumps=debug_numerics_max_dumps,
            check_grads=True,
        )
        debugger = NumericDebugger(debug_cfg)
        print(
            "  Debug numerics: enabled "
            f"(fail_fast={'on' if debug_cfg.fail_fast else 'off'}, "
            f"every={debug_cfg.every}, dump_dir={debug_dump_dir})"
        )
    else:
        debugger = None

    validation_report = None
    if check_chkpts:
        validation_report = validate_checkpoint_dir(ckpt_dir, strict=True, validate_load=True)
        print(
            f"Checkpoint validation: total={validation_report['total']} "
            f"valid={validation_report['valid']} invalid={len(validation_report['invalid'])}"
        )
        if validation_report["latest_path"]:
            print(f"  Latest valid checkpoint: {validation_report['latest_path']}")
        if validation_report["latest_state"]:
            print(
                f"  last_completed_epoch={validation_report['last_completed_epoch']}, "
                f"resume_epoch={validation_report['resume_epoch']}, "
                f"resume_batch={validation_report['resume_batch']}, "
                f"resume_global_step={validation_report['resume_global_step']}"
            )

        if resume_from is None and validation_report["latest_path"]:
            resume_from = str(validation_report["latest_path"])
    # Determine which data loader to use
    use_mlx_stream = use_mlx_data and HAS_MLX_DATA
    if use_mlx_data and not HAS_MLX_DATA:
        print("  Note: mlx-data not available, using PrefetchDataLoader")
    elif use_mlx_stream:
        print(f"  Using MLXDataStream (workers={num_workers}, prefetch={prefetch_size})")

    # Create data stream/loader
    data_checkpoint_path = ckpt_dir / "data_checkpoint.json"
    train_stream: MLXDataStream | None = None
    data_resume_progress: dict[str, Any] | None = None
    data_resume_source: str | None = None

    if use_mlx_stream:
        # Check for data checkpoint to resume from
        if resume_data_from:
            train_stream = MLXDataStream.from_checkpoint(
                dataset=dataset,
                checkpoint_path=resume_data_from,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                num_workers=num_workers,
            )
            print(f"  Resuming data from: {resume_data_from}")
            data_resume_progress = train_stream.get_progress()
            data_resume_source = resume_data_from
            print(
                f"  Data checkpoint: epoch {data_resume_progress['epoch']}, " f"batch {data_resume_progress['batch']}"
            )
        elif data_checkpoint_path.exists():
            # Auto-resume from last data checkpoint
            try:
                train_stream = MLXDataStream.from_checkpoint(
                    dataset=dataset,
                    checkpoint_path=data_checkpoint_path,
                    batch_size=batch_size,
                    prefetch_size=prefetch_size,
                    num_workers=num_workers,
                )
                data_resume_progress = train_stream.get_progress()
                data_resume_source = str(data_checkpoint_path)
                print(
                    "  Auto-resuming from data checkpoint: "
                    f"epoch {data_resume_progress['epoch']}, batch {data_resume_progress['batch']}"
                )
            except Exception as e:
                print(f"  Warning: Could not load data checkpoint: {e}")
                train_stream = None

        if train_stream is None:
            train_stream = MLXDataStream(
                dataset=dataset,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                num_workers=num_workers,
            )

    return DataPipelineResult(
        ckpt_dir=ckpt_dir,
        debugger=debugger,
        debug_dump_dir=debug_dump_dir,
        validation_report=validation_report,
        use_mlx_stream=use_mlx_stream,
        train_stream=train_stream,
        data_checkpoint_path=data_checkpoint_path,
        data_resume_progress=data_resume_progress,
        data_resume_source=data_resume_source,
        resume_from=resume_from,
    )
