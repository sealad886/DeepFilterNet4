"""Class-based API for DeepFilterNet4 dynamic training.

Provides :class:`TrainingSession` as a thin, kwargs-driven wrapper around
:func:`~df_mlx.train_dynamic.train`.  Callers can construct a session from
either explicit keyword arguments or a :class:`~df_mlx.run_config.RunConfig`,
then invoke ``session.run()`` to execute the full training pipeline.

Key exports:
    - TrainingSession: High-level session object wrapping train().
    - _kwargs_from_run_config: Convert a RunConfig into train() kwargs dict.
    - _TRAIN_KWARGS: Canonical ordered tuple of all train() keyword argument names.

Relationship to train_dynamic:
    All public symbols are re-exported via train_dynamic.py for backward
    compatibility.  TrainingSession.run() delegates directly to train().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from df_mlx.run_config import RunConfig

_SENTINEL = object()

# All keyword argument names accepted by ``train()`` in parameter order.
_TRAIN_KWARGS: tuple[str, ...] = (
    "cache_dir",
    "speech_list",
    "noise_list",
    "rir_list",
    "config_path",
    "epochs",
    "batch_size",
    "learning_rate",
    "learning_rate_min",
    "weight_decay",
    "checkpoint_dir",
    "resume_from",
    "resume_data_from",
    "validate_every",
    "save_strategy",
    "save_steps",
    "save_total_limit",
    "checkpoint_batches",
    "max_grad_norm",
    "warmup_epochs",
    "patience",
    "num_workers",
    "prefetch_size",
    "p_reverb",
    "p_clipping",
    "use_mlx_data",
    "use_fp16",
    "grad_accumulation_steps",
    "eval_frequency",
    "backbone_type",
    "model_variant",
    "verbose",
    "snr_range",
    "snr_range_extreme",
    "snr_range_very_low",
    "p_extreme_snr",
    "p_very_low_snr",
    "p_interfer_speech",
    "curriculum_warmup_epochs",
    "speech_gain_range",
    "noise_gain_range",
    "dynamic_loss",
    "pipeline_stages",
    "awesome_loss_weight",
    "awesome_mask_sharpness",
    "awesome_warmup_steps",
    "gan_enabled",
    "gan_start_epoch",
    "gan_ramp_epochs",
    "gan_adv_weight",
    "gan_fm_weight",
    "gan_disc_type",
    "gan_mpd_periods",
    "gan_msd_scales",
    "gan_disc_lr",
    "gan_disc_weight_decay",
    "gan_disc_grad_clip",
    "gan_disc_update_freq",
    "gan_disc_max_samples",
    "gan_cache_gen_waveforms",
    "gan_disc_gradient_checkpoint",
    "gan_gen_gradient_checkpoint",
    "gan_eval_frequency",
    "gan_mpd_channels",
    "gan_msd_channels",
    "experimental_compiled_gan",
    "vad_proxy_enabled",
    "vad_loss_weight",
    "vad_threshold",
    "vad_margin",
    "vad_speech_loss_weight",
    "vad_warmup_epochs",
    "vad_snr_gate_db",
    "vad_snr_gate_width",
    "vad_band_low_hz",
    "vad_band_high_hz",
    "vad_z_threshold",
    "vad_z_slope",
    "vad_eval_mode",
    "vad_eval_every",
    "vad_eval_batches",
    "vad_eval_max_seconds",
    "vad_silero_model_path",
    "vad_silero_sample_rate",
    "vad_train_prob",
    "vad_train_every_steps",
    "eval_sisdr",
    "max_train_batches",
    "max_valid_batches",
    "check_chkpts",
    "seed",
    "debug_numerics",
    "debug_numerics_fail_fast",
    "debug_numerics_every",
    "debug_numerics_dump_dir",
    "debug_numerics_dump_arrays",
    "debug_numerics_max_dumps",
    "nan_skip_batch",
    "sync_mode",
    "model_config",
    "dataset_overrides",
    "mrstft_config",
    "train_config_path",
)


def _kwargs_from_run_config(run_config: RunConfig) -> dict[str, Any]:
    """Extract ``train()`` keyword arguments from a :class:`RunConfig`.

    This mirrors the mapping used in ``training_cli_main.main()`` so that
    ``TrainingSession.from_run_config`` stays in sync with the CLI path.
    """
    cfg = run_config
    return dict(
        cache_dir=cfg.dataset.cache_dir,
        speech_list=cfg.dataset.speech_list,
        noise_list=cfg.dataset.noise_list,
        rir_list=cfg.dataset.rir_list,
        config_path=cfg.dataset.config,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        learning_rate_min=cfg.training.learning_rate_min,
        weight_decay=cfg.training.weight_decay,
        checkpoint_dir=cfg.checkpoint.checkpoint_dir,
        resume_from=None,  # resolved externally (CLI handles auto-resume)
        resume_data_from=None,
        validate_every=cfg.checkpoint.validate_every,
        save_strategy=cast(Literal["no", "epoch", "steps"], cfg.checkpoint.save_strategy),
        save_steps=cfg.checkpoint.save_steps,
        save_total_limit=cfg.checkpoint.save_total_limit,
        checkpoint_batches=cfg.checkpoint.checkpoint_batches,
        max_grad_norm=cfg.training.max_grad_norm,
        warmup_epochs=cfg.training.warmup_epochs,
        patience=cfg.training.patience,
        num_workers=cfg.dataloader.num_workers,
        prefetch_size=cfg.dataloader.prefetch_size,
        p_reverb=cfg.augmentation.p_reverb,
        p_clipping=cfg.augmentation.p_clipping,
        use_mlx_data=cfg.dataloader.use_mlx_data,
        use_fp16=cfg.training.fp16,
        grad_accumulation_steps=cfg.training.grad_accumulation_steps,
        eval_frequency=cfg.training.eval_frequency,
        backbone_type=cast(Literal["mamba", "gru", "attention"], cfg.model.backbone_type),
        model_variant=cast(Literal["full", "lite"], cfg.model.variant),
        verbose=cfg.debug.verbose,
        snr_range=cfg.dataset.snr_range,
        snr_range_extreme=cfg.dataset.snr_range_extreme,
        snr_range_very_low=cfg.dataset.snr_range_very_low,
        p_extreme_snr=cfg.dataset.p_extreme_snr,
        p_very_low_snr=cfg.dataset.p_very_low_snr,
        p_interfer_speech=cfg.dataset.p_interfer_speech,
        curriculum_warmup_epochs=cfg.training.curriculum_warmup_epochs,
        speech_gain_range=cfg.dataset.speech_gain_range,
        noise_gain_range=cfg.dataset.noise_gain_range,
        dynamic_loss=cast(
            Literal["baseline", "awesome", "pipeline_awesome"],
            cfg.loss.dynamic_loss,
        ),
        pipeline_stages=cfg.loss.pipeline_stages,
        awesome_loss_weight=cfg.loss.awesome.loss_weight,
        awesome_mask_sharpness=cfg.loss.awesome.mask_sharpness,
        awesome_warmup_steps=cfg.loss.awesome.warmup_steps,
        gan_enabled=cfg.gan.enabled,
        gan_start_epoch=cfg.gan.start_epoch,
        gan_ramp_epochs=cfg.gan.ramp_epochs,
        gan_adv_weight=cfg.gan.adv_weight,
        gan_fm_weight=cfg.gan.fm_weight,
        gan_disc_type=cast(Literal["combined", "mpd", "msd"], cfg.gan.discriminator),
        gan_mpd_periods=(tuple(cfg.gan.mpd_periods) if cfg.gan.mpd_periods else None),
        gan_msd_scales=cfg.gan.msd_scales,
        gan_disc_lr=cfg.gan.disc_lr,
        gan_disc_weight_decay=cfg.gan.disc_weight_decay,
        gan_disc_grad_clip=cfg.gan.disc_grad_clip,
        gan_disc_update_freq=cfg.gan.disc_update_freq,
        gan_disc_max_samples=cfg.gan.disc_max_samples,
        gan_mpd_channels=cfg.gan.mpd_channels,
        gan_msd_channels=cfg.gan.msd_channels,
        experimental_compiled_gan=cfg.gan.experimental_compile,
        gan_cache_gen_waveforms=cfg.gan.cache_gen_waveforms,
        gan_disc_gradient_checkpoint=cfg.gan.disc_gradient_checkpoint,
        gan_gen_gradient_checkpoint=cfg.gan.gen_gradient_checkpoint,
        gan_eval_frequency=cfg.gan.eval_frequency,
        vad_proxy_enabled=cfg.loss.awesome.proxy_enabled,
        vad_loss_weight=cfg.vad.loss_weight,
        vad_threshold=cfg.vad.threshold,
        vad_margin=cfg.vad.margin,
        vad_speech_loss_weight=cfg.vad.speech_loss_weight,
        vad_warmup_epochs=cfg.vad.warmup_epochs,
        vad_snr_gate_db=cfg.vad.snr_gate_db,
        vad_snr_gate_width=cfg.vad.snr_gate_width,
        vad_band_low_hz=cfg.vad.band_low_hz,
        vad_band_high_hz=cfg.vad.band_high_hz,
        vad_z_threshold=cfg.vad.z_threshold,
        vad_z_slope=cfg.vad.z_slope,
        vad_eval_mode=cast(Literal["auto", "proxy", "silero", "off"], cfg.vad.eval.mode),
        vad_eval_every=cfg.vad.eval.every,
        vad_eval_batches=cfg.vad.eval.batches,
        vad_eval_max_seconds=cfg.vad.eval.max_seconds,
        vad_silero_model_path=cfg.vad.eval.silero_model_path,
        vad_silero_sample_rate=cfg.vad.eval.silero_sample_rate,
        vad_train_prob=cfg.vad.train.prob,
        vad_train_every_steps=cfg.vad.train.every_steps,
        eval_sisdr=cfg.metrics.eval_sisdr,
        check_chkpts=cfg.checkpoint.check_chkpts,
        max_train_batches=cfg.dataloader.max_train_batches,
        max_valid_batches=cfg.dataloader.max_valid_batches,
        seed=cfg.training.seed,
        debug_numerics=cfg.debug.debug_numerics,
        debug_numerics_fail_fast=cfg.debug.debug_numerics_fail_fast,
        debug_numerics_every=cfg.debug.debug_numerics_every,
        debug_numerics_dump_dir=cfg.debug.debug_numerics_dump_dir,
        debug_numerics_dump_arrays=cfg.debug.debug_numerics_dump_arrays,
        debug_numerics_max_dumps=cfg.debug.debug_numerics_max_dumps,
        nan_skip_batch=cfg.debug.nan_skip_batch,
        sync_mode=cfg.debug.sync_mode,
        model_config=None,
        dataset_overrides=None,
        mrstft_config=cfg.loss.mrstft,
        train_config_path=cfg.training.train_config,
    )


class TrainingSession:
    """Class-based API for DfNet4 dynamic training.

    Wraps :func:`~df_mlx.train_dynamic.train` with an object-oriented
    interface.  Currently a thin delegation layer; future phases will
    incrementally migrate the ``train()`` body into class methods.

    Usage::

        # Direct keyword construction (mirrors train() signature)
        session = TrainingSession(epochs=50, batch_size=16, learning_rate=3e-4)
        session.setup()
        session.run()

        # From a RunConfig object
        session = TrainingSession.from_run_config(run_config, epochs=200)
        session.setup()
        session.run()
    """

    def __init__(self, **kwargs: Any) -> None:
        unknown = set(kwargs) - set(_TRAIN_KWARGS)
        if unknown:
            raise TypeError(f"TrainingSession received unexpected keyword arguments: {sorted(unknown)}")
        self._kwargs: dict[str, Any] = kwargs
        self._ready: bool = False
        # Extension-point attributes — reserved for future refactoring phases
        # that will move model init, dataset creation, and optimizer setup into
        # the session object.  Kept as typed placeholders so the interface is
        # stable when that work lands.
        self.state: Any | None = None
        self.step: Any | None = None
        self.validation: Any | None = None
        self.loop: Any | None = None

    @classmethod
    def from_run_config(cls, run_config: RunConfig, **overrides: Any) -> TrainingSession:
        """Create a session from a :class:`~df_mlx.run_config.RunConfig`.

        Extracts all ``train()`` keyword arguments using the same mapping as
        ``training_cli_main.main()``.  Any *overrides* are applied on top,
        allowing callers to tweak individual parameters without mutating the
        config object.
        """
        kwargs = _kwargs_from_run_config(run_config)
        kwargs.update(overrides)
        return cls(**kwargs)

    def setup(self) -> None:
        """Prepare the session for execution.

        For now this is intentionally lightweight and only marks the
        session as ready — "ready" simply means ``setup()`` has been
        called; no validation of arguments or initialisation of external
        resources occurs here.  The heavy lifting remains in ``train()``.
        """
        self._ready = True

    def run(self) -> None:
        """Execute the training loop."""
        if not self._ready:
            self.setup()
        from df_mlx.train_dynamic import train

        train(**self._kwargs)

    @property
    def kwargs(self) -> dict[str, Any]:
        """Return a copy of the stored keyword arguments."""
        return dict(self._kwargs)
