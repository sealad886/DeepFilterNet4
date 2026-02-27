#!/usr/bin/env python3
"""Orchestrator for MLX DeepFilterNet4 dynamic training.

This module is the central hub for the DFNet4 training pipeline.  It defines
the ``train()`` function, which contains the loss closures (``loss_fn``,
``loss_fn_gan``) and their ``mx.compile``-wrapped training steps — these must
remain as closures due to MLX autograd/compile semantics.  All other concerns
(dataset setup, checkpointing, validation, metrics, CLI, etc.) are delegated
to purpose-specific ``training_*.py`` modules.

For backward compatibility, this module re-exports every public symbol from
eight helper modules (see the ``# noqa: F401`` block and
``test_train_dynamic_reexports.py``).

Key exports:
    - train: Main entry-point — builds closures, runs epoch/batch loop.
    - (re-exports): All public symbols from training_checkpoints,
      training_cli, training_cli_main, training_losses, training_ops,
      training_session, training_signals, and training_waveform.

Usage:
    python -m df_mlx.train_dynamic \\
        --speech-list /path/to/speech_files.txt \\
        --noise-list /path/to/noise_files.txt \\
        --rir-list /path/to/rir_files.txt \\
        --epochs 100 --batch-size 8 --checkpoint-dir ./checkpoints
"""

from __future__ import annotations

# ── Standard library + third-party ──────────────────────────────────
import gc
import math
import os
import random
import sys
import time
from itertools import islice
from typing import TYPE_CHECKING, Any, Literal, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm.auto import tqdm

# ── Imports used locally (also re-exported for backward compat) ──────
from df_mlx.hardware import print_hardware_diagnostics
from df_mlx.run_config import SyncMode
from df_mlx.training_checkpoints import (
    _IN_PROGRESS_KINDS,
    _TRAIN_MODE_EAGER,
    _write_epoch_complete_marker,
    cleanup_checkpoints,
    maybe_skip_resume_batches,
    reconcile_resume,
    resolve_epoch_train_mode,
    save_checkpoint,
)
from df_mlx.training_cli import _resolve_pipeline_stage
from df_mlx.training_cli_main import main
from df_mlx.training_diagnostics import (
    DiagnosticContext,
)
from df_mlx.training_diagnostics import diagnose_nonfinite as _diagnose_nonfinite_impl
from df_mlx.training_helpers import (
    TrainingLoopState,
    _resolve_pipeline_stage_by_index,
)
from df_mlx.training_helpers import build_setup_panel_line as _build_setup_panel_line
from df_mlx.training_helpers import clip_gan_scores as _clip_gan_scores
from df_mlx.training_helpers import (
    curriculum_schedule,
)
from df_mlx.training_helpers import is_vad_train_reg_enabled as _is_vad_train_reg_enabled  # noqa: F401
from df_mlx.training_helpers import (
    print_compiled_step_eligibility,
)
from df_mlx.training_losses import (
    _compute_awesome_losses,
    _compute_pipeline_awesome_losses,
    _compute_speech_band_logmag_loss,
    _compute_vad_loss,
    _compute_vad_reg_loss,
)
from df_mlx.training_metrics import (
    collect_sync_metrics,
    compute_epoch_averages,
    create_epoch_accums,
    update_progress_bar,
)
from df_mlx.training_ops import (
    _tree_all_finite,
    accumulate_grads,
    clip_grad_norm,
    scale_grads,
)
from df_mlx.training_setup import (
    _sync_model_config_with_dataset,
    build_train_config,
    finalize_training,
    print_epoch_summary,
    print_training_config,
    setup_auxiliary_losses,
    setup_data_pipeline,
    setup_dataset,
)
from df_mlx.training_signals import (
    _interrupt_state,
    _register_sigint_handler,
    _update_interrupt_state,
)
from df_mlx.training_validation import ValidationContext
from df_mlx.training_validation import run_validation as _run_validation
from df_mlx.training_waveform import (
    _disc_crop_waveform,
    _gan_waveform_view,
    specs_to_wavs,
)

# isort: split
# ── Pure re-exports (backward compat — see test_train_dynamic_reexports.py) ──
from df_mlx.training_checkpoints import (  # noqa: F401
    _CHECKPOINT_KINDS,
    _COMPLETED_KINDS,
    _COUNTER_SEMANTICS_VERSION,
    _TRAIN_MODE_COMPILED,
    CheckpointManifest,
    CheckpointRecord,
    ResumeResult,
    _disc_weights_path,
    _is_disc_weights,
    _record_sort_key,
    _validate_checkpoint_pair,
    compute_resume_epoch,
    find_latest_checkpoint,
    load_checkpoint,
    resolve_resume_batch_count,
    validate_checkpoint_dir,
)
from df_mlx.training_cli import (  # noqa: F401
    _apply_cli_overrides,
    _flag_in_argv,
    _parse_pipeline_stages_cli,
)
from df_mlx.training_losses import (  # noqa: F401
    _AWESOME_ENERGY_BOOST_DB,
    _AWESOME_ENERGY_BOOST_WIDTH,
    _AWESOME_LOW_ENERGY_WEIGHT,
    _AWESOME_LOW_SNR_WEIGHT,
    _AWESOME_MASK_LOGIT_CLAMP,
    _AWESOME_MOD_THRESHOLD,
    _AWESOME_MOD_WIDTH,
    _AWESOME_MUSIC_FLUX_THR,
    _AWESOME_MUSIC_FLUX_WIDTH,
    _AWESOME_MUSICNESS_THR,
    _AWESOME_MUSICNESS_WIDTH,
    _AWESOME_PROXY_RATIO_FLOOR,
    _AWESOME_PROXY_RATIO_SCALE,
    _AWESOME_SMOOTH_WEIGHT,
    _EPS,
    _PIPELINE_ARTIFACT_SMOOTH_WEIGHT,
    _PIPELINE_LOW_ENERGY_ADDITIVE,
    _PIPELINE_LOW_SNR_ADDITIVE,
    _PIPELINE_MIN_MASK_FLOOR,
    _PIPELINE_MUSIC_SUPPRESSION_WEIGHT,
    _PIPELINE_PITCH_STABILITY_THR,
    _PIPELINE_PROXY_FLOOR,
    _PIPELINE_VOCAL_HARMONIC_THR,
    _VAD_LOGIT_CLAMP,
    _build_speech_band_mask,
    _compute_harmonic_ratio,
    _compute_improved_musicness,
    _compute_musicness,
    _compute_pitch_stability,
    _compute_proxy_gates,
    _compute_vad_eval_metrics,
    _compute_vad_probs,
    _log1p_mag,
    _snr_bucket_name,
)
from df_mlx.training_ops import (  # noqa: F401
    NumericDebugConfig,
    NumericDebugger,
    _batch_to_float,
)
from df_mlx.training_session import (  # noqa: F401
    _SENTINEL,
    _TRAIN_KWARGS,
    TrainingSession,
    _kwargs_from_run_config,
)
from df_mlx.training_signals import _handle_sigint  # noqa: F401
from df_mlx.training_waveform import compute_mrstft_loss  # noqa: F401

if TYPE_CHECKING:
    from df_mlx.config import ModelParams4
    from df_mlx.run_config import MultiResSpecLossConfig

# Canonical definition lives in training_helpers; alias here for local use.
from df_mlx.training_helpers import SCALAR_ZERO  # noqa: E402

# =============================================================================
# tqdm configuration
# =============================================================================
# Write progress bars to stderr so stdout can be redirected to a log file without
# capturing the progress bar spam. Also auto-disable tqdm when stderr isn't a TTY
# (e.g., when piping/redirecting), which prevents log files from being flooded.
_tqdm_env = os.getenv("DFNET_TQDM", "").strip().lower()
if _tqdm_env in {"1", "true", "yes", "on"}:
    _tqdm_disable = False
elif _tqdm_env in {"0", "false", "no", "off"}:
    _tqdm_disable = True
else:
    # Default: disable when stderr isn't interactive (prevents log spam when piped).
    _tqdm_disable = not sys.stderr.isatty()

_TQDM_KWARGS = {
    "file": sys.stderr,
    "disable": _tqdm_disable,
    "mininterval": 1.0,
    "maxinterval": 10.0,
    "dynamic_ncols": True,
}

_tqdm_panels_env = os.getenv("DFNET_TQDM_PANELS", "").strip().lower()
if _tqdm_panels_env in {"1", "true", "yes", "on"}:
    _tqdm_panels = True
elif _tqdm_panels_env in {"0", "false", "no", "off"}:
    _tqdm_panels = False
else:
    # Default on interactive terminals only.
    _tqdm_panels = not _tqdm_disable


def train(
    cache_dir: str | None = None,
    speech_list: str | None = None,
    noise_list: str | None = None,
    rir_list: str | None = None,
    config_path: str | None = None,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    learning_rate_min: float | None = None,
    weight_decay: float = 0.0,
    checkpoint_dir: str = "checkpoints",
    resume_from: str | None = None,
    resume_data_from: str | None = None,
    validate_every: int = 1,
    save_strategy: Literal["no", "epoch", "steps"] = "epoch",
    save_steps: int = 500,
    save_total_limit: int | None = None,
    checkpoint_batches: int = 0,
    max_grad_norm: float = 1.0,
    warmup_epochs: int = 5,
    patience: int = 10,
    num_workers: int = 4,
    prefetch_size: int = 8,
    p_reverb: float = 0.5,
    p_clipping: float = 0.0,
    use_mlx_data: bool = True,
    use_fp16: bool | None = None,
    grad_accumulation_steps: int = 1,
    eval_frequency: int = 10,
    backbone_type: Literal["mamba", "gru", "attention"] = "mamba",
    model_variant: Literal["full", "lite"] = "full",
    verbose: bool = False,
    snr_range: Tuple[float, float] | None = None,
    snr_range_extreme: Tuple[float, float] | None = None,
    snr_range_very_low: Tuple[float, float] | None = None,
    p_extreme_snr: float | None = None,
    p_very_low_snr: float | None = None,
    p_interfer_speech: float | None = None,
    curriculum_warmup_epochs: int = 0,
    speech_gain_range: Tuple[float, float] | None = None,
    noise_gain_range: Tuple[float, float] | None = None,
    dynamic_loss: Literal["baseline", "awesome", "pipeline_awesome"] = "baseline",
    pipeline_stages: list[dict[str, Any]] | None = None,
    awesome_loss_weight: float = 0.4,
    awesome_mask_sharpness: float = 6.0,
    awesome_warmup_steps: int = 0,
    gan_enabled: bool = False,
    gan_start_epoch: int = 0,
    gan_ramp_epochs: int = 0,
    gan_adv_weight: float = 0.0,
    gan_fm_weight: float = 0.0,
    gan_disc_type: Literal["combined", "mpd", "msd"] = "combined",
    gan_mpd_periods: Tuple[int, ...] | None = None,
    gan_msd_scales: int = 3,
    gan_disc_lr: float = 1e-4,
    gan_disc_weight_decay: float = 0.0,
    gan_disc_grad_clip: float = 1.0,
    gan_disc_update_freq: int = 1,
    gan_disc_max_samples: int = 48000,
    gan_cache_gen_waveforms: bool = True,
    gan_disc_gradient_checkpoint: bool = False,
    gan_gen_gradient_checkpoint: bool = False,
    gan_eval_frequency: int = 2,
    gan_mpd_channels: int = 32,
    gan_msd_channels: int = 128,
    experimental_compiled_gan: bool = False,
    vad_proxy_enabled: bool = True,
    vad_loss_weight: float = 0.05,
    vad_threshold: float = 0.6,
    vad_margin: float = 0.05,
    vad_speech_loss_weight: float = 0.0,
    vad_warmup_epochs: int = 5,
    vad_snr_gate_db: float = -10.0,
    vad_snr_gate_width: float = 6.0,
    vad_band_low_hz: float = 300.0,
    vad_band_high_hz: float = 3400.0,
    vad_z_threshold: float = 0.0,
    vad_z_slope: float = 1.0,
    vad_eval_mode: Literal["auto", "proxy", "silero", "off"] = "auto",
    vad_eval_every: int = 1,
    vad_eval_batches: int = 8,
    vad_eval_max_seconds: float = 0.0,
    vad_silero_model_path: str | None = None,
    vad_silero_sample_rate: int = 16000,
    vad_train_prob: float = 0.0,
    vad_train_every_steps: int = 0,
    eval_sisdr: bool = False,
    max_train_batches: int | None = None,
    max_valid_batches: int | None = None,
    check_chkpts: bool = False,
    seed: int | None = None,
    debug_numerics: bool = False,
    debug_numerics_fail_fast: bool = True,
    debug_numerics_every: int = 1,
    debug_numerics_dump_dir: str | None = None,
    debug_numerics_dump_arrays: bool = False,
    debug_numerics_max_dumps: int = 5,
    nan_skip_batch: bool = False,
    sync_mode: str = "normal",
    model_config: ModelParams4 | None = None,
    dataset_overrides: dict[str, Any] | None = None,
    mrstft_config: MultiResSpecLossConfig | None = None,
    train_config_path: str | None = None,
) -> None:
    """Train DfNet4 model with dynamic on-the-fly mixing.

    Args:
        cache_dir: Path to pre-built audio cache (from build_audio_cache.py)
        speech_list: Path to file containing speech file paths (if no cache)
        noise_list: Path to file containing noise file paths (if no cache)
        rir_list: Path to file containing RIR file paths (if no cache)
        config_path: Optional path to JSON config file
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        learning_rate_min: Minimum learning rate for cosine schedule
        weight_decay: Weight decay for AdamW optimizer
        checkpoint_dir: Directory for checkpoints
        resume_from: Optional model checkpoint to resume from
        resume_data_from: Optional data checkpoint for resuming interrupted epoch
        validate_every: Validate every N epochs
        save_strategy: Additional checkpoint cadence ("no", "epoch", or "steps"). End-of-epoch checkpoints are always saved for resume integrity.
        save_steps: Number of steps between checkpoints (when save_strategy="steps")
        save_total_limit: Maximum number of checkpoints to keep (None=unlimited)
        checkpoint_batches: Save data checkpoint every N batches (0=disabled)
        max_grad_norm: Maximum gradient norm for clipping
        warmup_epochs: Number of warmup epochs
        patience: Early stopping patience
        num_workers: Number of data loading workers
        prefetch_size: Number of batches to prefetch (for MLXDataStream)
        p_reverb: Probability of applying reverb
        p_clipping: Probability of clipping distortion
        use_mlx_data: Use MLXDataStream if available (faster, with checkpointing)
        use_fp16: Use FP16 (half-precision) training. None=auto-detect from hardware
        grad_accumulation_steps: Number of steps to accumulate gradients (effective batch = batch_size * grad_accumulation_steps)
        eval_frequency: Evaluate loss every N batches (reduces synchronization overhead)
        model_variant: Model size variant ("full" or "lite")
        verbose: Enable detailed timing and diagnostic output
        snr_range: Optional override for base SNR range (dB)
        snr_range_extreme: Optional override for extreme SNR range (dB)
        snr_range_very_low: Optional override for very-low SNR range (dB), for whisper/distant mic
        p_extreme_snr: Optional override for extreme SNR sampling probability
        p_very_low_snr: Optional override for very-low SNR sampling probability
        p_interfer_speech: Optional override for interfering speaker probability (simulates vocals/competing talker)
        curriculum_warmup_epochs: Number of warmup epochs for curriculum learning (0=disabled).
            During warmup, SNR/interferer probabilities ramp from 0 to target values.
        speech_gain_range: Optional override for speech gain range (dB)
        noise_gain_range: Optional override for noise gain range (dB)
        dynamic_loss: Which dynamic loss to use ("baseline", "awesome", or "pipeline_awesome")
        pipeline_stages: Optional staged loss schedule with entries containing
            start_epoch and optional overrides for awesome_loss_weight,
            vad_loss_weight, and vad_speech_loss_weight.
        awesome_loss_weight: Weight for awesome loss term (only if enabled)
        awesome_mask_sharpness: Sharpness for speech/noise dominance mask
        awesome_warmup_steps: Warmup steps for awesome loss weight ramp
        vad_proxy_enabled: Enable cheap VAD proxy gating for awesome loss
        vad_loss_weight: Weight for VAD speech-preservation loss
        vad_threshold: VAD probability threshold for speech gating
        vad_margin: Margin for VAD consistency loss
        vad_speech_loss_weight: Weight for VAD-weighted speech-structure loss
        vad_warmup_epochs: Warmup epochs for ramping VAD loss weight
        vad_snr_gate_db: SNR threshold for VAD gating (dB)
        vad_snr_gate_width: SNR gate softness (dB)
        vad_band_low_hz: Low cutoff for speech band (Hz)
        vad_band_high_hz: High cutoff for speech band (Hz)
        vad_z_threshold: Z-score threshold for VAD sigmoid
        vad_z_slope: Z-score slope for VAD sigmoid
        vad_eval_mode: VAD evaluation mode ("auto", "proxy", "silero", "off")
        vad_eval_every: Evaluate VAD metrics every N epochs
        vad_eval_batches: Number of validation batches used for VAD metrics
        vad_eval_max_seconds: Max seconds per clip for VAD eval (0 disables)
        vad_silero_model_path: Optional path to silero_vad.onnx
        vad_silero_sample_rate: Sample rate for Silero VAD (Hz)
        vad_train_prob: Probability of applying sparse VAD regularizer per batch
        vad_train_every_steps: Apply VAD regularizer every N steps (0 disables)
        eval_sisdr: Compute SI-SDR during validation (slower)
        max_train_batches: Limit number of train batches per epoch (None = full epoch)
        max_valid_batches: Limit number of validation batches (None = full validation)
        check_chkpts: Validate checkpoints before starting/resuming
        seed: Optional RNG seed override (sets Python/NumPy/MLX RNGs)
        debug_numerics: Enable numeric debug mode with finite checks and fail-fast behavior
        debug_numerics_fail_fast: Raise on first non-finite when debug_numerics enabled
        debug_numerics_every: Check every N steps in debug mode
        debug_numerics_dump_dir: Directory for numeric debug dumps (default: checkpoint_dir/debug_numerics)
        debug_numerics_dump_arrays: Save small tensor slices alongside JSON dumps
        debug_numerics_max_dumps: Maximum number of non-finite dumps to write
        nan_skip_batch: Skip optimizer update when loss/grads are non-finite (debug-friendly)
        sync_mode: Sync barrier budget (fast | normal | debug | profile)
        model_config: Optional MLX model config overrides (ModelParams4)
        dataset_overrides: Optional dataset config overrides (applied before CLI overrides)
        mrstft_config: Optional multi-res STFT loss config
        train_config_path: Optional path to INI train config (stored in metadata)
    """
    # Reset the session-scoped non-finite loss counter so consecutive
    # train() invocations in the same process don't carry over stale counts.
    train._nonfinite_loss_count = 0  # type: ignore[attr-defined]

    from df_mlx.config import get_default_config
    from df_mlx.dynamic_dataset import (
        DynamicDataset,
        PrefetchDataLoader,
    )
    from df_mlx.hardware import HardwareConfig
    from df_mlx.model import count_parameters, init_model
    from df_mlx.train import WarmupCosineSchedule, spectral_loss

    print("=" * 60)
    print("MLX DeepFilterNet4 Training - Dynamic On-the-Fly Mixing")
    print("=" * 60)

    # Detect hardware and get optimal settings
    hw_config = HardwareConfig.detect(verbose=verbose)

    # Determine FP16 setting
    if use_fp16 is None:
        use_fp16 = hw_config.use_fp16
    print(f"  Mixed precision (BF16): {'enabled' if use_fp16 else 'disabled'}")

    # Print hardware diagnostics in verbose mode
    if verbose:
        print_hardware_diagnostics()

    ds_result = setup_dataset(
        cache_dir=cache_dir,
        config_path=config_path,
        speech_list=speech_list,
        noise_list=noise_list,
        rir_list=rir_list,
        p_reverb=p_reverb,
        p_clipping=p_clipping,
        num_workers=num_workers,
        dataset_overrides=dataset_overrides,
        snr_range=snr_range,
        snr_range_extreme=snr_range_extreme,
        snr_range_very_low=snr_range_very_low,
        p_extreme_snr=p_extreme_snr,
        p_very_low_snr=p_very_low_snr,
        p_interfer_speech=p_interfer_speech,
        speech_gain_range=speech_gain_range,
        noise_gain_range=noise_gain_range,
        debug_numerics=debug_numerics,
        max_train_batches=max_train_batches,
        max_valid_batches=max_valid_batches,
        eval_frequency=eval_frequency,
        prefetch_size=prefetch_size,
        use_mlx_data=use_mlx_data,
        seed=seed,
    )
    config = ds_result.config
    seed = ds_result.seed
    max_train_batches = ds_result.max_train_batches
    max_valid_batches = ds_result.max_valid_batches
    eval_frequency = ds_result.eval_frequency
    num_workers = ds_result.num_workers
    prefetch_size = ds_result.prefetch_size
    use_mlx_data = ds_result.use_mlx_data

    # Create dataset (this populates config.*_files from cache index if using cache)
    print("\nInitializing dynamic dataset...")
    dataset = DynamicDataset(config)

    _aux = setup_auxiliary_losses(
        config=config,
        dynamic_loss=dynamic_loss,
        pipeline_stages=pipeline_stages,
        awesome_loss_weight=awesome_loss_weight,
        vad_loss_weight=vad_loss_weight,
        vad_speech_loss_weight=vad_speech_loss_weight,
        mrstft_config=mrstft_config,
        gan_enabled=gan_enabled,
        gan_adv_weight=gan_adv_weight,
        gan_fm_weight=gan_fm_weight,
        gan_disc_type=gan_disc_type,
        gan_mpd_periods=gan_mpd_periods,
        gan_mpd_channels=gan_mpd_channels,
        gan_msd_scales=gan_msd_scales,
        gan_msd_channels=gan_msd_channels,
        gan_disc_lr=gan_disc_lr,
        gan_disc_weight_decay=gan_disc_weight_decay,
        gan_disc_update_freq=gan_disc_update_freq,
        vad_eval_mode=vad_eval_mode,
        vad_silero_model_path=vad_silero_model_path,
        vad_silero_sample_rate=vad_silero_sample_rate,
        vad_eval_max_seconds=vad_eval_max_seconds,
        vad_band_low_hz=vad_band_low_hz,
        vad_band_high_hz=vad_band_high_hz,
        vad_train_prob=vad_train_prob,
        vad_train_every_steps=vad_train_every_steps,
    )
    use_awesome_loss = _aux.use_awesome_loss
    use_pipeline_awesome_loss = _aux.use_pipeline_awesome_loss
    pipeline_stage_defs = _aux.pipeline_stage_defs
    base_awesome_loss_weight = _aux.base_awesome_loss_weight
    base_vad_loss_weight = _aux.base_vad_loss_weight
    base_vad_speech_loss_weight = _aux.base_vad_speech_loss_weight
    mrstft_cfg = _aux.mrstft_cfg
    use_mrstft_loss = _aux.use_mrstft_loss
    mrstft_loss_fn = _aux.mrstft_loss_fn
    mrstft_istft = _aux.mrstft_istft
    mrstft_target_len = _aux.mrstft_target_len
    gan_enabled = _aux.gan_enabled
    gan_target_len = _aux.gan_target_len
    gan_istft = _aux.gan_istft
    gan_disc_type = _aux.gan_disc_type
    gan_disc_update_freq = _aux.gan_disc_update_freq
    discriminator = _aux.discriminator
    disc_optimizer = _aux.disc_optimizer
    feature_match_loss = _aux.feature_match_loss
    gan_loss_fns = _aux.gan_loss_fns
    vad_eval_enabled = _aux.vad_eval_enabled
    vad_eval_mode = _aux.vad_eval_mode
    silero_vad = _aux.silero_vad
    vad_band_mask = _aux.vad_band_mask
    vad_band_bins = _aux.vad_band_bins
    use_vad_loss = _aux.use_vad_loss
    use_vad_train_reg = _aux.use_vad_train_reg
    del _aux

    min_lr = learning_rate_min if learning_rate_min is not None else learning_rate * 0.01

    vad_enabled = print_training_config(
        config,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        min_lr=min_lr,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir,
        dynamic_loss=dynamic_loss,
        mrstft_cfg=mrstft_cfg,
        awesome_loss_weight=awesome_loss_weight,
        awesome_mask_sharpness=awesome_mask_sharpness,
        awesome_warmup_steps=awesome_warmup_steps,
        vad_proxy_enabled=vad_proxy_enabled,
        gan_enabled=gan_enabled,
        gan_adv_weight=gan_adv_weight,
        gan_fm_weight=gan_fm_weight,
        gan_start_epoch=gan_start_epoch,
        gan_ramp_epochs=gan_ramp_epochs,
        gan_disc_type=gan_disc_type,
        gan_mpd_periods=gan_mpd_periods,
        gan_msd_scales=gan_msd_scales,
        gan_disc_update_freq=gan_disc_update_freq,
        gan_disc_max_samples=gan_disc_max_samples,
        gan_mpd_channels=gan_mpd_channels,
        gan_msd_channels=gan_msd_channels,
        vad_loss_weight=vad_loss_weight,
        vad_speech_loss_weight=vad_speech_loss_weight,
        vad_threshold=vad_threshold,
        vad_margin=vad_margin,
        vad_warmup_epochs=vad_warmup_epochs,
        vad_snr_gate_db=vad_snr_gate_db,
        vad_snr_gate_width=vad_snr_gate_width,
        vad_band_low_hz=vad_band_low_hz,
        vad_band_high_hz=vad_band_high_hz,
        vad_eval_mode=vad_eval_mode,
        vad_eval_every=vad_eval_every,
        vad_eval_batches=vad_eval_batches,
        vad_eval_max_seconds=vad_eval_max_seconds,
        vad_silero_sample_rate=vad_silero_sample_rate,
        vad_silero_model_path=vad_silero_model_path,
        use_vad_train_reg=use_vad_train_reg,
        vad_train_prob=vad_train_prob,
        vad_train_every_steps=vad_train_every_steps,
        pipeline_stage_defs=pipeline_stage_defs,
    )

    tqdm_setup_panel = None
    tqdm_train_position = 0
    tqdm_valid_position = 0
    if _tqdm_panels:
        setup_line = _build_setup_panel_line(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dynamic_loss=dynamic_loss,
            gan_enabled=gan_enabled,
            vad_enabled=vad_enabled,
            checkpoint_dir=checkpoint_dir,
            use_fp16=bool(use_fp16),
        )
        tqdm_setup_panel = tqdm(
            total=1,
            desc=setup_line,
            bar_format="{desc}",
            position=0,
            leave=True,
            **_TQDM_KWARGS,
        )
        tqdm_setup_panel.update(1)
        tqdm_train_position = 1
        tqdm_valid_position = 2

    train_config = build_train_config(
        config,
        mrstft_cfg=mrstft_cfg,
        gan_mpd_periods=gan_mpd_periods,
        pipeline_stage_defs=pipeline_stage_defs,
        train_config_path=train_config_path,
        dynamic_loss=dynamic_loss,
        awesome_loss_weight=awesome_loss_weight,
        awesome_mask_sharpness=awesome_mask_sharpness,
        awesome_warmup_steps=awesome_warmup_steps,
        vad_proxy_enabled=vad_proxy_enabled,
        gan_enabled=gan_enabled,
        gan_start_epoch=gan_start_epoch,
        gan_ramp_epochs=gan_ramp_epochs,
        gan_adv_weight=gan_adv_weight,
        gan_fm_weight=gan_fm_weight,
        gan_disc_type=gan_disc_type,
        gan_msd_scales=gan_msd_scales,
        gan_disc_lr=gan_disc_lr,
        gan_disc_weight_decay=gan_disc_weight_decay,
        gan_disc_grad_clip=gan_disc_grad_clip,
        gan_disc_update_freq=gan_disc_update_freq,
        gan_cache_gen_waveforms=gan_cache_gen_waveforms,
        gan_disc_gradient_checkpoint=gan_disc_gradient_checkpoint,
        gan_gen_gradient_checkpoint=gan_gen_gradient_checkpoint,
        gan_eval_frequency=gan_eval_frequency,
        experimental_compiled_gan=experimental_compiled_gan,
        vad_loss_weight=vad_loss_weight,
        vad_threshold=vad_threshold,
        vad_margin=vad_margin,
        vad_speech_loss_weight=vad_speech_loss_weight,
        vad_warmup_epochs=vad_warmup_epochs,
        vad_snr_gate_db=vad_snr_gate_db,
        vad_snr_gate_width=vad_snr_gate_width,
        vad_band_low_hz=vad_band_low_hz,
        vad_band_high_hz=vad_band_high_hz,
        vad_z_threshold=vad_z_threshold,
        vad_z_slope=vad_z_slope,
        vad_eval_mode=vad_eval_mode,
        vad_eval_every=vad_eval_every,
        vad_eval_batches=vad_eval_batches,
        vad_eval_max_seconds=vad_eval_max_seconds,
        vad_silero_model_path=vad_silero_model_path,
        vad_silero_sample_rate=vad_silero_sample_rate,
        vad_train_prob=vad_train_prob,
        vad_train_every_steps=vad_train_every_steps,
        eval_sisdr=eval_sisdr,
        max_train_batches=max_train_batches,
        max_valid_batches=max_valid_batches,
        seed=seed,
        learning_rate_min=learning_rate_min,
        weight_decay=weight_decay,
        model_variant=model_variant,
        debug_numerics=debug_numerics,
        debug_numerics_fail_fast=debug_numerics_fail_fast,
        debug_numerics_every=debug_numerics_every,
        nan_skip_batch=nan_skip_batch,
    )

    dataset.set_split("train")

    print(f"  Train samples: {len(dataset):,}")

    # Create validation dataset (with reproducible indices)
    dataset.set_split("valid")
    print(f"  Valid samples: {len(dataset):,}")

    # Reset to training
    dataset.set_split("train")
    dataset.set_epoch(0)

    pipeline = setup_data_pipeline(
        dataset=dataset,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
        use_mlx_data=use_mlx_data,
        resume_from=resume_from,
        resume_data_from=resume_data_from,
        debug_numerics=debug_numerics,
        debug_numerics_fail_fast=debug_numerics_fail_fast,
        debug_numerics_every=debug_numerics_every,
        debug_numerics_dump_dir=debug_numerics_dump_dir,
        debug_numerics_dump_arrays=debug_numerics_dump_arrays,
        debug_numerics_max_dumps=debug_numerics_max_dumps,
        nan_skip_batch=nan_skip_batch,
        check_chkpts=check_chkpts,
    )
    ckpt_dir = pipeline.ckpt_dir
    debugger = pipeline.debugger
    validation_report = pipeline.validation_report
    use_mlx_stream = pipeline.use_mlx_stream
    train_stream = pipeline.train_stream
    data_checkpoint_path = pipeline.data_checkpoint_path
    data_resume_progress = pipeline.data_resume_progress
    data_resume_source = pipeline.data_resume_source
    resume_from = pipeline.resume_from

    # Connect data pipeline to interrupt handler
    if use_mlx_stream and train_stream is not None:
        _interrupt_state["data_checkpoint_path"] = data_checkpoint_path
        _interrupt_state["train_stream"] = train_stream

    # Initialize model with config
    print("\nInitializing model...")
    if model_config is None:
        model_config = get_default_config()
    _sync_model_config_with_dataset(model_config, config)
    model_config.backbone.backbone_type = backbone_type  # type: ignore[assignment]
    print(f"  Backbone type: {backbone_type} | Variant: {model_variant}")
    model = init_model(config=model_config, variant=model_variant)
    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,}")

    # Counter semantics:
    # - micro_batches_per_epoch: number of dataloader micro-batches consumed per epoch
    # - optimizer_steps_per_epoch: number of optimizer updates per epoch
    #   (with accumulation this is floor(micro_batches / grad_accumulation_steps))
    approx_samples_per_epoch = len(dataset)
    micro_batches_per_epoch = approx_samples_per_epoch // batch_size
    if micro_batches_per_epoch < 1:
        raise ValueError(
            f"Dataset too small for batch_size={batch_size}: "
            f"{approx_samples_per_epoch} samples -> 0 micro-batches/epoch"
        )

    optimizer_steps_per_epoch = micro_batches_per_epoch // grad_accumulation_steps
    if optimizer_steps_per_epoch < 1:
        optimizer_steps_per_epoch = 1
        print(
            "Warning: "
            f"grad_accumulation_steps={grad_accumulation_steps} >= "
            f"micro_batches_per_epoch={micro_batches_per_epoch}; "
            "using 1 optimizer step/epoch for scheduler bookkeeping"
        )
    total_steps = epochs * optimizer_steps_per_epoch
    warmup_steps = warmup_epochs * optimizer_steps_per_epoch
    vad_warmup_steps = vad_warmup_epochs * optimizer_steps_per_epoch if use_vad_loss else 0
    awesome_warmup_steps = max(int(awesome_warmup_steps), 0) if (use_awesome_loss or use_pipeline_awesome_loss) else 0

    schedule = WarmupCosineSchedule(
        base_lr=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr,
    )

    # Optimizer - create before loading checkpoint to allow optimizer state restoration
    # Use fixed learning rate (schedule applied manually before each step)
    # This is required because schedule callbacks can't run inside mx.compile()
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    # Resume from checkpoint if provided (AFTER optimizer creation)
    resume_result = reconcile_resume(
        model=model,
        optimizer=optimizer,
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
        resume_from=resume_from,
        train_stream=train_stream,
        data_resume_progress=data_resume_progress,
        data_resume_source=data_resume_source,
        pipeline_stage_defs=pipeline_stage_defs,
        epochs=epochs,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        tqdm_setup_panel=tqdm_setup_panel,
        validation_report=validation_report,
    )
    start_epoch = resume_result.start_epoch
    best_valid_loss = resume_result.best_valid_loss
    epochs_without_improvement = resume_result.epochs_without_improvement
    last_completed_epoch = resume_result.last_completed_epoch
    resume_global_step = resume_result.resume_global_step
    resume_batch_idx = resume_result.resume_batch_idx
    resume_checkpoint_kind = resume_result.resume_checkpoint_kind
    resume_stage_index = resume_result.resume_stage_index
    resume_stage_name = resume_result.resume_stage_name
    data_resume_progress = resume_result.data_resume_progress
    if resume_result.should_return_early:
        return

    _interrupt_state["last_completed_epoch"] = last_completed_epoch

    # Bare `gan_active` needed for closure capture (loss_fn, loss_fn_gan, diagnostics).
    # Will be synced with loop_state.gan_active in the epoch loop.
    gan_active = False

    # Mutable holder for late-bound compiled disc inference (GAN-P1).
    # Populated after the compiled-GAN section; used inside loss_fn/loss_fn_gan.
    _compiled_disc_infer_holder: list = [None]

    # Loss function - define as a pure function for compilation
    # Loss formula:
    #   L_total = L_spec
    #           + w_awesome * L_awesome
    #           + w_vad * L_vad + w_speech * L_speech
    #           + w_vad_reg * L_vad_reg (sparse, proxy-gated)
    #   L_vad = mean( gate * relu(p_ref - p_out - margin) )
    #   gate = sigmoid((snr - snr_gate_db)/snr_gate_width) * clip((p_ref - vad_thr)/(1 - vad_thr))
    #   p_ref/p_out from speech-band log-energy (z-scored per utterance)
    #   L_speech = mean( gate * |log_mag_out - log_mag_ref|_speechband )
    #   L_awesome = speech-preserving contrastive log-mag + noise suppression + smoothness
    def loss_fn(
        model,
        noisy_real,
        noisy_imag,
        feat_erb,
        feat_spec,
        clean_real,
        clean_imag,
        snr,
        vad_weight,
        speech_weight,
        awesome_weight,
        vad_reg_weight,
        gan_weight,
        fm_weight,
    ):
        """Compute training loss."""
        # Model expects spec as tuple (real, imag)
        noisy_spec = (noisy_real, noisy_imag)
        target_spec = (clean_real, clean_imag)

        if gan_gen_gradient_checkpoint and gan_active:
            _gen_fn = mx.checkpoint(model)
        else:
            _gen_fn = model
        out = _gen_fn(noisy_spec, feat_erb, feat_spec, return_vad=True)

        # Unpack model output (Option C: Multi-task VAD head)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], tuple):
            spec_out, vad_logits = out
        else:
            spec_out = out
            vad_logits = None

        spec_loss = spectral_loss(spec_out, target_spec)
        total_loss = spec_loss

        out_wav = None
        clean_wav = None
        if (use_mrstft_loss or gan_active) and gan_istft is not None:
            out_wav, clean_wav = specs_to_wavs(
                spec_out,
                target_spec,
                istft_fn=gan_istft,
                n_fft=config.fft_size,
                hop_length=config.hop_size,
                target_len=gan_target_len,
                force_fp32=use_mrstft_loss,
            )

        if use_mrstft_loss and mrstft_loss_fn is not None and out_wav is not None and clean_wav is not None:
            mrstft_loss = mrstft_loss_fn(out_wav, clean_wav)
            total_loss = total_loss + mrstft_loss

        if gan_active and gan_loss_fns is not None and discriminator is not None and out_wav is not None:
            gen_loss_fn, _ = gan_loss_fns
            gan_out_wav = _gan_waveform_view(out_wav, use_fp16=bool(use_fp16))
            gan_clean_wav = _gan_waveform_view(clean_wav, use_fp16=bool(use_fp16))
            gan_out_wav, crop_start = _disc_crop_waveform(gan_out_wav, gan_disc_max_samples)
            gan_clean_wav, _ = _disc_crop_waveform(gan_clean_wav, gan_disc_max_samples, crop_start)
            if _compiled_disc_infer_holder[0] is not None and not gan_disc_gradient_checkpoint:
                disc_fake, fake_feats, disc_real, real_feats = _compiled_disc_infer_holder[0](
                    gan_out_wav, gan_clean_wav
                )
            else:
                _need_feats = feature_match_loss is not None and gan_fm_weight > 0
                if gan_disc_gradient_checkpoint:
                    _disc_fn = mx.checkpoint(discriminator)
                else:
                    _disc_fn = discriminator
                disc_fake, fake_feats = _disc_fn(gan_out_wav, return_features=_need_feats)
                disc_real, real_feats = _disc_fn(mx.stop_gradient(gan_clean_wav), return_features=_need_feats)
            disc_fake = _clip_gan_scores(disc_fake)
            gan_g_loss = gen_loss_fn(disc_fake)
            total_loss = total_loss + gan_weight * gan_g_loss
            if feature_match_loss is not None and gan_fm_weight > 0:
                fm_loss = feature_match_loss(real_feats, fake_feats)
                total_loss = total_loss + fm_weight * fm_loss

        if use_awesome_loss:
            awesome_loss, _, _, _, _, _, _, _, _, _, _, _ = _compute_awesome_losses(
                noisy_real,
                noisy_imag,
                clean_real,
                clean_imag,
                spec_out[0],
                spec_out[1],
                snr,
                vad_band_mask,
                vad_band_bins,
                awesome_mask_sharpness,
                vad_z_threshold,
                vad_z_slope,
                vad_snr_gate_db,
                vad_snr_gate_width,
                vad_proxy_enabled,
            )
            total_loss = total_loss + awesome_weight * awesome_loss

        if use_pipeline_awesome_loss:
            pipeline_loss, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _compute_pipeline_awesome_losses(
                noisy_real,
                noisy_imag,
                clean_real,
                clean_imag,
                spec_out[0],
                spec_out[1],
                snr,
                vad_band_mask,
                vad_band_bins,
                awesome_mask_sharpness,
                vad_z_threshold,
                vad_z_slope,
                vad_snr_gate_db,
                vad_snr_gate_width,
                vad_proxy_enabled,
            )
            total_loss = total_loss + awesome_weight * pipeline_loss

        if use_vad_loss:
            vad_loss, p_ref, _, gate = _compute_vad_loss(
                clean_real,
                clean_imag,
                spec_out[0],
                spec_out[1],
                snr,
                vad_band_mask,
                vad_band_bins,
                vad_threshold,
                vad_margin,
                vad_snr_gate_db,
                vad_snr_gate_width,
                vad_z_threshold,
                vad_z_slope,
            )
            # IMPORTANT: avoid Python control flow on runtime weight tensors in
            # compiled paths. speech_weight may be an mx.array scalar, and
            # branching on it would force evaluation or raise inside mx.compile.
            speech_loss = _compute_speech_band_logmag_loss(
                clean_real,
                clean_imag,
                spec_out[0],
                spec_out[1],
                vad_band_mask,
                vad_band_bins,
                gate,
            )
            total_loss = total_loss + vad_weight * vad_loss + speech_weight * speech_loss

            # Option C: Multi-task VAD head BCE loss (logits path)
            if vad_logits is not None:
                p_ref_expanded = mx.expand_dims(p_ref, axis=-1)
                vad_head_loss = nn.losses.binary_cross_entropy(
                    vad_logits, p_ref_expanded, with_logits=True, reduction="mean"
                )
                total_loss = total_loss + vad_weight * vad_head_loss

        if use_vad_train_reg:
            vad_reg_loss, _, _, _, _, _, _ = _compute_vad_reg_loss(
                clean_real,
                clean_imag,
                noisy_real,
                noisy_imag,
                spec_out[0],
                spec_out[1],
                snr,
                vad_band_mask,
                vad_band_bins,
                vad_threshold,
                vad_margin,
                vad_z_threshold,
                vad_z_slope,
                vad_snr_gate_db,
                vad_snr_gate_width,
            )
            total_loss = total_loss + vad_reg_weight * vad_reg_loss

        # Return model output as auxiliary data so callers can reuse it for
        # logging/discriminator updates without triggering a second forward.
        # out_wav/clean_wav are the raw ISTFT outputs (possibly fp32); callers
        # apply _gan_waveform_view / stop_gradient / crop independently.
        return total_loss, spec_out, out_wav, clean_wav

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # -- Experimental compiled-GAN support ----------------------------------
    # When experimental_compiled_gan is True, create a separate loss function
    # with GAN paths always active (hardcoded True instead of the `gan_active`
    # closure variable). mx.compile traces Python booleans at trace time, so
    # `if gan_active` in the original loss_fn would be captured as False during
    # pre-GAN tracing and never re-traced when it flips to True. A separate
    # function ensures the compiled graph always includes generator adversarial
    # loss paths.
    loss_and_grad_gan = None

    if experimental_compiled_gan and gan_enabled:
        print("  [EXPERIMENTAL] Compiled-GAN experiment enabled (gen-only, Variant B)")

        def loss_fn_gan(
            model,
            noisy_real,
            noisy_imag,
            feat_erb,
            feat_spec,
            clean_real,
            clean_imag,
            snr,
            vad_weight,
            speech_weight,
            awesome_weight,
            vad_reg_weight,
            gan_weight,
            fm_weight,
        ):
            """Loss function with GAN generator paths always active (compiled-GAN experiment)."""
            noisy_spec = (noisy_real, noisy_imag)
            target_spec = (clean_real, clean_imag)

            if gan_gen_gradient_checkpoint:
                _gen_fn = mx.checkpoint(model)
            else:
                _gen_fn = model
            out = _gen_fn(noisy_spec, feat_erb, feat_spec, return_vad=True)

            # Unpack model output (Option C: Multi-task VAD head)
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], tuple):
                spec_out, vad_logits = out
            else:
                spec_out = out
                vad_logits = None

            spec_loss = spectral_loss(spec_out, target_spec)
            total_loss = spec_loss

            out_wav = None
            clean_wav = None
            # GAN always active: always compute waveforms
            if gan_istft is not None:
                out_wav, clean_wav = specs_to_wavs(
                    spec_out,
                    target_spec,
                    istft_fn=gan_istft,
                    n_fft=config.fft_size,
                    hop_length=config.hop_size,
                    target_len=gan_target_len,
                    force_fp32=use_mrstft_loss,
                )

            if use_mrstft_loss and mrstft_loss_fn is not None and out_wav is not None and clean_wav is not None:
                mrstft_loss = mrstft_loss_fn(out_wav, clean_wav)
                total_loss = total_loss + mrstft_loss

            # GAN generator loss — always active (hardcoded)
            if gan_loss_fns is not None and discriminator is not None and out_wav is not None:
                gen_loss_fn, _ = gan_loss_fns
                gan_out_wav = _gan_waveform_view(out_wav, use_fp16=bool(use_fp16))
                gan_clean_wav = _gan_waveform_view(clean_wav, use_fp16=bool(use_fp16))
                gan_out_wav, crop_start = _disc_crop_waveform(gan_out_wav, gan_disc_max_samples)
                gan_clean_wav, _ = _disc_crop_waveform(gan_clean_wav, gan_disc_max_samples, crop_start)
                if _compiled_disc_infer_holder[0] is not None and not gan_disc_gradient_checkpoint:
                    disc_fake, fake_feats, disc_real, real_feats = _compiled_disc_infer_holder[0](
                        gan_out_wav, gan_clean_wav
                    )
                else:
                    _need_feats = feature_match_loss is not None and gan_fm_weight > 0
                    if gan_disc_gradient_checkpoint:
                        _disc_fn = mx.checkpoint(discriminator)
                    else:
                        _disc_fn = discriminator
                    disc_fake, fake_feats = _disc_fn(gan_out_wav, return_features=_need_feats)
                    disc_real, real_feats = _disc_fn(mx.stop_gradient(gan_clean_wav), return_features=_need_feats)
                disc_fake = _clip_gan_scores(disc_fake)
                gan_g_loss = gen_loss_fn(disc_fake)
                total_loss = total_loss + gan_weight * gan_g_loss
                if feature_match_loss is not None and gan_fm_weight > 0:
                    fm_loss = feature_match_loss(real_feats, fake_feats)
                    total_loss = total_loss + fm_weight * fm_loss

            if use_awesome_loss:
                awesome_loss, _, _, _, _, _, _, _, _, _, _, _ = _compute_awesome_losses(
                    noisy_real,
                    noisy_imag,
                    clean_real,
                    clean_imag,
                    spec_out[0],
                    spec_out[1],
                    snr,
                    vad_band_mask,
                    vad_band_bins,
                    awesome_mask_sharpness,
                    vad_z_threshold,
                    vad_z_slope,
                    vad_snr_gate_db,
                    vad_snr_gate_width,
                    vad_proxy_enabled,
                )
                total_loss = total_loss + awesome_weight * awesome_loss

            if use_pipeline_awesome_loss:
                pipeline_loss, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _compute_pipeline_awesome_losses(
                    noisy_real,
                    noisy_imag,
                    clean_real,
                    clean_imag,
                    spec_out[0],
                    spec_out[1],
                    snr,
                    vad_band_mask,
                    vad_band_bins,
                    awesome_mask_sharpness,
                    vad_z_threshold,
                    vad_z_slope,
                    vad_snr_gate_db,
                    vad_snr_gate_width,
                    vad_proxy_enabled,
                )
                total_loss = total_loss + awesome_weight * pipeline_loss

            if use_vad_loss:
                vad_loss, p_ref, _, gate = _compute_vad_loss(
                    clean_real,
                    clean_imag,
                    spec_out[0],
                    spec_out[1],
                    snr,
                    vad_band_mask,
                    vad_band_bins,
                    vad_threshold,
                    vad_margin,
                    vad_snr_gate_db,
                    vad_snr_gate_width,
                    vad_z_threshold,
                    vad_z_slope,
                )
                # IMPORTANT: avoid Python control flow on runtime weight tensors
                # in compiled paths (see loss_fn comment above).
                speech_loss = _compute_speech_band_logmag_loss(
                    clean_real,
                    clean_imag,
                    spec_out[0],
                    spec_out[1],
                    vad_band_mask,
                    vad_band_bins,
                    gate,
                )
                total_loss = total_loss + vad_weight * vad_loss + speech_weight * speech_loss

                # Option C: Multi-task VAD head BCE loss (logits path)
                if vad_logits is not None:
                    p_ref_expanded = mx.expand_dims(p_ref, axis=-1)
                    vad_head_loss = nn.losses.binary_cross_entropy(
                        vad_logits, p_ref_expanded, with_logits=True, reduction="mean"
                    )
                    total_loss = total_loss + vad_weight * vad_head_loss

            if use_vad_train_reg:
                vad_reg_loss, _, _, _, _, _, _ = _compute_vad_reg_loss(
                    clean_real,
                    clean_imag,
                    noisy_real,
                    noisy_imag,
                    spec_out[0],
                    spec_out[1],
                    snr,
                    vad_band_mask,
                    vad_band_bins,
                    vad_threshold,
                    vad_margin,
                    vad_z_threshold,
                    vad_z_slope,
                    vad_snr_gate_db,
                    vad_snr_gate_width,
                )
                total_loss = total_loss + vad_reg_weight * vad_reg_loss

            return total_loss, spec_out, out_wav, clean_wav

        loss_and_grad_gan = nn.value_and_grad(model, loss_fn_gan)

    # Diagnostic context — groups all immutable state needed by diagnose_nonfinite.
    _diag_ctx = DiagnosticContext(
        model=model,
        debugger=debugger,
        spectral_loss_fn=spectral_loss,
        use_mrstft_loss=use_mrstft_loss,
        mrstft_loss_fn=mrstft_loss_fn,
        mrstft_istft=mrstft_istft,
        fft_size=config.fft_size,
        hop_size=config.hop_size,
        mrstft_target_len=mrstft_target_len,
        gan_loss_fns=gan_loss_fns,
        discriminator=discriminator,
        gan_istft=gan_istft,
        gan_target_len=gan_target_len,
        feature_match_loss=feature_match_loss,
        gan_fm_weight=gan_fm_weight,
        clip_gan_scores_fn=_clip_gan_scores,
        use_awesome_loss=use_awesome_loss,
        use_pipeline_awesome_loss=use_pipeline_awesome_loss,
        vad_band_mask=vad_band_mask,
        vad_band_bins=vad_band_bins,
        awesome_mask_sharpness=awesome_mask_sharpness,
        vad_z_threshold=vad_z_threshold,
        vad_z_slope=vad_z_slope,
        vad_snr_gate_db=vad_snr_gate_db,
        vad_snr_gate_width=vad_snr_gate_width,
        vad_proxy_enabled=vad_proxy_enabled,
        use_vad_loss=use_vad_loss,
        vad_threshold=vad_threshold,
        vad_margin=vad_margin,
        vad_speech_loss_weight=vad_speech_loss_weight,
        use_vad_train_reg=use_vad_train_reg,
    )

    def _diagnose_nonfinite(
        noisy_real: mx.array,
        noisy_imag: mx.array,
        feat_erb: mx.array,
        feat_spec: mx.array,
        clean_real: mx.array,
        clean_imag: mx.array,
        snr: mx.array,
        debug_ctx: dict[str, Any],
    ) -> None:
        """Thin wrapper — delegates to training_diagnostics.diagnose_nonfinite."""
        _diagnose_nonfinite_impl(
            _diag_ctx,
            noisy_real,
            noisy_imag,
            feat_erb,
            feat_spec,
            clean_real,
            clean_imag,
            snr,
            debug_ctx,
            gan_active=gan_active,
        )

    # -- Compile-boundary shape guardrails ----------------------------------
    def _assert_compile_boundary_shapes(
        noisy: mx.array,
        clean: mx.array,
        expected_batch_size: int,
        *,
        check_dtype: bool = True,
        expected_dtype: mx.Dtype = mx.float32,
    ) -> None:
        """Validate shape invariants at compile boundary to prevent retracing.

        Must be called *before* entering a compiled function so that any
        violation surfaces as a clear Python error rather than an opaque
        retrace or silent correctness issue.
        """
        if noisy.shape[0] != expected_batch_size:
            raise ValueError(
                f"Compile boundary shape violation: batch_size={noisy.shape[0]}, "
                f"expected={expected_batch_size}. This would trigger an expensive retrace."
            )
        if noisy.shape != clean.shape:
            raise ValueError(f"Compile boundary shape mismatch: noisy={noisy.shape}, clean={clean.shape}")
        if check_dtype and noisy.dtype != expected_dtype:
            raise ValueError(f"Compile boundary dtype mismatch: got {noisy.dtype}, " f"expected {expected_dtype}")

    _compile_retrace_count: int = 0

    def _log_compile_retrace_warning(context: str = "") -> None:
        """Log a warning when a compiled function retrace is detected.

        Call this when a shape/dtype change is observed that would force MLX
        to re-trace the compiled graph.
        """
        nonlocal _compile_retrace_count
        _compile_retrace_count += 1
        msg = f"[RETRACE WARNING #{_compile_retrace_count}] " f"Compiled function retrace detected. {context}"
        tqdm.write(msg)

    # Compiled training step for performance optimization
    # Captures model and optimizer state for graph tracing
    state = [model.state, optimizer.state]

    from functools import partial

    @partial(mx.compile, inputs=state, outputs=state)
    def compiled_step(
        noisy_real,
        noisy_imag,
        feat_erb,
        feat_spec,
        clean_real,
        clean_imag,
        snr,
        vad_weight,
        speech_weight,
        awesome_weight,
        vad_reg_weight,
        gan_weight,
        fm_weight,
        max_grad_norm_val,
    ):
        """JIT-compiled training step for faster training.

        This compiles the forward pass, backward pass, and optimizer update
        into a single optimized computation graph.
        """
        (loss, out, cached_out_wav, cached_clean_wav), grads = loss_and_grad(
            model,
            noisy_real,
            noisy_imag,
            feat_erb,
            feat_spec,
            clean_real,
            clean_imag,
            snr,
            vad_weight,
            speech_weight,
            awesome_weight,
            vad_reg_weight,
            gan_weight,
            fm_weight,
        )
        # Gradient clipping inline
        if max_grad_norm_val > 0:
            grads, _ = clip_grad_norm(grads, max_grad_norm_val)
        optimizer.update(model, grads)
        return loss, out, cached_out_wav, cached_clean_wav

    # Compiled forward/backward step (no optimizer update).
    # Used when gradient accumulation is enabled so updates remain aligned to
    # optimizer-step semantics while still compiling the expensive fwd+bwd path.
    @partial(mx.compile, inputs=[model.state], outputs=[model.state])
    def compiled_loss_and_grad_step(
        noisy_real,
        noisy_imag,
        feat_erb,
        feat_spec,
        clean_real,
        clean_imag,
        snr,
        vad_weight,
        speech_weight,
        awesome_weight,
        vad_reg_weight,
        gan_weight,
        fm_weight,
    ):
        (loss, out, cached_out_wav, cached_clean_wav), grads = loss_and_grad(
            model,
            noisy_real,
            noisy_imag,
            feat_erb,
            feat_spec,
            clean_real,
            clean_imag,
            snr,
            vad_weight,
            speech_weight,
            awesome_weight,
            vad_reg_weight,
            gan_weight,
            fm_weight,
        )
        return loss, out, cached_out_wav, cached_clean_wav, grads

    # -- Compiled GAN training steps (experimental) -------------------------
    # Mirror of compiled_step / compiled_loss_and_grad_step but using
    # loss_and_grad_gan so the generator adversarial path is always traced.
    compiled_gan_step = None
    compiled_gan_loss_and_grad_step = None

    if experimental_compiled_gan and loss_and_grad_gan is not None:
        gan_state = [model.state, optimizer.state]
        _lag_gan = loss_and_grad_gan  # capture non-None ref for Pyright

        @partial(mx.compile, inputs=gan_state, outputs=gan_state)
        def _compiled_gan_step(
            noisy_real,
            noisy_imag,
            feat_erb,
            feat_spec,
            clean_real,
            clean_imag,
            snr,
            vad_weight,
            speech_weight,
            awesome_weight,
            vad_reg_weight,
            gan_weight,
            fm_weight,
            max_grad_norm_val,
        ):
            """Compiled gen step with GAN paths always active (experimental)."""
            (loss, out, cached_out_wav, cached_clean_wav), grads = _lag_gan(
                model,
                noisy_real,
                noisy_imag,
                feat_erb,
                feat_spec,
                clean_real,
                clean_imag,
                snr,
                vad_weight,
                speech_weight,
                awesome_weight,
                vad_reg_weight,
                gan_weight,
                fm_weight,
            )
            if max_grad_norm_val > 0:
                grads, _ = clip_grad_norm(grads, max_grad_norm_val)
            optimizer.update(model, grads)
            return loss, out, cached_out_wav, cached_clean_wav

        @partial(mx.compile, inputs=[model.state], outputs=[model.state])
        def _compiled_gan_loss_and_grad_step(
            noisy_real,
            noisy_imag,
            feat_erb,
            feat_spec,
            clean_real,
            clean_imag,
            snr,
            vad_weight,
            speech_weight,
            awesome_weight,
            vad_reg_weight,
            gan_weight,
            fm_weight,
        ):
            """Compiled gen fwd+bwd with GAN paths always active (experimental)."""
            (loss, out, cached_out_wav, cached_clean_wav), grads = _lag_gan(
                model,
                noisy_real,
                noisy_imag,
                feat_erb,
                feat_spec,
                clean_real,
                clean_imag,
                snr,
                vad_weight,
                speech_weight,
                awesome_weight,
                vad_reg_weight,
                gan_weight,
                fm_weight,
            )
            return loss, out, cached_out_wav, cached_clean_wav, grads

        compiled_gan_step = _compiled_gan_step
        compiled_gan_loss_and_grad_step = _compiled_gan_loss_and_grad_step

    # -- Compiled discriminator update step (GAN-P2) -------------------------
    compiled_disc_update_step = None

    if (
        experimental_compiled_gan
        and discriminator is not None
        and disc_optimizer is not None
        and gan_loss_fns is not None
    ):
        disc_update_state = [discriminator.state, disc_optimizer.state]
        _, _disc_loss_fn_ref = gan_loss_fns  # capture discriminator_loss

        @partial(mx.compile, inputs=disc_update_state, outputs=disc_update_state)
        def _compiled_disc_update_step(
            clean_wav_d,
            pred_wav_d,
            max_disc_grad_norm,
        ):
            """Compiled discriminator update: fwd+bwd+optimizer in one graph."""

            def _disc_loss_inner(disc):
                real_out, _ = disc(clean_wav_d, return_features=False)
                fake_out, _ = disc(pred_wav_d, return_features=False)
                real_out = _clip_gan_scores(real_out)
                fake_out = _clip_gan_scores(fake_out)
                total_loss, _, _ = _disc_loss_fn_ref(real_out, fake_out)
                return total_loss

            d_loss, d_grads = nn.value_and_grad(discriminator, _disc_loss_inner)(discriminator)
            if max_disc_grad_norm > 0:
                d_grads, _ = clip_grad_norm(d_grads, max_disc_grad_norm)
            disc_optimizer.update(discriminator, d_grads)
            return d_loss

        compiled_disc_update_step = _compiled_disc_update_step

    # -- Compiled discriminator inference for gen loss path (GAN-P1) ----------
    compiled_disc_infer = None

    if experimental_compiled_gan and discriminator is not None:
        disc_infer_state = [discriminator.state]

        @partial(mx.compile, inputs=disc_infer_state, outputs=disc_infer_state)
        def _compiled_disc_infer(
            fake_wav,
            real_wav,
        ):
            """Compiled disc forward for gen loss path."""
            disc_fake, fake_feats = discriminator(fake_wav)
            disc_real, real_feats = discriminator(mx.stop_gradient(real_wav))
            return disc_fake, fake_feats, disc_real, real_feats

        compiled_disc_infer = _compiled_disc_infer
        _compiled_disc_infer_holder[0] = compiled_disc_infer

    # Base compiled-step eligibility (epoch-level mode selection may still choose eager).
    # Gradient accumulation is supported via compiled fwd+bwd with eager optimizer updates.
    base_compiled_step_enabled = print_compiled_step_eligibility(
        debug_numerics=debug_numerics,
        nan_skip_batch=nan_skip_batch,
        gan_enabled=gan_enabled,
        gan_start_epoch=gan_start_epoch,
        experimental_compiled_gan=experimental_compiled_gan,
        grad_accumulation_steps=grad_accumulation_steps,
        batch_size=batch_size,
    )

    scheduled_start_stage = _resolve_pipeline_stage(start_epoch, pipeline_stage_defs)
    scheduled_start_stage_index = int(scheduled_start_stage["index"])
    if resume_stage_index is None:
        initial_stage_index = scheduled_start_stage_index
    else:
        if resume_stage_index < scheduled_start_stage_index:
            print(
                "ℹ️  Clamping resume stage to scheduled stage floor: "
                f"resume_stage={resume_stage_index} → scheduled_stage={scheduled_start_stage_index}."
            )
        initial_stage_index = max(resume_stage_index, scheduled_start_stage_index)
    initial_stage = _resolve_pipeline_stage_by_index(initial_stage_index, pipeline_stage_defs)
    initial_stage_name = str(initial_stage["name"])
    if resume_stage_name and resume_stage_name != initial_stage_name:
        print(
            "ℹ️  Resume stage name normalized from checkpoint metadata: " f"{resume_stage_name} → {initial_stage_name}."
        )

    # Register SIGINT handler for graceful shutdown
    _register_sigint_handler(
        model,
        optimizer,
        ckpt_dir,
        train_config,
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
        last_completed_epoch=last_completed_epoch,
        pipeline_stage_index=initial_stage_index,
        pipeline_stage_name=initial_stage_name,
    )
    print("  SIGINT handler registered (CTRL+C will save checkpoint before exit)")

    # Training loop
    # Sync cadence derived from sync_mode (see docs/SYNC_BARRIER_POLICY.md)
    mode = SyncMode(sync_mode)
    emit_detailed_metrics = mode.emit_detailed_metrics
    print(f"\nStarting training (epoch {start_epoch + 1} to {epochs})...")
    print(f"  Sync mode: {sync_mode} (eval_frequency={eval_frequency})")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Est. total steps: {total_steps:,}")
    print()

    loop_state = TrainingLoopState(
        global_step=resume_global_step if resume_from else start_epoch * optimizer_steps_per_epoch,
        final_epoch=start_epoch,
        last_completed_epoch=max(last_completed_epoch, start_epoch - 1),
        best_valid_loss=best_valid_loss,
        epochs_without_improvement=epochs_without_improvement,
        active_stage_name=initial_stage_name,
        active_stage_index=initial_stage_index,
        epoch_awesome_loss_weight=base_awesome_loss_weight,
        epoch_vad_loss_weight=base_vad_loss_weight,
        epoch_vad_speech_loss_weight=base_vad_speech_loss_weight,
    )
    # Keep bare `gan_active` for closure capture (loss_fn reads it)
    gan_active = loop_state.gan_active

    def _sync_data_stream_stage(stage_index: int, stage_name: str) -> None:
        if train_stream is None:
            return
        train_stream._checkpoint.pipeline_stage_index = int(stage_index)
        train_stream._checkpoint.pipeline_stage_name = str(stage_name)

    max_train_batches = train_config.get("max_train_batches")
    max_valid_batches = train_config.get("max_valid_batches")

    # Cache config-constant mx.array values outside the training loop
    _gan_disc_grad_clip_mx = mx.array(float(gan_disc_grad_clip), dtype=mx.float32)

    # Validation context — groups all immutable state needed by run_validation.
    _valid_ctx = ValidationContext(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        fft_size=config.fft_size,
        hop_size=config.hop_size,
        sample_rate=config.sample_rate,
        spectral_loss_fn=spectral_loss,
        use_awesome_loss=use_awesome_loss,
        use_pipeline_awesome_loss=use_pipeline_awesome_loss,
        use_vad_loss=use_vad_loss,
        use_vad_train_reg=use_vad_train_reg,
        use_mrstft_loss=use_mrstft_loss,
        mrstft_loss_fn=mrstft_loss_fn,
        mrstft_istft=mrstft_istft,
        mrstft_target_len=mrstft_target_len,
        awesome_mask_sharpness=awesome_mask_sharpness,
        awesome_warmup_steps=awesome_warmup_steps,
        vad_band_mask=vad_band_mask,
        vad_band_bins=vad_band_bins,
        vad_z_threshold=vad_z_threshold,
        vad_z_slope=vad_z_slope,
        vad_snr_gate_db=vad_snr_gate_db,
        vad_snr_gate_width=vad_snr_gate_width,
        vad_proxy_enabled=vad_proxy_enabled,
        vad_threshold=vad_threshold,
        vad_margin=vad_margin,
        vad_eval_mode=vad_eval_mode,
        vad_eval_batches=vad_eval_batches,
        silero_vad=silero_vad,
        debugger=debugger,
        eval_sisdr=eval_sisdr,
        emit_detailed_metrics=emit_detailed_metrics,
        max_valid_batches=max_valid_batches,
        use_mlx_stream=use_mlx_stream,
        prefetch_size=prefetch_size,
        num_workers=num_workers,
        ckpt_dir=ckpt_dir,
        dynamic_loss=dynamic_loss,
        tqdm_valid_position=tqdm_valid_position,
        tqdm_panels=_tqdm_panels,
        tqdm_kwargs=_TQDM_KWARGS,
    )

    start_display = f"{start_epoch + 1}/{epochs} (idx {start_epoch})"
    lc_display = (
        f"{loop_state.last_completed_epoch + 1} (idx {loop_state.last_completed_epoch})"
        if loop_state.last_completed_epoch >= 0
        else "none"
    )
    print(f"Starting training at epoch {start_display} | last_completed_epoch={lc_display}")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        loop_state.final_epoch = epoch

        scheduled_stage = _resolve_pipeline_stage(epoch, pipeline_stage_defs)
        scheduled_stage_index = int(scheduled_stage["index"])
        next_stage_index = max(loop_state.active_stage_index, scheduled_stage_index)

        if next_stage_index != loop_state.active_stage_index:
            print(
                "\n🔄 Pipeline stage advanced "
                f"({loop_state.active_stage_index} -> {next_stage_index}) by schedule. Resetting loop_state.best_valid_loss."
            )
            loop_state.best_valid_loss = float("inf")
            loop_state.epochs_without_improvement = 0

        loop_state.active_stage_index = next_stage_index
        active_stage = _resolve_pipeline_stage_by_index(loop_state.active_stage_index, pipeline_stage_defs)
        loop_state.active_stage_name = str(active_stage["name"])
        loop_state.epoch_awesome_loss_weight = float(
            active_stage["awesome_loss_weight"]
            if active_stage["awesome_loss_weight"] is not None
            else base_awesome_loss_weight
        )
        loop_state.epoch_vad_loss_weight = float(
            active_stage["vad_loss_weight"] if active_stage["vad_loss_weight"] is not None else base_vad_loss_weight
        )
        loop_state.epoch_vad_speech_loss_weight = float(
            active_stage["vad_speech_loss_weight"]
            if active_stage["vad_speech_loss_weight"] is not None
            else base_vad_speech_loss_weight
        )
        train_config["pipeline_stage_active"] = {
            "index": loop_state.active_stage_index,
            "name": loop_state.active_stage_name,
            "start_epoch": int(active_stage["start_epoch"]),
            "awesome_loss_weight": loop_state.epoch_awesome_loss_weight,
            "vad_loss_weight": loop_state.epoch_vad_loss_weight,
            "vad_speech_loss_weight": loop_state.epoch_vad_speech_loss_weight,
        }
        print(
            "  Stage "
            f"{loop_state.active_stage_index} ({loop_state.active_stage_name}) | "
            f"awesome_w={loop_state.epoch_awesome_loss_weight:.4f} "
            f"vad_w={loop_state.epoch_vad_loss_weight:.4f} speech_w={loop_state.epoch_vad_speech_loss_weight:.4f}"
        )
        _sync_data_stream_stage(loop_state.active_stage_index, loop_state.active_stage_name)

        # Set epoch for reproducible shuffling
        dataset.set_split("train")
        dataset.set_epoch(epoch)

        # ====== Curriculum Learning Schedule ======
        if curriculum_warmup_epochs > 0:
            target_p_extreme = p_extreme_snr if p_extreme_snr is not None else config.p_extreme_snr
            target_p_very_low = p_very_low_snr if p_very_low_snr is not None else config.p_very_low_snr
            target_p_interfer = p_interfer_speech if p_interfer_speech is not None else config.p_interfer_speech
            cur_p_extreme, cur_p_very_low, cur_p_interfer = curriculum_schedule(
                epoch=epoch,
                total_epochs=epochs,
                warmup_epochs=curriculum_warmup_epochs,
                target_p_extreme=target_p_extreme,
                target_p_very_low=target_p_very_low,
                target_p_interfer=target_p_interfer,
            )
            # Update dataset config with scheduled probabilities
            dataset.config.p_extreme_snr = cur_p_extreme
            dataset.config.p_very_low_snr = cur_p_very_low
            dataset.config.p_interfer_speech = cur_p_interfer
            if epoch < curriculum_warmup_epochs or (epoch == curriculum_warmup_epochs and verbose):
                print(
                    f"  Curriculum (epoch {epoch + 1}/{curriculum_warmup_epochs}): "
                    f"p_extreme={cur_p_extreme:.3f}, p_very_low={cur_p_very_low:.3f}, p_interfer={cur_p_interfer:.3f}"
                )

        gan_scale = 0.0
        if gan_enabled and epoch >= gan_start_epoch:
            if gan_ramp_epochs > 0:
                gan_scale = min(1.0, (epoch - gan_start_epoch + 1) / gan_ramp_epochs)
            else:
                gan_scale = 1.0
        gan_weight = gan_adv_weight * gan_scale
        fm_weight = gan_fm_weight * gan_scale
        gan_weight_mx = mx.array(gan_weight, dtype=mx.float32)
        fm_weight_mx = mx.array(fm_weight, dtype=mx.float32)
        gan_active = gan_enabled and gan_scale > 0.0
        loop_state.gan_active = gan_active

        # GAN epochs use a tighter eval_frequency to bound lazy-graph
        # accumulation.  With per-step compiled disc and single-eval enabled
        # the graph is bounded, so the user-configured gan_eval_frequency
        # (default 2) is safe.  0 means "no override".
        epoch_eval_frequency = eval_frequency
        if gan_active and gan_eval_frequency > 0:
            epoch_eval_frequency = min(eval_frequency, gan_eval_frequency)
            if epoch == gan_start_epoch:
                print(
                    f"  GAN active: eval_frequency capped to {epoch_eval_frequency} "
                    f"(gan.eval_frequency={gan_eval_frequency}, training.eval_frequency={eval_frequency})"
                )

        prev_train_mode = loop_state.train_mode
        loop_state.train_mode, epoch_use_compiled_step = resolve_epoch_train_mode(
            compiled_step_base_enabled=base_compiled_step_enabled,
            gan_enabled=gan_enabled,
            gan_active=gan_active,
            previous_mode=loop_state.train_mode,
            experimental_compiled_gan=experimental_compiled_gan,
        )
        if not experimental_compiled_gan:
            if prev_train_mode == _TRAIN_MODE_EAGER and loop_state.train_mode != _TRAIN_MODE_EAGER:
                raise RuntimeError(
                    "Invariant violation: training mode switched from EAGER back to COMPILED. "
                    "Mode switches must be one-way to preserve deterministic behavior after GAN activation."
                )
        if gan_active and epoch_use_compiled_step and not experimental_compiled_gan:
            raise RuntimeError(
                "Invariant violation: GAN active epoch cannot run compiled step. "
                f"epoch={epoch}, gan_start_epoch={gan_start_epoch}"
            )

        # Determine whether we're using the GAN-specific compiled step for this epoch
        use_compiled_gan_step = (
            experimental_compiled_gan and gan_active and epoch_use_compiled_step and compiled_gan_step is not None
        )

        if loop_state.train_mode != prev_train_mode:
            if not base_compiled_step_enabled:
                mode_reason = "compiled_blocked"
            elif gan_enabled and gan_active and not experimental_compiled_gan:
                mode_reason = "gan_active"
            elif gan_enabled and gan_active and experimental_compiled_gan:
                mode_reason = "experimental_compiled_gan"
            else:
                mode_reason = "gan_inactive"
            print(f"  TRAIN_MODE={loop_state.train_mode} (epoch {epoch + 1}/{epochs}, reason={mode_reason})")
            if use_compiled_gan_step:
                print(f"  [EXPERIMENTAL] Using compiled-GAN step (gen compiled, disc eager) " f"epoch={epoch + 1}")

        if gan_enabled and verbose:
            print(
                f"  GAN schedule (epoch {epoch + 1}/{epochs}): "
                f"scale={gan_scale:.3f}, adv={gan_weight:.4f}, fm={fm_weight:.4f}"
            )

        # ====== Training ======
        model.train()
        train_loss = 0.0
        train_gan_d_loss = 0.0
        train_gan_d_updates = 0
        _epoch_accums = create_epoch_accums()
        partial_batch_fallbacks = 0
        partial_batch_warning_emitted = False
        num_train_batches = 0
        samples_processed = 0
        grad_norm = 0.0
        loss_val = 0.0  # Initialize for async eval

        # Update interrupt state at start of epoch
        _update_interrupt_state(
            epoch,
            0.0,
            loop_state.best_valid_loss,
            batch_idx=0,
            global_step=loop_state.global_step,
            last_completed_epoch=loop_state.last_completed_epoch,
            pipeline_stage_index=loop_state.active_stage_index,
            pipeline_stage_name=loop_state.active_stage_name,
        )

        # Timing accumulators for verbose diagnostics
        total_data_time = 0.0
        total_forward_time = 0.0  # Used for compiled step timing

        # Gradient accumulation tracking (only used when grad_accumulation_steps > 1)
        accumulated_grads: dict | None = None
        accumulated_loss = SCALAR_ZERO
        micro_batches_in_accum = 0

        # Cached mx.array weight scalars — avoid per-batch mx.array() allocation
        # when the Python float hasn't changed.
        _prev_vad_w: float | None = None
        _prev_vad_w_mx = SCALAR_ZERO
        _prev_speech_w: float | None = None
        _prev_speech_w_mx = SCALAR_ZERO
        _prev_awesome_w: float | None = None
        _prev_awesome_w_mx = SCALAR_ZERO
        _prev_vad_reg_w: float | None = None
        _prev_vad_reg_w_mx = SCALAR_ZERO

        # Create data iterator (MLXDataStream or PrefetchDataLoader)
        resume_batches_for_epoch = 0
        if resume_from and resume_checkpoint_kind in _IN_PROGRESS_KINDS and epoch == start_epoch:
            resume_batches_for_epoch = resume_batch_idx

        epoch_target_micro_batches = micro_batches_per_epoch
        if max_train_batches is not None:
            epoch_target_micro_batches = min(epoch_target_micro_batches, max_train_batches)
        if resume_batches_for_epoch > epoch_target_micro_batches:
            raise RuntimeError(
                "Resume micro-batch position exceeds epoch boundary. "
                f"resume_micro_batch={resume_batches_for_epoch}, "
                f"epoch_target_micro_batches={epoch_target_micro_batches}."
            )
        train_total = max(epoch_target_micro_batches - resume_batches_for_epoch, 0)

        if use_mlx_stream and train_stream is not None:
            if data_resume_progress is not None and epoch == data_resume_progress.get("epoch"):
                # Continue from saved data checkpoint without resetting epoch state.
                data_iterator = train_stream
                progress = train_stream.get_progress()
                if progress["batch"] != resume_batches_for_epoch:
                    raise RuntimeError(
                        "Data stream resume position does not match model resume position: "
                        f"data={progress['batch']}, model={resume_batches_for_epoch}."
                    )
                if resume_batches_for_epoch > 0:
                    print(f"  Resuming epoch {epoch + 1} from micro-batch {progress['batch']}")
                data_resume_progress = None
            elif resume_batches_for_epoch > 0:
                train_stream.set_resume_position(epoch=epoch, batch_idx=resume_batches_for_epoch, split="train")
                data_iterator = train_stream
                print(f"  Resuming epoch {epoch + 1} from micro-batch {resume_batches_for_epoch}")
            else:
                train_stream.set_epoch(epoch)
                data_iterator = train_stream
        else:
            data_iterator = PrefetchDataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=config.num_workers,
                prefetch_factor=2,
            )
            data_iterator, did_skip = maybe_skip_resume_batches(
                data_iterator,
                resume_from=resume_from,
                epoch=epoch,
                start_epoch=start_epoch,
                resume_batch_idx=resume_batches_for_epoch,
            )
            if did_skip:
                print(f"  Resuming epoch {epoch + 1} from micro-batch {resume_batches_for_epoch}")

        train_tqdm_kwargs = dict(_TQDM_KWARGS)
        if _tqdm_panels:
            train_tqdm_kwargs["position"] = tqdm_train_position

        train_pbar = tqdm(
            enumerate(islice(data_iterator, train_total)),
            total=train_total,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=True,
            **train_tqdm_kwargs,
        )

        # Throughput tracking: accumulate samples and wall-clock time over sync windows
        window_samples = 0
        window_start = time.perf_counter()

        data_start = time.perf_counter()
        for batch_idx, batch in train_pbar:
            data_time = time.perf_counter() - data_start
            total_data_time += data_time

            # Unpack batch
            noisy_real = batch["noisy_real"]
            noisy_imag = batch["noisy_imag"]
            clean_real = batch["clean_real"]
            clean_imag = batch["clean_imag"]
            feat_erb = batch["feat_erb"]
            feat_spec = batch["feat_spec"]
            snr = batch["snr"]

            debug_ctx = {
                "phase": "train",
                "epoch": epoch,
                "batch": batch_idx,
                "global_step": loop_state.global_step,
            }
            if debugger is not None:
                debugger.check("batch.noisy_real", noisy_real, debug_ctx)
                debugger.check("batch.noisy_imag", noisy_imag, debug_ctx)
                debugger.check("batch.clean_real", clean_real, debug_ctx)
                debugger.check("batch.clean_imag", clean_imag, debug_ctx)
                debugger.check("batch.feat_erb", feat_erb, debug_ctx)
                debugger.check("batch.feat_spec", feat_spec, debug_ctx)
                debugger.check("batch.snr", snr, debug_ctx)

            # Convert to BF16 if enabled (mixed precision training)
            # BF16 has the same exponent range as FP32 (8 bits), eliminating
            # the gradient overflow/underflow NaN issues seen with FP16.
            if use_fp16:
                noisy_real = noisy_real.astype(mx.bfloat16)
                noisy_imag = noisy_imag.astype(mx.bfloat16)
                clean_real = clean_real.astype(mx.bfloat16)
                clean_imag = clean_imag.astype(mx.bfloat16)
                feat_erb = feat_erb.astype(mx.bfloat16)
                feat_spec = feat_spec.astype(mx.bfloat16)

            current_batch_size = noisy_real.shape[0]

            # Update learning rate from schedule (must be done outside compiled step)
            current_lr = schedule(loop_state.global_step)
            optimizer.learning_rate = current_lr

            warmup_frac = 1.0
            if use_vad_loss and vad_warmup_steps > 0:
                warmup_frac = min(1.0, loop_state.global_step / max(vad_warmup_steps, 1))

            vad_weight = loop_state.epoch_vad_loss_weight * warmup_frac
            speech_weight = loop_state.epoch_vad_speech_loss_weight * warmup_frac
            if vad_weight != _prev_vad_w:
                _prev_vad_w = vad_weight
                _prev_vad_w_mx = mx.array(vad_weight, dtype=mx.float32)
            vad_weight_mx = _prev_vad_w_mx
            if speech_weight != _prev_speech_w:
                _prev_speech_w = speech_weight
                _prev_speech_w_mx = mx.array(speech_weight, dtype=mx.float32)
            speech_weight_mx = _prev_speech_w_mx
            awesome_frac = 1.0
            if (use_awesome_loss or use_pipeline_awesome_loss) and awesome_warmup_steps > 0:
                awesome_frac = min(1.0, loop_state.global_step / max(awesome_warmup_steps, 1))
            awesome_weight = loop_state.epoch_awesome_loss_weight * awesome_frac
            if awesome_weight != _prev_awesome_w:
                _prev_awesome_w = awesome_weight
                _prev_awesome_w_mx = mx.array(awesome_weight, dtype=mx.float32)
            awesome_weight_mx = _prev_awesome_w_mx

            apply_vad_reg = False
            if use_vad_train_reg:
                if vad_train_every_steps > 0 and loop_state.global_step % vad_train_every_steps == 0:
                    apply_vad_reg = True
                elif vad_train_prob > 0:
                    apply_vad_reg = random.random() < vad_train_prob
            vad_reg_weight = vad_weight if apply_vad_reg else 0.0
            if vad_reg_weight != _prev_vad_reg_w:
                _prev_vad_reg_w = vad_reg_weight
                _prev_vad_reg_w_mx = mx.array(vad_reg_weight, dtype=mx.float32)
            vad_reg_weight_mx = _prev_vad_reg_w_mx

            # Track whether optimizer was updated this iteration (for gradient accumulation)
            did_optimizer_update = False

            # Forward, backward, and update (either compiled or standard)
            fwd_start = time.perf_counter()

            model_out = None
            cached_out_wav = None
            cached_clean_wav = None
            use_compiled_step_for_batch = epoch_use_compiled_step and current_batch_size == batch_size
            if epoch_use_compiled_step:
                if not use_compiled_step_for_batch:
                    partial_batch_fallbacks += 1
                    if not partial_batch_warning_emitted:
                        _log_compile_retrace_warning(
                            context=(
                                "Detected non-canonical batch shape at compile boundary "
                                f"(got {current_batch_size}, expected {batch_size}); "
                                "falling back to eager for this batch to avoid retrace."
                            )
                        )
                        partial_batch_warning_emitted = True

            if use_compiled_step_for_batch:
                _assert_compile_boundary_shapes(
                    noisy_real,
                    clean_real,
                    batch_size,
                    check_dtype=use_fp16,
                    expected_dtype=mx.bfloat16 if use_fp16 else mx.float32,
                )
                should_sync = (batch_idx + 1) % epoch_eval_frequency == 0

                # Select the appropriate compiled functions. When the
                # experimental compiled-GAN flag is active AND GAN is active,
                # use the GAN-specific compiled functions whose computation
                # graph always includes generator adversarial loss paths.
                active_compiled_step = compiled_gan_step if use_compiled_gan_step else compiled_step
                active_compiled_lag = (
                    compiled_gan_loss_and_grad_step if use_compiled_gan_step else compiled_loss_and_grad_step
                )

                if grad_accumulation_steps > 1:
                    # Compiled fwd+bwd with eager accumulated optimizer updates.
                    loss, model_out, cached_out_wav, cached_clean_wav, grads = active_compiled_lag(
                        noisy_real,
                        noisy_imag,
                        feat_erb,
                        feat_spec,
                        clean_real,
                        clean_imag,
                        snr,
                        vad_weight_mx,
                        speech_weight_mx,
                        awesome_weight_mx,
                        vad_reg_weight_mx,
                        gan_weight_mx,
                        fm_weight_mx,
                    )
                    accumulated_grads = accumulate_grads(accumulated_grads, grads)
                    accumulated_loss = accumulated_loss + loss
                    micro_batches_in_accum += 1

                    is_accum_complete = micro_batches_in_accum >= grad_accumulation_steps
                    if is_accum_complete:
                        did_optimizer_update = True
                        final_grads = scale_grads(accumulated_grads, 1.0 / grad_accumulation_steps)
                        if max_grad_norm > 0:
                            final_grads, grad_norm_arr = clip_grad_norm(final_grads, max_grad_norm)
                            if should_sync:
                                grad_norm = float(grad_norm_arr)
                        if _tree_all_finite(final_grads):
                            optimizer.update(model, final_grads)
                        else:
                            did_optimizer_update = False
                            tqdm.write(
                                "⚠️  Non-finite grads after clipping; skipping optimizer update "
                                f"(step={loop_state.global_step})"
                            )
                        accumulated_grads = None
                        accumulated_loss = SCALAR_ZERO
                        micro_batches_in_accum = 0

                    if should_sync:
                        if did_optimizer_update:
                            mx.eval(loss, model.parameters(), optimizer.state)
                        else:
                            mx.eval(loss)
                else:
                    # Fully compiled training step (fwd+bwd+update) for best throughput.
                    did_optimizer_update = True
                    loss, model_out, cached_out_wav, cached_clean_wav = active_compiled_step(
                        noisy_real,
                        noisy_imag,
                        feat_erb,
                        feat_spec,
                        clean_real,
                        clean_imag,
                        snr,
                        vad_weight_mx,
                        speech_weight_mx,
                        awesome_weight_mx,
                        vad_reg_weight_mx,
                        gan_weight_mx,
                        fm_weight_mx,
                        max_grad_norm,
                    )

                    # One-time correctness verification for compiled-GAN step
                    if (
                        use_compiled_gan_step
                        and not loop_state.compiled_gan_correctness_verified
                        and loss_and_grad_gan is not None
                    ):
                        loop_state.compiled_gan_correctness_verified = True
                        # Run an eager forward pass for comparison
                        (eager_loss, _, _, _), _ = loss_and_grad_gan(
                            model,
                            noisy_real,
                            noisy_imag,
                            feat_erb,
                            feat_spec,
                            clean_real,
                            clean_imag,
                            snr,
                            vad_weight_mx,
                            speech_weight_mx,
                            awesome_weight_mx,
                            vad_reg_weight_mx,
                            gan_weight_mx,
                            fm_weight_mx,
                        )
                        mx.eval(loss, eager_loss)
                        compiled_val = float(loss)
                        eager_val = float(eager_loss)
                        if abs(compiled_val - eager_val) > 1e-5 + 1e-4 * abs(eager_val):
                            tqdm.write(
                                f"  [EXPERIMENTAL] WARNING: compiled-GAN correctness check FAILED. "
                                f"compiled_loss={compiled_val:.6f}, eager_loss={eager_val:.6f}, "
                                f"diff={abs(compiled_val - eager_val):.2e}"
                            )
                        else:
                            tqdm.write(
                                f"  [EXPERIMENTAL] Compiled-GAN correctness check PASSED. "
                                f"compiled_loss={compiled_val:.6f}, eager_loss={eager_val:.6f}"
                            )

                    # OPTIMIZATION: Only sync periodically to reduce GPU stalls
                    # This allows MLX to batch operations for better throughput
                    if should_sync:
                        mx.eval(state)
                    # Grad norm not tracked in the fully-compiled (non-accumulation) path.
                    grad_norm = float("nan")
            else:
                # Standard training step
                (loss, model_out, cached_out_wav, cached_clean_wav), grads = loss_and_grad(
                    model,
                    noisy_real,
                    noisy_imag,
                    feat_erb,
                    feat_spec,
                    clean_real,
                    clean_imag,
                    snr,
                    vad_weight_mx,
                    speech_weight_mx,
                    awesome_weight_mx,
                    vad_reg_weight_mx,
                    gan_weight_mx,
                    fm_weight_mx,
                )

                # Detach cached waveforms from the gen backward graph.
                # They are only used for the disc update which doesn't
                # need gen gradients.  Releasing graph refs here frees
                # ~50-80 MB of intermediate activations before the disc
                # forward pass adds its own.
                if cached_out_wav is not None:
                    cached_out_wav = mx.stop_gradient(cached_out_wav)
                if cached_clean_wav is not None:
                    cached_clean_wav = mx.stop_gradient(cached_clean_wav)

                # Build lazy finiteness check — no sync barrier here.
                # The actual bool extraction is deferred to the should_sync
                # boundary.  Non-finite grads are safely zeroed by
                # clip_grad_norm, so the optimizer update is always safe.
                loss_finite_arr = mx.all(mx.isfinite(loss))

                should_sync = (batch_idx + 1) % epoch_eval_frequency == 0

                # Always accumulate gradients (the old skip_update gate is
                # removed: clip_grad_norm zeros NaN grads, making the update
                # a harmless no-op for non-finite batches).
                accumulated_grads = accumulate_grads(accumulated_grads, grads)
                accumulated_loss = accumulated_loss + loss
                micro_batches_in_accum += 1

                # Check if accumulation window is complete
                is_accum_complete = micro_batches_in_accum >= grad_accumulation_steps
                grad_norm_arr = None
                if is_accum_complete:
                    did_optimizer_update = True

                    # Scale by 1/grad_accumulation_steps for proper averaging
                    final_grads = scale_grads(accumulated_grads, 1.0 / grad_accumulation_steps)

                    # Gradient clipping (returns clipped grads and norm as
                    # MLX array).  clip_grad_norm zeros non-finite grads.
                    if max_grad_norm > 0:
                        final_grads, grad_norm_arr = clip_grad_norm(final_grads, max_grad_norm)

                    if _tree_all_finite(final_grads):
                        optimizer.update(model, final_grads)
                    else:
                        did_optimizer_update = False
                        tqdm.write(
                            "⚠️  Non-finite grads in eager path; skipping optimizer update "
                            f"(step={loop_state.global_step})"
                        )

                    # Reset accumulator for next window
                    accumulated_grads = None
                    accumulated_loss = SCALAR_ZERO
                    micro_batches_in_accum = 0

                # ---- Single sync point per eval_frequency batches ----
                if should_sync:
                    # Eval gen state now — before disc forward starts.
                    # This releases gen backward graph, reducing peak
                    # memory during the disc forward+backward pass.
                    _eval_targets: list[Any] = [
                        loss,
                        loss_finite_arr,
                        model.parameters(),
                        optimizer.state,
                    ]
                    if grad_norm_arr is not None:
                        _eval_targets.append(grad_norm_arr)

                    mx.eval(*_eval_targets)
                    # Release cached allocations before disc forward to
                    # reduce peak memory overlap between gen and disc.
                    mx.clear_cache()
                    # Extract deferred scalars (free — already eval'd)
                    loss_finite = bool(loss_finite_arr)
                    if not loss_finite:
                        tqdm.write(
                            f"⚠️  Non-finite loss detected (step={loop_state.global_step}); "
                            "grads were zeroed by clip_grad_norm"
                        )
                        if debugger is not None:
                            _diagnose_nonfinite(
                                noisy_real,
                                noisy_imag,
                                feat_erb,
                                feat_spec,
                                clean_real,
                                clean_imag,
                                snr,
                                debug_ctx,
                            )
                    if grad_norm_arr is not None:
                        grad_norm = float(grad_norm_arr)

            pred_spec_for_logging = None
            if model_out is not None:
                pred_spec_for_logging = (
                    mx.stop_gradient(model_out[0]),
                    mx.stop_gradient(model_out[1]),
                )
                # Release the original model_out (and its backward graph)
                # now that we have the stop_gradient copies for logging.
                del model_out

            gan_d_loss_val = 0.0
            if gan_active and discriminator is not None and disc_optimizer is not None and gan_loss_fns is not None:
                do_disc_update = did_optimizer_update and ((loop_state.global_step % gan_disc_update_freq) == 0)
                if do_disc_update:
                    _, disc_loss_fn = gan_loss_fns

                    if pred_spec_for_logging is None:
                        pred_spec = model((noisy_real, noisy_imag), feat_erb, feat_spec)
                        # pred_spec is (real, imag) — no return_vad needed here
                        pred_spec = (
                            mx.stop_gradient(pred_spec[0]),
                            mx.stop_gradient(pred_spec[1]),
                        )
                    else:
                        pred_spec = pred_spec_for_logging
                    pred_spec_for_logging = pred_spec
                    if gan_istft is not None:
                        if gan_cache_gen_waveforms and cached_out_wav is not None and cached_clean_wav is not None:
                            pred_wav = cached_out_wav
                            clean_wav = cached_clean_wav
                        else:
                            pred_wav, clean_wav = specs_to_wavs(
                                pred_spec,
                                (clean_real, clean_imag),
                                istft_fn=gan_istft,
                                n_fft=config.fft_size,
                                hop_length=config.hop_size,
                                target_len=gan_target_len,
                                force_fp32=use_mrstft_loss,
                            )
                        pred_wav = _gan_waveform_view(pred_wav, use_fp16=bool(use_fp16))
                        clean_wav = _gan_waveform_view(clean_wav, use_fp16=bool(use_fp16))
                        pred_wav = mx.stop_gradient(pred_wav)

                        # Crop to disc_max_samples (same offset for real/fake alignment)
                        clean_wav_d, d_crop = _disc_crop_waveform(clean_wav, gan_disc_max_samples)
                        pred_wav_d, _ = _disc_crop_waveform(pred_wav, gan_disc_max_samples, crop_start=d_crop)

                        if compiled_disc_update_step is not None:
                            disc_loss = compiled_disc_update_step(
                                clean_wav_d,
                                pred_wav_d,
                                _gan_disc_grad_clip_mx,
                            )
                        else:

                            def disc_loss_wrapper(disc):
                                real_out, _ = disc(clean_wav_d, return_features=False)
                                fake_out, _ = disc(pred_wav_d, return_features=False)
                                real_out = _clip_gan_scores(real_out)
                                fake_out = _clip_gan_scores(fake_out)
                                total_loss, _, _ = disc_loss_fn(real_out, fake_out)
                                return total_loss

                            disc_loss, disc_grads = nn.value_and_grad(discriminator, disc_loss_wrapper)(discriminator)

                            if gan_disc_grad_clip > 0:
                                disc_grads, _ = clip_grad_norm(disc_grads, gan_disc_grad_clip)

                            if _tree_all_finite(disc_grads):
                                disc_optimizer.update(discriminator, disc_grads)
                            else:
                                tqdm.write(
                                    f"\u26a0\ufe0f  Non-finite disc grads; skipping disc update (step={loop_state.global_step})"
                                )

                        if should_sync:
                            # Gen state was already eval'd above.
                            # Eval disc state separately to avoid
                            # materializing gen+disc graphs at once.
                            mx.eval(
                                disc_loss,
                                discriminator.parameters(),
                                disc_optimizer.state,
                            )
                            gan_d_loss_val = float(disc_loss)
                            train_gan_d_updates += 1

            fwd_time = time.perf_counter() - fwd_start
            total_forward_time += fwd_time

            # Only convert loss to float when synced (avoids blocking)
            _loss_was_nonfinite = False
            if should_sync:
                loss_val = float(loss)
                if not math.isfinite(loss_val):
                    _loss_was_nonfinite = True
                    # Non-finite loss was already handled by clip_grad_norm
                    # (grads zeroed) and optionally diagnosed above.
                    # Substitute zero so epoch averaging isn't poisoned, and
                    # count it so we can abort if too many accumulate.
                    nonfinite_loss_count = getattr(train, "_nonfinite_loss_count", 0) + 1
                    train._nonfinite_loss_count = nonfinite_loss_count  # type: ignore[attr-defined]
                    tqdm.write(
                        f"⚠️  Non-finite loss_val at sync point "
                        f"(epoch={epoch}, batch={batch_idx}, step={loop_state.global_step}, "
                        f"cumulative_nonfinite={nonfinite_loss_count})"
                    )
                    _MAX_NONFINITE_LOSSES = 50
                    if nonfinite_loss_count >= _MAX_NONFINITE_LOSSES:
                        raise FloatingPointError(
                            f"Aborting: {nonfinite_loss_count} non-finite losses "
                            f"in this session (epoch={epoch}, step={loop_state.global_step}). "
                            "Model is likely diverged."
                        )
                    loss_val = 0.0
                train_loss += loss_val * epoch_eval_frequency  # Approximate accumulated loss
                if gan_active and gan_d_loss_val:
                    train_gan_d_loss += gan_d_loss_val

                # Debug mode: log per-step gradient norm for full observability
                if sync_mode == "debug" and math.isfinite(grad_norm):
                    tqdm.write(
                        f"  [debug] step={loop_state.global_step} grad_norm={grad_norm:.4f} " f"loss={loss_val:.6f}"
                    )

                # Profile mode: log step-level timing breakdown
                if sync_mode == "profile":
                    tqdm.write(
                        f"  [profile] step={loop_state.global_step} "
                        f"data={data_time * 1000:.1f}ms "
                        f"fwd={fwd_time * 1000:.1f}ms "
                        f"total={(data_time + fwd_time) * 1000:.1f}ms"
                    )
            num_train_batches += 1
            samples_processed += current_batch_size
            window_samples += current_batch_size
            # Only increment loop_state.global_step when optimizer actually updates
            # (for gradient accumulation > 1, updates happen every N batches)
            if did_optimizer_update:
                loop_state.global_step += 1

            # Track progress for interruption-safe resume metadata
            _update_interrupt_state(
                epoch,
                loss_val,
                loop_state.best_valid_loss,
                batch_idx=num_train_batches,
                global_step=loop_state.global_step,
                last_completed_epoch=loop_state.last_completed_epoch,
                pipeline_stage_index=loop_state.active_stage_index,
                pipeline_stage_name=loop_state.active_stage_name,
            )

            # Stop early for benchmarking if requested
            if max_train_batches is not None and num_train_batches >= max_train_batches:
                break

            # Update progress bar with real-time metrics (only on sync)
            if should_sync:
                lr = float(schedule(loop_state.global_step))
                # Throughput: samples processed in this sync window / wall-clock time
                window_elapsed = time.perf_counter() - window_start
                samples_per_sec = window_samples / max(window_elapsed, 1e-6)
                window_samples = 0
                window_start = time.perf_counter()

                _display = collect_sync_metrics(
                    noisy_real=noisy_real,
                    noisy_imag=noisy_imag,
                    clean_real=clean_real,
                    clean_imag=clean_imag,
                    snr=snr,
                    model=model,
                    feat_erb=feat_erb,
                    feat_spec=feat_spec,
                    pred_spec_for_logging=pred_spec_for_logging,
                    loss_val=loss_val,
                    loss_was_nonfinite=_loss_was_nonfinite,
                    epoch_eval_frequency=epoch_eval_frequency,
                    use_mrstft_loss=use_mrstft_loss,
                    use_vad_loss=use_vad_loss,
                    use_awesome_loss=use_awesome_loss,
                    use_pipeline_awesome_loss=use_pipeline_awesome_loss,
                    use_vad_train_reg=use_vad_train_reg,
                    use_fp16=use_fp16,
                    gan_active=gan_active,
                    emit_detailed_metrics=emit_detailed_metrics,
                    apply_vad_reg=apply_vad_reg,
                    debug_numerics=debug_numerics,
                    speech_weight=speech_weight,
                    spectral_loss_fn=spectral_loss,
                    mrstft_loss_fn=mrstft_loss_fn,
                    mrstft_istft=mrstft_istft,
                    mrstft_target_len=mrstft_target_len,
                    discriminator=discriminator,
                    feature_match_loss=feature_match_loss,
                    gan_loss_fns=gan_loss_fns,
                    gan_istft=gan_istft,
                    gan_fm_weight=gan_fm_weight,
                    gan_disc_max_samples=gan_disc_max_samples,
                    gan_target_len=gan_target_len,
                    config_fft_size=config.fft_size,
                    config_hop_size=config.hop_size,
                    config_sample_rate=config.sample_rate,
                    vad_band_mask=vad_band_mask,
                    vad_band_bins=vad_band_bins,
                    vad_threshold=vad_threshold,
                    vad_margin=vad_margin,
                    vad_snr_gate_db=vad_snr_gate_db,
                    vad_snr_gate_width=vad_snr_gate_width,
                    vad_z_threshold=vad_z_threshold,
                    vad_z_slope=vad_z_slope,
                    awesome_mask_sharpness=awesome_mask_sharpness,
                    vad_proxy_enabled=vad_proxy_enabled,
                    debugger=debugger,
                    debug_ctx=debug_ctx,
                    accums=_epoch_accums,
                )

                update_progress_bar(
                    train_pbar,
                    _display,
                    loss_val=loss_val,
                    train_loss=train_loss,
                    num_train_batches=num_train_batches,
                    gan_d_loss_val=gan_d_loss_val,
                    lr=lr,
                    grad_norm=grad_norm,
                    samples_per_sec=samples_per_sec,
                    data_time=data_time,
                    fwd_time=fwd_time,
                    global_step=loop_state.global_step,
                    verbose=verbose,
                    use_mrstft_loss=use_mrstft_loss,
                    use_vad_loss=use_vad_loss,
                    use_awesome_loss=use_awesome_loss,
                    use_pipeline_awesome_loss=use_pipeline_awesome_loss,
                    use_vad_train_reg=use_vad_train_reg,
                    gan_active=gan_active,
                )

            # Save data checkpoint periodically (for resume capability)
            if checkpoint_batches > 0 and use_mlx_stream and train_stream is not None:
                if (batch_idx + 1) % checkpoint_batches == 0:
                    _sync_data_stream_stage(loop_state.active_stage_index, loop_state.active_stage_name)
                    train_stream.save_checkpoint(data_checkpoint_path)

            # Save model checkpoint by steps (HuggingFace-style)
            if save_strategy == "steps" and save_steps > 0 and loop_state.global_step % save_steps == 0:
                # Force sync before checkpoint to get accurate loss
                mx.eval(state)
                loss_val = float(loss)

                ckpt_path = ckpt_dir / f"step_{loop_state.global_step:06d}.safetensors"
                step_saved = save_checkpoint(
                    model,
                    ckpt_path,
                    epoch=epoch,
                    batch_idx=num_train_batches,
                    global_step=loop_state.global_step,
                    loss=train_loss / num_train_batches if num_train_batches > 0 else loss_val,
                    best_valid_loss=loop_state.best_valid_loss,
                    config=train_config,
                    optimizer=optimizer,
                    discriminator=discriminator,
                    disc_optimizer=disc_optimizer,
                    last_completed_epoch=loop_state.last_completed_epoch,
                    pipeline_stage_index=loop_state.active_stage_index,
                    pipeline_stage_name=loop_state.active_stage_name,
                    kind="step",
                )
                if step_saved:
                    tqdm.write(f"  📦 Checkpoint saved: {ckpt_path.name} (step {loop_state.global_step})")
                else:
                    tqdm.write(f"  ⚠️  Checkpoint save failed: {ckpt_path.name} (step {loop_state.global_step})")

                # Cleanup old checkpoints if limit is set
                if save_total_limit is not None:
                    cleanup_checkpoints(ckpt_dir, save_total_limit)

            # Start timing for next data fetch
            data_start = time.perf_counter()

        train_pbar.close()

        # Force sync at epoch end to ensure accurate loss
        mx.eval(state)

        # Save data checkpoint at end of epoch (for clean resume at epoch boundary)
        if use_mlx_stream and train_stream is not None:
            _sync_data_stream_stage(loop_state.active_stage_index, loop_state.active_stage_name)
            train_stream.save_checkpoint(data_checkpoint_path)

        _n = max(num_train_batches, 1)
        loop_state.avg_train_loss = train_loss / _n
        epoch_avgs = compute_epoch_averages(
            _epoch_accums,
            train_loss=train_loss,
            num_train_batches=num_train_batches,
            train_gan_d_loss=train_gan_d_loss,
            train_gan_d_updates=train_gan_d_updates,
        )

        # Print detailed timing breakdown in verbose mode
        if verbose and num_train_batches > 0:
            total_time = total_data_time + total_forward_time
            print(f"\n  [Timing Breakdown - Epoch {epoch + 1}]")
            print(f"    Data loading:       {total_data_time:6.1f}s ({100 * total_data_time / total_time:5.1f}%)")
            print(
                f"    Train step (fwd+bwd+upd): {total_forward_time:6.1f}s ({100 * total_forward_time / total_time:5.1f}%)"
            )
            print(f"    TOTAL:              {total_time:6.1f}s")
            print(f"    Compiled training:  {'enabled' if epoch_use_compiled_step else 'disabled'}")
            if total_data_time > total_forward_time:
                print("    ⚠️  DATA LOADING IS BOTTLENECK - consider more workers or faster storage")

        if partial_batch_fallbacks > 0:
            print(
                "  Compile boundary fallback: "
                f"{partial_batch_fallbacks} batch(es) ran eager due to non-canonical batch size"
            )

        # ====== Validation ======
        avg_valid_loss = float("inf")
        best_saved = False
        if (epoch + 1) % validate_every == 0:
            do_vad_eval = vad_eval_enabled and (vad_eval_every > 0) and ((epoch + 1) % vad_eval_every == 0)
            avg_valid_loss = _run_validation(
                _valid_ctx,
                epoch=epoch,
                global_step=loop_state.global_step,
                epoch_awesome_loss_weight=loop_state.epoch_awesome_loss_weight,
                epoch_vad_loss_weight=loop_state.epoch_vad_loss_weight,
                epoch_vad_speech_loss_weight=loop_state.epoch_vad_speech_loss_weight,
                active_stage_index=loop_state.active_stage_index,
                active_stage_name=loop_state.active_stage_name,
                train_mode=loop_state.train_mode or "EAGER",
                label="  Validating",
                do_vad_eval=do_vad_eval,
            )
            loop_state.last_valid_loss = avg_valid_loss
            loop_state.last_valid_epoch = epoch

            # Early stopping check
            if avg_valid_loss < loop_state.best_valid_loss:
                loop_state.best_valid_loss = avg_valid_loss
                loop_state.epochs_without_improvement = 0

                # Save best model
                best_path = ckpt_dir / "best.safetensors"
                best_saved = save_checkpoint(
                    model,
                    best_path,
                    epoch=epoch,
                    batch_idx=None,
                    global_step=loop_state.global_step,
                    loss=loop_state.avg_train_loss,
                    best_valid_loss=loop_state.best_valid_loss,
                    config=train_config,
                    optimizer=optimizer,
                    discriminator=discriminator,
                    disc_optimizer=disc_optimizer,
                    last_completed_epoch=epoch,
                    pipeline_stage_index=loop_state.active_stage_index,
                    pipeline_stage_name=loop_state.active_stage_name,
                    kind="best",
                )
                if best_saved:
                    loop_state.last_completed_epoch = max(loop_state.last_completed_epoch, epoch)
                    _update_interrupt_state(
                        epoch,
                        loop_state.avg_train_loss,
                        loop_state.best_valid_loss,
                        batch_idx=num_train_batches,
                        global_step=loop_state.global_step,
                        last_completed_epoch=loop_state.last_completed_epoch,
                        pipeline_stage_index=loop_state.active_stage_index,
                        pipeline_stage_name=loop_state.active_stage_name,
                    )
                else:
                    print("⚠️  Best checkpoint save failed; epoch completion not updated.")
            else:
                loop_state.epochs_without_improvement += 1

        # ====== Epoch Summary ======
        epoch_time = time.perf_counter() - epoch_start

        # Update interrupt state with final epoch metrics
        _update_interrupt_state(
            epoch,
            loop_state.avg_train_loss,
            loop_state.best_valid_loss,
            batch_idx=num_train_batches,
            global_step=loop_state.global_step,
            last_completed_epoch=loop_state.last_completed_epoch,
            pipeline_stage_index=loop_state.active_stage_index,
            pipeline_stage_name=loop_state.active_stage_name,
        )

        print_epoch_summary(
            epoch_avgs,
            epoch=epoch,
            epochs=epochs,
            avg_valid_loss=avg_valid_loss,
            best_valid_loss=loop_state.best_valid_loss,
            samples_processed=samples_processed,
            epoch_time=epoch_time,
            use_vad_loss=use_vad_loss,
            use_awesome_loss=use_awesome_loss,
            use_pipeline_awesome_loss=use_pipeline_awesome_loss,
            use_mrstft_loss=use_mrstft_loss,
            use_vad_train_reg=use_vad_train_reg,
            gan_enabled=gan_enabled,
            gan_fm_weight=gan_fm_weight,
            verbose=verbose,
            debug_numerics=debug_numerics,
            num_debug_logs=_epoch_accums["num_debug_logs"],
            train_mask_clip_rate=_epoch_accums["mask_clip_rate"],
            train_eps_clean_rate=_epoch_accums["eps_clean_rate"],
            train_eps_noise_rate=_epoch_accums["eps_noise_rate"],
            train_mask_logit_min=_epoch_accums["mask_logit_min"],
            train_mask_logit_max=_epoch_accums["mask_logit_max"],
            num_vad_logs=_epoch_accums["num_vad_logs"],
            train_vad_clip_ref=_epoch_accums["vad_clip_ref"],
            train_vad_clip_out=_epoch_accums["vad_clip_out"],
        )

        # ====== Early Stopping / Curriculum Advance ======
        should_stop = False
        if patience > 0 and loop_state.epochs_without_improvement >= patience:
            if loop_state.active_stage_index + 1 < len(pipeline_stage_defs):
                prev_stage_index = loop_state.active_stage_index
                loop_state.active_stage_index += 1
                next_stage = _resolve_pipeline_stage_by_index(loop_state.active_stage_index, pipeline_stage_defs)
                loop_state.active_stage_name = str(next_stage["name"])
                print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                print(
                    "Moving to next pipeline stage "
                    f"'{loop_state.active_stage_name}' early ({prev_stage_index} -> {loop_state.active_stage_index})."
                )
                loop_state.best_valid_loss = float("inf")
                loop_state.epochs_without_improvement = 0
                train_config["pipeline_stage_active"] = {
                    "index": loop_state.active_stage_index,
                    "name": loop_state.active_stage_name,
                    "start_epoch": int(next_stage["start_epoch"]),
                    "awesome_loss_weight": (
                        float(next_stage["awesome_loss_weight"])
                        if next_stage["awesome_loss_weight"] is not None
                        else base_awesome_loss_weight
                    ),
                    "vad_loss_weight": (
                        float(next_stage["vad_loss_weight"])
                        if next_stage["vad_loss_weight"] is not None
                        else base_vad_loss_weight
                    ),
                    "vad_speech_loss_weight": (
                        float(next_stage["vad_speech_loss_weight"])
                        if next_stage["vad_speech_loss_weight"] is not None
                        else base_vad_speech_loss_weight
                    ),
                }
                _sync_data_stream_stage(loop_state.active_stage_index, loop_state.active_stage_name)
                _update_interrupt_state(
                    epoch,
                    loop_state.avg_train_loss,
                    loop_state.best_valid_loss,
                    batch_idx=num_train_batches,
                    global_step=loop_state.global_step,
                    last_completed_epoch=loop_state.last_completed_epoch,
                    pipeline_stage_index=loop_state.active_stage_index,
                    pipeline_stage_name=loop_state.active_stage_name,
                )
            else:
                should_stop = True

        # ====== End-of-Epoch Checkpointing (authoritative completion) ======
        ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.safetensors"
        epoch_saved = save_checkpoint(
            model,
            ckpt_path,
            epoch=epoch,
            batch_idx=None,
            global_step=loop_state.global_step,
            loss=loop_state.avg_train_loss,
            best_valid_loss=loop_state.best_valid_loss,
            config=train_config,
            optimizer=optimizer,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
            last_completed_epoch=epoch,
            pipeline_stage_index=loop_state.active_stage_index,
            pipeline_stage_name=loop_state.active_stage_name,
            kind="epoch_end",
        )
        epoch_completed = epoch_saved or best_saved
        if epoch_saved:
            loop_state.last_completed_epoch = epoch
            _update_interrupt_state(
                epoch,
                loop_state.avg_train_loss,
                loop_state.best_valid_loss,
                batch_idx=num_train_batches,
                global_step=loop_state.global_step,
                last_completed_epoch=loop_state.last_completed_epoch,
                pipeline_stage_index=loop_state.active_stage_index,
                pipeline_stage_name=loop_state.active_stage_name,
            )
            _write_epoch_complete_marker(ckpt_dir, epoch, ckpt_path)
            print(f"  📦 Checkpoint saved: {ckpt_path.name}")
            if save_total_limit is not None:
                cleanup_checkpoints(ckpt_dir, save_total_limit)
        else:
            if epoch_completed:
                print("⚠️  End-of-epoch checkpoint failed; relying on best checkpoint for completion.")
            else:
                print("⚠️  End-of-epoch checkpoint failed; epoch not marked as complete.")

        if should_stop:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break

        # Clear memory periodically
        if (epoch + 1) % 10 == 0:
            gc.collect()

    # Final validation to compare against best checkpoint.
    def _run_final_validation() -> float:
        return _run_validation(
            _valid_ctx,
            epoch=loop_state.final_epoch,
            global_step=loop_state.global_step,
            epoch_awesome_loss_weight=loop_state.epoch_awesome_loss_weight,
            epoch_vad_loss_weight=loop_state.epoch_vad_loss_weight,
            epoch_vad_speech_loss_weight=loop_state.epoch_vad_speech_loss_weight,
            active_stage_index=loop_state.active_stage_index,
            active_stage_name=loop_state.active_stage_name,
            train_mode=loop_state.train_mode or "EAGER",
            label="  Final validation",
            do_vad_eval=vad_eval_enabled,
        )

    finalize_training(
        final_epoch=loop_state.final_epoch,
        global_step=loop_state.global_step,
        avg_train_loss=loop_state.avg_train_loss,
        best_valid_loss=loop_state.best_valid_loss,
        last_completed_epoch=loop_state.last_completed_epoch,
        last_valid_epoch=loop_state.last_valid_epoch,
        last_valid_loss=loop_state.last_valid_loss,
        model=model,
        optimizer=optimizer,
        state=state,
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
        ckpt_dir=ckpt_dir,
        train_config=train_config,
        active_stage_index=loop_state.active_stage_index,
        active_stage_name=loop_state.active_stage_name,
        tqdm_setup_panel=tqdm_setup_panel,
        run_validation_fn=_run_final_validation,
    )


# main() is now in df_mlx.training_cli_main (re-exported above).


if __name__ == "__main__":
    main()
