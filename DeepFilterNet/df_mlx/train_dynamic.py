#!/usr/bin/env python3
"""Train MLX DeepFilterNet4 with dynamic on-the-fly mixing.

This script provides training using the dynamic dataset which mirrors the
original Rust DataLoader:
- Dynamic speech + noise + RIR mixing each epoch
- Full dataset diversity (all files available each epoch)
- Same speech can appear with different noise/RIR/SNR each epoch

Usage:
    python -m df_mlx.train_dynamic \
        --speech-list /path/to/speech_files.txt \
        --noise-list /path/to/noise_files.txt \
        --rir-list /path/to/rir_files.txt \
        --epochs 100 \
        --batch-size 8 \
        --checkpoint-dir ./checkpoints

    # Or with a config file
    python -m df_mlx.train_dynamic \
        --config dataset_config.json \
        --epochs 100

    # Or with a train.py-compatible INI config
    python -m df_mlx.train_dynamic \
        --config dataset_config.json \
        --train-config training_config.ini \
        --epochs 100

Features:
    - Dynamic on-the-fly mixing (matches original training strategy)
    - Full dataset diversity each epoch
    - Automatic learning rate scheduling
    - Gradient clipping for stability
    - Periodic checkpointing
    - Validation with fixed noise/RIR for reproducibility
    - Optional GAN adversarial + feature matching loss for perceptual cleanup
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from df_mlx.hardware import print_hardware_diagnostics  # noqa: E402, F401
from df_mlx.run_config import (  # noqa: E402, F401
    RunConfig,
    SyncMode,
    generate_run_config_example,
    load_preset_config,
    load_run_config,
    validate_run_config,
)
from df_mlx.train_dynamic_config import (  # noqa: E402, F401
    apply_train_ini_config,
    apply_train_ini_tables,
)
from df_mlx.training_checkpoints import (  # noqa: E402, F401
    _CHECKPOINT_KINDS,
    _COMPLETED_KINDS,
    _COUNTER_SEMANTICS_VERSION,
    _IN_PROGRESS_KINDS,
    _TRAIN_MODE_COMPILED,
    _TRAIN_MODE_EAGER,
    CheckpointManifest,
    CheckpointRecord,
    _disc_weights_path,
    _is_disc_weights,
    _record_sort_key,
    _validate_checkpoint_pair,
    _write_epoch_complete_marker,
    cleanup_checkpoints,
    compute_resume_epoch,
    find_latest_checkpoint,
    load_checkpoint,
    maybe_skip_resume_batches,
    resolve_epoch_train_mode,
    resolve_resume_batch_count,
    save_checkpoint,
    validate_checkpoint_dir,
)
from df_mlx.training_cli import (  # noqa: E402, F401
    _apply_cli_overrides,
    _flag_in_argv,
    _parse_pipeline_stages_cli,
    _resolve_pipeline_stage,
)
from df_mlx.training_cli_main import main  # noqa: E402, F401
from df_mlx.training_losses import (  # noqa: E402, F401
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
    _compute_awesome_losses,
    _compute_harmonic_ratio,
    _compute_improved_musicness,
    _compute_musicness,
    _compute_pipeline_awesome_losses,
    _compute_pitch_stability,
    _compute_proxy_gates,
    _compute_speech_band_logmag_loss,
    _compute_vad_eval_metrics,
    _compute_vad_loss,
    _compute_vad_probs,
    _compute_vad_reg_loss,
    _log1p_mag,
    _snr_bucket_name,
    _sync_model_config_with_dataset,
)
from df_mlx.training_ops import (  # noqa: E402, F401
    NumericDebugConfig,
    NumericDebugger,
    _batch_to_float,
    _tree_all_finite,
    accumulate_grads,
    clip_grad_norm,
    scale_grads,
)
from df_mlx.training_session import (  # noqa: E402, F401
    _SENTINEL,
    _TRAIN_KWARGS,
    TrainingSession,
    _kwargs_from_run_config,
)
from df_mlx.training_signals import (  # noqa: E402, F401
    _handle_sigint,
    _interrupt_state,
    _register_sigint_handler,
    _update_interrupt_state,
)
from df_mlx.training_waveform import (  # noqa: E402, F401
    _disc_crop_waveform,
    _gan_waveform_view,
    compute_mrstft_loss,
    specs_to_wavs,
)

if TYPE_CHECKING:
    from df_mlx.config import ModelParams4
    from df_mlx.run_config import MultiResSpecLossConfig

# Cached scalar zero — reused for default loss placeholders in validation and
# accumulated-loss resets.  Avoids repeated micro-allocations.  MLX arrays are
# value-immutable, so sharing a single instance is safe.
_SCALAR_ZERO = mx.array(0.0)

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


def _build_setup_panel_line(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dynamic_loss: str,
    gan_enabled: bool,
    vad_enabled: bool,
    checkpoint_dir: str,
    use_fp16: bool,
) -> str:
    """Build single-line setup metadata for the persistent setup panel."""
    return (
        "SETUP │ "
        f"epochs={epochs} "
        f"bs={batch_size} "
        f"lr={learning_rate:.1e} "
        f"loss={dynamic_loss} "
        f"gan={'on' if gan_enabled else 'off'} "
        f"vad={'on' if vad_enabled else 'off'} "
        f"fp16={'on' if use_fp16 else 'off'} "
        f"ckpt={checkpoint_dir}"
    )


# =============================================================================
# Curriculum Learning Scheduler
# =============================================================================


def curriculum_schedule(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    target_p_extreme: float,
    target_p_very_low: float,
    target_p_interfer: float,
) -> tuple[float, float, float]:
    """Compute curriculum-scheduled SNR and interferer probabilities.

    During warmup, we start with easy (high SNR) samples and gradually
    introduce harder samples. After warmup, we use the full target distribution.

    Schedule:
    - Epoch 0 to warmup_epochs: linear ramp from 0 to target values
    - After warmup_epochs: use full target values

    Args:
        epoch: Current training epoch (0-indexed)
        total_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs for curriculum
        target_p_extreme: Final probability for extreme SNR
        target_p_very_low: Final probability for very-low SNR
        target_p_interfer: Final probability for interfering speech

    Returns:
        Tuple of (p_extreme_snr, p_very_low_snr, p_interfer_speech)
    """
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        # Past warmup: use full target distribution
        return target_p_extreme, target_p_very_low, target_p_interfer

    # Linear ramp during warmup
    progress = epoch / warmup_epochs
    return (
        progress * target_p_extreme,
        progress * target_p_very_low,
        progress * target_p_interfer,
    )


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
    from df_mlx.config import get_default_config
    from df_mlx.dynamic_dataset import (
        HAS_MLX_DATA,
        DatasetConfig,
        DynamicDataset,
        MLXDataStream,
        PrefetchDataLoader,
        read_file_list,
    )
    from df_mlx.hardware import HardwareConfig
    from df_mlx.model import count_parameters, init_model
    from df_mlx.train import MultiResolutionSTFTLoss, WarmupCosineSchedule, spectral_loss

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

    # Load or create config
    if cache_dir:
        # Load config from pre-built audio cache
        cache_path = Path(cache_dir)
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

    # Create dataset (this populates config.*_files from cache index if using cache)
    print("\nInitializing dynamic dataset...")
    dataset = DynamicDataset(config)

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

    discriminator = None
    disc_optimizer = None
    feature_match_loss = None
    gan_loss_fns = None

    if gan_enabled:
        from functools import partial

        from df_mlx.discriminator import (
            CombinedDiscriminator,
            MultiPeriodDiscriminator,
            MultiScaleDiscriminator,
        )
        from df_mlx.loss import FeatureMatchingLoss, discriminator_loss, generator_loss
        from df_mlx.ops import istft

        if gan_istft is None:
            gan_istft = partial(istft)

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
    use_vad_train_reg = (vad_train_prob > 0 or vad_train_every_steps > 0) and vad_loss_weight > 0

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
        vad_band_mask = _SCALAR_ZERO
        vad_band_bins = 1.0

    min_lr = learning_rate_min if learning_rate_min is not None else learning_rate * 0.01

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

    train_config = {
        **config.__dict__,
        "train_config_path": train_config_path,
        "dynamic_loss": dynamic_loss,
        "pipeline_stages": pipeline_stage_defs,
        "awesome_loss_weight": awesome_loss_weight,
        "awesome_mask_sharpness": awesome_mask_sharpness,
        "awesome_warmup_steps": awesome_warmup_steps,
        "vad_proxy_enabled": vad_proxy_enabled,
        "mrstft_factor": mrstft_cfg.factor if mrstft_cfg is not None else 0.0,
        "mrstft_gamma": mrstft_cfg.gamma if mrstft_cfg is not None else 1.0,
        "mrstft_f_complex": mrstft_cfg.f_complex if mrstft_cfg is not None else None,
        "mrstft_fft_sizes": list(mrstft_cfg.fft_sizes) if mrstft_cfg is not None else None,
        "mrstft_hop_sizes": (list(mrstft_cfg.hop_sizes) if (mrstft_cfg and mrstft_cfg.hop_sizes) else None),
        "gan_enabled": gan_enabled,
        "gan_start_epoch": gan_start_epoch,
        "gan_ramp_epochs": gan_ramp_epochs,
        "gan_adv_weight": gan_adv_weight,
        "gan_fm_weight": gan_fm_weight,
        "gan_disc_type": gan_disc_type,
        "gan_mpd_periods": list(gan_mpd_periods) if gan_mpd_periods else [2, 3, 5, 7, 11],
        "gan_msd_scales": gan_msd_scales,
        "gan_disc_lr": gan_disc_lr,
        "gan_disc_weight_decay": gan_disc_weight_decay,
        "gan_disc_grad_clip": gan_disc_grad_clip,
        "gan_disc_update_freq": gan_disc_update_freq,
        "gan_cache_gen_waveforms": gan_cache_gen_waveforms,
        "gan_disc_gradient_checkpoint": gan_disc_gradient_checkpoint,
        "gan_gen_gradient_checkpoint": gan_gen_gradient_checkpoint,
        "gan_eval_frequency": gan_eval_frequency,
        "experimental_compiled_gan": experimental_compiled_gan,
        "vad_loss_weight": vad_loss_weight,
        "vad_threshold": vad_threshold,
        "vad_margin": vad_margin,
        "vad_speech_loss_weight": vad_speech_loss_weight,
        "vad_warmup_epochs": vad_warmup_epochs,
        "vad_snr_gate_db": vad_snr_gate_db,
        "vad_snr_gate_width": vad_snr_gate_width,
        "vad_band_low_hz": vad_band_low_hz,
        "vad_band_high_hz": vad_band_high_hz,
        "vad_z_threshold": vad_z_threshold,
        "vad_z_slope": vad_z_slope,
        "vad_eval_mode": vad_eval_mode,
        "vad_eval_every": vad_eval_every,
        "vad_eval_batches": vad_eval_batches,
        "vad_eval_max_seconds": vad_eval_max_seconds,
        "vad_silero_model_path": vad_silero_model_path,
        "vad_silero_sample_rate": vad_silero_sample_rate,
        "vad_train_prob": vad_train_prob,
        "vad_train_every_steps": vad_train_every_steps,
        "eval_sisdr": eval_sisdr,
        "max_train_batches": max_train_batches,
        "max_valid_batches": max_valid_batches,
        "seed": seed,
        "learning_rate_min": learning_rate_min,
        "weight_decay": weight_decay,
        "model_variant": model_variant,
        "debug_numerics": debug_numerics,
        "debug_numerics_fail_fast": debug_numerics_fail_fast,
        "debug_numerics_every": debug_numerics_every,
        "nan_skip_batch": nan_skip_batch,
    }

    dataset.set_split("train")

    print(f"  Train samples: {len(dataset):,}")

    # Create validation dataset (with reproducible indices)
    dataset.set_split("valid")
    print(f"  Valid samples: {len(dataset):,}")

    # Reset to training
    dataset.set_split("train")
    dataset.set_epoch(0)

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

        # Make data checkpoint path available to the interrupt handler
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
    start_epoch = 0
    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    last_completed_epoch = -1
    resume_global_step = 0
    resume_batch_idx = 0
    resume_checkpoint_kind = "epoch_end"

    if resume_from:
        state = load_checkpoint(
            model,
            resume_from,
            optimizer=optimizer,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
        )
        if state:
            ckpt_epoch = int(state.get("epoch", 0))
            ckpt_kind = state.get("kind", "epoch_end")
            resume_checkpoint_kind = ckpt_kind if isinstance(ckpt_kind, str) else "epoch_end"
            resume_global_step = state.get(
                "optimizer_steps_completed",
                state.get("global_step", ckpt_epoch * optimizer_steps_per_epoch),
            )
            start_epoch = compute_resume_epoch(state)
            completed_kinds = {"epoch_end", "best", "best_final", "final"}
            if ckpt_kind in completed_kinds:
                last_completed_epoch = state.get("last_completed_epoch", ckpt_epoch)
            else:
                last_completed_epoch = state.get("last_completed_epoch", ckpt_epoch - 1)
            if ckpt_kind in _IN_PROGRESS_KINDS:
                resume_batch_idx = resolve_resume_batch_count(state)
            best_valid_loss = state.get("best_valid_loss", float("inf"))
            print(
                "  Resumed from: "
                f"{resume_from} (epoch {start_epoch}, kind={ckpt_kind}, "
                f"last_completed={last_completed_epoch})"
            )
            print(
                "  Resume target: "
                f"epoch {start_epoch + 1} (idx {start_epoch}), "
                f"micro_batch {resume_batch_idx}, global_step {resume_global_step}"
            )
            if start_epoch >= epochs:
                print(f"✅ Training already complete (checkpoint epoch {ckpt_epoch}/{epochs}).")
                if tqdm_setup_panel is not None:
                    tqdm_setup_panel.close()
                return

    if validation_report and validation_report["last_completed_epoch"] > last_completed_epoch:
        last_completed_epoch = validation_report["last_completed_epoch"]

    if train_stream is not None and data_resume_progress is not None:
        data_epoch = data_resume_progress.get("epoch")
        data_batch = data_resume_progress.get("batch")
        if not isinstance(data_epoch, int) or not isinstance(data_batch, int):
            raise RuntimeError(
                "Data checkpoint progress is malformed. "
                f"source={data_resume_source}, progress={data_resume_progress}"
            )

        resume_requires_mid_epoch = resume_from is not None and resume_checkpoint_kind in _IN_PROGRESS_KINDS
        if resume_requires_mid_epoch:
            if data_epoch != start_epoch or data_batch != resume_batch_idx:
                # The model checkpoint is authoritative for how many micro-batches
                # were fully processed.  The data stream's counter can be ±1 ahead
                # because its iterator pre-increments before yield, so an interrupt
                # may capture a higher count.  Auto-correct when the epoch matches
                # and the batch delta is small; reject only on large or cross-epoch
                # mismatches.
                batch_delta = abs(data_batch - resume_batch_idx)
                if data_epoch == start_epoch and batch_delta <= 1:
                    print(
                        f"ℹ️  Auto-correcting data checkpoint batch position: "
                        f"data={data_batch} → model={resume_batch_idx} "
                        f"(delta={batch_delta}, epoch={start_epoch})."
                    )
                    train_stream.set_resume_position(epoch=start_epoch, batch_idx=resume_batch_idx)
                else:
                    raise RuntimeError(
                        "Model checkpoint and data checkpoint disagree on resume position. "
                        f"model=(epoch={start_epoch}, micro_batch={resume_batch_idx}, kind={resume_checkpoint_kind}), "
                        f"data=(epoch={data_epoch}, micro_batch={data_batch}) from {data_resume_source}. "
                        "Remediation: remove stale data_checkpoint.json or choose matching resume artifacts."
                    )
        else:
            # Resuming from an epoch-boundary checkpoint should always restart at batch 0.
            if data_epoch != start_epoch or data_batch > 0:
                print(
                    "ℹ️  Ignoring mid-epoch data checkpoint for epoch-boundary resume: "
                    f"data=(epoch={data_epoch}, micro_batch={data_batch}), resume_epoch={start_epoch}."
                )
                train_stream.set_epoch(start_epoch)
                data_resume_progress = None
            elif data_batch == 0:
                data_resume_progress = None

    if resume_from:
        lc_display = f"{last_completed_epoch + 1} (idx {last_completed_epoch})" if last_completed_epoch >= 0 else "none"
        print(f"  last_completed_epoch: {lc_display}")

    _interrupt_state["last_completed_epoch"] = last_completed_epoch

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
        out = _gen_fn(noisy_spec, feat_erb, feat_spec)
        spec_loss = spectral_loss(out, target_spec)
        total_loss = spec_loss

        out_wav = None
        clean_wav = None
        if (use_mrstft_loss or gan_active) and gan_istft is not None:
            out_wav, clean_wav = specs_to_wavs(
                out,
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
                out[0],
                out[1],
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
                out[0],
                out[1],
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
            vad_loss, _, _, gate = _compute_vad_loss(
                clean_real,
                clean_imag,
                out[0],
                out[1],
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
            speech_loss = _SCALAR_ZERO
            if vad_speech_loss_weight > 0:
                speech_loss = _compute_speech_band_logmag_loss(
                    clean_real,
                    clean_imag,
                    out[0],
                    out[1],
                    vad_band_mask,
                    vad_band_bins,
                    gate,
                )
            total_loss = total_loss + vad_weight * vad_loss + speech_weight * speech_loss

        if use_vad_train_reg:
            vad_reg_loss, _, _, _, _, _, _ = _compute_vad_reg_loss(
                clean_real,
                clean_imag,
                noisy_real,
                noisy_imag,
                out[0],
                out[1],
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
        return total_loss, out, out_wav, clean_wav

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
    _compiled_gan_correctness_verified = False

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
            out = _gen_fn(noisy_spec, feat_erb, feat_spec)
            spec_loss = spectral_loss(out, target_spec)
            total_loss = spec_loss

            out_wav = None
            clean_wav = None
            # GAN always active: always compute waveforms
            if gan_istft is not None:
                out_wav, clean_wav = specs_to_wavs(
                    out,
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
                    out[0],
                    out[1],
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
                    out[0],
                    out[1],
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
                vad_loss, _, _, gate = _compute_vad_loss(
                    clean_real,
                    clean_imag,
                    out[0],
                    out[1],
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
                speech_loss = _SCALAR_ZERO
                if vad_speech_loss_weight > 0:
                    speech_loss = _compute_speech_band_logmag_loss(
                        clean_real,
                        clean_imag,
                        out[0],
                        out[1],
                        vad_band_mask,
                        vad_band_bins,
                        gate,
                    )
                total_loss = total_loss + vad_weight * vad_loss + speech_weight * speech_loss

            if use_vad_train_reg:
                vad_reg_loss, _, _, _, _, _, _ = _compute_vad_reg_loss(
                    clean_real,
                    clean_imag,
                    noisy_real,
                    noisy_imag,
                    out[0],
                    out[1],
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

            return total_loss, out, out_wav, clean_wav

        loss_and_grad_gan = nn.value_and_grad(model, loss_fn_gan)

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
        """Run a diagnostic forward pass with detailed finite checks.

        Uses a non-fail-fast debugger so all components are checked and
        logged even when multiple contain non-finite values.
        """
        if debugger is None:
            return
        # Use a non-fail-fast copy so diagnosis completes fully instead of
        # crashing on the first non-finite intermediate value.
        from dataclasses import replace as _dc_replace

        diag_cfg = _dc_replace(debugger.config, fail_fast=False)
        diag = NumericDebugger(diag_cfg)
        tqdm.write("  [diagnose] Running non-finite diagnostic pass...")
        findings: list[str] = []

        def _diag_check(name: str, tensor: mx.array) -> None:
            if not diag.check(name, tensor, debug_ctx):
                findings.append(name)

        out = model((noisy_real, noisy_imag), feat_erb, feat_spec)
        _diag_check("model.out_real", out[0])
        _diag_check("model.out_imag", out[1])
        spec_loss = spectral_loss(out, (clean_real, clean_imag))
        _diag_check("spec_loss", spec_loss)
        if use_mrstft_loss and mrstft_loss_fn is not None and mrstft_istft is not None:
            mrstft_loss = compute_mrstft_loss(
                out,
                (clean_real, clean_imag),
                istft_fn=mrstft_istft,
                loss_fn=mrstft_loss_fn,
                n_fft=config.fft_size,
                hop_length=config.hop_size,
                target_len=mrstft_target_len,
                force_fp32=True,
            )
            _diag_check("mrstft_loss", mrstft_loss)
        if gan_active and gan_loss_fns is not None and discriminator is not None and gan_istft is not None:
            out_wav, clean_wav = specs_to_wavs(
                out,
                (clean_real, clean_imag),
                istft_fn=gan_istft,
                n_fft=config.fft_size,
                hop_length=config.hop_size,
                target_len=gan_target_len,
                force_fp32=True,
            )
            gen_loss_fn, _ = gan_loss_fns
            disc_fake, fake_feats = discriminator(out_wav)
            disc_real, real_feats = discriminator(clean_wav)
            gan_g_loss = gen_loss_fn(disc_fake)
            _diag_check("gan_g_loss", gan_g_loss)
            if feature_match_loss is not None and gan_fm_weight > 0:
                fm_loss = feature_match_loss(real_feats, fake_feats)
                _diag_check("gan_fm_loss", fm_loss)
        if use_awesome_loss:
            _compute_awesome_losses(
                noisy_real,
                noisy_imag,
                clean_real,
                clean_imag,
                out[0],
                out[1],
                snr,
                vad_band_mask,
                vad_band_bins,
                awesome_mask_sharpness,
                vad_z_threshold,
                vad_z_slope,
                vad_snr_gate_db,
                vad_snr_gate_width,
                vad_proxy_enabled,
                debug=diag,
                debug_ctx=debug_ctx,
            )
        if use_pipeline_awesome_loss:
            _compute_pipeline_awesome_losses(
                noisy_real,
                noisy_imag,
                clean_real,
                clean_imag,
                out[0],
                out[1],
                snr,
                vad_band_mask,
                vad_band_bins,
                awesome_mask_sharpness,
                vad_z_threshold,
                vad_z_slope,
                vad_snr_gate_db,
                vad_snr_gate_width,
                vad_proxy_enabled,
                debug=diag,
                debug_ctx=debug_ctx,
            )
        if use_vad_loss:
            _compute_vad_loss(
                clean_real,
                clean_imag,
                out[0],
                out[1],
                snr,
                vad_band_mask,
                vad_band_bins,
                vad_threshold,
                vad_margin,
                vad_snr_gate_db,
                vad_snr_gate_width,
                vad_z_threshold,
                vad_z_slope,
                debug=diag,
                debug_ctx=debug_ctx,
            )
            if vad_speech_loss_weight > 0:
                gate = mx.ones((clean_real.shape[0], clean_real.shape[1]))
                _compute_speech_band_logmag_loss(
                    clean_real,
                    clean_imag,
                    out[0],
                    out[1],
                    vad_band_mask,
                    vad_band_bins,
                    gate,
                    debug=diag,
                    debug_ctx=debug_ctx,
                )
        if use_vad_train_reg:
            _compute_vad_reg_loss(
                clean_real,
                clean_imag,
                noisy_real,
                noisy_imag,
                out[0],
                out[1],
                snr,
                vad_band_mask,
                vad_band_bins,
                vad_threshold,
                vad_margin,
                vad_z_threshold,
                vad_z_slope,
                vad_snr_gate_db,
                vad_snr_gate_width,
                debug=diag,
                debug_ctx=debug_ctx,
            )

        if findings:
            tqdm.write(f"  [diagnose] Non-finite in: {', '.join(findings)}")
        else:
            tqdm.write("  [diagnose] All individual components finite — NaN likely in backward pass")

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

    def run_validation(label: str = "  Validating", *, do_vad_eval: bool = False) -> float:
        """Run validation on the fixed validation split and return average loss."""
        model.eval()

        dataset.set_split("valid")
        dataset.set_epoch(0)  # Fixed epoch for reproducible validation

        if len(dataset) == 0:
            return float("inf")

        valid_loss = 0.0
        valid_spec_loss = 0.0
        valid_mrstft_loss = 0.0
        valid_vad_loss = 0.0
        valid_speech_loss = 0.0
        valid_awesome_loss = 0.0
        valid_awesome_speech = 0.0
        valid_awesome_noise = 0.0
        valid_awesome_smooth = 0.0
        valid_music_supp_loss = 0.0
        valid_mask_sat_loss = 0.0
        valid_mask_mean = 0.0
        valid_mask_high = 0.0
        valid_mask_low = 0.0
        valid_proxy_mean = 0.0
        valid_speech_ratio = 0.0
        valid_music_gate = 0.0
        valid_musicness = 0.0
        valid_mod_energy = 0.0
        valid_energy_boost = 0.0
        valid_snr_boost = 0.0
        valid_vad_reg_loss = 0.0
        valid_p_ref = 0.0
        valid_p_out = 0.0
        valid_gate_pct = 0.0
        valid_residual = 0.0
        valid_sisdr = 0.0
        bucket_metrics: dict[str, dict[str, float]] = {}
        vad_eval_p_ref = 0.0
        vad_eval_p_out = 0.0
        vad_eval_delta = 0.0
        vad_eval_batches_done = 0
        vad_eval_seconds = 0.0
        vad_eval_clips = 0
        num_valid_batches = 0
        valid_steps = len(dataset) // batch_size
        if max_valid_batches is not None:
            valid_steps = min(valid_steps, max_valid_batches)

        if use_mlx_stream:
            valid_loader = MLXDataStream(
                dataset=dataset,
                batch_size=batch_size,
                prefetch_size=max(1, prefetch_size // 2),
                num_workers=max(1, min(num_workers, 4)),
            )
            valid_loader.set_split("valid")
            valid_loader.set_epoch(0)
        else:
            valid_loader = PrefetchDataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=max(1, num_workers),
                prefetch_factor=2,
            )

        valid_tqdm_kwargs = dict(_TQDM_KWARGS)
        if _tqdm_panels:
            valid_tqdm_kwargs["position"] = tqdm_valid_position

        valid_pbar = tqdm(
            valid_loader,
            total=valid_steps,
            desc=label,
            unit="batch",
            leave=False,
            **valid_tqdm_kwargs,
        )

        sisdr_fn = None
        if eval_sisdr:
            from df_mlx.loss import si_sdr
            from df_mlx.ops import istft

            sisdr_fn = (si_sdr, istft)

        silero_istft = None
        if do_vad_eval and vad_eval_mode == "silero":
            from df_mlx.ops import istft

            silero_istft = istft

        for batch_idx, batch in enumerate(valid_pbar):
            noisy_real = batch["noisy_real"]
            noisy_imag = batch["noisy_imag"]
            clean_real = batch["clean_real"]
            clean_imag = batch["clean_imag"]
            feat_erb = batch["feat_erb"]
            feat_spec = batch["feat_spec"]
            snr = batch["snr"]
            debug_ctx = {
                "phase": "valid",
                "epoch": epoch,
                "batch": batch_idx,
                "global_step": global_step,
            }
            if debugger is not None:
                debugger.check("batch.noisy_real", noisy_real, debug_ctx)
                debugger.check("batch.noisy_imag", noisy_imag, debug_ctx)
                debugger.check("batch.clean_real", clean_real, debug_ctx)
                debugger.check("batch.clean_imag", clean_imag, debug_ctx)
                debugger.check("batch.feat_erb", feat_erb, debug_ctx)
                debugger.check("batch.feat_spec", feat_spec, debug_ctx)
                debugger.check("batch.snr", snr, debug_ctx)

            # Model expects spec as tuple (real, imag)
            noisy_spec = (noisy_real, noisy_imag)
            target_spec = (clean_real, clean_imag)

            out = model(noisy_spec, feat_erb, feat_spec)
            if debugger is not None:
                debugger.check("model.out_real", out[0], debug_ctx)
                debugger.check("model.out_imag", out[1], debug_ctx)
            spec_loss = spectral_loss(out, target_spec)
            mrstft_loss = _SCALAR_ZERO
            if use_mrstft_loss and mrstft_loss_fn is not None and mrstft_istft is not None:
                mrstft_loss = compute_mrstft_loss(
                    out,
                    target_spec,
                    istft_fn=mrstft_istft,
                    loss_fn=mrstft_loss_fn,
                    n_fft=config.fft_size,
                    hop_length=config.hop_size,
                    target_len=mrstft_target_len,
                    force_fp32=True,
                )

            awesome_loss = _SCALAR_ZERO
            awesome_speech = _SCALAR_ZERO
            awesome_noise = _SCALAR_ZERO
            awesome_smooth = _SCALAR_ZERO
            music_suppression_loss = _SCALAR_ZERO
            mask_saturation_loss = _SCALAR_ZERO
            mask = _SCALAR_ZERO
            proxy_frame = _SCALAR_ZERO
            speech_ratio = _SCALAR_ZERO
            music_gate = _SCALAR_ZERO
            musicness = _SCALAR_ZERO
            mod_energy = _SCALAR_ZERO
            energy_boost = _SCALAR_ZERO
            snr_boost = _SCALAR_ZERO

            if use_awesome_loss:
                (
                    awesome_loss,
                    awesome_speech,
                    awesome_noise,
                    awesome_smooth,
                    mask,
                    proxy_frame,
                    speech_ratio,
                    music_gate,
                    musicness,
                    mod_energy,
                    energy_boost,
                    snr_boost,
                ) = _compute_awesome_losses(
                    noisy_real,
                    noisy_imag,
                    clean_real,
                    clean_imag,
                    out[0],
                    out[1],
                    snr,
                    vad_band_mask,
                    vad_band_bins,
                    awesome_mask_sharpness,
                    vad_z_threshold,
                    vad_z_slope,
                    vad_snr_gate_db,
                    vad_snr_gate_width,
                    vad_proxy_enabled,
                    debug=debugger,
                    debug_ctx=debug_ctx,
                )

            if use_pipeline_awesome_loss:
                (
                    awesome_loss,
                    awesome_speech,
                    awesome_noise,
                    awesome_smooth,
                    music_suppression_loss,
                    mask_saturation_loss,
                    mask,
                    proxy_frame,
                    speech_ratio,
                    music_gate,
                    musicness,
                    _,  # vocal_gate
                    _,  # instrument_gate
                    mod_energy,
                    energy_boost,
                    snr_boost,
                ) = _compute_pipeline_awesome_losses(
                    noisy_real,
                    noisy_imag,
                    clean_real,
                    clean_imag,
                    out[0],
                    out[1],
                    snr,
                    vad_band_mask,
                    vad_band_bins,
                    awesome_mask_sharpness,
                    vad_z_threshold,
                    vad_z_slope,
                    vad_snr_gate_db,
                    vad_snr_gate_width,
                    vad_proxy_enabled,
                    debug=debugger,
                    debug_ctx=debug_ctx,
                )

            if use_vad_loss:
                vad_loss, p_ref, p_out, gate = _compute_vad_loss(
                    clean_real,
                    clean_imag,
                    out[0],
                    out[1],
                    snr,
                    vad_band_mask,
                    vad_band_bins,
                    vad_threshold,
                    vad_margin,
                    vad_snr_gate_db,
                    vad_snr_gate_width,
                    vad_z_threshold,
                    vad_z_slope,
                    debug=debugger,
                    debug_ctx=debug_ctx,
                )
                speech_loss = _SCALAR_ZERO
                if vad_speech_loss_weight > 0:
                    speech_loss = _compute_speech_band_logmag_loss(
                        clean_real,
                        clean_imag,
                        out[0],
                        out[1],
                        vad_band_mask,
                        vad_band_bins,
                        gate,
                        debug=debugger,
                        debug_ctx=debug_ctx,
                    )
            else:
                vad_loss = _SCALAR_ZERO
                speech_loss = _SCALAR_ZERO
                p_ref = _SCALAR_ZERO
                p_out = _SCALAR_ZERO
                gate = _SCALAR_ZERO

            vad_reg_loss = _SCALAR_ZERO
            if use_vad_train_reg:
                vad_reg_loss, _, _, _, _, _, _ = _compute_vad_reg_loss(
                    clean_real,
                    clean_imag,
                    noisy_real,
                    noisy_imag,
                    out[0],
                    out[1],
                    snr,
                    vad_band_mask,
                    vad_band_bins,
                    vad_threshold,
                    vad_margin,
                    vad_z_threshold,
                    vad_z_slope,
                    vad_snr_gate_db,
                    vad_snr_gate_width,
                    debug=debugger,
                    debug_ctx=debug_ctx,
                )

            awesome_weight_val = epoch_awesome_loss_weight
            if (use_awesome_loss or use_pipeline_awesome_loss) and awesome_warmup_steps > 0:
                awesome_weight_val = epoch_awesome_loss_weight * min(1.0, global_step / max(awesome_warmup_steps, 1))

            loss = spec_loss
            if use_mrstft_loss:
                loss = loss + mrstft_loss
            if use_awesome_loss or use_pipeline_awesome_loss:
                loss = loss + awesome_weight_val * awesome_loss
            if use_vad_loss:
                loss = loss + epoch_vad_loss_weight * vad_loss + epoch_vad_speech_loss_weight * speech_loss

            residual = mx.mean((out[0] - clean_real) ** 2 + (out[1] - clean_imag) ** 2)
            residual_by_sample = mx.mean((out[0] - clean_real) ** 2 + (out[1] - clean_imag) ** 2, axis=(1, 2))

            (
                loss_val,
                spec_loss_val,
                mrstft_loss_val,
                vad_loss_val,
                speech_loss_val,
                awesome_loss_val,
                awesome_speech_val,
                awesome_noise_val,
                awesome_smooth_val,
                music_suppression_loss_val,
                mask_saturation_loss_val,
                vad_reg_loss_val,
                residual_val,
            ) = _batch_to_float(
                loss,
                spec_loss,
                mrstft_loss,
                vad_loss,
                speech_loss,
                awesome_loss,
                awesome_speech,
                awesome_noise,
                awesome_smooth,
                music_suppression_loss,
                mask_saturation_loss,
                vad_reg_loss,
                residual,
            )

            valid_loss += loss_val
            valid_spec_loss += spec_loss_val
            valid_mrstft_loss += mrstft_loss_val
            valid_vad_loss += vad_loss_val
            valid_speech_loss += speech_loss_val
            valid_awesome_loss += awesome_loss_val
            valid_awesome_speech += awesome_speech_val
            valid_awesome_noise += awesome_noise_val
            valid_awesome_smooth += awesome_smooth_val
            valid_music_supp_loss += music_suppression_loss_val
            valid_mask_sat_loss += mask_saturation_loss_val
            valid_vad_reg_loss += vad_reg_loss_val
            valid_residual += residual_val
            num_valid_batches += 1

            if use_vad_loss:
                _p_ref_m = mx.mean(p_ref)
                _p_out_m = mx.mean(p_out)
                _gate_m = mx.mean(mx.where(gate > 0.0, 1.0, 0.0))
                _p_ref_f, _p_out_f, _gate_f = _batch_to_float(_p_ref_m, _p_out_m, _gate_m)
                valid_p_ref += _p_ref_f
                valid_p_out += _p_out_f
                valid_gate_pct += 100.0 * _gate_f

            if emit_detailed_metrics:
                snr_np = np.asarray(snr, dtype=np.float32).reshape(-1)
                residual_np = np.asarray(residual_by_sample, dtype=np.float32).reshape(-1)
                if use_vad_loss:
                    vad_delta_np = np.asarray(
                        mx.mean(mx.maximum(p_ref - p_out - vad_margin, 0.0), axis=1),
                        dtype=np.float32,
                    )
                else:
                    vad_delta_np = np.zeros_like(snr_np, dtype=np.float32)
                if use_awesome_loss or use_pipeline_awesome_loss:
                    if isinstance(musicness, mx.array):
                        musicness_np = np.asarray(musicness, dtype=np.float32).reshape(-1)
                    else:
                        musicness_np = np.zeros_like(snr_np, dtype=np.float32)
                    if musicness_np.shape[0] != snr_np.shape[0]:
                        musicness_np = np.full_like(snr_np, float(np.mean(musicness_np)), dtype=np.float32)
                else:
                    musicness_np = np.zeros_like(snr_np, dtype=np.float32)

                for i, snr_val in enumerate(snr_np):
                    bucket = _snr_bucket_name(float(snr_val))
                    metric = bucket_metrics.setdefault(
                        bucket,
                        {
                            "count": 0.0,
                            "residual_sum": 0.0,
                            "vad_delta_sum": 0.0,
                            "musicness_sum": 0.0,
                        },
                    )
                    metric["count"] += 1.0
                    metric["residual_sum"] += float(residual_np[i])
                    metric["vad_delta_sum"] += float(vad_delta_np[i])
                    metric["musicness_sum"] += float(musicness_np[i])

            if use_awesome_loss and emit_detailed_metrics:
                _mask_m = mx.mean(mask)
                _mask_hi = mx.mean(mx.where(mask > 0.8, 1.0, 0.0))
                _mask_lo = mx.mean(mx.where(mask < 0.2, 1.0, 0.0))
                _proxy_m = mx.mean(proxy_frame)
                _sr_m = mx.mean(speech_ratio)
                _mg_m = mx.mean(music_gate)
                _mu_m = mx.mean(musicness)
                _me_m = mx.mean(mod_energy)
                _eb_m = mx.mean(energy_boost)
                _sb_m = mx.mean(snr_boost)
                (
                    mask_mean,
                    mask_high,
                    mask_low,
                    proxy_mean,
                    speech_ratio_mean,
                    music_gate_mean,
                    musicness_mean,
                    mod_energy_mean,
                    energy_boost_mean,
                    snr_boost_mean,
                ) = _batch_to_float(
                    _mask_m,
                    _mask_hi,
                    _mask_lo,
                    _proxy_m,
                    _sr_m,
                    _mg_m,
                    _mu_m,
                    _me_m,
                    _eb_m,
                    _sb_m,
                )
                mask_high *= 100.0
                mask_low *= 100.0

                valid_mask_mean += mask_mean
                valid_mask_high += mask_high
                valid_mask_low += mask_low
                valid_proxy_mean += proxy_mean
                valid_speech_ratio += speech_ratio_mean
                valid_music_gate += music_gate_mean
                valid_musicness += musicness_mean
                valid_mod_energy += mod_energy_mean
                valid_energy_boost += energy_boost_mean
                valid_snr_boost += snr_boost_mean

            if do_vad_eval and vad_eval_batches_done < vad_eval_batches:
                if vad_eval_mode == "proxy":
                    p_ref_eval, p_out_eval = _compute_vad_probs(
                        clean_real.astype(mx.float32),
                        clean_imag.astype(mx.float32),
                        out[0].astype(mx.float32),
                        out[1].astype(mx.float32),
                        vad_band_mask,
                        vad_band_bins,
                        vad_z_threshold,
                        vad_z_slope,
                    )
                    p_ref_mean, p_out_mean, vad_dec = _compute_vad_eval_metrics(
                        p_ref_eval,
                        p_out_eval,
                        vad_margin,
                    )
                    vad_eval_p_ref += float(p_ref_mean)
                    vad_eval_p_out += float(p_out_mean)
                    vad_eval_delta += float(vad_dec)
                    vad_eval_batches_done += 1
                elif vad_eval_mode == "silero":
                    if silero_vad is None or silero_istft is None:
                        raise RuntimeError("Silero VAD requested but not initialized")
                    vad_start = time.perf_counter()
                    clean_wav = silero_istft(target_spec, n_fft=config.fft_size, hop_length=config.hop_size)
                    out_wav = silero_istft(out, n_fft=config.fft_size, hop_length=config.hop_size)
                    mx.eval(clean_wav, out_wav)
                    clean_np = np.asarray(clean_wav, dtype=np.float32)
                    out_np = np.asarray(out_wav, dtype=np.float32)
                    p_ref_batch = silero_vad.mean_probs(clean_np, config.sample_rate)
                    p_out_batch = silero_vad.mean_probs(out_np, config.sample_rate)
                    vad_eval_p_ref += float(np.mean(p_ref_batch))
                    vad_eval_p_out += float(np.mean(p_out_batch))
                    vad_eval_delta += float(np.mean(np.maximum(p_ref_batch - p_out_batch - vad_margin, 0.0)))
                    vad_eval_batches_done += 1
                    vad_eval_clips += int(len(p_ref_batch))
                    vad_eval_seconds += time.perf_counter() - vad_start

            if sisdr_fn is not None:
                si_sdr_fn, istft_fn = sisdr_fn
                clean_wav = istft_fn(target_spec, n_fft=config.fft_size, hop_length=config.hop_size)
                out_wav = istft_fn(out, n_fft=config.fft_size, hop_length=config.hop_size)
                sisdr_val = float(si_sdr_fn(out_wav, clean_wav))
                if math.isfinite(sisdr_val):
                    valid_sisdr += sisdr_val
                else:
                    print("⚠️  SI-SDR non-finite; skipping metric for this batch")

            valid_pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                avg=f"{valid_loss / num_valid_batches:.4f}",
            )

            if max_valid_batches is not None and (batch_idx + 1) >= max_valid_batches:
                break

        valid_pbar.close()

        if num_valid_batches > 0:
            avg_spec = valid_spec_loss / num_valid_batches
            avg_mrstft = valid_mrstft_loss / num_valid_batches
            avg_vad = valid_vad_loss / num_valid_batches
            avg_speech = valid_speech_loss / num_valid_batches
            avg_awesome = valid_awesome_loss / num_valid_batches
            avg_awesome_speech = valid_awesome_speech / num_valid_batches
            avg_awesome_noise = valid_awesome_noise / num_valid_batches
            avg_awesome_smooth = valid_awesome_smooth / num_valid_batches
            avg_music_supp = valid_music_supp_loss / num_valid_batches
            avg_mask_sat = valid_mask_sat_loss / num_valid_batches
            avg_vad_reg = valid_vad_reg_loss / num_valid_batches
            avg_residual = valid_residual / num_valid_batches
            avg_p_ref = valid_p_ref / num_valid_batches if use_vad_loss else 0.0
            avg_p_out = valid_p_out / num_valid_batches if use_vad_loss else 0.0
            avg_gate = valid_gate_pct / num_valid_batches if use_vad_loss else 0.0
            avg_sisdr = valid_sisdr / num_valid_batches if eval_sisdr else None
            use_awesome_metrics = use_awesome_loss or use_pipeline_awesome_loss
            avg_mask_mean = valid_mask_mean / num_valid_batches if use_awesome_metrics else 0.0
            avg_mask_high = valid_mask_high / num_valid_batches if use_awesome_metrics else 0.0
            avg_mask_low = valid_mask_low / num_valid_batches if use_awesome_metrics else 0.0
            avg_proxy = valid_proxy_mean / num_valid_batches if use_awesome_metrics else 0.0
            avg_speech_ratio = valid_speech_ratio / num_valid_batches if use_awesome_metrics else 0.0
            avg_music_gate = valid_music_gate / num_valid_batches if use_awesome_metrics else 0.0
            avg_musicness = valid_musicness / num_valid_batches if use_awesome_metrics else 0.0
            avg_mod = valid_mod_energy / num_valid_batches if use_awesome_metrics else 0.0
            avg_energy_boost = valid_energy_boost / num_valid_batches if use_awesome_metrics else 0.0
            avg_snr_boost = valid_snr_boost / num_valid_batches if use_awesome_metrics else 0.0
            avg_vad_eval_p_ref = (
                vad_eval_p_ref / vad_eval_batches_done if do_vad_eval and vad_eval_batches_done > 0 else 0.0
            )
            avg_vad_eval_p_out = (
                vad_eval_p_out / vad_eval_batches_done if do_vad_eval and vad_eval_batches_done > 0 else 0.0
            )
            avg_vad_eval_delta = (
                vad_eval_delta / vad_eval_batches_done if do_vad_eval and vad_eval_batches_done > 0 else 0.0
            )
            vad_eval_time = vad_eval_seconds
            vad_eval_clips_total = vad_eval_clips

            if (
                use_vad_loss
                or eval_sisdr
                or use_awesome_loss
                or use_pipeline_awesome_loss
                or use_vad_train_reg
                or do_vad_eval
                or use_mrstft_loss
            ):
                extras = [f"spec={avg_spec:.4f}", f"resid={avg_residual:.4f}"]
                if use_mrstft_loss:
                    extras.append(f"mrstft={avg_mrstft:.4f}")
                if use_vad_loss:
                    extras.extend([f"vad={avg_vad:.4f}", f"speech={avg_speech:.4f}"])
                if use_awesome_metrics:
                    extras.extend(
                        [
                            f"awesome={avg_awesome:.4f}",
                            f"aw_s={avg_awesome_speech:.4f}",
                            f"aw_n={avg_awesome_noise:.4f}",
                            f"aw_sm={avg_awesome_smooth:.4f}",
                        ]
                    )
                if use_pipeline_awesome_loss:
                    extras.extend(
                        [
                            f"mus_sup={avg_music_supp:.4f}",
                            f"mask_sat={avg_mask_sat:.4f}",
                        ]
                    )
                if use_vad_train_reg:
                    extras.append(f"vad_reg={avg_vad_reg:.4f}")
                if use_vad_loss:
                    extras.append(f"p_ref={avg_p_ref:.2f}")
                    extras.append(f"p_out={avg_p_out:.2f}")
                    extras.append(f"gate={avg_gate:.0f}%")
                if use_awesome_metrics:
                    extras.extend(
                        [
                            f"mask={avg_mask_mean:.2f}",
                            f"mask_hi={avg_mask_high:.0f}%",
                            f"mask_lo={avg_mask_low:.0f}%",
                            f"proxy={avg_proxy:.2f}",
                            f"ratio={avg_speech_ratio:.2f}",
                            f"music_gate={avg_music_gate:.2f}",
                            f"music={avg_musicness:.2f}",
                            f"mod={avg_mod:.2f}",
                            f"e_boost={avg_energy_boost:.2f}",
                            f"snr_boost={avg_snr_boost:.2f}",
                        ]
                    )
                if do_vad_eval and vad_eval_batches_done > 0:
                    extras.append(f"vad_eval_ref={avg_vad_eval_p_ref:.2f}")
                    extras.append(f"vad_eval_out={avg_vad_eval_p_out:.2f}")
                    extras.append(f"vad_eval_dec={avg_vad_eval_delta:.2f}")
                    if vad_eval_mode == "silero":
                        extras.append(f"vad_eval_s={vad_eval_time:.1f}")
                        extras.append(f"vad_eval_clips={vad_eval_clips_total}")
                if avg_sisdr is not None:
                    extras.append(f"si-sdr={avg_sisdr:.2f}dB")
                print(f"{label} metrics: " + " | ".join(extras))

            if bucket_metrics:
                bucket_parts = []
                bucket_summary: dict[str, dict[str, float]] = {}
                for bucket_name in sorted(bucket_metrics.keys()):
                    bm = bucket_metrics[bucket_name]
                    count = max(bm["count"], 1.0)
                    residual_mean = bm["residual_sum"] / count
                    vad_delta_mean = bm["vad_delta_sum"] / count
                    musicness_mean = bm["musicness_sum"] / count
                    bucket_summary[bucket_name] = {
                        "count": float(count),
                        "residual": float(residual_mean),
                        "vad_delta": float(vad_delta_mean),
                        "musicness": float(musicness_mean),
                    }
                    bucket_parts.append(
                        f"{bucket_name}:n={int(count)} resid={residual_mean:.4f} vadΔ={vad_delta_mean:.4f} mus={musicness_mean:.3f}"
                    )
                print(f"{label} buckets: " + " | ".join(bucket_parts))

                ablation_row = {
                    "epoch": int(epoch + 1),
                    "stage_index": int(active_stage_index),
                    "stage_name": active_stage_name,
                    "dynamic_loss": dynamic_loss,
                    "train_mode": train_mode,
                    "valid_loss": float(valid_loss / max(num_valid_batches, 1)),
                    "awesome": {
                        "music_suppression": float(avg_music_supp),
                        "mask_saturation": float(avg_mask_sat),
                    },
                    "buckets": bucket_summary,
                }
                ablation_path = ckpt_dir / "ablation_metrics.jsonl"
                try:
                    with open(ablation_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(ablation_row) + "\n")
                except OSError as exc:
                    tqdm.write(f"\u26a0\ufe0f  Failed to write ablation metrics: {exc}")

        return valid_loss / max(num_valid_batches, 1)

    # Base compiled-step eligibility (epoch-level mode selection may still choose eager).
    # Gradient accumulation is supported via compiled fwd+bwd with eager optimizer updates.
    base_compiled_step_enabled = not (debug_numerics or nan_skip_batch)
    compiled_disable_reasons: list[str] = []
    if debug_numerics:
        compiled_disable_reasons.append("debug_numerics")
    if nan_skip_batch:
        compiled_disable_reasons.append("nan_skip_batch")

    print(f"  Compiled-step base eligibility: {base_compiled_step_enabled}")
    if base_compiled_step_enabled:
        if gan_enabled and gan_start_epoch <= 0 and not experimental_compiled_gan:
            print("  GAN starts at epoch 1: training will run eager from the first epoch")
        elif gan_enabled and gan_start_epoch <= 0 and experimental_compiled_gan:
            print("  [EXPERIMENTAL] GAN starts at epoch 1: compiled-GAN experiment keeps compiled mode")
        elif gan_enabled and not experimental_compiled_gan:
            print(
                "  GAN delayed start: training will use compiled mode until GAN activation "
                f"(gan_start_epoch={gan_start_epoch + 1})"
            )
        elif gan_enabled and experimental_compiled_gan:
            print(
                "  [EXPERIMENTAL] GAN delayed start: compiled-GAN experiment will keep compiled "
                f"mode through GAN activation (gan_start_epoch={gan_start_epoch + 1})"
            )
    else:
        joined = ", ".join(compiled_disable_reasons) if compiled_disable_reasons else "unknown"
        print(f"  Compiled-step disabled by: {joined}")
        if experimental_compiled_gan:
            print(
                "  [EXPERIMENTAL] WARNING: compiled-GAN experiment requested but compiled mode "
                f"is globally disabled ({joined}). Experiment will not activate."
            )
    if grad_accumulation_steps > 1:
        print(
            f"  Gradient accumulation: {grad_accumulation_steps} steps (effective batch = {batch_size * grad_accumulation_steps})"
        )
        if base_compiled_step_enabled:
            print("  Gradient accumulation: compiled forward/backward enabled; optimizer updates remain accumulated")
        else:
            print("  Gradient accumulation: compiled training step disabled")
    if nan_skip_batch:
        print("  nan-skip-batch: enabled (will skip updates on non-finite loss/grads)")

    # Register SIGINT handler for graceful shutdown
    _register_sigint_handler(
        model,
        optimizer,
        ckpt_dir,
        train_config,
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
        last_completed_epoch=last_completed_epoch,
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

    global_step = resume_global_step if resume_from else start_epoch * optimizer_steps_per_epoch
    final_epoch = start_epoch
    last_completed_epoch = max(last_completed_epoch, start_epoch - 1)
    avg_train_loss = float("nan")
    last_valid_loss: float | None = None
    last_valid_epoch: int | None = None
    train_mode: Literal["COMPILED", "EAGER"] | None = None
    active_stage_name = "default"
    active_stage_index = 0
    epoch_awesome_loss_weight = base_awesome_loss_weight
    epoch_vad_loss_weight = base_vad_loss_weight
    epoch_vad_speech_loss_weight = base_vad_speech_loss_weight

    max_train_batches = train_config.get("max_train_batches")
    max_valid_batches = train_config.get("max_valid_batches")

    # Cache config-constant mx.array values outside the training loop
    _gan_disc_grad_clip_mx = mx.array(float(gan_disc_grad_clip), dtype=mx.float32)

    start_display = f"{start_epoch + 1}/{epochs} (idx {start_epoch})"
    lc_display = f"{last_completed_epoch + 1} (idx {last_completed_epoch})" if last_completed_epoch >= 0 else "none"
    print(f"Starting training at epoch {start_display} | last_completed_epoch={lc_display}")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        final_epoch = epoch

        active_stage = _resolve_pipeline_stage(epoch, pipeline_stage_defs)
        active_stage_index = int(active_stage["index"])
        active_stage_name = str(active_stage["name"])
        epoch_awesome_loss_weight = float(
            active_stage["awesome_loss_weight"]
            if active_stage["awesome_loss_weight"] is not None
            else base_awesome_loss_weight
        )
        epoch_vad_loss_weight = float(
            active_stage["vad_loss_weight"] if active_stage["vad_loss_weight"] is not None else base_vad_loss_weight
        )
        epoch_vad_speech_loss_weight = float(
            active_stage["vad_speech_loss_weight"]
            if active_stage["vad_speech_loss_weight"] is not None
            else base_vad_speech_loss_weight
        )
        train_config["pipeline_stage_active"] = {
            "index": active_stage_index,
            "name": active_stage_name,
            "start_epoch": int(active_stage["start_epoch"]),
            "awesome_loss_weight": epoch_awesome_loss_weight,
            "vad_loss_weight": epoch_vad_loss_weight,
            "vad_speech_loss_weight": epoch_vad_speech_loss_weight,
        }
        print(
            "  Stage "
            f"{active_stage_index} ({active_stage_name}) | "
            f"awesome_w={epoch_awesome_loss_weight:.4f} "
            f"vad_w={epoch_vad_loss_weight:.4f} speech_w={epoch_vad_speech_loss_weight:.4f}"
        )

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

        prev_train_mode = train_mode
        train_mode, epoch_use_compiled_step = resolve_epoch_train_mode(
            compiled_step_base_enabled=base_compiled_step_enabled,
            gan_enabled=gan_enabled,
            gan_active=gan_active,
            previous_mode=train_mode,
            experimental_compiled_gan=experimental_compiled_gan,
        )
        if not experimental_compiled_gan:
            if prev_train_mode == _TRAIN_MODE_EAGER and train_mode != _TRAIN_MODE_EAGER:
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

        if train_mode != prev_train_mode:
            if not base_compiled_step_enabled:
                mode_reason = "compiled_blocked"
            elif gan_enabled and gan_active and not experimental_compiled_gan:
                mode_reason = "gan_active"
            elif gan_enabled and gan_active and experimental_compiled_gan:
                mode_reason = "experimental_compiled_gan"
            else:
                mode_reason = "gan_inactive"
            print(f"  TRAIN_MODE={train_mode} (epoch {epoch + 1}/{epochs}, reason={mode_reason})")
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
        train_spec_loss = 0.0
        train_mrstft_loss = 0.0
        train_gan_g_loss = 0.0
        train_gan_d_loss = 0.0
        train_gan_fm_loss = 0.0
        train_gan_d_updates = 0
        train_vad_loss = 0.0
        train_speech_loss = 0.0
        train_awesome_loss = 0.0
        train_awesome_speech = 0.0
        train_awesome_noise = 0.0
        train_awesome_smooth = 0.0
        train_music_supp_loss = 0.0
        train_mask_sat_loss = 0.0
        train_vad_reg_loss = 0.0
        train_mask_mean = 0.0
        train_mask_high = 0.0
        train_mask_low = 0.0
        train_proxy_mean = 0.0
        train_speech_ratio = 0.0
        train_music_gate = 0.0
        train_musicness = 0.0
        train_mod_energy = 0.0
        train_energy_boost = 0.0
        train_snr_boost = 0.0
        train_p_ref = 0.0
        train_p_out = 0.0
        train_gate_pct = 0.0
        train_mask_logit_min = float("inf")
        train_mask_logit_max = float("-inf")
        train_mask_clip_rate = 0.0
        train_eps_clean_rate = 0.0
        train_eps_noise_rate = 0.0
        train_vad_clip_ref = 0.0
        train_vad_clip_out = 0.0
        num_debug_logs = 0
        num_vad_logs = 0
        num_awesome_logs = 0
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
            best_valid_loss,
            batch_idx=0,
            global_step=global_step,
            last_completed_epoch=last_completed_epoch,
        )

        # Timing accumulators for verbose diagnostics
        total_data_time = 0.0
        total_forward_time = 0.0  # Used for compiled step timing

        # Gradient accumulation tracking (only used when grad_accumulation_steps > 1)
        accumulated_grads: dict | None = None
        accumulated_loss = _SCALAR_ZERO
        micro_batches_in_accum = 0

        # Cached mx.array weight scalars — avoid per-batch mx.array() allocation
        # when the Python float hasn't changed.
        _prev_vad_w: float | None = None
        _prev_vad_w_mx = _SCALAR_ZERO
        _prev_speech_w: float | None = None
        _prev_speech_w_mx = _SCALAR_ZERO
        _prev_awesome_w: float | None = None
        _prev_awesome_w_mx = _SCALAR_ZERO
        _prev_vad_reg_w: float | None = None
        _prev_vad_reg_w_mx = _SCALAR_ZERO

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
                "global_step": global_step,
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
            current_lr = schedule(global_step)
            optimizer.learning_rate = current_lr

            warmup_frac = 1.0
            if use_vad_loss and vad_warmup_steps > 0:
                warmup_frac = min(1.0, global_step / max(vad_warmup_steps, 1))

            vad_weight = epoch_vad_loss_weight * warmup_frac
            speech_weight = epoch_vad_speech_loss_weight * warmup_frac
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
                awesome_frac = min(1.0, global_step / max(awesome_warmup_steps, 1))
            awesome_weight = epoch_awesome_loss_weight * awesome_frac
            if awesome_weight != _prev_awesome_w:
                _prev_awesome_w = awesome_weight
                _prev_awesome_w_mx = mx.array(awesome_weight, dtype=mx.float32)
            awesome_weight_mx = _prev_awesome_w_mx

            apply_vad_reg = False
            if use_vad_train_reg:
                if vad_train_every_steps > 0 and global_step % vad_train_every_steps == 0:
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
                                "⚠️  Non-finite grads after clipping; skipping optimizer update " f"(step={global_step})"
                            )
                        accumulated_grads = None
                        accumulated_loss = _SCALAR_ZERO
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
                        and not _compiled_gan_correctness_verified
                        and loss_and_grad_gan is not None
                    ):
                        _compiled_gan_correctness_verified = True
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
                grad_norm = float("nan")  # Not tracked in compiled path
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
                            "⚠️  Non-finite grads in eager path; skipping optimizer update " f"(step={global_step})"
                        )

                    # Reset accumulator for next window
                    accumulated_grads = None
                    accumulated_loss = _SCALAR_ZERO
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
                            f"⚠️  Non-finite loss detected (step={global_step}); " "grads were zeroed by clip_grad_norm"
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
                do_disc_update = did_optimizer_update and ((global_step % gan_disc_update_freq) == 0)
                if do_disc_update:
                    _, disc_loss_fn = gan_loss_fns

                    if pred_spec_for_logging is None:
                        pred_spec = model((noisy_real, noisy_imag), feat_erb, feat_spec)
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
                                total_loss, _, _ = disc_loss_fn(real_out, fake_out)
                                return total_loss

                            disc_loss, disc_grads = nn.value_and_grad(discriminator, disc_loss_wrapper)(discriminator)

                            if gan_disc_grad_clip > 0:
                                disc_grads, _ = clip_grad_norm(disc_grads, gan_disc_grad_clip)

                            if _tree_all_finite(disc_grads):
                                disc_optimizer.update(discriminator, disc_grads)
                            else:
                                tqdm.write(
                                    f"\u26a0\ufe0f  Non-finite disc grads; skipping disc update (step={global_step})"
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
                        f"(epoch={epoch}, batch={batch_idx}, step={global_step}, "
                        f"cumulative_nonfinite={nonfinite_loss_count})"
                    )
                    _MAX_NONFINITE_LOSSES = 50
                    if nonfinite_loss_count >= _MAX_NONFINITE_LOSSES:
                        raise FloatingPointError(
                            f"Aborting: {nonfinite_loss_count} non-finite losses "
                            f"in this session (epoch={epoch}, step={global_step}). "
                            "Model is likely diverged."
                        )
                    loss_val = 0.0
                train_loss += loss_val * epoch_eval_frequency  # Approximate accumulated loss
                if gan_active and gan_d_loss_val:
                    train_gan_d_loss += gan_d_loss_val

                # Debug mode: log per-step gradient norm for full observability
                if sync_mode == "debug" and math.isfinite(grad_norm):
                    tqdm.write(f"  [debug] step={global_step} grad_norm={grad_norm:.4f} " f"loss={loss_val:.6f}")

                # Profile mode: log step-level timing breakdown
                if sync_mode == "profile":
                    tqdm.write(
                        f"  [profile] step={global_step} "
                        f"data={data_time * 1000:.1f}ms "
                        f"fwd={fwd_time * 1000:.1f}ms "
                        f"total={(data_time + fwd_time) * 1000:.1f}ms"
                    )
            num_train_batches += 1
            samples_processed += current_batch_size
            window_samples += current_batch_size
            # Only increment global_step when optimizer actually updates
            # (for gradient accumulation > 1, updates happen every N batches)
            if did_optimizer_update:
                global_step += 1

            # Track progress for interruption-safe resume metadata
            _update_interrupt_state(
                epoch,
                loss_val,
                best_valid_loss,
                batch_idx=num_train_batches,
                global_step=global_step,
                last_completed_epoch=last_completed_epoch,
            )

            # Stop early for benchmarking if requested
            if max_train_batches is not None and num_train_batches >= max_train_batches:
                break

            # Update progress bar with real-time metrics (only on sync)
            if should_sync:
                lr = float(schedule(global_step))
                # Throughput: samples processed in this sync window / wall-clock time
                window_elapsed = time.perf_counter() - window_start
                samples_per_sec = window_samples / max(window_elapsed, 1e-6)
                window_samples = 0
                window_start = time.perf_counter()

                # Defaults for logging
                spec_loss_val = loss_val
                mrstft_loss_val = 0.0
                gan_g_loss_val = 0.0
                gan_fm_loss_val = 0.0
                vad_loss_val = 0.0
                speech_loss_val = 0.0
                p_ref_mean = 0.0
                p_out_mean = 0.0
                gate_pct = 0.0
                awesome_loss_val = 0.0
                awesome_speech_val = 0.0
                awesome_noise_val = 0.0
                awesome_smooth_val = 0.0
                mask_mean = 0.0
                mask_high = 0.0
                mask_low = 0.0
                proxy_mean = 0.0
                speech_ratio_mean = 0.0
                music_gate_mean = 0.0
                musicness_mean = 0.0
                mod_energy_mean = 0.0
                energy_boost_mean = 0.0
                snr_boost_mean = 0.0
                vad_reg_loss_val = 0.0

                # Compute model output for any metric block that needs it.
                # This must be outside the emit_detailed_metrics guard because
                # use_vad_loss / use_awesome_loss / use_pipeline_awesome_loss
                # reference out[0]/out[1] regardless of sync mode.
                # Skip entirely when the loss was non-finite — model output
                # contains NaN so all metric computations would be garbage
                # and debug checks would crash with fail_fast=True.
                needs_model_out = not _loss_was_nonfinite and (
                    use_vad_loss
                    or use_awesome_loss
                    or use_pipeline_awesome_loss
                    or use_vad_train_reg
                    or (emit_detailed_metrics and (use_mrstft_loss or gan_active))
                )
                if needs_model_out:
                    out = pred_spec_for_logging
                    if out is None:
                        out = model((noisy_real, noisy_imag), feat_erb, feat_spec)
                        out = (
                            mx.stop_gradient(out[0]),
                            mx.stop_gradient(out[1]),
                        )
                    if debugger is not None:
                        debugger.check("model.out_real", out[0], debug_ctx)
                        debugger.check("model.out_imag", out[1], debug_ctx)

                if emit_detailed_metrics and needs_model_out:
                    spec_loss = spectral_loss(out, (clean_real, clean_imag))
                    spec_loss_val = float(spec_loss)
                    train_spec_loss += spec_loss_val * epoch_eval_frequency
                    if use_mrstft_loss and mrstft_loss_fn is not None and mrstft_istft is not None:
                        mrstft_loss_val = float(
                            compute_mrstft_loss(
                                out,
                                (clean_real, clean_imag),
                                istft_fn=mrstft_istft,
                                loss_fn=mrstft_loss_fn,
                                n_fft=config.fft_size,
                                hop_length=config.hop_size,
                                target_len=mrstft_target_len,
                                force_fp32=True,
                            )
                        )
                        train_mrstft_loss += mrstft_loss_val * epoch_eval_frequency
                    if gan_active and gan_loss_fns is not None and discriminator is not None and gan_istft is not None:
                        out_wav, clean_wav = specs_to_wavs(
                            out,
                            (clean_real, clean_imag),
                            istft_fn=gan_istft,
                            n_fft=config.fft_size,
                            hop_length=config.hop_size,
                            target_len=gan_target_len,
                            force_fp32=use_mrstft_loss,
                        )
                        out_wav = _gan_waveform_view(out_wav, use_fp16=bool(use_fp16))
                        clean_wav = _gan_waveform_view(clean_wav, use_fp16=bool(use_fp16))
                        gen_loss_fn, _ = gan_loss_fns
                        disc_fake, fake_feats = discriminator(out_wav)
                        disc_real, real_feats = discriminator(clean_wav)
                        gan_g_loss_val = float(gen_loss_fn(disc_fake))
                        train_gan_g_loss += gan_g_loss_val * epoch_eval_frequency
                        if feature_match_loss is not None and gan_fm_weight > 0:
                            gan_fm_loss_val = float(feature_match_loss(real_feats, fake_feats))
                            train_gan_fm_loss += gan_fm_loss_val * epoch_eval_frequency

                if use_vad_loss and needs_model_out:
                    vad_loss, p_ref, p_out, gate = _compute_vad_loss(
                        clean_real,
                        clean_imag,
                        out[0],
                        out[1],
                        snr,
                        vad_band_mask,
                        vad_band_bins,
                        vad_threshold,
                        vad_margin,
                        vad_snr_gate_db,
                        vad_snr_gate_width,
                        vad_z_threshold,
                        vad_z_slope,
                        debug=debugger,
                        debug_ctx=debug_ctx,
                    )
                    speech_loss = _SCALAR_ZERO
                    if vad_speech_loss_weight > 0:
                        speech_loss = _compute_speech_band_logmag_loss(
                            clean_real,
                            clean_imag,
                            out[0],
                            out[1],
                            vad_band_mask,
                            vad_band_bins,
                            gate,
                            debug=debugger,
                            debug_ctx=debug_ctx,
                        )
                    _p_ref_m = mx.mean(p_ref)
                    _p_out_m = mx.mean(p_out)
                    _gate_m = mx.mean(mx.where(gate > 0.0, 1.0, 0.0))
                    (
                        vad_loss_val,
                        speech_loss_val,
                        p_ref_mean,
                        p_out_mean,
                        _gate_f,
                    ) = _batch_to_float(vad_loss, speech_loss, _p_ref_m, _p_out_m, _gate_m)
                    gate_pct = 100.0 * _gate_f

                    train_vad_loss += vad_loss_val * epoch_eval_frequency
                    train_speech_loss += speech_loss_val * epoch_eval_frequency
                    train_p_ref += p_ref_mean
                    train_p_out += p_out_mean
                    train_gate_pct += gate_pct
                    num_vad_logs += 1

                    if debug_numerics:
                        clean_power_dbg = clean_real.astype(mx.float32) ** 2 + clean_imag.astype(mx.float32) ** 2
                        out_power_dbg = out[0].astype(mx.float32) ** 2 + out[1].astype(mx.float32) ** 2
                        clean_band_dbg = mx.sum(clean_power_dbg * vad_band_mask, axis=-1) / (vad_band_bins + _EPS)
                        out_band_dbg = mx.sum(out_power_dbg * vad_band_mask, axis=-1) / (vad_band_bins + _EPS)
                        log_clean_dbg = mx.log10(clean_band_dbg + _EPS)
                        mu_dbg = mx.mean(log_clean_dbg, axis=1, keepdims=True)
                        sigma_dbg = mx.sqrt(mx.mean((log_clean_dbg - mu_dbg) ** 2, axis=1, keepdims=True) + _EPS)
                        z_ref_dbg = (log_clean_dbg - mu_dbg) / (sigma_dbg + _EPS)
                        z_out_dbg = (mx.log10(out_band_dbg + _EPS) - mu_dbg) / (sigma_dbg + _EPS)
                        clip_ref = 100.0 * float(mx.mean(mx.where(mx.abs(z_ref_dbg) > _VAD_LOGIT_CLAMP, 1.0, 0.0)))
                        clip_out = 100.0 * float(mx.mean(mx.where(mx.abs(z_out_dbg) > _VAD_LOGIT_CLAMP, 1.0, 0.0)))
                        train_vad_clip_ref += clip_ref
                        train_vad_clip_out += clip_out

                if use_awesome_loss and needs_model_out:
                    (
                        awesome_loss,
                        awesome_speech,
                        awesome_noise,
                        awesome_smooth,
                        mask,
                        proxy_frame,
                        speech_ratio,
                        music_gate,
                        musicness,
                        mod_energy,
                        energy_boost,
                        snr_boost,
                    ) = _compute_awesome_losses(
                        noisy_real,
                        noisy_imag,
                        clean_real,
                        clean_imag,
                        out[0],
                        out[1],
                        snr,
                        vad_band_mask,
                        vad_band_bins,
                        awesome_mask_sharpness,
                        vad_z_threshold,
                        vad_z_slope,
                        vad_snr_gate_db,
                        vad_snr_gate_width,
                        vad_proxy_enabled,
                        debug=debugger,
                        debug_ctx=debug_ctx,
                    )
                    _mask_m = mx.mean(mask)
                    _mask_hi = mx.mean(mx.where(mask > 0.8, 1.0, 0.0))
                    _mask_lo = mx.mean(mx.where(mask < 0.2, 1.0, 0.0))
                    _proxy_m = mx.mean(proxy_frame)
                    _sr_m = mx.mean(speech_ratio)
                    _mg_m = mx.mean(music_gate)
                    _mu_m = mx.mean(musicness)
                    _me_m = mx.mean(mod_energy)
                    _eb_m = mx.mean(energy_boost)
                    _sb_m = mx.mean(snr_boost)
                    (
                        awesome_loss_val,
                        awesome_speech_val,
                        awesome_noise_val,
                        awesome_smooth_val,
                        mask_mean,
                        mask_high,
                        mask_low,
                        proxy_mean,
                        speech_ratio_mean,
                        music_gate_mean,
                        musicness_mean,
                        mod_energy_mean,
                        energy_boost_mean,
                        snr_boost_mean,
                    ) = _batch_to_float(
                        awesome_loss,
                        awesome_speech,
                        awesome_noise,
                        awesome_smooth,
                        _mask_m,
                        _mask_hi,
                        _mask_lo,
                        _proxy_m,
                        _sr_m,
                        _mg_m,
                        _mu_m,
                        _me_m,
                        _eb_m,
                        _sb_m,
                    )
                    mask_high *= 100.0
                    mask_low *= 100.0

                    train_awesome_loss += awesome_loss_val * epoch_eval_frequency
                    train_awesome_speech += awesome_speech_val * epoch_eval_frequency
                    train_awesome_noise += awesome_noise_val * epoch_eval_frequency
                    train_awesome_smooth += awesome_smooth_val * epoch_eval_frequency
                    train_mask_mean += mask_mean
                    train_mask_high += mask_high
                    train_mask_low += mask_low
                    train_proxy_mean += proxy_mean
                    train_speech_ratio += speech_ratio_mean
                    train_music_gate += music_gate_mean
                    train_musicness += musicness_mean
                    train_mod_energy += mod_energy_mean
                    train_energy_boost += energy_boost_mean
                    train_snr_boost += snr_boost_mean
                    num_awesome_logs += 1

                    if debug_numerics:
                        clean_power_dbg = clean_real.astype(mx.float32) ** 2 + clean_imag.astype(mx.float32) ** 2
                        noise_real_dbg = noisy_real.astype(mx.float32) - clean_real.astype(mx.float32)
                        noise_imag_dbg = noisy_imag.astype(mx.float32) - clean_imag.astype(mx.float32)
                        noise_power_dbg = noise_real_dbg**2 + noise_imag_dbg**2
                        clean_band_dbg = mx.sum(clean_power_dbg * vad_band_mask, axis=-1) / (vad_band_bins + _EPS)
                        noise_band_dbg = mx.sum(noise_power_dbg * vad_band_mask, axis=-1) / (vad_band_bins + _EPS)
                        mask_logits_raw = awesome_mask_sharpness * (
                            _log1p_mag(clean_real, clean_imag) - _log1p_mag(noise_real_dbg, noise_imag_dbg)
                        )
                        mask_logit_min = float(mx.min(mask_logits_raw))
                        mask_logit_max = float(mx.max(mask_logits_raw))
                        mask_clip_rate = 100.0 * float(
                            mx.mean(mx.where(mx.abs(mask_logits_raw) > _AWESOME_MASK_LOGIT_CLAMP, 1.0, 0.0))
                        )
                        clean_eps_rate = 100.0 * float(mx.mean(mx.where(clean_band_dbg <= _EPS, 1.0, 0.0)))
                        noise_eps_rate = 100.0 * float(mx.mean(mx.where(noise_band_dbg <= _EPS, 1.0, 0.0)))
                        train_mask_logit_min = min(train_mask_logit_min, mask_logit_min)
                        train_mask_logit_max = max(train_mask_logit_max, mask_logit_max)
                        train_mask_clip_rate += mask_clip_rate
                        train_eps_clean_rate += clean_eps_rate
                        train_eps_noise_rate += noise_eps_rate
                        num_debug_logs += 1

                if use_pipeline_awesome_loss and needs_model_out:
                    (
                        awesome_loss,
                        awesome_speech,
                        awesome_noise,
                        awesome_smooth,
                        music_supp_loss,
                        mask_sat_loss,
                        mask,
                        proxy_frame,
                        speech_ratio,
                        music_gate,
                        musicness,
                        vocal_gate,
                        instrument_gate,
                        mod_energy,
                        energy_boost,
                        snr_boost,
                    ) = _compute_pipeline_awesome_losses(
                        noisy_real,
                        noisy_imag,
                        clean_real,
                        clean_imag,
                        out[0],
                        out[1],
                        snr,
                        vad_band_mask,
                        vad_band_bins,
                        awesome_mask_sharpness,
                        vad_z_threshold,
                        vad_z_slope,
                        vad_snr_gate_db,
                        vad_snr_gate_width,
                        vad_proxy_enabled,
                        debug=debugger,
                        debug_ctx=debug_ctx,
                    )
                    _mask_m = mx.mean(mask)
                    _mask_hi = mx.mean(mx.where(mask > 0.8, 1.0, 0.0))
                    _mask_lo = mx.mean(mx.where(mask < 0.2, 1.0, 0.0))
                    _proxy_m = mx.mean(proxy_frame)
                    _sr_m = mx.mean(speech_ratio)
                    _mg_m = mx.mean(music_gate)
                    _mu_m = mx.mean(musicness)
                    _me_m = mx.mean(mod_energy)
                    _eb_m = mx.mean(energy_boost)
                    _sb_m = mx.mean(snr_boost)
                    (
                        awesome_loss_val,
                        awesome_speech_val,
                        awesome_noise_val,
                        awesome_smooth_val,
                        music_supp_loss_val,
                        mask_sat_loss_val,
                        mask_mean,
                        mask_high,
                        mask_low,
                        proxy_mean,
                        speech_ratio_mean,
                        music_gate_mean,
                        musicness_mean,
                        mod_energy_mean,
                        energy_boost_mean,
                        snr_boost_mean,
                    ) = _batch_to_float(
                        awesome_loss,
                        awesome_speech,
                        awesome_noise,
                        awesome_smooth,
                        music_supp_loss,
                        mask_sat_loss,
                        _mask_m,
                        _mask_hi,
                        _mask_lo,
                        _proxy_m,
                        _sr_m,
                        _mg_m,
                        _mu_m,
                        _me_m,
                        _eb_m,
                        _sb_m,
                    )
                    mask_high *= 100.0
                    mask_low *= 100.0

                    train_awesome_loss += awesome_loss_val * epoch_eval_frequency
                    train_awesome_speech += awesome_speech_val * epoch_eval_frequency
                    train_awesome_noise += awesome_noise_val * epoch_eval_frequency
                    train_awesome_smooth += awesome_smooth_val * epoch_eval_frequency
                    train_music_supp_loss += music_supp_loss_val * epoch_eval_frequency
                    train_mask_sat_loss += mask_sat_loss_val * epoch_eval_frequency
                    train_mask_mean += mask_mean
                    train_mask_high += mask_high
                    train_mask_low += mask_low
                    train_proxy_mean += proxy_mean
                    train_speech_ratio += speech_ratio_mean
                    train_music_gate += music_gate_mean
                    train_musicness += musicness_mean
                    train_mod_energy += mod_energy_mean
                    train_energy_boost += energy_boost_mean
                    train_snr_boost += snr_boost_mean
                    num_awesome_logs += 1

                if use_vad_train_reg and apply_vad_reg and needs_model_out:
                    vad_reg_loss, vad_dec, gate, _, _, _, _ = _compute_vad_reg_loss(
                        clean_real,
                        clean_imag,
                        noisy_real,
                        noisy_imag,
                        out[0],
                        out[1],
                        snr,
                        vad_band_mask,
                        vad_band_bins,
                        vad_threshold,
                        vad_margin,
                        vad_z_threshold,
                        vad_z_slope,
                        vad_snr_gate_db,
                        vad_snr_gate_width,
                        debug=debugger,
                        debug_ctx=debug_ctx,
                    )
                    vad_reg_loss_val = float(vad_reg_loss)
                    train_vad_reg_loss += vad_reg_loss_val * epoch_eval_frequency

                if verbose:
                    train_pbar.set_postfix(
                        loss=f"{loss_val:.4f}",
                        spec=(
                            f"{spec_loss_val:.4f}"
                            if (use_vad_loss or use_awesome_loss or use_pipeline_awesome_loss or use_vad_train_reg)
                            else f"{loss_val:.4f}"
                        ),
                        mrstft=f"{mrstft_loss_val:.4f}" if use_mrstft_loss else "0.0000",
                        gan_g=f"{gan_g_loss_val:.4f}" if gan_active else "0.0000",
                        gan_d=f"{gan_d_loss_val:.4f}" if gan_active else "0.0000",
                        fm=f"{gan_fm_loss_val:.4f}" if gan_active else "0.0000",
                        vad=f"{vad_loss_val:.4f}" if use_vad_loss else "0.0000",
                        speech=f"{speech_loss_val:.4f}" if use_vad_loss else "0.0000",
                        awesome=(
                            f"{awesome_loss_val:.4f}" if (use_awesome_loss or use_pipeline_awesome_loss) else "0.0000"
                        ),
                        mask=(f"{mask_mean:.2f}" if (use_awesome_loss or use_pipeline_awesome_loss) else "0.00"),
                        lr=f"{lr:.1e}",
                        data=f"{data_time * 1000:.0f}ms",
                        fwd=f"{fwd_time * 1000:.0f}ms",
                        spd=f"{samples_per_sec:.0f}/s",
                        gstep=global_step,
                    )
                else:
                    grad_display = f"{grad_norm:.2f}" if math.isfinite(grad_norm) else "n/a"
                    train_pbar.set_postfix(
                        loss=f"{loss_val:.4f}",
                        avg=f"{train_loss / num_train_batches:.4f}",
                        gan_g=f"{gan_g_loss_val:.4f}" if gan_active else "0.0000",
                        gan_d=f"{gan_d_loss_val:.4f}" if gan_active else "0.0000",
                        fm=f"{gan_fm_loss_val:.4f}" if gan_active else "0.0000",
                        vad=f"{vad_loss_val:.4f}" if use_vad_loss else "0.0000",
                        speech=f"{speech_loss_val:.4f}" if use_vad_loss else "0.0000",
                        awesome=(
                            f"{awesome_loss_val:.4f}" if (use_awesome_loss or use_pipeline_awesome_loss) else "0.0000"
                        ),
                        mask=(f"{mask_mean:.2f}" if (use_awesome_loss or use_pipeline_awesome_loss) else "0.00"),
                        p_ref=f"{p_ref_mean:.2f}" if use_vad_loss else "0.00",
                        p_out=f"{p_out_mean:.2f}" if use_vad_loss else "0.00",
                        gate=f"{gate_pct:.0f}%" if use_vad_loss else "0%",
                        vad_reg=f"{vad_reg_loss_val:.4f}" if use_vad_train_reg else "0.0000",
                        lr=f"{lr:.1e}",
                        grad=grad_display,
                        spd=f"{samples_per_sec:.0f}/s",
                        gstep=global_step,
                    )

            # Save data checkpoint periodically (for resume capability)
            if checkpoint_batches > 0 and use_mlx_stream and train_stream is not None:
                if (batch_idx + 1) % checkpoint_batches == 0:
                    train_stream.save_checkpoint(data_checkpoint_path)

            # Save model checkpoint by steps (HuggingFace-style)
            if save_strategy == "steps" and save_steps > 0 and global_step % save_steps == 0:
                # Force sync before checkpoint to get accurate loss
                mx.eval(state)
                loss_val = float(loss)

                ckpt_path = ckpt_dir / f"step_{global_step:06d}.safetensors"
                step_saved = save_checkpoint(
                    model,
                    ckpt_path,
                    epoch=epoch,
                    batch_idx=num_train_batches,
                    global_step=global_step,
                    loss=train_loss / num_train_batches if num_train_batches > 0 else loss_val,
                    best_valid_loss=best_valid_loss,
                    config=train_config,
                    optimizer=optimizer,
                    discriminator=discriminator,
                    disc_optimizer=disc_optimizer,
                    last_completed_epoch=last_completed_epoch,
                    kind="step",
                )
                if step_saved:
                    tqdm.write(f"  📦 Checkpoint saved: {ckpt_path.name} (step {global_step})")
                else:
                    tqdm.write(f"  ⚠️  Checkpoint save failed: {ckpt_path.name} (step {global_step})")

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
            train_stream.save_checkpoint(data_checkpoint_path)

        avg_train_loss = train_loss / max(num_train_batches, 1)
        avg_train_spec_loss = train_spec_loss / max(num_train_batches, 1)
        avg_train_mrstft_loss = train_mrstft_loss / max(num_train_batches, 1)
        avg_train_gan_g_loss = train_gan_g_loss / max(num_train_batches, 1)
        avg_train_gan_fm_loss = train_gan_fm_loss / max(num_train_batches, 1)
        avg_train_gan_d_loss = train_gan_d_loss / max(train_gan_d_updates, 1)
        avg_train_vad_loss = train_vad_loss / max(num_train_batches, 1)
        avg_train_speech_loss = train_speech_loss / max(num_train_batches, 1)
        avg_train_awesome_loss = train_awesome_loss / max(num_train_batches, 1)
        avg_train_awesome_speech = train_awesome_speech / max(num_train_batches, 1)
        avg_train_awesome_noise = train_awesome_noise / max(num_train_batches, 1)
        avg_train_awesome_smooth = train_awesome_smooth / max(num_train_batches, 1)
        avg_train_music_supp = train_music_supp_loss / max(num_train_batches, 1)
        avg_train_mask_sat = train_mask_sat_loss / max(num_train_batches, 1)
        avg_train_vad_reg_loss = train_vad_reg_loss / max(num_train_batches, 1)
        avg_train_p_ref = train_p_ref / max(num_vad_logs, 1)
        avg_train_p_out = train_p_out / max(num_vad_logs, 1)
        avg_train_gate = train_gate_pct / max(num_vad_logs, 1)
        avg_train_mask_mean = train_mask_mean / max(num_awesome_logs, 1)
        avg_train_mask_high = train_mask_high / max(num_awesome_logs, 1)
        avg_train_mask_low = train_mask_low / max(num_awesome_logs, 1)
        avg_train_proxy = train_proxy_mean / max(num_awesome_logs, 1)
        avg_train_speech_ratio = train_speech_ratio / max(num_awesome_logs, 1)
        avg_train_music_gate = train_music_gate / max(num_awesome_logs, 1)
        avg_train_musicness = train_musicness / max(num_awesome_logs, 1)
        avg_train_mod = train_mod_energy / max(num_awesome_logs, 1)
        avg_train_energy_boost = train_energy_boost / max(num_awesome_logs, 1)
        avg_train_snr_boost = train_snr_boost / max(num_awesome_logs, 1)

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
            avg_valid_loss = run_validation("  Validating", do_vad_eval=do_vad_eval)
            last_valid_loss = avg_valid_loss
            last_valid_epoch = epoch

            # Early stopping check
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                epochs_without_improvement = 0

                # Save best model
                best_path = ckpt_dir / "best.safetensors"
                best_saved = save_checkpoint(
                    model,
                    best_path,
                    epoch=epoch,
                    batch_idx=None,
                    global_step=global_step,
                    loss=avg_train_loss,
                    best_valid_loss=best_valid_loss,
                    config=train_config,
                    optimizer=optimizer,
                    discriminator=discriminator,
                    disc_optimizer=disc_optimizer,
                    last_completed_epoch=epoch,
                    kind="best",
                )
                if best_saved:
                    last_completed_epoch = max(last_completed_epoch, epoch)
                    _update_interrupt_state(
                        epoch,
                        avg_train_loss,
                        best_valid_loss,
                        batch_idx=num_train_batches,
                        global_step=global_step,
                        last_completed_epoch=last_completed_epoch,
                    )
                else:
                    print("⚠️  Best checkpoint save failed; epoch completion not updated.")
            else:
                epochs_without_improvement += 1

        # ====== Epoch Summary ======
        epoch_time = time.perf_counter() - epoch_start
        epoch_throughput = samples_processed / epoch_time if epoch_time > 0 else 0

        # Update interrupt state with final epoch metrics
        _update_interrupt_state(
            epoch,
            avg_train_loss,
            best_valid_loss,
            batch_idx=num_train_batches,
            global_step=global_step,
            last_completed_epoch=last_completed_epoch,
        )

        # Improved epoch summary with throughput
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
            loss_parts = [f"Spec: {avg_train_spec_loss:.4f}"]
            if use_mrstft_loss:
                loss_parts.append(f"MRSTFT: {avg_train_mrstft_loss:.4f}")
            if gan_enabled:
                loss_parts.append(f"GAN_G: {avg_train_gan_g_loss:.4f}")
                loss_parts.append(f"GAN_D: {avg_train_gan_d_loss:.4f}")
                if gan_fm_weight > 0:
                    loss_parts.append(f"FM: {avg_train_gan_fm_loss:.4f}")
            if use_vad_loss:
                loss_parts.extend(
                    [
                        f"VAD: {avg_train_vad_loss:.4f}",
                        f"Speech: {avg_train_speech_loss:.4f}",
                    ]
                )
            if use_awesome_loss or use_pipeline_awesome_loss:
                loss_parts.extend(
                    [
                        f"Awesome: {avg_train_awesome_loss:.4f}",
                        f"AwS: {avg_train_awesome_speech:.4f}",
                        f"AwN: {avg_train_awesome_noise:.4f}",
                        f"AwSm: {avg_train_awesome_smooth:.4f}",
                    ]
                )
            if use_pipeline_awesome_loss:
                loss_parts.extend(
                    [
                        f"MusSup: {avg_train_music_supp:.4f}",
                        f"MaskSat: {avg_train_mask_sat:.4f}",
                    ]
                )
            if use_vad_train_reg:
                loss_parts.append(f"VADreg: {avg_train_vad_reg_loss:.4f}")
            loss_summary = " | " + " | ".join(loss_parts)

        print(
            f"✓ Epoch {epoch + 1}/{epochs} complete | "
            f"Train: {avg_train_loss:.4f}{loss_summary} | "
            f"Valid: {avg_valid_loss:.4f} {improvement_marker}| "
            f"Best: {best_valid_loss:.4f} | "
            f"{samples_processed:,} samples @ {epoch_throughput:.0f}/s | "
            f"{epoch_time:.1f}s"
        )

        if use_vad_loss and verbose:
            print(
                f"  VAD stats: p_ref={avg_train_p_ref:.2f} | "
                f"p_out={avg_train_p_out:.2f} | gate={avg_train_gate:.0f}%"
            )
        if (use_awesome_loss or use_pipeline_awesome_loss) and verbose:
            print(
                "  Awesome stats: "
                f"mask={avg_train_mask_mean:.2f} (hi {avg_train_mask_high:.0f}%, lo {avg_train_mask_low:.0f}%) | "
                f"proxy={avg_train_proxy:.2f} ratio={avg_train_speech_ratio:.2f} | "
                f"music_gate={avg_train_music_gate:.2f} music={avg_train_musicness:.2f} | "
                f"mod={avg_train_mod:.2f} e_boost={avg_train_energy_boost:.2f} snr_boost={avg_train_snr_boost:.2f}"
            )
        if debug_numerics:
            parts = []
            if (use_awesome_loss or use_pipeline_awesome_loss) and num_debug_logs > 0:
                avg_mask_clip = train_mask_clip_rate / num_debug_logs
                avg_eps_clean = train_eps_clean_rate / num_debug_logs
                avg_eps_noise = train_eps_noise_rate / num_debug_logs
                parts.append(
                    f"mask_logit=[{train_mask_logit_min:.1f},{train_mask_logit_max:.1f}] "
                    f"clip={avg_mask_clip:.1f}% eps_clean={avg_eps_clean:.1f}% eps_noise={avg_eps_noise:.1f}%"
                )
            if use_vad_loss and num_vad_logs > 0:
                avg_vad_clip_ref = train_vad_clip_ref / num_vad_logs
                avg_vad_clip_out = train_vad_clip_out / num_vad_logs
                parts.append(f"vad_clip_ref={avg_vad_clip_ref:.1f}% vad_clip_out={avg_vad_clip_out:.1f}%")
            if parts:
                print("  Debug numerics: " + " | ".join(parts))

        # ====== End-of-Epoch Checkpointing (authoritative completion) ======
        ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.safetensors"
        epoch_saved = save_checkpoint(
            model,
            ckpt_path,
            epoch=epoch,
            batch_idx=None,
            global_step=global_step,
            loss=avg_train_loss,
            best_valid_loss=best_valid_loss,
            config=train_config,
            optimizer=optimizer,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
            last_completed_epoch=epoch,
            kind="epoch_end",
        )
        epoch_completed = epoch_saved or best_saved
        if epoch_saved:
            last_completed_epoch = epoch
            _update_interrupt_state(
                epoch,
                avg_train_loss,
                best_valid_loss,
                batch_idx=num_train_batches,
                global_step=global_step,
                last_completed_epoch=last_completed_epoch,
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

        # ====== Early Stopping ======
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break

        # Clear memory periodically
        if (epoch + 1) % 10 == 0:
            gc.collect()

    # Final validation to compare against best checkpoint.
    final_valid_loss = float("inf")
    if last_valid_epoch == final_epoch and last_valid_loss is not None:
        final_valid_loss = last_valid_loss
    else:
        final_valid_loss = run_validation("  Final validation", do_vad_eval=vad_eval_enabled)
        last_valid_loss = final_valid_loss
        last_valid_epoch = final_epoch

    if final_valid_loss < best_valid_loss:
        best_valid_loss = final_valid_loss
        best_path = ckpt_dir / "best.safetensors"
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
            kind="best_final",
        )
        if best_final_saved:
            print(f"  ✅ Final weights set new best: {best_valid_loss:.4f}")
        else:
            print("  ⚠️  Failed to save final best checkpoint.")

    # Save final weights (even if not aligned to checkpoint interval).
    mx.eval(state)
    final_path = ckpt_dir / "final.safetensors"
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
    print(f"Best checkpoint: {ckpt_dir / 'best.safetensors'}")
    print(f"Checkpoints:     {ckpt_dir}")

    if tqdm_setup_panel is not None:
        tqdm_setup_panel.close()


# main() is now in df_mlx.training_cli_main (re-exported above).


if __name__ == "__main__":
    main()
