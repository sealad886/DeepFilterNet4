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

import argparse
import gc
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Tuple, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from df_mlx.grad_utils import clip_grad_norm_tree  # noqa: E402
from df_mlx.run_config import (  # noqa: E402
    RunConfig,
    SyncMode,
    generate_run_config_example,
    load_preset_config,
    load_run_config,
    set_by_path,
    validate_run_config,
)
from df_mlx.train_dynamic_config import apply_train_ini_config, apply_train_ini_tables  # noqa: E402

if TYPE_CHECKING:
    from df_mlx.config import ModelParams4
    from df_mlx.run_config import MultiResSpecLossConfig

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

# =============================================================================
# VAD-based speech preservation helpers
# =============================================================================

_EPS = 1e-8

# =============================================================================
# Awesome loss (speech-preserving contrastive) + proxy VAD constants
# =============================================================================
_AWESOME_PROXY_RATIO_FLOOR = 0.3
_AWESOME_PROXY_RATIO_SCALE = 0.7
_AWESOME_LOW_ENERGY_WEIGHT = 0.7
_AWESOME_LOW_SNR_WEIGHT = 0.7
_AWESOME_MOD_THRESHOLD = 0.25
_AWESOME_MOD_WIDTH = 0.15
_AWESOME_ENERGY_BOOST_DB = -3.5
_AWESOME_ENERGY_BOOST_WIDTH = 1.5
_AWESOME_SMOOTH_WEIGHT = 0.2
_AWESOME_MUSICNESS_THR = 0.55
_AWESOME_MUSICNESS_WIDTH = 0.15
_AWESOME_MUSIC_FLUX_THR = 0.08
_AWESOME_MUSIC_FLUX_WIDTH = 0.05
_AWESOME_MASK_LOGIT_CLAMP = 30.0
_VAD_LOGIT_CLAMP = 20.0

# =============================================================================
# Pipeline Awesome loss constants (improved speech preservation + music suppression)
# =============================================================================
_PIPELINE_MIN_MASK_FLOOR = 0.08  # Prevent complete suppression
_PIPELINE_LOW_ENERGY_ADDITIVE = 0.25  # Additive boost for quiet speech
_PIPELINE_LOW_SNR_ADDITIVE = 0.25  # Additive boost for low-SNR
_PIPELINE_PROXY_FLOOR = 0.15  # Higher minimum proxy weight
_PIPELINE_SPEECH_BAND_WEIGHT = 2.0  # Extra weight on speech band (300-3400 Hz)
_PIPELINE_MUSIC_SUPPRESSION_WEIGHT = 1.5  # Music suppression strength
_PIPELINE_VOCAL_HARMONIC_THR = 0.4  # Threshold for vocal harmonic detection
_PIPELINE_PITCH_STABILITY_THR = 0.3  # Threshold for pitch stability (vocals)
_PIPELINE_ARTIFACT_SMOOTH_WEIGHT = 0.3  # Temporal smoothing for artifact control
_PIPELINE_MASK_SATURATION_PENALTY = 0.1  # Penalty for extreme mask values


def _batch_to_float(*arrays: mx.array) -> tuple[float, ...]:
    """Evaluate multiple MLX arrays in one sync, then extract Python floats.

    Reduces N individual ``float(mx_array)`` sync barriers to a single ``mx.eval()``.
    """
    mx.eval(*arrays)
    return tuple(float(a) for a in arrays)


@dataclass
class NumericDebugConfig:
    enabled: bool = False
    fail_fast: bool = True
    skip_batch: bool = False
    every: int = 1
    dump_dir: Path | None = None
    dump_arrays: bool = False
    max_dumps: int = 5
    check_grads: bool = True


class NumericDebugger:
    """Helper for fail-fast finite checks and debug dumps."""

    def __init__(self, config: NumericDebugConfig):
        self.config = config
        self.dump_count = 0

    def _should_check(self, ctx: dict[str, Any] | None) -> bool:
        if not self.config.enabled:
            return False
        if ctx is None:
            return True
        step = ctx.get("global_step")
        if isinstance(step, int):
            return (step % max(self.config.every, 1)) == 0
        return True

    def _dump_stats(self, name: str, tensor: mx.array, ctx: dict[str, Any] | None) -> None:
        if self.config.dump_dir is None:
            return
        if self.dump_count >= self.config.max_dumps:
            return
        self.config.dump_dir.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(tensor, dtype=np.float32)
        finite_mask = np.isfinite(arr)
        finite_vals = arr[finite_mask]
        if finite_vals.size > 0:
            stats = {
                "min": float(finite_vals.min()),
                "max": float(finite_vals.max()),
                "mean": float(finite_vals.mean()),
            }
        else:
            stats = {"min": None, "max": None, "mean": None}
        dump = {
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite_pct": float(100.0 * finite_mask.mean()),
            "nonfinite_count": int(arr.size - finite_mask.sum()),
            "stats": stats,
            "context": ctx or {},
        }
        out_path = self.config.dump_dir / f"nonfinite_{self.dump_count:03d}_{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)
        if self.config.dump_arrays:
            slices = tuple(slice(0, min(dim, 8)) for dim in arr.shape)
            sample = arr[slices]
            np.savez_compressed(
                self.config.dump_dir / f"nonfinite_{self.dump_count:03d}_{name}.npz",
                sample=sample,
            )
        self.dump_count += 1

    def check(self, name: str, tensor: mx.array, ctx: dict[str, Any] | None = None) -> bool:
        if not self._should_check(ctx):
            return True
        is_finite = mx.isfinite(tensor)
        if bool(mx.all(is_finite)):
            return True
        self._dump_stats(name, tensor, ctx)
        message = f"Non-finite detected in {name}"
        if ctx:
            message += f" | ctx={ctx}"
        if self.config.fail_fast:
            raise FloatingPointError(message)
        return False

    def check_tree(self, name: str, tree: Any, ctx: dict[str, Any] | None = None) -> bool:
        """Check gradient tree for non-finite values.

        Never raises even when fail_fast is set — gradient non-finiteness is
        handled by skipping the optimizer update, not by crashing.  Dumps are
        still written for post-mortem analysis.
        """
        if not self._should_check(ctx) or not self.config.check_grads:
            return True
        from mlx.utils import tree_flatten

        all_finite = True
        for key, value in tree_flatten(tree):
            if value is None:
                continue
            if not bool(mx.all(mx.isfinite(value))):
                key_name = f"{name}.{key}"
                self._dump_stats(key_name, value, ctx)
                all_finite = False
        if not all_finite:
            from tqdm import tqdm

            tqdm.write(f"⚠️  Non-finite gradients in {name} " f"(ctx={ctx}) — skipping optimizer update")
        return all_finite


def _tree_all_finite(tree: Any) -> bool:
    """Fast tree-wide finite check (no dumps)."""
    from mlx.utils import tree_flatten

    for _, value in tree_flatten(tree):
        if value is None:
            continue
        if not bool(mx.all(mx.isfinite(value))):
            return False
    return True


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


def _flag_in_argv(flags: list[str], argv: list[str]) -> bool:
    for arg in argv:
        for flag in flags:
            if arg == flag or arg.startswith(f"{flag}="):
                return True
    return False


def _parse_pipeline_stages_cli(raw: str | None) -> list[dict[str, Any]]:
    """Parse --pipeline-stages JSON string into a normalized stage list."""
    if raw is None or raw.strip() == "":
        return []

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--pipeline-stages must be valid JSON") from exc

    if not isinstance(value, list):
        raise ValueError("--pipeline-stages must be a JSON array of stage objects")

    normalized: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    for i, stage in enumerate(value):
        if not isinstance(stage, dict):
            raise ValueError(f"pipeline stage at index {i} must be an object")
        if "start_epoch" not in stage:
            raise ValueError(f"pipeline stage at index {i} is missing required key 'start_epoch'")

        start_epoch = int(stage["start_epoch"])
        if start_epoch < 0:
            raise ValueError("pipeline stage start_epoch must be >= 0")
        if start_epoch in seen_epochs:
            raise ValueError(f"duplicate pipeline stage start_epoch={start_epoch}")
        seen_epochs.add(start_epoch)

        item: dict[str, Any] = {"start_epoch": start_epoch}
        if "name" in stage and stage["name"] is not None:
            item["name"] = str(stage["name"])

        for key in ("awesome_loss_weight", "vad_loss_weight", "vad_speech_loss_weight"):
            if key in stage and stage[key] is not None:
                val = float(stage[key])
                if val < 0.0:
                    raise ValueError(f"pipeline stage {key} must be >= 0")
                item[key] = val

        normalized.append(item)

    normalized.sort(key=lambda x: int(x["start_epoch"]))
    return normalized


def _resolve_pipeline_stage(epoch: int, stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Return active stage metadata for the provided epoch."""
    if not stages:
        return {
            "index": 0,
            "name": "default",
            "start_epoch": 0,
            "awesome_loss_weight": None,
            "vad_loss_weight": None,
            "vad_speech_loss_weight": None,
        }

    active_idx = 0
    for i, stage in enumerate(stages):
        if epoch >= int(stage["start_epoch"]):
            active_idx = i
        else:
            break

    active = stages[active_idx]
    return {
        "index": active_idx,
        "name": str(active.get("name", f"stage_{active_idx}")),
        "start_epoch": int(active["start_epoch"]),
        "awesome_loss_weight": active.get("awesome_loss_weight"),
        "vad_loss_weight": active.get("vad_loss_weight"),
        "vad_speech_loss_weight": active.get("vad_speech_loss_weight"),
    }


def _apply_cli_overrides(cfg: RunConfig, args: argparse.Namespace, argv: list[str]) -> None:
    overrides: list[tuple[list[str], str, Any]] = [
        (["--cache-dir"], "dataset.cache_dir", getattr(args, "cache_dir", None)),
        (["--speech-list"], "dataset.speech_list", getattr(args, "speech_list", None)),
        (["--noise-list"], "dataset.noise_list", getattr(args, "noise_list", None)),
        (["--rir-list"], "dataset.rir_list", getattr(args, "rir_list", None)),
        (["--config"], "dataset.config", getattr(args, "config", None)),
        (["--train-config"], "training.train_config", getattr(args, "train_config", None)),
        (["--snr-range"], "dataset.snr_range", getattr(args, "snr_range", None)),
        (
            ["--snr-range-extreme"],
            "dataset.snr_range_extreme",
            getattr(args, "snr_range_extreme", None),
        ),
        (
            ["--snr-range-very-low"],
            "dataset.snr_range_very_low",
            getattr(args, "snr_range_very_low", None),
        ),
        (["--p-extreme-snr"], "dataset.p_extreme_snr", getattr(args, "p_extreme_snr", None)),
        (["--p-very-low-snr"], "dataset.p_very_low_snr", getattr(args, "p_very_low_snr", None)),
        (
            ["--p-interfer-speech"],
            "dataset.p_interfer_speech",
            getattr(args, "p_interfer_speech", None),
        ),
        (
            ["--curriculum-warmup-epochs"],
            "training.curriculum_warmup_epochs",
            getattr(args, "curriculum_warmup_epochs", None),
        ),
        (
            ["--speech-gain-range"],
            "dataset.speech_gain_range",
            getattr(args, "speech_gain_range", None),
        ),
        (
            ["--noise-gain-range"],
            "dataset.noise_gain_range",
            getattr(args, "noise_gain_range", None),
        ),
        (["--p-reverb"], "augmentation.p_reverb", getattr(args, "p_reverb", None)),
        (["--p-clipping"], "augmentation.p_clipping", getattr(args, "p_clipping", None)),
        (["--epochs"], "training.epochs", getattr(args, "epochs", None)),
        (["--batch-size"], "training.batch_size", getattr(args, "batch_size", None)),
        (["--learning-rate"], "training.learning_rate", getattr(args, "learning_rate", None)),
        (
            ["--learning-rate-min"],
            "training.learning_rate_min",
            getattr(args, "learning_rate_min", None),
        ),
        (["--weight-decay"], "training.weight_decay", getattr(args, "weight_decay", None)),
        (["--warmup-epochs"], "training.warmup_epochs", getattr(args, "warmup_epochs", None)),
        (["--patience"], "training.patience", getattr(args, "patience", None)),
        (
            ["--grad-accumulation-steps"],
            "training.grad_accumulation_steps",
            getattr(args, "grad_accumulation_steps", None),
        ),
        (["--max-grad-norm"], "training.max_grad_norm", getattr(args, "max_grad_norm", None)),
        (["--eval-frequency"], "training.eval_frequency", getattr(args, "eval_frequency", None)),
        (["--seed"], "training.seed", getattr(args, "seed", None)),
        (["--num-workers"], "dataloader.num_workers", getattr(args, "num_workers", None)),
        (["--prefetch-size"], "dataloader.prefetch_size", getattr(args, "prefetch_size", None)),
        (
            ["--max-train-batches"],
            "dataloader.max_train_batches",
            getattr(args, "max_train_batches", None),
        ),
        (
            ["--max-valid-batches"],
            "dataloader.max_valid_batches",
            getattr(args, "max_valid_batches", None),
        ),
        (["--checkpoint-dir"], "checkpoint.checkpoint_dir", getattr(args, "checkpoint_dir", None)),
        (["--save-strategy"], "checkpoint.save_strategy", getattr(args, "save_strategy", None)),
        (["--save-steps"], "checkpoint.save_steps", getattr(args, "save_steps", None)),
        (
            ["--save-total-limit"],
            "checkpoint.save_total_limit",
            getattr(args, "save_total_limit", None),
        ),
        (
            ["--checkpoint-batches"],
            "checkpoint.checkpoint_batches",
            getattr(args, "checkpoint_batches", None),
        ),
        (["--validate-every"], "checkpoint.validate_every", getattr(args, "validate_every", None)),
        (["--resume"], "checkpoint.resume", getattr(args, "resume", None)),
        (["--resume-data"], "checkpoint.resume_data", getattr(args, "resume_data", None)),
        (["--check-chkpts"], "checkpoint.check_chkpts", getattr(args, "check_chkpts", None)),
        (
            ["--backbone", "--backbone-type"],
            "model.backbone_type",
            getattr(args, "backbone_type", None),
        ),
        (["--model-variant"], "model.variant", getattr(args, "model_variant", None)),
        (["--dynamic-loss"], "loss.dynamic_loss", getattr(args, "dynamic_loss", None)),
        (
            ["--awesome-loss-weight"],
            "loss.awesome.loss_weight",
            getattr(args, "awesome_loss_weight", None),
        ),
        (
            ["--awesome-mask-sharpness"],
            "loss.awesome.mask_sharpness",
            getattr(args, "awesome_mask_sharpness", None),
        ),
        (
            ["--awesome-warmup-steps"],
            "loss.awesome.warmup_steps",
            getattr(args, "awesome_warmup_steps", None),
        ),
        (["--mrstft-factor"], "loss.mrstft.factor", getattr(args, "mrstft_factor", None)),
        (["--mrstft-gamma"], "loss.mrstft.gamma", getattr(args, "mrstft_gamma", None)),
        (["--mrstft-f-complex"], "loss.mrstft.f_complex", getattr(args, "mrstft_f_complex", None)),
        (["--mrstft-fft-sizes"], "loss.mrstft.fft_sizes", getattr(args, "mrstft_fft_sizes", None)),
        (["--mrstft-hop-sizes"], "loss.mrstft.hop_sizes", getattr(args, "mrstft_hop_sizes", None)),
        (["--gan-enabled"], "gan.enabled", getattr(args, "gan_enabled", None)),
        (["--gan-start-epoch"], "gan.start_epoch", getattr(args, "gan_start_epoch", None)),
        (["--gan-ramp-epochs"], "gan.ramp_epochs", getattr(args, "gan_ramp_epochs", None)),
        (["--gan-adv-weight"], "gan.adv_weight", getattr(args, "gan_adv_weight", None)),
        (["--gan-fm-weight"], "gan.fm_weight", getattr(args, "gan_fm_weight", None)),
        (["--gan-discriminator"], "gan.discriminator", getattr(args, "gan_discriminator", None)),
        (["--gan-mpd-periods"], "gan.mpd_periods", getattr(args, "gan_mpd_periods", None)),
        (["--gan-msd-scales"], "gan.msd_scales", getattr(args, "gan_msd_scales", None)),
        (["--gan-disc-lr"], "gan.disc_lr", getattr(args, "gan_disc_lr", None)),
        (
            ["--gan-disc-weight-decay"],
            "gan.disc_weight_decay",
            getattr(args, "gan_disc_weight_decay", None),
        ),
        (["--gan-disc-grad-clip"], "gan.disc_grad_clip", getattr(args, "gan_disc_grad_clip", None)),
        (
            ["--gan-disc-update-freq"],
            "gan.disc_update_freq",
            getattr(args, "gan_disc_update_freq", None),
        ),
        (["--vad-loss-weight"], "vad.loss_weight", getattr(args, "vad_loss_weight", None)),
        (["--vad-threshold"], "vad.threshold", getattr(args, "vad_threshold", None)),
        (["--vad-margin"], "vad.margin", getattr(args, "vad_margin", None)),
        (
            ["--vad-speech-loss-weight"],
            "vad.speech_loss_weight",
            getattr(args, "vad_speech_loss_weight", None),
        ),
        (["--vad-warmup-epochs"], "vad.warmup_epochs", getattr(args, "vad_warmup_epochs", None)),
        (["--vad-snr-gate"], "vad.snr_gate_db", getattr(args, "vad_snr_gate", None)),
        (["--vad-snr-gate-width"], "vad.snr_gate_width", getattr(args, "vad_snr_gate_width", None)),
        (["--vad-band-low"], "vad.band_low_hz", getattr(args, "vad_band_low", None)),
        (["--vad-band-high"], "vad.band_high_hz", getattr(args, "vad_band_high", None)),
        (["--vad-z-threshold"], "vad.z_threshold", getattr(args, "vad_z_threshold", None)),
        (["--vad-z-slope"], "vad.z_slope", getattr(args, "vad_z_slope", None)),
        (["--vad-eval-mode"], "vad.eval.mode", getattr(args, "vad_eval_mode", None)),
        (["--vad-eval-every"], "vad.eval.every", getattr(args, "vad_eval_every", None)),
        (["--vad-eval-batches"], "vad.eval.batches", getattr(args, "vad_eval_batches", None)),
        (
            ["--vad-eval-max-seconds"],
            "vad.eval.max_seconds",
            getattr(args, "vad_eval_max_seconds", None),
        ),
        (
            ["--vad-silero-model-path"],
            "vad.eval.silero_model_path",
            getattr(args, "vad_silero_model_path", None),
        ),
        (
            ["--vad-silero-sample-rate"],
            "vad.eval.silero_sample_rate",
            getattr(args, "vad_silero_sample_rate", None),
        ),
        (["--vad-train-prob"], "vad.train.prob", getattr(args, "vad_train_prob", None)),
        (
            ["--vad-train-every-steps"],
            "vad.train.every_steps",
            getattr(args, "vad_train_every_steps", None),
        ),
        (["--eval-sisdr"], "metrics.eval_sisdr", getattr(args, "eval_sisdr", None)),
        (["-v", "--verbose"], "debug.verbose", getattr(args, "verbose", None)),
        (["--debug-numerics"], "debug.debug_numerics", getattr(args, "debug_numerics", None)),
        (
            ["--debug-numerics-no-fail-fast"],
            "debug.debug_numerics_fail_fast",
            not getattr(args, "debug_numerics_no_fail_fast", False),
        ),
        (
            ["--debug-numerics-every"],
            "debug.debug_numerics_every",
            getattr(args, "debug_numerics_every", None),
        ),
        (
            ["--debug-numerics-dump-dir"],
            "debug.debug_numerics_dump_dir",
            getattr(args, "debug_numerics_dump_dir", None),
        ),
        (
            ["--debug-numerics-dump-arrays"],
            "debug.debug_numerics_dump_arrays",
            getattr(args, "debug_numerics_dump_arrays", None),
        ),
        (
            ["--debug-numerics-max-dumps"],
            "debug.debug_numerics_max_dumps",
            getattr(args, "debug_numerics_max_dumps", None),
        ),
        (["--nan-skip-batch"], "debug.nan_skip_batch", getattr(args, "nan_skip_batch", None)),
    ]

    if _flag_in_argv(["--fp16"], argv) and _flag_in_argv(["--no-fp16"], argv):
        raise ValueError("Cannot pass both --fp16 and --no-fp16.")
    if _flag_in_argv(["--fp16"], argv):
        set_by_path(cfg, "training.fp16", True)
    if _flag_in_argv(["--no-fp16"], argv):
        set_by_path(cfg, "training.fp16", False)
    if _flag_in_argv(["--no-mlx-data"], argv):
        set_by_path(cfg, "dataloader.use_mlx_data", False)
    if _flag_in_argv(["--no-vad-proxy"], argv):
        set_by_path(cfg, "loss.awesome.proxy_enabled", False)

    for flags, path, value in overrides:
        if _flag_in_argv(flags, argv):
            set_by_path(cfg, path, value)

    if _flag_in_argv(["--pipeline-stages"], argv):
        parsed_stages = _parse_pipeline_stages_cli(getattr(args, "pipeline_stages", None))
        set_by_path(cfg, "loss.pipeline_stages", parsed_stages)


def _build_speech_band_mask(
    n_freqs: int,
    sample_rate: int,
    band_low_hz: float,
    band_high_hz: float,
) -> tuple[mx.array, float]:
    """Build a fixed speech-band mask for STFT bins."""
    freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float32)
    mask = ((freqs >= band_low_hz) & (freqs <= band_high_hz)).astype(np.float32)
    band_bins = float(mask.sum())
    if band_bins < 1:
        raise ValueError(
            f"Speech band [{band_low_hz}, {band_high_hz}] Hz has no bins for " f"n_freqs={n_freqs}, sr={sample_rate}."
        )
    return mx.array(mask), band_bins


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


def _compute_vad_probs(
    clean_real: mx.array,
    clean_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[mx.array, mx.array]:
    """Compute soft VAD probabilities from log-band energy (z-scored per utterance)."""
    clean_real = clean_real.astype(mx.float32)
    clean_imag = clean_imag.astype(mx.float32)
    out_real = out_real.astype(mx.float32)
    out_imag = out_imag.astype(mx.float32)
    clean_power = clean_real**2 + clean_imag**2
    out_power = out_real**2 + out_imag**2

    clean_band = mx.sum(clean_power * band_mask, axis=-1) / (band_bins + eps)
    out_band = mx.sum(out_power * band_mask, axis=-1) / (band_bins + eps)

    log_clean = mx.log10(clean_band + eps)
    mu = mx.mean(log_clean, axis=1, keepdims=True)
    # Edge case: ensure minimum variance to avoid instability on silence
    variance = mx.mean((log_clean - mu) ** 2, axis=1, keepdims=True)
    _MIN_VARIANCE = 1e-4
    sigma = mx.sqrt(mx.maximum(variance, _MIN_VARIANCE) + eps)

    z_ref_raw = (log_clean - mu) / (sigma + eps)
    z_out_raw = (mx.log10(out_band + eps) - mu) / (sigma + eps)
    z_ref = mx.clip(z_ref_raw, -_VAD_LOGIT_CLAMP, _VAD_LOGIT_CLAMP)
    z_out = mx.clip(z_out_raw, -_VAD_LOGIT_CLAMP, _VAD_LOGIT_CLAMP)

    z_slope = max(vad_z_slope, 1e-3)
    p_ref = mx.sigmoid((z_ref - vad_z_threshold) / z_slope)
    p_out = mx.sigmoid((z_out - vad_z_threshold) / z_slope)
    if debug is not None:
        debug.check("vad.clean_band", clean_band, debug_ctx)
        debug.check("vad.out_band", out_band, debug_ctx)
        debug.check("vad.log_clean", log_clean, debug_ctx)
        debug.check("vad.sigma", sigma, debug_ctx)
        debug.check("vad.z_ref_raw", z_ref_raw, debug_ctx)
        debug.check("vad.z_out_raw", z_out_raw, debug_ctx)
        debug.check("vad.z_ref", z_ref, debug_ctx)
        debug.check("vad.z_out", z_out, debug_ctx)
        debug.check("vad.p_ref", p_ref, debug_ctx)
        debug.check("vad.p_out", p_out, debug_ctx)
    return p_ref, p_out


def _compute_vad_loss(
    clean_real: mx.array,
    clean_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    snr: mx.array,
    band_mask: mx.array,
    band_bins: float,
    vad_threshold: float,
    vad_margin: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute soft VAD loss and diagnostics.

    Penalizes decreases in VAD probability relative to reference speech.
    """
    p_ref, p_out = _compute_vad_probs(
        clean_real,
        clean_imag,
        out_real,
        out_imag,
        band_mask,
        band_bins,
        vad_z_threshold,
        vad_z_slope,
        debug=debug,
        debug_ctx=debug_ctx,
    )

    speech_gate = mx.clip((p_ref - vad_threshold) / (1.0 - vad_threshold + _EPS), 0.0, 1.0)
    snr_scale = max(vad_snr_gate_width, 1e-3)
    snr_gate = mx.sigmoid((snr[:, None] - vad_snr_gate_db) / snr_scale)
    gate = mx.stop_gradient(speech_gate * snr_gate)

    vad_loss = mx.mean(mx.maximum(p_ref - p_out - vad_margin, 0.0) * gate)
    if debug is not None:
        debug.check("vad.speech_gate", speech_gate, debug_ctx)
        debug.check("vad.snr_gate", snr_gate, debug_ctx)
        debug.check("vad.gate", gate, debug_ctx)
        debug.check("vad.loss", vad_loss, debug_ctx)
    return vad_loss, p_ref, p_out, gate


def _compute_speech_band_logmag_loss(
    clean_real: mx.array,
    clean_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    gate: mx.array,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> mx.array:
    """Compute speech-band log-magnitude L1 loss weighted by VAD gate."""
    clean_real = clean_real.astype(mx.float32)
    clean_imag = clean_imag.astype(mx.float32)
    out_real = out_real.astype(mx.float32)
    out_imag = out_imag.astype(mx.float32)
    clean_mag = mx.sqrt(clean_real**2 + clean_imag**2 + eps)
    out_mag = mx.sqrt(out_real**2 + out_imag**2 + eps)

    clean_log = mx.log10(clean_mag + eps)
    out_log = mx.log10(out_mag + eps)

    clean_band = mx.sum(clean_log * band_mask, axis=-1) / (band_bins + eps)
    out_band = mx.sum(out_log * band_mask, axis=-1) / (band_bins + eps)

    loss = mx.mean(mx.abs(out_band - clean_band) * gate)
    if debug is not None:
        debug.check("speech_band.clean_band", clean_band, debug_ctx)
        debug.check("speech_band.out_band", out_band, debug_ctx)
        debug.check("speech_band.loss", loss, debug_ctx)
    return loss


def _log1p_mag(real: mx.array, imag: mx.array, eps: float = _EPS) -> mx.array:
    """Compute log1p magnitude for complex STFT."""
    real = real.astype(mx.float32)
    imag = imag.astype(mx.float32)
    mag = mx.sqrt(real**2 + imag**2 + eps)
    return mx.log1p(mag)


def _compute_musicness(
    mag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[mx.array, mx.array]:
    """Compute a cheap musicness score and its inverse gate.

    Uses spectral flatness (tonalness) and temporal flux stability.
    Returns per-sample musicness and a [0,1] gate (1 = keep speech bias).
    """
    # Spectral flatness over speech band
    mag = mag.astype(mx.float32)
    log_mag = mx.log(mag + eps)
    mean_log = mx.sum(log_mag * band_mask, axis=-1) / (band_bins + eps)
    geom_mean = mx.exp(mean_log)
    arith_mean = mx.sum(mag * band_mask, axis=-1) / (band_bins + eps)
    flatness = geom_mean / (arith_mean + eps)
    tonal = 1.0 - mx.clip(flatness, 0.0, 1.0)
    tonal_mean = mx.mean(tonal, axis=1, keepdims=True)

    # Temporal flux (lower flux => more music-like)
    # Edge case: with single frame, no flux can be computed - assume speech-like
    band_mag = mag * band_mask
    if mag.shape[1] > 1:
        flux = mx.sum(mx.abs(band_mag[:, 1:, :] - band_mag[:, :-1, :]), axis=-1) / (band_bins + eps)
        flux = mx.mean(flux, axis=1, keepdims=True)
    else:
        flux = mx.zeros((mag.shape[0], 1))
    flux_gate = mx.sigmoid((_AWESOME_MUSIC_FLUX_THR - flux) / _AWESOME_MUSIC_FLUX_WIDTH)

    musicness = mx.clip(tonal_mean * flux_gate, 0.0, 1.0)
    music_gate = 1.0 - mx.sigmoid((musicness - _AWESOME_MUSICNESS_THR) / _AWESOME_MUSICNESS_WIDTH)
    musicness = musicness.squeeze(-1)
    music_gate = music_gate.squeeze(-1)
    if debug is not None:
        debug.check("musicness.score", musicness, debug_ctx)
        debug.check("musicness.gate", music_gate, debug_ctx)
    return musicness, music_gate


def _compute_proxy_gates(
    clean_real: mx.array,
    clean_imag: mx.array,
    noisy_real: mx.array,
    noisy_imag: mx.array,
    snr: mx.array,
    band_mask: mx.array,
    band_bins: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    proxy_enabled: bool,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Compute proxy VAD gates and statistics.

    Returns:
        proxy_frame: (B, T) speech presence proxy
        speech_ratio: (B, T) speech energy ratio in speech band
        music_gate: (B,) gate to downweight music-like frames
        musicness: (B,) musicness score
        mod_energy: (B, 1) modulation energy proxy
        energy_boost: (B, 1) low-energy boost
        snr_boost: (B, 1) low-SNR boost
    """
    clean_real = clean_real.astype(mx.float32)
    clean_imag = clean_imag.astype(mx.float32)
    noisy_real = noisy_real.astype(mx.float32)
    noisy_imag = noisy_imag.astype(mx.float32)
    clean_power = clean_real**2 + clean_imag**2
    noise_real = noisy_real - clean_real
    noise_imag = noisy_imag - clean_imag
    noise_power = noise_real**2 + noise_imag**2

    clean_band = mx.sum(clean_power * band_mask, axis=-1) / (band_bins + eps)
    noise_band = mx.sum(noise_power * band_mask, axis=-1) / (band_bins + eps)
    speech_ratio = clean_band / (clean_band + noise_band + eps)

    log_clean = mx.log10(clean_band + eps)
    mu = mx.mean(log_clean, axis=1, keepdims=True)
    # Edge case: ensure minimum variance to avoid instability on silence
    variance = mx.mean((log_clean - mu) ** 2, axis=1, keepdims=True)
    _MIN_VARIANCE = 1e-4
    sigma = mx.sqrt(mx.maximum(variance, _MIN_VARIANCE) + eps)
    z_ref_raw = (log_clean - mu) / (sigma + eps)
    z_ref = mx.clip(z_ref_raw, -_VAD_LOGIT_CLAMP, _VAD_LOGIT_CLAMP)

    z_slope = max(vad_z_slope, 1e-3)
    p_ref = mx.sigmoid((z_ref - vad_z_threshold) / z_slope)

    # Modulation proxy from z-scored energy trajectory
    # Edge case: if only 1 frame, no modulation can be computed
    if z_ref.shape[1] > 1:
        mod_energy = mx.mean(mx.abs(z_ref[:, 1:] - z_ref[:, :-1]), axis=1, keepdims=True)
    else:
        mod_energy = mx.zeros((z_ref.shape[0], 1))
    mod_gate = mx.sigmoid((mod_energy - _AWESOME_MOD_THRESHOLD) / _AWESOME_MOD_WIDTH)

    mean_log = mx.mean(log_clean, axis=1, keepdims=True)
    energy_boost = mx.sigmoid((_AWESOME_ENERGY_BOOST_DB - mean_log) / _AWESOME_ENERGY_BOOST_WIDTH)

    snr_scale = max(vad_snr_gate_width, 1e-3)
    snr_boost = mx.sigmoid((vad_snr_gate_db - snr[:, None]) / snr_scale)

    # Musicness gate from noisy magnitude
    noisy_mag = mx.sqrt(noisy_real**2 + noisy_imag**2 + eps)
    musicness, music_gate = _compute_musicness(
        noisy_mag,
        band_mask,
        band_bins,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
    )

    if not proxy_enabled:
        proxy_frame = mx.ones_like(clean_band)
    else:
        proxy_frame = p_ref * (_AWESOME_PROXY_RATIO_FLOOR + _AWESOME_PROXY_RATIO_SCALE * speech_ratio)
        proxy_frame = proxy_frame * mod_gate * music_gate[:, None]
        proxy_frame = proxy_frame * (
            1.0 + _AWESOME_LOW_ENERGY_WEIGHT * energy_boost + _AWESOME_LOW_SNR_WEIGHT * snr_boost
        )
        proxy_frame = mx.clip(proxy_frame, 0.0, 5.0)

    proxy_frame = mx.stop_gradient(proxy_frame)
    if debug is not None:
        debug.check("proxy.z_ref_raw", z_ref_raw, debug_ctx)
        debug.check("proxy.z_ref", z_ref, debug_ctx)
        debug.check("proxy.speech_ratio", speech_ratio, debug_ctx)
        debug.check("proxy.p_ref", p_ref, debug_ctx)
        debug.check("proxy.mod_energy", mod_energy, debug_ctx)
        debug.check("proxy.energy_boost", energy_boost, debug_ctx)
        debug.check("proxy.snr_boost", snr_boost, debug_ctx)
        debug.check("proxy.frame", proxy_frame, debug_ctx)
    return proxy_frame, speech_ratio, music_gate, musicness, mod_energy, energy_boost, snr_boost


def _compute_awesome_losses(
    noisy_real: mx.array,
    noisy_imag: mx.array,
    clean_real: mx.array,
    clean_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    snr: mx.array,
    band_mask: mx.array,
    band_bins: float,
    mask_sharpness: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    proxy_enabled: bool,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    """Compute awesome loss components and diagnostic gates."""
    clean_log = _log1p_mag(clean_real, clean_imag, eps=eps)
    out_log = _log1p_mag(out_real, out_imag, eps=eps)

    noise_real = noisy_real.astype(mx.float32) - clean_real.astype(mx.float32)
    noise_imag = noisy_imag.astype(mx.float32) - clean_imag.astype(mx.float32)
    noise_log = _log1p_mag(noise_real, noise_imag, eps=eps)

    mask_logits = mx.clip(
        mask_sharpness * (clean_log - noise_log),
        -_AWESOME_MASK_LOGIT_CLAMP,
        _AWESOME_MASK_LOGIT_CLAMP,
    )
    mask = mx.sigmoid(mask_logits)
    mask = mx.stop_gradient(mask)
    if debug is not None:
        debug.check("awesome.clean_log", clean_log, debug_ctx)
        debug.check("awesome.noise_log", noise_log, debug_ctx)
        debug.check("awesome.mask_logits", mask_logits, debug_ctx)
        debug.check("awesome.mask", mask, debug_ctx)

    (
        proxy_frame,
        speech_ratio,
        music_gate,
        musicness,
        mod_energy,
        energy_boost,
        snr_boost,
    ) = _compute_proxy_gates(
        clean_real,
        clean_imag,
        noisy_real,
        noisy_imag,
        snr,
        band_mask,
        band_bins,
        vad_z_threshold,
        vad_z_slope,
        vad_snr_gate_db,
        vad_snr_gate_width,
        proxy_enabled,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
    )

    proxy_frame = proxy_frame[:, :, None]
    speech_loss = mx.mean(mx.abs(out_log - clean_log) * mask * proxy_frame)
    noise_loss = mx.mean(mx.abs(out_log) * (1.0 - mask))

    if out_log.shape[1] > 1:
        smooth_mask = 1.0 - mask[:, 1:, :]
        smooth_loss = mx.mean(mx.abs(out_log[:, 1:, :] - out_log[:, :-1, :]) * smooth_mask)
    else:
        smooth_loss = mx.array(0.0)

    awesome_loss = speech_loss + noise_loss + _AWESOME_SMOOTH_WEIGHT * smooth_loss
    if debug is not None:
        debug.check("awesome.speech_loss", speech_loss, debug_ctx)
        debug.check("awesome.noise_loss", noise_loss, debug_ctx)
        debug.check("awesome.smooth_loss", smooth_loss, debug_ctx)
        debug.check("awesome.loss", awesome_loss, debug_ctx)

    return (
        awesome_loss,
        speech_loss,
        noise_loss,
        smooth_loss,
        mask,
        proxy_frame.squeeze(-1),
        speech_ratio,
        music_gate,
        musicness,
        mod_energy,
        energy_boost,
        snr_boost,
    )


def _compute_pitch_stability(
    mag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    eps: float = _EPS,
) -> mx.array:
    """Compute pitch stability metric to detect sustained vocals vs speech.

    Vocals tend to have more stable pitch (lower frame-to-frame variation)
    while speech has more dynamic pitch contours.

    Returns per-sample pitch stability in [0, 1], where 1 = very stable (vocal-like).
    """
    mag = mag.astype(mx.float32)
    band_mag = mag * band_mask

    # Compute spectral centroid per frame
    freq_weights = mx.arange(band_mag.shape[-1], dtype=mx.float32)
    centroid = mx.sum(band_mag * freq_weights, axis=-1) / (mx.sum(band_mag, axis=-1) + eps)

    # Pitch stability = inverse of centroid variation
    if centroid.shape[1] > 1:
        centroid_diff = mx.abs(centroid[:, 1:] - centroid[:, :-1])
        centroid_var = mx.mean(centroid_diff, axis=1, keepdims=True)
        # Normalize and invert: low variation = high stability
        stability = mx.exp(-centroid_var / 10.0)
    else:
        stability = mx.ones((mag.shape[0], 1))

    return mx.clip(stability, 0.0, 1.0).squeeze(-1)


def _compute_harmonic_ratio(
    mag: mx.array,
    eps: float = _EPS,
) -> mx.array:
    """Compute harmonic-to-noise ratio to detect tonal content (vocals/music).

    Uses autocorrelation proxy: high HNR = more harmonic/tonal content.
    Returns per-sample HNR score in [0, 1].
    """
    mag = mag.astype(mx.float32)

    # Simple proxy: ratio of peak to mean energy in low-mid frequencies
    # Harmonic content creates spectral peaks
    low_mid_mag = mag[:, :, : mag.shape[-1] // 2]  # Lower half of spectrum
    peak_energy = mx.max(low_mid_mag, axis=-1)
    mean_energy = mx.mean(low_mid_mag, axis=-1) + eps

    hnr_proxy = peak_energy / mean_energy
    # Normalize to [0, 1] using sigmoid
    hnr_score = mx.sigmoid((hnr_proxy - 3.0) / 1.0)  # Center at ratio=3

    return mx.mean(hnr_score, axis=1)


def _compute_improved_musicness(
    mag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    snr: mx.array,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute improved musicness score with vocal detection.

    Returns:
        musicness: (B,) overall musicness score
        vocal_gate: (B,) gate for vocal content (1 = protect as speech)
        instrument_gate: (B,) gate for instrumental content (1 = suppress)
    """
    mag = mag.astype(mx.float32)

    # Original spectral flatness
    log_mag = mx.log(mag + eps)
    mean_log = mx.sum(log_mag * band_mask, axis=-1) / (band_bins + eps)
    geom_mean = mx.exp(mean_log)
    arith_mean = mx.sum(mag * band_mask, axis=-1) / (band_bins + eps)
    flatness = geom_mean / (arith_mean + eps)
    tonal = 1.0 - mx.clip(flatness, 0.0, 1.0)
    tonal_mean = mx.mean(tonal, axis=1, keepdims=True)

    # Temporal flux
    # Edge case: with single frame, no flux can be computed - assume speech-like
    band_mag = mag * band_mask
    if mag.shape[1] > 1:
        flux = mx.sum(mx.abs(band_mag[:, 1:, :] - band_mag[:, :-1, :]), axis=-1) / (band_bins + eps)
        flux_mean = mx.mean(flux, axis=1, keepdims=True)
    else:
        flux_mean = mx.zeros((mag.shape[0], 1))
    flux_gate = mx.sigmoid((_AWESOME_MUSIC_FLUX_THR - flux_mean) / _AWESOME_MUSIC_FLUX_WIDTH)

    # Pitch stability (vocals = less stable than instruments)
    pitch_stability = _compute_pitch_stability(mag, band_mask, band_bins, eps)

    # Harmonic ratio
    harmonic_ratio = _compute_harmonic_ratio(mag, eps)

    # Musicness from original features
    musicness_base = mx.clip(tonal_mean.squeeze(-1) * flux_gate.squeeze(-1), 0.0, 1.0)

    # Vocal detection: high tonality + moderate pitch stability + present in speech band
    # Vocals: tonal but with more pitch variation than instruments
    vocal_indicator = tonal_mean.squeeze(-1) * (1.0 - pitch_stability) * harmonic_ratio
    vocal_gate = mx.sigmoid((vocal_indicator - _PIPELINE_VOCAL_HARMONIC_THR) / 0.15)

    # Instrumental: high tonality + high pitch stability (sustained notes)
    instrument_indicator = tonal_mean.squeeze(-1) * pitch_stability * flux_gate.squeeze(-1)
    instrument_gate = mx.sigmoid((instrument_indicator - _PIPELINE_PITCH_STABILITY_THR) / 0.15)

    # Adjust musicness: reduce for vocals (they should be preserved as speech-like)
    musicness = musicness_base * (1.0 - 0.5 * vocal_gate)

    if debug is not None:
        debug.check("improved_musicness.tonal", tonal_mean, debug_ctx)
        debug.check("improved_musicness.flux", flux_mean, debug_ctx)
        debug.check("improved_musicness.pitch_stab", pitch_stability, debug_ctx)
        debug.check("improved_musicness.harmonic", harmonic_ratio, debug_ctx)
        debug.check("improved_musicness.vocal_gate", vocal_gate, debug_ctx)
        debug.check("improved_musicness.instrument_gate", instrument_gate, debug_ctx)

    return musicness, vocal_gate, instrument_gate


def _compute_pipeline_awesome_losses(
    noisy_real: mx.array,
    noisy_imag: mx.array,
    clean_real: mx.array,
    clean_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    snr: mx.array,
    band_mask: mx.array,
    band_bins: float,
    mask_sharpness: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    proxy_enabled: bool,
    min_mask_floor: float = _PIPELINE_MIN_MASK_FLOOR,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[
    mx.array,  # total loss
    mx.array,  # speech loss
    mx.array,  # noise loss
    mx.array,  # smooth loss
    mx.array,  # music suppression loss
    mx.array,  # mask saturation loss
    mx.array,  # mask
    mx.array,  # proxy_frame
    mx.array,  # speech_ratio
    mx.array,  # music_gate
    mx.array,  # musicness
    mx.array,  # vocal_gate
    mx.array,  # instrument_gate
    mx.array,  # mod_energy
    mx.array,  # energy_boost
    mx.array,  # snr_boost
]:
    """Compute pipeline_awesome loss with improved speech preservation and music suppression.

    Key improvements over basic awesome loss:
    1. Minimum mask floor to prevent complete speech suppression
    2. Additive (not multiplicative) boosts for low-energy and low-SNR speech
    3. Improved musicness detection with vocal/instrument separation
    4. Speech-band weighted loss
    5. Mask saturation penalty to encourage confident predictions
    6. Explicit music suppression loss

    Note: The mask saturation penalty uses mask entropy: mask*(1-mask).
    This is minimized when mask is near 0 or 1 (confident), and maximized
    at 0.5 (uncertain). We want to PENALIZE uncertainty, so we use it directly.
    """
    # Compute log magnitudes (same as awesome loss)
    clean_log = _log1p_mag(clean_real, clean_imag, eps=eps)
    out_log = _log1p_mag(out_real, out_imag, eps=eps)

    noise_real = noisy_real.astype(mx.float32) - clean_real.astype(mx.float32)
    noise_imag = noisy_imag.astype(mx.float32) - clean_imag.astype(mx.float32)
    noise_log = _log1p_mag(noise_real, noise_imag, eps=eps)

    # Compute speech/noise dominance mask with floor
    mask_logits = mx.clip(
        mask_sharpness * (clean_log - noise_log),
        -_AWESOME_MASK_LOGIT_CLAMP,
        _AWESOME_MASK_LOGIT_CLAMP,
    )
    raw_mask = mx.sigmoid(mask_logits)
    # Apply minimum floor to prevent complete suppression
    mask = mx.maximum(raw_mask, min_mask_floor)
    mask = mx.stop_gradient(mask)

    if debug is not None:
        debug.check("pipeline.clean_log", clean_log, debug_ctx)
        debug.check("pipeline.noise_log", noise_log, debug_ctx)
        debug.check("pipeline.mask_logits", mask_logits, debug_ctx)
        debug.check("pipeline.raw_mask", raw_mask, debug_ctx)
        debug.check("pipeline.mask", mask, debug_ctx)

    # Compute improved proxy gates with additive boosts
    clean_real_f32 = clean_real.astype(mx.float32)
    clean_imag_f32 = clean_imag.astype(mx.float32)
    noisy_real_f32 = noisy_real.astype(mx.float32)
    noisy_imag_f32 = noisy_imag.astype(mx.float32)

    clean_power = clean_real_f32**2 + clean_imag_f32**2
    noise_real_f32 = noisy_real_f32 - clean_real_f32
    noise_imag_f32 = noisy_imag_f32 - clean_imag_f32
    noise_power = noise_real_f32**2 + noise_imag_f32**2

    clean_band = mx.sum(clean_power * band_mask, axis=-1) / (band_bins + eps)
    noise_band = mx.sum(noise_power * band_mask, axis=-1) / (band_bins + eps)
    speech_ratio = clean_band / (clean_band + noise_band + eps)

    # Z-scored log energy for VAD proxy
    # Edge case handling: if variance is near-zero (silence), use neutral z-scores
    log_clean = mx.log10(clean_band + eps)
    mu = mx.mean(log_clean, axis=1, keepdims=True)
    variance = mx.mean((log_clean - mu) ** 2, axis=1, keepdims=True)
    # Use a minimum variance threshold to avoid division instability on silence
    _MIN_VARIANCE = 1e-4
    sigma = mx.sqrt(mx.maximum(variance, _MIN_VARIANCE) + eps)
    # When variance is too low, z-scores become unreliable; clamp them
    z_ref_raw = (log_clean - mu) / (sigma + eps)
    z_ref = mx.clip(z_ref_raw, -_VAD_LOGIT_CLAMP, _VAD_LOGIT_CLAMP)

    z_slope = max(vad_z_slope, 1e-3)
    p_ref = mx.sigmoid((z_ref - vad_z_threshold) / z_slope)

    # Modulation proxy
    # Edge case: with single frame, no modulation can be computed
    if z_ref.shape[1] > 1:
        mod_energy = mx.mean(mx.abs(z_ref[:, 1:] - z_ref[:, :-1]), axis=1, keepdims=True)
    else:
        mod_energy = mx.zeros((z_ref.shape[0], 1))
    mod_gate = mx.sigmoid((mod_energy - _AWESOME_MOD_THRESHOLD) / _AWESOME_MOD_WIDTH)

    # Energy and SNR boosts (ADDITIVE, not multiplicative)
    mean_log = mx.mean(log_clean, axis=1, keepdims=True)
    energy_boost = mx.sigmoid((_AWESOME_ENERGY_BOOST_DB - mean_log) / _AWESOME_ENERGY_BOOST_WIDTH)

    snr_scale = max(vad_snr_gate_width, 1e-3)
    snr_boost = mx.sigmoid((vad_snr_gate_db - snr[:, None]) / snr_scale)

    # Improved musicness detection
    noisy_mag = mx.sqrt(noisy_real_f32**2 + noisy_imag_f32**2 + eps)
    musicness, vocal_gate, instrument_gate = _compute_improved_musicness(
        noisy_mag,
        band_mask,
        band_bins,
        snr,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
    )

    # Music gate: downweight for instrumental, but preserve vocal-like content
    music_gate = 1.0 - mx.sigmoid((musicness - _AWESOME_MUSICNESS_THR) / _AWESOME_MUSICNESS_WIDTH)
    # Boost back for vocals (they should be preserved)
    music_gate = music_gate + 0.5 * vocal_gate * (1.0 - music_gate)

    if not proxy_enabled:
        proxy_frame = mx.ones_like(clean_band)
    else:
        # Base proxy from VAD and speech ratio (with higher floor)
        base_proxy = p_ref * (_PIPELINE_PROXY_FLOOR + (1.0 - _PIPELINE_PROXY_FLOOR) * speech_ratio)
        base_proxy = base_proxy * mod_gate * music_gate[:, None]

        # ADDITIVE boosts (key improvement for low-signal speech)
        proxy_frame = base_proxy + _PIPELINE_LOW_ENERGY_ADDITIVE * energy_boost + _PIPELINE_LOW_SNR_ADDITIVE * snr_boost
        proxy_frame = mx.clip(proxy_frame, _PIPELINE_PROXY_FLOOR, 5.0)

    proxy_frame = mx.stop_gradient(proxy_frame)

    if debug is not None:
        debug.check("pipeline.z_ref", z_ref, debug_ctx)
        debug.check("pipeline.p_ref", p_ref, debug_ctx)
        debug.check("pipeline.speech_ratio", speech_ratio, debug_ctx)
        debug.check("pipeline.energy_boost", energy_boost, debug_ctx)
        debug.check("pipeline.snr_boost", snr_boost, debug_ctx)
        debug.check("pipeline.music_gate", music_gate, debug_ctx)
        debug.check("pipeline.proxy_frame", proxy_frame, debug_ctx)

    # ========== Loss components ==========

    # 1. Speech preservation loss (weighted by proxy)
    proxy_frame_3d = proxy_frame[:, :, None]
    speech_loss = mx.mean(mx.abs(out_log - clean_log) * mask * proxy_frame_3d)

    # 2. Noise suppression loss
    noise_loss = mx.mean(mx.abs(out_log) * (1.0 - mask))

    # 3. Temporal smoothness for artifact control (stronger than base awesome)
    if out_log.shape[1] > 1:
        smooth_mask = 1.0 - mask[:, 1:, :]
        smooth_loss = mx.mean(mx.abs(out_log[:, 1:, :] - out_log[:, :-1, :]) * smooth_mask)
    else:
        smooth_loss = mx.array(0.0)

    # 4. Music suppression loss: penalize output energy where instrumental music detected
    instrument_weight = instrument_gate[:, None, None] * (1.0 - mask)  # Only where noise dominant
    music_suppression_loss = mx.mean(mx.abs(out_log) * instrument_weight)

    # 5. Mask saturation penalty: encourage confident mask predictions
    # mask * (1-mask) is an entropy-like term:
    #   - Minimized at 0 or 1 (confident predictions)
    #   - Maximized at 0.5 (uncertain predictions)
    # We PENALIZE uncertainty by using this term directly as the loss.
    # FIX: Previous implementation inverted this (1.0 - 4.0*entropy), which
    # rewarded uncertainty. Now we penalize uncertainty directly.
    mask_entropy = mx.mean(raw_mask * (1.0 - raw_mask))
    # Scale to [0, 1]: max entropy at mask=0.5 is 0.25, so multiply by 4
    mask_saturation_loss = 4.0 * mask_entropy

    # Total loss
    total_loss = (
        speech_loss
        + noise_loss
        + _PIPELINE_ARTIFACT_SMOOTH_WEIGHT * smooth_loss
        + _PIPELINE_MUSIC_SUPPRESSION_WEIGHT * music_suppression_loss
        + _PIPELINE_MASK_SATURATION_PENALTY * mask_saturation_loss
    )

    if debug is not None:
        debug.check("pipeline.speech_loss", speech_loss, debug_ctx)
        debug.check("pipeline.noise_loss", noise_loss, debug_ctx)
        debug.check("pipeline.smooth_loss", smooth_loss, debug_ctx)
        debug.check("pipeline.music_suppression_loss", music_suppression_loss, debug_ctx)
        debug.check("pipeline.mask_saturation_loss", mask_saturation_loss, debug_ctx)
        debug.check("pipeline.total_loss", total_loss, debug_ctx)

    return (
        total_loss,
        speech_loss,
        noise_loss,
        smooth_loss,
        music_suppression_loss,
        mask_saturation_loss,
        mask,
        proxy_frame,
        speech_ratio,
        music_gate,
        musicness,
        vocal_gate,
        instrument_gate,
        mod_energy.squeeze(-1),
        energy_boost.squeeze(-1),
        snr_boost.squeeze(-1),
    )


def _compute_vad_reg_loss(
    clean_real: mx.array,
    clean_imag: mx.array,
    noisy_real: mx.array,
    noisy_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    snr: mx.array,
    band_mask: mx.array,
    band_bins: float,
    vad_threshold: float,
    vad_margin: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Compute sparse VAD regularizer loss gated by speech ratio and musicness.

    Uses VAD probabilities only as stop-grad weights (non-differentiable).
    """
    p_ref, p_out = _compute_vad_probs(
        clean_real,
        clean_imag,
        out_real,
        out_imag,
        band_mask,
        band_bins,
        vad_z_threshold,
        vad_z_slope,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
    )

    vad_decrease = mx.maximum(p_ref - p_out - vad_margin, 0.0)

    proxy_frame, speech_ratio, music_gate, musicness, _, _, _ = _compute_proxy_gates(
        clean_real,
        clean_imag,
        noisy_real,
        noisy_imag,
        snr,
        band_mask,
        band_bins,
        vad_z_threshold,
        vad_z_slope,
        vad_snr_gate_db,
        vad_snr_gate_width,
        proxy_enabled=True,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
    )

    ratio_gate = mx.sigmoid((speech_ratio - vad_threshold) / 0.1)
    gate = mx.stop_gradient(vad_decrease * ratio_gate * music_gate[:, None])

    speech_loss = _compute_speech_band_logmag_loss(
        clean_real,
        clean_imag,
        out_real,
        out_imag,
        band_mask,
        band_bins,
        gate,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
    )

    return (
        speech_loss,
        vad_decrease,
        gate,
        p_ref,
        p_out,
        speech_ratio,
        musicness,
    )


def _compute_vad_eval_metrics(
    p_ref: mx.array,
    p_out: mx.array,
    vad_margin: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute VAD evaluation metrics (mean p_ref/p_out and decrease)."""
    p_ref_mean = mx.mean(p_ref)
    p_out_mean = mx.mean(p_out)
    vad_decrease = mx.mean(mx.maximum(p_ref - p_out - vad_margin, 0.0))
    return p_ref_mean, p_out_mean, vad_decrease


def _snr_bucket_name(snr_db: float) -> str:
    """Map SNR to a stable scenario bucket label."""
    if snr_db <= -20.0:
        return "very_low"
    if snr_db <= -5.0:
        return "extreme"
    if snr_db <= 5.0:
        return "low"
    if snr_db <= 20.0:
        return "mid"
    return "high"


# ============================================================================
# Signal Handling for Graceful Interrupt
# ============================================================================

# Global state for signal handler
_interrupt_state = {
    "checkpoint_dir": None,
    "epoch": 0,
    "batch_idx": 0,
    "global_step": 0,
    "model": None,
    "optimizer": None,
    "discriminator": None,
    "disc_optimizer": None,
    "loss": 0.0,
    "best_valid_loss": float("inf"),
    "config": {},
    "interrupted": False,
    "train_stream": None,
    "data_checkpoint_path": None,
    "last_completed_epoch": -1,
}


def _handle_sigint(signum, frame):
    """Handle SIGINT (CTRL+C) to save final checkpoint before exit.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    if _interrupt_state["interrupted"]:
        print("\n❌ Force exit (SIGINT received again)")
        sys.exit(1)

    _interrupt_state["interrupted"] = True
    signal_name = "SIGINT"
    if signum == signal.SIGTERM:
        signal_name = "SIGTERM"
    print("\n" + "=" * 60)
    print(f"⚠️  Training interrupted ({signal_name})")
    print("=" * 60)

    # Save final checkpoint
    if (
        _interrupt_state["model"] is not None
        and _interrupt_state["optimizer"] is not None
        and _interrupt_state["checkpoint_dir"] is not None
    ):
        try:
            print("💾 Saving final checkpoint before exit...")
            ckpt_dir = Path(_interrupt_state["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            epoch_idx = _interrupt_state.get("epoch", 0)
            batch_idx = _interrupt_state.get("batch_idx", 0)
            gstep = _interrupt_state.get("global_step", 0)
            last_completed = _interrupt_state.get("last_completed_epoch", -1)

            final_path = ckpt_dir / f"interrupted_epoch_{epoch_idx + 1:03d}.safetensors"
            saved = save_checkpoint(
                _interrupt_state["model"],
                final_path,
                epoch=epoch_idx,
                batch_idx=batch_idx,
                global_step=gstep,
                loss=_interrupt_state["loss"],
                best_valid_loss=_interrupt_state["best_valid_loss"],
                config=_interrupt_state["config"],
                optimizer=_interrupt_state["optimizer"],
                discriminator=_interrupt_state.get("discriminator"),
                disc_optimizer=_interrupt_state.get("disc_optimizer"),
                last_completed_epoch=last_completed,
                kind="interrupted",
            )
            if saved:
                print(f"✅ Final checkpoint saved to {final_path}")
            else:
                print(f"❌ Failed to save final checkpoint to {final_path}")

            # Also persist MLXDataStream state so --resume-data works after interrupts.
            train_stream = _interrupt_state.get("train_stream")
            data_ckpt_path = _interrupt_state.get("data_checkpoint_path")
            if train_stream is not None and data_ckpt_path is not None:
                try:
                    train_stream.save_checkpoint(data_ckpt_path)
                    print(f"✅ Data checkpoint saved to {data_ckpt_path}")
                except Exception as e_data:
                    print(f"❌ Failed to save data checkpoint: {data_ckpt_path} ({e_data})")
        except Exception as e:
            print(f"❌ Failed to save final checkpoint: {e}")

    print("Exiting...")
    raise KeyboardInterrupt()


def _register_sigint_handler(
    model,
    optimizer,
    checkpoint_dir,
    config,
    *,
    discriminator=None,
    disc_optimizer=None,
    last_completed_epoch: int = -1,
):
    """Register SIGINT handler for graceful training shutdown.

    Args:
        model: Model to save on interrupt
        optimizer: Optimizer to save state on interrupt
        checkpoint_dir: Directory to save checkpoint to
        config: Training configuration dict
        last_completed_epoch: Last fully completed epoch when registering
    """
    _interrupt_state["model"] = model
    _interrupt_state["optimizer"] = optimizer
    _interrupt_state["discriminator"] = discriminator
    _interrupt_state["disc_optimizer"] = disc_optimizer
    _interrupt_state["checkpoint_dir"] = checkpoint_dir
    _interrupt_state["config"] = config
    _interrupt_state["last_completed_epoch"] = last_completed_epoch
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)


def _update_interrupt_state(epoch, loss, best_valid_loss, *, batch_idx=0, global_step=0, last_completed_epoch=-1):
    """Update global state for interrupt handler.

    Args:
        epoch: Current epoch
        loss: Current training loss
        best_valid_loss: Best validation loss so far
        batch_idx: Current batch index within epoch
        global_step: Global training step
        last_completed_epoch: Last fully completed epoch index
    """
    _interrupt_state["epoch"] = epoch
    _interrupt_state["batch_idx"] = batch_idx
    _interrupt_state["global_step"] = global_step
    _interrupt_state["loss"] = loss
    _interrupt_state["best_valid_loss"] = best_valid_loss
    _interrupt_state["last_completed_epoch"] = last_completed_epoch


def print_hardware_diagnostics():
    """Print comprehensive hardware and MLX diagnostics."""
    print("\n" + "=" * 70)
    print("HARDWARE DIAGNOSTICS")
    print("=" * 70)

    # System info
    import platform

    print("\n[System]")
    print(f"  Platform:     {platform.platform()}")
    print(f"  Python:       {platform.python_version()}")
    print(f"  Processor:    {platform.processor() or 'Unknown'}")

    # MLX device info
    print("\n[MLX]")
    print(f"  Default device: {mx.default_device()}")
    print(f"  MLX version:    {mx.__version__ if hasattr(mx, '__version__') else 'Unknown'}")  # type: ignore

    # Try to get Apple Silicon info
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  CPU:            {result.stdout.strip()}")
    except Exception:
        pass

    # Memory info
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            mem_bytes = int(result.stdout.strip())
            mem_gb = mem_bytes / (1024**3)
            print(f"  Total RAM:      {mem_gb:.1f} GB")
    except Exception:
        pass

    # GPU cores (Apple Silicon)
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Total Number of Cores" in line:
                    print(f"  GPU Cores:      {line.split(':')[-1].strip()}")
                    break
    except Exception:
        pass

    # CPU core info
    try:
        perf_cores = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True,
            text=True,
        )
        eff_cores = subprocess.run(
            ["sysctl", "-n", "hw.perflevel1.logicalcpu"],
            capture_output=True,
            text=True,
        )
        if perf_cores.returncode == 0 and eff_cores.returncode == 0:
            p = perf_cores.stdout.strip()
            e = eff_cores.stdout.strip()
            print(f"  CPU Cores:      {p} performance + {e} efficiency")
    except Exception:
        pass

    # Current process CPU affinity / thread count
    print("\n[Process]")
    print(f"  PID:            {os.getpid()}")
    import multiprocessing

    print(f"  CPU count:      {multiprocessing.cpu_count()}")

    # MLX memory (if available)
    try:
        # MLX doesn't have direct memory query, but we can check metal
        result = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  GPU Wired Limit: {result.stdout.strip()} MB")
    except Exception:
        pass

    print("=" * 70 + "\n")


def clip_grad_norm(grads, max_norm: float) -> Tuple[dict, mx.array]:
    """Clip gradients by global norm.

    Returns:
        Tuple of (clipped_grads, grad_norm) where grad_norm is an MLX array.
        Call float(grad_norm) outside compiled functions to get the scalar value.
    """
    clipped, total_norm = clip_grad_norm_tree(grads, max_norm)
    return cast(dict, clipped), total_norm


def accumulate_grads(accumulated: Any | None, new_grads: Any) -> Any:
    """Accumulate gradients by summing them element-wise.

    Args:
        accumulated: Previous accumulated gradients (None for first batch)
        new_grads: New gradients to add

    Returns:
        Combined gradient tree
    """
    if accumulated is None:
        return new_grads

    def add_trees(a: Any, b: Any) -> Any:
        if isinstance(a, mx.array) and isinstance(b, mx.array):
            return a + b
        elif isinstance(a, dict) and isinstance(b, dict):
            return {k: add_trees(a[k], b[k]) for k in a.keys()}
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            result = [add_trees(av, bv) for av, bv in zip(a, b)]
            return type(a)(result)
        return b  # fallback (shouldn't happen with valid grad trees)

    return add_trees(accumulated, new_grads)


def scale_grads(grads: Any, scale: float) -> Any:
    """Scale all gradients by a constant factor.

    Args:
        grads: Gradient tree
        scale: Scale factor (e.g., 1/grad_accumulation_steps)

    Returns:
        Scaled gradient tree
    """
    scale_arr = mx.array(scale, dtype=mx.float32)

    def apply_scale(x: Any) -> Any:
        if isinstance(x, mx.array):
            return x * scale_arr
        elif isinstance(x, dict):
            return {k: apply_scale(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [apply_scale(v) for v in x]
        elif isinstance(x, tuple):
            return tuple(apply_scale(v) for v in x)
        return x

    return apply_scale(grads)


def specs_to_wavs(
    out_spec: tuple[mx.array, mx.array],
    clean_spec: tuple[mx.array, mx.array],
    *,
    istft_fn: Callable[..., mx.array],
    n_fft: int,
    hop_length: int,
    target_len: int,
    force_fp32: bool = True,
) -> tuple[mx.array, mx.array]:
    """Convert complex specs to waveforms with optional FP32 stabilization."""
    if force_fp32:
        out_spec = (out_spec[0].astype(mx.float32), out_spec[1].astype(mx.float32))
        clean_spec = (clean_spec[0].astype(mx.float32), clean_spec[1].astype(mx.float32))

    clean_wav = istft_fn(
        clean_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        length=target_len,
    )
    out_wav = istft_fn(
        out_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        length=target_len,
    )

    if force_fp32:
        if clean_wav.dtype != mx.float32:
            clean_wav = clean_wav.astype(mx.float32)
        if out_wav.dtype != mx.float32:
            out_wav = out_wav.astype(mx.float32)

    return out_wav, clean_wav


def compute_mrstft_loss(
    out_spec: tuple[mx.array, mx.array],
    clean_spec: tuple[mx.array, mx.array],
    *,
    istft_fn: Callable[..., mx.array],
    loss_fn: Callable[[mx.array, mx.array], mx.array],
    n_fft: int,
    hop_length: int,
    target_len: int,
    force_fp32: bool = True,
) -> mx.array:
    """Compute MRSTFT loss from complex specs with optional FP32 stabilization.

    MRSTFT involves magnitude squaring and power compression, which can overflow
    in FP16 when the model outputs large spectral magnitudes. We optionally cast
    to FP32 for this path to keep losses finite while the rest of the training
    stays in mixed precision.
    """
    if istft_fn is None or loss_fn is None:
        return mx.array(0.0)

    out_wav, clean_wav = specs_to_wavs(
        out_spec,
        clean_spec,
        istft_fn=istft_fn,
        n_fft=n_fft,
        hop_length=hop_length,
        target_len=target_len,
        force_fp32=force_fp32,
    )

    return loss_fn(out_wav, clean_wav)


def _gan_waveform_view(wav: mx.array, *, use_fp16: bool) -> mx.array:
    """Return GAN discriminator waveform view in the desired precision.

    GAN discriminator activations are a major memory contributor when adversarial
    training activates. Keeping this path in model precision (FP16 when enabled)
    reduces peak memory while MRSTFT can still run in FP32 for stability.
    """
    if use_fp16 and wav.dtype != mx.float16:
        return wav.astype(mx.float16)
    return wav


def _disc_crop_waveform(wav: mx.array, max_samples: int, crop_start: int | None = None) -> tuple[mx.array, int]:
    """Random-crop waveform along the time axis for discriminator input.

    Waveform-domain discriminators (MPD/MSD) produce enormous activation tensors
    proportional to input length.  Cropping to a shorter segment (e.g. 1 s at
    48 kHz = 48 000 samples) cuts discriminator memory by the ratio
    ``original_len / max_samples`` with negligible quality impact — the
    discriminator only needs to assess local perceptual quality.

    Args:
        wav: Waveform tensor ``(batch, samples)``.
        max_samples: Maximum number of samples to keep (0 = no crop).
        crop_start: If given, reuse this start index (keeps fake/real aligned).

    Returns:
        (cropped_wav, crop_start) so the same offset can be reused for the
        paired waveform.
    """
    if max_samples <= 0 or wav.shape[-1] <= max_samples:
        return wav, 0
    if crop_start is None:
        crop_start = random.randint(0, wav.shape[-1] - max_samples)
    return wav[:, crop_start : crop_start + max_samples], crop_start


_CHECKPOINT_KINDS = {"step", "epoch_end", "best", "best_final", "final", "interrupted"}
_COMPLETED_KINDS = {"epoch_end", "best", "best_final", "final"}
_IN_PROGRESS_KINDS = {"step", "interrupted"}
_COUNTER_SEMANTICS_VERSION = 2


@dataclass(frozen=True)
class CheckpointManifest:
    """Manifest describing checkpoint file layout and naming patterns."""

    weights_ext: str = ".safetensors"
    state_ext: str = ".state.json"
    tmp_suffixes: tuple[str, ...] = (".tmp", ".partial")
    epoch_complete_suffix: str = ".complete"

    step_re: re.Pattern[str] = re.compile(r"^step_(\d+)\.safetensors$")
    epoch_re: re.Pattern[str] = re.compile(r"^epoch_(\d+)\.safetensors$")
    interrupted_re: re.Pattern[str] = re.compile(r"^interrupted_epoch_(\d+)\.safetensors$")
    complete_re: re.Pattern[str] = re.compile(r"^epoch_(\d+)\.complete$")

    def state_path(self, weights_path: Path) -> Path:
        return weights_path.with_suffix(self.state_ext)

    def is_temporary(self, path: Path) -> bool:
        name = path.name
        return any(suffix in name for suffix in self.tmp_suffixes)

    def expected_from_name(self, path: Path) -> dict:
        name = path.name
        if match := self.step_re.match(name):
            return {"kind": "step", "global_step": int(match.group(1))}
        if match := self.epoch_re.match(name):
            return {"kind": "epoch_end", "epoch": int(match.group(1)) - 1}
        if match := self.interrupted_re.match(name):
            return {"kind": "interrupted", "epoch": int(match.group(1)) - 1}
        if name == "best.safetensors":
            return {"kinds": {"best", "best_final"}}
        if name == "final.safetensors":
            return {"kinds": {"final"}}
        return {}

    def marker_epoch(self, path: Path) -> int | None:
        if match := self.complete_re.match(path.name):
            return int(match.group(1)) - 1
        return None


def _disc_weights_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.disc{path.suffix}")


def _is_disc_weights(path: Path, manifest: CheckpointManifest | None = None) -> bool:
    manifest = manifest or CheckpointManifest()
    return path.name.endswith(f".disc{manifest.weights_ext}")


@dataclass
class CheckpointRecord:
    """Parsed checkpoint metadata for validation and resume planning."""

    path: Path
    state_path: Path
    mtime: float
    state: dict[str, Any] | None = None
    kind: str | None = None
    epoch: int | None = None
    batch_idx: int | None = None
    global_step: int | None = None
    last_completed_epoch: int | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def _record_sort_key(record: CheckpointRecord) -> tuple[int, float]:
    """Sort checkpoints by global_step when available, falling back to mtime."""
    if record.global_step is None:
        return (-1, record.mtime)
    return (record.global_step, record.mtime)


def _validate_checkpoint_pair(checkpoint_path: Path, *, manifest: CheckpointManifest | None = None) -> bool:
    """Validate that both weights and state files exist and are non-empty.

    Args:
        checkpoint_path: Path to checkpoint (.safetensors file)

    Returns:
        True if both files exist and are valid, False otherwise
    """
    manifest = manifest or CheckpointManifest()
    weights_file = checkpoint_path
    state_file = manifest.state_path(checkpoint_path)

    # Check both files exist
    if not weights_file.exists():
        print(f"⚠️  Checkpoint missing: {weights_file.name}")
        return False
    if not state_file.exists():
        print(f"⚠️  Checkpoint missing state file: {state_file.name}")
        return False

    # Check files are not empty (indicates incomplete write)
    if weights_file.stat().st_size == 0:
        print(f"⚠️  Checkpoint is empty: {weights_file.name}")
        return False
    if state_file.stat().st_size == 0:
        print(f"⚠️  Checkpoint state file is empty: {state_file.name}")
        return False

    return True


def compute_resume_epoch(state: dict) -> int:
    """Determine the epoch index to resume from based on checkpoint kind."""
    epoch = int(state.get("epoch", 0))
    kind = state.get("kind", "epoch_end")
    if kind in _COMPLETED_KINDS:
        return epoch + 1
    return epoch


def resolve_resume_batch_count(state: dict[str, Any]) -> int:
    """Resolve resume micro-batch count from checkpoint state.

    Returns the number of micro-batches already consumed in the in-progress
    epoch. For legacy checkpoints (without counter_semantics_version),
    batch_idx is interpreted as a 0-based index of the last processed batch and
    is converted to a processed-count via +1.
    """
    kind = state.get("kind", "epoch_end")
    if kind not in _IN_PROGRESS_KINDS:
        return 0

    raw_counter = state.get("micro_batches_completed", state.get("batch_idx"))
    if not isinstance(raw_counter, int) or raw_counter < 0:
        return 0

    version_raw = state.get("counter_semantics_version", 1)
    version = version_raw if isinstance(version_raw, int) else 1
    if version >= _COUNTER_SEMANTICS_VERSION:
        return raw_counter
    return raw_counter + 1


def maybe_skip_resume_batches(
    data_iterator,
    *,
    resume_from: str | None,
    epoch: int,
    start_epoch: int,
    resume_batch_idx: int,
):
    """Skip already-processed micro-batches when resuming an in-progress epoch.

    Returns a tuple of (iterator, did_skip).
    """
    if resume_from and epoch == start_epoch and resume_batch_idx > 0:
        return islice(data_iterator, resume_batch_idx, None), True
    return data_iterator, False


_TRAIN_MODE_COMPILED = "COMPILED"
_TRAIN_MODE_EAGER = "EAGER"


def resolve_epoch_train_mode(
    *,
    compiled_step_base_enabled: bool,
    gan_enabled: bool,
    gan_active: bool,
    previous_mode: Literal["COMPILED", "EAGER"] | None,
    experimental_compiled_gan: bool = False,
) -> tuple[Literal["COMPILED", "EAGER"], bool]:
    """Resolve training mode for an epoch with one-way COMPILED->EAGER semantics.

    Rules:
    - If compiled mode is globally blocked (debug, nan-skip, grad accumulation), use EAGER.
    - If GAN is active for this epoch, use EAGER — unless experimental_compiled_gan is True.
    - Once EAGER is entered, do not switch back to COMPILED in later epochs
      (unless experimental_compiled_gan keeps compiled mode through GAN activation).
    """
    if not experimental_compiled_gan and previous_mode == _TRAIN_MODE_EAGER:
        return _TRAIN_MODE_EAGER, False
    if not compiled_step_base_enabled:
        return _TRAIN_MODE_EAGER, False
    if gan_enabled and gan_active and not experimental_compiled_gan:
        return _TRAIN_MODE_EAGER, False
    return _TRAIN_MODE_COMPILED, True


def validate_checkpoint_dir(
    checkpoint_dir: Path,
    strict: bool = True,
    *,
    validate_load: bool = False,
) -> dict:
    """Validate checkpoints in a directory and return a resume plan.

    Args:
        checkpoint_dir: Directory containing checkpoints
        strict: If True, raise on any validation errors
        validate_load: If True, attempt to load checkpoint weights for integrity
    """
    manifest = CheckpointManifest()
    report = {
        "total": 0,
        "valid": 0,
        "invalid": [],
        "latest_path": None,
        "latest_state": None,
        "last_completed_epoch": -1,
        "resume_epoch": 0,
        "resume_batch": 0,
        "resume_global_step": None,
        "warnings": [],
    }

    if not checkpoint_dir.exists():
        return report

    tmp_files = [p for p in checkpoint_dir.iterdir() if manifest.is_temporary(p)]
    for tmp in tmp_files:
        report["invalid"].append((tmp, "temporary checkpoint residue"))

    ckpt_files = sorted(
        [
            p
            for p in checkpoint_dir.glob(f"*{manifest.weights_ext}")
            if not manifest.is_temporary(p) and not _is_disc_weights(p, manifest)
        ],
        key=lambda p: p.stat().st_mtime,
    )

    records: list[CheckpointRecord] = []

    for ckpt in ckpt_files:
        report["total"] += 1
        state_path = manifest.state_path(ckpt)
        record = CheckpointRecord(path=ckpt, state_path=state_path, mtime=ckpt.stat().st_mtime)

        if not ckpt.exists():
            record.errors.append("weights missing")
        elif ckpt.stat().st_size == 0:
            record.errors.append("weights file is empty")

        if not state_path.exists():
            record.errors.append("state missing")
        elif state_path.stat().st_size == 0:
            record.errors.append("state file is empty")

        if record.errors:
            records.append(record)
            report["invalid"].append((ckpt, "; ".join(record.errors)))
            continue

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            record.errors.append(f"state load error: {e}")
            records.append(record)
            report["invalid"].append((ckpt, "; ".join(record.errors)))
            continue

        record.state = state
        kind = state.get("kind")
        epoch = state.get("epoch")
        last_completed = state.get("last_completed_epoch")
        batch_idx = state.get("micro_batches_completed", state.get("batch_idx"))
        global_step = state.get("optimizer_steps_completed", state.get("global_step"))

        if kind not in _CHECKPOINT_KINDS:
            record.errors.append("missing/invalid kind")
        if not isinstance(epoch, int):
            record.errors.append("missing/invalid epoch")
        if not isinstance(last_completed, int):
            record.errors.append("missing/invalid last_completed_epoch")
        if batch_idx is not None and not isinstance(batch_idx, int):
            record.errors.append("invalid batch_idx")
        if global_step is not None and not isinstance(global_step, int):
            record.errors.append("invalid global_step")

        record.kind = kind if isinstance(kind, str) else None
        record.epoch = epoch if isinstance(epoch, int) else None
        record.batch_idx = batch_idx if isinstance(batch_idx, int) else None
        record.global_step = global_step if isinstance(global_step, int) else None
        record.last_completed_epoch = last_completed if isinstance(last_completed, int) else None

        expected = manifest.expected_from_name(ckpt)
        if expected:
            expected_kind = expected.get("kind")
            expected_kinds = expected.get("kinds")
            if expected_kind and kind != expected_kind:
                record.errors.append(f"kind mismatch (expected {expected_kind})")
            if expected_kinds and kind not in expected_kinds:
                record.errors.append(f"kind mismatch (expected {sorted(expected_kinds)})")
            if expected.get("epoch") is not None and isinstance(epoch, int):
                if epoch != expected["epoch"]:
                    record.errors.append(f"epoch mismatch (state {epoch} vs name {expected['epoch']})")
            if expected.get("global_step") is not None and isinstance(global_step, int):
                if global_step != expected["global_step"]:
                    record.errors.append(
                        f"global_step mismatch (state {global_step} vs name {expected['global_step']})"
                    )
        else:
            record.errors.append("unrecognized checkpoint filename")

        if isinstance(kind, str) and isinstance(epoch, int) and isinstance(last_completed, int):
            if kind in _COMPLETED_KINDS:
                if last_completed < epoch:
                    record.errors.append("completed kind but last_completed_epoch < epoch")
            elif kind in _IN_PROGRESS_KINDS:
                if last_completed > epoch - 1:
                    record.errors.append("in-progress kind but last_completed_epoch too high")
            if kind in _IN_PROGRESS_KINDS and record.batch_idx is None:
                record.errors.append("in-progress checkpoint missing batch_idx")
            if kind == "step" and record.global_step is None:
                record.errors.append("step checkpoint missing global_step")

        checkpoint_kind = state.get("checkpoint_kind")
        if checkpoint_kind is not None:
            expected_checkpoint_kind = "end_of_epoch" if kind in _COMPLETED_KINDS else "in_progress"
            if checkpoint_kind != expected_checkpoint_kind:
                record.errors.append("checkpoint_kind mismatch")

        if state.get("current_epoch") is not None and state.get("current_epoch") != epoch:
            record.errors.append("current_epoch mismatch")
        if state.get("last_saved_global_step") is not None and state.get("last_saved_global_step") != global_step:
            record.errors.append("last_saved_global_step mismatch")
        if state.get("last_saved_batch_idx") is not None and state.get("last_saved_batch_idx") != batch_idx:
            record.errors.append("last_saved_batch_idx mismatch")

        if validate_load and not record.errors:
            try:
                _ = mx.load(str(ckpt))
            except Exception as e:
                record.errors.append(f"weights load error: {e}")

        records.append(record)
        if record.valid:
            report["valid"] += 1
            if record.last_completed_epoch is not None:
                report["last_completed_epoch"] = max(report["last_completed_epoch"], record.last_completed_epoch)
        else:
            report["invalid"].append((ckpt, "; ".join(record.errors)))

    marker_files = list(checkpoint_dir.glob(f"epoch_*{manifest.epoch_complete_suffix}"))
    marker_epochs = {}
    for marker in marker_files:
        if marker.stat().st_size == 0:
            report["invalid"].append((marker, "epoch complete marker is empty"))
            continue
        marker_epoch = manifest.marker_epoch(marker)
        if marker_epoch is None:
            report["invalid"].append((marker, "unrecognized epoch complete marker name"))
            continue
        marker_epochs[marker_epoch] = marker

    if marker_epochs:
        completed_epochs = {
            rec.epoch for rec in records if rec.valid and rec.kind in _COMPLETED_KINDS and rec.epoch is not None
        }
        for epoch_idx, marker in marker_epochs.items():
            if epoch_idx not in completed_epochs:
                report["invalid"].append((marker, "epoch complete marker without valid end-of-epoch checkpoint"))

    valid_records = [rec for rec in records if rec.valid]
    if valid_records:
        latest = max(valid_records, key=_record_sort_key)
        report["latest_path"] = latest.path
        report["latest_state"] = latest.state
        if latest.state:
            report["resume_epoch"] = compute_resume_epoch(latest.state)
            report["resume_batch"] = resolve_resume_batch_count(latest.state)
            report["resume_global_step"] = latest.state.get(
                "optimizer_steps_completed", latest.state.get("global_step")
            )

    # Detect monotonicity issues across valid checkpoints (by modification time).
    valid_by_time = sorted(valid_records, key=lambda rec: rec.mtime)
    last_epoch_seen = None
    last_step_seen = None
    last_completed_seen = None
    for rec in valid_by_time:
        if rec.epoch is not None:
            if last_epoch_seen is not None and rec.epoch < last_epoch_seen:
                report["invalid"].append((rec.path, "epoch decreased relative to earlier checkpoint"))
            last_epoch_seen = rec.epoch
        if rec.global_step is not None:
            if last_step_seen is not None and rec.global_step < last_step_seen:
                report["invalid"].append((rec.path, "global_step decreased relative to earlier checkpoint"))
            last_step_seen = rec.global_step
        if rec.last_completed_epoch is not None:
            if last_completed_seen is not None and rec.last_completed_epoch < last_completed_seen:
                report["invalid"].append((rec.path, "last_completed_epoch decreased relative to earlier checkpoint"))
            last_completed_seen = rec.last_completed_epoch

    data_ckpt = checkpoint_dir / "data_checkpoint.json"
    if data_ckpt.exists():
        try:
            with open(data_ckpt, "r", encoding="utf-8") as f:
                data_state = json.load(f)
            data_epoch = data_state.get("epoch")
            data_batch = data_state.get("batch_idx")
            if not isinstance(data_epoch, int) or data_epoch < 0:
                report["invalid"].append((data_ckpt, "data checkpoint has invalid epoch"))
            if not isinstance(data_batch, int) or data_batch < 0:
                report["invalid"].append((data_ckpt, "data checkpoint has invalid batch_idx"))
            if report["latest_state"] and isinstance(data_epoch, int):
                latest_epoch = report["latest_state"].get("epoch")
                if isinstance(latest_epoch, int) and data_epoch > latest_epoch:
                    report["invalid"].append((data_ckpt, "data checkpoint epoch exceeds latest model checkpoint epoch"))
        except Exception as e:
            report["invalid"].append((data_ckpt, f"data checkpoint load error: {e}"))

    if report["invalid"] and strict:
        msgs = [f"{p.name}: {reason}" for p, reason in report["invalid"]]
        raise RuntimeError(
            "Checkpoint validation failed:\n  "
            + "\n  ".join(msgs)
            + "\nRemediation: remove or move corrupted checkpoints/markers and retry."
        )

    return report


def _write_epoch_complete_marker(checkpoint_dir: Path, epoch: int, checkpoint_path: Path) -> bool:
    """Write an epoch completion marker after a successful end-of-epoch checkpoint."""
    manifest = CheckpointManifest()
    marker_path = checkpoint_dir / f"epoch_{epoch + 1:03d}{manifest.epoch_complete_suffix}"
    tmp_marker = marker_path.with_name(f"{marker_path.name}.tmp")
    marker_state = {
        "epoch": epoch,
        "checkpoint": checkpoint_path.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(tmp_marker, "w", encoding="utf-8") as f:
            json.dump(marker_state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_marker.replace(marker_path)
        return True
    except Exception as e:
        print(f"⚠️  Failed to write epoch completion marker: {e}")
        return False


def save_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    epoch: int,
    batch_idx: int | None = None,
    global_step: int | None = None,
    loss: float,
    best_valid_loss: float,
    config: dict,
    optimizer: optim.Optimizer | None = None,
    discriminator: nn.Module | None = None,
    disc_optimizer: optim.Optimizer | None = None,
    last_completed_epoch: int = -1,
    kind: str = "epoch_end",
    raise_on_error: bool = False,
) -> bool:
    """Save a training checkpoint with model weights, training state, and optimizer state.

    Args:
        model: Model to save
        path: Path to checkpoint file (.safetensors)
        epoch: Current epoch index (0-based)
        batch_idx: Number of micro-batches completed within the current epoch
        global_step: Number of optimizer updates completed globally
        loss: Current training loss
        best_valid_loss: Best validation loss so far
        config: Training configuration dict
        optimizer: Optional optimizer to save state from
        last_completed_epoch: Last fully completed epoch index (-1 if none)
        kind: Checkpoint kind: step | epoch_end | best | final | interrupted
        raise_on_error: Raise on failure instead of returning False
    Returns:
        True if checkpoint was saved and validated, False otherwise.
    """
    from mlx.utils import tree_flatten

    manifest = CheckpointManifest()

    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_weights = path.with_name(f"{path.stem}.tmp{path.suffix}")

        # Flatten nested params for safetensors
        params = model.parameters()
        flat_params = tree_flatten(params)
        weights = {k: v for k, v in flat_params}

        # Ensure tensors are materialized before writing and retry once if needed
        if weights:
            mx.eval(*weights.values())
        mx.save_safetensors(str(tmp_weights), weights)
        if not tmp_weights.exists():
            mx.save_safetensors(str(tmp_weights), weights)

        # Prepare optimizer state for serialization
        optimizer_state_dict = {}
        if optimizer is not None and hasattr(optimizer, "state") and optimizer.state:
            try:
                # Flatten optimizer state for JSON serialization
                flat_state = tree_flatten(optimizer.state)
                # Convert arrays to lists, preserve scalar types (int, float, bool)
                for k, v in flat_state:
                    if isinstance(v, mx.array):
                        optimizer_state_dict[k] = v.tolist()  # Array → list
                    else:
                        optimizer_state_dict[k] = v  # Scalar → keep as-is
            except Exception as e:
                print(f"⚠️  Failed to serialize optimizer state: {e}")

        disc_optimizer_state_dict = {}
        if disc_optimizer is not None and hasattr(disc_optimizer, "state") and disc_optimizer.state:
            try:
                flat_state = tree_flatten(disc_optimizer.state)
                for k, v in flat_state:
                    if isinstance(v, mx.array):
                        disc_optimizer_state_dict[k] = v.tolist()
                    else:
                        disc_optimizer_state_dict[k] = v
            except Exception as e:
                print(f"⚠️  Failed to serialize discriminator optimizer state: {e}")

        checkpoint_kind = "end_of_epoch" if kind in _COMPLETED_KINDS else "in_progress"

        # Save training state and metadata
        state_path = manifest.state_path(path)
        tmp_state_path = state_path.with_name(f"{state_path.stem}.tmp{state_path.suffix}")
        state = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "micro_batches_completed": batch_idx,
            "global_step": global_step,
            "optimizer_steps_completed": global_step,
            "loss": loss,
            "best_valid_loss": best_valid_loss,
            "config": config,
            "optimizer_state": optimizer_state_dict,
            "disc_optimizer_state": disc_optimizer_state_dict,
            "last_completed_epoch": last_completed_epoch,
            "kind": kind,
            "checkpoint_kind": checkpoint_kind,
            "counter_semantics_version": _COUNTER_SEMANTICS_VERSION,
            "batch_unit": "microbatch_count",
            "step_unit": "optimizer_step",
            "current_epoch": epoch,
            "last_saved_global_step": global_step,
            "last_saved_batch_idx": batch_idx,
        }
        with open(tmp_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        tmp_weights.replace(path)
        tmp_state_path.replace(state_path)

        # Save discriminator weights after main checkpoint is safely written
        if discriminator is not None:
            disc_path = _disc_weights_path(path)
            tmp_disc = disc_path.with_name(f"{disc_path.stem}.tmp{disc_path.suffix}")
            disc_params = discriminator.parameters()
            flat_disc = tree_flatten(disc_params)
            disc_weights = {k: v for k, v in flat_disc}
            if disc_weights:
                mx.eval(*disc_weights.values())
            mx.save_safetensors(str(tmp_disc), disc_weights)
            if not tmp_disc.exists():
                mx.save_safetensors(str(tmp_disc), disc_weights)
            tmp_disc.replace(disc_path)

        if not _validate_checkpoint_pair(path, manifest=manifest):
            msg = f"Checkpoint validation failed after save: {path.name}"
            if raise_on_error:
                raise RuntimeError(msg)
            print(f"⚠️  {msg}")
            return False

        if optimizer_state_dict:
            print(f"✅ Saved checkpoint with optimizer state: {path.name}")
        return True
    except Exception as e:
        if raise_on_error:
            raise
        print(f"❌ Failed to save checkpoint {Path(path).name}: {e}")
        return False


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: optim.Optimizer | None = None,
    discriminator: nn.Module | None = None,
    disc_optimizer: optim.Optimizer | None = None,
) -> dict:
    """Load a training checkpoint and restore model weights and optimizer state.

    Args:
        model: Model to load weights into
        path: Path to checkpoint file
        optimizer: Optional optimizer to restore state into

    Returns:
        Training state dict containing epoch, loss, etc.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    ckpt_path = Path(path)
    manifest = CheckpointManifest()

    # Validate checkpoint pair before loading
    if not _validate_checkpoint_pair(ckpt_path, manifest=manifest):
        print(f"⚠️  Checkpoint validation failed: {ckpt_path.name}")
        return {}

    try:
        # Load weights
        weights = mx.load(str(ckpt_path))

        # Align checkpoint weights with model's parameter tree
        flat_model = tree_flatten(model.parameters())
        pairs = []
        missing = []
        for name, param in flat_model:
            if isinstance(weights, dict) and name in weights:
                pairs.append((name, weights[name]))
            else:
                pairs.append((name, param))
                missing.append(name)

        nested_weights = tree_unflatten(pairs)
        model.update(nested_weights)

        if missing:
            print(f"⚠️  {len(missing)} parameters were missing in checkpoint")

        # Load training state
        state_path = manifest.state_path(ckpt_path)
        state = {}
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)

        # Restore optimizer state if provided
        if optimizer is not None and "optimizer_state" in state:
            try:
                optimizer_state_dict = state.get("optimizer_state", {})
                if optimizer_state_dict:
                    # Convert all values back to mx.array (including scalars from .tolist())
                    restored = {}
                    for k, v in optimizer_state_dict.items():
                        # All serialized optimizer state values should become mx.array
                        restored[k] = mx.array(v)
                    # Reconstruct optimizer state from flat dict
                    state_pairs = list(restored.items())
                    nested_state = tree_unflatten(state_pairs)
                    optimizer.state = nested_state
                    print("✅ Restored optimizer state from checkpoint")
            except Exception as e:
                print(f"⚠️  Failed to restore optimizer state: {e}")

        # Restore discriminator weights/optimizer if provided
        if discriminator is not None:
            disc_path = _disc_weights_path(ckpt_path)
            if disc_path.exists():
                try:
                    disc_weights = mx.load(str(disc_path))
                    flat_disc = tree_flatten(discriminator.parameters())
                    disc_pairs = []
                    missing_disc = []
                    for name, param in flat_disc:
                        if isinstance(disc_weights, dict) and name in disc_weights:
                            disc_pairs.append((name, disc_weights[name]))
                        else:
                            disc_pairs.append((name, param))
                            missing_disc.append(name)
                    discriminator.update(tree_unflatten(disc_pairs))
                    if missing_disc:
                        print(f"⚠️  {len(missing_disc)} discriminator parameters missing in checkpoint")
                except Exception as e:
                    print(f"⚠️  Failed to load discriminator weights: {e}")
            else:
                print(f"⚠️  Discriminator checkpoint missing: {disc_path.name}")

        if disc_optimizer is not None and "disc_optimizer_state" in state:
            try:
                disc_state_dict = state.get("disc_optimizer_state", {})
                if disc_state_dict:
                    restored = {k: mx.array(v) for k, v in disc_state_dict.items()}
                    disc_pairs = list(restored.items())
                    disc_nested = tree_unflatten(disc_pairs)
                    disc_optimizer.state = disc_nested
                    print("✅ Restored discriminator optimizer state from checkpoint")
            except Exception as e:
                print(f"⚠️  Failed to restore discriminator optimizer state: {e}")

        epoch = state.get("epoch", 0)
        kind = state.get("kind", "epoch_end")
        completed_kinds = {"epoch_end", "best", "best_final", "final"}
        last_completed = state.get("last_completed_epoch", epoch if kind in completed_kinds else epoch - 1)
        print(f"✅ Loaded checkpoint from epoch {epoch} (kind={kind}, last_completed={last_completed})")
        return state

    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        return {}


def cleanup_checkpoints(
    checkpoint_dir: Path,
    save_total_limit: int,
    keep_best: bool = True,
) -> None:
    """Remove old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        save_total_limit: Maximum number of checkpoints to keep
        keep_best: If True, always keep best.safetensors (doesn't count towards limit)
    """
    if save_total_limit <= 0:
        return

    manifest = CheckpointManifest()

    # Find all checkpoint files (epoch_*.safetensors and step_*.safetensors)
    ckpt_files = []
    for pattern in ["epoch_*.safetensors", "step_*.safetensors"]:
        ckpt_files.extend([p for p in checkpoint_dir.glob(pattern) if not _is_disc_weights(p, manifest)])

    # Sort by modification time (oldest first)
    ckpt_files.sort(key=lambda p: p.stat().st_mtime)

    # Calculate how many to remove
    num_to_remove = len(ckpt_files) - save_total_limit

    if num_to_remove <= 0:
        return

    # Remove oldest checkpoints
    for ckpt_path in ckpt_files[:num_to_remove]:
        # Remove the safetensors file
        ckpt_path.unlink(missing_ok=True)
        # Remove discriminator weights if present
        _disc_weights_path(ckpt_path).unlink(missing_ok=True)

        # Also remove the accompanying state.json
        state_path = manifest.state_path(ckpt_path)
        state_path.unlink(missing_ok=True)

        # Remove epoch completion marker if present
        marker_epoch = manifest.expected_from_name(ckpt_path).get("epoch")
        if marker_epoch is not None:
            marker_path = checkpoint_dir / f"epoch_{marker_epoch + 1:03d}{manifest.epoch_complete_suffix}"
            marker_path.unlink(missing_ok=True)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the most recent checkpoint in the checkpoint directory.

    Returns the latest valid checkpoint based on metadata and modification time.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to most recent checkpoint, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None

    manifest = CheckpointManifest()
    candidates = [
        p
        for p in checkpoint_dir.glob(f"*{manifest.weights_ext}")
        if not manifest.is_temporary(p) and not _is_disc_weights(p, manifest)
    ]

    valid_pairs: list[Path] = []
    for ckpt in candidates:
        state_path = manifest.state_path(ckpt)
        if not ckpt.exists() or ckpt.stat().st_size == 0:
            continue
        if not state_path.exists() or state_path.stat().st_size == 0:
            continue
        valid_pairs.append(ckpt)

    if not valid_pairs:
        return None

    # Fast path: use latest mtime without loading large state JSON files.
    return max(valid_pairs, key=lambda p: p.stat().st_mtime)


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
    print(f"  Mixed precision (FP16): {'enabled' if use_fp16 else 'disabled'}")

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
        vad_band_mask = mx.array(0.0)
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

        out = model(noisy_spec, feat_erb, feat_spec)
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
            disc_fake, fake_feats = discriminator(gan_out_wav)
            disc_real, real_feats = discriminator(mx.stop_gradient(gan_clean_wav))
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
            speech_loss = mx.array(0.0)
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
        return total_loss, out

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

            out = model(noisy_spec, feat_erb, feat_spec)
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
                disc_fake, fake_feats = discriminator(gan_out_wav)
                disc_real, real_feats = discriminator(mx.stop_gradient(gan_clean_wav))
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
                speech_loss = mx.array(0.0)
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

            return total_loss, out

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
        """Run a diagnostic forward pass with detailed finite checks."""
        if debugger is None:
            return
        out = model((noisy_real, noisy_imag), feat_erb, feat_spec)
        debugger.check("model.out_real", out[0], debug_ctx)
        debugger.check("model.out_imag", out[1], debug_ctx)
        spec_loss = spectral_loss(out, (clean_real, clean_imag))
        debugger.check("spec_loss", spec_loss, debug_ctx)
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
            debugger.check("mrstft_loss", mrstft_loss, debug_ctx)
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
            debugger.check("gan_g_loss", gan_g_loss, debug_ctx)
            if feature_match_loss is not None and gan_fm_weight > 0:
                fm_loss = feature_match_loss(real_feats, fake_feats)
                debugger.check("gan_fm_loss", fm_loss, debug_ctx)
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
                debug=debugger,
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
                debug=debugger,
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
                debug=debugger,
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
                    debug=debugger,
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
                debug=debugger,
                debug_ctx=debug_ctx,
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
        (loss, out), grads = loss_and_grad(
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
        return loss, out

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
        (loss, out), grads = loss_and_grad(
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
        return loss, out, grads

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
            (loss, out), grads = _lag_gan(
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
            return loss, out

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
            (loss, out), grads = _lag_gan(
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
            return loss, out, grads

        compiled_gan_step = _compiled_gan_step
        compiled_gan_loss_and_grad_step = _compiled_gan_loss_and_grad_step

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

        valid_pbar = tqdm(
            valid_loader,
            total=valid_steps,
            desc=label,
            unit="batch",
            leave=False,
            **_TQDM_KWARGS,
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
            mrstft_loss = mx.array(0.0)
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

            awesome_loss = mx.array(0.0)
            awesome_speech = mx.array(0.0)
            awesome_noise = mx.array(0.0)
            awesome_smooth = mx.array(0.0)
            music_suppression_loss = mx.array(0.0)
            mask_saturation_loss = mx.array(0.0)
            mask = mx.array(0.0)
            proxy_frame = mx.array(0.0)
            speech_ratio = mx.array(0.0)
            music_gate = mx.array(0.0)
            musicness = mx.array(0.0)
            mod_energy = mx.array(0.0)
            energy_boost = mx.array(0.0)
            snr_boost = mx.array(0.0)

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
                speech_loss = mx.array(0.0)
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
                vad_loss = mx.array(0.0)
                speech_loss = mx.array(0.0)
                p_ref = mx.array(0.0)
                p_out = mx.array(0.0)
                gate = mx.array(0.0)

            vad_reg_loss = mx.array(0.0)
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
                with open(ablation_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ablation_row) + "\n")

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
        gan_active = gan_enabled and gan_scale > 0.0

        # GAN epochs MUST sync every step to prevent lazy-graph accumulation.
        # With eval_frequency > 1 the discriminator forward/backward graphs
        # pile up across batches, easily exceeding unified-memory limits.
        epoch_eval_frequency = eval_frequency
        if gan_active and eval_frequency > 1:
            epoch_eval_frequency = 1
            if epoch == gan_start_epoch:
                print(
                    f"  GAN active: overriding eval_frequency {eval_frequency} → 1 "
                    "(mandatory per-step sync to prevent OOM from graph accumulation)"
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
        accumulated_loss = mx.array(0.0)
        micro_batches_in_accum = 0

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

        train_pbar = tqdm(
            enumerate(islice(data_iterator, train_total)),
            total=train_total,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=True,
            **_TQDM_KWARGS,
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

            # Convert to FP16 if enabled (mixed precision training)
            if use_fp16:
                noisy_real = noisy_real.astype(mx.float16)
                noisy_imag = noisy_imag.astype(mx.float16)
                clean_real = clean_real.astype(mx.float16)
                clean_imag = clean_imag.astype(mx.float16)
                feat_erb = feat_erb.astype(mx.float16)
                feat_spec = feat_spec.astype(mx.float16)

            current_batch_size = noisy_real.shape[0]

            # Update learning rate from schedule (must be done outside compiled step)
            current_lr = schedule(global_step)
            optimizer.learning_rate = current_lr

            warmup_frac = 1.0
            if use_vad_loss and vad_warmup_steps > 0:
                warmup_frac = min(1.0, global_step / max(vad_warmup_steps, 1))

            vad_weight = epoch_vad_loss_weight * warmup_frac
            speech_weight = epoch_vad_speech_loss_weight * warmup_frac
            vad_weight_mx = mx.array(vad_weight, dtype=mx.float32)
            speech_weight_mx = mx.array(speech_weight, dtype=mx.float32)
            awesome_frac = 1.0
            if (use_awesome_loss or use_pipeline_awesome_loss) and awesome_warmup_steps > 0:
                awesome_frac = min(1.0, global_step / max(awesome_warmup_steps, 1))
            awesome_weight = epoch_awesome_loss_weight * awesome_frac
            awesome_weight_mx = mx.array(awesome_weight, dtype=mx.float32)

            apply_vad_reg = False
            if use_vad_train_reg:
                if vad_train_every_steps > 0 and global_step % vad_train_every_steps == 0:
                    apply_vad_reg = True
                elif vad_train_prob > 0:
                    apply_vad_reg = random.random() < vad_train_prob
            vad_reg_weight = vad_weight if apply_vad_reg else 0.0
            vad_reg_weight_mx = mx.array(vad_reg_weight, dtype=mx.float32)
            gan_weight_mx = mx.array(gan_weight, dtype=mx.float32)
            fm_weight_mx = mx.array(fm_weight, dtype=mx.float32)

            # Track whether optimizer was updated this iteration (for gradient accumulation)
            did_optimizer_update = False

            # Forward, backward, and update (either compiled or standard)
            fwd_start = time.perf_counter()

            model_out = None
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
                    expected_dtype=mx.float16 if use_fp16 else mx.float32,
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
                    loss, model_out, grads = active_compiled_lag(
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
                                f"(step={global_step})"
                            )
                        accumulated_grads = None
                        accumulated_loss = mx.array(0.0)
                        micro_batches_in_accum = 0

                    if should_sync:
                        if did_optimizer_update:
                            mx.eval(loss, model.parameters(), optimizer.state)
                        else:
                            mx.eval(loss)
                else:
                    # Fully compiled training step (fwd+bwd+update) for best throughput.
                    did_optimizer_update = True
                    loss, model_out = active_compiled_step(
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
                        (eager_loss, _), _ = loss_and_grad_gan(
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
                (loss, model_out), grads = loss_and_grad(
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
                loss_finite = bool(mx.all(mx.isfinite(loss)))
                if debugger is not None and not loss_finite:
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
                    debugger.check("train.loss", loss, debug_ctx)

                # Grad finiteness is checked AFTER clipping (below).
                # Non-finite raw grads are expected during early GAN epochs;
                # clip_grad_norm_tree zeros them, and we skip the update.
                skip_update = False
                if not loss_finite:
                    skip_update = True
                    tqdm.write("⚠️  Non-finite loss detected; skipping optimizer update")
                # Only sync periodically
                should_sync = (batch_idx + 1) % epoch_eval_frequency == 0
                if should_sync:
                    mx.eval(loss)

                if not skip_update:
                    # Accumulate gradients (for grad_accumulation_steps > 1)
                    accumulated_grads = accumulate_grads(accumulated_grads, grads)
                    accumulated_loss = accumulated_loss + loss
                    micro_batches_in_accum += 1

                    # Check if accumulation window is complete
                    is_accum_complete = micro_batches_in_accum >= grad_accumulation_steps
                    if is_accum_complete:
                        did_optimizer_update = True

                        # Scale by 1/grad_accumulation_steps for proper averaging
                        final_grads = scale_grads(accumulated_grads, 1.0 / grad_accumulation_steps)

                        # Gradient clipping (returns clipped grads and norm as MLX array)
                        # clip_grad_norm zeros grads when norm is non-finite.
                        if max_grad_norm > 0:
                            final_grads, grad_norm_arr = clip_grad_norm(final_grads, max_grad_norm)
                            if should_sync:
                                grad_norm = float(grad_norm_arr)

                        # Post-clip finite check: skip update if grads are
                        # still non-finite after clipping (shouldn't happen
                        # with the zeroing logic, but guard defensively).
                        grads_finite = _tree_all_finite(final_grads)
                        if not grads_finite:
                            if debugger is not None:
                                debugger.check_tree("train.grads_clipped", final_grads, debug_ctx)
                            tqdm.write(
                                "⚠️  Non-finite grads after clipping; skipping optimizer update "
                                f"(step={global_step})"
                            )
                            did_optimizer_update = False
                        else:
                            # Update parameters
                            optimizer.update(model, final_grads)

                        # Reset accumulator for next window
                        accumulated_grads = None
                        accumulated_loss = mx.array(0.0)
                        micro_batches_in_accum = 0

                # Only sync periodically for better throughput
                if should_sync:
                    mx.eval(loss, model.parameters(), optimizer.state)

            pred_spec_for_logging = None
            if model_out is not None:
                pred_spec_for_logging = (
                    mx.stop_gradient(model_out[0]),
                    mx.stop_gradient(model_out[1]),
                )

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

                        def disc_loss_wrapper(disc):
                            real_out, _ = disc(clean_wav_d)
                            fake_out, _ = disc(pred_wav_d)
                            total_loss, _, _ = disc_loss_fn(real_out, fake_out)
                            return total_loss

                        disc_loss, disc_grads = nn.value_and_grad(discriminator, disc_loss_wrapper)(discriminator)

                        if gan_disc_grad_clip > 0:
                            disc_grads, _ = clip_grad_norm(disc_grads, gan_disc_grad_clip)

                        disc_optimizer.update(discriminator, disc_grads)

                        if should_sync:
                            mx.eval(disc_loss, discriminator.parameters(), disc_optimizer.state)
                            gan_d_loss_val = float(disc_loss)
                            train_gan_d_updates += 1

            fwd_time = time.perf_counter() - fwd_start
            total_forward_time += fwd_time

            # Only convert loss to float when synced (avoids blocking)
            if should_sync:
                loss_val = float(loss)
                if not math.isfinite(loss_val):
                    raise FloatingPointError(
                        "Non-finite loss detected "
                        f"(epoch={epoch}, batch={batch_idx}, step={global_step}). "
                        "Re-run with --debug-numerics for detailed diagnostics."
                    )
                train_loss += loss_val * eval_frequency  # Approximate accumulated loss
                if gan_active and gan_d_loss_val:
                    train_gan_d_loss += gan_d_loss_val * eval_frequency

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
                needs_model_out = (
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
                    train_spec_loss += spec_loss_val * eval_frequency
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
                        train_mrstft_loss += mrstft_loss_val * eval_frequency
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
                        train_gan_g_loss += gan_g_loss_val * eval_frequency
                        if feature_match_loss is not None and gan_fm_weight > 0:
                            gan_fm_loss_val = float(feature_match_loss(real_feats, fake_feats))
                            train_gan_fm_loss += gan_fm_loss_val * eval_frequency

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
                    speech_loss = mx.array(0.0)
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

                    train_vad_loss += vad_loss_val * eval_frequency
                    train_speech_loss += speech_loss_val * eval_frequency
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

                    train_awesome_loss += awesome_loss_val * eval_frequency
                    train_awesome_speech += awesome_speech_val * eval_frequency
                    train_awesome_noise += awesome_noise_val * eval_frequency
                    train_awesome_smooth += awesome_smooth_val * eval_frequency
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

                if use_pipeline_awesome_loss:
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

                    train_awesome_loss += awesome_loss_val * eval_frequency
                    train_awesome_speech += awesome_speech_val * eval_frequency
                    train_awesome_noise += awesome_noise_val * eval_frequency
                    train_awesome_smooth += awesome_smooth_val * eval_frequency
                    train_music_supp_loss += music_supp_loss_val * eval_frequency
                    train_mask_sat_loss += mask_sat_loss_val * eval_frequency
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

                if use_vad_train_reg and apply_vad_reg:
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
                    train_vad_reg_loss += vad_reg_loss_val * eval_frequency

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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train DfNet4 with dynamic on-the-fly mixing. "
            "--config refers to the dataset/mixer JSON config, "
            "--train-config is the train.py-style INI config, "
            "and --run-config refers to CLI/runtime settings (TOML)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data sources (priority: cache_dir > config > file lists)
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Path to pre-built audio cache (from build_audio_cache.py)",
    )
    parser.add_argument(
        "--speech-list",
        type=str,
        help="Path to file containing speech file paths (one per line)",
    )
    parser.add_argument(
        "--noise-list",
        type=str,
        help="Path to file containing noise file paths (one per line)",
    )
    parser.add_argument(
        "--rir-list",
        type=str,
        help="Path to file containing RIR file paths (one per line)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to dataset/mixer JSON config file (alternative to file lists)",
    )
    parser.add_argument(
        "--run-config",
        type=str,
        help="Path to run-config TOML file (CLI/runtime settings)",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        help="Path to train.py-compatible INI config (model + training settings)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["entry", "pro", "max", "ultra", "debug"],
        default=None,
        help=(
            "Load a named hardware preset as the base config. "
            "Values from --run-config and explicit CLI flags override preset defaults. "
            "See docs/RUN_CONFIG_PRESETS.md for details."
        ),
    )
    parser.add_argument(
        "--print-run-config",
        action="store_true",
        help="Print a commented run-config TOML example and exit",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--learning-rate-min",
        type=float,
        default=None,
        help="Minimum learning rate for cosine schedule (defaults to 1%% of base)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume from checkpoint. If no path given, auto-finds latest in checkpoint-dir",
    )
    parser.add_argument(
        "--resume-data",
        nargs="?",
        const=True,
        default=False,
        help="Resume data loading state. If no path given, uses data_checkpoint.json in checkpoint-dir",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=1,
        help="Validate every N epochs",
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="epoch",
        choices=["no", "epoch", "steps"],
        help=(
            "Checkpoint save strategy for additional checkpoints: "
            "'no' (only best + required epoch_end), "
            "'epoch' (every epoch), "
            "'steps' (every N steps)"
        ),
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (only when --save-strategy=steps)",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep (oldest removed first, best model always kept)",
    )
    parser.add_argument(
        "--checkpoint-batches",
        type=int,
        default=0,
        help="Save data checkpoint every N batches (0=disabled, for resume)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--p-reverb",
        type=float,
        default=0.5,
        help="Probability of applying reverb",
    )
    parser.add_argument(
        "--p-clipping",
        type=float,
        default=0.0,
        help="Probability of clipping distortion",
    )

    # Other parameters
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--prefetch-size",
        type=int,
        default=8,
        help="Number of batches to prefetch (for MLXDataStream)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--no-mlx-data",
        action="store_true",
        help="Disable mlx-data (use PrefetchDataLoader instead)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=None,
        help="Enable FP16 (half-precision) training for faster performance",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 training (use FP32 for full precision)",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (effective batch = batch_size * grad_accumulation_steps)",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=10,
        help="Sync with GPU every N batches (higher = faster but less responsive logging)",
    )
    parser.add_argument(
        "--backbone",
        "--backbone-type",
        dest="backbone_type",
        type=str,
        choices=["mamba", "gru", "attention"],
        default="mamba",
        help="Backbone type: 'mamba' (parallel scan SSM), 'gru' (recurrent), or 'attention' (fastest backward)",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=["full", "lite"],
        default="full",
        help="Model variant: 'full' or 'lite'",
    )
    parser.add_argument(
        "--snr-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Override base SNR range in dB (e.g., --snr-range -5 40)",
    )
    parser.add_argument(
        "--snr-range-extreme",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Override extreme SNR range in dB (e.g., --snr-range-extreme -20 -5)",
    )
    parser.add_argument(
        "--snr-range-very-low",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Override very-low SNR range in dB (e.g., --snr-range-very-low -30 -20)",
    )
    parser.add_argument(
        "--p-extreme-snr",
        type=float,
        help="Probability of sampling from extreme SNR range (0-1)",
    )
    parser.add_argument(
        "--p-very-low-snr",
        type=float,
        help="Probability of sampling from very-low SNR range (0-1)",
    )
    parser.add_argument(
        "--p-interfer-speech",
        type=float,
        help="Probability of adding interfering speaker (0-1, simulates vocals/competing talker)",
    )
    parser.add_argument(
        "--curriculum-warmup-epochs",
        type=int,
        default=0,
        help="Number of warmup epochs for curriculum learning (0=disabled). "
        "SNR/interferer probabilities ramp linearly from 0 to target values.",
    )
    parser.add_argument(
        "--speech-gain-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Override speech gain range in dB (e.g., --speech-gain-range -12 12)",
    )
    parser.add_argument(
        "--noise-gain-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Override noise gain range in dB (e.g., --noise-gain-range -12 12)",
    )
    parser.add_argument(
        "--dynamic-loss",
        type=str,
        choices=["baseline", "awesome", "pipeline_awesome"],
        default="baseline",
        help="Dynamic loss: 'baseline' (spectral + legacy VAD), 'awesome' (speech-preserving contrastive), or 'pipeline_awesome' (improved speech preservation + music suppression)",
    )
    parser.add_argument(
        "--pipeline-stages",
        type=str,
        default=None,
        help=(
            "JSON array of stage configs with start_epoch and optional overrides. "
            'Example: \'[{"start_epoch":0,"name":"bootstrap","awesome_loss_weight":0.2},'
            '{"start_epoch":5,"name":"refine","awesome_loss_weight":0.4}]\''
        ),
    )
    parser.add_argument(
        "--awesome-loss-weight",
        type=float,
        default=0.4,
        help="Weight for awesome speech-preserving contrastive loss",
    )
    parser.add_argument(
        "--awesome-mask-sharpness",
        type=float,
        default=6.0,
        help="Sharpness for speech/noise dominance mask in awesome loss",
    )
    parser.add_argument(
        "--awesome-warmup-steps",
        type=int,
        default=0,
        help="Warmup steps for ramping awesome loss weight",
    )
    parser.add_argument(
        "--mrstft-factor",
        type=float,
        default=None,
        help="Multi-res STFT loss weight (0 disables)",
    )
    parser.add_argument(
        "--mrstft-gamma",
        type=float,
        default=None,
        help="Multi-res STFT magnitude compression exponent",
    )
    parser.add_argument(
        "--mrstft-f-complex",
        type=float,
        default=None,
        help="Multi-res STFT complex loss weight (None disables)",
    )
    parser.add_argument(
        "--mrstft-fft-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Multi-res STFT FFT sizes (e.g., --mrstft-fft-sizes 512 1024 2048)",
    )
    parser.add_argument(
        "--mrstft-hop-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Multi-res STFT hop sizes (defaults to fft_size//4)",
    )
    parser.add_argument(
        "--gan-enabled",
        action="store_true",
        help="Enable GAN adversarial training",
    )
    parser.add_argument(
        "--gan-start-epoch",
        type=int,
        default=0,
        help="Epoch to start GAN training (0-based)",
    )
    parser.add_argument(
        "--gan-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp GAN weights over N epochs (0 disables ramp)",
    )
    parser.add_argument(
        "--gan-adv-weight",
        type=float,
        default=0.0,
        help="GAN adversarial loss weight",
    )
    parser.add_argument(
        "--gan-fm-weight",
        type=float,
        default=0.0,
        help="GAN feature matching loss weight",
    )
    parser.add_argument(
        "--gan-discriminator",
        type=str,
        default="combined",
        choices=["combined", "mpd", "msd"],
        help="Discriminator type for GAN training",
    )
    parser.add_argument(
        "--gan-mpd-periods",
        type=int,
        nargs="+",
        default=None,
        help="MPD periods for GAN discriminator (e.g., --gan-mpd-periods 2 3 5 7 11)",
    )
    parser.add_argument(
        "--gan-msd-scales",
        type=int,
        default=3,
        help="MSD scales for GAN discriminator",
    )
    parser.add_argument(
        "--gan-disc-lr",
        type=float,
        default=1e-4,
        help="GAN discriminator learning rate",
    )
    parser.add_argument(
        "--gan-disc-weight-decay",
        type=float,
        default=0.0,
        help="GAN discriminator weight decay",
    )
    parser.add_argument(
        "--gan-disc-grad-clip",
        type=float,
        default=1.0,
        help="GAN discriminator gradient clipping",
    )
    parser.add_argument(
        "--gan-disc-update-freq",
        type=int,
        default=1,
        help="Update discriminator every N steps",
    )
    parser.add_argument(
        "--no-vad-proxy",
        action="store_true",
        help="Disable cheap VAD proxy gating in awesome loss",
    )
    parser.add_argument(
        "--vad-loss-weight",
        type=float,
        default=0.05,
        help="Weight for VAD speech-preservation loss (0 disables)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.6,
        help="VAD probability threshold for speech gating",
    )
    parser.add_argument(
        "--vad-margin",
        type=float,
        default=0.05,
        help="Margin for VAD consistency loss",
    )
    parser.add_argument(
        "--vad-speech-loss-weight",
        type=float,
        default=0.0,
        help="Weight for VAD-weighted speech-structure loss",
    )
    parser.add_argument(
        "--vad-warmup-epochs",
        type=int,
        default=5,
        help="Warmup epochs to ramp VAD loss from 0 to target weight",
    )
    parser.add_argument(
        "--vad-snr-gate",
        type=float,
        default=-10.0,
        help="SNR threshold (dB) for VAD gating",
    )
    parser.add_argument(
        "--vad-snr-gate-width",
        type=float,
        default=6.0,
        help="Softness of SNR gating in dB",
    )
    parser.add_argument(
        "--vad-band-low",
        type=float,
        default=300.0,
        help="Low cutoff for speech band in Hz",
    )
    parser.add_argument(
        "--vad-band-high",
        type=float,
        default=3400.0,
        help="High cutoff for speech band in Hz",
    )
    parser.add_argument(
        "--vad-z-threshold",
        type=float,
        default=0.0,
        help="Z-score threshold for VAD sigmoid",
    )
    parser.add_argument(
        "--vad-z-slope",
        type=float,
        default=1.0,
        help="Z-score slope for VAD sigmoid",
    )
    parser.add_argument(
        "--vad-eval-mode",
        type=str,
        choices=["auto", "proxy", "silero", "off"],
        default="auto",
        help="VAD eval mode for periodic metrics (auto enables proxy for awesome loss)",
    )
    parser.add_argument(
        "--vad-eval-every",
        type=int,
        default=1,
        help="Evaluate VAD metrics every N epochs",
    )
    parser.add_argument(
        "--vad-eval-batches",
        type=int,
        default=8,
        help="Number of validation batches used for VAD metrics",
    )
    parser.add_argument(
        "--vad-eval-max-seconds",
        type=float,
        default=0.0,
        help="Max seconds per clip for VAD eval (0 disables)",
    )
    parser.add_argument(
        "--vad-silero-model-path",
        type=str,
        default=None,
        help="Path to silero_vad.onnx (defaults to silero-vad package data)",
    )
    parser.add_argument(
        "--vad-silero-sample-rate",
        type=int,
        default=16000,
        help="Sample rate for Silero VAD evaluation (Hz)",
    )
    parser.add_argument(
        "--vad-train-prob",
        type=float,
        default=0.0,
        help="Probability of applying sparse VAD regularizer per batch (0 disables)",
    )
    parser.add_argument(
        "--vad-train-every-steps",
        type=int,
        default=0,
        help="Apply sparse VAD regularizer every N steps (0 disables)",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit number of training batches per epoch (for fast benchmarking)",
    )
    parser.add_argument(
        "--max-valid-batches",
        type=int,
        default=None,
        help="Limit number of validation batches (for fast benchmarking)",
    )
    parser.add_argument(
        "--eval-sisdr",
        action="store_true",
        help="Compute SI-SDR during validation (slower)",
    )
    parser.add_argument(
        "--check-chkpts",
        action="store_true",
        help="Validate checkpoints and metadata before starting/resuming",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed override (enables deterministic sampling)",
    )
    parser.add_argument(
        "--debug-numerics",
        action="store_true",
        help="Enable numeric debug mode (fail-fast finite checks, short run, deterministic)",
    )
    parser.add_argument(
        "--debug-numerics-no-fail-fast",
        action="store_true",
        help="Disable fail-fast behavior in debug-numerics mode",
    )
    parser.add_argument(
        "--debug-numerics-every",
        type=int,
        default=1,
        help="Check tensors every N steps in debug-numerics mode",
    )
    parser.add_argument(
        "--debug-numerics-dump-dir",
        type=str,
        default=None,
        help="Directory for numeric debug dumps (default: checkpoint_dir/debug_numerics)",
    )
    parser.add_argument(
        "--debug-numerics-dump-arrays",
        action="store_true",
        help="Save small tensor slices alongside numeric debug JSON dumps",
    )
    parser.add_argument(
        "--debug-numerics-max-dumps",
        type=int,
        default=5,
        help="Maximum number of non-finite dumps to write in debug mode",
    )
    parser.add_argument(
        "--nan-skip-batch",
        action="store_true",
        help="Skip optimizer update when loss/grads are non-finite (debug-friendly)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed timing diagnostics and hardware info",
    )

    # Keep parser defaults aligned with RunConfig so --help always reports
    # accurate effective defaults. CLI application still keys off explicit
    # argv presence in _apply_cli_overrides(), so these defaults are display-
    # oriented and do not change precedence semantics.
    default_cfg = RunConfig()
    parser.set_defaults(
        cache_dir=default_cfg.dataset.cache_dir,
        speech_list=default_cfg.dataset.speech_list,
        noise_list=default_cfg.dataset.noise_list,
        rir_list=default_cfg.dataset.rir_list,
        config=default_cfg.dataset.config,
        train_config=default_cfg.training.train_config,
        epochs=default_cfg.training.epochs,
        batch_size=default_cfg.training.batch_size,
        learning_rate=default_cfg.training.learning_rate,
        learning_rate_min=default_cfg.training.learning_rate_min,
        weight_decay=default_cfg.training.weight_decay,
        checkpoint_dir=default_cfg.checkpoint.checkpoint_dir,
        resume=default_cfg.checkpoint.resume,
        resume_data=default_cfg.checkpoint.resume_data,
        validate_every=default_cfg.checkpoint.validate_every,
        save_strategy=default_cfg.checkpoint.save_strategy,
        save_steps=default_cfg.checkpoint.save_steps,
        save_total_limit=default_cfg.checkpoint.save_total_limit,
        checkpoint_batches=default_cfg.checkpoint.checkpoint_batches,
        p_reverb=default_cfg.augmentation.p_reverb,
        p_clipping=default_cfg.augmentation.p_clipping,
        num_workers=default_cfg.dataloader.num_workers,
        prefetch_size=default_cfg.dataloader.prefetch_size,
        max_grad_norm=default_cfg.training.max_grad_norm,
        warmup_epochs=default_cfg.training.warmup_epochs,
        patience=default_cfg.training.patience,
        fp16=default_cfg.training.fp16,
        grad_accumulation_steps=default_cfg.training.grad_accumulation_steps,
        eval_frequency=default_cfg.training.eval_frequency,
        backbone_type=default_cfg.model.backbone_type,
        model_variant=default_cfg.model.variant,
        snr_range=default_cfg.dataset.snr_range,
        snr_range_extreme=default_cfg.dataset.snr_range_extreme,
        snr_range_very_low=default_cfg.dataset.snr_range_very_low,
        p_extreme_snr=default_cfg.dataset.p_extreme_snr,
        p_very_low_snr=default_cfg.dataset.p_very_low_snr,
        p_interfer_speech=default_cfg.dataset.p_interfer_speech,
        curriculum_warmup_epochs=default_cfg.training.curriculum_warmup_epochs,
        speech_gain_range=default_cfg.dataset.speech_gain_range,
        noise_gain_range=default_cfg.dataset.noise_gain_range,
        dynamic_loss=default_cfg.loss.dynamic_loss,
        pipeline_stages=list(default_cfg.loss.pipeline_stages),
        awesome_loss_weight=default_cfg.loss.awesome.loss_weight,
        awesome_mask_sharpness=default_cfg.loss.awesome.mask_sharpness,
        awesome_warmup_steps=default_cfg.loss.awesome.warmup_steps,
        mrstft_factor=default_cfg.loss.mrstft.factor,
        mrstft_gamma=default_cfg.loss.mrstft.gamma,
        mrstft_f_complex=default_cfg.loss.mrstft.f_complex,
        mrstft_fft_sizes=list(default_cfg.loss.mrstft.fft_sizes),
        mrstft_hop_sizes=default_cfg.loss.mrstft.hop_sizes,
        gan_enabled=default_cfg.gan.enabled,
        gan_start_epoch=default_cfg.gan.start_epoch,
        gan_ramp_epochs=default_cfg.gan.ramp_epochs,
        gan_adv_weight=default_cfg.gan.adv_weight,
        gan_fm_weight=default_cfg.gan.fm_weight,
        gan_discriminator=default_cfg.gan.discriminator,
        gan_mpd_periods=list(default_cfg.gan.mpd_periods),
        gan_msd_scales=default_cfg.gan.msd_scales,
        gan_disc_lr=default_cfg.gan.disc_lr,
        gan_disc_weight_decay=default_cfg.gan.disc_weight_decay,
        gan_disc_grad_clip=default_cfg.gan.disc_grad_clip,
        gan_disc_update_freq=default_cfg.gan.disc_update_freq,
        experimental_compiled_gan=default_cfg.gan.experimental_compile,
        vad_loss_weight=default_cfg.vad.loss_weight,
        vad_threshold=default_cfg.vad.threshold,
        vad_margin=default_cfg.vad.margin,
        vad_speech_loss_weight=default_cfg.vad.speech_loss_weight,
        vad_warmup_epochs=default_cfg.vad.warmup_epochs,
        vad_snr_gate=default_cfg.vad.snr_gate_db,
        vad_snr_gate_width=default_cfg.vad.snr_gate_width,
        vad_band_low=default_cfg.vad.band_low_hz,
        vad_band_high=default_cfg.vad.band_high_hz,
        vad_z_threshold=default_cfg.vad.z_threshold,
        vad_z_slope=default_cfg.vad.z_slope,
        vad_eval_mode=default_cfg.vad.eval.mode,
        vad_eval_every=default_cfg.vad.eval.every,
        vad_eval_batches=default_cfg.vad.eval.batches,
        vad_eval_max_seconds=default_cfg.vad.eval.max_seconds,
        vad_silero_model_path=default_cfg.vad.eval.silero_model_path,
        vad_silero_sample_rate=default_cfg.vad.eval.silero_sample_rate,
        vad_train_prob=default_cfg.vad.train.prob,
        vad_train_every_steps=default_cfg.vad.train.every_steps,
        max_train_batches=default_cfg.dataloader.max_train_batches,
        max_valid_batches=default_cfg.dataloader.max_valid_batches,
        eval_sisdr=default_cfg.metrics.eval_sisdr,
        check_chkpts=default_cfg.checkpoint.check_chkpts,
        seed=default_cfg.training.seed,
        verbose=default_cfg.debug.verbose,
        debug_numerics=default_cfg.debug.debug_numerics,
        debug_numerics_fail_fast=default_cfg.debug.debug_numerics_fail_fast,
        debug_numerics_every=default_cfg.debug.debug_numerics_every,
        debug_numerics_dump_dir=default_cfg.debug.debug_numerics_dump_dir,
        debug_numerics_dump_arrays=default_cfg.debug.debug_numerics_dump_arrays,
        debug_numerics_max_dumps=default_cfg.debug.debug_numerics_max_dumps,
        nan_skip_batch=default_cfg.debug.nan_skip_batch,
    )

    args = parser.parse_args()

    if args.print_run_config:
        print(generate_run_config_example(), end="")
        return

    run_cfg = RunConfig()
    if args.preset:
        run_cfg = load_preset_config(args.preset, base=run_cfg)
    if args.run_config:
        run_cfg = load_run_config(args.run_config, base=run_cfg)
    train_config_path = args.train_config or run_cfg.training.train_config
    from df_mlx.config import get_default_config

    model_cfg = get_default_config()
    dataset_overrides: dict[str, Any] = {}
    ini_warnings: list[str] = []
    if train_config_path:
        ini_overrides = apply_train_ini_config(train_config_path, run_cfg, model_cfg)
        dataset_overrides.update(ini_overrides.dataset_overrides)
        ini_warnings.extend(ini_overrides.warnings)
    # Enforce documented precedence: defaults < train-config < run-config < CLI.
    if args.run_config:
        run_cfg = load_run_config(args.run_config, base=run_cfg)
    # Single-file mode: apply INI-compatible sections in run-config.
    # Then re-apply run-config so explicit top-level TOML values win over train_ini.* compatibility tables.
    if run_cfg.train_ini:
        toml_ini_overrides = apply_train_ini_tables(run_cfg.train_ini, run_cfg, model_cfg)
        dataset_overrides.update(toml_ini_overrides.dataset_overrides)
        ini_warnings.extend(toml_ini_overrides.warnings)
        if args.run_config:
            run_cfg = load_run_config(args.run_config, base=run_cfg)
    _apply_cli_overrides(run_cfg, args, sys.argv[1:])
    validate_run_config(run_cfg)
    if ini_warnings:
        print("Train-config compatibility warnings:")
        for warning in ini_warnings:
            print(f"  - {warning}")
    # Ensure backbone override from CLI/run-config wins
    model_cfg.backbone.backbone_type = run_cfg.model.backbone_type  # type: ignore[assignment]

    def _resolve_resume(resume_setting: bool | str, checkpoint_dir: str, label: str) -> str | None:
        if not resume_setting:
            return None
        if isinstance(resume_setting, str):
            return resume_setting
        ckpt_dir = Path(checkpoint_dir)
        if label == "resume":
            latest = find_latest_checkpoint(ckpt_dir)
            if latest:
                resume_path = str(latest)
                print(f"Auto-resuming from: {resume_path}")
                return resume_path
            print(f"Warning: resume requested but no checkpoint found in {ckpt_dir}")
            return None
        data_ckpt = ckpt_dir / "data_checkpoint.json"
        if data_ckpt.exists():
            resume_path = str(data_ckpt)
            print(f"Auto-resuming data from: {resume_path}")
            return resume_path
        print(f"Warning: resume-data requested but {data_ckpt} not found")
        return None

    resume_from = _resolve_resume(run_cfg.checkpoint.resume, run_cfg.checkpoint.checkpoint_dir, "resume")
    resume_data_from = _resolve_resume(
        run_cfg.checkpoint.resume_data,
        run_cfg.checkpoint.checkpoint_dir,
        "resume_data",
    )

    train(
        cache_dir=run_cfg.dataset.cache_dir,
        speech_list=run_cfg.dataset.speech_list,
        noise_list=run_cfg.dataset.noise_list,
        rir_list=run_cfg.dataset.rir_list,
        config_path=run_cfg.dataset.config,
        epochs=run_cfg.training.epochs,
        batch_size=run_cfg.training.batch_size,
        learning_rate=run_cfg.training.learning_rate,
        learning_rate_min=run_cfg.training.learning_rate_min,
        weight_decay=run_cfg.training.weight_decay,
        checkpoint_dir=run_cfg.checkpoint.checkpoint_dir,
        resume_from=resume_from,
        resume_data_from=resume_data_from,
        validate_every=run_cfg.checkpoint.validate_every,
        save_strategy=cast(Literal["no", "epoch", "steps"], run_cfg.checkpoint.save_strategy),
        save_steps=run_cfg.checkpoint.save_steps,
        save_total_limit=run_cfg.checkpoint.save_total_limit,
        checkpoint_batches=run_cfg.checkpoint.checkpoint_batches,
        max_grad_norm=run_cfg.training.max_grad_norm,
        warmup_epochs=run_cfg.training.warmup_epochs,
        patience=run_cfg.training.patience,
        num_workers=run_cfg.dataloader.num_workers,
        prefetch_size=run_cfg.dataloader.prefetch_size,
        p_reverb=run_cfg.augmentation.p_reverb,
        p_clipping=run_cfg.augmentation.p_clipping,
        use_mlx_data=run_cfg.dataloader.use_mlx_data,
        use_fp16=run_cfg.training.fp16,
        grad_accumulation_steps=run_cfg.training.grad_accumulation_steps,
        eval_frequency=run_cfg.training.eval_frequency,
        backbone_type=cast(Literal["mamba", "gru", "attention"], run_cfg.model.backbone_type),
        model_variant=cast(Literal["full", "lite"], run_cfg.model.variant),
        verbose=run_cfg.debug.verbose,
        snr_range=run_cfg.dataset.snr_range,
        snr_range_extreme=run_cfg.dataset.snr_range_extreme,
        snr_range_very_low=run_cfg.dataset.snr_range_very_low,
        p_extreme_snr=run_cfg.dataset.p_extreme_snr,
        p_very_low_snr=run_cfg.dataset.p_very_low_snr,
        p_interfer_speech=run_cfg.dataset.p_interfer_speech,
        curriculum_warmup_epochs=run_cfg.training.curriculum_warmup_epochs,
        speech_gain_range=run_cfg.dataset.speech_gain_range,
        noise_gain_range=run_cfg.dataset.noise_gain_range,
        dynamic_loss=cast(Literal["baseline", "awesome", "pipeline_awesome"], run_cfg.loss.dynamic_loss),
        pipeline_stages=run_cfg.loss.pipeline_stages,
        awesome_loss_weight=run_cfg.loss.awesome.loss_weight,
        awesome_mask_sharpness=run_cfg.loss.awesome.mask_sharpness,
        awesome_warmup_steps=run_cfg.loss.awesome.warmup_steps,
        gan_enabled=run_cfg.gan.enabled,
        gan_start_epoch=run_cfg.gan.start_epoch,
        gan_ramp_epochs=run_cfg.gan.ramp_epochs,
        gan_adv_weight=run_cfg.gan.adv_weight,
        gan_fm_weight=run_cfg.gan.fm_weight,
        gan_disc_type=cast(Literal["combined", "mpd", "msd"], run_cfg.gan.discriminator),
        gan_mpd_periods=tuple(run_cfg.gan.mpd_periods) if run_cfg.gan.mpd_periods else None,
        gan_msd_scales=run_cfg.gan.msd_scales,
        gan_disc_lr=run_cfg.gan.disc_lr,
        gan_disc_weight_decay=run_cfg.gan.disc_weight_decay,
        gan_disc_grad_clip=run_cfg.gan.disc_grad_clip,
        gan_disc_update_freq=run_cfg.gan.disc_update_freq,
        gan_disc_max_samples=run_cfg.gan.disc_max_samples,
        gan_mpd_channels=run_cfg.gan.mpd_channels,
        gan_msd_channels=run_cfg.gan.msd_channels,
        experimental_compiled_gan=run_cfg.gan.experimental_compile,
        vad_proxy_enabled=run_cfg.loss.awesome.proxy_enabled,
        vad_loss_weight=run_cfg.vad.loss_weight,
        vad_threshold=run_cfg.vad.threshold,
        vad_margin=run_cfg.vad.margin,
        vad_speech_loss_weight=run_cfg.vad.speech_loss_weight,
        vad_warmup_epochs=run_cfg.vad.warmup_epochs,
        vad_snr_gate_db=run_cfg.vad.snr_gate_db,
        vad_snr_gate_width=run_cfg.vad.snr_gate_width,
        vad_band_low_hz=run_cfg.vad.band_low_hz,
        vad_band_high_hz=run_cfg.vad.band_high_hz,
        vad_z_threshold=run_cfg.vad.z_threshold,
        vad_z_slope=run_cfg.vad.z_slope,
        vad_eval_mode=cast(Literal["auto", "proxy", "silero", "off"], run_cfg.vad.eval.mode),
        vad_eval_every=run_cfg.vad.eval.every,
        vad_eval_batches=run_cfg.vad.eval.batches,
        vad_eval_max_seconds=run_cfg.vad.eval.max_seconds,
        vad_silero_model_path=run_cfg.vad.eval.silero_model_path,
        vad_silero_sample_rate=run_cfg.vad.eval.silero_sample_rate,
        vad_train_prob=run_cfg.vad.train.prob,
        vad_train_every_steps=run_cfg.vad.train.every_steps,
        eval_sisdr=run_cfg.metrics.eval_sisdr,
        check_chkpts=run_cfg.checkpoint.check_chkpts,
        max_train_batches=run_cfg.dataloader.max_train_batches,
        max_valid_batches=run_cfg.dataloader.max_valid_batches,
        seed=run_cfg.training.seed,
        debug_numerics=run_cfg.debug.debug_numerics,
        debug_numerics_fail_fast=run_cfg.debug.debug_numerics_fail_fast,
        debug_numerics_every=run_cfg.debug.debug_numerics_every,
        debug_numerics_dump_dir=run_cfg.debug.debug_numerics_dump_dir,
        debug_numerics_dump_arrays=run_cfg.debug.debug_numerics_dump_arrays,
        debug_numerics_max_dumps=run_cfg.debug.debug_numerics_max_dumps,
        nan_skip_batch=run_cfg.debug.nan_skip_batch,
        sync_mode=run_cfg.debug.sync_mode,
        model_config=model_cfg,
        dataset_overrides=dataset_overrides,
        mrstft_config=run_cfg.loss.mrstft,
        train_config_path=train_config_path,
    )


if __name__ == "__main__":
    main()
