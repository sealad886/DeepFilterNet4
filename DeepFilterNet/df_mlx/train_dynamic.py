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

Features:
    - Dynamic on-the-fly mixing (matches original training strategy)
    - Full dataset diversity each epoch
    - Automatic learning rate scheduling
    - Gradient clipping for stability
    - Periodic checkpointing
    - Validation with fixed noise/RIR for reproducibility
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Tuple, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

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
) -> tuple[mx.array, mx.array]:
    """Compute soft VAD probabilities from log-band energy (z-scored per utterance)."""
    clean_power = clean_real**2 + clean_imag**2
    out_power = out_real**2 + out_imag**2

    clean_band = mx.sum(clean_power * band_mask, axis=-1) / (band_bins + eps)
    out_band = mx.sum(out_power * band_mask, axis=-1) / (band_bins + eps)

    log_clean = mx.log10(clean_band + eps)
    mu = mx.mean(log_clean, axis=1, keepdims=True)
    sigma = mx.sqrt(mx.mean((log_clean - mu) ** 2, axis=1, keepdims=True) + eps)

    z_ref = (log_clean - mu) / (sigma + eps)
    z_out = (mx.log10(out_band + eps) - mu) / (sigma + eps)

    z_slope = max(vad_z_slope, 1e-3)
    p_ref = mx.sigmoid((z_ref - vad_z_threshold) / z_slope)
    p_out = mx.sigmoid((z_out - vad_z_threshold) / z_slope)
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
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute soft VAD loss and diagnostics.

    Penalizes decreases in VAD probability relative to reference speech.
    """
    clean_real = clean_real.astype(mx.float32)
    clean_imag = clean_imag.astype(mx.float32)
    out_real = out_real.astype(mx.float32)
    out_imag = out_imag.astype(mx.float32)

    p_ref, p_out = _compute_vad_probs(
        clean_real,
        clean_imag,
        out_real,
        out_imag,
        band_mask,
        band_bins,
        vad_z_threshold,
        vad_z_slope,
    )

    speech_gate = mx.clip((p_ref - vad_threshold) / (1.0 - vad_threshold + _EPS), 0.0, 1.0)
    snr_scale = max(vad_snr_gate_width, 1e-3)
    snr_gate = mx.sigmoid((snr[:, None] - vad_snr_gate_db) / snr_scale)
    gate = mx.stop_gradient(speech_gate * snr_gate)

    vad_loss = mx.mean(mx.maximum(p_ref - p_out - vad_margin, 0.0) * gate)
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
) -> mx.array:
    """Compute speech-band log-magnitude L1 loss weighted by VAD gate."""
    clean_mag = mx.sqrt(clean_real**2 + clean_imag**2 + eps)
    out_mag = mx.sqrt(out_real**2 + out_imag**2 + eps)

    clean_log = mx.log10(clean_mag + eps)
    out_log = mx.log10(out_mag + eps)

    clean_band = mx.sum(clean_log * band_mask, axis=-1) / (band_bins + eps)
    out_band = mx.sum(out_log * band_mask, axis=-1) / (band_bins + eps)

    return mx.mean(mx.abs(out_band - clean_band) * gate)


def _log1p_mag(real: mx.array, imag: mx.array, eps: float = _EPS) -> mx.array:
    """Compute log1p magnitude for complex STFT."""
    mag = mx.sqrt(real**2 + imag**2 + eps)
    return mx.log1p(mag)


def _compute_musicness(
    mag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    eps: float = _EPS,
) -> tuple[mx.array, mx.array]:
    """Compute a cheap musicness score and its inverse gate.

    Uses spectral flatness (tonalness) and temporal flux stability.
    Returns per-sample musicness and a [0,1] gate (1 = keep speech bias).
    """
    # Spectral flatness over speech band
    log_mag = mx.log(mag + eps)
    mean_log = mx.sum(log_mag * band_mask, axis=-1) / (band_bins + eps)
    geom_mean = mx.exp(mean_log)
    arith_mean = mx.sum(mag * band_mask, axis=-1) / (band_bins + eps)
    flatness = geom_mean / (arith_mean + eps)
    tonal = 1.0 - mx.clip(flatness, 0.0, 1.0)
    tonal_mean = mx.mean(tonal, axis=1, keepdims=True)

    # Temporal flux (lower flux => more music-like)
    band_mag = mag * band_mask
    flux = mx.sum(mx.abs(band_mag[:, 1:, :] - band_mag[:, :-1, :]), axis=-1) / (band_bins + eps)
    flux = mx.mean(flux, axis=1, keepdims=True)
    flux_gate = mx.sigmoid((_AWESOME_MUSIC_FLUX_THR - flux) / _AWESOME_MUSIC_FLUX_WIDTH)

    musicness = mx.clip(tonal_mean * flux_gate, 0.0, 1.0)
    music_gate = 1.0 - mx.sigmoid((musicness - _AWESOME_MUSICNESS_THR) / _AWESOME_MUSICNESS_WIDTH)
    return musicness.squeeze(-1), music_gate.squeeze(-1)


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
    clean_power = clean_real**2 + clean_imag**2
    noise_real = noisy_real - clean_real
    noise_imag = noisy_imag - clean_imag
    noise_power = noise_real**2 + noise_imag**2

    clean_band = mx.sum(clean_power * band_mask, axis=-1) / (band_bins + eps)
    noise_band = mx.sum(noise_power * band_mask, axis=-1) / (band_bins + eps)
    speech_ratio = clean_band / (clean_band + noise_band + eps)

    log_clean = mx.log10(clean_band + eps)
    mu = mx.mean(log_clean, axis=1, keepdims=True)
    sigma = mx.sqrt(mx.mean((log_clean - mu) ** 2, axis=1, keepdims=True) + eps)
    z_ref = (log_clean - mu) / (sigma + eps)

    z_slope = max(vad_z_slope, 1e-3)
    p_ref = mx.sigmoid((z_ref - vad_z_threshold) / z_slope)

    # Modulation proxy from z-scored energy trajectory
    mod_energy = mx.mean(mx.abs(z_ref[:, 1:] - z_ref[:, :-1]), axis=1, keepdims=True)
    mod_gate = mx.sigmoid((mod_energy - _AWESOME_MOD_THRESHOLD) / _AWESOME_MOD_WIDTH)

    mean_log = mx.mean(log_clean, axis=1, keepdims=True)
    energy_boost = mx.sigmoid((_AWESOME_ENERGY_BOOST_DB - mean_log) / _AWESOME_ENERGY_BOOST_WIDTH)

    snr_scale = max(vad_snr_gate_width, 1e-3)
    snr_boost = mx.sigmoid((vad_snr_gate_db - snr[:, None]) / snr_scale)

    # Musicness gate from noisy magnitude
    noisy_mag = mx.sqrt(noisy_real**2 + noisy_imag**2 + eps)
    musicness, music_gate = _compute_musicness(noisy_mag, band_mask, band_bins, eps=eps)

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
]:
    """Compute awesome loss components and diagnostic gates."""
    clean_log = _log1p_mag(clean_real, clean_imag, eps=eps)
    out_log = _log1p_mag(out_real, out_imag, eps=eps)

    noise_real = noisy_real - clean_real
    noise_imag = noisy_imag - clean_imag
    noise_log = _log1p_mag(noise_real, noise_imag, eps=eps)

    mask = mx.sigmoid(mask_sharpness * (clean_log - noise_log))
    mask = mx.stop_gradient(mask)

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


def _register_sigint_handler(model, optimizer, checkpoint_dir, config, last_completed_epoch: int = -1):
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
    flat_grads = []

    def flatten(x):
        if isinstance(x, mx.array):
            flat_grads.append(x.reshape(-1))
        elif isinstance(x, dict):
            for v in x.values():
                flatten(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                flatten(v)

    flatten(grads)

    if not flat_grads:
        return grads, mx.array(0.0)

    total_norm_sq = sum(mx.sum(g**2) for g in flat_grads)
    total_norm = mx.sqrt(total_norm_sq)

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = mx.minimum(clip_coef, mx.array(1.0))

    def apply_clip(x):
        if isinstance(x, mx.array):
            return x * clip_coef
        elif isinstance(x, dict):
            return {k: apply_clip(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [apply_clip(v) for v in x]
        elif isinstance(x, tuple):
            return tuple(apply_clip(v) for v in x)
        return x

    return cast(dict, apply_clip(grads)), total_norm


_CHECKPOINT_KINDS = {"step", "epoch_end", "best", "best_final", "final", "interrupted"}
_COMPLETED_KINDS = {"epoch_end", "best", "best_final", "final"}
_IN_PROGRESS_KINDS = {"step", "interrupted"}


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
        [p for p in checkpoint_dir.glob(f"*{manifest.weights_ext}") if not manifest.is_temporary(p)],
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
        batch_idx = state.get("batch_idx")
        global_step = state.get("global_step")

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
            resume_batch = latest.state.get("batch_idx")
            if latest.kind in _IN_PROGRESS_KINDS and isinstance(resume_batch, int):
                report["resume_batch"] = resume_batch
            report["resume_global_step"] = latest.state.get("global_step")

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
    last_completed_epoch: int = -1,
    kind: str = "epoch_end",
    raise_on_error: bool = False,
) -> bool:
    """Save a training checkpoint with model weights, training state, and optimizer state.

    Args:
        model: Model to save
        path: Path to checkpoint file (.safetensors)
        epoch: Current epoch index (0-based)
        batch_idx: Batch index within epoch (for in-progress checkpoints)
        global_step: Global training step
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

        checkpoint_kind = "end_of_epoch" if kind in _COMPLETED_KINDS else "in_progress"

        # Save training state and metadata
        state_path = manifest.state_path(path)
        tmp_state_path = state_path.with_name(f"{state_path.stem}.tmp{state_path.suffix}")
        state = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": global_step,
            "loss": loss,
            "best_valid_loss": best_valid_loss,
            "config": config,
            "optimizer_state": optimizer_state_dict,
            "last_completed_epoch": last_completed_epoch,
            "kind": kind,
            "checkpoint_kind": checkpoint_kind,
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
        ckpt_files.extend(checkpoint_dir.glob(pattern))

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

    report = validate_checkpoint_dir(checkpoint_dir, strict=False, validate_load=False)
    latest = report.get("latest_path")
    if isinstance(latest, Path):
        return latest
    return None


def train(
    cache_dir: str | None = None,
    speech_list: str | None = None,
    noise_list: str | None = None,
    rir_list: str | None = None,
    config_path: str | None = None,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
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
    verbose: bool = False,
    snr_range: Tuple[float, float] | None = None,
    snr_range_extreme: Tuple[float, float] | None = None,
    p_extreme_snr: float | None = None,
    speech_gain_range: Tuple[float, float] | None = None,
    noise_gain_range: Tuple[float, float] | None = None,
    dynamic_loss: Literal["baseline", "awesome"] = "baseline",
    awesome_loss_weight: float = 0.4,
    awesome_mask_sharpness: float = 6.0,
    awesome_warmup_steps: int = 0,
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
    vad_train_prob: float = 0.0,
    vad_train_every_steps: int = 0,
    eval_sisdr: bool = False,
    max_train_batches: int | None = None,
    max_valid_batches: int | None = None,
    check_chkpts: bool = False,
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
        verbose: Enable detailed timing and diagnostic output
        snr_range: Optional override for base SNR range (dB)
        snr_range_extreme: Optional override for extreme SNR range (dB)
        p_extreme_snr: Optional override for extreme SNR sampling probability
        speech_gain_range: Optional override for speech gain range (dB)
        noise_gain_range: Optional override for noise gain range (dB)
        dynamic_loss: Which dynamic loss to use ("baseline" or "awesome")
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
        vad_train_prob: Probability of applying sparse VAD regularizer per batch
        vad_train_every_steps: Apply VAD regularizer every N steps (0 disables)
        eval_sisdr: Compute SI-SDR during validation (slower)
        max_train_batches: Limit number of train batches per epoch (None = full epoch)
        max_valid_batches: Limit number of validation batches (None = full validation)
        check_chkpts: Validate checkpoints before starting/resuming
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
    from df_mlx.train import WarmupCosineSchedule, spectral_loss

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

    if snr_range is not None:
        config.snr_range = snr_range
    if snr_range_extreme is not None:
        config.snr_range_extreme = snr_range_extreme
    if p_extreme_snr is not None:
        config.p_extreme_snr = p_extreme_snr
    if speech_gain_range is not None:
        config.speech_gain_range = speech_gain_range
    if noise_gain_range is not None:
        config.noise_gain_range = noise_gain_range

    # Create dataset (this populates config.*_files from cache index if using cache)
    print("\nInitializing dynamic dataset...")
    dataset = DynamicDataset(config)

    use_awesome_loss = dynamic_loss == "awesome"

    if vad_eval_mode == "auto":
        vad_eval_mode = "proxy" if use_awesome_loss else "off"
    vad_eval_enabled = vad_eval_mode != "off"
    if vad_eval_mode == "silero":
        print("⚠️  Silero VAD not integrated in-core; falling back to proxy VAD for eval metrics.")
        vad_eval_mode = "proxy"

    use_vad_loss = vad_loss_weight > 0 or vad_speech_loss_weight > 0
    use_vad_train_reg = (vad_train_prob > 0 or vad_train_every_steps > 0) and vad_loss_weight > 0

    need_band_mask = use_vad_loss or use_awesome_loss or vad_eval_enabled or use_vad_train_reg
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

    # Print file counts after dataset init (so cache files are included)
    print(f"Speech files:   {len(config.speech_files):,}")
    print(f"Noise files:    {len(config.noise_files):,}")
    print(f"RIR files:      {len(config.rir_files):,}")
    print(f"Epochs:         {epochs}")
    print(f"Batch size:     {batch_size}")
    print(f"Learning rate:  {learning_rate}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"P(reverb):      {config.p_reverb}")
    print(f"P(clipping):    {config.p_clipping}")
    print(f"SNR range:      {config.snr_range} dB")
    print(f"SNR extreme:    {config.snr_range_extreme} dB (p={config.p_extreme_snr})")
    print(f"Speech gain:    {config.speech_gain_range} dB")
    print(f"Noise gain:     {config.noise_gain_range} dB")
    print(f"Dynamic loss:   {dynamic_loss}")
    if use_awesome_loss:
        print(
            f"  Awesome loss: weight={awesome_loss_weight}, mask_sharpness={awesome_mask_sharpness}, "
            f"warmup_steps={awesome_warmup_steps}, proxy={'on' if vad_proxy_enabled else 'off'}"
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
    if use_vad_train_reg:
        print(
            "  VAD train:     " f"prob={vad_train_prob} every_steps={vad_train_every_steps} (weight={vad_loss_weight})"
        )
    print("=" * 60)

    train_config = {
        **config.__dict__,
        "dynamic_loss": dynamic_loss,
        "awesome_loss_weight": awesome_loss_weight,
        "awesome_mask_sharpness": awesome_mask_sharpness,
        "awesome_warmup_steps": awesome_warmup_steps,
        "vad_proxy_enabled": vad_proxy_enabled,
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
        "vad_train_prob": vad_train_prob,
        "vad_train_every_steps": vad_train_every_steps,
        "eval_sisdr": eval_sisdr,
        "max_train_batches": max_train_batches,
        "max_valid_batches": max_valid_batches,
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
    model_config = get_default_config()
    model_config.backbone.backbone_type = backbone_type  # type: ignore[assignment]
    print(f"  Backbone type: {backbone_type}")
    model = init_model(config=model_config)
    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,}")

    # Estimate steps per epoch (approximate since samples may be skipped)
    approx_samples_per_epoch = len(dataset)
    steps_per_epoch = approx_samples_per_epoch // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    vad_warmup_steps = vad_warmup_epochs * steps_per_epoch if use_vad_loss else 0
    awesome_warmup_steps = max(int(awesome_warmup_steps), 0) if use_awesome_loss else 0

    schedule = WarmupCosineSchedule(
        base_lr=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=learning_rate * 0.01,
    )

    # Optimizer - create before loading checkpoint to allow optimizer state restoration
    # Use fixed learning rate (schedule applied manually before each step)
    # This is required because schedule callbacks can't run inside mx.compile()
    optimizer = optim.AdamW(learning_rate=learning_rate)

    # Resume from checkpoint if provided (AFTER optimizer creation)
    start_epoch = 0
    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    last_completed_epoch = -1
    resume_global_step = 0
    resume_batch_idx = 0

    if resume_from:
        state = load_checkpoint(model, resume_from, optimizer=optimizer)
        if state:
            ckpt_epoch = int(state.get("epoch", 0))
            ckpt_kind = state.get("kind", "epoch_end")
            resume_global_step = state.get("global_step", ckpt_epoch * steps_per_epoch)
            start_epoch = compute_resume_epoch(state)
            completed_kinds = {"epoch_end", "best", "best_final", "final"}
            if ckpt_kind in completed_kinds:
                last_completed_epoch = state.get("last_completed_epoch", ckpt_epoch)
            else:
                last_completed_epoch = state.get("last_completed_epoch", ckpt_epoch - 1)
            if ckpt_kind in _IN_PROGRESS_KINDS:
                batch_val = state.get("batch_idx")
                if isinstance(batch_val, int) and batch_val >= 0:
                    resume_batch_idx = batch_val
            best_valid_loss = state.get("best_valid_loss", float("inf"))
            print(
                "  Resumed from: "
                f"{resume_from} (epoch {start_epoch}, kind={ckpt_kind}, "
                f"last_completed={last_completed_epoch})"
            )
            print(
                "  Resume target: "
                f"epoch {start_epoch + 1} (idx {start_epoch}), "
                f"batch {resume_batch_idx}, global_step {resume_global_step}"
            )
            if start_epoch >= epochs:
                print(f"✅ Training already complete (checkpoint epoch {ckpt_epoch}/{epochs}).")
                return

    if validation_report and validation_report["last_completed_epoch"] > last_completed_epoch:
        last_completed_epoch = validation_report["last_completed_epoch"]

    if train_stream is not None and data_resume_progress is not None:
        data_epoch = data_resume_progress.get("epoch")
        if isinstance(data_epoch, int) and data_epoch != start_epoch:
            print(
                "⚠️  Data checkpoint epoch does not match resume epoch "
                f"(data={data_epoch}, resume={start_epoch}). "
                f"Ignoring data checkpoint: {data_resume_source}"
            )
            train_stream.set_epoch(start_epoch)
            data_resume_progress = None
        elif data_resume_progress.get("batch", 0) in (0, None):
            data_resume_progress = None

    if resume_from:
        lc_display = f"{last_completed_epoch + 1} (idx {last_completed_epoch})" if last_completed_epoch >= 0 else "none"
        print(f"  last_completed_epoch: {lc_display}")

    _interrupt_state["last_completed_epoch"] = last_completed_epoch

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
    ):
        """Compute training loss."""
        # Model expects spec as tuple (real, imag)
        noisy_spec = (noisy_real, noisy_imag)
        target_spec = (clean_real, clean_imag)

        out = model(noisy_spec, feat_erb, feat_spec)
        spec_loss = spectral_loss(out, target_spec)
        total_loss = spec_loss

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

        return total_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

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
        max_grad_norm_val,
    ):
        """JIT-compiled training step for faster training.

        This compiles the forward pass, backward pass, and optimizer update
        into a single optimized computation graph.
        """
        loss, grads = loss_and_grad(
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
        )
        # Gradient clipping inline
        if max_grad_norm_val > 0:
            grads, _ = clip_grad_norm(grads, max_grad_norm_val)
        optimizer.update(model, grads)
        return loss

    def run_validation(label: str = "  Validating", *, do_vad_eval: bool = False) -> float:
        """Run validation on the fixed validation split and return average loss."""
        model.eval()

        dataset.set_split("valid")
        dataset.set_epoch(0)  # Fixed epoch for reproducible validation

        if len(dataset) == 0:
            return float("inf")

        valid_loss = 0.0
        valid_spec_loss = 0.0
        valid_vad_loss = 0.0
        valid_speech_loss = 0.0
        valid_awesome_loss = 0.0
        valid_awesome_speech = 0.0
        valid_awesome_noise = 0.0
        valid_awesome_smooth = 0.0
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
        vad_eval_p_ref = 0.0
        vad_eval_p_out = 0.0
        vad_eval_delta = 0.0
        vad_eval_batches_done = 0
        num_valid_batches = 0
        valid_steps = len(dataset) // batch_size
        if max_valid_batches is not None:
            valid_steps = min(valid_steps, max_valid_batches)

        valid_loader = PrefetchDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=2,
            prefetch_factor=1,
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

        for batch_idx, batch in enumerate(valid_pbar):
            noisy_real = batch["noisy_real"]
            noisy_imag = batch["noisy_imag"]
            clean_real = batch["clean_real"]
            clean_imag = batch["clean_imag"]
            feat_erb = batch["feat_erb"]
            feat_spec = batch["feat_spec"]
            snr = batch["snr"]

            # Model expects spec as tuple (real, imag)
            noisy_spec = (noisy_real, noisy_imag)
            target_spec = (clean_real, clean_imag)

            out = model(noisy_spec, feat_erb, feat_spec)
            spec_loss = spectral_loss(out, target_spec)

            awesome_loss = mx.array(0.0)
            awesome_speech = mx.array(0.0)
            awesome_noise = mx.array(0.0)
            awesome_smooth = mx.array(0.0)
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
                )

            awesome_weight_val = awesome_loss_weight
            if use_awesome_loss and awesome_warmup_steps > 0:
                awesome_weight_val = awesome_loss_weight * min(1.0, global_step / max(awesome_warmup_steps, 1))

            loss = spec_loss
            if use_awesome_loss:
                loss = loss + awesome_weight_val * awesome_loss
            if use_vad_loss:
                loss = loss + vad_loss_weight * vad_loss + vad_speech_loss_weight * speech_loss

            residual = mx.mean((out[0] - clean_real) ** 2 + (out[1] - clean_imag) ** 2)

            loss_val = float(loss)
            spec_loss_val = float(spec_loss)
            vad_loss_val = float(vad_loss)
            speech_loss_val = float(speech_loss)
            awesome_loss_val = float(awesome_loss)
            awesome_speech_val = float(awesome_speech)
            awesome_noise_val = float(awesome_noise)
            awesome_smooth_val = float(awesome_smooth)
            vad_reg_loss_val = float(vad_reg_loss)
            residual_val = float(residual)

            valid_loss += loss_val
            valid_spec_loss += spec_loss_val
            valid_vad_loss += vad_loss_val
            valid_speech_loss += speech_loss_val
            valid_awesome_loss += awesome_loss_val
            valid_awesome_speech += awesome_speech_val
            valid_awesome_noise += awesome_noise_val
            valid_awesome_smooth += awesome_smooth_val
            valid_vad_reg_loss += vad_reg_loss_val
            valid_residual += residual_val
            num_valid_batches += 1

            if use_vad_loss:
                valid_p_ref += float(mx.mean(p_ref))
                valid_p_out += float(mx.mean(p_out))
                valid_gate_pct += 100.0 * float(mx.mean(mx.where(gate > 0.0, 1.0, 0.0)))

            if use_awesome_loss:
                mask_mean = float(mx.mean(mask))
                mask_high = 100.0 * float(mx.mean(mx.where(mask > 0.8, 1.0, 0.0)))
                mask_low = 100.0 * float(mx.mean(mx.where(mask < 0.2, 1.0, 0.0)))
                proxy_mean = float(mx.mean(proxy_frame))
                speech_ratio_mean = float(mx.mean(speech_ratio))
                music_gate_mean = float(mx.mean(music_gate))
                musicness_mean = float(mx.mean(musicness))
                mod_energy_mean = float(mx.mean(mod_energy))
                energy_boost_mean = float(mx.mean(energy_boost))
                snr_boost_mean = float(mx.mean(snr_boost))

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

            if sisdr_fn is not None:
                si_sdr_fn, istft_fn = sisdr_fn
                clean_wav = istft_fn(target_spec, n_fft=config.fft_size, hop_length=config.hop_size)
                out_wav = istft_fn(out, n_fft=config.fft_size, hop_length=config.hop_size)
                sisdr_val = float(si_sdr_fn(out_wav, clean_wav))
                valid_sisdr += sisdr_val

            valid_pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                avg=f"{valid_loss / num_valid_batches:.4f}",
            )

            if max_valid_batches is not None and (batch_idx + 1) >= max_valid_batches:
                break

        valid_pbar.close()

        if num_valid_batches > 0:
            avg_spec = valid_spec_loss / num_valid_batches
            avg_vad = valid_vad_loss / num_valid_batches
            avg_speech = valid_speech_loss / num_valid_batches
            avg_awesome = valid_awesome_loss / num_valid_batches
            avg_awesome_speech = valid_awesome_speech / num_valid_batches
            avg_awesome_noise = valid_awesome_noise / num_valid_batches
            avg_awesome_smooth = valid_awesome_smooth / num_valid_batches
            avg_vad_reg = valid_vad_reg_loss / num_valid_batches
            avg_residual = valid_residual / num_valid_batches
            avg_p_ref = valid_p_ref / num_valid_batches if use_vad_loss else 0.0
            avg_p_out = valid_p_out / num_valid_batches if use_vad_loss else 0.0
            avg_gate = valid_gate_pct / num_valid_batches if use_vad_loss else 0.0
            avg_sisdr = valid_sisdr / num_valid_batches if eval_sisdr else None
            avg_mask_mean = valid_mask_mean / num_valid_batches if use_awesome_loss else 0.0
            avg_mask_high = valid_mask_high / num_valid_batches if use_awesome_loss else 0.0
            avg_mask_low = valid_mask_low / num_valid_batches if use_awesome_loss else 0.0
            avg_proxy = valid_proxy_mean / num_valid_batches if use_awesome_loss else 0.0
            avg_speech_ratio = valid_speech_ratio / num_valid_batches if use_awesome_loss else 0.0
            avg_music_gate = valid_music_gate / num_valid_batches if use_awesome_loss else 0.0
            avg_musicness = valid_musicness / num_valid_batches if use_awesome_loss else 0.0
            avg_mod = valid_mod_energy / num_valid_batches if use_awesome_loss else 0.0
            avg_energy_boost = valid_energy_boost / num_valid_batches if use_awesome_loss else 0.0
            avg_snr_boost = valid_snr_boost / num_valid_batches if use_awesome_loss else 0.0
            avg_vad_eval_p_ref = (
                vad_eval_p_ref / vad_eval_batches_done if do_vad_eval and vad_eval_batches_done > 0 else 0.0
            )
            avg_vad_eval_p_out = (
                vad_eval_p_out / vad_eval_batches_done if do_vad_eval and vad_eval_batches_done > 0 else 0.0
            )
            avg_vad_eval_delta = (
                vad_eval_delta / vad_eval_batches_done if do_vad_eval and vad_eval_batches_done > 0 else 0.0
            )

            if use_vad_loss or eval_sisdr or use_awesome_loss or use_vad_train_reg or do_vad_eval:
                extras = [f"spec={avg_spec:.4f}", f"resid={avg_residual:.4f}"]
                if use_vad_loss:
                    extras.extend([f"vad={avg_vad:.4f}", f"speech={avg_speech:.4f}"])
                if use_awesome_loss:
                    extras.extend(
                        [
                            f"awesome={avg_awesome:.4f}",
                            f"aw_s={avg_awesome_speech:.4f}",
                            f"aw_n={avg_awesome_noise:.4f}",
                            f"aw_sm={avg_awesome_smooth:.4f}",
                        ]
                    )
                if use_vad_train_reg:
                    extras.append(f"vad_reg={avg_vad_reg:.4f}")
                if use_vad_loss:
                    extras.append(f"p_ref={avg_p_ref:.2f}")
                    extras.append(f"p_out={avg_p_out:.2f}")
                    extras.append(f"gate={avg_gate:.0f}%")
                if use_awesome_loss:
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
                if avg_sisdr is not None:
                    extras.append(f"si-sdr={avg_sisdr:.2f}dB")
                print(f"{label} metrics: " + " | ".join(extras))

        return valid_loss / max(num_valid_batches, 1)

    # Flag to use compiled step (can be disabled for debugging)
    use_compiled_step = True
    print(f"  Using compiled training step: {use_compiled_step}")

    # Register SIGINT handler for graceful shutdown
    _register_sigint_handler(model, optimizer, ckpt_dir, train_config, last_completed_epoch=last_completed_epoch)
    print("  SIGINT handler registered (CTRL+C will save checkpoint before exit)")

    # Training loop
    print(f"\nStarting training (epoch {start_epoch + 1} to {epochs})...")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Est. total steps: {total_steps:,}")
    print()

    global_step = resume_global_step if resume_from else start_epoch * steps_per_epoch
    final_epoch = start_epoch
    last_completed_epoch = max(last_completed_epoch, start_epoch - 1)
    avg_train_loss = float("nan")
    last_valid_loss: float | None = None
    last_valid_epoch: int | None = None

    max_train_batches = train_config.get("max_train_batches")
    max_valid_batches = train_config.get("max_valid_batches")

    start_display = f"{start_epoch + 1}/{epochs} (idx {start_epoch})"
    lc_display = f"{last_completed_epoch + 1} (idx {last_completed_epoch})" if last_completed_epoch >= 0 else "none"
    print(f"Starting training at epoch {start_display} | last_completed_epoch={lc_display}")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        final_epoch = epoch

        # Set epoch for reproducible shuffling
        dataset.set_split("train")
        dataset.set_epoch(epoch)

        # ====== Training ======
        model.train()
        train_loss = 0.0
        train_spec_loss = 0.0
        train_vad_loss = 0.0
        train_speech_loss = 0.0
        train_awesome_loss = 0.0
        train_awesome_speech = 0.0
        train_awesome_noise = 0.0
        train_awesome_smooth = 0.0
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
        num_vad_logs = 0
        num_awesome_logs = 0
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

        # Create data iterator (MLXDataStream or PrefetchDataLoader)
        if use_mlx_stream and train_stream is not None:
            if data_resume_progress is not None and epoch == data_resume_progress.get("epoch"):
                # Continue from saved data checkpoint without resetting epoch state.
                data_iterator = train_stream
                progress = train_stream.get_progress()
                print(f"  Resuming epoch {epoch + 1} from batch {progress['batch']}")
                data_resume_progress = None
            else:
                train_stream.set_epoch(epoch)
                data_iterator = train_stream
                # Check if resuming mid-epoch
                progress = train_stream.get_progress()
                if progress["batch"] > 0 and epoch == progress["epoch"]:
                    print(f"  Resuming epoch {epoch + 1} from batch {progress['batch']}")
        else:
            data_iterator = PrefetchDataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=config.num_workers,
                prefetch_factor=2,
            )

        # Training progress bar
        train_total = steps_per_epoch
        if max_train_batches is not None:
            train_total = min(train_total, max_train_batches)

        train_pbar = tqdm(
            enumerate(data_iterator),
            total=train_total,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=True,
            **_TQDM_KWARGS,
        )

        data_start = time.time()
        for batch_idx, batch in train_pbar:
            data_time = time.time() - data_start
            total_data_time += data_time
            step_start = time.time()

            # Unpack batch
            noisy_real = batch["noisy_real"]
            noisy_imag = batch["noisy_imag"]
            clean_real = batch["clean_real"]
            clean_imag = batch["clean_imag"]
            feat_erb = batch["feat_erb"]
            feat_spec = batch["feat_spec"]
            snr = batch["snr"]

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

            vad_weight = vad_loss_weight * warmup_frac
            speech_weight = vad_speech_loss_weight * warmup_frac
            vad_weight_mx = mx.array(vad_weight, dtype=mx.float32)
            speech_weight_mx = mx.array(speech_weight, dtype=mx.float32)
            awesome_frac = 1.0
            if use_awesome_loss and awesome_warmup_steps > 0:
                awesome_frac = min(1.0, global_step / max(awesome_warmup_steps, 1))
            awesome_weight = awesome_loss_weight * awesome_frac
            awesome_weight_mx = mx.array(awesome_weight, dtype=mx.float32)

            apply_vad_reg = False
            if use_vad_train_reg:
                if vad_train_every_steps > 0 and global_step % vad_train_every_steps == 0:
                    apply_vad_reg = True
                elif vad_train_prob > 0:
                    apply_vad_reg = random.random() < vad_train_prob
            vad_reg_weight = vad_weight if apply_vad_reg else 0.0
            vad_reg_weight_mx = mx.array(vad_reg_weight, dtype=mx.float32)

            # Forward, backward, and update (either compiled or standard)
            fwd_start = time.time()

            if use_compiled_step:
                # Use compiled training step for better performance
                loss = compiled_step(
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
                    max_grad_norm,
                )
                # OPTIMIZATION: Only sync periodically to reduce GPU stalls
                # This allows MLX to batch operations for better throughput
                should_sync = (batch_idx + 1) % eval_frequency == 0
                if should_sync:
                    mx.eval(state)
                grad_norm = 0.0  # Not tracked in compiled step
            else:
                # Standard training step
                loss, grads = loss_and_grad(
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
                )
                # Only sync periodically
                should_sync = (batch_idx + 1) % eval_frequency == 0
                if should_sync:
                    mx.eval(loss)

                # Gradient clipping (returns clipped grads and norm as MLX array)
                if max_grad_norm > 0:
                    grads, grad_norm_arr = clip_grad_norm(grads, max_grad_norm)
                    if should_sync:
                        grad_norm = float(grad_norm_arr)

                # Update parameters
                optimizer.update(model, grads)

                # Only sync periodically for better throughput
                if should_sync:
                    mx.eval(model.parameters(), optimizer.state)

            fwd_time = time.time() - fwd_start
            total_forward_time += fwd_time

            step_time = time.time() - step_start

            # Only convert loss to float when synced (avoids blocking)
            if should_sync:
                loss_val = float(loss)
                train_loss += loss_val * eval_frequency  # Approximate accumulated loss
            num_train_batches += 1
            samples_processed += current_batch_size
            global_step += 1

            # Track progress for interruption-safe resume metadata
            _update_interrupt_state(
                epoch,
                loss_val,
                best_valid_loss,
                batch_idx=batch_idx,
                global_step=global_step,
                last_completed_epoch=last_completed_epoch,
            )

            # Stop early for benchmarking if requested
            if max_train_batches is not None and num_train_batches >= max_train_batches:
                break

            # Update progress bar with real-time metrics (only on sync)
            if should_sync:
                lr = float(schedule(global_step))
                samples_per_sec = (
                    (current_batch_size * eval_frequency) / (step_time * eval_frequency) if step_time > 0 else 0
                )

                # Defaults for logging
                spec_loss_val = loss_val
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

                if use_vad_loss or use_awesome_loss or use_vad_train_reg:
                    # Extra forward pass for component logging (only on sync steps)
                    out = model((noisy_real, noisy_imag), feat_erb, feat_spec)
                    spec_loss = spectral_loss(out, (clean_real, clean_imag))
                    spec_loss_val = float(spec_loss)
                    train_spec_loss += spec_loss_val * eval_frequency

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
                    vad_loss_val = float(vad_loss)
                    speech_loss_val = float(speech_loss)
                    p_ref_mean = float(mx.mean(p_ref))
                    p_out_mean = float(mx.mean(p_out))
                    gate_pct = 100.0 * float(mx.mean(mx.where(gate > 0.0, 1.0, 0.0)))

                    train_vad_loss += vad_loss_val * eval_frequency
                    train_speech_loss += speech_loss_val * eval_frequency
                    train_p_ref += p_ref_mean
                    train_p_out += p_out_mean
                    train_gate_pct += gate_pct
                    num_vad_logs += 1

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
                    )
                    awesome_loss_val = float(awesome_loss)
                    awesome_speech_val = float(awesome_speech)
                    awesome_noise_val = float(awesome_noise)
                    awesome_smooth_val = float(awesome_smooth)

                    mask_mean = float(mx.mean(mask))
                    mask_high = 100.0 * float(mx.mean(mx.where(mask > 0.8, 1.0, 0.0)))
                    mask_low = 100.0 * float(mx.mean(mx.where(mask < 0.2, 1.0, 0.0)))
                    proxy_mean = float(mx.mean(proxy_frame))
                    speech_ratio_mean = float(mx.mean(speech_ratio))
                    music_gate_mean = float(mx.mean(music_gate))
                    musicness_mean = float(mx.mean(musicness))
                    mod_energy_mean = float(mx.mean(mod_energy))
                    energy_boost_mean = float(mx.mean(energy_boost))
                    snr_boost_mean = float(mx.mean(snr_boost))

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
                    )
                    vad_reg_loss_val = float(vad_reg_loss)
                    train_vad_reg_loss += vad_reg_loss_val * eval_frequency

                if verbose:
                    train_pbar.set_postfix(
                        loss=f"{loss_val:.4f}",
                        spec=(
                            f"{spec_loss_val:.4f}"
                            if (use_vad_loss or use_awesome_loss or use_vad_train_reg)
                            else f"{loss_val:.4f}"
                        ),
                        vad=f"{vad_loss_val:.4f}" if use_vad_loss else "0.0000",
                        speech=f"{speech_loss_val:.4f}" if use_vad_loss else "0.0000",
                        awesome=f"{awesome_loss_val:.4f}" if use_awesome_loss else "0.0000",
                        mask=f"{mask_mean:.2f}" if use_awesome_loss else "0.00",
                        lr=f"{lr:.1e}",
                        data=f"{data_time * 1000:.0f}ms",
                        step=f"{fwd_time * 1000:.0f}ms",
                        spd=f"{samples_per_sec:.0f}/s",
                        sync=f"1/{eval_frequency}",
                    )
                else:
                    train_pbar.set_postfix(
                        loss=f"{loss_val:.4f}",
                        avg=f"{train_loss / num_train_batches:.4f}",
                        vad=f"{vad_loss_val:.4f}" if use_vad_loss else "0.0000",
                        speech=f"{speech_loss_val:.4f}" if use_vad_loss else "0.0000",
                        awesome=f"{awesome_loss_val:.4f}" if use_awesome_loss else "0.0000",
                        mask=f"{mask_mean:.2f}" if use_awesome_loss else "0.00",
                        p_ref=f"{p_ref_mean:.2f}" if use_vad_loss else "0.00",
                        p_out=f"{p_out_mean:.2f}" if use_vad_loss else "0.00",
                        gate=f"{gate_pct:.0f}%" if use_vad_loss else "0%",
                        vad_reg=f"{vad_reg_loss_val:.4f}" if use_vad_train_reg else "0.0000",
                        lr=f"{lr:.1e}",
                        grad=f"{grad_norm:.2f}",
                        speed=f"{samples_per_sec:.0f}s/s",
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
                    batch_idx=batch_idx,
                    global_step=global_step,
                    loss=train_loss / num_train_batches if num_train_batches > 0 else loss_val,
                    best_valid_loss=best_valid_loss,
                    config=train_config,
                    optimizer=optimizer,
                    last_completed_epoch=last_completed_epoch,
                    kind="step",
                )
                if step_saved:
                    print(f"\n  📦 Checkpoint saved: {ckpt_path.name} (step {global_step})")
                else:
                    print(f"\n  ⚠️  Checkpoint save failed: {ckpt_path.name} (step {global_step})")

                # Cleanup old checkpoints if limit is set
                if save_total_limit is not None:
                    cleanup_checkpoints(ckpt_dir, save_total_limit)

            # Start timing for next data fetch
            data_start = time.time()

        train_pbar.close()

        # Force sync at epoch end to ensure accurate loss
        mx.eval(state)

        # Save data checkpoint at end of epoch (for clean resume at epoch boundary)
        if use_mlx_stream and train_stream is not None:
            train_stream.save_checkpoint(data_checkpoint_path)

        avg_train_loss = train_loss / max(num_train_batches, 1)
        avg_train_spec_loss = train_spec_loss / max(num_train_batches, 1)
        avg_train_vad_loss = train_vad_loss / max(num_train_batches, 1)
        avg_train_speech_loss = train_speech_loss / max(num_train_batches, 1)
        avg_train_awesome_loss = train_awesome_loss / max(num_train_batches, 1)
        avg_train_awesome_speech = train_awesome_speech / max(num_train_batches, 1)
        avg_train_awesome_noise = train_awesome_noise / max(num_train_batches, 1)
        avg_train_awesome_smooth = train_awesome_smooth / max(num_train_batches, 1)
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
            print(f"    Compiled training:  {'enabled' if use_compiled_step else 'disabled'}")
            if total_data_time > total_forward_time:
                print("    ⚠️  DATA LOADING IS BOTTLENECK - consider more workers or faster storage")

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
        epoch_time = time.time() - epoch_start
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
        if use_vad_loss or use_awesome_loss or use_vad_train_reg:
            loss_parts = [f"Spec: {avg_train_spec_loss:.4f}"]
            if use_vad_loss:
                loss_parts.extend(
                    [
                        f"VAD: {avg_train_vad_loss:.4f}",
                        f"Speech: {avg_train_speech_loss:.4f}",
                    ]
                )
            if use_awesome_loss:
                loss_parts.extend(
                    [
                        f"Awesome: {avg_train_awesome_loss:.4f}",
                        f"AwS: {avg_train_awesome_speech:.4f}",
                        f"AwN: {avg_train_awesome_noise:.4f}",
                        f"AwSm: {avg_train_awesome_smooth:.4f}",
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
        if use_awesome_loss and verbose:
            print(
                "  Awesome stats: "
                f"mask={avg_train_mask_mean:.2f} (hi {avg_train_mask_high:.0f}%, lo {avg_train_mask_low:.0f}%) | "
                f"proxy={avg_train_proxy:.2f} ratio={avg_train_speech_ratio:.2f} | "
                f"music_gate={avg_train_music_gate:.2f} music={avg_train_musicness:.2f} | "
                f"mod={avg_train_mod:.2f} e_boost={avg_train_energy_boost:.2f} snr_boost={avg_train_snr_boost:.2f}"
            )

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
    parser = argparse.ArgumentParser(description="Train DfNet4 with dynamic on-the-fly mixing")

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
        help="Path to JSON config file (alternative to file lists)",
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
        "--backbone-type",
        type=str,
        choices=["mamba", "gru", "attention"],
        default="mamba",
        help="Backbone type: 'mamba' (parallel scan SSM), 'gru' (recurrent), or 'attention' (fastest backward)",
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
        "--p-extreme-snr",
        type=float,
        help="Probability of sampling from extreme SNR range (0-1)",
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
        choices=["baseline", "awesome"],
        default="baseline",
        help="Dynamic loss to use: 'baseline' (spectral + legacy VAD) or 'awesome'",
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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed timing diagnostics and hardware info",
    )

    args = parser.parse_args()

    # Determine FP16 setting from arguments
    use_fp16: bool | None = None
    if args.fp16:
        use_fp16 = True
    elif args.no_fp16:
        use_fp16 = False

    # Resolve resume paths
    # --resume can be: False (not set), True (flag only), or str (explicit path)
    resume_from: str | None = None
    if args.resume:
        if isinstance(args.resume, str):
            resume_from = args.resume
        else:
            # Auto-find latest checkpoint in checkpoint_dir
            ckpt_dir = Path(args.checkpoint_dir)
            latest = find_latest_checkpoint(ckpt_dir)
            if latest:
                resume_from = str(latest)
                print(f"Auto-resuming from: {resume_from}")
            else:
                print(f"Warning: --resume specified but no checkpoint found in {ckpt_dir}")

    # --resume-data can be: False (not set), True (flag only), or str (explicit path)
    resume_data_from: str | None = None
    if args.resume_data:
        if isinstance(args.resume_data, str):
            resume_data_from = args.resume_data
        else:
            # Auto-use data_checkpoint.json in checkpoint_dir
            data_ckpt = Path(args.checkpoint_dir) / "data_checkpoint.json"
            if data_ckpt.exists():
                resume_data_from = str(data_ckpt)
                print(f"Auto-resuming data from: {resume_data_from}")
            else:
                print(f"Warning: --resume-data specified but {data_ckpt} not found")

    train(
        cache_dir=args.cache_dir,
        speech_list=args.speech_list,
        noise_list=args.noise_list,
        rir_list=args.rir_list,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=resume_from,
        resume_data_from=resume_data_from,
        validate_every=args.validate_every,
        save_strategy=cast(Literal["no", "epoch", "steps"], args.save_strategy),
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        checkpoint_batches=args.checkpoint_batches,
        max_grad_norm=args.max_grad_norm,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        prefetch_size=args.prefetch_size,
        p_reverb=args.p_reverb,
        p_clipping=args.p_clipping,
        use_mlx_data=not args.no_mlx_data,
        use_fp16=use_fp16,
        grad_accumulation_steps=args.grad_accumulation_steps,
        eval_frequency=args.eval_frequency,
        backbone_type=cast(Literal["mamba", "gru", "attention"], args.backbone_type),
        verbose=args.verbose,
        snr_range=tuple(args.snr_range) if args.snr_range else None,
        snr_range_extreme=tuple(args.snr_range_extreme) if args.snr_range_extreme else None,
        p_extreme_snr=args.p_extreme_snr,
        speech_gain_range=tuple(args.speech_gain_range) if args.speech_gain_range else None,
        noise_gain_range=tuple(args.noise_gain_range) if args.noise_gain_range else None,
        dynamic_loss=cast(Literal["baseline", "awesome"], args.dynamic_loss),
        awesome_loss_weight=args.awesome_loss_weight,
        awesome_mask_sharpness=args.awesome_mask_sharpness,
        awesome_warmup_steps=args.awesome_warmup_steps,
        vad_proxy_enabled=not args.no_vad_proxy,
        vad_loss_weight=args.vad_loss_weight,
        vad_threshold=args.vad_threshold,
        vad_margin=args.vad_margin,
        vad_speech_loss_weight=args.vad_speech_loss_weight,
        vad_warmup_epochs=args.vad_warmup_epochs,
        vad_snr_gate_db=args.vad_snr_gate,
        vad_snr_gate_width=args.vad_snr_gate_width,
        vad_band_low_hz=args.vad_band_low,
        vad_band_high_hz=args.vad_band_high,
        vad_z_threshold=args.vad_z_threshold,
        vad_z_slope=args.vad_z_slope,
        vad_eval_mode=cast(Literal["auto", "proxy", "silero", "off"], args.vad_eval_mode),
        vad_eval_every=args.vad_eval_every,
        vad_eval_batches=args.vad_eval_batches,
        vad_train_prob=args.vad_train_prob,
        vad_train_every_steps=args.vad_train_every_steps,
        eval_sisdr=args.eval_sisdr,
        check_chkpts=args.check_chkpts,
        max_train_batches=args.max_train_batches,
        max_valid_batches=args.max_valid_batches,
    )


if __name__ == "__main__":
    main()
