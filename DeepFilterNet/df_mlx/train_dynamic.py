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
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, Tuple, cast

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


# ============================================================================
# Signal Handling for Graceful Interrupt
# ============================================================================

# Global state for signal handler
_interrupt_state = {
    "checkpoint_dir": None,
    "epoch": 0,
    "model": None,
    "optimizer": None,
    "loss": 0.0,
    "best_valid_loss": float("inf"),
    "config": {},
    "interrupted": False,
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
    print("\n" + "=" * 60)
    print("⚠️  Training interrupted by user (CTRL+C)")
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

            final_path = ckpt_dir / f"interrupted_epoch_{_interrupt_state['epoch']:03d}.safetensors"
            save_checkpoint(
                _interrupt_state["model"],
                final_path,
                epoch=_interrupt_state["epoch"],
                loss=_interrupt_state["loss"],
                best_valid_loss=_interrupt_state["best_valid_loss"],
                config=_interrupt_state["config"],
                optimizer=_interrupt_state["optimizer"],
            )
            print(f"✅ Final checkpoint saved to {final_path}")
        except Exception as e:
            print(f"❌ Failed to save final checkpoint: {e}")

    print("Exiting...")
    raise KeyboardInterrupt()


def _register_sigint_handler(model, optimizer, checkpoint_dir, config):
    """Register SIGINT handler for graceful training shutdown.

    Args:
        model: Model to save on interrupt
        optimizer: Optimizer to save state on interrupt
        checkpoint_dir: Directory to save checkpoint to
        config: Training configuration dict
    """
    _interrupt_state["model"] = model
    _interrupt_state["optimizer"] = optimizer
    _interrupt_state["checkpoint_dir"] = checkpoint_dir
    _interrupt_state["config"] = config
    signal.signal(signal.SIGINT, _handle_sigint)


def _update_interrupt_state(epoch, loss, best_valid_loss):
    """Update global state for interrupt handler.

    Args:
        epoch: Current epoch
        loss: Current training loss
        best_valid_loss: Best validation loss so far
    """
    _interrupt_state["epoch"] = epoch
    _interrupt_state["loss"] = loss
    _interrupt_state["best_valid_loss"] = best_valid_loss


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


def _validate_checkpoint_pair(checkpoint_path: Path) -> bool:
    """Validate that both .safetensors and .state.json files exist and are non-empty.

    Args:
        checkpoint_path: Path to checkpoint (.safetensors file)

    Returns:
        True if both files exist and are valid, False otherwise
    """
    weights_file = checkpoint_path
    state_file = checkpoint_path.with_suffix(".state.json")

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


def save_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    epoch: int,
    loss: float,
    best_valid_loss: float,
    config: dict,
    optimizer: optim.Optimizer | None = None,
    completed: bool = False,
) -> None:
    """Save a training checkpoint with model weights, training state, and optimizer state.

    Args:
        model: Model to save
        path: Path to checkpoint file (.safetensors)
        epoch: Current epoch number
        loss: Current training loss
        best_valid_loss: Best validation loss so far
        config: Training configuration dict
        optimizer: Optional optimizer to save state from
        completed: Whether this checkpoint marks a completed training run
    """
    from mlx.utils import tree_flatten

    path = Path(path)

    # Flatten nested params for safetensors
    params = model.parameters()
    flat_params = tree_flatten(params)
    weights = {k: v for k, v in flat_params}
    mx.save_safetensors(str(path), weights)

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

    # Save training state and metadata
    state_path = path.with_suffix(".state.json")
    state = {
        "epoch": epoch,
        "loss": loss,
        "best_valid_loss": best_valid_loss,
        "config": config,
        "optimizer_state": optimizer_state_dict,
        "completed": completed,
    }
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    if optimizer_state_dict:
        print(f"✅ Saved checkpoint with optimizer state: {path.name}")


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

    # Validate checkpoint pair before loading
    if not _validate_checkpoint_pair(ckpt_path):
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
        state_path = ckpt_path.with_suffix(".state.json")
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
        print(f"✅ Loaded checkpoint from epoch {epoch}")
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
        state_path = ckpt_path.with_suffix(".state.json")
        state_path.unlink(missing_ok=True)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the most recent checkpoint in the checkpoint directory.

    Checks for step_*.safetensors, epoch_*.safetensors, and final.safetensors files,
    returning the most recently modified one.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to most recent checkpoint, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None

    ckpt_files = []
    for pattern in ["step_*.safetensors", "epoch_*.safetensors", "final.safetensors"]:
        ckpt_files.extend(checkpoint_dir.glob(pattern))

    if not ckpt_files:
        # Also check for best.safetensors
        best_path = checkpoint_dir / "best.safetensors"
        if best_path.exists():
            return best_path
        return None

    # Return most recently modified
    return max(ckpt_files, key=lambda p: p.stat().st_mtime)


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
    eval_sisdr: bool = False,
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
        save_strategy: When to save checkpoints - "no", "epoch", or "steps"
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
        eval_sisdr: Compute SI-SDR during validation (slower)
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

    use_vad_loss = vad_loss_weight > 0 or vad_speech_loss_weight > 0
    if use_vad_loss:
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
    print("=" * 60)

    train_config = {
        **config.__dict__,
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
        "eval_sisdr": eval_sisdr,
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

    # Determine which data loader to use
    use_mlx_stream = use_mlx_data and HAS_MLX_DATA
    if use_mlx_data and not HAS_MLX_DATA:
        print("  Note: mlx-data not available, using PrefetchDataLoader")
    elif use_mlx_stream:
        print(f"  Using MLXDataStream (workers={num_workers}, prefetch={prefetch_size})")

    # Create data stream/loader
    data_checkpoint_path = ckpt_dir / "data_checkpoint.json"
    train_stream: MLXDataStream | None = None

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
            progress = train_stream.get_progress()
            print(f"  Data checkpoint: epoch {progress['epoch']}, batch {progress['batch']}")
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
                progress = train_stream.get_progress()
                print(f"  Auto-resuming from data checkpoint: epoch {progress['epoch']}, batch {progress['batch']}")
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

    if resume_from:
        state = load_checkpoint(model, resume_from, optimizer=optimizer)
        start_epoch = state.get("epoch", 0)
        best_valid_loss = state.get("best_valid_loss", float("inf"))
        completed = bool(state.get("completed", False))
        print(f"  Resumed from: {resume_from} (epoch {start_epoch})")
        if completed and start_epoch >= epochs:
            print(f"✅ Training already complete (checkpoint epoch {start_epoch}/{epochs}).")
            return
        if not completed and start_epoch >= epochs:
            print(
                f"⚠️  Checkpoint epoch {start_epoch} >= requested {epochs} but not marked complete; "
                f"resuming from epoch {max(epochs - 1, 0)}."
            )
            start_epoch = max(epochs - 1, 0)

    # Loss function - define as a pure function for compilation
    # Loss formula:
    #   L_total = L_spec + w_vad * L_vad + w_speech * L_speech
    #   L_vad = mean( gate * relu(p_ref - p_out - margin) )
    #   gate = sigmoid((snr - snr_gate_db)/snr_gate_width) * clip((p_ref - vad_thr)/(1 - vad_thr))
    #   p_ref/p_out from speech-band log-energy (z-scored per utterance)
    #   L_speech = mean( gate * |log_mag_out - log_mag_ref|_speechband )
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
    ):
        """Compute training loss."""
        # Model expects spec as tuple (real, imag)
        noisy_spec = (noisy_real, noisy_imag)
        target_spec = (clean_real, clean_imag)

        out = model(noisy_spec, feat_erb, feat_spec)
        spec_loss = spectral_loss(out, target_spec)

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

            return spec_loss + vad_weight * vad_loss + speech_weight * speech_loss

        return spec_loss

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
        )
        # Gradient clipping inline
        if max_grad_norm_val > 0:
            grads, _ = clip_grad_norm(grads, max_grad_norm_val)
        optimizer.update(model, grads)
        return loss

    def run_validation(label: str = "  Validating") -> float:
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
        valid_p_ref = 0.0
        valid_p_out = 0.0
        valid_gate_pct = 0.0
        valid_residual = 0.0
        valid_sisdr = 0.0
        num_valid_batches = 0
        valid_steps = len(dataset) // batch_size

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

        for batch in valid_pbar:
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
                loss = spec_loss + vad_loss_weight * vad_loss
                loss = loss + vad_speech_loss_weight * speech_loss
            else:
                vad_loss = mx.array(0.0)
                speech_loss = mx.array(0.0)
                p_ref = mx.array(0.0)
                p_out = mx.array(0.0)
                gate = mx.array(0.0)
                loss = spec_loss

            residual = mx.mean((out[0] - clean_real) ** 2 + (out[1] - clean_imag) ** 2)

            loss_val = float(loss)
            spec_loss_val = float(spec_loss)
            vad_loss_val = float(vad_loss)
            speech_loss_val = float(speech_loss)
            residual_val = float(residual)

            valid_loss += loss_val
            valid_spec_loss += spec_loss_val
            valid_vad_loss += vad_loss_val
            valid_speech_loss += speech_loss_val
            valid_residual += residual_val
            num_valid_batches += 1

            if use_vad_loss:
                valid_p_ref += float(mx.mean(p_ref))
                valid_p_out += float(mx.mean(p_out))
                valid_gate_pct += float(mx.mean(mx.where(gate > 0.0, 1.0, 0.0)))

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

        valid_pbar.close()

        if num_valid_batches > 0:
            avg_spec = valid_spec_loss / num_valid_batches
            avg_vad = valid_vad_loss / num_valid_batches
            avg_speech = valid_speech_loss / num_valid_batches
            avg_residual = valid_residual / num_valid_batches
            avg_p_ref = valid_p_ref / num_valid_batches if use_vad_loss else 0.0
            avg_p_out = valid_p_out / num_valid_batches if use_vad_loss else 0.0
            avg_gate = valid_gate_pct / num_valid_batches if use_vad_loss else 0.0
            avg_sisdr = valid_sisdr / num_valid_batches if eval_sisdr else None

            if use_vad_loss or eval_sisdr:
                extras = [
                    f"spec={avg_spec:.4f}",
                    f"vad={avg_vad:.4f}",
                    f"speech={avg_speech:.4f}",
                    f"resid={avg_residual:.4f}",
                ]
                if use_vad_loss:
                    extras.append(f"p_ref={avg_p_ref:.2f}")
                    extras.append(f"p_out={avg_p_out:.2f}")
                    extras.append(f"gate={avg_gate:.0f}%")
                if avg_sisdr is not None:
                    extras.append(f"si-sdr={avg_sisdr:.2f}dB")
                print(f"{label} metrics: " + " | ".join(extras))

        return valid_loss / max(num_valid_batches, 1)

    # Flag to use compiled step (can be disabled for debugging)
    use_compiled_step = True
    print(f"  Using compiled training step: {use_compiled_step}")

    # Register SIGINT handler for graceful shutdown
    _register_sigint_handler(model, optimizer, ckpt_dir, train_config)
    print("  SIGINT handler registered (CTRL+C will save checkpoint before exit)")

    # Training loop
    print(f"\nStarting training (epoch {start_epoch + 1} to {epochs})...")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Est. total steps: {total_steps:,}")
    print()

    global_step = start_epoch * steps_per_epoch
    final_epoch = start_epoch
    avg_train_loss = float("nan")
    last_valid_loss: float | None = None
    last_valid_epoch: int | None = None

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
        train_p_ref = 0.0
        train_p_out = 0.0
        train_gate_pct = 0.0
        num_vad_logs = 0
        num_train_batches = 0
        samples_processed = 0
        grad_norm = 0.0
        loss_val = 0.0  # Initialize for async eval

        # Update interrupt state at start of epoch
        _update_interrupt_state(epoch + 1, 0.0, best_valid_loss)

        # Timing accumulators for verbose diagnostics
        total_data_time = 0.0
        total_forward_time = 0.0  # Used for compiled step timing

        # Create data iterator (MLXDataStream or PrefetchDataLoader)
        if use_mlx_stream and train_stream is not None:
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
        train_pbar = tqdm(
            enumerate(data_iterator),
            total=steps_per_epoch,
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

            # Update progress bar with real-time metrics (only on sync)
            if should_sync:
                lr = float(schedule(global_step))
                samples_per_sec = (
                    (current_batch_size * eval_frequency) / (step_time * eval_frequency) if step_time > 0 else 0
                )

                if use_vad_loss:
                    # Extra forward pass for component logging (only on sync steps)
                    out = model((noisy_real, noisy_imag), feat_erb, feat_spec)
                    spec_loss = spectral_loss(out, (clean_real, clean_imag))
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
                    spec_loss_val = float(spec_loss)
                    vad_loss_val = float(vad_loss)
                    speech_loss_val = float(speech_loss)
                    p_ref_mean = float(mx.mean(p_ref))
                    p_out_mean = float(mx.mean(p_out))
                    gate_pct = float(mx.mean(mx.where(gate > 0.0, 1.0, 0.0)))

                    train_spec_loss += spec_loss_val * eval_frequency
                    train_vad_loss += vad_loss_val * eval_frequency
                    train_speech_loss += speech_loss_val * eval_frequency
                    train_p_ref += p_ref_mean
                    train_p_out += p_out_mean
                    train_gate_pct += gate_pct
                    num_vad_logs += 1

                if verbose:
                    train_pbar.set_postfix(
                        loss=f"{loss_val:.4f}",
                        spec=f"{spec_loss_val:.4f}" if use_vad_loss else f"{loss_val:.4f}",
                        vad=f"{vad_loss_val:.4f}" if use_vad_loss else "0.0000",
                        speech=f"{speech_loss_val:.4f}" if use_vad_loss else "0.0000",
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
                        p_ref=f"{p_ref_mean:.2f}" if use_vad_loss else "0.00",
                        p_out=f"{p_out_mean:.2f}" if use_vad_loss else "0.00",
                        gate=f"{gate_pct:.0f}%" if use_vad_loss else "0%",
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
                save_checkpoint(
                    model,
                    ckpt_path,
                    epoch=epoch + 1,
                    loss=train_loss / num_train_batches if num_train_batches > 0 else loss_val,
                    best_valid_loss=best_valid_loss,
                    config=train_config,
                    optimizer=optimizer,
                )
                print(f"\n  📦 Checkpoint saved: {ckpt_path.name} (step {global_step})")

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
        avg_train_p_ref = train_p_ref / max(num_vad_logs, 1)
        avg_train_p_out = train_p_out / max(num_vad_logs, 1)
        avg_train_gate = train_gate_pct / max(num_vad_logs, 1)

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
        if (epoch + 1) % validate_every == 0:
            avg_valid_loss = run_validation("  Validating")
            last_valid_loss = avg_valid_loss
            last_valid_epoch = epoch

            # Early stopping check
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                epochs_without_improvement = 0

                # Save best model
                best_path = ckpt_dir / "best.safetensors"
                save_checkpoint(
                    model,
                    best_path,
                    epoch=epoch + 1,
                    loss=avg_train_loss,
                    best_valid_loss=best_valid_loss,
                    config=train_config,
                    optimizer=optimizer,
                )
            else:
                epochs_without_improvement += 1

        # ====== Epoch Summary ======
        epoch_time = time.time() - epoch_start
        epoch_throughput = samples_processed / epoch_time if epoch_time > 0 else 0

        # Update interrupt state with final epoch metrics
        _update_interrupt_state(epoch + 1, avg_train_loss, best_valid_loss)

        # Improved epoch summary with throughput
        improvement_marker = "★" if avg_valid_loss <= best_valid_loss else ""
        vad_summary = ""
        if use_vad_loss:
            vad_summary = (
                f" | Spec: {avg_train_spec_loss:.4f}"
                f" | VAD: {avg_train_vad_loss:.4f}"
                f" | Speech: {avg_train_speech_loss:.4f}"
            )

        print(
            f"✓ Epoch {epoch + 1}/{epochs} complete | "
            f"Train: {avg_train_loss:.4f}{vad_summary} | "
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

        # ====== Epoch-based Checkpointing ======
        if save_strategy == "epoch":
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.safetensors"
            save_checkpoint(
                model,
                ckpt_path,
                epoch=epoch + 1,
                loss=avg_train_loss,
                best_valid_loss=best_valid_loss,
                config=train_config,
                optimizer=optimizer,
            )
            print(f"  📦 Checkpoint saved: {ckpt_path.name}")
            if save_total_limit is not None:
                cleanup_checkpoints(ckpt_dir, save_total_limit)

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
        final_valid_loss = run_validation("  Final validation")
        last_valid_loss = final_valid_loss
        last_valid_epoch = final_epoch

    if final_valid_loss < best_valid_loss:
        best_valid_loss = final_valid_loss
        best_path = ckpt_dir / "best.safetensors"
        save_checkpoint(
            model,
            best_path,
            epoch=final_epoch + 1,
            loss=avg_train_loss,
            best_valid_loss=best_valid_loss,
            config=train_config,
            optimizer=optimizer,
        )
        print(f"  ✅ Final weights set new best: {best_valid_loss:.4f}")

    # Save final weights (even if not aligned to checkpoint interval).
    mx.eval(state)
    final_path = ckpt_dir / "final.safetensors"
    save_checkpoint(
        model,
        final_path,
        epoch=final_epoch + 1,
        loss=avg_train_loss,
        best_valid_loss=best_valid_loss,
        config=train_config,
        optimizer=optimizer,
        completed=True,
    )
    print(f"  📦 Final checkpoint saved: {final_path.name}")

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
        help="Checkpoint save strategy: 'no' (only best model), 'epoch' (every epoch), 'steps' (every N steps)",
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
        "--eval-sisdr",
        action="store_true",
        help="Compute SI-SDR during validation (slower)",
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
        eval_sisdr=args.eval_sisdr,
    )


if __name__ == "__main__":
    main()
