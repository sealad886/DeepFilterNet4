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
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


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
    print(f"  MLX version:    {mx.__version__ if hasattr(mx, '__version__') else 'Unknown'}")

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


def clip_grad_norm(grads, max_norm: float) -> Tuple[dict, float]:
    """Clip gradients by global norm.

    Returns:
        Tuple of (clipped_grads, grad_norm) for monitoring.
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
        return grads, 0.0

    total_norm_sq = sum(mx.sum(g**2) for g in flat_grads)
    total_norm = mx.sqrt(total_norm_sq)
    grad_norm_val = float(total_norm)

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

    return apply_clip(grads), grad_norm_val


def save_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    epoch: int,
    loss: float,
    best_valid_loss: float,
    config: dict,
) -> None:
    """Save a training checkpoint."""
    from mlx.utils import tree_flatten

    path = Path(path)

    # Flatten nested params for safetensors
    params = model.parameters()
    flat_params = tree_flatten(params)
    weights = {k: v for k, v in flat_params}
    mx.save_safetensors(str(path), weights)

    # Save training state
    state_path = path.with_suffix(".state.json")
    state = {
        "epoch": epoch,
        "loss": loss,
        "best_valid_loss": best_valid_loss,
        "config": config,
    }
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def load_checkpoint(model: nn.Module, path: str | Path) -> dict:
    """Load a training checkpoint."""
    ckpt_path = Path(path)

    # Load weights
    weights = mx.load(str(ckpt_path))
    model.load_weights(weights)

    # Load training state
    state_path = ckpt_path.with_suffix(".state.json")
    state = {}
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)

    return state


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
    checkpoint_every: int = 5,
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
    verbose: bool = False,
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
        checkpoint_every: Save checkpoint every N epochs
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
        verbose: Enable detailed timing and diagnostic output
    """
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

    # Create dataset (this populates config.*_files from cache index if using cache)
    print("\nInitializing dynamic dataset...")
    dataset = DynamicDataset(config)

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
    print("=" * 60)

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

    # Initialize model
    print("\nInitializing model...")
    model = init_model()
    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,}")

    # Resume from checkpoint if provided
    start_epoch = 0
    best_valid_loss = float("inf")
    epochs_without_improvement = 0

    if resume_from:
        state = load_checkpoint(model, resume_from)
        start_epoch = state.get("epoch", 0)
        best_valid_loss = state.get("best_valid_loss", float("inf"))
        print(f"  Resumed from: {resume_from} (epoch {start_epoch})")

    # Estimate steps per epoch (approximate since samples may be skipped)
    approx_samples_per_epoch = len(dataset)
    steps_per_epoch = approx_samples_per_epoch // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    schedule = WarmupCosineSchedule(
        base_lr=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=learning_rate * 0.01,
    )

    # Optimizer
    optimizer = optim.AdamW(learning_rate=schedule)

    # Loss function - define as a pure function for compilation
    def loss_fn(model, noisy_real, noisy_imag, feat_erb, feat_spec, clean_real, clean_imag):
        """Compute training loss."""
        # Model expects spec as tuple (real, imag)
        noisy_spec = (noisy_real, noisy_imag)
        target_spec = (clean_real, clean_imag)

        out = model(noisy_spec, feat_erb, feat_spec)
        return spectral_loss(out, target_spec)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Compiled training step for performance optimization
    # Captures model and optimizer state for graph tracing
    state = [model.state, optimizer.state]

    from functools import partial

    @partial(mx.compile, inputs=state, outputs=state)
    def compiled_step(noisy_real, noisy_imag, feat_erb, feat_spec, clean_real, clean_imag, max_grad_norm_val):
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
        )
        # Gradient clipping inline
        if max_grad_norm_val > 0:
            grads, _ = clip_grad_norm(grads, max_grad_norm_val)
        optimizer.update(model, grads)
        return loss

    # Flag to use compiled step (can be disabled for debugging)
    use_compiled_step = True
    print(f"  Using compiled training step: {use_compiled_step}")

    # Training loop
    print(f"\nStarting training (epoch {start_epoch + 1} to {epochs})...")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Est. total steps: {total_steps:,}")
    print()

    global_step = start_epoch * steps_per_epoch
    final_epoch = start_epoch

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        final_epoch = epoch

        # Set epoch for reproducible shuffling
        dataset.set_split("train")
        dataset.set_epoch(epoch)

        # ====== Training ======
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        samples_processed = 0
        grad_norm = 0.0

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
            dynamic_ncols=True,
            leave=True,
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

            # Convert to FP16 if enabled (mixed precision training)
            if use_fp16:
                noisy_real = noisy_real.astype(mx.float16)
                noisy_imag = noisy_imag.astype(mx.float16)
                clean_real = clean_real.astype(mx.float16)
                clean_imag = clean_imag.astype(mx.float16)
                feat_erb = feat_erb.astype(mx.float16)
                feat_spec = feat_spec.astype(mx.float16)

            current_batch_size = noisy_real.shape[0]

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
                    max_grad_norm,
                )
                # Evaluate state
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
                )
                # Force evaluation to get accurate timing
                mx.eval(loss)

                # Gradient clipping (returns clipped grads and norm)
                if max_grad_norm > 0:
                    grads, grad_norm = clip_grad_norm(grads, max_grad_norm)

                # Update parameters
                optimizer.update(model, grads)

                # Force evaluation
                mx.eval(model.parameters(), optimizer.state)

            fwd_time = time.time() - fwd_start
            total_forward_time += fwd_time

            step_time = time.time() - step_start
            loss_val = float(loss)
            train_loss += loss_val
            num_train_batches += 1
            samples_processed += current_batch_size
            global_step += 1

            # Update progress bar with real-time metrics
            lr = float(schedule(global_step))
            samples_per_sec = current_batch_size / step_time if step_time > 0 else 0

            if verbose:
                # Show timing breakdown in verbose mode
                train_pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    lr=f"{lr:.1e}",
                    data=f"{data_time * 1000:.0f}ms",
                    step=f"{fwd_time * 1000:.0f}ms",
                    spd=f"{samples_per_sec:.0f}/s",
                    compiled="✓" if use_compiled_step else "✗",
                )
            else:
                train_pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    avg=f"{train_loss / num_train_batches:.4f}",
                    lr=f"{lr:.1e}",
                    grad=f"{grad_norm:.2f}",
                    speed=f"{samples_per_sec:.0f}s/s",
                )

            # Save data checkpoint periodically (for resume capability)
            if checkpoint_batches > 0 and use_mlx_stream and train_stream is not None:
                if (batch_idx + 1) % checkpoint_batches == 0:
                    train_stream.save_checkpoint(data_checkpoint_path)

            # Start timing for next data fetch
            data_start = time.time()

        train_pbar.close()

        # Save data checkpoint at end of epoch (for clean resume at epoch boundary)
        if use_mlx_stream and train_stream is not None:
            train_stream.save_checkpoint(data_checkpoint_path)

        avg_train_loss = train_loss / max(num_train_batches, 1)

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
            model.eval()

            dataset.set_split("valid")
            dataset.set_epoch(0)  # Fixed epoch for reproducible validation

            valid_loss = 0.0
            num_valid_batches = 0
            valid_steps = len(dataset) // batch_size

            valid_loader = PrefetchDataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=2,
                prefetch_factor=1,
            )

            # Validation progress bar
            valid_pbar = tqdm(
                valid_loader,
                total=valid_steps,
                desc="  Validating",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )

            for batch in valid_pbar:
                noisy_real = batch["noisy_real"]
                noisy_imag = batch["noisy_imag"]
                clean_real = batch["clean_real"]
                clean_imag = batch["clean_imag"]
                feat_erb = batch["feat_erb"]
                feat_spec = batch["feat_spec"]

                # Model expects spec as tuple (real, imag)
                noisy_spec = (noisy_real, noisy_imag)
                target_spec = (clean_real, clean_imag)

                out = model(noisy_spec, feat_erb, feat_spec)
                loss = spectral_loss(out, target_spec)

                loss_val = float(loss)
                valid_loss += loss_val
                num_valid_batches += 1

                # Update validation progress
                valid_pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    avg=f"{valid_loss / num_valid_batches:.4f}",
                )

            valid_pbar.close()
            avg_valid_loss = valid_loss / max(num_valid_batches, 1)

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
                    config=config.__dict__,
                )
            else:
                epochs_without_improvement += 1

        # ====== Epoch Summary ======
        epoch_time = time.time() - epoch_start
        epoch_throughput = samples_processed / epoch_time if epoch_time > 0 else 0

        # Improved epoch summary with throughput
        improvement_marker = "★" if avg_valid_loss <= best_valid_loss else ""
        print(
            f"✓ Epoch {epoch + 1}/{epochs} complete | "
            f"Train: {avg_train_loss:.4f} | "
            f"Valid: {avg_valid_loss:.4f} {improvement_marker}| "
            f"Best: {best_valid_loss:.4f} | "
            f"{samples_processed:,} samples @ {epoch_throughput:.0f}/s | "
            f"{epoch_time:.1f}s"
        )

        # ====== Checkpointing ======
        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.safetensors"
            save_checkpoint(
                model,
                ckpt_path,
                epoch=epoch + 1,
                loss=avg_train_loss,
                best_valid_loss=best_valid_loss,
                config=config.__dict__,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

        # ====== Early Stopping ======
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break

        # Clear memory periodically
        if (epoch + 1) % 10 == 0:
            gc.collect()

    # ====== Final Summary ======
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final epoch:     {final_epoch + 1}")
    print(f"Best valid loss: {best_valid_loss:.4f}")
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
        type=str,
        help="Resume model from checkpoint",
    )
    parser.add_argument(
        "--resume-data",
        type=str,
        help="Resume data loading from checkpoint (for interrupted epochs)",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=1,
        help="Validate every N epochs",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
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
        resume_from=args.resume,
        resume_data_from=args.resume_data,
        validate_every=args.validate_every,
        checkpoint_every=args.checkpoint_every,
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
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
