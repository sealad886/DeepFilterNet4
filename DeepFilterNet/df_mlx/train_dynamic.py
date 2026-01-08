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
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent))


def clip_grad_norm(grads, max_norm: float):
    """Clip gradients by global norm."""
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
        return grads

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

    return apply_clip(grads)


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
    speech_list: str | None = None,
    noise_list: str | None = None,
    rir_list: str | None = None,
    config_path: str | None = None,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    resume_from: str | None = None,
    validate_every: int = 1,
    checkpoint_every: int = 5,
    max_grad_norm: float = 1.0,
    warmup_epochs: int = 5,
    patience: int = 10,
    num_workers: int = 4,
    p_reverb: float = 0.5,
    p_clipping: float = 0.0,
) -> None:
    """Train DfNet4 model with dynamic on-the-fly mixing.

    Args:
        speech_list: Path to file containing speech file paths
        noise_list: Path to file containing noise file paths
        rir_list: Path to file containing RIR file paths
        config_path: Optional path to JSON config file
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        checkpoint_dir: Directory for checkpoints
        resume_from: Optional checkpoint to resume from
        validate_every: Validate every N epochs
        checkpoint_every: Save checkpoint every N epochs
        max_grad_norm: Maximum gradient norm for clipping
        warmup_epochs: Number of warmup epochs
        patience: Early stopping patience
        num_workers: Number of data loading workers
        p_reverb: Probability of applying reverb
        p_clipping: Probability of clipping distortion
    """
    from df_mlx.dynamic_dataset import DatasetConfig, DynamicDataset, PrefetchDataLoader, read_file_list
    from df_mlx.model import count_parameters, init_model
    from df_mlx.train import WarmupCosineSchedule, spectral_loss

    print("=" * 60)
    print("MLX DeepFilterNet4 Training - Dynamic On-the-Fly Mixing")
    print("=" * 60)

    # Load or create config
    if config_path:
        config = DatasetConfig.from_json(config_path)
        print(f"Loaded config from: {config_path}")
    else:
        if not speech_list:
            raise ValueError("Either --config or --speech-list required")

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

    # Create dataset
    print("\nInitializing dynamic dataset...")
    dataset = DynamicDataset(config)
    dataset.set_split("train")

    print(f"  Train samples: {len(dataset):,}")

    # Create validation dataset (with reproducible indices)
    dataset.set_split("valid")
    print(f"  Valid samples: {len(dataset):,}")

    # Reset to training
    dataset.set_split("train")
    dataset.set_epoch(0)

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

    # Create checkpoint directory
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Loss function
    def loss_fn(model, noisy_real, noisy_imag, feat_erb, feat_spec, clean_real, clean_imag):
        """Compute training loss."""
        # Reconstruct complex spec as stacked real/imag
        noisy_spec = mx.stack([noisy_real, noisy_imag], axis=-1)
        target_spec = mx.stack([clean_real, clean_imag], axis=-1)

        out = model(noisy_spec, feat_erb, feat_spec)
        return spectral_loss(out, target_spec)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

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

        data_loader = PrefetchDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=config.num_workers,
            prefetch_factor=2,
        )

        for batch_idx, batch in enumerate(data_loader):
            step_start = time.time()

            # Unpack batch
            noisy_real = batch["noisy_real"]
            noisy_imag = batch["noisy_imag"]
            clean_real = batch["clean_real"]
            clean_imag = batch["clean_imag"]
            feat_erb = batch["feat_erb"]
            feat_spec = batch["feat_spec"]

            # Forward and backward pass
            loss, grads = loss_and_grad(
                model,
                noisy_real,
                noisy_imag,
                feat_erb,
                feat_spec,
                clean_real,
                clean_imag,
            )

            # Gradient clipping
            if max_grad_norm > 0:
                grads = clip_grad_norm(grads, max_grad_norm)

            # Update parameters
            optimizer.update(model, grads)

            # Force evaluation
            mx.eval(model.parameters(), optimizer.state)

            train_loss += float(loss)
            num_train_batches += 1
            global_step += 1

            # Progress update
            step_time = time.time() - step_start
            if batch_idx % 50 == 0:
                lr = schedule(global_step)
                print(
                    f"  Epoch {epoch + 1}/{epochs} | "
                    f"Step {batch_idx}/{steps_per_epoch} | "
                    f"Loss: {float(loss):.4f} | "
                    f"LR: {float(lr):.2e} | "
                    f"Time: {step_time * 1000:.0f}ms"
                )

        avg_train_loss = train_loss / max(num_train_batches, 1)

        # ====== Validation ======
        avg_valid_loss = float("inf")
        if (epoch + 1) % validate_every == 0:
            model.eval()

            dataset.set_split("valid")
            dataset.set_epoch(0)  # Fixed epoch for reproducible validation

            valid_loss = 0.0
            num_valid_batches = 0

            valid_loader = PrefetchDataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=2,
                prefetch_factor=1,
            )

            for batch in valid_loader:
                noisy_real = batch["noisy_real"]
                noisy_imag = batch["noisy_imag"]
                clean_real = batch["clean_real"]
                clean_imag = batch["clean_imag"]
                feat_erb = batch["feat_erb"]
                feat_spec = batch["feat_spec"]

                noisy_spec = mx.stack([noisy_real, noisy_imag], axis=-1)
                target_spec = mx.stack([clean_real, clean_imag], axis=-1)

                out = model(noisy_spec, feat_erb, feat_spec)
                loss = spectral_loss(out, target_spec)

                valid_loss += float(loss)
                num_valid_batches += 1

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
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Valid: {avg_valid_loss:.4f} | "
            f"Best: {best_valid_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
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

    # Data sources
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
        help="Resume from checkpoint",
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

    args = parser.parse_args()

    train(
        speech_list=args.speech_list,
        noise_list=args.noise_list,
        rir_list=args.rir_list,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        validate_every=args.validate_every,
        checkpoint_every=args.checkpoint_every,
        max_grad_norm=args.max_grad_norm,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        p_reverb=args.p_reverb,
        p_clipping=args.p_clipping,
    )


if __name__ == "__main__":
    main()
