#!/usr/bin/env python3
"""Train MLX DeepFilterNet4 with real data from MLX datastore.

This script provides a complete training pipeline using the MLX datastore
format with pre-computed spectral features.

Usage:
    python -m df_mlx.train_with_data \
        --datastore /path/to/mlx_datastore \
        --epochs 100 \
        --batch-size 8 \
        --checkpoint-dir ./checkpoints

    # Resume from checkpoint
    python -m df_mlx.train_with_data \
        --datastore /path/to/mlx_datastore \
        --resume checkpoints/epoch_050.safetensors

Features:
    - Automatic learning rate scheduling (warmup + cosine decay)
    - Gradient clipping for stability
    - Periodic checkpointing
    - Validation loss tracking
    - Early stopping support
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    path: Path,
    *,
    epoch: int,
    loss: float,
    best_valid_loss: float,
) -> None:
    """Save a training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        path: Path to save to
        epoch: Current epoch
        loss: Current loss
        best_valid_loss: Best validation loss so far
    """
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
    }
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def load_checkpoint(model: nn.Module, path: str | Path) -> dict:
    """Load a training checkpoint.

    Args:
        model: Model to load weights into
        path: Path to checkpoint

    Returns:
        Training state dict
    """
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

    return state  # type: ignore[return-value]


def clip_grad_norm(grads, max_norm: float) -> Tuple[dict, float]:
    """Clip gradients by global norm.

    Args:
        grads: Nested dict/list of gradients
        max_norm: Maximum norm

    Returns:
        Tuple of (clipped_grads, grad_norm)
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

    return cast(dict, apply_clip(grads)), grad_norm_val


def train(
    datastore_dir: str,
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
    streaming: bool = False,
) -> None:
    """Train DfNet4 model with MLX datastore.

    Args:
        datastore_dir: Path to MLX datastore
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
        streaming: Use streaming data loader for large datasets
    """
    from df_mlx.datastore import MLXDataLoader, StreamingMLXDataLoader
    from df_mlx.model import count_parameters, init_model
    from df_mlx.train import ModelStatistics, WarmupCosineSchedule, spectral_loss

    print("=" * 60)
    print("MLX DeepFilterNet4 Training")
    print("=" * 60)
    print(f"Datastore:      {datastore_dir}")
    print(f"Epochs:         {epochs}")
    print(f"Batch size:     {batch_size}")
    print(f"Learning rate:  {learning_rate}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("=" * 60)

    # Create data loaders
    print("\nInitializing data loaders...")
    LoaderClass = StreamingMLXDataLoader if streaming else MLXDataLoader

    train_loader = LoaderClass(
        datastore_dir,
        split="train",
        batch_size=batch_size,
    )
    valid_loader = LoaderClass(
        datastore_dir,
        split="valid",
        batch_size=batch_size,
    )

    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Valid batches: {len(valid_loader):,}")

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

    # Learning rate schedule
    steps_per_epoch = len(train_loader)
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

    # Statistics tracker
    stats = ModelStatistics()

    # Create checkpoint directory
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Loss function
    def loss_fn(model, spec, feat_erb, feat_spec, target):
        """Compute training loss."""
        out = model(spec, feat_erb, feat_spec)
        return spectral_loss(out, target)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    print(f"\nStarting training (epoch {start_epoch + 1} to {epochs})...")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Total steps:  {total_steps:,}")
    print()

    global_step = start_epoch * steps_per_epoch
    final_epoch = start_epoch

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        final_epoch = epoch

        # ====== Training ======
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        samples_processed = 0
        grad_norm = 0.0

        # Training progress bar
        train_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )

        for batch_idx, batch in train_pbar:
            step_start = time.time()

            # Unpack batch - format from datastore
            spec = batch["spec"]
            feat_erb = batch["feat_erb"]
            feat_spec = batch["feat_spec"]
            target = batch["target"]

            current_batch_size = feat_erb.shape[0]  # type: ignore[union-attr]

            # Forward and backward pass
            loss, grads = loss_and_grad(model, spec, feat_erb, feat_spec, target)

            # Gradient clipping (returns clipped grads and norm)
            if max_grad_norm > 0:
                grads, grad_norm = clip_grad_norm(grads, max_grad_norm)

            # Update parameters
            optimizer.update(model, grads)

            # Force evaluation
            mx.eval(loss, model.parameters())

            # Update learning rate
            current_lr = schedule(global_step)

            # Track statistics
            step_time = time.time() - step_start
            loss_val = float(loss)
            stats.update(loss=loss_val, lr=current_lr, step_time=step_time)

            train_loss += loss_val
            num_train_batches += 1
            samples_processed += current_batch_size
            global_step += 1

            # Update progress bar with real-time metrics
            samples_per_sec = current_batch_size / step_time if step_time > 0 else 0
            train_pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                avg=f"{train_loss / num_train_batches:.4f}",
                lr=f"{current_lr:.1e}",
                grad=f"{grad_norm:.2f}",
                speed=f"{samples_per_sec:.0f}s/s",
            )

        train_pbar.close()
        avg_train_loss = train_loss / max(num_train_batches, 1)

        # ====== Validation ======
        avg_valid_loss = 0.0
        if (epoch + 1) % validate_every == 0 and len(valid_loader) > 0:
            model.eval()
            valid_loss = 0.0
            num_valid_batches = 0

            # Validation progress bar
            valid_pbar = tqdm(
                valid_loader,
                total=len(valid_loader),
                desc="  Validating",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )

            for batch in valid_pbar:
                spec = batch["spec"]
                feat_erb = batch["feat_erb"]
                feat_spec = batch["feat_spec"]
                target = batch["target"]

                out = model(spec, feat_erb, feat_spec)
                loss = spectral_loss(out, target)
                mx.eval(loss)
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
                save_checkpoint(
                    model,
                    optimizer,
                    ckpt_dir / "best.safetensors",
                    epoch=epoch + 1,
                    loss=avg_valid_loss,
                    best_valid_loss=best_valid_loss,
                )
            else:
                epochs_without_improvement += 1

        # Print epoch summary with throughput
        epoch_time = time.time() - epoch_start
        epoch_throughput = samples_processed / epoch_time if epoch_time > 0 else 0
        improvement_marker = "★" if avg_valid_loss <= best_valid_loss and avg_valid_loss > 0 else ""
        print(
            f"✓ Epoch {epoch + 1}/{epochs} complete | "
            f"Train: {avg_train_loss:.4f} | "
            f"Valid: {avg_valid_loss:.4f} {improvement_marker}| "
            f"Best: {best_valid_loss:.4f} | "
            f"{samples_processed:,} samples @ {epoch_throughput:.0f}/s | "
            f"{epoch_time:.1f}s"
        )

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:04d}.safetensors"
            save_checkpoint(
                model,
                optimizer,
                ckpt_path,
                epoch=epoch + 1,
                loss=avg_train_loss,
                best_valid_loss=best_valid_loss,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break

    # Save final checkpoint
    final_path = ckpt_dir / "final.safetensors"
    save_checkpoint(
        model,
        optimizer,
        final_path,
        epoch=final_epoch + 1,
        loss=avg_train_loss,
        best_valid_loss=best_valid_loss,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Final checkpoint:     {final_path}")
    print(f"Best checkpoint:      {ckpt_dir / 'best.safetensors'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train MLX DeepFilterNet4 with real data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datastore",
        required=True,
        help="Path to MLX datastore directory",
    )
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
        default="checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume",
        default=None,
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
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm (0 to disable clipping)",
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
        help="Early stopping patience (0 to disable)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming data loader for large datasets",
    )

    args = parser.parse_args()

    train(
        datastore_dir=args.datastore,
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
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()
