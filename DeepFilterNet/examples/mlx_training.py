#!/usr/bin/env python3
"""MLX DeepFilterNet4 training example.

This example demonstrates how to:
1. Set up training configuration
2. Create a training loop with the Trainer class
3. Handle checkpointing and logging
4. Use learning rate scheduling

Requirements:
    pip install mlx soundfile h5py numpy

Usage:
    python mlx_training.py --data train_data.hdf5 --epochs 100
    python mlx_training.py --data train.hdf5 --resume checkpoint.safetensors

This is a simplified training loop. For production training,
see the full df_mlx/train.py module and training scripts.
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_synthetic_batch(batch_size: int = 4, time_steps: int = 50):
    """Create synthetic training batch for demonstration.

    In real training, you would load from your HDF5 dataset.

    Args:
        batch_size: Number of samples per batch
        time_steps: Number of time frames

    Returns:
        Dictionary with spec, feat_erb, feat_spec, and target
    """
    n_freqs = 481  # FFT bins
    erb_bands = 32
    df_bins = 96

    # Create random spectral data (real and imaginary components)
    spec_real = mx.random.normal(shape=(batch_size, time_steps, n_freqs)) * 0.1
    spec_imag = mx.random.normal(shape=(batch_size, time_steps, n_freqs)) * 0.1

    # ERB features
    feat_erb = mx.random.normal(shape=(batch_size, time_steps, erb_bands))

    # DF-band spectral features (real/imag stacked)
    feat_spec = mx.random.normal(shape=(batch_size, time_steps, df_bins, 2))

    # Target (clean spectrum) - similar to input for synthetic data
    target_real = mx.random.normal(shape=(batch_size, time_steps, n_freqs)) * 0.1
    target_imag = mx.random.normal(shape=(batch_size, time_steps, n_freqs)) * 0.1

    return {
        "spec": (spec_real, spec_imag),
        "feat_erb": feat_erb,
        "feat_spec": feat_spec,
        "target": (target_real, target_imag),
    }


def simple_training_loop(
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    resume_from: str | None = None,
):
    """Simple training loop demonstrating MLX DeepFilterNet4 training.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        checkpoint_dir: Directory for checkpoints
        resume_from: Optional checkpoint to resume from
    """
    from df_mlx.model import count_parameters, init_model
    from df_mlx.train import ModelStatistics, WarmupCosineSchedule, combined_loss, save_checkpoint

    # Initialize model
    print("Initializing model...")
    model = init_model()
    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,}")

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from:
        from df_mlx.train import load_checkpoint

        checkpoint = load_checkpoint(model, resume_from)
        start_epoch = checkpoint.get("epoch", 0)
        print(f"  Resumed from: {resume_from} (epoch {start_epoch})")

    # Create optimizer with learning rate schedule
    steps_per_epoch = 100  # Simulated
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(500, total_steps // 10)

    schedule = WarmupCosineSchedule(
        base_lr=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=learning_rate * 0.01,
    )

    optimizer = optim.AdamW(learning_rate=schedule)

    # Initialize statistics tracker
    stats = ModelStatistics()

    # Create checkpoint directory
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps:  {total_steps}")
    print(f"  Learning rate: {learning_rate}")
    print()

    global_step = start_epoch * steps_per_epoch

    # Loss and gradient function
    def loss_fn(model, spec, feat_erb, feat_spec, target):
        """Compute loss for a batch."""
        # Forward pass
        out_real, out_imag = model(spec, feat_erb, feat_spec)

        # Compute combined loss
        loss = combined_loss(
            pred=(out_real, out_imag),
            target=target,
            alpha_spec=1.0,
            alpha_time=0.1,
        )

        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx in range(steps_per_epoch):
            step_start = time.time()

            # Get batch (synthetic data for demo)
            batch = create_synthetic_batch(batch_size)

            # Forward and backward pass
            loss, grads = loss_and_grad(
                model,
                batch["spec"],
                batch["feat_erb"],
                batch["feat_spec"],
                batch["target"],
            )

            # Update parameters
            optimizer.update(model, grads)

            # Force evaluation
            mx.eval(loss, model.parameters())

            # Update learning rate
            current_lr = schedule(global_step)

            # Track statistics
            step_time = time.time() - step_start
            stats.update(
                loss=float(loss),
                lr=current_lr,
                step_time=step_time,
            )

            epoch_loss += float(loss)
            num_batches += 1
            global_step += 1

            # Print progress occasionally
            if (batch_idx + 1) % 20 == 0:
                avg_loss = epoch_loss / num_batches
                print(
                    f"  Epoch {epoch + 1}/{num_epochs} | "
                    f"Step {batch_idx + 1}/{steps_per_epoch} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e}"
                )

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches

        print(f"\nEpoch {epoch + 1} complete:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.safetensors"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                path=str(ckpt_path),
                epoch=epoch + 1,
                step=global_step,
                loss=avg_loss,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

        print()

    # Save final statistics
    stats_path = ckpt_dir / "training_stats.json"
    stats.save(str(stats_path))
    print(f"Training complete! Stats saved to {stats_path}")

    # Print summary
    summary = stats.summary()
    print("\nTraining Summary:")
    print(f"  Total steps: {summary['steps']}")
    print(f"  Final loss: {summary['loss_mean']:.4f}")
    print(f"  Avg step time: {summary['step_time_mean'] * 1000:.1f}ms")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train MLX DeepFilterNet4 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training (uses synthetic data)
    python mlx_training.py --epochs 10

    # With checkpoint directory
    python mlx_training.py --epochs 100 --checkpoint-dir ./checkpoints

    # Resume from checkpoint
    python mlx_training.py --epochs 100 --resume checkpoints/epoch_050.safetensors

Note: This example uses synthetic data for demonstration.
For real training, modify create_synthetic_batch() to load your dataset.
        """,
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "-c",
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints)",
    )
    parser.add_argument(
        "--resume",
        "-r",
        help="Resume from checkpoint",
        default=None,
    )

    args = parser.parse_args()

    print("MLX DeepFilterNet4 Training Example")
    print("=" * 50)

    simple_training_loop(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
