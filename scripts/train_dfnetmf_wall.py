#!/usr/bin/env python3
"""Training script for DFNetMF on Wall camera dataset.

This script trains DFNetMF (Multi-Frame filtering) model to remove
stationary noise (clicking, interference) from security camera audio.

Usage:
    python train_dfnetmf_wall.py --dataset /Users/andrew/DataDump/datasets/wall_processed
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Add DeepFilterNet to path
SCRIPT_DIR = Path(__file__).parent
DFNET_DIR = SCRIPT_DIR.parent / "DeepFilterNet"
sys.path.insert(0, str(DFNET_DIR))


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Dataset
    dataset_dir: str = "/Users/andrew/DataDump/datasets/wall_processed"
    checkpoint_dir: str = "/Users/andrew/DataDump/checkpoints/dfnetmf_wall"

    # Model
    mfop_method: str = "WF"  # "WF" or "MVDR"
    df_order: int = 5
    nb_erb: int = 32
    nb_df: int = 96

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    warmup_steps: int = 1000

    # Audio
    sr: int = 48000
    fft_size: int = 960
    hop_size: int = 480
    segment_length: int = 48000 * 2  # 2 seconds

    # Logging
    log_interval: int = 10
    save_interval: int = 500
    val_interval: int = 100


def load_audio_pair(noisy_path: Path, clean_path: Path, config: TrainingConfig) -> Optional[Tuple[mx.array, mx.array]]:
    """Load a noisy/clean audio pair."""
    try:
        import soundfile as sf

        noisy, sr = sf.read(noisy_path)
        clean, _ = sf.read(clean_path)

        # Ensure same length
        min_len = min(len(noisy), len(clean))
        noisy = noisy[:min_len]
        clean = clean[:min_len]

        # Resample if needed
        if sr != config.sr:
            import librosa

            noisy = librosa.resample(noisy, orig_sr=sr, target_sr=config.sr)
            clean = librosa.resample(clean, orig_sr=sr, target_sr=config.sr)

        # Normalize
        max_val = max(np.abs(noisy).max(), np.abs(clean).max(), 1e-8)
        noisy = noisy / max_val
        clean = clean / max_val

        return mx.array(noisy.astype(np.float32)), mx.array(clean.astype(np.float32))
    except Exception as e:
        print(f"Error loading pair {noisy_path}: {e}")
        return None


class WallDataset:
    """Dataset loader for Wall camera training pairs."""

    def __init__(self, dataset_dir: Path, config: TrainingConfig, split: str = "train"):
        self.dataset_dir = dataset_dir
        self.config = config
        self.split = split

        # Load manifest
        manifest_path = dataset_dir / "manifests" / "train.tsv"
        self.pairs = []

        if manifest_path.exists():
            with open(manifest_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        noisy, clean = parts
                        if Path(noisy).exists() and Path(clean).exists():
                            self.pairs.append((Path(noisy), Path(clean)))

        # Split into train/val (90/10)
        np.random.seed(42)
        indices = np.random.permutation(len(self.pairs))
        split_idx = int(len(indices) * 0.9)

        if split == "train":
            self.pairs = [self.pairs[i] for i in indices[:split_idx]]
        else:
            self.pairs = [self.pairs[i] for i in indices[split_idx:]]

        print(f"Loaded {len(self.pairs)} {split} pairs")

    def __len__(self):
        return len(self.pairs)

    def get_batch(self, batch_size: int) -> Optional[Tuple[mx.array, mx.array]]:
        """Get a random batch of audio pairs."""
        if len(self.pairs) == 0:
            return None

        indices = np.random.choice(len(self.pairs), min(batch_size, len(self.pairs)), replace=False)

        noisy_batch = []
        clean_batch = []

        for idx in indices:
            noisy_path, clean_path = self.pairs[idx]
            result = load_audio_pair(noisy_path, clean_path, self.config)
            if result:
                noisy, clean = result
                # Ensure consistent length
                target_len = self.config.segment_length
                if len(noisy) >= target_len:
                    start = np.random.randint(0, len(noisy) - target_len + 1)
                    noisy = noisy[start : start + target_len]
                    clean = clean[start : start + target_len]
                else:
                    # Pad
                    pad_len = target_len - len(noisy)
                    noisy = mx.pad(noisy, [(0, pad_len)])
                    clean = mx.pad(clean, [(0, pad_len)])

                noisy_batch.append(noisy)
                clean_batch.append(clean)

        if not noisy_batch:
            return None

        return mx.stack(noisy_batch), mx.stack(clean_batch)


def stft(audio: mx.array, fft_size: int, hop_size: int) -> mx.array:
    """Compute STFT.

    Args:
        audio: [B, T] audio signal
        fft_size: FFT size
        hop_size: Hop size

    Returns:
        spec: [B, T, F, 2] complex spectrum as real
    """
    from df_mlx.ops import stft as mlx_stft

    # mlx_stft returns (real, imag) tuple
    spec_real, spec_imag = mlx_stft(audio, fft_size, hop_size)
    # Stack to [B, T, F, 2]
    return mx.stack([spec_real, spec_imag], axis=-1)


def istft(spec: mx.array, fft_size: int, hop_size: int) -> mx.array:
    """Compute inverse STFT.

    Args:
        spec: [B, T, F, 2] complex spectrum as real

    Returns:
        audio: [B, T] reconstructed audio
    """
    from df_mlx.ops import istft as mlx_istft

    return mlx_istft(spec, fft_size, hop_size)


def compute_erb_features(spec: mx.array, erb_fb: mx.array) -> mx.array:
    """Compute ERB features from spectrum.

    Args:
        spec: [B, T, F, 2] complex spectrum
        erb_fb: [F, E] ERB filterbank

    Returns:
        feat_erb: [B, T, E, 1] ERB features
    """
    # Compute magnitude
    mag = mx.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2 + 1e-8)
    # Apply ERB filterbank
    erb = mx.matmul(mag, erb_fb)  # [B, T, E]
    # Log scale
    erb = mx.log(erb + 1e-8)
    # Add channel dim
    return mx.expand_dims(erb, axis=-1)


def compute_loss(
    model: nn.Module,
    noisy_audio: mx.array,
    clean_audio: mx.array,
    erb_fb: mx.array,
    erb_inv: mx.array,
    config: TrainingConfig,
) -> Tuple[mx.array, Dict[str, float]]:
    """Compute training loss.

    Uses a combination of:
    - L1 loss on waveform
    - Multi-resolution STFT loss
    - ERB band loss
    """
    fft_size = config.fft_size
    hop_size = config.hop_size
    nb_df = config.nb_df

    # Compute spectrograms
    noisy_spec = stft(noisy_audio, fft_size, hop_size)  # [B, T, F, 2]
    clean_spec = stft(clean_audio, fft_size, hop_size)

    # Compute features
    feat_erb = compute_erb_features(noisy_spec, erb_fb)  # [B, T, E, 1]
    feat_spec = noisy_spec[:, :, :nb_df, :]  # [B, T, nb_df, 2]

    # Forward pass
    enhanced_spec, mask, lsnr, _ = model(noisy_spec, feat_erb, feat_spec)

    # Reconstruct audio
    enhanced_audio = istft(enhanced_spec, fft_size, hop_size)

    # Ensure same length
    min_len = min(enhanced_audio.shape[-1], clean_audio.shape[-1])
    enhanced_audio = enhanced_audio[..., :min_len]
    clean_audio = clean_audio[..., :min_len]

    # L1 waveform loss
    l1_loss = mx.mean(mx.abs(enhanced_audio - clean_audio))

    # Spectral loss (magnitude)
    enhanced_mag = mx.sqrt(enhanced_spec[..., 0] ** 2 + enhanced_spec[..., 1] ** 2 + 1e-8)
    clean_mag = mx.sqrt(clean_spec[..., 0] ** 2 + clean_spec[..., 1] ** 2 + 1e-8)

    # Match time dimension
    min_t = min(enhanced_mag.shape[1], clean_mag.shape[1])
    enhanced_mag = enhanced_mag[:, :min_t, :]
    clean_mag = clean_mag[:, :min_t, :]

    spec_loss = mx.mean(mx.abs(enhanced_mag - clean_mag))

    # Log magnitude loss
    log_spec_loss = mx.mean(mx.abs(mx.log(enhanced_mag + 1e-8) - mx.log(clean_mag + 1e-8)))

    # Total loss
    total_loss = l1_loss + 0.5 * spec_loss + 0.25 * log_spec_loss

    metrics = {
        "total": float(total_loss),
        "l1": float(l1_loss),
        "spec": float(spec_loss),
        "log_spec": float(log_spec_loss),
    }

    return total_loss, metrics


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    noisy_audio: mx.array,
    clean_audio: mx.array,
    erb_fb: mx.array,
    erb_inv: mx.array,
    config: TrainingConfig,
) -> Dict[str, float]:
    """Single training step."""

    def loss_fn(model):
        loss, metrics = compute_loss(model, noisy_audio, clean_audio, erb_fb, erb_inv, config)
        return loss, metrics

    # Compute gradients
    (loss, metrics), grads = nn.value_and_grad(model, loss_fn)(model)

    # Update parameters
    optimizer.update(model, grads)

    return metrics


def validate(
    model: nn.Module,
    val_dataset: WallDataset,
    erb_fb: mx.array,
    erb_inv: mx.array,
    config: TrainingConfig,
    num_batches: int = 10,
) -> Dict[str, float]:
    """Run validation."""
    total_metrics = {"total": 0.0, "l1": 0.0, "spec": 0.0, "log_spec": 0.0}
    count = 0

    for _ in range(num_batches):
        batch = val_dataset.get_batch(config.batch_size)
        if batch is None:
            continue

        noisy, clean = batch
        _, metrics = compute_loss(model, noisy, clean, erb_fb, erb_inv, config)
        mx.eval(metrics)

        for k, v in metrics.items():
            total_metrics[k] += v
        count += 1

    if count > 0:
        for k in total_metrics:
            total_metrics[k] /= count

    return total_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path,
):
    """Save training checkpoint."""
    from mlx.utils import tree_flatten

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    weights_path = checkpoint_dir / f"step_{step:06d}.safetensors"
    flat_params = tree_flatten(model.parameters())
    mx.save_safetensors(str(weights_path), dict(flat_params))

    # Save training state
    state = {
        "step": step,
        "epoch": epoch,
        "metrics": metrics,
    }
    state_path = checkpoint_dir / f"step_{step:06d}_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"Saved checkpoint: {weights_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DFNetMF on Wall camera dataset")
    parser.add_argument("--dataset", type=str, default="/Users/andrew/DataDump/datasets/wall_processed")
    parser.add_argument("--checkpoint-dir", type=str, default="/Users/andrew/DataDump/checkpoints/dfnetmf_wall")
    parser.add_argument("--method", type=str, default="WF", choices=["WF", "MVDR"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    args = parser.parse_args()

    print("=" * 60)
    print("DFNetMF Training - Wall Camera Dataset")
    print("=" * 60)

    # Create config
    config = TrainingConfig(
        dataset_dir=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        mfop_method=args.method,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )

    print("\nConfiguration:")
    print(f"  Dataset: {config.dataset_dir}")
    print(f"  Checkpoint: {config.checkpoint_dir}")
    print(f"  Method: {config.mfop_method}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")

    # Load dataset
    print("\nLoading dataset...")
    dataset_dir = Path(config.dataset_dir)
    train_dataset = WallDataset(dataset_dir, config, split="train")
    val_dataset = WallDataset(dataset_dir, config, split="val")

    if len(train_dataset) == 0:
        print("❌ No training data found!")
        print(f"   Looking for manifest at: {dataset_dir / 'manifests' / 'train.tsv'}")
        sys.exit(1)

    # Create model
    print("\nInitializing model...")
    from df_mlx.deepfilternetmf import DFNetMF, ModelParamsMF
    from df_mlx.ops import erb_fb_and_inverse

    # Get ERB filterbanks
    erb_fb, erb_inv = erb_fb_and_inverse(
        sr=config.sr,
        fft_size=config.fft_size,
        nb_bands=config.nb_erb,
    )

    # Create model params
    model_params = ModelParamsMF(
        sr=config.sr,
        fft_size=config.fft_size,
        hop_size=config.hop_size,
        nb_erb=config.nb_erb,
        nb_df=config.nb_df,
        df_order=config.df_order,
        mfop_method=config.mfop_method,
    )

    model = DFNetMF(erb_fb, erb_inv, run_df=True, train_mask=True, params=model_params)

    # Count parameters
    from mlx.utils import tree_flatten

    flat_params = tree_flatten(model.parameters())
    num_params = sum(p.size for _, p in flat_params)  # type: ignore
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)

    # Training loop
    print("\nStarting training...")
    print("-" * 60)

    step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        epoch_metrics = {"total": 0.0, "l1": 0.0, "spec": 0.0, "log_spec": 0.0}
        epoch_steps = 0

        # Training epoch
        steps_per_epoch = max(len(train_dataset) // config.batch_size, 1)
        for batch_idx in range(steps_per_epoch):
            batch = train_dataset.get_batch(config.batch_size)
            if batch is None:
                continue

            noisy, clean = batch

            metrics = train_step(model, optimizer, noisy, clean, erb_fb, erb_inv, config)
            mx.eval(model.parameters())

            step += 1
            epoch_steps += 1

            for k, v in metrics.items():
                epoch_metrics[k] += v

            # Log
            if step % config.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs} | "
                    f"Step {step} | "
                    f"Loss: {metrics['total']:.4f} | "
                    f"L1: {metrics['l1']:.4f} | "
                    f"Spec: {metrics['spec']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Validation
            if step % config.val_interval == 0:
                val_metrics = validate(model, val_dataset, erb_fb, erb_inv, config)
                print(f"  [VAL] Loss: {val_metrics['total']:.4f} | L1: {val_metrics['l1']:.4f}")

                if val_metrics["total"] < best_val_loss:
                    best_val_loss = val_metrics["total"]
                    save_checkpoint(
                        model,
                        optimizer,
                        step,
                        epoch,
                        val_metrics,
                        Path(config.checkpoint_dir) / "best",
                    )

            # Save checkpoint
            if step % config.save_interval == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    epoch,
                    metrics,
                    Path(config.checkpoint_dir),
                )

        # Epoch summary
        epoch_time = time.time() - epoch_start
        if epoch_steps > 0:
            for k in epoch_metrics:
                epoch_metrics[k] /= epoch_steps
        print(f"\nEpoch {epoch + 1} complete | Avg Loss: {epoch_metrics['total']:.4f} | Time: {epoch_time:.1f}s\n")

    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
