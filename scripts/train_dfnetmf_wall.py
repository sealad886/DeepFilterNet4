#!/usr/bin/env python3
"""Training script for DFNetMF on Wall camera dataset.

This script trains DFNetMF (Multi-Frame filtering) model to remove
stationary noise (clicking, interference) from security camera audio.

Usage:
    python train_dfnetmf_wall.py --dataset /Users/andrew/DataDump/datasets/wall_processed
"""

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_unflatten
from tqdm.auto import tqdm

# Add DeepFilterNet to path
SCRIPT_DIR = Path(__file__).parent
DFNET_DIR = SCRIPT_DIR.parent / "DeepFilterNet"
sys.path.insert(0, str(DFNET_DIR))

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
    save_total_limit: int = 5
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

    with tqdm(total=num_batches, desc="Validation", leave=False, unit="batch", **_TQDM_KWARGS) as pbar:
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

            pbar.set_postfix({"loss": f"{metrics['total']:.4f}"})
            pbar.update(1)

    if count > 0:
        for k in total_metrics:
            total_metrics[k] /= count

    return total_metrics


def _validate_checkpoint_pair(checkpoint_dir: Path, step: int) -> bool:
    """Validate that both .safetensors and state.json files exist for a checkpoint step.

    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Checkpoint step number

    Returns:
        True if both files exist and are valid, False otherwise
    """
    weights_file = checkpoint_dir / f"step_{step:06d}.safetensors"
    state_file = checkpoint_dir / f"step_{step:06d}_state.json"

    # Check both files exist
    if not weights_file.exists():
        print(f"⚠️  Checkpoint step {step}: missing .safetensors file")
        return False
    if not state_file.exists():
        print(f"⚠️  Checkpoint step {step}: missing state.json file")
        return False

    # Check files are not empty (indicates incomplete write)
    if weights_file.stat().st_size == 0:
        print(f"⚠️  Checkpoint step {step}: .safetensors file is empty")
        return False
    if state_file.stat().st_size == 0:
        print(f"⚠️  Checkpoint step {step}: state.json file is empty")
        return False

    return True


def load_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
) -> tuple[int, int]:
    """Load latest checkpoint and restore model weights and optimizer state.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model: Model to restore weights into
        optimizer: Optional optimizer to restore state into

    Returns:
        (step, epoch) tuple, or (0, 0) if no checkpoint found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return 0, 0

    # Find latest checkpoint by state file
    state_files = sorted(checkpoint_dir.glob("step_*_state.json"))
    if not state_files:
        return 0, 0

    # Find the latest valid checkpoint (with consistency check)
    for state_file in reversed(state_files):
        latest_step = int(state_file.stem.split("_")[1])

        # Validate checkpoint pair
        if not _validate_checkpoint_pair(checkpoint_dir, latest_step):
            print("  Trying previous checkpoint...")
            continue

        break
    else:
        print("❌ No valid checkpoint pair found")
        return 0, 0

    # Load state
    with open(state_file) as f:
        state = json.load(f)

    step = state.get("step", 0)
    epoch = state.get("epoch", 0)

    # Load model weights
    weights_file = checkpoint_dir / f"step_{latest_step:06d}.safetensors"
    try:
        flat_weights = mx.load(str(weights_file))

        # Align checkpoint weights with the model's parameter tree to avoid
        # shape/name mismatches. Missing parameters (should be none) fall
        # back to current values so update() never fails.
        from mlx.utils import tree_flatten

        flat_model = tree_flatten(model.parameters())
        pairs = []
        missing = []
        for name, param in flat_model:
            if isinstance(flat_weights, dict) and name in flat_weights:
                pairs.append((name, flat_weights[name]))
            else:
                pairs.append((name, param))
                missing.append(name)

        nested_weights = tree_unflatten(pairs)
        model.update(nested_weights)

        if missing:
            print(f"⚠️  {len(missing)} parameters were missing in checkpoint (e.g., {missing[:5]})")

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
                    # Reconstruct optimizer state from flat dict using tree_unflatten
                    state_pairs = list(restored.items())
                    nested_state = tree_unflatten(state_pairs)
                    optimizer.state = nested_state
                    print(f"✅ Restored optimizer state (step {step})")
            except Exception as e:
                print(f"⚠️  Failed to restore optimizer state: {e}")

        print(f"✅ Loaded checkpoint from step {step}, epoch {epoch}")
        return step, epoch
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        return 0, 0


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path,
    completed: bool = False,
):
    """Save training checkpoint including optimizer state.

    Args:
        model: Model to save weights from
        optimizer: Optimizer to save state from
        step: Current training step
        epoch: Current epoch
        metrics: Dictionary of training metrics
        checkpoint_dir: Directory to save checkpoint to
        completed: Whether this checkpoint marks a completed training run
    """
    from mlx.utils import tree_flatten

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    weights_path = checkpoint_dir / f"step_{step:06d}.safetensors"
    flat_params = tree_flatten(model.parameters())
    mx.save_safetensors(str(weights_path), dict(flat_params))

    # Save optimizer state
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
    state = {
        "step": step,
        "epoch": epoch,
        "metrics": metrics,
        "optimizer_state": optimizer_state_dict,
        "completed": completed,
    }
    state_path = checkpoint_dir / f"step_{step:06d}_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"Saved checkpoint: {weights_path}")
    if optimizer_state_dict:
        print(f"✅ Saved optimizer state to checkpoint (step {step})")


# ============================================================================
# Signal Handling for Graceful Interrupt
# ============================================================================

# Global state for signal handler
_interrupt_state = {
    "checkpoint_dir": None,
    "step": 0,
    "epoch": 0,
    "model": None,
    "optimizer": None,
    "metrics": {},
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
            save_checkpoint(
                _interrupt_state["model"],
                _interrupt_state["optimizer"],
                _interrupt_state["step"],
                _interrupt_state["epoch"],
                _interrupt_state["metrics"],
                _interrupt_state["checkpoint_dir"],
            )
            print(f"✅ Final checkpoint saved to {_interrupt_state['checkpoint_dir']}")
        except Exception as e:
            print(f"❌ Failed to save final checkpoint: {e}")

    print("Exiting...")
    raise KeyboardInterrupt()


def _register_sigint_handler(model, optimizer, checkpoint_dir):
    """Register SIGINT handler for graceful training shutdown.

    Args:
        model: Model to save on interrupt
        optimizer: Optimizer to save state on interrupt
        checkpoint_dir: Directory to save checkpoint to
    """
    _interrupt_state["model"] = model
    _interrupt_state["optimizer"] = optimizer
    _interrupt_state["checkpoint_dir"] = checkpoint_dir
    signal.signal(signal.SIGINT, _handle_sigint)


def _update_interrupt_state(step, epoch, metrics):
    """Update global state for interrupt handler.

    Args:
        step: Current training step
        epoch: Current epoch
        metrics: Current metrics dictionary
    """
    _interrupt_state["step"] = step
    _interrupt_state["epoch"] = epoch
    _interrupt_state["metrics"] = metrics


def main():
    parser = argparse.ArgumentParser(description="Train DFNetMF on Wall camera dataset")
    parser.add_argument("--dataset", type=str, default="/Users/andrew/DataDump/datasets/wall_processed")
    parser.add_argument("--checkpoint-dir", type=str, default="/Users/andrew/DataDump/checkpoints/dfnetmf_wall")
    parser.add_argument("--method", type=str, default="WF", choices=["WF", "MVDR"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=5,
        help="Keep only the most recent N checkpoints in the main checkpoint directory (best checkpoint is kept separately)",
    )

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
        save_total_limit=args.save_total_limit,
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

    # Resume from checkpoint if requested
    step = 0
    best_val_loss = float("inf")
    resume_epoch = 0

    if args.resume:
        checkpoint_path = Path(config.checkpoint_dir)
        if checkpoint_path.exists():
            loaded_step, loaded_epoch = load_checkpoint(checkpoint_path, model, optimizer=optimizer)
            if loaded_step > 0:
                step = loaded_step
                resume_epoch = loaded_epoch
                print(f"📂 Resuming from checkpoint: step={step}, epoch={loaded_epoch}")

                # Check if the latest checkpoint marks a completed run.
                state_path = checkpoint_path / f"step_{step:06d}_state.json"
                completed = False
                if state_path.exists():
                    try:
                        with open(state_path) as f:
                            state = json.load(f)
                        completed = bool(state.get("completed", False))
                    except Exception as e:
                        print(f"⚠️  Failed to read checkpoint state: {e}")

                if completed:
                    if loaded_epoch + 1 >= config.num_epochs:
                        print(
                            f"✅ Training already complete at epoch {loaded_epoch + 1}/{config.num_epochs}. "
                            "Nothing to do."
                        )
                        return
                    resume_epoch = loaded_epoch + 1
                    print(f"-> Checkpoint marked complete; continuing at epoch {resume_epoch + 1}/{config.num_epochs}")
        else:
            print(f"⚠️  No checkpoint directory found at {checkpoint_path}, starting fresh")

    # Training loop
    print("\nStarting training...")
    print("-" * 60)

    # Track the latest metrics so we can always save a final checkpoint on clean exit.
    last_metrics: Dict[str, float] = {}
    last_epoch: int = resume_epoch

    # Register signal handler for graceful CTRL+C handling
    _register_sigint_handler(model, optimizer, Path(config.checkpoint_dir))

    with tqdm(
        total=config.num_epochs - resume_epoch,
        desc="Training",
        unit="epoch",
        position=0,
        initial=resume_epoch,
        **_TQDM_KWARGS,
    ) as epoch_pbar:
        for epoch in range(resume_epoch, config.num_epochs):
            epoch_start = time.time()
            epoch_metrics = {"total": 0.0, "l1": 0.0, "spec": 0.0, "log_spec": 0.0}
            epoch_steps = 0

            # Training epoch
            steps_per_epoch = max(len(train_dataset) // config.batch_size, 1)

            with tqdm(
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{config.num_epochs}",
                leave=False,
                unit="batch",
                position=1,
                **_TQDM_KWARGS,
            ) as step_pbar:
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

                    # Update latest metrics for final checkpoint
                    last_metrics = metrics
                    last_epoch = epoch
                    # Update interrupt state with current training progress
                    _update_interrupt_state(step, epoch, metrics)

                    # Update step progress bar with current metrics
                    step_pbar.set_postfix(
                        {
                            "loss": f"{metrics['total']:.4f}",
                            "l1": f"{metrics['l1']:.4f}",
                            "spec": f"{metrics['spec']:.4f}",
                            "best_val": f"{best_val_loss:.4f}" if best_val_loss != float("inf") else "N/A",
                        }
                    )
                    step_pbar.update(1)

                    # Validation
                    if step % config.val_interval == 0:
                        val_metrics = validate(model, val_dataset, erb_fb, erb_inv, config)
                        step_pbar.write(
                            f"  [VAL @ step {step}] Loss: {val_metrics['total']:.4f} | "
                            f"L1: {val_metrics['l1']:.4f} | Spec: {val_metrics['spec']:.4f}"
                        )

                        if val_metrics["total"] < best_val_loss:
                            best_val_loss = val_metrics["total"]
                            best_dir = Path(config.checkpoint_dir) / "best"
                            best_dir.mkdir(parents=True, exist_ok=True)
                            for p in best_dir.glob("step_*.safetensors"):
                                p.unlink(missing_ok=True)
                            for p in best_dir.glob("step_*_state.json"):
                                p.unlink(missing_ok=True)
                            save_checkpoint(
                                model,
                                optimizer,
                                step,
                                epoch,
                                val_metrics,
                                best_dir,
                            )
                            step_pbar.write(f"  ✅ New best validation loss: {best_val_loss:.4f}")

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
                        step_pbar.write(f"  💾 Checkpoint saved at step {step}")
                        # Keep only the most recent N checkpoints (weights + state) in the main dir.
                        # Best checkpoint is stored separately under checkpoint_dir/best.
                        if getattr(config, "save_total_limit", 0) and config.save_total_limit > 0:
                            ckpt_dir = Path(config.checkpoint_dir)
                            ckpt_files = list(ckpt_dir.glob("step_*.safetensors"))
                            # Sort by step number encoded in the filename: step_000500.safetensors
                            ckpt_files.sort(key=lambda p: int(p.stem.split("_")[1]))
                            to_remove = ckpt_files[: -config.save_total_limit]
                            for weights_path in to_remove:
                                weights_path.unlink(missing_ok=True)
                                state_path = weights_path.with_name(f"{weights_path.stem}_state.json")
                                state_path.unlink(missing_ok=True)

            # Epoch summary
            epoch_time = time.time() - epoch_start
            if epoch_steps > 0:
                for k in epoch_metrics:
                    epoch_metrics[k] /= epoch_steps

            # Update epoch progress bar
            epoch_pbar.set_postfix(
                {
                    "avg_loss": f"{epoch_metrics['total']:.4f}",
                    "best_val": f"{best_val_loss:.4f}" if best_val_loss != float("inf") else "N/A",
                    "time": f"{epoch_time:.1f}s",
                }
            )
            epoch_pbar.update(1)

    # Final validation to compare against best checkpoint.
    if len(val_dataset) > 0:
        try:
            final_val_metrics = validate(model, val_dataset, erb_fb, erb_inv, config)
            if final_val_metrics["total"] < best_val_loss:
                best_val_loss = final_val_metrics["total"]
                best_dir = Path(config.checkpoint_dir) / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                for p in best_dir.glob("step_*.safetensors"):
                    p.unlink(missing_ok=True)
                for p in best_dir.glob("step_*_state.json"):
                    p.unlink(missing_ok=True)
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    last_epoch,
                    final_val_metrics,
                    best_dir,
                )
                print(f"✅ Final weights set new best validation loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"⚠️  Final validation failed: {e}")

    # Always save a final checkpoint on clean completion.
    # This guarantees the last in-memory weights are persisted even if `step` isn't
    # aligned to `save_interval` and no new best checkpoint happened at the end.
    try:
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        weights_path = ckpt_dir / f"step_{step:06d}.safetensors"
        state_path = ckpt_dir / f"step_{step:06d}_state.json"

        # Avoid redundant writes if the final step already produced a checkpoint.
        if not (weights_path.exists() and state_path.exists()):
            save_checkpoint(
                model,
                optimizer,
                step,
                last_epoch,
                last_metrics or {"total": float("nan")},
                ckpt_dir,
                completed=True,
            )
            print(f"💾 Final checkpoint saved at step {step}")
        else:
            try:
                with open(state_path) as f:
                    state = json.load(f)
                if not state.get("completed", False):
                    state["completed"] = True
                    with open(state_path, "w") as f:
                        json.dump(state, f, indent=2)
                    print("✅ Marked final checkpoint as completed")
            except Exception as e:
                print(f"⚠️  Failed to mark final checkpoint as completed: {e}")

        # Enforce checkpoint retention after the final save.
        if getattr(config, "save_total_limit", 0) and config.save_total_limit > 0:
            ckpt_files = list(ckpt_dir.glob("step_*.safetensors"))
            ckpt_files.sort(key=lambda p: int(p.stem.split("_")[1]))
            to_remove = ckpt_files[: -config.save_total_limit]
            for weights_path in to_remove:
                weights_path.unlink(missing_ok=True)
                state_path = weights_path.with_name(f"{weights_path.stem}_state.json")
                state_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"⚠️  Failed to save/prune final checkpoint: {e}")

    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
