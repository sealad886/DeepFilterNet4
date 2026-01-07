"""Training utilities for MLX DeepFilterNet4.

This module provides training functionality for the MLX implementation,
including:
- Training loop with gradient computation
- Learning rate scheduling
- Loss functions
- Checkpointing
- Weight conversion between PyTorch and MLX

Optimized for Apple Silicon with:
- Lazy evaluation for memory efficiency
- Unified memory utilization
- Metal-accelerated operations
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .config import TrainConfig
from .model import DfNet4, count_parameters

# ============================================================================
# Loss Functions
# ============================================================================


def spectral_loss(
    pred: Tuple[mx.array, mx.array],
    target: Tuple[mx.array, mx.array],
    alpha: float = 0.5,
) -> mx.array:
    """Combined magnitude and complex spectral loss.

    Args:
        pred: Predicted spectrum as (real, imag)
        target: Target spectrum as (real, imag)
        alpha: Weight for complex loss (1-alpha for magnitude)

    Returns:
        Scalar loss value
    """
    pred_real, pred_imag = pred
    target_real, target_imag = target

    # Magnitude loss
    pred_mag = mx.sqrt(pred_real**2 + pred_imag**2 + 1e-8)
    target_mag = mx.sqrt(target_real**2 + target_imag**2 + 1e-8)
    mag_loss = mx.mean(mx.abs(pred_mag - target_mag))

    # Complex loss
    complex_loss = mx.mean(mx.abs(pred_real - target_real) + mx.abs(pred_imag - target_imag))

    return (1 - alpha) * mag_loss + alpha * complex_loss


def multi_resolution_stft_loss(
    pred: mx.array,
    target: mx.array,
    fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
    hop_sizes: Optional[Tuple[int, ...]] = None,
) -> mx.array:
    """Multi-resolution STFT loss.

    Computes spectral loss at multiple resolutions for better
    time-frequency trade-off.

    Args:
        pred: Predicted waveform (batch, samples)
        target: Target waveform (batch, samples)
        fft_sizes: Tuple of FFT sizes
        hop_sizes: Tuple of hop sizes (defaults to fft_size // 4)

    Returns:
        Scalar loss value
    """
    from .ops import stft

    if hop_sizes is None:
        hop_sizes = tuple(fft // 4 for fft in fft_sizes)

    total_loss = mx.array(0.0)

    for fft_size, hop_size in zip(fft_sizes, hop_sizes):
        pred_spec = stft(pred, n_fft=fft_size, hop_length=hop_size)
        target_spec = stft(target, n_fft=fft_size, hop_length=hop_size)

        total_loss = total_loss + spectral_loss(pred_spec, target_spec)

    return total_loss / len(fft_sizes)


def snr_loss(
    pred: mx.array,
    target: mx.array,
    eps: float = 1e-8,
) -> mx.array:
    """Signal-to-noise ratio loss (negative SNR for minimization).

    Args:
        pred: Predicted signal
        target: Target signal
        eps: Small constant for numerical stability

    Returns:
        Negative SNR (to minimize)
    """
    noise = pred - target
    signal_power = mx.sum(target**2) + eps
    noise_power = mx.sum(noise**2) + eps

    snr = 10 * mx.log10(signal_power / noise_power)
    return -snr  # Negative for minimization


# ============================================================================
# Learning Rate Scheduling
# ============================================================================


class WarmupCosineSchedule:
    """Cosine learning rate schedule with warmup.

    Args:
        base_lr: Base learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step: int) -> float:
        """Get learning rate for given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * step / max(self.warmup_steps, 1)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


# ============================================================================
# Training Loop
# ============================================================================


class Trainer:
    """Training manager for DeepFilterNet4.

    Handles the training loop, checkpointing, logging, and evaluation.

    Args:
        model: DfNet4 model to train
        config: Training configuration
        optimizer: MLX optimizer (defaults to AdamW)
    """

    def __init__(
        self,
        model: DfNet4,
        config: TrainConfig,
        optimizer: Optional[optim.Optimizer] = None,
    ):
        self.model = model
        self.config = config

        # Initialize optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Learning rate schedule
        self.lr_schedule = WarmupCosineSchedule(
            base_lr=config.learning_rate,
            warmup_steps=config.warmup_steps,
            total_steps=config.max_steps,
        )

        # Training state
        self.step = 0
        self.best_loss = float("inf")

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def loss_fn(
        self,
        model: DfNet4,
        spec: Tuple[mx.array, mx.array],
        feat_erb: mx.array,
        feat_spec: mx.array,
        target: Tuple[mx.array, mx.array],
    ) -> mx.array:
        """Compute loss for a batch.

        Args:
            model: Model to evaluate
            spec: Input spectrum (real, imag)
            feat_erb: ERB features
            feat_spec: DF features
            target: Target spectrum (real, imag)

        Returns:
            Scalar loss value
        """
        pred = model(spec, feat_erb, feat_spec)
        return spectral_loss(pred, target)

    def train_step(
        self,
        spec: Tuple[mx.array, mx.array],
        feat_erb: mx.array,
        feat_spec: mx.array,
        target: Tuple[mx.array, mx.array],
    ) -> float:
        """Execute a single training step.

        Args:
            spec: Input spectrum
            feat_erb: ERB features
            feat_spec: DF features
            target: Target spectrum

        Returns:
            Loss value for this step
        """
        # Update learning rate
        lr = self.lr_schedule(self.step)
        self.optimizer.learning_rate = lr

        # Compute loss and gradients
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad_fn(self.model, spec, feat_erb, feat_spec, target)

        # Gradient clipping
        if self.config.grad_clip > 0:
            grads, _ = optim.clip_grad_norm(grads, self.config.grad_clip)

        # Update parameters
        self.optimizer.update(self.model, grads)

        # Evaluate to get actual loss value
        mx.eval(loss)

        self.step += 1
        return float(loss)

    def train(
        self,
        train_loader: Iterator,
        val_loader: Optional[Iterator] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run training loop.

        Args:
            train_loader: Training data iterator
            val_loader: Optional validation data iterator
            num_steps: Number of steps (defaults to config.max_steps)

        Returns:
            Training history dictionary
        """
        if num_steps is None:
            num_steps = self.config.max_steps

        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        print(f"Starting training for {num_steps} steps...")
        print(f"Model parameters: {count_parameters(self.model):,}")

        start_time = time.time()
        running_loss = 0.0

        for batch in train_loader:
            if self.step >= num_steps:
                break

            # Unpack batch
            spec, feat_erb, feat_spec, target = batch

            # Training step
            loss = self.train_step(spec, feat_erb, feat_spec, target)
            running_loss += loss

            # Logging
            if self.step % self.config.log_every == 0:
                avg_loss = running_loss / self.config.log_every
                lr = self.lr_schedule(self.step)
                elapsed = time.time() - start_time
                steps_per_sec = self.step / elapsed if elapsed > 0 else 0

                print(
                    f"Step {self.step:6d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Steps/s: {steps_per_sec:.2f}"
                )

                history["train_loss"].append(avg_loss)
                history["learning_rate"].append(lr)
                running_loss = 0.0

            # Validation
            if val_loader is not None and self.step % self.config.eval_every == 0:
                val_loss = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                print(f"  Validation loss: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best.safetensors")

            # Checkpointing
            if self.step % self.config.save_every == 0:
                self.save_checkpoint(f"step_{self.step}.safetensors")

        # Final save
        self.save_checkpoint("final.safetensors")

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best validation loss: {self.best_loss:.4f}")

        return history

    def evaluate(
        self,
        val_loader: Iterator,
        num_batches: int = 10,
    ) -> float:
        """Evaluate model on validation data.

        Args:
            val_loader: Validation data iterator
            num_batches: Number of batches to evaluate

        Returns:
            Average validation loss
        """
        total_loss = 0.0
        count = 0

        for batch in val_loader:
            if count >= num_batches:
                break

            spec, feat_erb, feat_spec, target = batch

            # Forward pass only (no gradient)
            pred = self.model(spec, feat_erb, feat_spec)
            loss = spectral_loss(pred, target)
            mx.eval(loss)

            total_loss += float(loss)
            count += 1

        return total_loss / max(count, 1)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        from mlx.utils import tree_flatten

        path = self.checkpoint_dir / filename

        # Save weights - flatten the nested dict to a flat dict for safetensors
        params = self.model.parameters()
        flat_params = tree_flatten(params)
        weights = {k: v for k, v in flat_params}
        mx.save_safetensors(str(path), weights)

        # Save training state
        state_path = path.with_suffix(".json")
        state = {
            "step": self.step,
            "best_loss": self.best_loss,
            "config": {
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "warmup_steps": self.config.warmup_steps,
            },
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        path = Path(path)

        # Load weights
        weights = mx.load(str(path))
        self.model.load_weights(weights)

        # Load training state if exists
        state_path = path.with_suffix(".json")
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.step = state.get("step", 0)
            self.best_loss = state.get("best_loss", float("inf"))

        print(f"Loaded checkpoint: {path} (step {self.step})")


# ============================================================================
# Weight Conversion
# ============================================================================


def convert_pytorch_weights(
    pytorch_state_dict: Dict[str, np.ndarray],
) -> Dict[str, mx.array]:
    """Convert PyTorch state dict to MLX format.

    Handles differences in:
    - Parameter naming conventions
    - Tensor layouts (PyTorch NCHW -> MLX NHWC for convolutions)
    - Data types

    Args:
        pytorch_state_dict: PyTorch state dictionary (numpy arrays)

    Returns:
        MLX-compatible weight dictionary
    """
    mlx_weights = {}

    for name, param in pytorch_state_dict.items():
        # Convert numpy array to MLX
        mlx_param = mx.array(param)

        # Handle convolution weight transposition
        # PyTorch: (out_ch, in_ch, H, W) -> MLX: (out_ch, H, W, in_ch)
        if "conv" in name.lower() and "weight" in name and len(param.shape) == 4:
            mlx_param = mx.transpose(mlx_param, (0, 2, 3, 1))

        # Map PyTorch names to MLX names
        mlx_name = name
        mlx_name = mlx_name.replace(".weight", ".weight")
        mlx_name = mlx_name.replace(".bias", ".bias")
        mlx_name = mlx_name.replace("running_mean", "running_mean")
        mlx_name = mlx_name.replace("running_var", "running_var")

        mlx_weights[mlx_name] = mlx_param

    return mlx_weights


def load_pytorch_checkpoint(
    model: DfNet4,
    checkpoint_path: str,
) -> DfNet4:
    """Load weights from a PyTorch checkpoint.

    Args:
        model: MLX model to load weights into
        checkpoint_path: Path to PyTorch checkpoint (.pth or .pt)

    Returns:
        Model with loaded weights
    """
    import torch

    # Load PyTorch checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Convert to numpy
    numpy_state = {k: v.numpy() for k, v in state_dict.items()}

    # Convert to MLX
    mlx_weights = convert_pytorch_weights(numpy_state)

    # Load into model
    model.load_weights(mlx_weights)

    print(f"Loaded PyTorch checkpoint: {checkpoint_path}")
    return model


# ============================================================================
# Convenience Functions
# ============================================================================


def train(
    model: DfNet4,
    train_loader: Iterator,
    val_loader: Optional[Iterator] = None,
    config: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """Convenience function for training.

    Args:
        model: Model to train
        train_loader: Training data iterator
        val_loader: Optional validation iterator
        config: Training configuration

    Returns:
        Training history
    """
    if config is None:
        config = TrainConfig()

    trainer = Trainer(model, config)
    return trainer.train(train_loader, val_loader)
