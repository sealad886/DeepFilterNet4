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

from .config import LossConfig, TrainConfig
from .model import DfNet4, count_parameters

# ============================================================================
# Running Statistics
# ============================================================================


class RunningStats(nn.Module):
    """Track running mean and variance statistics.

    Useful for input feature normalization, online statistics computation,
    and monitoring training dynamics.

    The statistics are updated using exponential moving average:
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_var

    Attributes:
        num_features: Number of features to track
        momentum: EMA momentum (higher = slower update)
        eps: Small constant for numerical stability

    Example:
        >>> stats = RunningStats(num_features=256)
        >>> for batch in data_loader:
        ...     normalized = stats(batch, training=True)
        >>> # During inference
        >>> normalized = stats(data, training=False)
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Running statistics - initialized to 0 mean, 1 variance
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.num_batches_tracked = mx.array(0)

    def update(self, x: mx.array) -> None:
        """Update running statistics with a batch of data.

        Args:
            x: Input data (..., num_features)
        """
        # Compute batch statistics (reduce over all dims except last)
        batch_mean = mx.mean(x, axis=tuple(range(x.ndim - 1)))
        batch_var = mx.var(x, axis=tuple(range(x.ndim - 1)))

        # Update running statistics with EMA
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        self.num_batches_tracked = self.num_batches_tracked + 1

    def normalize(
        self,
        x: mx.array,
        use_running: bool = True,
    ) -> mx.array:
        """Normalize input using statistics.

        Args:
            x: Input data (..., num_features)
            use_running: If True, use running stats; else compute from x

        Returns:
            Normalized data
        """
        if use_running:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = mx.mean(x, axis=tuple(range(x.ndim - 1)), keepdims=True)
            var = mx.var(x, axis=tuple(range(x.ndim - 1)), keepdims=True)
            mean = mx.squeeze(mean)
            var = mx.squeeze(var)

        return (x - mean) / mx.sqrt(var + self.eps)

    def __call__(
        self,
        x: mx.array,
        training: bool = True,
    ) -> mx.array:
        """Update stats (if training) and normalize.

        Args:
            x: Input data (..., num_features)
            training: If True, update running stats

        Returns:
            Normalized data
        """
        if training:
            self.update(x)
            # During training, use batch stats for normalization
            return self.normalize(x, use_running=False)
        else:
            # During inference, use running stats
            return self.normalize(x, use_running=True)


class FeatureNormalizer(nn.Module):
    """Per-sample feature normalizer with EMA smoothing.

    Implements a causal normalizer similar to the libdf unit_norm function.
    Uses exponential moving average to track feature magnitudes over time.

    This is useful for normalizing input features where the statistics
    need to adapt over time within each sample.

    Args:
        num_features: Number of features per time step
        alpha: EMA smoothing factor (0 = no smoothing, 1 = full smoothing)
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        num_features: int,
        alpha: float = 0.9,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.num_features = num_features
        self.alpha = alpha
        self.eps = eps

        # Initial state for normalization
        self._init_state = mx.ones((num_features,))

    def __call__(
        self,
        x: mx.array,
        state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Normalize features with EMA smoothing.

        Args:
            x: Input features (batch, time, features) or (time, features)
            state: Optional initial state (batch, features)

        Returns:
            Tuple of (normalized features, final state)
        """
        # Handle input dimensions
        input_2d = x.ndim == 2
        if input_2d:
            x = mx.expand_dims(x, axis=0)

        batch, time_steps, features = x.shape

        # Initialize state
        if state is None:
            state = mx.broadcast_to(self._init_state, (batch, features))

        # Compute magnitude per time step
        x_mag = mx.sqrt(mx.sum(x**2, axis=-1, keepdims=True) + self.eps)

        # Process time steps
        outputs = []
        for t in range(time_steps):
            # EMA update: state = alpha * state + (1 - alpha) * x_mag
            state = self.alpha * state + (1 - self.alpha) * x_mag[:, t, :]

            # Normalize
            norm_factor = mx.sqrt(state + self.eps)
            normalized = x[:, t, :] / norm_factor
            outputs.append(normalized)

        output = mx.stack(outputs, axis=1)

        if input_2d:
            output = mx.squeeze(output, axis=0)
            state = mx.squeeze(state, axis=0)

        return output, state


class ModelStatistics:
    """Track model training statistics.

    Monitors various metrics during training including:
    - Loss history
    - Gradient norms
    - Parameter statistics
    - Learning rate

    Example:
        >>> stats = ModelStatistics()
        >>> for batch in data_loader:
        ...     loss = train_step(batch)
        ...     stats.update(loss=float(loss), lr=scheduler.get_lr())
        >>> stats.summary()
    """

    def __init__(self):
        self.history: Dict[str, list] = {
            "loss": [],
            "grad_norm": [],
            "lr": [],
            "step_time": [],
        }
        self.step_count = 0

    def update(
        self,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
        step_time: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Update statistics with current step values.

        Args:
            loss: Training loss
            grad_norm: Gradient L2 norm
            lr: Current learning rate
            step_time: Time for this step (seconds)
            **kwargs: Additional metrics to track
        """
        if loss is not None:
            self.history["loss"].append(loss)
        if grad_norm is not None:
            self.history["grad_norm"].append(grad_norm)
        if lr is not None:
            self.history["lr"].append(lr)
        if step_time is not None:
            self.history["step_time"].append(step_time)

        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        self.step_count += 1

    def get_recent(self, key: str, n: int = 100) -> list:
        """Get recent values for a metric.

        Args:
            key: Metric name
            n: Number of recent values

        Returns:
            List of recent values
        """
        return self.history.get(key, [])[-n:]

    def get_mean(self, key: str, n: int = 100) -> float:
        """Get mean of recent values.

        Args:
            key: Metric name
            n: Number of recent values to average

        Returns:
            Mean value
        """
        values = self.get_recent(key, n)
        return sum(values) / max(len(values), 1)

    def summary(self) -> Dict[str, float]:
        """Get summary statistics.

        Returns:
            Dictionary of summary metrics
        """
        return {
            "steps": self.step_count,
            "loss_mean": self.get_mean("loss"),
            "loss_std": float(np.std(self.get_recent("loss"))) if self.history["loss"] else 0.0,
            "grad_norm_mean": self.get_mean("grad_norm"),
            "step_time_mean": self.get_mean("step_time"),
        }

    def save(self, path: str) -> None:
        """Save statistics to JSON file.

        Args:
            path: Output file path
        """
        with open(path, "w") as f:
            json.dump(
                {
                    "history": self.history,
                    "step_count": self.step_count,
                },
                f,
                indent=2,
            )

    def load(self, path: str) -> None:
        """Load statistics from JSON file.

        Args:
            path: Input file path
        """
        with open(path) as f:
            data = json.load(f)
        self.history = data.get("history", {})
        self.step_count = data.get("step_count", 0)


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


class MultiResolutionSTFTLoss:
    """Multi-resolution STFT loss for speech enhancement training.

    Computes spectral loss at multiple resolutions to capture both
    fine-grained details and broader spectral structure. Matches the
    PyTorch MultiResSpecLoss implementation.

    The loss combines:
    - Magnitude loss: MSE on (optionally compressed) magnitudes
    - Complex loss (optional): MSE on real/imag components

    Args:
        fft_sizes: List of FFT sizes to use
        hop_sizes: List of hop sizes (defaults to fft_size // 4)
        gamma: Magnitude compression exponent (1.0 = no compression)
        factor: Weight for magnitude loss
        f_complex: Weight for complex loss (None to disable)
        eps: Small constant for numerical stability

    Example:
        >>> loss_fn = MultiResolutionSTFTLoss(
        ...     fft_sizes=[512, 1024, 2048],
        ...     gamma=0.5,  # Compressed magnitude
        ...     factor=1.0,
        ...     f_complex=0.5,  # Enable complex loss
        ... )
        >>> loss = loss_fn(pred_waveform, target_waveform)
    """

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Optional[Tuple[int, ...]] = None,
        gamma: float = 1.0,
        factor: float = 1.0,
        f_complex: Optional[float] = None,
        eps: float = 1e-12,
    ):
        self.fft_sizes = fft_sizes
        if hop_sizes is None:
            self.hop_sizes = tuple(fft // 4 for fft in fft_sizes)
        else:
            self.hop_sizes = hop_sizes

        assert len(self.fft_sizes) == len(
            self.hop_sizes
        ), f"fft_sizes ({len(fft_sizes)}) and hop_sizes ({len(hop_sizes)}) must have same length"

        self.gamma = gamma
        self.factor = factor
        self.f_complex = f_complex
        self.eps = eps

    def __call__(self, pred: mx.array, target: mx.array) -> mx.array:
        """Compute multi-resolution STFT loss.

        Args:
            pred: Predicted waveform (batch, samples) or (samples,)
            target: Target waveform (batch, samples) or (samples,)

        Returns:
            Scalar loss value
        """
        from .ops import stft

        # Handle 1D input
        if pred.ndim == 1:
            pred = mx.expand_dims(pred, axis=0)
        if target.ndim == 1:
            target = mx.expand_dims(target, axis=0)

        total_loss = mx.array(0.0)

        for fft_size, hop_size in zip(self.fft_sizes, self.hop_sizes):
            # Compute STFTs
            pred_real, pred_imag = stft(pred, n_fft=fft_size, hop_length=hop_size)
            target_real, target_imag = stft(target, n_fft=fft_size, hop_length=hop_size)

            # Compute magnitudes
            pred_mag = mx.sqrt(pred_real**2 + pred_imag**2 + self.eps)
            target_mag = mx.sqrt(target_real**2 + target_imag**2 + self.eps)

            # Apply gamma compression if needed
            if self.gamma != 1.0:
                pred_mag_comp = mx.power(mx.maximum(pred_mag, self.eps), self.gamma)
                target_mag_comp = mx.power(mx.maximum(target_mag, self.eps), self.gamma)
            else:
                pred_mag_comp = pred_mag
                target_mag_comp = target_mag

            # Magnitude loss (MSE)
            mag_loss = mx.mean((pred_mag_comp - target_mag_comp) ** 2) * self.factor
            total_loss = total_loss + mag_loss

            # Complex loss (optional)
            if self.f_complex is not None and self.f_complex > 0:
                if self.gamma != 1.0:
                    # Reconstruct complex with compressed magnitude
                    # pred_complex = pred_mag_comp * exp(i * angle(pred))
                    pred_angle = mx.arctan2(pred_imag, pred_real + self.eps)
                    pred_real_comp = pred_mag_comp * mx.cos(pred_angle)
                    pred_imag_comp = pred_mag_comp * mx.sin(pred_angle)

                    target_angle = mx.arctan2(target_imag, target_real + self.eps)
                    target_real_comp = target_mag_comp * mx.cos(target_angle)
                    target_imag_comp = target_mag_comp * mx.sin(target_angle)
                else:
                    pred_real_comp = pred_real
                    pred_imag_comp = pred_imag
                    target_real_comp = target_real
                    target_imag_comp = target_imag

                # MSE on real/imag components
                complex_loss = (
                    mx.mean((pred_real_comp - target_real_comp) ** 2)
                    + mx.mean((pred_imag_comp - target_imag_comp) ** 2)
                ) * self.f_complex
                total_loss = total_loss + complex_loss

        # Average over resolutions
        return total_loss / len(self.fft_sizes)

    def compute_per_resolution(self, pred: mx.array, target: mx.array) -> Dict[str, mx.array]:
        """Compute loss breakdown per resolution (for logging).

        Args:
            pred: Predicted waveform
            target: Target waveform

        Returns:
            Dictionary with loss values per FFT size
        """
        from .ops import stft

        if pred.ndim == 1:
            pred = mx.expand_dims(pred, axis=0)
        if target.ndim == 1:
            target = mx.expand_dims(target, axis=0)

        losses = {}

        for fft_size, hop_size in zip(self.fft_sizes, self.hop_sizes):
            pred_real, pred_imag = stft(pred, n_fft=fft_size, hop_length=hop_size)
            target_real, target_imag = stft(target, n_fft=fft_size, hop_length=hop_size)

            pred_mag = mx.sqrt(pred_real**2 + pred_imag**2 + self.eps)
            target_mag = mx.sqrt(target_real**2 + target_imag**2 + self.eps)

            if self.gamma != 1.0:
                pred_mag = mx.power(mx.maximum(pred_mag, self.eps), self.gamma)
                target_mag = mx.power(mx.maximum(target_mag, self.eps), self.gamma)

            losses[f"mrsl_{fft_size}"] = mx.mean((pred_mag - target_mag) ** 2) * self.factor

        losses["mrsl_total"] = sum(losses.values()) / len(self.fft_sizes)
        return losses

    @classmethod
    def from_config(cls, config: LossConfig) -> "MultiResolutionSTFTLoss":
        """Create loss from configuration.

        Args:
            config: LossConfig instance

        Returns:
            Configured MultiResolutionSTFTLoss instance
        """
        hop_sizes = config.mrsl_hop_sizes
        if hop_sizes is not None:
            hop_sizes = tuple(hop_sizes)

        return cls(
            fft_sizes=tuple(config.mrsl_fft_sizes),
            hop_sizes=hop_sizes,
            gamma=config.mrsl_gamma,
            factor=config.mrsl_factor,
            f_complex=config.mrsl_f_complex,
        )


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


def lsnr_loss(
    pred_lsnr: mx.array,
    target_lsnr: mx.array,
    lsnr_min: float = -15.0,
    lsnr_max: float = 40.0,
) -> mx.array:
    """LSNR (Local SNR) prediction loss.

    L1 loss between predicted and target LSNR values, clipped to valid range.

    Args:
        pred_lsnr: Predicted LSNR (batch, time, 1)
        target_lsnr: Target LSNR (batch, time, 1)
        lsnr_min: Minimum valid LSNR value
        lsnr_max: Maximum valid LSNR value

    Returns:
        Scalar loss value
    """
    # Clip to valid range
    pred_clipped = mx.clip(pred_lsnr, lsnr_min, lsnr_max)
    target_clipped = mx.clip(target_lsnr, lsnr_min, lsnr_max)

    return mx.mean(mx.abs(pred_clipped - target_clipped))


# ============================================================================
# Combined Loss Functions (Module-Level)
# ============================================================================


def combined_loss(
    pred: Tuple[mx.array, mx.array],
    target: Tuple[mx.array, mx.array],
    alpha_spec: float = 1.0,
    alpha_time: float = 0.1,
) -> mx.array:
    """Combined spectral and time-domain loss for training.

    This is a convenience function combining spectral loss and time-domain loss
    for use in custom training loops outside the Trainer class.

    Args:
        pred: Predicted spectrum as (real, imag)
        target: Target spectrum as (real, imag)
        alpha_spec: Weight for spectral loss (magnitude + complex)
        alpha_time: Weight for time-domain loss (waveform)

    Returns:
        Combined scalar loss value

    Example:
        >>> loss = combined_loss(
        ...     pred=(out_real, out_imag),
        ...     target=(target_real, target_imag),
        ...     alpha_spec=1.0,
        ...     alpha_time=0.1,
        ... )
    """
    # Spectral loss (magnitude + complex)
    spec_loss = spectral_loss(pred, target, alpha=0.5)

    # Time-domain loss via ISTFT approximation
    pred_real, pred_imag = pred
    target_real, target_imag = target

    # Approximate time-domain error using magnitude difference
    pred_mag = mx.sqrt(pred_real**2 + pred_imag**2 + 1e-8)
    target_mag = mx.sqrt(target_real**2 + target_imag**2 + 1e-8)
    time_loss = mx.mean((pred_mag - target_mag) ** 2)

    return alpha_spec * spec_loss + alpha_time * time_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    path: str | Path = "checkpoint.safetensors",
    epoch: int = 0,
    step: int = 0,
    loss: float = 0.0,
    **extra_state: Any,
) -> Path:
    """Save model checkpoint to file.

    This is a standalone function for saving checkpoints outside the Trainer class.

    Args:
        model: Model to save
        optimizer: Optional optimizer (state saved separately if provided)
        path: Path to save checkpoint
        epoch: Current epoch number
        step: Current training step
        loss: Current loss value
        **extra_state: Additional state to save in metadata

    Returns:
        Path to saved checkpoint

    Example:
        >>> save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     path="checkpoints/epoch_010.safetensors",
        ...     epoch=10,
        ...     step=1000,
        ...     loss=0.123,
        ... )
    """
    from mlx.utils import tree_flatten

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save model weights
    params = model.parameters()
    flat_params = tree_flatten(params)
    weights = {k: v for k, v in flat_params}
    mx.save_safetensors(str(path), weights)

    # Save training state as JSON
    state_path = path.with_suffix(".json")
    state = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        **extra_state,
    }
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    return path


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model checkpoint from file.

    This is a standalone function for loading checkpoints outside the Trainer class.

    Args:
        model: Model to load weights into
        path: Path to checkpoint file
        strict: Whether to require all keys to match

    Returns:
        Dictionary with training state (epoch, step, loss, etc.)

    Example:
        >>> state = load_checkpoint(model, "checkpoints/epoch_010.safetensors")
        >>> start_epoch = state.get("epoch", 0)
    """
    path = Path(path)

    # Load model weights - mx.load returns Dict[str, mx.array] for safetensors
    weights: Dict[str, mx.array] = mx.load(str(path))  # type: ignore[assignment]
    model.load_weights(list(weights.items()))

    # Load training state
    state_path = path.with_suffix(".json")
    state: Dict[str, Any] = {}
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)

    return state


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

    def load_checkpoint(self, path: str | Path):
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
