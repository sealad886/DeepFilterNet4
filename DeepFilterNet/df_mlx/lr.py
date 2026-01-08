"""Learning rate schedulers for MLX DeepFilterNet4 training.

Ported from the PyTorch implementation (df/lr.py), providing:
- Cosine scheduler with warmup (ConvNeXt-style)
- Cyclic decay with configurable decay multiplier
- Integration with MLX optimizers

The main scheduler is designed for long training runs with:
1. Linear warmup phase
2. Cosine decay to minimum LR
3. Optional cyclic restarts with decay
"""

from typing import Iterator, Optional, Tuple

import numpy as np


def cosine_scheduler(
    base_lr: float,
    min_lr: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
    warmup_steps: int = 0,
    cycle_decay: float = 1.0,
    cycle_mul: float = 1.0,
) -> Iterator[Tuple[int, int, float]]:
    """Cosine learning rate scheduler with warmup and cyclic decay.

    This is a ConvNeXt-style scheduler that provides:
    1. Linear warmup from 0 to base_lr
    2. Cosine decay from base_lr to min_lr
    3. Optional cyclic restarts with decay

    The scheduler yields (epoch, step, lr) tuples for each training step,
    allowing fine-grained control over learning rate.

    Args:
        base_lr: Base (maximum) learning rate after warmup
        min_lr: Minimum learning rate at end of cycle
        epochs: Total number of training epochs
        steps_per_epoch: Number of steps (batches) per epoch
        warmup_epochs: Number of warmup epochs (linear warmup)
        warmup_steps: Additional warmup steps beyond warmup_epochs
        cycle_decay: Multiplier for base_lr at each cycle restart
            (1.0 = no decay, 0.9 = 10% reduction each cycle)
        cycle_mul: Multiplier for cycle length at each restart
            (1.0 = same length, 2.0 = double length each cycle)

    Yields:
        Tuple of (epoch, step, learning_rate) for each training step

    Example:
        >>> scheduler = cosine_scheduler(
        ...     base_lr=1e-3,
        ...     min_lr=1e-6,
        ...     epochs=100,
        ...     steps_per_epoch=1000,
        ...     warmup_epochs=5,
        ...     cycle_decay=0.9,  # 10% decay per cycle
        ...     cycle_mul=2.0,    # Double cycle length
        ... )
        >>> for epoch, step, lr in scheduler:
        ...     optimizer.learning_rate = lr
        ...     train_step()
    """
    warmup_total = warmup_epochs * steps_per_epoch + warmup_steps

    current_lr = base_lr
    cycle_steps = (epochs - warmup_epochs) * steps_per_epoch - warmup_steps

    global_step = 0
    cycle_start = warmup_total
    cycle_length = cycle_steps
    cycle_num = 0

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            if global_step < warmup_total:
                # Linear warmup
                lr = base_lr * global_step / max(warmup_total, 1)
            else:
                # Position within current cycle
                cycle_pos = global_step - cycle_start

                # Check for cycle completion
                if cycle_pos >= cycle_length:
                    # Start new cycle
                    cycle_num += 1
                    cycle_start = global_step
                    cycle_pos = 0
                    current_lr = base_lr * (cycle_decay**cycle_num)
                    cycle_length = int(cycle_length * cycle_mul)

                # Cosine decay within cycle
                progress = cycle_pos / max(cycle_length, 1)
                progress = min(progress, 1.0)
                cosine = 0.5 * (1 + np.cos(np.pi * progress))
                lr = min_lr + (current_lr - min_lr) * cosine

            yield epoch, step, lr
            global_step += 1


class CosineScheduler:
    """Cosine learning rate scheduler with warmup.

    Object-oriented wrapper around cosine_scheduler for easier integration
    with training loops.

    Args:
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        epochs: Total training epochs
        steps_per_epoch: Steps per epoch
        warmup_epochs: Warmup epochs
        warmup_steps: Additional warmup steps
        cycle_decay: Decay factor per cycle
        cycle_mul: Cycle length multiplier

    Example:
        >>> scheduler = CosineScheduler(
        ...     base_lr=1e-3,
        ...     min_lr=1e-6,
        ...     epochs=100,
        ...     steps_per_epoch=1000,
        ...     warmup_epochs=5,
        ... )
        >>> for step in range(total_steps):
        ...     lr = scheduler.step()
        ...     optimizer.learning_rate = lr
    """

    def __init__(
        self,
        base_lr: float,
        min_lr: float,
        epochs: int,
        steps_per_epoch: int,
        warmup_epochs: int = 0,
        warmup_steps: int = 0,
        cycle_decay: float = 1.0,
        cycle_mul: float = 1.0,
    ):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_steps
        self.cycle_decay = cycle_decay
        self.cycle_mul = cycle_mul

        self._generator = cosine_scheduler(
            base_lr=base_lr,
            min_lr=min_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=warmup_epochs,
            warmup_steps=warmup_steps,
            cycle_decay=cycle_decay,
            cycle_mul=cycle_mul,
        )

        self.current_epoch = 0
        self.current_step = 0
        self.current_lr = 0.0

    def step(self) -> float:
        """Advance scheduler by one step and return new learning rate.

        Returns:
            Learning rate for the current step

        Raises:
            StopIteration: If all steps have been exhausted
        """
        epoch, step, lr = next(self._generator)
        self.current_epoch = epoch
        self.current_step = step
        self.current_lr = lr
        return lr

    def get_lr(self) -> float:
        """Get current learning rate without advancing.

        Returns:
            Current learning rate
        """
        return self.current_lr

    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing.

        Returns:
            Dictionary with scheduler state
        """
        return {
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "current_lr": self.current_lr,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "warmup_epochs": self.warmup_epochs,
            "warmup_steps": self.warmup_steps,
            "cycle_decay": self.cycle_decay,
            "cycle_mul": self.cycle_mul,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load scheduler state from checkpoint.

        This recreates the generator and advances it to the saved position.

        Args:
            state: State dictionary from state_dict()
        """
        # Restore parameters
        self.base_lr = state["base_lr"]
        self.min_lr = state["min_lr"]
        self.epochs = state["epochs"]
        self.steps_per_epoch = state["steps_per_epoch"]
        self.warmup_epochs = state["warmup_epochs"]
        self.warmup_steps = state["warmup_steps"]
        self.cycle_decay = state["cycle_decay"]
        self.cycle_mul = state["cycle_mul"]

        # Recreate generator
        self._generator = cosine_scheduler(
            base_lr=self.base_lr,
            min_lr=self.min_lr,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_steps=self.warmup_steps,
            cycle_decay=self.cycle_decay,
            cycle_mul=self.cycle_mul,
        )

        # Advance to saved position
        target_global_step = state["current_epoch"] * self.steps_per_epoch + state["current_step"]
        for _ in range(target_global_step):
            try:
                next(self._generator)
            except StopIteration:
                break

        self.current_epoch = state["current_epoch"]
        self.current_step = state["current_step"]
        self.current_lr = state["current_lr"]


class WarmupScheduler:
    """Simple warmup scheduler that wraps another scheduler.

    Provides linear warmup followed by any base schedule.

    Args:
        base_lr: Target learning rate after warmup
        warmup_steps: Number of warmup steps
        base_scheduler: Optional underlying scheduler to use after warmup

    Example:
        >>> warmup = WarmupScheduler(base_lr=1e-3, warmup_steps=1000)
        >>> for step in range(total_steps):
        ...     lr = warmup.step()
        ...     optimizer.learning_rate = lr
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        base_scheduler: Optional["CosineScheduler"] = None,
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.current_step = 0
        self.current_lr = 0.0

    def step(self) -> float:
        """Advance scheduler by one step.

        Returns:
            Learning rate for current step
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup
            self.current_lr = self.base_lr * self.current_step / max(self.warmup_steps, 1)
        elif self.base_scheduler is not None:
            self.current_lr = self.base_scheduler.step()
        else:
            self.current_lr = self.base_lr

        self.current_step += 1
        return self.current_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


class LinearDecayScheduler:
    """Linear learning rate decay scheduler.

    Linearly decays learning rate from base_lr to min_lr over total_steps.

    Args:
        base_lr: Starting learning rate
        min_lr: Final learning rate
        total_steps: Total number of steps for decay
        warmup_steps: Optional warmup steps at start

    Example:
        >>> scheduler = LinearDecayScheduler(
        ...     base_lr=1e-3,
        ...     min_lr=1e-6,
        ...     total_steps=100000,
        ...     warmup_steps=1000,
        ... )
    """

    def __init__(
        self,
        base_lr: float,
        min_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
    ):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.current_lr = 0.0

    def step(self) -> float:
        """Advance scheduler by one step.

        Returns:
            Learning rate for current step
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup
            self.current_lr = self.base_lr * self.current_step / max(self.warmup_steps, 1)
        else:
            # Linear decay
            decay_step = self.current_step - self.warmup_steps
            decay_total = self.total_steps - self.warmup_steps
            progress = min(decay_step / max(decay_total, 1), 1.0)
            self.current_lr = self.base_lr - (self.base_lr - self.min_lr) * progress

        self.current_step += 1
        return self.current_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def state_dict(self) -> dict:
        """Get scheduler state."""
        return {
            "current_step": self.current_step,
            "current_lr": self.current_lr,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load scheduler state."""
        self.current_step = state["current_step"]
        self.current_lr = state["current_lr"]


class ExponentialDecayScheduler:
    """Exponential learning rate decay scheduler.

    Multiplies learning rate by decay factor at each step.

    Args:
        base_lr: Starting learning rate
        min_lr: Minimum learning rate (floor)
        decay_rate: Decay multiplier per step (e.g., 0.9999)
        warmup_steps: Optional warmup steps

    Example:
        >>> scheduler = ExponentialDecayScheduler(
        ...     base_lr=1e-3,
        ...     min_lr=1e-6,
        ...     decay_rate=0.9999,
        ... )
    """

    def __init__(
        self,
        base_lr: float,
        min_lr: float = 1e-7,
        decay_rate: float = 0.9999,
        warmup_steps: int = 0,
    ):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.current_lr = 0.0

    def step(self) -> float:
        """Advance scheduler by one step.

        Returns:
            Learning rate for current step
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup
            self.current_lr = self.base_lr * self.current_step / max(self.warmup_steps, 1)
        else:
            # Exponential decay
            decay_step = self.current_step - self.warmup_steps
            self.current_lr = max(self.base_lr * (self.decay_rate**decay_step), self.min_lr)

        self.current_step += 1
        return self.current_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


def create_scheduler(
    scheduler_type: str,
    **kwargs,
) -> CosineScheduler | LinearDecayScheduler | ExponentialDecayScheduler:
    """Factory function to create schedulers.

    Args:
        scheduler_type: Type of scheduler ("cosine", "linear", "exponential")
        **kwargs: Arguments passed to scheduler constructor

    Returns:
        Configured scheduler instance

    Example:
        >>> scheduler = create_scheduler(
        ...     "cosine",
        ...     base_lr=1e-3,
        ...     min_lr=1e-6,
        ...     epochs=100,
        ...     steps_per_epoch=1000,
        ... )
    """
    schedulers = {
        "cosine": CosineScheduler,
        "linear": LinearDecayScheduler,
        "exponential": ExponentialDecayScheduler,
    }

    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. " f"Available: {list(schedulers.keys())}")

    return schedulers[scheduler_type](**kwargs)
