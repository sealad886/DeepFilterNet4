"""Checkpoint management for MLX DeepFilterNet4 training.

Provides checkpoint save/load functionality with:
- Patience tracking for early stopping
- Best model selection based on validation loss
- Complete state management (model, optimizer, scheduler, epoch)
- Safe atomic saves with temporary files

Ported from df/checkpoint.py with MLX-specific adaptations.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .lr import CosineScheduler

# ============================================================================
# Constants
# ============================================================================

CHECKPOINT_FILE = "checkpoint.safetensors"
STATE_FILE = "state.json"
BEST_CHECKPOINT = "best_checkpoint.safetensors"
PATIENCE_FILE = "patience.json"


# ============================================================================
# Patience Tracking
# ============================================================================


@dataclass
class PatienceState:
    """State for patience-based early stopping.

    Attributes:
        best_loss: Best validation loss seen so far
        best_epoch: Epoch where best loss was achieved
        patience_count: Number of epochs without improvement
        max_patience: Maximum patience before stopping
        min_delta: Minimum improvement to reset patience
    """

    best_loss: float = float("inf")
    best_epoch: int = 0
    patience_count: int = 0
    max_patience: int = 10
    min_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "patience_count": self.patience_count,
            "max_patience": self.max_patience,
            "min_delta": self.min_delta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatienceState":
        """Create from dictionary."""
        return cls(**d)


def check_patience(
    current_loss: float,
    state: PatienceState,
) -> Tuple[bool, bool, PatienceState]:
    """Check if training should stop based on patience.

    Args:
        current_loss: Current validation loss
        state: Current patience state

    Returns:
        Tuple of:
            - is_best: Whether this is the best loss
            - should_stop: Whether to stop training
            - new_state: Updated patience state
    """
    improvement = state.best_loss - current_loss

    if improvement > state.min_delta:
        # New best - reset patience
        new_state = PatienceState(
            best_loss=current_loss,
            best_epoch=state.best_epoch + state.patience_count + 1,
            patience_count=0,
            max_patience=state.max_patience,
            min_delta=state.min_delta,
        )
        return True, False, new_state
    else:
        # No improvement
        new_patience = state.patience_count + 1
        should_stop = new_patience >= state.max_patience
        new_state = PatienceState(
            best_loss=state.best_loss,
            best_epoch=state.best_epoch,
            patience_count=new_patience,
            max_patience=state.max_patience,
            min_delta=state.min_delta,
        )
        return False, should_stop, new_state


def read_patience(checkpoint_dir: Path | str) -> PatienceState:
    """Read patience state from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing patience file

    Returns:
        Patience state (default state if file doesn't exist)
    """
    path = Path(checkpoint_dir) / PATIENCE_FILE
    if path.exists():
        with open(path) as f:
            return PatienceState.from_dict(json.load(f))
    return PatienceState()


def write_patience(checkpoint_dir: Path | str, state: PatienceState) -> None:
    """Write patience state to checkpoint directory.

    Args:
        checkpoint_dir: Directory to save patience file
        state: Patience state to save
    """
    path = Path(checkpoint_dir) / PATIENCE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state.to_dict(), f, indent=2)


# ============================================================================
# Checkpoint Management
# ============================================================================


@dataclass
class CheckpointState:
    """Complete training state for checkpointing.

    Attributes:
        epoch: Current epoch number
        step: Current global step
        loss: Current/last loss value
        best_loss: Best validation loss
        patience: Patience tracking state
        config: Optional config dictionary
    """

    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    best_loss: float = float("inf")
    patience: Optional[PatienceState] = None
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "best_loss": self.best_loss,
        }
        if self.patience is not None:
            d["patience"] = self.patience.to_dict()
        if self.config is not None:
            d["config"] = self.config
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointState":
        """Create from dictionary."""
        patience = None
        if "patience" in d:
            patience = PatienceState.from_dict(d["patience"])
        return cls(
            epoch=d.get("epoch", 0),
            step=d.get("step", 0),
            loss=d.get("loss", 0.0),
            best_loss=d.get("best_loss", float("inf")),
            patience=patience,
            config=d.get("config"),
        )


def save_checkpoint(
    checkpoint_dir: Path | str,
    model: nn.Module,
    state: CheckpointState,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[CosineScheduler] = None,
    is_best: bool = False,
) -> Path:
    """Save model checkpoint with complete training state.

    Uses atomic writes with temporary files to prevent corruption.

    Args:
        checkpoint_dir: Directory to save checkpoint
        model: Model to save
        state: Training state
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        is_best: Whether this is the best checkpoint

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILE
    temp_path = checkpoint_path.with_suffix(".tmp")

    params = model.parameters()
    flat_params = tree_flatten(params)
    weights = {k: v for k, v in flat_params}
    mx.save_safetensors(str(temp_path), weights)

    # Atomic rename
    shutil.move(str(temp_path), str(checkpoint_path))

    # Build state dict
    state_dict = state.to_dict()

    # Add optimizer state if provided
    if optimizer is not None:
        opt_state = optimizer.state
        if opt_state:
            # Flatten optimizer state for JSON serialization
            flat_opt = tree_flatten(opt_state)
            # Convert arrays to lists for JSON
            opt_dict = {}
            for k, v in flat_opt:
                if isinstance(v, mx.array):
                    opt_dict[k] = v.tolist()
                else:
                    opt_dict[k] = v
            state_dict["optimizer"] = opt_dict

    # Add scheduler state if provided
    if scheduler is not None:
        state_dict["scheduler"] = scheduler.state_dict()

    # Save state
    state_path = checkpoint_dir / STATE_FILE
    with open(state_path, "w") as f:
        json.dump(state_dict, f, indent=2)

    # Save patience separately for easy access
    if state.patience is not None:
        write_patience(checkpoint_dir, state.patience)

    # Copy to best if needed
    if is_best:
        best_path = checkpoint_dir / BEST_CHECKPOINT
        shutil.copy(str(checkpoint_path), str(best_path))

    return checkpoint_path


def load_checkpoint(
    checkpoint_dir: Path | str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[CosineScheduler] = None,
    load_best: bool = False,
) -> CheckpointState:
    """Load model checkpoint and restore training state.

    Args:
        checkpoint_dir: Directory containing checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        load_best: Whether to load best checkpoint instead of latest

    Returns:
        Restored training state

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Choose checkpoint file
    if load_best:
        checkpoint_path = checkpoint_dir / BEST_CHECKPOINT
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_dir / CHECKPOINT_FILE
    else:
        checkpoint_path = checkpoint_dir / CHECKPOINT_FILE

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model weights
    weights: Dict[str, mx.array] = mx.load(str(checkpoint_path))  # type: ignore
    model.load_weights(list(weights.items()))

    # Load state
    state_path = checkpoint_dir / STATE_FILE
    if state_path.exists():
        with open(state_path) as f:
            state_dict = json.load(f)
    else:
        state_dict = {}

    state = CheckpointState.from_dict(state_dict)

    # Restore optimizer state
    if optimizer is not None and "optimizer" in state_dict:
        opt_dict = state_dict["optimizer"]
        # Convert lists back to arrays
        restored = {}
        for k, v in opt_dict.items():
            if isinstance(v, list):
                restored[k] = mx.array(v)
            else:
                restored[k] = v
        # Unflatten and set
        try:
            unflat = tree_unflatten(list(restored.items()))
            optimizer.state = unflat
        except Exception:
            pass  # Optimizer state mismatch - start fresh

    # Restore scheduler state
    if scheduler is not None and "scheduler" in state_dict:
        scheduler.load_state_dict(state_dict["scheduler"])

    # Load patience
    state.patience = read_patience(checkpoint_dir)

    return state


def checkpoint_exists(checkpoint_dir: Path | str) -> bool:
    """Check if a checkpoint exists in the directory.

    Args:
        checkpoint_dir: Directory to check

    Returns:
        True if checkpoint exists
    """
    checkpoint_dir = Path(checkpoint_dir)
    return (checkpoint_dir / CHECKPOINT_FILE).exists()


def get_latest_checkpoint(base_dir: Path | str) -> Optional[Path]:
    """Find the latest checkpoint in a base directory.

    Looks for numbered checkpoint directories (e.g., checkpoint_001)
    and returns the highest-numbered one.

    Args:
        base_dir: Base directory to search

    Returns:
        Path to latest checkpoint directory, or None
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None

    # Look for checkpoint_XXX directories
    checkpoint_dirs = list(base_dir.glob("checkpoint_*"))
    if not checkpoint_dirs:
        # Check if base_dir itself is a checkpoint
        if checkpoint_exists(base_dir):
            return base_dir
        return None

    # Sort by number
    def extract_num(p: Path) -> int:
        try:
            return int(p.name.split("_")[-1])
        except ValueError:
            return -1

    checkpoint_dirs.sort(key=extract_num, reverse=True)

    for d in checkpoint_dirs:
        if checkpoint_exists(d):
            return d

    return None


# ============================================================================
# Helper Functions
# ============================================================================


def load_model(
    model: nn.Module,
    checkpoint_path: Path | str,
    strict: bool = True,
) -> nn.Module:
    """Load model weights from checkpoint file.

    Simple wrapper for loading just model weights without full state.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to safetensors file
        strict: Whether to require all keys to match

    Returns:
        Model with loaded weights
    """
    checkpoint_path = Path(checkpoint_path)
    weights: Dict[str, mx.array] = mx.load(str(checkpoint_path))  # type: ignore
    model.load_weights(list(weights.items()), strict=strict)
    return model


def save_model(
    model: nn.Module,
    path: Path | str,
) -> Path:
    """Save model weights to file.

    Simple wrapper for saving just model weights.

    Args:
        model: Model to save
        path: Output path

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    params = model.parameters()
    flat_params = tree_flatten(params)
    weights = {k: v for k, v in flat_params}
    mx.save_safetensors(str(path), weights)

    return path


class CheckpointManager:
    """Manager for handling multiple checkpoints with rotation.

    Keeps track of the N most recent checkpoints and automatically
    removes old ones.

    Args:
        checkpoint_dir: Base directory for checkpoints
        max_to_keep: Maximum number of checkpoints to keep
        save_best: Whether to save best checkpoint separately

    Example:
        >>> manager = CheckpointManager("checkpoints", max_to_keep=5)
        >>> for epoch in range(100):
        ...     loss = train_epoch()
        ...     manager.save(model, epoch, loss, is_best=(loss < best_loss))
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        max_to_keep: int = 5,
        save_best: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_to_keep = max_to_keep
        self.save_best = save_best
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoints: list[Path] = []
        self._scan_existing()

    def _scan_existing(self) -> None:
        """Scan for existing checkpoints."""
        checkpoint_dirs = list(self.checkpoint_dir.glob("checkpoint_*"))

        def extract_num(p: Path) -> int:
            try:
                return int(p.name.split("_")[-1])
            except ValueError:
                return -1

        checkpoint_dirs.sort(key=extract_num)
        self._checkpoints = [d for d in checkpoint_dirs if checkpoint_exists(d)]

    def save(
        self,
        model: nn.Module,
        epoch: int,
        loss: float,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[CosineScheduler] = None,
        is_best: bool = False,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a new checkpoint.

        Args:
            model: Model to save
            epoch: Current epoch
            loss: Current loss
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            is_best: Whether this is the best checkpoint
            extra_state: Optional extra state to save

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint directory
        checkpoint_subdir = self.checkpoint_dir / f"checkpoint_{epoch:04d}"

        # Build state
        state = CheckpointState(
            epoch=epoch,
            step=0,  # Can be updated if needed
            loss=loss,
            config=extra_state,
        )

        # Save checkpoint
        save_checkpoint(
            checkpoint_subdir,
            model,
            state,
            optimizer=optimizer,
            scheduler=scheduler,
            is_best=is_best and self.save_best,
        )

        self._checkpoints.append(checkpoint_subdir)

        # Remove old checkpoints
        while len(self._checkpoints) > self.max_to_keep:
            old = self._checkpoints.pop(0)
            if old.exists():
                shutil.rmtree(old)

        return checkpoint_subdir

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[CosineScheduler] = None,
    ) -> Optional[CheckpointState]:
        """Load the latest checkpoint.

        Args:
            model: Model to load into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore

        Returns:
            Checkpoint state, or None if no checkpoint exists
        """
        if not self._checkpoints:
            return None

        latest = self._checkpoints[-1]
        return load_checkpoint(latest, model, optimizer, scheduler)

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[CosineScheduler] = None,
    ) -> Optional[CheckpointState]:
        """Load the best checkpoint.

        Args:
            model: Model to load into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore

        Returns:
            Checkpoint state, or None if no checkpoint exists
        """
        best_path = self.checkpoint_dir / BEST_CHECKPOINT
        if best_path.exists():
            # Load from top-level best checkpoint
            weights: Dict[str, mx.array] = mx.load(str(best_path))  # type: ignore
            model.load_weights(list(weights.items()))
            # Try to load state from latest
            if self._checkpoints:
                state_path = self._checkpoints[-1] / STATE_FILE
                if state_path.exists():
                    with open(state_path) as f:
                        state_dict = json.load(f)
                    return CheckpointState.from_dict(state_dict)
            return CheckpointState()

        # Fall back to latest
        return self.load_latest(model, optimizer, scheduler)

    @property
    def latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        return self._checkpoints[-1] if self._checkpoints else None

    @property
    def num_checkpoints(self) -> int:
        """Get number of saved checkpoints."""
        return len(self._checkpoints)
