"""Signal handling and graceful interrupt management for training."""

from __future__ import annotations

import signal
import sys
from pathlib import Path

from df_mlx.training_checkpoints import save_checkpoint

# Global state for signal handler
_interrupt_state = {
    "checkpoint_dir": None,
    "epoch": 0,
    "batch_idx": 0,
    "global_step": 0,
    "model": None,
    "optimizer": None,
    "discriminator": None,
    "disc_optimizer": None,
    "loss": 0.0,
    "best_valid_loss": float("inf"),
    "config": {},
    "interrupted": False,
    "train_stream": None,
    "data_checkpoint_path": None,
    "last_completed_epoch": -1,
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
    signal_name = "SIGINT"
    if signum == signal.SIGTERM:
        signal_name = "SIGTERM"
    print("\n" + "=" * 60)
    print(f"⚠️  Training interrupted ({signal_name})")
    print("=" * 60)

    # Save final checkpoint
    if (
        _interrupt_state["model"] is not None
        and _interrupt_state["optimizer"] is not None
        and _interrupt_state["checkpoint_dir"] is not None
    ):
        try:
            print("💾 Saving final checkpoint before exit...")
            ckpt_dir = Path(_interrupt_state["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            epoch_idx = _interrupt_state.get("epoch", 0)
            batch_idx = _interrupt_state.get("batch_idx", 0)
            gstep = _interrupt_state.get("global_step", 0)
            last_completed = _interrupt_state.get("last_completed_epoch", -1)

            final_path = ckpt_dir / f"interrupted_epoch_{epoch_idx + 1:03d}.safetensors"
            saved = save_checkpoint(
                _interrupt_state["model"],
                final_path,
                epoch=epoch_idx,
                batch_idx=batch_idx,
                global_step=gstep,
                loss=_interrupt_state["loss"],
                best_valid_loss=_interrupt_state["best_valid_loss"],
                config=_interrupt_state["config"],
                optimizer=_interrupt_state["optimizer"],
                discriminator=_interrupt_state.get("discriminator"),
                disc_optimizer=_interrupt_state.get("disc_optimizer"),
                last_completed_epoch=last_completed,
                kind="interrupted",
            )
            if saved:
                print(f"✅ Final checkpoint saved to {final_path}")
            else:
                print(f"❌ Failed to save final checkpoint to {final_path}")

            # Also persist MLXDataStream state so --resume-data works after interrupts.
            train_stream = _interrupt_state.get("train_stream")
            data_ckpt_path = _interrupt_state.get("data_checkpoint_path")
            if train_stream is not None and data_ckpt_path is not None:
                try:
                    # Sync data stream position with model's authoritative batch count.
                    # The data iterator may have pre-incremented its counter for the
                    # *next* batch before the training loop finished processing the
                    # current one, so the model's batch_idx is the true count of
                    # fully-processed micro-batches.
                    train_stream._checkpoint.epoch = epoch_idx
                    train_stream._checkpoint.batch_idx = batch_idx
                    train_stream._batch_count = batch_idx
                    train_stream.save_checkpoint(data_ckpt_path)
                    print(f"✅ Data checkpoint saved to {data_ckpt_path}")
                except Exception as e_data:
                    print(f"❌ Failed to save data checkpoint: {data_ckpt_path} ({e_data})")
        except Exception as e:
            print(f"❌ Failed to save final checkpoint: {e}")

    print("Exiting...")
    raise KeyboardInterrupt()


def _register_sigint_handler(
    model,
    optimizer,
    checkpoint_dir,
    config,
    *,
    discriminator=None,
    disc_optimizer=None,
    last_completed_epoch: int = -1,
):
    """Register SIGINT handler for graceful training shutdown.

    Args:
        model: Model to save on interrupt
        optimizer: Optimizer to save state on interrupt
        checkpoint_dir: Directory to save checkpoint to
        config: Training configuration dict
        last_completed_epoch: Last fully completed epoch when registering
    """
    _interrupt_state["model"] = model
    _interrupt_state["optimizer"] = optimizer
    _interrupt_state["discriminator"] = discriminator
    _interrupt_state["disc_optimizer"] = disc_optimizer
    _interrupt_state["checkpoint_dir"] = checkpoint_dir
    _interrupt_state["config"] = config
    _interrupt_state["last_completed_epoch"] = last_completed_epoch
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)


def _update_interrupt_state(epoch, loss, best_valid_loss, *, batch_idx=0, global_step=0, last_completed_epoch=-1):
    """Update global state for interrupt handler.

    Args:
        epoch: Current epoch
        loss: Current training loss
        best_valid_loss: Best validation loss so far
        batch_idx: Current batch index within epoch
        global_step: Global training step
        last_completed_epoch: Last fully completed epoch index
    """
    _interrupt_state["epoch"] = epoch
    _interrupt_state["batch_idx"] = batch_idx
    _interrupt_state["global_step"] = global_step
    _interrupt_state["loss"] = loss
    _interrupt_state["best_valid_loss"] = best_valid_loss
    _interrupt_state["last_completed_epoch"] = last_completed_epoch
