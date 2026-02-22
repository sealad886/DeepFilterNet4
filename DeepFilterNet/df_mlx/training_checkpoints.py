"""Checkpoint management for dynamic training."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

_CHECKPOINT_KINDS = {"step", "epoch_end", "best", "best_final", "final", "interrupted"}
_COMPLETED_KINDS = {"epoch_end", "best", "best_final", "final"}
_IN_PROGRESS_KINDS = {"step", "interrupted"}
_COUNTER_SEMANTICS_VERSION = 2


@dataclass(frozen=True)
class CheckpointManifest:
    """Manifest describing checkpoint file layout and naming patterns."""

    weights_ext: str = ".safetensors"
    state_ext: str = ".state.json"
    tmp_suffixes: tuple[str, ...] = (".tmp", ".partial")
    epoch_complete_suffix: str = ".complete"

    step_re: re.Pattern[str] = re.compile(r"^step_(\d+)\.safetensors$")
    epoch_re: re.Pattern[str] = re.compile(r"^epoch_(\d+)\.safetensors$")
    interrupted_re: re.Pattern[str] = re.compile(r"^interrupted_epoch_(\d+)\.safetensors$")
    complete_re: re.Pattern[str] = re.compile(r"^epoch_(\d+)\.complete$")

    def state_path(self, weights_path: Path) -> Path:
        return weights_path.with_suffix(self.state_ext)

    def is_temporary(self, path: Path) -> bool:
        name = path.name
        return any(suffix in name for suffix in self.tmp_suffixes)

    def expected_from_name(self, path: Path) -> dict:
        name = path.name
        if match := self.step_re.match(name):
            return {"kind": "step", "global_step": int(match.group(1))}
        if match := self.epoch_re.match(name):
            return {"kind": "epoch_end", "epoch": int(match.group(1)) - 1}
        if match := self.interrupted_re.match(name):
            return {"kind": "interrupted", "epoch": int(match.group(1)) - 1}
        if name == "best.safetensors":
            return {"kinds": {"best", "best_final"}}
        if name == "final.safetensors":
            return {"kinds": {"final"}}
        return {}

    def marker_epoch(self, path: Path) -> int | None:
        if match := self.complete_re.match(path.name):
            return int(match.group(1)) - 1
        return None


def _disc_weights_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.disc{path.suffix}")


def _is_disc_weights(path: Path, manifest: CheckpointManifest | None = None) -> bool:
    manifest = manifest or CheckpointManifest()
    return path.name.endswith(f".disc{manifest.weights_ext}")


@dataclass
class CheckpointRecord:
    """Parsed checkpoint metadata for validation and resume planning."""

    path: Path
    state_path: Path
    mtime: float
    state: dict[str, Any] | None = None
    kind: str | None = None
    epoch: int | None = None
    batch_idx: int | None = None
    global_step: int | None = None
    last_completed_epoch: int | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def _record_sort_key(record: CheckpointRecord) -> tuple[int, float]:
    """Sort checkpoints by global_step when available, falling back to mtime."""
    if record.global_step is None:
        return (-1, record.mtime)
    return (record.global_step, record.mtime)


def _validate_checkpoint_pair(checkpoint_path: Path, *, manifest: CheckpointManifest | None = None) -> bool:
    """Validate that both weights and state files exist and are non-empty.

    Args:
        checkpoint_path: Path to checkpoint (.safetensors file)

    Returns:
        True if both files exist and are valid, False otherwise
    """
    manifest = manifest or CheckpointManifest()
    weights_file = checkpoint_path
    state_file = manifest.state_path(checkpoint_path)

    # Check both files exist
    if not weights_file.exists():
        print(f"⚠️  Checkpoint missing: {weights_file.name}")
        return False
    if not state_file.exists():
        print(f"⚠️  Checkpoint missing state file: {state_file.name}")
        return False

    # Check files are not empty (indicates incomplete write)
    if weights_file.stat().st_size == 0:
        print(f"⚠️  Checkpoint is empty: {weights_file.name}")
        return False
    if state_file.stat().st_size == 0:
        print(f"⚠️  Checkpoint state file is empty: {state_file.name}")
        return False

    return True


def compute_resume_epoch(state: dict) -> int:
    """Determine the epoch index to resume from based on checkpoint kind."""
    epoch = int(state.get("epoch", 0))
    kind = state.get("kind", "epoch_end")
    if kind in _COMPLETED_KINDS:
        return epoch + 1
    return epoch


def resolve_resume_batch_count(state: dict[str, Any]) -> int:
    """Resolve resume micro-batch count from checkpoint state.

    Returns the number of micro-batches already consumed in the in-progress
    epoch. For legacy checkpoints (without counter_semantics_version),
    batch_idx is interpreted as a 0-based index of the last processed batch and
    is converted to a processed-count via +1.
    """
    kind = state.get("kind", "epoch_end")
    if kind not in _IN_PROGRESS_KINDS:
        return 0

    raw_counter = state.get("micro_batches_completed", state.get("batch_idx"))
    if not isinstance(raw_counter, int) or raw_counter < 0:
        return 0

    version_raw = state.get("counter_semantics_version", 1)
    version = version_raw if isinstance(version_raw, int) else 1
    if version >= _COUNTER_SEMANTICS_VERSION:
        return raw_counter
    return raw_counter + 1


def maybe_skip_resume_batches(
    data_iterator,
    *,
    resume_from: str | None,
    epoch: int,
    start_epoch: int,
    resume_batch_idx: int,
):
    """Skip already-processed micro-batches when resuming an in-progress epoch.

    Returns a tuple of (iterator, did_skip).
    """
    if resume_from and epoch == start_epoch and resume_batch_idx > 0:
        return islice(data_iterator, resume_batch_idx, None), True
    return data_iterator, False


_TRAIN_MODE_COMPILED = "COMPILED"
_TRAIN_MODE_EAGER = "EAGER"


def resolve_epoch_train_mode(
    *,
    compiled_step_base_enabled: bool,
    gan_enabled: bool,
    gan_active: bool,
    previous_mode: Literal["COMPILED", "EAGER"] | None,
    experimental_compiled_gan: bool = False,
) -> tuple[Literal["COMPILED", "EAGER"], bool]:
    """Resolve training mode for an epoch with one-way COMPILED->EAGER semantics.

    Rules:
    - If compiled mode is globally blocked (debug, nan-skip, grad accumulation), use EAGER.
    - If GAN is active for this epoch, use EAGER — unless experimental_compiled_gan is True.
    - Once EAGER is entered, do not switch back to COMPILED in later epochs
      (unless experimental_compiled_gan keeps compiled mode through GAN activation).
    """
    if not experimental_compiled_gan and previous_mode == _TRAIN_MODE_EAGER:
        return _TRAIN_MODE_EAGER, False
    if not compiled_step_base_enabled:
        return _TRAIN_MODE_EAGER, False
    if gan_enabled and gan_active and not experimental_compiled_gan:
        return _TRAIN_MODE_EAGER, False
    return _TRAIN_MODE_COMPILED, True


def validate_checkpoint_dir(
    checkpoint_dir: Path,
    strict: bool = True,
    *,
    validate_load: bool = False,
) -> dict:
    """Validate checkpoints in a directory and return a resume plan.

    Args:
        checkpoint_dir: Directory containing checkpoints
        strict: If True, raise on any validation errors
        validate_load: If True, attempt to load checkpoint weights for integrity
    """
    manifest = CheckpointManifest()
    report = {
        "total": 0,
        "valid": 0,
        "invalid": [],
        "latest_path": None,
        "latest_state": None,
        "last_completed_epoch": -1,
        "resume_epoch": 0,
        "resume_batch": 0,
        "resume_global_step": None,
        "warnings": [],
    }

    if not checkpoint_dir.exists():
        return report

    tmp_files = [p for p in checkpoint_dir.iterdir() if manifest.is_temporary(p)]
    for tmp in tmp_files:
        report["invalid"].append((tmp, "temporary checkpoint residue"))

    ckpt_files = sorted(
        [
            p
            for p in checkpoint_dir.glob(f"*{manifest.weights_ext}")
            if not manifest.is_temporary(p) and not _is_disc_weights(p, manifest)
        ],
        key=lambda p: p.stat().st_mtime,
    )

    records: list[CheckpointRecord] = []

    for ckpt in ckpt_files:
        report["total"] += 1
        state_path = manifest.state_path(ckpt)
        record = CheckpointRecord(path=ckpt, state_path=state_path, mtime=ckpt.stat().st_mtime)

        if not ckpt.exists():
            record.errors.append("weights missing")
        elif ckpt.stat().st_size == 0:
            record.errors.append("weights file is empty")

        if not state_path.exists():
            record.errors.append("state missing")
        elif state_path.stat().st_size == 0:
            record.errors.append("state file is empty")

        if record.errors:
            records.append(record)
            report["invalid"].append((ckpt, "; ".join(record.errors)))
            continue

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            record.errors.append(f"state load error: {e}")
            records.append(record)
            report["invalid"].append((ckpt, "; ".join(record.errors)))
            continue

        record.state = state
        kind = state.get("kind")
        epoch = state.get("epoch")
        last_completed = state.get("last_completed_epoch")
        batch_idx = state.get("micro_batches_completed", state.get("batch_idx"))
        global_step = state.get("optimizer_steps_completed", state.get("global_step"))

        if kind not in _CHECKPOINT_KINDS:
            record.errors.append("missing/invalid kind")
        if not isinstance(epoch, int):
            record.errors.append("missing/invalid epoch")
        if not isinstance(last_completed, int):
            record.errors.append("missing/invalid last_completed_epoch")
        if batch_idx is not None and not isinstance(batch_idx, int):
            record.errors.append("invalid batch_idx")
        if global_step is not None and not isinstance(global_step, int):
            record.errors.append("invalid global_step")

        record.kind = kind if isinstance(kind, str) else None
        record.epoch = epoch if isinstance(epoch, int) else None
        record.batch_idx = batch_idx if isinstance(batch_idx, int) else None
        record.global_step = global_step if isinstance(global_step, int) else None
        record.last_completed_epoch = last_completed if isinstance(last_completed, int) else None

        expected = manifest.expected_from_name(ckpt)
        if expected:
            expected_kind = expected.get("kind")
            expected_kinds = expected.get("kinds")
            if expected_kind and kind != expected_kind:
                record.errors.append(f"kind mismatch (expected {expected_kind})")
            if expected_kinds and kind not in expected_kinds:
                record.errors.append(f"kind mismatch (expected {sorted(expected_kinds)})")
            if expected.get("epoch") is not None and isinstance(epoch, int):
                if epoch != expected["epoch"]:
                    record.errors.append(f"epoch mismatch (state {epoch} vs name {expected['epoch']})")
            if expected.get("global_step") is not None and isinstance(global_step, int):
                if global_step != expected["global_step"]:
                    record.errors.append(
                        f"global_step mismatch (state {global_step} vs name {expected['global_step']})"
                    )
        else:
            record.errors.append("unrecognized checkpoint filename")

        if isinstance(kind, str) and isinstance(epoch, int) and isinstance(last_completed, int):
            if kind in _COMPLETED_KINDS:
                if last_completed < epoch:
                    record.errors.append("completed kind but last_completed_epoch < epoch")
            elif kind in _IN_PROGRESS_KINDS:
                if last_completed > epoch - 1:
                    record.errors.append("in-progress kind but last_completed_epoch too high")
            if kind in _IN_PROGRESS_KINDS and record.batch_idx is None:
                record.errors.append("in-progress checkpoint missing batch_idx")
            if kind == "step" and record.global_step is None:
                record.errors.append("step checkpoint missing global_step")

        checkpoint_kind = state.get("checkpoint_kind")
        if checkpoint_kind is not None:
            expected_checkpoint_kind = "end_of_epoch" if kind in _COMPLETED_KINDS else "in_progress"
            if checkpoint_kind != expected_checkpoint_kind:
                record.errors.append("checkpoint_kind mismatch")

        if state.get("current_epoch") is not None and state.get("current_epoch") != epoch:
            record.errors.append("current_epoch mismatch")
        if state.get("last_saved_global_step") is not None and state.get("last_saved_global_step") != global_step:
            record.errors.append("last_saved_global_step mismatch")
        if state.get("last_saved_batch_idx") is not None and state.get("last_saved_batch_idx") != batch_idx:
            record.errors.append("last_saved_batch_idx mismatch")

        if validate_load and not record.errors:
            try:
                _ = mx.load(str(ckpt))
            except Exception as e:
                record.errors.append(f"weights load error: {e}")

        records.append(record)
        if record.valid:
            report["valid"] += 1
            if record.last_completed_epoch is not None:
                report["last_completed_epoch"] = max(report["last_completed_epoch"], record.last_completed_epoch)
        else:
            report["invalid"].append((ckpt, "; ".join(record.errors)))

    marker_files = list(checkpoint_dir.glob(f"epoch_*{manifest.epoch_complete_suffix}"))
    marker_epochs = {}
    for marker in marker_files:
        if marker.stat().st_size == 0:
            report["invalid"].append((marker, "epoch complete marker is empty"))
            continue
        marker_epoch = manifest.marker_epoch(marker)
        if marker_epoch is None:
            report["invalid"].append((marker, "unrecognized epoch complete marker name"))
            continue
        marker_epochs[marker_epoch] = marker

    if marker_epochs:
        completed_epochs = {
            rec.epoch for rec in records if rec.valid and rec.kind in _COMPLETED_KINDS and rec.epoch is not None
        }
        for epoch_idx, marker in marker_epochs.items():
            if epoch_idx not in completed_epochs:
                report["invalid"].append((marker, "epoch complete marker without valid end-of-epoch checkpoint"))

    valid_records = [rec for rec in records if rec.valid]
    if valid_records:
        latest = max(valid_records, key=_record_sort_key)
        report["latest_path"] = latest.path
        report["latest_state"] = latest.state
        if latest.state:
            report["resume_epoch"] = compute_resume_epoch(latest.state)
            report["resume_batch"] = resolve_resume_batch_count(latest.state)
            report["resume_global_step"] = latest.state.get(
                "optimizer_steps_completed", latest.state.get("global_step")
            )

    # Detect monotonicity issues across valid checkpoints (by modification time).
    valid_by_time = sorted(valid_records, key=lambda rec: rec.mtime)
    last_epoch_seen = None
    last_step_seen = None
    last_completed_seen = None
    for rec in valid_by_time:
        if rec.epoch is not None:
            if last_epoch_seen is not None and rec.epoch < last_epoch_seen:
                report["invalid"].append((rec.path, "epoch decreased relative to earlier checkpoint"))
            last_epoch_seen = rec.epoch
        if rec.global_step is not None:
            if last_step_seen is not None and rec.global_step < last_step_seen:
                report["invalid"].append((rec.path, "global_step decreased relative to earlier checkpoint"))
            last_step_seen = rec.global_step
        if rec.last_completed_epoch is not None:
            if last_completed_seen is not None and rec.last_completed_epoch < last_completed_seen:
                report["invalid"].append((rec.path, "last_completed_epoch decreased relative to earlier checkpoint"))
            last_completed_seen = rec.last_completed_epoch

    data_ckpt = checkpoint_dir / "data_checkpoint.json"
    if data_ckpt.exists():
        try:
            with open(data_ckpt, "r", encoding="utf-8") as f:
                data_state = json.load(f)
            data_epoch = data_state.get("epoch")
            data_batch = data_state.get("batch_idx")
            if not isinstance(data_epoch, int) or data_epoch < 0:
                report["invalid"].append((data_ckpt, "data checkpoint has invalid epoch"))
            if not isinstance(data_batch, int) or data_batch < 0:
                report["invalid"].append((data_ckpt, "data checkpoint has invalid batch_idx"))
            if report["latest_state"] and isinstance(data_epoch, int):
                latest_epoch = report["latest_state"].get("epoch")
                if isinstance(latest_epoch, int) and data_epoch > latest_epoch:
                    report["invalid"].append((data_ckpt, "data checkpoint epoch exceeds latest model checkpoint epoch"))
        except Exception as e:
            report["invalid"].append((data_ckpt, f"data checkpoint load error: {e}"))

    if report["invalid"] and strict:
        msgs = [f"{p.name}: {reason}" for p, reason in report["invalid"]]
        raise RuntimeError(
            "Checkpoint validation failed:\n  "
            + "\n  ".join(msgs)
            + "\nRemediation: remove or move corrupted checkpoints/markers and retry."
        )

    return report


def _write_epoch_complete_marker(checkpoint_dir: Path, epoch: int, checkpoint_path: Path) -> bool:
    """Write an epoch completion marker after a successful end-of-epoch checkpoint."""
    manifest = CheckpointManifest()
    marker_path = checkpoint_dir / f"epoch_{epoch + 1:03d}{manifest.epoch_complete_suffix}"
    tmp_marker = marker_path.with_name(f"{marker_path.name}.tmp")
    marker_state = {
        "epoch": epoch,
        "checkpoint": checkpoint_path.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(tmp_marker, "w", encoding="utf-8") as f:
            json.dump(marker_state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_marker.replace(marker_path)
        return True
    except Exception as e:
        print(f"⚠️  Failed to write epoch completion marker: {e}")
        return False


def save_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    epoch: int,
    batch_idx: int | None = None,
    global_step: int | None = None,
    loss: float,
    best_valid_loss: float,
    config: dict,
    optimizer: optim.Optimizer | None = None,
    discriminator: nn.Module | None = None,
    disc_optimizer: optim.Optimizer | None = None,
    last_completed_epoch: int = -1,
    kind: str = "epoch_end",
    raise_on_error: bool = False,
) -> bool:
    """Save a training checkpoint with model weights, training state, and optimizer state.

    Args:
        model: Model to save
        path: Path to checkpoint file (.safetensors)
        epoch: Current epoch index (0-based)
        batch_idx: Number of micro-batches completed within the current epoch
        global_step: Number of optimizer updates completed globally
        loss: Current training loss
        best_valid_loss: Best validation loss so far
        config: Training configuration dict
        optimizer: Optional optimizer to save state from
        last_completed_epoch: Last fully completed epoch index (-1 if none)
        kind: Checkpoint kind: step | epoch_end | best | final | interrupted
        raise_on_error: Raise on failure instead of returning False
    Returns:
        True if checkpoint was saved and validated, False otherwise.
    """
    from mlx.utils import tree_flatten

    manifest = CheckpointManifest()
    tmp_weights: Path | None = None
    tmp_state_path: Path | None = None
    tmp_disc: Path | None = None

    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_weights = path.with_name(f"{path.stem}.tmp{path.suffix}")

        # Flatten nested params for safetensors
        params = model.parameters()
        flat_params = tree_flatten(params)
        weights = {k: v for k, v in flat_params}

        # Ensure tensors are materialized before writing and retry once if needed
        if weights:
            mx.eval(*weights.values())
        mx.save_safetensors(str(tmp_weights), weights)
        if not tmp_weights.exists():
            mx.save_safetensors(str(tmp_weights), weights)

        # Prepare optimizer state for serialization
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

        disc_optimizer_state_dict = {}
        if disc_optimizer is not None and hasattr(disc_optimizer, "state") and disc_optimizer.state:
            try:
                flat_state = tree_flatten(disc_optimizer.state)
                for k, v in flat_state:
                    if isinstance(v, mx.array):
                        disc_optimizer_state_dict[k] = v.tolist()
                    else:
                        disc_optimizer_state_dict[k] = v
            except Exception as e:
                print(f"⚠️  Failed to serialize discriminator optimizer state: {e}")

        checkpoint_kind = "end_of_epoch" if kind in _COMPLETED_KINDS else "in_progress"

        # Save training state and metadata
        state_path = manifest.state_path(path)
        tmp_state_path = state_path.with_name(f"{state_path.stem}.tmp{state_path.suffix}")
        state = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "micro_batches_completed": batch_idx,
            "global_step": global_step,
            "optimizer_steps_completed": global_step,
            "loss": loss,
            "best_valid_loss": best_valid_loss,
            "config": config,
            "optimizer_state": optimizer_state_dict,
            "disc_optimizer_state": disc_optimizer_state_dict,
            "last_completed_epoch": last_completed_epoch,
            "kind": kind,
            "checkpoint_kind": checkpoint_kind,
            "counter_semantics_version": _COUNTER_SEMANTICS_VERSION,
            "batch_unit": "microbatch_count",
            "step_unit": "optimizer_step",
            "current_epoch": epoch,
            "last_saved_global_step": global_step,
            "last_saved_batch_idx": batch_idx,
        }
        with open(tmp_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Prepare discriminator weights to temp file BEFORE main rename,
        # so that crash between main and disc renames only loses the disc
        # temp file (which is cleaned up as residue on next validate).
        if discriminator is not None:
            disc_path = _disc_weights_path(path)
            tmp_disc = disc_path.with_name(f"{disc_path.stem}.tmp{disc_path.suffix}")
            disc_params = discriminator.parameters()
            flat_disc = tree_flatten(disc_params)
            disc_weights = {k: v for k, v in flat_disc}
            if disc_weights:
                mx.eval(*disc_weights.values())
            mx.save_safetensors(str(tmp_disc), disc_weights)

        # Atomic rename — main checkpoint first, then discriminator
        tmp_weights.replace(path)
        tmp_state_path.replace(state_path)

        if tmp_disc is not None:
            disc_path = _disc_weights_path(path)
            tmp_disc.replace(disc_path)

        if not _validate_checkpoint_pair(path, manifest=manifest):
            msg = f"Checkpoint validation failed after save: {path.name}"
            if raise_on_error:
                raise RuntimeError(msg)
            print(f"⚠️  {msg}")
            return False

        if optimizer_state_dict:
            print(f"✅ Saved checkpoint with optimizer state: {path.name}")
        return True
    except Exception as e:
        # Clean up temp files to avoid disk leaks
        for tmp in (tmp_weights, tmp_state_path, tmp_disc):
            if tmp is not None:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except OSError:
                    pass
        if raise_on_error:
            raise
        print(f"❌ Failed to save checkpoint {Path(path).name}: {e}")
        return False


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: optim.Optimizer | None = None,
    discriminator: nn.Module | None = None,
    disc_optimizer: optim.Optimizer | None = None,
) -> dict:
    """Load a training checkpoint and restore model weights and optimizer state.

    Args:
        model: Model to load weights into
        path: Path to checkpoint file
        optimizer: Optional optimizer to restore state into

    Returns:
        Training state dict containing epoch, loss, etc.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    ckpt_path = Path(path)
    manifest = CheckpointManifest()

    # Validate checkpoint pair before loading
    if not _validate_checkpoint_pair(ckpt_path, manifest=manifest):
        print(f"⚠️  Checkpoint validation failed: {ckpt_path.name}")
        return {}

    try:
        # Load weights
        weights = mx.load(str(ckpt_path))

        # Align checkpoint weights with model's parameter tree
        flat_model = tree_flatten(model.parameters())
        pairs = []
        missing = []
        for name, param in flat_model:
            if isinstance(weights, dict) and name in weights:
                pairs.append((name, weights[name]))
            else:
                pairs.append((name, param))
                missing.append(name)

        nested_weights = tree_unflatten(pairs)
        model.update(nested_weights)

        if missing:
            print(f"⚠️  {len(missing)} parameters were missing in checkpoint")

        # Load training state
        state_path = manifest.state_path(ckpt_path)
        state = {}
        if state_path.exists():
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)

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
                    # Reconstruct optimizer state from flat dict
                    state_pairs = list(restored.items())
                    nested_state = tree_unflatten(state_pairs)
                    optimizer.state = nested_state
                    print("✅ Restored optimizer state from checkpoint")
            except Exception as e:
                print(f"⚠️  Failed to restore optimizer state: {e}")

        # Restore discriminator weights/optimizer if provided
        if discriminator is not None:
            disc_path = _disc_weights_path(ckpt_path)
            if disc_path.exists():
                try:
                    disc_weights = mx.load(str(disc_path))
                    flat_disc = tree_flatten(discriminator.parameters())
                    disc_pairs = []
                    missing_disc = []
                    for name, param in flat_disc:
                        if isinstance(disc_weights, dict) and name in disc_weights:
                            disc_pairs.append((name, disc_weights[name]))
                        else:
                            disc_pairs.append((name, param))
                            missing_disc.append(name)
                    discriminator.update(tree_unflatten(disc_pairs))
                    if missing_disc:
                        print(f"⚠️  {len(missing_disc)} discriminator parameters missing in checkpoint")
                except Exception as e:
                    print(f"⚠️  Failed to load discriminator weights: {e}")
            else:
                print(f"⚠️  Discriminator checkpoint missing: {disc_path.name}")

        if disc_optimizer is not None and "disc_optimizer_state" in state:
            try:
                disc_state_dict = state.get("disc_optimizer_state", {})
                if disc_state_dict:
                    restored = {k: mx.array(v) for k, v in disc_state_dict.items()}
                    disc_pairs = list(restored.items())
                    disc_nested = tree_unflatten(disc_pairs)
                    disc_optimizer.state = disc_nested
                    print("✅ Restored discriminator optimizer state from checkpoint")
            except Exception as e:
                print(f"⚠️  Failed to restore discriminator optimizer state: {e}")

        epoch = state.get("epoch", 0)
        kind = state.get("kind", "epoch_end")
        completed_kinds = {"epoch_end", "best", "best_final", "final"}
        last_completed = state.get("last_completed_epoch", epoch if kind in completed_kinds else epoch - 1)
        print(f"✅ Loaded checkpoint from epoch {epoch} (kind={kind}, last_completed={last_completed})")
        return state

    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        return {}


def cleanup_checkpoints(
    checkpoint_dir: Path,
    save_total_limit: int,
    keep_best: bool = True,
) -> None:
    """Remove old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        save_total_limit: Maximum number of checkpoints to keep
        keep_best: If True, always keep best.safetensors (doesn't count towards limit)
    """
    if save_total_limit <= 0:
        return

    manifest = CheckpointManifest()

    # Find all checkpoint files (epoch_*, step_*, and interrupted_epoch_*)
    ckpt_files = []
    for pattern in ["epoch_*.safetensors", "step_*.safetensors", "interrupted_epoch_*.safetensors"]:
        ckpt_files.extend([p for p in checkpoint_dir.glob(pattern) if not _is_disc_weights(p, manifest)])

    # Sort by modification time (oldest first)
    ckpt_files.sort(key=lambda p: p.stat().st_mtime)

    # Calculate how many to remove
    num_to_remove = len(ckpt_files) - save_total_limit

    if num_to_remove <= 0:
        return

    # Remove oldest checkpoints
    for ckpt_path in ckpt_files[:num_to_remove]:
        # Remove the safetensors file
        ckpt_path.unlink(missing_ok=True)
        # Remove discriminator weights if present
        _disc_weights_path(ckpt_path).unlink(missing_ok=True)

        # Also remove the accompanying state.json
        state_path = manifest.state_path(ckpt_path)
        state_path.unlink(missing_ok=True)

        # Remove epoch completion marker if present
        marker_epoch = manifest.expected_from_name(ckpt_path).get("epoch")
        if marker_epoch is not None:
            marker_path = checkpoint_dir / f"epoch_{marker_epoch + 1:03d}{manifest.epoch_complete_suffix}"
            marker_path.unlink(missing_ok=True)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the most recent checkpoint in the checkpoint directory.

    Returns the latest valid checkpoint based on metadata and modification time.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to most recent checkpoint, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None

    manifest = CheckpointManifest()
    candidates = [
        p
        for p in checkpoint_dir.glob(f"*{manifest.weights_ext}")
        if not manifest.is_temporary(p) and not _is_disc_weights(p, manifest)
    ]

    valid_pairs: list[Path] = []
    for ckpt in candidates:
        state_path = manifest.state_path(ckpt)
        if not ckpt.exists() or ckpt.stat().st_size == 0:
            continue
        if not state_path.exists() or state_path.stat().st_size == 0:
            continue
        valid_pairs.append(ckpt)

    if not valid_pairs:
        return None

    # Fast path: use latest mtime without loading large state JSON files.
    return max(valid_pairs, key=lambda p: p.stat().st_mtime)
