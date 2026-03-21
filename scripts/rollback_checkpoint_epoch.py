#!/usr/bin/env python3
"""Step a checkpoint directory back to a chosen resume epoch.

This helper trims newer model checkpoints and validates whether auto-resume
artifacts are coherent with df_mlx runtime semantics:

- ``--resume`` auto-resolves via ``find_latest_checkpoint(checkpoint_dir)``
- ``--resume-data`` auto-resolves via ``checkpoint_dir/data_checkpoint.json``

By default, the command runs in dry-run mode and only reports what would
change. Use ``--apply`` to perform file changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TextIO

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_ROOT = _REPO_ROOT / "DeepFilterNet"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from df_mlx.training_checkpoints import (  # noqa: E402
    _IN_PROGRESS_KINDS,
    CheckpointManifest,
    compute_resume_epoch,
    find_latest_checkpoint,
    resolve_resume_batch_count,
    validate_checkpoint_dir,
)


@dataclass(frozen=True)
class CheckpointSnapshot:
    """Model checkpoint metadata needed for rollback planning."""

    path: Path
    state_path: Path
    kind: str
    epoch: int
    resume_epoch: int
    resume_batch: int
    global_step: int | None
    stage_index: int | None
    stage_name: str | None
    mtime: float
    state: dict[str, Any]


@dataclass(frozen=True)
class DataCheckpointAssessment:
    """Coherence assessment for data checkpoint vs selected model checkpoint."""

    status: Literal[
        "missing",
        "coherent",
        "auto_correctable",
        "ignored_on_epoch_boundary",
        "error",
    ]
    message: str
    epoch: int | None
    batch_idx: int | None
    stage_index: int | None
    stage_name: str | None


@dataclass(frozen=True)
class RollbackPlan:
    """Full rollback plan before any file mutation."""

    checkpoint_dir: Path
    target_resume_epoch: int
    selected: CheckpointSnapshot
    projected_latest: CheckpointSnapshot
    removable: tuple[CheckpointSnapshot, ...]
    removable_markers: tuple[Path, ...]
    data_assessment: DataCheckpointAssessment


@dataclass
class ProgressReporter:
    """Emit robust user-facing progress updates for long-running stages."""

    enabled: bool = True
    progress_every: int = 100
    stream: TextIO | None = None

    def _emit(self, message: str) -> None:
        if self.enabled:
            print(message, file=self.stream or sys.stderr, flush=True)

    def info(self, message: str) -> None:
        self._emit(message)

    def start(self, label: str) -> float:
        self._emit(f"⏳ {label}...")
        return time.perf_counter()

    def done(self, label: str, start_time: float, *, detail: str | None = None) -> None:
        elapsed = time.perf_counter() - start_time
        suffix = f" — {detail}" if detail else ""
        self._emit(f"✅ {label} ({elapsed:.2f}s){suffix}")

    def loop(self, label: str, index: int, total: int, *, item: str | None = None) -> None:
        if not self.enabled or total <= 0:
            return
        should_emit = total <= 10 or index in {1, total} or index % max(self.progress_every, 1) == 0
        if not should_emit:
            return
        pct = 100.0 * index / total
        item_suffix = f" — {item}" if item and (total <= 10 or index in {1, total}) else ""
        self._emit(f"   {label}: {index}/{total} ({pct:.1f}%){item_suffix}")


def _disc_path(weights_path: Path) -> Path:
    return weights_path.with_name(f"{weights_path.stem}.disc{weights_path.suffix}")


def _resolve_target_resume_epoch(
    *,
    target_resume_epoch: int | None,
    target_epoch: int | None,
) -> int:
    if target_resume_epoch is not None and target_epoch is not None:
        raise ValueError("Use either --target-resume-epoch or --target-epoch, not both")
    if target_resume_epoch is None and target_epoch is None:
        raise ValueError("One of --target-resume-epoch or --target-epoch is required")

    if target_resume_epoch is not None:
        if target_resume_epoch < 0:
            raise ValueError(f"target resume epoch must be >= 0, got {target_resume_epoch}")
        return target_resume_epoch

    assert target_epoch is not None
    if target_epoch <= 0:
        raise ValueError(f"target epoch must be >= 1, got {target_epoch}")
    return target_epoch - 1


def _load_data_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_snapshots(checkpoint_dir: Path, progress: ProgressReporter) -> list[CheckpointSnapshot]:
    manifest = CheckpointManifest()
    ckpt_files = sorted(
        [
            p
            for p in checkpoint_dir.glob(f"*{manifest.weights_ext}")
            if not manifest.is_temporary(p) and not p.name.endswith(f".disc{manifest.weights_ext}")
        ],
        key=lambda p: p.stat().st_mtime,
    )
    progress.info(f"   Found {len(ckpt_files)} checkpoint weight files to inspect")
    stage_start = progress.start("Loading checkpoint metadata")

    snapshots: list[CheckpointSnapshot] = []
    total = len(ckpt_files)
    for idx, ckpt in enumerate(ckpt_files, start=1):
        progress.loop("Parsed metadata", idx, total, item=ckpt.name)
        state_path = manifest.state_path(ckpt)
        if not state_path.exists():
            continue
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to read state for {ckpt.name}: {exc}") from exc

        kind = state.get("kind")
        epoch = state.get("epoch")
        if not isinstance(kind, str) or kind == "":
            raise RuntimeError(f"Checkpoint {ckpt.name} has missing/invalid kind")
        if not isinstance(epoch, int) or epoch < 0:
            raise RuntimeError(f"Checkpoint {ckpt.name} has missing/invalid epoch")

        global_step = state.get("optimizer_steps_completed", state.get("global_step"))
        if not isinstance(global_step, int):
            global_step = None

        stage_index = state.get("pipeline_stage_index")
        if not isinstance(stage_index, int):
            stage_index = None

        stage_name = state.get("pipeline_stage_name")
        if not isinstance(stage_name, str):
            stage_name = None

        snapshots.append(
            CheckpointSnapshot(
                path=ckpt,
                state_path=state_path,
                kind=kind,
                epoch=epoch,
                resume_epoch=compute_resume_epoch(state),
                resume_batch=resolve_resume_batch_count(state),
                global_step=global_step,
                stage_index=stage_index,
                stage_name=stage_name,
                mtime=ckpt.stat().st_mtime,
                state=state,
            )
        )

    if not snapshots:
        raise RuntimeError(f"No valid checkpoint pairs found in {checkpoint_dir}")

    progress.done("Loading checkpoint metadata", stage_start, detail=f"{len(snapshots)} usable checkpoints")
    return snapshots


def _assess_data_checkpoint(
    data_state: dict[str, Any] | None,
    selected: CheckpointSnapshot,
) -> DataCheckpointAssessment:
    if data_state is None:
        return DataCheckpointAssessment(
            status="missing",
            message="data_checkpoint.json not found",
            epoch=None,
            batch_idx=None,
            stage_index=None,
            stage_name=None,
        )

    data_epoch = data_state.get("epoch")
    data_batch = data_state.get("batch_idx")
    data_stage_idx = data_state.get("pipeline_stage_index")
    data_stage_name = data_state.get("pipeline_stage_name")

    if not isinstance(data_epoch, int) or data_epoch < 0:
        return DataCheckpointAssessment(
            status="error",
            message=f"data checkpoint has invalid epoch: {data_epoch}",
            epoch=None,
            batch_idx=None,
            stage_index=None,
            stage_name=None,
        )
    if not isinstance(data_batch, int) or data_batch < 0:
        return DataCheckpointAssessment(
            status="error",
            message=f"data checkpoint has invalid batch_idx: {data_batch}",
            epoch=data_epoch,
            batch_idx=None,
            stage_index=None,
            stage_name=None,
        )
    if data_stage_idx is not None and (not isinstance(data_stage_idx, int) or data_stage_idx < 0):
        return DataCheckpointAssessment(
            status="error",
            message=f"data checkpoint has invalid pipeline_stage_index: {data_stage_idx}",
            epoch=data_epoch,
            batch_idx=data_batch,
            stage_index=None,
            stage_name=None,
        )
    if data_stage_name is not None and not isinstance(data_stage_name, str):
        return DataCheckpointAssessment(
            status="error",
            message=f"data checkpoint has invalid pipeline_stage_name: {data_stage_name}",
            epoch=data_epoch,
            batch_idx=data_batch,
            stage_index=data_stage_idx if isinstance(data_stage_idx, int) else None,
            stage_name=None,
        )

    if isinstance(data_stage_idx, int) and isinstance(selected.stage_index, int):
        if data_stage_idx > selected.stage_index:
            return DataCheckpointAssessment(
                status="error",
                message=(
                    "data checkpoint stage exceeds model stage "
                    f"(data={data_stage_idx}, model={selected.stage_index})"
                ),
                epoch=data_epoch,
                batch_idx=data_batch,
                stage_index=data_stage_idx,
                stage_name=data_stage_name if isinstance(data_stage_name, str) else None,
            )

    if selected.kind in _IN_PROGRESS_KINDS:
        if data_epoch == selected.resume_epoch and data_batch == selected.resume_batch:
            return DataCheckpointAssessment(
                status="coherent",
                message="data checkpoint matches in-progress model resume position",
                epoch=data_epoch,
                batch_idx=data_batch,
                stage_index=data_stage_idx if isinstance(data_stage_idx, int) else None,
                stage_name=data_stage_name if isinstance(data_stage_name, str) else None,
            )

        if data_epoch == selected.resume_epoch and abs(data_batch - selected.resume_batch) <= 1:
            return DataCheckpointAssessment(
                status="auto_correctable",
                message=(
                    "data checkpoint differs by <=1 micro-batch in same epoch "
                    f"(data={data_batch}, model={selected.resume_batch})"
                ),
                epoch=data_epoch,
                batch_idx=data_batch,
                stage_index=data_stage_idx if isinstance(data_stage_idx, int) else None,
                stage_name=data_stage_name if isinstance(data_stage_name, str) else None,
            )

        return DataCheckpointAssessment(
            status="error",
            message=(
                "data checkpoint disagrees with in-progress model resume position "
                f"(data=(epoch={data_epoch}, batch={data_batch}), "
                f"model=(epoch={selected.resume_epoch}, batch={selected.resume_batch}))"
            ),
            epoch=data_epoch,
            batch_idx=data_batch,
            stage_index=data_stage_idx if isinstance(data_stage_idx, int) else None,
            stage_name=data_stage_name if isinstance(data_stage_name, str) else None,
        )

    if data_epoch == selected.resume_epoch and data_batch == 0:
        return DataCheckpointAssessment(
            status="coherent",
            message="data checkpoint matches epoch-boundary resume position",
            epoch=data_epoch,
            batch_idx=data_batch,
            stage_index=data_stage_idx if isinstance(data_stage_idx, int) else None,
            stage_name=data_stage_name if isinstance(data_stage_name, str) else None,
        )

    return DataCheckpointAssessment(
        status="ignored_on_epoch_boundary",
        message=(
            "data checkpoint is mid-epoch and would be ignored/reset by epoch-boundary resume "
            f"(data=(epoch={data_epoch}, batch={data_batch}), resume_epoch={selected.resume_epoch})"
        ),
        epoch=data_epoch,
        batch_idx=data_batch,
        stage_index=data_stage_idx if isinstance(data_stage_idx, int) else None,
        stage_name=data_stage_name if isinstance(data_stage_name, str) else None,
    )


def build_rollback_plan(
    checkpoint_dir: Path,
    target_resume_epoch: int,
    *,
    progress: ProgressReporter,
) -> RollbackPlan:
    validation_start = progress.start("Validating checkpoint directory")
    report = validate_checkpoint_dir(checkpoint_dir, strict=False, validate_load=False)
    progress.done(
        "Validating checkpoint directory",
        validation_start,
        detail=f"total={report['total']} valid={report['valid']} invalid={len(report['invalid'])}",
    )
    non_data_invalid = [(path, reason) for path, reason in report["invalid"] if path.name != "data_checkpoint.json"]
    if non_data_invalid:
        invalid_msgs = "; ".join(f"{path.name}: {reason}" for path, reason in non_data_invalid)
        raise RuntimeError(
            "Checkpoint directory contains invalid artifacts. " "Clean these first before rollback: " f"{invalid_msgs}"
        )

    snapshots = _load_snapshots(checkpoint_dir, progress)
    target_candidates = [snapshot for snapshot in snapshots if snapshot.resume_epoch == target_resume_epoch]
    if not target_candidates:
        available_epochs = sorted({snapshot.resume_epoch for snapshot in snapshots})
        raise RuntimeError(
            f"No checkpoint resolves to resume epoch {target_resume_epoch}. "
            f"Available resume epochs: {available_epochs}"
        )

    selected = max(target_candidates, key=lambda snapshot: snapshot.mtime)
    removable_paths = {snapshot.path for snapshot in snapshots if snapshot.resume_epoch > target_resume_epoch}
    removable = tuple(snapshot for snapshot in snapshots if snapshot.path in removable_paths)
    remaining = [snapshot for snapshot in snapshots if snapshot.path not in removable_paths]
    projected_latest = max(remaining, key=lambda snapshot: snapshot.mtime)
    if projected_latest.resume_epoch != target_resume_epoch:
        raise RuntimeError(
            "Rollback would not make the target resume epoch latest for auto --resume. "
            f"Target={target_resume_epoch}, projected_latest={projected_latest.resume_epoch} "
            f"({projected_latest.path.name})."
        )

    manifest = CheckpointManifest()
    marker_scan_start = progress.start("Scanning epoch completion markers")
    removable_markers: list[Path] = []
    markers = sorted(checkpoint_dir.glob(f"epoch_*{manifest.epoch_complete_suffix}"))
    marker_total = len(markers)
    for idx, marker in enumerate(markers, start=1):
        progress.loop("Scanned markers", idx, marker_total, item=marker.name)
        marker_epoch = manifest.marker_epoch(marker)
        if marker_epoch is None:
            continue
        marker_resume_epoch = marker_epoch + 1
        if marker_resume_epoch > target_resume_epoch:
            removable_markers.append(marker)
    progress.done(
        "Scanning epoch completion markers",
        marker_scan_start,
        detail=f"{len(removable_markers)} marker(s) will be removed",
    )

    data_assess_start = progress.start("Assessing data checkpoint coherence")
    data_state = _load_data_checkpoint(checkpoint_dir / "data_checkpoint.json")
    data_assessment = _assess_data_checkpoint(data_state, projected_latest)
    progress.done(
        "Assessing data checkpoint coherence",
        data_assess_start,
        detail=f"status={data_assessment.status}",
    )
    progress.info(
        "✅ Rollback planning complete "
        f"(selected={selected.path.name}, projected_latest={projected_latest.path.name}, "
        f"remove={len(removable)}, target_resume_epoch={target_resume_epoch})"
    )

    return RollbackPlan(
        checkpoint_dir=checkpoint_dir,
        target_resume_epoch=target_resume_epoch,
        selected=selected,
        projected_latest=projected_latest,
        removable=removable,
        removable_markers=tuple(sorted(removable_markers)),
        data_assessment=data_assessment,
    )


def _normalized_data_checkpoint(existing: dict[str, Any] | None, selected: CheckpointSnapshot) -> dict[str, Any]:
    payload: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}

    old_batch = payload.get("batch_idx")
    old_samples = payload.get("samples_processed")
    samples_per_batch = 1
    if isinstance(old_batch, int) and old_batch > 0 and isinstance(old_samples, int) and old_samples >= 0:
        samples_per_batch = max(old_samples // old_batch, 1)

    payload["epoch"] = selected.resume_epoch
    payload["batch_idx"] = selected.resume_batch
    payload["samples_processed"] = selected.resume_batch * samples_per_batch
    payload["seed"] = payload.get("seed", 42)
    payload["split"] = payload.get("split", "train")

    if isinstance(selected.stage_index, int):
        payload["pipeline_stage_index"] = selected.stage_index
    elif "pipeline_stage_index" not in payload:
        payload["pipeline_stage_index"] = 0

    if isinstance(selected.stage_name, str) and selected.stage_name:
        payload["pipeline_stage_name"] = selected.stage_name
    elif "pipeline_stage_name" not in payload:
        payload["pipeline_stage_name"] = "default"

    payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def apply_rollback_plan(
    plan: RollbackPlan,
    *,
    sync_data_checkpoint: bool,
    progress: ProgressReporter,
) -> None:
    remove_start = progress.start("Removing superseded checkpoint files")
    total_remove = len(plan.removable)
    for idx, snapshot in enumerate(plan.removable, start=1):
        progress.loop("Removed checkpoints", idx, total_remove, item=snapshot.path.name)
        snapshot.path.unlink(missing_ok=True)
        snapshot.state_path.unlink(missing_ok=True)
        _disc_path(snapshot.path).unlink(missing_ok=True)
    progress.done("Removing superseded checkpoint files", remove_start, detail=f"removed={total_remove}")

    marker_remove_start = progress.start("Removing epoch completion markers")
    total_markers = len(plan.removable_markers)
    for idx, marker in enumerate(plan.removable_markers, start=1):
        progress.loop("Removed markers", idx, total_markers, item=marker.name)
        marker.unlink(missing_ok=True)
    progress.done("Removing epoch completion markers", marker_remove_start, detail=f"removed={total_markers}")

    if sync_data_checkpoint:
        sync_start = progress.start("Synchronizing data checkpoint")
        data_path = plan.checkpoint_dir / "data_checkpoint.json"
        existing_data = _load_data_checkpoint(data_path)
        normalized = _normalized_data_checkpoint(existing_data, plan.projected_latest)
        _write_json_atomic(data_path, normalized)
        progress.done("Synchronizing data checkpoint", sync_start, detail=str(data_path))


def _plan_to_dict(plan: RollbackPlan) -> dict[str, Any]:
    return {
        "checkpoint_dir": str(plan.checkpoint_dir),
        "target_resume_epoch": plan.target_resume_epoch,
        "selected_checkpoint": str(plan.selected.path),
        "selected_kind": plan.selected.kind,
        "selected_resume_epoch": plan.selected.resume_epoch,
        "selected_resume_batch": plan.selected.resume_batch,
        "projected_latest_checkpoint": str(plan.projected_latest.path),
        "remove_count": len(plan.removable),
        "remove_markers_count": len(plan.removable_markers),
        "remove_checkpoints": [str(snapshot.path) for snapshot in plan.removable],
        "remove_markers": [str(marker) for marker in plan.removable_markers],
        "data_assessment": asdict(plan.data_assessment),
    }


def _print_human_plan(plan: RollbackPlan, *, apply: bool, sync_data_checkpoint: bool) -> None:
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[{mode}] rollback plan for {plan.checkpoint_dir}")
    print(
        "  target resume epoch: "
        f"{plan.target_resume_epoch} | selected: {plan.selected.path.name} "
        f"(kind={plan.selected.kind}, resume_batch={plan.selected.resume_batch})"
    )
    print(f"  projected latest after rollback: {plan.projected_latest.path.name}")
    print(f"  checkpoints to remove: {len(plan.removable)}")
    for snapshot in plan.removable:
        print(
            f"    - {snapshot.path.name} "
            f"(resume_epoch={snapshot.resume_epoch}, kind={snapshot.kind}, epoch={snapshot.epoch})"
        )
    if plan.removable_markers:
        print(f"  epoch markers to remove: {len(plan.removable_markers)}")
        for marker in plan.removable_markers:
            print(f"    - {marker.name}")
    print("  data checkpoint assessment: " f"{plan.data_assessment.status} — {plan.data_assessment.message}")
    if sync_data_checkpoint:
        print("  data checkpoint sync: enabled")
    else:
        print("  data checkpoint sync: disabled")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Step a df_mlx checkpoint directory back to a chosen resume epoch and "
            "validate --resume / --resume-data coherence."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoit-dir",
        type=Path,
        required=True,
        help="Directory containing .safetensors checkpoints and state JSON files",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--target-resume-epoch",
        "--target-resume",
        type=int,
        help="Target resume epoch index (0-based, internal train loop index)",
    )
    group.add_argument(
        "--target-epoch",
        type=int,
        help="Target epoch number (1-based human-friendly alias for resume epoch)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply rollback changes. Default is dry-run preview.",
    )
    parser.add_argument(
        "--no-sync-data-checkpoint",
        action="store_true",
        help="Do not rewrite/create data_checkpoint.json during --apply.",
    )
    parser.add_argument(
        "--require-resume-data",
        action="store_true",
        help="Return non-zero if data_checkpoint.json is missing or incoherent.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON plan/report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress updates.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Emit loop progress every N items for large checkpoint sets.",
    )

    args = parser.parse_args(argv)

    try:
        progress = ProgressReporter(enabled=not args.quiet, progress_every=max(args.progress_every, 1))
        mode = "APPLY" if args.apply else "DRY-RUN"
        progress.info(f"🚀 Starting rollback helper in {mode} mode")

        target_resume_epoch = _resolve_target_resume_epoch(
            target_resume_epoch=args.target_resume_epoch,
            target_epoch=args.target_epoch,
        )
        checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
        if not checkpoint_dir.exists():
            print(f"Error: checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
            return 2

        planning_start = progress.start("Building rollback plan")
        plan = build_rollback_plan(checkpoint_dir, target_resume_epoch, progress=progress)
        progress.done("Building rollback plan", planning_start)
        sync_data_checkpoint = not args.no_sync_data_checkpoint

        if args.json:
            print(json.dumps(_plan_to_dict(plan), indent=2))
        else:
            _print_human_plan(plan, apply=args.apply, sync_data_checkpoint=sync_data_checkpoint)

        if plan.data_assessment.status == "error" and not (args.apply and sync_data_checkpoint):
            print(
                "Validation failure: data checkpoint is incompatible with selected model checkpoint. "
                "Use --apply (with sync enabled) to normalize it.",
                file=sys.stderr,
            )
            return 1
        if (
            args.require_resume_data
            and plan.data_assessment.status == "missing"
            and not (args.apply and sync_data_checkpoint)
        ):
            print("Validation failure: data_checkpoint.json is required but missing.", file=sys.stderr)
            return 1

        if args.apply:
            apply_start = progress.start("Applying rollback changes")
            apply_rollback_plan(plan, sync_data_checkpoint=sync_data_checkpoint, progress=progress)
            progress.done("Applying rollback changes", apply_start)

            post_validate_start = progress.start("Validating post-rollback state")
            latest = find_latest_checkpoint(checkpoint_dir, prefer_completed=False)
            if latest is None:
                print("Error: no checkpoints remain after rollback", file=sys.stderr)
                return 2
            if latest != plan.projected_latest.path:
                print(
                    "Error: latest checkpoint after rollback does not match projection "
                    f"({latest.name} != {plan.projected_latest.path.name})",
                    file=sys.stderr,
                )
                return 2

            post_report = validate_checkpoint_dir(checkpoint_dir, strict=True, validate_load=False)
            if post_report["resume_epoch"] != target_resume_epoch:
                print(
                    "Error: post-rollback resume epoch mismatch "
                    f"({post_report['resume_epoch']} != {target_resume_epoch})",
                    file=sys.stderr,
                )
                return 2

            post_data_state = _load_data_checkpoint(checkpoint_dir / "data_checkpoint.json")
            post_assessment = _assess_data_checkpoint(post_data_state, plan.projected_latest)
            if args.require_resume_data and post_assessment.status == "missing":
                print("Validation failure: data_checkpoint.json is required but missing.", file=sys.stderr)
                return 1
            if post_assessment.status == "error":
                print(f"Validation failure: {post_assessment.message}", file=sys.stderr)
                return 1
            progress.done(
                "Validating post-rollback state",
                post_validate_start,
                detail=(
                    f"latest={latest.name}, resume_epoch={post_report['resume_epoch']}, "
                    f"resume_batch={post_report['resume_batch']}"
                ),
            )

            if args.json:
                print(
                    json.dumps(
                        {
                            "applied": True,
                            "latest_checkpoint": str(latest),
                            "resume_epoch": post_report["resume_epoch"],
                            "resume_batch": post_report["resume_batch"],
                            "data_assessment": asdict(post_assessment),
                        },
                        indent=2,
                    )
                )
            else:
                print("Rollback applied and validated.")

        progress.info("✅ Rollback helper completed")

        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Validation failure: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
