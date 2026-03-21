"""Tests for scripts/rollback_checkpoint_epoch.py."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.nn as nn
import mlx.optimizers as optim

# Import script module directly from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from rollback_checkpoint_epoch import main  # type: ignore[import-not-found]  # noqa: E402

from df_mlx.training_checkpoints import find_latest_checkpoint, save_checkpoint, validate_checkpoint_dir  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def __call__(self, x):
        return self.lin(x)


def _make_checkpoint(
    tmpdir: Path,
    *,
    epoch: int,
    kind: str,
    last_completed: int,
    batch_idx: int | None = None,
    global_step: int | None = None,
    pipeline_stage_index: int | None = None,
    pipeline_stage_name: str | None = None,
) -> Path:
    model = TinyModel()
    optimizer = optim.AdamW(learning_rate=0.001)
    if kind == "step":
        global_step = global_step if global_step is not None else epoch * 100 + 1
        ckpt_path = tmpdir / f"step_{global_step:06d}.safetensors"
    elif kind == "epoch_end":
        ckpt_path = tmpdir / f"epoch_{epoch + 1:03d}.safetensors"
    elif kind == "interrupted":
        ckpt_path = tmpdir / f"interrupted_epoch_{epoch + 1:03d}.safetensors"
    elif kind in {"best", "best_final"}:
        ckpt_path = tmpdir / "best.safetensors"
    elif kind == "final":
        ckpt_path = tmpdir / "final.safetensors"
    else:
        ckpt_path = tmpdir / f"{kind}.safetensors"

    ok = save_checkpoint(
        model,
        ckpt_path,
        epoch=epoch,
        batch_idx=batch_idx,
        global_step=global_step,
        loss=0.1,
        best_valid_loss=0.2,
        config={},
        optimizer=optimizer,
        last_completed_epoch=last_completed,
        pipeline_stage_index=pipeline_stage_index,
        pipeline_stage_name=pipeline_stage_name,
        kind=kind,
    )
    assert ok
    return ckpt_path


def test_target_resume_epoch_missing_returns_error(tmp_path: Path):
    _make_checkpoint(tmp_path, epoch=3, kind="epoch_end", last_completed=3, global_step=301)

    rc = main(
        [
            "--checkpoint-dir",
            str(tmp_path),
            "--target-resume-epoch",
            "2",
        ]
    )

    assert rc == 1


def test_apply_rolls_back_newer_checkpoints_and_syncs_data_checkpoint(tmp_path: Path):
    _make_checkpoint(tmp_path, epoch=3, kind="epoch_end", last_completed=3, global_step=301)
    time.sleep(0.02)
    target_step = _make_checkpoint(
        tmp_path,
        epoch=4,
        kind="step",
        last_completed=3,
        batch_idx=5,
        global_step=406,
        pipeline_stage_index=2,
        pipeline_stage_name="focus",
    )
    time.sleep(0.02)
    newer = _make_checkpoint(
        tmp_path,
        epoch=4,
        kind="epoch_end",
        last_completed=4,
        global_step=499,
        pipeline_stage_index=2,
        pipeline_stage_name="focus",
    )

    (tmp_path / "data_checkpoint.json").write_text(
        json.dumps(
            {
                "epoch": 5,
                "batch_idx": 0,
                "samples_processed": 0,
                "seed": 42,
                "split": "train",
                "pipeline_stage_index": 3,
                "pipeline_stage_name": "too_far",
            }
        )
    )

    rc = main(
        [
            "--checkpoint-dir",
            str(tmp_path),
            "--target-resume-epoch",
            "4",
            "--apply",
        ]
    )
    assert rc == 0
    assert not newer.exists()

    latest = find_latest_checkpoint(tmp_path, prefer_completed=False)
    assert latest == target_step

    report = validate_checkpoint_dir(tmp_path, strict=True, validate_load=False)
    assert report["resume_epoch"] == 4

    data_state = json.loads((tmp_path / "data_checkpoint.json").read_text())
    assert data_state["epoch"] == 4
    assert data_state["batch_idx"] == 5
    assert data_state["pipeline_stage_index"] == 2
    assert data_state["pipeline_stage_name"] == "focus"


def test_dry_run_does_not_remove_files(tmp_path: Path):
    _make_checkpoint(tmp_path, epoch=3, kind="epoch_end", last_completed=3, global_step=301)
    time.sleep(0.02)
    _make_checkpoint(tmp_path, epoch=4, kind="step", last_completed=3, batch_idx=2, global_step=402)
    time.sleep(0.02)
    newer = _make_checkpoint(tmp_path, epoch=4, kind="epoch_end", last_completed=4, global_step=480)

    rc = main(
        [
            "--checkpoint-dir",
            str(tmp_path),
            "--target-resume-epoch",
            "4",
        ]
    )
    assert rc == 0
    assert newer.exists(), "dry-run must not mutate checkpoint files"


def test_in_progress_data_mismatch_fails_without_apply(tmp_path: Path):
    _make_checkpoint(
        tmp_path,
        epoch=4,
        kind="step",
        last_completed=3,
        batch_idx=5,
        global_step=405,
        pipeline_stage_index=1,
        pipeline_stage_name="s1",
    )
    (tmp_path / "data_checkpoint.json").write_text(
        json.dumps(
            {
                "epoch": 4,
                "batch_idx": 50,
                "samples_processed": 100,
                "seed": 42,
                "split": "train",
                "pipeline_stage_index": 1,
                "pipeline_stage_name": "s1",
            }
        )
    )

    rc = main(
        [
            "--checkpoint-dir",
            str(tmp_path),
            "--target-resume-epoch",
            "4",
        ]
    )
    assert rc == 1


def test_require_resume_data_fails_when_missing(tmp_path: Path):
    _make_checkpoint(tmp_path, epoch=4, kind="step", last_completed=3, batch_idx=3, global_step=403)

    rc = main(
        [
            "--checkpoint-dir",
            str(tmp_path),
            "--target-resume-epoch",
            "4",
            "--require-resume-data",
        ]
    )
    assert rc == 1


def test_apply_creates_data_checkpoint_when_missing(tmp_path: Path):
    _make_checkpoint(
        tmp_path,
        epoch=4,
        kind="step",
        last_completed=3,
        batch_idx=7,
        global_step=407,
        pipeline_stage_index=1,
        pipeline_stage_name="stage_1",
    )

    data_path = tmp_path / "data_checkpoint.json"
    assert not data_path.exists()

    rc = main(
        [
            "--checkpoint-dir",
            str(tmp_path),
            "--target-resume-epoch",
            "4",
            "--apply",
            "--require-resume-data",
        ]
    )
    assert rc == 0
    assert data_path.exists()

    payload = json.loads(data_path.read_text())
    assert payload["epoch"] == 4
    assert payload["batch_idx"] == 7
    assert payload["pipeline_stage_index"] == 1
    assert payload["pipeline_stage_name"] == "stage_1"


def test_progress_feedback_emits_stage_updates(tmp_path: Path, capsys):
    _make_checkpoint(tmp_path, epoch=3, kind="epoch_end", last_completed=3, global_step=301)
    _make_checkpoint(tmp_path, epoch=4, kind="step", last_completed=3, batch_idx=2, global_step=402)

    rc = main(
        [
            "--checkpoint-dir",
            str(tmp_path),
            "--target-resume-epoch",
            "4",
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    assert "Starting rollback helper" in captured.err
    assert "Validating checkpoint directory" in captured.err
    assert "Loading checkpoint metadata" in captured.err
    assert "Rollback planning complete" in captured.err


def test_cli_aliases_are_supported(tmp_path: Path):
    _make_checkpoint(tmp_path, epoch=4, kind="step", last_completed=3, batch_idx=1, global_step=401)

    rc = main(
        [
            "--checkpoit-dir",
            str(tmp_path),
            "--target-resume",
            "4",
            "--quiet",
        ]
    )
    assert rc == 0
