import json
import sys
import tempfile
from pathlib import Path

import mlx.nn as nn
import mlx.optimizers as optim
import pytest

# Ensure the df_mlx package is importable when running tests from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from df_mlx.train_dynamic import compute_resume_epoch, save_checkpoint, validate_checkpoint_dir  # noqa: E402


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
):
    model = TinyModel()
    optimizer = optim.AdamW(learning_rate=0.001)
    if kind == "step":
        global_step = global_step if global_step is not None else epoch * 10 + 1
        ckpt_path = tmpdir / f"step_{global_step:06d}.safetensors"
    elif kind == "epoch_end":
        ckpt_path = tmpdir / f"epoch_{epoch + 1:03d}.safetensors"
    elif kind == "interrupted":
        ckpt_path = tmpdir / f"interrupted_epoch_{epoch + 1:03d}.safetensors"
    elif kind == "best":
        ckpt_path = tmpdir / "best.safetensors"
    elif kind == "final":
        ckpt_path = tmpdir / "final.safetensors"
    else:
        ckpt_path = tmpdir / f"{kind}.safetensors"
    save_checkpoint(
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
        kind=kind,
    )
    return ckpt_path


def test_resume_in_progress_epoch_not_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        ckpt_path = _make_checkpoint(
            tmpdir,
            epoch=3,
            kind="step",
            last_completed=2,
            batch_idx=5,
            global_step=301,
        )

        report = validate_checkpoint_dir(tmpdir, strict=True, validate_load=False)
        assert report["valid"] == 1
        assert report["latest_path"] == ckpt_path
        assert report["last_completed_epoch"] == 2
        assert report["resume_epoch"] == 3  # resume same epoch, not epoch+1

        state_path = ckpt_path.with_suffix(".state.json")
        state = json.loads(state_path.read_text())
        assert compute_resume_epoch(state) == 3


def test_resume_after_completed_epoch_advances():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        ckpt_path = _make_checkpoint(tmpdir, epoch=4, kind="epoch_end", last_completed=4, global_step=401)

        report = validate_checkpoint_dir(tmpdir, strict=True, validate_load=False)
        assert report["valid"] == 1
        assert report["latest_path"] == ckpt_path
        assert report["last_completed_epoch"] == 4
        assert report["resume_epoch"] == 5  # completed epoch advances by one


def test_resume_prefers_interrupted_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_checkpoint(tmpdir, epoch=2, kind="epoch_end", last_completed=2, global_step=201)
        interrupted = _make_checkpoint(
            tmpdir,
            epoch=3,
            kind="interrupted",
            last_completed=2,
            batch_idx=7,
            global_step=305,
        )

        report = validate_checkpoint_dir(tmpdir, strict=True, validate_load=False)
        assert report["latest_path"] == interrupted
        assert report["resume_epoch"] == 3


def test_corrupted_checkpoint_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # Create an empty weights file without state
        bad = tmpdir / "epoch_001.safetensors"
        bad.write_bytes(b"")
        with pytest.raises(RuntimeError):
            validate_checkpoint_dir(tmpdir, strict=True, validate_load=False)
