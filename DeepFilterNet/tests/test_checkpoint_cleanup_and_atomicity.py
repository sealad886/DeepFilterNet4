"""Tests for checkpoint cleanup (including interrupted) and discriminator atomicity."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

from df_mlx.training_checkpoints import (  # noqa: E402
    CheckpointManifest,
    cleanup_checkpoints,
    save_checkpoint,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def __call__(self, x):
        return self.lin(x)


def _save_ckpt(tmpdir: Path, name: str, *, epoch: int = 0, kind: str = "epoch_end") -> Path:
    model = TinyModel()
    optimizer = optim.AdamW(learning_rate=0.001)
    path = tmpdir / name
    save_checkpoint(
        model,
        path,
        epoch=epoch,
        batch_idx=0,
        global_step=epoch * 10,
        loss=0.1,
        best_valid_loss=0.2,
        config={},
        optimizer=optimizer,
        last_completed_epoch=epoch,
        kind=kind,
    )
    return path


class TestCleanupIncludesInterrupted:
    """cleanup_checkpoints must remove interrupted_epoch_* files when over limit."""

    def test_interrupted_checkpoints_cleaned_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            # Create 3 epoch checkpoints and 2 interrupted checkpoints
            import time

            paths = []
            for i in range(3):
                p = _save_ckpt(tmpdir, f"epoch_{i + 1:03d}.safetensors", epoch=i, kind="epoch_end")
                paths.append(p)
                time.sleep(0.05)  # Ensure distinct mtimes

            for i in range(3, 5):
                p = _save_ckpt(
                    tmpdir,
                    f"interrupted_epoch_{i + 1:03d}.safetensors",
                    epoch=i,
                    kind="interrupted",
                )
                paths.append(p)
                time.sleep(0.05)

            # Keep only 2 — should remove the 3 oldest
            cleanup_checkpoints(tmpdir, save_total_limit=2)

            remaining = sorted(p.name for p in tmpdir.glob("*.safetensors"))

            # Should keep only the 2 newest (the interrupted ones)
            assert len([f for f in remaining if not f.endswith(".state.json")]) == 2
            # Make sure interrupted files can be part of what remains
            assert any("interrupted" in f for f in remaining)

    def test_interrupted_state_json_also_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            _save_ckpt(
                tmpdir,
                "interrupted_epoch_001.safetensors",
                epoch=0,
                kind="interrupted",
            )
            import time

            time.sleep(0.05)
            _save_ckpt(tmpdir, "epoch_002.safetensors", epoch=1, kind="epoch_end")

            # Keep 1 — should remove the interrupted one
            cleanup_checkpoints(tmpdir, save_total_limit=1)

            assert not (tmpdir / "interrupted_epoch_001.safetensors").exists()
            manifest = CheckpointManifest()
            state_path = manifest.state_path(tmpdir / "interrupted_epoch_001.safetensors")
            assert not state_path.exists()
            assert (tmpdir / "epoch_002.safetensors").exists()


class TestDiscriminatorAtomicity:
    """Discriminator weights should be written to temp before main rename."""

    def test_disc_weights_saved_alongside_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            model = TinyModel()
            disc = TinyModel()
            optimizer = optim.AdamW(learning_rate=0.001)
            disc_optimizer = optim.AdamW(learning_rate=0.001)

            path = tmpdir / "epoch_001.safetensors"
            saved = save_checkpoint(
                model,
                path,
                epoch=0,
                batch_idx=10,
                global_step=100,
                loss=0.5,
                best_valid_loss=0.4,
                config={},
                optimizer=optimizer,
                discriminator=disc,
                disc_optimizer=disc_optimizer,
                last_completed_epoch=0,
                kind="epoch_end",
            )
            assert saved

            disc_path = path.with_name(f"{path.stem}.disc{path.suffix}")
            assert disc_path.exists(), "Discriminator weights file should exist"
            assert disc_path.stat().st_size > 0

            # No temp files should remain
            tmp_files = list(tmpdir.glob("*.tmp*"))
            assert tmp_files == [], f"Temp files should be cleaned up: {tmp_files}"

    def test_no_disc_when_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            model = TinyModel()
            optimizer = optim.AdamW(learning_rate=0.001)

            path = tmpdir / "epoch_001.safetensors"
            saved = save_checkpoint(
                model,
                path,
                epoch=0,
                batch_idx=10,
                global_step=100,
                loss=0.5,
                best_valid_loss=0.4,
                config={},
                optimizer=optimizer,
                last_completed_epoch=0,
                kind="epoch_end",
            )
            assert saved

            disc_path = path.with_name(f"{path.stem}.disc{path.suffix}")
            assert not disc_path.exists(), "No disc weights when discriminator is None"
