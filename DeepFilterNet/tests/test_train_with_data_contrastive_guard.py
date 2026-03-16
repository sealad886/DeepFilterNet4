from pathlib import Path

import mlx.core as mx
import pytest

from df_mlx.train_with_data import _checkpoint_has_contrastive_projector, train


def _save_checkpoint(path: Path, weights: dict[str, mx.array]) -> None:
    mx.save_safetensors(str(path), weights)


def test_checkpoint_has_contrastive_projector_detects_projector_weights(tmp_path: Path) -> None:
    ckpt = tmp_path / "contrastive.safetensors"
    _save_checkpoint(
        ckpt,
        {
            "encoder.weight": mx.zeros((2, 2), dtype=mx.float32),
            "contrastive_projector.layers.0.weight": mx.zeros((4, 4), dtype=mx.float32),
        },
    )

    assert _checkpoint_has_contrastive_projector(ckpt) is True


def test_checkpoint_has_contrastive_projector_ignores_regular_checkpoints(tmp_path: Path) -> None:
    ckpt = tmp_path / "baseline.safetensors"
    _save_checkpoint(
        ckpt,
        {
            "encoder.weight": mx.zeros((2, 2), dtype=mx.float32),
            "decoder.weight": mx.zeros((2, 2), dtype=mx.float32),
        },
    )

    assert _checkpoint_has_contrastive_projector(ckpt) is False


def test_train_with_data_rejects_contrastive_resume_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "contrastive.safetensors"
    _save_checkpoint(
        ckpt,
        {
            "contrastive_projector.layers.0.weight": mx.zeros((4, 4), dtype=mx.float32),
        },
    )

    with pytest.raises(ValueError, match="Contrastive AWESOME checkpoints are only supported"):
        train(datastore_dir=str(tmp_path / "datastore"), resume_from=str(ckpt), epochs=1)
