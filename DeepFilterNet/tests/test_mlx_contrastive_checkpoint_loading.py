from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

# Ensure the df_mlx package is importable when running tests from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from df_mlx.checkpoint import load_model  # noqa: E402
from df_mlx.model import init_model  # noqa: E402
from df_mlx.train import load_checkpoint  # noqa: E402


def _save_model_checkpoint(path: Path, model) -> tuple[str, mx.array]:
    weights = {name: value for name, value in tree_flatten(model.parameters())}
    mx.eval(*weights.values())
    mx.save_safetensors(str(path), weights)
    probe_name = next(name for name in weights if not str(name).startswith("contrastive_projector."))
    return probe_name, weights[probe_name]


def test_train_load_checkpoint_ignores_train_only_contrastive_projector_weights(tmp_path: Path) -> None:
    contrastive_model = init_model(contrastive_hidden_dim=256, contrastive_embedding_dim=128)
    checkpoint_path = tmp_path / "contrastive.safetensors"
    probe_name, probe_value = _save_model_checkpoint(checkpoint_path, contrastive_model)

    inference_model = init_model()
    state = load_checkpoint(inference_model, checkpoint_path)

    loaded_weights = dict(tree_flatten(inference_model.parameters()))
    assert state == {}
    assert inference_model.has_contrastive_projector is False
    assert np.allclose(np.array(loaded_weights[probe_name]), np.array(probe_value))


def test_checkpoint_load_model_ignores_train_only_contrastive_projector_weights(tmp_path: Path) -> None:
    contrastive_model = init_model(contrastive_hidden_dim=256, contrastive_embedding_dim=128)
    checkpoint_path = tmp_path / "contrastive.safetensors"
    probe_name, probe_value = _save_model_checkpoint(checkpoint_path, contrastive_model)

    inference_model = init_model()
    load_model(inference_model, checkpoint_path)

    loaded_weights = dict(tree_flatten(inference_model.parameters()))
    assert inference_model.has_contrastive_projector is False
    assert np.allclose(np.array(loaded_weights[probe_name]), np.array(probe_value))


def test_train_load_checkpoint_still_rejects_unrelated_unexpected_weights(tmp_path: Path) -> None:
    model = init_model()
    checkpoint_path = tmp_path / "bad_extra.safetensors"
    weights = {name: value for name, value in tree_flatten(model.parameters())}
    weights["unexpected.weight"] = mx.zeros((1,), dtype=mx.float32)
    mx.eval(*weights.values())
    mx.save_safetensors(str(checkpoint_path), weights)

    with pytest.raises(ValueError, match="Received .* not in model|unexpected.weight"):
        load_checkpoint(init_model(), checkpoint_path)
