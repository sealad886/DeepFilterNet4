"""Tests for the TrainingSession class-based training API."""

from __future__ import annotations

import pytest


class TestTrainingSessionInit:
    """TrainingSession.__init__ stores kwargs and validates names."""

    def test_accepts_valid_kwargs(self):
        from df_mlx.training_session import TrainingSession

        session = TrainingSession(epochs=50, batch_size=16)
        assert session.kwargs["epochs"] == 50
        assert session.kwargs["batch_size"] == 16

    def test_rejects_unknown_kwargs(self):
        from df_mlx.training_session import TrainingSession

        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            TrainingSession(not_a_real_param=42)

    def test_empty_is_valid(self):
        from df_mlx.training_session import TrainingSession

        session = TrainingSession()
        assert session.kwargs == {}

    def test_kwargs_returns_copy(self):
        from df_mlx.training_session import TrainingSession

        session = TrainingSession(epochs=10)
        kw = session.kwargs
        kw["epochs"] = 999
        assert session.kwargs["epochs"] == 10


class TestTrainingSessionSetup:
    """TrainingSession.setup sets _ready flag."""

    def test_setup_sets_ready(self):
        from df_mlx.training_session import TrainingSession

        session = TrainingSession()
        assert not session._ready
        session.setup()
        assert session._ready


class TestTrainingSessionSetupInit:
    """TrainingSession.setup() creates model, optimizer, and hw_config."""

    def test_setup_creates_model(self):
        from df_mlx.training_session import TrainingSession

        session = TrainingSession(backbone_type="mamba", model_variant="full")
        session.setup()
        assert session.model is not None
        assert hasattr(session.model, "parameters")

    def test_setup_creates_optimizer(self):
        from df_mlx.training_session import TrainingSession

        session = TrainingSession(learning_rate=3e-4, weight_decay=0.01)
        session.setup()
        assert session.optimizer is not None

    def test_setup_detects_hardware(self):
        from df_mlx.training_session import TrainingSession

        session = TrainingSession()
        session.setup()
        assert session.hw_config is not None

    def test_setup_counts_parameters(self):
        from df_mlx.training_session import TrainingSession

        session = TrainingSession(backbone_type="mamba", model_variant="full")
        session.setup()
        assert session.param_count > 0

    def test_setup_stores_model_config(self):
        from df_mlx.training_session import TrainingSession

        session = TrainingSession(backbone_type="gru")
        session.setup()
        assert session.model_config is not None
        assert session.model_config.backbone.backbone_type == "gru"

    def test_setup_lite_variant(self):
        from df_mlx.training_session import TrainingSession

        full_session = TrainingSession(model_variant="full")
        full_session.setup()

        lite_session = TrainingSession(model_variant="lite")
        lite_session.setup()

        assert lite_session.param_count < full_session.param_count

    def test_setup_contrastive_awesome_adds_projector(self):
        from df_mlx.training_session import TrainingSession

        base_session = TrainingSession(dynamic_loss="baseline")
        base_session.setup()

        contrastive_session = TrainingSession(dynamic_loss="contrastive_awesome")
        contrastive_session.setup()

        assert contrastive_session.param_count > base_session.param_count

    def test_setup_idempotent(self):
        from df_mlx.training_session import TrainingSession

        session = TrainingSession()
        session.setup()
        model_id = id(session.model)
        session.setup()
        assert id(session.model) != model_id  # re-creates each time


class TestTrainingSessionFromRunConfig:
    """TrainingSession.from_run_config extracts kwargs from RunConfig."""

    def test_creates_session_from_default_run_config(self):
        from df_mlx.run_config import RunConfig
        from df_mlx.training_session import TrainingSession

        cfg = RunConfig()
        session = TrainingSession.from_run_config(cfg)
        assert session.kwargs["epochs"] == cfg.training.epochs
        assert session.kwargs["batch_size"] == cfg.training.batch_size
        assert session.kwargs["learning_rate"] == cfg.training.learning_rate
        assert session.kwargs["checkpoint_dir"] == cfg.checkpoint.checkpoint_dir

    def test_overrides_applied_on_top(self):
        from df_mlx.run_config import RunConfig
        from df_mlx.training_session import TrainingSession

        cfg = RunConfig()
        session = TrainingSession.from_run_config(cfg, epochs=999, batch_size=64)
        assert session.kwargs["epochs"] == 999
        assert session.kwargs["batch_size"] == 64

    def test_all_train_kwargs_present(self):
        from df_mlx.run_config import RunConfig
        from df_mlx.training_session import _TRAIN_KWARGS, TrainingSession

        cfg = RunConfig()
        session = TrainingSession.from_run_config(cfg)
        missing = set(_TRAIN_KWARGS) - set(session.kwargs)
        assert missing == set(), f"Missing kwargs from from_run_config: {sorted(missing)}"


class TestKwargsFromRunConfig:
    """_kwargs_from_run_config produces a complete mapping."""

    def test_all_keys_are_valid_train_kwargs(self):
        from df_mlx.run_config import RunConfig
        from df_mlx.training_session import _TRAIN_KWARGS, _kwargs_from_run_config

        cfg = RunConfig()
        result = _kwargs_from_run_config(cfg)
        extra = set(result) - set(_TRAIN_KWARGS)
        assert extra == set(), f"Keys not in _TRAIN_KWARGS: {sorted(extra)}"

    def test_no_train_kwarg_missing(self):
        from df_mlx.run_config import RunConfig
        from df_mlx.training_session import _TRAIN_KWARGS, _kwargs_from_run_config

        cfg = RunConfig()
        result = _kwargs_from_run_config(cfg)
        missing = set(_TRAIN_KWARGS) - set(result)
        assert missing == set(), f"Missing keys: {sorted(missing)}"


class TestTrainKwargsRegistry:
    """_TRAIN_KWARGS must match the actual train() signature."""

    def test_train_kwargs_matches_signature(self):
        import inspect

        from df_mlx.train_dynamic import train
        from df_mlx.training_session import _TRAIN_KWARGS

        sig = inspect.signature(train)
        sig_params = set(sig.parameters.keys())
        registry = set(_TRAIN_KWARGS)
        assert registry == sig_params, (
            f"Mismatch between _TRAIN_KWARGS and train() signature.\n"
            f"  Extra in registry: {sorted(registry - sig_params)}\n"
            f"  Missing from registry: {sorted(sig_params - registry)}"
        )
