import pytest

from df_mlx.run_config import (
    SYNC_MODE_EVAL_FREQUENCY,
    RunConfig,
    apply_run_config_dict,
    resolve_run_config,
)


class TestSyncModeChoices:
    """sync_mode field accepts exactly (fast, normal, debug, profile)."""

    @pytest.mark.parametrize("mode", ["fast", "normal", "debug", "profile"])
    def test_valid_sync_modes(self, mode: str) -> None:
        cfg = RunConfig()
        apply_run_config_dict(cfg, {"debug": {"sync_mode": mode}})
        assert cfg.debug.sync_mode == mode

    def test_invalid_sync_mode_rejected(self) -> None:
        cfg = RunConfig()
        with pytest.raises(ValueError, match="Invalid value"):
            apply_run_config_dict(cfg, {"debug": {"sync_mode": "turbo"}})

    def test_default_is_normal(self) -> None:
        cfg = RunConfig()
        assert cfg.debug.sync_mode == "normal"


class TestEvalFrequencyOverride:
    """resolve_run_config overrides eval_frequency when at default."""

    def test_fast_overrides_eval_frequency(self) -> None:
        cfg = RunConfig()
        apply_run_config_dict(cfg, {"debug": {"sync_mode": "fast"}})
        resolve_run_config(cfg)
        assert cfg.training.eval_frequency == SYNC_MODE_EVAL_FREQUENCY["fast"]

    def test_debug_overrides_eval_frequency(self) -> None:
        cfg = RunConfig()
        apply_run_config_dict(cfg, {"debug": {"sync_mode": "debug"}})
        resolve_run_config(cfg)
        assert cfg.training.eval_frequency == 1

    def test_profile_overrides_eval_frequency(self) -> None:
        cfg = RunConfig()
        apply_run_config_dict(cfg, {"debug": {"sync_mode": "profile"}})
        resolve_run_config(cfg)
        assert cfg.training.eval_frequency == 5

    def test_normal_keeps_default(self) -> None:
        cfg = RunConfig()
        apply_run_config_dict(cfg, {"debug": {"sync_mode": "normal"}})
        resolve_run_config(cfg)
        assert cfg.training.eval_frequency == 10

    def test_explicit_eval_frequency_not_overridden(self) -> None:
        cfg = RunConfig()
        apply_run_config_dict(
            cfg,
            {
                "training": {"eval_frequency": 25},
                "debug": {"sync_mode": "fast"},
            },
        )
        resolve_run_config(cfg)
        assert cfg.training.eval_frequency == 25

    def test_explicit_eval_frequency_one_not_overridden(self) -> None:
        cfg = RunConfig()
        apply_run_config_dict(
            cfg,
            {
                "training": {"eval_frequency": 1},
                "debug": {"sync_mode": "fast"},
            },
        )
        resolve_run_config(cfg)
        assert cfg.training.eval_frequency == 1


class TestDebugNumericsInteraction:
    """debug_numerics forces eager regardless of sync_mode."""

    def test_debug_numerics_independent_of_sync_mode(self) -> None:
        cfg = RunConfig()
        apply_run_config_dict(
            cfg,
            {
                "debug": {"sync_mode": "fast", "debug_numerics": True},
            },
        )
        resolve_run_config(cfg)
        assert cfg.debug.debug_numerics is True
        assert cfg.debug.sync_mode == "fast"

    def test_nan_skip_batch_independent_of_sync_mode(self) -> None:
        cfg = RunConfig()
        apply_run_config_dict(
            cfg,
            {
                "debug": {"sync_mode": "fast", "nan_skip_batch": True},
            },
        )
        resolve_run_config(cfg)
        assert cfg.debug.nan_skip_batch is True
        assert cfg.debug.sync_mode == "fast"


class TestSyncModeEvalFrequencyMap:
    """SYNC_MODE_EVAL_FREQUENCY covers all valid modes."""

    def test_all_modes_have_mapping(self) -> None:
        expected = {"fast", "normal", "debug", "profile"}
        assert set(SYNC_MODE_EVAL_FREQUENCY.keys()) == expected

    def test_values_are_positive_ints(self) -> None:
        for mode, freq in SYNC_MODE_EVAL_FREQUENCY.items():
            assert isinstance(freq, int), f"{mode} frequency is not int"
            assert freq >= 1, f"{mode} frequency must be >= 1"
