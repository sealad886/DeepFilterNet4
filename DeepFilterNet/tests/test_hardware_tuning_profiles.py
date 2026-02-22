"""Tests for hardware tuning profiles and auto-tune dataloader logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from df_mlx.run_config import (
    HARDWARE_PROFILES,
    DataloaderConfig,
    RunConfig,
    apply_run_config_dict,
    detect_hardware_class,
    get_hardware_tuning_profile,
    resolve_run_config,
)

# ---------------------------------------------------------------------------
# Profile completeness
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"loader", "num_workers", "prefetch_size"}
EXPECTED_CLASSES = {"entry", "pro", "max", "ultra"}


def test_all_hardware_classes_have_profiles():
    assert set(HARDWARE_PROFILES.keys()) == EXPECTED_CLASSES


@pytest.mark.parametrize("hw_class", sorted(EXPECTED_CLASSES))
def test_profile_has_required_keys(hw_class: str):
    profile = HARDWARE_PROFILES[hw_class]
    assert REQUIRED_KEYS.issubset(profile.keys())


@pytest.mark.parametrize("hw_class", sorted(EXPECTED_CLASSES))
def test_profile_values_are_positive(hw_class: str):
    profile = HARDWARE_PROFILES[hw_class]
    assert profile["num_workers"] > 0
    assert profile["prefetch_size"] > 0
    assert profile["loader"] in ("prefetch", "mlx_data")


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "brand_string, expected_class",
    [
        ("Apple M1", "entry"),
        ("Apple M2", "entry"),
        ("Apple M3", "entry"),
        ("Apple M4", "entry"),
        ("Apple M1 Pro", "pro"),
        ("Apple M2 Pro", "pro"),
        ("Apple M3 Pro", "pro"),
        ("Apple M1 Max", "max"),
        ("Apple M2 Max", "max"),
        ("Apple M3 Max", "max"),
        ("Apple M1 Ultra", "ultra"),
        ("Apple M2 Ultra", "ultra"),
    ],
)
def test_detect_hardware_class_known_chips(brand_string: str, expected_class: str):
    import subprocess as _subprocess

    with patch("df_mlx.run_config.subprocess") as mock_sub:
        mock_sub.check_output.return_value = brand_string.encode()
        mock_sub.CalledProcessError = _subprocess.CalledProcessError
        mock_sub.DEVNULL = _subprocess.DEVNULL
        result = detect_hardware_class()
    assert result == expected_class


def test_detect_hardware_class_unknown_chip():
    import subprocess as _subprocess

    with patch("df_mlx.run_config.subprocess") as mock_sub:
        mock_sub.check_output.return_value = b"Intel Core i9-13900K"
        mock_sub.CalledProcessError = _subprocess.CalledProcessError
        mock_sub.DEVNULL = _subprocess.DEVNULL
        result = detect_hardware_class()
    assert result == "unknown"


def test_detect_hardware_class_command_fails():
    import subprocess as _subprocess

    with patch("df_mlx.run_config.subprocess") as mock_sub:
        mock_sub.CalledProcessError = _subprocess.CalledProcessError
        mock_sub.check_output.side_effect = _subprocess.CalledProcessError(1, "sysctl")
        result = detect_hardware_class()
    assert result == "unknown"


def test_detect_hardware_class_missing_sysctl():
    import subprocess as _subprocess

    with patch("df_mlx.run_config.subprocess") as mock_sub:
        mock_sub.check_output.side_effect = FileNotFoundError("sysctl")
        mock_sub.CalledProcessError = _subprocess.CalledProcessError
        mock_sub.DEVNULL = _subprocess.DEVNULL
        result = detect_hardware_class()
    assert result == "unknown"


# ---------------------------------------------------------------------------
# get_hardware_tuning_profile
# ---------------------------------------------------------------------------


def test_profile_recommendation_returns_expected_keys():
    with patch("df_mlx.run_config.detect_hardware_class", return_value="pro"):
        profile = get_hardware_tuning_profile()
    assert REQUIRED_KEYS.issubset(profile.keys())
    assert "hardware_class" in profile
    assert profile["hardware_class"] == "pro"


def test_profile_recommendation_unknown_falls_back():
    with patch("df_mlx.run_config.detect_hardware_class", return_value="unknown"):
        profile = get_hardware_tuning_profile()
    assert profile["num_workers"] == 2
    assert profile["prefetch_size"] == 4
    assert profile["loader"] == "prefetch"
    assert profile["hardware_class"] == "unknown"


# ---------------------------------------------------------------------------
# auto_tune_dataloader integration with resolve_run_config
# ---------------------------------------------------------------------------


def test_auto_tune_applies_profile_when_no_explicit_values():
    cfg = RunConfig()
    apply_run_config_dict(cfg, {"dataloader": {"auto_tune_dataloader": True}})

    with patch("df_mlx.run_config.detect_hardware_class", return_value="max"):
        resolve_run_config(cfg)

    assert cfg.dataloader.num_workers == 6
    assert cfg.dataloader.prefetch_size == 12
    assert cfg.dataloader.use_mlx_data is True


def test_auto_tune_does_not_override_explicit_num_workers():
    cfg = RunConfig()
    apply_run_config_dict(
        cfg,
        {"dataloader": {"auto_tune_dataloader": True, "num_workers": 3}},
    )

    with patch("df_mlx.run_config.detect_hardware_class", return_value="ultra"):
        resolve_run_config(cfg)

    assert cfg.dataloader.num_workers == 3
    assert cfg.dataloader.prefetch_size == 16
    assert cfg.dataloader.use_mlx_data is True


def test_auto_tune_does_not_override_explicit_prefetch_size():
    cfg = RunConfig()
    apply_run_config_dict(
        cfg,
        {"dataloader": {"auto_tune_dataloader": True, "prefetch_size": 5}},
    )

    with patch("df_mlx.run_config.detect_hardware_class", return_value="ultra"):
        resolve_run_config(cfg)

    assert cfg.dataloader.num_workers == 8
    assert cfg.dataloader.prefetch_size == 5


def test_auto_tune_does_not_override_explicit_use_mlx_data():
    cfg = RunConfig()
    apply_run_config_dict(
        cfg,
        {"dataloader": {"auto_tune_dataloader": True, "use_mlx_data": False}},
    )

    with patch("df_mlx.run_config.detect_hardware_class", return_value="max"):
        resolve_run_config(cfg)

    assert cfg.dataloader.use_mlx_data is False
    assert cfg.dataloader.num_workers == 6


def test_auto_tune_not_applied_when_disabled():
    cfg = RunConfig()
    apply_run_config_dict(cfg, {"dataloader": {"auto_tune_dataloader": False}})
    resolve_run_config(cfg)

    defaults = DataloaderConfig()
    assert cfg.dataloader.num_workers == defaults.num_workers
    assert cfg.dataloader.prefetch_size == defaults.prefetch_size


def test_auto_tune_not_applied_by_default():
    cfg = RunConfig()
    resolve_run_config(cfg)

    defaults = DataloaderConfig()
    assert cfg.dataloader.num_workers == defaults.num_workers
    assert cfg.dataloader.prefetch_size == defaults.prefetch_size
