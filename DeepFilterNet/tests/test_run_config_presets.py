"""Tests for run-config preset loading and --preset CLI integration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from df_mlx.run_config import PRESET_NAMES, RunConfig, apply_run_config_dict, load_preset_config, load_run_config

PRESETS_DIR = Path(__file__).resolve().parent.parent.parent / "schemas" / "presets"


# ------------------------------------------------------------------
# Preset TOML parsing
# ------------------------------------------------------------------


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_toml_parses(name: str) -> None:
    """Each preset TOML file must be valid TOML."""
    path = PRESETS_DIR / f"{name}.toml"
    assert path.exists(), f"Missing preset file: {path}"
    with open(path, "rb") as f:
        data = tomllib.load(f)
    assert isinstance(data, dict)


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_has_required_keys(name: str) -> None:
    """Every preset must define the core tuning knobs."""
    path = PRESETS_DIR / f"{name}.toml"
    with open(path, "rb") as f:
        data = tomllib.load(f)

    assert "training" in data, f"Preset '{name}' missing [training]"
    assert "batch_size" in data["training"], f"Preset '{name}' missing training.batch_size"
    assert "eval_frequency" in data["training"], f"Preset '{name}' missing training.eval_frequency"

    assert "dataloader" in data, f"Preset '{name}' missing [dataloader]"
    assert "num_workers" in data["dataloader"], f"Preset '{name}' missing dataloader.num_workers"
    assert "prefetch_size" in data["dataloader"], f"Preset '{name}' missing dataloader.prefetch_size"
    assert "use_mlx_data" in data["dataloader"], f"Preset '{name}' missing dataloader.use_mlx_data"

    assert "debug" in data, f"Preset '{name}' missing [debug]"
    assert "sync_mode" in data["debug"], f"Preset '{name}' missing debug.sync_mode"


# ------------------------------------------------------------------
# load_preset_config
# ------------------------------------------------------------------


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_load_preset_config_returns_run_config(name: str) -> None:
    """load_preset_config must produce a valid RunConfig for each preset."""
    cfg = load_preset_config(name)
    assert isinstance(cfg, RunConfig)


def test_load_preset_config_unknown_name() -> None:
    """Unknown preset name must raise ValueError with available list."""
    with pytest.raises(ValueError, match="Unknown preset"):
        load_preset_config("nonexistent")


def test_load_preset_values_entry() -> None:
    """Verify specific values for the entry preset."""
    cfg = load_preset_config("entry")
    assert cfg.training.batch_size == 2
    assert cfg.training.eval_frequency == 10
    assert cfg.dataloader.num_workers == 2
    assert cfg.dataloader.prefetch_size == 4
    assert cfg.dataloader.use_mlx_data is False
    assert cfg.debug.sync_mode == "normal"


def test_load_preset_values_max() -> None:
    """Verify specific values for the max preset."""
    cfg = load_preset_config("max")
    assert cfg.training.batch_size == 8
    assert cfg.training.eval_frequency == 25
    assert cfg.dataloader.num_workers == 6
    assert cfg.dataloader.prefetch_size == 12
    assert cfg.dataloader.use_mlx_data is True
    assert cfg.debug.sync_mode == "fast"


def test_load_preset_values_debug() -> None:
    """Verify specific values for the debug preset."""
    cfg = load_preset_config("debug")
    assert cfg.training.batch_size == 2
    assert cfg.training.eval_frequency == 1
    assert cfg.training.fp16 is False
    assert cfg.dataloader.use_mlx_data is False
    assert cfg.debug.sync_mode == "debug"
    assert cfg.debug.verbose is True


# ------------------------------------------------------------------
# Preset + run-config overlay
# ------------------------------------------------------------------


def test_run_config_overrides_preset(tmp_path: Path) -> None:
    """A run-config TOML must override preset values."""
    override_toml = tmp_path / "override.toml"
    override_toml.write_text(
        "[training]\nbatch_size = 16\n[dataloader]\nnum_workers = 10\n",
        encoding="utf-8",
    )

    cfg = load_preset_config("entry")
    assert cfg.training.batch_size == 2  # preset value
    cfg = load_run_config(override_toml, base=cfg)
    assert cfg.training.batch_size == 16  # overridden
    assert cfg.dataloader.num_workers == 10  # overridden
    # Non-overridden values stay from preset
    assert cfg.dataloader.prefetch_size == 4
    assert cfg.debug.sync_mode == "normal"


def test_cli_style_overrides_on_preset() -> None:
    """Simulated CLI override via apply_run_config_dict on top of preset."""
    cfg = load_preset_config("pro")
    assert cfg.training.batch_size == 4

    apply_run_config_dict(cfg, {"training": {"batch_size": 12}})
    assert cfg.training.batch_size == 12
    # Other preset values preserved
    assert cfg.dataloader.num_workers == 4


# ------------------------------------------------------------------
# Preset names constant
# ------------------------------------------------------------------


def test_preset_names_match_toml_files() -> None:
    """PRESET_NAMES must match available TOML files in schemas/presets/."""
    toml_files = sorted(p.stem for p in PRESETS_DIR.glob("*.toml"))
    assert sorted(PRESET_NAMES) == toml_files


# ------------------------------------------------------------------
# CLI integration (subprocess)
# ------------------------------------------------------------------


def test_preset_flag_in_help() -> None:
    """--preset must appear in train_dynamic.py --help output."""
    result = subprocess.run(
        [sys.executable, "-m", "df_mlx.train_dynamic", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--preset" in result.stdout
    assert "entry" in result.stdout
    assert "pro" in result.stdout
