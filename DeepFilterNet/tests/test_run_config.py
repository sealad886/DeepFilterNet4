try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from pathlib import Path

import pytest

from df_mlx.run_config import (
    RunConfig,
    apply_run_config_dict,
    generate_run_config_example,
    load_run_config,
    set_by_path,
)
from df_mlx.training_setup import setup_dataset


def test_print_run_config_parses_to_defaults():
    text = generate_run_config_example()
    data = tomllib.loads(text)
    cfg = RunConfig()
    apply_run_config_dict(cfg, data)
    assert cfg == RunConfig()


def test_run_config_precedence_cli_wins():
    cfg = RunConfig()
    apply_run_config_dict(cfg, {"training": {"learning_rate": 1e-4}})
    set_by_path(cfg, "training.learning_rate", 3e-5)
    assert cfg.training.learning_rate == 3e-5


def test_unknown_key_errors_with_suggestion():
    cfg = RunConfig()
    try:
        apply_run_config_dict(cfg, {"trainng": {"epochs": 2}})
    except ValueError as exc:
        msg = str(exc)
        assert "Unknown key" in msg
        assert "training" in msg
    else:
        raise AssertionError("Expected ValueError for unknown key")


def test_load_run_config_roundtrip(tmp_path):
    text = generate_run_config_example()
    path = tmp_path / "run.toml"
    path.write_text(text, encoding="utf-8")
    cfg = load_run_config(path)
    assert cfg == RunConfig()


def test_run_config_accepts_embedded_train_ini_tables():
    cfg = RunConfig()
    apply_run_config_dict(
        cfg,
        {
            "train_ini": {
                "df": {"sr": 44100, "fft_size": 1024},
                "train": {"max_epochs": 12},
                "MultiResSpecLoss": {"factor": 0.5},
            }
        },
    )
    assert cfg.train_ini["df"]["sr"] == 44100
    assert cfg.train_ini["df"]["fft_size"] == 1024
    assert cfg.train_ini["train"]["max_epochs"] == 12
    assert cfg.train_ini["MultiResSpecLoss"]["factor"] == 0.5


def test_run_config_accepts_pipeline_awesome_dynamic_loss():
    cfg = RunConfig()
    apply_run_config_dict(cfg, {"loss": {"dynamic_loss": "pipeline_awesome"}})
    assert cfg.loss.dynamic_loss == "pipeline_awesome"


def test_run_config_accepts_pipeline_stages_table_list():
    cfg = RunConfig()
    apply_run_config_dict(
        cfg,
        {
            "loss": {
                "pipeline_stages": [
                    {"start_epoch": 0, "name": "bootstrap", "awesome_loss_weight": 0.2},
                    {"start_epoch": 5, "name": "refine", "vad_loss_weight": 0.05},
                ]
            }
        },
    )

    assert len(cfg.loss.pipeline_stages) == 2
    assert cfg.loss.pipeline_stages[0]["start_epoch"] == 0
    assert cfg.loss.pipeline_stages[1]["start_epoch"] == 5


def test_run_config_accepts_speech_boost_options():
    cfg = RunConfig()
    apply_run_config_dict(
        cfg,
        {
            "enhance": {
                "speech_boost_db": 4.5,
                "speech_boost_threshold": 0.65,
                "speech_boost_min_speech_ms": 180,
                "speech_boost_min_silence_ms": 90,
                "speech_boost_pad_ms": 40,
                "speech_boost_ramp_ms": 10.0,
                "speech_boost_peak_limit": 0.95,
                "speech_boost_silero_model_path": "models/silero_vad.onnx",
                "speech_boost_silero_sample_rate": 16000,
            }
        },
    )
    assert cfg.enhance.speech_boost_db == 4.5
    assert cfg.enhance.speech_boost_threshold == 0.65
    assert cfg.enhance.speech_boost_min_speech_ms == 180
    assert cfg.enhance.speech_boost_min_silence_ms == 90
    assert cfg.enhance.speech_boost_pad_ms == 40
    assert cfg.enhance.speech_boost_ramp_ms == 10.0
    assert cfg.enhance.speech_boost_peak_limit == 0.95
    assert cfg.enhance.speech_boost_silero_model_path == "models/silero_vad.onnx"
    assert cfg.enhance.speech_boost_silero_sample_rate == 16000


def test_run_config_example_includes_speech_boost_descriptions():
    text = generate_run_config_example()
    assert "[enhance]" in text
    assert "# Boost dB applied only to Silero-detected speech segments (0 disables)" in text
    assert "speech_boost_db = 0.0" in text
    assert "# Silero speech probability threshold for segment detection" in text
    assert "speech_boost_threshold = 0.5" in text


def test_run_profiles_do_not_reference_removed_cleaned_datastore() -> None:
    profiles_dir = Path(__file__).resolve().parents[1] / "df_mlx" / "configs" / "run_profiles"
    stale_profiles = [path.name for path in profiles_dir.glob("*.toml") if "mlx_datastore_cleaned" in path.read_text()]
    assert stale_profiles == []


def test_setup_dataset_falls_back_from_removed_cleaned_cache_dir(tmp_path: Path, capsys) -> None:
    actual_cache_dir = tmp_path / "mlx_datastore"
    actual_cache_dir.mkdir()
    (actual_cache_dir / "config.json").write_text('{"sample_rate": 48000}\n', encoding="utf-8")

    result = setup_dataset(cache_dir=str(tmp_path / "mlx_datastore_cleaned"), use_mlx_data=False)

    assert result.config.cache_dir == str(actual_cache_dir)
    output = capsys.readouterr().out
    assert "mlx_datastore_cleaned/config.json" in output
    assert str(actual_cache_dir / "config.json") in output


def test_setup_dataset_missing_cache_lists_checked_candidates(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as exc_info:
        setup_dataset(cache_dir=str(tmp_path / "mlx_datastore_cleaned"), use_mlx_data=False)

    message = str(exc_info.value)
    assert "mlx_datastore_cleaned/config.json" in message
    assert "mlx_datastore/config.json" in message
