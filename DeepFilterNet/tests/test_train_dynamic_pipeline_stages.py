import pytest

from df_mlx.train_dynamic import _parse_pipeline_stages_cli, _resolve_pipeline_stage, _snr_bucket_name


def test_parse_pipeline_stages_cli_valid_json():
    stages = _parse_pipeline_stages_cli(
        '[{"start_epoch": 0, "name": "bootstrap", "awesome_loss_weight": 0.2}, '
        '{"start_epoch": 5, "name": "refine", "vad_loss_weight": 0.05}]'
    )

    assert len(stages) == 2
    assert stages[0]["start_epoch"] == 0
    assert stages[0]["name"] == "bootstrap"
    assert stages[1]["start_epoch"] == 5
    assert stages[1]["vad_loss_weight"] == pytest.approx(0.05)


def test_parse_pipeline_stages_cli_rejects_duplicate_start_epoch():
    with pytest.raises(ValueError, match="duplicate pipeline stage"):
        _parse_pipeline_stages_cli('[{"start_epoch": 0}, {"start_epoch": 0}]')


def test_resolve_pipeline_stage_uses_latest_started_stage():
    stages = [
        {"start_epoch": 0, "name": "bootstrap", "awesome_loss_weight": 0.2},
        {"start_epoch": 3, "name": "stabilize", "awesome_loss_weight": 0.35},
        {"start_epoch": 7, "name": "refine", "awesome_loss_weight": 0.45},
    ]

    stage_e0 = _resolve_pipeline_stage(0, stages)
    stage_e5 = _resolve_pipeline_stage(5, stages)
    stage_e9 = _resolve_pipeline_stage(9, stages)

    assert stage_e0["name"] == "bootstrap"
    assert stage_e5["name"] == "stabilize"
    assert stage_e9["name"] == "refine"


def test_snr_bucket_name_boundaries():
    assert _snr_bucket_name(-25.0) == "very_low"
    assert _snr_bucket_name(-10.0) == "extreme"
    assert _snr_bucket_name(0.0) == "low"
    assert _snr_bucket_name(10.0) == "mid"
    assert _snr_bucket_name(30.0) == "high"
