from pathlib import Path


def test_setup_sh_splits_local_workspace_dependencies_from_python_install():
    setup_sh = Path(__file__).resolve().parents[2] / "setup.sh"
    text = setup_sh.read_text(encoding="utf-8")

    assert "split_project_python_requirements()" in text
    assert "deepfilterlib)" in text
    assert "deepfilterdataloader)" in text
    assert "Installing external Python dependencies" in text
    assert "python -m pip install --no-deps" in text
    assert "dependencies handled separately" in text


def test_setup_sh_pins_repo_supported_maturin_range():
    setup_sh = Path(__file__).resolve().parents[2] / "setup.sh"
    text = setup_sh.read_text(encoding="utf-8")

    assert 'MATURIN_REQUIREMENT="maturin>=1.3,<1.5"' in text
    assert "Ensuring compatible maturin (${MATURIN_REQUIREMENT}) is installed" in text


def test_setup_sh_defaults_include_dnsmos_and_platform_asr():
    setup_sh = Path(__file__).resolve().parents[2] / "setup.sh"
    text = setup_sh.read_text(encoding="utf-8")

    assert "default_asr_extra()" in text
    assert 'echo "asr-mlx"' in text
    assert 'echo "asr"' in text
    assert 'printf "%s\\n" "$asr_extra" "dnsmos-local" "train" "eval"' in text
    assert "mapfile -t DEFAULT_EXTRAS < <(default_extras)" in text


def test_setup_sh_help_text_matches_platform_aware_defaults():
    setup_sh = Path(__file__).resolve().parents[2] / "setup.sh"
    text = setup_sh.read_text(encoding="utf-8")

    assert "platform-appropriate ASR extra" in text
    assert "dnsmos-local" in text
    assert "adds dev on top of the default extras" in text
