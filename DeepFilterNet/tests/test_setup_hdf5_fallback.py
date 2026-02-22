from pathlib import Path


def test_setup_sh_contains_hdf5_homebrew_2x_fallback_logic():
    setup_sh = Path(__file__).resolve().parents[2] / "setup.sh"
    text = setup_sh.read_text(encoding="utf-8")

    assert "H5pubconf.h" in text
    assert "Homebrew HDF5" in text
    assert "hdf5-static" in text
    assert "retrying with bundled static HDF5" in text
