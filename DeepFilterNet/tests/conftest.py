"""Pytest configuration and fixtures for DeepFilterNet tests."""

import platform
from typing import Optional, Tuple

import pytest
import torch


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "mps: mark test as requiring MPS backend")


def _get_macos_version() -> Optional[Tuple[int, int]]:
    """Get macOS version as (major, minor) tuple, or None if not macOS."""
    if platform.system() != "Darwin":
        return None
    try:
        version = platform.mac_ver()[0]
        parts = version.split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except (ValueError, IndexError):
        return None


def _mps_available() -> bool:
    """Check if MPS backend is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _mps_supports_complex() -> bool:
    """Check if MPS supports complex operations (macOS 14+)."""
    if not _mps_available():
        return False
    macos_version = _get_macos_version()
    if macos_version is None:
        return False
    return macos_version[0] >= 14


@pytest.fixture
def mps_device():
    """Fixture providing an MPS device, skipping if not available."""
    if not _mps_available():
        pytest.skip("MPS not available")
    return torch.device("mps")


@pytest.fixture
def mps_complex_device():
    """Fixture providing MPS device for complex ops, skipping if not supported."""
    if not _mps_available():
        pytest.skip("MPS not available")
    if not _mps_supports_complex():
        pytest.skip("MPS complex operations require macOS 14+")
    return torch.device("mps")


@pytest.fixture
def cpu_device():
    """Fixture providing a CPU device."""
    return torch.device("cpu")


@pytest.fixture
def any_device():
    """Fixture providing the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="session", autouse=False)
def df_config():
    """Initialize df.config with minimal settings for testing.

    Use this fixture for tests that need Loss class or other config-dependent code.
    """
    import os
    import tempfile

    from df.config import config

    # Create a minimal config file
    config_content = """
[train]
MODEL = deepfilternet3
BATCH_SIZE = 2
MAX_EPOCHS = 1
GAN_ENABLED = false

[df]
sr = 48000
fft_size = 960
hop_size = 480
nb_erb = 32
nb_df = 96
lsnr_max = 35
lsnr_min = -15
min_nb_freqs = 2

[MaskLoss]
factor = 1

[SpectralLoss]
factor_magnitude = 0
factor_complex = 0

[MultiResSpecLoss]
factor = 0

[SdrLoss]
factor = 0

[LocalSnrLoss]
factor = 0

[ASRLoss]
factor = 0
factor_lm = 0

[GANLoss]
factor = 0

[FeatureMatchingLoss]
factor = 0

[SpeakerLoss]
factor = 0

[optim]
LR = 1e-4
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        config.load(config_path)
        yield config
    finally:
        os.unlink(config_path)


# Auto-skip MPS tests on non-macOS platforms
def pytest_collection_modifyitems(config, items):
    """Automatically skip MPS-marked tests on non-MPS platforms."""
    skip_mps = pytest.mark.skip(reason="MPS not available on this platform")
    for item in items:
        if "mps" in item.keywords and not _mps_available():
            item.add_marker(skip_mps)
