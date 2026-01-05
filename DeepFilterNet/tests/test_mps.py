"""MPS backend tests for DeepFilterNet.

These tests validate MPS (Metal Performance Shaders) backend functionality
on Apple Silicon Macs. Tests are automatically skipped on non-macOS platforms.
"""

import platform
from typing import Optional, Tuple

import pytest
import torch


# === Device Detection Tests ===


def test_get_macos_version_returns_tuple_on_darwin():
    """Test that get_macos_version returns proper tuple on macOS."""
    from df.utils import get_macos_version
    
    version = get_macos_version()
    if platform.system() == "Darwin":
        assert version is not None
        assert isinstance(version, tuple)
        assert len(version) == 2
        assert isinstance(version[0], int)
        assert isinstance(version[1], int)
        assert version[0] >= 10  # Minimum expected major version
    else:
        assert version is None


def test_mps_supports_complex_returns_bool():
    """Test that mps_supports_complex returns a boolean."""
    from df.utils import mps_supports_complex
    
    result = mps_supports_complex()
    assert isinstance(result, bool)


def test_mps_supports_complex_true_on_non_mps():
    """Test that mps_supports_complex returns True when MPS not available."""
    from df.utils import mps_supports_complex
    
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        # When MPS is not available, function should return True (no restriction)
        assert mps_supports_complex() is True


def test_get_device_explicit_cpu():
    """Test explicit CPU device selection."""
    from df.utils import get_device
    
    device = get_device("cpu")
    assert device.type == "cpu"


def test_get_device_auto_detection():
    """Test auto device detection returns valid device."""
    from df.utils import get_device
    
    device = get_device()
    assert device.type in ("cuda", "mps", "cpu")


@pytest.mark.mps
def test_get_device_explicit_mps(mps_device):
    """Test explicit MPS device selection."""
    from df.utils import get_device
    
    device = get_device("mps")
    assert device.type == "mps"


# === MPS Safe Norm Tests ===


def test_mps_safe_norm_real_tensor_cpu():
    """Test mps_safe_norm works on real CPU tensors."""
    from df.utils import mps_safe_norm
    
    x = torch.randn(4, 8)
    norm = mps_safe_norm(x, dim=1)
    expected = torch.norm(x, dim=1)
    assert torch.allclose(norm, expected)


def test_mps_safe_norm_real_tensor_keepdim():
    """Test mps_safe_norm preserves dimensions when keepdim=True."""
    from df.utils import mps_safe_norm
    
    x = torch.randn(4, 8)
    norm = mps_safe_norm(x, dim=1, keepdim=True)
    assert norm.shape == (4, 1)


@pytest.mark.mps
def test_mps_safe_norm_real_tensor_mps(mps_device):
    """Test mps_safe_norm works on real MPS tensors."""
    from df.utils import mps_safe_norm
    
    x = torch.randn(4, 8, device=mps_device)
    norm = mps_safe_norm(x, dim=1)
    assert norm.device.type == "mps"
    assert norm.shape == (4,)


@pytest.mark.mps
def test_mps_safe_norm_complex_tensor_mps(mps_complex_device):
    """Test mps_safe_norm handles complex MPS tensors via CPU fallback."""
    from df.utils import mps_safe_norm
    
    x = torch.randn(4, 8, dtype=torch.complex64, device=mps_complex_device)
    norm = mps_safe_norm(x, dim=1)
    # Result should be back on MPS
    assert norm.device.type == "mps"
    assert norm.shape == (4,)
    # Result should be real
    assert not norm.is_complex()


# === DfOp Forward Method Tests ===


@pytest.mark.mps
def test_dfop_forward_real_unfold_mps(mps_device):
    """Test forward_real_unfold works on MPS."""
    from df.modules import DfOp
    
    df_bins = 96
    df_order = 5
    
    dfop = DfOp(df_bins=df_bins, df_order=df_order, method="real_unfold")
    dfop = dfop.to(mps_device)
    
    # Create test inputs [B, 1, T, F, 2]
    batch, time, freq = 1, 10, 256
    spec = torch.randn(batch, 1, time, freq, 2, device=mps_device)
    # Coefficients [B, T, O, F, 2]
    coefs = torch.randn(batch, time, df_order, df_bins, 2, device=mps_device)
    
    output = dfop.forward_real_unfold(spec, coefs)
    
    assert output.device.type == "mps"
    assert output.shape == spec.shape
    assert not torch.isnan(output).any()


@pytest.mark.mps
def test_dfop_forward_complex_strided_mps(mps_complex_device):
    """Test forward_complex_strided works on MPS with macOS 14+."""
    from df.modules import DfOp
    
    df_bins = 96
    df_order = 5
    
    dfop = DfOp(df_bins=df_bins, df_order=df_order, method="complex_strided")
    dfop = dfop.to(mps_complex_device)
    
    # Create test inputs [B, 1, T, F, 2]
    batch, time, freq = 1, 10, 256
    spec = torch.randn(batch, 1, time, freq, 2, device=mps_complex_device)
    # Coefficients [B, T, O, F, 2]
    coefs = torch.randn(batch, time, df_order, df_bins, 2, device=mps_complex_device)
    
    output = dfop.forward_complex_strided(spec, coefs)
    
    assert output.device.type == "mps"
    assert output.shape == spec.shape
    assert not torch.isnan(output).any()


@pytest.mark.mps
def test_dfop_auto_method_selection_mps(mps_device):
    """Test DfOp auto-selects appropriate method based on MPS complex support."""
    from df.modules import DfOp
    from df.utils import mps_supports_complex
    
    # Create DfOp with default complex_strided method
    dfop = DfOp(df_bins=96, df_order=5, method="complex_strided")
    
    # If MPS doesn't support complex, forward should be forward_real_unfold
    if not mps_supports_complex():
        assert dfop.forward == dfop.forward_real_unfold
    else:
        assert dfop.forward == dfop.forward_complex_strided


# === Basic Tensor Operations on MPS ===


@pytest.mark.mps
def test_basic_tensor_ops_mps(mps_device):
    """Test basic tensor operations work on MPS."""
    x = torch.randn(10, 10, device=mps_device)
    y = torch.randn(10, 10, device=mps_device)
    
    # Matrix multiplication
    z = x @ y
    assert z.device.type == "mps"
    
    # Element-wise ops
    z = x * y + x
    assert z.device.type == "mps"
    
    # Reduction
    s = x.sum()
    assert s.device.type == "mps"


@pytest.mark.mps
def test_complex_tensor_creation_mps(mps_complex_device):
    """Test complex tensor creation on MPS (macOS 14+)."""
    # Create real tensor and view as complex
    real = torch.randn(10, 10, 2, device=mps_complex_device)
    c = torch.view_as_complex(real)
    
    assert c.is_complex()
    assert c.device.type == "mps"
    
    # View back as real
    real2 = torch.view_as_real(c)
    assert not real2.is_complex()
    assert torch.allclose(real, real2)
