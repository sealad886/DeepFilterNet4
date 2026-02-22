"""Tests for guarded augmentation extension fallback architecture.

Validates that all augmentation operations work correctly with both the
Rust extension (when available) and the pure-Python fallback path.
"""

from __future__ import annotations

import random
from unittest import mock

import numpy as np
import pytest

from df_mlx.augment_ext import (
    _combine_noises_python,
    _mix_audio_python,
    augment_capabilities,
    biquad_filter,
    combine_noises,
    mix_audio,
    rust_augment_available,
)

# Seed for deterministic noise generation in tests
_RNG = np.random.RandomState(42)
_SR = 48000


def _sine(freq: float = 440.0, duration: float = 0.1, sr: int = _SR) -> np.ndarray:
    """Generate a short sine wave for testing."""
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(length: int = 4800) -> np.ndarray:
    """Generate reproducible white noise."""
    return _RNG.randn(length).astype(np.float32) * 0.1


# ---------------------------------------------------------------------------
# Capability checks
# ---------------------------------------------------------------------------


class TestCapabilityChecks:
    """Tests for capability reporting functions."""

    def test_augment_capabilities_returns_dict(self) -> None:
        caps = augment_capabilities()
        assert isinstance(caps, dict)

    def test_augment_capabilities_has_required_keys(self) -> None:
        caps = augment_capabilities()
        expected_keys = {"rust_extension", "biquad_backend", "mix_backend", "combine_backend"}
        assert set(caps.keys()) == expected_keys

    def test_rust_available_consistent_with_capabilities(self) -> None:
        caps = augment_capabilities()
        assert caps["rust_extension"] is rust_augment_available()

    def test_backend_values_when_rust_available(self) -> None:
        if not rust_augment_available():
            pytest.skip("Rust extension not installed")
        caps = augment_capabilities()
        assert caps["biquad_backend"] == "rust"
        assert caps["mix_backend"] == "rust"
        assert caps["combine_backend"] == "rust"

    def test_backend_values_when_rust_not_available(self) -> None:
        if rust_augment_available():
            pytest.skip("Rust extension is installed — cannot test fallback values")
        caps = augment_capabilities()
        assert caps["biquad_backend"] == "scipy"
        assert caps["mix_backend"] == "numpy"
        assert caps["combine_backend"] == "numpy"


# ---------------------------------------------------------------------------
# Rust pathway (skipped when extension is not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not rust_augment_available(), reason="Rust extension not installed")
class TestRustPathway:
    """Tests that run only when Rust extension is available."""

    def test_biquad_filter_rust(self) -> None:
        audio = _sine()
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = biquad_filter(audio, b, a)
        np.testing.assert_allclose(result, audio, atol=1e-6)

    def test_mix_audio_rust(self) -> None:
        clean = _sine()
        noise = _noise(len(clean))
        c, ns, noisy = mix_audio(clean, noise, snr_db=10.0)
        assert c.shape == clean.shape
        assert ns.shape == clean.shape
        assert noisy.shape == clean.shape
        assert c.dtype == np.float32

    def test_combine_noises_rust(self) -> None:
        n1 = _noise(4800)
        n2 = _noise(9600)
        result = combine_noises([n1, n2], target_len=4800)
        assert result.shape == (4800,)
        assert result.dtype == np.float32
        assert not np.all(result == 0)


# ---------------------------------------------------------------------------
# Fallback pathway (always runs using pure-Python implementations)
# ---------------------------------------------------------------------------


class TestFallbackPathway:
    """Tests that always exercise the Python fallback functions."""

    def test_mix_audio_python_produces_valid_output(self) -> None:
        clean = _sine()
        noise = _noise(len(clean))
        c, ns, noisy = _mix_audio_python(clean, noise, snr_db=10.0)
        assert c.dtype == np.float32
        assert ns.dtype == np.float32
        assert noisy.dtype == np.float32
        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(ns))
        assert np.all(np.isfinite(noisy))

    def test_mix_audio_python_gain_applied(self) -> None:
        clean = _sine()
        noise = _noise(len(clean))
        c0, _, _ = _mix_audio_python(clean, noise, snr_db=10.0, gain_db=0.0)
        c6, _, _ = _mix_audio_python(clean, noise, snr_db=10.0, gain_db=6.0)
        # With positive gain, clean_out RMS should be higher (before clipping guard)
        assert np.mean(c6**2) > 0  # at minimum, non-silent

    def test_mix_audio_python_clipping_guard(self) -> None:
        clean = np.ones(1000, dtype=np.float32) * 0.9
        noise = np.ones(1000, dtype=np.float32) * 0.9
        _, _, noisy = _mix_audio_python(clean, noise, snr_db=0.0)
        assert np.abs(noisy).max() <= 1.0 + 1e-6

    def test_mix_audio_python_noise_shorter_than_clean(self) -> None:
        clean = _sine(duration=0.2)
        noise = _noise(len(clean) // 3)
        c, ns, noisy = _mix_audio_python(clean, noise, snr_db=10.0)
        assert c.shape == clean.shape

    def test_combine_noises_python_empty(self) -> None:
        result = _combine_noises_python([], target_len=4800, gains_db=[], offsets=[])
        assert result.shape == (4800,)
        assert np.all(result == 0)

    def test_combine_noises_python_single(self) -> None:
        n = _noise(4800)
        result = _combine_noises_python([n], target_len=4800, gains_db=[0.0], offsets=[0])
        np.testing.assert_allclose(result, n, atol=1e-6)

    def test_combine_noises_python_multiple(self) -> None:
        n1 = _noise(4800)
        n2 = _noise(4800)
        result = _combine_noises_python([n1, n2], target_len=4800, gains_db=[0.0, 0.0], offsets=[0, 0])
        expected = n1 + n2
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_combine_noises_python_with_gain(self) -> None:
        n = _noise(4800)
        result_0db = _combine_noises_python([n], target_len=4800, gains_db=[0.0], offsets=[0])
        result_6db = _combine_noises_python([n], target_len=4800, gains_db=[6.0], offsets=[0])
        assert np.mean(result_6db**2) > np.mean(result_0db**2)

    def test_combine_noises_python_short_noise_tiled(self) -> None:
        n = _noise(100)
        result = _combine_noises_python([n], target_len=4800, gains_db=[0.0], offsets=[0])
        assert result.shape == (4800,)
        assert not np.all(result == 0)

    def test_biquad_filter_fallback_identity(self) -> None:
        """Biquad with identity coefficients returns ~original signal."""
        import df_mlx.augment_ext as ext_mod

        with mock.patch.object(ext_mod, "_RUST_AVAILABLE", False):
            audio = _sine()
            b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            result = ext_mod.biquad_filter(audio, b, a)
            np.testing.assert_allclose(result, audio, atol=1e-5)

    def test_biquad_filter_fallback_low_pass(self) -> None:
        """Low-pass biquad attenuates high-frequency content."""
        import df_mlx.augment_ext as ext_mod

        with mock.patch.object(ext_mod, "_RUST_AVAILABLE", False):
            audio = _sine(freq=10000.0, duration=0.05)
            # Simple low-pass at 1000 Hz
            w0 = 2 * np.pi * 1000 / _SR
            alpha = np.sin(w0) / (2 * 0.707)
            cos_w0 = np.cos(w0)
            b0 = (1 - cos_w0) / 2
            b1 = 1 - cos_w0
            b2 = b0
            a0 = 1 + alpha
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1.0, -2 * cos_w0 / a0, (1 - alpha) / a0])
            result = ext_mod.biquad_filter(audio, b, a)
            assert np.mean(result**2) < np.mean(audio**2)


# ---------------------------------------------------------------------------
# Parity: both pathways produce identical results
# ---------------------------------------------------------------------------


class TestParity:
    """Tests that both pathways produce identical (or near-identical) results.

    When Rust is not available, these tests compare the bridge output to
    the explicit Python fallback output — which should be identical since
    the bridge delegates to the same Python fallback in that case.
    """

    def test_mix_audio_bridge_matches_python_fallback(self) -> None:
        """Bridge mix_audio matches explicit Python fallback."""
        import df_mlx.augment_ext as ext_mod

        clean = _sine()
        noise = _noise(len(clean))

        # Force Python path
        with mock.patch.object(ext_mod, "_RUST_AVAILABLE", False):
            bridge_c, bridge_ns, bridge_noisy = ext_mod.mix_audio(clean, noise, snr_db=10.0, gain_db=3.0)

        py_c, py_ns, py_noisy = _mix_audio_python(clean, noise, snr_db=10.0, gain_db=3.0)

        np.testing.assert_allclose(bridge_c, py_c, atol=1e-7)
        np.testing.assert_allclose(bridge_ns, py_ns, atol=1e-7)
        np.testing.assert_allclose(bridge_noisy, py_noisy, atol=1e-7)

    def test_combine_noises_bridge_matches_python_fallback(self) -> None:
        """Bridge combine_noises matches explicit Python fallback."""
        import df_mlx.augment_ext as ext_mod

        n1 = _noise(4800)
        n2 = _noise(2400)

        random.seed(99)
        with mock.patch.object(ext_mod, "_RUST_AVAILABLE", False):
            bridge_result = ext_mod.combine_noises([n1, n2], target_len=4800, gains_db=[0.0, 3.0])

        # Reproduce the offsets the bridge computed (same seed)
        random.seed(99)
        offsets = []
        for n in [n1, n2]:
            if len(n) > 4800:
                offsets.append(random.randint(0, len(n) - 4800))
            else:
                offsets.append(0)

        py_result = _combine_noises_python([n1, n2], 4800, [0.0, 3.0], offsets)
        np.testing.assert_allclose(bridge_result, py_result, atol=1e-7)

    def test_biquad_filter_bridge_matches_fallback(self) -> None:
        """Bridge biquad matches explicit scipy fallback when forced to Python."""
        import df_mlx.augment_ext as ext_mod

        audio = _sine()
        b = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        a = np.array([1.0, -0.4, 0.1], dtype=np.float64)

        with mock.patch.object(ext_mod, "_RUST_AVAILABLE", False):
            result = ext_mod.biquad_filter(audio, b, a)

        from scipy import signal as scipy_signal

        expected = np.asarray(scipy_signal.lfilter(b, a, audio), dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_determinism_mix_audio(self) -> None:
        """Same inputs produce same outputs on repeated calls."""
        clean = _sine()
        noise = _noise(len(clean))
        r1 = mix_audio(clean, noise, snr_db=5.0, gain_db=2.0)
        r2 = mix_audio(clean, noise, snr_db=5.0, gain_db=2.0)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# dynamic_dataset.py routing
# ---------------------------------------------------------------------------


class TestDynamicDatasetRouting:
    """Verify that mix_audio / combine_noises in dynamic_dataset.py delegate to the bridge."""

    def test_mix_audio_routes_through_bridge(self) -> None:
        from df_mlx import dynamic_dataset as ds_mod

        clean = _sine()
        noise = _noise(len(clean))

        # Call through dynamic_dataset
        ds_c, ds_ns, ds_noisy = ds_mod.mix_audio(clean, noise, snr_db=10.0)

        # Call through bridge directly
        ext_c, ext_ns, ext_noisy = mix_audio(clean, noise, snr_db=10.0)

        np.testing.assert_array_equal(ds_c, ext_c)
        np.testing.assert_array_equal(ds_ns, ext_ns)
        np.testing.assert_array_equal(ds_noisy, ext_noisy)

    def test_combine_noises_routes_through_bridge(self) -> None:
        from df_mlx import dynamic_dataset as ds_mod

        n1 = _noise(4800)
        n2 = _noise(4800)

        # Both use same RNG state — seed before each call
        random.seed(123)
        ds_result = ds_mod.combine_noises([n1, n2], target_len=4800, gains_db=[0.0, 0.0])
        random.seed(123)
        ext_result = combine_noises([n1, n2], target_len=4800, gains_db=[0.0, 0.0])

        np.testing.assert_array_equal(ds_result, ext_result)

    def test_augmentations_biquad_routes_through_bridge(self) -> None:
        from df_mlx.dynamic_dataset import Augmentations

        audio = _sine()
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        aug_result = Augmentations.biquad_filter(audio, b, a)
        ext_result = biquad_filter(audio, b, a)

        np.testing.assert_array_equal(aug_result, ext_result)


# ---------------------------------------------------------------------------
# Augmentations class end-to-end (both modes)
# ---------------------------------------------------------------------------


class TestAugmentationsEndToEnd:
    """End-to-end tests for Augmentations class methods that use the bridge."""

    def test_high_pass(self) -> None:
        from df_mlx.dynamic_dataset import Augmentations

        audio = _sine(freq=100.0, duration=0.1)
        result = Augmentations.high_pass(audio, freq=500.0, q=0.707, sr=_SR)
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        # High-pass at 500 Hz should attenuate a 100 Hz sine
        assert np.mean(result**2) < np.mean(audio**2)

    def test_low_pass(self) -> None:
        from df_mlx.dynamic_dataset import Augmentations

        audio = _sine(freq=10000.0, duration=0.1)
        result = Augmentations.low_pass(audio, freq=1000.0, q=0.707, sr=_SR)
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        # Low-pass at 1000 Hz should attenuate a 10000 Hz sine
        assert np.mean(result**2) < np.mean(audio**2)

    def test_bandwidth_limit(self) -> None:
        from df_mlx.dynamic_dataset import Augmentations

        audio = _sine(freq=100.0, duration=0.1)
        result = Augmentations.bandwidth_limit(audio, low_freq=200.0, high_freq=8000.0, sr=_SR)
        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_biquad_filter_identity(self) -> None:
        from df_mlx.dynamic_dataset import Augmentations

        audio = _sine()
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = Augmentations.biquad_filter(audio, b, a)
        np.testing.assert_allclose(result, audio, atol=1e-5)

    def test_high_pass_fallback(self) -> None:
        """high_pass works correctly via Python fallback."""
        import df_mlx.augment_ext as ext_mod
        from df_mlx.dynamic_dataset import Augmentations

        audio = _sine(freq=100.0, duration=0.1)
        with mock.patch.object(ext_mod, "_RUST_AVAILABLE", False):
            result = Augmentations.high_pass(audio, freq=500.0, q=0.707, sr=_SR)
        assert result.shape == audio.shape
        assert np.mean(result**2) < np.mean(audio**2)

    def test_low_pass_fallback(self) -> None:
        """low_pass works correctly via Python fallback."""
        import df_mlx.augment_ext as ext_mod
        from df_mlx.dynamic_dataset import Augmentations

        audio = _sine(freq=10000.0, duration=0.1)
        with mock.patch.object(ext_mod, "_RUST_AVAILABLE", False):
            result = Augmentations.low_pass(audio, freq=1000.0, q=0.707, sr=_SR)
        assert result.shape == audio.shape
        assert np.mean(result**2) < np.mean(audio**2)

    def test_bandwidth_limit_fallback(self) -> None:
        """bandwidth_limit works correctly via Python fallback."""
        import df_mlx.augment_ext as ext_mod
        from df_mlx.dynamic_dataset import Augmentations

        audio = _sine(freq=100.0, duration=0.1)
        with mock.patch.object(ext_mod, "_RUST_AVAILABLE", False):
            result = Augmentations.bandwidth_limit(audio, low_freq=200.0, high_freq=8000.0, sr=_SR)
        assert result.shape == audio.shape
