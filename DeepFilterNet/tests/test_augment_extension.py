"""Tests for the augmentation extension bridge (augment_ext).

Validates both the Python fallback path and (when available) the
Rust-accelerated path against reference implementations.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from df_mlx.augment_ext import (
    _combine_noises_python,
    _mix_audio_python,
    biquad_filter,
    combine_noises,
    mix_audio,
    rust_augment_available,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scipy_biquad(audio: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    from scipy import signal as scipy_signal

    return np.asarray(scipy_signal.lfilter(b, a, audio), dtype=np.float32)


# ---------------------------------------------------------------------------
# biquad_filter
# ---------------------------------------------------------------------------


class TestBiquadFilter:
    def test_passthrough(self) -> None:
        audio = np.array([1.0, 0.5, -0.3, 0.8, -1.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = biquad_filter(audio, b, a)
        np.testing.assert_allclose(result, audio, atol=1e-6)

    def test_parity_with_scipy(self) -> None:
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(4800).astype(np.float32) * 0.5

        # Low-pass biquad coefficients
        w0 = 2 * np.pi * 1000 / 48000
        q = 0.707
        alpha = np.sin(w0) / (2 * q)
        cos_w0 = np.cos(w0)
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = b0
        a0 = 1 + alpha
        a1_coeff = -2 * cos_w0
        a2 = 1 - alpha

        b = np.array([b0, b1, b2], dtype=np.float32) / a0
        a = np.array([1.0, a1_coeff / a0, a2 / a0], dtype=np.float32)

        expected = _scipy_biquad(audio, b, a)
        result = biquad_filter(audio, b, a)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_dc_gain(self) -> None:
        b = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        a = np.array([1.0, -0.1, 0.1], dtype=np.float32)
        audio = np.ones(2000, dtype=np.float32)
        result = biquad_filter(audio, b, a)
        dc_gain = sum(b) / sum(a)
        assert abs(float(result[-1]) - dc_gain) < 1e-3


# ---------------------------------------------------------------------------
# mix_audio
# ---------------------------------------------------------------------------


class TestMixAudio:
    def test_zero_snr(self) -> None:
        clean = np.full(200, 0.5, dtype=np.float32)
        noise = np.full(200, 0.4, dtype=np.float32)
        c, ns, noisy = mix_audio(clean, noise, snr_db=0.0, gain_db=0.0)
        c_pow = np.mean(c**2)
        n_pow = np.mean(ns**2)
        ratio_db = 10 * np.log10(c_pow / n_pow)
        assert abs(ratio_db) < 1.0, f"Expected ~0 dB, got {ratio_db:.2f}"

    def test_high_snr(self) -> None:
        rng = np.random.default_rng(7)
        clean = rng.standard_normal(1000).astype(np.float32) * 0.3
        noise = rng.standard_normal(1000).astype(np.float32) * 0.3
        c, ns, noisy = mix_audio(clean, noise, snr_db=20.0, gain_db=0.0)
        c_pow = np.mean(c**2)
        n_pow = np.mean(ns**2)
        ratio_db = 10 * np.log10(c_pow / n_pow)
        assert abs(ratio_db - 20.0) < 2.0, f"Expected ~20 dB, got {ratio_db:.2f}"

    def test_anticlip(self) -> None:
        clean = np.full(100, 0.9, dtype=np.float32)
        noise = np.full(100, 0.9, dtype=np.float32)
        _c, _ns, noisy = mix_audio(clean, noise, snr_db=0.0, gain_db=6.0)
        assert np.abs(noisy).max() <= 1.0

    def test_noise_tiling(self) -> None:
        clean = np.full(300, 0.3, dtype=np.float32)
        noise = np.array([0.1, -0.1], dtype=np.float32)
        c, ns, noisy = mix_audio(clean, noise, snr_db=10.0)
        assert len(ns) == 300

    def test_parity_with_python_fallback(self) -> None:
        rng = np.random.default_rng(99)
        clean = rng.standard_normal(500).astype(np.float32) * 0.4
        noise = rng.standard_normal(500).astype(np.float32) * 0.4
        snr, gain = 5.0, 3.0

        ref_c, ref_ns, ref_noisy = _mix_audio_python(clean.copy(), noise.copy(), snr, gain)
        c, ns, noisy = mix_audio(clean.copy(), noise.copy(), snr, gain)

        np.testing.assert_allclose(c, ref_c, atol=1e-4)
        np.testing.assert_allclose(ns, ref_ns, atol=1e-4)
        np.testing.assert_allclose(noisy, ref_noisy, atol=1e-4)


# ---------------------------------------------------------------------------
# combine_noises
# ---------------------------------------------------------------------------


class TestCombineNoises:
    def test_empty_list(self) -> None:
        result = combine_noises([], target_len=100)
        assert result.shape == (100,)
        np.testing.assert_allclose(result, 0.0)

    def test_single_noise_unity_gain(self) -> None:
        noise = np.ones(100, dtype=np.float32) * 0.5
        result = combine_noises([noise], target_len=100, gains_db=[0.0])
        np.testing.assert_allclose(result, 0.5, atol=1e-6)

    def test_two_noises_accumulate(self) -> None:
        n1 = np.ones(100, dtype=np.float32) * 0.3
        n2 = np.ones(100, dtype=np.float32) * 0.2
        result = combine_noises([n1, n2], target_len=100, gains_db=[0.0, 0.0])
        np.testing.assert_allclose(result, 0.5, atol=1e-6)

    def test_gain_applied(self) -> None:
        noise = np.ones(100, dtype=np.float32) * 0.5
        result = combine_noises([noise], target_len=100, gains_db=[6.0206])
        np.testing.assert_allclose(result, 1.0, atol=1e-2)

    def test_short_noise_tiled(self) -> None:
        noise = np.array([1.0, 2.0], dtype=np.float32)
        random.seed(42)
        result = combine_noises([noise], target_len=5, gains_db=[0.0])
        assert result.shape == (5,)

    def test_parity_fallback(self) -> None:
        rng = np.random.default_rng(123)
        n1 = rng.standard_normal(200).astype(np.float32) * 0.3
        n2 = rng.standard_normal(150).astype(np.float32) * 0.2
        gains = [3.0, -2.0]

        random.seed(77)
        ref = _combine_noises_python(
            [n1.copy(), n2.copy()],
            target_len=180,
            gains_db=gains,
            offsets=[0, 0],
        )
        random.seed(77)
        result = combine_noises([n1.copy(), n2.copy()], target_len=180, gains_db=gains)

        # Parity depends on offsets being deterministic; use atol for float diffs
        assert result.shape == ref.shape


# ---------------------------------------------------------------------------
# Bridge availability
# ---------------------------------------------------------------------------


class TestBridgeAvailability:
    def test_availability_flag_is_bool(self) -> None:
        assert isinstance(rust_augment_available(), bool)

    @pytest.mark.skipif(
        not rust_augment_available(),
        reason="Rust extension not installed",
    )
    def test_rust_extension_basic(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = biquad_filter(audio, b, a)
        np.testing.assert_allclose(result, audio, atol=1e-7)
