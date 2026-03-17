"""Correctness tests for complex-spectral helpers and split epsilon policy.

Validates the pure-MLX helper functions (complex_mag, log1p_mag,
band_energy) and the split epsilon constants (_EPS_F, _BAND_ENERGY_EPS_F).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from df_mlx.metal_kernels import (
    _BAND_ENERGY_EPS_F,
    _EPS_F,
    band_energy,
    complex_mag,
    log1p_mag,
)
from df_mlx.train import spectral_loss
from df_mlx.training_losses import _log1p_mag as _training_log1p_mag


def _rand_complex(shape, dtype=mx.float32):
    """Generate random complex components in a realistic spectral range."""
    rng = np.random.default_rng(42)
    real = mx.array(rng.standard_normal(shape).astype(np.float32))
    imag = mx.array(rng.standard_normal(shape).astype(np.float32))
    if dtype != mx.float32:
        real = real.astype(dtype)
        imag = imag.astype(dtype)
    return real, imag


# ==================================================================
# log1p_mag
# ==================================================================


class TestLog1pMag:
    def test_forward_basic(self):
        """Basic forward pass produces expected values."""
        real, imag = _rand_complex((4, 100, 64))
        result = log1p_mag(real, imag)
        ref = mx.log1p(mx.sqrt(real * real + imag * imag + _EPS_F))
        mx.eval(result, ref)
        np.testing.assert_allclose(np.array(result), np.array(ref), rtol=1e-5, atol=1e-6)

    def test_forward_zeros(self):
        """All-zero input (tests eps stability)."""
        real = mx.zeros((2, 10, 32))
        imag = mx.zeros((2, 10, 32))
        result = log1p_mag(real, imag)
        mx.eval(result)
        expected = np.full((2, 10, 32), np.log1p(np.sqrt(_EPS_F)), dtype=np.float32)
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-6, atol=1e-7)

    def test_default_eps_uses_magnitude_floor_constant(self):
        """Default log1p magnitude should use the dedicated magnitude epsilon."""
        real = mx.zeros((1, 2, 3), dtype=mx.float32)
        imag = mx.zeros((1, 2, 3), dtype=mx.float32)
        result = log1p_mag(real, imag)
        mx.eval(result)
        expected = np.full((1, 2, 3), np.log1p(np.sqrt(_EPS_F)), dtype=np.float32)
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-6, atol=1e-7)

    def test_forward_respects_custom_eps(self):
        """Output changes with caller-provided epsilon."""
        real, imag = _rand_complex((2, 8, 8))
        eps = 1e-4
        result = log1p_mag(real, imag, eps=eps)
        ref = mx.log1p(mx.sqrt(real * real + imag * imag + eps))
        mx.eval(result, ref)
        np.testing.assert_allclose(np.array(result), np.array(ref), rtol=1e-5, atol=1e-6)

    def test_forward_preserves_shape(self):
        """Output shape matches input shape."""
        real, imag = _rand_complex((3, 50, 128))
        result = log1p_mag(real, imag)
        mx.eval(result)
        assert result.shape == real.shape

    def test_reduced_precision_dtype_is_preserved(self):
        """Raw helper keeps reduced precision unless the caller upcasts first."""
        real, imag = _rand_complex((2, 8, 8), dtype=mx.float16)
        result = log1p_mag(real, imag)
        mx.eval(result)
        assert result.dtype == mx.float16

    def test_backward_produces_gradients(self):
        """Backward pass produces finite gradients."""
        real, imag = _rand_complex((2, 10, 16))

        def fn(r, i):
            return mx.sum(log1p_mag(r, i))

        grad_fn = mx.grad(fn, argnums=[0, 1])
        g_r, g_i = grad_fn(real, imag)
        mx.eval(g_r, g_i)
        assert np.all(np.isfinite(np.array(g_r)))
        assert np.all(np.isfinite(np.array(g_i)))

    def test_training_wrapper_forwards_eps(self):
        """The training helper must preserve caller-provided epsilon."""
        real, imag = _rand_complex((2, 8, 8))
        eps = 1e-4
        wrapper = _training_log1p_mag(real, imag, eps=eps, _assume_float32=True)
        direct = log1p_mag(real, imag, eps=eps)
        mx.eval(wrapper, direct)
        np.testing.assert_allclose(np.array(wrapper), np.array(direct), rtol=1e-5, atol=1e-6)


# ==================================================================
# complex_mag
# ==================================================================


class TestComplexMag:
    def test_forward_basic(self):
        real, imag = _rand_complex((4, 100, 64))
        result = complex_mag(real, imag)
        ref = mx.sqrt(real * real + imag * imag + _EPS_F)
        mx.eval(result, ref)
        np.testing.assert_allclose(np.array(result), np.array(ref), rtol=1e-5, atol=1e-6)

    def test_forward_zeros(self):
        real = mx.zeros((2, 10, 32))
        imag = mx.zeros((2, 10, 32))
        result = complex_mag(real, imag)
        mx.eval(result)
        expected = np.full((2, 10, 32), np.sqrt(_EPS_F), dtype=np.float32)
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-5, atol=1e-6)

    def test_forward_respects_custom_eps(self):
        real, imag = _rand_complex((2, 8, 8))
        eps = 1e-4
        result = complex_mag(real, imag, eps=eps)
        ref = mx.sqrt(real * real + imag * imag + eps)
        mx.eval(result, ref)
        np.testing.assert_allclose(np.array(result), np.array(ref), rtol=1e-5, atol=1e-6)

    def test_reduced_precision_dtype_is_preserved(self):
        """Raw helper keeps reduced precision unless the caller upcasts first."""
        real, imag = _rand_complex((2, 8, 8), dtype=mx.float16)
        result = complex_mag(real, imag)
        mx.eval(result)
        assert result.dtype == mx.float16

    def test_backward_produces_gradients(self):
        real, imag = _rand_complex((2, 10, 16))

        def fn(r, i):
            return mx.sum(complex_mag(r, i))

        grad_fn = mx.grad(fn, argnums=[0, 1])
        g_r, g_i = grad_fn(real, imag)
        mx.eval(g_r, g_i)
        assert np.all(np.isfinite(np.array(g_r)))
        assert np.all(np.isfinite(np.array(g_i)))


class TestSpectralLossWiring:
    def test_train_spectral_loss_matches_reference(self):
        real_pred, imag_pred = _rand_complex((2, 16, 32))
        real_target, imag_target = _rand_complex((2, 16, 32))

        result = spectral_loss((real_pred, imag_pred), (real_target, imag_target), alpha=0.5)

        pred_mag = complex_mag(real_pred, imag_pred, eps=_EPS_F)
        target_mag = complex_mag(real_target, imag_target, eps=_EPS_F)
        ref_mag_loss = mx.mean(mx.abs(pred_mag - target_mag))
        ref_complex_loss = mx.mean(mx.abs(real_pred - real_target) + mx.abs(imag_pred - imag_target))
        ref = 0.5 * ref_mag_loss + 0.5 * ref_complex_loss

        mx.eval(result, ref)
        np.testing.assert_allclose(np.array(result), np.array(ref), rtol=1e-5, atol=1e-6)


# ==================================================================
# band_energy
# ==================================================================


class TestBandEnergy:
    @staticmethod
    def _make_band_mask(F: int, active_bins: int = 20):
        """Create a realistic band mask: first `active_bins` bins are 1."""
        mask = np.zeros(F, dtype=np.float32)
        mask[:active_bins] = 1.0
        return mx.array(mask), float(active_bins)

    def test_forward_basic(self):
        B, T, F = 4, 100, 64
        real, imag = _rand_complex((B, T, F))
        mask, bins = self._make_band_mask(F)
        f_band, f_log = band_energy(real, imag, mask, bins)
        mx.eval(f_band, f_log)
        # Verify against inline computation
        power = real * real + imag * imag
        ref_band = mx.sum(power * mask, axis=-1) / (bins + _BAND_ENERGY_EPS_F)
        ref_log = mx.log10(ref_band + _BAND_ENERGY_EPS_F)
        mx.eval(ref_band, ref_log)
        np.testing.assert_allclose(np.array(f_band), np.array(ref_band), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.array(f_log), np.array(ref_log), rtol=1e-4, atol=1e-5)

    def test_output_shapes(self):
        B, T, F = 2, 50, 32
        real, imag = _rand_complex((B, T, F))
        mask, bins = self._make_band_mask(F, active_bins=10)
        f_band, f_log = band_energy(real, imag, mask, bins)
        mx.eval(f_band, f_log)
        assert f_band.shape == (B, T)
        assert f_log.shape == (B, T)

    def test_zeros_input(self):
        B, T, F = 2, 10, 16
        real = mx.zeros((B, T, F))
        imag = mx.zeros((B, T, F))
        mask, bins = self._make_band_mask(F, active_bins=8)
        f_band, f_log = band_energy(real, imag, mask, bins)
        mx.eval(f_band, f_log)
        expected_log = np.full((B, T), np.log10(_BAND_ENERGY_EPS_F), dtype=np.float32)
        np.testing.assert_allclose(np.array(f_log), expected_log, rtol=1e-6, atol=1e-7)

    def test_default_eps_uses_band_energy_floor_constant(self):
        """Band-energy defaults should keep the stronger reduction epsilon."""
        real = mx.zeros((1, 2, 16), dtype=mx.float32)
        imag = mx.zeros((1, 2, 16), dtype=mx.float32)
        mask, bins = self._make_band_mask(16, active_bins=8)
        _band, log_band = band_energy(real, imag, mask, bins)
        mx.eval(log_band)
        expected = np.full((1, 2), np.log10(_BAND_ENERGY_EPS_F), dtype=np.float32)
        np.testing.assert_allclose(np.array(log_band), expected, rtol=1e-6, atol=1e-7)
        assert not np.isclose(np.log10(_BAND_ENERGY_EPS_F), np.log10(_EPS_F))

    @pytest.mark.parametrize("band_bins", [0.0, -1.0])
    def test_rejects_non_positive_band_bins(self, band_bins: float):
        """Invalid band definitions should fail loudly instead of masking bugs."""
        real, imag = _rand_complex((1, 2, 16))
        mask = mx.zeros((16,), dtype=mx.float32)

        with pytest.raises(ValueError, match="band_bins must be positive"):
            band_energy(real, imag, mask, band_bins)

    def test_bfloat16_inputs_accumulate_in_float32(self):
        """Reduced-precision inputs should produce stable float32 outputs."""
        B, T, F = 2, 12, 24
        real, imag = _rand_complex((B, T, F), dtype=mx.bfloat16)
        mask, bins = self._make_band_mask(F, active_bins=12)
        f_band, f_log = band_energy(real, imag, mask, bins)
        # Compare with float32 reference
        r_power = real.astype(mx.float32) ** 2 + imag.astype(mx.float32) ** 2
        r_band = mx.sum(r_power * mask, axis=-1) / (bins + _BAND_ENERGY_EPS_F)
        r_log = mx.log10(r_band + _BAND_ENERGY_EPS_F)
        mx.eval(f_band, f_log, r_band, r_log)
        assert f_band.dtype == mx.float32
        assert f_log.dtype == mx.float32
        np.testing.assert_allclose(np.array(f_band), np.array(r_band), rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(np.array(f_log), np.array(r_log), rtol=2e-3, atol=2e-3)


# ==================================================================
# Split epsilon regression tests
# ==================================================================


class TestSplitEpsilonDefaults:
    def test_magnitude_eps_is_1e_minus_10(self):
        """_EPS_F must remain 1e-10 for magnitude/log-magnitude paths."""
        assert _EPS_F == 1e-10

    def test_band_energy_eps_is_1e_minus_8(self):
        """_BAND_ENERGY_EPS_F must remain 1e-8 for reduction/denominator paths."""
        assert _BAND_ENERGY_EPS_F == 1e-8
