"""Correctness tests for fused Metal GPU kernels.

Validates that each custom Metal kernel produces results numerically
equivalent to the reference pure-MLX implementation, for both forward
and backward (VJP) passes.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from df_mlx.metal_kernels import (
    _ref_band_energy,
    _ref_complex_mag,
    _ref_log1p_mag,
    fused_band_energy,
    fused_complex_mag,
    fused_log1p_mag,
)
from df_mlx.train import spectral_loss


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
# fused_log1p_mag — forward
# ==================================================================


class TestFusedLog1pMag:
    def test_forward_matches_reference(self):
        """Forward pass matches pure-MLX reference."""
        real, imag = _rand_complex((4, 100, 64))
        fused = fused_log1p_mag(real, imag)
        ref = _ref_log1p_mag(real, imag)
        mx.eval(fused, ref)
        np.testing.assert_allclose(np.array(fused), np.array(ref), rtol=1e-5, atol=1e-6)

    def test_forward_scalar(self):
        """Single-element input."""
        real = mx.array([1.0])
        imag = mx.array([1.0])
        fused = fused_log1p_mag(real, imag)
        ref = _ref_log1p_mag(real, imag)
        mx.eval(fused, ref)
        np.testing.assert_allclose(np.array(fused), np.array(ref), rtol=1e-5, atol=1e-6)

    def test_forward_zeros(self):
        """All-zero input (tests eps stability)."""
        real = mx.zeros((2, 10, 32))
        imag = mx.zeros((2, 10, 32))
        fused = fused_log1p_mag(real, imag)
        ref = _ref_log1p_mag(real, imag)
        mx.eval(fused, ref)
        np.testing.assert_allclose(np.array(fused), np.array(ref), rtol=1e-5, atol=1e-6)

    def test_forward_respects_custom_eps(self):
        """Kernel output changes with caller-provided epsilon."""
        real, imag = _rand_complex((2, 8, 8))
        eps = 1e-4
        fused = fused_log1p_mag(real, imag, eps=eps)
        ref = mx.log1p(mx.sqrt(real * real + imag * imag + eps))
        mx.eval(fused, ref)
        np.testing.assert_allclose(np.array(fused), np.array(ref), rtol=1e-5, atol=1e-6)

    def test_forward_preserves_shape(self):
        """Output shape matches input shape."""
        real, imag = _rand_complex((3, 50, 128))
        fused = fused_log1p_mag(real, imag)
        mx.eval(fused)
        assert fused.shape == real.shape

    def test_backward_matches_reference(self):
        """Backward (VJP) is numerically consistent with finite differences."""
        real, imag = _rand_complex((2, 10, 16))

        def fused_fn(r, i):
            return mx.sum(fused_log1p_mag(r, i))

        def ref_fn(r, i):
            return mx.sum(_ref_log1p_mag(r, i))

        fused_grad_fn = mx.grad(fused_fn, argnums=[0, 1])
        ref_grad_fn = mx.grad(ref_fn, argnums=[0, 1])

        fg_r, fg_i = fused_grad_fn(real, imag)
        rg_r, rg_i = ref_grad_fn(real, imag)
        mx.eval(fg_r, fg_i, rg_r, rg_i)

        np.testing.assert_allclose(np.array(fg_r), np.array(rg_r), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.array(fg_i), np.array(rg_i), rtol=1e-4, atol=1e-5)


# ==================================================================
# fused_complex_mag — forward + backward
# ==================================================================


class TestFusedComplexMag:
    def test_forward_matches_reference(self):
        real, imag = _rand_complex((4, 100, 64))
        fused = fused_complex_mag(real, imag)
        ref = _ref_complex_mag(real, imag)
        mx.eval(fused, ref)
        np.testing.assert_allclose(np.array(fused), np.array(ref), rtol=1e-5, atol=1e-6)

    def test_forward_zeros(self):
        real = mx.zeros((2, 10, 32))
        imag = mx.zeros((2, 10, 32))
        fused = fused_complex_mag(real, imag)
        ref = _ref_complex_mag(real, imag)
        mx.eval(fused, ref)
        np.testing.assert_allclose(np.array(fused), np.array(ref), rtol=1e-5, atol=1e-6)

    def test_forward_respects_custom_eps(self):
        real, imag = _rand_complex((2, 8, 8))
        eps = 1e-4
        fused = fused_complex_mag(real, imag, eps=eps)
        ref = _ref_complex_mag(real, imag, eps=eps)
        mx.eval(fused, ref)
        np.testing.assert_allclose(np.array(fused), np.array(ref), rtol=1e-5, atol=1e-6)

    def test_backward_matches_reference(self):
        real, imag = _rand_complex((2, 10, 16))

        def fused_fn(r, i):
            return mx.sum(fused_complex_mag(r, i))

        def ref_fn(r, i):
            return mx.sum(_ref_complex_mag(r, i))

        fused_grad_fn = mx.grad(fused_fn, argnums=[0, 1])
        ref_grad_fn = mx.grad(ref_fn, argnums=[0, 1])

        fg_r, fg_i = fused_grad_fn(real, imag)
        rg_r, rg_i = ref_grad_fn(real, imag)
        mx.eval(fg_r, fg_i, rg_r, rg_i)

        np.testing.assert_allclose(np.array(fg_r), np.array(rg_r), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.array(fg_i), np.array(rg_i), rtol=1e-4, atol=1e-5)


class TestSpectralLossWiring:
    def test_train_spectral_loss_matches_reference(self):
        real_pred, imag_pred = _rand_complex((2, 16, 32))
        real_target, imag_target = _rand_complex((2, 16, 32))

        fused = spectral_loss((real_pred, imag_pred), (real_target, imag_target), alpha=0.5)

        pred_mag = _ref_complex_mag(real_pred, imag_pred, eps=1e-8)
        target_mag = _ref_complex_mag(real_target, imag_target, eps=1e-8)
        ref_mag_loss = mx.mean(mx.abs(pred_mag - target_mag))
        ref_complex_loss = mx.mean(mx.abs(real_pred - real_target) + mx.abs(imag_pred - imag_target))
        ref = 0.5 * ref_mag_loss + 0.5 * ref_complex_loss

        mx.eval(fused, ref)
        np.testing.assert_allclose(np.array(fused), np.array(ref), rtol=1e-5, atol=1e-6)


# ==================================================================
# fused_band_energy — forward only
# ==================================================================


class TestFusedBandEnergy:
    @staticmethod
    def _make_band_mask(F: int, active_bins: int = 20):
        """Create a realistic band mask: first `active_bins` bins are 1."""
        mask = np.zeros(F, dtype=np.float32)
        mask[:active_bins] = 1.0
        return mx.array(mask), float(active_bins)

    def test_forward_matches_reference(self):
        B, T, F = 4, 100, 64
        real, imag = _rand_complex((B, T, F))
        mask, bins = self._make_band_mask(F)
        f_band, f_log = fused_band_energy(real, imag, mask, bins)
        r_band, r_log = _ref_band_energy(real, imag, mask, bins)
        mx.eval(f_band, f_log, r_band, r_log)
        np.testing.assert_allclose(np.array(f_band), np.array(r_band), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.array(f_log), np.array(r_log), rtol=1e-4, atol=1e-5)

    def test_output_shapes(self):
        B, T, F = 2, 50, 32
        real, imag = _rand_complex((B, T, F))
        mask, bins = self._make_band_mask(F, active_bins=10)
        f_band, f_log = fused_band_energy(real, imag, mask, bins)
        mx.eval(f_band, f_log)
        assert f_band.shape == (B, T)
        assert f_log.shape == (B, T)

    def test_zeros_input(self):
        B, T, F = 2, 10, 16
        real = mx.zeros((B, T, F))
        imag = mx.zeros((B, T, F))
        mask, bins = self._make_band_mask(F, active_bins=8)
        f_band, f_log = fused_band_energy(real, imag, mask, bins)
        r_band, r_log = _ref_band_energy(real, imag, mask, bins)
        mx.eval(f_band, f_log, r_band, r_log)
        np.testing.assert_allclose(np.array(f_band), np.array(r_band), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.array(f_log), np.array(r_log), rtol=1e-5, atol=1e-6)

    def test_single_frame(self):
        """B=1, T=1 edge case."""
        B, T, F = 1, 1, 32
        real, imag = _rand_complex((B, T, F))
        mask, bins = self._make_band_mask(F)
        f_band, f_log = fused_band_energy(real, imag, mask, bins)
        r_band, r_log = _ref_band_energy(real, imag, mask, bins)
        mx.eval(f_band, f_log, r_band, r_log)
        np.testing.assert_allclose(np.array(f_band), np.array(r_band), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.array(f_log), np.array(r_log), rtol=1e-4, atol=1e-5)

    def test_bfloat16_inputs_accumulate_in_float32(self):
        """Reduced-precision inputs should compile and return stable float32 outputs."""
        B, T, F = 2, 12, 24
        real, imag = _rand_complex((B, T, F), dtype=mx.bfloat16)
        mask, bins = self._make_band_mask(F, active_bins=12)
        f_band, f_log = fused_band_energy(real, imag, mask, bins)
        r_band, r_log = _ref_band_energy(real.astype(mx.float32), imag.astype(mx.float32), mask, bins)
        mx.eval(f_band, f_log, r_band, r_log)
        assert f_band.dtype == mx.float32
        assert f_log.dtype == mx.float32
        np.testing.assert_allclose(np.array(f_band), np.array(r_band), rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(np.array(f_log), np.array(r_log), rtol=2e-3, atol=2e-3)

    def test_full_mask(self):
        """All bins active."""
        B, T, F = 2, 20, 48
        real, imag = _rand_complex((B, T, F))
        mask = mx.ones((F,))
        bins = float(F)
        f_band, f_log = fused_band_energy(real, imag, mask, bins)
        r_band, r_log = _ref_band_energy(real, imag, mask, bins)
        mx.eval(f_band, f_log, r_band, r_log)
        np.testing.assert_allclose(np.array(f_band), np.array(r_band), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.array(f_log), np.array(r_log), rtol=1e-4, atol=1e-5)
