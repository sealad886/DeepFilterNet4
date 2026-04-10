"""Tests for FusedMultiResSpectralLoss parity with FusedSpectralLoss."""

import mlx.core as mx
import pytest

from df_mlx.loss import FusedMultiResSpectralLoss, FusedSpectralLoss, SpectralLoss

ATOL = 1e-5
RTOL = 1e-4
SAMPLE_LEN = 4096
FFT_SIZES_DEFAULT = (512, 1024, 2048)


def _random_waveform(batch: int = 2, length: int = SAMPLE_LEN) -> mx.array:
    return mx.random.normal((batch, length))


def _assert_close(
    a: mx.array, b: mx.array, atol: float = ATOL, rtol: float = RTOL
) -> None:
    diff = mx.abs(a - b)
    tol = atol + rtol * mx.abs(b)
    assert bool(mx.all(diff <= tol)), (
        f"Max diff {float(mx.max(diff)):.2e}, "
        f"max tol {float(mx.max(tol)):.2e}, "
        f"a={float(a):.6f}, b={float(b):.6f}"
    )


class TestParityWithFusedSpectralLoss:
    """Verify FusedMultiResSpectralLoss matches FusedSpectralLoss numerically."""

    def test_parity_no_gamma_no_complex(self):
        pred, target = _random_waveform(), _random_waveform()
        fused = FusedSpectralLoss(fft_sizes=FFT_SIZES_DEFAULT)
        multi_res = FusedMultiResSpectralLoss(fft_sizes=FFT_SIZES_DEFAULT)

        loss_fused = fused(pred, target)
        loss_multi_res = multi_res(pred, target)
        mx.eval(loss_fused, loss_multi_res)
        _assert_close(loss_multi_res, loss_fused)

    def test_parity_with_gamma(self):
        pred, target = _random_waveform(), _random_waveform()
        kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "gamma": 0.3}
        fused = FusedSpectralLoss(**kwargs)
        multi_res = FusedMultiResSpectralLoss(**kwargs)

        loss_fused = fused(pred, target)
        loss_multi_res = multi_res(pred, target)
        mx.eval(loss_fused, loss_multi_res)
        _assert_close(loss_multi_res, loss_fused)

    def test_parity_with_complex_loss(self):
        pred, target = _random_waveform(), _random_waveform()
        kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "factor_complex": 0.5}
        fused = FusedSpectralLoss(**kwargs)
        multi_res = FusedMultiResSpectralLoss(**kwargs)

        loss_fused = fused(pred, target)
        loss_multi_res = multi_res(pred, target)
        mx.eval(loss_fused, loss_multi_res)
        _assert_close(loss_multi_res, loss_fused)

    def test_parity_with_gamma_and_complex(self):
        pred, target = _random_waveform(), _random_waveform()
        kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "gamma": 0.3, "factor_complex": 0.5}
        fused = FusedSpectralLoss(**kwargs)
        multi_res = FusedMultiResSpectralLoss(**kwargs)

        loss_fused = fused(pred, target)
        loss_multi_res = multi_res(pred, target)
        mx.eval(loss_fused, loss_multi_res)
        _assert_close(loss_multi_res, loss_fused)

    def test_parity_matches_original_spectral_loss(self):
        pred, target = _random_waveform(), _random_waveform()
        kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "gamma": 0.3, "factor_complex": 0.5}
        orig = SpectralLoss(**kwargs)
        multi_res = FusedMultiResSpectralLoss(**kwargs)

        loss_orig = orig(pred, target)
        loss_multi_res = multi_res(pred, target)
        mx.eval(loss_orig, loss_multi_res)
        _assert_close(loss_multi_res, loss_orig)

    def test_parity_different_fft_sizes(self):
        pred, target = _random_waveform(length=8192), _random_waveform(length=8192)
        fft_sizes = (256, 512, 4096)
        kwargs = {"fft_sizes": fft_sizes, "gamma": 0.5}
        fused = FusedSpectralLoss(**kwargs)
        multi_res = FusedMultiResSpectralLoss(**kwargs)

        loss_fused = fused(pred, target)
        loss_multi_res = multi_res(pred, target)
        mx.eval(loss_fused, loss_multi_res)
        _assert_close(loss_multi_res, loss_fused)

    def test_parity_single_resolution(self):
        pred, target = _random_waveform(), _random_waveform()
        kwargs = {"fft_sizes": (1024,), "factor_complex": 0.3, "gamma": 0.6}
        fused = FusedSpectralLoss(**kwargs)
        multi_res = FusedMultiResSpectralLoss(**kwargs)

        loss_fused = fused(pred, target)
        loss_multi_res = multi_res(pred, target)
        mx.eval(loss_fused, loss_multi_res)
        _assert_close(loss_multi_res, loss_fused)


class TestGradientParity:
    """Verify gradients are correct."""

    def test_gradient_parity_with_fused(self):
        mx.random.seed(42)
        pred = mx.random.normal((1, SAMPLE_LEN))
        target = mx.random.normal((1, SAMPLE_LEN))

        kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "gamma": 0.3, "factor_complex": 0.5}
        fused = FusedSpectralLoss(**kwargs)
        multi_res = FusedMultiResSpectralLoss(**kwargs)

        grad_fused = mx.grad(lambda p: fused(p, target))(pred)
        grad_multi_res = mx.grad(lambda p: multi_res(p, target))(pred)
        mx.eval(grad_fused, grad_multi_res)

        diff = mx.max(mx.abs(grad_fused - grad_multi_res))
        scale = mx.maximum(mx.max(mx.abs(grad_fused)), mx.array(1e-8))
        rel = diff / scale
        mx.eval(diff, rel)
        assert float(rel) < 1e-3, (
            f"Gradient relative diff {float(rel):.2e} exceeds 1e-3"
        )

    def test_gradient_parity_with_original(self):
        mx.random.seed(42)
        pred = mx.random.normal((1, SAMPLE_LEN))
        target = mx.random.normal((1, SAMPLE_LEN))

        kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "gamma": 0.3, "factor_complex": 0.5}
        orig = SpectralLoss(**kwargs)
        multi_res = FusedMultiResSpectralLoss(**kwargs)

        grad_orig = mx.grad(lambda p: orig(p, target))(pred)
        grad_multi_res = mx.grad(lambda p: multi_res(p, target))(pred)
        mx.eval(grad_orig, grad_multi_res)

        diff = mx.max(mx.abs(grad_orig - grad_multi_res))
        scale = mx.maximum(mx.max(mx.abs(grad_orig)), mx.array(1e-8))
        rel = diff / scale
        mx.eval(diff, rel)
        assert float(rel) < 1e-3, (
            f"Gradient relative diff {float(rel):.2e} exceeds 1e-3"
        )


class TestVariousShapes:
    """Test with different batch sizes and sample counts."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_different_batch_sizes(self, batch_size: int):
        pred = _random_waveform(batch=batch_size)
        target = _random_waveform(batch=batch_size)
        kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "gamma": 0.3}
        fused = FusedSpectralLoss(**kwargs)
        multi_res = FusedMultiResSpectralLoss(**kwargs)

        loss_fused = fused(pred, target)
        loss_multi_res = multi_res(pred, target)
        mx.eval(loss_fused, loss_multi_res)
        _assert_close(loss_multi_res, loss_fused)

    @pytest.mark.parametrize("length", [1024, 2048, 4096, 8192])
    def test_different_sample_lengths(self, length: int):
        pred = _random_waveform(length=length)
        target = _random_waveform(length=length)
        kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "gamma": 0.3, "factor_complex": 0.5}
        fused = FusedSpectralLoss(**kwargs)
        multi_res = FusedMultiResSpectralLoss(**kwargs)

        loss_fused = fused(pred, target)
        loss_multi_res = multi_res(pred, target)
        mx.eval(loss_fused, loss_multi_res)
        _assert_close(loss_multi_res, loss_fused)

    def test_1d_input(self):
        signal_1d = mx.random.normal((SAMPLE_LEN,))
        signal_2d = mx.expand_dims(signal_1d, axis=0)
        target_1d = mx.random.normal((SAMPLE_LEN,))
        target_2d = mx.expand_dims(target_1d, axis=0)

        multi_res = FusedMultiResSpectralLoss(fft_sizes=FFT_SIZES_DEFAULT)
        loss_1d = multi_res(signal_1d, target_1d)
        loss_2d = multi_res(signal_2d, target_2d)
        mx.eval(loss_1d, loss_2d)
        _assert_close(loss_1d, loss_2d)


class TestFiniteOutput:
    """Verify no NaN/Inf outputs."""

    def test_no_nan_inf_zeros(self):
        pred = mx.zeros((2, SAMPLE_LEN))
        target = mx.zeros((2, SAMPLE_LEN))
        multi_res = FusedMultiResSpectralLoss(fft_sizes=FFT_SIZES_DEFAULT)
        loss = multi_res(pred, target)
        mx.eval(loss)
        assert mx.all(mx.isfinite(loss)), f"Loss contains NaN/Inf: {float(loss)}"

    def test_no_nan_inf_ones(self):
        pred = mx.ones((2, SAMPLE_LEN))
        target = mx.ones((2, SAMPLE_LEN))
        multi_res = FusedMultiResSpectralLoss(fft_sizes=FFT_SIZES_DEFAULT)
        loss = multi_res(pred, target)
        mx.eval(loss)
        assert mx.all(mx.isfinite(loss)), f"Loss contains NaN/Inf: {float(loss)}"

    def test_no_nan_inf_random(self):
        pred = _random_waveform()
        target = _random_waveform()
        kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "gamma": 0.3, "factor_complex": 0.5}
        multi_res = FusedMultiResSpectralLoss(**kwargs)
        loss = multi_res(pred, target)
        mx.eval(loss)
        assert mx.all(mx.isfinite(loss)), f"Loss contains NaN/Inf: {float(loss)}"

    def test_no_nan_inf_extreme_values(self):
        pred = mx.random.uniform(-1e3, 1e3, (2, SAMPLE_LEN))
        target = mx.random.uniform(-1e3, 1e3, (2, SAMPLE_LEN))
        multi_res = FusedMultiResSpectralLoss(fft_sizes=FFT_SIZES_DEFAULT)
        loss = multi_res(pred, target)
        mx.eval(loss)
        assert mx.all(mx.isfinite(loss)), f"Loss contains NaN/Inf: {float(loss)}"


class TestCompiledCompute:
    """Verify compiled compute attribute exists and is callable."""

    def test_compiled_compute_attribute(self):
        multi_res = FusedMultiResSpectralLoss()
        assert hasattr(multi_res, "_compiled_compute")
        assert callable(multi_res._compiled_compute)

    def test_windows_precomputed(self):
        multi_res = FusedMultiResSpectralLoss(fft_sizes=(512, 1024))
        assert len(multi_res._windows) == 2
        for window in multi_res._windows:
            assert isinstance(window, mx.array)
