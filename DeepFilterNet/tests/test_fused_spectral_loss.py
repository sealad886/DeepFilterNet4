"""Tests for FusedSpectralLoss parity with SpectralLoss."""

import mlx.core as mx
import pytest

from df_mlx.loss import FusedSpectralLoss, SpectralLoss

ATOL = 1e-5
RTOL = 1e-4
SAMPLE_LEN = 4096
FFT_SIZES_DEFAULT = (512, 1024, 2048)


def _random_waveform(batch: int = 2, length: int = SAMPLE_LEN) -> mx.array:
    return mx.random.normal((batch, length))


def _assert_close(a: mx.array, b: mx.array, atol: float = ATOL, rtol: float = RTOL) -> None:
    diff = mx.abs(a - b)
    tol = atol + rtol * mx.abs(b)
    assert bool(mx.all(diff <= tol)), (
        f"Max diff {float(mx.max(diff)):.2e}, "
        f"max tol {float(mx.max(tol)):.2e}, "
        f"a={float(a):.6f}, b={float(b):.6f}"
    )


# ------------------------------------------------------------------
# 1. Basic parity (no gamma, no complex loss)
# ------------------------------------------------------------------


def test_fused_matches_original():
    pred, target = _random_waveform(), _random_waveform()
    orig = SpectralLoss(fft_sizes=FFT_SIZES_DEFAULT)
    fused = FusedSpectralLoss(fft_sizes=FFT_SIZES_DEFAULT)

    loss_orig = orig(pred, target)
    loss_fused = fused(pred, target)
    mx.eval(loss_orig, loss_fused)
    _assert_close(loss_fused, loss_orig)


# ------------------------------------------------------------------
# 2. Gamma compression parity
# ------------------------------------------------------------------


def test_fused_matches_with_gamma():
    pred, target = _random_waveform(), _random_waveform()
    kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "gamma": 0.3}
    loss_orig = SpectralLoss(**kwargs)(pred, target)
    loss_fused = FusedSpectralLoss(**kwargs)(pred, target)
    mx.eval(loss_orig, loss_fused)
    _assert_close(loss_fused, loss_orig)


# ------------------------------------------------------------------
# 3. Complex loss parity
# ------------------------------------------------------------------


def test_fused_matches_with_complex():
    pred, target = _random_waveform(), _random_waveform()
    kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "factor_complex": 0.5}
    loss_orig = SpectralLoss(**kwargs)(pred, target)
    loss_fused = FusedSpectralLoss(**kwargs)(pred, target)
    mx.eval(loss_orig, loss_fused)
    _assert_close(loss_fused, loss_orig)


# ------------------------------------------------------------------
# 4. Full config parity (gamma + complex)
# ------------------------------------------------------------------


def test_fused_matches_full():
    pred, target = _random_waveform(), _random_waveform()
    kwargs = {
        "fft_sizes": FFT_SIZES_DEFAULT,
        "gamma": 0.3,
        "factor": 1.0,
        "factor_complex": 0.5,
    }
    loss_orig = SpectralLoss(**kwargs)(pred, target)
    loss_fused = FusedSpectralLoss(**kwargs)(pred, target)
    mx.eval(loss_orig, loss_fused)
    _assert_close(loss_fused, loss_orig)


# ------------------------------------------------------------------
# 5. 1D input produces same result as 2D
# ------------------------------------------------------------------


def test_fused_1d_input():
    signal_1d = mx.random.normal((SAMPLE_LEN,))
    signal_2d = mx.expand_dims(signal_1d, axis=0)
    target_1d = mx.random.normal((SAMPLE_LEN,))
    target_2d = mx.expand_dims(target_1d, axis=0)

    fused = FusedSpectralLoss(fft_sizes=FFT_SIZES_DEFAULT)
    loss_1d = fused(signal_1d, target_1d)
    loss_2d = fused(signal_2d, target_2d)
    mx.eval(loss_1d, loss_2d)
    _assert_close(loss_1d, loss_2d)


# ------------------------------------------------------------------
# 6. Non-default FFT sizes
# ------------------------------------------------------------------


def test_fused_different_fft_sizes():
    pred, target = _random_waveform(length=8192), _random_waveform(length=8192)
    fft_sizes = (256, 512, 4096)
    kwargs = {"fft_sizes": fft_sizes, "gamma": 0.5}
    loss_orig = SpectralLoss(**kwargs)(pred, target)
    loss_fused = FusedSpectralLoss(**kwargs)(pred, target)
    mx.eval(loss_orig, loss_fused)
    _assert_close(loss_fused, loss_orig)


# ------------------------------------------------------------------
# 7. Single resolution
# ------------------------------------------------------------------


def test_fused_single_resolution():
    pred, target = _random_waveform(), _random_waveform()
    kwargs = {"fft_sizes": (1024,), "factor_complex": 0.3, "gamma": 0.6}
    loss_orig = SpectralLoss(**kwargs)(pred, target)
    loss_fused = FusedSpectralLoss(**kwargs)(pred, target)
    mx.eval(loss_orig, loss_fused)
    _assert_close(loss_fused, loss_orig)


# ------------------------------------------------------------------
# 8. Gradient parity
# ------------------------------------------------------------------


def test_fused_gradient_parity():
    mx.random.seed(42)
    pred = mx.random.normal((1, SAMPLE_LEN))
    target = mx.random.normal((1, SAMPLE_LEN))

    kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "gamma": 0.3, "factor_complex": 0.5}
    orig = SpectralLoss(**kwargs)
    fused = FusedSpectralLoss(**kwargs)

    grad_orig = mx.grad(lambda p: orig(p, target))(pred)
    grad_fused = mx.grad(lambda p: fused(p, target))(pred)
    mx.eval(grad_orig, grad_fused)

    diff = mx.max(mx.abs(grad_orig - grad_fused))
    scale = mx.maximum(mx.max(mx.abs(grad_orig)), mx.array(1e-8))
    rel = diff / scale
    mx.eval(diff, rel)
    assert float(rel) < 1e-3, f"Gradient relative diff {float(rel):.2e} exceeds 1e-3"


# ------------------------------------------------------------------
# 9. _compiled_loss attribute exists and is callable
# ------------------------------------------------------------------


def test_fused_loss_is_compiled():
    fused = FusedSpectralLoss()
    assert hasattr(fused, "_compiled_loss")
    assert callable(fused._compiled_loss)


# ------------------------------------------------------------------
# 10. Multiple batch sizes
# ------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fused_batch_sizes(batch_size: int):
    pred = _random_waveform(batch=batch_size)
    target = _random_waveform(batch=batch_size)
    kwargs = {"fft_sizes": FFT_SIZES_DEFAULT, "gamma": 0.3}
    loss_orig = SpectralLoss(**kwargs)(pred, target)
    loss_fused = FusedSpectralLoss(**kwargs)(pred, target)
    mx.eval(loss_orig, loss_fused)
    _assert_close(loss_fused, loss_orig)
