import mlx.core as mx

from df_mlx.loss import si_sdr
from df_mlx.train import spectral_loss
from df_mlx.train_dynamic import _build_speech_band_mask, _compute_awesome_losses


def _finite(x: mx.array) -> bool:
    return bool(mx.all(mx.isfinite(x)))


def test_spectral_loss_fp16_finite():
    pred_real = mx.ones((2, 4, 257), dtype=mx.float16) * mx.array(600.0, dtype=mx.float16)
    pred_imag = mx.ones((2, 4, 257), dtype=mx.float16) * mx.array(600.0, dtype=mx.float16)
    target_real = pred_real * mx.array(0.95, dtype=mx.float16)
    target_imag = pred_imag * mx.array(0.95, dtype=mx.float16)

    loss = spectral_loss((pred_real, pred_imag), (target_real, target_imag))
    assert _finite(loss)


def test_awesome_loss_fp16_finite():
    batch, frames, freqs = 2, 4, 257
    clean_real = mx.ones((batch, frames, freqs), dtype=mx.float16) * mx.array(600.0, dtype=mx.float16)
    clean_imag = mx.ones((batch, frames, freqs), dtype=mx.float16) * mx.array(600.0, dtype=mx.float16)
    noisy_real = clean_real + mx.ones((batch, frames, freqs), dtype=mx.float16) * mx.array(10.0, dtype=mx.float16)
    noisy_imag = clean_imag + mx.ones((batch, frames, freqs), dtype=mx.float16) * mx.array(10.0, dtype=mx.float16)
    out_real = clean_real * mx.array(0.9, dtype=mx.float16)
    out_imag = clean_imag * mx.array(0.9, dtype=mx.float16)
    snr = mx.zeros((batch,), dtype=mx.float32)

    band_mask, band_bins = _build_speech_band_mask(freqs, 48000, 300.0, 3400.0)

    (
        awesome_loss,
        _,
        _,
        _,
        mask,
        proxy_frame,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _compute_awesome_losses(
        noisy_real,
        noisy_imag,
        clean_real,
        clean_imag,
        out_real,
        out_imag,
        snr,
        band_mask,
        band_bins,
        6.0,
        0.0,
        1.0,
        -10.0,
        6.0,
        True,
    )

    assert _finite(awesome_loss)
    assert _finite(mask)
    assert _finite(proxy_frame)


def test_sisdr_fp16_finite():
    pred = mx.ones((1, 2048), dtype=mx.float16) * mx.array(0.1, dtype=mx.float16)
    target = mx.zeros((1, 2048), dtype=mx.float16)
    val = si_sdr(pred, target)
    assert _finite(val)
