"""Tests for the fused mel power+log Metal kernel.

Validates numerical parity between the Metal kernel and the pure-MLX fallback
path across a range of shapes, and checks integration with the full DNSMOS
proxy pipeline.
"""

import mlx.core as mx
import numpy as np
import pytest

from df_mlx.kernels import mel_power_log_kernel, metal_kernels_available

KERNEL_AVAILABLE = metal_kernels_available()
requires_kernel = pytest.mark.skipif(not KERNEL_AVAILABLE, reason="Metal kernels unavailable")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _reference_mel(
    spec_complex: mx.array,
    mel_fb: mx.array,
) -> mx.array:
    """Pure-MLX reference: power -> mel projection -> log."""
    power = mx.abs(spec_complex) ** 2
    mel_spec = mx.matmul(power, mx.transpose(mel_fb))
    return mx.log(mx.maximum(mel_spec, 1e-10))


def _random_mel_fb(n_mels: int, n_freqs: int) -> mx.array:
    """Create a random non-negative mel filterbank for testing."""
    return mx.abs(mx.random.normal((n_mels, n_freqs)))


# ------------------------------------------------------------------
# Kernel wrapper direct tests
# ------------------------------------------------------------------


@requires_kernel
@pytest.mark.parametrize(
    "batch_size, n_frames, n_fft, n_mels",
    [
        (1, 10, 512, 64),
        (2, 20, 256, 40),
        (4, 5, 1024, 80),
        (1, 1, 128, 16),
        (3, 50, 400, 64),
    ],
)
def test_kernel_vs_reference_parity(
    batch_size: int,
    n_frames: int,
    n_fft: int,
    n_mels: int,
) -> None:
    """Kernel output matches the pure-MLX reference within tolerance."""
    n_freqs = n_fft // 2 + 1
    spec_complex = mx.random.normal((batch_size, n_frames, n_freqs)) + 1j * mx.random.normal(
        (batch_size, n_frames, n_freqs)
    )
    mel_fb = _random_mel_fb(n_mels, n_freqs)

    expected = _reference_mel(spec_complex, mel_fb)

    spec_real = mx.real(spec_complex)
    spec_imag = mx.imag(spec_complex)
    result = mel_power_log_kernel(spec_real, spec_imag, mel_fb, batch_size, n_frames, n_mels)

    mx.eval(expected, result)
    np.testing.assert_allclose(
        np.array(result),
        np.array(expected),
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"Parity failed for shape ({batch_size}, {n_frames}, {n_fft}, {n_mels})",
    )


@requires_kernel
def test_kernel_output_shape() -> None:
    """Kernel produces the expected output shape."""
    batch_size, n_frames, n_fft, n_mels = 2, 15, 512, 64
    n_freqs = n_fft // 2 + 1

    spec_real = mx.random.normal((batch_size, n_frames, n_freqs))
    spec_imag = mx.random.normal((batch_size, n_frames, n_freqs))
    mel_fb = _random_mel_fb(n_mels, n_freqs)

    result = mel_power_log_kernel(spec_real, spec_imag, mel_fb, batch_size, n_frames, n_mels)
    mx.eval(result)

    assert result.shape == (batch_size, n_frames, n_mels)


@requires_kernel
def test_kernel_log_floor() -> None:
    """Kernel applies log(max(x, 1e-10)) correctly for near-zero power."""
    batch_size, n_frames, n_freqs, n_mels = 1, 5, 33, 8
    spec_real = mx.zeros((batch_size, n_frames, n_freqs))
    spec_imag = mx.zeros((batch_size, n_frames, n_freqs))
    mel_fb = _random_mel_fb(n_mels, n_freqs)

    result = mel_power_log_kernel(spec_real, spec_imag, mel_fb, batch_size, n_frames, n_mels)
    mx.eval(result)

    expected_floor = float(np.log(1e-10))
    np.testing.assert_allclose(
        np.array(result),
        expected_floor,
        atol=1e-4,
        err_msg="Log floor not applied correctly for zero-power input",
    )


def test_kernel_raises_without_metal(monkeypatch: pytest.MonkeyPatch) -> None:
    """mel_power_log_kernel raises RuntimeError when Metal is unavailable."""
    import df_mlx.kernels as kmod

    monkeypatch.setattr(kmod, "_mel_power_log_kernel", None)

    with pytest.raises(RuntimeError, match="metal_kernel"):
        mel_power_log_kernel(
            mx.zeros((1, 1, 5)),
            mx.zeros((1, 1, 5)),
            mx.zeros((4, 5)),
            batch_size=1,
            n_frames=1,
            n_mels=4,
        )


# ------------------------------------------------------------------
# MelSpectrogram integration tests
# ------------------------------------------------------------------


@requires_kernel
def test_mel_spectrogram_kernel_vs_fallback() -> None:
    """MelSpectrogram with kernel matches fallback output."""
    from df_mlx.dnsmos_proxy import MelSpectrogram

    audio = mx.random.normal((2, 8000))

    mel_kernel = MelSpectrogram(n_fft=512, hop_length=256, n_mels=64, use_metal_kernel=True)
    mel_fallback = MelSpectrogram(n_fft=512, hop_length=256, n_mels=64, use_metal_kernel=False)

    # Share filterbank and window so only the computation path differs
    mel_fallback._mel_fb = mel_kernel._mel_fb
    mel_fallback._window = mel_kernel._window

    out_kernel = mel_kernel(audio)
    out_fallback = mel_fallback(audio)
    mx.eval(out_kernel, out_fallback)

    np.testing.assert_allclose(
        np.array(out_kernel),
        np.array(out_fallback),
        rtol=1e-4,
        atol=1e-5,
    )


def test_mel_spectrogram_fallback_flag() -> None:
    """MelSpectrogram with use_metal_kernel=False never uses the kernel path."""
    from df_mlx.dnsmos_proxy import MelSpectrogram

    mel = MelSpectrogram(use_metal_kernel=False)
    assert not mel._use_kernel

    audio = mx.random.normal((1, 4000))
    out = mel(audio)
    mx.eval(out)
    assert out.ndim == 3


@requires_kernel
@pytest.mark.parametrize(
    "n_fft, hop, n_mels",
    [
        (256, 128, 40),
        (400, 160, 64),
        (512, 256, 80),
        (1024, 512, 128),
    ],
)
def test_mel_spectrogram_various_configs(n_fft: int, hop: int, n_mels: int) -> None:
    """MelSpectrogram kernel path produces correct shapes for various configs."""
    from df_mlx.dnsmos_proxy import MelSpectrogram

    audio = mx.random.normal((2, 16000))
    mel = MelSpectrogram(n_fft=n_fft, hop_length=hop, n_mels=n_mels, use_metal_kernel=True)
    out = mel(audio)
    mx.eval(out)

    expected_frames = (16000 - n_fft) // hop + 1
    assert out.shape == (2, n_mels, expected_frames)


# ------------------------------------------------------------------
# Full DNSMOS pipeline smoke test
# ------------------------------------------------------------------


@requires_kernel
def test_dnsmos_proxy_pipeline() -> None:
    """Full DNSMOSProxy pipeline runs end-to-end with the kernel path."""
    from df_mlx.dnsmos_proxy import DNSMOSProxy

    proxy = DNSMOSProxy()
    audio = mx.random.normal((1, 16000))
    scores = proxy(audio)
    mx.eval(scores["sig"], scores["bak"], scores["ovl"])

    for key in ("sig", "bak", "ovl"):
        assert scores[key].shape == (1,), f"{key} has wrong shape: {scores[key].shape}"
