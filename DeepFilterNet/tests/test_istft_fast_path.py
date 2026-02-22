"""Tests for iSTFT optimized overlap-add fast path.

Validates the cached window normalization and vectorized overlap-add
optimizations in df_mlx.ops.istft.
"""

import mlx.core as mx
import numpy as np
import pytest

from df_mlx.ops import _cached_window_norm, istft, stft

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(batch: int, length: int, seed: int = 42) -> mx.array:
    """Generate a deterministic random signal for testing."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((batch, length)).astype(np.float32)
    return mx.array(data)


def _numpy_istft_reference(
    real: np.ndarray,
    imag: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_type: str,
    center: bool,
) -> np.ndarray:
    """Minimal NumPy reference iSTFT for parity checking."""
    if window_type == "sqrt_hann":
        win = np.sqrt(np.hanning(win_length)).astype(np.float32)
    elif window_type == "hann":
        win = np.hanning(win_length).astype(np.float32)
    else:
        raise ValueError(f"Unsupported window: {window_type}")

    if win_length < n_fft:
        pad_l = (n_fft - win_length) // 2
        pad_r = n_fft - win_length - pad_l
        win = np.pad(win, (pad_l, pad_r))

    complex_spec = real + 1j * imag
    batch_size, num_frames, _ = complex_spec.shape
    frames = np.fft.irfft(complex_spec, n=n_fft, axis=-1).astype(np.float32)
    frames *= win

    output_length = (num_frames - 1) * hop_length + n_fft
    output = np.zeros((batch_size, output_length), dtype=np.float32)
    window_sum = np.zeros(output_length, dtype=np.float32)
    win_sq = win * win

    for i in range(num_frames):
        start = i * hop_length
        output[:, start : start + n_fft] += frames[:, i, :]
        window_sum[start : start + n_fft] += win_sq

    window_sum = np.maximum(window_sum, 1e-8)
    output /= window_sum[None, :]

    if center:
        pad_amount = n_fft // 2
        output = output[:, pad_amount:-pad_amount]

    return output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCachedWindowNorm:
    """Tests for _cached_window_norm caching and correctness."""

    def test_same_object_on_repeat_call(self) -> None:
        """Calling with identical args should return the cached array."""
        _cached_window_norm.cache_clear()
        a = _cached_window_norm("sqrt_hann", 960, 960, 480, 20)
        b = _cached_window_norm("sqrt_hann", 960, 960, 480, 20)
        assert a is b

    def test_different_params_differ(self) -> None:
        """Different parameters should produce different arrays."""
        _cached_window_norm.cache_clear()
        a = _cached_window_norm("sqrt_hann", 960, 960, 480, 20)
        b = _cached_window_norm("sqrt_hann", 960, 960, 240, 20)
        assert a.shape != b.shape or not mx.array_equal(a, b)

    def test_output_shape(self) -> None:
        num_frames = 30
        n_fft = 960
        hop = 480
        out = _cached_window_norm("sqrt_hann", n_fft, n_fft, hop, num_frames)
        expected_len = (num_frames - 1) * hop + n_fft
        assert out.shape == (1, expected_len)

    def test_minimum_clamp(self) -> None:
        """All values should be >= 1e-8 (the clamp floor)."""
        out = _cached_window_norm("sqrt_hann", 960, 960, 480, 10)
        assert mx.all(out >= 1e-8).item()


class TestIstftRoundTrip:
    """STFT -> iSTFT round-trip recovery tests."""

    @pytest.mark.parametrize(
        "n_fft,hop",
        [
            (960, 480),  # nover=2 (default config)
            (960, 240),  # nover=4
            (480, 160),  # nover=3
        ],
    )
    def test_round_trip(self, n_fft: int, hop: int) -> None:
        """stft -> istft should recover the original signal within tolerance."""
        _cached_window_norm.cache_clear()
        length = 48000
        sig = _make_signal(1, length)

        spec = stft(sig, n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=True)
        reconstructed = istft(spec, n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=True, length=length)

        mx.eval(reconstructed)
        np.testing.assert_allclose(np.array(reconstructed), np.array(sig), atol=1e-4, rtol=1e-4)

    def test_round_trip_no_center(self) -> None:
        """Round-trip with center=False (skip boundary samples lacking full overlap)."""
        _cached_window_norm.cache_clear()
        n_fft, hop, length = 960, 480, 48000
        sig = _make_signal(1, length)

        spec = stft(sig, n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=False)
        reconstructed = istft(spec, n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=False)

        mx.eval(reconstructed)
        recovered_len = min(reconstructed.shape[-1], length)
        # Skip first/last n_fft samples — without center padding the
        # boundary frames don't have full overlap for perfect reconstruction.
        margin = n_fft
        np.testing.assert_allclose(
            np.array(reconstructed)[..., margin : recovered_len - margin],
            np.array(sig)[..., margin : recovered_len - margin],
            atol=1e-4,
            rtol=1e-4,
        )


class TestIstftNumpyParity:
    """Parity tests against a reference NumPy iSTFT."""

    def test_parity_960_480(self) -> None:
        """Default config: n_fft=960, hop=480."""
        _cached_window_norm.cache_clear()
        n_fft, hop, win_len = 960, 480, 960
        sig = _make_signal(2, 48000)

        spec = stft(sig, n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=True)
        real_np = np.array(spec[0])
        imag_np = np.array(spec[1])

        mlx_out = istft(spec, n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=True)
        mx.eval(mlx_out)

        np_out = _numpy_istft_reference(real_np, imag_np, n_fft, hop, win_len, "sqrt_hann", center=True)

        np.testing.assert_allclose(np.array(mlx_out), np_out, atol=1e-4, rtol=1e-4)


class TestIstftBatchDim:
    """Batch dimension handling."""

    @pytest.mark.parametrize("batch", [1, 4])
    def test_batch_sizes(self, batch: int) -> None:
        _cached_window_norm.cache_clear()
        n_fft, hop, length = 960, 480, 48000
        sig = _make_signal(batch, length)

        spec = stft(sig, n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=True)
        reconstructed = istft(spec, n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=True, length=length)

        mx.eval(reconstructed)
        assert reconstructed.shape == (batch, length)
        np.testing.assert_allclose(np.array(reconstructed), np.array(sig), atol=1e-4, rtol=1e-4)


class TestIstftEdgeCases:
    """Edge cases for iSTFT."""

    def test_single_frame(self) -> None:
        """num_frames=1 should not crash."""
        _cached_window_norm.cache_clear()
        n_fft, hop = 960, 480
        n_freqs = n_fft // 2 + 1
        real = mx.zeros((1, 1, n_freqs))
        imag = mx.zeros((1, 1, n_freqs))

        result = istft((real, imag), n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=False)
        mx.eval(result)
        assert result.shape[-1] == n_fft

    def test_length_trimming(self) -> None:
        """The length parameter should trim the output correctly."""
        _cached_window_norm.cache_clear()
        n_fft, hop, target = 960, 480, 12345
        sig = _make_signal(1, 48000)

        spec = stft(sig, n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=True)
        out = istft(spec, n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=True, length=target)
        mx.eval(out)
        assert out.shape[-1] == target

    def test_1d_input(self) -> None:
        """Unbatched (2-D spec) input should work and return 1-D output."""
        _cached_window_norm.cache_clear()
        n_fft, hop = 960, 480
        sig = _make_signal(1, 48000)
        sig_1d = sig.squeeze(0)

        spec = stft(sig_1d, n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=True)
        out = istft(spec, n_fft=n_fft, hop_length=hop, window="sqrt_hann", center=True)
        mx.eval(out)
        assert out.ndim == 1


class TestIstftFallbackPath:
    """Ensure the fallback (non-integer overlap ratio) path still works."""

    def test_non_integer_ratio(self) -> None:
        """n_fft=1000, hop=300 -> 1000 % 300 != 0 -> fallback."""
        _cached_window_norm.cache_clear()
        n_fft, hop, length = 1000, 300, 48000
        sig = _make_signal(1, length)

        spec = stft(sig, n_fft=n_fft, hop_length=hop, window="hann", center=True)
        out = istft(spec, n_fft=n_fft, hop_length=hop, window="hann", center=True, length=length)
        mx.eval(out)
        assert out.shape[-1] == length
        np.testing.assert_allclose(np.array(out), np.array(sig), atol=1e-3, rtol=1e-3)
