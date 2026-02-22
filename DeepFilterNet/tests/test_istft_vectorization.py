"""Tests for vectorized iSTFT overlap-add implementation."""

import mlx.core as mx
import numpy as np
import pytest

from df_mlx.ops import get_window, istft, stft

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_signal(batch_size: int, length: int, *, seed: int = 42) -> mx.array:
    """Generate a deterministic random signal."""
    np.random.seed(seed)
    return mx.array(np.random.randn(batch_size, length).astype(np.float32))


def _istft_loop(
    spec,
    n_fft: int = 960,
    hop_length: int = 480,
    window: str = "sqrt_hann",
    center: bool = True,
    length=None,
):
    """Reference iSTFT using the original loop-based overlap-add."""
    win_length = n_fft

    if isinstance(spec, tuple):
        real, imag = spec
    else:
        real = spec[..., 0]
        imag = spec[..., 1]

    input_1d = real.ndim == 2
    if input_1d:
        real = mx.expand_dims(real, axis=0)
        imag = mx.expand_dims(imag, axis=0)

    batch_size, num_frames, num_freqs = real.shape
    complex_spec = real + 1j * imag
    frames = mx.fft.irfft(complex_spec, n=n_fft, axis=-1)

    win = get_window(window, win_length)
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        win = mx.pad(win, [(pad_left, pad_right)])

    frames = frames * win

    output_length = (num_frames - 1) * hop_length + n_fft
    output = mx.zeros((batch_size, output_length))
    window_sum = mx.zeros((output_length,))
    win_squared = win * win

    for i in range(num_frames):
        start = i * hop_length
        window_sum = window_sum.at[start : start + n_fft].add(win_squared)

    for i in range(num_frames):
        start = i * hop_length
        output = output.at[:, start : start + n_fft].add(frames[:, i, :])

    window_sum = mx.maximum(window_sum, 1e-8)
    output = output / window_sum

    if center:
        pad_amount = n_fft // 2
        output = output[:, pad_amount:-pad_amount]

    if length is not None:
        output = output[:, :length]

    if input_1d:
        output = mx.squeeze(output, axis=0)

    return output


# ---------------------------------------------------------------------------
# Round-trip reconstruction tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """stft -> istft should reconstruct the original signal."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_batch_sizes(self, batch_size: int):
        sig = _random_signal(batch_size, 9600)
        spec = stft(sig, n_fft=960, hop_length=480, window="sqrt_hann", return_complex=True)
        recon = istft(spec, n_fft=960, hop_length=480, window="sqrt_hann", length=9600)
        mx.eval(recon)
        err = mx.max(mx.abs(sig - recon)).item()
        assert err < 1e-5, f"Round-trip error {err:.2e} exceeds threshold"

    @pytest.mark.parametrize(
        "length",
        [4800, 9600, 48000],
        ids=["short", "medium", "long"],
    )
    def test_signal_lengths(self, length: int):
        sig = _random_signal(2, length)
        spec = stft(sig, n_fft=960, hop_length=480, window="sqrt_hann", return_complex=True)
        recon = istft(spec, n_fft=960, hop_length=480, window="sqrt_hann", length=length)
        mx.eval(recon)
        err = mx.max(mx.abs(sig - recon)).item()
        assert err < 1e-5, f"Round-trip error {err:.2e} for length={length}"

    def test_default_deepfilternet_params(self):
        """Default n_fft=960, hop_length=480 used by DeepFilterNet."""
        sig = _random_signal(1, 48000)
        spec = stft(sig, n_fft=960, hop_length=480, window="sqrt_hann", return_complex=True)
        recon = istft(spec, n_fft=960, hop_length=480, window="sqrt_hann", length=48000)
        mx.eval(recon)
        err = mx.max(mx.abs(sig - recon)).item()
        assert err < 1e-5, f"Default-param round-trip error {err:.2e}"


# ---------------------------------------------------------------------------
# Vectorized vs loop equivalence
# ---------------------------------------------------------------------------


class TestVectorizedVsLoop:
    """Vectorized path must match loop-based reference."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_output_equivalence(self, batch_size: int):
        sig = _random_signal(batch_size, 9600)
        spec = stft(sig, n_fft=960, hop_length=480, window="sqrt_hann", return_complex=True)

        vec_out = istft(spec, n_fft=960, hop_length=480, window="sqrt_hann", length=9600)
        loop_out = _istft_loop(spec, n_fft=960, hop_length=480, window="sqrt_hann", length=9600)
        mx.eval(vec_out, loop_out)

        err = mx.max(mx.abs(vec_out - loop_out)).item()
        assert err < 1e-6, f"Vec vs loop mismatch {err:.2e}"

    def test_equivalence_long_signal(self):
        sig = _random_signal(2, 48000)
        spec = stft(sig, n_fft=960, hop_length=480, window="sqrt_hann", return_complex=True)

        vec_out = istft(spec, n_fft=960, hop_length=480, window="sqrt_hann", length=48000)
        loop_out = _istft_loop(spec, n_fft=960, hop_length=480, window="sqrt_hann", length=48000)
        mx.eval(vec_out, loop_out)

        err = mx.max(mx.abs(vec_out - loop_out)).item()
        assert err < 1e-6, f"Vec vs loop mismatch on long signal {err:.2e}"


# ---------------------------------------------------------------------------
# Window type tests
# ---------------------------------------------------------------------------


class TestWindowTypes:
    """Various window types should all reconstruct cleanly."""

    @pytest.mark.parametrize("window", ["hann", "hamming", "sqrt_hann"])
    def test_window_round_trip(self, window: str):
        sig = _random_signal(2, 9600)
        spec = stft(sig, n_fft=960, hop_length=480, window=window, return_complex=True)
        recon = istft(spec, n_fft=960, hop_length=480, window=window, length=9600)
        mx.eval(recon)
        err = mx.max(mx.abs(sig - recon)).item()
        assert err < 1e-4, f"Round-trip error {err:.2e} for window={window}"

    @pytest.mark.parametrize("window", ["hann", "hamming", "sqrt_hann"])
    def test_window_vec_vs_loop(self, window: str):
        sig = _random_signal(2, 9600)
        spec = stft(sig, n_fft=960, hop_length=480, window=window, return_complex=True)

        vec_out = istft(spec, n_fft=960, hop_length=480, window=window, length=9600)
        loop_out = _istft_loop(spec, n_fft=960, hop_length=480, window=window, length=9600)
        mx.eval(vec_out, loop_out)

        err = mx.max(mx.abs(vec_out - loop_out)).item()
        assert err < 1e-6, f"Vec vs loop mismatch for window={window}: {err:.2e}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge and boundary conditions."""

    def test_1d_input(self):
        """Unbatched (1-D) signal should work identically."""
        sig = _random_signal(1, 9600)[0]  # squeeze to 1-D
        spec = stft(sig, n_fft=960, hop_length=480, window="sqrt_hann", return_complex=True)
        recon = istft(spec, n_fft=960, hop_length=480, window="sqrt_hann", length=9600)
        mx.eval(recon)
        assert recon.ndim == 1
        err = mx.max(mx.abs(sig - recon)).item()
        assert err < 1e-5, f"1-D round-trip error {err:.2e}"

    def test_no_length_trim(self):
        """When length=None, output should still be valid."""
        sig = _random_signal(1, 9600)
        spec = stft(sig, n_fft=960, hop_length=480, window="sqrt_hann", return_complex=True)
        recon = istft(spec, n_fft=960, hop_length=480, window="sqrt_hann", length=None)
        mx.eval(recon)
        assert recon.shape[1] > 0

    def test_stacked_format_input(self):
        """iSTFT should accept (..., freq, 2) stacked format."""
        sig = _random_signal(2, 9600)
        spec_stacked = stft(sig, n_fft=960, hop_length=480, window="sqrt_hann", return_complex=False)
        recon = istft(spec_stacked, n_fft=960, hop_length=480, window="sqrt_hann", length=9600)
        mx.eval(recon)
        err = mx.max(mx.abs(sig - recon)).item()
        assert err < 1e-5, f"Stacked-format round-trip error {err:.2e}"
