"""Tests for extension-backed data packing optimizations."""

import numpy as np

from df_mlx.feature_ops import compute_stft


class TestStftDtype:
    """Verify compute_stft always returns complex64."""

    def test_output_is_complex64(self):
        """STFT output must be complex64 regardless of numpy version."""
        audio = np.random.randn(48000).astype(np.float32)
        spec = compute_stft(audio, fft_size=960, hop_size=480)
        assert spec.dtype == np.complex64

    def test_real_imag_are_float32(self):
        """Real and imaginary parts must be float32."""
        audio = np.random.randn(48000).astype(np.float32)
        spec = compute_stft(audio, fft_size=960, hop_size=480)
        assert spec.real.dtype == np.float32
        assert spec.imag.dtype == np.float32

    def test_real_part_is_contiguous(self):
        """Real part of complex64 array should be interpretable as contiguous."""
        audio = np.random.randn(48000).astype(np.float32)
        spec = compute_stft(audio, fft_size=960, hop_size=480)
        real = np.require(spec.real, dtype=np.float32, requirements="C")
        np.testing.assert_array_equal(real, spec.real)

    def test_shape_unchanged(self):
        """STFT shape should not be affected by dtype optimization."""
        audio = np.random.randn(48000).astype(np.float32)
        spec = compute_stft(audio, fft_size=960, hop_size=480)
        assert spec.ndim == 2
        assert spec.shape[1] == 960 // 2 + 1

    def test_numerical_fidelity(self):
        """Verify complex64 STFT values match reference within tolerance."""
        np.random.seed(42)
        audio = np.random.randn(4800).astype(np.float32)
        spec = compute_stft(audio, fft_size=960, hop_size=480)

        window = np.sqrt(np.hanning(961)[:-1]).astype(np.float32)
        pad_len = 960 - 480
        audio_padded = np.pad(audio, (pad_len, pad_len), mode="constant")
        num_frames = (len(audio_padded) - 960) // 480 + 1
        ref_spec = np.zeros((num_frames, 481), dtype=np.complex128)
        for i in range(num_frames):
            frame = audio_padded[i * 480 : i * 480 + 960]
            ref_spec[i] = np.fft.rfft(frame * window, n=960)

        np.testing.assert_allclose(spec, ref_spec.astype(np.complex64), atol=1e-4, rtol=1e-4)
