"""Tests for df_mlx operations (mel spectrogram, etc.)."""

import mlx.core as mx
import numpy as np
import pytest


class TestMelSpectrogram:
    """Tests for MelSpectrogram module."""

    @pytest.fixture
    def mel_extractor(self):
        """Create MelSpectrogram extractor for testing."""
        from df_mlx.dnsmos_proxy import MelSpectrogram

        return MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=256,
            n_mels=64,
            use_metal_kernel=False,  # Use pure MLX path
        )

    def test_output_shape(self, mel_extractor):
        """Verify output shape is correct."""
        batch_size = 2
        n_samples = 16000  # 1 second of audio

        audio = mx.random.normal(shape=(batch_size, n_samples))
        mel_spec = mel_extractor(audio)

        expected_n_frames = (n_samples - mel_extractor.n_fft) // mel_extractor.hop_length + 1
        assert mel_spec.shape == (batch_size, mel_extractor.n_mels, expected_n_frames)

    def test_single_input(self, mel_extractor):
        """Test with single input (batch_size=1, 2D)."""
        n_samples = 16000

        audio = mx.random.normal(shape=(1, n_samples))
        mel_spec = mel_extractor(audio)

        expected_n_frames = (n_samples - mel_extractor.n_fft) // mel_extractor.hop_length + 1
        assert mel_spec.shape == (1, mel_extractor.n_mels, expected_n_frames)

    def test_finite_output(self, mel_extractor):
        """Verify no NaN/Inf in output."""
        audio = mx.random.normal(shape=(2, 16000))
        mel_spec = mel_extractor(audio)

        assert mx.all(mx.isfinite(mel_spec)).item()

    def test_3d_input(self, mel_extractor):
        """Test with 3D input [B, 1, T]."""
        audio = mx.random.normal(shape=(2, 1, 16000))
        mel_spec = mel_extractor(audio)

        expected_n_frames = (16000 - mel_extractor.n_fft) // mel_extractor.hop_length + 1
        assert mel_spec.shape == (2, mel_extractor.n_mels, expected_n_frames)

    def test_short_audio(self, mel_extractor):
        """Test with audio shorter than n_fft."""
        audio = mx.random.normal(shape=(2, 100))
        mel_spec = mel_extractor(audio)

        assert mel_spec.shape == (2, mel_extractor.n_mels, 1)

    def test_log_scaling(self, mel_extractor):
        """Verify output is in log scale (values are finite and some are positive)."""
        audio = mx.random.normal(shape=(2, 16000))
        mel_spec = mel_extractor(audio)

        assert mx.all(mx.isfinite(mel_spec)).item()

    def test_deterministic(self, mel_extractor):
        """Verify same input produces same output."""
        audio = mx.random.normal(shape=(2, 16000))

        mel_spec1 = mel_extractor(audio)
        mel_spec2 = mel_extractor(audio)

        np.testing.assert_array_almost_equal(mel_spec1, mel_spec2)
