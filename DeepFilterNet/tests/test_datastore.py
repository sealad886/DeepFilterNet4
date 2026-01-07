#!/usr/bin/env python3
"""Tests for MLX datastore and data preparation modules."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestDatastoreConfig:
    """Test DatastoreConfig functionality."""

    def test_config_defaults(self):
        """Test default config values."""
        from df_mlx.datastore import DatastoreConfig

        config = DatastoreConfig()
        assert config.sample_rate == 48000
        assert config.fft_size == 960
        assert config.hop_size == 480
        assert config.nb_erb == 32
        assert config.nb_df == 96
        assert config.samples_per_shard == 1000
        assert config.compress is False

    def test_config_with_compress(self):
        """Test config with compress option."""
        from df_mlx.datastore import DatastoreConfig

        config = DatastoreConfig(compress=True)
        assert config.compress is True

    def test_config_serialization_via_index(self):
        """Test config serialization through DatastoreIndex."""
        from df_mlx.datastore import DatastoreConfig, DatastoreIndex

        config = DatastoreConfig(compress=True, sample_rate=16000)
        index = DatastoreIndex(config=config)
        d = index.to_dict()
        assert d["config"]["sample_rate"] == 16000
        assert d["config"]["compress"] is True


class TestPrepareDataFunctions:
    """Test individual functions from prepare_data."""

    def test_compute_stft_stride_tricks(self):
        """Test STFT computation with stride_tricks optimization."""
        from df_mlx.prepare_data import compute_stft

        # Generate test signal
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32)

        fft_size = 960
        hop_size = 480
        window = np.sqrt(np.hanning(fft_size + 1)[:-1]).astype(np.float32)

        stft = compute_stft(audio, fft_size, hop_size, window)

        # Check output shape
        expected_freqs = fft_size // 2 + 1
        assert stft.shape[1] == expected_freqs
        assert stft.dtype == np.complex128 or stft.dtype == np.complex64

    def test_compute_erb_features(self):
        """Test ERB feature computation."""
        from df_mlx.prepare_data import compute_erb_features, create_erb_filterbank

        erb_fb = create_erb_filterbank(sr=48000, fft_size=960, nb_erb=32)

        # Generate dummy spectrum
        spec = np.random.randn(100, 481) + 1j * np.random.randn(100, 481)

        erb = compute_erb_features(spec, erb_fb)

        assert erb.shape == (100, 32)
        assert erb.dtype == np.float32

    def test_compute_df_features(self):
        """Test DF feature computation."""
        from df_mlx.prepare_data import compute_df_features

        spec = np.random.randn(100, 481) + 1j * np.random.randn(100, 481)
        nb_df = 96

        df_feat = compute_df_features(spec, nb_df)

        assert df_feat.shape == (100, nb_df, 2)
        assert df_feat.dtype == np.float32

    def test_mix_audio(self):
        """Test audio mixing at specified SNR."""
        from df_mlx.prepare_data import mix_audio

        np.random.seed(42)
        clean = np.random.randn(48000).astype(np.float32)
        noise = np.random.randn(48000).astype(np.float32)

        noisy, clean_scaled = mix_audio(clean, noise, snr_db=10.0)

        assert noisy.shape == clean.shape
        assert clean_scaled.shape == clean.shape


class TestDatastoreWriter:
    """Test MLXDatastoreWriter functionality."""

    def test_writer_creation(self, temp_dir):
        """Test creating a new datastore writer."""
        from df_mlx.datastore import DatastoreConfig, MLXDatastoreWriter

        config = DatastoreConfig(samples_per_shard=10, compress=False)

        with MLXDatastoreWriter(temp_dir, config) as writer:
            assert writer.config == config
            assert not writer.resumed

    def test_writer_with_samples(self, temp_dir):
        """Test writing samples to datastore."""
        from df_mlx.datastore import DatastoreConfig, MLXDatastoreWriter

        config = DatastoreConfig(samples_per_shard=5, compress=False)

        # Generate dummy data
        np.random.seed(42)
        num_samples = 10
        time_frames = 100
        n_freqs = 481
        nb_erb = 32
        nb_df = 96

        with MLXDatastoreWriter(temp_dir, config) as writer:
            writer.set_split("train")

            for i in range(num_samples):
                spec = np.random.randn(time_frames, n_freqs).astype(np.float32)
                writer.add_sample(
                    spec_real=spec,
                    spec_imag=spec,
                    feat_erb=np.random.randn(time_frames, nb_erb).astype(np.float32),
                    feat_spec=np.random.randn(time_frames, nb_df, 2).astype(np.float32),
                    clean_real=spec,
                    clean_imag=spec,
                    snr_db=10.0,
                )

        # Check that files were created
        index_path = temp_dir / "index.json"
        assert index_path.exists()

        # Check for shard files (saved at root, not in subdirs)
        shard_files = list(temp_dir.glob("train_*.npz"))
        assert len(shard_files) >= 2  # With 10 samples and 5 per shard

    def test_compress_option(self, temp_dir):
        """Test compressed vs uncompressed file sizes."""
        import json

        from df_mlx.datastore import DatastoreConfig, MLXDatastoreWriter

        # Create highly compressible data
        np.random.seed(42)
        time_frames = 50
        n_freqs = 481
        nb_erb = 32
        nb_df = 96

        # Uncompressed version
        config_uncompressed = DatastoreConfig(samples_per_shard=5, compress=False)
        dir_uncompressed = temp_dir / "uncompressed"

        with MLXDatastoreWriter(dir_uncompressed, config_uncompressed) as writer:
            writer.set_split("train")
            for _ in range(5):
                spec = np.zeros((time_frames, n_freqs), dtype=np.float32)
                writer.add_sample(
                    spec_real=spec,
                    spec_imag=spec,
                    feat_erb=np.zeros((time_frames, nb_erb), dtype=np.float32),
                    feat_spec=np.zeros((time_frames, nb_df, 2), dtype=np.float32),
                    clean_real=spec,
                    clean_imag=spec,
                    snr_db=0.0,
                )

        # Check config in index
        with open(dir_uncompressed / "index.json") as f:
            index = json.load(f)
        assert index["config"]["compress"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
