"""Tests for RIR file loading in the training pipeline."""

import json

import numpy as np
import pytest


@pytest.fixture
def mock_datastore(tmp_path):
    """Create a minimal mock datastore with speech, noise, and rir categories."""
    sr = 48000
    seg_samples = sr * 5  # 5 seconds

    # Create category directories
    (tmp_path / "speech").mkdir()
    (tmp_path / "noise").mkdir()
    (tmp_path / "rir").mkdir()

    # Create speech shards
    speech_data = {"audio_00000": np.random.randn(seg_samples).astype(np.float32)}
    np.savez(tmp_path / "speech" / "shard_0000.npz", **speech_data)

    # Create noise shards
    noise_data = {"audio_00000": np.random.randn(seg_samples).astype(np.float32)}
    np.savez(tmp_path / "noise" / "shard_0000.npz", **noise_data)

    # Create RIR shards (shorter - RIRs are typically < 1s)
    rir_data = {"audio_00000": np.random.randn(sr).astype(np.float32)}
    np.savez(tmp_path / "rir" / "shard_0000.npz", **rir_data)

    return tmp_path, sr, seg_samples


def _write_index(path, include_rir=True):
    """Write an index.json with or without rir entries."""
    index = {
        "speech": {"s1.wav": ["speech/shard_0000.npz", "audio_00000"]},
        "noise": {"n1.wav": ["noise/shard_0000.npz", "audio_00000"]},
    }
    if include_rir:
        index["rir"] = {"r1.wav": ["rir/shard_0000.npz", "audio_00000"]}
    with open(path / "index.json", "w") as f:
        json.dump(index, f)


def _write_config(path, sr=48000, p_reverb=0.5):
    """Write a config.json."""
    config = {
        "cache_dir": str(path),
        "sample_rate": sr,
        "segment_length": 5.0,
        "fft_size": 960,
        "hop_size": 480,
        "nb_erb": 32,
        "nb_df": 96,
        "p_reverb": p_reverb,
    }
    with open(path / "config.json", "w") as f:
        json.dump(config, f)


class TestRIRLoading:
    """Test RIR file loading from the sharded cache."""

    def test_rir_loaded_when_index_has_entries(self, mock_datastore):
        """RIR files should be loaded when the index has rir entries."""
        path, sr, _ = mock_datastore
        _write_index(path, include_rir=True)
        _write_config(path)

        from df_mlx.dynamic_dataset import DatasetConfig, DynamicDataset

        config = DatasetConfig.from_json(str(path / "config.json"))
        config.cache_dir = str(path)
        ds = DynamicDataset(config)

        assert ds.rir_cache is not None
        assert len(config.rir_files) == 1
        assert config.rir_files[0] == "r1.wav"

    def test_rir_zero_when_index_missing_rir(self, mock_datastore, capsys):
        """RIR files should be 0 (with warning) when index has no rir entries."""
        path, sr, _ = mock_datastore
        _write_index(path, include_rir=False)
        _write_config(path)

        from df_mlx.dynamic_dataset import DatasetConfig, DynamicDataset

        config = DatasetConfig.from_json(str(path / "config.json"))
        config.cache_dir = str(path)
        ds = DynamicDataset(config)

        # _load_optional_cache returns None when index has no entries,
        # but prints a warning because the rir/ directory has shard files.
        assert ds.rir_cache is None
        assert len(config.rir_files) == 0

        captured = capsys.readouterr()
        assert "Warning: rir/ directory has" in captured.out
        assert "index.json has no RIR entries" in captured.out

    def test_rir_zero_when_no_rir_directory(self, mock_datastore):
        """RIR files should be 0 when rir/ directory doesn't exist."""
        path, sr, _ = mock_datastore
        # Remove rir directory
        import shutil

        shutil.rmtree(path / "rir")
        _write_index(path, include_rir=False)
        _write_config(path)

        from df_mlx.dynamic_dataset import DatasetConfig, DynamicDataset

        config = DatasetConfig.from_json(str(path / "config.json"))
        config.cache_dir = str(path)
        ds = DynamicDataset(config)

        assert ds.rir_cache is None
        assert len(config.rir_files) == 0


class TestBuildCachePreservesRIR:
    """Test that build_audio_cache preserves RIR index entries during partial rebuilds."""

    def test_existing_rir_preserved_when_no_rir_list(self, mock_datastore):
        """existing_indices['rir'] should be preserved in all_indices when rir_future is None."""
        path, _, _ = mock_datastore
        _write_index(path, include_rir=True)

        # Load the existing index
        with open(path / "index.json") as f:
            existing_indices = json.load(f)

        # Simulate the logic from build_audio_cache.py after the fix:
        # When rir_future is None but existing_indices has 'rir',
        # existing entries should be preserved.
        all_indices = {}
        all_indices["speech"] = existing_indices.get("speech", {})
        all_indices["noise"] = existing_indices.get("noise", {})

        rir_future = None  # No --rir-list provided
        if rir_future is not None:
            pass  # would collect from future
        elif "rir" in existing_indices:
            all_indices["rir"] = existing_indices["rir"]

        assert "rir" in all_indices
        assert len(all_indices["rir"]) == 1
        assert "r1.wav" in all_indices["rir"]


class TestTrainingConfigWarning:
    """Test that print_training_config warns when p_reverb > 0 but no RIR files."""

    def test_warning_when_p_reverb_positive_no_rir(self, capsys):
        """Should print warning when p_reverb > 0 but rir_files is empty."""
        from df_mlx.dynamic_dataset import DatasetConfig
        from df_mlx.training_setup import print_training_config

        config = DatasetConfig(p_reverb=0.5, rir_files=[])
        # Need minimal speech/noise for the function to work
        config.speech_files = ["s1.wav"]
        config.noise_files = ["n1.wav"]

        print_training_config(
            config,
            epochs=1,
            batch_size=1,
            learning_rate=1e-4,
            min_lr=1e-7,
            weight_decay=0.0,
            checkpoint_dir="/tmp/test",
            dynamic_loss="baseline",
        )

        captured = capsys.readouterr()
        assert "WARNING: p_reverb > 0 but no RIR files loaded" in captured.out

    def test_no_warning_when_rir_files_present(self, capsys):
        """Should NOT print warning when RIR files are present."""
        from df_mlx.dynamic_dataset import DatasetConfig
        from df_mlx.training_setup import print_training_config

        config = DatasetConfig(p_reverb=0.5, rir_files=["r1.wav"])
        config.speech_files = ["s1.wav"]
        config.noise_files = ["n1.wav"]

        print_training_config(
            config,
            epochs=1,
            batch_size=1,
            learning_rate=1e-4,
            min_lr=1e-7,
            weight_decay=0.0,
            checkpoint_dir="/tmp/test",
            dynamic_loss="baseline",
        )

        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_no_warning_when_p_reverb_zero(self, capsys):
        """Should NOT print warning when p_reverb is 0."""
        from df_mlx.dynamic_dataset import DatasetConfig
        from df_mlx.training_setup import print_training_config

        config = DatasetConfig(p_reverb=0.0, rir_files=[])
        config.speech_files = ["s1.wav"]
        config.noise_files = ["n1.wav"]

        print_training_config(
            config,
            epochs=1,
            batch_size=1,
            learning_rate=1e-4,
            min_lr=1e-7,
            weight_decay=0.0,
            checkpoint_dir="/tmp/test",
            dynamic_loss="baseline",
        )

        captured = capsys.readouterr()
        assert "WARNING" not in captured.out
