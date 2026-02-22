#!/usr/bin/env python3
"""Test script for MLXDataStream integration."""

import os
import tempfile
import time

import numpy as np

from df_mlx.dynamic_dataset import (
    HAS_MLX_DATA,
    CheckpointState,
    MLXDataStream,
    Sample,
    compute_df_features,
    compute_erb_features,
    compute_stft,
    create_erb_filterbank,
)


class MockDynamicDataset:
    """Mock dataset for testing MLXDataStream without real audio files."""

    def __init__(self, n_samples: int = 100, segment_samples: int = 96000):
        self.n_samples = n_samples
        self.segment_samples = segment_samples
        self.fft_size = 960
        self.hop_size = 480
        self.sample_rate = 48000

        # Simulate config
        class MockConfig:
            seed = 42
            nb_erb = 32
            nb_df = 96

        self.config = MockConfig()

        # Pre-compute filterbank and window
        self.erb_fb = create_erb_filterbank(
            sr=self.sample_rate,
            fft_size=self.fft_size,
            nb_erb=self.config.nb_erb,
        )
        self.window = np.sqrt(np.hanning(self.fft_size + 1)[:-1]).astype(np.float32)

        self._epoch = 0
        self._current_split = "train"

    def __len__(self):
        return self.n_samples

    def set_split(self, split: str):
        self._current_split = split

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def get_sample(self, idx: int) -> Sample:
        """Generate a synthetic sample."""
        # Create deterministic but varied audio
        np.random.seed(self.config.seed + self._epoch * 1000 + idx)

        # Generate synthetic noisy and clean signals
        noisy = np.random.randn(self.segment_samples).astype(np.float32) * 0.1
        clean = np.random.randn(self.segment_samples).astype(np.float32) * 0.05

        # Compute spectrograms
        noisy_spec = compute_stft(noisy, self.fft_size, self.hop_size, self.window)
        clean_spec = compute_stft(clean, self.fft_size, self.hop_size, self.window)

        # Compute features
        feat_erb = compute_erb_features(noisy_spec, self.erb_fb)
        feat_spec = compute_df_features(noisy_spec, self.config.nb_df)

        return Sample(
            noisy_spec=noisy_spec,
            clean_spec=clean_spec,
            feat_erb=feat_erb,
            feat_spec=feat_spec,
            snr=np.random.uniform(-5, 20),
            gain=np.random.uniform(-6, 6),
        )


def test_checkpoint_state():
    """Test CheckpointState serialization."""
    print("Testing CheckpointState...")

    cp = CheckpointState(epoch=5, batch_idx=100, samples_processed=800, seed=123)

    # Test to_dict
    d = cp.to_dict()
    assert d["epoch"] == 5
    assert d["batch_idx"] == 100
    assert d["samples_processed"] == 800
    assert d["seed"] == 123

    # Test from_dict
    cp2 = CheckpointState.from_dict(d)
    assert cp2.epoch == cp.epoch
    assert cp2.batch_idx == cp.batch_idx

    # Test save/load
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cp.save(f.name)
        cp3 = CheckpointState.load(f.name)
        assert cp3.epoch == cp.epoch
        os.unlink(f.name)

    print("✓ CheckpointState tests passed")


def test_mlx_data_stream():
    """Test MLXDataStream with mock dataset."""
    print("\nTesting MLXDataStream...")
    print(f"mlx-data available: {HAS_MLX_DATA}")

    if not HAS_MLX_DATA:
        print("⚠ Skipping MLXDataStream test (mlx-data not installed)")
        return

    # Create mock dataset
    dataset = MockDynamicDataset(n_samples=50)
    print(f"Mock dataset: {len(dataset)} samples")

    # Create stream
    batch_size = 4
    stream = MLXDataStream(
        dataset=dataset,
        batch_size=batch_size,
        prefetch_size=4,
        num_workers=4,
    )
    print("Created MLXDataStream")

    # Time batch loading
    stream.set_epoch(0)
    start = time.time()
    batch_count = 0
    first_batch = None

    for batch in stream:
        batch_count += 1
        if batch_count == 1:
            first_batch = batch
            print("First batch shapes:")
            for k, v in batch.items():
                print(f"  {k}: {v.shape}")
        if batch_count >= 10:
            break

    elapsed = time.time() - start
    samples = batch_count * batch_size
    rate = samples / elapsed if elapsed > 0 else 0
    print(f"Loaded {batch_count} batches ({samples} samples) in {elapsed:.2f}s ({rate:.1f} samples/s)")

    assert batch_count > 0, "Should have loaded at least one batch"
    assert first_batch is not None, "First batch should exist"

    # Verify batch shapes
    assert first_batch["noisy_real"].shape[0] == batch_size
    assert first_batch["feat_erb"].ndim == 3  # (B, T, E)
    assert first_batch["feat_spec"].ndim == 4  # (B, T, D, 2)

    # Test checkpoint
    cp_path = "/tmp/test_mlx_stream_checkpoint.json"
    stream.save_checkpoint(cp_path)
    print(f"Saved checkpoint: epoch={stream.checkpoint.epoch}, batch={stream.checkpoint.batch_idx}")

    # Test resume
    stream2 = MLXDataStream.from_checkpoint(
        dataset=dataset,
        checkpoint_path=cp_path,
        batch_size=batch_size,
        prefetch_size=4,
        num_workers=4,
    )
    progress = stream2.get_progress()
    print(f"Resumed: epoch={progress['epoch']}, batch={progress['batch']}")

    os.unlink(cp_path)
    print("✓ MLXDataStream tests passed")


def main():
    print("=" * 60)
    print("MLXDataStream Integration Tests")
    print("=" * 60)

    test_checkpoint_state()
    test_mlx_data_stream()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
