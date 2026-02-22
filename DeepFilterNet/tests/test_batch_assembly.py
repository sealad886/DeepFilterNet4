"""Tests for _assemble_batch pre-allocated batch assembly."""

import numpy as np
import pytest

from df_mlx.dynamic_dataset import Sample, _assemble_batch


def _make_sample(
    time_frames: int = 100,
    n_freqs: int = 481,
    n_erb: int = 32,
    n_df: int = 96,
    snr: float = 10.0,
    gain: float = 1.0,
) -> Sample:
    """Create a synthetic Sample with realistic shapes."""
    spec = np.random.randn(time_frames, n_freqs).astype(np.float32) + 1j * np.random.randn(time_frames, n_freqs).astype(
        np.float32
    )
    clean_spec = np.random.randn(time_frames, n_freqs).astype(np.float32) + 1j * np.random.randn(
        time_frames, n_freqs
    ).astype(np.float32)
    feat_erb = np.random.randn(time_frames, n_erb).astype(np.float32)
    feat_spec = np.random.randn(time_frames, n_df, 2).astype(np.float32)
    return Sample(
        noisy_spec=spec,
        clean_spec=clean_spec,
        feat_erb=feat_erb,
        feat_spec=feat_spec,
        snr=snr,
        gain=gain,
    )


class TestAssembleBatch:
    def test_single_sample(self):
        s = _make_sample()
        batch = _assemble_batch([s])
        assert batch["noisy_real"].shape == (1, 100, 481)
        assert batch["snr"].shape == (1,)

    def test_batch_shapes(self):
        samples = [_make_sample() for _ in range(4)]
        batch = _assemble_batch(samples)
        assert batch["noisy_real"].shape == (4, 100, 481)
        assert batch["clean_imag"].shape == (4, 100, 481)
        assert batch["feat_erb"].shape == (4, 100, 32)
        assert batch["feat_spec"].shape == (4, 100, 96, 2)
        assert batch["snr"].shape == (4,)

    def test_values_preserved(self):
        s = _make_sample(snr=42.0)
        batch = _assemble_batch([s])
        np.testing.assert_allclose(np.array(batch["noisy_real"]), s.noisy_spec.real[None], atol=1e-6)
        np.testing.assert_allclose(np.array(batch["snr"]), [42.0], atol=1e-6)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _assemble_batch([])

    def test_all_keys_present(self):
        batch = _assemble_batch([_make_sample()])
        expected_keys = {
            "noisy_real",
            "noisy_imag",
            "clean_real",
            "clean_imag",
            "feat_erb",
            "feat_spec",
            "snr",
        }
        assert set(batch.keys()) == expected_keys

    def test_multiple_snr_values(self):
        samples = [_make_sample(snr=float(i)) for i in range(8)]
        batch = _assemble_batch(samples)
        expected = np.arange(8, dtype=np.float32)
        np.testing.assert_allclose(np.array(batch["snr"]), expected, atol=1e-6)
