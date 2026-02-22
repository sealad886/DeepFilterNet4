"""Tests for resume determinism after S2.1/S2.2 loader refactors.

Validates that _assemble_batch (pre-allocated buffers instead of np.stack),
complex64 STFT enforcement, and iter_batches/PrefetchDataLoader all preserve
identical output semantics.
"""

import mlx.core as mx
import numpy as np
import pytest

from df_mlx.dynamic_dataset import DatasetConfig, DynamicDataset, PrefetchDataLoader, Sample, _assemble_batch
from df_mlx.feature_ops import compute_stft

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    seed: int = 0,
    time_frames: int = 100,
    n_freqs: int = 481,
    n_erb: int = 32,
    n_df: int = 96,
) -> Sample:
    """Create a deterministic synthetic Sample."""
    rng = np.random.RandomState(seed)
    spec = (rng.randn(time_frames, n_freqs) + 1j * rng.randn(time_frames, n_freqs)).astype(np.complex64)
    clean = (rng.randn(time_frames, n_freqs) + 1j * rng.randn(time_frames, n_freqs)).astype(np.complex64)
    return Sample(
        noisy_spec=spec,
        clean_spec=clean,
        feat_erb=rng.randn(time_frames, n_erb).astype(np.float32),
        feat_spec=rng.randn(time_frames, n_df, 2).astype(np.float32),
        snr=float(rng.uniform(-5, 30)),
        gain=1.0,
    )


def _old_stack_assemble(samples):
    """Reference implementation using the old np.stack path."""
    noisy_real = np.stack([s.noisy_spec.real for s in samples])
    noisy_imag = np.stack([s.noisy_spec.imag for s in samples])
    clean_real = np.stack([s.clean_spec.real for s in samples])
    clean_imag = np.stack([s.clean_spec.imag for s in samples])
    feat_erb = np.stack([s.feat_erb for s in samples])
    feat_spec = np.stack([s.feat_spec for s in samples])
    snr_arr = np.array([s.snr for s in samples], dtype=np.float32)
    return {
        "noisy_real": mx.array(noisy_real),
        "noisy_imag": mx.array(noisy_imag),
        "clean_real": mx.array(clean_real),
        "clean_imag": mx.array(clean_imag),
        "feat_erb": mx.array(feat_erb),
        "feat_spec": mx.array(feat_spec),
        "snr": mx.array(snr_arr),
    }


def _make_mock_dataset(n_samples: int = 10, seed: int = 42):
    """Build a DynamicDataset with mocked audio loaders for deterministic tests."""
    file_list = [f"spk_{i}.wav" for i in range(max(n_samples, 3))]
    cfg = DatasetConfig(
        speech_files=file_list,
        noise_files=["noise_a.wav"],
        rir_files=[],
        sample_rate=8000,
        segment_length=0.1,
        fft_size=64,
        hop_size=32,
        nb_erb=8,
        nb_df=16,
        p_reverb=0.0,
        p_clipping=0.0,
        p_bandwidth_ext=0.0,
        n_noise_min=1,
        n_noise_max=1,
        p_random_noise=0.0,
        snr_range=(-5.0, -5.0),
        snr_range_extreme=(-20.0, -20.0),
        snr_range_very_low=(-30.0, -30.0),
        p_very_low_snr=0.0,
        p_extreme_snr=0.0,
        p_interfer_speech=0.0,
        interfer_speech_snr_range=(0.0, 0.0),
        seed=seed,
    )
    ds = DynamicDataset(cfg)
    ds.set_split("train")
    ds.set_epoch(0)
    n = ds.segment_samples

    def _mock_load_speech(idx: int, rng):
        local_rng = np.random.RandomState(seed + idx)
        return local_rng.randn(n).astype(np.float32) * 0.3

    def _mock_load_noise(rng):
        return np.zeros((n,), dtype=np.float32), 0.0

    ds._load_speech = _mock_load_speech  # type: ignore[method-assign]
    ds._load_noise = _mock_load_noise  # type: ignore[method-assign]
    return ds


# ---------------------------------------------------------------------------
# 1. Batch output equivalence: _assemble_batch vs old np.stack path
# ---------------------------------------------------------------------------


class TestBatchOutputEquivalence:
    def test_single_sample_equivalence(self):
        sample = _make_sample(seed=0)
        new_batch = _assemble_batch([sample])
        old_batch = _old_stack_assemble([sample])
        for key in old_batch:
            np.testing.assert_allclose(
                np.array(new_batch[key]),
                np.array(old_batch[key]),
                atol=0.0,
                err_msg=f"Mismatch for key {key!r}",
            )

    def test_multi_sample_equivalence(self):
        samples = [_make_sample(seed=i) for i in range(8)]
        new_batch = _assemble_batch(samples)
        old_batch = _old_stack_assemble(samples)
        for key in old_batch:
            np.testing.assert_allclose(
                np.array(new_batch[key]),
                np.array(old_batch[key]),
                atol=0.0,
                err_msg=f"Mismatch for key {key!r}",
            )

    def test_key_sets_identical(self):
        samples = [_make_sample(seed=0)]
        assert set(_assemble_batch(samples).keys()) == set(_old_stack_assemble(samples).keys())


# ---------------------------------------------------------------------------
# 2. iter_batches determinism: same dataset/epoch/seed => identical batches
# ---------------------------------------------------------------------------


class TestIterBatchesDeterminism:
    def test_two_passes_produce_identical_batches(self):
        ds = _make_mock_dataset(n_samples=6, seed=99)
        batch_size = 2

        ds.set_epoch(0)
        batches_a = list(ds.iter_batches(batch_size=batch_size, drop_last=True))

        ds.set_epoch(0)
        batches_b = list(ds.iter_batches(batch_size=batch_size, drop_last=True))

        assert len(batches_a) == len(batches_b), "Number of batches differs across runs"
        for i, (ba, bb) in enumerate(zip(batches_a, batches_b)):
            assert set(ba.keys()) == set(bb.keys())
            for key in ba:
                np.testing.assert_allclose(
                    np.array(ba[key]),
                    np.array(bb[key]),
                    atol=1e-6,
                    err_msg=f"Batch {i}, key {key!r} differs across deterministic runs",
                )


# ---------------------------------------------------------------------------
# 3. PrefetchDataLoader batch parity: same keys/shapes as iter_batches
# ---------------------------------------------------------------------------


class TestPrefetchDataLoaderBatchParity:
    def test_keys_and_shapes_match_iter_batches(self):
        ds = _make_mock_dataset(n_samples=6, seed=77)
        batch_size = 2

        ds.set_epoch(0)
        iter_batches = list(ds.iter_batches(batch_size=batch_size, drop_last=True))

        ds.set_epoch(0)
        loader = PrefetchDataLoader(
            ds,
            batch_size=batch_size,
            num_workers=1,
            prefetch_factor=1,
            drop_last=True,
            strict_failures=True,
        )
        loader_batches = list(loader)

        assert len(loader_batches) == len(iter_batches), (
            f"PrefetchDataLoader yielded {len(loader_batches)} batches, " f"iter_batches yielded {len(iter_batches)}"
        )
        for i, (lb, ib) in enumerate(zip(loader_batches, iter_batches)):
            assert set(lb.keys()) == set(ib.keys()), f"Key mismatch at batch {i}"
            for key in ib:
                assert lb[key].shape == ib[key].shape, (
                    f"Shape mismatch at batch {i}, key {key!r}: " f"{lb[key].shape} vs {ib[key].shape}"
                )


# ---------------------------------------------------------------------------
# 4. Complex64 round-trip: compute_stft forces complex64
# ---------------------------------------------------------------------------


class TestComplex64RoundTrip:
    def test_stft_returns_complex64(self):
        rng = np.random.RandomState(12)
        audio = rng.randn(4800).astype(np.float32)
        spec = compute_stft(audio, fft_size=960, hop_size=480)
        assert spec.dtype == np.complex64, f"Expected complex64, got {spec.dtype}"

    def test_stft_from_float64_input_still_complex64(self):
        rng = np.random.RandomState(13)
        audio = rng.randn(4800).astype(np.float64)
        spec = compute_stft(audio, fft_size=960, hop_size=480)
        assert spec.dtype == np.complex64, f"Expected complex64, got {spec.dtype}"

    def test_stft_real_imag_are_float32(self):
        rng = np.random.RandomState(14)
        audio = rng.randn(4800).astype(np.float32)
        spec = compute_stft(audio, fft_size=960, hop_size=480)
        assert spec.real.dtype == np.float32
        assert spec.imag.dtype == np.float32

    def test_stft_values_within_tolerance(self):
        rng = np.random.RandomState(15)
        audio = rng.randn(4800).astype(np.float32)
        spec = compute_stft(audio, fft_size=960, hop_size=480)
        # Compute with float64 for reference, then compare
        audio_f64 = audio.astype(np.float64)
        spec_f64 = compute_stft(audio_f64, fft_size=960, hop_size=480)
        # Both return complex64 now, so compare directly
        np.testing.assert_allclose(
            spec.real,
            spec_f64.real,
            atol=1e-5,
            err_msg="Real parts diverge beyond tolerance",
        )
        np.testing.assert_allclose(
            spec.imag,
            spec_f64.imag,
            atol=1e-5,
            err_msg="Imaginary parts diverge beyond tolerance",
        )


# ---------------------------------------------------------------------------
# 5. Resume index consistency: batch count matches expected
# ---------------------------------------------------------------------------


class TestResumeIndexConsistency:
    def test_batch_count_matches_expected(self):
        n_samples = 9
        batch_size = 3
        ds = _make_mock_dataset(n_samples=n_samples, seed=55)
        ds.set_epoch(0)
        batches = list(ds.iter_batches(batch_size=batch_size, drop_last=True))
        n_dataset = len(ds)
        expected_batches = n_dataset // batch_size
        assert (
            len(batches) == expected_batches
        ), f"Expected {expected_batches} batches (from {n_dataset} samples), got {len(batches)}"

    def test_no_silent_duplicates_or_skips(self):
        n_samples = 6
        batch_size = 2
        ds = _make_mock_dataset(n_samples=n_samples, seed=56)
        ds.set_epoch(0)
        batches = list(ds.iter_batches(batch_size=batch_size, drop_last=True))
        total_samples = sum(np.array(b["snr"]).shape[0] for b in batches)
        n_dataset = len(ds)
        expected_total = (n_dataset // batch_size) * batch_size
        assert total_samples == expected_total, (
            f"Total samples across batches ({total_samples}) != expected ({expected_total}). "
            "Possible silent skip or duplicate."
        )

    def test_three_batch_iterate(self):
        """Iterate exactly 3 batches and confirm count."""
        # 20 files * 0.9 train_split = 18 train samples; batch_size=3 => 6 batches
        ds = _make_mock_dataset(n_samples=20, seed=57)
        batch_size = 3
        ds.set_epoch(0)
        batches = []
        for batch in ds.iter_batches(batch_size=batch_size, drop_last=True):
            batches.append(batch)
            if len(batches) == 3:
                break
        assert len(batches) == 3


# ---------------------------------------------------------------------------
# 6. Empty / partial batch edge cases
# ---------------------------------------------------------------------------


class TestEmptyPartialBatchEdgeCases:
    def test_assemble_batch_empty_raises_valueerror(self):
        with pytest.raises(ValueError, match="Cannot assemble empty batch"):
            _assemble_batch([])

    def test_partial_batch_drop_last(self):
        ds = _make_mock_dataset(n_samples=5, seed=60)
        batch_size = 3
        ds.set_epoch(0)
        batches = list(ds.iter_batches(batch_size=batch_size, drop_last=True))
        for b in batches:
            assert np.array(b["snr"]).shape[0] == batch_size

    def test_partial_batch_keep_last(self):
        ds = _make_mock_dataset(n_samples=5, seed=61)
        batch_size = 3
        ds.set_epoch(0)
        batches = list(ds.iter_batches(batch_size=batch_size, drop_last=False))
        assert len(batches) >= 1
        last_size = np.array(batches[-1]["snr"]).shape[0]
        n_dataset = len(ds)
        remainder = n_dataset % batch_size
        if remainder > 0:
            assert last_size == remainder


# ---------------------------------------------------------------------------
# 7. SNR array dtype consistency
# ---------------------------------------------------------------------------


class TestSnrDtypeConsistency:
    def test_snr_field_is_float32(self):
        samples = [_make_sample(seed=i) for i in range(4)]
        batch = _assemble_batch(samples)
        snr_np = np.array(batch["snr"])
        assert snr_np.dtype == np.float32, f"Expected float32, got {snr_np.dtype}"

    def test_snr_preserves_values(self):
        expected_snrs = [1.5, -3.0, 20.0, 0.0]
        samples = []
        for i, snr_val in enumerate(expected_snrs):
            s = _make_sample(seed=i)
            s = Sample(
                noisy_spec=s.noisy_spec,
                clean_spec=s.clean_spec,
                feat_erb=s.feat_erb,
                feat_spec=s.feat_spec,
                snr=snr_val,
                gain=s.gain,
            )
            samples.append(s)
        batch = _assemble_batch(samples)
        np.testing.assert_allclose(
            np.array(batch["snr"]),
            np.array(expected_snrs, dtype=np.float32),
            atol=1e-7,
        )

    def test_snr_dtype_from_iter_batches(self):
        ds = _make_mock_dataset(n_samples=4, seed=70)
        ds.set_epoch(0)
        for batch in ds.iter_batches(batch_size=2, drop_last=True):
            snr_np = np.array(batch["snr"])
            assert snr_np.dtype == np.float32, f"Expected float32, got {snr_np.dtype}"
            break  # one batch is enough
