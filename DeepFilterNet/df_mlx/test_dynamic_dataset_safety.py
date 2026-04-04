import random

import numpy as np
import pytest
from scipy.io import wavfile

from df_mlx.dynamic_dataset import (
    HAS_MLX_DATA,
    DatasetConfig,
    DynamicDataset,
    MLXDataStream,
    PrefetchDataLoader,
    _build_retry_metadata,
)


def _write_wav(path, sr: int = 16000, seconds: float = 1.0) -> None:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    audio = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    wavfile.write(path, sr, audio)


def _make_mlx_dataset(tmp_path, *, speech_entries: int = 1, seconds: float = 0.6, seed: int = 123) -> DynamicDataset:
    speech_path = tmp_path / "speech.wav"
    _write_wav(speech_path, seconds=seconds)

    cfg = DatasetConfig(
        speech_files=[str(speech_path)] * speech_entries,
        noise_files=[],
        rir_files=[],
        sample_rate=16000,
        segment_length=0.5,
        train_split=1.0,
        valid_split=0.0,
        p_random_noise=0.0,
        n_noise_min=1,
        n_noise_max=1,
        seed=seed,
    )
    return DynamicDataset(cfg)


def test_dynamic_dataset_initializes_indices_for_immediate_get_sample(tmp_path):
    speech_path = tmp_path / "speech.wav"
    _write_wav(speech_path)

    cfg = DatasetConfig(
        speech_files=[str(speech_path)],
        noise_files=[],
        rir_files=[],
        sample_rate=16000,
        segment_length=0.5,
        train_split=1.0,
        valid_split=0.0,
    )
    dataset = DynamicDataset(cfg)
    sample = dataset.get_sample(0)
    assert sample is not None


def test_dynamic_dataset_pads_short_speech_instead_of_skipping(tmp_path):
    speech_path = tmp_path / "short_speech.wav"
    _write_wav(speech_path, seconds=0.2)

    cfg = DatasetConfig(
        speech_files=[str(speech_path)],
        noise_files=[],
        rir_files=[],
        sample_rate=16000,
        segment_length=0.5,
        train_split=1.0,
        valid_split=0.0,
    )
    dataset = DynamicDataset(cfg)
    sample = dataset.get_sample(0)

    assert sample is not None
    assert sample.noisy_spec.shape[0] > 0
    assert sample.clean_spec.shape == sample.noisy_spec.shape


@pytest.mark.skipif(not HAS_MLX_DATA, reason="mlx-data not installed")
def test_mlx_data_stream_transforms_short_speech_without_fallback_failure(tmp_path):
    speech_path = tmp_path / "short_speech.wav"
    _write_wav(speech_path, seconds=0.2)

    cfg = DatasetConfig(
        speech_files=[str(speech_path)],
        noise_files=[],
        rir_files=[],
        sample_rate=16000,
        segment_length=0.5,
        train_split=1.0,
        valid_split=0.0,
        p_random_noise=0.0,
        n_noise_min=1,
        n_noise_max=1,
    )
    dataset = DynamicDataset(cfg)
    stream = MLXDataStream(dataset, batch_size=1, prefetch_size=1, num_workers=1, drop_last=False)

    transformed = stream._sample_transform(
        {"idx": np.array(0, dtype=np.int32), "fallbacks": np.array([], dtype=np.int32)}
    )

    assert transformed["noisy_real"].ndim == 2
    assert transformed["clean_real"].shape == transformed["noisy_real"].shape
    assert transformed["feat_erb"].ndim == 2


def test_build_retry_metadata_preserves_wrapped_fallback_order():
    indices, fallback_matrix = _build_retry_metadata(np.array([10, 20, 30], dtype=np.int32), fallback_count=4)

    np.testing.assert_array_equal(indices, np.array([10, 20, 30], dtype=np.int32))
    np.testing.assert_array_equal(
        fallback_matrix,
        np.array(
            [
                [20, 30, 10, 20],
                [30, 10, 20, 30],
                [10, 20, 30, 10],
            ],
            dtype=np.int32,
        ),
    )


@pytest.mark.skipif(not HAS_MLX_DATA, reason="mlx-data not installed")
def test_mlx_data_stream_scalar_idx_tries_fallbacks_in_order(tmp_path):
    dataset = _make_mlx_dataset(tmp_path)
    stream = MLXDataStream(dataset, batch_size=1, prefetch_size=1, num_workers=1, drop_last=False)

    original_get_sample = dataset.get_sample
    seen: list[int] = []

    def _recording_get_sample(idx: int):
        seen.append(idx)
        if idx == 2:
            return original_get_sample(0)
        return None

    dataset.get_sample = _recording_get_sample  # type: ignore[method-assign]

    transformed = stream._sample_transform({"idx": np.int32(0), "fallbacks": np.array([1, 2, 3], dtype=np.int32)})

    assert seen == [0, 1, 2]
    assert transformed["noisy_real"].ndim == 2


@pytest.mark.skipif(not HAS_MLX_DATA, reason="mlx-data not installed")
def test_mlx_data_stream_raises_when_all_retry_indices_fail(tmp_path):
    dataset = _make_mlx_dataset(tmp_path)
    stream = MLXDataStream(dataset, batch_size=1, prefetch_size=1, num_workers=1, drop_last=False)

    dataset.get_sample = lambda idx: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match=r"Failed to load sample after trying 4 indices\. Primary index: 0\."):
        stream._sample_transform({"idx": np.int32(0), "fallbacks": np.array([1, 2, 3], dtype=np.int32)})


@pytest.mark.skipif(not HAS_MLX_DATA, reason="mlx-data not installed")
def test_mlx_data_stream_iterates_scalar_retry_metadata_through_mlx_data(tmp_path):
    dataset = _make_mlx_dataset(tmp_path, speech_entries=4)
    stream = MLXDataStream(dataset, batch_size=2, prefetch_size=1, num_workers=1, drop_last=False)

    batch = next(iter(stream))

    assert batch["noisy_real"].shape[0] == 2
    assert batch["clean_real"].shape == batch["noisy_real"].shape
    assert batch["snr"].shape == (2,)


@pytest.mark.skipif(not HAS_MLX_DATA, reason="mlx-data not installed")
def test_mlx_data_stream_resume_skips_shuffled_prefix_deterministically(tmp_path):
    dataset = _make_mlx_dataset(tmp_path, speech_entries=6, seed=321)
    original_get_sample = dataset.get_sample
    seen: list[int] = []

    def _recording_get_sample(idx: int):
        seen.append(idx)
        return original_get_sample(idx)

    dataset.get_sample = _recording_get_sample  # type: ignore[method-assign]

    stream = MLXDataStream(dataset, batch_size=2, prefetch_size=1, num_workers=1, drop_last=False)
    stream.set_resume_position(epoch=3, batch_idx=1)

    _ = next(iter(stream))

    expected_order = list(range(len(dataset)))
    random.Random(dataset.config.seed + 3).shuffle(expected_order)
    assert seen[:2] == expected_order[2:4]


def test_prefetch_loader_raises_when_no_samples_can_be_loaded(tmp_path):
    missing = tmp_path / "missing.wav"
    cfg = DatasetConfig(
        speech_files=[str(missing)],
        noise_files=[],
        rir_files=[],
        sample_rate=16000,
        segment_length=0.5,
        train_split=1.0,
        valid_split=0.0,
    )
    dataset = DynamicDataset(cfg)
    dataset.set_split("train")
    dataset.set_epoch(0)

    loader = PrefetchDataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        prefetch_factor=1,
        drop_last=False,
    )
    with pytest.raises(RuntimeError, match="(?i)failed to load any samples|failed while loading sample"):
        list(loader)
