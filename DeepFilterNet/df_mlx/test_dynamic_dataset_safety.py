import numpy as np
import pytest
from scipy.io import wavfile

from df_mlx.dynamic_dataset import DatasetConfig, DynamicDataset, PrefetchDataLoader


def _write_wav(path, sr: int = 16000, seconds: float = 1.0) -> None:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    audio = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    wavfile.write(path, sr, audio)


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
