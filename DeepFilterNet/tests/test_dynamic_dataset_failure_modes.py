import numpy as np

from df_mlx.dynamic_dataset import DatasetConfig, DynamicDataset


def _make_dataset(
    *,
    p_very_low_snr: float = 0.0,
    p_extreme_snr: float = 0.0,
    p_interfer_speech: float = 0.0,
    p_background_music: float = 0.0,
):
    cfg = DatasetConfig(
        speech_files=["spk_a.wav", "spk_b.wav", "spk_c.wav"],
        noise_files=["noise_a.wav"],
        music_files=["music_a.wav"] if p_background_music > 0 else [],
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
        p_very_low_snr=p_very_low_snr,
        p_extreme_snr=p_extreme_snr,
        p_interfer_speech=p_interfer_speech,
        p_background_music=p_background_music,
        background_music_gain_range=(0.0, 0.0),
        interfer_speech_snr_range=(0.0, 0.0),
        seed=123,
    )
    ds = DynamicDataset(cfg)
    ds.set_split("train")
    ds.set_epoch(0)

    n = ds.segment_samples

    def _mock_load_speech(idx: int, rng):
        # deterministic, index-dependent signal
        amp = 0.3 + 0.1 * float(idx)
        return np.full((n,), amp, dtype=np.float32)

    def _mock_load_noise(rng):
        return np.zeros((n,), dtype=np.float32), 0.0

    def _mock_load_music(rng):
        return np.full((n,), 0.2, dtype=np.float32), 0.0

    ds._load_speech = _mock_load_speech  # type: ignore[method-assign]
    ds._load_noise = _mock_load_noise  # type: ignore[method-assign]
    ds._load_music = _mock_load_music  # type: ignore[method-assign]
    return ds


def test_very_low_snr_sampling_path_is_reachable():
    ds = _make_dataset(p_very_low_snr=1.0, p_extreme_snr=0.0)
    sample = ds.get_sample(0)
    assert sample is not None
    assert sample.snr == -30.0


def test_extreme_snr_sampling_path_is_reachable():
    ds = _make_dataset(p_very_low_snr=0.0, p_extreme_snr=1.0)
    sample = ds.get_sample(0)
    assert sample is not None
    assert sample.snr == -20.0


def test_interfering_speech_mixing_changes_noisy_spectrum():
    baseline = _make_dataset(p_interfer_speech=0.0)
    with_interfer = _make_dataset(p_interfer_speech=1.0)

    sample_base = baseline.get_sample(0)
    sample_interfer = with_interfer.get_sample(0)

    assert sample_base is not None
    assert sample_interfer is not None

    # Interfering speech path should alter resulting noisy spectrum
    diff = np.mean(np.abs(sample_interfer.noisy_spec - sample_base.noisy_spec))
    assert diff > 1e-6
    assert sample_interfer.interference_spec.shape == sample_interfer.noisy_spec.shape
    assert np.mean(np.abs(sample_interfer.interference_spec - sample_base.interference_spec)) > 1e-6


def test_background_music_mixing_changes_noisy_spectrum():
    baseline = _make_dataset(p_background_music=0.0)
    with_music = _make_dataset(p_background_music=1.0)

    sample_base = baseline.get_sample(0)
    sample_music = with_music.get_sample(0)

    assert sample_base is not None
    assert sample_music is not None

    diff = np.mean(np.abs(sample_music.noisy_spec - sample_base.noisy_spec))
    assert diff > 1e-6
    assert sample_music.interference_spec.shape == sample_music.noisy_spec.shape
    assert np.mean(np.abs(sample_music.interference_spec - sample_base.interference_spec)) > 1e-6
