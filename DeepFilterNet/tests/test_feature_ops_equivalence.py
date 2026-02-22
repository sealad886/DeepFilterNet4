import numpy as np

from df_mlx import dynamic_dataset, prepare_data
from df_mlx.feature_ops import (
    compute_df_features,
    compute_erb_features,
    compute_stft,
    create_erb_filterbank,
)


def _legacy_compute_stft(
    audio: np.ndarray,
    fft_size: int = 960,
    hop_size: int = 480,
) -> np.ndarray:
    window = np.sqrt(np.hanning(fft_size + 1)[:-1]).astype(np.float32)
    pad_len = fft_size - hop_size
    audio_padded = np.pad(audio, (pad_len, pad_len), mode="constant")
    num_frames = (len(audio_padded) - fft_size) // hop_size + 1
    shape = (num_frames, fft_size)
    strides = (audio_padded.strides[0] * hop_size, audio_padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio_padded, shape=shape, strides=strides, writeable=False)
    spec = np.fft.rfft(frames * window, n=fft_size, axis=-1)
    return spec.astype(np.complex64) if spec.dtype != np.complex64 else spec


def _legacy_create_erb_filterbank(
    sr: int = 48000,
    fft_size: int = 960,
    nb_erb: int = 32,
    min_freq: float = 20.0,
    max_freq: float | None = None,
) -> np.ndarray:
    if max_freq is None:
        max_freq = sr / 2
    n_freqs = fft_size // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)

    def hz_to_erb(f):
        return 9.265 * np.log(1 + f / (24.7 * 9.265))

    def erb_to_hz(erb):
        return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)

    erb_min = hz_to_erb(min_freq)
    erb_max = hz_to_erb(max_freq)
    erb_centers = np.linspace(erb_min, erb_max, nb_erb)
    center_freqs = erb_to_hz(erb_centers)

    fb = np.zeros((n_freqs, nb_erb), dtype=np.float32)
    for i in range(nb_erb):
        center = center_freqs[i]
        erb_bandwidth = 24.7 * (4.37 * center / 1000 + 1)
        low = center - erb_bandwidth / 2
        high = center + erb_bandwidth / 2
        for j, freq in enumerate(freqs):
            if low <= freq <= center:
                fb[j, i] = (freq - low) / (center - low + 1e-10)
            elif center < freq <= high:
                fb[j, i] = (high - freq) / (high - center + 1e-10)
    return fb / (fb.sum(axis=0, keepdims=True) + 1e-10)


def _legacy_compute_erb_features(spec: np.ndarray, erb_fb: np.ndarray) -> np.ndarray:
    mag_sq = np.abs(spec) ** 2
    erb = np.matmul(mag_sq, erb_fb)
    erb = np.log10(np.maximum(erb, 1e-10))
    return erb.astype(np.float32)


def _legacy_compute_df_features(spec: np.ndarray, nb_df: int = 96) -> np.ndarray:
    df_spec = spec[:, :nb_df]
    return np.stack([df_spec.real, df_spec.imag], axis=-1).astype(np.float32)


def test_feature_helpers_match_legacy_behavior() -> None:
    rng = np.random.default_rng(1234)
    audio = rng.standard_normal(4096, dtype=np.float32)

    stft_new = compute_stft(audio, fft_size=512, hop_size=128)
    stft_old = _legacy_compute_stft(audio, fft_size=512, hop_size=128)
    np.testing.assert_allclose(stft_new, stft_old, rtol=0.0, atol=0.0)

    fb_new = create_erb_filterbank(sr=16000, fft_size=512, nb_erb=24)
    fb_old = _legacy_create_erb_filterbank(sr=16000, fft_size=512, nb_erb=24)
    np.testing.assert_allclose(fb_new, fb_old, rtol=1e-6, atol=1e-7)

    erb_new = compute_erb_features(stft_new, fb_new)
    erb_old = _legacy_compute_erb_features(stft_old, fb_old)
    np.testing.assert_allclose(erb_new, erb_old, rtol=1e-6, atol=1e-7)

    df_new = compute_df_features(stft_new, nb_df=24)
    df_old = _legacy_compute_df_features(stft_old, nb_df=24)
    np.testing.assert_allclose(df_new, df_old, rtol=0.0, atol=0.0)


def test_prepare_data_and_dynamic_dataset_use_canonical_feature_helpers() -> None:
    assert dynamic_dataset.compute_stft is compute_stft
    assert dynamic_dataset.create_erb_filterbank is create_erb_filterbank
    assert dynamic_dataset.compute_erb_features is compute_erb_features
    assert dynamic_dataset.compute_df_features is compute_df_features

    assert prepare_data.compute_stft is compute_stft
    assert prepare_data.create_erb_filterbank is create_erb_filterbank
    assert prepare_data.compute_erb_features is compute_erb_features
    assert prepare_data.compute_df_features is compute_df_features
