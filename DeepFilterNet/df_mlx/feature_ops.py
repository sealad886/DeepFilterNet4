"""Shared NumPy feature extraction helpers for df_mlx datasets.

These functions are used by both dynamic dataset generation and
offline data-preparation scripts. Keeping them in one module avoids
drift between training and preprocessing paths.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_stft(
    audio: np.ndarray,
    fft_size: int = 960,
    hop_size: int = 480,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute STFT of audio signal."""
    if window is None:
        window = np.sqrt(np.hanning(fft_size + 1)[:-1]).astype(np.float32)

    pad_len = fft_size - hop_size
    audio_padded = np.pad(audio, (pad_len, pad_len), mode="constant")

    num_frames = (len(audio_padded) - fft_size) // hop_size + 1
    shape = (num_frames, fft_size)
    strides = (audio_padded.strides[0] * hop_size, audio_padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio_padded, shape=shape, strides=strides, writeable=False)

    windowed = frames * window
    spec = np.fft.rfft(windowed, n=fft_size, axis=-1)
    return spec.astype(np.complex64) if spec.dtype != np.complex64 else spec


def create_erb_filterbank(
    sr: int = 48000,
    fft_size: int = 960,
    nb_erb: int = 32,
    min_freq: float = 20.0,
    max_freq: Optional[float] = None,
) -> np.ndarray:
    """Create an ERB triangular filterbank matrix."""
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

        for j, f in enumerate(freqs):
            if low <= f <= center:
                fb[j, i] = (f - low) / (center - low + 1e-10)
            elif center < f <= high:
                fb[j, i] = (high - f) / (high - center + 1e-10)

    return fb / (fb.sum(axis=0, keepdims=True) + 1e-10)


def compute_erb_features(spec: np.ndarray, erb_fb: np.ndarray) -> np.ndarray:
    """Compute ERB-band features from complex spectrum."""
    mag_sq = np.abs(spec) ** 2
    erb = np.matmul(mag_sq, erb_fb)
    erb = np.log10(np.maximum(erb, 1e-10))
    return erb.astype(np.float32)


def compute_df_features(spec: np.ndarray, nb_df: int = 96) -> np.ndarray:
    """Compute low-frequency DF complex features."""
    df_spec = spec[:, :nb_df]
    df_feat = np.stack([df_spec.real, df_spec.imag], axis=-1)
    return df_feat.astype(np.float32)
