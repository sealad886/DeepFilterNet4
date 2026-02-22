#!/usr/bin/env python3
"""Debug script to find NaN source in single-frame case."""

import sys

sys.path.insert(0, "/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/df_mlx")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from train_dynamic import _build_speech_band_mask, _compute_awesome_losses  # noqa: E402

batch_size = 2
n_frames = 1
n_freqs = 481

np.random.seed(42)
clean_real = mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32))
clean_imag = mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32))
noisy_real = clean_real + mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32) * 0.5)
noisy_imag = clean_imag + mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32) * 0.5)
out_real = clean_real
out_imag = clean_imag
snr = mx.array([10.0, 5.0])

band_mask_arr, band_bins = _build_speech_band_mask(n_freqs, 48000, 300.0, 3400.0)

result = _compute_awesome_losses(
    noisy_real,
    noisy_imag,
    clean_real,
    clean_imag,
    out_real,
    out_imag,
    snr,
    band_mask_arr,
    band_bins,
    mask_sharpness=6.0,
    vad_z_threshold=0.0,
    vad_z_slope=1.0,
    vad_snr_gate_db=-10.0,
    vad_snr_gate_width=6.0,
    proxy_enabled=True,
)

mx.eval(result)

names = [
    "awesome_loss",
    "speech_loss",
    "noise_loss",
    "smooth_loss",
    "mask",
    "proxy_frame",
    "speech_ratio",
    "music_gate",
    "musicness",
    "mod_energy",
    "energy_boost",
    "snr_boost",
]

for i, (name, arr) in enumerate(zip(names, result)):
    arr_np = np.asarray(arr)
    is_finite = np.all(np.isfinite(arr_np))
    print(f"{name}: shape={arr.shape}, finite={is_finite}")
    if not is_finite:
        print(f"  PROBLEM VALUES: {arr_np}")
