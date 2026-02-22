#!/usr/bin/env python3
"""Debug script to investigate single-frame NaN issue."""

import mlx.core as mx
import numpy as np

batch_size = 2
n_frames = 1
n_freqs = 481
eps = 1e-7

np.random.seed(42)
clean_real = mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32))
clean_power = clean_real**2

# Create band mask for speech band (300-3400 Hz at 48kHz, 960 fft)
sr = 48000
fft_size = 960
freq_per_bin = sr / fft_size
low_bin = int(300 / freq_per_bin)
high_bin = int(3400 / freq_per_bin)
band_mask = mx.array(
    np.where((np.arange(n_freqs) >= low_bin) & (np.arange(n_freqs) <= high_bin), 1.0, 0.0).astype(np.float32)
)
band_bins = float(mx.sum(band_mask).item())

clean_band = mx.sum(clean_power * band_mask, axis=-1) / (band_bins + eps)
print(f"clean_band shape: {clean_band.shape}")
mx.eval(clean_band)
print(f"clean_band values: {np.asarray(clean_band)}")

log_clean = mx.log10(clean_band + eps)
print(f"log_clean shape: {log_clean.shape}")
mx.eval(log_clean)
print(f"log_clean values: {np.asarray(log_clean)}")

# Mean over time dimension (axis=1)
mu = mx.mean(log_clean, axis=1, keepdims=True)
print(f"mu shape: {mu.shape}")
mx.eval(mu)
print(f"mu values: {np.asarray(mu)}")

# Variance over time dimension (axis=1) - PROBLEM: with 1 frame, variance is 0!
variance = mx.mean((log_clean - mu) ** 2, axis=1, keepdims=True)
print(f"variance shape: {variance.shape}")
mx.eval(variance)
print(f"variance values: {np.asarray(variance)}")

_MIN_VARIANCE = 1e-4
sigma = mx.sqrt(mx.maximum(variance, _MIN_VARIANCE) + eps)
print(f"sigma shape: {sigma.shape}")
mx.eval(sigma)
print(f"sigma values: {np.asarray(sigma)}")

z = (log_clean - mu) / (sigma + eps)
print(f"z shape: {z.shape}")
mx.eval(z)
print(f"z values: {np.asarray(z)}")
print(f"z is finite: {np.all(np.isfinite(np.asarray(z)))}")
