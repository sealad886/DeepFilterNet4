#!/usr/bin/env python3
"""Baseline comparison benchmarks for df_mlx optimizations.

This script compares optimized implementations against baseline (pre-optimization)
versions to prove speedup.
"""

import time
from typing import Tuple

import mlx.core as mx
import numpy as np


def spectral_loss_baseline(
    pred: mx.array,
    target: mx.array,
    windows: list,
    eps: float = 1e-10,
    gamma: float = 0.3,
) -> mx.array:
    """Baseline FusedSpectralLoss with scalar accumulation."""

    if pred.ndim == 1:
        pred = mx.expand_dims(pred, axis=0)
    if target.ndim == 1:
        target = mx.expand_dims(target, axis=0)

    total_loss = mx.array(0.0)
    for i, (fft_size, hop_size) in enumerate([(512, 128), (1024, 256), (2048, 512)]):
        window = windows[i]

        pred_real, pred_imag = _stft_inline_baseline(pred, fft_size, hop_size, window)
        target_real, target_imag = _stft_inline_baseline(target, fft_size, hop_size, window)

        pred_mag = mx.sqrt(pred_real**2 + pred_imag**2 + eps)
        target_mag = mx.sqrt(target_real**2 + target_imag**2 + eps)

        if gamma != 1.0:
            pred_mag_c = mx.power(pred_mag, gamma)
            target_mag_c = mx.power(target_mag, gamma)
        else:
            pred_mag_c = pred_mag
            target_mag_c = target_mag

        mag_loss = mx.mean((pred_mag_c - target_mag_c) ** 2)
        total_loss = total_loss + mag_loss

    return total_loss / 3


def _stft_inline_baseline(x: mx.array, n_fft: int, hop_length: int, window: mx.array) -> Tuple[mx.array, mx.array]:
    """Original STFT inline without optimizations."""
    pad_amount = n_fft // 2
    x_padded = mx.pad(x, [(0, 0), (pad_amount, pad_amount)])
    num_samples = x_padded.shape[1]
    num_frames = (num_samples - n_fft) // hop_length + 1

    frames = []
    for i in range(num_frames):
        start = i * hop_length
        frame = x_padded[0, start : start + n_fft] * window
        frames.append(frame)

    frames_stacked = mx.stack(frames, axis=0)
    frames_stacked = mx.expand_dims(frames_stacked, axis=0)
    fft_out = mx.fft.rfft(frames_stacked, axis=-1)
    return mx.real(fft_out)[0], mx.imag(fft_out)[0]


def dfop_vjp_baseline(
    spec_real_pad: mx.array,
    spec_imag_pad: mx.array,
    coef_real: mx.array,
    coef_imag: mx.array,
    d_out_real: mx.array,
    d_out_imag: mx.array,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Baseline VJP with Python loop (pre-optimization)."""
    df_order = coef_real.shape[-1]
    output_time = coef_real.shape[1]
    batch_size = coef_real.shape[0]
    nb_df = coef_real.shape[2]

    d_coef_real = mx.zeros_like(coef_real)
    d_coef_imag = mx.zeros_like(coef_imag)
    d_spec_real_pad = mx.zeros_like(spec_real_pad)
    d_spec_imag_pad = mx.zeros_like(spec_imag_pad)

    for k in range(df_order):
        cr_k = coef_real[:, :, :, k]
        ci_k = coef_imag[:, :, :, k]

        grad_r = cr_k * d_out_real + ci_k * d_out_imag
        grad_i = cr_k * d_out_imag - ci_k * d_out_real

        d_spec_real_pad = d_spec_real_pad.at[:, k : k + output_time, :].add(grad_r)
        d_spec_imag_pad = d_spec_imag_pad.at[:, k : k + output_time, :].add(grad_i)

        for b in range(batch_size):
            for t in range(output_time):
                for f in range(nb_df):
                    s_r = spec_real_pad[b, t + k, f]
                    s_i = spec_imag_pad[b, t + k, f]
                    d_coef_real[b, t, f, k] = s_r * d_out_real[b, t, f] + s_i * d_out_imag[b, t, f]
                    d_coef_imag[b, t, f, k] = s_r * d_out_imag[b, t, f] - s_i * d_out_real[b, t, f]

    return d_spec_real_pad, d_spec_imag_pad, d_coef_real, d_coef_imag


def dfop_vjp_optimized(
    spec_real_pad: mx.array,
    spec_imag_pad: mx.array,
    coef_real: mx.array,
    coef_imag: mx.array,
    d_out_real: mx.array,
    d_out_imag: mx.array,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Optimized VJP with vectorization (post-optimization)."""
    df_order = coef_real.shape[-1]
    output_time = coef_real.shape[1]
    batch_size = coef_real.shape[0]
    nb_df = coef_real.shape[2]
    n_freqs = spec_real_pad.shape[2]

    frame_starts = mx.arange(output_time)
    offsets = mx.arange(df_order)
    indices = frame_starts[:, None] + offsets[None, :]
    flat_idx = indices.flatten()

    in_real = mx.take(spec_real_pad, flat_idx, axis=1).reshape(batch_size, output_time, df_order, n_freqs)
    in_imag = mx.take(spec_imag_pad, flat_idx, axis=1).reshape(batch_size, output_time, df_order, n_freqs)
    in_real = mx.transpose(in_real, (0, 1, 3, 2))
    in_imag = mx.transpose(in_imag, (0, 1, 3, 2))
    in_real = in_real[:, :, :nb_df, :]
    in_imag = in_imag[:, :, :nb_df, :]

    d_out_r = mx.expand_dims(d_out_real, axis=-1)
    d_out_i = mx.expand_dims(d_out_imag, axis=-1)

    d_coef_real = in_real * d_out_r + in_imag * d_out_i
    d_coef_imag = in_real * d_out_i - in_imag * d_out_r

    d_spec_real_pad = mx.zeros_like(spec_real_pad)
    d_spec_imag_pad = mx.zeros_like(spec_imag_pad)

    grad_r_all = coef_real * mx.expand_dims(d_out_real, axis=-1) + coef_imag * mx.expand_dims(d_out_imag, axis=-1)
    grad_i_all = coef_real * mx.expand_dims(d_out_imag, axis=-1) - coef_imag * mx.expand_dims(d_out_real, axis=-1)

    for k in range(df_order):
        d_spec_real_pad = d_spec_real_pad.at[:, k : k + output_time, :nb_df].add(grad_r_all[:, :, :, k])
        d_spec_imag_pad = d_spec_imag_pad.at[:, k : k + output_time, :nb_df].add(grad_i_all[:, :, :, k])

    return d_spec_real_pad, d_spec_imag_pad, d_coef_real, d_coef_imag


def batch_assemble_baseline(samples: list) -> dict:
    """Baseline batch assembly with loop."""
    n = len(samples)
    s0 = samples[0]
    spec_shape = s0.noisy_spec.real.shape

    noisy_real = np.empty((n, *spec_shape), dtype=np.float32)
    noisy_imag = np.empty((n, *spec_shape), dtype=np.float32)
    clean_real = np.empty((n, *spec_shape), dtype=np.float32)
    clean_imag = np.empty((n, *spec_shape), dtype=np.float32)

    for i, s in enumerate(samples):
        noisy_real[i] = s.noisy_spec.real
        noisy_imag[i] = s.noisy_spec.imag
        clean_real[i] = s.clean_spec.real
        clean_imag[i] = s.clean_spec.imag

    return {
        "noisy_real": mx.array(noisy_real),
        "noisy_imag": mx.array(noisy_imag),
        "clean_real": mx.array(clean_real),
        "clean_imag": mx.array(clean_imag),
    }


def benchmark(fn, *args, warmup=10, iterations=50):
    """Run benchmark and return mean time in ms."""
    for _ in range(warmup):
        fn(*args)
        mx.eval()

    t0 = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
        mx.eval()
    t1 = time.perf_counter()

    return (t1 - t0) / iterations * 1000


def main():
    print("=" * 70)
    print("DF_MLX OPTIMIZATION BASELINE COMPARISON")
    print("=" * 70)

    results = {}

    # 1. Spectral Loss
    print("\n1. SPECTRAL LOSS (batch=4, 48k samples)")
    print("-" * 50)

    from df_mlx.loss import FusedMultiResSpectralLoss, FusedSpectralLoss
    from df_mlx.ops import get_window

    pred = mx.random.normal((4, 48000))
    target = mx.random.normal((4, 48000))
    configs = {"fft_sizes": (512, 1024, 2048), "gamma": 0.3}
    windows = [
        get_window("sqrt_hann", 512),
        get_window("sqrt_hann", 1024),
        get_window("sqrt_hann", 2048),
    ]

    # Baseline
    t_baseline = benchmark(lambda p, t: spectral_loss_baseline(p, t, windows), pred, target)
    print(f"  Baseline (scalar accumulation):     {t_baseline:.2f} ms")

    # Original FusedSpectralLoss
    fused_orig = FusedSpectralLoss(**configs)
    t_orig = benchmark(fused_orig, pred, target)
    print(f"  FusedSpectralLoss (original):       {t_orig:.2f} ms")

    # New FusedMultiResSpectralLoss
    fused_new = FusedMultiResSpectralLoss(**configs)
    t_new = benchmark(fused_new, pred, target)
    print(f"  FusedMultiResSpectralLoss (new):    {t_new:.2f} ms")

    print(f"\n  Speedup vs Baseline:      {t_baseline / t_new:.2f}x")
    print(f"  Speedup vs Original:      {t_orig / t_new:.2f}x")
    results["spectral_loss"] = (t_baseline, t_orig, t_new)

    # 2. DfOp VJP
    print("\n2. DFOP VJP (batch=4, df_order=5)")
    print("-" * 50)

    nb_df, df_order, n_frames, batch_size = 96, 5, 97, 4
    spec_real_pad = mx.random.normal((batch_size, n_frames + df_order - 1, nb_df)).astype(mx.float32)
    spec_imag_pad = mx.random.normal((batch_size, n_frames + df_order - 1, nb_df)).astype(mx.float32)
    coef_real = mx.random.normal((batch_size, n_frames, nb_df, df_order)).astype(mx.float32)
    coef_imag = mx.random.normal((batch_size, n_frames, nb_df, df_order)).astype(mx.float32)
    d_out_real = mx.random.normal((batch_size, n_frames, nb_df)).astype(mx.float32)
    d_out_imag = mx.random.normal((batch_size, n_frames, nb_df)).astype(mx.float32)

    # Baseline VJP (Python loop)
    print("  Running baseline VJP (this may take a moment)...")
    t_baseline = benchmark(
        lambda sr, si, cr, ci, dr, di: dfop_vjp_baseline(sr, si, cr, ci, dr, di),
        spec_real_pad,
        spec_imag_pad,
        coef_real,
        coef_imag,
        d_out_real,
        d_out_imag,
        iterations=10,
    )
    print(f"  Baseline (Python loop):            {t_baseline:.2f} ms")

    # Optimized VJP (vectorized)
    t_optimized = benchmark(
        lambda sr, si, cr, ci, dr, di: dfop_vjp_optimized(sr, si, cr, ci, dr, di),
        spec_real_pad,
        spec_imag_pad,
        coef_real,
        coef_imag,
        d_out_real,
        d_out_imag,
    )
    print(f"  Optimized (vectorized):           {t_optimized:.2f} ms")

    print(f"\n  Speedup:                          {t_baseline / t_optimized:.2f}x")
    results["dfop_vjp"] = (t_baseline, t_optimized)

    # 3. Batch Assembly
    print("\n3. BATCH ASSEMBLY (batch=8)")
    print("-" * 50)

    from df_mlx.dynamic_dataset import Sample, _assemble_batch

    def make_samples(n):
        return [
            Sample(
                noisy_spec=np.random.randn(97, 481) + 1j * np.random.randn(97, 481),
                clean_spec=np.random.randn(97, 481) + 1j * np.random.randn(97, 481),
                interference_spec=np.random.randn(97, 481) + 1j * np.random.randn(97, 481),
                feat_erb=np.random.randn(97, 32).astype(np.float32),
                feat_spec=np.random.randn(97, 96, 2).astype(np.float32),
                snr=0.0,
                gain=0.0,
            )
            for _ in range(n)
        ]

    samples = make_samples(8)

    # Baseline
    t_baseline = benchmark(lambda s: batch_assemble_baseline(s), samples)
    print(f"  Baseline (no contiguous):          {t_baseline:.2f} ms")

    # Optimized
    t_optimized = benchmark(lambda s: _assemble_batch(s), samples)
    print(f"  Optimized (contiguous + fastpath):{t_optimized:.2f} ms")

    print(f"\n  Speedup:                          {t_baseline / t_optimized:.2f}x")
    results["batch_assembly"] = (t_baseline, t_optimized)

    # 4. Mel Spectrogram (already vectorized)
    print("\n4. MEL SPECTROGRAM (already vectorized)")
    print("-" * 50)
    print("  Status: Already using mx.matmul - no change needed")
    print("  See benchmark for current performance: 0.30 ms (batch=4)")
    results["mel"] = None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Component':<25} {'Baseline':<12} {'Optimized':<12} {'Speedup':<10}")
    print("-" * 70)

    if results["spectral_loss"]:
        b, o, n = results["spectral_loss"]
        print(f"{'Spectral Loss (new)':<25} {b:>8.2f} ms   {n:>8.2f} ms   {b / n:>6.2f}x")
    if results["dfop_vjp"]:
        b, n = results["dfop_vjp"]
        print(f"{'DfOp VJP':<25} {b:>8.2f} ms   {n:>8.2f} ms   {b / n:>6.2f}x")
    if results["batch_assembly"]:
        b, n = results["batch_assembly"]
        print(f"{'Batch Assembly':<25} {b:>8.2f} ms   {n:>8.2f} ms   {b / n:>6.2f}x")

    print("=" * 70)


if __name__ == "__main__":
    main()
