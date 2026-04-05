#!/usr/bin/env python3
"""Microbenchmark for z-scoring cache optimization.

Compares:
- Baseline: _z_score_clean_energy called multiple times (once per loss)
- Optimized: _z_score_clean_energy called once, result passed via _precomputed_z

Example:
    python -m df_mlx.benchmark_zscoring_cache --batch-sizes 4,8 --iters 50 --warmup 10
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import mlx.core as mx


def _make_test_inputs(batch_size: int):
    """Create test inputs for benchmarking."""
    n_freqs, n_frames = 481, 97

    clean_real = mx.random.normal((batch_size, n_frames, n_freqs)).astype(mx.float32)
    clean_imag = mx.random.normal((batch_size, n_frames, n_freqs)).astype(mx.float32)
    band_mask = mx.random.uniform(0, 1, (n_freqs,)).astype(mx.float32)
    band_bins = mx.sum(band_mask)
    vad_z_threshold = -1.0
    vad_z_slope = 0.1
    eps = 1e-8

    mx.eval(clean_real, clean_imag, band_mask)
    return (
        clean_real,
        clean_imag,
        band_mask,
        band_bins,
        vad_z_threshold,
        vad_z_slope,
        eps,
    )


def band_energy(
    real: mx.array,
    imag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    eps: float = 1e-8,
):
    """Compute band energy - simplified version for benchmark."""
    power = real**2 + imag**2
    band = mx.sum(power * band_mask, axis=-1) / (band_bins + eps)
    log_band = mx.log10(band + eps)
    return band, log_band


def _z_score_clean_energy(
    clean_real: mx.array,
    clean_imag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    eps: float = 1e-8,
):
    """Z-score clean energy - actual implementation."""
    clean_band, log_clean = band_energy(clean_real, clean_imag, band_mask, band_bins, eps)
    mu = mx.mean(log_clean, axis=1, keepdims=True)
    variance = mx.mean((log_clean - mu) ** 2, axis=1, keepdims=True)
    sigma = mx.sqrt(mx.maximum(variance, 1e-10) + eps)
    z_ref_raw = (log_clean - mu) / (sigma + eps)
    z_ref = mx.clip(z_ref_raw, -10.0, 10.0)
    z_slope = max(vad_z_slope, 1e-3)
    p_ref = mx.sigmoid((z_ref - vad_z_threshold) / z_slope)
    return clean_band, log_clean, z_ref_raw, z_ref, p_ref


def benchmark_baseline(batch_size: int, num_calls: int = 4, iters: int = 50, warmup: int = 10):
    """Baseline: call _z_score_clean_energy multiple times."""
    clean_real, clean_imag, band_mask, band_bins, vad_z_threshold, vad_z_slope, eps = _make_test_inputs(batch_size)

    # Warmup
    for _ in range(warmup):
        results = []
        for _ in range(num_calls):
            result = _z_score_clean_energy(
                clean_real,
                clean_imag,
                band_mask,
                band_bins,
                vad_z_threshold,
                vad_z_slope,
                eps,
            )
            results.append(result)
        mx.eval(*results)

    # Benchmark
    times = []
    for _ in range(iters):
        results = []
        start = time.perf_counter()
        for _ in range(num_calls):
            result = _z_score_clean_energy(
                clean_real,
                clean_imag,
                band_mask,
                band_bins,
                vad_z_threshold,
                vad_z_slope,
                eps,
            )
            results.append(result)
        mx.eval(*results)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return times


def benchmark_optimized(batch_size: int, num_calls: int = 4, iters: int = 50, warmup: int = 10):
    """Optimized: call _z_score_clean_energy once, reuse result."""
    clean_real, clean_imag, band_mask, band_bins, vad_z_threshold, vad_z_slope, eps = _make_test_inputs(batch_size)

    # Warmup
    for _ in range(warmup):
        precomputed = _z_score_clean_energy(
            clean_real,
            clean_imag,
            band_mask,
            band_bins,
            vad_z_threshold,
            vad_z_slope,
            eps,
        )
        # Simulate passing precomputed_z to 4 loss functions (just access, no recompute)
        _ = precomputed[0]  # clean_band
        _ = precomputed[1]  # log_clean
        _ = precomputed[2]  # z_ref_raw
        _ = precomputed[3]  # z_ref
        _ = precomputed[4]  # p_ref
        mx.eval()

    # Benchmark
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        precomputed = _z_score_clean_energy(
            clean_real,
            clean_imag,
            band_mask,
            band_bins,
            vad_z_threshold,
            vad_z_slope,
            eps,
        )
        # Simulate passing precomputed_z to 4 loss functions (just access, no recompute)
        _ = precomputed[0]
        _ = precomputed[1]
        _ = precomputed[2]
        _ = precomputed[3]
        _ = precomputed[4]
        mx.eval()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return times


@dataclass
class BenchmarkResult:
    batch_size: int
    baseline_mean_ms: float
    baseline_std_ms: float
    optimized_mean_ms: float
    optimized_std_ms: float
    speedup_percent: float
    calls_per_batch: int


def run_benchmarks(batch_sizes: list[int], iters: int = 50, warmup: int = 10, num_calls: int = 4):
    """Run benchmarks for multiple batch sizes."""
    results = []

    for bs in batch_sizes:
        print(f"\n=== Batch size: {bs} ===")
        print(f"Simulating {num_calls} loss functions calling z-scoring...")

        baseline_times = benchmark_baseline(bs, num_calls, iters, warmup)
        optimized_times = benchmark_optimized(bs, num_calls, iters, warmup)

        baseline_mean = statistics.mean(baseline_times)
        baseline_std = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0
        optimized_mean = statistics.mean(optimized_times)
        optimized_std = statistics.stdev(optimized_times) if len(optimized_times) > 1 else 0

        speedup = ((baseline_mean - optimized_mean) / baseline_mean) * 100

        result = BenchmarkResult(
            batch_size=bs,
            baseline_mean_ms=baseline_mean,
            baseline_std_ms=baseline_std,
            optimized_mean_ms=optimized_mean,
            optimized_std_ms=optimized_std,
            speedup_percent=speedup,
            calls_per_batch=num_calls,
        )
        results.append(result)

        print(f"Baseline:  {baseline_mean:.3f} ± {baseline_std:.3f} ms")
        print(f"Optimized: {optimized_mean:.3f} ± {optimized_std:.3f} ms")
        print(f"Speedup:   {speedup:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark z-scoring cache optimization")
    parser.add_argument("--batch-sizes", type=str, default="4,8", help="Comma-separated batch sizes")
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--calls", type=int, default=4, help="Number of loss function calls per batch")
    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print("=" * 60)
    print("Z-Scoring Cache Optimization Benchmark")
    print("=" * 60)
    print(f"Batch sizes: {batch_sizes}")
    print(f"Iterations: {args.iters}")
    print(f"Warmup: {args.warmup}")
    print(f"Loss calls simulated: {args.calls}")
    print("=" * 60)

    results = run_benchmarks(batch_sizes, args.iters, args.warmup, args.calls)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Batch':>6} | {'Baseline (ms)':>14} | {'Optimized (ms)':>15} | {'Speedup':>8}")
    print("-" * 60)
    for r in results:
        print(
            f"{r.batch_size:>6} | {r.baseline_mean_ms:>14.3f} | {r.optimized_mean_ms:>15.3f} | {r.speedup_percent:>7.1f}%"
        )

    avg_speedup = statistics.mean([r.speedup_percent for r in results])
    print("-" * 60)
    print(f"Average speedup: {avg_speedup:.1f}%")


if __name__ == "__main__":
    main()
