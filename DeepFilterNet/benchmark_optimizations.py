#!/usr/bin/env python3
"""Benchmark all optimized operations for performance measurement."""

import time
from typing import Callable, Tuple

import mlx.core as mx


def benchmark_fn(
    fn: Callable,
    args: Tuple,
    warmup: int = 3,
    runs: int = 10,
    sync: bool = True,
) -> Tuple[float, float]:
    """Benchmark a function with warmup and multiple runs.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        if sync:
            mx.eval(result)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn(*args)
        if sync:
            mx.eval(result)
        times.append((time.perf_counter() - start) * 1000)

    import numpy as np

    return np.mean(times), np.std(times)


def benchmark_stft():
    """Benchmark STFT operations."""
    from df_mlx.ops import istft, stft

    print("=" * 70)
    print("STFT BENCHMARKS")
    print("=" * 70)

    # Test different batch sizes and lengths
    configs = [
        (1, 48000, "1 sec mono"),
        (4, 48000, "1 sec batch=4"),
        (8, 48000, "1 sec batch=8"),
        (4, 96000, "2 sec batch=4"),
        (8, 96000, "2 sec batch=8"),
    ]

    for batch, samples, desc in configs:
        x = mx.random.normal((batch, samples))
        mx.eval(x)

        # STFT benchmark
        mean_ms, std_ms = benchmark_fn(lambda x: stft(x, n_fft=960, hop_length=480), (x,))
        print(f"STFT [{desc:20s}]: {mean_ms:7.2f} ± {std_ms:.2f} ms")

        # iSTFT benchmark
        real, imag = stft(x, n_fft=960, hop_length=480)
        mx.eval(real, imag)
        mean_ms, std_ms = benchmark_fn(
            lambda r, i: istft((r, i), n_fft=960, hop_length=480, length=samples),
            (real, imag),
        )
        print(f"iSTFT [{desc:20s}]: {mean_ms:7.2f} ± {std_ms:.2f} ms")

    print()


def benchmark_multiframe():
    """Benchmark multiframe operations."""
    from df_mlx.multiframe import DF, DFreal

    print("=" * 70)
    print("MULTIFRAME BENCHMARKS")
    print("=" * 70)

    # Test configurations
    configs = [
        (1, 50, "T=50 batch=1"),
        (4, 50, "T=50 batch=4"),
        (8, 50, "T=50 batch=8"),
        (4, 100, "T=100 batch=4"),
        (8, 100, "T=100 batch=8"),
        (8, 200, "T=200 batch=8"),
    ]

    df = DF(num_freqs=96, frame_size=5, lookahead=0)
    df_real = DFreal(num_freqs=96, frame_size=5, lookahead=0)

    for batch, T, desc in configs:
        spec = mx.random.normal((batch, 1, T, 96, 2))
        mx.eval(spec)

        # spec_unfold benchmark
        mean_ms, std_ms = benchmark_fn(df.spec_unfold, (spec,))
        print(f"spec_unfold [{desc:20s}]: {mean_ms:7.3f} ± {std_ms:.3f} ms")

        # spec_unfold_real benchmark
        mean_ms, std_ms = benchmark_fn(df_real.spec_unfold_real, (spec,))
        print(f"spec_unfold_real [{desc:20s}]: {mean_ms:7.3f} ± {std_ms:.3f} ms")

    print()


def benchmark_loss():
    """Benchmark loss functions."""
    from df_mlx.loss import SegmentalSiSdrLoss

    print("=" * 70)
    print("LOSS FUNCTION BENCHMARKS")
    print("=" * 70)

    loss_fn = SegmentalSiSdrLoss(segment_size=512, overlap=0.5)

    # Test configurations
    configs = [
        (1, 8000, "~0.5 sec batch=1"),
        (4, 8000, "~0.5 sec batch=4"),
        (8, 8000, "~0.5 sec batch=8"),
        (4, 48000, "1 sec batch=4"),
        (8, 48000, "1 sec batch=8"),
        (8, 96000, "2 sec batch=8"),
    ]

    for batch, samples, desc in configs:
        clean = mx.random.normal((batch, samples))
        enhanced = mx.random.normal((batch, samples))
        mx.eval(clean, enhanced)

        mean_ms, std_ms = benchmark_fn(loss_fn, (clean, enhanced))
        print(f"SegmentalSiSdrLoss [{desc:20s}]: {mean_ms:7.3f} ± {std_ms:.3f} ms")

    print()


def benchmark_full_forward():
    """Benchmark full model forward pass with optimizations."""
    from df_mlx.model import DfNet4

    print("=" * 70)
    print("FULL MODEL FORWARD PASS BENCHMARKS")
    print("=" * 70)

    model = DfNet4()
    mx.eval(model.parameters())

    # Test configurations
    configs = [
        (1, 50, "T=50 batch=1"),
        (4, 50, "T=50 batch=4"),
        (8, 50, "T=50 batch=8"),
        (4, 100, "T=100 batch=4"),
    ]

    for batch, T, desc in configs:
        feat_erb = mx.random.normal((batch, T, 32))
        feat_spec = mx.random.normal((batch, T, 96, 2))
        spec_real = mx.random.normal((batch, T, 481))
        spec_imag = mx.random.normal((batch, T, 481))
        mx.eval(feat_erb, feat_spec, spec_real, spec_imag)

        def forward(erb, spec_feat, real, imag):
            return model(
                spec=(real, imag),
                feat_erb=erb,
                feat_spec=spec_feat,
                training=False,
            )

        mean_ms, std_ms = benchmark_fn(forward, (feat_erb, feat_spec, spec_real, spec_imag))
        throughput = (batch * T * 1000) / mean_ms  # frames/sec
        print(f"Forward [{desc:20s}]: {mean_ms:7.2f} ± {std_ms:.2f} ms  ({throughput:.0f} frames/sec)")

    print()


def benchmark_compiled_vs_uncompiled():
    """Compare compiled vs uncompiled operations."""
    print("=" * 70)
    print("COMPILED VS UNCOMPILED OPERATIONS")
    print("=" * 70)

    # Simple matmul operation for comparison
    A = mx.random.normal((256, 512))
    B = mx.random.normal((512, 256))
    mx.eval(A, B)

    def matmul_uncompiled(a, b):
        return mx.matmul(a, b)

    @mx.compile
    def matmul_compiled(a, b):
        return mx.matmul(a, b)

    # Benchmark uncompiled
    times_uncompiled = []
    for _ in range(10):  # Warmup
        result = matmul_uncompiled(A, B)
        mx.eval(result)

    for _ in range(50):
        start = time.perf_counter()
        result = matmul_uncompiled(A, B)
        mx.eval(result)
        times_uncompiled.append((time.perf_counter() - start) * 1000)

    import numpy as np

    print(f"Uncompiled matmul (256x512 @ 512x256): {np.mean(times_uncompiled):.3f} ± {np.std(times_uncompiled):.3f} ms")

    # Benchmark compiled
    times_compiled = []
    for _ in range(10):  # Warmup
        result = matmul_compiled(A, B)
        mx.eval(result)

    for _ in range(50):
        start = time.perf_counter()
        result = matmul_compiled(A, B)
        mx.eval(result)
        times_compiled.append((time.perf_counter() - start) * 1000)

    print(f"Compiled matmul (256x512 @ 512x256):   {np.mean(times_compiled):.3f} ± {np.std(times_compiled):.3f} ms")

    if np.mean(times_uncompiled) > np.mean(times_compiled):
        speedup = np.mean(times_uncompiled) / np.mean(times_compiled)
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("Note: mx.compile overhead may exceed benefit for simple ops")

    print()


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("  MLX OPTIMIZATION BENCHMARK SUITE")
    print("  Hardware: Apple Silicon (MLX)")
    print("=" * 70 + "\n")

    benchmark_stft()
    benchmark_multiframe()
    benchmark_loss()
    benchmark_full_forward()
    benchmark_compiled_vs_uncompiled()

    print("=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
