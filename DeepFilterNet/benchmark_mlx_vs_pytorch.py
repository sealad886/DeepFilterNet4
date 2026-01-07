#!/usr/bin/env python
"""Benchmark MLX vs PyTorch-MPS implementations of DeepFilterNet4.

This script compares inference performance between:
1. MLX implementation (Apple Silicon native)
2. PyTorch with MPS backend (Metal Performance Shaders)

Usage:
    python benchmark_mlx_vs_pytorch.py
"""

import gc
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List

# Check frameworks availability
PYTORCH_AVAILABLE = False
MLX_AVAILABLE = False

try:
    import torch

    PYTORCH_AVAILABLE = torch.backends.mps.is_available()
    if PYTORCH_AVAILABLE:
        print(f"✓ PyTorch {torch.__version__} with MPS backend available")
    else:
        print("✗ PyTorch MPS not available")
except ImportError:
    print("✗ PyTorch not installed")

try:
    import mlx.core as _mx  # noqa: F401 - imported to check availability

    MLX_AVAILABLE = True
    print("✓ MLX available")
except ImportError:
    print("✗ MLX not installed")

if not PYTORCH_AVAILABLE and not MLX_AVAILABLE:
    print("\nNo backends available for benchmarking!")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    framework: str
    batch_size: int
    seq_length: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    samples_per_sec: float
    memory_mb: float


def get_pytorch_model():
    """Initialize PyTorch DfNet4 model on MPS."""
    import os

    from df.config import config as df_config

    # Load configuration from file
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/apple/config.ini")
    if os.path.exists(config_path):
        df_config.load(config_path)
    else:
        # Create a minimal config if file not found
        raise FileNotFoundError(f"Config file not found: {config_path}")

    from df.deepfilternet4 import DfNet4, ModelParams4
    from df.modules import erb_fb
    from libdf import DF

    p = ModelParams4()
    df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb = erb_fb(df_state.erb_widths(), p.sr, inverse=False)
    erb_inverse = erb_fb(df_state.erb_widths(), p.sr, inverse=True)

    # Create simple params object for benchmark
    class SimpleParams:
        def __init__(self, mp):
            self.sr = mp.sr
            self.fft_size = mp.fft_size
            self.hop_size = mp.hop_size
            self.nb_erb = mp.nb_erb
            self.nb_df = mp.nb_df

    params = SimpleParams(p)

    model = DfNet4(erb, erb_inverse, run_df=True, train_mask=False)
    model = model.to("mps")
    model.eval()

    return model, params


def get_mlx_model():
    """Initialize MLX DfNet4 model."""
    from df_mlx.config import get_default_config
    from df_mlx.model import init_model

    model = init_model()
    config = get_default_config()

    # Create params object with expected attributes
    class MLXParams:
        def __init__(self, cfg):
            self.fft_size = cfg.fft_size
            self.nb_fft = cfg.n_freqs  # fft_size // 2 + 1
            self.nb_erb = cfg.nb_erb
            self.nb_df = cfg.nb_df

    params = MLXParams(config)
    return model, params


def benchmark_pytorch(
    model, params, batch_sizes: List[int], seq_length: int, num_warmup: int, num_runs: int
) -> List[BenchmarkResult]:
    """Benchmark PyTorch model on MPS."""
    import torch

    results = []
    n_freqs = params.fft_size // 2 + 1

    for batch_size in batch_sizes:
        # Create input tensors with correct shapes for DfNet4
        # spec: [B, 1, T, F, 2] - complex spectrum as real/imag
        spec = torch.randn(batch_size, 1, seq_length, n_freqs, 2, device="mps")
        # feat_erb: [B, 1, T, E] - ERB features
        feat_erb = torch.randn(batch_size, 1, seq_length, params.nb_erb, device="mps")
        # feat_spec: [B, 1, T, F', 2] - complex spectrogram features
        feat_spec = torch.randn(batch_size, 1, seq_length, params.nb_df, 2, device="mps")

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(spec, feat_erb, feat_spec)
                torch.mps.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(spec, feat_erb, feat_spec)
                torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)  # Convert to ms

        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0
        samples_per_sec = (batch_size * 1000) / mean_ms

        results.append(
            BenchmarkResult(
                framework="PyTorch-MPS",
                batch_size=batch_size,
                seq_length=seq_length,
                mean_ms=mean_ms,
                std_ms=std_ms,
                min_ms=min(times),
                max_ms=max(times),
                samples_per_sec=samples_per_sec,
                memory_mb=0,  # MPS memory tracking is limited
            )
        )

        # Clean up
        del spec, feat_erb, feat_spec
        gc.collect()

    return results


def benchmark_mlx(
    model, params, batch_sizes: List[int], seq_length: int, num_warmup: int, num_runs: int
) -> List[BenchmarkResult]:
    """Benchmark MLX model."""
    import mlx.core as mx

    results = []

    for batch_size in batch_sizes:
        # Create input tensors
        spec_real = mx.random.normal((batch_size, seq_length, params.nb_fft))
        spec_imag = mx.random.normal((batch_size, seq_length, params.nb_fft))
        feat_erb = mx.random.normal((batch_size, seq_length, params.nb_erb))
        feat_spec = mx.random.normal((batch_size, seq_length, params.nb_df, 2))

        # Warmup
        for _ in range(num_warmup):
            _ = model((spec_real, spec_imag), feat_erb, feat_spec)
            mx.eval(model.parameters())

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            out = model((spec_real, spec_imag), feat_erb, feat_spec)
            mx.eval(out)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0
        samples_per_sec = (batch_size * 1000) / mean_ms

        results.append(
            BenchmarkResult(
                framework="MLX",
                batch_size=batch_size,
                seq_length=seq_length,
                mean_ms=mean_ms,
                std_ms=std_ms,
                min_ms=min(times),
                max_ms=max(times),
                samples_per_sec=samples_per_sec,
                memory_mb=0,
            )
        )

    return results


def print_results(pytorch_results: List[BenchmarkResult], mlx_results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS: MLX vs PyTorch-MPS")
    print("=" * 80)

    # Header
    print(
        f"\n{'Batch':<8} {'Framework':<14} {'Mean (ms)':<12} {'Std (ms)':<10} "
        f"{'Min (ms)':<10} {'Max (ms)':<10} {'Samples/s':<12}"
    )
    print("-" * 80)

    # Combine and sort by batch size
    all_results = pytorch_results + mlx_results
    all_results.sort(key=lambda x: (x.batch_size, x.framework))

    for r in all_results:
        print(
            f"{r.batch_size:<8} {r.framework:<14} {r.mean_ms:<12.2f} {r.std_ms:<10.2f} "
            f"{r.min_ms:<10.2f} {r.max_ms:<10.2f} {r.samples_per_sec:<12.1f}"
        )

    # Speedup comparison
    print("\n" + "=" * 80)
    print("SPEEDUP COMPARISON (MLX vs PyTorch-MPS)")
    print("=" * 80)
    print(f"\n{'Batch':<8} {'PyTorch (ms)':<14} {'MLX (ms)':<14} {'Speedup':<12} {'Winner':<10}")
    print("-" * 60)

    for pt_r in pytorch_results:
        mlx_r = next((r for r in mlx_results if r.batch_size == pt_r.batch_size), None)
        if mlx_r:
            speedup = pt_r.mean_ms / mlx_r.mean_ms
            winner = "MLX" if speedup > 1 else "PyTorch"
            speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1 / speedup:.2f}x slower"
            print(
                f"{pt_r.batch_size:<8} {pt_r.mean_ms:<14.2f} {mlx_r.mean_ms:<14.2f} " f"{speedup_str:<12} {winner:<10}"
            )

    # Throughput comparison
    print("\n" + "=" * 80)
    print("THROUGHPUT COMPARISON (samples/second)")
    print("=" * 80)
    print(f"\n{'Batch':<8} {'PyTorch':<14} {'MLX':<14} {'Difference':<14}")
    print("-" * 50)

    for pt_r in pytorch_results:
        mlx_r = next((r for r in mlx_results if r.batch_size == pt_r.batch_size), None)
        if mlx_r:
            diff = mlx_r.samples_per_sec - pt_r.samples_per_sec
            diff_pct = (diff / pt_r.samples_per_sec) * 100
            print(
                f"{pt_r.batch_size:<8} {pt_r.samples_per_sec:<14.1f} "
                f"{mlx_r.samples_per_sec:<14.1f} {diff_pct:+.1f}%"
            )


def main():
    """Run the benchmark."""
    print("\n" + "=" * 80)
    print("DeepFilterNet4 Benchmark: MLX vs PyTorch-MPS")
    print("=" * 80)

    # Configuration
    batch_sizes = [1, 2, 4, 8, 12, 16]
    seq_length = 100  # ~1 second of audio at typical hop size
    num_warmup = 5
    num_runs = 20

    print("\nConfiguration:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Sequence length: {seq_length} frames")
    print(f"  Warmup runs: {num_warmup}")
    print(f"  Benchmark runs: {num_runs}")

    pytorch_results = []
    mlx_results = []

    # Benchmark PyTorch-MPS
    if PYTORCH_AVAILABLE:
        print("\n" + "-" * 40)
        print("Benchmarking PyTorch-MPS...")
        print("-" * 40)

        try:
            model, params = get_pytorch_model()
            print("  Model loaded on MPS")
            print(f"  FFT size: {params.fft_size}, ERB bands: {params.nb_erb}, DF bins: {params.nb_df}")

            pytorch_results = benchmark_pytorch(model, params, batch_sizes, seq_length, num_warmup, num_runs)

            for r in pytorch_results:
                print(
                    f"  Batch {r.batch_size:2d}: {r.mean_ms:6.2f}ms ± {r.std_ms:5.2f}ms ({r.samples_per_sec:.1f} samples/s)"
                )

            # Clean up
            del model
            import torch

            torch.mps.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"  ✗ PyTorch benchmark failed: {e}")
            import traceback

            traceback.print_exc()

    # Benchmark MLX
    if MLX_AVAILABLE:
        print("\n" + "-" * 40)
        print("Benchmarking MLX...")
        print("-" * 40)

        try:
            model, params = get_mlx_model()
            print("  Model loaded")
            print(f"  FFT bins: {params.nb_fft}, ERB bands: {params.nb_erb}, DF bins: {params.nb_df}")

            mlx_results = benchmark_mlx(model, params, batch_sizes, seq_length, num_warmup, num_runs)

            for r in mlx_results:
                print(
                    f"  Batch {r.batch_size:2d}: {r.mean_ms:6.2f}ms ± {r.std_ms:5.2f}ms ({r.samples_per_sec:.1f} samples/s)"
                )

            # Clean up
            del model
            gc.collect()

        except Exception as e:
            print(f"  ✗ MLX benchmark failed: {e}")
            import traceback

            traceback.print_exc()

    # Print comparison
    if pytorch_results and mlx_results:
        print_results(pytorch_results, mlx_results)
    elif pytorch_results:
        print("\nOnly PyTorch results available (MLX benchmark failed)")
    elif mlx_results:
        print("\nOnly MLX results available (PyTorch benchmark failed)")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
