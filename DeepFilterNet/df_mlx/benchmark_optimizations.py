#!/usr/bin/env python3
"""Benchmark script for df_mlx training optimizations.

Benchmarks:
- FusedMultiResSpectralLoss
- DfOp VJP
- Mel spectrogram
- Batch assembly

Example:
    python -m df_mlx.benchmark_optimizations --batch-sizes 1,4,8 --iters 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

from df_mlx.benchmark_common import safe_percentile
from df_mlx.dnsmos_proxy import MelSpectrogram
from df_mlx.dynamic_dataset import Sample, _assemble_batch
from df_mlx.loss import FusedMultiResSpectralLoss
from df_mlx.modules import DfOp


@dataclass
class OptimizationResult:
    op_name: str
    batch_size: int
    mean_ms: float
    std_ms: float
    p5_ms: float
    p50_ms: float
    p95_ms: float
    throughput_ops_per_sec: float


def _make_spectral_loss_input(
    case_batch_size: int,
) -> Tuple[FusedMultiResSpectralLoss, mx.array, mx.array]:
    loss_fn = FusedMultiResSpectralLoss(
        fft_sizes=(512, 1024, 2048),
        gamma=0.3,
        factor=1.0,
        factor_complex=0.5,
    )
    pred = mx.random.normal((case_batch_size, 48000))
    target = mx.random.normal((case_batch_size, 48000))
    mx.eval(pred, target)
    return loss_fn, pred, target


def _make_dfop_vjp_input(
    case_batch_size: int,
) -> Tuple[DfOp, mx.array, mx.array, mx.array]:
    nb_df = 96
    df_order = 5
    n_fft = 960
    n_freqs = n_fft // 2 + 1
    n_frames = 97

    op = DfOp(nb_df=nb_df, df_order=df_order, df_lookahead=0)
    spec_real = mx.random.normal((case_batch_size, n_frames, n_freqs))
    spec_imag = mx.random.normal((case_batch_size, n_frames, n_freqs))
    coef = mx.random.normal((case_batch_size, n_frames, nb_df, df_order, 2))
    mx.eval(spec_real, spec_imag, coef)
    return op, spec_real, spec_imag, coef


def _make_mel_input(case_batch_size: int) -> Tuple[MelSpectrogram, mx.array]:
    mel = MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=64,
    )
    audio = mx.random.normal((case_batch_size, 16000))
    mx.eval(audio)
    return mel, audio


def _assemble_batch_python(samples: List[Sample]) -> Dict[str, mx.array]:
    return _assemble_batch(samples)


def _make_batch_assembly_input(case_batch_size: int) -> List[Sample]:
    np.random.seed(42)
    n_freqs, n_frames = 481, 97
    erb_bins = 64
    spec_bins = 64
    samples = []
    for _ in range(case_batch_size):
        sample = Sample(
            noisy_spec=np.random.randn(n_frames, n_freqs) + 1j * np.random.randn(n_frames, n_freqs),
            clean_spec=np.random.randn(n_frames, n_freqs) + 1j * np.random.randn(n_frames, n_freqs),
            interference_spec=np.random.randn(n_frames, n_freqs) + 1j * np.random.randn(n_frames, n_freqs),
            feat_erb=np.random.randn(n_frames, erb_bins).astype(np.float32),
            feat_spec=np.random.randn(n_frames, spec_bins).astype(np.float32),
            snr=np.random.uniform(-10, 20),
            gain=np.random.uniform(0.5, 1.5),
        )
        samples.append(sample)
    return samples


def benchmark_fused_spectral_loss(batch_size: int, iters: int, warmup: int) -> OptimizationResult:
    loss_fn, pred, target = _make_spectral_loss_input(batch_size)

    for _ in range(warmup):
        _ = loss_fn(pred, target)
        mx.eval(_)

    latencies_ms: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        loss = loss_fn(pred, target)
        mx.eval(loss)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

    return OptimizationResult(
        op_name="fused_spectral_loss",
        batch_size=batch_size,
        mean_ms=mean_ms,
        std_ms=std_ms,
        p5_ms=safe_percentile(latencies_ms, 5),
        p50_ms=safe_percentile(latencies_ms, 50),
        p95_ms=safe_percentile(latencies_ms, 95),
        throughput_ops_per_sec=iters / (sum(latencies_ms) / 1000.0),
    )


def benchmark_dfop_vjp(batch_size: int, iters: int, warmup: int) -> OptimizationResult:
    op, spec_real, spec_imag, coef = _make_dfop_vjp_input(batch_size)

    def run():
        def loss_fn(coef):
            r, i = op((spec_real, spec_imag), coef)
            return mx.mean(r**2 + i**2)

        grad_fn = mx.grad(loss_fn)
        return grad_fn(coef)

    for _ in range(warmup):
        _ = run()
        mx.eval(_)

    latencies_ms: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = run()
        mx.eval(_)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

    return OptimizationResult(
        op_name="dfop_vjp",
        batch_size=batch_size,
        mean_ms=mean_ms,
        std_ms=std_ms,
        p5_ms=safe_percentile(latencies_ms, 5),
        p50_ms=safe_percentile(latencies_ms, 50),
        p95_ms=safe_percentile(latencies_ms, 95),
        throughput_ops_per_sec=iters / (sum(latencies_ms) / 1000.0),
    )


def benchmark_mel_spectrogram(batch_size: int, iters: int, warmup: int) -> OptimizationResult:
    mel, audio = _make_mel_input(batch_size)

    for _ in range(warmup):
        _ = mel(audio)
        mx.eval(_)

    latencies_ms: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = mel(audio)
        mx.eval(out)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

    return OptimizationResult(
        op_name="mel_spectrogram",
        batch_size=batch_size,
        mean_ms=mean_ms,
        std_ms=std_ms,
        p5_ms=safe_percentile(latencies_ms, 5),
        p50_ms=safe_percentile(latencies_ms, 50),
        p95_ms=safe_percentile(latencies_ms, 95),
        throughput_ops_per_sec=iters / (sum(latencies_ms) / 1000.0),
    )


def benchmark_batch_assembly(batch_size: int, iters: int, warmup: int) -> OptimizationResult:
    samples = _make_batch_assembly_input(batch_size)

    for _ in range(warmup):
        _ = _assemble_batch_python(samples)

    latencies_ms: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _assemble_batch_python(samples)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

    return OptimizationResult(
        op_name="batch_assembly",
        batch_size=batch_size,
        mean_ms=mean_ms,
        std_ms=std_ms,
        p5_ms=safe_percentile(latencies_ms, 5),
        p50_ms=safe_percentile(latencies_ms, 50),
        p95_ms=safe_percentile(latencies_ms, 95),
        throughput_ops_per_sec=iters / (sum(latencies_ms) / 1000.0),
    )


def print_table(results: List[OptimizationResult]) -> None:
    header = f"{'Op':<20} {'BS':>3} {'Mean ms':>9} {'Std ms':>8} {'P5 ms':>8} {'P50 ms':>8} {'P95 ms':>8} {'Ops/s':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.op_name:<20} {r.batch_size:>3} {r.mean_ms:>9.3f} {r.std_ms:>8.3f} "
            f"{r.p5_ms:>8.3f} {r.p50_ms:>8.3f} {r.p95_ms:>8.3f} {r.throughput_ops_per_sec:>10.1f}"
        )


def run_benchmarks(batch_sizes: List[int], iters: int, warmup: int) -> List[OptimizationResult]:
    results: List[OptimizationResult] = []

    for bs in batch_sizes:
        print(f"Benchmarking batch_size={bs} ...")

        r = benchmark_fused_spectral_loss(bs, iters, warmup)
        results.append(r)
        print(f"  fused_spectral_loss: {r.mean_ms:.3f} ms")

        r = benchmark_dfop_vjp(bs, iters, warmup)
        results.append(r)
        print(f"  dfop_vjp: {r.mean_ms:.3f} ms")

        r = benchmark_mel_spectrogram(bs, iters, warmup)
        results.append(r)
        print(f"  mel_spectrogram: {r.mean_ms:.3f} ms")

        r = benchmark_batch_assembly(bs, iters, warmup)
        results.append(r)
        print(f"  batch_assembly: {r.mean_ms:.3f} ms")

    return results


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmark_optimizations",
        description="Benchmark df_mlx training optimizations",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8",
        help="Comma-separated batch sizes (default: 1,4,8)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Number of measured iterations per case (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations per case (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/optimization_benchmark_latest.jsonl",
        help="Output JSONL file path",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> List[OptimizationResult]:
    args = parse_args(argv)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    results = run_benchmarks(batch_sizes, args.iters, args.warmup)

    print()
    print_table(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "benchmark": "optimizations",
        "batch_sizes": batch_sizes,
        "iters": args.iters,
        "warmup": args.warmup,
    }
    with open(output_path, "w") as f:
        f.write(json.dumps({"type": "metadata", **metadata}) + "\n")
        for r in results:
            f.write(json.dumps({"type": "result", **asdict(r)}) + "\n")

    print(f"\nResults written to {output_path}")
    return results


if __name__ == "__main__":
    main()
