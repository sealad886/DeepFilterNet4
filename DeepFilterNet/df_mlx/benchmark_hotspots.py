#!/usr/bin/env python3
"""Microbenchmark harness for hotspot ops: STFT, iSTFT, MelSpectrogram, DfOp, SpectralLoss.

Measures per-op latency (mean, std, p5, p50, p95) and throughput across
configurable batch sizes.  Outputs JSONL to ``logs/hotspot_benchmark_latest.jsonl``
with reproducibility metadata, and prints a human-readable table to stdout.

Example:
    python -m df_mlx.benchmark_hotspots --batch-sizes 1,4,8 --iters 50 --warmup 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import mlx.core as mx

from df_mlx.benchmark_train_step import collect_reproducibility_metadata
from df_mlx.dnsmos_proxy import MelSpectrogram
from df_mlx.loss import SpectralLoss
from df_mlx.modules import DfOp
from df_mlx.ops import istft, stft

# ---------------------------------------------------------------------------
# Benchmark config / result dataclasses
# ---------------------------------------------------------------------------

OP_NAMES = ("stft", "istft", "mel_spec", "dfop", "spectral_loss")


@dataclass(frozen=True)
class HotspotCase:
    """Describes a single benchmark configuration."""

    op_name: str
    batch_size: int
    n_fft: int
    hop_length: int
    segment_samples: int
    warmup_iters: int
    bench_iters: int


@dataclass
class HotspotResult:
    """Stores timing results for a single benchmark run."""

    op_name: str
    batch_size: int
    mean_ms: float
    std_ms: float
    p5_ms: float
    p50_ms: float
    p95_ms: float
    throughput_ops_per_sec: float


# ---------------------------------------------------------------------------
# Regression-threshold helpers (mirrors benchmark_train_step)
# ---------------------------------------------------------------------------

THRESHOLD_LATENCY_FACTOR = 1.15
THRESHOLD_CV_MAX = 0.20


def check_hotspot_regression(
    result: HotspotResult,
    baseline_p50_ms: float,
    baseline_p95_ms: float,
) -> Dict[str, Any]:
    """Evaluate pass/fail for a single result against a baseline.

    Returns a dict with per-gate results and an overall ``passed`` flag.
    """
    import os

    override = os.environ.get("BENCHMARK_OVERRIDE", "") == "1"

    latency_ok = result.p95_ms <= baseline_p95_ms * THRESHOLD_LATENCY_FACTOR
    cv = (result.std_ms / result.mean_ms) if result.mean_ms > 0 else float("inf")
    variance_ok = cv <= THRESHOLD_CV_MAX

    passed = latency_ok and variance_ok
    if override and not passed:
        passed = True

    return {
        "passed": passed,
        "override": override and not (latency_ok and variance_ok),
        "latency": {
            "ok": latency_ok,
            "new_p50_ms": result.p50_ms,
            "new_p95_ms": result.p95_ms,
            "baseline_p50_ms": baseline_p50_ms,
            "baseline_p95_ms": baseline_p95_ms,
        },
        "variance": {"ok": variance_ok, "cv": cv},
    }


# ---------------------------------------------------------------------------
# Default benchmark matrix
# ---------------------------------------------------------------------------


def build_default_matrix(
    batch_sizes: List[int],
    bench_iters: int = 50,
    warmup_iters: int = 10,
) -> List[HotspotCase]:
    """Build the default set of benchmark cases across all ops and batch sizes."""
    cases: List[HotspotCase] = []
    for bs in batch_sizes:
        cases.append(HotspotCase("stft", bs, 960, 480, 48000, warmup_iters, bench_iters))
        cases.append(HotspotCase("istft", bs, 960, 480, 48000, warmup_iters, bench_iters))
        cases.append(HotspotCase("mel_spec", bs, 512, 160, 16000, warmup_iters, bench_iters))
        cases.append(HotspotCase("dfop", bs, 960, 480, 48000, warmup_iters, bench_iters))
        cases.append(HotspotCase("spectral_loss", bs, 960, 480, 48000, warmup_iters, bench_iters))
    return cases


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------


def _make_stft_input(case: HotspotCase) -> mx.array:
    return mx.random.normal((case.batch_size, case.segment_samples))


def _make_istft_input(case: HotspotCase) -> Tuple[mx.array, mx.array]:
    audio = mx.random.normal((case.batch_size, case.segment_samples))
    spec = stft(audio, n_fft=case.n_fft, hop_length=case.hop_length)
    mx.eval(spec[0], spec[1])
    return spec


def _make_mel_input(case: HotspotCase) -> mx.array:
    return mx.random.normal((case.batch_size, case.segment_samples))


def _make_dfop_input(case: HotspotCase) -> Tuple[DfOp, Tuple[mx.array, mx.array], mx.array]:
    nb_df = 96
    df_order = 5
    n_freqs = case.n_fft // 2 + 1
    n_frames = (case.segment_samples - case.n_fft) // case.hop_length + 1

    op = DfOp(nb_df=nb_df, df_order=df_order, df_lookahead=0)
    spec_real = mx.random.normal((case.batch_size, n_frames, n_freqs))
    spec_imag = mx.random.normal((case.batch_size, n_frames, n_freqs))
    coef = mx.random.normal((case.batch_size, n_frames, nb_df, df_order, 2))
    mx.eval(spec_real, spec_imag, coef)
    return op, (spec_real, spec_imag), coef


def _make_spectral_loss_input(case: HotspotCase) -> Tuple[SpectralLoss, mx.array, mx.array]:
    loss_fn = SpectralLoss(fft_sizes=(512, 1024, 2048), gamma=0.3, factor=1.0, factor_complex=0.5)
    pred = mx.random.normal((case.batch_size, case.segment_samples))
    target = mx.random.normal((case.batch_size, case.segment_samples))
    mx.eval(pred, target)
    return loss_fn, pred, target


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _safe_percentile(values: List[float], pct: float) -> float:
    """Return the *pct*-th percentile of *values* (0-100 scale)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (pct / 100.0) * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _build_op_fn(case: HotspotCase) -> Tuple[Callable[[], None], Any]:
    """Return ``(run_once_fn, cached_data)`` for the given case.

    ``run_once_fn()`` calls the op *and* ``mx.eval`` so that timings
    reflect actual GPU work, not just dispatch.
    """
    if case.op_name == "stft":
        audio = _make_stft_input(case)
        mx.eval(audio)

        def run() -> None:
            r, i = stft(audio, n_fft=case.n_fft, hop_length=case.hop_length)
            mx.eval(r, i)

        return run, audio

    if case.op_name == "istft":
        spec = _make_istft_input(case)

        def run() -> None:
            out = istft(spec, n_fft=case.n_fft, hop_length=case.hop_length)
            mx.eval(out)

        return run, spec

    if case.op_name == "mel_spec":
        audio = _make_mel_input(case)
        mx.eval(audio)
        mel = MelSpectrogram(
            sample_rate=case.segment_samples,
            n_fft=case.n_fft,
            hop_length=case.hop_length,
            n_mels=64,
        )

        def run() -> None:
            out = mel(audio)
            mx.eval(out)

        return run, (mel, audio)

    if case.op_name == "dfop":
        op, spec, coef = _make_dfop_input(case)

        def run() -> None:
            r, i = op(spec, coef)
            mx.eval(r, i)

        return run, (op, spec, coef)

    if case.op_name == "spectral_loss":
        loss_fn, pred, target = _make_spectral_loss_input(case)

        def run() -> None:
            loss = loss_fn(pred, target)
            mx.eval(loss)

        return run, (loss_fn, pred, target)

    raise ValueError(f"Unknown op_name: {case.op_name!r}")


def run_case(case: HotspotCase) -> HotspotResult:
    """Run warmup + benchmark iterations for *case* and return a ``HotspotResult``."""
    run_fn, _data = _build_op_fn(case)

    # Warmup (not measured)
    for _ in range(case.warmup_iters):
        run_fn()

    # Measured iterations
    latencies_ms: List[float] = []
    total_start = time.perf_counter()
    for _ in range(case.bench_iters):
        t0 = time.perf_counter()
        run_fn()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    total_elapsed = time.perf_counter() - total_start

    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    p5_ms = _safe_percentile(latencies_ms, 5)
    p50_ms = _safe_percentile(latencies_ms, 50)
    p95_ms = _safe_percentile(latencies_ms, 95)
    throughput = case.bench_iters / total_elapsed if total_elapsed > 0 else 0.0

    return HotspotResult(
        op_name=case.op_name,
        batch_size=case.batch_size,
        mean_ms=mean_ms,
        std_ms=std_ms,
        p5_ms=p5_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        throughput_ops_per_sec=throughput,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_table(results: List[HotspotResult]) -> None:
    """Print a human-readable table of benchmark results to stdout."""
    header = (
        f"{'Op':<16} {'BS':>3} {'Mean ms':>9} {'Std ms':>8} " f"{'P5 ms':>8} {'P50 ms':>8} {'P95 ms':>8} {'Ops/s':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.op_name:<16} {r.batch_size:>3} {r.mean_ms:>9.3f} {r.std_ms:>8.3f} "
            f"{r.p5_ms:>8.3f} {r.p50_ms:>8.3f} {r.p95_ms:>8.3f} "
            f"{r.throughput_ops_per_sec:>10.1f}"
        )


def write_jsonl(
    results: List[HotspotResult],
    output_path: Path,
    metadata: Dict[str, Any],
) -> None:
    """Write results as JSONL (one line per result), with metadata as the first line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(json.dumps({"type": "metadata", **metadata}) + "\n")
        for r in results:
            f.write(json.dumps({"type": "result", **asdict(r)}) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="benchmark_hotspots",
        description="Microbenchmark harness for STFT, iSTFT, MelSpectrogram, DfOp, SpectralLoss",
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
        default=50,
        help="Number of measured iterations per case (default: 50)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations per case (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/hotspot_benchmark_latest.jsonl",
        help="Output JSONL file path (default: logs/hotspot_benchmark_latest.jsonl)",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=",".join(OP_NAMES),
        help=f"Comma-separated ops to benchmark (default: {','.join(OP_NAMES)})",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> List[HotspotResult]:
    """Entry point: parse args, run benchmarks, write output."""
    args = parse_args(argv)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    selected_ops = set(args.ops.split(","))

    cases = build_default_matrix(batch_sizes, bench_iters=args.iters, warmup_iters=args.warmup)
    cases = [c for c in cases if c.op_name in selected_ops]

    metadata = collect_reproducibility_metadata(
        config={
            "benchmark": "hotspot_microbench",
            "batch_sizes": batch_sizes,
            "bench_iters": args.iters,
            "warmup_iters": args.warmup,
            "ops": sorted(selected_ops),
        }
    )

    results: List[HotspotResult] = []
    for case in cases:
        print(f"Benchmarking {case.op_name} (batch_size={case.batch_size}) ...", flush=True)
        result = run_case(case)
        results.append(result)

    print()
    print_table(results)

    output_path = Path(args.output)
    write_jsonl(results, output_path, metadata)
    print(f"\nResults written to {output_path}")

    return results


if __name__ == "__main__":
    main()
