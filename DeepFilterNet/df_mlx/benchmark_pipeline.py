#!/usr/bin/env python3
"""Benchmark MLX dynamic data pipeline throughput and latency tails.

This benchmark targets the data loading path used by df_mlx training:
- PrefetchDataLoader (thread pool prefetch)
- MLXDataStream (mlx-data prefetch) when available

It reports:
- Mean/p50/p95/p99 batch fetch latency
- Batches/s and samples/s throughput
- Total measured batches/samples

Example:
    python -m df_mlx.benchmark_pipeline \
        --cache-dir /path/to/audio_cache \
        --batch-size 8 \
        --batches 200 \
        --workers 1,2,4,8 \
        --backends prefetch,mlx_stream
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import mlx.core as mx

from df_mlx.benchmark_common import batch_size_from_batch as _batch_size_from_batch
from df_mlx.benchmark_common import build_dataset as _build_dataset
from df_mlx.benchmark_common import load_source_lists as _load_source_lists
from df_mlx.benchmark_common import (
    parse_backend_list,
    parse_bool_list,
    parse_float_list,
    parse_int_list,
    parse_split_list,
)
from df_mlx.benchmark_common import require_min as _require_min
from df_mlx.benchmark_common import require_positive_float as _require_positive_float
from df_mlx.benchmark_common import safe_percentile as _safe_percentile
from df_mlx.dynamic_dataset import (
    HAS_MLX_DATA,
    MLXDataStream,
    PrefetchDataLoader,
)


@dataclass
class BenchmarkResult:
    """Latency and throughput summary for one loader configuration."""

    backend: str
    split: str
    epoch: int
    workers: int
    prefetch: int
    batch_size: int
    batches_requested: int
    warmup_batches: int
    repeats: int
    sync_arrays: bool
    sample_rate: int
    segment_length: float
    fft_size: int
    hop_size: int
    nb_erb: int
    nb_df: int
    seed: int
    measured_batches: int
    measured_samples: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    total_seconds: float
    batches_per_sec: float
    samples_per_sec: float


@dataclass(frozen=True)
class BenchmarkCase:
    """A single benchmark matrix configuration."""

    backend: str
    split: str
    epoch: int
    workers: int
    batch_size: int
    batches: int
    warmup_batches: int
    repeats: int
    sync_arrays: bool
    sample_rate: int
    segment_length: float
    fft_size: int
    hop_size: int
    nb_erb: int
    nb_df: int
    seed: int
    prefetch_factor: int
    prefetch_size: int

    @property
    def prefetch(self) -> int:
        return self.prefetch_factor if self.backend == "prefetch" else self.prefetch_size


def _choices_for_backend(backends: Sequence[str]) -> Set[str]:
    return set(backends)


def _build_benchmark_cases(args: argparse.Namespace) -> List[BenchmarkCase]:
    _require_min("epoch", args.epoch, 0)
    _require_min("batch-size", args.batch_size, 1)
    _require_min("batches", args.batches, 1)
    _require_min("warmup-batches", args.warmup_batches, 0)
    _require_min("repeats", args.repeats, 1)
    _require_min("workers", args.workers, 1)
    _require_min("sample-rate", args.sample_rate, 1)
    _require_positive_float("segment-length", args.segment_length)
    _require_min("fft-size", args.fft_size, 1)
    _require_min("hop-size", args.hop_size, 1)
    _require_min("nb-erb", args.nb_erb, 1)
    _require_min("nb-df", args.nb_df, 1)
    _require_min("prefetch-factor", args.prefetch_factor, 1)
    _require_min("prefetch-size", args.prefetch_size, 1)

    if not args.backends:
        raise ValueError("At least one backend is required")
    if not args.split:
        raise ValueError("At least one split is required")

    backend_choices = _choices_for_backend(args.backends)
    prefetch_factors = args.prefetch_factor if "prefetch" in backend_choices else [2]
    prefetch_sizes = args.prefetch_size if "mlx_stream" in backend_choices else [8]

    cases: List[BenchmarkCase] = []
    base_product = itertools.product(
        args.backends,
        args.split,
        args.epoch,
        args.workers,
        args.batch_size,
        args.batches,
        args.warmup_batches,
        args.repeats,
        args.sync_arrays,
        args.sample_rate,
        args.segment_length,
        args.fft_size,
        args.hop_size,
        args.nb_erb,
        args.nb_df,
        args.seed,
    )

    for (
        backend,
        split,
        epoch,
        workers,
        batch_size,
        batches,
        warmup_batches,
        repeats,
        sync_arrays,
        sample_rate,
        segment_length,
        fft_size,
        hop_size,
        nb_erb,
        nb_df,
        seed,
    ) in base_product:
        if backend == "prefetch":
            for prefetch_factor in prefetch_factors:
                cases.append(
                    BenchmarkCase(
                        backend=backend,
                        split=split,
                        epoch=epoch,
                        workers=workers,
                        batch_size=batch_size,
                        batches=batches,
                        warmup_batches=warmup_batches,
                        repeats=repeats,
                        sync_arrays=sync_arrays,
                        sample_rate=sample_rate,
                        segment_length=segment_length,
                        fft_size=fft_size,
                        hop_size=hop_size,
                        nb_erb=nb_erb,
                        nb_df=nb_df,
                        seed=seed,
                        prefetch_factor=prefetch_factor,
                        prefetch_size=1,
                    )
                )
        else:
            for prefetch_size in prefetch_sizes:
                cases.append(
                    BenchmarkCase(
                        backend=backend,
                        split=split,
                        epoch=epoch,
                        workers=workers,
                        batch_size=batch_size,
                        batches=batches,
                        warmup_batches=warmup_batches,
                        repeats=repeats,
                        sync_arrays=sync_arrays,
                        sample_rate=sample_rate,
                        segment_length=segment_length,
                        fft_size=fft_size,
                        hop_size=hop_size,
                        nb_erb=nb_erb,
                        nb_df=nb_df,
                        seed=seed,
                        prefetch_factor=1,
                        prefetch_size=prefetch_size,
                    )
                )

    return cases


def _materialize_batch(batch: Dict[str, mx.array]) -> None:
    # Synchronize all arrays to include host->device staging costs in latency.
    mx.eval(*batch.values())


def _benchmark_loader_once(
    loader: Iterable[Dict[str, mx.array]],
    warmup_batches: int,
    measured_batches: int,
    sync_arrays: bool,
) -> Dict[str, Any]:
    iterator = iter(loader)

    for _ in range(warmup_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if sync_arrays:
            _materialize_batch(batch)

    latencies_ms: List[float] = []
    samples = 0
    batches = 0
    start = time.perf_counter()

    while batches < measured_batches:
        t0 = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if sync_arrays:
            _materialize_batch(batch)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        samples += _batch_size_from_batch(batch)
        batches += 1

    elapsed = time.perf_counter() - start
    return {
        "latencies_ms": latencies_ms,
        "samples": samples,
        "batches": batches,
        "elapsed_s": elapsed,
    }


def _aggregate_results(
    case: BenchmarkCase,
    run_stats: List[Dict[str, Any]],
) -> BenchmarkResult:
    all_latencies: List[float] = []
    measured_samples = 0
    measured_batches = 0
    total_seconds = 0.0

    for stats in run_stats:
        all_latencies.extend(stats["latencies_ms"])
        measured_samples += int(stats["samples"])
        measured_batches += int(stats["batches"])
        total_seconds += float(stats["elapsed_s"])

    mean_ms = statistics.mean(all_latencies) if all_latencies else math.nan
    std_ms = statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0.0
    p50_ms = _safe_percentile(all_latencies, 50)
    p95_ms = _safe_percentile(all_latencies, 95)
    p99_ms = _safe_percentile(all_latencies, 99)
    min_ms = min(all_latencies) if all_latencies else math.nan
    max_ms = max(all_latencies) if all_latencies else math.nan
    batches_per_sec = measured_batches / total_seconds if total_seconds > 0 else 0.0
    samples_per_sec = measured_samples / total_seconds if total_seconds > 0 else 0.0

    return BenchmarkResult(
        backend=case.backend,
        split=case.split,
        epoch=case.epoch,
        workers=case.workers,
        prefetch=case.prefetch,
        batch_size=case.batch_size,
        batches_requested=case.batches,
        warmup_batches=case.warmup_batches,
        repeats=case.repeats,
        sync_arrays=case.sync_arrays,
        sample_rate=case.sample_rate,
        segment_length=case.segment_length,
        fft_size=case.fft_size,
        hop_size=case.hop_size,
        nb_erb=case.nb_erb,
        nb_df=case.nb_df,
        seed=case.seed,
        measured_batches=measured_batches,
        measured_samples=measured_samples,
        mean_ms=mean_ms,
        std_ms=std_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        total_seconds=total_seconds,
        batches_per_sec=batches_per_sec,
        samples_per_sec=samples_per_sec,
    )


def print_summary(results: List[BenchmarkResult]) -> None:
    print("\n" + "=" * 200)
    print("MLX DATA PIPELINE BENCHMARK RESULTS")
    print("=" * 200)
    print(
        f"{'Backend':<11} {'Split':<6} {'Epoch':>5} {'Workers':>7} {'Prefetch':>8} "
        f"{'Batch':>6} {'ReqB':>5} {'Warm':>5} {'Rep':>4} {'Sync':>5} "
        f"{'SR':>7} {'Seg':>5} {'FFT':>5} {'Hop':>5} {'ERB':>4} {'DF':>4} {'Seed':>6} "
        f"{'Mean(ms)':>10} {'P95':>9} {'P99':>9} {'Batches/s':>11} {'Samples/s':>11}"
    )
    print("-" * 200)

    sorted_results = sorted(
        results,
        key=lambda r: (
            -r.samples_per_sec,
            r.backend,
            r.workers,
            r.prefetch,
            r.batch_size,
            r.sample_rate,
        ),
    )
    for r in sorted_results:
        print(
            f"{r.backend:<11} {r.split:<6} {r.epoch:>5d} {r.workers:>7d} {r.prefetch:>8d} "
            f"{r.batch_size:>6d} {r.batches_requested:>5d} {r.warmup_batches:>5d} {r.repeats:>4d} "
            f"{str(r.sync_arrays):>5} {r.sample_rate:>7d} {r.segment_length:>5.1f} "
            f"{r.fft_size:>5d} {r.hop_size:>5d} {r.nb_erb:>4d} {r.nb_df:>4d} {r.seed:>6d} "
            f"{r.mean_ms:>10.2f} {r.p95_ms:>9.2f} {r.p99_ms:>9.2f} {r.batches_per_sec:>11.2f} "
            f"{r.samples_per_sec:>11.2f}"
        )

    print("-" * 200)
    if sorted_results:
        best = sorted_results[0]
        print(
            "Best throughput: "
            f"{best.backend} split={best.split} epoch={best.epoch} workers={best.workers} "
            f"batch={best.batch_size} prefetch={best.prefetch} "
            f"({best.samples_per_sec:.2f} samples/s, p95={best.p95_ms:.2f} ms)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLX dynamic data pipeline")
    parser.add_argument("--speech-list", type=str, default=None, help="Speech file list (required without --cache-dir)")
    parser.add_argument("--noise-list", type=str, default=None, help="Noise file list (required without --cache-dir)")
    parser.add_argument("--rir-list", type=str, default=None, help="Optional RIR file list")
    parser.add_argument("--cache-dir", type=str, default=None, help="Path to cache dir from build_audio_cache.py")
    parser.add_argument(
        "--split",
        type=parse_split_list,
        default=parse_split_list("train"),
        help="Comma-separated split values: train,valid,test",
    )
    parser.add_argument("--epoch", type=parse_int_list, default=parse_int_list("0"), help="Dataset epoch seed(s)")
    parser.add_argument("--batch-size", type=parse_int_list, default=parse_int_list("8"))
    parser.add_argument(
        "--batches", type=parse_int_list, default=parse_int_list("200"), help="Batches measured per repeat"
    )
    parser.add_argument(
        "--warmup-batches", type=parse_int_list, default=parse_int_list("10"), help="Warmup batches per repeat"
    )
    parser.add_argument(
        "--repeats", type=parse_int_list, default=parse_int_list("3"), help="Number of repeats per configuration"
    )
    parser.add_argument(
        "--workers",
        type=parse_int_list,
        default=parse_int_list("1,2,4,8"),
        help="Comma-separated worker counts, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--backends",
        type=parse_backend_list,
        default=parse_backend_list("prefetch,mlx_stream"),
        help="Comma-separated backends: prefetch,mlx_stream",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=parse_int_list,
        default=parse_int_list("2"),
        help="Prefetch queue size(s) for PrefetchDataLoader",
    )
    parser.add_argument(
        "--prefetch-size",
        type=parse_int_list,
        default=parse_int_list("8"),
        help="Prefetch size(s) for MLXDataStream",
    )
    parser.add_argument("--sample-rate", type=parse_int_list, default=parse_int_list("48000"))
    parser.add_argument("--segment-length", type=parse_float_list, default=parse_float_list("5.0"))
    parser.add_argument("--fft-size", type=parse_int_list, default=parse_int_list("960"))
    parser.add_argument("--hop-size", type=parse_int_list, default=parse_int_list("480"))
    parser.add_argument("--nb-erb", type=parse_int_list, default=parse_int_list("32"))
    parser.add_argument("--nb-df", type=parse_int_list, default=parse_int_list("96"))
    parser.add_argument("--seed", type=parse_int_list, default=parse_int_list("42"))
    parser.add_argument(
        "--sync-arrays",
        type=parse_bool_list,
        default=parse_bool_list("true"),
        help="Comma-separated booleans; true calls mx.eval() per batch, false skips it",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Deprecated alias for --sync-arrays false",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.no_sync:
        args.sync_arrays = [False]

    cases = _build_benchmark_cases(args)
    if not cases:
        raise RuntimeError("No benchmark cases generated from argument matrix")
    source_lists = _load_source_lists(args)

    print(f"Benchmark matrix size: {len(cases)} configuration(s)")
    print(
        f"Backends={args.backends} splits={args.split} workers={args.workers} "
        f"batch_size={args.batch_size} prefetch_factor={args.prefetch_factor} prefetch_size={args.prefetch_size}"
    )
    if "mlx_stream" in args.backends and not HAS_MLX_DATA:
        print("Skipping mlx_stream backend: mlx-data is not installed.")
    results: List[BenchmarkResult] = []
    for idx, case in enumerate(cases, start=1):
        if case.backend == "mlx_stream" and not HAS_MLX_DATA:
            continue

        dataset = _build_dataset(args, case, source_lists)
        available_batches = len(dataset) // case.batch_size
        run_stats: List[Dict[str, Any]] = []
        print(
            f"\n[{idx}/{len(cases)}] backend={case.backend} split={case.split} epoch={case.epoch} "
            f"workers={case.workers} prefetch={case.prefetch} batch={case.batch_size} "
            f"batches={case.batches} warmup={case.warmup_batches} repeats={case.repeats} "
            f"sync={case.sync_arrays} sr={case.sample_rate} seg={case.segment_length} "
            f"fft={case.fft_size} hop={case.hop_size} erb={case.nb_erb} df={case.nb_df} seed={case.seed} "
            f"available_batches={available_batches}"
        )

        for run in range(case.repeats):
            dataset.set_split(case.split)
            dataset.set_epoch(case.epoch + run)

            if case.backend == "prefetch":
                loader: Iterable[Dict[str, mx.array]] = PrefetchDataLoader(
                    dataset=dataset,
                    batch_size=case.batch_size,
                    num_workers=case.workers,
                    prefetch_factor=case.prefetch_factor,
                    drop_last=True,
                )
            else:
                stream = MLXDataStream(
                    dataset=dataset,
                    batch_size=case.batch_size,
                    prefetch_size=case.prefetch_size,
                    num_workers=case.workers,
                    drop_last=True,
                )
                stream.set_split(case.split)
                stream.set_epoch(case.epoch + run)
                loader = stream

            stats = _benchmark_loader_once(
                loader=loader,
                warmup_batches=case.warmup_batches,
                measured_batches=case.batches,
                sync_arrays=case.sync_arrays,
            )
            run_stats.append(stats)
            print(
                f"  repeat {run + 1}/{case.repeats}: batches={stats['batches']} "
                f"samples={stats['samples']} elapsed={stats['elapsed_s']:.2f}s"
            )
            if stats["batches"] < case.batches:
                print(
                    "  warning: dataset exhausted before requested measured batches. "
                    "Increase dataset size or lower --batches."
                )

        results.append(_aggregate_results(case=case, run_stats=run_stats))

    if not results:
        raise RuntimeError("No benchmark results generated")

    print_summary(results)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nWrote JSON results to {output_path}")


if __name__ == "__main__":
    main()
