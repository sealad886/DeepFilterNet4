#!/usr/bin/env python3
"""Focused benchmark harness for tuning ``fused_complex_mag``.

Benchmarks:
1. Raw forward dispatch across threadgroup sizes.
2. End-to-end ``spectral_loss`` candidate comparison: native vs fused.

This is intended to tune the threadgroup size used by the Metal kernel before
it is wired into the training loss path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import mlx.core as mx
import numpy as np

from df_mlx.metal_kernels import (
    _EPS_F,
    _dispatch_complex_mag_forward,
    _make_params,
    _ref_complex_mag,
)
from df_mlx.train import spectral_loss as train_spectral_loss


@dataclass(frozen=True)
class BenchCase:
    name: str
    shape: tuple[int, int, int]


@dataclass
class BenchResult:
    case: str
    variant: str
    mean_ms: float
    p50_ms: float
    p95_ms: float


def _safe_percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def _benchmark(fn: Callable[[], None], warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    latencies: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(latencies), _safe_percentile(latencies, 50), _safe_percentile(latencies, 95)


def _parse_cases(values: Iterable[str]) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for item in values:
        shape = tuple(int(part) for part in item.split("x"))
        if len(shape) != 3:
            raise ValueError(f"Expected BxTxF shape, got: {item!r}")
        cases.append(BenchCase(name=item, shape=shape))
    return cases


def _spectral_loss_native(
    pred_real: mx.array,
    pred_imag: mx.array,
    target_real: mx.array,
    target_imag: mx.array,
    alpha: float = 0.5,
    eps: float = _EPS_F,
) -> mx.array:
    pred_mag = _ref_complex_mag(pred_real, pred_imag, eps=eps)
    target_mag = _ref_complex_mag(target_real, target_imag, eps=eps)
    mag_loss = mx.mean(mx.abs(pred_mag - target_mag))
    complex_loss = mx.mean(mx.abs(pred_real - target_real) + mx.abs(pred_imag - target_imag))
    return (1 - alpha) * mag_loss + alpha * complex_loss


def run(case: BenchCase, *, warmup: int, iters: int, threadgroups: list[int]) -> list[BenchResult]:
    B, T, F = case.shape
    rng = np.random.default_rng(42)
    pred_real = mx.array(rng.standard_normal((B, T, F)).astype(np.float32))
    pred_imag = mx.array(rng.standard_normal((B, T, F)).astype(np.float32))
    target_real = mx.array(rng.standard_normal((B, T, F)).astype(np.float32))
    target_imag = mx.array(rng.standard_normal((B, T, F)).astype(np.float32))
    mx.eval(pred_real, pred_imag, target_real, target_imag)

    results: list[BenchResult] = []
    params = _make_params(pred_real, _EPS_F)

    def baseline_forward() -> None:
        out = _ref_complex_mag(pred_real, pred_imag)
        mx.eval(out)

    mean_ms, p50_ms, p95_ms = _benchmark(baseline_forward, warmup, iters)
    results.append(
        BenchResult(case=case.name, variant="complex_mag_native", mean_ms=mean_ms, p50_ms=p50_ms, p95_ms=p95_ms)
    )

    for tg in threadgroups:

        def fused_forward() -> None:
            out = _dispatch_complex_mag_forward(pred_real, pred_imag, params, threadgroup_size=tg)
            mx.eval(out)

        mean_ms, p50_ms, p95_ms = _benchmark(fused_forward, warmup, iters)
        results.append(
            BenchResult(
                case=case.name, variant=f"complex_mag_fused_tg{tg}", mean_ms=mean_ms, p50_ms=p50_ms, p95_ms=p95_ms
            )
        )

    def native_spectral() -> None:
        loss = _spectral_loss_native(pred_real, pred_imag, target_real, target_imag)
        mx.eval(loss)

    mean_ms, p50_ms, p95_ms = _benchmark(native_spectral, warmup, iters)
    results.append(
        BenchResult(case=case.name, variant="spectral_loss_native", mean_ms=mean_ms, p50_ms=p50_ms, p95_ms=p95_ms)
    )

    def wired_spectral() -> None:
        loss = train_spectral_loss((pred_real, pred_imag), (target_real, target_imag))
        mx.eval(loss)

    mean_ms, p50_ms, p95_ms = _benchmark(wired_spectral, warmup, iters)
    results.append(
        BenchResult(case=case.name, variant="spectral_loss_train_wired", mean_ms=mean_ms, p50_ms=p50_ms, p95_ms=p95_ms)
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fused_complex_mag threadgroup sizes and spectral_loss impact."
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="1x150x481,4x150x481,8x150x481,8x500x481",
        help="Comma-separated BxTxF shapes representative of training spectra.",
    )
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--threadgroups", type=str, default="64,128,256,512")
    args = parser.parse_args()

    cases = _parse_cases(args.cases.split(","))
    threadgroups = [int(v) for v in args.threadgroups.split(",") if v.strip()]

    all_results: list[BenchResult] = []
    for case in cases:
        all_results.extend(run(case, warmup=args.warmup, iters=args.iters, threadgroups=threadgroups))

    print(json.dumps([asdict(r) for r in all_results], indent=2))


if __name__ == "__main__":
    main()
