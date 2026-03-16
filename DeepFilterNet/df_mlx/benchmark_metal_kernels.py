#!/usr/bin/env python3
"""Reusable benchmark and epsilon-analysis harness for df_mlx Metal kernels.

This entrypoint benchmarks the native and fused paths for the custom Metal
kernels and can also run an adversarial epsilon-sensitivity analysis for the
stabilized magnitude math used by the kernels.

Examples:
    python -m df_mlx.benchmark_metal_kernels --mode bench
    python -m df_mlx.benchmark_metal_kernels --mode eps --eps-values 1e-10,1e-8,1e-6
    python -m df_mlx.benchmark_metal_kernels --mode all --iters 35 --warmup 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from df_mlx.benchmark_common import parse_float_list, parse_int_list, safe_percentile
from df_mlx.benchmark_train_step import collect_reproducibility_metadata
from df_mlx.metal_kernels import (
    _BAND_ENERGY_EPS_F,
    _EPS_F,
    _dispatch_complex_mag_forward,
    _dispatch_log1p_mag_forward,
    _make_params,
    _ref_band_energy,
    _ref_complex_mag,
    _ref_log1p_mag,
    fused_band_energy,
    fused_complex_mag,
    fused_log1p_mag,
)

DEFAULT_CASES = "small=2x150x481,mid=12x150x481,trainish=36x150x481,long=8x500x481"
DEFAULT_THREADGROUPS = "64,128,256,512"
DEFAULT_EPS_VALUES = "1e-10,1e-9,1e-8,1e-7,1e-6"
DEFAULT_SCALES = "1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3"


@dataclass(frozen=True)
class KernelCase:
    label: str
    shape: tuple[int, int, int]


@dataclass
class MetricSummary:
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def _parse_cases(value: str) -> list[KernelCase]:
    cases: list[KernelCase] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            label, dims = token.split("=", 1)
        else:
            label, dims = token, token
        shape = tuple(int(part) for part in dims.split("x"))
        if len(shape) != 3:
            raise ValueError(f"Expected BxTxF case, got {token!r}")
        cases.append(KernelCase(label=label.strip(), shape=shape))
    if not cases:
        raise ValueError("At least one case is required")
    return cases


def _benchmark(fn: Callable[[], None], warmup: int, iters: int) -> MetricSummary:
    for _ in range(warmup):
        fn()
    latencies: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return MetricSummary(
        mean_ms=statistics.mean(latencies),
        std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        p50_ms=safe_percentile(latencies, 50),
        p95_ms=safe_percentile(latencies, 95),
        min_ms=min(latencies),
        max_ms=max(latencies),
    )


def _band_mask(n_freqs: int) -> tuple[mx.array, float]:
    mask = np.zeros(n_freqs, dtype=np.float32)
    mask[: max(1, n_freqs // 6)] = 1.0
    return mx.array(mask), float(mask.sum())


def _speedup(native: MetricSummary, candidate: MetricSummary) -> float:
    if candidate.mean_ms <= 0.0:
        return float("inf")
    return native.mean_ms / candidate.mean_ms


def run_path_benchmarks(
    cases: list[KernelCase],
    *,
    warmup: int,
    iters: int,
    threadgroups: list[int],
    mag_eps: float,
    band_eps: float,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(42)
    results: list[dict[str, Any]] = []
    for case in cases:
        bsz, frames, n_freqs = case.shape
        real = mx.array(rng.standard_normal((bsz, frames, n_freqs)).astype(np.float32))
        imag = mx.array(rng.standard_normal((bsz, frames, n_freqs)).astype(np.float32))
        mask, bins = _band_mask(n_freqs)
        params = _make_params(real, mag_eps)
        mx.eval(real, imag, mask, params)

        log1p_native = _benchmark(lambda: mx.eval(_ref_log1p_mag(real, imag, eps=mag_eps)), warmup, iters)
        log1p_fused = _benchmark(lambda: mx.eval(fused_log1p_mag(real, imag, eps=mag_eps)), warmup, iters)
        complex_native = _benchmark(lambda: mx.eval(_ref_complex_mag(real, imag, eps=mag_eps)), warmup, iters)
        complex_fused = _benchmark(lambda: mx.eval(fused_complex_mag(real, imag, eps=mag_eps)), warmup, iters)
        band_native = _benchmark(
            lambda: mx.eval(*_ref_band_energy(real, imag, mask, bins, eps=band_eps)), warmup, iters
        )
        band_fused = _benchmark(
            lambda: mx.eval(*fused_band_energy(real, imag, mask, bins, eps=band_eps)), warmup, iters
        )

        log1p_tgs = {
            str(tg): asdict(
                _benchmark(
                    lambda tg=tg: mx.eval(_dispatch_log1p_mag_forward(real, imag, params, threadgroup_size=tg)),
                    warmup,
                    iters,
                )
            )
            for tg in threadgroups
        }
        complex_tgs = {
            str(tg): asdict(
                _benchmark(
                    lambda tg=tg: mx.eval(_dispatch_complex_mag_forward(real, imag, params, threadgroup_size=tg)),
                    warmup,
                    iters,
                )
            )
            for tg in threadgroups
        }

        results.append(
            {
                "case": case.label,
                "shape": list(case.shape),
                "elements": bsz * frames * n_freqs,
                "log1p_mag": {
                    "native": asdict(log1p_native),
                    "fused": asdict(log1p_fused),
                    "speedup": _speedup(log1p_native, log1p_fused),
                    "threadgroups": log1p_tgs,
                },
                "complex_mag": {
                    "native": asdict(complex_native),
                    "fused": asdict(complex_fused),
                    "speedup": _speedup(complex_native, complex_fused),
                    "threadgroups": complex_tgs,
                },
                "band_energy": {
                    "native": asdict(band_native),
                    "fused": asdict(band_fused),
                    "speedup": _speedup(band_native, band_fused),
                },
            }
        )
    return results


def run_eps_analysis(eps_values: list[float], scales: list[float]) -> list[dict[str, Any]]:
    """Measure how candidate eps values bias quiet-bin magnitudes and logs."""
    results: list[dict[str, Any]] = []
    for eps in eps_values:
        rows: list[dict[str, float]] = []
        max_rel_mag_bias = 0.0
        max_rel_log_bias = 0.0
        first_below_1pct_mag = None
        first_below_1pct_log = None

        for scale in scales:
            true_mag = float(np.sqrt(2.0) * scale)
            stabilized_mag = float(np.sqrt(2.0 * scale * scale + eps))
            true_log = float(np.log1p(true_mag))
            stabilized_log = float(np.log1p(stabilized_mag))
            mag_abs_bias = stabilized_mag - true_mag
            log_abs_bias = stabilized_log - true_log
            mag_rel_bias = mag_abs_bias / max(true_mag, 1e-30)
            log_rel_bias = log_abs_bias / max(abs(true_log), 1e-30)
            max_rel_mag_bias = max(max_rel_mag_bias, mag_rel_bias)
            max_rel_log_bias = max(max_rel_log_bias, log_rel_bias)
            if first_below_1pct_mag is None and mag_rel_bias < 0.01:
                first_below_1pct_mag = scale
            if first_below_1pct_log is None and log_rel_bias < 0.01:
                first_below_1pct_log = scale
            rows.append(
                {
                    "scale": scale,
                    "true_mag": true_mag,
                    "stabilized_mag": stabilized_mag,
                    "mag_abs_bias": mag_abs_bias,
                    "mag_rel_bias": mag_rel_bias,
                    "true_log1p": true_log,
                    "stabilized_log1p": stabilized_log,
                    "log_abs_bias": log_abs_bias,
                    "log_rel_bias": log_rel_bias,
                }
            )

        results.append(
            {
                "eps": eps,
                "magnitude_floor": float(np.sqrt(eps)),
                "zero_log1p_floor": float(np.log1p(np.sqrt(eps))),
                "first_scale_below_1pct_mag_bias": first_below_1pct_mag,
                "first_scale_below_1pct_log_bias": first_below_1pct_log,
                "max_rel_mag_bias": max_rel_mag_bias,
                "max_rel_log_bias": max_rel_log_bias,
                "rows": rows,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark df_mlx custom Metal kernel paths and epsilon sensitivity.")
    parser.add_argument("--mode", choices=("bench", "eps", "all"), default="all")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=35)
    parser.add_argument("--threadgroups", type=parse_int_list, default=parse_int_list(DEFAULT_THREADGROUPS))
    parser.add_argument("--mag-eps", type=float, default=_EPS_F)
    parser.add_argument("--band-eps", type=float, default=_BAND_ENERGY_EPS_F)
    parser.add_argument("--eps-values", type=parse_float_list, default=parse_float_list(DEFAULT_EPS_VALUES))
    parser.add_argument("--scales", type=parse_float_list, default=parse_float_list(DEFAULT_SCALES))
    parser.add_argument("--output", type=Path, default=Path("logs") / "metal_kernel_benchmark_latest.json")
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "metadata": collect_reproducibility_metadata(
            {
                "mode": args.mode,
                "cases": args.cases,
                "warmup": args.warmup,
                "iters": args.iters,
                "threadgroups": args.threadgroups,
                "mag_eps": args.mag_eps,
                "band_eps": args.band_eps,
                "eps_values": args.eps_values,
                "scales": args.scales,
            }
        )
    }

    if args.mode in {"bench", "all"}:
        payload["benchmarks"] = run_path_benchmarks(
            _parse_cases(args.cases),
            warmup=args.warmup,
            iters=args.iters,
            threadgroups=args.threadgroups,
            mag_eps=args.mag_eps,
            band_eps=args.band_eps,
        )

    if args.mode in {"eps", "all"}:
        payload["eps_analysis"] = run_eps_analysis(args.eps_values, args.scales)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
