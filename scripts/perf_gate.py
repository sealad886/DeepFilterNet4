#!/usr/bin/env python3
"""Performance regression gate for DeepFilterNet MLX training.

Compares two JSONL benchmark result files (baseline vs candidate) and applies
the pass/fail thresholds from ``check_regression()`` in
``df_mlx.benchmark_train_step``.

Usage:
    python scripts/perf_gate.py --baseline baseline.jsonl --candidate candidate.jsonl
    python scripts/perf_gate.py --baseline baseline.jsonl --candidate candidate.jsonl \\
        --report gate_report.md --strict

Exit codes:
    0 — all configs pass
    1 — at least one config fails a regression gate
    2 — error (missing data, incompatible inputs, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _config_key(cfg: Dict[str, Any]) -> str:
    """Produce a canonical string key for matching baseline/candidate configs."""
    backbone = cfg.get("backbone", cfg.get("model_variant", "unknown"))
    bs = cfg.get("batch_size", "?")
    compiled = cfg.get("compiled", False)
    grad_accum = cfg.get("grad_accumulation", cfg.get("grad_accum", 1))
    fp16 = cfg.get("fp16", False)
    mode = "compiled" if compiled else "eager"
    fp_tag = "fp16" if fp16 else "fp32"
    return f"{backbone}/bs{bs}/{mode}/ga{grad_accum}/{fp_tag}"


def load_results(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load benchmark results from a JSONL or JSON file.

    Supports two formats:
    - JSONL: one JSON object per line (each is a result record)
    - JSON: ``{"metadata": {...}, "results": [...]}``

    Returns ``(metadata, results)`` where *results* is a list of dicts each
    containing at least ``config`` and ``metrics`` (or flat metric keys).
    """
    text = path.read_text()

    # Try JSON first (single object with metadata + results)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "results" in data:
            return data.get("metadata", {}), data["results"]
    except json.JSONDecodeError:
        pass

    # JSONL: one object per line
    records: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"Warning: skipping malformed line {line_no} in {path}", file=sys.stderr)
            continue
        if "config" not in obj and "backbone" not in obj and "model_variant" not in obj:
            if not metadata:
                metadata = obj
            continue
        records.append(obj)

    return metadata, records


def _extract_config(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the config sub-dict from a result record."""
    if "config" in record:
        return record["config"]
    # Flat record: synthesise config from known keys
    return {
        "backbone": record.get("backbone", record.get("model_variant", "unknown")),
        "batch_size": record.get("batch_size"),
        "compiled": record.get("compiled", False),
        "grad_accumulation": record.get("grad_accumulation", record.get("grad_accum", 1)),
        "fp16": record.get("fp16", False),
    }


def _extract_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the metrics sub-dict from a result record."""
    if "metrics" in record:
        return record["metrics"]
    return record


def _safe_get(metrics: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in metrics:
            val = metrics[k]
            if val is not None:
                return float(val)
    return default


def compare(
    baseline_records: List[Dict[str, Any]],
    candidate_records: List[Dict[str, Any]],
    *,
    threshold_throughput: float = 0.90,
    threshold_latency: float = 1.15,
    threshold_cv: float = 0.20,
    strict: bool = False,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Compare baseline and candidate results.

    Returns ``(passed, rows)`` where *rows* is a list of per-config comparison
    dicts suitable for report generation.
    """
    # Index baseline by config key
    baseline_by_key: Dict[str, Dict[str, Any]] = {}
    for rec in baseline_records:
        cfg = _extract_config(rec)
        key = _config_key(cfg)
        baseline_by_key[key] = rec

    rows: List[Dict[str, Any]] = []
    all_passed = True

    for rec in candidate_records:
        cfg = _extract_config(rec)
        key = _config_key(cfg)
        bl = baseline_by_key.get(key)
        if bl is None:
            rows.append(
                {
                    "config_key": key,
                    "status": "SKIP",
                    "reason": "no baseline",
                    "details": {},
                }
            )
            continue

        bl_m = _extract_metrics(bl)
        cd_m = _extract_metrics(rec)

        bl_p5 = _safe_get(bl_m, "samples_per_sec_p5", "samples_per_sec")
        cd_p5 = _safe_get(cd_m, "samples_per_sec_p5", "samples_per_sec")
        bl_p95 = _safe_get(bl_m, "step_p95_ms")
        cd_p95 = _safe_get(cd_m, "step_p95_ms")
        cd_mean = _safe_get(cd_m, "samples_per_sec_mean", "samples_per_sec")
        cd_std = _safe_get(cd_m, "samples_per_sec_std", default=0.0)

        throughput_ok = cd_p5 >= bl_p5 * threshold_throughput if bl_p5 > 0 else True
        latency_ok = cd_p95 <= bl_p95 * threshold_latency if bl_p95 > 0 else True
        cv = (cd_std / cd_mean) if cd_mean > 0 else float("inf")
        variance_ok = cv <= threshold_cv

        passed = throughput_ok and latency_ok and variance_ok
        if strict and not passed:
            all_passed = False
        elif not passed:
            all_passed = False

        throughput_delta = ((cd_p5 - bl_p5) / bl_p5 * 100) if bl_p5 > 0 else 0.0
        latency_delta = ((cd_p95 - bl_p95) / bl_p95 * 100) if bl_p95 > 0 else 0.0

        rows.append(
            {
                "config_key": key,
                "status": "PASS" if passed else "FAIL",
                "throughput": {
                    "baseline": bl_p5,
                    "candidate": cd_p5,
                    "delta_pct": throughput_delta,
                    "ok": throughput_ok,
                },
                "latency": {
                    "baseline": bl_p95,
                    "candidate": cd_p95,
                    "delta_pct": latency_delta,
                    "ok": latency_ok,
                },
                "variance": {"cv": cv, "ok": variance_ok},
                "details": {"passed": passed},
            }
        )

    return all_passed, rows


def generate_report(
    rows: List[Dict[str, Any]],
    passed: bool,
    baseline_meta: Dict[str, Any],
    candidate_meta: Dict[str, Any],
) -> str:
    """Generate a human-readable markdown report."""
    lines: List[str] = []
    lines.append("=== Performance Regression Gate ===")

    bl_commit = baseline_meta.get("commit", "unknown")
    cd_commit = candidate_meta.get("commit", "unknown")
    bl_ts = baseline_meta.get("timestamp", "")
    cd_ts = candidate_meta.get("timestamp", "")

    hw = baseline_meta.get("hardware", {})
    chip = hw.get("chip", "unknown")
    cores = hw.get("gpu_cores", "?")
    mem = hw.get("memory_gb", "?")
    hw_str = f"{chip} ({cores} cores, {mem}GB)"

    lines.append(f"Baseline: commit {bl_commit} ({bl_ts[:10] if bl_ts else 'N/A'})")
    lines.append(f"Candidate: commit {cd_commit} ({cd_ts[:10] if cd_ts else 'N/A'})")
    lines.append(f"Hardware: {hw_str}")
    lines.append("")
    lines.append("| Config | Metric | Baseline | Candidate | Delta | Status |")
    lines.append("|--------|--------|----------|-----------|-------|--------|")

    fail_count = 0
    for row in rows:
        key = row["config_key"]
        status = row["status"]
        if status == "SKIP":
            lines.append(f"| {key} | — | — | — | — | SKIP |")
            continue

        tp = row["throughput"]
        lt = row["latency"]

        tp_status = "PASS" if tp["ok"] else "FAIL"
        lt_status = "PASS" if lt["ok"] else "FAIL"
        var_status = "PASS" if row["variance"]["ok"] else "FAIL"

        if not tp["ok"] or not lt["ok"] or not row["variance"]["ok"]:
            fail_count += 1

        lines.append(
            f"| {key} | samples/s | {tp['baseline']:.1f} | {tp['candidate']:.1f} "
            f"| {tp['delta_pct']:+.1f}% | {tp_status} |"
        )
        lines.append(
            f"| {key} | step_p95_ms | {lt['baseline']:.1f} | {lt['candidate']:.1f} "
            f"| {lt['delta_pct']:+.1f}% | {lt_status} |"
        )
        cv_pct = row["variance"]["cv"] * 100
        lines.append(f"| {key} | CV | — | — | {cv_pct:.1f}% | {var_status} |")

    lines.append("")
    if passed:
        lines.append("Result: PASS")
    else:
        lines.append(f"Result: FAIL ({fail_count} regression(s) detected)")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Performance regression gate for DeepFilterNet MLX training.")
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline JSONL/JSON benchmark results.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to candidate JSONL/JSON benchmark results.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output path for the markdown report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any regression (no tolerance).",
    )
    parser.add_argument(
        "--threshold-throughput",
        type=float,
        default=0.90,
        help="Throughput pass factor (default: 0.90 = 10%% tolerance).",
    )
    parser.add_argument(
        "--threshold-latency",
        type=float,
        default=1.15,
        help="Latency pass factor (default: 1.15 = 15%% tolerance).",
    )
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        print(f"Error: baseline file not found: {args.baseline}", file=sys.stderr)
        return 2
    if not args.candidate.exists():
        print(f"Error: candidate file not found: {args.candidate}", file=sys.stderr)
        return 2

    try:
        bl_meta, bl_results = load_results(args.baseline)
        cd_meta, cd_results = load_results(args.candidate)
    except Exception as e:
        print(f"Error loading results: {e}", file=sys.stderr)
        return 2

    if not bl_results:
        print("Error: no baseline results found.", file=sys.stderr)
        return 2
    if not cd_results:
        print("Error: no candidate results found.", file=sys.stderr)
        return 2

    passed, rows = compare(
        bl_results,
        cd_results,
        threshold_throughput=args.threshold_throughput,
        threshold_latency=args.threshold_latency,
        strict=args.strict,
    )

    report = generate_report(rows, passed, bl_meta, cd_meta)
    print(report)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n")
        print(f"\nReport written to {args.report}", file=sys.stderr)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
