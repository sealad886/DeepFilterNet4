#!/usr/bin/env python3
"""Compare two benchmark result JSON files and produce a performance report.

Usage:
    python scripts/compare_benchmarks.py /tmp/bench_before.json /tmp/bench_after.json
"""

import argparse
import json
from typing import Any, Dict


def compare_timing(name: str, before: Dict[str, float], after: Dict[str, float], width: int = 45) -> str:
    """Compare timing results between before and after."""
    b_mean = before["mean_ms"]
    a_mean = after["mean_ms"]

    if b_mean > 0:
        speedup = b_mean / a_mean
        pct_change = ((b_mean - a_mean) / b_mean) * 100
    else:
        speedup = 1.0
        pct_change = 0.0

    direction = "faster" if pct_change > 0 else "slower"
    line = (
        f"  {name:<{width}}: {b_mean:8.2f}ms -> {a_mean:8.2f}ms"
        f"  ({speedup:.2f}x, {abs(pct_change):.1f}% {direction})"
    )

    # Memory comparison if available
    if "peak_memory_mb" in before and "peak_memory_mb" in after:
        b_mem = before["peak_memory_mb"]
        a_mem = after["peak_memory_mb"]
        if b_mem > 0:
            mem_pct = ((b_mem - a_mem) / b_mem) * 100
            mem_dir = "less" if mem_pct > 0 else "more"
            line += f"\n  {'  (memory)':<{width}}: {b_mem:.1f}MB -> {a_mem:.1f}MB ({abs(mem_pct):.1f}% {mem_dir})"

    return line


def print_section(title: str, before_section: Dict, after_section: Dict) -> Dict[str, Any]:
    """Compare and print a section of results."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    section_summary = {}

    # Find common keys
    common_keys = sorted(set(before_section.keys()) & set(after_section.keys()))
    before_only = sorted(set(before_section.keys()) - set(after_section.keys()))
    after_only = sorted(set(after_section.keys()) - set(before_section.keys()))

    for key in common_keys:
        b = before_section[key]
        a = after_section[key]
        if isinstance(b, dict) and "mean_ms" in b and isinstance(a, dict) and "mean_ms" in a:
            print(compare_timing(key, b, a))
            speedup = b["mean_ms"] / a["mean_ms"] if a["mean_ms"] > 0 else 1.0
            section_summary[key] = {
                "before_ms": b["mean_ms"],
                "after_ms": a["mean_ms"],
                "speedup": speedup,
                "pct_improvement": ((b["mean_ms"] - a["mean_ms"]) / b["mean_ms"]) * 100 if b["mean_ms"] > 0 else 0.0,
            }
            # Throughput comparison
            if "throughput_audio_sec_per_wall_sec" in b and "throughput_audio_sec_per_wall_sec" in a:
                b_tp = b["throughput_audio_sec_per_wall_sec"]
                a_tp = a["throughput_audio_sec_per_wall_sec"]
                tp_improvement = ((a_tp - b_tp) / b_tp) * 100 if b_tp > 0 else 0.0
                print(f"    -> Throughput: {b_tp:.1f} -> {a_tp:.1f} sec-audio/sec" f"  ({tp_improvement:+.1f}%)")
                section_summary[key]["throughput_before"] = b_tp
                section_summary[key]["throughput_after"] = a_tp

    if before_only:
        print(f"\n  [Before only]: {', '.join(before_only)}")
    if after_only:
        print(f"\n  [After only]: {', '.join(after_only)}")

    return section_summary


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("before", type=str, help="Path to before-optimization benchmark JSON")
    parser.add_argument("after", type=str, help="Path to after-optimization benchmark JSON")
    parser.add_argument("--output", type=str, help="Output comparison JSON file")
    args = parser.parse_args()

    with open(args.before) as f:
        before = json.load(f)
    with open(args.after) as f:
        after = json.load(f)

    print("=" * 80)
    print("  PERFORMANCE COMPARISON REPORT")
    print("=" * 80)

    # Environment info
    b_env = before.get("environment", {})
    a_env = after.get("environment", {})
    print(f"\n  Before: {b_env.get('git_branch', '?')}@{b_env.get('git_commit', '?')}")
    print(f"  After:  {a_env.get('git_branch', '?')}@{a_env.get('git_commit', '?')}")
    print(f"  MLX:    {b_env.get('mlx_version', '?')} / {a_env.get('mlx_version', '?')}")

    summary: Dict[str, Any] = {}

    # Compare each section
    sections = [
        ("DfNet4 Forward Pass", "dfnet4_forward"),
        ("DfNet4 Training Step", "dfnet4_training"),
        ("Mamba Selective Scan", "mamba_scan"),
        ("Post-Filter", "post_filter"),
        ("SqueezedAttention", "attention"),
    ]

    for title, key in sections:
        if key in before and key in after:
            summary[key] = print_section(title, before[key], after[key])

    # Memory comparison
    if "memory" in before and "memory" in after:
        print(f"\n{'=' * 80}")
        print("  Peak Memory Usage")
        print(f"{'=' * 80}")
        for k in sorted(set(before["memory"].keys()) & set(after["memory"].keys())):
            b = before["memory"][k]
            a = after["memory"][k]
            if isinstance(b, dict) and "peak_memory_mb" in b:
                b_peak = b["peak_memory_mb"]
                a_peak = a["peak_memory_mb"]
                pct = ((b_peak - a_peak) / b_peak * 100) if b_peak > 0 else 0
                direction = "less" if pct > 0 else "more"
                print(f"  {k:<45}: {b_peak:.1f}MB -> {a_peak:.1f}MB" f"  ({abs(pct):.1f}% {direction})")

    # Overall summary
    print(f"\n{'=' * 80}")
    print("  OVERALL SUMMARY")
    print(f"{'=' * 80}")

    all_speedups = []
    for section_data in summary.values():
        for bench_data in section_data.values():
            if "speedup" in bench_data:
                all_speedups.append(bench_data["speedup"])

    if all_speedups:
        import numpy as np

        print(f"  Benchmarks compared: {len(all_speedups)}")
        print(f"  Mean speedup:        {np.mean(all_speedups):.2f}x")
        print(f"  Median speedup:      {np.median(all_speedups):.2f}x")
        print(f"  Min speedup:         {np.min(all_speedups):.2f}x")
        print(f"  Max speedup:         {np.max(all_speedups):.2f}x")

        faster = sum(1 for s in all_speedups if s > 1.01)
        slower = sum(1 for s in all_speedups if s < 0.99)
        neutral = len(all_speedups) - faster - slower
        print(f"  Faster: {faster}  Neutral: {neutral}  Slower: {slower}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "before_env": b_env,
                    "after_env": a_env,
                    "sections": summary,
                    "overall_speedups": all_speedups,
                },
                f,
                indent=2,
            )
        print(f"\n  Comparison saved to {args.output}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
