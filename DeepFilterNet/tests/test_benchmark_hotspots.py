"""Tests for df_mlx.benchmark_hotspots harness."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from df_mlx.benchmark_hotspots import (
    HotspotCase,
    HotspotResult,
    build_default_matrix,
    check_hotspot_regression,
    main,
    run_case,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quick_case(op_name: str, batch_size: int = 1) -> HotspotCase:
    """Return a minimal-iteration case for fast testing."""
    if op_name == "mel_spec":
        return HotspotCase(op_name, batch_size, 512, 160, 16000, 1, 1)
    return HotspotCase(op_name, batch_size, 960, 480, 48000, 1, 1)


# ---------------------------------------------------------------------------
# Per-op smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", ["stft", "istft", "mel_spec", "dfop", "spectral_loss"])
def test_op_runs_without_error(op_name: str) -> None:
    """Each op benchmark runs with 1 warmup / 1 measured iter without crashing."""
    case = _quick_case(op_name)
    result = run_case(case)

    assert isinstance(result, HotspotResult)
    assert result.op_name == op_name
    assert result.batch_size == 1
    assert result.mean_ms > 0
    assert result.throughput_ops_per_sec > 0


# ---------------------------------------------------------------------------
# JSONL output validity
# ---------------------------------------------------------------------------


def test_jsonl_output_is_valid() -> None:
    """Verify the JSONL file written by write_jsonl is parseable."""
    results: List[HotspotResult] = [
        HotspotResult("stft", 1, 0.5, 0.1, 0.4, 0.5, 0.6, 2000.0),
        HotspotResult("istft", 1, 0.8, 0.2, 0.6, 0.8, 1.0, 1250.0),
    ]
    metadata = {"type": "metadata", "commit": "abc1234"}

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "bench.jsonl"
        write_jsonl(results, out_path, metadata)

        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == 3  # 1 metadata + 2 results

        meta_line = json.loads(lines[0])
        assert meta_line["type"] == "metadata"

        for line in lines[1:]:
            obj = json.loads(line)
            assert obj["type"] == "result"
            assert "op_name" in obj
            assert "mean_ms" in obj
            assert "throughput_ops_per_sec" in obj


# ---------------------------------------------------------------------------
# Regression threshold checking
# ---------------------------------------------------------------------------


def test_regression_pass() -> None:
    """A result within threshold should pass."""
    result = HotspotResult("stft", 1, 1.0, 0.05, 0.9, 1.0, 1.1, 1000.0)
    verdict = check_hotspot_regression(result, baseline_p50_ms=1.0, baseline_p95_ms=1.1)
    assert verdict["passed"] is True
    assert verdict["latency"]["ok"] is True
    assert verdict["variance"]["ok"] is True


def test_regression_fail_latency() -> None:
    """A result with P95 significantly above baseline should fail the latency gate."""
    result = HotspotResult("stft", 1, 2.0, 0.05, 1.8, 2.0, 2.5, 500.0)
    verdict = check_hotspot_regression(result, baseline_p50_ms=1.0, baseline_p95_ms=1.1)
    assert verdict["passed"] is False
    assert verdict["latency"]["ok"] is False


def test_regression_fail_variance() -> None:
    """A result with high CV should fail the variance gate."""
    result = HotspotResult("stft", 1, 1.0, 0.5, 0.9, 1.0, 1.1, 1000.0)
    verdict = check_hotspot_regression(result, baseline_p50_ms=1.0, baseline_p95_ms=1.1)
    assert verdict["passed"] is False
    assert verdict["variance"]["ok"] is False


# ---------------------------------------------------------------------------
# CLI / integration
# ---------------------------------------------------------------------------


def test_cli_entrypoint_produces_jsonl() -> None:
    """Verify the ``main()`` CLI entrypoint writes JSONL and returns results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_out.jsonl"
        results = main(
            [
                "--batch-sizes",
                "1",
                "--iters",
                "2",
                "--warmup",
                "1",
                "--ops",
                "stft",
                "--output",
                str(out_path),
            ]
        )

        assert len(results) == 1
        assert results[0].op_name == "stft"
        assert out_path.exists()

        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == 2  # metadata + 1 result


def test_default_matrix_structure() -> None:
    """build_default_matrix returns cases for all ops x batch_sizes."""
    cases = build_default_matrix([1, 4], bench_iters=5, warmup_iters=2)
    op_names = {c.op_name for c in cases}
    bs_set = {c.batch_size for c in cases}

    assert op_names == {"stft", "istft", "mel_spec", "dfop", "spectral_loss"}
    assert bs_set == {1, 4}
    assert all(c.bench_iters == 5 for c in cases)
    assert all(c.warmup_iters == 2 for c in cases)
