"""Tests for scripts/perf_gate.py — performance regression gate."""

from __future__ import annotations

import json

# Import the gate module directly by manipulating the path so we can test
# without installing it as a package.
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from perf_gate import _config_key, compare, generate_report, load_results, main  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_record(
    backbone: str = "dfnet4",
    batch_size: int = 4,
    compiled: bool = True,
    grad_accumulation: int = 1,
    fp16: bool = False,
    samples_per_sec_p5: float = 120.0,
    samples_per_sec_mean: float = 125.0,
    samples_per_sec_std: float = 2.0,
    step_p95_ms: float = 33.0,
) -> Dict[str, Any]:
    return {
        "config": {
            "backbone": backbone,
            "batch_size": batch_size,
            "compiled": compiled,
            "grad_accumulation": grad_accumulation,
            "fp16": fp16,
        },
        "metrics": {
            "samples_per_sec_p5": samples_per_sec_p5,
            "samples_per_sec_mean": samples_per_sec_mean,
            "samples_per_sec_std": samples_per_sec_std,
            "step_p95_ms": step_p95_ms,
        },
    }


def _make_metadata(commit: str = "abc1234") -> Dict[str, Any]:
    return {
        "commit": commit,
        "timestamp": "2026-02-13T12:00:00+00:00",
        "hardware": {"chip": "Apple M3 Max", "gpu_cores": 40, "memory_gb": 48},
    }


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _write_json(path: Path, records: List[Dict[str, Any]], metadata: Dict[str, Any] | None = None) -> None:
    data = {"metadata": metadata or {}, "results": records}
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# config key tests
# ---------------------------------------------------------------------------


class TestConfigKey:
    def test_basic_key(self) -> None:
        key = _config_key({"backbone": "dfnet4", "batch_size": 4, "compiled": True, "fp16": False})
        assert key == "dfnet4/bs4/compiled/ga1/fp32"

    def test_eager_fp16(self) -> None:
        key = _config_key(
            {
                "backbone": "mamba",
                "batch_size": 8,
                "compiled": False,
                "grad_accumulation": 2,
                "fp16": True,
            }
        )
        assert key == "mamba/bs8/eager/ga2/fp16"

    def test_model_variant_fallback(self) -> None:
        key = _config_key({"model_variant": "dfnet4", "batch_size": 1, "compiled": True})
        assert key.startswith("dfnet4/")


# ---------------------------------------------------------------------------
# load_results tests
# ---------------------------------------------------------------------------


class TestLoadResults:
    def test_load_jsonl(self, tmp_path: Path) -> None:
        rec = _make_record()
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [rec])
        meta, results = load_results(p)
        assert len(results) == 1
        assert results[0]["config"]["backbone"] == "dfnet4"

    def test_load_json(self, tmp_path: Path) -> None:
        rec = _make_record()
        md = _make_metadata()
        p = tmp_path / "data.json"
        _write_json(p, [rec], md)
        meta, results = load_results(p)
        assert meta["commit"] == "abc1234"
        assert len(results) == 1

    def test_load_jsonl_with_metadata_line(self, tmp_path: Path) -> None:
        md = _make_metadata()
        rec = _make_record()
        p = tmp_path / "data.jsonl"
        p.write_text(json.dumps(md) + "\n" + json.dumps(rec) + "\n")
        meta, results = load_results(p)
        assert meta.get("commit") == "abc1234"
        assert len(results) == 1


# ---------------------------------------------------------------------------
# compare tests
# ---------------------------------------------------------------------------


class TestCompare:
    def test_all_pass(self) -> None:
        bl = [_make_record(samples_per_sec_p5=120.0, step_p95_ms=33.0)]
        cd = [_make_record(samples_per_sec_p5=115.0, step_p95_ms=35.0)]
        passed, rows = compare(bl, cd)
        assert passed is True
        assert rows[0]["status"] == "PASS"

    def test_throughput_fail(self) -> None:
        bl = [_make_record(samples_per_sec_p5=120.0)]
        cd = [_make_record(samples_per_sec_p5=100.0)]  # -16.7%
        passed, rows = compare(bl, cd)
        assert passed is False
        assert rows[0]["throughput"]["ok"] is False

    def test_latency_fail(self) -> None:
        bl = [_make_record(step_p95_ms=33.0)]
        cd = [_make_record(step_p95_ms=40.0)]  # +21.2%
        passed, rows = compare(bl, cd)
        assert passed is False
        assert rows[0]["latency"]["ok"] is False

    def test_variance_fail(self) -> None:
        bl = [_make_record()]
        cd = [_make_record(samples_per_sec_mean=100.0, samples_per_sec_std=25.0)]  # CV=0.25
        passed, rows = compare(bl, cd)
        assert passed is False
        assert rows[0]["variance"]["ok"] is False

    def test_missing_baseline_skipped(self) -> None:
        bl = [_make_record(backbone="dfnet4")]
        cd = [_make_record(backbone="mamba")]
        passed, rows = compare(bl, cd)
        assert passed is True  # no failures, just skip
        assert rows[0]["status"] == "SKIP"

    def test_threshold_override_throughput(self) -> None:
        bl = [_make_record(samples_per_sec_p5=120.0)]
        cd = [_make_record(samples_per_sec_p5=112.0)]  # -6.7%, within 10%
        passed, rows = compare(bl, cd, threshold_throughput=0.95)
        assert passed is False  # stricter threshold: need 95%
        assert rows[0]["throughput"]["ok"] is False

    def test_threshold_override_latency(self) -> None:
        bl = [_make_record(step_p95_ms=33.0)]
        cd = [_make_record(step_p95_ms=36.0)]  # +9.1%, within 15%
        passed, rows = compare(bl, cd, threshold_latency=1.05)
        assert passed is False  # stricter threshold: 5%
        assert rows[0]["latency"]["ok"] is False

    def test_multiple_configs_mixed(self) -> None:
        bl = [
            _make_record(backbone="dfnet4", batch_size=4, samples_per_sec_p5=120.0),
            _make_record(backbone="dfnet4", batch_size=8, samples_per_sec_p5=230.0),
        ]
        cd = [
            _make_record(backbone="dfnet4", batch_size=4, samples_per_sec_p5=118.0),
            _make_record(backbone="dfnet4", batch_size=8, samples_per_sec_p5=195.0),  # -15.2%
        ]
        passed, rows = compare(bl, cd)
        assert passed is False
        statuses = [r["status"] for r in rows]
        assert "PASS" in statuses
        assert "FAIL" in statuses


# ---------------------------------------------------------------------------
# report generation tests
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_pass(self) -> None:
        bl = [_make_record()]
        cd = [_make_record()]
        passed, rows = compare(bl, cd)
        report = generate_report(rows, passed, _make_metadata("aaa"), _make_metadata("bbb"))
        assert "Result: PASS" in report
        assert "FAIL" not in report.split("Result:")[1]

    def test_report_fail(self) -> None:
        bl = [_make_record(samples_per_sec_p5=120.0)]
        cd = [_make_record(samples_per_sec_p5=90.0)]
        passed, rows = compare(bl, cd)
        report = generate_report(rows, passed, _make_metadata(), _make_metadata())
        assert "Result: FAIL" in report
        assert "regression" in report.lower()

    def test_report_contains_hardware(self) -> None:
        bl = [_make_record()]
        cd = [_make_record()]
        passed, rows = compare(bl, cd)
        meta = _make_metadata()
        report = generate_report(rows, passed, meta, meta)
        assert "Apple M3 Max" in report
        assert "40 cores" in report

    def test_report_skip_row(self) -> None:
        bl = [_make_record(backbone="dfnet4")]
        cd = [_make_record(backbone="mamba")]
        passed, rows = compare(bl, cd)
        report = generate_report(rows, passed, _make_metadata(), _make_metadata())
        assert "SKIP" in report


# ---------------------------------------------------------------------------
# CLI / exit code tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_exit_0_on_pass(self, tmp_path: Path) -> None:
        bl = [_make_record()]
        cd = [_make_record()]
        bl_path = tmp_path / "bl.jsonl"
        cd_path = tmp_path / "cd.jsonl"
        _write_jsonl(bl_path, bl)
        _write_jsonl(cd_path, cd)
        rc = main(["--baseline", str(bl_path), "--candidate", str(cd_path)])
        assert rc == 0

    def test_exit_1_on_fail(self, tmp_path: Path) -> None:
        bl = [_make_record(samples_per_sec_p5=120.0)]
        cd = [_make_record(samples_per_sec_p5=90.0)]
        bl_path = tmp_path / "bl.jsonl"
        cd_path = tmp_path / "cd.jsonl"
        _write_jsonl(bl_path, bl)
        _write_jsonl(cd_path, cd)
        rc = main(["--baseline", str(bl_path), "--candidate", str(cd_path)])
        assert rc == 1

    def test_exit_2_on_missing_file(self, tmp_path: Path) -> None:
        bl_path = tmp_path / "nonexistent.jsonl"
        cd_path = tmp_path / "also_missing.jsonl"
        rc = main(["--baseline", str(bl_path), "--candidate", str(cd_path)])
        assert rc == 2

    def test_exit_2_on_empty_results(self, tmp_path: Path) -> None:
        bl_path = tmp_path / "bl.jsonl"
        cd_path = tmp_path / "cd.jsonl"
        bl_path.write_text("")
        cd_path.write_text(json.dumps(_make_record()) + "\n")
        rc = main(["--baseline", str(bl_path), "--candidate", str(cd_path)])
        assert rc == 2

    def test_report_written(self, tmp_path: Path) -> None:
        bl = [_make_record()]
        cd = [_make_record()]
        bl_path = tmp_path / "bl.jsonl"
        cd_path = tmp_path / "cd.jsonl"
        report_path = tmp_path / "out" / "report.md"
        _write_jsonl(bl_path, bl)
        _write_jsonl(cd_path, cd)
        rc = main(
            [
                "--baseline",
                str(bl_path),
                "--candidate",
                str(cd_path),
                "--report",
                str(report_path),
            ]
        )
        assert rc == 0
        assert report_path.exists()
        content = report_path.read_text()
        assert "Result: PASS" in content

    def test_strict_flag(self, tmp_path: Path) -> None:
        bl = [_make_record(samples_per_sec_p5=120.0)]
        cd = [_make_record(samples_per_sec_p5=100.0)]
        bl_path = tmp_path / "bl.jsonl"
        cd_path = tmp_path / "cd.jsonl"
        _write_jsonl(bl_path, bl)
        _write_jsonl(cd_path, cd)
        rc = main(["--baseline", str(bl_path), "--candidate", str(cd_path), "--strict"])
        assert rc == 1

    def test_threshold_override_cli(self, tmp_path: Path) -> None:
        bl = [_make_record(samples_per_sec_p5=120.0)]
        cd = [_make_record(samples_per_sec_p5=115.0)]  # -4.2%
        bl_path = tmp_path / "bl.jsonl"
        cd_path = tmp_path / "cd.jsonl"
        _write_jsonl(bl_path, bl)
        _write_jsonl(cd_path, cd)
        # Default threshold (0.90) should pass
        rc_pass = main(["--baseline", str(bl_path), "--candidate", str(cd_path)])
        assert rc_pass == 0
        # Strict threshold (0.99) should fail
        rc_fail = main(
            [
                "--baseline",
                str(bl_path),
                "--candidate",
                str(cd_path),
                "--threshold-throughput",
                "0.99",
            ]
        )
        assert rc_fail == 1
