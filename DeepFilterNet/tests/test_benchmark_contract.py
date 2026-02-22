"""Tests for benchmark contract: metadata collection, thresholds, and matrix generation."""

from __future__ import annotations

from unittest.mock import patch

from df_mlx.benchmark_train_step import (
    CONTRACT_BACKBONES,
    CONTRACT_BATCH_SIZES,
    CONTRACT_COMPILED,
    CONTRACT_FP16,
    CONTRACT_GRAD_ACCUM,
    THRESHOLD_CV_MAX,
    THRESHOLD_LATENCY_FACTOR,
    THRESHOLD_THROUGHPUT_FACTOR,
    check_regression,
    collect_reproducibility_metadata,
    generate_contract_matrix,
)

# ---------------------------------------------------------------------------
# Metadata collection
# ---------------------------------------------------------------------------


class TestCollectReproducibilityMetadata:
    def test_required_top_level_keys(self) -> None:
        meta = collect_reproducibility_metadata()
        for key in ("hardware", "os", "runtime", "commit", "timestamp"):
            assert key in meta, f"Missing required key: {key}"

    def test_hardware_subkeys(self) -> None:
        meta = collect_reproducibility_metadata()
        hw = meta["hardware"]
        assert "chip" in hw
        assert "gpu_cores" in hw
        assert "memory_gb" in hw
        assert isinstance(hw["chip"], str) and len(hw["chip"]) > 0

    def test_os_subkeys(self) -> None:
        meta = collect_reproducibility_metadata()
        os_info = meta["os"]
        assert "name" in os_info
        assert "version" in os_info
        assert isinstance(os_info["name"], str)

    def test_runtime_subkeys(self) -> None:
        meta = collect_reproducibility_metadata()
        rt = meta["runtime"]
        assert "python" in rt
        assert "mlx" in rt
        assert isinstance(rt["python"], str)
        assert isinstance(rt["mlx"], str)

    def test_commit_is_string(self) -> None:
        meta = collect_reproducibility_metadata()
        assert isinstance(meta["commit"], str)

    def test_timestamp_iso_format(self) -> None:
        from datetime import datetime

        meta = collect_reproducibility_metadata()
        ts = meta["timestamp"]
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None

    def test_config_and_hash_included_when_config_provided(self) -> None:
        cfg = {"backbone": "dfnet4", "batch_size": 4}
        meta = collect_reproducibility_metadata(config=cfg)
        assert meta["config"] == cfg
        assert "reproducibility_hash" in meta
        assert len(meta["reproducibility_hash"]) == 64  # SHA-256 hex

    def test_no_config_key_when_none(self) -> None:
        meta = collect_reproducibility_metadata()
        assert "config" not in meta
        assert "reproducibility_hash" not in meta

    def test_hash_deterministic(self) -> None:
        cfg = {"backbone": "dfnet4", "batch_size": 8, "compiled": True}
        h1 = collect_reproducibility_metadata(config=cfg)["reproducibility_hash"]
        h2 = collect_reproducibility_metadata(config=cfg)["reproducibility_hash"]
        assert h1 == h2

    def test_hash_changes_with_config(self) -> None:
        cfg_a = {"backbone": "dfnet4", "batch_size": 4}
        cfg_b = {"backbone": "mamba", "batch_size": 4}
        h_a = collect_reproducibility_metadata(config=cfg_a)["reproducibility_hash"]
        h_b = collect_reproducibility_metadata(config=cfg_b)["reproducibility_hash"]
        assert h_a != h_b


# ---------------------------------------------------------------------------
# Threshold / pass-fail logic
# ---------------------------------------------------------------------------


class TestCheckRegression:
    def test_all_pass(self) -> None:
        result = check_regression(
            new_p5=100.0,
            baseline_p5=100.0,
            new_p95=50.0,
            baseline_p95=50.0,
            new_std=5.0,
            new_mean=100.0,
        )
        assert result["passed"] is True
        assert result["throughput"]["ok"] is True
        assert result["latency"]["ok"] is True
        assert result["variance"]["ok"] is True
        assert result["override"] is False

    def test_throughput_regression_fails(self) -> None:
        result = check_regression(
            new_p5=80.0,
            baseline_p5=100.0,  # threshold = 90
            new_p95=50.0,
            baseline_p95=50.0,
            new_std=5.0,
            new_mean=100.0,
        )
        assert result["passed"] is False
        assert result["throughput"]["ok"] is False

    def test_throughput_at_boundary_passes(self) -> None:
        result = check_regression(
            new_p5=90.0,
            baseline_p5=100.0,
            new_p95=50.0,
            baseline_p95=50.0,
            new_std=5.0,
            new_mean=100.0,
        )
        assert result["throughput"]["ok"] is True
        assert result["passed"] is True

    def test_latency_regression_fails(self) -> None:
        result = check_regression(
            new_p5=100.0,
            baseline_p5=100.0,
            new_p95=60.0,
            baseline_p95=50.0,  # threshold = 57.5
            new_std=5.0,
            new_mean=100.0,
        )
        assert result["passed"] is False
        assert result["latency"]["ok"] is False

    def test_latency_at_boundary_passes(self) -> None:
        # Use a value clearly within the threshold (slightly below boundary)
        result = check_regression(
            new_p5=100.0,
            baseline_p5=100.0,
            new_p95=114.0,
            baseline_p95=100.0,
            new_std=5.0,
            new_mean=100.0,
        )
        assert result["latency"]["ok"] is True

    def test_high_variance_fails(self) -> None:
        result = check_regression(
            new_p5=100.0,
            baseline_p5=100.0,
            new_p95=50.0,
            baseline_p95=50.0,
            new_std=25.0,
            new_mean=100.0,  # CV = 0.25 > 0.20
        )
        assert result["passed"] is False
        assert result["variance"]["ok"] is False

    def test_variance_at_boundary_passes(self) -> None:
        result = check_regression(
            new_p5=100.0,
            baseline_p5=100.0,
            new_p95=50.0,
            baseline_p95=50.0,
            new_std=20.0,
            new_mean=100.0,  # CV = 0.20
        )
        assert result["variance"]["ok"] is True

    def test_zero_mean_fails_variance(self) -> None:
        result = check_regression(
            new_p5=100.0,
            baseline_p5=100.0,
            new_p95=50.0,
            baseline_p95=50.0,
            new_std=1.0,
            new_mean=0.0,
        )
        assert result["variance"]["ok"] is False

    @patch.dict("os.environ", {"BENCHMARK_OVERRIDE": "1"})
    def test_override_forces_pass(self) -> None:
        result = check_regression(
            new_p5=50.0,
            baseline_p5=100.0,  # would fail throughput
            new_p95=100.0,
            baseline_p95=50.0,  # would fail latency
            new_std=30.0,
            new_mean=100.0,  # would fail variance
        )
        assert result["passed"] is True
        assert result["override"] is True

    @patch.dict("os.environ", {"BENCHMARK_OVERRIDE": "0"})
    def test_override_zero_does_not_force_pass(self) -> None:
        result = check_regression(
            new_p5=50.0,
            baseline_p5=100.0,
            new_p95=100.0,
            baseline_p95=50.0,
            new_std=30.0,
            new_mean=100.0,
        )
        assert result["passed"] is False
        assert result["override"] is False

    def test_threshold_constants(self) -> None:
        assert THRESHOLD_THROUGHPUT_FACTOR == 0.90
        assert THRESHOLD_LATENCY_FACTOR == 1.15
        assert THRESHOLD_CV_MAX == 0.20


# ---------------------------------------------------------------------------
# Contract matrix generation
# ---------------------------------------------------------------------------


class TestGenerateContractMatrix:
    def test_matrix_length(self) -> None:
        matrix = generate_contract_matrix()
        expected = (
            len(CONTRACT_BACKBONES)
            * len(CONTRACT_BATCH_SIZES)
            * len(CONTRACT_COMPILED)
            * len(CONTRACT_GRAD_ACCUM)
            * len(CONTRACT_FP16)
        )
        assert len(matrix) == expected

    def test_all_entries_are_dicts(self) -> None:
        for entry in generate_contract_matrix():
            assert isinstance(entry, dict)

    def test_required_keys_present(self) -> None:
        required = {"backbone", "batch_size", "compiled", "grad_accumulation", "fp16"}
        for entry in generate_contract_matrix():
            assert required.issubset(entry.keys()), f"Missing keys in {entry}"

    def test_backbone_values(self) -> None:
        backbones = {e["backbone"] for e in generate_contract_matrix()}
        assert backbones == set(CONTRACT_BACKBONES)

    def test_batch_size_values(self) -> None:
        sizes = {e["batch_size"] for e in generate_contract_matrix()}
        assert sizes == set(CONTRACT_BATCH_SIZES)

    def test_compiled_values(self) -> None:
        compiled = {e["compiled"] for e in generate_contract_matrix()}
        assert compiled == {True, False}

    def test_grad_accum_values(self) -> None:
        accums = {e["grad_accumulation"] for e in generate_contract_matrix()}
        assert accums == set(CONTRACT_GRAD_ACCUM)

    def test_fp16_values(self) -> None:
        fp16_vals = {e["fp16"] for e in generate_contract_matrix()}
        assert fp16_vals == {True, False}

    def test_no_duplicates(self) -> None:
        matrix = generate_contract_matrix()
        seen: set[tuple[str, int, bool, int, bool]] = set()
        for entry in matrix:
            key = (
                entry["backbone"],
                entry["batch_size"],
                entry["compiled"],
                entry["grad_accumulation"],
                entry["fp16"],
            )
            assert key not in seen, f"Duplicate config: {entry}"
            seen.add(key)
