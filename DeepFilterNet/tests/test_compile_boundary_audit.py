"""Tests for compile-boundary shape guardrails and retrace-risk invariants.

Validates that:
- _assert_compile_boundary_shapes accepts valid shapes and rejects mismatches
- Compiled step functions exist with expected signatures
- DataLoaders default to drop_last=True
"""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_TRAIN_DYNAMIC_PATH = Path(__file__).resolve().parent.parent / "df_mlx" / "train_dynamic.py"
_DYNAMIC_DATASET_PATH = Path(__file__).resolve().parent.parent / "df_mlx" / "dynamic_dataset.py"


# ---------------------------------------------------------------------------
# Standalone replica of the guardrail function for unit testing.
# The canonical implementation lives inside train() in train_dynamic.py;
# we replicate here so tests don't need to import the full training module.
# ---------------------------------------------------------------------------
def _assert_compile_boundary_shapes(
    noisy: mx.array,
    clean: mx.array,
    expected_batch_size: int,
    *,
    check_dtype: bool = True,
    expected_dtype: mx.Dtype = mx.float32,
) -> None:
    """Validate shape invariants at compile boundary to prevent retracing."""
    if noisy.shape[0] != expected_batch_size:
        raise ValueError(
            f"Compile boundary shape violation: batch_size={noisy.shape[0]}, "
            f"expected={expected_batch_size}. This would trigger an expensive retrace."
        )
    if noisy.shape != clean.shape:
        raise ValueError(f"Compile boundary shape mismatch: noisy={noisy.shape}, clean={clean.shape}")
    if check_dtype and noisy.dtype != expected_dtype:
        raise ValueError(f"Compile boundary dtype mismatch: got {noisy.dtype}, " f"expected {expected_dtype}")


# ---------------------------------------------------------------------------
# Test: _assert_compile_boundary_shapes — valid inputs
# ---------------------------------------------------------------------------
class TestAssertCompileBoundaryShapes:
    """Tests for the compile-boundary shape assertion guard."""

    def test_valid_shapes_accepted(self):
        noisy = mx.zeros((4, 33, 100))
        clean = mx.zeros((4, 33, 100))
        _assert_compile_boundary_shapes(noisy, clean, 4)

    def test_batch_size_mismatch_raises(self):
        noisy = mx.zeros((3, 33, 100))
        clean = mx.zeros((3, 33, 100))
        with pytest.raises(ValueError, match="batch_size=3.*expected=4"):
            _assert_compile_boundary_shapes(noisy, clean, 4)

    def test_shape_mismatch_raises(self):
        noisy = mx.zeros((4, 33, 100))
        clean = mx.zeros((4, 33, 80))
        with pytest.raises(ValueError, match="shape mismatch"):
            _assert_compile_boundary_shapes(noisy, clean, 4)

    def test_dtype_mismatch_raises(self):
        noisy = mx.zeros((4, 33, 100), dtype=mx.float16)
        clean = mx.zeros((4, 33, 100), dtype=mx.float16)
        with pytest.raises(ValueError, match="dtype mismatch"):
            _assert_compile_boundary_shapes(noisy, clean, 4, check_dtype=True, expected_dtype=mx.float32)

    def test_dtype_check_disabled(self):
        noisy = mx.zeros((4, 33, 100), dtype=mx.float16)
        clean = mx.zeros((4, 33, 100), dtype=mx.float16)
        _assert_compile_boundary_shapes(noisy, clean, 4, check_dtype=False)

    def test_fp16_consistent_passes(self):
        noisy = mx.zeros((4, 33, 100), dtype=mx.float16)
        clean = mx.zeros((4, 33, 100), dtype=mx.float16)
        _assert_compile_boundary_shapes(noisy, clean, 4, check_dtype=True, expected_dtype=mx.float16)


# ---------------------------------------------------------------------------
# Test: guardrail functions exist in the source
# ---------------------------------------------------------------------------
class TestGuardrailFunctionsExist:
    """Confirm that _assert_compile_boundary_shapes and _log_compile_retrace_warning
    are defined inside train_dynamic.py."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _TRAIN_DYNAMIC_PATH.read_text()

    def test_assert_fn_exists(self):
        assert "def _assert_compile_boundary_shapes(" in self.source

    def test_retrace_warning_fn_exists(self):
        assert "def _log_compile_retrace_warning(" in self.source

    def test_retrace_warning_has_context_param(self):
        match = re.search(r"def _log_compile_retrace_warning\(([^)]*)\)", self.source)
        assert match is not None
        assert "context" in match.group(1)


# ---------------------------------------------------------------------------
# Test: compiled function signatures
# ---------------------------------------------------------------------------
class TestCompiledFunctionSignatures:
    """Confirm compiled_step and compiled_loss_and_grad_step exist with
    expected argument counts in the source code."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _TRAIN_DYNAMIC_PATH.read_text()

    def test_compiled_step_exists(self):
        assert "def compiled_step(" in self.source

    def test_compiled_loss_and_grad_step_exists(self):
        assert "def compiled_loss_and_grad_step(" in self.source

    def test_compiled_step_arg_count(self):
        """compiled_step should have 14 explicit parameters (excluding self)."""
        match = re.search(r"def compiled_step\((.*?)\):", self.source, re.DOTALL)
        assert match is not None
        args = [a.strip() for a in match.group(1).split(",") if a.strip()]
        assert len(args) == 14, f"Expected 14 args, got {len(args)}: {args}"

    def test_compiled_loss_and_grad_step_arg_count(self):
        """compiled_loss_and_grad_step should have 13 explicit parameters."""
        match = re.search(r"def compiled_loss_and_grad_step\((.*?)\):", self.source, re.DOTALL)
        assert match is not None
        args = [a.strip() for a in match.group(1).split(",") if a.strip()]
        assert len(args) == 13, f"Expected 13 args, got {len(args)}: {args}"


# ---------------------------------------------------------------------------
# Test: drop_last defaults
# ---------------------------------------------------------------------------
class TestDropLastDefaults:
    """Verify that both data loaders default to drop_last=True."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _DYNAMIC_DATASET_PATH.read_text()

    def test_prefetch_data_loader_drop_last_default(self):
        assert "drop_last: bool = True" in self.source
        # Ensure it's within the PrefetchDataLoader class
        pdl_pos = self.source.find("class PrefetchDataLoader")
        mds_pos = self.source.find("class MLXDataStream")
        drop_pos = self.source.find("drop_last: bool = True", pdl_pos)
        assert pdl_pos < drop_pos < mds_pos

    def test_mlx_data_stream_drop_last_default(self):
        mds_pos = self.source.find("class MLXDataStream")
        drop_pos = self.source.find("drop_last: bool = True", mds_pos)
        assert drop_pos > mds_pos


# ---------------------------------------------------------------------------
# Test: shape assertion is invoked at compile boundary
# ---------------------------------------------------------------------------
class TestShapeAssertionIntegration:
    """Validate that _assert_compile_boundary_shapes is called before the
    compiled step in the training loop source."""

    def test_assertion_before_compiled_step_call(self):
        source = _TRAIN_DYNAMIC_PATH.read_text()
        assert_pos = source.find("_assert_compile_boundary_shapes(")
        compiled_call_pos = source.find("compiled_loss_and_grad_step(")
        assert assert_pos != -1, "Shape assertion not found in source"
        assert compiled_call_pos != -1, "compiled_loss_and_grad_step call not found"
        assert assert_pos < compiled_call_pos, "Shape assertion must appear before compiled step call"

    def test_compiled_path_has_partial_batch_fallback(self):
        source = _TRAIN_DYNAMIC_PATH.read_text()
        assert "use_compiled_step_for_batch = epoch_use_compiled_step and current_batch_size == batch_size" in source
        assert "falling back to eager for this batch to avoid retrace" in source
