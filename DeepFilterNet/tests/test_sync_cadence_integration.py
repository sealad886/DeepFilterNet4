"""Tests for sync-cadence integration in train_dynamic.py.

Validates that sync_mode from SYNC_BARRIER_POLICY.md is properly integrated
into the training loop: parameter is accepted, fast mode suppresses detailed
metrics, debug mode enables per-step grad_norm logging, profile mode enables
step-level timing, and mx.eval() calls stay within should_sync guards in the
compiled path.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_DF_MLX_DIR = Path(__file__).resolve().parents[1] / "df_mlx"
_train_src = (_DF_MLX_DIR / "train_dynamic.py").read_text()
_cli_main_src = (_DF_MLX_DIR / "training_cli_main.py").read_text()
# Combine both files so source-level grep tests cover the train() definition
# AND the main() caller that was extracted to training_cli_main.py.
TRAIN_SOURCE: str = _train_src + "\n" + _cli_main_src
RUN_CONFIG_SOURCE: str = (_DF_MLX_DIR / "run_config.py").read_text()


# ---------------------------------------------------------------------------
# 1. sync_mode parameter plumbing
# ---------------------------------------------------------------------------


class TestSyncModeParameterPlumbing:
    """Verify sync_mode flows from config through caller to training function."""

    def test_train_function_accepts_sync_mode_parameter(self) -> None:
        assert "sync_mode: str = " in TRAIN_SOURCE

    def test_caller_passes_sync_mode_from_run_config(self) -> None:
        assert "sync_mode=run_cfg.debug.sync_mode" in TRAIN_SOURCE

    def test_sync_mode_logged_at_startup(self) -> None:
        assert "Sync mode: {sync_mode}" in TRAIN_SOURCE

    def test_run_config_defines_sync_mode_field(self) -> None:
        assert "sync_mode: str = cfg_field(" in RUN_CONFIG_SOURCE

    def test_sync_mode_eval_frequency_dict_exists(self) -> None:
        assert "SYNC_MODE_EVAL_FREQUENCY" in RUN_CONFIG_SOURCE

    def test_resolve_run_config_applies_sync_mode_override(self) -> None:
        assert "SYNC_MODE_EVAL_FREQUENCY.get(cfg.debug.sync_mode)" in RUN_CONFIG_SOURCE


# ---------------------------------------------------------------------------
# 2. Fast mode: suppress detailed metrics
# ---------------------------------------------------------------------------


class TestFastModeSuppression:
    """Fast sync_mode skips expensive per-window component metrics."""

    def test_emit_detailed_metrics_flag_defined(self) -> None:
        assert "emit_detailed_metrics = mode.emit_detailed_metrics" in TRAIN_SOURCE

    def test_detailed_metrics_guard_uses_emit_flag(self) -> None:
        assert "emit_detailed_metrics" in TRAIN_SOURCE
        pattern = re.compile(
            r"if\s+emit_detailed_metrics\s+and\s+",
        )
        assert pattern.search(
            TRAIN_SOURCE
        ), "Component loss block should be gated with 'if emit_detailed_metrics and ...'"

    def test_scalar_loss_still_logged_regardless_of_mode(self) -> None:
        """loss_val and throughput must always be logged, even in fast mode."""
        assert "loss_val = float(loss)" in TRAIN_SOURCE
        assert "samples_per_sec" in TRAIN_SOURCE


# ---------------------------------------------------------------------------
# 3. Debug mode: per-step grad norm logging
# ---------------------------------------------------------------------------


class TestDebugModeGradNorm:
    """Debug sync_mode logs grad_norm at every sync point."""

    def test_debug_mode_grad_norm_logging_present(self) -> None:
        assert 'sync_mode == "debug"' in TRAIN_SOURCE

    def test_debug_grad_norm_log_contains_step_and_value(self) -> None:
        pattern = re.compile(r"\[debug\].*step=.*grad_norm=")
        assert pattern.search(TRAIN_SOURCE), "Debug mode should log step and grad_norm"


# ---------------------------------------------------------------------------
# 4. Profile mode: step-level timing
# ---------------------------------------------------------------------------


class TestProfileModeTiming:
    """Profile sync_mode logs step-level timing breakdowns."""

    def test_profile_mode_timing_present(self) -> None:
        assert 'sync_mode == "profile"' in TRAIN_SOURCE

    def test_profile_timing_includes_data_and_fwd(self) -> None:
        assert "data={data_time" in TRAIN_SOURCE, "Profile mode should log data timing"
        assert "fwd={fwd_time" in TRAIN_SOURCE, "Profile mode should log fwd timing"


# ---------------------------------------------------------------------------
# 5. Resume determinism: sync_mode has no checkpoint impact
# ---------------------------------------------------------------------------


class TestResumeDeterminism:
    """sync_mode must not affect checkpoint fields (resume determinism)."""

    def test_save_checkpoint_no_sync_mode_dependency(self) -> None:
        """save_checkpoint call sites should not reference sync_mode."""
        save_blocks = [m.start() for m in re.finditer(r"save_checkpoint\(", TRAIN_SOURCE)]
        assert len(save_blocks) > 0, "save_checkpoint should be called in train_dynamic.py"

        for pos in save_blocks:
            block_end = TRAIN_SOURCE.find(")", pos) + 1
            call_text = TRAIN_SOURCE[pos:block_end]
            assert (
                "sync_mode" not in call_text
            ), f"save_checkpoint call should not reference sync_mode: {call_text[:120]}..."

    def test_checkpoint_state_json_no_sync_mode(self) -> None:
        """Checkpoint state JSON keys should not include sync_mode."""
        assert (
            '"sync_mode"'
            not in TRAIN_SOURCE.replace("sync_mode: str", "")
            .replace("sync_mode=run_cfg", "")
            .split("def save_checkpoint")[0]
        )


# ---------------------------------------------------------------------------
# 6. No accidental barriers: mx.eval() in compiled path stays inside should_sync
# ---------------------------------------------------------------------------


class TestNoAccidentalBarriers:
    """mx.eval() in the compiled training path must be inside should_sync guards."""

    def _extract_compiled_path_block(self) -> str:
        """Extract the compiled training step block from source."""
        start_marker = "if epoch_use_compiled_step:"
        idx = TRAIN_SOURCE.find(start_marker)
        assert idx >= 0, "Could not find compiled step block"

        end_marker = "else:\n                # Standard training step"
        end_idx = TRAIN_SOURCE.find(end_marker, idx)
        assert end_idx >= 0, "Could not find end of compiled step block"
        return TRAIN_SOURCE[idx:end_idx]

    def test_compiled_path_mx_eval_inside_should_sync(self) -> None:
        """All mx.eval() calls in compiled path must be preceded by should_sync check
        or be in the discriminator update section (which always evals)."""
        block = self._extract_compiled_path_block()
        eval_positions = [m.start() for m in re.finditer(r"mx\.eval\(", block)]
        assert len(eval_positions) > 0, "Expected mx.eval() calls in compiled path"

        for pos in eval_positions:
            context_before = block[max(0, pos - 1000) : pos]
            in_disc_update = "do_disc_update" in context_before or "disc_update" in context_before
            in_should_sync = "should_sync" in context_before[-300:]
            in_correctness_check = (
                "_compiled_gan_correctness_verified" in context_before or "eager_loss" in context_before
            )
            assert in_should_sync or in_disc_update or in_correctness_check, (
                f"mx.eval() at char {pos} in compiled block not guarded by should_sync, "
                f"disc update, or correctness check. Context: ...{context_before[-100:]}"
            )

    def test_should_sync_uses_eval_frequency(self) -> None:
        """should_sync must be based on eval_frequency modular arithmetic."""
        pattern = re.compile(r"should_sync\s*=\s*\(batch_idx \+ 1\)\s*%\s*epoch_eval_frequency\s*==\s*0")
        matches = pattern.findall(TRAIN_SOURCE)
        assert len(matches) >= 2, (
            "Expected at least 2 should_sync = (batch_idx + 1) % epoch_eval_frequency == 0 "
            f"(compiled and eager paths), found {len(matches)}"
        )


# ---------------------------------------------------------------------------
# 7. Cross-check: SYNC_BARRIER_POLICY.md modes match run_config.py
# ---------------------------------------------------------------------------


class TestPolicyDocAlignment:
    """Sync modes defined in docs match implementation."""

    def test_all_documented_modes_in_run_config(self) -> None:
        policy_path = Path(__file__).resolve().parents[2] / "docs" / "SYNC_BARRIER_POLICY.md"
        if not policy_path.exists():
            pytest.skip("SYNC_BARRIER_POLICY.md not found")

        policy_text = policy_path.read_text()
        for mode in ("fast", "normal", "debug", "profile"):
            assert (
                f"`{mode}`" in policy_text or f'"{mode}"' in policy_text
            ), f"Mode '{mode}' not documented in SYNC_BARRIER_POLICY.md"
            assert f'"{mode}"' in RUN_CONFIG_SOURCE, f"Mode '{mode}' not found in run_config.py"
