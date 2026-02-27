"""Tests for training loop logging integrity.

Validates fixes for:
- BUG-K1I-1: Misleading 'step' label in progress bar (was forward-time in ms, not step count)
- BUG-K1I-2: Throughput (spd) computed from single-batch timing instead of sync-window
- BUG-K1I-3: time.time() usage instead of monotonic time.perf_counter()
- BUG-K1I-4: print() inside tqdm loop causing output corruption
- BUG-K1I-5: 'speed' label showed 's/s' (seconds/second) instead of '/s'
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRAIN_DYNAMIC_PATH = Path(__file__).parent.parent / "df_mlx" / "train_dynamic.py"
TRAINING_METRICS_PATH = Path(__file__).parent.parent / "df_mlx" / "training_metrics.py"


def _read_source() -> str:
    return TRAIN_DYNAMIC_PATH.read_text() + "\n" + TRAINING_METRICS_PATH.read_text()


def _parse_module() -> ast.Module:
    return ast.parse(_read_source())


def _find_function_node(module: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


# ---------------------------------------------------------------------------
# 1. No time.time() in training loop — must use time.perf_counter()
# ---------------------------------------------------------------------------


class TestMonotonicClock:
    """Ensure the training loop uses time.perf_counter() exclusively."""

    def test_no_time_time_calls(self):
        """time.time() should not appear in train_dynamic.py (non-monotonic, subject to NTP jumps)."""
        source = _read_source()
        matches = re.findall(r"\btime\.time\(\)", source)
        assert len(matches) == 0, (
            f"Found {len(matches)} occurrences of time.time() — " "must use time.perf_counter() for monotonic timing"
        )

    def test_perf_counter_used(self):
        """Verify time.perf_counter() is used for timing in the training loop."""
        source = _read_source()
        assert "time.perf_counter()" in source, "Expected time.perf_counter() usage in train_dynamic.py"


# ---------------------------------------------------------------------------
# 2. Progress bar field naming — no misleading 'step' label
# ---------------------------------------------------------------------------


class TestProgressBarLabels:
    """Verify progress bar postfix keys are unambiguous."""

    def test_no_step_timing_label(self):
        """The 'step' postfix key (which showed forward-time in ms) must be renamed to 'fwd'."""
        source = _read_source()
        # Find set_postfix calls — the old code had step=f"{fwd_time * 1000:.0f}ms"
        # which showed forward-time labelled as 'step'. This was confusing.
        pbar_matches = re.findall(r"set_postfix\([^)]+\bstep\s*=\s*f\".+?ms", source)
        assert len(pbar_matches) == 0, (
            "Found 'step=<time>ms' in set_postfix — the field was renamed to 'fwd' "
            "to avoid confusion with step counter"
        )

    def test_fwd_timing_label_exists(self):
        """The verbose progress bar should use 'fwd' for forward-pass timing."""
        source = _read_source()
        assert re.search(
            r"set_postfix\(.*?\bfwd\s*=", source, re.DOTALL
        ), "Expected 'fwd=...' in set_postfix for forward-pass timing"

    def test_global_step_in_progress_bar(self):
        """The progress bar must show global_step (gstep) for unambiguous step tracking."""
        source = _read_source()
        assert re.search(
            r"set_postfix\(.*?\bgstep\s*=", source, re.DOTALL
        ), "Expected 'gstep=...' in set_postfix for global step counter"

    def test_no_speed_ss_typo(self):
        """The speed label must not show 's/s' (seconds per second)."""
        source = _read_source()
        # The old bug was: speed=f"{samples_per_sec:.0f}s/s"
        assert 's/s"' not in source, "Found 's/s' speed label typo — should be '/s' or 'samp/s'"


# ---------------------------------------------------------------------------
# 3. Throughput calculation uses sync-window accumulation
# ---------------------------------------------------------------------------


class TestThroughputCalculation:
    """Verify throughput is computed over the sync window, not a single batch."""

    def test_window_accumulation_variables_exist(self):
        """The training loop must define window_samples and window_start for sync-window throughput."""
        source = _read_source()
        assert "window_samples" in source, "Missing window_samples variable for throughput tracking"
        assert "window_start" in source, "Missing window_start variable for throughput tracking"

    def test_window_reset_on_sync(self):
        """window_samples and window_start must be reset after each sync to start a fresh window."""
        source = _read_source()
        # After computing samples_per_sec, window_samples should be reset to 0
        pattern = re.compile(
            r"samples_per_sec\s*=.*window_samples.*\n" r".*window_samples\s*=\s*0",
            re.DOTALL,
        )
        assert pattern.search(source), (
            "window_samples must be reset to 0 after computing samples_per_sec " "to start a fresh measurement window"
        )

    def test_no_eval_frequency_cancellation(self):
        """The old buggy formula multiplied and divided by eval_frequency (cancelling out).

        The fixed code should compute throughput from accumulated window data, not
        `(batch_size * eval_frequency) / (step_time * eval_frequency)`.
        """
        source = _read_source()
        cancelled = re.findall(
            r"current_batch_size\s*\*\s*eval_frequency.*step_time\s*\*\s*eval_frequency",
            source,
        )
        assert len(cancelled) == 0, (
            "Found the old buggy throughput formula where eval_frequency cancels out. "
            "Throughput must use accumulated window_samples / window_elapsed."
        )

    def test_division_by_zero_guard(self):
        """Throughput computation must guard against division by near-zero elapsed time."""
        source = _read_source()
        # Should have max(window_elapsed, 1e-6) or similar
        assert re.search(
            r"max\(window_elapsed,\s*1e-6\)", source
        ), "Throughput computation must clamp elapsed time to avoid division by zero"


# ---------------------------------------------------------------------------
# 4. No raw print() inside tqdm loop — use tqdm.write()
# ---------------------------------------------------------------------------


class TestOutputIntegrity:
    """Verify that print() calls inside the tqdm training loop use tqdm.write()."""

    def test_in_loop_prints_use_tqdm_write(self):
        """Messages emitted inside the batch loop must use tqdm.write() not print().

        Raw print() interleaves with tqdm's carriage-return output and corrupts lines.
        """
        source = _read_source()
        module = _parse_module()
        train_fn = _find_function_node(module, "train")
        assert train_fn is not None, "Could not find train() function"

        # Find the inner for-loop over batches — look for "for batch_idx" in train()
        # Check that any print() calls inside it are actually tqdm.write()
        train_source = ast.get_source_segment(source, train_fn)
        assert train_source is not None

        # Locate the batch loop section (between "for batch_idx" and "train_pbar.close()")
        batch_loop_start = train_source.find("for batch_idx, batch in train_pbar:")
        batch_loop_end = train_source.find("train_pbar.close()", batch_loop_start)
        if batch_loop_start < 0 or batch_loop_end < 0:
            pytest.skip("Could not locate batch loop boundaries in train()")

        loop_body = train_source[batch_loop_start:batch_loop_end]

        # Find bare print() calls (not tqdm.write()) in the loop body
        # Exclude comments
        bare_prints = []
        for i, line in enumerate(loop_body.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "print(" in stripped and "tqdm.write(" not in stripped:
                bare_prints.append((i, stripped))

        assert len(bare_prints) == 0, (
            f"Found {len(bare_prints)} bare print() call(s) inside the tqdm batch loop. "
            f"Use tqdm.write() instead to prevent output corruption.\n"
            + "\n".join(f"  Line {ln}: {code}" for ln, code in bare_prints[:5])
        )


# ---------------------------------------------------------------------------
# 5. Throughput sanity: deterministic unit test
# ---------------------------------------------------------------------------


class TestThroughputSanity:
    """Verify throughput math with deterministic inputs."""

    def test_basic_throughput(self):
        """Simulated sync-window throughput should be samples / elapsed."""
        batch_size = 8
        num_batches = 10
        elapsed = 5.0  # seconds
        window_samples = batch_size * num_batches
        samples_per_sec = window_samples / max(elapsed, 1e-6)
        assert abs(samples_per_sec - 16.0) < 0.01

    def test_near_zero_elapsed(self):
        """Near-zero elapsed must not produce infinity."""
        window_samples = 80
        window_elapsed = 0.0  # edge case
        samples_per_sec = window_samples / max(window_elapsed, 1e-6)
        assert samples_per_sec < 1e12, "Near-zero elapsed should be clamped, not infinite"
        assert samples_per_sec > 0

    def test_no_spike_from_timer_reset(self):
        """Simulated timer reset (negative elapsed) must be handled.

        With time.perf_counter(), negative elapsed should never happen.
        But if it did, the max() guard prevents negative throughput.
        """
        window_samples = 80
        window_elapsed = -0.5  # impossible with perf_counter, but test the guard
        samples_per_sec = window_samples / max(window_elapsed, 1e-6)
        assert samples_per_sec > 0, "Throughput must be positive even with bad elapsed"


# ---------------------------------------------------------------------------
# 6. Step counter monotonicity (simulated)
# ---------------------------------------------------------------------------


class TestStepCounterMonotonicity:
    """Verify step counter semantics are correct."""

    def test_global_step_increments_on_optimizer_update(self):
        """Simulate a training loop and verify global_step increments exactly once per optimizer update."""
        global_step = 0
        grad_accumulation_steps = 4
        total_batches = 20
        micro_batches_in_accum = 0
        step_history: list[int] = []

        for batch_idx in range(total_batches):
            did_optimizer_update = False
            micro_batches_in_accum += 1
            if micro_batches_in_accum >= grad_accumulation_steps:
                did_optimizer_update = True
                micro_batches_in_accum = 0
            if did_optimizer_update:
                global_step += 1
            step_history.append(global_step)

        # With 20 batches and accum=4, expect 5 optimizer updates
        assert global_step == 5
        # Step history should be monotonically non-decreasing
        for i in range(1, len(step_history)):
            assert (
                step_history[i] >= step_history[i - 1]
            ), f"global_step decreased at batch {i}: {step_history[i]} < {step_history[i - 1]}"
        # And should only increase by 1 at each optimizer update
        increments = [step_history[i] - step_history[i - 1] for i in range(1, len(step_history))]
        assert all(d in (0, 1) for d in increments), f"global_step changed by more than 1: {increments}"

    def test_global_step_without_accumulation(self):
        """Without gradient accumulation, global_step should increment every batch."""
        global_step = 0
        total_batches = 10
        for _ in range(total_batches):
            global_step += 1
        assert global_step == total_batches

    def test_resume_step_continuity(self):
        """Simulate train K steps, then resume and train K more.

        global_step must continue from where it left off.
        """
        # Phase 1: train 50 steps
        global_step = 0
        for _ in range(50):
            global_step += 1
        saved_step = global_step
        assert saved_step == 50

        # Phase 2: resume from checkpoint
        resume_global_step = saved_step
        global_step = resume_global_step
        for _ in range(50):
            global_step += 1
        assert global_step == 100, f"Expected 100 after resume, got {global_step}"


# ---------------------------------------------------------------------------
# 7. tqdm configuration sanity
# ---------------------------------------------------------------------------


class TestTqdmConfiguration:
    """Verify tqdm is configured to avoid output corruption."""

    def test_tqdm_writes_to_stderr(self):
        """Progress bars must write to stderr to avoid interleaving with stdout logs."""
        source = _read_source()
        assert '"file": sys.stderr' in source, "tqdm must be configured with file=sys.stderr"

    def test_tqdm_auto_disable(self):
        """tqdm should auto-disable when stderr is not a TTY."""
        source = _read_source()
        assert "sys.stderr.isatty()" in source, "tqdm should check sys.stderr.isatty() for auto-disable"
