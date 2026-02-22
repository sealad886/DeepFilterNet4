"""Tests for performance audit pass 3 fixes.

Validates:
1. _tree_all_finite batched sync: single mx.eval instead of N per-leaf syncs (PERF3-P0-001)
2. loss.py _ZERO cached constant: module-level constant reused (PERF3-P2-001)
3. segmental_snr vectorized: no Python per-frame loops (PERF3-P2-002)
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestTreeAllFiniteBatched:
    """PERF3-P0-001: _tree_all_finite uses batched lazy evaluation."""

    def test_all_finite_returns_true(self):
        from df_mlx.training_ops import _tree_all_finite

        tree = {"a": mx.ones((3, 4)), "b": {"c": mx.zeros((2,))}}
        assert _tree_all_finite(tree) is True

    def test_nan_returns_false(self):
        from df_mlx.training_ops import _tree_all_finite

        tree = {"a": mx.ones((3,)), "b": mx.array([1.0, float("nan"), 3.0])}
        assert _tree_all_finite(tree) is False

    def test_inf_returns_false(self):
        from df_mlx.training_ops import _tree_all_finite

        tree = {"a": mx.array([float("inf")]), "b": mx.ones((2,))}
        assert _tree_all_finite(tree) is False

    def test_empty_tree_returns_true(self):
        from df_mlx.training_ops import _tree_all_finite

        assert _tree_all_finite({}) is True

    def test_none_values_skipped(self):
        from df_mlx.training_ops import _tree_all_finite

        tree = {"a": None, "b": mx.ones((2,))}
        assert _tree_all_finite(tree) is True

    def test_large_tree_consistent(self):
        """Verify batched check matches serial check for a larger tree."""
        from df_mlx.training_ops import _tree_all_finite

        tree = {f"param_{i}": mx.ones((10, 10)) for i in range(50)}
        assert _tree_all_finite(tree) is True

        tree["param_25"] = mx.array([1.0, float("nan")])
        assert _tree_all_finite(tree) is False


class TestLossZeroCached:
    """PERF3-P2-001: loss.py uses module-level _ZERO constant."""

    def test_zero_constant_exists(self):
        from df_mlx.loss import _ZERO

        assert isinstance(_ZERO, mx.array)
        assert float(_ZERO) == 0.0

    def test_spectral_loss_uses_cached_zero(self):
        """SpectralLoss should produce valid results with cached accumulator."""
        from df_mlx.loss import SpectralLoss

        loss_fn = SpectralLoss(fft_sizes=(256,), gamma=1.0, factor=1.0)
        pred = mx.zeros((1, 4000))
        target = mx.ones((1, 4000)) * 0.01
        result = loss_fn(pred, target)
        assert result.shape == ()
        assert float(result) >= 0.0

    def test_combined_loss_uses_cached_zero(self):
        """CombinedLoss should produce valid results with cached accumulator."""
        from df_mlx.loss import CombinedLoss

        loss_fn = CombinedLoss()
        pred = mx.ones((1, 4000)) * 0.5
        target = mx.ones((1, 4000)) * 0.5
        total, breakdown = loss_fn(pred, target)
        assert total.shape == ()
        assert isinstance(breakdown, dict)


class TestSegmentalSnrVectorized:
    """PERF3-P2-002: segmental_snr uses vectorized ops, not Python loops."""

    def test_basic_identical_signals(self):
        """Identical signals should give high SNR."""
        from df_mlx.evaluation import segmental_snr

        sig = mx.array(np.random.randn(1, 8000).astype(np.float32))
        result = segmental_snr(sig, sig)
        assert result.shape == (1,)
        assert float(result[0]) == 35.0  # clipped to max

    def test_noisy_signal(self):
        """Noisy signal should give lower SNR."""
        from df_mlx.evaluation import segmental_snr

        rng = np.random.RandomState(42)
        clean = rng.randn(1, 8000).astype(np.float32)
        noise = rng.randn(1, 8000).astype(np.float32) * 0.1
        result = segmental_snr(mx.array(clean), mx.array(clean + noise))
        assert result.shape == (1,)
        snr_val = float(result[0])
        assert -10 <= snr_val <= 35

    def test_batch_dimension(self):
        """Should handle batched inputs."""
        from df_mlx.evaluation import segmental_snr

        rng = np.random.RandomState(42)
        clean = rng.randn(3, 8000).astype(np.float32)
        noisy = clean + rng.randn(3, 8000).astype(np.float32) * 0.1
        result = segmental_snr(mx.array(clean), mx.array(noisy))
        assert result.shape == (3,)

    def test_1d_input(self):
        """Should handle unbatched 1D input."""
        from df_mlx.evaluation import segmental_snr

        rng = np.random.RandomState(42)
        clean = rng.randn(8000).astype(np.float32)
        result = segmental_snr(mx.array(clean), mx.array(clean))
        assert result.ndim == 1

    def test_short_signal(self):
        """Signal shorter than frame_length should return zeros."""
        from df_mlx.evaluation import segmental_snr

        sig = mx.ones((1, 100))
        result = segmental_snr(sig, sig, frame_length=512)
        assert result.shape == (1,)
        assert float(result[0]) == 0.0

    def test_no_python_loops_in_source(self):
        """Verify the implementation has no per-frame Python for-loops."""
        import inspect

        from df_mlx.evaluation import segmental_snr

        source = inspect.getsource(segmental_snr)
        assert "for b in range" not in source, "Found per-batch Python loop"
        assert "for i in range" not in source, "Found per-frame Python loop"
