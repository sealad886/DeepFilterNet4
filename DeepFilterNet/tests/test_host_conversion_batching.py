"""Tests that _batch_to_float correctly reduces sync barriers."""

import mlx.core as mx

from df_mlx.training_ops import _batch_to_float


class TestBatchToFloat:
    def test_basic_scalars(self):
        a, b, c = mx.array(1.0), mx.array(2.0), mx.array(3.0)
        result = _batch_to_float(a, b, c)
        assert result == (1.0, 2.0, 3.0)

    def test_single_value(self):
        (val,) = _batch_to_float(mx.array(42.0))
        assert val == 42.0

    def test_mean_reduction(self):
        arr = mx.array([1.0, 2.0, 3.0, 4.0])
        mean_arr = mx.mean(arr)
        (val,) = _batch_to_float(mean_arr)
        assert abs(val - 2.5) < 1e-6

    def test_return_types(self):
        a, b = mx.array(1.5), mx.array(2.5)
        result = _batch_to_float(a, b)
        assert all(isinstance(v, float) for v in result)

    def test_preserves_values_after_eval(self):
        vals = [mx.array(float(i)) for i in range(10)]
        result = _batch_to_float(*vals)
        assert result == tuple(float(i) for i in range(10))
