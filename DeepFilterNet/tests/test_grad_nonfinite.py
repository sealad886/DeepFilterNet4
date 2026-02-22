"""Tests for non-finite gradient handling in clip_grad_norm_tree."""

import mlx.core as mx

from df_mlx.grad_utils import clip_grad_norm_tree


def _all_finite(tree):
    from mlx.utils import tree_flatten

    for _, v in tree_flatten(tree):
        if v is None:
            continue
        if not bool(mx.all(mx.isfinite(v))):
            return False
    return True


def _all_zero(tree):
    from mlx.utils import tree_flatten

    for _, v in tree_flatten(tree):
        if v is None:
            continue
        if not bool(mx.all(v == 0.0)):
            return False
    return True


def test_normal_grads_clipped():
    grads = {"a": mx.array([1.0, 2.0, 3.0]), "b": mx.array([4.0, 5.0])}
    clipped, norm = clip_grad_norm_tree(grads, 1.0)
    mx.eval(clipped["a"], clipped["b"], norm)
    assert _all_finite(clipped)
    assert float(norm) > 0


def test_inf_grads_zeroed():
    grads = {"a": mx.array([float("inf"), 2.0, 3.0]), "b": mx.array([4.0, 5.0])}
    clipped, norm = clip_grad_norm_tree(grads, 1.0)
    mx.eval(clipped["a"], clipped["b"], norm)
    assert _all_zero(clipped), f"Expected all zeros, got a={clipped['a'].tolist()}, b={clipped['b'].tolist()}"
    assert not bool(mx.isfinite(norm)), "Norm should be inf when grads contain inf"


def test_nan_grads_zeroed():
    grads = {"a": mx.array([float("nan"), 2.0, 3.0]), "b": mx.array([4.0, 5.0])}
    clipped, norm = clip_grad_norm_tree(grads, 1.0)
    mx.eval(clipped["a"], clipped["b"], norm)
    assert _all_zero(clipped), f"Expected all zeros, got a={clipped['a'].tolist()}, b={clipped['b'].tolist()}"


def test_mixed_inf_nan_zeroed():
    grads = {"layer": {"w": mx.array([float("inf"), float("nan")]), "b": mx.array([1.0])}}
    clipped, norm = clip_grad_norm_tree(grads, 0.5)
    mx.eval(norm)
    assert _all_zero(clipped)


def test_small_grads_not_clipped():
    grads = {"a": mx.array([0.01, 0.02])}
    clipped, norm = clip_grad_norm_tree(grads, 10.0)
    mx.eval(clipped["a"], norm)
    # Small grads should pass through unchanged
    assert abs(float(clipped["a"][0]) - 0.01) < 1e-5
    assert abs(float(clipped["a"][1]) - 0.02) < 1e-5


def test_empty_grads():
    grads = {}
    clipped, norm = clip_grad_norm_tree(grads, 1.0)
    mx.eval(norm)
    assert float(norm) == 0.0
