import mlx.core as mx
import numpy as np

from df_mlx.benchmark_train_step import _clip_grad_norm as benchmark_clip_grad_norm
from df_mlx.grad_utils import clip_grad_norm_tree
from df_mlx.train_with_data import clip_grad_norm as train_with_data_clip_grad_norm
from df_mlx.training_ops import clip_grad_norm as train_dynamic_clip_grad_norm


def _tree_to_numpy(tree):
    if isinstance(tree, mx.array):
        return np.asarray(tree)
    if isinstance(tree, dict):
        return {k: _tree_to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_to_numpy(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_to_numpy(v) for v in tree)
    return tree


def _assert_tree_close(a, b, atol=1e-6):
    if isinstance(a, np.ndarray):
        np.testing.assert_allclose(a, b, atol=atol)
        return
    if isinstance(a, dict):
        assert set(a.keys()) == set(b.keys())
        for key in a:
            _assert_tree_close(a[key], b[key], atol=atol)
        return
    if isinstance(a, list):
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            _assert_tree_close(av, bv, atol=atol)
        return
    if isinstance(a, tuple):
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            _assert_tree_close(av, bv, atol=atol)
        return
    assert a == b


def test_clip_grad_norm_tree_matches_wrappers():
    grads = {
        "w1": mx.array([[3.0, 4.0]], dtype=mx.float32),
        "nested": [mx.array([1.0, 2.0, 2.0], dtype=mx.float32)],
    }
    max_norm = 2.5

    ref_tree, ref_norm = clip_grad_norm_tree(grads, max_norm)
    td_tree, td_norm = train_dynamic_clip_grad_norm(grads, max_norm)
    twd_tree, twd_norm = train_with_data_clip_grad_norm(grads, max_norm)
    bench_tree, bench_norm = benchmark_clip_grad_norm(grads, max_norm)

    ref_np = _tree_to_numpy(ref_tree)
    td_np = _tree_to_numpy(td_tree)
    twd_np = _tree_to_numpy(twd_tree)
    bench_np = _tree_to_numpy(bench_tree)

    _assert_tree_close(ref_np, td_np)
    _assert_tree_close(ref_np, twd_np)
    _assert_tree_close(ref_np, bench_np)

    np.testing.assert_allclose(float(td_norm), float(ref_norm), atol=1e-6)
    np.testing.assert_allclose(float(twd_norm), float(ref_norm), atol=1e-6)
    np.testing.assert_allclose(float(bench_norm), float(ref_norm), atol=1e-6)
