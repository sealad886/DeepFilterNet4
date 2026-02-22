"""Gradient-tree utilities shared across training/benchmark entrypoints."""

from __future__ import annotations

from typing import Any, Tuple

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

# Module-level cached scalars — reused across calls to avoid repeated
# micro-allocations in eager (non-compiled) hot paths.  MLX arrays are
# value-immutable so sharing is safe.
_ZERO = mx.array(0.0)
_ONE = mx.array(1.0)


def clip_grad_norm_tree(grads: Any, max_norm: float) -> Tuple[Any, mx.array]:
    """Clip a gradient tree by global norm.

    Returns:
        (clipped_tree, total_norm_array)
    """
    leaves = tree_flatten(grads)
    if not leaves:
        return grads, _ZERO

    total_norm_sq = _ZERO
    for _, g in leaves:
        total_norm_sq = total_norm_sq + mx.sum(g * g)
    total_norm = mx.sqrt(total_norm_sq)
    # When total_norm is non-finite (inf/nan from exploding grads),
    # zero all gradients rather than propagating nan through the model.
    norm_finite = mx.isfinite(total_norm)
    safe_norm = mx.where(norm_finite, total_norm, _ONE)
    clip_coef = mx.minimum(max_norm / (safe_norm + 1e-6), _ONE)

    def _clip_leaf(g: mx.array) -> mx.array:
        clipped = g * clip_coef
        return mx.where(norm_finite, clipped, mx.zeros_like(g))

    return tree_map(_clip_leaf, grads), total_norm
