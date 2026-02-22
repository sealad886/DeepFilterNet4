"""Gradient-tree utilities shared across training/benchmark entrypoints."""

from __future__ import annotations

from typing import Any, List, Tuple

import mlx.core as mx


def clip_grad_norm_tree(grads: Any, max_norm: float) -> Tuple[Any, mx.array]:
    """Clip a gradient tree by global norm.

    Returns:
        (clipped_tree, total_norm_array)
    """
    flat_grads: List[mx.array] = []

    def flatten(x: Any) -> None:
        if isinstance(x, mx.array):
            flat_grads.append(x.reshape(-1))
        elif isinstance(x, dict):
            for v in x.values():
                flatten(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                flatten(v)

    flatten(grads)
    if not flat_grads:
        return grads, mx.array(0.0)

    total_norm_sq = sum(mx.sum(g**2) for g in flat_grads)
    total_norm = mx.sqrt(total_norm_sq)
    # When total_norm is non-finite (inf/nan from exploding grads),
    # zero all gradients rather than propagating nan through the model.
    norm_finite = mx.isfinite(total_norm)
    safe_norm = mx.where(norm_finite, total_norm, mx.array(1.0))
    clip_coef = mx.minimum(max_norm / (safe_norm + 1e-6), mx.array(1.0))

    def apply_clip(x: Any) -> Any:
        if isinstance(x, mx.array):
            # Use where instead of multiply to avoid inf*0=nan
            clipped = x * clip_coef
            return mx.where(norm_finite, clipped, mx.zeros_like(x))
        if isinstance(x, dict):
            return {k: apply_clip(v) for k, v in x.items()}
        if isinstance(x, list):
            return [apply_clip(v) for v in x]
        if isinstance(x, tuple):
            return tuple(apply_clip(v) for v in x)
        return x

    return apply_clip(grads), total_norm
