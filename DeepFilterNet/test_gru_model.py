#!/usr/bin/env python3
"""Test GRU model forward and backward pass for numerical issues."""

import mlx.core as mx
import mlx.nn as nn

from df_mlx.config import get_default_config
from df_mlx.model import DfNet4
from df_mlx.train import spectral_loss


def main():
    # Create GRU model
    config = get_default_config()
    config.backbone.backbone_type = "gru"
    model = DfNet4(config)

    def count_params(params):
        total = 0
        for v in params.values():
            if isinstance(v, mx.array):
                total += v.size
            elif isinstance(v, dict):
                total += count_params(v)
        return total

    num_params = count_params(model.parameters())
    print(f"Model created with {num_params:,} parameters")

    # Create test data matching what train_dynamic.py passes
    batch, time_steps = 4, 500
    noisy_spec = (
        mx.random.normal((batch, time_steps, 481)),
        mx.random.normal((batch, time_steps, 481)),
    )
    feat_erb = mx.random.normal((batch, time_steps, 32))
    feat_spec = mx.random.normal((batch, time_steps, 96, 2))
    target_spec = (
        mx.random.normal((batch, time_steps, 481)),
        mx.random.normal((batch, time_steps, 481)),
    )

    print("\nTesting forward pass...")

    # Forward pass
    out = model(noisy_spec, feat_erb, feat_spec)
    mx.eval(out)

    print(f"Output type: {type(out)}")
    print(f"Output[0] shape: {out[0].shape}")
    print(f"Output[0] min/max: {float(mx.min(out[0])):.6f}/{float(mx.max(out[0])):.6f}")
    print(f"Output[0] any NaN: {bool(mx.any(mx.isnan(out[0])))}")
    print(f"Output[0] any Inf: {bool(mx.any(mx.isinf(out[0])))}")
    print(f"Output[1] shape: {out[1].shape}")
    print(f"Output[1] min/max: {float(mx.min(out[1])):.6f}/{float(mx.max(out[1])):.6f}")

    # Test loss
    print("\nTesting loss computation...")
    loss = spectral_loss(out, target_spec)
    mx.eval(loss)
    print(f"Loss value: {float(loss):.6f}")
    print(f"Loss is NaN: {bool(mx.isnan(loss))}")
    print(f"Loss is Inf: {bool(mx.isinf(loss))}")

    # Test backward pass
    print("\nTesting backward pass...")

    def loss_fn(model, noisy_spec, feat_erb, feat_spec, target_spec):
        out = model(noisy_spec, feat_erb, feat_spec)
        return spectral_loss(out, target_spec)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss_val, grads = loss_and_grad(model, noisy_spec, feat_erb, feat_spec, target_spec)
    mx.eval(loss_val, grads)

    print(f"Loss value: {float(loss_val):.6f}")
    print(f"Loss is Inf: {bool(mx.isinf(loss_val))}")

    # Check gradients
    def check_grads(grads, prefix=""):
        for k, v in grads.items():
            if isinstance(v, mx.array):
                has_nan = bool(mx.any(mx.isnan(v)))
                has_inf = bool(mx.any(mx.isinf(v)))
                if has_nan or has_inf:
                    print(f"  {prefix}{k}: NaN={has_nan}, Inf={has_inf}")
            elif isinstance(v, dict):
                check_grads(v, prefix=f"{prefix}{k}.")

    print("\nChecking for NaN/Inf in gradients...")
    check_grads(grads)

    def get_all_grads(grads, acc=None):
        if acc is None:
            acc = []
        for k, v in grads.items():
            if isinstance(v, mx.array):
                acc.append(v)
            elif isinstance(v, dict):
                get_all_grads(v, acc)
        return acc

    all_grads = get_all_grads(grads)
    if all_grads:
        grad_max = max(float(mx.max(mx.abs(g))) for g in all_grads)
        print(f"Max gradient: {grad_max:.6f}")
    else:
        print("No gradients found!")

    print("\nGRU model forward/backward OK!")


if __name__ == "__main__":
    main()
