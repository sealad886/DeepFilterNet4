#!/usr/bin/env python3
"""Test GRU training with synthetic data to verify the training loop works."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from df_mlx.config import get_default_config
from df_mlx.model import DfNet4
from df_mlx.train import spectral_loss


def count_params(params):
    total = 0
    for v in params.values():
        if isinstance(v, mx.array):
            total += v.size
        elif isinstance(v, dict):
            total += count_params(v)
    return total


def test_training(use_fp16: bool = False):
    dtype_str = "FP16" if use_fp16 else "FP32"
    print(f"\nTesting GRU training with {dtype_str} data...")

    # Create GRU model
    config = get_default_config()
    config.backbone.backbone_type = "gru"
    model = DfNet4(config)

    num_params = count_params(model.parameters())
    print(f"Model: {num_params:,} parameters")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=1e-4)

    # Loss function
    def loss_fn(model, noisy_spec, feat_erb, feat_spec, target_spec):
        out = model(noisy_spec, feat_erb, feat_spec)
        return spectral_loss(out, target_spec)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop with synthetic data
    batch_size = 4
    time_steps = 500
    num_steps = 5

    print(f"Running {num_steps} training steps...")
    losses = []
    dtype = mx.float16 if use_fp16 else mx.float32

    for step in range(num_steps):
        # Generate synthetic data
        noisy_spec = (
            mx.random.normal((batch_size, time_steps, 481)).astype(dtype) * 0.1,
            mx.random.normal((batch_size, time_steps, 481)).astype(dtype) * 0.1,
        )
        feat_erb = mx.random.normal((batch_size, time_steps, 32)).astype(dtype) * 0.1
        feat_spec = mx.random.normal((batch_size, time_steps, 96, 2)).astype(dtype) * 0.1
        target_spec = (
            mx.random.normal((batch_size, time_steps, 481)).astype(dtype) * 0.1,
            mx.random.normal((batch_size, time_steps, 481)).astype(dtype) * 0.1,
        )

        # Forward and backward
        loss, grads = loss_and_grad(model, noisy_spec, feat_erb, feat_spec, target_spec)

        # Update
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)

        loss_val = float(loss)
        losses.append(loss_val)
        print(f"  Step {step + 1}/{num_steps}: loss = {loss_val:.6f}")

    print(f"\n{dtype_str} Training Summary:")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss:   {losses[-1]:.6f}")

    if losses[-1] < losses[0]:
        print(f"  OK: {dtype_str} GRU training works - loss decreased!")
        return True
    else:
        print(f"  WARNING: Loss did not decrease with {dtype_str}")
        return False


def main():
    print("=" * 60)
    print("GRU Training Validation")
    print("=" * 60)

    # Test FP32
    fp32_ok = test_training(use_fp16=False)

    # Test FP16
    fp16_ok = test_training(use_fp16=True)

    print("\n" + "=" * 60)
    print("Final Results:")
    print(f"  FP32 training: {'OK' if fp32_ok else 'FAILED'}")
    print(f"  FP16 training: {'OK' if fp16_ok else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
