#!/usr/bin/env python3
"""Test attention backbone training stability."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from df_mlx.config import get_default_config
from df_mlx.model import DfNet4
from df_mlx.train import spectral_loss


def main():
    print("Testing attention backbone training stability...")

    # Setup
    config = get_default_config()
    config.backbone.backbone_type = "attention"
    model = DfNet4(config)
    optimizer = optim.AdamW(learning_rate=1e-4)

    # Dummy data
    batch_size, time_steps = 4, 500
    noisy_spec = (
        mx.random.normal((batch_size, time_steps, 481)),
        mx.random.normal((batch_size, time_steps, 481)),
    )
    feat_erb = mx.random.normal((batch_size, time_steps, 32))
    feat_spec = mx.random.normal((batch_size, time_steps, 96, 2))
    target_spec = (
        mx.random.normal((batch_size, time_steps, 481)),
        mx.random.normal((batch_size, time_steps, 481)),
    )

    def loss_fn(model, nr, ni, erb, spec, cr, ci):
        out = model((nr, ni), erb, spec)
        return spectral_loss(out, (cr, ci))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    print("Running 10 training steps...")
    losses = []
    for step in range(10):
        loss, grads = loss_and_grad(
            model,
            noisy_spec[0],
            noisy_spec[1],
            feat_erb,
            feat_spec,
            target_spec[0],
            target_spec[1],
        )
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())
        print(f"  Step {step + 1}: loss = {loss.item():.4f}")

    # Verify loss is decreasing (or stable)
    print(f"\nLoss trend: {losses[0]:.4f} -> {losses[-1]:.4f}")
    if losses[-1] < losses[0] * 1.5:  # Allow some fluctuation
        print("✅ Training appears stable!")
    else:
        print("⚠️ Training may be unstable")


if __name__ == "__main__":
    main()
