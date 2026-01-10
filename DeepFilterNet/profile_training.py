#!/usr/bin/env python3
"""Profile training step performance with different backbone types."""

import time

import mlx.core as mx
import mlx.nn as nn

from df_mlx.config import get_default_config
from df_mlx.model import DfNet4
from df_mlx.train import spectral_loss


def profile_backbone(backbone_type: str):
    print(f"\n{'=' * 60}")
    print(f"PROFILING: {backbone_type.upper()} BACKBONE")
    print("=" * 60)

    config = get_default_config()
    config.backbone.backbone_type = backbone_type  # type: ignore
    model = DfNet4(config)

    batch_size = 8
    time_steps = 500  # Reduced for faster testing

    # Create dummy data
    noisy_spec = (mx.random.normal((batch_size, time_steps, 481)), mx.random.normal((batch_size, time_steps, 481)))
    feat_erb = mx.random.normal((batch_size, time_steps, 32))
    feat_spec = mx.random.normal((batch_size, time_steps, 96, 2))
    target_spec = (mx.random.normal((batch_size, time_steps, 481)), mx.random.normal((batch_size, time_steps, 481)))

    # Warmup
    print("Warming up JIT...")
    for _ in range(2):
        out = model(noisy_spec, feat_erb, feat_spec)
        mx.eval(out)

    # Time forward pass
    print("\n=== Forward Pass ===")
    times = []
    for i in range(3):
        start = time.time()
        out = model(noisy_spec, feat_erb, feat_spec)
        mx.eval(out)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  Forward {i + 1}: {elapsed:.1f}ms")
    fwd_avg = sum(times) / len(times)
    print(f"  Average: {fwd_avg:.1f}ms")

    # Time backward pass
    print("\n=== Backward Pass ===")

    def loss_fn(model, nr, ni, erb, spec, cr, ci):
        out = model((nr, ni), erb, spec)
        return spectral_loss(out, (cr, ci))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Warmup backward
    for _ in range(2):
        loss, grads = loss_and_grad(
            model, noisy_spec[0], noisy_spec[1], feat_erb, feat_spec, target_spec[0], target_spec[1]
        )
        mx.eval(loss, grads)

    times = []
    for i in range(3):
        start = time.time()
        loss, grads = loss_and_grad(
            model, noisy_spec[0], noisy_spec[1], feat_erb, feat_spec, target_spec[0], target_spec[1]
        )
        mx.eval(loss, grads)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  Backward {i + 1}: {elapsed:.1f}ms")
    bwd_avg = sum(times) / len(times)
    print(f"  Average: {bwd_avg:.1f}ms")

    print(f"\n=== Summary for {backbone_type.upper()} ===")
    print(f"  Forward:  {fwd_avg:.1f}ms")
    print(f"  Backward: {bwd_avg:.1f}ms")
    print(f"  Ratio (bwd/fwd): {bwd_avg / fwd_avg:.1f}x")

    return fwd_avg, bwd_avg


def main():
    print("Comparing backbone types for training performance")
    print("Batch size: 8, Sequence length: 500 frames")

    results = {}
    for backbone in ["mamba", "gru", "attention"]:
        fwd, bwd = profile_backbone(backbone)
        results[backbone] = {"forward": fwd, "backward": bwd}

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for name, times in results.items():
        total = times["forward"] + times["backward"]
        print(
            f"{name.upper():10s}: Forward {times['forward']:.0f}ms, "
            f"Backward {times['backward']:.0f}ms, Total {total:.0f}ms"
        )

    # Calculate speedups
    if "mamba" in results and "gru" in results:
        speedup = results["mamba"]["backward"] / results["gru"]["backward"]
        print(f"\nGRU backward is {speedup:.1f}x faster than Mamba!")

    if "mamba" in results and "attention" in results:
        speedup = results["mamba"]["backward"] / results["attention"]["backward"]
        print(f"Attention backward is {speedup:.1f}x faster than Mamba!")

    if "gru" in results and "attention" in results:
        speedup = results["gru"]["backward"] / results["attention"]["backward"]
        print(f"Attention backward is {speedup:.1f}x faster than GRU!")


if __name__ == "__main__":
    main()
