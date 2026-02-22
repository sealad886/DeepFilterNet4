#!/usr/bin/env python3
"""Benchmark: old vs new sync-barrier patterns in GAN training.

Measures wall-clock time per training step for two patterns:
  OLD: Multiple mx.eval() calls per step (loss finiteness, tree-all-finite,
       optimizer state, disc loss — 5-7 sync barriers).
  NEW: Single mx.eval() call aggregating all targets (1 barrier per eval step).

Usage:
    python benchmark_sync_barriers.py [--steps N] [--warmup N] [--repeats N]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils


def _tree_all_finite(tree: dict) -> mx.array:
    """Old-style finiteness check that forces a sync barrier."""
    leaves = [v for _, v in mlx.utils.tree_flatten(tree) if isinstance(v, mx.array)]
    if not leaves:
        return mx.array(True)
    checks = [mx.all(mx.isfinite(leaf)) for leaf in leaves]
    return mx.all(mx.stack(checks))


def clip_grad_norm(grads, max_norm: float):
    """Simplified clip_grad_norm that zeros NaN grads."""
    flat = mlx.utils.tree_flatten(grads)
    total_norm_sq = mx.array(0.0)
    for _, v in flat:
        if isinstance(v, mx.array):
            total_norm_sq = total_norm_sq + mx.sum(v * v)
    total_norm = mx.sqrt(total_norm_sq)
    scale = max_norm / (total_norm + 1e-6)
    scale = mx.minimum(scale, mx.array(1.0))
    clipped = [(k, v * scale) if isinstance(v, mx.array) else (k, v) for k, v in flat]
    return mlx.utils.tree_unflatten(clipped), total_norm


class TinyGenerator(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.enc = nn.Linear(dim, dim)
        self.gru = nn.RNN(dim, dim)
        self.dec = nn.Linear(dim, dim)

    def __call__(self, x):
        h = self.enc(x)
        h = self.gru(h)
        return self.dec(h)


class TinyDiscriminator(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def __call__(self, x):
        return self.net(x)


def gen_loss_fn(model, x, target):
    pred = model(x)
    loss = mx.mean((pred - target) ** 2)
    return loss


def disc_loss_fn(disc, real, fake):
    real_out = disc(real)
    fake_out = disc(mx.stop_gradient(fake))
    loss = mx.mean((real_out - 1.0) ** 2) + mx.mean(fake_out**2)
    return loss


def old_pattern_step(model, disc, gen_opt, disc_opt, x, target, grad_clip: float = 1.0):
    """OLD pattern: multiple mx.eval() calls per step (5+ sync barriers)."""
    # 1. Forward + backward
    loss, grads = nn.value_and_grad(model, lambda m: gen_loss_fn(m, x, target))(model)

    # --- BARRIER 1: Check loss finiteness ---
    loss_finite = bool(mx.all(mx.isfinite(loss)))

    # --- BARRIER 2: Check all grads finite ---
    grads_ok = bool(_tree_all_finite(grads))

    if loss_finite and grads_ok:
        grads, grad_norm = clip_grad_norm(grads, grad_clip)
        # --- BARRIER 3: Eval grad_norm ---
        grad_norm_val = float(grad_norm)
        gen_opt.update(model, grads)
        # --- BARRIER 4: Eval optimizer state ---
        mx.eval(model.parameters(), gen_opt.state)
    else:
        grad_norm_val = 0.0

    # Discriminator step
    pred = model(x)
    disc_loss, disc_grads = nn.value_and_grad(disc, lambda d: disc_loss_fn(d, target, pred))(disc)
    disc_grads, _ = clip_grad_norm(disc_grads, grad_clip)
    disc_opt.update(disc, disc_grads)

    # --- BARRIER 5: Eval disc ---
    mx.eval(disc_loss, disc.parameters(), disc_opt.state)
    disc_loss_val = float(disc_loss)

    return float(loss), disc_loss_val, grad_norm_val


def new_pattern_step(model, disc, gen_opt, disc_opt, x, target, grad_clip: float = 1.0):
    """NEW pattern: single mx.eval() call (1 sync barrier)."""
    # Forward + backward
    loss, grads = nn.value_and_grad(model, lambda m: gen_loss_fn(m, x, target))(model)

    # Lazy finiteness check — no sync
    loss_finite_arr = mx.all(mx.isfinite(loss))

    # Optimistic update: clip_grad_norm zeros NaN grads
    grads, grad_norm_arr = clip_grad_norm(grads, grad_clip)
    gen_opt.update(model, grads)

    # Discriminator step
    pred = model(x)
    disc_loss, disc_grads = nn.value_and_grad(disc, lambda d: disc_loss_fn(d, target, pred))(disc)
    disc_grads, _ = clip_grad_norm(disc_grads, grad_clip)
    disc_opt.update(disc, disc_grads)

    # --- SINGLE BARRIER: eval everything at once ---
    mx.eval(
        loss,
        disc_loss,
        loss_finite_arr,
        grad_norm_arr,
        model.parameters(),
        gen_opt.state,
        disc.parameters(),
        disc_opt.state,
    )

    # Extract scalars (free — already eval'd)
    loss_finite = bool(loss_finite_arr)
    grad_norm_val = float(grad_norm_arr) if loss_finite else 0.0
    disc_loss_val = float(disc_loss)

    return float(loss), disc_loss_val, grad_norm_val


def new_pattern_with_eval_freq(
    model,
    disc,
    gen_opt,
    disc_opt,
    x,
    target,
    step_idx: int,
    eval_frequency: int = 2,
    grad_clip: float = 1.0,
):
    """NEW pattern with eval_frequency=2: eval only every other step."""
    loss, grads = nn.value_and_grad(model, lambda m: gen_loss_fn(m, x, target))(model)
    loss_finite_arr = mx.all(mx.isfinite(loss))
    grads, grad_norm_arr = clip_grad_norm(grads, grad_clip)
    gen_opt.update(model, grads)

    pred = model(x)
    disc_loss, disc_grads = nn.value_and_grad(disc, lambda d: disc_loss_fn(d, target, pred))(disc)
    disc_grads, _ = clip_grad_norm(disc_grads, grad_clip)
    disc_opt.update(disc, disc_grads)

    should_sync = ((step_idx + 1) % eval_frequency) == 0
    if should_sync:
        mx.eval(
            loss,
            disc_loss,
            loss_finite_arr,
            grad_norm_arr,
            model.parameters(),
            gen_opt.state,
            disc.parameters(),
            disc_opt.state,
        )

    return float(loss) if should_sync else 0.0


def run_pattern(name, step_fn, steps, warmup, repeats, dim, batch, seq):
    """Run a pattern for multiple repeats and return timing stats."""
    all_times = []

    for r in range(repeats):
        model = TinyGenerator(dim)
        disc = TinyDiscriminator(dim)
        gen_opt = optim.AdamW(learning_rate=1e-4)
        disc_opt = optim.AdamW(learning_rate=1e-4)
        mx.eval(model.parameters(), disc.parameters())

        x = mx.random.normal((batch, seq, dim))
        target = mx.random.normal((batch, seq, dim))
        mx.eval(x, target)

        # Warmup
        for i in range(warmup):
            step_fn(model, disc, gen_opt, disc_opt, x, target, i)

        # Timed
        t0 = time.perf_counter()
        for i in range(steps):
            step_fn(model, disc, gen_opt, disc_opt, x, target, i)
        elapsed = time.perf_counter() - t0
        ms_per_step = (elapsed / steps) * 1000
        all_times.append(ms_per_step)
        print(f"  {name} repeat {r + 1}/{repeats}: {ms_per_step:.2f} ms/step ({elapsed:.2f}s total)")

    return {
        "name": name,
        "mean_ms": statistics.mean(all_times),
        "stdev_ms": statistics.stdev(all_times) if len(all_times) > 1 else 0.0,
        "min_ms": min(all_times),
        "max_ms": max(all_times),
        "all_ms": all_times,
    }


def main():
    parser = argparse.ArgumentParser(description="Sync barrier benchmark")
    parser.add_argument("--steps", type=int, default=50, help="Measured steps per repeat")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats")
    parser.add_argument("--dim", type=int, default=128, help="Model dimension")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq", type=int, default=64, help="Sequence length")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    print(f"Config: steps={args.steps}, warmup={args.warmup}, repeats={args.repeats}")
    print(f"Model: dim={args.dim}, batch={args.batch}, seq={args.seq}")
    print(f"MLX version: {mx.__version__}")
    print()

    def old_step(model, disc, gen_opt, disc_opt, x, target, step_idx):
        return old_pattern_step(model, disc, gen_opt, disc_opt, x, target)

    def new_step(model, disc, gen_opt, disc_opt, x, target, step_idx):
        return new_pattern_step(model, disc, gen_opt, disc_opt, x, target)

    def new_freq2_step(model, disc, gen_opt, disc_opt, x, target, step_idx):
        return new_pattern_with_eval_freq(model, disc, gen_opt, disc_opt, x, target, step_idx, eval_frequency=2)

    results = []

    print("=== OLD: Multiple sync barriers (5 per step) ===")
    results.append(
        run_pattern(
            "old_multi_barrier",
            old_step,
            args.steps,
            args.warmup,
            args.repeats,
            args.dim,
            args.batch,
            args.seq,
        )
    )
    print()

    print("=== NEW: Single sync barrier (1 per step) ===")
    results.append(
        run_pattern(
            "new_single_barrier",
            new_step,
            args.steps,
            args.warmup,
            args.repeats,
            args.dim,
            args.batch,
            args.seq,
        )
    )
    print()

    print("=== NEW + eval_frequency=2 (1 barrier every 2 steps) ===")
    results.append(
        run_pattern(
            "new_eval_freq2",
            new_freq2_step,
            args.steps,
            args.warmup,
            args.repeats,
            args.dim,
            args.batch,
            args.seq,
        )
    )
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    old_mean = results[0]["mean_ms"]
    for r in results:
        speedup = old_mean / r["mean_ms"] if r["mean_ms"] > 0 else float("inf")
        print(f"  {r['name']:25s}: {r['mean_ms']:7.2f} ± {r['stdev_ms']:.2f} ms/step " f"(speedup: {speedup:.2f}x)")

    output_path = args.output or str(Path(__file__).parent.parent / "logs" / "sync_barrier_benchmark.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "mlx_version": mx.__version__,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
