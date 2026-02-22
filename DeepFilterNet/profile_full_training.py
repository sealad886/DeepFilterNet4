#!/usr/bin/env python3
"""Profile full training step to identify bottlenecks.

This profiles each component separately to identify where time is spent:
1. Data loading
2. FP16 conversion (if enabled)
3. Forward pass
4. Loss computation
5. Backward pass
6. Optimizer update
7. mx.eval() synchronization

Usage:
    python DeepFilterNet/profile_full_training.py
"""

import sys
import time
from pathlib import Path

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from df_mlx.config import get_default_config  # noqa: E402
from df_mlx.model import DfNet4  # noqa: E402
from df_mlx.train import spectral_loss  # noqa: E402


def create_batch(batch_size: int, seq_len: int, dtype=mx.float32):
    """Create synthetic batch data."""
    return {
        "noisy_real": mx.random.normal((batch_size, seq_len, 481)).astype(dtype),
        "noisy_imag": mx.random.normal((batch_size, seq_len, 481)).astype(dtype),
        "clean_real": mx.random.normal((batch_size, seq_len, 481)).astype(dtype),
        "clean_imag": mx.random.normal((batch_size, seq_len, 481)).astype(dtype),
        "feat_erb": mx.random.normal((batch_size, seq_len, 32)).astype(dtype),
        "feat_spec": mx.random.normal((batch_size, seq_len, 96, 2)).astype(dtype),
    }


def profile_component(name: str, fn, n_iters: int = 5, warmup: int = 2):
    """Profile a component function."""
    # Warmup
    for _ in range(warmup):
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        elif result is not None:
            mx.eval(result)

    # Time
    times = []
    for _ in range(n_iters):
        start = time.time()
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        elif result is not None:
            mx.eval(result)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    print(f"  {name:30s}: {avg:8.1f}ms (±{std:.1f}ms)")
    return avg


def main():
    print("=" * 70)
    print("FULL TRAINING STEP PROFILER")
    print("=" * 70)

    batch_size = 8
    seq_len = 500  # ~5 seconds at 100Hz frame rate

    for backbone_type in ["gru", "mamba"]:
        print(f"\n{'=' * 70}")
        print(f"BACKBONE: {backbone_type.upper()}")
        print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
        print("=" * 70)

        # Create model
        config = get_default_config()
        config.backbone.backbone_type = backbone_type  # type: ignore
        model = DfNet4(config)
        model.train()
        mx.eval(model.parameters())

        # Create optimizer
        optimizer = optim.AdamW(learning_rate=1e-4)

        # Create loss and grad function
        def loss_fn(model, nr, ni, erb, spec, cr, ci):
            out = model((nr, ni), erb, spec)
            return spectral_loss(out, (cr, ci))

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        # Create state for compiled step
        state = [model.state, optimizer.state]

        # Profile each component
        print("\n--- Component Timing ---")

        # 1. Batch creation (simulates data loading overhead)
        profile_component(
            "1. Batch creation (FP32)",
            lambda: create_batch(batch_size, seq_len, mx.float32),
        )

        # 2. FP16 conversion
        batch_fp32 = create_batch(batch_size, seq_len, mx.float32)
        mx.eval(list(batch_fp32.values()))

        def convert_to_fp16():
            return {k: v.astype(mx.float16) for k, v in batch_fp32.items()}

        profile_component("2. FP16 conversion", convert_to_fp16)

        # 3. Forward pass only
        batch = create_batch(batch_size, seq_len, mx.float32)
        mx.eval(list(batch.values()))

        def forward_only():
            return model(
                (batch["noisy_real"], batch["noisy_imag"]),
                batch["feat_erb"],
                batch["feat_spec"],
            )

        fwd_time = profile_component("3. Forward pass", forward_only)

        # 4. Loss computation only (from model output)
        out = model(
            (batch["noisy_real"], batch["noisy_imag"]),
            batch["feat_erb"],
            batch["feat_spec"],
        )
        mx.eval(out)
        target = (batch["clean_real"], batch["clean_imag"])

        profile_component("4. Loss only", lambda: spectral_loss(out, target))

        # 5. Forward + backward (value_and_grad)
        def fwd_bwd():
            return loss_and_grad(
                model,
                batch["noisy_real"],
                batch["noisy_imag"],
                batch["feat_erb"],
                batch["feat_spec"],
                batch["clean_real"],
                batch["clean_imag"],
            )

        fwd_bwd_time = profile_component("5. Forward + backward", fwd_bwd)

        # 6. Optimizer update only
        _, grads = loss_and_grad(
            model,
            batch["noisy_real"],
            batch["noisy_imag"],
            batch["feat_erb"],
            batch["feat_spec"],
            batch["clean_real"],
            batch["clean_imag"],
        )
        mx.eval(grads)

        profile_component("6. Optimizer update", lambda: optimizer.update(model, grads))

        # 7. Full training step (uncompiled)
        def full_step_uncompiled():
            loss, grads = loss_and_grad(
                model,
                batch["noisy_real"],
                batch["noisy_imag"],
                batch["feat_erb"],
                batch["feat_spec"],
                batch["clean_real"],
                batch["clean_imag"],
            )
            optimizer.update(model, grads)
            return loss

        uncompiled_time = profile_component("7. Full step (uncompiled)", full_step_uncompiled)

        # 8. Compiled training step
        from functools import partial

        @partial(mx.compile, inputs=state, outputs=state)
        def compiled_step():
            loss, grads = loss_and_grad(
                model,
                batch["noisy_real"],
                batch["noisy_imag"],
                batch["feat_erb"],
                batch["feat_spec"],
                batch["clean_real"],
                batch["clean_imag"],
            )
            optimizer.update(model, grads)
            return loss

        compiled_time = profile_component("8. Full step (compiled)", compiled_step)

        # Summary
        print(f"\n--- Summary for {backbone_type.upper()} ---")
        print(f"  Forward only:        {fwd_time:.1f}ms")
        print(f"  Forward + backward:  {fwd_bwd_time:.1f}ms")
        print(f"  Backward overhead:   {fwd_bwd_time - fwd_time:.1f}ms")
        print(f"  Uncompiled step:     {uncompiled_time:.1f}ms")
        print(f"  Compiled step:       {compiled_time:.1f}ms")
        print(f"  Compilation speedup: {uncompiled_time / compiled_time:.2f}x")

        # Clean up memory between backbone tests
        del model, optimizer
        mx.metal.clear_cache() if hasattr(mx, "metal") else None

    # Test larger batch sizes with GRU
    print("\n" + "=" * 70)
    print("BATCH SIZE SCALING (GRU backbone)")
    print("=" * 70)

    config = get_default_config()
    config.backbone.backbone_type = "gru"  # type: ignore
    model = DfNet4(config)
    optimizer = optim.AdamW(learning_rate=1e-4)

    def loss_fn_v2(model, nr, ni, erb, spec, cr, ci):
        out = model((nr, ni), erb, spec)
        return spectral_loss(out, (cr, ci))

    loss_and_grad = nn.value_and_grad(model, loss_fn_v2)
    state = [model.state, optimizer.state]

    for bs in [4, 8, 12, 16, 24]:
        try:
            batch = create_batch(bs, seq_len, mx.float32)
            mx.eval(list(batch.values()))

            from functools import partial

            @partial(mx.compile, inputs=state, outputs=state)
            def compiled_step_bs():
                loss, grads = loss_and_grad(
                    model,
                    batch["noisy_real"],
                    batch["noisy_imag"],
                    batch["feat_erb"],
                    batch["feat_spec"],
                    batch["clean_real"],
                    batch["clean_imag"],
                )
                optimizer.update(model, grads)
                return loss

            # Warmup
            for _ in range(2):
                _ = compiled_step_bs()  # noqa: F841
                mx.eval(state)

            # Time
            times = []
            for _ in range(5):
                start = time.time()
                _ = compiled_step_bs()  # noqa: F841
                mx.eval(state)
                times.append((time.time() - start) * 1000)

            avg = sum(times) / len(times)
            samples_per_sec = bs * 1000 / avg
            print(f"  Batch {bs:2d}: {avg:8.1f}ms ({samples_per_sec:.1f} samples/sec)")

        except Exception as e:
            print(f"  Batch {bs:2d}: FAILED - {e}")

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
