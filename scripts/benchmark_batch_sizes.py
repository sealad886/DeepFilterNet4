#!/usr/bin/env python
"""Benchmark different batch sizes to find optimal settings for Apple Silicon."""

import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DeepFilterNet"))


def main():
    # Test if MPS is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing on device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    # Import the model
    from df.config import config
    from df.deepfilternet4 import init_model

    # Load config
    config_path = "/Volumes/TrainingData/output/config.ini"
    if os.path.exists(config_path):
        config.load(config_path)
    else:
        print(f"Config not found at {config_path}, using defaults")

    # Initialize model
    model = init_model(df_state=None, run_df=True, train_mask=True)
    model.train()

    # Test parameters
    SR = 48000
    FFT_SIZE = 960
    HOP_SIZE = 480
    NB_ERB = 32
    NB_DF = 96
    SAMPLE_LEN_S = 5.0
    NUM_FRAMES = int(SAMPLE_LEN_S * SR / HOP_SIZE)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Sample length: {SAMPLE_LEN_S}s ({NUM_FRAMES} frames)")
    print()

    batch_sizes = [1, 2, 3, 4, 6, 8]
    results = []

    print("=" * 80)
    print(
        f"{'Batch':<8} {'Forward (ms)':<14} {'Backward (ms)':<14} {'Total (ms)':<14} {'Throughput':<12} {'Status':<8}"
    )
    print("=" * 80)

    for batch_size in batch_sizes:
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()

        try:
            # Create dummy inputs matching model expectations:
            # spec: [B, 1, T, F, 2] - real/imag as last dim
            # feat_erb: [B, 1, T, E]
            # feat_spec: [B, 1, T, F', 2]
            spec = torch.randn(batch_size, 1, NUM_FRAMES, FFT_SIZE // 2 + 1, 2, device=device)
            feat_erb = torch.randn(batch_size, 1, NUM_FRAMES, NB_ERB, device=device)
            feat_spec = torch.randn(batch_size, 1, NUM_FRAMES, NB_DF, 2, device=device)

            # Warmup
            with torch.no_grad():
                _ = model(spec.clone(), feat_erb.clone(), feat_spec.clone())
            if device.type == "mps":
                torch.mps.synchronize()

            # Benchmark forward pass
            start = time.perf_counter()
            enh, m, lsnr, _ = model(spec.clone(), feat_erb.clone(), feat_spec.clone())
            if device.type == "mps":
                torch.mps.synchronize()
            forward_time = (time.perf_counter() - start) * 1000

            # Benchmark backward pass
            loss = enh.abs().mean() + m.abs().mean()
            start = time.perf_counter()
            loss.backward()
            if device.type == "mps":
                torch.mps.synchronize()
            backward_time = (time.perf_counter() - start) * 1000

            total_time = forward_time + backward_time
            throughput = batch_size / (total_time / 1000)  # samples/sec
            status = "OK"

            results.append(
                {
                    "batch_size": batch_size,
                    "forward_ms": forward_time,
                    "backward_ms": backward_time,
                    "total_ms": total_time,
                    "throughput": throughput,
                    "status": status,
                }
            )

            print(
                f"{batch_size:<8} {forward_time:<14.1f} {backward_time:<14.1f} {total_time:<14.1f} {throughput:<12.2f} {status:<8}"
            )

            # Cleanup
            del spec, feat_erb, feat_spec, enh, m, lsnr, loss
            model.zero_grad()

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS" in str(e):
                print(f"{batch_size:<8} {'--':<14} {'--':<14} {'--':<14} {'--':<12} {'OOM':<8}")
                results.append({"batch_size": batch_size, "status": "OOM"})
            else:
                print(f"{batch_size:<8} {'--':<14} {'--':<14} {'--':<14} {'--':<12} {'ERROR':<8}")
                print(f"  Error: {e}")
                results.append({"batch_size": batch_size, "status": "ERROR"})

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    print("=" * 80)
    print()

    # Find optimal batch size
    successful = [r for r in results if r["status"] == "OK"]
    if successful:
        best = max(successful, key=lambda x: x["throughput"])
        safest = successful[0]  # Smallest batch that works

        print("RECOMMENDATIONS:")
        print(f"  Safest batch size:    {safest['batch_size']} ({safest['throughput']:.2f} samples/sec)")
        print(f"  Best throughput:      {best['batch_size']} ({best['throughput']:.2f} samples/sec)")

        # Calculate efficiency
        efficiency = best["throughput"] / safest["throughput"]
        print(f"  Throughput gain:      {efficiency:.1f}x with batch {best['batch_size']} vs batch 1")
        print()
        print("NOTE: Larger batches give better throughput but use more memory.")
        print("Start conservative and increase if your system remains stable.")
    else:
        print("All batch sizes failed! Check your GPU memory.")


if __name__ == "__main__":
    main()
