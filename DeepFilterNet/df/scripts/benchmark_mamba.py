#!/usr/bin/env python
"""Benchmark script comparing Mamba vs GRU performance.

This script measures inference latency, throughput, and memory usage
for Mamba-based and GRU-based sequence models.

Usage:
    python -m df.scripts.benchmark_mamba [--device cpu|cuda|mps] [--output results.md]
"""

import argparse
import gc
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from df.mamba import Mamba, SqueezedMamba
from df.modules import SqueezedGRU_S


def get_memory_usage(device: torch.device) -> float:
    """Get current memory usage in MB."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    elif device.type == "mps":
        # MPS doesn't expose memory stats directly
        return 0.0
    else:
        return 0.0


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    device: torch.device,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> Dict[str, float]:
    """Benchmark a model's inference performance.
    
    Args:
        model: The model to benchmark
        input_shape: (batch_size, seq_length, features)
        device: Device to run on
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of benchmark iterations
        
    Returns:
        Dictionary with latency_ms, throughput_samples_per_sec, memory_mb
    """
    model = model.to(device)
    model.eval()
    
    batch_size, seq_length, features = input_shape
    x = torch.randn(*input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    latencies = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
                
            start = time.perf_counter()
            _ = model(x)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
                
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    # Get peak memory
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    else:
        peak_memory_mb = 0.0
    
    mean_latency = sum(latencies) / len(latencies)
    throughput = (batch_size * benchmark_runs) / (sum(latencies) / 1000)
    
    return {
        "latency_ms": mean_latency,
        "latency_std_ms": (sum((l - mean_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5,
        "throughput_samples_per_sec": throughput,
        "peak_memory_mb": peak_memory_mb,
    }


def create_mamba_model(
    input_size: int,
    hidden_size: int,
    output_size: int,
    num_layers: int,
) -> nn.Module:
    """Create a SqueezedMamba model."""
    return SqueezedMamba(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
    )


def create_gru_model(
    input_size: int,
    hidden_size: int,
    output_size: int,
    num_layers: int,
) -> nn.Module:
    """Create a SqueezedGRU_S model."""
    return SqueezedGRU_S(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        linear_groups=8,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_benchmarks(
    device: torch.device,
    configurations: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Run benchmarks for various configurations.
    
    Args:
        device: Device to run on
        configurations: List of configs, each with input_size, hidden_size, etc.
        
    Returns:
        List of benchmark results
    """
    if configurations is None:
        configurations = [
            # Small model (like DfNet embedding)
            {"input_size": 96, "hidden_size": 256, "output_size": 96, "num_layers": 2,
             "batch_size": 1, "seq_length": 100, "name": "Small (DfNet-like)"},
            # Medium model
            {"input_size": 256, "hidden_size": 512, "output_size": 256, "num_layers": 2,
             "batch_size": 1, "seq_length": 100, "name": "Medium"},
            # Large model
            {"input_size": 512, "hidden_size": 1024, "output_size": 512, "num_layers": 3,
             "batch_size": 1, "seq_length": 100, "name": "Large"},
            # Long sequence
            {"input_size": 96, "hidden_size": 256, "output_size": 96, "num_layers": 2,
             "batch_size": 1, "seq_length": 500, "name": "Long Sequence"},
            # Batched
            {"input_size": 96, "hidden_size": 256, "output_size": 96, "num_layers": 2,
             "batch_size": 8, "seq_length": 100, "name": "Batched (8)"},
        ]
    
    results = []
    
    for config in configurations:
        print(f"\nBenchmarking: {config['name']}")
        print("-" * 50)
        
        input_shape = (config["batch_size"], config["seq_length"], config["input_size"])
        
        # Create models
        mamba_model = create_mamba_model(
            config["input_size"],
            config["hidden_size"],
            config["output_size"],
            config["num_layers"],
        )
        
        gru_model = create_gru_model(
            config["input_size"],
            config["hidden_size"],
            config["output_size"],
            config["num_layers"],
        )
        
        # Count parameters
        mamba_params = count_parameters(mamba_model)
        gru_params = count_parameters(gru_model)
        
        print(f"  Mamba params: {mamba_params:,}")
        print(f"  GRU params: {gru_params:,}")
        
        # Benchmark Mamba
        print("  Benchmarking Mamba...")
        mamba_results = benchmark_model(mamba_model, input_shape, device)
        
        # Benchmark GRU
        print("  Benchmarking GRU...")
        gru_results = benchmark_model(gru_model, input_shape, device)
        
        result = {
            "name": config["name"],
            "config": config,
            "mamba_params": mamba_params,
            "gru_params": gru_params,
            "mamba": mamba_results,
            "gru": gru_results,
            "speedup": gru_results["latency_ms"] / mamba_results["latency_ms"],
        }
        
        results.append(result)
        
        print(f"  Mamba latency: {mamba_results['latency_ms']:.2f} ms")
        print(f"  GRU latency: {gru_results['latency_ms']:.2f} ms")
        print(f"  Speedup: {result['speedup']:.2f}x")
    
    return results


def format_results_markdown(results: List[Dict], device: str) -> str:
    """Format benchmark results as markdown."""
    lines = [
        "# Mamba vs GRU Benchmark Results",
        "",
        f"**Device:** {device}",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Configuration | Mamba Params | GRU Params | Mamba (ms) | GRU (ms) | Speedup |",
        "|---------------|-------------|------------|------------|----------|---------|",
    ]
    
    for r in results:
        lines.append(
            f"| {r['name']} | {r['mamba_params']:,} | {r['gru_params']:,} | "
            f"{r['mamba']['latency_ms']:.2f} | {r['gru']['latency_ms']:.2f} | "
            f"{r['speedup']:.2f}x |"
        )
    
    lines.extend([
        "",
        "## Detailed Results",
        "",
    ])
    
    for r in results:
        lines.extend([
            f"### {r['name']}",
            "",
            f"- **Input:** batch={r['config']['batch_size']}, "
            f"seq_len={r['config']['seq_length']}, features={r['config']['input_size']}",
            f"- **Hidden:** {r['config']['hidden_size']}, Layers: {r['config']['num_layers']}",
            "",
            "| Metric | Mamba | GRU |",
            "|--------|-------|-----|",
            f"| Parameters | {r['mamba_params']:,} | {r['gru_params']:,} |",
            f"| Latency (ms) | {r['mamba']['latency_ms']:.2f} ± {r['mamba']['latency_std_ms']:.2f} | "
            f"{r['gru']['latency_ms']:.2f} ± {r['gru']['latency_std_ms']:.2f} |",
            f"| Throughput (samples/s) | {r['mamba']['throughput_samples_per_sec']:.1f} | "
            f"{r['gru']['throughput_samples_per_sec']:.1f} |",
            "",
        ])
    
    lines.extend([
        "## Notes",
        "",
        "- Mamba uses selective state space models with O(n) complexity",
        "- GRU uses standard recurrent processing with O(n) sequential operations",
        "- Mamba typically shows better performance on longer sequences due to parallelization",
        "- Parameter counts may differ due to architectural differences",
        "",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mamba vs GRU")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run benchmarks on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for markdown results",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of benchmark runs",
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Running benchmarks on: {device}")
    print("=" * 60)
    
    # Run benchmarks
    results = run_benchmarks(device)
    
    # Format and output results
    markdown = format_results_markdown(results, str(device))
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(markdown)
        print(f"\nResults written to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print(markdown)


if __name__ == "__main__":
    main()
