#!/usr/bin/env python3
"""Benchmark for comparing before/after performance optimization.

Measures throughput, latency, and memory for key components:
- DfNet4 full forward pass (inference)
- DfNet4 forward + backward (training step)
- MambaBlock selective scan (isolated)
- Post-filter operation (Metal kernel vs fallback)
- SqueezedAttention forward

Usage:
    # From the repo root (with venv activated):
    python scripts/benchmark_perf_comparison.py --output /tmp/bench_results.json

The script automatically detects whether the Metal post-filter kernel
is available and benchmarks accordingly.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DeepFilterNet"))


def get_memory_mb() -> float:
    """Get current active GPU memory in MB."""
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except AttributeError:
        try:
            return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            return 0.0


def get_peak_memory_mb() -> float:
    """Get peak GPU memory in MB."""
    try:
        return mx.get_peak_memory() / (1024 * 1024)
    except AttributeError:
        try:
            return mx.metal.get_peak_memory() / (1024 * 1024)
        except Exception:
            return 0.0


def reset_peak_memory() -> None:
    """Reset peak memory tracking."""
    try:
        mx.reset_peak_memory()
    except AttributeError:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass


def clear_cache() -> None:
    """Clear GPU cache."""
    try:
        mx.clear_cache()
    except AttributeError:
        try:
            mx.metal.clear_cache()
        except Exception:
            pass


def benchmark_fn(
    fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    warmup: int = 5,
    runs: int = 20,
    sync: bool = True,
    track_memory: bool = False,
) -> Dict[str, float]:
    """Benchmark a function with warmup, timing, and optional memory tracking.

    Returns dict with: mean_ms, std_ms, min_ms, max_ms, median_ms,
                       and optionally: peak_memory_mb, avg_memory_mb
    """
    if kwargs is None:
        kwargs = {}

    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        if sync:
            if isinstance(result, (tuple, list)):
                mx.eval(*[r for r in result if isinstance(r, mx.array)])
            elif isinstance(result, mx.array):
                mx.eval(result)
            elif isinstance(result, dict):
                mx.eval(*[v for v in result.values() if isinstance(v, mx.array)])

    gc.collect()
    clear_cache()

    # Timed runs
    times: List[float] = []
    peak_mems: List[float] = []
    avg_mems: List[float] = []

    for _ in range(runs):
        if track_memory:
            gc.collect()
            clear_cache()
            reset_peak_memory()
            mem_before = get_memory_mb()

        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if sync:
            if isinstance(result, (tuple, list)):
                mx.eval(*[r for r in result if isinstance(r, mx.array)])
            elif isinstance(result, mx.array):
                mx.eval(result)
            elif isinstance(result, dict):
                mx.eval(*[v for v in result.values() if isinstance(v, mx.array)])
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

        if track_memory:
            peak_mems.append(get_peak_memory_mb())
            avg_mems.append(get_memory_mb() - mem_before)

    result_dict: Dict[str, float] = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
    }

    if track_memory and peak_mems:
        result_dict["peak_memory_mb"] = float(np.max(peak_mems))
        result_dict["avg_memory_mb"] = float(np.mean(avg_mems))

    return result_dict


def format_result(name: str, result: Dict[str, float], width: int = 45) -> str:
    """Format a benchmark result as a readable line."""
    mem_str = ""
    if "peak_memory_mb" in result:
        mem_str = f"  | peak={result['peak_memory_mb']:.1f}MB"
    return (
        f"  {name:<{width}}: {result['mean_ms']:8.2f} ± {result['std_ms']:5.2f} ms"
        f"  (min={result['min_ms']:.2f}, max={result['max_ms']:.2f}){mem_str}"
    )


def bench_dfnet4_forward(batch_sizes: List[int], seq_len: int = 100) -> Dict[str, Any]:
    """Benchmark DfNet4 forward pass (inference)."""
    from df_mlx.config import ModelParams4
    from df_mlx.model import DfNet4

    print("\n" + "=" * 80)
    print("BENCHMARK: DfNet4 Forward Pass (Inference)")
    print("=" * 80)

    p = ModelParams4()
    model = DfNet4(p)
    mx.eval(model.parameters())

    results = {}
    for B in batch_sizes:
        spec_real = mx.random.normal((B, seq_len, 481))
        spec_imag = mx.random.normal((B, seq_len, 481))
        feat_erb = mx.random.normal((B, seq_len, 32))
        feat_spec = mx.random.normal((B, seq_len, 96, 2))
        mx.eval(spec_real, spec_imag, feat_erb, feat_spec)

        def fwd():
            return model(
                spec=(spec_real, spec_imag),
                feat_erb=feat_erb,
                feat_spec=feat_spec,
                training=False,
            )

        r = benchmark_fn(fwd, track_memory=True)
        name = f"forward B={B} T={seq_len}"
        print(format_result(name, r))
        results[name] = r

        throughput = (B * seq_len * 480 / 48000) / (r["mean_ms"] / 1000)
        print(f"    -> Throughput: {throughput:.1f} sec-audio/sec-wall")
        r["throughput_audio_sec_per_wall_sec"] = throughput

    return results


def bench_dfnet4_training_step(batch_sizes: List[int], seq_len: int = 100) -> Dict[str, Any]:
    """Benchmark DfNet4 forward + backward + optimizer (training step)."""
    from df_mlx.config import ModelParams4
    from df_mlx.loss import SpectralLoss
    from df_mlx.model import DfNet4

    print("\n" + "=" * 80)
    print("BENCHMARK: DfNet4 Training Step (Forward + Backward + Optimizer)")
    print("=" * 80)

    p = ModelParams4()
    model = DfNet4(p)
    mx.eval(model.parameters())

    loss_fn = SpectralLoss(fft_sizes=(512, 1024, 2048), gamma=0.3, factor=1.0)

    import mlx.optimizers as optim

    optimizer = optim.AdamW(learning_rate=1e-4)
    loss_and_grad = nn.value_and_grad(model, lambda m, *a, **kw: _train_loss(m, loss_fn, *a, **kw))

    results = {}
    for B in batch_sizes:
        spec_real = mx.random.normal((B, seq_len, 481))
        spec_imag = mx.random.normal((B, seq_len, 481))
        feat_erb = mx.random.normal((B, seq_len, 32))
        feat_spec = mx.random.normal((B, seq_len, 96, 2))
        target_wav = mx.random.normal((B, seq_len * 480))
        mx.eval(spec_real, spec_imag, feat_erb, feat_spec, target_wav)

        def train_step():
            loss, grads = loss_and_grad(model, spec_real, spec_imag, feat_erb, feat_spec, target_wav)
            optimizer.update(model, grads)
            return loss

        r = benchmark_fn(train_step, warmup=3, runs=10, track_memory=True)
        name = f"train_step B={B} T={seq_len}"
        print(format_result(name, r))
        results[name] = r

        throughput = (B * seq_len * 480 / 48000) / (r["mean_ms"] / 1000)
        print(f"    -> Throughput: {throughput:.1f} sec-audio/sec-wall")
        r["throughput_audio_sec_per_wall_sec"] = throughput

    return results


def _train_loss(
    model: nn.Module,
    loss_fn: Any,
    spec_real: mx.array,
    spec_imag: mx.array,
    feat_erb: mx.array,
    feat_spec: mx.array,
    target_wav: mx.array,
) -> mx.array:
    """Compute loss for training step benchmark."""
    from df_mlx.ops import istft

    out_real, out_imag = model(
        spec=(spec_real, spec_imag),
        feat_erb=feat_erb,
        feat_spec=feat_spec,
        training=True,
    )
    pred_wav = istft(
        (out_real, out_imag),
        n_fft=960,
        hop_length=480,
        length=target_wav.shape[-1],
    )
    # Align lengths in case of minor iSTFT rounding differences
    min_len = min(pred_wav.shape[-1], target_wav.shape[-1])
    pred_wav = pred_wav[..., :min_len]
    target_wav_aligned = target_wav[..., :min_len]
    loss = loss_fn(pred_wav, target_wav_aligned)
    return loss


def bench_mamba_scan(batch_sizes: List[int], seq_lens: List[int]) -> Dict[str, Any]:
    """Benchmark MambaBlock forward pass (contains selective scan)."""
    from df_mlx.mamba import MambaBlock

    print("\n" + "=" * 80)
    print("BENCHMARK: MambaBlock (Selective Scan)")
    print("=" * 80)

    block = MambaBlock(d_model=256, d_state=16, d_conv=4, expand_factor=2)
    mx.eval(block.parameters())

    results = {}
    for B in batch_sizes:
        for T in seq_lens:
            x = mx.random.normal((B, T, 256))
            mx.eval(x)

            def fwd():
                return block(x)

            r = benchmark_fn(fwd, track_memory=True)
            name = f"mamba_fwd B={B} T={T}"
            print(format_result(name, r))
            results[name] = r

    return results


def bench_post_filter(batch_sizes: List[int], seq_len: int = 100) -> Dict[str, Any]:
    """Benchmark post-filter: Metal kernel vs pure-MLX fallback."""
    print("\n" + "=" * 80)
    print("BENCHMARK: Post-Filter Operation")
    print("=" * 80)

    has_metal_kernel = False
    try:
        from df_mlx.kernels import metal_kernels_available, post_filter_kernel

        has_metal_kernel = metal_kernels_available()
        print(f"  Metal kernel available: {has_metal_kernel}")
    except ImportError:
        print("  Metal post-filter kernel not available (pre-optimization version)")

    results = {}

    for B in batch_sizes:
        enh_real = mx.random.normal((B, seq_len, 481))
        enh_imag = mx.random.normal((B, seq_len, 481))
        orig_real = mx.random.normal((B, seq_len, 481))
        orig_imag = mx.random.normal((B, seq_len, 481))
        mx.eval(enh_real, enh_imag, orig_real, orig_imag)

        # Pure MLX fallback (available in both versions)
        def pf_mlx():
            enh_mag = mx.sqrt(enh_real * enh_real + enh_imag * enh_imag + 1e-12)
            orig_mag = mx.sqrt(orig_real * orig_real + orig_imag * orig_imag + 1e-12)
            mask = enh_mag / mx.maximum(orig_mag, mx.array(1e-10))
            mask = mx.clip(mask, 0.0, 1.0)
            beta = 0.02
            mask_sin = mx.sin(1.5707963 * mask)
            pf = (1.0 - beta) * mask_sin + beta
            return enh_real * pf, enh_imag * pf

        r_mlx = benchmark_fn(pf_mlx, track_memory=True)
        name_mlx = f"post_filter_mlx B={B}"
        print(format_result(name_mlx, r_mlx))
        results[name_mlx] = r_mlx

        if has_metal_kernel:
            beta_arr = mx.array([0.02], dtype=enh_real.dtype)
            mx.eval(beta_arr)

            def pf_metal():
                return post_filter_kernel(enh_real, enh_imag, orig_real, orig_imag, beta_arr)

            r_metal = benchmark_fn(pf_metal, track_memory=True)
            name_metal = f"post_filter_metal B={B}"
            print(format_result(name_metal, r_metal))
            results[name_metal] = r_metal

            if r_mlx["mean_ms"] > 0:
                speedup = r_mlx["mean_ms"] / r_metal["mean_ms"]
                print(f"    -> Metal speedup: {speedup:.2f}x")

    return results


def bench_squeezed_attention(batch_sizes: List[int], seq_len: int = 100) -> Dict[str, Any]:
    """Benchmark SqueezedAttention forward pass."""
    from df_mlx.modules import SqueezedAttention

    print("\n" + "=" * 80)
    print("BENCHMARK: SqueezedAttention")
    print("=" * 80)

    attn = SqueezedAttention(
        input_size=256,
        hidden_size=256,
        output_size=256,
        num_layers=2,
        num_heads=4,
        linear_groups=8,
        gru_skip=True,
    )
    mx.eval(attn.parameters())

    results = {}
    for B in batch_sizes:
        x = mx.random.normal((B, seq_len, 256))
        mx.eval(x)

        def fwd():
            return attn(x)

        r = benchmark_fn(fwd, track_memory=True)
        name = f"attention_fwd B={B} T={seq_len}"
        print(format_result(name, r))
        results[name] = r

    return results


def bench_full_model_memory(batch_sizes: List[int], seq_len: int = 100) -> Dict[str, Any]:
    """Measure peak memory for full model forward and forward+backward."""
    from df_mlx.config import ModelParams4
    from df_mlx.loss import SpectralLoss
    from df_mlx.model import DfNet4

    print("\n" + "=" * 80)
    print("BENCHMARK: Peak Memory Usage")
    print("=" * 80)

    results = {}
    for B in batch_sizes:
        # Fresh model each time to get clean memory measurement
        gc.collect()
        clear_cache()
        reset_peak_memory()

        p = ModelParams4()
        model = DfNet4(p)
        mx.eval(model.parameters())

        spec_real = mx.random.normal((B, seq_len, 481))
        spec_imag = mx.random.normal((B, seq_len, 481))
        feat_erb = mx.random.normal((B, seq_len, 32))
        feat_spec = mx.random.normal((B, seq_len, 96, 2))
        target_wav = mx.random.normal((B, seq_len * 480))
        mx.eval(spec_real, spec_imag, feat_erb, feat_spec, target_wav)

        # Forward-only memory
        gc.collect()
        clear_cache()
        reset_peak_memory()
        mem_before = get_memory_mb()

        out_real, out_imag = model(
            spec=(spec_real, spec_imag),
            feat_erb=feat_erb,
            feat_spec=feat_spec,
            training=False,
        )
        mx.eval(out_real, out_imag)

        fwd_peak = get_peak_memory_mb()
        fwd_active = get_memory_mb()
        name_fwd = f"fwd_memory B={B}"
        results[name_fwd] = {
            "peak_memory_mb": fwd_peak,
            "active_memory_mb": fwd_active,
            "delta_memory_mb": fwd_active - mem_before,
        }
        print(
            f"  {name_fwd:<45}: peak={fwd_peak:.1f}MB"
            f"  active={fwd_active:.1f}MB  delta={fwd_active - mem_before:.1f}MB"
        )

        # Training step memory (forward + backward + optimizer)
        loss_fn = SpectralLoss(fft_sizes=(512, 1024, 2048), gamma=0.3)
        optimizer = __import__("mlx.optimizers", fromlist=["AdamW"]).AdamW(learning_rate=1e-4)
        loss_and_grad = nn.value_and_grad(model, lambda m, *a, **kw: _train_loss(m, loss_fn, *a, **kw))

        gc.collect()
        clear_cache()
        reset_peak_memory()
        mem_before = get_memory_mb()

        loss, grads = loss_and_grad(model, spec_real, spec_imag, feat_erb, feat_spec, target_wav)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())

        train_peak = get_peak_memory_mb()
        train_active = get_memory_mb()
        name_train = f"train_memory B={B}"
        results[name_train] = {
            "peak_memory_mb": train_peak,
            "active_memory_mb": train_active,
            "delta_memory_mb": train_active - mem_before,
        }
        print(
            f"  {name_train:<45}: peak={train_peak:.1f}MB"
            f"  active={train_active:.1f}MB  delta={train_active - mem_before:.1f}MB"
        )

    return results


def collect_environment_info() -> Dict[str, str]:
    """Collect environment information for the benchmark report."""
    info: Dict[str, str] = {
        "python_version": sys.version.split()[0],
        "mlx_version": mx.__version__,
        "metal_available": str(mx.metal.is_available()),
    }

    # Check if post-filter Metal kernel exists
    try:
        from df_mlx.kernels import metal_kernels_available, post_filter_kernel  # noqa: F401

        info["post_filter_kernel"] = str(metal_kernels_available())
    except ImportError:
        info["post_filter_kernel"] = "not_available"

    # Check git info
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        info["git_commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        info["git_branch"] = result.stdout.strip()
    except Exception:
        info["git_commit"] = "unknown"
        info["git_branch"] = "unknown"

    return info


def main():
    parser = argparse.ArgumentParser(description="Performance benchmark comparison")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path for results",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=100,
        help="Sequence length (time frames, default=100 = 1 sec at 48kHz)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer runs, smaller configs",
    )
    args = parser.parse_args()

    if args.quick:
        args.batch_sizes = [1, 4]

    env_info = collect_environment_info()

    print("=" * 80)
    print("PERFORMANCE BENCHMARK: DeepFilterNet4 MLX")
    print("=" * 80)
    print(f"  Python:     {env_info['python_version']}")
    print(f"  MLX:        {env_info['mlx_version']}")
    print(f"  Metal:      {env_info['metal_available']}")
    print(f"  PF kernel:  {env_info.get('post_filter_kernel', 'N/A')}")
    print(f"  Git:        {env_info['git_branch']}@{env_info['git_commit']}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Seq length:  {args.seq_len} frames ({args.seq_len * 480 / 48000:.1f}s)")

    all_results: Dict[str, Any] = {"environment": env_info}

    # Run all benchmarks
    all_results["dfnet4_forward"] = bench_dfnet4_forward(args.batch_sizes, args.seq_len)
    all_results["dfnet4_training"] = bench_dfnet4_training_step(args.batch_sizes, args.seq_len)
    all_results["mamba_scan"] = bench_mamba_scan(
        args.batch_sizes,
        [50, 100, 200] if not args.quick else [50, 100],
    )
    all_results["post_filter"] = bench_post_filter(args.batch_sizes, args.seq_len)
    all_results["attention"] = bench_squeezed_attention(args.batch_sizes, args.seq_len)
    all_results["memory"] = bench_full_model_memory(args.batch_sizes, args.seq_len)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    else:
        # Default output location
        label = env_info.get("git_branch", "unknown").replace("/", "_")
        default_path = Path(f"/tmp/bench_{label}_{env_info.get('git_commit', 'unknown')}.json")
        with open(default_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {default_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    main()
