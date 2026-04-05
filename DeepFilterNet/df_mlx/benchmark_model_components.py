#!/usr/bin/env python3
"""Component-level model forward/backward pass benchmark.

Measures per-component latency in DfNet4 to identify where GPU time goes.
Uses synthetic data — no audio files required.

Components profiled:
  encoder, backbone (Mamba), vad_head, erb_decoder, df_decoder, df_op,
  post_filter, loss_fn, backward_pass

Example:
    python -m df_mlx.benchmark_model_components --batch-sizes 1,4 --seq-len 40 --iters 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx
import mlx.nn as nn

from df_mlx.benchmark_common import get_chip_name, get_gpu_cores, get_memory_gb


@dataclass(frozen=True)
class ComponentResult:
    component: str
    batch_size: int
    seq_len: int
    compiled: bool
    dtype: str
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    pct_of_total: float  # percentage of full forward pass


def _percentile(data: List[float], p: float) -> float:
    """Simple percentile without numpy."""
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[-1]
    return data[f] + (k - f) * (data[c] - data[f])


def _time_fn(fn, iters: int, warmup: int) -> List[float]:
    """Time a function, returning list of per-call latencies in seconds."""
    for _ in range(warmup):
        out = fn()
        mx.eval(out) if isinstance(out, mx.array) else mx.eval(*_flatten(out))

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        mx.eval(out) if isinstance(out, mx.array) else mx.eval(*_flatten(out))
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return times


def _flatten(obj: Any) -> List[mx.array]:
    """Flatten nested tuples/lists of mx.arrays for mx.eval."""
    if isinstance(obj, mx.array):
        return [obj]
    if isinstance(obj, (tuple, list)):
        result: list[mx.array] = []
        for item in obj:
            result.extend(_flatten(item))
        return result
    return []


def _build_model_and_inputs(
    batch_size: int,
    seq_len: int,
    dtype: mx.Dtype = mx.float32,
) -> Dict[str, Any]:
    """Build DfNet4 model and synthetic inputs."""
    from df_mlx.config import get_default_config
    from df_mlx.model import DfNet4

    p = get_default_config()
    model = DfNet4(p)

    spec_real = mx.random.normal((batch_size, seq_len, p.n_freqs)).astype(dtype)
    spec_imag = mx.random.normal((batch_size, seq_len, p.n_freqs)).astype(dtype)
    feat_erb = mx.random.normal((batch_size, seq_len, p.nb_erb)).astype(dtype)
    feat_spec = mx.random.normal((batch_size, seq_len, p.nb_df, 2)).astype(dtype)

    # Force eval so inputs are materialized
    mx.eval(spec_real, spec_imag, feat_erb, feat_spec)

    return {
        "model": model,
        "p": p,
        "spec_real": spec_real,
        "spec_imag": spec_imag,
        "feat_erb": feat_erb,
        "feat_spec": feat_spec,
    }


def benchmark_components(
    batch_size: int,
    seq_len: int,
    iters: int = 30,
    warmup: int = 5,
    compiled: bool = False,
    use_bf16: bool = False,
) -> List[ComponentResult]:
    """Benchmark each model component individually."""
    dtype = mx.bfloat16 if use_bf16 else mx.float32
    dtype_name = "bf16" if use_bf16 else "f32"

    ctx = _build_model_and_inputs(batch_size, seq_len, dtype)
    model = ctx["model"]
    spec_real = ctx["spec_real"]
    spec_imag = ctx["spec_imag"]
    feat_erb = ctx["feat_erb"]
    feat_spec = ctx["feat_spec"]

    results: List[ComponentResult] = []

    # 1. Full forward pass (baseline for % calculation)
    def full_fwd():
        return model((spec_real, spec_imag), feat_erb, feat_spec, training=True)

    full_times = _time_fn(full_fwd, iters, warmup)
    full_mean = statistics.mean(full_times) * 1000

    full_times_sorted = sorted(full_times)
    results.append(
        ComponentResult(
            component="full_forward",
            batch_size=batch_size,
            seq_len=seq_len,
            compiled=compiled,
            dtype=dtype_name,
            mean_ms=full_mean,
            std_ms=statistics.stdev(full_times) * 1000 if len(full_times) > 1 else 0,
            p50_ms=_percentile(full_times_sorted, 50) * 1000,
            p95_ms=_percentile(full_times_sorted, 95) * 1000,
            pct_of_total=100.0,
        )
    )

    # 2. Encoder
    def encoder_fwd():
        fe = feat_erb
        fs = feat_spec
        if model.conv_lookahead > 0:
            fe = model._pad_features(fe, model.conv_lookahead)
            fs = model._pad_features(fs, model.conv_lookahead)
        return model.encoder(fe, fs)

    enc_times = _time_fn(encoder_fwd, iters, warmup)
    enc_mean = statistics.mean(enc_times) * 1000

    # Get encoder output for subsequent components
    fe = feat_erb
    fs = feat_spec
    if model.conv_lookahead > 0:
        fe = model._pad_features(fe, model.conv_lookahead)
        fs = model._pad_features(fs, model.conv_lookahead)
    emb, lsnr = model.encoder(fe, fs)
    mx.eval(emb, lsnr)

    enc_times_sorted = sorted(enc_times)
    results.append(
        ComponentResult(
            component="encoder",
            batch_size=batch_size,
            seq_len=seq_len,
            compiled=compiled,
            dtype=dtype_name,
            mean_ms=enc_mean,
            std_ms=statistics.stdev(enc_times) * 1000 if len(enc_times) > 1 else 0,
            p50_ms=_percentile(enc_times_sorted, 50) * 1000,
            p95_ms=_percentile(enc_times_sorted, 95) * 1000,
            pct_of_total=enc_mean / full_mean * 100,
        )
    )

    # 3. Backbone (Mamba)
    def backbone_fwd():
        return model.backbone(emb)

    bb_times = _time_fn(backbone_fwd, iters, warmup)
    bb_mean = statistics.mean(bb_times) * 1000
    bb_out, _ = model.backbone(emb)
    mx.eval(bb_out)

    bb_times_sorted = sorted(bb_times)
    results.append(
        ComponentResult(
            component="backbone_mamba",
            batch_size=batch_size,
            seq_len=seq_len,
            compiled=compiled,
            dtype=dtype_name,
            mean_ms=bb_mean,
            std_ms=statistics.stdev(bb_times) * 1000 if len(bb_times) > 1 else 0,
            p50_ms=_percentile(bb_times_sorted, 50) * 1000,
            p95_ms=_percentile(bb_times_sorted, 95) * 1000,
            pct_of_total=bb_mean / full_mean * 100,
        )
    )

    # 4. VAD head
    def vad_fwd():
        return model.vad_head(bb_out)

    vad_times = _time_fn(vad_fwd, iters, warmup)
    vad_mean = statistics.mean(vad_times) * 1000
    vad_times_sorted = sorted(vad_times)
    results.append(
        ComponentResult(
            component="vad_head",
            batch_size=batch_size,
            seq_len=seq_len,
            compiled=compiled,
            dtype=dtype_name,
            mean_ms=vad_mean,
            std_ms=statistics.stdev(vad_times) * 1000 if len(vad_times) > 1 else 0,
            p50_ms=_percentile(vad_times_sorted, 50) * 1000,
            p95_ms=_percentile(vad_times_sorted, 95) * 1000,
            pct_of_total=vad_mean / full_mean * 100,
        )
    )

    # 5. ERB decoder
    def erb_dec_fwd():
        return model.erb_decoder(bb_out)

    erb_times = _time_fn(erb_dec_fwd, iters, warmup)
    erb_mean = statistics.mean(erb_times) * 1000
    erb_mask = model.erb_decoder(bb_out)
    mx.eval(erb_mask)

    erb_times_sorted = sorted(erb_times)
    results.append(
        ComponentResult(
            component="erb_decoder",
            batch_size=batch_size,
            seq_len=seq_len,
            compiled=compiled,
            dtype=dtype_name,
            mean_ms=erb_mean,
            std_ms=statistics.stdev(erb_times) * 1000 if len(erb_times) > 1 else 0,
            p50_ms=_percentile(erb_times_sorted, 50) * 1000,
            p95_ms=_percentile(erb_times_sorted, 95) * 1000,
            pct_of_total=erb_mean / full_mean * 100,
        )
    )

    # 6. DF decoder
    def df_dec_fwd():
        return model.df_decoder(bb_out)

    df_times = _time_fn(df_dec_fwd, iters, warmup)
    df_mean = statistics.mean(df_times) * 1000
    df_times_sorted = sorted(df_times)
    results.append(
        ComponentResult(
            component="df_decoder",
            batch_size=batch_size,
            seq_len=seq_len,
            compiled=compiled,
            dtype=dtype_name,
            mean_ms=df_mean,
            std_ms=statistics.stdev(df_times) * 1000 if len(df_times) > 1 else 0,
            p50_ms=_percentile(df_times_sorted, 50) * 1000,
            p95_ms=_percentile(df_times_sorted, 95) * 1000,
            pct_of_total=df_mean / full_mean * 100,
        )
    )

    # 7. Mask expansion + DfOp (combined since they're sequential)
    mask = mx.matmul(erb_mask, model._erb_fb_T)
    masked_real = spec_real * mask
    masked_imag = spec_imag * mask
    df_out = model.df_decoder(bb_out)
    mx.eval(mask, masked_real, masked_imag, df_out)

    if model.df_op is not None:

        def dfop_fwd():
            return model.df_op((masked_real, masked_imag), df_out)

        dfop_times = _time_fn(dfop_fwd, iters, warmup)
        dfop_mean = statistics.mean(dfop_times) * 1000
        dfop_times_sorted = sorted(dfop_times)
        results.append(
            ComponentResult(
                component="df_op",
                batch_size=batch_size,
                seq_len=seq_len,
                compiled=compiled,
                dtype=dtype_name,
                mean_ms=dfop_mean,
                std_ms=statistics.stdev(dfop_times) * 1000 if len(dfop_times) > 1 else 0,
                p50_ms=_percentile(dfop_times_sorted, 50) * 1000,
                p95_ms=_percentile(dfop_times_sorted, 95) * 1000,
                pct_of_total=dfop_mean / full_mean * 100,
            )
        )

    # 8. Full forward + backward pass (value_and_grad)
    loss_fn = nn.value_and_grad(
        model,
        lambda m, sr, si, fe, fs: mx.mean(
            mx.square(m((sr, si), fe, fs, training=True)[0]) + mx.square(m((sr, si), fe, fs, training=True)[1])
        ),
    )

    def fwd_bwd():
        return loss_fn(model, spec_real, spec_imag, feat_erb, feat_spec)

    fwd_bwd_times = _time_fn(fwd_bwd, iters, warmup)
    fwd_bwd_mean = statistics.mean(fwd_bwd_times) * 1000
    fwd_bwd_times_sorted = sorted(fwd_bwd_times)
    results.append(
        ComponentResult(
            component="forward_backward",
            batch_size=batch_size,
            seq_len=seq_len,
            compiled=compiled,
            dtype=dtype_name,
            mean_ms=fwd_bwd_mean,
            std_ms=statistics.stdev(fwd_bwd_times) * 1000 if len(fwd_bwd_times) > 1 else 0,
            p50_ms=_percentile(fwd_bwd_times_sorted, 50) * 1000,
            p95_ms=_percentile(fwd_bwd_times_sorted, 95) * 1000,
            pct_of_total=fwd_bwd_mean / full_mean * 100,
        )
    )

    # 9. Mamba selective scan isolation
    # SqueezedMamba.layers[i] is Mamba → .mamba is MambaBlock → ._selective_scan
    mamba_block = None
    if hasattr(model.backbone, "layers"):
        for layer in model.backbone.layers:
            if hasattr(layer, "mamba") and isinstance(layer.mamba, nn.Module):
                mamba_block = layer.mamba  # MambaBlock
                break

    if mamba_block is not None:
        d_inner = mamba_block.d_inner
        d_state = mamba_block.d_state
        u = mx.random.normal((batch_size, seq_len, d_inner)).astype(dtype)
        delta = mx.abs(mx.random.normal((batch_size, seq_len, d_inner))).astype(dtype) + 0.01
        A = -mx.exp(mamba_block.A_log)
        B = mx.random.normal((batch_size, seq_len, d_state)).astype(dtype)
        C = mx.random.normal((batch_size, seq_len, d_state)).astype(dtype)
        D = mamba_block.D
        mx.eval(u, delta, A, B, C, D)

        def scan_fwd():
            return mamba_block._selective_scan(u, delta, A, B, C, D)

        scan_times = _time_fn(scan_fwd, iters, warmup)
        scan_mean = statistics.mean(scan_times) * 1000
        scan_times_sorted = sorted(scan_times)
        results.append(
            ComponentResult(
                component="mamba_selective_scan",
                batch_size=batch_size,
                seq_len=seq_len,
                compiled=compiled,
                dtype=dtype_name,
                mean_ms=scan_mean,
                std_ms=statistics.stdev(scan_times) * 1000 if len(scan_times) > 1 else 0,
                p50_ms=_percentile(scan_times_sorted, 50) * 1000,
                p95_ms=_percentile(scan_times_sorted, 95) * 1000,
                pct_of_total=scan_mean / full_mean * 100,
            )
        )

    return results


def print_results(all_results: List[ComponentResult]) -> None:
    """Print a formatted table of results."""
    # Group by batch_size
    batch_sizes = sorted(set(r.batch_size for r in all_results))

    for bs in batch_sizes:
        batch_results = [r for r in all_results if r.batch_size == bs]
        seq_len = batch_results[0].seq_len
        dtype = batch_results[0].dtype

        print(f"\n{'=' * 80}")
        print(f"  Batch={bs}  SeqLen={seq_len}  Dtype={dtype}")
        print(f"{'=' * 80}")
        print(f"  {'Component':<25} {'Mean':>8} {'Std':>8} {'P50':>8} {'P95':>8} {'%Total':>8}")
        print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

        for r in batch_results:
            print(
                f"  {r.component:<25} "
                f"{r.mean_ms:7.2f}ms "
                f"{r.std_ms:7.2f}ms "
                f"{r.p50_ms:7.2f}ms "
                f"{r.p95_ms:7.2f}ms "
                f"{r.pct_of_total:6.1f}%"
            )


def main():
    parser = argparse.ArgumentParser(description="DfNet4 component-level benchmark")
    parser.add_argument("--batch-sizes", type=str, default="1,4", help="Comma-separated batch sizes")
    parser.add_argument("--seq-len", type=int, default=40, help="Sequence length (time frames)")
    parser.add_argument("--iters", type=int, default=30, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--output", type=str, default=None, help="JSONL output file")
    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print("DfNet4 Component Benchmark")
    print(f"  Chip: {get_chip_name()}")
    print(f"  GPU cores: {get_gpu_cores()}")
    print(f"  Memory: {get_memory_gb()} GB")
    print(f"  Dtype: {'bf16' if args.bf16 else 'f32'}")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Iterations: {args.iters} (warmup: {args.warmup})")

    all_results: List[ComponentResult] = []
    for bs in batch_sizes:
        print(f"\n  Benchmarking batch_size={bs}...")
        results = benchmark_components(
            batch_size=bs,
            seq_len=args.seq_len,
            iters=args.iters,
            warmup=args.warmup,
            use_bf16=args.bf16,
        )
        all_results.extend(results)

    print_results(all_results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in all_results:
                f.write(json.dumps(asdict(r)) + "\n")
        print(f"\n  Results written to {out_path}")


if __name__ == "__main__":
    main()
