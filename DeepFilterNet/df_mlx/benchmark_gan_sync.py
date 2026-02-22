#!/usr/bin/env python3
"""Benchmark GAN training sync-barrier patterns.

Compares two sync strategies for the eager GAN training path:

  OLD  – Multiple mx.eval() calls per step (5 barriers):
         1. mx.eval(loss_finite)              — finiteness gate
         2. mx.eval(tree_all_finite)          — param finite check
         3. mx.eval(model.params, opt.state)  — gen materialization
         4. mx.eval(disc_loss)                — disc loss
         5. mx.eval(disc.params, disc_opt)    — disc materialization

  NEW  – Single consolidated mx.eval() per step (1 barrier):
         1. mx.eval(loss, disc_loss, loss_finite,
                    model.params, opt.state,
                    disc.params, disc_opt.state,
                    grad_norm)

Uses synthetic tensors so the measurement isolates GPU↔CPU sync
overhead, which is the dominant cost on Apple Silicon.

Example:
    python -m df_mlx.benchmark_gan_sync
    python -m df_mlx.benchmark_gan_sync --steps 60 --warmup 15 --batch-size 4
    python -m df_mlx.benchmark_gan_sync --json-out logs/gan_sync_benchmark.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from df_mlx.benchmark_common import get_chip_name as _get_chip_name
from df_mlx.benchmark_common import get_gpu_cores as _get_gpu_cores
from df_mlx.benchmark_common import safe_percentile as _safe_percentile
from df_mlx.config import get_default_config
from df_mlx.discriminator import CombinedDiscriminator
from df_mlx.grad_utils import clip_grad_norm_tree
from df_mlx.loss import discriminator_loss
from df_mlx.model import init_model
from df_mlx.train import spectral_loss


@dataclass
class SyncBenchmarkResult:
    pattern: str
    batch_size: int
    steps: int
    warmup_steps: int
    repeats: int
    step_mean_ms: float
    step_median_ms: float
    step_p95_ms: float
    step_p99_ms: float
    step_min_ms: float
    step_max_ms: float
    steps_per_sec: float
    total_seconds: float


def _collect_metadata() -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chip": _get_chip_name(),
        "gpu_cores": _get_gpu_cores(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "metal_memory_gb": round(mx.device_info()["memory_size"] / (1024**3), 1),
    }


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

_FFT_SIZE = 960
_HOP_SIZE = 480
_NB_ERB = 32
_NB_DF = 96
_SAMPLE_RATE = 48000
_SEGMENT_SEC = 2.0  # shorter than training for faster benchmarking
_NUM_FRAMES = int(_SEGMENT_SEC * _SAMPLE_RATE / _HOP_SIZE)
_NUM_FREQS = _FFT_SIZE // 2 + 1

# Waveform length for disc
_WAV_LEN = int(_SEGMENT_SEC * _SAMPLE_RATE)
_DISC_MAX_SAMPLES = 48000


def _make_batch(batch_size: int) -> Dict[str, mx.array]:
    return {
        "noisy_real": mx.random.normal((batch_size, _NUM_FRAMES, _NUM_FREQS)),
        "noisy_imag": mx.random.normal((batch_size, _NUM_FRAMES, _NUM_FREQS)),
        "feat_erb": mx.random.normal((batch_size, _NUM_FRAMES, _NB_ERB)),
        "feat_spec": mx.random.normal((batch_size, _NUM_FRAMES, _NB_DF, 2)),
        "clean_real": mx.random.normal((batch_size, _NUM_FRAMES, _NUM_FREQS)),
        "clean_imag": mx.random.normal((batch_size, _NUM_FRAMES, _NUM_FREQS)),
    }


def _make_waveforms(
    batch_size: int, max_samples: int = _DISC_MAX_SAMPLES
) -> Tuple[mx.array, mx.array]:
    pred_wav = mx.random.normal((batch_size, max_samples))
    clean_wav = mx.random.normal((batch_size, max_samples))
    return pred_wav, clean_wav


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def _build_models(
    batch_size: int,
    mpd_channels: int = 32,
    msd_channels: int = 128,
    disc_max_samples: int = _DISC_MAX_SAMPLES,
) -> Tuple[nn.Module, optim.Optimizer, nn.Module, optim.Optimizer]:
    """Build gen + disc + optimizers with warmed-up states.

    Args:
        batch_size: Batch size for warmup.
        mpd_channels: MPD channels for discriminator.
        msd_channels: MSD channels for discriminator.
        disc_max_samples: Waveform length for discriminator warmup.
    """
    cfg = get_default_config()
    cfg.audio.sr = _SAMPLE_RATE
    cfg.audio.fft_size = _FFT_SIZE
    cfg.audio.hop_size = _HOP_SIZE
    cfg.audio.nb_freqs = _NUM_FREQS
    cfg.audio.n_freqs = _NUM_FREQS
    cfg.erb.nb_erb = _NB_ERB
    cfg.df.nb_df = _NB_DF

    model = init_model(config=cfg, variant="full")
    model.train()
    gen_opt = optim.Adam(learning_rate=1e-4)

    disc = CombinedDiscriminator(
        mpd_periods=(2, 3, 5, 7, 11),
        mpd_channels=mpd_channels,
        msd_scales=3,
        msd_channels=msd_channels,
    )
    disc.train()
    disc_opt = optim.Adam(learning_rate=1e-4)

    # Warm up parameters so optimizer states are initialized
    batch = _make_batch(batch_size)
    pred_wav, clean_wav = _make_waveforms(batch_size, disc_max_samples)

    # Gen forward
    gen_loss_and_grad = nn.value_and_grad(
        model,
        lambda m, nr, ni, fe, fs, cr, ci: spectral_loss(m((nr, ni), fe, fs), (cr, ci)),
    )
    loss, grads = gen_loss_and_grad(
        model,
        batch["noisy_real"],
        batch["noisy_imag"],
        batch["feat_erb"],
        batch["feat_spec"],
        batch["clean_real"],
        batch["clean_imag"],
    )
    grads, _ = clip_grad_norm_tree(grads, 1.0)
    gen_opt.update(model, grads)
    mx.eval(loss, model.parameters(), gen_opt.state)

    # Disc forward
    def _disc_loss_fn(d):
        real_out, _ = d(clean_wav, return_features=False)
        fake_out, _ = d(mx.stop_gradient(pred_wav), return_features=False)
        total, _, _ = discriminator_loss(real_out, fake_out)
        return total

    disc_loss, disc_grads = nn.value_and_grad(disc, _disc_loss_fn)(disc)
    disc_grads, _ = clip_grad_norm_tree(disc_grads, 1.0)
    disc_opt.update(disc, disc_grads)
    mx.eval(disc_loss, disc.parameters(), disc_opt.state)

    return model, gen_opt, disc, disc_opt


# ---------------------------------------------------------------------------
# Step functions: OLD pattern vs NEW pattern
# ---------------------------------------------------------------------------


def _old_pattern_step(
    model: nn.Module,
    gen_opt: optim.Optimizer,
    disc: nn.Module,
    disc_opt: optim.Optimizer,
    batch: Dict[str, mx.array],
    pred_wav: mx.array,
    clean_wav: mx.array,
    gen_loss_and_grad: Any,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """OLD pattern: 5 sync barriers per step."""
    # 1. Gen forward + backward (lazy)
    loss, grads = gen_loss_and_grad(
        model,
        batch["noisy_real"],
        batch["noisy_imag"],
        batch["feat_erb"],
        batch["feat_spec"],
        batch["clean_real"],
        batch["clean_imag"],
    )

    # BARRIER 1: finiteness check
    loss_finite = mx.all(mx.isfinite(loss))
    mx.eval(loss_finite)
    is_finite = bool(loss_finite)

    if is_finite:
        # BARRIER 2: clip + check all-finite
        grads, grad_norm = clip_grad_norm_tree(grads, max_grad_norm)
        mx.eval(grad_norm)

        # BARRIER 3: optimizer update + materialize
        gen_opt.update(model, grads)
        mx.eval(model.parameters(), gen_opt.state)

    # Disc step
    def _disc_loss_fn(d):
        real_out, _ = d(clean_wav, return_features=False)
        fake_out, _ = d(mx.stop_gradient(pred_wav), return_features=False)
        total, _, _ = discriminator_loss(real_out, fake_out)
        return total

    disc_loss, disc_grads = nn.value_and_grad(disc, _disc_loss_fn)(disc)
    disc_grads, _ = clip_grad_norm_tree(disc_grads, max_grad_norm)

    # BARRIER 4: disc loss eval
    mx.eval(disc_loss)

    # BARRIER 5: disc optimizer + materialize
    disc_opt.update(disc, disc_grads)
    mx.eval(disc.parameters(), disc_opt.state)

    return float(loss), float(disc_loss)


def _new_pattern_step(
    model: nn.Module,
    gen_opt: optim.Optimizer,
    disc: nn.Module,
    disc_opt: optim.Optimizer,
    batch: Dict[str, mx.array],
    pred_wav: mx.array,
    clean_wav: mx.array,
    gen_loss_and_grad: Any,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """NEW pattern: 1 sync barrier per step."""
    # Gen forward + backward (lazy)
    loss, grads = gen_loss_and_grad(
        model,
        batch["noisy_real"],
        batch["noisy_imag"],
        batch["feat_erb"],
        batch["feat_spec"],
        batch["clean_real"],
        batch["clean_imag"],
    )

    # Lazy finiteness — no sync
    loss_finite_arr = mx.all(mx.isfinite(loss))

    # Optimistic clip + update (clip_grad_norm zeros NaN grads)
    grads, grad_norm_arr = clip_grad_norm_tree(grads, max_grad_norm)
    gen_opt.update(model, grads)

    # Disc step (all lazy)
    def _disc_loss_fn(d):
        real_out, _ = d(clean_wav, return_features=False)
        fake_out, _ = d(mx.stop_gradient(pred_wav), return_features=False)
        total, _, _ = discriminator_loss(real_out, fake_out)
        return total

    disc_loss, disc_grads = nn.value_and_grad(disc, _disc_loss_fn)(disc)
    disc_grads, _ = clip_grad_norm_tree(disc_grads, max_grad_norm)
    disc_opt.update(disc, disc_grads)

    # SINGLE BARRIER: materialize everything at once
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

    return float(loss), float(disc_loss)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_pattern(
    pattern_name: str,
    step_fn,
    model: nn.Module,
    gen_opt: optim.Optimizer,
    disc: nn.Module,
    disc_opt: optim.Optimizer,
    gen_loss_and_grad: Any,
    batch_size: int,
    warmup_steps: int,
    measured_steps: int,
    repeats: int,
) -> SyncBenchmarkResult:
    all_latencies: List[float] = []

    for rep in range(repeats):
        # Warmup
        for _ in range(warmup_steps):
            batch = _make_batch(batch_size)
            pred_wav, clean_wav = _make_waveforms(batch_size)
            step_fn(
                model,
                gen_opt,
                disc,
                disc_opt,
                batch,
                pred_wav,
                clean_wav,
                gen_loss_and_grad,
            )

        # Measured
        for _ in range(measured_steps):
            batch = _make_batch(batch_size)
            pred_wav, clean_wav = _make_waveforms(batch_size)
            t0 = time.perf_counter()
            step_fn(
                model,
                gen_opt,
                disc,
                disc_opt,
                batch,
                pred_wav,
                clean_wav,
                gen_loss_and_grad,
            )
            t1 = time.perf_counter()
            all_latencies.append((t1 - t0) * 1000.0)

    total_secs = sum(all_latencies) / 1000.0
    return SyncBenchmarkResult(
        pattern=pattern_name,
        batch_size=batch_size,
        steps=len(all_latencies),
        warmup_steps=warmup_steps,
        repeats=repeats,
        step_mean_ms=statistics.mean(all_latencies),
        step_median_ms=statistics.median(all_latencies),
        step_p95_ms=_safe_percentile(all_latencies, 95),
        step_p99_ms=_safe_percentile(all_latencies, 99),
        step_min_ms=min(all_latencies),
        step_max_ms=max(all_latencies),
        steps_per_sec=len(all_latencies) / total_secs if total_secs > 0 else 0,
        total_seconds=total_secs,
    )


def _print_comparison(old: SyncBenchmarkResult, new: SyncBenchmarkResult) -> None:
    speedup_mean = old.step_mean_ms / new.step_mean_ms if new.step_mean_ms > 0 else 0
    speedup_p95 = old.step_p95_ms / new.step_p95_ms if new.step_p95_ms > 0 else 0
    saved_ms = old.step_mean_ms - new.step_mean_ms

    print("\n" + "=" * 80)
    print("GAN SYNC BARRIER BENCHMARK")
    print("=" * 80)
    print(f"  Batch size: {old.batch_size}")
    print(f"  Steps: {old.steps} (warmup: {old.warmup_steps}, repeats: {old.repeats})")
    print()
    print(f"  {'Metric':<20} {'OLD (5 barriers)':>18} {'NEW (1 barrier)':>18} {'Speedup':>10}")
    print(f"  {'-' * 20} {'-' * 18} {'-' * 18} {'-' * 10}")
    print(
        f"  {'Mean (ms)':<20} {old.step_mean_ms:>18.2f} {new.step_mean_ms:>18.2f} "
        f"{speedup_mean:>9.2f}x"
    )
    print(
        f"  {'Median (ms)':<20} {old.step_median_ms:>18.2f} {new.step_median_ms:>18.2f} "
        f"{old.step_median_ms / new.step_median_ms if new.step_median_ms > 0 else 0:>9.2f}x"
    )
    print(
        f"  {'P95 (ms)':<20} {old.step_p95_ms:>18.2f} {new.step_p95_ms:>18.2f} "
        f"{speedup_p95:>9.2f}x"
    )
    print(
        f"  {'P99 (ms)':<20} {old.step_p99_ms:>18.2f} {new.step_p99_ms:>18.2f} "
        f"{old.step_p99_ms / new.step_p99_ms if new.step_p99_ms > 0 else 0:>9.2f}x"
    )
    print(f"  {'Min (ms)':<20} {old.step_min_ms:>18.2f} {new.step_min_ms:>18.2f}")
    print(f"  {'Max (ms)':<20} {old.step_max_ms:>18.2f} {new.step_max_ms:>18.2f}")
    print(f"  {'Steps/s':<20} {old.steps_per_sec:>18.2f} {new.steps_per_sec:>18.2f}")
    print()
    print(f"  Sync overhead saved: {saved_ms:.2f} ms/step ({speedup_mean:.2f}x speedup)")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GAN sync-barrier patterns (old=5 barriers vs new=1)"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps per repeat")
    parser.add_argument("--steps", type=int, default=40, help="Measured steps per repeat")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats")
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--metadata",
        action="store_true",
        default=False,
        help="Include hardware metadata in JSON output",
    )
    args = parser.parse_args()

    print("Building models (gen + disc + optimizers)...")
    model, gen_opt, disc, disc_opt = _build_models(args.batch_size)

    gen_loss_and_grad = nn.value_and_grad(
        model,
        lambda m, nr, ni, fe, fs, cr, ci: spectral_loss(m((nr, ni), fe, fs), (cr, ci)),
    )

    print(
        f"Running benchmark: batch_size={args.batch_size}, "
        f"warmup={args.warmup}, steps={args.steps}, repeats={args.repeats}"
    )

    # --- Run OLD pattern ---
    print("\n--- OLD pattern (5 sync barriers) ---")
    old_result = _run_pattern(
        "old_5_barriers",
        _old_pattern_step,
        model,
        gen_opt,
        disc,
        disc_opt,
        gen_loss_and_grad,
        args.batch_size,
        args.warmup,
        args.steps,
        args.repeats,
    )
    print(f"  Mean: {old_result.step_mean_ms:.2f} ms/step")

    gc.collect()

    # --- Run NEW pattern ---
    print("\n--- NEW pattern (1 sync barrier) ---")
    new_result = _run_pattern(
        "new_1_barrier",
        _new_pattern_step,
        model,
        gen_opt,
        disc,
        disc_opt,
        gen_loss_and_grad,
        args.batch_size,
        args.warmup,
        args.steps,
        args.repeats,
    )
    print(f"  Mean: {new_result.step_mean_ms:.2f} ms/step")

    _print_comparison(old_result, new_result)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Any = {
            "results": [asdict(old_result), asdict(new_result)],
            "speedup_mean": (
                old_result.step_mean_ms / new_result.step_mean_ms
                if new_result.step_mean_ms > 0
                else 0
            ),
            "saved_ms_per_step": old_result.step_mean_ms - new_result.step_mean_ms,
        }
        if args.metadata:
            payload["metadata"] = _collect_metadata()
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON results to {out_path}")


if __name__ == "__main__":
    main()
