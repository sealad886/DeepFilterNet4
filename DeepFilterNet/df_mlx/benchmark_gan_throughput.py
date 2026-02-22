#!/usr/bin/env python3
"""Benchmark GAN training step throughput.

Sweeps key config parameters (batch_size, grad_accumulation, eval_frequency,
disc_update_freq, disc_max_samples, disc_gradient_checkpoint,
cache_gen_waveforms, mpd/msd channels) to find optimal settings for the
run_pipeline_awesome_gan_silero_single_oom_safe.toml config.

Uses synthetic data — no dataset required.  Measures the actual training
step time including gen forward+backward, disc update, and sync barriers,
mirroring the real eager GAN inner loop from train_dynamic.py.

Example:
    python -m df_mlx.benchmark_gan_throughput
    python -m df_mlx.benchmark_gan_throughput --full
    python -m df_mlx.benchmark_gan_throughput --batch-sizes 8,12 --eval-freqs 4,8
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from df_mlx.benchmark_gan_sync import (
    _build_models,
    _collect_metadata,
    _make_batch,
    _safe_percentile,
)
from df_mlx.benchmark_train_step import collect_reproducibility_metadata
from df_mlx.grad_utils import clip_grad_norm_tree
from df_mlx.loss import discriminator_loss
from df_mlx.train import spectral_loss
from df_mlx.training_ops import accumulate_grads as _accumulate_grads
from df_mlx.training_ops import scale_grads as _scale_grads

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_EFFECTIVE_BATCH = 64


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ThroughputResult:
    """Single benchmark case result."""

    batch_size: int
    grad_accumulation_steps: int
    eval_frequency: int
    disc_update_freq: int
    disc_max_samples: int
    disc_gradient_checkpoint: bool
    gen_gradient_checkpoint: bool
    cache_gen_waveforms: bool
    mpd_channels: int
    msd_channels: int
    steps: int
    warmup: int
    step_mean_ms: float
    step_p50_ms: float
    step_p95_ms: float
    effective_samples_per_sec: float
    peak_memory_gb: float


# ---------------------------------------------------------------------------
# Inner loop — mirrors train_dynamic.py eager GAN path
# ---------------------------------------------------------------------------


def _run_benchmark_case(
    model: nn.Module,
    gen_opt: optim.Optimizer,
    disc: nn.Module,
    disc_opt: optim.Optimizer,
    *,
    batch_size: int,
    grad_accumulation_steps: int,
    eval_frequency: int,
    disc_update_freq: int,
    disc_max_samples: int,
    disc_gradient_checkpoint: bool,
    gen_gradient_checkpoint: bool,
    cache_gen_waveforms: bool,
    warmup: int,
    measured_steps: int,
) -> ThroughputResult:
    """Run one benchmark case and return the result."""
    max_grad_norm = 1.0
    disc_grad_clip = 1.0

    if gen_gradient_checkpoint:
        gen_loss_and_grad = nn.value_and_grad(
            model,
            lambda m, nr, ni, fe, fs, cr, ci: spectral_loss(
                mx.checkpoint(m)((nr, ni), fe, fs), (cr, ci)
            ),
        )
    else:
        gen_loss_and_grad = nn.value_and_grad(
            model,
            lambda m, nr, ni, fe, fs, cr, ci: spectral_loss(m((nr, ni), fe, fs), (cr, ci)),
        )

    total_microbatches = warmup + measured_steps
    latencies: List[float] = []

    accumulated_grads: Dict[str, Any] | None = None
    micro_count = 0
    global_step = 0

    mx.reset_peak_memory()

    for mb_idx in range(total_microbatches):
        batch = _make_batch(batch_size)

        is_measured = mb_idx >= warmup
        t0 = time.perf_counter() if is_measured else 0.0

        # --- Gen forward + backward ---
        loss, grads = gen_loss_and_grad(
            model,
            batch["noisy_real"],
            batch["noisy_imag"],
            batch["feat_erb"],
            batch["feat_spec"],
            batch["clean_real"],
            batch["clean_imag"],
        )

        # Simulate cached waveforms (gen output → disc input)
        if cache_gen_waveforms:
            cached_out_wav = mx.random.normal((batch_size, disc_max_samples))
            cached_clean_wav = mx.random.normal((batch_size, disc_max_samples))
            cached_out_wav = mx.stop_gradient(cached_out_wav)
            cached_clean_wav = mx.stop_gradient(cached_clean_wav)
        else:
            cached_out_wav = None
            cached_clean_wav = None

        loss_finite_arr = mx.all(mx.isfinite(loss))

        # Accumulate grads
        accumulated_grads = _accumulate_grads(accumulated_grads, grads)
        micro_count += 1

        should_sync = (mb_idx + 1) % eval_frequency == 0

        disc_loss: mx.array | None = None

        if micro_count >= grad_accumulation_steps:

            final_grads = _scale_grads(
                accumulated_grads,  # type: ignore[arg-type]
                1.0 / grad_accumulation_steps,
            )
            if max_grad_norm > 0:
                final_grads, grad_norm_arr = clip_grad_norm_tree(final_grads, max_grad_norm)
            gen_opt.update(model, final_grads)

            accumulated_grads = None
            micro_count = 0
            global_step += 1

            # --- Disc update ---
            do_disc = global_step % disc_update_freq == 0
            if do_disc:
                if cache_gen_waveforms and cached_out_wav is not None:
                    pred_wav = cached_out_wav
                    clean_wav = cached_clean_wav
                else:
                    pred_wav = mx.random.normal((batch_size, disc_max_samples))
                    clean_wav = mx.random.normal((batch_size, disc_max_samples))

                pred_wav_d = mx.stop_gradient(pred_wav)
                clean_wav_d = clean_wav

                if disc_gradient_checkpoint:

                    def _disc_loss_ckpt(
                        d: nn.Module,
                    ) -> mx.array:
                        fwd = mx.checkpoint(d)
                        real_out, _ = fwd(
                            clean_wav_d,
                            return_features=False,
                        )
                        fake_out, _ = fwd(
                            pred_wav_d,
                            return_features=False,
                        )
                        total, _, _ = discriminator_loss(real_out, fake_out)
                        return total

                    disc_loss, disc_grads = nn.value_and_grad(disc, _disc_loss_ckpt)(disc)
                else:

                    def _disc_loss_fn(
                        d: nn.Module,
                    ) -> mx.array:
                        real_out, _ = d(
                            clean_wav_d,
                            return_features=False,
                        )
                        fake_out, _ = d(
                            pred_wav_d,
                            return_features=False,
                        )
                        total, _, _ = discriminator_loss(real_out, fake_out)
                        return total

                    disc_loss, disc_grads = nn.value_and_grad(disc, _disc_loss_fn)(disc)

                if disc_grad_clip > 0:
                    disc_grads, _ = clip_grad_norm_tree(disc_grads, disc_grad_clip)
                disc_opt.update(disc, disc_grads)

        # --- Sync barrier ---
        if should_sync:
            eval_targets: list[Any] = [
                loss,
                loss_finite_arr,
                model.parameters(),
                gen_opt.state,
            ]
            if disc_loss is not None:
                eval_targets.extend(
                    [
                        disc_loss,
                        disc.parameters(),
                        disc_opt.state,
                    ]
                )
            mx.eval(*eval_targets)

        if is_measured:
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    peak_mem_gb = mx.get_peak_memory() / (1024**3)

    step_mean = float(np.mean(latencies)) if latencies else float("nan")
    step_p50 = float(np.median(latencies)) if latencies else float("nan")
    step_p95 = _safe_percentile(latencies, 95)

    total_elapsed_s = sum(latencies) / 1000.0
    eff_samples = batch_size * len(latencies)
    eff_sps = eff_samples / total_elapsed_s if total_elapsed_s > 0 else 0.0

    return ThroughputResult(
        batch_size=batch_size,
        grad_accumulation_steps=grad_accumulation_steps,
        eval_frequency=eval_frequency,
        disc_update_freq=disc_update_freq,
        disc_max_samples=disc_max_samples,
        disc_gradient_checkpoint=disc_gradient_checkpoint,
        gen_gradient_checkpoint=gen_gradient_checkpoint,
        cache_gen_waveforms=cache_gen_waveforms,
        mpd_channels=-1,  # filled by caller
        msd_channels=-1,
        steps=measured_steps,
        warmup=warmup,
        step_mean_ms=round(step_mean, 2),
        step_p50_ms=round(step_p50, 2),
        step_p95_ms=round(step_p95, 2),
        effective_samples_per_sec=round(eff_sps, 2),
        peak_memory_gb=round(peak_mem_gb, 3),
    )


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """Defines the parameter space for a throughput sweep."""

    batch_sizes: List[int]
    grad_accum_steps: List[int]
    eval_frequencies: List[int]
    disc_update_freqs: List[int]
    disc_max_samples_list: List[int]
    disc_gradient_checkpoints: List[bool]
    gen_gradient_checkpoints: List[bool]
    cache_gen_waveforms_list: List[bool]
    mpd_channels_list: List[int]
    msd_channels_list: List[int]


def _quick_config() -> SweepConfig:
    return SweepConfig(
        batch_sizes=[24],
        grad_accum_steps=[2],
        eval_frequencies=[6],
        disc_update_freqs=[2],
        disc_max_samples_list=[24000],
        disc_gradient_checkpoints=[True],
        gen_gradient_checkpoints=[True],
        cache_gen_waveforms_list=[True],
        mpd_channels_list=[16],
        msd_channels_list=[64],
    )


def _full_config() -> SweepConfig:
    return SweepConfig(
        batch_sizes=[8, 12, 16],
        grad_accum_steps=[1, 2, 4],
        eval_frequencies=[2, 4, 8, 16],
        disc_update_freqs=[1, 2, 3],
        disc_max_samples_list=[24000, 48000, 96000],
        disc_gradient_checkpoints=[True, False],
        gen_gradient_checkpoints=[True, False],
        cache_gen_waveforms_list=[True, False],
        mpd_channels_list=[16, 32],
        msd_channels_list=[64, 128],
    )


def _enumerate_cases(
    cfg: SweepConfig,
) -> List[Dict[str, Any]]:
    """Enumerate all valid sweep cases, skipping impractical combos."""
    cases: List[Dict[str, Any]] = []
    for (
        bs,
        ga,
        ef,
        duf,
        dms,
        dgc,
        ggc,
        cgw,
        mpd,
        msd,
    ) in product(
        cfg.batch_sizes,
        cfg.grad_accum_steps,
        cfg.eval_frequencies,
        cfg.disc_update_freqs,
        cfg.disc_max_samples_list,
        cfg.disc_gradient_checkpoints,
        cfg.gen_gradient_checkpoints,
        cfg.cache_gen_waveforms_list,
        cfg.mpd_channels_list,
        cfg.msd_channels_list,
    ):
        if bs * ga > _MAX_EFFECTIVE_BATCH:
            continue
        cases.append(
            {
                "batch_size": bs,
                "grad_accumulation_steps": ga,
                "eval_frequency": ef,
                "disc_update_freq": duf,
                "disc_max_samples": dms,
                "disc_gradient_checkpoint": dgc,
                "gen_gradient_checkpoint": ggc,
                "cache_gen_waveforms": cgw,
                "mpd_channels": mpd,
                "msd_channels": msd,
            }
        )
    return cases


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def _print_results_table(
    results: List[ThroughputResult],
    top_n: int = 3,
) -> None:
    """Print sorted results table highlighting top configs."""
    sorted_results = sorted(
        results,
        key=lambda r: r.effective_samples_per_sec,
        reverse=True,
    )

    hdr = (
        f"{'#':>3}  "
        f"{'BS':>3} "
        f"{'GA':>3} "
        f"{'EF':>3} "
        f"{'DUF':>3} "
        f"{'DMS':>6} "
        f"{'DGC':>4} "
        f"{'GGC':>4} "
        f"{'CGW':>4} "
        f"{'MPD':>4} "
        f"{'MSD':>4} "
        f"{'Mean':>8} "
        f"{'P50':>8} "
        f"{'P95':>8} "
        f"{'Samp/s':>9} "
        f"{'PkMem':>7}"
    )

    print("\n" + "=" * len(hdr))
    print("GAN THROUGHPUT BENCHMARK RESULTS")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for i, r in enumerate(sorted_results):
        marker = " ***" if i < top_n else ""
        dgc_str = "Y" if r.disc_gradient_checkpoint else "N"
        ggc_str = "Y" if r.gen_gradient_checkpoint else "N"
        cgw_str = "Y" if r.cache_gen_waveforms else "N"
        dms_k = r.disc_max_samples // 1000
        print(
            f"{i + 1:>3}  "
            f"{r.batch_size:>3} "
            f"{r.grad_accumulation_steps:>3} "
            f"{r.eval_frequency:>3} "
            f"{r.disc_update_freq:>3} "
            f"{dms_k:>5}k "
            f"{dgc_str:>4} "
            f"{ggc_str:>4} "
            f"{cgw_str:>4} "
            f"{r.mpd_channels:>4} "
            f"{r.msd_channels:>4} "
            f"{r.step_mean_ms:>7.1f} "
            f"{r.step_p50_ms:>7.1f} "
            f"{r.step_p95_ms:>7.1f} "
            f"{r.effective_samples_per_sec:>9.1f} "
            f"{r.peak_memory_gb:>6.2f}"
            f"{marker}"
        )

    print("-" * len(hdr))
    print(
        "Legend: BS=batch_size, GA=grad_accum, EF=eval_freq, "
        "DUF=disc_update_freq, DMS=disc_max_samples, "
        "DGC=disc_gradient_checkpoint, GGC=gen_gradient_checkpoint, CGW=cache_gen_wavs, "
        "MPD=mpd_ch, MSD=msd_ch"
    )
    print("*** = top-3 configs by effective samples/sec")
    print("=" * len(hdr))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Benchmark GAN training step throughput " "by sweeping config parameters."),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Quick sweep (default): limited parameter space.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Full sweep: complete parameter matrix.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Measured microbatch steps (default: 40).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup microbatch steps (default: 10).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="logs/gan_throughput_benchmark.jsonl",
        help="JSONL output path.",
    )
    p.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes override.",
    )
    p.add_argument(
        "--eval-freqs",
        type=str,
        default=None,
        help="Comma-separated eval_frequency override.",
    )
    p.add_argument(
        "--disc-max-samples",
        type=str,
        default=None,
        help="Comma-separated disc_max_samples override.",
    )
    p.add_argument(
        "--disc-update-freqs",
        type=str,
        default=None,
        help="Comma-separated disc_update_freq override.",
    )
    p.add_argument(
        "--grad-accum",
        type=str,
        default=None,
        help="Comma-separated grad_accumulation_steps override.",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the GAN throughput benchmark."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.full:
        sweep = _full_config()
    else:
        sweep = _quick_config()

    # Apply CLI overrides
    if args.batch_sizes:
        sweep.batch_sizes = _parse_int_list(args.batch_sizes)
    if args.eval_freqs:
        sweep.eval_frequencies = _parse_int_list(args.eval_freqs)
    if args.disc_max_samples:
        sweep.disc_max_samples_list = _parse_int_list(args.disc_max_samples)
    if args.disc_update_freqs:
        sweep.disc_update_freqs = _parse_int_list(args.disc_update_freqs)
    if args.grad_accum:
        sweep.grad_accum_steps = _parse_int_list(args.grad_accum)

    cases = _enumerate_cases(sweep)
    total = len(cases)
    print(f"GAN throughput benchmark: {total} cases, " f"warmup={args.warmup}, steps={args.steps}")

    results: List[ThroughputResult] = []
    jsonl_path = Path(args.output)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Group cases by (mpd_channels, msd_channels, batch_size) to reuse
    # models where only loop params change.
    cases_sorted = sorted(
        cases,
        key=lambda c: (
            c["mpd_channels"],
            c["msd_channels"],
            c["batch_size"],
        ),
    )

    current_model_key: Tuple[int, int, int] | None = None
    model: nn.Module | None = None
    gen_opt: optim.Optimizer | None = None
    disc: nn.Module | None = None
    disc_opt: optim.Optimizer | None = None

    metadata = _collect_metadata()
    repro = collect_reproducibility_metadata()

    for idx, case in enumerate(cases_sorted):
        model_key = (
            case["mpd_channels"],
            case["msd_channels"],
            case["batch_size"],
        )

        if model_key != current_model_key:
            # Clean up old models
            if model is not None:
                del model, gen_opt, disc, disc_opt
                gc.collect()
                mx.reset_peak_memory()

            print(
                f"\n  Building models: mpd_ch={model_key[0]}, "
                f"msd_ch={model_key[1]}, bs={model_key[2]}..."
            )
            model, gen_opt, disc, disc_opt = _build_models(
                batch_size=model_key[2],
                mpd_channels=model_key[0],
                msd_channels=model_key[1],
            )
            current_model_key = model_key
        else:
            gc.collect()
            mx.reset_peak_memory()

        assert model is not None
        assert gen_opt is not None
        assert disc is not None
        assert disc_opt is not None

        label = (
            f"[{idx + 1}/{total}] bs={case['batch_size']} "
            f"ga={case['grad_accumulation_steps']} "
            f"ef={case['eval_frequency']} "
            f"duf={case['disc_update_freq']} "
            f"dms={case['disc_max_samples']} "
            f"dgc={case['disc_gradient_checkpoint']} "
            f"ggc={case['gen_gradient_checkpoint']} "
            f"cgw={case['cache_gen_waveforms']} "
            f"mpd={case['mpd_channels']} "
            f"msd={case['msd_channels']}"
        )
        print(f"  {label} ... ", end="", flush=True)

        try:
            result = _run_benchmark_case(
                model,
                gen_opt,
                disc,
                disc_opt,
                batch_size=case["batch_size"],
                grad_accumulation_steps=case["grad_accumulation_steps"],
                eval_frequency=case["eval_frequency"],
                disc_update_freq=case["disc_update_freq"],
                disc_max_samples=case["disc_max_samples"],
                disc_gradient_checkpoint=case["disc_gradient_checkpoint"],
                gen_gradient_checkpoint=case["gen_gradient_checkpoint"],
                cache_gen_waveforms=case["cache_gen_waveforms"],
                warmup=args.warmup,
                measured_steps=args.steps,
            )
            result.mpd_channels = case["mpd_channels"]
            result.msd_channels = case["msd_channels"]
            results.append(result)

            print(
                f"mean={result.step_mean_ms:.1f}ms "
                f"samp/s={result.effective_samples_per_sec:.1f} "
                f"mem={result.peak_memory_gb:.2f}GB"
            )

            # Append JSONL
            record: Dict[str, Any] = asdict(result)
            record["metadata"] = metadata
            record["reproducibility"] = repro
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

    if not results:
        print("\nNo successful benchmark cases.")
        sys.exit(1)

    _print_results_table(results)
    print(f"\nResults written to {jsonl_path}")


if __name__ == "__main__":
    main()
