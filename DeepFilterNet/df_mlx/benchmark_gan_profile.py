#!/usr/bin/env python3
"""GPU/CPU/memory profiling for GAN training inner-loop phases.

Instruments each phase of a single GAN training step with:
  - wall-clock time (time.perf_counter)
  - GPU compute time (mx.eval barrier per phase)
  - memory tracking (active, peak via mx.get_active_memory / mx.get_peak_memory)
  - lazy-vs-materialized overhead breakdown

Two profiling modes:
  isolated   – mx.eval() after every phase (per-phase GPU cost)
  end-to-end – single mx.eval() at the end (real training pattern)
  both       – run both and print side-by-side

Example:
    python -m df_mlx.benchmark_gan_profile
    python -m df_mlx.benchmark_gan_profile --mode both --steps 10
    python -m df_mlx.benchmark_gan_profile --gpu-trace --steps 1
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from df_mlx.benchmark_gan_sync import (
    _DISC_MAX_SAMPLES,
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

_MB = 1024**2


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PhaseProfile:
    """Timing and memory snapshot for a single training phase."""

    name: str
    wall_time_ms: float
    eval_time_ms: float
    memory_before_mb: float
    memory_after_mb: float
    peak_memory_mb: float

    @property
    def lazy_overhead_ms(self) -> float:
        return self.wall_time_ms - self.eval_time_ms

    @property
    def memory_delta_mb(self) -> float:
        return self.memory_after_mb - self.memory_before_mb


@dataclass
class StepProfile:
    """Aggregated profile for one full GAN training step."""

    phases: List[PhaseProfile]
    total_wall_ms: float
    total_eval_ms: float
    peak_memory_mb: float

    @property
    def total_lazy_ms(self) -> float:
        return self.total_wall_ms - self.total_eval_ms

    @property
    def gpu_fraction(self) -> float:
        return self.total_eval_ms / self.total_wall_ms if self.total_wall_ms > 0 else 0.0


@dataclass
class AggregatedPhaseStats:
    """Mean / P50 / P95 for a single phase across multiple steps."""

    name: str
    wall_mean_ms: float
    wall_p50_ms: float
    wall_p95_ms: float
    eval_mean_ms: float
    eval_p50_ms: float
    eval_p95_ms: float
    lazy_mean_ms: float
    peak_mem_mean_mb: float


@dataclass
class ProfileRunResult:
    """Complete benchmark run result for one mode."""

    mode: str
    batch_size: int
    steps: int
    warmup: int
    step_profiles: List[StepProfile]
    aggregated: List[AggregatedPhaseStats]
    total_mean_ms: float
    total_p50_ms: float
    total_p95_ms: float
    peak_memory_mb: float


# ---------------------------------------------------------------------------
# Phase instrumentation
# ---------------------------------------------------------------------------


def _profile_phase(
    name: str,
    phase_fn: Callable[[], Sequence[mx.array] | mx.array | None],
    *,
    do_eval: bool = True,
) -> Tuple[PhaseProfile, Any]:
    """Run *phase_fn*, optionally mx.eval the result, and record timing/memory."""
    mx.reset_peak_memory()
    mem_before = mx.get_active_memory() / _MB

    t0 = time.perf_counter()
    result = phase_fn()
    t1 = time.perf_counter()

    eval_elapsed = 0.0
    if do_eval and result is not None:
        eval_t0 = time.perf_counter()
        if isinstance(result, (list, tuple)):
            mx.eval(*result)
        else:
            mx.eval(result)
        eval_elapsed = time.perf_counter() - eval_t0

    mem_after = mx.get_active_memory() / _MB
    peak = mx.get_peak_memory() / _MB

    wall_ms = (t1 - t0 + eval_elapsed) * 1000.0
    eval_ms = eval_elapsed * 1000.0

    profile = PhaseProfile(
        name=name,
        wall_time_ms=wall_ms,
        eval_time_ms=eval_ms,
        memory_before_mb=mem_before,
        memory_after_mb=mem_after,
        peak_memory_mb=peak,
    )
    return profile, result


# ---------------------------------------------------------------------------
# Isolated-mode step profiler
# ---------------------------------------------------------------------------


def _profile_step_isolated(
    model: nn.Module,
    gen_opt: optim.Optimizer,
    disc: nn.Module,
    disc_opt: optim.Optimizer,
    batch: Dict[str, mx.array],
    *,
    disc_max_samples: int,
    max_grad_norm: float,
    disc_grad_clip: float,
    disc_gradient_checkpoint: bool,
    gen_gradient_checkpoint: bool,
    grad_accumulation_steps: int,
) -> StepProfile:
    """Profile each phase with its own eval barrier."""
    phases: List[PhaseProfile] = []
    step_t0 = time.perf_counter()
    total_eval_ms = 0.0
    mx.reset_peak_memory()

    # Mutable holders for cross-phase data
    gen_result: Dict[str, Any] = {}

    # ---- Phase 1: gen forward + backward ----
    gen_loss_and_grad = nn.value_and_grad(
        model,
        lambda m, nr, ni, fe, fs, cr, ci: spectral_loss(
            (mx.checkpoint(m) if gen_gradient_checkpoint else m)((nr, ni), fe, fs), (cr, ci)
        ),
    )

    def _gen_fwd_bwd() -> Tuple[mx.array, ...]:
        loss, grads = gen_loss_and_grad(
            model,
            batch["noisy_real"],
            batch["noisy_imag"],
            batch["feat_erb"],
            batch["feat_spec"],
            batch["clean_real"],
            batch["clean_imag"],
        )
        gen_result["loss"] = loss
        gen_result["grads"] = grads
        return (loss,)

    p, _ = _profile_phase("gen_forward_backward", _gen_fwd_bwd)
    phases.append(p)
    total_eval_ms += p.eval_time_ms

    # ---- Phase 2: stop_gradient on cached waveforms ----
    cached_out_wav_holder: Dict[str, mx.array] = {}
    cached_clean_wav_holder: Dict[str, mx.array] = {}

    def _stop_grad() -> Tuple[mx.array, mx.array]:
        raw_out = mx.random.normal((batch["noisy_real"].shape[0], disc_max_samples))
        raw_clean = mx.random.normal((batch["noisy_real"].shape[0], disc_max_samples))
        out_sg = mx.stop_gradient(raw_out)
        clean_sg = mx.stop_gradient(raw_clean)
        cached_out_wav_holder["wav"] = out_sg
        cached_clean_wav_holder["wav"] = clean_sg
        return (out_sg, clean_sg)

    p, _ = _profile_phase("stop_gradient", _stop_grad)
    phases.append(p)
    total_eval_ms += p.eval_time_ms

    # ---- Phase 3: grad accumulation (if applicable) ----
    accumulated_grads = gen_result["grads"]

    def _grad_accum() -> Tuple[mx.array, ...]:
        nonlocal accumulated_grads
        if grad_accumulation_steps > 1:
            accumulated_grads = _accumulate_grads(accumulated_grads, gen_result["grads"])
            accumulated_grads = _scale_grads(accumulated_grads, 1.0 / grad_accumulation_steps)
        # Return a representative array so we have something to eval
        leaves = tree_flatten(accumulated_grads)
        return tuple(v for _, v in leaves[:1])

    p, _ = _profile_phase("grad_accumulation", _grad_accum)
    phases.append(p)
    total_eval_ms += p.eval_time_ms

    # ---- Phase 4: gen grad clip ----
    clipped_grads_holder: Dict[str, Any] = {}

    def _gen_clip() -> Tuple[mx.array, ...]:
        clipped, norm_arr = clip_grad_norm_tree(accumulated_grads, max_grad_norm)
        clipped_grads_holder["grads"] = clipped
        clipped_grads_holder["norm"] = norm_arr
        return (norm_arr,)

    p, _ = _profile_phase("gen_grad_clip", _gen_clip)
    phases.append(p)
    total_eval_ms += p.eval_time_ms

    # ---- Phase 5: gen optimizer update ----
    def _gen_opt_update() -> Tuple[mx.array, ...]:
        gen_opt.update(model, clipped_grads_holder["grads"])
        return (gen_result["loss"],)

    p, _ = _profile_phase("gen_optimizer_update", _gen_opt_update)
    phases.append(p)
    total_eval_ms += p.eval_time_ms

    # ---- Phase 6: gen eval (materialize params + opt state) ----
    def _gen_eval() -> Tuple[Any, ...]:
        mx.eval(gen_result["loss"], model.parameters(), gen_opt.state)
        return None  # type: ignore[return-value]

    p_gen_eval, _ = _profile_phase("gen_eval", _gen_eval, do_eval=False)
    # The wall_time IS the eval time for this phase
    p_gen_eval = PhaseProfile(
        name="gen_eval",
        wall_time_ms=p_gen_eval.wall_time_ms,
        eval_time_ms=p_gen_eval.wall_time_ms,
        memory_before_mb=p_gen_eval.memory_before_mb,
        memory_after_mb=p_gen_eval.memory_after_mb,
        peak_memory_mb=p_gen_eval.peak_memory_mb,
    )
    phases.append(p_gen_eval)
    total_eval_ms += p_gen_eval.eval_time_ms

    # ---- Phase 7: disc forward + backward ----
    disc_result: Dict[str, Any] = {}
    pred_wav_d = mx.stop_gradient(cached_out_wav_holder["wav"])
    clean_wav_d = cached_clean_wav_holder["wav"]

    def _disc_fwd_bwd() -> Tuple[mx.array, ...]:
        if disc_gradient_checkpoint:

            def _loss_fn(d: nn.Module) -> mx.array:
                fwd = mx.checkpoint(d)
                real_out, _ = fwd(clean_wav_d, return_features=False)
                fake_out, _ = fwd(pred_wav_d, return_features=False)
                total, _, _ = discriminator_loss(real_out, fake_out)
                return total

        else:

            def _loss_fn(d: nn.Module) -> mx.array:
                real_out, _ = d(clean_wav_d, return_features=False)
                fake_out, _ = d(pred_wav_d, return_features=False)
                total, _, _ = discriminator_loss(real_out, fake_out)
                return total

        d_loss, d_grads = nn.value_and_grad(disc, _loss_fn)(disc)
        disc_result["loss"] = d_loss
        disc_result["grads"] = d_grads
        return (d_loss,)

    p, _ = _profile_phase("disc_forward_backward", _disc_fwd_bwd)
    phases.append(p)
    total_eval_ms += p.eval_time_ms

    # ---- Phase 8: disc grad clip ----
    disc_clipped_holder: Dict[str, Any] = {}

    def _disc_clip() -> Tuple[mx.array, ...]:
        clipped, norm_arr = clip_grad_norm_tree(disc_result["grads"], disc_grad_clip)
        disc_clipped_holder["grads"] = clipped
        return (norm_arr,)

    p, _ = _profile_phase("disc_grad_clip", _disc_clip)
    phases.append(p)
    total_eval_ms += p.eval_time_ms

    # ---- Phase 9: disc optimizer update ----
    def _disc_opt_update() -> Tuple[mx.array, ...]:
        disc_opt.update(disc, disc_clipped_holder["grads"])
        return (disc_result["loss"],)

    p, _ = _profile_phase("disc_optimizer_update", _disc_opt_update)
    phases.append(p)
    total_eval_ms += p.eval_time_ms

    # ---- Phase 10: disc eval ----
    def _disc_eval() -> None:
        mx.eval(disc_result["loss"], disc.parameters(), disc_opt.state)
        return None

    p_disc_eval, _ = _profile_phase("disc_eval", _disc_eval, do_eval=False)
    p_disc_eval = PhaseProfile(
        name="disc_eval",
        wall_time_ms=p_disc_eval.wall_time_ms,
        eval_time_ms=p_disc_eval.wall_time_ms,
        memory_before_mb=p_disc_eval.memory_before_mb,
        memory_after_mb=p_disc_eval.memory_after_mb,
        peak_memory_mb=p_disc_eval.peak_memory_mb,
    )
    phases.append(p_disc_eval)
    total_eval_ms += p_disc_eval.eval_time_ms

    step_t1 = time.perf_counter()
    step_peak = mx.get_peak_memory() / _MB

    return StepProfile(
        phases=phases,
        total_wall_ms=(step_t1 - step_t0) * 1000.0,
        total_eval_ms=total_eval_ms,
        peak_memory_mb=step_peak,
    )


# ---------------------------------------------------------------------------
# End-to-end mode step profiler
# ---------------------------------------------------------------------------


def _profile_step_e2e(
    model: nn.Module,
    gen_opt: optim.Optimizer,
    disc: nn.Module,
    disc_opt: optim.Optimizer,
    batch: Dict[str, mx.array],
    *,
    disc_max_samples: int,
    max_grad_norm: float,
    disc_grad_clip: float,
    disc_gradient_checkpoint: bool,
    gen_gradient_checkpoint: bool,
    grad_accumulation_steps: int,
) -> StepProfile:
    """Profile all phases lazily, then one big mx.eval at the end."""
    phases: List[PhaseProfile] = []
    step_t0 = time.perf_counter()
    mx.reset_peak_memory()

    bs = batch["noisy_real"].shape[0]

    gen_loss_and_grad = nn.value_and_grad(
        model,
        lambda m, nr, ni, fe, fs, cr, ci: spectral_loss(
            (mx.checkpoint(m) if gen_gradient_checkpoint else m)((nr, ni), fe, fs), (cr, ci)
        ),
    )

    # --- Gen forward+backward (lazy) ---
    mem_before = mx.get_active_memory() / _MB
    t0 = time.perf_counter()
    loss, grads = gen_loss_and_grad(
        model,
        batch["noisy_real"],
        batch["noisy_imag"],
        batch["feat_erb"],
        batch["feat_spec"],
        batch["clean_real"],
        batch["clean_imag"],
    )
    t1 = time.perf_counter()
    phases.append(
        PhaseProfile(
            "gen_forward_backward",
            (t1 - t0) * 1000.0,
            0.0,
            mem_before,
            mx.get_active_memory() / _MB,
            mx.get_peak_memory() / _MB,
        )
    )

    # --- stop_gradient ---
    t0 = time.perf_counter()
    cached_out = mx.stop_gradient(mx.random.normal((bs, disc_max_samples)))
    cached_clean = mx.stop_gradient(mx.random.normal((bs, disc_max_samples)))
    t1 = time.perf_counter()
    phases.append(
        PhaseProfile(
            "stop_gradient",
            (t1 - t0) * 1000.0,
            0.0,
            phases[-1].memory_after_mb,
            mx.get_active_memory() / _MB,
            mx.get_peak_memory() / _MB,
        )
    )

    # --- grad accumulation ---
    t0 = time.perf_counter()
    if grad_accumulation_steps > 1:
        grads = _accumulate_grads(grads, grads)
        grads = _scale_grads(grads, 1.0 / grad_accumulation_steps)
    t1 = time.perf_counter()
    phases.append(
        PhaseProfile(
            "grad_accumulation",
            (t1 - t0) * 1000.0,
            0.0,
            phases[-1].memory_after_mb,
            mx.get_active_memory() / _MB,
            mx.get_peak_memory() / _MB,
        )
    )

    # --- gen grad clip ---
    t0 = time.perf_counter()
    grads, grad_norm_arr = clip_grad_norm_tree(grads, max_grad_norm)
    t1 = time.perf_counter()
    phases.append(
        PhaseProfile(
            "gen_grad_clip",
            (t1 - t0) * 1000.0,
            0.0,
            phases[-1].memory_after_mb,
            mx.get_active_memory() / _MB,
            mx.get_peak_memory() / _MB,
        )
    )

    # --- gen optimizer update ---
    t0 = time.perf_counter()
    gen_opt.update(model, grads)
    t1 = time.perf_counter()
    phases.append(
        PhaseProfile(
            "gen_optimizer_update",
            (t1 - t0) * 1000.0,
            0.0,
            phases[-1].memory_after_mb,
            mx.get_active_memory() / _MB,
            mx.get_peak_memory() / _MB,
        )
    )

    # --- disc forward+backward (lazy) ---
    pred_wav_d = mx.stop_gradient(cached_out)
    clean_wav_d = cached_clean

    if disc_gradient_checkpoint:

        def _disc_loss_fn(d: nn.Module) -> mx.array:
            fwd = mx.checkpoint(d)
            real_out, _ = fwd(clean_wav_d, return_features=False)
            fake_out, _ = fwd(pred_wav_d, return_features=False)
            total, _, _ = discriminator_loss(real_out, fake_out)
            return total

    else:

        def _disc_loss_fn(d: nn.Module) -> mx.array:
            real_out, _ = d(clean_wav_d, return_features=False)
            fake_out, _ = d(pred_wav_d, return_features=False)
            total, _, _ = discriminator_loss(real_out, fake_out)
            return total

    t0 = time.perf_counter()
    disc_loss, disc_grads = nn.value_and_grad(disc, _disc_loss_fn)(disc)
    t1 = time.perf_counter()
    phases.append(
        PhaseProfile(
            "disc_forward_backward",
            (t1 - t0) * 1000.0,
            0.0,
            phases[-1].memory_after_mb,
            mx.get_active_memory() / _MB,
            mx.get_peak_memory() / _MB,
        )
    )

    # --- disc grad clip ---
    t0 = time.perf_counter()
    disc_grads, _ = clip_grad_norm_tree(disc_grads, disc_grad_clip)
    t1 = time.perf_counter()
    phases.append(
        PhaseProfile(
            "disc_grad_clip",
            (t1 - t0) * 1000.0,
            0.0,
            phases[-1].memory_after_mb,
            mx.get_active_memory() / _MB,
            mx.get_peak_memory() / _MB,
        )
    )

    # --- disc optimizer update ---
    t0 = time.perf_counter()
    disc_opt.update(disc, disc_grads)
    t1 = time.perf_counter()
    phases.append(
        PhaseProfile(
            "disc_optimizer_update",
            (t1 - t0) * 1000.0,
            0.0,
            phases[-1].memory_after_mb,
            mx.get_active_memory() / _MB,
            mx.get_peak_memory() / _MB,
        )
    )

    # --- single big eval ---
    eval_t0 = time.perf_counter()
    mx.eval(
        loss,
        disc_loss,
        grad_norm_arr,
        model.parameters(),
        gen_opt.state,
        disc.parameters(),
        disc_opt.state,
    )
    eval_t1 = time.perf_counter()
    total_eval_ms = (eval_t1 - eval_t0) * 1000.0

    # Add the consolidated eval as a final phase
    phases.append(
        PhaseProfile(
            "consolidated_eval",
            total_eval_ms,
            total_eval_ms,
            phases[-1].memory_after_mb,
            mx.get_active_memory() / _MB,
            mx.get_peak_memory() / _MB,
        )
    )

    step_t1 = time.perf_counter()
    step_peak = mx.get_peak_memory() / _MB

    return StepProfile(
        phases=phases,
        total_wall_ms=(step_t1 - step_t0) * 1000.0,
        total_eval_ms=total_eval_ms,
        peak_memory_mb=step_peak,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_phase_stats(
    step_profiles: List[StepProfile],
) -> List[AggregatedPhaseStats]:
    """Compute mean/P50/P95 for each phase across multiple steps."""
    if not step_profiles:
        return []

    phase_names = [p.name for p in step_profiles[0].phases]
    aggregated: List[AggregatedPhaseStats] = []

    for idx, name in enumerate(phase_names):
        walls = [sp.phases[idx].wall_time_ms for sp in step_profiles if idx < len(sp.phases)]
        evals = [sp.phases[idx].eval_time_ms for sp in step_profiles if idx < len(sp.phases)]
        lazys = [w - e for w, e in zip(walls, evals)]
        peaks = [sp.phases[idx].peak_memory_mb for sp in step_profiles if idx < len(sp.phases)]

        aggregated.append(
            AggregatedPhaseStats(
                name=name,
                wall_mean_ms=float(np.mean(walls)),
                wall_p50_ms=float(np.median(walls)),
                wall_p95_ms=_safe_percentile(walls, 95),
                eval_mean_ms=float(np.mean(evals)),
                eval_p50_ms=float(np.median(evals)),
                eval_p95_ms=_safe_percentile(evals, 95),
                lazy_mean_ms=float(np.mean(lazys)),
                peak_mem_mean_mb=float(np.mean(peaks)),
            )
        )

    return aggregated


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def _run_profile(
    model: nn.Module,
    gen_opt: optim.Optimizer,
    disc: nn.Module,
    disc_opt: optim.Optimizer,
    *,
    mode: str,
    batch_size: int,
    steps: int,
    warmup: int,
    disc_max_samples: int,
    max_grad_norm: float,
    disc_grad_clip: float,
    disc_gradient_checkpoint: bool,
    gen_gradient_checkpoint: bool,
    grad_accumulation_steps: int,
) -> ProfileRunResult:
    """Run warmup + measured steps, return aggregated results."""
    step_fn = _profile_step_isolated if mode == "isolated" else _profile_step_e2e
    total_iters = warmup + steps
    step_profiles: List[StepProfile] = []

    for i in range(total_iters):
        gc.collect()
        batch = _make_batch(batch_size)

        sp = step_fn(
            model,
            gen_opt,
            disc,
            disc_opt,
            batch,
            disc_max_samples=disc_max_samples,
            max_grad_norm=max_grad_norm,
            disc_grad_clip=disc_grad_clip,
            disc_gradient_checkpoint=disc_gradient_checkpoint,
            gen_gradient_checkpoint=gen_gradient_checkpoint,
            grad_accumulation_steps=grad_accumulation_steps,
        )

        if i >= warmup:
            step_profiles.append(sp)

        label = "warmup" if i < warmup else f"step {i - warmup + 1}/{steps}"
        print(f"  [{mode}] {label}: wall={sp.total_wall_ms:.1f}ms eval={sp.total_eval_ms:.1f}ms")

    aggregated = _aggregate_phase_stats(step_profiles)
    totals = [sp.total_wall_ms for sp in step_profiles]
    peak = max((sp.peak_memory_mb for sp in step_profiles), default=0.0)

    return ProfileRunResult(
        mode=mode,
        batch_size=batch_size,
        steps=steps,
        warmup=warmup,
        step_profiles=step_profiles,
        aggregated=aggregated,
        total_mean_ms=float(np.mean(totals)) if totals else 0.0,
        total_p50_ms=float(np.median(totals)) if totals else 0.0,
        total_p95_ms=_safe_percentile(totals, 95),
        peak_memory_mb=peak,
    )


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


_DIVIDER = "\u2500"


def _print_phase_table(result: ProfileRunResult) -> None:
    """Print per-phase timing/memory table."""
    mode_label = "Isolated Phases — each eval'd separately"
    if result.mode == "end-to-end":
        mode_label = "End-to-End — single eval at end"

    print()
    print("=" * 100)
    print(f"GAN TRAINING STEP PROFILE ({mode_label})")
    print("=" * 100)

    hdr = (
        f"{'Phase':<26} "
        f"{'Wall(ms)':>9} "
        f"{'Eval(ms)':>9} "
        f"{'Lazy(ms)':>9} "
        f"{'MemBef(MB)':>11} "
        f"{'MemAft(MB)':>11} "
        f"{'Peak(MB)':>9}"
    )
    print(hdr)
    print(_DIVIDER * 100)

    total_wall = 0.0
    total_eval = 0.0

    for ag in result.aggregated:
        total_wall += ag.wall_mean_ms
        total_eval += ag.eval_mean_ms

        # Gather memory averages from step profiles
        idx = next(
            (i for i, a in enumerate(result.aggregated) if a.name == ag.name),
            -1,
        )
        mem_bef_vals = [
            sp.phases[idx].memory_before_mb for sp in result.step_profiles if idx < len(sp.phases)
        ]
        mem_aft_vals = [
            sp.phases[idx].memory_after_mb for sp in result.step_profiles if idx < len(sp.phases)
        ]
        peak_vals = [
            sp.phases[idx].peak_memory_mb for sp in result.step_profiles if idx < len(sp.phases)
        ]

        mem_bef = float(np.mean(mem_bef_vals)) if mem_bef_vals else 0.0
        mem_aft = float(np.mean(mem_aft_vals)) if mem_aft_vals else 0.0
        peak = float(np.mean(peak_vals)) if peak_vals else 0.0

        print(
            f"{ag.name:<26} "
            f"{ag.wall_mean_ms:>9.1f} "
            f"{ag.eval_mean_ms:>9.1f} "
            f"{ag.lazy_mean_ms:>9.1f} "
            f"{mem_bef:>11.1f} "
            f"{mem_aft:>11.1f} "
            f"{peak:>9.1f}"
        )

    print(_DIVIDER * 100)

    total_lazy = total_wall - total_eval
    gpu_frac = (total_eval / total_wall * 100) if total_wall > 0 else 0
    cpu_frac = 100.0 - gpu_frac

    print(f"{'TOTAL':<26} " f"{total_wall:>9.1f} " f"{total_eval:>9.1f} " f"{total_lazy:>9.1f}")
    print(f"GPU fraction: {gpu_frac:.1f}%    CPU/lazy fraction: {cpu_frac:.1f}%")
    print(f"Step P50: {result.total_p50_ms:.1f}ms   P95: {result.total_p95_ms:.1f}ms")
    print(f"Peak memory: {result.peak_memory_mb:.1f} MB")


def _print_breakdown_summary(result: ProfileRunResult) -> None:
    """Print percentage breakdown and optimization recommendations."""
    if not result.aggregated:
        return

    total_wall = sum(ag.wall_mean_ms for ag in result.aggregated)
    if total_wall <= 0:
        return

    print()
    print("=" * 80)
    print("PHASE BREAKDOWN")
    print("=" * 80)

    sorted_phases = sorted(result.aggregated, key=lambda a: a.wall_mean_ms, reverse=True)

    gen_total = 0.0
    disc_total = 0.0

    for ag in sorted_phases:
        pct = ag.wall_mean_ms / total_wall * 100
        bar_len = int(pct / 2)
        bar = "\u2588" * bar_len
        print(f"  {ag.name:<26} {pct:>5.1f}%  {bar}")

        if ag.name.startswith("gen_") or ag.name in ("stop_gradient", "grad_accumulation"):
            gen_total += ag.wall_mean_ms
        elif ag.name.startswith("disc_"):
            disc_total += ag.wall_mean_ms
        elif ag.name == "consolidated_eval":
            pass  # shared between gen and disc

    print()
    gen_pct = gen_total / total_wall * 100 if total_wall > 0 else 0
    disc_pct = disc_total / total_wall * 100 if total_wall > 0 else 0
    print(f"  Generator phases:      {gen_pct:>5.1f}% ({gen_total:.1f} ms)")
    print(f"  Discriminator phases:  {disc_pct:>5.1f}% ({disc_total:.1f} ms)")

    # Memory spike analysis
    print()
    print("MEMORY SPIKES")
    print("-" * 40)
    for ag in result.aggregated:
        idx = next((i for i, a in enumerate(result.aggregated) if a.name == ag.name), -1)
        deltas = [
            sp.phases[idx].memory_delta_mb for sp in result.step_profiles if idx < len(sp.phases)
        ]
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        if abs(mean_delta) > 10:
            direction = "+" if mean_delta > 0 else ""
            print(f"  {ag.name:<26} {direction}{mean_delta:.1f} MB")

    # Optimization recommendations
    print()
    print("OPTIMIZATION RECOMMENDATIONS")
    print("-" * 40)

    rec_idx = 1
    dominant = sorted_phases[0]
    dominant_pct = dominant.wall_mean_ms / total_wall * 100

    if result.mode == "isolated":
        # In isolated mode, gen_eval and disc_eval absorb lazy work from earlier phases.
        # The *real* dominant cost is gen fwd+bwd + everything eval'd in gen_eval.
        gen_compute = sum(
            ag.wall_mean_ms
            for ag in result.aggregated
            if ag.name
            in (
                "gen_forward_backward",
                "grad_accumulation",
                "gen_grad_clip",
                "gen_optimizer_update",
                "gen_eval",
            )
        )
        disc_compute = sum(
            ag.wall_mean_ms
            for ag in result.aggregated
            if ag.name
            in ("disc_forward_backward", "disc_grad_clip", "disc_optimizer_update", "disc_eval")
        )
        gen_compute_pct = gen_compute / total_wall * 100 if total_wall > 0 else 0
        disc_compute_pct = disc_compute / total_wall * 100 if total_wall > 0 else 0

        print(
            f"  {rec_idx}. Generator compute: {gen_compute_pct:.0f}% ({gen_compute:.0f}ms)"
            " — includes fwd+bwd, grad clip, opt update, and eval"
        )
        rec_idx += 1
        print(
            f"  {rec_idx}. Discriminator compute: {disc_compute_pct:.0f}% ({disc_compute:.0f}ms)"
            " — includes fwd+bwd, grad clip, opt update, and eval"
        )
        rec_idx += 1

        # Note about isolated-mode eval semantics
        print(
            f"  {rec_idx}. NOTE: In isolated mode, mx.eval() per phase forces sync."
            " This shows WHERE GPU time goes but slightly inflates total step time."
            " Use end-to-end mode (--mode both) for realistic step latency."
        )
        rec_idx += 1
    else:
        # End-to-end: consolidated_eval dominates (expected)
        print(
            f"  {rec_idx}. {dominant.name} shows {dominant_pct:.0f}% — expected for lazy eval."
            " All GPU work is deferred to the sync barrier."
        )
        rec_idx += 1

    total_lazy = sum(ag.lazy_mean_ms for ag in result.aggregated)
    lazy_pct = total_lazy / total_wall * 100 if total_wall > 0 else 0
    if lazy_pct > 25:
        print(
            f"  {rec_idx}. Lazy overhead is {lazy_pct:.0f}% — graph construction"
            " is significant, consider mx.compile for hot paths"
        )
        rec_idx += 1

    if disc_total > gen_total * 0.5 and result.mode == "isolated":
        print(
            f"  {rec_idx}. Disc is {disc_pct:.0f}% of step — reduce mpd_channels"
            "/msd_channels or increase disc_update_freq"
        )
        rec_idx += 1

    if result.peak_memory_mb > 4000:
        print(
            f"  {rec_idx}. Peak memory {result.peak_memory_mb:.0f} MB is high"
            " — enable disc_gradient_checkpoint or reduce batch_size"
        )
        rec_idx += 1

    print("=" * 80)


def _print_side_by_side(
    iso: ProfileRunResult,
    e2e: ProfileRunResult,
) -> None:
    """Print isolated vs end-to-end results side by side."""
    print()
    print("=" * 110)
    print("SIDE-BY-SIDE: ISOLATED vs END-TO-END")
    print("=" * 110)

    hdr = (
        f"{'Phase':<26} "
        f"{'Iso Wall':>9} "
        f"{'Iso Eval':>9} "
        f"{'E2E Wall':>9} "
        f"{'E2E Eval':>9} "
        f"{'Delta':>8}"
    )
    print(hdr)
    print(_DIVIDER * 110)

    iso_by_name = {a.name: a for a in iso.aggregated}
    e2e_by_name = {a.name: a for a in e2e.aggregated}
    all_names = list(
        dict.fromkeys([a.name for a in iso.aggregated] + [a.name for a in e2e.aggregated])
    )

    for name in all_names:
        ia = iso_by_name.get(name)
        ea = e2e_by_name.get(name)
        iw = ia.wall_mean_ms if ia else 0.0
        ie = ia.eval_mean_ms if ia else 0.0
        ew = ea.wall_mean_ms if ea else 0.0
        ee = ea.eval_mean_ms if ea else 0.0
        delta = ew - iw

        print(
            f"{name:<26} "
            f"{iw:>9.1f} "
            f"{ie:>9.1f} "
            f"{ew:>9.1f} "
            f"{ee:>9.1f} "
            f"{delta:>+8.1f}"
        )

    print(_DIVIDER * 110)
    iso_total = iso.total_mean_ms
    e2e_total = e2e.total_mean_ms
    speedup = iso_total / e2e_total if e2e_total > 0 else 0
    print(
        f"{'TOTAL':<26} "
        f"{iso_total:>9.1f} "
        f"{iso.total_mean_ms - iso.total_p50_ms + iso.total_p50_ms:>9.1f} "
        f"{e2e_total:>9.1f} "
        f"{'':>9} "
        f"{e2e_total - iso_total:>+8.1f}"
    )
    print(f"End-to-end speedup: {speedup:.2f}x (lazy eval amortization)")
    print("=" * 110)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def _results_to_json(
    results: List[ProfileRunResult],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Serialize results to a JSON-safe dict."""

    def _serialize_result(r: ProfileRunResult) -> Dict[str, Any]:
        return {
            "mode": r.mode,
            "batch_size": r.batch_size,
            "steps": r.steps,
            "warmup": r.warmup,
            "total_mean_ms": round(r.total_mean_ms, 2),
            "total_p50_ms": round(r.total_p50_ms, 2),
            "total_p95_ms": round(r.total_p95_ms, 2),
            "peak_memory_mb": round(r.peak_memory_mb, 1),
            "phases": [asdict(a) for a in r.aggregated],
            "per_step": [
                {
                    "total_wall_ms": round(sp.total_wall_ms, 2),
                    "total_eval_ms": round(sp.total_eval_ms, 2),
                    "peak_memory_mb": round(sp.peak_memory_mb, 1),
                    "phases": [
                        {
                            "name": p.name,
                            "wall_time_ms": round(p.wall_time_ms, 2),
                            "eval_time_ms": round(p.eval_time_ms, 2),
                            "memory_before_mb": round(p.memory_before_mb, 1),
                            "memory_after_mb": round(p.memory_after_mb, 1),
                            "peak_memory_mb": round(p.peak_memory_mb, 1),
                        }
                        for p in sp.phases
                    ],
                }
                for sp in r.step_profiles
            ],
        }

    return {
        "metadata": metadata,
        "results": [_serialize_result(r) for r in results],
    }


# ---------------------------------------------------------------------------
# GPU trace support
# ---------------------------------------------------------------------------


def _try_start_gpu_trace() -> bool:
    """Start Metal GPU trace capture if available."""
    try:
        mx.metal.start_capture("gan_profile_trace")
        return True
    except Exception as exc:
        print(f"  Warning: GPU trace capture failed: {exc}")
        print("  Set MTL_CAPTURE_ENABLED=1 and re-run to enable GPU traces.")
        return False


def _try_stop_gpu_trace() -> None:
    try:
        mx.metal.stop_capture()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Profile GAN training step phases (timing + memory).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Profiled steps (default: 10)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup steps (default: 5)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size (default: 12)",
    )
    p.add_argument(
        "--disc-max-samples",
        type=int,
        default=_DISC_MAX_SAMPLES,
        help=f"Disc waveform length (default: {_DISC_MAX_SAMPLES})",
    )
    p.add_argument(
        "--disc-gradient-checkpoint",
        action="store_true",
        default=True,
        help="Enable disc gradient checkpointing (default: True)",
    )
    p.add_argument(
        "--no-disc-gradient-checkpoint",
        action="store_false",
        dest="disc_gradient_checkpoint",
        help="Disable disc gradient checkpointing",
    )
    p.add_argument(
        "--gen-gradient-checkpoint",
        action="store_true",
        default=False,
        help="Enable gen gradient checkpointing (saves ~7 GB, ~20%% slower gen)",
    )
    p.add_argument(
        "--no-gen-gradient-checkpoint",
        action="store_false",
        dest="gen_gradient_checkpoint",
        help="Disable gen gradient checkpointing",
    )
    p.add_argument(
        "--mpd-channels",
        type=int,
        default=16,
        help="MPD channels (default: 16)",
    )
    p.add_argument(
        "--msd-channels",
        type=int,
        default=64,
        help="MSD channels (default: 64)",
    )
    p.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (default: 1.0)",
    )
    p.add_argument(
        "--disc-grad-clip",
        type=float,
        default=1.0,
        help="Disc gradient clip norm (default: 1.0)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="logs/gan_profile.json",
        help="JSON output path (default: logs/gan_profile.json)",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["isolated", "end-to-end", "both"],
        default="isolated",
        help="Profiling mode (default: isolated)",
    )
    p.add_argument(
        "--gpu-trace",
        action="store_true",
        help="Capture Metal GPU trace (requires MTL_CAPTURE_ENABLED=1)",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    print("Building models (gen + disc + optimizers)...")
    model, gen_opt, disc, disc_opt = _build_models(
        batch_size=args.batch_size,
        mpd_channels=args.mpd_channels,
        msd_channels=args.msd_channels,
        disc_max_samples=args.disc_max_samples,
    )

    metadata = _collect_metadata()
    repro = collect_reproducibility_metadata()
    metadata["reproducibility"] = repro
    metadata["config"] = {
        "batch_size": args.batch_size,
        "disc_max_samples": args.disc_max_samples,
        "disc_gradient_checkpoint": args.disc_gradient_checkpoint,
        "gen_gradient_checkpoint": args.gen_gradient_checkpoint,
        "mpd_channels": args.mpd_channels,
        "msd_channels": args.msd_channels,
        "grad_accumulation_steps": args.grad_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "disc_grad_clip": args.disc_grad_clip,
    }

    print(
        f"Config: batch_size={args.batch_size}, "
        f"disc_max_samples={args.disc_max_samples}, "
        f"mpd={args.mpd_channels}, msd={args.msd_channels}, "
        f"disc_ckpt={args.disc_gradient_checkpoint}, "
        f"gen_ckpt={args.gen_gradient_checkpoint}, "
        f"ga={args.grad_accumulation_steps}"
    )
    print(f"Warmup: {args.warmup}  Steps: {args.steps}  Mode: {args.mode}")

    gpu_trace_active = False
    if args.gpu_trace:
        gpu_trace_active = _try_start_gpu_trace()

    run_kwargs = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "disc_max_samples": args.disc_max_samples,
        "max_grad_norm": args.max_grad_norm,
        "disc_grad_clip": args.disc_grad_clip,
        "disc_gradient_checkpoint": args.disc_gradient_checkpoint,
        "gen_gradient_checkpoint": args.gen_gradient_checkpoint,
        "grad_accumulation_steps": args.grad_accumulation_steps,
    }

    results: List[ProfileRunResult] = []

    modes_to_run = ["isolated", "end-to-end"] if args.mode == "both" else [args.mode]

    for mode in modes_to_run:
        print(f"\n--- Running {mode} profile ---")
        result = _run_profile(
            model,
            gen_opt,
            disc,
            disc_opt,
            mode=mode,
            **run_kwargs,
        )
        results.append(result)
        _print_phase_table(result)
        _print_breakdown_summary(result)

    if gpu_trace_active:
        _try_stop_gpu_trace()
        print("\nGPU trace saved (check ~/Library/Developer/ or Instruments).")

    if args.mode == "both" and len(results) == 2:
        _print_side_by_side(results[0], results[1])

    # Write JSON output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _results_to_json(results, metadata)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote JSON results to {out_path}")


if __name__ == "__main__":
    main()
