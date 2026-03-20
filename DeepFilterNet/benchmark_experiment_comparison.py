#!/usr/bin/env python3
"""Benchmark throughput & memory for experiment configs E0-E3.

Profiles training-step cost for each GAN experiment configuration using
synthetic data.  Reports per-phase timings, total step time, and estimated
throughput so experiments can be compared before committing to long runs.

Experiment matrix:
  E0 — Baseline: full combined disc (MPD 5 + MSD 3)
  E1 — MPD-only reduced (3 periods, no MSD)
  E2 — No GAN, enhanced MRSTFT (5 resolutions)
  E3 — Frozen disc (same as E0 but disc frozen → no backward/optimizer)

Usage:
    cd DeepFilterNet && python benchmark_experiment_comparison.py [--steps 20] [--warmup 5] [--batch 4] [--seq-len 200] [--output results.json]
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent))

from df_mlx.config import get_default_config  # noqa: E402
from df_mlx.discriminator import (  # noqa: E402
    CombinedDiscriminator,
    MultiPeriodDiscriminator,
)
from df_mlx.loss import (  # noqa: E402
    FeatureMatchingLoss,
    FusedSpectralLoss,
    discriminator_loss,
    generator_loss,
)
from df_mlx.model import DfNet4  # noqa: E402
from df_mlx.ops import istft  # noqa: E402


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    name: str
    description: str
    gan_enabled: bool
    disc_type: str  # "combined" | "mpd" | "none"
    mpd_periods: tuple
    mpd_channels: int
    msd_scales: int
    msd_channels: int
    disc_frozen: bool  # simulate post-freeze phase
    mrstft_fft_sizes: tuple
    mrstft_factor: float


EXPERIMENTS = {
    "E0": ExperimentConfig(
        name="E0-Baseline",
        description="Full combined disc (MPD×5 + MSD×3, channels 16/64)",
        gan_enabled=True,
        disc_type="combined",
        mpd_periods=(2, 3, 5, 7, 11),
        mpd_channels=16,
        msd_scales=3,
        msd_channels=64,
        disc_frozen=False,
        mrstft_fft_sizes=(512, 1024, 2048),
        mrstft_factor=0.2,
    ),
    "E1": ExperimentConfig(
        name="E1-MPD-Only",
        description="MPD-only 3 periods (no MSD), channels 16",
        gan_enabled=True,
        disc_type="mpd",
        mpd_periods=(2, 3, 5),
        mpd_channels=16,
        msd_scales=0,
        msd_channels=0,
        disc_frozen=False,
        mrstft_fft_sizes=(512, 1024, 2048),
        mrstft_factor=0.2,
    ),
    "E2": ExperimentConfig(
        name="E2-NoGAN-MRSTFT",
        description="No disc, enhanced MRSTFT (5 FFT sizes, factor 0.5)",
        gan_enabled=False,
        disc_type="none",
        mpd_periods=(),
        mpd_channels=0,
        msd_scales=0,
        msd_channels=0,
        disc_frozen=False,
        mrstft_fft_sizes=(256, 512, 1024, 2048, 4096),
        mrstft_factor=0.5,
    ),
    "E3-pre": ExperimentConfig(
        name="E3-Pre-Freeze",
        description="Full combined disc BEFORE freeze (same as E0)",
        gan_enabled=True,
        disc_type="combined",
        mpd_periods=(2, 3, 5, 7, 11),
        mpd_channels=16,
        msd_scales=3,
        msd_channels=64,
        disc_frozen=False,
        mrstft_fft_sizes=(512, 1024, 2048),
        mrstft_factor=0.2,
    ),
    "E3-post": ExperimentConfig(
        name="E3-Post-Freeze",
        description="Full combined disc AFTER freeze (FM only, no disc backward)",
        gan_enabled=True,
        disc_type="combined",
        mpd_periods=(2, 3, 5, 7, 11),
        mpd_channels=16,
        msd_scales=3,
        msd_channels=64,
        disc_frozen=True,
        mrstft_fft_sizes=(512, 1024, 2048),
        mrstft_factor=0.2,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_batch(batch_size: int, seq_len: int, n_freq: int = 481, nb_erb: int = 32, nb_df: int = 96):
    """Synthetic training batch."""
    return {
        "noisy_real": mx.random.normal((batch_size, seq_len, n_freq)),
        "noisy_imag": mx.random.normal((batch_size, seq_len, n_freq)),
        "clean_real": mx.random.normal((batch_size, seq_len, n_freq)),
        "clean_imag": mx.random.normal((batch_size, seq_len, n_freq)),
        "feat_erb": mx.random.normal((batch_size, seq_len, nb_erb)),
        "feat_spec": mx.random.normal((batch_size, seq_len, nb_df, 2)),
    }


def count_params(module: nn.Module) -> int:
    """Count parameters in an nn.Module."""
    return sum(p.size for p in module.parameters().values() if isinstance(p, mx.array))


def count_params_tree(tree) -> int:
    """Count all mx.array leaves in a nested param tree."""
    total = 0
    if isinstance(tree, mx.array):
        return tree.size
    if isinstance(tree, dict):
        for v in tree.values():
            total += count_params_tree(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            total += count_params_tree(v)
    return total


def make_disc(cfg: ExperimentConfig):
    """Build discriminator for an experiment config.  Returns None for E2."""
    if not cfg.gan_enabled or cfg.disc_type == "none":
        return None
    if cfg.disc_type == "combined":
        return CombinedDiscriminator(
            mpd_periods=cfg.mpd_periods,
            mpd_channels=cfg.mpd_channels,
            msd_scales=cfg.msd_scales,
            msd_channels=cfg.msd_channels,
        )
    if cfg.disc_type == "mpd":
        return MultiPeriodDiscriminator(
            periods=cfg.mpd_periods,
            channels=cfg.mpd_channels,
        )
    raise ValueError(f"Unknown disc_type: {cfg.disc_type}")


def _flat_eval(*args):
    """mx.eval that handles nested structures."""
    flat = []
    for a in args:
        if isinstance(a, mx.array):
            flat.append(a)
        elif isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, mx.array):
                    flat.append(item)
                elif isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, mx.array):
                            flat.append(sub)
    if flat:
        mx.eval(*flat)


# ---------------------------------------------------------------------------
# Per-phase profiling
# ---------------------------------------------------------------------------
def profile_phase(name: str, fn, n_iters: int, warmup: int) -> dict:
    """Run fn for warmup+n_iters, return timing stats."""
    for _ in range(warmup):
        result = fn()
        _flat_eval(result) if not isinstance(result, mx.array) else mx.eval(result)

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = fn()
        _flat_eval(result) if not isinstance(result, mx.array) else mx.eval(result)
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    p95 = sorted(times)[int(0.95 * len(times))] if len(times) >= 2 else avg
    return {"mean_ms": round(avg, 2), "std_ms": round(std, 2), "p95_ms": round(p95, 2)}


def benchmark_experiment(
    exp_key: str,
    cfg: ExperimentConfig,
    batch_size: int,
    seq_len: int,
    n_steps: int,
    warmup: int,
) -> dict:
    """Run full benchmark for one experiment, return results dict."""
    print(f"\n{'=' * 70}")
    print(f"  {cfg.name}: {cfg.description}")
    print(f"  batch={batch_size}  seq_len={seq_len}  steps={n_steps}  warmup={warmup}")
    print(f"{'=' * 70}")

    # --- Build model ---
    model_cfg = get_default_config()
    model = DfNet4(model_cfg)
    model.train()
    mx.eval(model.parameters())
    gen_params = count_params_tree(model.parameters())
    gen_optimizer = optim.AdamW(learning_rate=1e-4)

    # --- Build discriminator ---
    disc = make_disc(cfg)
    disc_params = 0
    disc_optimizer: Optional[optim.AdamW] = None
    if disc is not None:
        mx.eval(disc.parameters())
        disc_params = count_params_tree(disc.parameters())
        if cfg.disc_frozen:
            disc.freeze()
        else:
            disc_optimizer = optim.AdamW(learning_rate=1e-4)

    # --- Build MRSTFT loss ---
    mrstft_loss = FusedSpectralLoss(
        fft_sizes=cfg.mrstft_fft_sizes,
        factor=cfg.mrstft_factor,
    )
    fm_loss = FeatureMatchingLoss(factor=2.0)

    print(f"  Gen params:  {gen_params:>12,}")
    print(f"  Disc params: {disc_params:>12,} {'(FROZEN)' if cfg.disc_frozen else ''}")
    print(f"  MRSTFT:      {len(cfg.mrstft_fft_sizes)} resolutions, factor={cfg.mrstft_factor}")

    # --- Create synthetic batch (reused) ---
    batch = create_batch(batch_size, seq_len)
    mx.eval(list(batch.values()))

    n_fft = 960
    hop = 480
    target_wav_len = seq_len * hop

    # Shorthand
    nr, ni = batch["noisy_real"], batch["noisy_imag"]
    cr, ci = batch["clean_real"], batch["clean_imag"]
    erb, spec = batch["feat_erb"], batch["feat_spec"]

    # -----------------------------------------------------------------------
    # Phase 1: Generator forward + spectral loss + MRSTFT
    # -----------------------------------------------------------------------
    def gen_forward():
        out = model((nr, ni), erb, spec, training=True)
        out_r, out_i = out if isinstance(out, tuple) and len(out) == 2 else out[0]
        # Spectral L1 (simplified stand-in)
        spec_loss = mx.mean(mx.abs(out_r - cr) + mx.abs(out_i - ci))
        # iSTFT → waveform for MRSTFT
        out_wav = istft((out_r, out_i), n_fft=n_fft, hop_length=hop, length=target_wav_len)
        clean_wav = istft((cr, ci), n_fft=n_fft, hop_length=hop, length=target_wav_len)
        mrstft = mrstft_loss(out_wav, clean_wav)
        total = spec_loss + mrstft
        return total, out_wav, clean_wav

    print("\n--- Phase timings ---")
    stats_gen_fwd = profile_phase("Gen forward+loss", gen_forward, n_steps, warmup)
    print(f"  {'Gen forward+loss':30s}: {stats_gen_fwd['mean_ms']:8.1f}ms (±{stats_gen_fwd['std_ms']:.1f})")

    # -----------------------------------------------------------------------
    # Phase 2: Generator forward+backward (value_and_grad)
    # -----------------------------------------------------------------------
    def gen_loss_fn(model_):
        out = model_((nr, ni), erb, spec, training=True)
        out_r, out_i = out if isinstance(out, tuple) and len(out) == 2 else out[0]
        spec_loss = mx.mean(mx.abs(out_r - cr) + mx.abs(out_i - ci))
        out_wav = istft((out_r, out_i), n_fft=n_fft, hop_length=hop, length=target_wav_len)
        clean_wav = istft((cr, ci), n_fft=n_fft, hop_length=hop, length=target_wav_len)
        mrstft = mrstft_loss(out_wav, clean_wav)

        total = spec_loss + mrstft

        # GAN gen loss (disc in forward-only mode)
        if cfg.gan_enabled and disc is not None:
            disc_out_fake, fake_fmaps = disc(out_wav, return_features=True)
            gen_loss = generator_loss(disc_out_fake)
            adv_w = 0.0 if cfg.disc_frozen else 0.06
            total = total + adv_w * gen_loss

            # Feature matching — disc on clean is detached from gen graph
            _, real_fmaps = disc(mx.stop_gradient(clean_wav), return_features=True)
            fm = fm_loss(real_fmaps, fake_fmaps)
            total = total + 1.0 * fm

        return total

    gen_value_and_grad = nn.value_and_grad(model, gen_loss_fn)

    def gen_train_step():
        loss, grads = gen_value_and_grad(model)
        gen_optimizer.update(model, grads)
        return loss

    stats_gen_step = profile_phase("Gen train step", gen_train_step, n_steps, warmup)
    print(f"  {'Gen train step':30s}: {stats_gen_step['mean_ms']:8.1f}ms (±{stats_gen_step['std_ms']:.1f})")

    # -----------------------------------------------------------------------
    # Phase 3: Discriminator train step (if not frozen and GAN enabled)
    # -----------------------------------------------------------------------
    stats_disc_step = {"mean_ms": 0.0, "std_ms": 0.0, "p95_ms": 0.0}
    if cfg.gan_enabled and disc is not None and not cfg.disc_frozen and disc_optimizer is not None:
        # Pre-generate detached waveforms
        model.eval()
        out = model((nr, ni), erb, spec, training=False)
        out_r, out_i = out if isinstance(out, tuple) and len(out) == 2 else out[0]
        fake_wav = mx.stop_gradient(istft((out_r, out_i), n_fft=n_fft, hop_length=hop, length=target_wav_len))
        real_wav = mx.stop_gradient(istft((cr, ci), n_fft=n_fft, hop_length=hop, length=target_wav_len))
        model.train()
        mx.eval(fake_wav, real_wav)

        def disc_loss_fn(disc_):
            real_out, _ = disc_(real_wav, return_features=False)
            fake_out, _ = disc_(mx.stop_gradient(fake_wav), return_features=False)
            total, _, _ = discriminator_loss(real_out, fake_out)
            return total

        disc_vag = nn.value_and_grad(disc, disc_loss_fn)

        def disc_train_step():
            loss, grads = disc_vag(disc)
            disc_optimizer.update(disc, grads)
            return loss

        stats_disc_step = profile_phase("Disc train step", disc_train_step, n_steps, warmup)
        print(f"  {'Disc train step':30s}: {stats_disc_step['mean_ms']:8.1f}ms (±{stats_disc_step['std_ms']:.1f})")
    else:
        reason = "frozen" if cfg.disc_frozen else ("no GAN" if not cfg.gan_enabled else "N/A")
        print(f"  {'Disc train step':30s}: SKIPPED ({reason})")

    # -----------------------------------------------------------------------
    # Phase 4: Full step (gen + disc combined)
    # -----------------------------------------------------------------------
    total_step_ms = stats_gen_step["mean_ms"] + stats_disc_step["mean_ms"]
    throughput_steps_per_sec = 1000.0 / total_step_ms if total_step_ms > 0 else float("inf")
    audio_sec_per_step = (seq_len * hop) / 48000.0
    rtf = audio_sec_per_step * throughput_steps_per_sec  # real-time factor (× realtime)

    print(f"\n--- Summary for {cfg.name} ---")
    print(f"  Total step time: {total_step_ms:8.1f}ms")
    print(f"  Throughput:      {throughput_steps_per_sec:8.2f} steps/sec")
    print(f"  RTF:             {rtf:8.2f}× realtime  ({audio_sec_per_step:.2f}s audio/step)")
    if cfg.gan_enabled and disc is not None and not cfg.disc_frozen:
        disc_frac = stats_disc_step["mean_ms"] / total_step_ms * 100
        print(f"  Disc overhead:   {disc_frac:8.1f}% of step time")

    return {
        "experiment": exp_key,
        "name": cfg.name,
        "description": cfg.description,
        "gen_params": gen_params,
        "disc_params": disc_params,
        "disc_frozen": cfg.disc_frozen,
        "mrstft_resolutions": len(cfg.mrstft_fft_sizes),
        "gen_forward_ms": stats_gen_fwd,
        "gen_step_ms": stats_gen_step,
        "disc_step_ms": stats_disc_step,
        "total_step_ms": round(total_step_ms, 2),
        "throughput_steps_sec": round(throughput_steps_per_sec, 3),
        "rtf": round(rtf, 3),
        "audio_sec_per_step": round(audio_sec_per_step, 3),
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(results: list[dict]):
    """Pretty-print comparison table."""
    print(f"\n{'=' * 90}")
    print("  EXPERIMENT COMPARISON")
    print(f"{'=' * 90}")

    headers = ["Experiment", "Gen (ms)", "Disc (ms)", "Total (ms)", "Steps/s", "RTF", "Disc Params"]
    widths = [20, 12, 12, 12, 10, 10, 14]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(f"  {header_line}")
    print(f"  {'-' * sum(widths) + '-' * (len(widths) - 1) * 2}")

    baseline_total = results[0]["total_step_ms"] if results else 1.0

    for r in results:
        speedup = baseline_total / r["total_step_ms"] if r["total_step_ms"] > 0 else float("inf")
        disc_p = f"{r['disc_params']:,}" if r["disc_params"] > 0 else "—"
        if r["disc_frozen"]:
            disc_p += " ❄️"
        row = [
            r["name"],
            f"{r['gen_step_ms']['mean_ms']:.1f}",
            f"{r['disc_step_ms']['mean_ms']:.1f}" if r["disc_step_ms"]["mean_ms"] > 0 else "—",
            f"{r['total_step_ms']:.1f}",
            f"{r['throughput_steps_sec']:.2f}",
            f"{r['rtf']:.2f}×",
            disc_p,
        ]
        row_line = "  ".join(str(v).ljust(w) for v, w in zip(row, widths))
        suffix = f"  ({speedup:.2f}× E0)" if r["experiment"] != "E0" else "  (baseline)"
        print(f"  {row_line}{suffix}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark E0-E3 experiment configs")
    parser.add_argument("--steps", type=int, default=20, help="Measured steps per phase")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps per phase")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=200, help="Sequence length (frames, ~2s at hop=480)")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Subset of experiments to run (e.g. E0 E1). Default: all",
    )
    args = parser.parse_args()

    exp_keys = args.experiments or list(EXPERIMENTS.keys())
    for k in exp_keys:
        if k not in EXPERIMENTS:
            print(f"ERROR: Unknown experiment '{k}'. Available: {list(EXPERIMENTS.keys())}")
            sys.exit(1)

    print("=" * 70)
    print("  GAN EXPERIMENT THROUGHPUT BENCHMARK")
    print(f"  batch={args.batch}  seq_len={args.seq_len}  steps={args.steps}  warmup={args.warmup}")
    print(f"  experiments: {exp_keys}")
    print("=" * 70)

    results = []
    for key in exp_keys:
        cfg = EXPERIMENTS[key]
        r = benchmark_experiment(key, cfg, args.batch, args.seq_len, args.steps, args.warmup)
        results.append(r)
        # Free memory between experiments
        mx.clear_cache()

    print_comparison(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "benchmark": "experiment_comparison",
                    "batch_size": args.batch,
                    "seq_len": args.seq_len,
                    "steps": args.steps,
                    "warmup": args.warmup,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
