#!/usr/bin/env python3
"""Evaluate quality metrics (PESQ, SI-SDR, STOI) for experiment checkpoints.

Loads each trained model from the MLX training checkpoint format
(no config.ini — builds model from default config, loads .safetensors weights),
enhances held-out noisy test pairs from the MLX datastore, and computes
PESQ (wb), SI-SDR, and STOI.

Usage:
    python evaluate_experiments.py                       # evaluate all found
    python evaluate_experiments.py --experiments E0 E2   # evaluate specific
    python evaluate_experiments.py --n-samples 50        # more test samples
    python evaluate_experiments.py --output results.json # custom output path
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from df_mlx.checkpoint import load_model as load_checkpoint_weights  # noqa: E402
from df_mlx.config import get_default_config  # noqa: E402
from df_mlx.enhance import enhance  # noqa: E402
from df_mlx.evaluation import PESQMetric, SiSDRMetric, STOIMetric  # noqa: E402
from df_mlx.model import init_model  # noqa: E402

SR = 48000
CACHE_DIR = "/Users/andrew/DataDump/datasets/mlx_datastore"
EXPERIMENT_CONFIGS = {
    "E0": {
        "name": "E0-Baseline (Full GAN)",
        "checkpoint_dir": "/Users/andrew/DataDump/checkpoints/gan_from_40",
    },
    "E1": {
        "name": "E1-MPD-Only",
        "checkpoint_dir": "/Users/andrew/DataDump/checkpoints/e1_mpd_only_reduced",
    },
    "E2": {
        "name": "E2-NoGAN-MRSTFT",
        "checkpoint_dir": "/Users/andrew/DataDump/checkpoints/e2_mrstft_enhanced_no_gan",
    },
    "E3": {
        "name": "E3-Frozen-Disc-FM",
        "checkpoint_dir": "/Users/andrew/DataDump/checkpoints/e3_frozen_disc_fm",
    },
}


def load_test_pairs(cache_dir: str, n_samples: int, seed: int = 42) -> list:
    """Load clean/noisy pair samples from the MLX datastore for evaluation.

    Constructs noisy mixtures from clean speech + noise at random SNRs,
    matching the training pipeline's mixing approach.
    """
    rng = np.random.default_rng(seed)
    config_path = Path(cache_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    segment_samples = int(config["segment_length"] * SR)
    snr_lo, snr_hi = config["snr_range"]

    speech_dir = Path(cache_dir) / "speech"
    noise_dir = Path(cache_dir) / "noise"

    speech_shards = sorted(speech_dir.glob("*.npz"))
    noise_shards = sorted(noise_dir.glob("*.npz"))

    if not speech_shards or not noise_shards:
        raise FileNotFoundError(f"No shards found in {cache_dir}")

    pairs = []
    attempts = 0
    max_attempts = n_samples * 10

    while len(pairs) < n_samples and attempts < max_attempts:
        attempts += 1
        sp_shard = rng.choice(speech_shards)
        ns_shard = rng.choice(noise_shards)

        try:
            sp_data = np.load(sp_shard, allow_pickle=True)
            ns_data = np.load(ns_shard, allow_pickle=True)
        except Exception:
            continue

        sp_keys = [k for k in sp_data.files if sp_data[k].dtype == np.float32]
        ns_keys = [k for k in ns_data.files if ns_data[k].dtype == np.float32]
        if not sp_keys or not ns_keys:
            continue

        sp_audio = sp_data[rng.choice(sp_keys)].astype(np.float32)
        ns_audio = ns_data[rng.choice(ns_keys)].astype(np.float32)

        if len(sp_audio) < segment_samples or len(ns_audio) < segment_samples:
            continue

        start_sp = rng.integers(0, len(sp_audio) - segment_samples + 1)
        start_ns = rng.integers(0, len(ns_audio) - segment_samples + 1)
        clean = sp_audio[start_sp : start_sp + segment_samples]
        noise = ns_audio[start_ns : start_ns + segment_samples]

        snr_db = rng.uniform(snr_lo, snr_hi)
        clean_rms = np.sqrt(np.mean(clean**2) + 1e-10)
        noise_rms = np.sqrt(np.mean(noise**2) + 1e-10)
        target_noise_rms = clean_rms / (10 ** (snr_db / 20))
        noise_scaled = noise * (target_noise_rms / noise_rms)
        noisy = clean + noise_scaled

        peak = max(np.abs(noisy).max(), 1e-7)
        if peak > 0.99:
            noisy = noisy * (0.99 / peak)
            clean = clean * (0.99 / peak)

        pairs.append(
            {
                "clean": clean,
                "noisy": noisy,
                "snr_db": float(snr_db),
                "name": f"sample_{len(pairs):03d}",
            }
        )

    if len(pairs) < n_samples:
        print(f"Warning: only created {len(pairs)}/{n_samples} test pairs")

    return pairs


def _find_best_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, int]:
    """Find best or latest checkpoint in the training checkpoint directory.

    MLX training saves checkpoints as:
      best.safetensors, final.safetensors, epoch_NNN.safetensors,
      interrupted_epoch_NNN.safetensors

    Returns (path, epoch) or (None, 0) if not found.
    """
    # Prefer best.safetensors
    best = checkpoint_dir / "best.safetensors"
    if best.exists():
        # Try to extract epoch from associated state
        state_file = checkpoint_dir / "best.state.json"
        epoch = 0
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                epoch = state.get("epoch", state.get("last_completed_epoch", 0))
            except (json.JSONDecodeError, KeyError):
                pass
        return best, epoch

    # Try final.safetensors
    final = checkpoint_dir / "final.safetensors"
    if final.exists():
        state_file = checkpoint_dir / "final.state.json"
        epoch = 0
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                epoch = state.get("epoch", state.get("last_completed_epoch", 0))
            except (json.JSONDecodeError, KeyError):
                pass
        return final, epoch

    # Try epoch_NNN.safetensors (highest number)
    epoch_files = sorted(checkpoint_dir.glob("epoch_*.safetensors"))
    # Filter out state files
    epoch_files = [f for f in epoch_files if ".state" not in f.name and ".complete" not in f.name]
    if epoch_files:
        latest = epoch_files[-1]
        # Extract epoch number
        try:
            epoch = int(latest.stem.split("_")[1])
        except (IndexError, ValueError):
            epoch = 0
        return latest, epoch

    # Try interrupted checkpoints
    interrupted = sorted(checkpoint_dir.glob("interrupted_epoch_*.safetensors"))
    interrupted = [f for f in interrupted if ".state" not in f.name]
    if interrupted:
        latest = interrupted[-1]
        try:
            epoch = int(latest.stem.split("_")[-1])
        except (IndexError, ValueError):
            epoch = 0
        return latest, epoch

    return None, 0


def _build_model_and_params(cache_dir: str):
    """Build a DfNet4 model from default config, synced with dataset params.

    This matches exactly what the MLX training pipeline does:
    get_default_config() → sync with dataset → init_model.
    """
    from df_mlx.training_setup import _sync_model_config_with_dataset

    # Load dataset config
    config_path = Path(cache_dir) / "config.json"
    with open(config_path) as f:
        ds_config = json.load(f)

    # Build model config (same as training)
    model_cfg = get_default_config()

    # Sync with dataset params using a simple namespace
    class _DsCfg:
        pass

    ds = _DsCfg()
    ds.sample_rate = ds_config["sample_rate"]
    ds.fft_size = ds_config["fft_size"]
    ds.hop_size = ds_config["hop_size"]
    ds.nb_erb = ds_config["nb_erb"]
    ds.nb_df = ds_config["nb_df"]
    _sync_model_config_with_dataset(model_cfg, ds)

    model = init_model(config=model_cfg)
    return model, model_cfg


def evaluate_checkpoint(
    checkpoint_dir: str,
    test_pairs: list,
    cache_dir: str,
    epoch: str = "best",
) -> dict:
    """Load a model from checkpoint_dir and evaluate on test_pairs."""
    cp_path = Path(checkpoint_dir)
    if not cp_path.exists():
        return {"error": f"Checkpoint dir not found: {checkpoint_dir}"}

    # Find best checkpoint file
    ckpt_path, loaded_epoch = _find_best_checkpoint(cp_path)
    if ckpt_path is None:
        return {"error": f"No checkpoint files found in {checkpoint_dir}"}

    # Build model from default config (same arch as training)
    model, model_cfg = _build_model_and_params(cache_dir)

    # Load trained weights
    load_checkpoint_weights(model, ckpt_path)
    model.eval()
    print(f"  Loaded {ckpt_path.name} (epoch {loaded_epoch}) from {cp_path.name}")

    pesq_metric = PESQMetric(sr=SR)
    sisdr_metric = SiSDRMetric(source_sr=SR)
    stoi_metric = STOIMetric(sr=SR)

    per_sample = []
    for pair in test_pairs:
        noisy_mx = mx.array(pair["noisy"])
        enhanced_mx = enhance(model, noisy_mx, model_cfg, compensate_delay=True)
        mx.eval(enhanced_mx)
        enhanced_np = np.array(enhanced_mx).squeeze()
        clean_np = pair["clean"]
        noisy_np = pair["noisy"]

        min_len = min(len(clean_np), len(enhanced_np))
        clean_np = clean_np[:min_len]
        enhanced_np = enhanced_np[:min_len]
        noisy_np = noisy_np[:min_len]

        pesq_r = pesq_metric.add(clean_np, enhanced_np, noisy_np, pair["name"])
        sisdr_r = sisdr_metric.add(clean_np, enhanced_np, noisy_np, pair["name"])
        stoi_r = stoi_metric.add(clean_np, enhanced_np, noisy_np, pair["name"])

        per_sample.append(
            {
                "name": pair["name"],
                "snr_db": pair["snr_db"],
                "pesq_enh": pesq_r.enhanced,
                "pesq_noisy": pesq_r.noisy,
                "sisdr_enh": sisdr_r.enhanced,
                "sisdr_noisy": sisdr_r.noisy,
                "stoi_enh": stoi_r.enhanced,
                "stoi_noisy": stoi_r.noisy,
            }
        )

    return {
        "epoch": loaded_epoch,
        "pesq_mean": pesq_metric.mean(),
        "sisdr_mean": sisdr_metric.mean(),
        "stoi_mean": stoi_metric.mean(),
        "pesq_noisy_mean": float(np.mean([s["pesq_noisy"] for s in per_sample if s["pesq_noisy"] is not None])),
        "sisdr_noisy_mean": float(np.mean([s["sisdr_noisy"] for s in per_sample if s["sisdr_noisy"] is not None])),
        "stoi_noisy_mean": float(np.mean([s["stoi_noisy"] for s in per_sample if s["stoi_noisy"] is not None])),
        "n_samples": len(per_sample),
        "per_sample": per_sample,
    }


def print_comparison_table(results: dict):
    """Print a comparison table of all experiment results."""
    print("\n" + "=" * 90)
    print("QUALITY EVALUATION RESULTS")
    print("=" * 90)

    header = f"{'Experiment':<25} {'Epoch':>6} {'PESQ':>8} {'SI-SDR':>8} {'STOI':>8} {'N':>4}"
    print(header)
    print("-" * 90)

    e0_pesq = None
    for exp_id, data in sorted(results.items()):
        if "error" in data:
            print(f"{exp_id:<25} {'ERROR':>6}  {data['error']}")
            continue
        name = EXPERIMENT_CONFIGS.get(exp_id, {}).get("name", exp_id)
        pesq_val = data["pesq_mean"]
        sisdr_val = data["sisdr_mean"]
        stoi_val = data["stoi_mean"]
        epoch = data["epoch"]
        n = data["n_samples"]
        print(f"{name:<25} {epoch:>6} {pesq_val:>8.3f} {sisdr_val:>8.2f} {stoi_val:>8.4f} {n:>4}")
        if exp_id == "E0":
            e0_pesq = pesq_val

    print("-" * 90)

    # Print noisy baseline
    any_result = next((d for d in results.values() if "error" not in d), None)
    if any_result:
        print(
            f"{'(Noisy input)':<25} {'':>6} "
            f"{any_result['pesq_noisy_mean']:>8.3f} "
            f"{any_result['sisdr_noisy_mean']:>8.2f} "
            f"{any_result['stoi_noisy_mean']:>8.4f}"
        )

    if e0_pesq and e0_pesq > 0:
        print("\n--- Relative to E0 baseline ---")
        for exp_id, data in sorted(results.items()):
            if "error" in data or exp_id == "E0":
                continue
            name = EXPERIMENT_CONFIGS.get(exp_id, {}).get("name", exp_id)
            pct = data["pesq_mean"] / e0_pesq * 100
            delta_sisdr = data["sisdr_mean"] - results.get("E0", {}).get("sisdr_mean", 0)
            print(f"  {name}: PESQ={pct:.1f}% of E0, SI-SDR delta={delta_sisdr:+.2f} dB")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment quality metrics")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENT_CONFIGS.keys()),
        help="Experiment IDs to evaluate (default: all)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Number of test samples to evaluate (default: 30)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/Users/andrew/DataDump/datasets/mlx_datastore",
        help="Path to MLX datastore cache",
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default="best",
        help="Checkpoint epoch to load (best, latest, or integer)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("Loading test pairs from datastore...")
    test_pairs = load_test_pairs(args.cache_dir, args.n_samples, seed=args.seed)
    print(
        f"Created {len(test_pairs)} test pairs (SNR range: "
        f"{min(p['snr_db'] for p in test_pairs):.1f} to "
        f"{max(p['snr_db'] for p in test_pairs):.1f} dB)"
    )

    results = {}
    for exp_id in args.experiments:
        if exp_id not in EXPERIMENT_CONFIGS:
            print(f"Unknown experiment: {exp_id}, skipping")
            continue

        cfg = EXPERIMENT_CONFIGS[exp_id]
        print(f"\nEvaluating {cfg['name']}...")
        t0 = time.time()
        results[exp_id] = evaluate_checkpoint(
            cfg["checkpoint_dir"],
            test_pairs,
            cache_dir=args.cache_dir,
            epoch=args.epoch,
        )
        elapsed = time.time() - t0
        if "error" not in results[exp_id]:
            print(
                f"  Done in {elapsed:.1f}s "
                f"(PESQ={results[exp_id]['pesq_mean']:.3f}, "
                f"SI-SDR={results[exp_id]['sisdr_mean']:.2f}dB, "
                f"STOI={results[exp_id]['stoi_mean']:.4f})"
            )
        else:
            print(f"  {results[exp_id]['error']}")

    print_comparison_table(results)

    output_path = args.output or "logs/quality_evaluation_results.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    serializable = {}
    for k, v in results.items():
        if "error" in v:
            serializable[k] = v
        else:
            sv = dict(v)
            sv.pop("per_sample", None)
            serializable[k] = sv
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
