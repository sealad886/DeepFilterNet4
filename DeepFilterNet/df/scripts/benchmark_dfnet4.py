#!/usr/bin/env python
"""Benchmark script for DeepFilterNet4 performance evaluation.

This script evaluates DFNet4 models on standard speech enhancement datasets
(VoiceBank-DEMAND) and reports PESQ, STOI, and DNSMOS metrics.

Usage:
    python -m df.scripts.benchmark_dfnet4 --checkpoint model.ckpt --test-dir /path/to/voicebank
    python -m df.scripts.benchmark_dfnet4 --checkpoint model.ckpt --compare-dfnet3 model3.ckpt
    python -m df.scripts.benchmark_dfnet4 --checkpoint model.ckpt --rtf-only
"""

import argparse
import os
import time
from math import floor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from df.config import config
from df.evaluation_utils import evaluation_loop
from df.logger import init_logger
from df.model import ModelParams
from df.utils import get_device
from libdf import DF

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, model_type: str = "deepfilternet4"):
    """Load a model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Model type (deepfilternet4, deepfilternet3)

    Returns:
        model: Loaded model in eval mode
        df_state: DF state for feature extraction
    """
    if model_type == "deepfilternet4":
        from df.deepfilternet4 import ModelParams4, init_model

        p = ModelParams4()
    else:
        from df.deepfilternet3 import init_model

        p = ModelParams()

    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
    )

    model = init_model(df_state=df_state, run_df=True, train_mask=True)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}, using random weights")

    model = model.to(get_device())
    model.eval()

    return model, df_state


def measure_rtf(
    model,
    df_state: DF,
    duration_sec: float = 10.0,
    num_runs: int = 10,
    warmup_runs: int = 3,
) -> Dict[str, float]:
    """Measure Real-Time Factor (RTF) for the model.

    Args:
        model: Model to benchmark
        df_state: DF state
        duration_sec: Simulated audio duration in seconds
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs

    Returns:
        Dict with rtf_mean, rtf_std, latency_ms
    """
    device = get_device()
    sr = df_state.sr()
    hop_size = df_state.hop_size()
    fft_size = df_state.fft_size()
    nb_erb = df_state.nb_erb()

    # Calculate dimensions
    audio_len = int(duration_sec * sr)
    num_frames = audio_len // hop_size
    num_freqs = fft_size // 2 + 1

    # Get nb_df from model or params
    nb_df = getattr(model, "nb_df", 96)

    # Create random inputs
    spec = torch.randn(1, 1, num_frames, num_freqs, 2, device=device)
    feat_erb = torch.randn(1, 1, num_frames, nb_erb, device=device)
    feat_spec = torch.randn(1, 1, num_frames, nb_df, 2, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(spec, feat_erb, feat_spec)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif hasattr(torch, "mps") and device.type == "mps":
                torch.mps.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(spec, feat_erb, feat_spec)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif hasattr(torch, "mps") and device.type == "mps":
                torch.mps.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times)
    rtf_mean = times.mean() / duration_sec
    rtf_std = times.std() / duration_sec
    latency_ms = (times.mean() / num_frames) * 1000  # ms per frame

    return {
        "rtf_mean": rtf_mean,
        "rtf_std": rtf_std,
        "latency_ms": latency_ms,
        "process_time_sec": times.mean(),
        "audio_duration_sec": duration_sec,
    }


def count_parameters(model) -> Dict[str, int]:
    """Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dict with total_params, trainable_params
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_params_m": floor(total / 1e6),
    }


def get_voicebank_files(
    test_dir: str,
) -> Tuple[List[str], List[str]]:
    """Get clean and noisy file lists from VoiceBank-DEMAND test set.

    Args:
        test_dir: Path to test directory

    Returns:
        clean_files: List of clean audio paths
        noisy_files: List of noisy audio paths
    """
    test_path = Path(test_dir)

    # Try common VoiceBank-DEMAND directory structures
    clean_dirs = [
        test_path / "clean_testset_wav",
        test_path / "clean",
        test_path / "clean_test",
    ]
    noisy_dirs = [
        test_path / "noisy_testset_wav",
        test_path / "noisy",
        test_path / "noisy_test",
    ]

    clean_dir = None
    noisy_dir = None

    for cd, nd in zip(clean_dirs, noisy_dirs):
        if cd.exists() and nd.exists():
            clean_dir = cd
            noisy_dir = nd
            break

    if clean_dir is None or noisy_dir is None:
        raise ValueError(
            f"Could not find clean/noisy directories in {test_dir}. "
            f"Expected structure: clean_testset_wav/, noisy_testset_wav/"
        )

    clean_files = sorted(list(clean_dir.glob("*.wav")))
    noisy_files = sorted(list(noisy_dir.glob("*.wav")))

    # Match files by name
    clean_names = {f.stem: f for f in clean_files}
    matched_clean = []
    matched_noisy = []

    for noisy_file in noisy_files:
        name = noisy_file.stem
        if name in clean_names:
            matched_clean.append(str(clean_names[name]))
            matched_noisy.append(str(noisy_file))

    if len(matched_clean) == 0:
        raise ValueError(f"No matching clean/noisy file pairs found in {test_dir}")

    logger.info(f"Found {len(matched_clean)} test files")

    return matched_clean, matched_noisy


def run_evaluation(
    model,
    df_state: DF,
    clean_files: List[str],
    noisy_files: List[str],
    metrics: List[str] = ["pesq", "stoi", "composite"],
    n_workers: int = 4,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Run evaluation on a dataset.

    Args:
        model: Model to evaluate
        df_state: DF state
        clean_files: List of clean audio paths
        noisy_files: List of noisy audio paths
        metrics: List of metrics to compute
        n_workers: Number of parallel workers
        max_samples: Maximum samples to evaluate (None for all)

    Returns:
        Dict with metric results
    """
    if max_samples is not None:
        clean_files = clean_files[:max_samples]
        noisy_files = noisy_files[:max_samples]

    results = evaluation_loop(
        df_state=df_state,
        model=model,
        clean_files=clean_files,
        noisy_files=noisy_files,
        metrics=metrics,
        n_workers=n_workers,
        log_percent=25,
    )

    return results


def format_results(
    results: Dict[str, float],
    rtf: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, int]] = None,
    model_name: str = "DFNet4",
) -> str:
    """Format benchmark results as a table.

    Args:
        results: Evaluation metrics dict
        rtf: RTF benchmark dict
        params: Parameter count dict
        model_name: Name of the model

    Returns:
        Formatted string
    """
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f" Benchmark Results: {model_name}")
    lines.append(f"{'=' * 60}")

    if params:
        lines.append("\n📊 Model Parameters:")
        lines.append(f"   Total: {params['total_params']:,} ({params['total_params_m']:.2f}M)")

    if rtf:
        lines.append("\n⏱️  Real-Time Factor:")
        lines.append(f"   RTF: {rtf['rtf_mean']:.4f} ± {rtf['rtf_std']:.4f}")
        lines.append(f"   Latency: {rtf['latency_ms']:.2f} ms/frame")
        lines.append(
            f"   ({rtf['audio_duration_sec']:.1f}s audio in {rtf['process_time_sec']:.2f}s)"
        )

    lines.append("\n📈 Quality Metrics:")

    # Group and format metrics
    metric_groups = {
        "PESQ": ["pesq", "pesq_wb", "pesq_nb"],
        "STOI": ["stoi"],
        "Composite": ["csig", "cbak", "covl"],
        "SI-SDR": ["sisdr"],
        "DNSMOS": ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovl"],
    }

    for group_name, metric_keys in metric_groups.items():
        found_metrics = [(k, v) for k, v in results.items() if k.lower() in metric_keys]
        if found_metrics:
            lines.append(f"   {group_name}:")
            for k, v in found_metrics:
                lines.append(f"      {k}: {v:.4f}")

    # Any remaining metrics
    grouped_keys = set(k for keys in metric_groups.values() for k in keys)
    remaining = [(k, v) for k, v in results.items() if k.lower() not in grouped_keys]
    if remaining:
        lines.append("   Other:")
        for k, v in remaining:
            lines.append(f"      {k}: {v:.4f}")

    lines.append(f"{'=' * 60}\n")

    return "\n".join(lines)


def compare_models(
    results1: Dict[str, float],
    results2: Dict[str, float],
    name1: str = "DFNet4",
    name2: str = "DFNet3",
) -> str:
    """Generate comparison table between two models.

    Args:
        results1: First model results
        results2: Second model results
        name1: First model name
        name2: Second model name

    Returns:
        Formatted comparison string
    """
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f" Model Comparison: {name1} vs {name2}")
    lines.append(f"{'=' * 70}")
    lines.append(f"\n{'Metric':<15} | {name1:<12} | {name2:<12} | {'Diff':<10} | {'%':<8}")
    lines.append(f"{'-' * 15}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 8}")

    all_keys = set(results1.keys()) | set(results2.keys())
    for key in sorted(all_keys):
        v1 = results1.get(key, float("nan"))
        v2 = results2.get(key, float("nan"))
        diff = v1 - v2
        pct = (diff / v2 * 100) if v2 != 0 else 0

        indicator = "🔼" if diff > 0.01 else ("🔽" if diff < -0.01 else "➖")

        lines.append(
            f"{key:<15} | {v1:<12.4f} | {v2:<12.4f} | {diff:<+10.4f} | {indicator} {pct:+.1f}%"
        )

    lines.append(f"{'=' * 70}\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepFilterNet4 model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to DFNet4 checkpoint (or use random weights if not provided)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Path to VoiceBank-DEMAND test directory",
    )
    parser.add_argument(
        "--compare-dfnet3",
        type=str,
        default=None,
        help="Path to DFNet3 checkpoint for comparison",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["pesq", "stoi", "composite"],
        help="Metrics to compute (pesq, stoi, composite, sisdr, dnsmos5)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel workers for evaluation",
    )
    parser.add_argument(
        "--rtf-only",
        action="store_true",
        help="Only measure RTF (no quality metrics)",
    )
    parser.add_argument(
        "--rtf-duration",
        type=float,
        default=10.0,
        help="Audio duration for RTF measurement (seconds)",
    )
    parser.add_argument(
        "--rtf-runs",
        type=int,
        default=10,
        help="Number of RTF benchmark runs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON format)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Initialize logging
    init_logger(level="DEBUG" if args.debug else "INFO", file=None)

    # Load config
    if args.config:
        config.load(args.config, allow_reload=True)

    # Set device
    if args.device:
        os.environ["DF_DEVICE"] = args.device

    logger.info(f"Using device: {get_device()}")

    # Load DFNet4 model
    logger.info("Loading DFNet4 model...")
    model4, df_state = load_model(args.checkpoint or "", model_type="deepfilternet4")

    # Count parameters
    params4 = count_parameters(model4)
    logger.info(f"DFNet4 parameters: {params4['total_params_m']:.2f}M")

    # Measure RTF
    logger.info("Measuring Real-Time Factor...")
    rtf4 = measure_rtf(
        model4,
        df_state,
        duration_sec=args.rtf_duration,
        num_runs=args.rtf_runs,
    )
    logger.info(f"DFNet4 RTF: {rtf4['rtf_mean']:.4f}")

    results4 = {}
    results3 = None

    # Run quality evaluation if test directory provided
    if args.test_dir and not args.rtf_only:
        logger.info("Loading test dataset...")
        clean_files, noisy_files = get_voicebank_files(args.test_dir)

        logger.info("Evaluating DFNet4...")
        results4 = run_evaluation(
            model4,
            df_state,
            clean_files,
            noisy_files,
            metrics=args.metrics,
            n_workers=args.n_workers,
            max_samples=args.max_samples,
        )

        # Compare with DFNet3 if requested
        if args.compare_dfnet3:
            logger.info("Loading DFNet3 model for comparison...")
            model3, df_state3 = load_model(args.compare_dfnet3, model_type="deepfilternet3")

            count_parameters(model3)
            _ = measure_rtf(model3, df_state3, duration_sec=args.rtf_duration)

            logger.info("Evaluating DFNet3...")
            results3 = run_evaluation(
                model3,
                df_state3,
                clean_files,
                noisy_files,
                metrics=args.metrics,
                n_workers=args.n_workers,
                max_samples=args.max_samples,
            )

    # Print results
    print(format_results(results4, rtf4, params4, "DFNet4"))

    if results3 is not None:
        print(compare_models(results4, results3, "DFNet4", "DFNet3"))

    # Save results
    if args.output:
        import json

        output_data = {
            "dfnet4": {
                "metrics": results4,
                "rtf": rtf4,
                "params": params4,
            }
        }
        if results3 is not None:
            output_data["dfnet3"] = {
                "metrics": results3,
            }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Print target comparison
    logger.info("\n" + "=" * 60)
    logger.info(" Target Performance Comparison")
    logger.info("=" * 60)
    logger.info(" Metric       | Target | Achieved")
    logger.info(" -------------|--------|--------")

    targets = {
        "pesq": 3.45,
        "stoi": 0.96,
        "rtf": 0.25,
    }

    achieved = {
        "pesq": results4.get("pesq", results4.get("pesq_wb", 0)),
        "stoi": results4.get("stoi", 0),
        "rtf": rtf4["rtf_mean"],
    }

    for metric, target in targets.items():
        value = achieved.get(metric, 0)
        if metric == "rtf":
            status = "✅" if value < target else "❌"
            logger.info(f" {metric.upper():<12} | <{target:<5} | {value:.4f} {status}")
        else:
            status = "✅" if value >= target else "❌"
            logger.info(f" {metric.upper():<12} | >={target:<5} | {value:.4f} {status}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
