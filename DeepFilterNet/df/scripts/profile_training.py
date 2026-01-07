#!/usr/bin/env python
"""Profile DeepFilterNet training to measure MPS GPU utilization.

This script profiles training iterations to identify:
- MPS vs CPU time distribution
- Per-operation timing breakdown
- Memory transfers between CPU and GPU
- Potential bottlenecks (data loading, model ops, loss computation)

Usage:
    # Profile a few iterations
    python -m df.scripts.profile_training \
        datasets/dataset.cfg data/hdf5 output/profile \
        --num-iterations 10

    # Profile with chrome trace output for visualization
    python -m df.scripts.profile_training \
        datasets/dataset.cfg data/hdf5 output/profile \
        --num-iterations 20 --chrome-trace profile_trace.json

    # Profile specific components
    python -m df.scripts.profile_training \
        datasets/dataset.cfg data/hdf5 output/profile \
        --profile-dataloader --profile-forward --profile-backward
"""

import argparse
import os
import time
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

# MPS profiler for Apple Silicon OS Signpost traces
try:
    from torch.mps import profiler as mps_profiler

    HAS_MPS_PROFILER = True
except ImportError:
    HAS_MPS_PROFILER = False

from df.checkpoint import load_model
from df.config import config
from df.logger import init_logger
from df.loss import Istft, Loss
from df.modules import get_device
from df.utils import as_real, get_norm_alpha
from libdf import DF
from libdfdata import PytorchDataLoader as DataLoader


def measure_device_utilization(device: torch.device) -> Dict[str, float]:
    """Measure current device memory and utilization.

    Returns dict with memory stats for the device.
    """
    stats = {}

    if device.type == "mps":
        # MPS memory stats
        stats["mps_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
        stats["mps_driver_allocated_gb"] = torch.mps.driver_allocated_memory() / 1e9
    elif device.type == "cuda":
        stats["cuda_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
        stats["cuda_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
        stats["cuda_max_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

    return stats


def sync_device(device: torch.device):
    """Synchronize device operations for accurate timing."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def profile_dataloader_step(loader, split: str, seed: int) -> Dict[str, Any]:
    """Profile a single dataloader step."""
    start = time.perf_counter()
    iterator = loader.iter_epoch(split, seed)
    batch = next(iterator)
    elapsed = time.perf_counter() - start

    return {
        "dataloader_time_ms": elapsed * 1000,
        "batch_shape": str(batch.feat_erb.shape),
    }


def profile_forward_step(
    model: torch.nn.Module,
    batch,
    device: torch.device,
) -> Dict[str, float]:
    """Profile forward pass timing."""
    timings = {}

    # Transfer to device
    start = time.perf_counter()
    feat_erb = batch.feat_erb.to(device, non_blocking=True)
    feat_spec = as_real(batch.feat_spec.to(device, non_blocking=True))
    noisy = batch.noisy.to(device, non_blocking=True)
    sync_device(device)
    timings["transfer_time_ms"] = (time.perf_counter() - start) * 1000

    # Forward pass
    start = time.perf_counter()
    with torch.no_grad():
        enh, m, lsnr, other = model.forward(
            spec=as_real(noisy),
            feat_erb=feat_erb,
            feat_spec=feat_spec,
        )
    sync_device(device)
    timings["forward_time_ms"] = (time.perf_counter() - start) * 1000

    return timings


def profile_full_iteration(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    losses: Loss,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """Profile a complete training iteration."""
    timings = {}

    # Transfer to device
    start = time.perf_counter()
    feat_erb = batch.feat_erb.to(device, non_blocking=True)
    feat_spec = as_real(batch.feat_spec.to(device, non_blocking=True))
    noisy = batch.noisy.to(device, non_blocking=True)
    clean = batch.speech.to(device, non_blocking=True)
    snrs = batch.snr.to(device, non_blocking=True)
    sync_device(device)
    timings["transfer_time_ms"] = (time.perf_counter() - start) * 1000

    # Forward pass
    optimizer.zero_grad()
    start = time.perf_counter()
    enh, m, lsnr, other = model.forward(
        spec=as_real(noisy),
        feat_erb=feat_erb,
        feat_spec=feat_spec,
    )
    sync_device(device)
    timings["forward_time_ms"] = (time.perf_counter() - start) * 1000

    # Loss computation
    start = time.perf_counter()
    err = losses.forward(clean, noisy, enh, m, lsnr, snrs=snrs)
    sync_device(device)
    timings["loss_time_ms"] = (time.perf_counter() - start) * 1000

    # Backward pass
    start = time.perf_counter()
    err.backward()
    sync_device(device)
    timings["backward_time_ms"] = (time.perf_counter() - start) * 1000

    # Optimizer step
    start = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    sync_device(device)
    timings["optimizer_time_ms"] = (time.perf_counter() - start) * 1000

    timings["total_time_ms"] = sum(timings.values())

    return timings


def run_torch_profiler(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    losses: Loss,
    optimizer: torch.optim.Optimizer,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
    output_dir: str = ".",
    chrome_trace_path: Optional[str] = None,
) -> Dict[str, float]:
    """Run torch.profiler for detailed operation breakdown."""

    # Determine activities based on device
    activities = [torch.profiler.ProfilerActivity.CPU]
    # Note: MPS doesn't have dedicated ProfilerActivity, but CPU profiling
    # still captures Python-side timing which is useful

    schedule = torch.profiler.schedule(
        wait=1,
        warmup=warmup_iterations,
        active=num_iterations,
        repeat=1,
    )

    trace_handler = None
    if chrome_trace_path:
        trace_handler = torch.profiler.tensorboard_trace_handler(output_dir)

    all_timings: List[Dict[str, float]] = []

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i, batch in enumerate(loader.iter_epoch("train", seed=42)):
            if i >= 1 + warmup_iterations + num_iterations:
                break

            timings = profile_full_iteration(model, batch, device, losses, optimizer)
            all_timings.append(timings)

            prof.step()

            if i % 5 == 0:
                logger.info(f"Profiler step {i}: {timings}")

    # Export chrome trace if requested
    if chrome_trace_path:
        prof.export_chrome_trace(chrome_trace_path)
        logger.info(f"Chrome trace exported to {chrome_trace_path}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("PROFILER SUMMARY (sorted by CPU time)")
    logger.info("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    # Calculate average timings
    avg_timings = {}
    if all_timings:
        for key in all_timings[0].keys():
            values = [t[key] for t in all_timings[warmup_iterations:]]
            avg_timings[f"avg_{key}"] = sum(values) / len(values)

    return avg_timings


def run_mps_profiler(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    losses: Loss,
    optimizer: torch.optim.Optimizer,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
    mode: str = "interval",
) -> Dict[str, float]:
    """Run MPS profiler for OS Signpost traces.

    The generated signposts can be viewed in Xcode Instruments:
    1. Open Instruments (Xcode -> Open Developer Tool -> Instruments)
    2. Choose "Logging" template
    3. Run this script while Instruments is recording
    """
    if not HAS_MPS_PROFILER:
        logger.warning("MPS profiler not available")
        return {}

    if device.type != "mps":
        logger.warning("MPS profiler only works on MPS device")
        return {}

    logger.info(f"Starting MPS profiler in {mode} mode")
    logger.info("View traces in Xcode Instruments -> Logging template")

    all_timings: List[Dict[str, float]] = []

    # Warmup without profiling
    for i, batch in enumerate(loader.iter_epoch("train", seed=42)):
        if i >= warmup_iterations:
            break
        _ = profile_full_iteration(model, batch, device, losses, optimizer)

    # Profile iterations
    mps_profiler.start(mode=mode, wait_until_completed=True)
    try:
        for i, batch in enumerate(loader.iter_epoch("train", seed=43)):
            if i >= num_iterations:
                break
            timings = profile_full_iteration(model, batch, device, losses, optimizer)
            all_timings.append(timings)
            logger.info(f"MPS profiler iteration {i}: {timings}")
    finally:
        mps_profiler.stop()

    # Calculate averages
    avg_timings = {}
    if all_timings:
        for key in all_timings[0].keys():
            values = [t[key] for t in all_timings]
            avg_timings[f"avg_{key}"] = sum(values) / len(values)

    return avg_timings


def analyze_device_placement(model: torch.nn.Module, device: torch.device):
    """Analyze where model parameters are placed."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL DEVICE PLACEMENT ANALYSIS")
    logger.info("=" * 80)

    cpu_params = 0
    gpu_params = 0
    cpu_modules = []

    for name, param in model.named_parameters():
        if param.device.type == "cpu":
            cpu_params += param.numel()
            cpu_modules.append(name)
        else:
            gpu_params += param.numel()

    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            if param.device.type != device.type:
                logger.warning(f"Parameter {name}.{pname} is on {param.device}, expected {device}")

    logger.info(f"Expected device: {device}")
    logger.info(f"GPU parameters: {gpu_params:,} ({gpu_params / 1e6:.2f}M)")
    logger.info(f"CPU parameters: {cpu_params:,} ({cpu_params / 1e6:.2f}M)")

    if cpu_modules:
        logger.warning(f"Modules on CPU: {cpu_modules[:10]}...")

    return {"cpu_params": cpu_params, "gpu_params": gpu_params}


def print_timing_report(timings: Dict[str, float], title: str = "TIMING REPORT"):
    """Print formatted timing report."""
    logger.info("\n" + "=" * 80)
    logger.info(title)
    logger.info("=" * 80)

    total = timings.get("avg_total_time_ms", sum(v for k, v in timings.items() if "time" in k))

    for key, value in sorted(timings.items()):
        if "time" in key.lower():
            pct = (value / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 2)
            logger.info(f"{key:30s}: {value:8.2f} ms ({pct:5.1f}%) {bar}")


def main():
    parser = argparse.ArgumentParser(description="Profile DeepFilterNet training")
    parser.add_argument("data_config_file", type=str, help="Path to dataset config")
    parser.add_argument("data_dir", type=str, help="Path to HDF5 data directory")
    parser.add_argument("base_dir", type=str, help="Output directory for profiles")
    parser.add_argument("--num-iterations", type=int, default=10, help="Iterations to profile")
    parser.add_argument("--warmup-iterations", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--chrome-trace", type=str, default=None, help="Chrome trace output path")
    parser.add_argument("--mps-profiler", action="store_true", help="Use MPS OS Signpost profiler")
    parser.add_argument("--mps-mode", type=str, default="interval", choices=["interval", "event"])
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, mps, cuda)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for profiling")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)
    init_logger(file=os.path.join(args.base_dir, "profile.log"), level=args.log_level)

    # Load config if exists
    config_file = os.path.join(args.base_dir, "config.ini")
    if os.path.exists(config_file):
        config.load(config_file)

    # Set up device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    logger.info(f"Profiling on device: {device}")

    # Print device info
    logger.info("\n" + "=" * 80)
    logger.info("DEVICE INFORMATION")
    logger.info("=" * 80)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Device: {device}")
    if device.type == "mps":
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")
        logger.info(f"MPS built: {torch.backends.mps.is_built()}")
    elif device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(device)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    # Initialize model - set defaults if no config loaded
    if not config_file or not os.path.exists(config_file):
        # Load defaults for profiling
        config.use_defaults()
        config.set("MODEL", "deepfilternet4", str, section="train")
        config.set("FFT_SIZE", 960, int, section="df")
        config.set("HOP_SIZE", 480, int, section="df")
        config.set("NB_ERB", 32, int, section="df")
        config.set("NB_DF", 96, int, section="df")
        config.set("SR", 48000, int, section="df")
        logger.info("Using default DeepFilterNet4 configuration for profiling")

    # Import the correct model parameters class based on model type
    model_name = config("MODEL", default="deepfilternet4", section="train")
    if model_name == "deepfilternet4":
        from df.deepfilternet4 import ModelParams4 as ParamsClass
    else:
        from df.model import ModelParams as ParamsClass

    p = ParamsClass()
    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )

    checkpoint_dir = os.path.join(args.base_dir, "checkpoints")
    model, epoch = load_model(checkpoint_dir if os.path.exists(checkpoint_dir) else None, df_state)
    model = model.to(device)
    model.train()

    # Analyze device placement
    analyze_device_placement(model, device)

    # Initialize dataloader
    dataloader = DataLoader(
        ds_dir=args.data_dir,
        ds_config=args.data_config_file,
        sr=p.sr,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size,
        num_workers=2,
        pin_memory=device.type == "cuda",
        max_len_s=config("MAX_SAMPLE_LEN_S", 5.0, float, section="train"),
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_erb=p.nb_erb,
        nb_spec=p.nb_df,
        norm_alpha=get_norm_alpha(log=False),
        prefetch=8,
        seed=42,
    )

    # Initialize losses
    istft = Istft(p.fft_size, p.hop_size, torch.as_tensor(df_state.fft_window().copy())).to(device)
    losses = Loss(df_state, istft).to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Initial memory stats
    logger.info("\n" + "=" * 80)
    logger.info("INITIAL MEMORY STATS")
    logger.info("=" * 80)
    mem_stats = measure_device_utilization(device)
    for k, v in mem_stats.items():
        logger.info(f"{k}: {v:.3f} GB")

    # Profile dataloader
    logger.info("\n" + "=" * 80)
    logger.info("DATALOADER PROFILING")
    logger.info("=" * 80)
    dl_stats = profile_dataloader_step(dataloader, "train", seed=42)
    for k, v in dl_stats.items():
        logger.info(f"{k}: {v}")

    # Run torch profiler
    logger.info("\n" + "=" * 80)
    logger.info("TORCH PROFILER")
    logger.info("=" * 80)
    avg_timings = run_torch_profiler(
        model=model,
        loader=dataloader,
        device=device,
        losses=losses,
        optimizer=optimizer,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations,
        output_dir=args.base_dir,
        chrome_trace_path=args.chrome_trace,
    )
    print_timing_report(avg_timings, "AVERAGE ITERATION TIMING")

    # Calculate MPS utilization estimate
    if device.type == "mps" and avg_timings:
        total_time = avg_timings.get("avg_total_time_ms", 0)
        forward_time = avg_timings.get("avg_forward_time_ms", 0)
        backward_time = avg_timings.get("avg_backward_time_ms", 0)
        loss_time = avg_timings.get("avg_loss_time_ms", 0)

        gpu_time = forward_time + backward_time + loss_time
        cpu_overhead = total_time - gpu_time

        logger.info("\n" + "=" * 80)
        logger.info("MPS UTILIZATION ESTIMATE")
        logger.info("=" * 80)
        logger.info(f"Total iteration time: {total_time:.2f} ms")
        logger.info(f"GPU compute time (forward+backward+loss): {gpu_time:.2f} ms")
        logger.info(f"CPU/transfer overhead: {cpu_overhead:.2f} ms")
        logger.info(f"Estimated GPU utilization: {gpu_time / total_time * 100:.1f}%")

    # Run MPS profiler if requested
    if args.mps_profiler and device.type == "mps":
        logger.info("\n" + "=" * 80)
        logger.info("MPS OS SIGNPOST PROFILER")
        logger.info("=" * 80)
        mps_timings = run_mps_profiler(
            model=model,
            loader=dataloader,
            device=device,
            losses=losses,
            optimizer=optimizer,
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations,
            mode=args.mps_mode,
        )
        if mps_timings:
            print_timing_report(mps_timings, "MPS PROFILER TIMING")

    # Final memory stats
    logger.info("\n" + "=" * 80)
    logger.info("FINAL MEMORY STATS")
    logger.info("=" * 80)
    mem_stats = measure_device_utilization(device)
    for k, v in mem_stats.items():
        logger.info(f"{k}: {v:.3f} GB")

    logger.info("\n" + "=" * 80)
    logger.info("PROFILING COMPLETE")
    logger.info("=" * 80)

    # Summary recommendations
    if avg_timings:
        total = avg_timings.get("avg_total_time_ms", 0)
        transfer = avg_timings.get("avg_transfer_time_ms", 0)

        if transfer / total > 0.2:
            logger.warning(
                f"Data transfer takes {transfer / total * 100:.1f}% of iteration time. "
                "Consider increasing prefetch or using async data loading."
            )

        forward = avg_timings.get("avg_forward_time_ms", 0)
        backward = avg_timings.get("avg_backward_time_ms", 0)
        if backward > forward * 2:
            logger.info(
                f"Backward pass ({backward:.1f}ms) is significantly slower than forward ({forward:.1f}ms). "
                "This is normal for training but consider gradient checkpointing for memory."
            )


if __name__ == "__main__":
    main()
