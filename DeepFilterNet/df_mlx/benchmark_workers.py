#!/usr/bin/env python3
"""Benchmark different worker counts for MLX datastore building.

This script helps determine optimal parallelism settings by measuring
processing speed with different numbers of workers.

Usage:
    python -m df_mlx.benchmark_workers \
        --speech-list /path/to/speech.txt \
        --noise-list /path/to/noise.txt \
        --samples 500

Results will show samples/second for each worker configuration.
"""

import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

# Import from prepare_data
from df_mlx.prepare_data import create_erb_filterbank, load_audio, process_sample, read_file_list


def process_single_file(
    args_tuple: Tuple,
) -> bool:
    """Process a single file (for multiprocessing)."""
    (
        speech_path,
        noise_array,
        rir_array,
        snr_db,
        fft_size,
        hop_size,
        erb_fb,
        nb_df,
        window,
        sample_rate,
        segment_samples,
    ) = args_tuple

    try:
        clean = load_audio(speech_path, sample_rate)
        if len(clean) < segment_samples:
            return False

        if len(clean) > segment_samples:
            start = random.randint(0, len(clean) - segment_samples)
            clean = clean[start : start + segment_samples]

        result = process_sample(
            clean,
            noise_array,
            rir_array,
            snr_db,
            fft_size,
            hop_size,
            erb_fb,
            nb_df,
            window,
        )
        return result is not None
    except Exception:
        return False


def benchmark_single_threaded(
    speech_files: List[str],
    noise_cache: dict,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    erb_fb: np.ndarray,
    nb_df: int,
    window: np.ndarray,
    segment_samples: int,
    num_samples: int,
) -> Tuple[float, int]:
    """Benchmark single-threaded processing."""
    noise_keys = list(noise_cache.keys())
    start = time.perf_counter()
    processed = 0

    for speech_path in tqdm(speech_files[:num_samples], desc="Single-threaded"):
        try:
            clean = load_audio(speech_path, sample_rate)
            if len(clean) < segment_samples:
                continue
            if len(clean) > segment_samples:
                s = random.randint(0, len(clean) - segment_samples)
                clean = clean[s : s + segment_samples]

            noise_path = random.choice(noise_keys)
            noise = noise_cache[noise_path]
            snr_db = random.uniform(-5, 25)

            result = process_sample(clean, noise, None, snr_db, fft_size, hop_size, erb_fb, nb_df, window)
            if result:
                processed += 1
        except Exception:
            continue

    elapsed = time.perf_counter() - start
    return elapsed, processed


def benchmark_threaded(
    speech_files: List[str],
    noise_cache: dict,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    erb_fb: np.ndarray,
    nb_df: int,
    window: np.ndarray,
    segment_samples: int,
    num_samples: int,
    num_workers: int,
) -> Tuple[float, int]:
    """Benchmark multi-threaded processing."""
    noise_keys = list(noise_cache.keys())

    def process_one(speech_path):
        try:
            clean = load_audio(speech_path, sample_rate)
            if len(clean) < segment_samples:
                return False
            if len(clean) > segment_samples:
                s = random.randint(0, len(clean) - segment_samples)
                clean = clean[s : s + segment_samples]

            noise_path = random.choice(noise_keys)
            noise = noise_cache[noise_path]
            snr_db = random.uniform(-5, 25)

            result = process_sample(clean, noise, None, snr_db, fft_size, hop_size, erb_fb, nb_df, window)
            return result is not None
        except Exception:
            return False

    start = time.perf_counter()
    processed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_one, f) for f in speech_files[:num_samples]]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Threads({num_workers})"):
            if future.result():
                processed += 1

    elapsed = time.perf_counter() - start
    return elapsed, processed


def benchmark_multiprocess(
    speech_files: List[str],
    noise_cache: dict,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    erb_fb: np.ndarray,
    nb_df: int,
    window: np.ndarray,
    segment_samples: int,
    num_samples: int,
    num_workers: int,
) -> Tuple[float, int]:
    """Benchmark multi-process processing."""
    noise_keys = list(noise_cache.keys())

    # Prepare args for each file
    args_list = []
    for speech_path in speech_files[:num_samples]:
        noise_path = random.choice(noise_keys)
        noise = noise_cache[noise_path]
        snr_db = random.uniform(-5, 25)
        args_list.append(
            (
                speech_path,
                noise,
                None,
                snr_db,
                fft_size,
                hop_size,
                erb_fb,
                nb_df,
                window,
                sample_rate,
                segment_samples,
            )
        )

    start = time.perf_counter()
    processed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_file, args) for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processes({num_workers})"):
            if future.result():
                processed += 1

    elapsed = time.perf_counter() - start
    return elapsed, processed


def main():
    parser = argparse.ArgumentParser(description="Benchmark worker counts for datastore building")
    parser.add_argument("--speech-list", required=True, help="Path to speech file list")
    parser.add_argument("--noise-list", required=True, help="Path to noise file list")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to process")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate")
    parser.add_argument("--fft-size", type=int, default=960, help="FFT size")
    parser.add_argument("--hop-size", type=int, default=480, help="Hop size")
    parser.add_argument("--nb-erb", type=int, default=32, help="ERB bands")
    parser.add_argument("--nb-df", type=int, default=96, help="DF bands")
    parser.add_argument("--segment-length", type=float, default=5.0, help="Segment length in seconds")
    args = parser.parse_args()

    # Detect CPU count
    cpu_count = os.cpu_count() or 4
    print(f"Detected {cpu_count} CPU cores")
    print(f"Testing with {args.samples} samples")
    print()

    # Load file lists
    print("Loading file lists...")
    speech_files = read_file_list(args.speech_list)
    noise_files = read_file_list(args.noise_list)
    print(f"  Speech files: {len(speech_files):,}")
    print(f"  Noise files:  {len(noise_files):,}")

    # Shuffle for variety
    random.seed(42)
    random.shuffle(speech_files)

    # Preload noise
    print("\nPreloading noise files...")
    noise_cache = {}
    for nf in noise_files[: min(50, len(noise_files))]:
        try:
            noise_cache[nf] = load_audio(nf, args.sample_rate)
        except Exception:
            pass
    print(f"  Loaded {len(noise_cache)} noise files")

    # Setup
    erb_fb = create_erb_filterbank(args.sample_rate, args.fft_size, args.nb_erb)
    window = np.sqrt(np.hanning(args.fft_size + 1)[:-1]).astype(np.float32)
    segment_samples = int(args.segment_length * args.sample_rate)

    results = []

    # Benchmark single-threaded
    print("\n" + "=" * 60)
    print("Benchmarking single-threaded (baseline)...")
    elapsed, processed = benchmark_single_threaded(
        speech_files,
        noise_cache,
        args.sample_rate,
        args.fft_size,
        args.hop_size,
        erb_fb,
        args.nb_df,
        window,
        segment_samples,
        args.samples,
    )
    rate = processed / elapsed
    results.append(("Single-threaded", 1, elapsed, processed, rate))
    print(f"  Time: {elapsed:.2f}s, Processed: {processed}, Rate: {rate:.1f} samples/sec")

    # Benchmark threading (I/O bound)
    worker_counts = [2, 4, 8, 12, 16]
    if cpu_count > 16:
        worker_counts.append(cpu_count)

    print("\n" + "=" * 60)
    print("Benchmarking multi-threading (good for I/O bound)...")
    for n in worker_counts:
        if n > cpu_count * 2:
            continue
        elapsed, processed = benchmark_threaded(
            speech_files,
            noise_cache,
            args.sample_rate,
            args.fft_size,
            args.hop_size,
            erb_fb,
            args.nb_df,
            window,
            segment_samples,
            args.samples,
            n,
        )
        rate = processed / elapsed
        results.append(("Threading", n, elapsed, processed, rate))
        print(f"  Workers: {n:2d}, Time: {elapsed:.2f}s, Rate: {rate:.1f} samples/sec")

    # Benchmark multiprocessing (CPU bound)
    print("\n" + "=" * 60)
    print("Benchmarking multi-processing (good for CPU bound)...")
    for n in [2, 4, 8, min(12, cpu_count)]:
        if n > cpu_count:
            continue
        elapsed, processed = benchmark_multiprocess(
            speech_files,
            noise_cache,
            args.sample_rate,
            args.fft_size,
            args.hop_size,
            erb_fb,
            args.nb_df,
            window,
            segment_samples,
            args.samples,
            n,
        )
        rate = processed / elapsed
        results.append(("Multiprocessing", n, elapsed, processed, rate))
        print(f"  Workers: {n:2d}, Time: {elapsed:.2f}s, Rate: {rate:.1f} samples/sec")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'Workers':>8} {'Time (s)':>10} {'Rate':>12}")
    print("-" * 60)

    best_rate = 0
    best_config = ""
    for method, workers, elapsed, processed, rate in sorted(results, key=lambda x: -x[4]):
        print(f"{method:<20} {workers:>8} {elapsed:>10.2f} {rate:>10.1f}/s")
        if rate > best_rate:
            best_rate = rate
            best_config = f"{method} with {workers} workers"

    print("-" * 60)
    print(f"Best: {best_config} ({best_rate:.1f} samples/sec)")
    print()
    print("RECOMMENDATION:")
    print("  The workload is primarily I/O bound (loading audio files).")
    print("  Threading is usually faster than multiprocessing due to lower overhead.")
    print(f"  For your system ({cpu_count} cores), try: NUM_WORKERS={min(8, cpu_count)}")
    print()
    print("NOTE: The current prepare_data.py is single-threaded for simplicity.")
    print("      Multi-threading would require refactoring the processing loop.")


if __name__ == "__main__":
    main()
