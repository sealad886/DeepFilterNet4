#!/usr/bin/env python3
"""Build pre-processed audio cache for efficient MLX training.

This script pre-processes raw audio files and saves them in a format
optimized for fast loading during training:
- Resample to target sample rate
- Convert to mono float32
- Normalize peak amplitude
- Save as sharded NPZ files for efficient I/O

The resulting cache enables dynamic mixing (like the original Rust DataLoader)
while avoiding the overhead of decoding audio files during training.

Usage:
    python -m df_mlx.build_audio_cache \
        --speech-list /path/to/speech_files.txt \
        --noise-list /path/to/noise_files.txt \
        --rir-list /path/to/rir_files.txt \
        --output-dir /path/to/audio_cache \
        --sample-rate 48000 \
        --num-workers 8

Output structure:
    output_dir/
        speech/
            shard_0000.npz  # Contains multiple audio arrays
            shard_0001.npz
            ...
        noise/
            shard_0000.npz
            ...
        rir/
            shard_0000.npz
            ...
        index.json  # Maps file paths to shard locations
        config.json  # Dataset configuration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from tqdm import tqdm

# Audio loading with resampling
try:
    import soundfile as sf
    from scipy import signal as scipy_signal

    def load_audio_file(path: str, target_sr: int) -> Optional[np.ndarray]:
        """Load audio file and resample if needed."""
        try:
            audio, file_sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Convert to mono
            if file_sr != target_sr:
                num_samples = int(len(audio) * target_sr / file_sr)
                audio = scipy_signal.resample(audio, num_samples)
            return cast(np.ndarray, audio).astype(np.float32)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            return None

except ImportError:
    print("Error: soundfile and scipy required. Install with:")
    print("  pip install soundfile scipy")
    sys.exit(1)


@dataclass
class ShardWriter:
    """Writes audio arrays to sharded NPZ files."""

    output_dir: Path
    category: str  # 'speech', 'noise', or 'rir'
    shard_size: int = 500  # Files per shard
    resume_from_shard: int = 0  # Starting shard index when resuming
    base_dir: Optional[str] = None  # Base dir for relative paths in index

    def __post_init__(self):
        self.shard_dir = self.output_dir / self.category
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.current_shard: Dict[str, np.ndarray] = {}
        self.current_shard_idx = self.resume_from_shard
        self.index: Dict[str, Tuple[str, str]] = {}  # path -> (shard_file, key)
        self._lock = threading.Lock()
        self._base_path = Path(self.base_dir) if self.base_dir else None

    def _get_index_key(self, original_path: str) -> str:
        """Convert absolute path to index key (relative if base_dir set)."""
        if self._base_path:
            try:
                return str(Path(original_path).relative_to(self._base_path))
            except ValueError:
                # Path not under base_dir, use as-is
                return original_path
        return original_path

    def add(self, original_path: str, audio: np.ndarray) -> None:
        """Add an audio array to the current shard."""
        with self._lock:
            key = f"audio_{len(self.current_shard):05d}"
            self.current_shard[key] = audio
            # Store relative path: category/shard_NNNN.npz
            shard_rel_path = f"{self.category}/shard_{self.current_shard_idx:04d}.npz"
            index_key = self._get_index_key(original_path)
            self.index[index_key] = (shard_rel_path, key)

            if len(self.current_shard) >= self.shard_size:
                self._flush_shard()

    def _flush_shard(self) -> None:
        """Write current shard to disk."""
        if not self.current_shard:
            return

        shard_path = self.shard_dir / f"shard_{self.current_shard_idx:04d}.npz"
        np.savez_compressed(shard_path, **self.current_shard)
        self.current_shard = {}
        self.current_shard_idx += 1

    def finalize(self) -> Dict[str, Tuple[str, str]]:
        """Flush remaining data and return index."""
        with self._lock:
            self._flush_shard()
        return self.index


def read_file_list(path: str) -> List[str]:
    """Read file list from text file."""
    files = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if os.path.exists(line):
                    files.append(line)
                else:
                    print(f"Warning: File not found: {line}")
    return files


def process_file(
    file_path: str,
    target_sr: int,
    normalize: bool = True,
) -> Optional[Tuple[str, np.ndarray]]:
    """Load and process a single audio file."""
    audio = load_audio_file(file_path, target_sr)
    if audio is None:
        return None

    # Normalize to peak amplitude
    if normalize:
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak

    return (file_path, audio)


def build_cache_for_category(
    file_list: List[str],
    category: str,
    output_dir: Path,
    sample_rate: int,
    shard_size: int,
    num_workers: int,
    normalize: bool = True,
    existing_index: Optional[Dict[str, Tuple[str, str]]] = None,
    base_dir: Optional[str] = None,
) -> Tuple[Dict[str, Tuple[str, str]], Dict]:
    """Build cache for a single category (speech/noise/rir).

    Args:
        file_list: List of file paths to process
        category: Category name ('speech', 'noise', or 'rir')
        output_dir: Output directory for cache
        sample_rate: Target sample rate
        shard_size: Files per shard
        num_workers: Number of parallel workers
        normalize: Whether to normalize audio
        existing_index: Existing index from previous run (for resume)
        base_dir: Base directory for computing relative paths (None = absolute)

    Returns:
        Tuple of (index dict, stats dict)
    """
    if not file_list:
        return existing_index or {}, {"total": 0, "cached": 0, "failed": 0, "skipped": 0}

    # Helper to convert path to index key (same logic as ShardWriter)
    base_path = Path(base_dir) if base_dir else None

    def to_index_key(path: str) -> str:
        if base_path:
            try:
                return str(Path(path).relative_to(base_path))
            except ValueError:
                return path
        return path

    # Filter out already-processed files when resuming
    if existing_index:
        existing_paths = set(existing_index.keys())
        # Compare using index keys (relative paths if base_dir set)
        files_to_process = [f for f in file_list if to_index_key(f) not in existing_paths]
        skipped_count = len(file_list) - len(files_to_process)

        # Find highest existing shard index for this category
        existing_shards = set()
        for shard_file, _ in existing_index.values():
            # shard_file format: "category/shard_0000.npz"
            shard_path = Path(shard_file)
            if shard_path.parent.name == category:
                shard_num = int(shard_path.stem.split("_")[1])
                existing_shards.add(shard_num)

        resume_from_shard = max(existing_shards) + 1 if existing_shards else 0
        print(f"\nResuming {category}: {skipped_count:,} already cached, {len(files_to_process):,} remaining")
        print(f"  Starting from shard index: {resume_from_shard}")
    else:
        files_to_process = file_list
        skipped_count = 0
        resume_from_shard = 0
        print(f"\nProcessing {category}: {len(file_list):,} files")

    if not files_to_process:
        print("  All files already cached!")
        return existing_index or {}, {
            "total": len(file_list),
            "cached": 0,
            "failed": 0,
            "skipped": skipped_count,
        }

    writer = ShardWriter(output_dir, category, shard_size, resume_from_shard=resume_from_shard, base_dir=base_dir)

    # Stats
    total_files = len(files_to_process)
    cached_count = 0
    failed_count = 0
    total_samples = 0
    total_duration = 0.0

    # Use bounded queue to prevent OOM
    max_in_flight = num_workers * 4
    pending_futures: Queue[Future] = Queue(maxsize=max_in_flight)

    def submit_task(executor, path):
        future = executor.submit(process_file, path, sample_rate, normalize)
        pending_futures.put(future)

    def process_completed():
        nonlocal cached_count, failed_count, total_samples, total_duration
        future = pending_futures.get()
        result = future.result()
        if result is not None:
            path, audio = result
            writer.add(path, audio)
            cached_count += 1
            total_samples += len(audio)
            total_duration += len(audio) / sample_rate
        else:
            failed_count += 1

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        pbar = tqdm(total=total_files, desc=f"  {category}", unit="files")

        for i, file_path in enumerate(files_to_process):
            # If queue is full, process one completed task first
            if pending_futures.full():
                process_completed()
                pbar.update(1)

            submit_task(executor, file_path)

        # Process remaining tasks
        while not pending_futures.empty():
            process_completed()
            pbar.update(1)

        pbar.close()

    new_index = writer.finalize()

    # Merge with existing index
    if existing_index:
        merged_index = {**existing_index, **new_index}
    else:
        merged_index = new_index

    stats = {
        "total": len(file_list),
        "cached": cached_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "total_samples": total_samples,
        "total_duration_hours": total_duration / 3600,
        "num_shards": writer.current_shard_idx,
    }

    print(f"  Newly cached: {cached_count:,} files")
    print(f"  Duration: {total_duration / 3600:.1f} hours")
    print(f"  Total in index: {len(merged_index):,} files")

    return merged_index, stats


def main():
    parser = argparse.ArgumentParser(description="Build pre-processed audio cache for MLX training")

    # Input file lists
    parser.add_argument(
        "--speech-list",
        type=str,
        required=True,
        help="Text file with speech audio paths",
    )
    parser.add_argument(
        "--noise-list",
        type=str,
        required=True,
        help="Text file with noise audio paths",
    )
    parser.add_argument(
        "--rir-list",
        type=str,
        help="Text file with RIR audio paths (optional)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write audio cache",
    )

    # Resume support
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing index.json (skip already-cached files)",
    )

    # Index path storage
    parser.add_argument(
        "--base-dir",
        type=str,
        help="Base directory for computing relative paths in index (default: use absolute paths)",
    )

    # Audio parameters
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Target sample rate (default: 48000)",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=5.0,
        help="Segment length in seconds for training (default: 5.0)",
    )

    # Performance
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500,
        help="Files per shard (default: 500)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    # Augmentation config (for config.json)
    parser.add_argument("--p-reverb", type=float, default=0.5)
    parser.add_argument("--p-clipping", type=float, default=0.0)
    parser.add_argument("--snr-min", type=float, default=-5.0)
    parser.add_argument("--snr-max", type=float, default=40.0)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DeepFilterNet MLX Audio Cache Builder")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Shard size: {args.shard_size} files")
    print(f"Workers: {args.num_workers}")

    # Load existing index for resume support
    existing_indices: Dict[str, Dict[str, Tuple[str, str]]] = {}
    index_path = output_dir / "index.json"
    if args.resume and index_path.exists():
        print(f"\nResume mode: Loading existing index from {index_path}")
        with open(index_path) as f:
            existing_indices = json.load(f)
        for cat, idx in existing_indices.items():
            print(f"  {cat}: {len(idx):,} files already cached")
    elif args.resume:
        print("\nResume mode: No existing index found, starting fresh")

    # Read file lists
    print("\nReading file lists...")
    speech_files = read_file_list(args.speech_list)
    noise_files = read_file_list(args.noise_list)
    rir_files = read_file_list(args.rir_list) if args.rir_list else []

    print(f"  Speech files: {len(speech_files):,}")
    print(f"  Noise files: {len(noise_files):,}")
    print(f"  RIR files: {len(rir_files):,}")

    if not speech_files:
        print("Error: No speech files found!")
        sys.exit(1)
    if not noise_files:
        print("Error: No noise files found!")
        sys.exit(1)

    start_time = time.time()

    # Build caches
    all_indices = {}
    all_stats = {}

    # Speech cache
    speech_index, speech_stats = build_cache_for_category(
        speech_files,
        "speech",
        output_dir,
        args.sample_rate,
        args.shard_size,
        args.num_workers,
        normalize=True,
        existing_index=existing_indices.get("speech"),
        base_dir=args.base_dir,
    )
    all_indices["speech"] = speech_index
    all_stats["speech"] = speech_stats

    # Noise cache
    noise_index, noise_stats = build_cache_for_category(
        noise_files,
        "noise",
        output_dir,
        args.sample_rate,
        args.shard_size,
        args.num_workers,
        normalize=True,
        existing_index=existing_indices.get("noise"),
        base_dir=args.base_dir,
    )
    all_indices["noise"] = noise_index
    all_stats["noise"] = noise_stats

    # RIR cache (if provided)
    if rir_files:
        rir_index, rir_stats = build_cache_for_category(
            rir_files,
            "rir",
            output_dir,
            args.sample_rate,
            args.shard_size,
            args.num_workers,
            normalize=False,
            existing_index=existing_indices.get("rir"),
            base_dir=args.base_dir,
        )
        all_indices["rir"] = rir_index
        all_stats["rir"] = rir_stats

    elapsed = time.time() - start_time

    # Write index file
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(all_indices, f)
    print(f"\nWrote index to {index_path}")

    # Write config file for training
    config = {
        "cache_dir": str(output_dir),
        "sample_rate": args.sample_rate,
        "segment_length": args.segment_length,
        "fft_size": 960,
        "hop_size": 480,
        "nb_erb": 32,
        "nb_df": 96,
        "snr_range": [args.snr_min, args.snr_max],
        "gain_range": [-6.0, 6.0],
        "p_reverb": args.p_reverb if rir_files else 0.0,
        "p_clipping": args.p_clipping,
        "n_noise_min": 2,
        "n_noise_max": 5,
        "stats": all_stats,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote config to {config_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Build complete!")
    print("=" * 60)
    print(f"Elapsed time: {elapsed / 60:.1f} minutes")
    print(f"Cache directory: {output_dir}")
    print()
    print("To train with dynamic mixing:")
    print(f"  python -m df_mlx.train_dynamic --cache-dir {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
