#!/usr/bin/env python3
"""Generate file lists for dynamic dataset training.

This script scans directories or extracts from HDF5 datasets to create
file lists for the dynamic dataset training script.

Usage:
    # From directory of audio files
    python -m df_mlx.generate_file_lists \
        --speech-dirs /path/to/speech1 /path/to/speech2 \
        --noise-dirs /path/to/noise \
        --rir-dirs /path/to/rirs \
        --output-dir ./file_lists

    # From HDF5 datasets (original format)
    python -m df_mlx.generate_file_lists \
        --speech-hdf5 /path/to/speech_clean.hdf5 \
        --noise-hdf5 /path/to/noise_music.hdf5 \
        --rir-hdf5 /path/to/rir.hdf5 \
        --output-dir ./file_lists

Outputs:
    output_dir/
        speech_files.txt
        noise_files.txt
        rir_files.txt
        config.json  (for use with train_dynamic.py --config)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}


def find_audio_files(
    directories: List[str],
    extensions: Optional[set] = None,
    recursive: bool = True,
) -> List[str]:
    """Find all audio files in directories.

    Args:
        directories: List of directories to scan
        extensions: Set of valid extensions (default: common audio formats)
        recursive: Whether to scan recursively

    Returns:
        List of audio file paths
    """
    if extensions is None:
        extensions = AUDIO_EXTENSIONS

    files = []
    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue

        if recursive:
            for ext in extensions:
                files.extend(str(f) for f in path.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                files.extend(str(f) for f in path.glob(f"*{ext}"))

    return sorted(set(files))


def extract_from_hdf5(hdf5_path: str) -> List[str]:
    """Extract audio file paths from HDF5 dataset.

    The HDF5 format used by DeepFilterNet stores audio in groups,
    with attributes pointing to original files.

    Args:
        hdf5_path: Path to HDF5 file

    Returns:
        List of file paths (or keys if no paths stored)
    """
    try:
        import h5py
    except ImportError:
        print("Error: h5py required for HDF5 extraction. Install with: pip install h5py")
        return []

    files = []
    with h5py.File(hdf5_path, "r") as f:

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Try to get original file path from attributes
                if "file" in obj.attrs:
                    files.append(obj.attrs["file"])
                elif "path" in obj.attrs:
                    files.append(obj.attrs["path"])
                else:
                    files.append(name)

        f.visititems(visitor)

    return sorted(set(files))


def write_file_list(files: List[str], output_path: Path) -> None:
    """Write file list to text file."""
    with open(output_path, "w") as f:
        for file in files:
            f.write(f"{file}\n")
    print(f"  Wrote {len(files):,} files to {output_path}")


def generate_config(
    speech_files: List[str],
    noise_files: List[str],
    rir_files: List[str],
    output_dir: Path,
    sample_rate: int = 48000,
    segment_length: float = 5.0,
    p_reverb: float = 0.5,
) -> None:
    """Generate dataset config JSON file."""
    config = {
        "speech_files": speech_files,
        "noise_files": noise_files,
        "rir_files": rir_files,
        "sample_rate": sample_rate,
        "segment_length": segment_length,
        "fft_size": 960,
        "hop_size": 480,
        "nb_erb": 32,
        "nb_df": 96,
        "snr_range": [-5.0, 40.0],
        "snr_range_extreme": [-20.0, -5.0],
        "p_extreme_snr": 0.1,
        "gain_range": [-6.0, 6.0],
        "speech_gain_range": [-12.0, 12.0],
        "noise_gain_range": [-12.0, 12.0],
        "p_reverb": p_reverb,
        "p_clipping": 0.0,
        "p_bandwidth_ext": 0.0,
        "p_interfer_speech": 0.0,
        "n_noise_min": 2,
        "n_noise_max": 5,
        "p_random_noise": 0.05,
        "num_workers": 4,
        "prefetch_factor": 2,
        "seed": 42,
        "train_split": 0.9,
        "valid_split": 0.05,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Wrote config to {config_path}")


def read_file_list(file_path: str) -> List[str]:
    """Read file list from text file."""
    files = []
    path = Path(file_path)
    if not path.exists():
        print(f"Warning: File list not found: {file_path}")
        return files

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                files.append(line)

    return files


def main():
    parser = argparse.ArgumentParser(description="Generate file lists for dynamic dataset training")

    # Directory sources
    parser.add_argument(
        "--speech-dirs",
        nargs="+",
        type=str,
        help="Directories containing speech audio files",
    )
    parser.add_argument(
        "--noise-dirs",
        nargs="+",
        type=str,
        help="Directories containing noise audio files",
    )
    parser.add_argument(
        "--rir-dirs",
        nargs="+",
        type=str,
        help="Directories containing RIR audio files",
    )

    # File list sources (e.g., from existing .txt file lists)
    parser.add_argument(
        "--speech-list",
        type=str,
        help="Text file containing speech audio file paths (one per line)",
    )
    parser.add_argument(
        "--noise-list",
        type=str,
        help="Text file containing noise audio file paths (one per line)",
    )
    parser.add_argument(
        "--rir-list",
        type=str,
        help="Text file containing RIR audio file paths (one per line)",
    )

    # HDF5 sources
    parser.add_argument(
        "--speech-hdf5",
        type=str,
        help="HDF5 file containing speech audio",
    )
    parser.add_argument(
        "--noise-hdf5",
        type=str,
        help="HDF5 file containing noise audio",
    )
    parser.add_argument(
        "--rir-hdf5",
        type=str,
        help="HDF5 file containing RIRs",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write file lists",
    )

    # Options
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Sample rate for config",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=5.0,
        help="Segment length in seconds for config",
    )
    parser.add_argument(
        "--p-reverb",
        type=float,
        default=0.5,
        help="Reverb probability for config",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't scan directories recursively",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Also generate config.json with all file paths embedded",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating file lists...")
    print("=" * 60)

    # Collect speech files
    speech_files = []
    if args.speech_list:
        print(f"Reading speech file list: {args.speech_list}")
        speech_files = read_file_list(args.speech_list)
    elif args.speech_dirs:
        print(f"Scanning speech directories: {args.speech_dirs}")
        speech_files = find_audio_files(args.speech_dirs, recursive=not args.no_recursive)
    elif args.speech_hdf5:
        print(f"Extracting from HDF5: {args.speech_hdf5}")
        speech_files = extract_from_hdf5(args.speech_hdf5)

    # Collect noise files
    noise_files = []
    if args.noise_list:
        print(f"Reading noise file list: {args.noise_list}")
        noise_files = read_file_list(args.noise_list)
    elif args.noise_dirs:
        print(f"Scanning noise directories: {args.noise_dirs}")
        noise_files = find_audio_files(args.noise_dirs, recursive=not args.no_recursive)
    elif args.noise_hdf5:
        print(f"Extracting from HDF5: {args.noise_hdf5}")
        noise_files = extract_from_hdf5(args.noise_hdf5)

    # Collect RIR files
    rir_files = []
    if args.rir_list:
        print(f"Reading RIR file list: {args.rir_list}")
        rir_files = read_file_list(args.rir_list)
    elif args.rir_dirs:
        print(f"Scanning RIR directories: {args.rir_dirs}")
        rir_files = find_audio_files(args.rir_dirs, recursive=not args.no_recursive)
    elif args.rir_hdf5:
        print(f"Extracting from HDF5: {args.rir_hdf5}")
        rir_files = extract_from_hdf5(args.rir_hdf5)

    print()
    print("Results:")
    print("-" * 40)

    # Write file lists
    if speech_files:
        write_file_list(speech_files, output_dir / "speech_files.txt")
    else:
        print("  Warning: No speech files found")

    if noise_files:
        write_file_list(noise_files, output_dir / "noise_files.txt")
    else:
        print("  Warning: No noise files found")

    if rir_files:
        write_file_list(rir_files, output_dir / "rir_files.txt")
    else:
        print("  Info: No RIR files found (reverb will be disabled)")

    # Optionally generate config
    if args.generate_config:
        print()
        print("Generating config.json...")
        generate_config(
            speech_files,
            noise_files,
            rir_files,
            output_dir,
            sample_rate=args.sample_rate,
            segment_length=args.segment_length,
            p_reverb=args.p_reverb if rir_files else 0.0,
        )

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Speech files: {len(speech_files):,}")
    print(f"  Noise files:  {len(noise_files):,}")
    print(f"  RIR files:    {len(rir_files):,}")
    print(f"  Output dir:   {output_dir}")

    print()
    print("Usage with train_dynamic.py:")
    print("  python -m df_mlx.train_dynamic \\")
    print(f"      --speech-list {output_dir}/speech_files.txt \\")
    print(f"      --noise-list {output_dir}/noise_files.txt \\")
    if rir_files:
        print(f"      --rir-list {output_dir}/rir_files.txt \\")
    print("      --epochs 100 --batch-size 8")

    if args.generate_config:
        print()
        print("Or with config file:")
        print("  python -m df_mlx.train_dynamic \\")
        print(f"      --config {output_dir}/config.json \\")
        print("      --epochs 100 --batch-size 8")


if __name__ == "__main__":
    main()
