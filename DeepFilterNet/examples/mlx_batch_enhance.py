#!/usr/bin/env python3
"""Batch audio enhancement with MLX DeepFilterNet4.

This example demonstrates how to:
1. Process multiple audio files in a directory
2. Use batch processing for efficiency
3. Handle various input formats
4. Track progress with statistics

Requirements:
    pip install mlx soundfile tqdm

Usage:
    python mlx_batch_enhance.py input_dir/ output_dir/
    python mlx_batch_enhance.py input_dir/ output_dir/ --model checkpoint.safetensors

All audio files in input_dir will be enhanced and saved to output_dir
with the same filenames.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator

import mlx.core as mx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Supported audio extensions
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def find_audio_files(directory: Path) -> Iterator[Path]:
    """Find all audio files in a directory (recursively).

    Args:
        directory: Directory to search

    Yields:
        Paths to audio files
    """
    for ext in AUDIO_EXTENSIONS:
        yield from directory.rglob(f"*{ext}")


def load_audio(path: Path, target_sr: int = 48000) -> tuple[mx.array, float]:
    """Load audio file and return samples with original duration.

    Args:
        path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Tuple of (audio array, duration in seconds)
    """
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")

    # Store original duration
    original_duration = len(audio) / sr

    # Convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        import numpy as np
        from scipy import signal

        num_samples = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, num_samples)
        audio = audio.astype(np.float32)

    return mx.array(audio), original_duration


def save_audio(audio: mx.array, path: Path, sr: int = 48000) -> None:
    """Save audio array to file.

    Args:
        audio: Audio samples
        path: Output path
        sr: Sample rate
    """
    import numpy as np
    import soundfile as sf

    audio_np = np.array(audio, dtype=np.float32)

    # Normalize to prevent clipping
    max_val = np.abs(audio_np).max()
    if max_val > 1.0:
        audio_np = audio_np / max_val * 0.95

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(path), audio_np, sr)


def batch_enhance(
    input_dir: Path,
    output_dir: Path,
    model_path: str | None = None,
    sample_rate: int = 48000,
    verbose: bool = True,
) -> dict:
    """Enhance all audio files in a directory.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        model_path: Optional model checkpoint path
        sample_rate: Target sample rate
        verbose: Print progress

    Returns:
        Statistics dictionary
    """
    from df_mlx.model import enhance, init_model, load_checkpoint

    # Find all audio files
    audio_files = list(find_audio_files(input_dir))
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return {"files_processed": 0}

    print(f"Found {len(audio_files)} audio files")

    # Initialize model
    model = init_model()
    if model_path:
        load_checkpoint(model, model_path)
        print(f"Loaded model: {model_path}")
    else:
        print("Warning: Using randomly initialized model (provide --model for real usage)")

    # Process files
    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "total_duration": 0.0,
        "total_time": 0.0,
    }

    try:
        from tqdm import tqdm

        progress = tqdm(audio_files, desc="Enhancing", unit="file")
    except ImportError:
        progress = audio_files
        if verbose:
            print("Install tqdm for progress bar: pip install tqdm")

    for audio_path in progress:
        try:
            # Compute relative path for output
            rel_path = audio_path.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix(".wav")

            # Load audio
            audio, duration = load_audio(audio_path, sample_rate)
            stats["total_duration"] += duration

            # Enhance
            start_time = time.time()
            enhanced = enhance(model, audio)
            mx.eval(enhanced)
            elapsed = time.time() - start_time
            stats["total_time"] += elapsed

            # Save
            save_audio(enhanced, output_path, sample_rate)
            stats["files_processed"] += 1

            if verbose and not hasattr(progress, "set_postfix"):
                rtf = elapsed / duration if duration > 0 else 0
                print(f"  {audio_path.name}: {duration:.1f}s @ {rtf:.2f}x RTF")

        except Exception as e:
            stats["files_failed"] += 1
            if verbose:
                print(f"  Error processing {audio_path.name}: {e}")

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch enhance audio files with MLX DeepFilterNet4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all files in a directory
    python mlx_batch_enhance.py input/ output/

    # With model checkpoint
    python mlx_batch_enhance.py input/ output/ --model checkpoint.safetensors

    # Recursively process subdirectories
    python mlx_batch_enhance.py data/noisy/ data/enhanced/ -m model.safetensors
        """,
    )
    parser.add_argument("input_dir", help="Input directory containing audio files")
    parser.add_argument("output_dir", help="Output directory for enhanced files")
    parser.add_argument(
        "--model",
        "-m",
        help="Path to model checkpoint (.safetensors)",
        default=None,
    )
    parser.add_argument(
        "--sample-rate",
        "-sr",
        type=int,
        default=48000,
        help="Sample rate (default: 48000)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: Input is not a directory: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Process files
    stats = batch_enhance(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=args.model,
        sample_rate=args.sample_rate,
        verbose=not args.quiet,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Files processed: {stats['files_processed']}")
    if stats["files_failed"]:
        print(f"  Files failed:    {stats['files_failed']}")
    print(f"  Total duration:  {stats['total_duration']:.1f}s")
    print(f"  Processing time: {stats['total_time']:.1f}s")

    if stats["total_duration"] > 0:
        rtf = stats["total_time"] / stats["total_duration"]
        print(f"  Real-time factor: {rtf:.2f}x")
        print(f"  Throughput: {stats['total_duration'] / stats['total_time']:.1f}x real-time")


if __name__ == "__main__":
    main()
