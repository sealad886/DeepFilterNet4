#!/usr/bin/env python3
"""Basic MLX DeepFilterNet4 inference example.

This example demonstrates how to:
1. Load an MLX DeepFilterNet4 model
2. Enhance a single audio file
3. Save the enhanced output

Requirements:
    pip install mlx soundfile

Usage:
    python mlx_inference.py input.wav output.wav
    python mlx_inference.py input.wav output.wav --model path/to/checkpoint.safetensors

The model removes background noise while preserving speech quality.
Optimized for Apple Silicon (M1/M2/M3) using MLX.
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_audio(path: str, target_sr: int = 48000) -> mx.array:
    """Load audio file and convert to MLX array.

    Args:
        path: Path to audio file
        target_sr: Target sample rate (default: 48000)

    Returns:
        Audio samples as MLX array
    """
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        import numpy as np
        from scipy import signal

        num_samples = int(len(audio) * target_sr / sr)
        resampled = signal.resample(audio, num_samples)
        audio = np.asarray(resampled, dtype=np.float32)

    return mx.array(audio)


def save_audio(audio: mx.array, path: str, sr: int = 48000) -> None:
    """Save MLX array as audio file.

    Args:
        audio: Audio samples as MLX array
        path: Output file path
        sr: Sample rate
    """
    import numpy as np
    import soundfile as sf

    # Convert to numpy
    audio_np = np.array(audio, dtype=np.float32)

    # Normalize to prevent clipping
    max_val = np.abs(audio_np).max()
    if max_val > 1.0:
        audio_np = audio_np / max_val * 0.95

    sf.write(path, audio_np, sr)


def enhance_audio(
    audio: mx.array,
    model_path: str | None = None,
) -> mx.array:
    """Enhance audio using MLX DeepFilterNet4.

    Args:
        audio: Input audio samples (1D array)
        model_path: Optional path to model checkpoint

    Returns:
        Enhanced audio samples
    """
    from df_mlx.model import init_model
    from df_mlx.train import load_checkpoint

    # Initialize model with default parameters
    model = init_model()

    # Load checkpoint if provided
    if model_path:
        load_checkpoint(model, model_path)
        print(f"Loaded model from: {model_path}")
    else:
        print("Using randomly initialized model (for demonstration)")
        print("For real usage, provide a trained checkpoint with --model")

    # Enhance audio using model's enhance method (return_spec=False returns mx.array)
    result = model.enhance(audio)
    assert isinstance(result, mx.array), "Expected mx.array from enhance()"

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhance audio using MLX DeepFilterNet4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python mlx_inference.py noisy.wav clean.wav

    # With model checkpoint
    python mlx_inference.py noisy.wav clean.wav --model checkpoint.safetensors
        """,
    )
    parser.add_argument("input", help="Input audio file (WAV, FLAC, etc.)")
    parser.add_argument("output", help="Output audio file")
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

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading audio: {args.input}")
    audio = load_audio(args.input, args.sample_rate)
    duration = len(audio) / args.sample_rate
    print(f"  Duration: {duration:.2f}s ({len(audio)} samples)")

    print("Enhancing...")
    start_time = time.time()
    enhanced = enhance_audio(audio, args.model)
    mx.eval(enhanced)  # Force evaluation
    elapsed = time.time() - start_time

    rtf = elapsed / duration if duration > 0 else 0
    print(f"  Processing time: {elapsed:.2f}s")
    print(f"  Real-time factor: {rtf:.2f}x")

    print(f"Saving output: {args.output}")
    save_audio(enhanced, args.output, args.sample_rate)
    print("Done!")


if __name__ == "__main__":
    main()
