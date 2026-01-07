#!/usr/bin/env python3
"""Real-time streaming audio enhancement with MLX DeepFilterNet4.

This example demonstrates how to:
1. Use the streaming API for frame-by-frame processing
2. Maintain state across frames for causal processing
3. Handle real-time audio with low latency
4. Process audio in chunks for live applications

Requirements:
    pip install mlx soundfile numpy

Usage:
    python mlx_streaming.py input.wav output.wav
    python mlx_streaming.py input.wav output.wav --chunk-size 960

The streaming API processes audio frame-by-frame, suitable for:
- Real-time audio applications
- Low-latency processing
- Memory-efficient handling of long recordings
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def streaming_enhance(
    audio: mx.array,
    model_path: str | None = None,
    chunk_samples: int = 480,
    verbose: bool = True,
) -> mx.array:
    """Enhance audio using streaming (frame-by-frame) processing.

    This demonstrates the StreamingDfNet4 API for real-time applications.

    Args:
        audio: Input audio samples (1D array)
        model_path: Optional path to model checkpoint
        chunk_samples: Samples per chunk (default: 480 = 10ms @ 48kHz)
        verbose: Print progress information

    Returns:
        Enhanced audio samples
    """
    from df_mlx.model import StreamingDfNet4, init_model
    from df_mlx.train import load_checkpoint

    # Initialize model
    model = init_model()
    if model_path:
        load_checkpoint(model, model_path)

    # Create streaming wrapper
    streaming = StreamingDfNet4(model)

    # Initialize state for batch_size=1
    state = streaming.init_state(batch_size=1)

    # Process audio in chunks
    num_samples = audio.shape[0]
    num_chunks = (num_samples + chunk_samples - 1) // chunk_samples

    if verbose:
        print(f"Processing {num_samples} samples in {num_chunks} chunks")
        print(f"Chunk size: {chunk_samples} samples ({chunk_samples / 48000 * 1000:.1f}ms)")

    output_chunks = []
    latencies = []

    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = min(start_idx + chunk_samples, num_samples)

        # Get input chunk
        chunk = audio[start_idx:end_idx]

        # Pad last chunk if needed
        if len(chunk) < chunk_samples:
            pad_len = chunk_samples - len(chunk)
            chunk = mx.pad(chunk, [(0, pad_len)])

        # Add batch dimension
        chunk = mx.expand_dims(chunk, axis=0)

        # Process frame
        start_time = time.time()
        enhanced_chunk, state = streaming.process_frame(chunk, state)
        mx.eval(enhanced_chunk)
        latency = time.time() - start_time
        latencies.append(latency)

        # Remove batch dimension and store
        if enhanced_chunk is not None:
            output_chunks.append(mx.squeeze(enhanced_chunk, axis=0))

        # Progress update
        if verbose and (i + 1) % 100 == 0:
            avg_latency = np.mean(latencies[-100:]) * 1000
            print(f"  Chunk {i + 1}/{num_chunks} - Avg latency: {avg_latency:.2f}ms")

    # Flush remaining samples
    remaining, _ = streaming.flush(state)
    if remaining is not None:
        output_chunks.append(mx.squeeze(remaining, axis=0))

    # Concatenate all chunks
    if output_chunks:
        enhanced = mx.concatenate(output_chunks, axis=0)
        # Trim to original length
        enhanced = enhanced[:num_samples]
    else:
        enhanced = mx.zeros_like(audio)

    if verbose:
        avg_latency = np.mean(latencies) * 1000
        max_latency = np.max(latencies) * 1000
        chunk_duration = chunk_samples / 48000 * 1000
        print("\nLatency statistics:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Maximum: {max_latency:.2f}ms")
        print(f"  Chunk duration: {chunk_duration:.1f}ms")
        print(f"  Real-time capable: {'Yes' if avg_latency < chunk_duration else 'No'}")

    return enhanced


def load_audio(path: str, target_sr: int = 48000) -> mx.array:
    """Load audio file."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        from scipy import signal

        num_samples = int(len(audio) * target_sr / sr)
        resampled = signal.resample(audio, num_samples)
        audio = np.asarray(resampled, dtype=np.float32)

    return mx.array(audio)


def save_audio(audio: mx.array, path: str, sr: int = 48000) -> None:
    """Save audio to file."""
    import soundfile as sf

    audio_np = np.array(audio, dtype=np.float32)
    max_val = np.abs(audio_np).max()
    if max_val > 1.0:
        audio_np = audio_np / max_val * 0.95

    sf.write(path, audio_np, sr)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Streaming audio enhancement with MLX DeepFilterNet4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic streaming processing
    python mlx_streaming.py input.wav output.wav

    # With smaller chunks for lower latency
    python mlx_streaming.py input.wav output.wav --chunk-size 240

    # With model checkpoint
    python mlx_streaming.py input.wav output.wav -m checkpoint.safetensors

Chunk sizes:
    - 480 samples = 10ms (default)
    - 240 samples = 5ms (lower latency)
    - 960 samples = 20ms (higher throughput)
        """,
    )
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument(
        "--model",
        "-m",
        help="Path to model checkpoint",
        default=None,
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=480,
        help="Samples per chunk (default: 480 = 10ms)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("MLX DeepFilterNet4 Streaming Example")
    print("=" * 50)

    print(f"Loading: {args.input}")
    audio = load_audio(args.input)
    duration = len(audio) / 48000
    print(f"Duration: {duration:.2f}s ({len(audio)} samples)")

    print("\nProcessing with streaming API...")
    start_time = time.time()
    enhanced = streaming_enhance(
        audio=audio,
        model_path=args.model,
        chunk_samples=args.chunk_size,
        verbose=not args.quiet,
    )
    total_time = time.time() - start_time

    print(f"\nTotal processing time: {total_time:.2f}s")
    print(f"Overall RTF: {total_time / duration:.2f}x")

    print(f"Saving: {args.output}")
    save_audio(enhanced, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
