#!/usr/bin/env python3
"""Build MLX datastore from audio files.

This script converts audio files (speech, noise, RIR) into pre-computed
spectral features stored in an MLX-native sharded format for efficient
training.

Usage:
    python -m df_mlx.prepare_data \
        --speech-list /path/to/clean_all.txt \
        --noise-list /path/to/noise_music.txt \
        --rir-list /path/to/rir_all.txt \
        --output-dir /path/to/mlx_dataset \
        --train-split 0.9 \
        --valid-split 0.05

The script will:
1. Load audio files from the provided lists
2. Create noisy mixtures with random SNR and optional RIR convolution
3. Compute spectral features (STFT, ERB, DF bands)
4. Save to sharded .npz files with an index.json manifest

Requirements:
    pip install soundfile numpy scipy tqdm
"""

import argparse
import random
import sys
from typing import List, Optional, Tuple, cast

import numpy as np
from scipy import signal as scipy_signal
from tqdm import tqdm

# Try to import soundfile, fall back to scipy.io.wavfile
try:
    import soundfile as sf

    def load_audio(path: str, sr: int) -> np.ndarray:
        """Load audio file and resample if needed."""
        audio, file_sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        if file_sr != sr:
            # Simple resampling using scipy
            num_samples = int(len(audio) * sr / file_sr)
            audio = scipy_signal.resample(audio, num_samples)
        return cast(np.ndarray, audio).astype(np.float32)

except ImportError:
    from scipy.io import wavfile

    def load_audio(path: str, sr: int) -> np.ndarray:
        """Load audio file and resample if needed."""
        file_sr, audio = wavfile.read(path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            num_samples = int(len(audio) * sr / file_sr)
            audio = scipy_signal.resample(audio, num_samples)
        return cast(np.ndarray, audio).astype(np.float32)


def read_file_list(path: str) -> List[str]:
    """Read list of audio file paths from text file."""
    files = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Handle tab-separated format (path\tduration)
                if "\t" in line:
                    line = line.split("\t")[0]
                files.append(line)
    return files


def compute_stft(
    audio: np.ndarray,
    fft_size: int = 960,
    hop_size: int = 480,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute STFT of audio signal (optimized).

    Args:
        audio: Input audio (samples,)
        fft_size: FFT size
        hop_size: Hop size
        window: Optional window function

    Returns:
        Complex STFT (time, freq)
    """
    if window is None:
        window = np.sqrt(np.hanning(fft_size + 1)[:-1]).astype(np.float32)

    # Pad audio to ensure we get complete frames
    pad_len = fft_size - hop_size
    audio_padded = np.pad(audio, (pad_len, pad_len), mode="constant")

    # Use stride_tricks for efficient zero-copy frame extraction
    num_frames = (len(audio_padded) - fft_size) // hop_size + 1

    # Create strided view (no copy)
    shape = (num_frames, fft_size)
    strides = (audio_padded.strides[0] * hop_size, audio_padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio_padded, shape=shape, strides=strides, writeable=False)

    # Apply window and compute FFT (copy happens here)
    windowed = frames * window
    stft = np.fft.rfft(windowed, n=fft_size, axis=-1)
    return stft


def compute_erb_features(
    spec: np.ndarray,
    erb_fb: np.ndarray,
) -> np.ndarray:
    """Compute ERB band features from spectrum.

    Args:
        spec: Complex spectrum (time, freq)
        erb_fb: ERB filterbank matrix (freq, erb_bands)

    Returns:
        ERB features (time, erb_bands)
    """
    # Compute magnitude squared
    mag_sq = np.abs(spec) ** 2

    # Apply ERB filterbank
    erb = np.matmul(mag_sq, erb_fb)

    # Log compression with floor
    erb = np.log10(np.maximum(erb, 1e-10))

    return erb.astype(np.float32)


def compute_df_features(
    spec: np.ndarray,
    nb_df: int = 96,
) -> np.ndarray:
    """Compute DF-band features (complex coefficients).

    Args:
        spec: Complex spectrum (time, freq)
        nb_df: Number of DF bands

    Returns:
        DF features (time, nb_df, 2) - real/imag stacked
    """
    # Take first nb_df frequency bins
    df_spec = spec[:, :nb_df]

    # Stack real and imaginary parts
    df_feat = np.stack([df_spec.real, df_spec.imag], axis=-1)

    return df_feat.astype(np.float32)


def create_erb_filterbank(
    sr: int = 48000,
    fft_size: int = 960,
    nb_erb: int = 32,
    min_freq: float = 20.0,
    max_freq: Optional[float] = None,
) -> np.ndarray:
    """Create ERB filterbank matrix.

    Args:
        sr: Sample rate
        fft_size: FFT size
        nb_erb: Number of ERB bands
        min_freq: Minimum frequency
        max_freq: Maximum frequency (default: sr/2)

    Returns:
        Filterbank matrix (n_freqs, nb_erb)
    """
    if max_freq is None:
        max_freq = sr / 2

    n_freqs = fft_size // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)

    # ERB scale conversion
    def hz_to_erb(f):
        return 9.265 * np.log(1 + f / (24.7 * 9.265))

    def erb_to_hz(erb):
        return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)

    # Create ERB-spaced center frequencies
    erb_min = hz_to_erb(min_freq)
    erb_max = hz_to_erb(max_freq)
    erb_centers = np.linspace(erb_min, erb_max, nb_erb)
    center_freqs = erb_to_hz(erb_centers)

    # Create triangular filterbank
    fb = np.zeros((n_freqs, nb_erb), dtype=np.float32)

    for i in range(nb_erb):
        center = center_freqs[i]

        # Bandwidth based on ERB
        erb_bandwidth = 24.7 * (4.37 * center / 1000 + 1)
        low = center - erb_bandwidth / 2
        high = center + erb_bandwidth / 2

        # Triangular filter
        for j, f in enumerate(freqs):
            if low <= f <= center:
                fb[j, i] = (f - low) / (center - low + 1e-10)
            elif center < f <= high:
                fb[j, i] = (high - f) / (high - center + 1e-10)

    # Normalize each filter
    fb = fb / (fb.sum(axis=0, keepdims=True) + 1e-10)

    return fb


def mix_audio(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mix clean speech with noise at specified SNR.

    Args:
        clean: Clean speech signal
        noise: Noise signal
        snr_db: Target SNR in dB

    Returns:
        Tuple of (noisy mixture, scaled clean signal)
    """
    # Match lengths
    if len(noise) < len(clean):
        # Repeat noise
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[: len(clean)]

    # Compute signal powers
    clean_power = np.mean(clean**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10

    # Scale noise to achieve target SNR
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(target_noise_power / noise_power)

    # Mix
    noisy = clean + noise_scaled

    # Normalize to prevent clipping
    max_val = np.abs(noisy).max()
    if max_val > 0.99:
        scale = 0.99 / max_val
        noisy *= scale
        clean_out = clean * scale
    else:
        clean_out = clean

    return noisy, clean_out


def apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Apply room impulse response to audio.

    Args:
        audio: Input audio
        rir: Room impulse response

    Returns:
        Convolved audio
    """
    # Find direct path (highest peak)
    direct_idx = np.argmax(np.abs(rir))

    # Convolve
    convolved = scipy_signal.fftconvolve(audio, rir, mode="full")

    # Align to direct path and trim
    convolved = convolved[direct_idx : direct_idx + len(audio)]

    # Ensure same length
    if len(convolved) < len(audio):
        convolved = np.pad(convolved, (0, len(audio) - len(convolved)))
    elif len(convolved) > len(audio):
        convolved = convolved[: len(audio)]

    return convolved.astype(np.float32)


def process_sample(
    clean_audio: np.ndarray,
    noise_audio: np.ndarray,
    rir_audio: Optional[np.ndarray],
    snr_db: float,
    fft_size: int,
    hop_size: int,
    erb_fb: np.ndarray,
    nb_df: int,
    window: np.ndarray,
) -> Optional[dict]:
    """Process a single training sample.

    Args:
        clean_audio: Clean speech
        noise_audio: Noise signal
        rir_audio: Optional RIR
        snr_db: Target SNR
        fft_size: FFT size
        hop_size: Hop size
        erb_fb: ERB filterbank
        nb_df: Number of DF bands
        window: STFT window

    Returns:
        Dict with processed features or None if failed
    """
    try:
        # Apply RIR to clean speech if provided
        if rir_audio is not None:
            clean_reverb = apply_rir(clean_audio, rir_audio)
        else:
            clean_reverb = clean_audio

        # Mix with noise
        noisy, clean_scaled = mix_audio(clean_reverb, noise_audio, snr_db)

        # Compute STFTs
        noisy_stft = compute_stft(noisy, fft_size, hop_size, window)
        clean_stft = compute_stft(clean_scaled, fft_size, hop_size, window)

        # Compute features
        feat_erb = compute_erb_features(noisy_stft, erb_fb)
        feat_spec = compute_df_features(noisy_stft, nb_df)

        return {
            "spec_real": noisy_stft.real.astype(np.float32),
            "spec_imag": noisy_stft.imag.astype(np.float32),
            "feat_erb": feat_erb,
            "feat_spec": feat_spec,
            "clean_real": clean_stft.real.astype(np.float32),
            "clean_imag": clean_stft.imag.astype(np.float32),
            "snr_db": snr_db,
        }
    except Exception as e:
        print(f"  Warning: Failed to process sample: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Build MLX datastore from audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--speech-list",
        required=True,
        help="Path to clean speech file list",
    )
    parser.add_argument(
        "--noise-list",
        required=True,
        help="Path to noise file list",
    )
    parser.add_argument(
        "--rir-list",
        default=None,
        help="Path to RIR file list (optional)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for MLX datastore",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Target sample rate (default: 48000)",
    )
    parser.add_argument(
        "--fft-size",
        type=int,
        default=960,
        help="FFT size (default: 960)",
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=480,
        help="Hop size (default: 480)",
    )
    parser.add_argument(
        "--nb-erb",
        type=int,
        default=32,
        help="Number of ERB bands (default: 32)",
    )
    parser.add_argument(
        "--nb-df",
        type=int,
        default=96,
        help="Number of DF bands (default: 96)",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=-5,
        help="Minimum SNR in dB (default: -5)",
    )
    parser.add_argument(
        "--snr-max",
        type=float,
        default=25,
        help="Maximum SNR in dB (default: 25)",
    )
    parser.add_argument(
        "--rir-prob",
        type=float,
        default=0.5,
        help="Probability of applying RIR (default: 0.5)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9)",
    )
    parser.add_argument(
        "--valid-split",
        type=float,
        default=0.05,
        help="Fraction of data for validation (default: 0.05)",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=500,
        help="Samples per shard file (default: 500)",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=5.0,
        help="Audio segment length in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to generate (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing datastore",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use compressed .npz files (slower I/O but smaller files)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("MLX Datastore Builder")
    print("=" * 60)
    print(f"Output:       {args.output_dir}")
    print(f"Sample rate:  {args.sample_rate}")
    print(f"FFT size:     {args.fft_size}")
    print(f"Hop size:     {args.hop_size}")
    print(f"SNR range:    [{args.snr_min}, {args.snr_max}] dB")
    print(f"Segment:      {args.segment_length}s")
    print("=" * 60)

    # Read file lists
    print("\nLoading file lists...")
    speech_files = read_file_list(args.speech_list)
    noise_files = read_file_list(args.noise_list)
    rir_files = read_file_list(args.rir_list) if args.rir_list else []

    print(f"  Speech files: {len(speech_files):,}")
    print(f"  Noise files:  {len(noise_files):,}")
    print(f"  RIR files:    {len(rir_files):,}")

    if not speech_files:
        print("Error: No speech files found!")
        sys.exit(1)
    if not noise_files:
        print("Error: No noise files found!")
        sys.exit(1)

    # Create ERB filterbank
    erb_fb = create_erb_filterbank(
        sr=args.sample_rate,
        fft_size=args.fft_size,
        nb_erb=args.nb_erb,
    )

    # Create window
    window = np.sqrt(np.hanning(args.fft_size + 1)[:-1]).astype(np.float32)

    # Calculate segment samples
    segment_samples = int(args.segment_length * args.sample_rate)

    # Initialize datastore writer
    from df_mlx.datastore import DatastoreConfig, MLXDatastoreWriter, is_shutdown_requested

    config = DatastoreConfig(
        sample_rate=args.sample_rate,
        fft_size=args.fft_size,
        hop_size=args.hop_size,
        nb_erb=args.nb_erb,
        nb_df=args.nb_df,
        samples_per_shard=args.samples_per_shard,
        compress=args.compress,
    )

    # Use context manager to ensure proper cleanup
    with MLXDatastoreWriter(args.output_dir, config, resume=not args.no_resume) as writer:

        # Check if resuming
        if writer.resumed:
            print("\n*** Resuming from existing datastore ***")

        # Determine splits
        num_speech = len(speech_files)
        if args.max_samples:
            num_speech = min(num_speech, args.max_samples)

        train_end = int(num_speech * args.train_split)
        valid_end = train_end + int(num_speech * args.valid_split)

        # Shuffle speech files (use seed for reproducibility on resume)
        random.shuffle(speech_files)

        splits = [
            ("train", speech_files[:train_end]),
            ("valid", speech_files[train_end:valid_end]),
            ("test", speech_files[valid_end:num_speech]),
        ]

        # Preload some noise files for efficiency
        print("\nPreloading noise files...")
        noise_cache = {}
        for i, nf in enumerate(tqdm(noise_files[: min(100, len(noise_files))], desc="Loading noise")):
            try:
                noise_cache[nf] = load_audio(nf, args.sample_rate)
            except Exception as e:
                print(f"  Warning: Failed to load {nf}: {e}")
        # Pre-compute keys list to avoid repeated list() calls in hot loop
        noise_cache_keys = list(noise_cache.keys())

        # Preload RIR files if available
        rir_cache = {}
        if rir_files:
            print("Preloading RIR files...")
            for rf in tqdm(rir_files[: min(50, len(rir_files))], desc="Loading RIRs"):
                try:
                    rir_cache[rf] = load_audio(rf, args.sample_rate)
                except Exception as e:
                    print(f"  Warning: Failed to load {rf}: {e}")
        # Pre-compute keys list to avoid repeated list() calls in hot loop
        rir_cache_keys = list(rir_cache.keys()) if rir_cache else []

        # Process each split
        interrupted = False
        for split_name, split_files in splits:
            if not split_files:
                continue

            if interrupted or is_shutdown_requested():
                break

            # Get existing progress for resume
            existing_samples = writer.get_sample_count(split_name)
            files_already_processed = writer.get_files_processed(split_name)

            print(f"\nProcessing {split_name} split ({len(split_files):,} files)...")
            if files_already_processed > 0:
                print(f"  Resuming: skipping {files_already_processed:,} already-processed files")
                print(f"            ({existing_samples:,} samples exist)")

            writer.set_split(split_name)

            samples_added = 0
            files_skipped = 0
            try:
                pbar = tqdm(
                    enumerate(split_files),
                    total=len(split_files),
                    desc=split_name,
                    initial=files_already_processed,
                )
                for file_idx, speech_path in pbar:
                    # Skip already-processed files on resume
                    if file_idx < files_already_processed:
                        files_skipped += 1
                        continue

                    # Check for interrupt
                    if is_shutdown_requested():
                        interrupted = True
                        break

                    # Track that we're processing this file
                    writer.increment_files_processed(split_name)

                    try:
                        # Load speech
                        clean = load_audio(speech_path, args.sample_rate)

                        # Skip if too short
                        if len(clean) < segment_samples:
                            continue

                        # Extract random segment
                        if len(clean) > segment_samples:
                            start = random.randint(0, len(clean) - segment_samples)
                            clean = clean[start : start + segment_samples]

                        # Select random noise
                        if noise_cache_keys:
                            noise_path = random.choice(noise_cache_keys)
                            noise = noise_cache[noise_path]
                        else:
                            noise_path = random.choice(noise_files)
                            noise = load_audio(noise_path, args.sample_rate)

                        # Select random RIR (optional)
                        rir = None
                        if rir_cache_keys and random.random() < args.rir_prob:
                            rir_path = random.choice(rir_cache_keys)
                            rir = rir_cache[rir_path]

                        # Random SNR
                        snr_db = random.uniform(args.snr_min, args.snr_max)

                        # Process sample
                        result = process_sample(
                            clean,
                            noise,
                            rir,
                            snr_db,
                            args.fft_size,
                            args.hop_size,
                            erb_fb,
                            args.nb_df,
                            window,
                        )

                        if result:
                            writer.add_sample(**result)
                            samples_added += 1
                            pbar.set_postfix({"new": samples_added, "total": existing_samples + samples_added})

                    except Exception as e:
                        print(f"  Warning: Failed to process {speech_path}: {e}")
                        continue

            except KeyboardInterrupt:
                print(f"\n\nInterrupted during {split_name} split.")
                interrupted = True

            print(f"  Added {samples_added:,} new samples to {split_name}")
            if files_skipped > 0:
                print(f"  (Skipped {files_skipped:,} already-processed files)")

        # Context manager handles finalize
        if interrupted:
            print("\nSaving progress before exit...")

    # After context manager exits
    print("\n" + "=" * 60)
    if not interrupted and not is_shutdown_requested():
        print("Build complete!")
    else:
        print("Build interrupted - progress saved!")
        print("Resume with: same command (will continue from saved progress)")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Index file:       {args.output_dir}/index.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
