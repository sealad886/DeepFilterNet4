#!/usr/bin/env python3
"""Preprocessing script for Wall camera security footage.

This script builds a training dataset for DFNetMF from security camera footage
with stationary noise (clicking, high-pitched interference from bad microphone).

The approach:
1. Extract audio from MP4 files
2. Resample to 48kHz (DeepFilterNet requirement)
3. Detect silent segments (noise-only) vs active segments
4. Build paired training data for self-supervised noise removal

Usage:
    python preprocess_wall_dataset.py --input /Volumes/HomeSecurityVideos/Wall \
        --output /Users/andrew/DataDump/datasets/wall_processed \
        --max-files 500  # Limit for initial testing
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


@dataclass
class AudioSegment:
    """Represents an audio segment with metadata."""

    source_file: str
    start_time: float
    end_time: float
    is_silent: bool  # True if mostly noise, False if has activity
    rms_energy: float
    output_path: Optional[str] = None


@dataclass
class DatasetStats:
    """Statistics about the processed dataset."""

    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_segments: int = 0
    silent_segments: int = 0
    active_segments: int = 0
    total_duration_seconds: float = 0.0
    noise_profile_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "total_segments": self.total_segments,
            "silent_segments": self.silent_segments,
            "active_segments": self.active_segments,
            "total_duration_seconds": self.total_duration_seconds,
            "noise_profile_files": self.noise_profile_files,
        }


def check_dependencies():
    """Check that required tools are available."""
    required = ["ffmpeg", "ffprobe"]
    missing = []
    for tool in required:
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        print(f"❌ Missing required tools: {missing}")
        print("   Install with: brew install ffmpeg")
        sys.exit(1)
    print("✅ Dependencies check passed")


def find_mp4_files(input_dir: Path, max_files: Optional[int] = None) -> List[Path]:
    """Find all MP4 files in the input directory."""
    files = sorted(input_dir.rglob("*.mp4"))
    if max_files:
        files = files[:max_files]
    return files


def extract_audio(
    mp4_path: Path,
    output_path: Path,
    target_sr: int = 48000,
) -> bool:
    """Extract audio from MP4 and convert to WAV at target sample rate."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(target_sr),
            "-ac",
            "1",  # Mono
            str(output_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0 and output_path.exists()
    except Exception as e:
        print(f"Error extracting audio from {mp4_path}: {e}")
        return False


def analyze_audio_energy(wav_path: Path, segment_duration: float = 1.0) -> List[Tuple[float, float, float]]:
    """Analyze audio energy in segments.

    Returns list of (start_time, end_time, rms_energy) tuples.
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(wav_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        segment_samples = int(segment_duration * sr)
        segments = []

        for i in range(0, len(audio), segment_samples):
            segment = audio[i : i + segment_samples]
            if len(segment) < segment_samples // 2:
                continue

            rms = np.sqrt(np.mean(segment**2))
            start_time = i / sr
            end_time = (i + len(segment)) / sr
            segments.append((start_time, end_time, rms))

        return segments
    except Exception as e:
        print(f"Error analyzing {wav_path}: {e}")
        return []


def detect_noise_threshold(energy_values: List[float], percentile: float = 25) -> float:
    """Detect noise floor threshold from energy distribution.

    Segments below this threshold are considered "silent" (noise-only).
    """
    if not energy_values:
        return 0.01
    return np.percentile(energy_values, percentile)  # type: ignore


def process_single_file(
    mp4_path: Path,
    output_dir: Path,
    temp_dir: Path,
    target_sr: int = 48000,
    segment_duration: float = 2.0,
) -> Optional[Tuple[List[AudioSegment], float]]:
    """Process a single MP4 file and extract segments.

    Returns (list of segments, total duration) or None on failure.
    """
    # Extract audio to temp file (unique name for parallel execution)
    temp_wav = temp_dir / f"{mp4_path.stem}_{uuid.uuid4().hex}.wav"
    if not extract_audio(mp4_path, temp_wav, target_sr):
        return None

    # Analyze energy
    segments_energy = analyze_audio_energy(temp_wav, segment_duration)
    if not segments_energy:
        temp_wav.unlink(missing_ok=True)
        return None

    # Get total duration
    try:
        import soundfile as sf

        info = sf.info(temp_wav)
        total_duration = info.duration
    except Exception:
        total_duration = segments_energy[-1][1] if segments_energy else 0

    # Calculate noise threshold for this file
    energy_values = [s[2] for s in segments_energy]
    noise_threshold = detect_noise_threshold(energy_values)

    # Create segment objects
    segments = []
    for start, end, rms in segments_energy:
        is_silent = rms < noise_threshold * 1.5  # Allow some margin
        segment = AudioSegment(
            source_file=str(mp4_path),
            start_time=start,
            end_time=end,
            is_silent=is_silent,
            rms_energy=float(rms),
        )
        segments.append(segment)

    # Keep temp wav for later processing
    # Move to output dir with proper naming
    date_dir = mp4_path.parent.name  # YYYYMMDD
    out_wav_dir = output_dir / "wav" / date_dir
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    out_wav_path = out_wav_dir / f"{mp4_path.stem}.wav"

    shutil.move(temp_wav, out_wav_path)

    # Update segment paths
    for seg in segments:
        seg.output_path = str(out_wav_path)

    return segments, total_duration


def extract_noise_samples(
    segments: List[AudioSegment],
    output_dir: Path,
    max_noise_files: int = 100,
    min_duration: float = 1.0,
    start_index: int = 0,
) -> List[str]:
    """Extract pure noise samples from silent segments."""
    import soundfile as sf

    noise_dir = output_dir / "noise_samples"
    noise_dir.mkdir(parents=True, exist_ok=True)

    # Find the most silent segments (lowest energy)
    silent_segments = [s for s in segments if s.is_silent and s.output_path]
    silent_segments.sort(key=lambda x: x.rms_energy)

    noise_files = []
    for i, seg in enumerate(silent_segments[:max_noise_files]):
        if seg.end_time - seg.start_time < min_duration:
            continue

        try:
            audio, sr = sf.read(seg.output_path)
            start_sample = int(seg.start_time * sr)
            end_sample = int(seg.end_time * sr)
            noise_segment = audio[start_sample:end_sample]

            out_path = noise_dir / f"noise_{start_index + i:04d}.wav"
            sf.write(out_path, noise_segment, sr)
            noise_files.append(str(out_path))
        except Exception as e:
            print(f"Error extracting noise sample: {e}")

    return noise_files


def create_training_pairs(
    segments: List[AudioSegment],
    output_dir: Path,
    noise_files: List[str],
    pairs_per_file: int = 3,
) -> int:
    """Create training pairs for self-supervised learning.

    For stationary noise, we use:
    - Active segments as "noisy" inputs
    - Same segments with noise estimation subtracted as "target"

    This is a simplified approach - the model will learn the noise pattern
    from the data itself.
    """
    import soundfile as sf

    pairs_dir = output_dir / "training_pairs"
    noisy_dir = pairs_dir / "noisy"
    clean_dir = pairs_dir / "clean"  # Actually "less noisy" targets
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Continue numbering from existing cache to avoid overwrites on resume
    existing_pairs = sorted(noisy_dir.glob("pair_*.wav"))
    pair_count = len(existing_pairs)

    # Load noise profile (average of noise samples)
    if noise_files:
        noise_samples = []
        for nf in noise_files[:20]:  # Use first 20 noise files
            try:
                audio, sr = sf.read(nf)
                noise_samples.append(audio)
            except Exception:
                pass

        if noise_samples:
            # Create averaged noise spectrum for spectral subtraction
            min_len = min(len(n) for n in noise_samples)
            noise_avg = np.mean([n[:min_len] for n in noise_samples], axis=0)
        else:
            noise_avg = None
    else:
        noise_avg = None

    # Get active segments
    active_segments = [s for s in segments if not s.is_silent and s.output_path]

    files_processed = set()

    for seg in active_segments:
        if seg.output_path in files_processed:
            continue
        files_processed.add(seg.output_path)

        try:
            audio, sr = sf.read(seg.output_path)

            # Create multiple segments from this file
            segment_length = int(2.0 * sr)  # 2 second segments
            for i in range(min(pairs_per_file, len(audio) // segment_length)):
                start = i * segment_length
                end = start + segment_length
                segment = audio[start:end]

                if len(segment) < segment_length:
                    continue

                # Save noisy version (original)
                noisy_path = noisy_dir / f"pair_{pair_count:06d}.wav"
                sf.write(noisy_path, segment, sr)

                # Create "clean" target using spectral subtraction
                # For stationary noise, simple spectral subtraction works
                if noise_avg is not None and len(noise_avg) >= len(segment):
                    # Simple noise reduction using spectral subtraction
                    clean_segment = spectral_subtract(segment, noise_avg[: len(segment)], sr)
                else:
                    # Fallback: use Wiener filtering approximation
                    clean_segment = simple_wiener_filter(segment)

                clean_path = clean_dir / f"pair_{pair_count:06d}.wav"
                sf.write(clean_path, clean_segment, sr)

                pair_count += 1

        except Exception as e:
            print(f"Error creating pairs from {seg.output_path}: {e}")

    return pair_count


def spectral_subtract(noisy: np.ndarray, noise_estimate: np.ndarray, sr: int, alpha: float = 2.0) -> np.ndarray:
    """Simple spectral subtraction for noise reduction."""
    # FFT parameters
    n_fft = 2048
    hop = n_fft // 4

    # Pad signals
    pad_len = n_fft - (len(noisy) % n_fft)
    noisy_padded = np.pad(noisy, (0, pad_len))
    noise_padded = np.pad(noise_estimate, (0, pad_len))

    # Estimate noise spectrum
    noise_spec = np.abs(np.fft.rfft(noise_padded[:n_fft]))

    # Process in frames
    output = np.zeros_like(noisy_padded)
    window = np.hanning(n_fft)

    for i in range(0, len(noisy_padded) - n_fft, hop):
        frame = noisy_padded[i : i + n_fft] * window
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        phase = np.angle(spec)

        # Spectral subtraction
        mag_clean = np.maximum(mag - alpha * noise_spec, 0.1 * mag)

        # Reconstruct
        clean_spec = mag_clean * np.exp(1j * phase)
        clean_frame = np.fft.irfft(clean_spec)
        output[i : i + n_fft] += clean_frame * window

    # Normalize overlap-add
    output = output / (n_fft / hop / 2)

    return output[: len(noisy)]


def simple_wiener_filter(signal: np.ndarray, noise_floor: float = 0.01) -> np.ndarray:
    """Simple Wiener-like filtering."""
    n_fft = 2048
    hop = n_fft // 4

    pad_len = n_fft - (len(signal) % n_fft)
    padded = np.pad(signal, (0, pad_len))

    output = np.zeros_like(padded)
    window = np.hanning(n_fft)

    for i in range(0, len(padded) - n_fft, hop):
        frame = padded[i : i + n_fft] * window
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        phase = np.angle(spec)

        # Estimate SNR and apply Wiener gain
        snr = mag**2 / (noise_floor**2 + 1e-10)
        gain = snr / (snr + 1)
        mag_clean = mag * gain

        clean_spec = mag_clean * np.exp(1j * phase)
        clean_frame = np.fft.irfft(clean_spec)
        output[i : i + n_fft] += clean_frame * window

    output = output / (n_fft / hop / 2)
    return output[: len(signal)]


def create_manifest(
    output_dir: Path,
    stats: DatasetStats,
    segments: List[AudioSegment],
):
    """Create dataset manifest files."""
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Save stats
    with open(manifest_dir / "stats.json", "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    # Create training manifest
    pairs_dir = output_dir / "training_pairs"
    noisy_dir = pairs_dir / "noisy"
    clean_dir = pairs_dir / "clean"

    if noisy_dir.exists():
        noisy_files = sorted(noisy_dir.glob("*.wav"))
        manifest_lines = []
        for nf in noisy_files:
            cf = clean_dir / nf.name
            if cf.exists():
                manifest_lines.append(f"{nf}\t{cf}\n")

        with open(manifest_dir / "train.tsv", "w") as f:
            f.writelines(manifest_lines)

        print(f"Created manifest with {len(manifest_lines)} training pairs")

    # Save segment info for debugging
    segment_info = [
        {
            "source": s.source_file,
            "start": float(s.start_time),
            "end": float(s.end_time),
            "is_silent": bool(s.is_silent),
            "rms": float(s.rms_energy),
        }
        for s in segments[:1000]  # Limit for file size
    ]
    with open(manifest_dir / "segments_sample.json", "w") as f:
        json.dump(segment_info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Preprocess Wall camera footage for DFNetMF training")
    parser.add_argument(
        "--input",
        type=str,
        default="/Volumes/HomeSecurityVideos/Wall",
        help="Input directory with MP4 files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/Users/andrew/DataDump/datasets/wall_processed",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=48000,
        help="Target sample rate",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=2.0,
        help="Segment duration in seconds",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Wall Camera Dataset Preprocessing")
    print("=" * 60)

    # Check dependencies
    check_dependencies()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Find files
    print("\nFinding MP4 files...")
    mp4_files = find_mp4_files(input_dir, args.max_files)
    print(f"Found {len(mp4_files)} MP4 files")

    if not mp4_files:
        print("❌ No MP4 files found")
        sys.exit(1)

    # Initialize stats
    stats = DatasetStats(total_files=len(mp4_files))

    # Process files (parallel when workers > 1)
    print(f"\nProcessing files with {args.workers} worker(s)...")

    # Filter out already processed files to support resume
    to_process = []
    for mp4_path in mp4_files:
        out_wav_path = output_dir / "wav" / mp4_path.parent.name / f"{mp4_path.stem}.wav"
        if not out_wav_path.exists():
            to_process.append(mp4_path)

    all_segments: List[AudioSegment] = []
    temp_root = Path(tempfile.mkdtemp())

    try:
        if args.workers and args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        process_single_file,
                        mp4_path,
                        output_dir,
                        temp_root,
                        args.target_sr,
                        args.segment_duration,
                    )
                    for mp4_path in to_process
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing files",
                    unit="file",
                    dynamic_ncols=True,
                    smoothing=0.1,
                    mininterval=1.0,
                ):
                    result = future.result()
                    if result:
                        segments, duration = result
                        all_segments.extend(segments)
                        stats.processed_files += 1
                        stats.total_duration_seconds += duration
                        stats.total_segments += len(segments)
                        stats.silent_segments += sum(1 for s in segments if s.is_silent)
                        stats.active_segments += sum(1 for s in segments if not s.is_silent)
                    else:
                        stats.failed_files += 1
        else:
            for mp4_path in tqdm(
                to_process,
                desc="Processing files",
                unit="file",
                dynamic_ncols=True,
                smoothing=0.1,
                mininterval=1.0,
            ):
                result = process_single_file(
                    mp4_path,
                    output_dir,
                    temp_root,
                    args.target_sr,
                    args.segment_duration,
                )

                if result:
                    segments, duration = result
                    all_segments.extend(segments)
                    stats.processed_files += 1
                    stats.total_duration_seconds += duration
                    stats.total_segments += len(segments)
                    stats.silent_segments += sum(1 for s in segments if s.is_silent)
                    stats.active_segments += sum(1 for s in segments if not s.is_silent)
                else:
                    stats.failed_files += 1
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    print(f"\n✅ Processed {stats.processed_files}/{stats.total_files} files")
    print(f"   Total duration: {stats.total_duration_seconds / 3600:.2f} hours")
    print(f"   Total segments: {stats.total_segments}")
    print(f"   Silent (noise-only): {stats.silent_segments}")
    print(f"   Active (with signal): {stats.active_segments}")

    # Extract noise samples
    print("\nExtracting noise samples...")
    noise_dir = output_dir / "noise_samples"
    existing_noise = sorted(noise_dir.glob("noise_*.wav")) if noise_dir.exists() else []
    noise_files = [str(p) for p in existing_noise]
    noise_files.extend(extract_noise_samples(all_segments, output_dir, start_index=len(existing_noise)))
    stats.noise_profile_files = noise_files
    print(f"   Created {len(noise_files)} noise sample files")

    # Create training pairs
    print("\nCreating training pairs...")
    pair_count = create_training_pairs(all_segments, output_dir, noise_files)
    print(f"   Created {pair_count} training pairs")

    # Create manifest
    print("\nCreating manifests...")
    create_manifest(output_dir, stats, all_segments)

    print("\n" + "=" * 60)
    print("✅ Preprocessing complete!")
    print("=" * 60)
    print(f"\nDataset saved to: {output_dir}")
    print(f"Training pairs: {pair_count}")
    print("\nNext step: Run training with:")
    print(f"  python train_dfnetmf_wall.py --dataset {output_dir}")


if __name__ == "__main__":
    main()
