#!/usr/bin/env python3
"""Batch video audio enhancement using DeepFilterNet MLX models.

This script extracts audio from video files, enhances it using trained MLX models,
and encodes the result to AAC format for use as sidecar audio tracks.

Features:
- Recursive video discovery (.mp4, .mov)
- FFmpeg-based audio extraction and AAC encoding
- MLX model audio enhancement
- Progress tracking with tqdm
- Error logging and recovery
- Enhancement manifest generation for playlist integration
- Skip-existing logic to avoid re-processing

Usage:
    # Basic enhancement
    python enhance_video_audio.py \\
        --video-dir /Volumes/HomeSecurityVideos/Wall \\
        --model-checkpoint ./checkpoints/best.safetensors

    # With custom output format and batch processing
    python enhance_video_audio.py \\
        --video-dir /Volumes/HomeSecurityVideos/Blink \\
        --model-checkpoint ./checkpoints/dfnet4_dynamic/best.safetensors \\
        --output-format aac \\
        --batch-size 8 \\
        --force

Example:
    $ python enhance_video_audio.py --video-dir ~/Videos --model-checkpoint model.safetensors
    Discovering videos in ~/Videos...
    Found 120 videos (15 already enhanced, 105 to process)
    Processing videos: 100%|███████████| 105/105 [2:15:30<00:00, 77.43s/video]
    ✅ Enhancement complete: 105 succeeded, 0 failed
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Add DeepFilterNet to path
sys.path.insert(0, str(Path(__file__).parent.parent / "DeepFilterNet"))

try:
    import mlx.core as mx

    from df_mlx.enhance import enhance_audio, load_model
except ImportError:
    print("❌ Error: MLX and df_mlx modules not found")
    print("Make sure you're running from the repository root and MLX is installed")
    sys.exit(1)


def setup_logging(log_file: Path = Path("errors.log")) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_file: Path to error log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("video_enhancement")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # File handler for errors
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def discover_videos(
    video_dir: Path,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Discover video files recursively in the given directory.

    Args:
        video_dir: Root directory to search for videos
        force: If True, include videos that already have enhanced audio
        logger: Optional logger for output

    Returns:
        List of video file paths to process
    """
    if logger:
        logger.info(f"Discovering videos in {video_dir}...")

    video_extensions = {".mp4", ".mov"}
    all_videos = []

    # Recursively find all video files
    for ext in video_extensions:
        all_videos.extend(video_dir.rglob(f"*{ext}"))

    if not all_videos:
        if logger:
            logger.warning(f"No video files found in {video_dir}")
        return []

    # Filter out videos that already have enhanced audio (unless force=True)
    to_process = []
    skipped = 0

    for video_path in all_videos:
        enhanced_audio_path = video_path.with_name(f"{video_path.stem}_enhanced.m4a")
        if enhanced_audio_path.exists() and not force:
            skipped += 1
        else:
            to_process.append(video_path)

    if logger:
        logger.info(f"Found {len(all_videos)} videos " f"({skipped} already enhanced, {len(to_process)} to process)")

    return to_process


def extract_audio(
    video_path: Path,
    output_wav: Path,
    sample_rate: int = 48000,
    logger: logging.Logger | None = None,
) -> tuple[bool, str]:
    """Extract audio from video file to PCM WAV format.

    Args:
        video_path: Path to input video file
        output_wav: Path to output WAV file
        sample_rate: Target sample rate (default: 48000 Hz)
        logger: Optional logger for errors

    Returns:
        (success, error_message) tuple
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-y",  # Overwrite output
            str(output_wav),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            error_msg = f"FFmpeg extraction failed: {result.stderr[-200:]}"
            if logger:
                logger.error(f"{video_path}: {error_msg}")
            return False, error_msg

        # Verify output file exists and is not empty
        if not output_wav.exists() or output_wav.stat().st_size == 0:
            error_msg = "Extracted audio file is empty or missing"
            if logger:
                logger.error(f"{video_path}: {error_msg}")
            return False, error_msg

        return True, ""

    except subprocess.TimeoutExpired:
        error_msg = "Audio extraction timeout (>5 minutes)"
        if logger:
            logger.error(f"{video_path}: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Audio extraction error: {str(e)}"
        if logger:
            logger.error(f"{video_path}: {error_msg}")
        return False, error_msg


def enhance_audio_file(
    input_wav: Path,
    output_wav: Path,
    model: Any,
    logger: logging.Logger | None = None,
) -> tuple[bool, str]:
    """Enhance audio file using MLX model.

    Args:
        input_wav: Path to input WAV file
        output_wav: Path to output enhanced WAV file
        model: Loaded MLX model
        logger: Optional logger for errors

    Returns:
        (success, error_message) tuple
    """
    try:
        # Load audio
        audio, sr = sf.read(input_wav)

        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Handle mono/stereo
        if audio.ndim == 1:
            # Mono audio
            audio_mx = mx.array(audio)
            enhanced_mx = enhance_audio(model, audio_mx, sr)
            enhanced = np.array(enhanced_mx)
        else:
            # Stereo/multi-channel - process each channel
            enhanced_channels = []
            for ch in range(audio.shape[1]):
                audio_mx = mx.array(audio[:, ch])
                enhanced_mx = enhance_audio(model, audio_mx, sr)
                enhanced_channels.append(np.array(enhanced_mx))
            enhanced = np.stack(enhanced_channels, axis=1)

        # Save enhanced audio
        sf.write(output_wav, enhanced, sr)

        return True, ""

    except Exception as e:
        error_msg = f"Audio enhancement error: {str(e)}"
        if logger:
            logger.error(f"{input_wav}: {error_msg}")
        return False, error_msg


def encode_aac(
    input_wav: Path,
    output_m4a: Path,
    bitrate: str = "192k",
    logger: logging.Logger | None = None,
) -> tuple[bool, str]:
    """Encode WAV audio to AAC format.

    Args:
        input_wav: Path to input WAV file
        output_m4a: Path to output M4A file
        bitrate: AAC bitrate (default: 192k)
        logger: Optional logger for errors

    Returns:
        (success, error_message) tuple
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            str(input_wav),
            "-c:a",
            "aac",
            "-b:a",
            bitrate,
            "-y",  # Overwrite output
            str(output_m4a),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        if result.returncode != 0:
            error_msg = f"FFmpeg AAC encoding failed: {result.stderr[-200:]}"
            if logger:
                logger.error(f"{input_wav}: {error_msg}")
            return False, error_msg

        # Verify output file exists and is not empty
        if not output_m4a.exists() or output_m4a.stat().st_size == 0:
            error_msg = "Encoded AAC file is empty or missing"
            if logger:
                logger.error(f"{input_wav}: {error_msg}")
            return False, error_msg

        return True, ""

    except subprocess.TimeoutExpired:
        error_msg = "AAC encoding timeout (>2 minutes)"
        if logger:
            logger.error(f"{input_wav}: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"AAC encoding error: {str(e)}"
        if logger:
            logger.error(f"{input_wav}: {error_msg}")
        return False, error_msg


def update_manifest(
    manifest_path: Path,
    video_path: Path,
    enhanced_audio_path: Path,
) -> None:
    """Update enhancement manifest with new entry.

    Args:
        manifest_path: Path to manifest JSON file
        video_path: Path to video file
        enhanced_audio_path: Path to enhanced audio file
    """
    # Load existing manifest or create new one
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"videos": []}

    # Add or update entry
    entry = {
        "video_path": str(video_path.absolute()),
        "video_path_relative": str(video_path.relative_to(video_path.parent.parent)),
        "enhanced_audio_path": str(enhanced_audio_path.absolute()),
        "enhanced_audio_path_relative": str(enhanced_audio_path.relative_to(enhanced_audio_path.parent.parent)),
        "timestamp": datetime.now().isoformat(),
    }

    # Check if entry already exists (update if so)
    existing_idx = None
    for idx, existing_entry in enumerate(manifest["videos"]):
        if existing_entry["video_path"] == entry["video_path"]:
            existing_idx = idx
            break

    if existing_idx is not None:
        manifest["videos"][existing_idx] = entry
    else:
        manifest["videos"].append(entry)

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def process_video(
    video_path: Path,
    model: Any,
    output_format: str,
    manifest_path: Path | None,
    logger: logging.Logger,
) -> tuple[bool, str]:
    """Process a single video: extract, enhance, encode.

    Args:
        video_path: Path to video file
        model: Loaded MLX enhancement model
        output_format: Output audio format ('aac' or 'wav')
        manifest_path: Optional path to enhancement manifest
        logger: Logger instance

    Returns:
        (success, error_message) tuple
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Extract audio
        extracted_wav = tmpdir / "extracted.wav"
        success, error = extract_audio(video_path, extracted_wav, logger=logger)
        if not success:
            return False, f"Extraction failed: {error}"

        # Step 2: Enhance audio
        enhanced_wav = tmpdir / "enhanced.wav"
        success, error = enhance_audio_file(extracted_wav, enhanced_wav, model, logger=logger)
        if not success:
            return False, f"Enhancement failed: {error}"

        # Step 3: Encode to output format
        if output_format == "aac":
            output_path = video_path.with_name(f"{video_path.stem}_enhanced.m4a")
            success, error = encode_aac(enhanced_wav, output_path, logger=logger)
            if not success:
                return False, f"Encoding failed: {error}"
        else:  # wav
            output_path = video_path.with_name(f"{video_path.stem}_enhanced.wav")
            enhanced_wav.rename(output_path)

        # Step 4: Update manifest
        if manifest_path:
            update_manifest(manifest_path, video_path, output_path)

        return True, ""


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch enhance video audio using DeepFilterNet MLX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Directory containing videos to process (searched recursively)",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        required=True,
        help="Path to trained MLX model checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--output-format",
        choices=["aac", "wav"],
        default="aac",
        help="Output audio format (default: aac)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for enhancement (default: 1, currently unused)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process videos that already have enhanced audio",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="Path to enhancement manifest JSON (default: video_dir/enhancement_manifest.json)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("errors.log"),
        help="Path to error log file (default: errors.log)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.video_dir.exists():
        print(f"❌ Error: Video directory not found: {args.video_dir}")
        return 1

    if not args.model_checkpoint.exists():
        print(f"❌ Error: Model checkpoint not found: {args.model_checkpoint}")
        return 1

    # Setup logging
    logger = setup_logging(args.log_file)

    # Default manifest path
    if args.manifest_path is None:
        args.manifest_path = args.video_dir / "enhancement_manifest.json"

    logger.info("=" * 60)
    logger.info("Video Audio Enhancement Pipeline")
    logger.info("=" * 60)
    logger.info(f"Video directory: {args.video_dir}")
    logger.info(f"Model checkpoint: {args.model_checkpoint}")
    logger.info(f"Output format: {args.output_format}")
    logger.info(f"Manifest: {args.manifest_path}")
    logger.info("")

    # Load model
    logger.info("Loading MLX model...")
    try:
        model = load_model(str(args.model_checkpoint))
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return 1

    # Discover videos
    videos = discover_videos(args.video_dir, force=args.force, logger=logger)
    if not videos:
        logger.info("No videos to process")
        return 0

    logger.info("")

    # Process videos with progress bar
    succeeded = 0
    failed = 0
    errors = []

    with tqdm(videos, desc="Processing videos", unit="video") as pbar:
        for video_path in pbar:
            pbar.set_postfix_str(video_path.name[:40])

            success, error = process_video(
                video_path,
                model,
                args.output_format,
                args.manifest_path,
                logger,
            )

            if success:
                succeeded += 1
            else:
                failed += 1
                errors.append((video_path, error))

            pbar.set_postfix(succeeded=succeeded, failed=failed)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Enhancement Complete")
    logger.info("=" * 60)
    logger.info(f"Total processed: {len(videos)}")
    logger.info(f"Succeeded: {succeeded}")
    logger.info(f"Failed: {failed}")

    if failed > 0:
        logger.info(f"\nErrors logged to: {args.log_file}")
        logger.info("\nFailed videos:")
        for video_path, error in errors[:10]:  # Show first 10
            logger.info(f"  - {video_path.name}: {error}")
        if len(errors) > 10:
            logger.info(f"  ... and {len(errors) - 10} more (see {args.log_file})")

    if succeeded > 0:
        logger.info(f"\nEnhancement manifest: {args.manifest_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
