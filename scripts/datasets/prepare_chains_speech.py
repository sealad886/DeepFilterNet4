#!/usr/bin/env python3
"""Prepare CHAINS clean-speech inputs for MLX datastore building.

This helper scans the CHAINS corpus, keeps the mono speaking styles as-is,
and extracts the speaker channel from the stereo RSI recordings into a stable,
resumable output tree. It then writes a deterministic file list that can be
merged into the clean-speech list used by ``build_mlx_datastore.sh``.
"""

from __future__ import annotations

import argparse
import os
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

MONO_STYLES = ("fast", "retell", "solo", "sync", "whsp")
RSI_STYLE = "rsi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CHAINS clean-speech sources for MLX datastore builds.")
    parser.add_argument("--chains-dir", required=True, help="Root of the CHAINS corpus.")
    parser.add_argument(
        "--prepared-root",
        required=True,
        help="Directory where generated RSI subject-channel wav files are written.",
    )
    parser.add_argument(
        "--output-list",
        required=True,
        help="Path to write the prepared CHAINS clean-speech list.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild extracted RSI subject-channel files even when resumable outputs already exist.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Max parallel workers for RSI extraction (default: min(4, cpu_count)).",
    )
    return parser.parse_args()


def write_output_list(paths: list[Path], output_list: Path) -> None:
    output_list.parent.mkdir(parents=True, exist_ok=True)
    temp_output_list = output_list.with_name(f"{output_list.name}.tmp.{os.getpid()}")
    with temp_output_list.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{path}\n")
    temp_output_list.replace(output_list)


def list_style_files(style_root: Path) -> list[Path]:
    return sorted(path.resolve() for path in style_root.rglob("*.wav") if path.is_file())


def read_wave_header(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as handle:
        return (
            handle.getnchannels(),
            handle.getsampwidth(),
            handle.getframerate(),
            handle.getnframes(),
        )


def validate_mono_files(style: str, files: list[Path]) -> list[Path]:
    validated: list[Path] = []
    for path in files:
        channels, _, _, frames = read_wave_header(path)
        if channels != 1:
            raise SystemExit(f"Expected mono CHAINS {style} file, found {channels} channels: {path}")
        if frames <= 0:
            raise SystemExit(f"CHAINS {style} file contains no frames: {path}")
        validated.append(path)
    return validated


def is_complete_output(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def build_rsi_output_path(source: Path, rsi_root: Path, prepared_root: Path) -> Path:
    relative = source.relative_to(rsi_root)
    return prepared_root / "rsi_subject" / relative


def should_refresh_output(source: Path, target: Path, *, overwrite: bool) -> bool:
    if overwrite or not is_complete_output(target):
        return True
    try:
        return target.stat().st_mtime_ns < source.stat().st_mtime_ns
    except OSError:
        return True


def _dtype_for_sample_width(sample_width: int):
    mapping = {
        1: np.dtype(np.uint8),
        2: np.dtype("<i2"),
        4: np.dtype("<i4"),
    }
    dtype = mapping.get(sample_width)
    if dtype is None:
        raise SystemExit(f"Unsupported CHAINS sample width {sample_width} bytes")
    return dtype


def extract_rsi_subject_channel(source: Path, target: Path) -> None:
    with wave.open(str(source), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
        raw_frames = reader.readframes(frame_count)

    if channels != 2:
        raise SystemExit(f"Expected stereo CHAINS RSI file, found {channels} channels: {source}")
    if frame_count <= 0:
        raise SystemExit(f"CHAINS RSI file contains no frames: {source}")

    dtype = _dtype_for_sample_width(sample_width)
    audio = np.frombuffer(raw_frames, dtype=dtype)
    if audio.size != frame_count * channels:
        raise SystemExit(f"Unexpected RSI frame layout for {source}")
    subject_channel = np.ascontiguousarray(audio.reshape(frame_count, channels)[:, 1])

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    if temp_target.exists():
        temp_target.unlink()
    try:
        with wave.open(str(temp_target), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(sample_width)
            writer.setframerate(sample_rate)
            writer.writeframes(subject_channel.tobytes())
        temp_target.replace(target)
    except Exception:
        if temp_target.exists():
            temp_target.unlink()
        raise


def require_style_root(chains_dir: Path, style: str) -> Path:
    style_root = chains_dir / style / "data" / style
    if not style_root.is_dir():
        raise SystemExit(f"Expected CHAINS style directory not found: {style_root}")
    return style_root


def main() -> int:
    args = parse_args()

    chains_dir = Path(args.chains_dir).expanduser().resolve()
    prepared_root = Path(args.prepared_root).expanduser().resolve()
    output_list = Path(args.output_list).expanduser().resolve()

    if not chains_dir.is_dir():
        raise SystemExit(f"CHAINS corpus root not found: {chains_dir}")

    mono_files: list[Path] = []
    mono_counts: dict[str, int] = {}
    for style in MONO_STYLES:
        style_root = require_style_root(chains_dir, style)
        style_files = validate_mono_files(style, list_style_files(style_root))
        mono_counts[style] = len(style_files)
        mono_files.extend(style_files)

    rsi_root = require_style_root(chains_dir, RSI_STYLE)
    rsi_sources = list_style_files(rsi_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    rsi_outputs: list[Path] = []
    rsi_rebuilt = 0
    rsi_reused = 0

    def _extract_one_rsi(source: Path) -> tuple[Path, str]:
        target = build_rsi_output_path(source, rsi_root, prepared_root).resolve()
        if should_refresh_output(source, target, overwrite=args.overwrite):
            extract_rsi_subject_channel(source, target)
            return target, "rebuilt"
        return target, "reused"

    num_workers = max(1, args.num_workers)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for target, status in pool.map(_extract_one_rsi, rsi_sources):
            rsi_outputs.append(target)
            if status == "rebuilt":
                rsi_rebuilt += 1
            else:
                rsi_reused += 1

    all_paths = sorted({path.resolve() for path in (*mono_files, *rsi_outputs)}, key=str)
    write_output_list(all_paths, output_list)

    print("=" * 60)
    print("CHAINS Clean Speech Preparation")
    print("=" * 60)
    print(f"CHAINS dir:       {chains_dir}")
    print(f"Prepared root:    {prepared_root}")
    print(f"Output list:      {output_list}")
    for style in MONO_STYLES:
        print(f"{style:16}{mono_counts[style]:6d} mono files")
    print(f"rsi stereo files:{len(rsi_sources):6d}")
    print(f"RSI extracted:   {rsi_rebuilt:6d}")
    print(f"RSI reused:      {rsi_reused:6d}")
    print(f"Total speech:    {len(all_paths):6d}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
