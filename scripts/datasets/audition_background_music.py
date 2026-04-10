#!/usr/bin/env python3
"""Render a small before/after audition pack for prepared background music.

This helper reuses the canonical DSP path from ``prepare_background_music.py``
so you can quickly listen to original vs prepared room/speaker/live-ish
variants before committing to a large cache build.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from prepare_background_music import (  # noqa: E402
    DEFAULT_PREPARE_SEED,
    DEFAULT_RIR_PROBABILITY,
    DEFAULT_STYLE,
    DEFAULT_VARIANTS_PER_SOURCE,
    STYLE_PRESETS,
    build_rng,
    describe_style,
    load_audio_file,
    load_rir_cached,
    normalize_peak,
    render_room_playback_variant,
    save_audio_file,
    select_rir_path,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "DeepFilterNet"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from df_mlx.file_lists import read_file_list  # noqa: E402

DEFAULT_AUDITION_COUNT = 6
DEFAULT_CLIP_SECONDS = 12.0
DEFAULT_GAP_SECONDS = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a small audition pack for prepared background music")
    parser.add_argument("--file-list", required=True, help="Input music file list")
    parser.add_argument("--output-dir", required=True, help="Directory for audition outputs")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="Target sample rate for rendered files")
    parser.add_argument(
        "--style",
        choices=tuple(STYLE_PRESETS.keys()),
        default=DEFAULT_STYLE,
        help=f"Playback-style preset used for prepared variants (default: {DEFAULT_STYLE})",
    )
    parser.add_argument("--rir-list", type=str, default=None, help="Optional RIR file list used while rendering")
    parser.add_argument(
        "--num-sources",
        type=int,
        default=DEFAULT_AUDITION_COUNT,
        help=f"How many source tracks to audition (default: {DEFAULT_AUDITION_COUNT})",
    )
    parser.add_argument(
        "--variants-per-source",
        type=int,
        default=DEFAULT_VARIANTS_PER_SOURCE,
        help=f"Prepared variants to render per source (default: {DEFAULT_VARIANTS_PER_SOURCE})",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=DEFAULT_CLIP_SECONDS,
        help=f"Max seconds per audition clip (default: {DEFAULT_CLIP_SECONDS}; use 0 for full file)",
    )
    parser.add_argument(
        "--rir-probability",
        type=float,
        default=DEFAULT_RIR_PROBABILITY,
        help=f"Probability of using an RIR for a prepared variant (default: {DEFAULT_RIR_PROBABILITY})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_PREPARE_SEED,
        help=f"Base seed for deterministic source selection/rendering (default: {DEFAULT_PREPARE_SEED})",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=DEFAULT_GAP_SECONDS,
        help=f"Silence gap between original and prepared in comparison clips (default: {DEFAULT_GAP_SECONDS})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite audition outputs even if they already exist")
    return parser.parse_args()


def choose_sources(source_paths: list[Path], num_sources: int, seed: int) -> list[Path]:
    if num_sources < 1:
        raise SystemExit("--num-sources must be >= 1")
    rng = np.random.default_rng(seed)
    ordered = list(source_paths)
    if len(ordered) > 1:
        rng.shuffle(ordered)
    return sorted(ordered[: min(num_sources, len(ordered))], key=lambda path: str(path))


def sanitize_label(path: Path) -> str:
    stem = path.stem.lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    return cleaned or "sample"


def extract_audition_clip(
    audio: np.ndarray,
    sample_rate: int,
    clip_seconds: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    normalized = normalize_peak(np.asarray(audio, dtype=np.float32))
    if clip_seconds <= 0:
        return normalized, 0
    max_samples = max(1, int(round(sample_rate * clip_seconds)))
    if normalized.shape[0] <= max_samples:
        return normalized, 0
    max_start = normalized.shape[0] - max_samples
    start = int(rng.integers(0, max_start + 1))
    end = start + max_samples
    return np.asarray(normalized[start:end], dtype=np.float32), start


def build_comparison_clip(
    original: np.ndarray, prepared: np.ndarray, sample_rate: int, gap_seconds: float
) -> np.ndarray:
    gap_samples = max(0, int(round(sample_rate * gap_seconds)))
    gap = np.zeros(gap_samples, dtype=np.float32)
    return np.asarray(np.concatenate([original, gap, prepared]), dtype=np.float32)


def build_audition_entry_dir(output_dir: Path, sample_index: int, source: Path) -> Path:
    return output_dir / f"{sample_index:02d}_{sanitize_label(source)}"


def write_manifest(
    output_dir: Path,
    manifest: list[dict[str, object]],
    *,
    style: str,
    style_description: str,
) -> None:
    manifest_path = output_dir / "audition_manifest.json"
    temp_path = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": 1,
                "style": style,
                "style_description": style_description,
                "samples": manifest,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    temp_path.replace(manifest_path)


def write_readme(
    output_dir: Path,
    manifest: list[dict[str, object]],
    *,
    sample_rate: int,
    clip_seconds: float,
    style: str,
    style_description: str,
) -> None:
    readme_path = output_dir / "README.md"
    lines = [
        "# Background Music Audition Pack",
        "",
        f"- Sample rate: {sample_rate} Hz",
        f"- Max clip seconds: {clip_seconds}",
        f"- Style preset: `{style}` — {style_description}",
        "- Each sample directory contains:",
        "  - `original.wav`",
        "  - `prepared_vXX.wav`",
        "  - `compare_vXX.wav` (original, short silence, then prepared)",
        "",
        "## Samples",
        "",
    ]
    for sample in manifest:
        lines.append(f"### {sample['sample_dir']}")
        lines.append(f"- Source: `{sample['source_path']}`")
        lines.append(f"- Clip start sample: {sample['clip_start_sample']}")
        for variant in sample["variants"]:
            lines.append(
                f"- Variant {variant['variant_index']:02d}: "
                f"`{variant['prepared_path']}` / `{variant['comparison_path']}`"
            )
        lines.append("")
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.variants_per_source < 1:
        raise SystemExit("--variants-per-source must be >= 1")
    if not (0.0 <= args.rir_probability <= 1.0):
        raise SystemExit("--rir-probability must be in [0, 1]")
    if args.gap_seconds < 0.0:
        raise SystemExit("--gap-seconds must be >= 0")

    file_list = Path(args.file_list).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = [Path(path).expanduser().resolve() for path in read_file_list(file_list, check_exists=True)]
    if not source_paths:
        raise SystemExit(f"No music files found in {file_list}")

    chosen_sources = choose_sources(source_paths, args.num_sources, args.seed)
    rir_paths: list[Path] = []
    if args.rir_list:
        rir_paths = [Path(path).expanduser().resolve() for path in read_file_list(args.rir_list, check_exists=True)]

    style_description = describe_style(args.style)
    rir_cache: dict[Path, np.ndarray] = {}
    manifest: list[dict[str, object]] = []

    print(f"[info] Rendering audition pack from {len(chosen_sources):,} source track(s)")
    print(f"[info] Style preset: {args.style} — {style_description}")
    for sample_index, source in enumerate(chosen_sources):
        sample_dir = build_audition_entry_dir(output_dir, sample_index, source)
        sample_dir.mkdir(parents=True, exist_ok=True)

        source_audio = load_audio_file(str(source), args.sample_rate)
        excerpt_rng = build_rng(source, -1, args.seed)
        excerpt_audio, clip_start_sample = extract_audition_clip(
            source_audio,
            args.sample_rate,
            args.clip_seconds,
            excerpt_rng,
        )

        original_path = sample_dir / "original.wav"
        if args.overwrite or not original_path.exists():
            save_audio_file(original_path, excerpt_audio, args.sample_rate)

        variant_entries: list[dict[str, object]] = []
        for variant_idx in range(args.variants_per_source):
            prepared_path = sample_dir / f"prepared_v{variant_idx:02d}.wav"
            comparison_path = sample_dir / f"compare_v{variant_idx:02d}.wav"

            if args.overwrite or not prepared_path.exists() or not comparison_path.exists():
                rng = build_rng(source, variant_idx, args.seed)
                rir_audio = None
                if rir_paths and rng.random() < args.rir_probability:
                    rir_path = select_rir_path(rir_paths, rng)
                    if rir_path is not None:
                        rir_audio = load_rir_cached(rir_path, args.sample_rate, rir_cache)
                prepared_audio = render_room_playback_variant(
                    excerpt_audio,
                    args.sample_rate,
                    rng,
                    style=args.style,
                    rir_audio=rir_audio,
                )
                comparison_audio = build_comparison_clip(
                    excerpt_audio,
                    prepared_audio,
                    args.sample_rate,
                    args.gap_seconds,
                )
                save_audio_file(prepared_path, prepared_audio, args.sample_rate)
                save_audio_file(comparison_path, comparison_audio, args.sample_rate)

            variant_entries.append(
                {
                    "variant_index": variant_idx,
                    "prepared_path": str(prepared_path),
                    "comparison_path": str(comparison_path),
                }
            )

        manifest.append(
            {
                "sample_dir": sample_dir.name,
                "source_path": str(source),
                "original_path": str(original_path),
                "clip_start_sample": clip_start_sample,
                "variants": variant_entries,
            }
        )

    write_manifest(output_dir, manifest, style=args.style, style_description=style_description)
    write_readme(
        output_dir,
        manifest,
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        style=args.style,
        style_description=style_description,
    )
    print(f"[ok] wrote audition pack -> {output_dir}")
    print(f"[info] manifest -> {output_dir / 'audition_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
