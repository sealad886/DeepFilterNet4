#!/usr/bin/env python3
"""Render deterministic room/speaker/live-style variants from a music file list.

This script is an optional pre-step for ``build_mlx_datastore.sh``. It mirrors
relative source paths under ``--base-dir`` and writes one or more synthetic
"dirty" variants per source file. The transforms intentionally bias toward
speaker-in-room playback rather than studio-clean music by combining:

- bandwidth limiting and resample round-trips,
- optional room impulse response convolution,
- mild saturation/clipping,
- low-level hiss/hum,
- deterministic per-file randomization.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "DeepFilterNet"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from df_mlx._audio_io import load_audio_file, resample_audio  # noqa: E402
from df_mlx.file_lists import read_file_list  # noqa: E402
from df_mlx.prepare_data import apply_rir  # noqa: E402

try:  # pragma: no branch - exercised in normal runtime
    import soundfile as sf
except ImportError:  # pragma: no cover - fallback only used in minimal envs
    sf = None
    from scipy.io import wavfile


DEFAULT_PREPARE_SEED = 1337
DEFAULT_VARIANTS_PER_SOURCE = 2
DEFAULT_RIR_PROBABILITY = 0.8
SPEAKER_RESAMPLE_TARGETS = (12_000, 16_000, 18_000, 22_050, 24_000, 32_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare degraded background-music variants for MLX datastore builds")
    parser.add_argument("--file-list", required=True, help="Input background-music file list")
    parser.add_argument("--output-root", required=True, help="Root directory for prepared music variants")
    parser.add_argument("--base-dir", required=True, help="Base directory used to mirror relative source paths")
    parser.add_argument("--output-list", required=True, help="Output text file listing prepared variants")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="Target sample rate for prepared outputs")
    parser.add_argument(
        "--rir-list", type=str, default=None, help="Optional RIR file list used for room-playback rendering"
    )
    parser.add_argument(
        "--variants-per-source",
        type=int,
        default=DEFAULT_VARIANTS_PER_SOURCE,
        help=f"Prepared variants per source file (default: {DEFAULT_VARIANTS_PER_SOURCE})",
    )
    parser.add_argument(
        "--rir-probability",
        type=float,
        default=DEFAULT_RIR_PROBABILITY,
        help=f"Probability of applying an RIR when one is available (default: {DEFAULT_RIR_PROBABILITY})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_PREPARE_SEED,
        help=f"Base seed for deterministic per-file rendering (default: {DEFAULT_PREPARE_SEED})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild prepared files even if they already exist")
    return parser.parse_args()


def write_output_list(paths: list[Path], output_list: Path) -> None:
    output_list.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_list.with_name(f"{output_list.name}.tmp.{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{path}\n")
    temp_path.replace(output_list)


def build_output_path(source: Path, output_root: Path, base_dir: Path, variant_idx: int) -> Path:
    try:
        relative = source.relative_to(base_dir)
    except ValueError:
        relative = Path("_external") / source.name
    suffix_label = source.suffix.lower().lstrip(".") or "audio"
    variant_dir = output_root / relative.parent / f"{relative.stem}__{suffix_label}"
    return variant_dir / f"{relative.stem}.speaker_room_v{variant_idx:02d}.wav"


def build_rng(source: Path, variant_idx: int, seed: int) -> np.random.Generator:
    material = f"{source.resolve()}::{variant_idx}::{seed}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    derived_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(derived_seed)


def select_rir_path(rir_paths: list[Path], rng: np.random.Generator) -> Path | None:
    if not rir_paths:
        return None
    return rir_paths[int(rng.integers(0, len(rir_paths)))]


def load_rir_cached(path: Path, sample_rate: int, cache: dict[Path, np.ndarray]) -> np.ndarray | None:
    if path in cache:
        return cache[path]
    try:
        rir = load_audio_file(str(path), sample_rate)
    except Exception as exc:  # pragma: no cover - operational safeguard
        print(f"[warn] Failed to load RIR {path}: {exc}", file=sys.stderr)
        return None
    cache[path] = rir
    return rir


def apply_filter_safely(audio: np.ndarray, sos: np.ndarray) -> np.ndarray:
    if audio.size < 32:
        return np.asarray(scipy_signal.sosfilt(sos, audio), dtype=np.float32)
    try:
        return np.asarray(scipy_signal.sosfiltfilt(sos, audio), dtype=np.float32)
    except ValueError:
        return np.asarray(scipy_signal.sosfilt(sos, audio), dtype=np.float32)


def apply_bandwidth_shape(audio: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    shaped = np.asarray(audio, dtype=np.float32)
    nyquist_margin = max(400.0, sample_rate / 2.0 - 500.0)
    low_cut_hz = float(rng.uniform(90.0, min(220.0, sample_rate / 8.0)))
    high_cut_hz = float(rng.uniform(2800.0, min(7800.0, nyquist_margin)))
    if high_cut_hz <= low_cut_hz + 200.0:
        high_cut_hz = min(sample_rate / 2.0 - 200.0, low_cut_hz + 400.0)

    hp_sos = scipy_signal.butter(2, low_cut_hz, btype="highpass", fs=sample_rate, output="sos")
    lp_sos = scipy_signal.butter(4, high_cut_hz, btype="lowpass", fs=sample_rate, output="sos")
    shaped = apply_filter_safely(shaped, hp_sos)
    shaped = apply_filter_safely(shaped, lp_sos)
    return np.asarray(shaped, dtype=np.float32)


def apply_speaker_resample_roundtrip(audio: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    lower_rates = [rate for rate in SPEAKER_RESAMPLE_TARGETS if rate < sample_rate]
    if not lower_rates:
        return np.asarray(audio, dtype=np.float32)
    intermediate_sr = int(lower_rates[int(rng.integers(0, len(lower_rates)))])
    degraded = resample_audio(audio, sample_rate, intermediate_sr)
    return np.asarray(resample_audio(degraded, intermediate_sr, sample_rate), dtype=np.float32)


def add_playback_noise(audio: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    noisy = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(noisy**2) + 1e-8))
    if rms <= 1e-7:
        return noisy

    hiss = rng.standard_normal(noisy.shape[0]).astype(np.float32)
    hiss_std = float(np.std(hiss) + 1e-8)
    hiss_db = float(rng.uniform(-36.0, -24.0))
    hiss_scale = rms * (10.0 ** (hiss_db / 20.0)) / hiss_std
    noisy = noisy + hiss * hiss_scale

    if rng.random() < 0.6:
        hum_freq = 50.0 if rng.random() < 0.5 else 60.0
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        timeline = np.arange(noisy.shape[0], dtype=np.float32) / float(max(1, sample_rate))
        hum = np.sin((2.0 * math.pi * hum_freq * timeline) + phase)
        hum += 0.35 * np.sin((2.0 * math.pi * hum_freq * 2.0 * timeline) + phase)
        hum = hum.astype(np.float32)
        hum_std = float(np.std(hum) + 1e-8)
        hum_db = float(rng.uniform(-40.0, -28.0))
        hum_scale = rms * (10.0 ** (hum_db / 20.0)) / hum_std
        noisy = noisy + hum * hum_scale

    return np.asarray(noisy, dtype=np.float32)


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        return np.asarray(audio * (0.99 / peak), dtype=np.float32)
    return np.asarray(audio, dtype=np.float32)


def render_room_playback_variant(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    rir_audio: np.ndarray | None = None,
) -> np.ndarray:
    original = normalize_peak(np.asarray(audio, dtype=np.float32))
    processed = original.copy()

    input_gain_db = float(rng.uniform(-2.0, 2.0))
    processed = processed * (10.0 ** (input_gain_db / 20.0))

    if rng.random() < 0.85:
        processed = apply_speaker_resample_roundtrip(processed, sample_rate, rng)

    processed = apply_bandwidth_shape(processed, sample_rate, rng)

    if rir_audio is not None:
        reverbed = apply_rir(processed, rir_audio)
        wet_mix = float(rng.uniform(0.55, 0.9))
        processed = ((wet_mix * reverbed) + ((1.0 - wet_mix) * processed)).astype(np.float32)

    drive = float(rng.uniform(1.1, 2.8))
    processed = np.tanh(processed * drive) / np.tanh(drive)

    if rng.random() < 0.7:
        clip_level = float(rng.uniform(0.35, 0.8))
        processed = np.clip(processed, -clip_level, clip_level) / clip_level

    processed = add_playback_noise(processed, sample_rate, rng)

    dry_blend = float(rng.uniform(0.08, 0.28))
    processed = ((1.0 - dry_blend) * processed) + (dry_blend * original)
    return normalize_peak(np.asarray(processed, dtype=np.float32))


def save_audio_file(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sf is not None:
        sf.write(path, audio, sample_rate, subtype="PCM_16")
        return
    int_audio = np.clip(audio, -1.0, 1.0)
    int_audio = (int_audio * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, int_audio)  # type: ignore[name-defined]


def is_complete_output(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def main() -> int:
    args = parse_args()
    if args.variants_per_source < 1:
        raise SystemExit("--variants-per-source must be >= 1")
    if not (0.0 <= args.rir_probability <= 1.0):
        raise SystemExit("--rir-probability must be in [0, 1]")

    input_list = Path(args.file_list).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    output_list = Path(args.output_list).expanduser().resolve()

    source_paths = [Path(path).expanduser().resolve() for path in read_file_list(input_list, check_exists=True)]
    if not source_paths:
        raise SystemExit(f"No music files found in {input_list}")

    rir_paths: list[Path] = []
    if args.rir_list:
        rir_paths = [Path(path).expanduser().resolve() for path in read_file_list(args.rir_list, check_exists=True)]

    prepared_paths: list[Path] = []
    rir_cache: dict[Path, np.ndarray] = {}

    print(
        f"[info] Preparing {len(source_paths):,} music sources into "
        f"{args.variants_per_source} room-playback variant(s) each"
    )
    if rir_paths:
        print(f"[info] Loaded RIR candidate list: {len(rir_paths):,} entries (p={args.rir_probability:.2f})")
    else:
        print("[info] No RIR list provided; using speaker/EQ/noise degradations only")

    processed_count = 0
    reused_count = 0
    for source in source_paths:
        audio = None
        for variant_idx in range(args.variants_per_source):
            output_path = build_output_path(source, output_root, base_dir, variant_idx)
            prepared_paths.append(output_path)
            if not args.overwrite and is_complete_output(output_path):
                reused_count += 1
                continue

            if audio is None:
                audio = load_audio_file(str(source), args.sample_rate)

            rng = build_rng(source, variant_idx, args.seed)
            rir_audio = None
            if rir_paths and rng.random() < args.rir_probability:
                selected_rir = select_rir_path(rir_paths, rng)
                if selected_rir is not None:
                    rir_audio = load_rir_cached(selected_rir, args.sample_rate, rir_cache)

            rendered = render_room_playback_variant(audio, args.sample_rate, rng, rir_audio=rir_audio)
            save_audio_file(output_path, rendered, args.sample_rate)
            processed_count += 1

    write_output_list(prepared_paths, output_list)
    print(f"[ok] wrote {len(prepared_paths):,} prepared music entries -> {output_list}")
    print(f"[info] prepared={processed_count:,} reused={reused_count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
