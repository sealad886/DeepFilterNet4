#!/usr/bin/env python3
"""Fast batch audio enhancement with DFNet4-MLX.

Three-stage pipeline that overlaps I/O with GPU work:
  1. Thread pool decodes audio files (ffmpeg for m4a/mp3, soundfile for wav/flac)
  2. Main thread runs MLX model enhancement (GPU)
  3. Thread pool saves results asynchronously (soundfile)

Single model initialization handles multiple input directories.

Usage:
    python scripts/fast_enhance.py \\
        --input /path/to/noisy1 --output /path/to/enhanced1 \\
        --input /path/to/noisy2 --output /path/to/enhanced2 \\
        --checkpoint-dir /path/to/checkpoints
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).parent.absolute()
DF_DIR = SCRIPT_DIR.parent / "DeepFilterNet"
sys.path.insert(0, str(DF_DIR))

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}
TARGET_SR = 48000


class AudioItem(NamedTuple):
    """An audio file ready for enhancement."""

    original_path: str
    output_path: str
    audio: np.ndarray
    duration: float


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------


def _decode_ffmpeg(path: str) -> np.ndarray:
    """Decode any audio file to float32 mono 48kHz via ffmpeg subprocess."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SR),
            "-",
        ],
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: {result.stderr.decode(errors='replace').strip()}")
    if len(result.stdout) == 0:
        raise RuntimeError(f"ffmpeg produced empty output for {path}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _decode_soundfile(path: str) -> np.ndarray:
    """Decode WAV/FLAC via soundfile (zero subprocess overhead)."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != TARGET_SR:
        from df_mlx._audio_io import resample_audio

        audio = resample_audio(audio, sr, TARGET_SR)
    return np.ascontiguousarray(audio, dtype=np.float32)


def decode_audio(path: str) -> np.ndarray:
    """Decode audio to float32 mono 48kHz, picking the fastest decoder."""
    ext = Path(path).suffix.lower()
    if ext in {".wav", ".flac"}:
        try:
            return _decode_soundfile(path)
        except Exception:
            return _decode_ffmpeg(path)
    return _decode_ffmpeg(path)


def _load_item(original_path: str, output_path: str) -> AudioItem:
    """Load and decode one audio file (runs in thread pool)."""
    audio = decode_audio(original_path)
    duration = len(audio) / TARGET_SR
    return AudioItem(original_path=original_path, output_path=output_path, audio=audio, duration=duration)


def _save_item(audio_np: np.ndarray, output_path: str) -> None:
    """Save enhanced audio (runs in thread pool)."""
    import soundfile as sf

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    max_val = np.abs(audio_np).max()
    if max_val > 1.0:
        audio_np = audio_np / max_val * 0.95
    sf.write(output_path, audio_np, TARGET_SR)


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find most recent checkpoint in a directory tree."""
    root = Path(checkpoint_dir)
    if not root.exists():
        return None

    def _scan(d: Path) -> list:
        found = []
        for pat in ("step_*.safetensors", "epoch_*.safetensors"):
            found.extend(d.glob(pat))
        return found

    def _number(p: Path) -> int:
        parts = p.stem.split("_")
        try:
            return int(parts[1]) if len(parts) >= 2 else 0
        except ValueError:
            return 0

    checkpoints = _scan(root)
    if not checkpoints:
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                checkpoints.extend(_scan(sub))

    if checkpoints:
        checkpoints.sort(key=_number, reverse=True)
        return str(checkpoints[0])

    for d in [root] + sorted(s for s in root.iterdir() if s.is_dir()):
        best = d / "best.safetensors"
        if best.exists():
            return str(best)

    return None


def init_model(checkpoint_path: str) -> Tuple:
    """Initialize DFNet4-MLX model with checkpoint. Returns (model, config)."""
    import mlx.core as mx
    from df_mlx.config import BackboneParams, ModelParams4
    from df_mlx.model import init_model as _init_model
    from df_mlx.train import load_checkpoint

    config = ModelParams4()

    # Load training config from state file
    ckpt = Path(checkpoint_path)
    for suffix in (".state.json", ".json"):
        state_path = ckpt.with_suffix(suffix)
        if state_path.exists():
            break
    else:
        state_path = None

    train_cfg: dict = {}
    if state_path and state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        train_cfg = state.get("config", {})

    if train_cfg:
        for key, attr_chain in [
            ("nb_df", ("df", "nb_df")),
            ("nb_erb", ("erb", "nb_erb")),
            ("hop_size", ("audio", "hop_size")),
        ]:
            if key in train_cfg:
                obj = config
                for part in attr_chain[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, attr_chain[-1], train_cfg[key])
        if "fft_size" in train_cfg:
            config.audio.fft_size = train_cfg["fft_size"]
            config.audio.nb_freqs = train_cfg["fft_size"] // 2 + 1
            config.audio.n_freqs = config.audio.nb_freqs

    # Detect backbone type from weights
    weights = mx.load(checkpoint_path)
    weight_keys = set(weights.keys())
    if any("attention_layers" in k for k in weight_keys):
        backbone_type = "attention"
    elif any("gru" in k.lower() for k in weight_keys):
        backbone_type = "gru"
    else:
        backbone_type = "mamba"
    del weights

    config.backbone = BackboneParams(backbone_type=backbone_type)

    model = _init_model(
        config=config,
        variant=train_cfg.get("model_variant", "full"),
    )
    load_checkpoint(model, checkpoint_path)

    # Warmup: force Metal shader compilation on a tiny input
    dummy = mx.zeros((1, TARGET_SR))
    mx.eval(model.enhance(dummy))

    return model, config


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def collect_files(input_dirs: List[str], output_dirs: List[str]) -> List[Tuple[str, str]]:
    """Collect (input_path, output_path) pairs from matched dir lists."""
    pairs: List[Tuple[str, str]] = []
    seen = set()

    for in_dir, out_dir in zip(input_dirs, output_dirs):
        p = Path(in_dir)
        if not p.is_dir():
            print(f"WARNING: skipping non-directory {in_dir}")
            continue
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
                abs_path = str(f.resolve())
                if abs_path not in seen:
                    seen.add(abs_path)
                    out_name = f"{f.stem}_enhanced.wav"
                    pairs.append((str(f), str(Path(out_dir) / out_name)))
    return pairs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    model,
    file_pairs: List[Tuple[str, str]],
    workers: int = 4,
) -> dict:
    """Run the three-stage pipeline: load → enhance → save."""
    import mlx.core as mx

    n = len(file_pairs)
    total_duration = 0.0
    total_enhance_time = 0.0
    successes = 0
    failures = 0

    t_wall_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Stage 1: submit all load jobs (pool runs them concurrently up to `workers`)
        load_futures = [pool.submit(_load_item, inp, outp) for inp, outp in file_pairs]

        save_futures = []

        for i, future in enumerate(load_futures):
            try:
                item = future.result()
            except Exception as e:
                fname = Path(file_pairs[i][0]).name
                print(f"  [{i + 1}/{n}] FAIL (load) {fname}: {e}")
                failures += 1
                continue

            # Stage 2: enhance on GPU (main thread — sequential)
            audio_mx = mx.array(item.audio)
            t0 = time.time()
            enhanced = model.enhance(audio_mx)
            mx.eval(enhanced)
            t1 = time.time()

            enhance_time = t1 - t0
            total_enhance_time += enhance_time
            total_duration += item.duration
            successes += 1

            rtf = enhance_time / item.duration if item.duration > 0 else 0
            fname = Path(item.original_path).name
            print(f"  [{i + 1}/{n}] {fname} ({item.duration:.1f}s) RTF={rtf:.3f}")

            # Stage 3: save asynchronously
            enhanced_np = np.array(enhanced, dtype=np.float32)
            save_futures.append(pool.submit(_save_item, enhanced_np, item.output_path))

        # Wait for all saves
        for f in save_futures:
            try:
                f.result()
            except Exception as e:
                print(f"  WARNING: save failed: {e}")
                failures += 1
                successes -= 1

    wall_time = time.time() - t_wall_start

    return {
        "files_processed": successes,
        "files_failed": failures,
        "total_audio_seconds": total_duration,
        "total_enhance_seconds": total_enhance_time,
        "wall_time_seconds": wall_time,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class PairedDirsAction(argparse.Action):
    """Accumulate --input/--output into paired lists on the namespace."""

    def __call__(self, parser, namespace, values, option_string=None):
        lst = getattr(namespace, self.dest) or []
        lst.append(values)
        setattr(namespace, self.dest, lst)


def main():
    parser = argparse.ArgumentParser(
        description="Fast batch audio enhancement with DFNet4-MLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Single directory
    python scripts/fast_enhance.py -i noisy/ -o enhanced/ --checkpoint-dir /path/to/ckpts

    # Multiple directories (one model load)
    python scripts/fast_enhance.py \\
        -i /data/to-clean -o /data/enhanced/main \\
        -i /data/to-clean/news -o /data/enhanced/news \\
        --checkpoint-dir /path/to/ckpts
""",
    )

    parser.add_argument("-i", "--input", dest="inputs", action=PairedDirsAction, required=True, help="Input directory")
    parser.add_argument(
        "-o", "--output", dest="outputs", action=PairedDirsAction, required=True, help="Output directory"
    )
    parser.add_argument("--checkpoint", help="Direct path to .safetensors checkpoint")
    parser.add_argument(
        "--checkpoint-dir",
        default=os.path.expanduser("~/DataDump/checkpoints"),
        help="Directory to search for latest checkpoint",
    )
    parser.add_argument("--workers", type=int, default=4, help="I/O thread pool size (default: 4)")

    args = parser.parse_args()

    if len(args.inputs) != len(args.outputs):
        parser.error("Each --input must have a matching --output")

    # Resolve checkpoint
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    if not checkpoint:
        print("ERROR: No checkpoint found. Use --checkpoint or --checkpoint-dir")
        sys.exit(1)
    print(f"Checkpoint: {Path(checkpoint).name}")

    # Collect files
    file_pairs = collect_files(args.inputs, args.outputs)
    if not file_pairs:
        print("No audio files found")
        sys.exit(1)
    print(
        f"Found {len(file_pairs)} audio files across {len(args.inputs)} director{'y' if len(args.inputs) == 1 else 'ies'}"
    )

    # Initialize model once
    print("Initializing model...")
    t0 = time.time()
    model, _ = init_model(checkpoint)
    init_time = time.time() - t0
    print(f"Model ready in {init_time:.1f}s (backbone detected, shaders compiled)")
    print()

    # Run pipeline
    stats = run_pipeline(model, file_pairs, workers=args.workers)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Files processed: {stats['files_processed']}")
    if stats["files_failed"]:
        print(f"  Files failed:    {stats['files_failed']}")
    total_min = stats["total_audio_seconds"] / 60
    print(f"  Total audio:     {stats['total_audio_seconds']:.1f}s ({total_min:.1f}min)")
    print(f"  Enhance time:    {stats['total_enhance_seconds']:.1f}s (GPU only)")
    print(f"  Wall time:       {stats['wall_time_seconds']:.1f}s (incl. I/O)")

    if stats["total_audio_seconds"] > 0:
        enhance_rtf = stats["total_enhance_seconds"] / stats["total_audio_seconds"]
        wall_rtf = stats["wall_time_seconds"] / stats["total_audio_seconds"]
        enhance_speed = 1 / enhance_rtf
        wall_speed = 1 / wall_rtf
        print(f"  Enhance RTF:     {enhance_rtf:.4f} ({enhance_speed:.0f}x real-time)")
        print(f"  Effective RTF:   {wall_rtf:.4f} ({wall_speed:.0f}x real-time, with I/O)")
    print("=" * 60)


if __name__ == "__main__":
    main()
