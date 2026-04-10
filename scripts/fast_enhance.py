#!/usr/bin/env python3
"""Fast batch audio enhancement with DFNet4-MLX and Rich TUI.

Three-stage pipeline that overlaps I/O with GPU work:
  1. Thread pool decodes audio files (ffmpeg for m4a/mp3/mp4, soundfile for wav/flac)
  2. Main thread runs MLX model enhancement (GPU)
  3. Thread pool saves results asynchronously (soundfile)

Supports multiple checkpoint directories — processes them sequentially (GPU is
the scarce resource; parallel models would thrash Metal and waste memory).

Usage:
    # Single checkpoint
    python scripts/fast_enhance.py \\
        -i /path/to/noisy1 -i /path/to/noisy2 \\
        --output-base /path/to/output \\
        --checkpoint-dir /path/to/checkpoints

    # Multiple checkpoints (one model load per checkpoint, shared file list)
    python scripts/fast_enhance.py \\
        -i /path/to/noisy1 -i /path/to/noisy2 \\
        --output-base /path/to/output \\
        --checkpoint-dir /path/to/ckpt1 --checkpoint-dir /path/to/ckpt2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).parent.absolute()
DF_DIR = SCRIPT_DIR.parent / "DeepFilterNet"
sys.path.insert(0, str(DF_DIR))

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".mp4", ".m4v"}
TARGET_SR = 48000


# ---------------------------------------------------------------------------
# Rich TUI helpers
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False


def _make_console() -> "Console":
    if _RICH:
        return Console()
    raise RuntimeError("rich is required")


def _make_progress(console: "Console") -> "Progress":
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[status]}"),
        console=console,
        transient=False,
    )


def _checkpoint_header(console: "Console", idx: int, total: int, ckpt_name: str, ckpt_file: str, info: dict) -> None:
    """Print a rich panel header for a checkpoint run."""
    lines = [
        f"[bold cyan]Checkpoint {idx}/{total}[/bold cyan]: [bold]{ckpt_name}[/bold]",
        f"  File: [dim]{ckpt_file}[/dim]",
    ]
    if info.get("backbone"):
        lines.append(f"  Backbone: [yellow]{info['backbone']}[/yellow]  Variant: {info.get('variant', 'full')}")
    console.print(Panel("\n".join(lines), border_style="blue", padding=(0, 1)))


def _checkpoint_summary(console: "Console", ckpt_name: str, stats: dict) -> None:
    """Print a rich table summarising one checkpoint run."""
    table = Table(title=f"[bold]{ckpt_name}[/bold] results", border_style="green", show_header=False, padding=(0, 2))
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Files processed", str(stats["files_processed"]))
    if stats["files_failed"]:
        table.add_row("Files failed", f"[red]{stats['files_failed']}[/red]")
    total_min = stats["total_audio_seconds"] / 60
    table.add_row("Audio duration", f"{stats['total_audio_seconds']:.0f}s ({total_min:.0f} min)")
    table.add_row("Enhance time (GPU)", f"{stats['total_enhance_seconds']:.1f}s")
    table.add_row("Wall time", f"{stats['wall_time_seconds']:.1f}s")
    if stats["total_audio_seconds"] > 0:
        e_rtf = stats["total_enhance_seconds"] / stats["total_audio_seconds"]
        w_rtf = stats["wall_time_seconds"] / stats["total_audio_seconds"]
        table.add_row("Enhance RTF", f"{e_rtf:.4f}  ({1 / e_rtf:.0f}× real-time)")
        table.add_row("Effective RTF", f"{w_rtf:.4f}  ({1 / w_rtf:.0f}× real-time)")
    console.print(table)
    console.print()


def _comparison_table(console: "Console", all_stats: Dict[str, dict]) -> None:
    """Print a comparison table across all checkpoint runs."""
    if len(all_stats) < 2:
        return
    table = Table(title="[bold]Checkpoint Comparison[/bold]", border_style="cyan")
    table.add_column("Checkpoint", style="bold")
    table.add_column("Files", justify="right")
    table.add_column("Audio", justify="right")
    table.add_column("GPU Time", justify="right")
    table.add_column("Wall Time", justify="right")
    table.add_column("Eff. RTF", justify="right")

    for name, s in all_stats.items():
        audio_min = s["total_audio_seconds"] / 60
        w_rtf = s["wall_time_seconds"] / s["total_audio_seconds"] if s["total_audio_seconds"] > 0 else 0
        speed = 1 / w_rtf if w_rtf > 0 else 0
        table.add_row(
            name,
            str(s["files_processed"]),
            f"{audio_min:.0f} min",
            f"{s['total_enhance_seconds']:.1f}s",
            f"{s['wall_time_seconds']:.1f}s",
            f"{w_rtf:.4f} ({speed:.0f}×)",
        )
    console.print(table)


def _final_totals(console: "Console", grand: dict) -> None:
    """Print grand totals panel."""
    lines = []
    lines.append(f"Checkpoints processed: [bold]{grand['checkpoints']}[/bold]")
    lines.append(f"Total files enhanced:  [bold]{grand['files']}[/bold]")
    total_min = grand["audio_seconds"] / 60
    lines.append(f"Total audio:           [bold]{total_min:.0f} min[/bold]")
    lines.append(f"Total wall time:       [bold]{grand['wall_seconds']:.0f}s[/bold]")
    console.print(Panel("\n".join(lines), title="[bold green]Complete[/bold green]", border_style="green"))


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


def _load_train_config(checkpoint_path: str) -> dict:
    """Load training config from the checkpoint's state file."""
    ckpt = Path(checkpoint_path)
    for suffix in (".state.json", ".json"):
        state_path = ckpt.with_suffix(suffix)
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            return state.get("config", {})
    return {}


def init_model(checkpoint_path: str) -> Tuple:
    """Initialize DFNet4-MLX model with checkpoint.

    Returns (model, info_dict) where info_dict has backbone/variant metadata.
    """
    import mlx.core as mx
    from df_mlx.config import BackboneParams, ModelParams4
    from df_mlx.model import init_model as _init_model
    from df_mlx.train import load_checkpoint

    config = ModelParams4()
    train_cfg = _load_train_config(checkpoint_path)

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
    variant = train_cfg.get("model_variant", "full")

    model = _init_model(config=config, variant=variant)
    load_checkpoint(model, checkpoint_path)

    # Warmup: force Metal shader compilation
    dummy = mx.zeros((1, TARGET_SR))
    mx.eval(model.enhance(dummy))

    info = {"backbone": backbone_type, "variant": variant}
    return model, info


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def collect_input_files(input_dirs: List[str]) -> List[Tuple[str, str]]:
    """Collect (abs_path, relative_subdir) for all audio files.

    ``relative_subdir`` is the path of the file's parent relative to the first
    input directory (treated as the base).  For files directly in the base this
    is ``"."``.
    """
    if not input_dirs:
        return []

    base = Path(input_dirs[0]).resolve()
    results: List[Tuple[str, str]] = []
    seen: set = set()

    for in_dir in input_dirs:
        p = Path(in_dir).resolve()
        if not p.is_dir():
            continue
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
                abs_path = str(f.resolve())
                if abs_path not in seen:
                    seen.add(abs_path)
                    try:
                        rel = str(f.resolve().parent.relative_to(base))
                    except ValueError:
                        rel = f.resolve().parent.name
                    results.append((abs_path, rel))
    return results


def make_output_pairs(
    input_files: List[Tuple[str, str]],
    output_base: str,
    ckpt_name: str,
    model_label: str,
) -> List[Tuple[str, str]]:
    """Build (input_path, output_path) pairs for one checkpoint."""
    pairs: List[Tuple[str, str]] = []
    for abs_path, rel_subdir in input_files:
        stem = Path(abs_path).stem
        out_dir = Path(output_base) / ckpt_name / rel_subdir / model_label
        pairs.append((abs_path, str(out_dir / f"{stem}_enhanced.wav")))
    return pairs


# ---------------------------------------------------------------------------
# Pipeline (with Rich progress)
# ---------------------------------------------------------------------------


def run_pipeline(
    model,
    file_pairs: List[Tuple[str, str]],
    console: "Console",
    workers: int = 4,
) -> dict:
    """Run the three-stage pipeline: decode → enhance → save."""
    import mlx.core as mx

    n = len(file_pairs)
    total_duration = 0.0
    total_enhance_time = 0.0
    successes = 0
    failures = 0

    t_wall_start = time.time()

    with _make_progress(console) as progress:
        task_id = progress.add_task("Processing", total=n, status="")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            load_futures = [pool.submit(_load_item, inp, outp) for inp, outp in file_pairs]
            save_futures: List[Future] = []

            for i, future in enumerate(load_futures):
                fname = Path(file_pairs[i][0]).name

                try:
                    progress.update(task_id, status=f"[dim]decode {fname}[/dim]")
                    item = future.result()
                except Exception as e:
                    console.print(f"  [red]✗ FAIL (load)[/red] {fname}: {e}")
                    failures += 1
                    progress.advance(task_id)
                    continue

                # Enhance on GPU
                progress.update(task_id, status=f"[cyan]enhance {fname}[/cyan]")
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
                progress.update(task_id, status=f"[green]{fname}[/green] RTF={rtf:.3f}")
                progress.advance(task_id)

                # Save asynchronously
                enhanced_np = np.array(enhanced, dtype=np.float32)
                save_futures.append(pool.submit(_save_item, enhanced_np, item.output_path))

            # Drain saves
            progress.update(task_id, status="[dim]waiting for saves…[/dim]")
            for f in save_futures:
                try:
                    f.result()
                except Exception as e:
                    console.print(f"  [red]✗ save failed:[/red] {e}")
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


class AppendAction(argparse.Action):
    """Accumulate repeated flags into a list."""

    def __call__(self, parser, namespace, values, option_string=None):
        lst = getattr(namespace, self.dest) or []
        lst.append(values)
        setattr(namespace, self.dest, lst)


def _resolve_checkpoint_dirs(raw_dirs: List[str], fallback_base: str) -> List[Tuple[str, str, str]]:
    """Resolve checkpoint dirs → list of (display_name, dir_path, ckpt_file).

    Each raw dir is either an absolute path or a name under ``fallback_base``.
    """
    resolved: List[Tuple[str, str, str]] = []
    for raw in raw_dirs:
        d = Path(raw)
        if not d.is_dir():
            candidate = Path(fallback_base) / raw
            if candidate.is_dir():
                d = candidate
            else:
                raise FileNotFoundError(f"Checkpoint directory not found: {raw} (also tried {candidate})")
        ckpt_file = find_latest_checkpoint(str(d))
        if not ckpt_file:
            raise FileNotFoundError(f"No checkpoint file found in {d}")
        resolved.append((d.name, str(d), ckpt_file))
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Fast batch audio enhancement with DFNet4-MLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Single checkpoint
    python scripts/fast_enhance.py \\
        -i /data/to-clean -i /data/to-clean/news \\
        --output-base /data/to-listen \\
        --checkpoint-dir my_checkpoint

    # Multiple checkpoints (processed sequentially, one model at a time)
    python scripts/fast_enhance.py \\
        -i /data/to-clean -i /data/to-clean/news \\
        --output-base /data/to-listen \\
        --checkpoint-dir ckpt_A --checkpoint-dir ckpt_B

Output structure:
    {output-base}/{checkpoint-name}/{relative-subdir}/{model-label}/{file}_enhanced.wav
""",
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="inputs",
        action=AppendAction,
        required=True,
        help="Input directory (repeat for multiple)",
    )
    parser.add_argument(
        "--output-base",
        required=True,
        help="Base output directory (checkpoint name / subdir structure created underneath)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dirs",
        action=AppendAction,
        required=True,
        help="Checkpoint directory or name (repeat for multiple)",
    )
    parser.add_argument(
        "--checkpoint-fallback-base",
        default=os.path.expanduser("~/DataDump/checkpoints"),
        help="Fallback base when checkpoint-dir is a bare name (default: ~/DataDump/checkpoints)",
    )
    parser.add_argument(
        "--model-label",
        default="DeepFilterNet4-MLX",
        help="Subdirectory name for model outputs (default: DeepFilterNet4-MLX)",
    )
    parser.add_argument("--workers", type=int, default=4, help="I/O thread pool size (default: 4)")

    args = parser.parse_args()

    if not _RICH:
        print("ERROR: rich is required for TUI display.  pip install rich", file=sys.stderr)
        sys.exit(1)

    console = _make_console()

    # Resolve all checkpoint dirs
    try:
        checkpoints = _resolve_checkpoint_dirs(args.checkpoint_dirs, args.checkpoint_fallback_base)
    except FileNotFoundError as e:
        console.print(f"[red bold]ERROR:[/red bold] {e}")
        sys.exit(1)

    # Collect input files once (shared across all checkpoints)
    input_files = collect_input_files(args.inputs)
    if not input_files:
        console.print("[red]No audio files found in input directories[/red]")
        sys.exit(1)

    n_dirs = len(args.inputs)
    console.print(
        Panel(
            f"[bold]{len(input_files)}[/bold] audio files from "
            f"[bold]{n_dirs}[/bold] director{'y' if n_dirs == 1 else 'ies'}\n"
            f"[bold]{len(checkpoints)}[/bold] checkpoint{'s' if len(checkpoints) != 1 else ''} to process "
            f"[dim](sequential — GPU is the scarce resource)[/dim]\n"
            f"Output base: [cyan]{args.output_base}[/cyan]",
            title="[bold]Fast Enhance[/bold]",
            border_style="blue",
        )
    )

    all_stats: Dict[str, dict] = {}
    grand = {"checkpoints": 0, "files": 0, "audio_seconds": 0.0, "wall_seconds": 0.0}
    total_ckpts = len(checkpoints)

    for idx, (ckpt_name, ckpt_dir, ckpt_file) in enumerate(checkpoints, 1):
        # Build output pairs for this checkpoint
        file_pairs = make_output_pairs(input_files, args.output_base, ckpt_name, args.model_label)

        # Init model
        console.print()
        _checkpoint_header(console, idx, total_ckpts, ckpt_name, Path(ckpt_file).name, {})

        console.print("  [dim]Loading model…[/dim]")
        t0 = time.time()
        model, info = init_model(ckpt_file)
        init_time = time.time() - t0
        console.print(
            f"  [green]✓[/green] Model ready in {init_time:.1f}s "
            f"[dim](backbone={info['backbone']}, variant={info['variant']})[/dim]"
        )

        # Run pipeline
        stats = run_pipeline(model, file_pairs, console, workers=args.workers)
        all_stats[ckpt_name] = stats

        _checkpoint_summary(console, ckpt_name, stats)

        grand["checkpoints"] += 1
        grand["files"] += stats["files_processed"]
        grand["audio_seconds"] += stats["total_audio_seconds"]
        grand["wall_seconds"] += stats["wall_time_seconds"] + init_time

        # Release model memory before loading next
        del model

    # Final comparison + totals
    _comparison_table(console, all_stats)
    _final_totals(console, grand)


if __name__ == "__main__":
    main()
