#!/usr/bin/env python3
"""Enhance a clean-speech corpus with DeepFilterNet3 and mirror it to a new tree.

This is intended as an optional pre-step before building the MLX datastore.
It preserves relative paths under ``--base-dir`` and emits a file list that can
be fed directly into ``build_mlx_datastore.sh`` / ``df_mlx.build_audio_cache``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Iterable, List, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "DeepFilterNet"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from df.enhance import AudioDataset, enhance, init_df  # noqa: E402
from df.io import resample, save_audio  # noqa: E402
from df.model import ModelParams  # noqa: E402

NON_SPEECH_PATH_MARKERS = frozenset(
    {
        "noise",
        "music",
        "musan",
        "fsd50k",
        "rir",
        "openair",
        "acousticrooms",
        "air",
    }
)

KNOWN_MLX_MODEL_NAMES = frozenset({"deepfilternet3-mlx", "deepfilternet4-mlx"})
KNOWN_TORCH_MODEL_NAMES = frozenset({"deepfilternet", "deepfilternet2", "deepfilternet3"})
MLX_CLEAR_CACHE_INTERVAL = 8
MLX_DEFAULT_ENHANCE_BATCH_SIZE = 4
_LIST_WRITE_INTERVAL = 30.0


class PreprocessProgressStats:
    def __init__(self, start_time: float) -> None:
        self.start_time = start_time
        self.enhance_count = 0
        self.enhance_seconds = 0.0
        self.processed_audio_seconds = 0.0
        self.save_count = 0
        self.save_seconds = 0.0
        self.queue_high_water = 0


class EnhanceBackend:
    def __init__(self, name: str, sample_rate: int, enhance_audio: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.name = name
        self.sample_rate = sample_rate
        self.enhance_audio = enhance_audio


class ProbeCacheEntry(NamedTuple):
    duration_seconds: float
    size_bytes: int
    mtime_ns: int


def read_file_list(path: Path) -> List[Path]:
    files: List[Path] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            files.append(Path(line).expanduser().resolve())
    return files


def build_output_path(source: Path, output_root: Path, base_dir: Path) -> Path:
    try:
        relative = source.relative_to(base_dir)
    except ValueError:
        relative = Path("_external") / source.name
    return output_root / relative


def write_output_list(paths: Iterable[Path], output_list: Path) -> None:
    output_list.parent.mkdir(parents=True, exist_ok=True)
    temp_output_list = output_list.with_name(f"{output_list.name}.tmp.{os.getpid()}")
    with temp_output_list.open("w") as handle:
        for path in paths:
            handle.write(f"{path}\n")
    temp_output_list.replace(output_list)


def write_resumable_output_list(output_paths: list[Path], completed_paths: set[Path], output_list: Path) -> None:
    write_output_list((path for path in output_paths if path in completed_paths), output_list)


def _maybe_write_resumable_output_list(
    output_paths: list[Path], completed_paths: set[Path], output_list: Path, *, force: bool = False
) -> None:
    now = time.monotonic()
    last = _maybe_write_resumable_output_list._last_write_time
    if not force and (now - last) < _LIST_WRITE_INTERVAL:
        return
    write_resumable_output_list(output_paths, completed_paths, output_list)
    _maybe_write_resumable_output_list._last_write_time = now


_maybe_write_resumable_output_list._last_write_time = 0.0


def resolve_probe_cache_path(output_list: Path, explicit_cache_path: str | None) -> Path:
    if explicit_cache_path:
        return Path(explicit_cache_path).expanduser().resolve()
    cache_stem = output_list.stem if output_list.suffix else output_list.name
    return output_list.with_name(f"{cache_stem}.ffprobe-cache.json")


def _build_probe_cache_entry(path: Path, duration_seconds: float) -> ProbeCacheEntry:
    stat = path.stat()
    return ProbeCacheEntry(
        duration_seconds=duration_seconds,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def load_probe_cache(path: Path) -> dict[Path, ProbeCacheEntry]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] Ignoring unreadable ffprobe cache {path}: {exc}", file=sys.stderr)
        return {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        return {}

    cache: dict[Path, ProbeCacheEntry] = {}
    for raw_path, raw_entry in entries.items():
        if not isinstance(raw_path, str) or not isinstance(raw_entry, dict):
            continue
        try:
            duration_seconds = float(raw_entry["duration_seconds"])
            size_bytes = int(raw_entry["size_bytes"])
            mtime_ns = int(raw_entry["mtime_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(duration_seconds) or duration_seconds < 0.0 or size_bytes < 0 or mtime_ns < 0:
            continue
        cache[Path(raw_path)] = ProbeCacheEntry(
            duration_seconds=duration_seconds,
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
        )
    return cache


def write_probe_cache(path: Path, entries: dict[Path, ProbeCacheEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "entries": {
            str(source_path): {
                "duration_seconds": entry.duration_seconds,
                "size_bytes": entry.size_bytes,
                "mtime_ns": entry.mtime_ns,
            }
            for source_path, entry in sorted(entries.items(), key=lambda item: str(item[0]))
        },
    }
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)


def find_output_path_collisions(sources: list[Path], output_paths: list[Path]) -> dict[Path, list[Path]]:
    collisions: dict[Path, list[Path]] = {}
    mapped_sources: dict[Path, list[Path]] = {}
    for source, output_path in zip(sources, output_paths):
        existing_sources = mapped_sources.setdefault(output_path, [])
        if source not in existing_sources:
            existing_sources.append(source)
    for output_path, output_sources in mapped_sources.items():
        if len(output_sources) > 1:
            collisions[output_path] = output_sources
    return collisions


def raise_on_output_path_collisions(sources: list[Path], output_paths: list[Path]) -> None:
    collisions = find_output_path_collisions(sources, output_paths)
    if not collisions:
        return
    preview_lines: list[str] = []
    for output_path, output_sources in list(collisions.items())[:10]:
        joined_sources = ", ".join(str(source) for source in output_sources[:3])
        if len(output_sources) > 3:
            joined_sources = f"{joined_sources}, ..."
        preview_lines.append(f"  {output_path} <- {joined_sources}")
    remainder = len(collisions) - min(len(collisions), 10)
    if remainder > 0:
        preview_lines.append(f"  ... and {remainder} more")
    raise SystemExit(
        "Multiple source files map to the same preprocess output path. "
        "Use a broader --base-dir or a different --output-root so each source mirrors uniquely.\n"
        + "\n".join(preview_lines)
    )


def path_labels(path: Path) -> set[str]:
    labels: set[str] = set()
    anchor = path.anchor.lower()
    for part in path.parts:
        lower_part = part.lower()
        if lower_part == anchor:
            continue
        labels.add(lower_part)
        stem = Path(lower_part).stem
        if stem:
            labels.add(stem)
    return labels


def find_ineligible_sources(paths: Iterable[Path]) -> List[Path]:
    return [path for path in paths if path_labels(path) & NON_SPEECH_PATH_MARKERS]


def running_on_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def model_requests_mlx(model_base_dir: str | None) -> bool:
    if not model_base_dir:
        return False
    lowered = str(model_base_dir).strip().lower()
    if lowered in KNOWN_MLX_MODEL_NAMES:
        return True
    path = Path(model_base_dir).expanduser()
    if lowered in KNOWN_TORCH_MODEL_NAMES:
        return False
    return "mlx" in lowered or path.name.lower() in KNOWN_MLX_MODEL_NAMES or path.is_dir()


def model_explicitly_requests_mlx(model_base_dir: str | None) -> bool:
    if not model_base_dir:
        return False
    lowered = str(model_base_dir).strip().lower()
    path = Path(model_base_dir).expanduser()
    return lowered in KNOWN_MLX_MODEL_NAMES or "mlx" in lowered or path.name.lower() in KNOWN_MLX_MODEL_NAMES


def should_prefer_mlx_backend(model_base_dir: str | None, requested_device: str | None) -> bool:
    if not running_on_apple_silicon():
        return False
    if requested_device and requested_device.lower() not in {"mps"}:
        return False
    return model_requests_mlx(model_base_dir)


def resolve_torch_fallback_model(model_base_dir: str | None) -> str:
    if model_explicitly_requests_mlx(model_base_dir):
        print("[warn] MLX model requested but MLX backend unavailable; falling back to DeepFilterNet3 (torch).")
        return "DeepFilterNet3"
    return model_base_dir or "DeepFilterNet3"


def is_complete_output(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def build_temp_output_path(target: Path) -> Path:
    return target.with_name(f".{target.stem}.partial.{os.getpid()}{target.suffix}")


def resolve_ffprobe_bin() -> str:
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None:
        raise SystemExit("ffprobe is required for duration-based preprocessing progress but was not found on PATH")
    return ffprobe_bin


def load_torch_backend(model_base_dir: str, requested_device: str | None) -> EnhanceBackend:
    model, df_state, _, _ = init_df(
        model_base_dir=model_base_dir,
        post_filter=False,
        log_level="INFO",
        log_file=None,
        config_allow_defaults=True,
        epoch="best",
        default_model="DeepFilterNet3",
        mask_only=False,
        device=requested_device,
    )
    df_sr = ModelParams().sr

    def enhance_audio(audio: torch.Tensor) -> torch.Tensor:
        return enhance(model, df_state, audio, pad=True, device=requested_device).detach().cpu()

    return EnhanceBackend(name="torch", sample_rate=df_sr, enhance_audio=enhance_audio)


def _import_mlx_enhance_module():
    from df_mlx import enhance as mlx_enhance_mod

    return mlx_enhance_mod


def clear_mlx_cache(mx_module) -> None:
    clear_cache = getattr(mx_module, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
        return
    metal = getattr(mx_module, "metal", None)
    if metal is not None:
        metal_clear_cache = getattr(metal, "clear_cache", None)
        if callable(metal_clear_cache):
            metal_clear_cache()


def is_mlx_resource_limit_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "metal::malloc" in message or ("resource limit" in message and "metal" in message)


def _run_mlx_enhancement(mlx_enhance_mod, model, audio_mx, params):
    enhanced = mlx_enhance_mod.enhance(
        model,
        audio_mx,
        params,
        compensate_delay=True,
    )
    mlx_enhance_mod.mx.eval(enhanced)
    return enhanced


def _mlx_output_to_torch(enhanced) -> torch.Tensor:
    enhanced_np = np.array(enhanced, copy=True)
    return torch.from_numpy(enhanced_np)


def load_mlx_backend(model_base_dir: str) -> EnhanceBackend:
    mlx_enhance_mod = _import_mlx_enhance_module()

    model, params, _, _ = mlx_enhance_mod.load_model(model_path=model_base_dir, epoch="best")
    inference_calls = 0

    def enhance_audio(audio: torch.Tensor) -> torch.Tensor:
        nonlocal inference_calls

        audio_np = audio.detach().cpu().numpy()
        audio_mx = None
        enhanced = None
        result = None
        try:
            audio_mx = mlx_enhance_mod.mx.array(audio_np)
            for attempt in range(2):
                try:
                    enhanced = _run_mlx_enhancement(mlx_enhance_mod, model, audio_mx, params)
                    break
                except Exception as exc:
                    if attempt == 0 and is_mlx_resource_limit_error(exc):
                        clear_mlx_cache(mlx_enhance_mod.mx)
                        gc.collect()
                        audio_mx = mlx_enhance_mod.mx.array(audio_np)
                        continue
                    clear_mlx_cache(mlx_enhance_mod.mx)
                    gc.collect()
                    raise
            result = _mlx_output_to_torch(enhanced)
            return result
        finally:
            inference_calls += 1
            if inference_calls % MLX_CLEAR_CACHE_INTERVAL == 0:
                clear_mlx_cache(mlx_enhance_mod.mx)
                gc.collect()
            del audio_mx
            del enhanced

    return EnhanceBackend(name="mlx", sample_rate=params.sr, enhance_audio=enhance_audio)


def resolve_backend(model_base_dir: str, requested_device: str | None) -> EnhanceBackend:
    if should_prefer_mlx_backend(model_base_dir, requested_device):
        try:
            return load_mlx_backend(model_base_dir)
        except Exception as exc:
            print(f"[warn] Failed to initialize MLX preprocessing backend, falling back to torch: {exc}")
    return load_torch_backend(resolve_torch_fallback_model(model_base_dir), requested_device)


def choose_enhance_batch_size(backend_name: str, override: int | None = None) -> int:
    if override is not None:
        return max(1, override)
    return MLX_DEFAULT_ENHANCE_BATCH_SIZE if backend_name == "mlx" else 1


def _can_batch_audio(audio: torch.Tensor) -> bool:
    return audio.ndim == 1 or (audio.ndim == 2 and audio.shape[0] == 1)


def _normalize_batched_audio(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1:
        return audio.detach().cpu()
    if audio.ndim == 2 and audio.shape[0] == 1:
        return audio.squeeze(0).detach().cpu()
    raise ValueError(f"Batched enhancement supports mono tensors only, got shape {tuple(audio.shape)}")


def enhance_audio_batch(backend: EnhanceBackend, audios: list[torch.Tensor]) -> tuple[list[torch.Tensor], float]:
    if not audios:
        return [], 0.0

    had_channel_dim = [audio.ndim == 2 for audio in audios]
    normalized = [_normalize_batched_audio(audio) for audio in audios]
    lengths = [int(audio.shape[-1]) for audio in normalized]
    max_length = max(lengths)
    padded = [F.pad(audio, (0, max_length - audio.shape[-1])) for audio in normalized]

    enhance_started = time.perf_counter()
    enhanced_batch = backend.enhance_audio(torch.stack(padded, dim=0))
    elapsed = time.perf_counter() - enhance_started

    if enhanced_batch.ndim == 1:
        enhanced_batch = enhanced_batch.unsqueeze(0)
    elif enhanced_batch.ndim == 3 and enhanced_batch.shape[1] == 1:
        enhanced_batch = enhanced_batch.squeeze(1)

    enhanced_items = []
    for i, length in enumerate(lengths):
        enhanced_item = enhanced_batch[i, :length].detach().cpu()
        if had_channel_dim[i]:
            enhanced_item = enhanced_item.unsqueeze(0)
        enhanced_items.append(enhanced_item)
    return enhanced_items, elapsed


def probe_audio_duration_seconds(path: Path, ffprobe_bin: str) -> float:
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "unknown ffprobe error"
        raise RuntimeError(stderr)
    raw_duration = result.stdout.strip()
    try:
        duration_seconds = float(raw_duration)
    except ValueError as exc:
        raise RuntimeError(f"invalid ffprobe duration output: {raw_duration!r}") from exc
    if not math.isfinite(duration_seconds) or duration_seconds < 0.0:
        raise RuntimeError(f"invalid non-finite duration: {duration_seconds!r}")
    return duration_seconds


def probe_audio_durations(
    paths: Iterable[Path],
    ffprobe_bin: str,
    *,
    num_workers: int,
    cache_path: Path | None = None,
) -> dict[Path, float]:
    ordered_paths = list(paths)
    if not ordered_paths:
        return {}

    cache_entries = load_probe_cache(cache_path) if cache_path is not None else {}
    durations: dict[Path, float] = {}
    failures: list[str] = []
    discovered_audio_seconds = 0.0
    start_time = time.perf_counter()
    probe_paths: list[Path] = []

    for path in ordered_paths:
        try:
            stat = path.stat()
        except OSError as exc:
            failures.append(f"{path}: {exc}")
            continue
        cached = cache_entries.get(path)
        if (
            cached is not None
            and cached.size_bytes == stat.st_size
            and cached.mtime_ns == stat.st_mtime_ns
            and math.isfinite(cached.duration_seconds)
            and cached.duration_seconds >= 0.0
        ):
            durations[path] = cached.duration_seconds
            discovered_audio_seconds += cached.duration_seconds
        else:
            probe_paths.append(path)

    max_inflight = max(1, min(len(probe_paths), max(4, num_workers * 4))) if probe_paths else 1
    completed_files = len(durations)

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as probe_pool:
        pending_futures: dict[Future[float], Path] = {}
        path_iter = iter(probe_paths)

        def schedule_probe_jobs() -> None:
            while len(pending_futures) < max_inflight:
                try:
                    path = next(path_iter)
                except StopIteration:
                    break
                pending_futures[probe_pool.submit(probe_audio_duration_seconds, path, ffprobe_bin)] = path

        with tqdm(
            total=len(ordered_paths),
            initial=completed_files,
            desc="ffprobe",
            unit="file",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.05,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        ) as progress:
            schedule_probe_jobs()
            progress.set_postfix_str(
                build_probe_postfix(
                    completed_files,
                    len(ordered_paths),
                    discovered_audio_seconds,
                    len(failures),
                    start_time=start_time,
                ),
                refresh=False,
            )
            while pending_futures:
                completed, _ = wait(set(pending_futures), return_when=FIRST_COMPLETED)
                for future in completed:
                    path = pending_futures.pop(future)
                    try:
                        duration_seconds = future.result()
                        durations[path] = duration_seconds
                        discovered_audio_seconds += duration_seconds
                        cache_entries[path] = _build_probe_cache_entry(path, duration_seconds)
                    except Exception as exc:  # pragma: no cover - exercised via main guard path
                        failures.append(f"{path}: {exc}")
                    completed_files += 1
                schedule_probe_jobs()
                progress.update(len(completed))
                progress.set_postfix_str(
                    build_probe_postfix(
                        completed_files,
                        len(ordered_paths),
                        discovered_audio_seconds,
                        len(failures),
                        start_time=start_time,
                    ),
                    refresh=False,
                )

    if cache_path is not None and cache_entries:
        write_probe_cache(cache_path, cache_entries)
    if failures:
        preview = "\n".join(f"  {item}" for item in failures[:10])
        remainder = len(failures) - min(len(failures), 10)
        if remainder > 0:
            preview = f"{preview}\n  ... and {remainder} more"
        raise SystemExit(f"ffprobe failed while probing source durations:\n{preview}")
    return durations


def resolve_effective_device(requested_device: str | None) -> str:
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_save_workers(loader_workers: int, effective_device: str) -> int:
    if effective_device.startswith("cpu"):
        return 1
    return max(2, min(8, max(1, loader_workers)))


def choose_probe_workers(loader_workers: int) -> int:
    cpu_count = os.cpu_count() or 4
    return max(1, min(32, max(4, max(1, loader_workers) * 4, cpu_count // 2)))


def _format_average_ms(total_seconds: float, count: int) -> str:
    if count <= 0:
        return "n/a"
    return f"{(total_seconds / count) * 1000.0:.0f}ms"


def _format_audio_progress_value(seconds: float) -> str:
    if seconds >= 3600.0:
        return f"{seconds / 3600.0:.1f}h"
    if seconds >= 60.0:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds:.1f}s"


def _format_probe_rate(files_per_second: float | None) -> str:
    if files_per_second is None or not math.isfinite(files_per_second) or files_per_second <= 0.0:
        return "warming"
    return f"{files_per_second:.1f}/s"


def _format_eta(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "warming"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_probe_postfix(
    completed_files: int,
    total_files: int,
    discovered_audio_seconds: float,
    failure_count: int,
    *,
    start_time: float,
    now: float | None = None,
) -> str:
    current_time = time.perf_counter() if now is None else now
    elapsed = max(current_time - start_time, 1e-9)
    file_rate = completed_files / elapsed if completed_files > 0 else None
    remaining_files = max(total_files - completed_files, 0)
    eta_seconds = None
    if file_rate is not None and file_rate > 0.0:
        eta_seconds = remaining_files / file_rate
    elif remaining_files <= 0:
        eta_seconds = 0.0

    return (
        f"files={completed_files:,}/{total_files:,}, "
        f"eta={_format_eta(eta_seconds)}, "
        f"probe={_format_probe_rate(file_rate)}, "
        f"audio={_format_audio_progress_value(discovered_audio_seconds)}, "
        f"fail={failure_count}"
    )


def build_progress_postfix(
    stats: PreprocessProgressStats,
    inflight_saves: int,
    completed_audio_seconds: float,
    total_audio_seconds: float,
    *,
    now: float | None = None,
) -> str:
    current_time = time.perf_counter() if now is None else now
    elapsed = max(current_time - stats.start_time, 1e-9)
    realtime_factor = stats.processed_audio_seconds / elapsed if stats.processed_audio_seconds > 0.0 else None
    remaining_audio_seconds = max(total_audio_seconds - completed_audio_seconds, 0.0)
    eta_seconds = None
    if realtime_factor is not None and realtime_factor > 0.0:
        eta_seconds = remaining_audio_seconds / realtime_factor
    elif remaining_audio_seconds <= 0.0:
        eta_seconds = 0.0

    rt_text = f"{realtime_factor:.2f}x" if realtime_factor is not None else "warming"
    return (
        f"audio={_format_audio_progress_value(completed_audio_seconds)}/"
        f"{_format_audio_progress_value(total_audio_seconds)}, "
        f"eta={_format_eta(eta_seconds)}, "
        f"rt={rt_text}, "
        f"save_q={inflight_saves}, "
        f"enh={_format_average_ms(stats.enhance_seconds, stats.enhance_count)}, "
        f"save={_format_average_ms(stats.save_seconds, stats.save_count)}"
    )


def save_enhanced_audio_atomically(target: Path, enhanced_audio: torch.Tensor, df_sr: int, orig_sr: int) -> float:
    temp_target = build_temp_output_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if temp_target.exists():
        temp_target.unlink()
    save_started = time.perf_counter()
    try:
        audio_to_save = enhanced_audio
        if orig_sr != df_sr:
            audio_to_save = resample(audio_to_save, df_sr, orig_sr)
        save_audio(str(temp_target), audio_to_save, sr=orig_sr, output_dir=None, suffix=None, log=False)
        temp_target.replace(target)
        return time.perf_counter() - save_started
    except Exception:
        if temp_target.exists():
            temp_target.unlink()
        raise


def collect_completed_saves(
    inflight_saves: dict[Future[float], tuple[Path, Path, float]],
    failures: list[str],
    stats: PreprocessProgressStats,
    completed_paths: set[Path],
    *,
    wait_for_completion: bool = False,
    block_until_all: bool = False,
) -> tuple[int, float]:
    if not inflight_saves:
        return 0, 0.0
    if wait_for_completion:
        return_when = ALL_COMPLETED if block_until_all else FIRST_COMPLETED
        completed, _ = wait(set(inflight_saves), return_when=return_when)
    else:
        completed = {future for future in inflight_saves if future.done()}
        if not completed:
            return 0, 0.0
    completed_count = 0
    completed_audio_seconds = 0.0
    for future in completed:
        source, target, duration_seconds = inflight_saves.pop(future)
        try:
            stats.save_seconds += future.result()
            stats.save_count += 1
            stats.processed_audio_seconds += duration_seconds
            completed_paths.add(target)
            completed_count += 1
            completed_audio_seconds += duration_seconds
        except Exception as exc:  # pragma: no cover - operational safeguard
            failures.append(f"{source}: {exc}")
    return completed_count, completed_audio_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess clean speech with DeepFilterNet3.")
    parser.add_argument("--file-list", required=True, help="Input clean-speech file list.")
    parser.add_argument("--output-root", required=True, help="Root directory for enhanced copies.")
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory used to preserve relative paths under the output root.",
    )
    parser.add_argument(
        "--output-list",
        required=True,
        help="Path to write the enhanced file list for downstream datastore building.",
    )
    parser.add_argument(
        "--model-base-dir",
        default="DeepFilterNet3",
        help="Pretrained model name or model directory (default: DeepFilterNet3).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional inference device override: cpu, cuda, mps, etc.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers used while reading source audio (resume is default unless --overwrite is set).",
    )
    parser.add_argument(
        "--probe-workers",
        type=int,
        default=None,
        help="Parallel ffprobe workers used to estimate pending audio duration before enhancement (default: auto).",
    )
    parser.add_argument(
        "--probe-cache",
        default=None,
        help="JSON cache for ffprobe duration results; defaults to a sibling of --output-list. Unchanged files reuse cached durations on reruns.",
    )
    parser.add_argument(
        "--enhance-batch-size",
        type=int,
        default=None,
        help="Batch size for MLX enhancement (default: auto, currently 4 for MLX, 1 for torch).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild outputs even when the mirrored file already exists.",
    )
    parser.add_argument(
        "--allow-non-speech-paths",
        action="store_true",
        help="Bypass the default guard that rejects obvious noise/music/RIR paths.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    file_list = Path(args.file_list).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    output_list = Path(args.output_list).expanduser().resolve()
    effective_device = resolve_effective_device(args.device)

    sources = read_file_list(file_list)
    if not sources:
        raise SystemExit(f"No input files found in {file_list}")

    if not args.allow_non_speech_paths:
        ineligible_sources = find_ineligible_sources(sources)
        if ineligible_sources:
            preview = "\n".join(f"  {path}" for path in ineligible_sources[:10])
            remainder = len(ineligible_sources) - min(len(ineligible_sources), 10)
            if remainder > 0:
                preview = f"{preview}\n  ... and {remainder} more"
            raise SystemExit(
                "Refusing to preprocess obvious non-speech inputs. "
                "Pass a clean speech list instead, or use --allow-non-speech-paths to override.\n"
                f"{preview}"
            )

    output_root.mkdir(parents=True, exist_ok=True)

    output_paths = [build_output_path(source, output_root, base_dir) for source in sources]
    raise_on_output_path_collisions(sources, output_paths)
    completed_paths = set() if args.overwrite else {path for path in output_paths if is_complete_output(path)}
    pending_pairs = [
        (source, target)
        for source, target in zip(sources, output_paths)
        if args.overwrite or target not in completed_paths
    ]
    pending_sources = [source for source, _ in pending_pairs]
    completed_count = len(sources) - len(pending_sources)
    save_workers = choose_save_workers(args.num_workers, effective_device)
    probe_workers = max(1, getattr(args, "probe_workers", 0) or choose_probe_workers(args.num_workers))
    probe_cache_path = resolve_probe_cache_path(output_list, getattr(args, "probe_cache", None))

    write_resumable_output_list(output_paths, completed_paths, output_list)

    print("=" * 60)
    print("Clean Speech Preprocessor")
    print("=" * 60)
    print(f"Input list:      {file_list}")
    print(f"Input files:     {len(sources):,}")
    print(f"Completed:       {completed_count:,}")
    print(f"Pending files:   {len(pending_sources):,}")
    print(f"Output root:     {output_root}")
    print(f"Output list:     {output_list}")
    print(f"Probe cache:     {probe_cache_path}")
    print(f"Base dir:        {base_dir}")
    print(f"Model:           {args.model_base_dir}")
    print(f"Device:          {effective_device}")
    print(f"Workers:         {args.num_workers}")
    print(f"Probe workers:   {probe_workers}")
    print(f"Save workers:    {save_workers}")
    print(f"Mode:            {'overwrite' if args.overwrite else 'resume'}")
    print("=" * 60)

    if not pending_sources:
        print("All mirrored outputs already exist; reusing them.")
        print(f"Wrote output list with {len(completed_paths):,} entries -> {output_list}")
        return 0

    ffprobe_bin = resolve_ffprobe_bin()
    print(f"Duration scan:   ffprobe over {len(pending_sources):,} pending files")
    pending_source_durations = probe_audio_durations(
        pending_sources,
        ffprobe_bin,
        num_workers=probe_workers,
        cache_path=probe_cache_path,
    )
    total_audio_seconds = sum(pending_source_durations.values())
    print(f"Pending audio:   {total_audio_seconds / 3600.0:.2f}h")

    backend = resolve_backend(args.model_base_dir, args.device)
    enhance_batch_size = choose_enhance_batch_size(backend.name, args.enhance_batch_size)
    print(f"Enhance backend: {backend.name}")
    print(f"Enhance batch:   {enhance_batch_size}")

    if enhance_batch_size > 1:
        pending_sources = sorted(pending_sources, key=lambda source: pending_source_durations[source])

    dataset = AudioDataset([str(path) for path in pending_sources], backend.sample_rate)
    loader_kwargs: dict[str, object] = {
        "num_workers": max(0, args.num_workers),
        "pin_memory": effective_device.startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = min(8, max(2, args.num_workers))
    loader = DataLoader(dataset, **loader_kwargs)

    failures: list[str] = []
    inflight_saves: dict[Future[float], tuple[Path, Path, float]] = {}
    max_inflight_saves = max(2, save_workers * 2)
    progress_stats = PreprocessProgressStats(start_time=time.perf_counter())
    pending_enhance_batch: list[tuple[Path, Path, float, int, torch.Tensor]] = []

    def _submit_save(enhanced, source, target, duration_seconds, orig_sr):
        future = save_pool.submit(
            save_enhanced_audio_atomically,
            target,
            enhanced,
            backend.sample_rate,
            orig_sr,
        )
        inflight_saves[future] = (source, target, duration_seconds)
        progress_stats.queue_high_water = max(progress_stats.queue_high_water, len(inflight_saves))

    def _flush_pending_enhance_batch() -> None:
        if not pending_enhance_batch:
            return
        batch_records = list(pending_enhance_batch)
        pending_enhance_batch.clear()
        source_audio = [audio for _, _, _, _, audio in batch_records]
        enhanced_items, elapsed = enhance_audio_batch(backend, source_audio)
        progress_stats.enhance_seconds += elapsed
        progress_stats.enhance_count += len(batch_records)
        for enhanced, (source, target, duration_seconds, orig_sr, _) in zip(enhanced_items, batch_records):
            _submit_save(enhanced, source, target, duration_seconds, orig_sr)

    def _drain_saves(progress, *, force: bool = False):
        """Collect completed saves and update progress."""
        completed_save_count, completed_duration_seconds = collect_completed_saves(
            inflight_saves,
            failures,
            progress_stats,
            completed_paths,
            wait_for_completion=force,
        )
        if completed_save_count:
            _maybe_write_resumable_output_list(output_paths, completed_paths, output_list)
            progress.update(completed_duration_seconds)
        return completed_save_count

    def _update_postfix(progress):
        progress.set_postfix_str(
            build_progress_postfix(
                progress_stats,
                len(inflight_saves),
                progress_stats.processed_audio_seconds,
                total_audio_seconds,
            ),
            refresh=False,
        )

    with ThreadPoolExecutor(max_workers=save_workers) as save_pool:
        with torch.inference_mode():
            with tqdm(
                total=total_audio_seconds,
                initial=0.0,
                desc="Enhancing",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}{postfix}",
                dynamic_ncols=True,
                mininterval=0.5,
                smoothing=0.05,
            ) as progress:
                progress.set_postfix_str(
                    build_progress_postfix(
                        progress_stats,
                        len(inflight_saves),
                        0.0,
                        total_audio_seconds,
                    )
                )
                for file_batch, audio_batch, orig_sr_batch in loader:
                    source = Path(file_batch[0]).expanduser().resolve()
                    target = build_output_path(source, output_root, base_dir)
                    duration_seconds = pending_source_durations[source]
                    try:
                        audio = audio_batch.squeeze(0)
                        orig_sr = int(orig_sr_batch[0])
                        if enhance_batch_size > 1 and _can_batch_audio(audio):
                            pending_enhance_batch.append((source, target, duration_seconds, orig_sr, audio))
                            if len(pending_enhance_batch) >= enhance_batch_size:
                                _flush_pending_enhance_batch()
                        else:
                            _flush_pending_enhance_batch()
                            t0 = time.perf_counter()
                            enhanced = backend.enhance_audio(audio)
                            elapsed = time.perf_counter() - t0
                            progress_stats.enhance_seconds += elapsed
                            progress_stats.enhance_count += 1
                            _submit_save(enhanced, source, target, duration_seconds, orig_sr)
                        _drain_saves(progress)
                        if len(inflight_saves) >= max_inflight_saves:
                            _drain_saves(progress, force=True)
                        _update_postfix(progress)
                    except Exception as exc:  # pragma: no cover - operational safeguard
                        failures.append(f"{source}: {exc}")
                        _update_postfix(progress)

                _flush_pending_enhance_batch()
                _drain_saves(progress)

                completed_save_count, completed_duration_seconds = collect_completed_saves(
                    inflight_saves,
                    failures,
                    progress_stats,
                    completed_paths,
                    wait_for_completion=True,
                    block_until_all=True,
                )
                if completed_save_count:
                    write_resumable_output_list(output_paths, completed_paths, output_list)
                    progress.update(completed_duration_seconds)
                _update_postfix(progress)

    elapsed = max(time.perf_counter() - progress_stats.start_time, 1e-9)
    print(
        "Preprocess summary: "
        f"{progress_stats.enhance_count:,} files / {progress_stats.processed_audio_seconds / 3600.0:.2f}h audio "
        f"enhanced in {elapsed:.1f}s "
        f"({progress_stats.processed_audio_seconds / elapsed:.2f}x realtime) | "
        f"avg enhance {_format_average_ms(progress_stats.enhance_seconds, progress_stats.enhance_count)} | "
        f"avg save {_format_average_ms(progress_stats.save_seconds, progress_stats.save_count)} | "
        f"save queue high-water {progress_stats.queue_high_water}"
    )

    if failures:
        print("The following files failed to preprocess:", file=sys.stderr)
        for item in failures[:20]:
            print(f"  {item}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
        raise SystemExit(1)

    write_resumable_output_list(output_paths, completed_paths, output_list)
    print(f"Wrote output list with {len(completed_paths):,} entries -> {output_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
