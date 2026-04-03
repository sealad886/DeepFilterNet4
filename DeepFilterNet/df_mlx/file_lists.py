"""Shared helpers for reading line-based audio file list files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class ResolvedDatasetFileLists:
    """Resolved sibling-aware dataset file-list bundle for raw-list training."""

    speech_list: str | None
    noise_list: str | None
    music_list: str | None
    rir_list: str | None
    auto_selected: dict[str, str]


def read_file_list(
    path: str | Path,
    *,
    split_tab: bool = False,
    check_exists: bool = False,
    warn_missing_entries: bool = False,
    warn_missing_list: bool = False,
) -> List[str]:
    """Read a text file containing one path per line.

    Args:
        path: Path to list file.
        split_tab: If True, keep only first tab-separated column.
        check_exists: If True, skip entries that do not exist on disk.
        warn_missing_entries: Print warning when a listed entry does not exist.
        warn_missing_list: Print warning and return [] if list file is missing.
    """
    file_path = Path(path)
    if warn_missing_list and not file_path.exists():
        print(f"Warning: File list not found: {file_path}")
        return []

    files: List[str] = []
    with open(file_path) as f:
        for line in f:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            if split_tab and "\t" in entry:
                entry = entry.split("\t", 1)[0]
            if check_exists and not os.path.exists(entry):
                if warn_missing_entries:
                    print(f"Warning: File not found: {entry}")
                continue
            files.append(entry)
    return files


def _normalize_optional_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser())


def _candidate_list_dirs(paths: Iterable[str | Path | None]) -> list[Path]:
    ordered: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        if raw_path is None:
            continue
        parent = Path(raw_path).expanduser().parent
        try:
            resolved = parent.resolve()
        except OSError:
            resolved = parent
        if resolved not in seen:
            ordered.append(resolved)
            seen.add(resolved)
    return ordered


def _pick_first_existing_list(
    candidate_dirs: list[Path],
    candidate_names: tuple[str, ...],
) -> str | None:
    for list_dir in candidate_dirs:
        for name in candidate_names:
            candidate = list_dir / name
            if candidate.is_file() and candidate.stat().st_size > 0:
                return str(candidate)
    return None


def resolve_dataset_file_lists(
    *,
    speech_list: str | Path | None,
    noise_list: str | Path | None = None,
    music_list: str | Path | None = None,
    rir_list: str | Path | None = None,
) -> ResolvedDatasetFileLists:
    """Resolve sibling dataset list defaults for raw-list training flows.

    Preference order matches the newer dataset-prep outputs:
    - noise: ``noise_all.txt`` → ``noise_music.txt``
    - music: ``background_music.prepared_merged.txt`` →
      ``background_music.txt`` → ``background_music_expanded.txt``
    - RIR: ``rir_all.txt``
    """

    resolved_speech = _normalize_optional_path(speech_list)
    resolved_noise = _normalize_optional_path(noise_list)
    resolved_music = _normalize_optional_path(music_list)
    resolved_rir = _normalize_optional_path(rir_list)
    auto_selected: dict[str, str] = {}

    candidate_dirs = _candidate_list_dirs((resolved_speech, resolved_noise, resolved_music, resolved_rir))

    if resolved_noise is None:
        resolved_noise = _pick_first_existing_list(candidate_dirs, ("noise_all.txt", "noise_music.txt"))
        if resolved_noise is not None:
            auto_selected["noise_list"] = resolved_noise

    if resolved_music is None:
        resolved_music = _pick_first_existing_list(
            candidate_dirs,
            (
                "background_music.prepared_merged.txt",
                "background_music.txt",
                "background_music_expanded.txt",
            ),
        )
        if resolved_music is not None:
            auto_selected["music_list"] = resolved_music

    if resolved_rir is None:
        resolved_rir = _pick_first_existing_list(candidate_dirs, ("rir_all.txt",))
        if resolved_rir is not None:
            auto_selected["rir_list"] = resolved_rir

    return ResolvedDatasetFileLists(
        speech_list=resolved_speech,
        noise_list=resolved_noise,
        music_list=resolved_music,
        rir_list=resolved_rir,
        auto_selected=auto_selected,
    )
