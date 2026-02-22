"""Shared helpers for reading line-based audio file list files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


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
