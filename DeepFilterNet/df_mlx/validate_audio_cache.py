#!/usr/bin/env python3
"""Validate audio cache shard integrity.

This script checks the validity of sharded NPZ cache files:
1. Each shard contains the files it's supposed to contain
2. No duplication between shards (each file appears exactly once)
3. Shard files are not corrupted
4. Index.json matches actual shard contents

Usage:
    python -m df_mlx.validate_audio_cache /path/to/audio_cache [--fix]

Options:
    --fix           Attempt to fix issues (rebuild index, remove corrupt shards)
    --verbose       Show detailed information about each shard
    --category      Only validate a specific category (speech/noise/rir)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm


class ValidationError:
    """Represents a validation error found during checking."""

    def __init__(self, error_type: str, message: str, shard_path: Optional[Path] = None, severity: str = "error"):
        self.error_type = error_type
        self.message = message
        self.shard_path = shard_path
        self.severity = severity  # 'error', 'warning', 'info'

    def __str__(self):
        prefix = f"[{self.severity.upper()}]"
        if self.shard_path:
            return f"{prefix} {self.shard_path.name}: {self.message}"
        return f"{prefix} {self.message}"


def load_shard_safely(shard_path: Path) -> Tuple[Optional[Dict], Optional[str]]:
    """Load a shard file and return its contents or error message."""
    try:
        with np.load(shard_path, allow_pickle=True) as data:
            return dict(data), None
    except Exception as e:
        return None, str(e)


def validate_shard_structure(shard_path: Path, shard_data: Dict, verbose: bool = False) -> List[ValidationError]:
    """Validate the internal structure of a single shard."""
    errors = []

    # Check for __paths__ array
    if "__paths__" not in shard_data:
        errors.append(
            ValidationError(
                "missing_paths", "Shard missing __paths__ array (legacy format)", shard_path, severity="error"
            )
        )
        return errors

    paths = shard_data["__paths__"]

    # Check paths is a valid array
    if not isinstance(paths, np.ndarray):
        errors.append(
            ValidationError("invalid_paths_type", f"__paths__ is not a numpy array: {type(paths)}", shard_path)
        )
        return errors

    # Count audio arrays
    audio_keys = [k for k in shard_data.keys() if k.startswith("audio_")]
    num_audio = len(audio_keys)
    num_paths = len(paths)

    if num_audio != num_paths:
        errors.append(
            ValidationError("count_mismatch", f"Mismatch: {num_paths} paths but {num_audio} audio arrays", shard_path)
        )

    # Verify audio array naming sequence
    expected_keys = {f"audio_{i:05d}" for i in range(num_paths)}
    actual_keys = set(audio_keys)

    missing_keys = expected_keys - actual_keys
    extra_keys = actual_keys - expected_keys

    if missing_keys:
        errors.append(
            ValidationError(
                "missing_audio_keys",
                f"Missing audio keys: {sorted(missing_keys)[:5]}{'...' if len(missing_keys) > 5 else ''}",
                shard_path,
            )
        )

    if extra_keys:
        errors.append(
            ValidationError(
                "extra_audio_keys",
                f"Unexpected audio keys: {sorted(extra_keys)[:5]}{'...' if len(extra_keys) > 5 else ''}",
                shard_path,
                severity="warning",
            )
        )

    # Check for empty audio arrays
    empty_count = 0
    for key in audio_keys:
        arr = shard_data[key]
        if not isinstance(arr, np.ndarray):
            errors.append(ValidationError("invalid_audio_type", f"{key} is not a numpy array: {type(arr)}", shard_path))
        elif arr.size == 0:
            empty_count += 1

    if empty_count > 0:
        errors.append(
            ValidationError("empty_audio", f"{empty_count} empty audio arrays found", shard_path, severity="warning")
        )

    # Check for empty/invalid paths
    empty_paths = sum(1 for p in paths if not p or str(p).strip() == "")
    if empty_paths > 0:
        errors.append(ValidationError("empty_paths", f"{empty_paths} empty path entries found", shard_path))

    if verbose and not errors:
        print(f"  ✓ {shard_path.name}: {num_paths} files, all valid")

    return errors


def validate_category(
    cache_dir: Path, category: str, verbose: bool = False
) -> Tuple[List[ValidationError], Dict[str, Set[Tuple[str, str]]]]:
    """Validate all shards for a category.

    Returns:
        Tuple of (errors list, dict mapping path -> set of (shard, key) locations)
    """
    errors = []
    path_locations: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    shard_dir = cache_dir / category
    if not shard_dir.exists():
        if verbose:
            print(f"  Category '{category}': no directory found")
        return errors, path_locations

    shard_files = sorted(shard_dir.glob("shard_*.npz"))
    if not shard_files:
        if verbose:
            print(f"  Category '{category}': no shard files found")
        return errors, path_locations

    print(f"\nValidating {category}: {len(shard_files)} shards")

    # Check for gaps in shard numbering
    shard_nums = []
    for sf in shard_files:
        try:
            num = int(sf.stem.split("_")[1])
            shard_nums.append(num)
        except (IndexError, ValueError):
            errors.append(ValidationError("invalid_shard_name", "Invalid shard filename format", sf))

    if shard_nums:
        shard_nums.sort()
        expected = list(range(shard_nums[0], shard_nums[-1] + 1))
        missing = set(expected) - set(shard_nums)
        if missing:
            errors.append(
                ValidationError(
                    "shard_gaps",
                    f"Gaps in shard sequence: missing shards {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}",
                    severity="warning",
                )
            )

    # Validate each shard
    total_files = 0
    for shard_path in tqdm(shard_files, desc=f"  Checking {category}", unit="shard"):
        shard_data, load_error = load_shard_safely(shard_path)

        if load_error or shard_data is None:
            errors.append(
                ValidationError("corrupt_shard", f"Failed to load: {load_error or 'unknown error'}", shard_path)
            )
            continue

        # Validate structure
        struct_errors = validate_shard_structure(shard_path, shard_data, verbose)
        errors.extend(struct_errors)

        # Track file locations for duplicate detection
        if "__paths__" in shard_data:
            paths = shard_data["__paths__"]
            shard_rel = f"{category}/{shard_path.name}"
            for i, path in enumerate(paths):
                path_str = str(path)
                key = f"audio_{i:05d}"
                path_locations[path_str].add((shard_rel, key))
            total_files += len(paths)

    print(f"  Total files in {category}: {total_files:,}")

    return errors, path_locations


def check_duplicates(path_locations: Dict[str, Set[Tuple[str, str]]]) -> List[ValidationError]:
    """Check for files appearing in multiple shards."""
    errors = []
    duplicates = {path: locs for path, locs in path_locations.items() if len(locs) > 1}

    if duplicates:
        errors.append(
            ValidationError(
                "duplicates_found", f"Found {len(duplicates)} files appearing in multiple shards", severity="error"
            )
        )

        # Show first few duplicates
        for i, (path, locs) in enumerate(list(duplicates.items())[:5]):
            loc_str = ", ".join(f"{s}:{k}" for s, k in sorted(locs))
            errors.append(ValidationError("duplicate_detail", f"  '{path}' in: {loc_str}", severity="info"))

        if len(duplicates) > 5:
            errors.append(
                ValidationError("duplicate_detail", f"  ... and {len(duplicates) - 5} more duplicates", severity="info")
            )

    return errors


def validate_index(
    cache_dir: Path, all_path_locations: Dict[str, Dict[str, Set[Tuple[str, str]]]], verbose: bool = False
) -> List[ValidationError]:
    """Validate index.json against actual shard contents."""
    errors = []
    index_path = cache_dir / "index.json"

    if not index_path.exists():
        errors.append(
            ValidationError(
                "missing_index", "index.json not found (can be rebuilt with --rebuild-index)", severity="warning"
            )
        )
        return errors

    try:
        with open(index_path) as f:
            index_data = json.load(f)
    except Exception as e:
        errors.append(ValidationError("corrupt_index", f"Failed to load index.json: {e}"))
        return errors

    print("\nValidating index.json...")

    for category in ["speech", "noise", "rir"]:
        if category not in index_data:
            if category in all_path_locations and all_path_locations[category]:
                errors.append(
                    ValidationError(
                        "missing_category_in_index",
                        f"Category '{category}' exists on disk but not in index",
                        severity="warning",
                    )
                )
            continue

        index_entries = index_data[category]
        shard_entries = all_path_locations.get(category, {})

        index_paths = set(index_entries.keys())
        shard_paths = set(shard_entries.keys())

        # Files in index but not in shards
        missing_from_shards = index_paths - shard_paths
        if missing_from_shards:
            errors.append(
                ValidationError(
                    "index_stale",
                    f"{category}: {len(missing_from_shards)} index entries point to missing files",
                    severity="error",
                )
            )
            if verbose:
                for path in list(missing_from_shards)[:3]:
                    shard, key = index_entries[path]
                    errors.append(
                        ValidationError(
                            "index_stale_detail", f"  '{path}' -> {shard}:{key} (not found)", severity="info"
                        )
                    )

        # Files in shards but not in index
        missing_from_index = shard_paths - index_paths
        if missing_from_index:
            errors.append(
                ValidationError(
                    "index_incomplete",
                    f"{category}: {len(missing_from_index)} shard entries not in index",
                    severity="warning",
                )
            )

        # Verify index entries match shard contents
        mismatches = 0
        for path, (idx_shard, idx_key) in index_entries.items():
            if path in shard_entries:
                actual_locs = shard_entries[path]
                expected_loc = (idx_shard, idx_key)
                if expected_loc not in actual_locs:
                    mismatches += 1
                    if verbose and mismatches <= 3:
                        actual_str = ", ".join(f"{s}:{k}" for s, k in sorted(actual_locs))
                        errors.append(
                            ValidationError(
                                "index_mismatch_detail",
                                f"  '{path}': index says {idx_shard}:{idx_key}, actual: {actual_str}",
                                severity="info",
                            )
                        )

        if mismatches > 0:
            errors.append(
                ValidationError(
                    "index_mismatch",
                    f"{category}: {mismatches} index entries don't match shard contents",
                    severity="error",
                )
            )

        if verbose:
            print(f"  {category}: {len(index_entries)} index entries, {len(shard_entries)} shard entries")

    return errors


def print_summary(errors: List[ValidationError]) -> Tuple[int, int, int]:
    """Print error summary and return counts by severity."""
    error_count = sum(1 for e in errors if e.severity == "error")
    warning_count = sum(1 for e in errors if e.severity == "warning")
    info_count = sum(1 for e in errors if e.severity == "info")

    if not errors:
        print("\n✅ All validations passed!")
        return 0, 0, 0

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    # Group by type
    by_type: Dict[str, List[ValidationError]] = defaultdict(list)
    for e in errors:
        by_type[e.error_type].append(e)

    for error_type, type_errors in sorted(by_type.items()):
        severity = type_errors[0].severity
        icon = "❌" if severity == "error" else "⚠️" if severity == "warning" else "ℹ️"
        print(f"\n{icon} {error_type.replace('_', ' ').title()} ({len(type_errors)})")
        for e in type_errors[:10]:
            print(f"   {e}")
        if len(type_errors) > 10:
            print(f"   ... and {len(type_errors) - 10} more")

    print("\n" + "-" * 60)
    print(f"Summary: {error_count} errors, {warning_count} warnings, {info_count} info")

    return error_count, warning_count, info_count


def generate_report(
    cache_dir: Path, all_path_locations: Dict[str, Dict[str, Set[Tuple[str, str]]]], errors: List[ValidationError]
) -> Dict:
    """Generate a structured report of the validation results."""
    report = {
        "cache_dir": str(cache_dir),
        "categories": {},
        "errors": [],
        "warnings": [],
        "duplicates": [],
    }

    for category, path_locs in all_path_locations.items():
        shard_dir = cache_dir / category
        shard_files = list(shard_dir.glob("shard_*.npz")) if shard_dir.exists() else []

        report["categories"][category] = {
            "num_shards": len(shard_files),
            "num_files": len(path_locs),
        }

        # Find duplicates for this category
        duplicates = {p: list(locs) for p, locs in path_locs.items() if len(locs) > 1}
        if duplicates:
            report["duplicates"].extend(
                [{"category": category, "path": p, "locations": locs} for p, locs in duplicates.items()]
            )

    for e in errors:
        entry = {
            "type": e.error_type,
            "message": e.message,
            "shard": str(e.shard_path) if e.shard_path else None,
        }
        if e.severity == "error":
            report["errors"].append(entry)
        elif e.severity == "warning":
            report["warnings"].append(entry)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate audio cache shard integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation
    python -m df_mlx.validate_audio_cache /path/to/cache

    # Verbose output
    python -m df_mlx.validate_audio_cache /path/to/cache --verbose

    # Check only speech category
    python -m df_mlx.validate_audio_cache /path/to/cache --category speech

    # Generate JSON report
    python -m df_mlx.validate_audio_cache /path/to/cache --json-report report.json
        """,
    )

    parser.add_argument(
        "cache_dir",
        type=str,
        help="Path to audio cache directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information about each shard",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["speech", "noise", "rir"],
        help="Only validate a specific category",
    )
    parser.add_argument(
        "--json-report",
        type=str,
        help="Write JSON report to file",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix issues by rebuilding index (requires running build_audio_cache.py with --rebuild-index)",
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        sys.exit(1)

    print("=" * 60)
    print("DeepFilterNet Audio Cache Validator")
    print("=" * 60)
    print(f"Cache directory: {cache_dir}")

    all_errors: List[ValidationError] = []
    all_path_locations: Dict[str, Dict[str, Set[Tuple[str, str]]]] = {}

    # Determine categories to validate
    categories = [args.category] if args.category else ["speech", "noise", "rir"]

    # Validate each category
    for category in categories:
        cat_errors, path_locations = validate_category(cache_dir, category, args.verbose)
        all_errors.extend(cat_errors)
        all_path_locations[category] = path_locations

        # Check for duplicates within category
        dup_errors = check_duplicates(path_locations)
        all_errors.extend(dup_errors)

    # Validate index.json
    index_errors = validate_index(cache_dir, all_path_locations, args.verbose)
    all_errors.extend(index_errors)

    # Print summary
    error_count, warning_count, info_count = print_summary(all_errors)

    # Generate JSON report if requested
    if args.json_report:
        report = generate_report(cache_dir, all_path_locations, all_errors)
        with open(args.json_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to {args.json_report}")

    # Suggest fixes
    if all_errors and not args.fix:
        print("\nTo attempt fixes, run:")
        print(f"  python -m df_mlx.build_audio_cache --output-dir {cache_dir} --rebuild-index --resume ...")

    # Exit code based on errors
    if error_count > 0:
        sys.exit(1)
    elif warning_count > 0:
        sys.exit(0)  # Warnings are not fatal
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
