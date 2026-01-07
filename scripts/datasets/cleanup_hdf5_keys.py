#!/usr/bin/env python3
"""Remove HDF5 dataset entries with keys that don't match the expected format.

Expected format: relative path with '/' replaced by '_'
  e.g., '.._raw_LibriSpeech_dev-clean_1272_128104_1272-128104-0000.flac'

Invalid format: basename only (no path separators)
  e.g., '1272-128104-0000.flac'
"""

import argparse
import sys
from typing import TYPE_CHECKING, cast

import h5py

if TYPE_CHECKING:
    from h5py import Group  # noqa: F401 (used by cast())


def is_valid_key(key: str) -> bool:
    """Check if key matches the expected format (relative path with _ separators).

    Valid keys have multiple _ separators from path components.
    Invalid keys are just basenames with no path structure.
    """
    # Valid keys have path structure: they contain multiple underscores
    # and typically start with '.._' or have directory-like patterns
    #
    # Examples of VALID keys:
    #   .._raw_LibriSpeech_dev-clean_1272_128104_1272-128104-0000.flac
    #   raw_VCTK_wav48_p227_p227_001_mic1.flac
    #
    # Examples of INVALID keys (basename only):
    #   1272-128104-0000.flac
    #   p227_001_mic1.flac

    # Heuristic: valid keys have at least 3 underscore-separated parts
    # that look like directory structure (not just filename parts)
    parts = key.split("_")

    # If it starts with '..' it's definitely a relative path
    if key.startswith(".."):
        return True

    # If it has many parts and some look like directory names, it's valid
    # Directory names are typically: 'raw', 'LibriSpeech', 'VCTK', 'wav48', etc.
    dir_patterns = {"raw", "LibriSpeech", "VCTK", "wav48", "dev", "test", "train", "clean", "other"}

    for part in parts:
        if part in dir_patterns or part.startswith("dev-") or part.startswith("test-") or part.startswith("train-"):
            return True

    # Fallback: if the key has fewer than 4 parts, it's likely just a filename
    return len(parts) >= 5


def cleanup_hdf5(file_path: str, dry_run: bool = True) -> dict:
    """Remove invalid keys from HDF5 file.

    Args:
        file_path: Path to HDF5 file
        dry_run: If True, only report what would be deleted

    Returns:
        Dict with counts of valid/invalid keys per group
    """
    mode = "r" if dry_run else "a"
    results = {}

    with h5py.File(file_path, mode) as f:
        print(f"File: {file_path}")
        print(f"Attributes: {dict(f.attrs)}")
        print()

        for group_name in f.keys():
            grp = cast("Group", f[group_name])
            all_keys = list(grp.keys())

            valid_keys = [k for k in all_keys if is_valid_key(k)]
            invalid_keys = [k for k in all_keys if not is_valid_key(k)]

            results[group_name] = {
                "total": len(all_keys),
                "valid": len(valid_keys),
                "invalid": len(invalid_keys),
                "invalid_keys": invalid_keys[:10],  # Sample for display
            }

            print(f"Group '{group_name}':")
            print(f"  Total entries: {len(all_keys)}")
            print(f"  Valid keys:    {len(valid_keys)}")
            print(f"  Invalid keys:  {len(invalid_keys)}")

            if invalid_keys:
                print("  Sample invalid keys:")
                for k in invalid_keys[:5]:
                    print(f"    - {k}")
                if len(invalid_keys) > 5:
                    print(f"    ... and {len(invalid_keys) - 5} more")

            if not dry_run and invalid_keys:
                print(f"  Deleting {len(invalid_keys)} invalid entries...")
                for k in invalid_keys:
                    del grp[k]
                print("  Done.")

            print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Remove HDF5 entries with invalid key formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("hdf5_file", help="Path to HDF5 file to clean up")
    parser.add_argument("--execute", action="store_true", help="Actually delete invalid keys (default is dry-run)")
    parser.add_argument("--show-all", action="store_true", help="Show all invalid keys (not just sample)")

    args = parser.parse_args()

    dry_run = not args.execute

    if dry_run:
        print("=" * 60)
        print("DRY RUN - No changes will be made")
        print("Use --execute to actually delete invalid keys")
        print("=" * 60)
        print()
    else:
        print("=" * 60)
        print("EXECUTING - Invalid keys will be DELETED")
        print("=" * 60)
        print()

        confirm = input("Are you sure? Type 'yes' to continue: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(1)
        print()

    results = cleanup_hdf5(args.hdf5_file, dry_run=dry_run)

    # Summary
    total_invalid = sum(r["invalid"] for r in results.values())
    total_valid = sum(r["valid"] for r in results.values())

    print("=" * 60)
    print("Summary:")
    print(f"  Total valid:   {total_valid}")
    print(f"  Total invalid: {total_invalid}")

    if dry_run and total_invalid > 0:
        print()
        print("Run with --execute to delete invalid entries.")


if __name__ == "__main__":
    main()
