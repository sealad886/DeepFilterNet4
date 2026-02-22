#!/usr/bin/env python3
import argparse
import os
import shutil
import sys

import audb


def cleanup_tmp(root_path: str | None) -> None:
    if not root_path:
        return
    tmp_path = root_path + "~"
    if os.path.isdir(tmp_path):
        try:
            shutil.rmtree(tmp_path)
            print(f"[warn] removed stale audb temp dir: {tmp_path}")
        except Exception as e:
            print(f"[warn] failed to remove temp dir {tmp_path}: {e}")


def normalize_version(version: str | None, name: str) -> str | None:
    if version is None:
        return None
    v = version.strip().lower()
    if v in ("", "latest", "none"):
        return None
    try:
        # audb.available() returns DataFrame of available datasets, filter by name
        available = audb.available()
        if name in available.index:
            db_versions = available.loc[name].get("version", [])
            if isinstance(db_versions, str):
                db_versions = [db_versions]
            if version not in db_versions:
                print(f"[warn] audb version '{version}' not found for {name}; using latest")
                return None
    except Exception:
        pass
    return version


def call_load(name: str, root: str | None, version: str | None):
    """Load audb dataset with version-agnostic parameter handling."""
    target_root = root or os.getcwd()
    cleanup_tmp(target_root)

    # Build kwargs for audb.load_to (preferred) or audb.load
    load_kwargs: dict[str, str] = {"name": name}
    if version:
        load_kwargs["version"] = version

    # Prefer load_to which accepts a root parameter
    if hasattr(audb, "load_to"):
        load_kwargs["root"] = target_root
        load_fn = audb.load_to
    else:
        load_fn = audb.load

    # Call the load function
    try:
        return load_fn(**load_kwargs)
    except RuntimeError as e:
        if "temporary directory" in str(e) or "audb~" in str(e):
            cleanup_tmp(target_root)
            return load_fn(**load_kwargs)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=os.environ.get("AUDB_NAME"))
    parser.add_argument("--version", default=os.environ.get("AUDB_VERSION"))
    parser.add_argument("--root", default=os.environ.get("AUDB_ROOT"))
    args = parser.parse_args()

    if not args.name:
        print("[error] AUDB_NAME is required", file=sys.stderr)
        return 2

    version = normalize_version(args.version, args.name)
    call_load(args.name, args.root, version)
    print(f"[ok] audb downloaded {args.name} -> {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
