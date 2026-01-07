#!/usr/bin/env python3
import argparse
import functools
import inspect
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

    # Build kwargs for audb.load
    load_kwargs = {"name": name}
    if version:
        load_kwargs["version"] = version

    # Check if audb.load_to or audb.load accepts root parameter
    load_fn = getattr(audb, "load_to", None) or audb.load

    # If load_fn is a partial with root already bound, don't pass it again
    if isinstance(load_fn, functools.partial):
        bound_keys = load_fn.keywords or {}
        if "root" not in bound_keys:
            load_kwargs["root"] = target_root
    else:
        # Check function signature for root parameter
        sig = inspect.signature(load_fn)
        params = list(sig.parameters.keys())

        # Handle different audb API versions
        if params and params[0] in ("root", "path"):
            # Old API: load_to(root, name, ...)
            try:
                if version:
                    return load_fn(target_root, name, version=version)
                return load_fn(target_root, name)
            except RuntimeError as e:
                if "temporary directory" in str(e) or "audb~" in str(e):
                    cleanup_tmp(target_root)
                    if version:
                        return load_fn(target_root, name, version=version)
                    return load_fn(target_root, name)
                raise
        elif "root" in params:
            load_kwargs["root"] = target_root

    # Call the load function with appropriate kwargs
    try:
        return load_fn(**load_kwargs)
    except RuntimeError as e:
        if "temporary directory" in str(e) or "audb~" in str(e):
            cleanup_tmp(target_root)
            return load_fn(**load_kwargs)
        raise
    except TypeError:
        # Fallback: try with positional arguments if kwargs fail
        if version:
            return load_fn(name, version=version)
        return load_fn(name)


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
