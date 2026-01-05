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
        available = audb.available(name)
        if version not in available:
            print(f"[warn] audb version '{version}' not found for {name}; using latest")
            return None
    except Exception:
        pass
    return version


def call_load(name: str, root: str | None, version: str | None):
    load = audb.load_to
    # If load_to is a partial with root already set, do not pass root again.
    if isinstance(load, functools.partial) and "root" in (load.keywords or {}):
        if version:
            return load(name, version=version)
        return load(name)

    params = list(inspect.signature(load).parameters)
    # Some audb versions use load_to(root, name, ...)
    if params and params[0] in ("root", "path"):
        target_root = root or os.getcwd()
        cleanup_tmp(target_root)
        try:
            if version:
                return load(target_root, name, version=version)
            return load(target_root, name)
        except RuntimeError as e:
            if "temporary directory" in str(e) or "audb~" in str(e):
                cleanup_tmp(target_root)
                if version:
                    return load(target_root, name, version=version)
                return load(target_root, name)
            raise

    # Default: load_to(name, version=?, root=?)
    if root is not None:
        cleanup_tmp(root)
        try:
            return load(name, version=version, root=root)
        except RuntimeError as e:
            if "temporary directory" in str(e) or "audb~" in str(e):
                cleanup_tmp(root)
                return load(name, version=version, root=root)
            raise

    if version:
        return load(name, version=version)
    return load(name)


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
