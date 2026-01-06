#!/usr/bin/env python3
"""Filter FSD50K audio files by license (CC0/CC-BY only by default).

Reads license info from the JSON metadata files (dev_clips_info_FSD50K.json
and eval_clips_info_FSD50K.json) provided by FSD50K.
"""
import argparse
import json
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter FSD50K clips by license type")
    parser.add_argument("--fsd50k-dir", default=os.environ.get("FSD50K_DIR"))
    parser.add_argument("--list-dir", default=os.environ.get("LIST_DIR"))
    parser.add_argument(
        "--allowed-patterns",
        nargs="*",
        default=[
            r"creativecommons\.org/publicdomain/zero",  # CC0
            r"creativecommons\.org/licenses/by/",  # CC-BY (any version)
        ],
        help="Regex patterns to match allowed license URLs",
    )
    return parser.parse_args()


def license_allowed(license_url: str, patterns: list[str]) -> bool:
    """Check if a license URL matches any of the allowed patterns."""
    if not license_url:
        return False
    for pattern in patterns:
        if re.search(pattern, license_url, re.IGNORECASE):
            return True
    return False


def main() -> int:
    args = parse_args()
    if not args.fsd50k_dir:
        raise SystemExit("FSD50K_DIR not set")
    if not args.list_dir:
        raise SystemExit("LIST_DIR not set")

    root = Path(args.fsd50k_dir)
    meta_dir = root / "FSD50K.metadata"

    # Load license info from JSON metadata files
    allowed_ids: set[str] = set()
    for json_file in ["dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"]:
        json_path = meta_dir / json_file
        if not json_path.exists():
            print(f"[warn] metadata file not found: {json_path}")
            continue
        with json_path.open() as f:
            data = json.load(f)
        for clip_id, info in data.items():
            license_url = info.get("license", "")
            if license_allowed(license_url, args.allowed_patterns):
                allowed_ids.add(clip_id)

    if not allowed_ids:
        raise SystemExit("No clips matched the allowed license patterns")

    # Find audio files
    candidates = []
    for sub in [root / "FSD50K.dev_audio", root / "FSD50K.eval_audio"]:
        if sub.exists():
            candidates.extend(sub.rglob("*.wav"))

    # Filter by allowed IDs (filename without extension)
    filtered = [p for p in candidates if p.stem in allowed_ids]

    out = Path(args.list_dir) / "fsd50k_filtered.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for p in sorted(filtered):
            f.write(str(p) + "\n")

    print(f"[ok] wrote {len(filtered)} entries -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
