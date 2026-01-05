#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsd50k-dir", default=os.environ.get("FSD50K_DIR"))
    parser.add_argument("--list-dir", default=os.environ.get("LIST_DIR"))
    parser.add_argument("--id-col", default=os.environ.get("FSD50K_ID_COL", "fname"))
    parser.add_argument("--license-col", default=os.environ.get("FSD50K_LICENSE_COL", "license"))
    parser.add_argument(
        "--allowed",
        nargs="*",
        default=["CC0", "CC-BY", "CC BY", "CC-BY-4.0", "CC BY 4.0"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.fsd50k_dir:
        raise SystemExit("FSD50K_DIR not set")
    if not args.list_dir:
        raise SystemExit("LIST_DIR not set")

    root = Path(args.fsd50k_dir)
    meta_dir = root / "FSD50K.metadata"
    csv_path = meta_dir / "FSD50K.metadata.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing metadata CSV at {csv_path}")

    allowed = set(args.allowed)

    allowed_ids = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            lic = (row.get(args.license_col) or "").strip()
            if lic in allowed:
                allowed_ids.append(row[args.id_col])

    candidates = []
    for sub in [root / "FSD50K.dev_audio", root / "FSD50K.eval_audio"]:
        if sub.exists():
            candidates.extend(sub.rglob("*.wav"))

    id_set = set(allowed_ids)
    filtered = [p for p in candidates if p.name in id_set]

    out = Path(args.list_dir) / "fsd50k_filtered.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for p in sorted(filtered):
            f.write(str(p) + "\n")

    print(f"[ok] wrote {len(filtered)} entries -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
