#!/usr/bin/env python3
"""Filter FSD50K audio files into generic-noise and targeted-music lists.

The script still applies the existing license gate (CC0/CC-BY by default), but
it now keeps music-oriented clips out of the generic-noise list and writes a
second file containing room/live/speaker-biased music examples. The music
targeting is intentionally opinionated for DeepFilterNet background-music
training: pop/rock/EDM/country-ish material with vocals/song cues is preferred,
and room/live playback context is strongly favored when metadata exposes it.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

LICENSE_KEY_CANDIDATES = ("license", "clip_license", "license_url")
TITLE_KEY_CANDIDATES = ("title", "clip_title")
DESCRIPTION_KEY_CANDIDATES = ("description", "clip_description")
TAGS_KEY_CANDIDATES = ("tags", "clip_tags")

MUSIC_CORE_TERMS = (
    "music",
    "song",
    "singing",
    "singer",
    "vocal",
    "vocals",
    "background music",
    "band",
    "choir",
    "karaoke",
)
TARGET_STYLE_TERMS = (
    "pop",
    "pop music",
    "pop rock",
    "rock",
    "rock music",
    "rock and roll",
    "indie rock",
    "alternative rock",
    "country",
    "country music",
    "electronic",
    "electronic music",
    "dance",
    "dance music",
    "edm",
    "house",
    "house music",
    "techno",
    "trance",
    "dubstep",
    "disco",
    "synthpop",
)
VOCAL_TERMS = (
    "song",
    "singing",
    "singer",
    "vocal",
    "vocals",
    "lyrics",
    "lyric",
    "chorus",
    "choir",
    "karaoke",
    "band",
)
ROOM_CAPTURE_TERMS = (
    "live",
    "concert",
    "audience",
    "crowd",
    "room",
    "hall",
    "club",
    "bar",
    "pub",
    "festival",
    "gig",
    "rehearsal",
    "stage",
    "speaker",
    "loudspeaker",
    "radio",
    "boombox",
    "phone",
    "cellphone",
    "television",
    "tv",
    "field recording",
    "microphone",
    "mic",
    "camcorder",
    "reverberation",
    "reverb",
    "echo",
    "distant",
    "another room",
    "amplifier",
    "amplified",
    "pa system",
)
STUDIO_TERMS = (
    "studio",
    "mastered",
    "mastering",
    "mixdown",
    "multitrack",
    "stem",
    "stems",
    "isolated vocal",
    "dry mix",
    "loop pack",
    "sample pack",
    "midi",
)

NORMALIZE_SEPARATORS_RE = re.compile(r"[_/]+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9+]+")
WHITESPACE_RE = re.compile(r"\s+")


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


def _normalize_text(value: str) -> str:
    value = value.lower()
    value = NORMALIZE_SEPARATORS_RE.sub(" ", value)
    value = NON_ALNUM_RE.sub(" ", value)
    return WHITESPACE_RE.sub(" ", value).strip()


def _extract_first_str(info: dict[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_text_list(info: dict[str, object], keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
    return []


def _load_clip_info(meta_dir: Path, allowed_patterns: list[str]) -> dict[str, dict[str, object]]:
    clip_info: dict[str, dict[str, object]] = {}
    for json_file in ("dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"):
        json_path = meta_dir / json_file
        if not json_path.exists():
            print(f"[warn] metadata file not found: {json_path}")
            continue
        with json_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            continue
        for clip_id, raw_info in data.items():
            if not isinstance(raw_info, dict):
                continue
            license_url = ""
            for key in LICENSE_KEY_CANDIDATES:
                value = raw_info.get(key)
                if isinstance(value, str) and value.strip():
                    license_url = value
                    break
            if license_allowed(license_url, allowed_patterns):
                clip_info[str(clip_id)] = raw_info
    return clip_info


def _load_ground_truth_labels(root: Path) -> dict[str, list[str]]:
    labels_by_id: dict[str, list[str]] = {}
    ground_truth_dir = root / "FSD50K.ground_truth"
    for csv_name in ("dev.csv", "eval.csv"):
        csv_path = ground_truth_dir / csv_name
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                clip_id = (row.get("fname") or row.get("clip_id") or "").strip()
                if not clip_id:
                    continue
                labels_raw = row.get("labels") or row.get("label") or ""
                labels = [part.strip() for part in labels_raw.split(",") if part.strip()]
                if labels:
                    labels_by_id[clip_id] = labels
    return labels_by_id


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(f" {term} " in padded for term in terms)


def _build_clip_text(info: dict[str, object], labels: list[str]) -> str:
    parts: list[str] = []
    title = _extract_first_str(info, TITLE_KEY_CANDIDATES)
    description = _extract_first_str(info, DESCRIPTION_KEY_CANDIDATES)
    tags = _extract_text_list(info, TAGS_KEY_CANDIDATES)
    if title:
        parts.append(title)
    if description:
        parts.append(description)
    parts.extend(tags)
    parts.extend(labels)
    return _normalize_text(" ".join(parts))


def _classify_clip(text: str) -> tuple[bool, bool]:
    has_music_core = _contains_any(text, MUSIC_CORE_TERMS)
    has_target_style = _contains_any(text, TARGET_STYLE_TERMS)
    has_vocal_cues = _contains_any(text, VOCAL_TERMS)
    has_room_capture = _contains_any(text, ROOM_CAPTURE_TERMS)
    has_studio_cues = _contains_any(text, STUDIO_TERMS)

    is_targeted_music = (
        has_music_core and (has_target_style or has_vocal_cues) and (has_room_capture or not has_studio_cues)
    )
    return has_music_core, is_targeted_music


def _write_list(paths: list[Path], out_path: Path) -> None:
    tmp_out = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_out.open("w", encoding="utf-8") as handle:
        for path in sorted(paths):
            handle.write(str(path) + "\n")
    tmp_out.replace(out_path)


def main() -> int:
    args = parse_args()
    if not args.fsd50k_dir:
        raise SystemExit("FSD50K_DIR not set")
    if not args.list_dir:
        raise SystemExit("LIST_DIR not set")

    root = Path(args.fsd50k_dir)
    meta_dir = root / "FSD50K.metadata"

    clip_info = _load_clip_info(meta_dir, args.allowed_patterns)
    allowed_ids = set(clip_info)

    if not allowed_ids:
        raise SystemExit("No clips matched the allowed license patterns")

    labels_by_id = _load_ground_truth_labels(root)

    # Find audio files
    candidates: list[Path] = []
    for sub in [root / "FSD50K.dev_audio", root / "FSD50K.eval_audio"]:
        if sub.exists():
            candidates.extend(sub.rglob("*.wav"))

    generic_noise: list[Path] = []
    targeted_music: list[Path] = []
    dropped_music_like = 0
    for candidate in sorted(candidates):
        clip_id = candidate.stem
        if clip_id not in allowed_ids:
            continue
        text = _build_clip_text(clip_info.get(clip_id, {}), labels_by_id.get(clip_id, []))
        has_music_core, is_targeted_music = _classify_clip(text)
        if is_targeted_music:
            targeted_music.append(candidate)
        elif has_music_core:
            dropped_music_like += 1
        else:
            generic_noise.append(candidate)

    generic_noise_out = Path(args.list_dir) / "fsd50k_filtered.txt"
    targeted_music_out = Path(args.list_dir) / "fsd50k_music_targeted.txt"
    _write_list(generic_noise, generic_noise_out)
    _write_list(targeted_music, targeted_music_out)

    print(f"[ok] wrote {len(generic_noise)} generic-noise entries -> {generic_noise_out}")
    print(f"[ok] wrote {len(targeted_music)} targeted-music entries -> {targeted_music_out}")
    if dropped_music_like:
        print(
            f"[info] dropped {dropped_music_like} music-like clips that did not match the targeted room/live/song heuristic"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
