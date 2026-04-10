#!/usr/bin/env python3
"""Curate chart-style background-music file lists from open music corpora.

This script builds the canonical dedicated background-music lists used by
DeepFilterNet training from larger open corpora. The current target is
roughly the genre/style mix associated with mainstream compilation CDs like
"Now That's What I Call Music": pop, pop-rock/alternative rock, dance/EDM,
country-pop/americana, and adjacent R&B/hip-hop crossover material.

Inputs:
- FMA metadata/audio (preferred automated source)
- MTG-Jamendo metadata/audio (preferred when already present; optional due size)

Outputs:
- background_music.txt: canonical curated list (target-count capped)
- background_music_expanded.txt: all eligible candidates (source-prioritized)
- background_music_fma.txt / background_music_mtg_jamendo.txt: source lists
- background_music_catalog.tsv: scored audit table
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_BUCKET_WEIGHTS: dict[str, float] = {
    "pop": 0.26,
    "rock": 0.24,
    "electronic": 0.24,
    "country": 0.12,
    "urban": 0.14,
}

_BUCKET_KEYWORDS: dict[str, set[str]] = {
    "pop": {
        "pop",
        "dancepop",
        "electropop",
        "synthpop",
        "indiepop",
        "teenpop",
        "commercial",
        "adultcontemporary",
        "poprock",
    },
    "rock": {
        "rock",
        "alternativerock",
        "indierock",
        "poprock",
        "hardrock",
        "punkrock",
        "newwave",
        "postpunk",
        "garagerock",
    },
    "electronic": {
        "electronic",
        "dance",
        "edm",
        "house",
        "deephouse",
        "electrohouse",
        "disco",
        "nu disco",
        "nudisco",
        "trance",
        "techno",
        "synthwave",
        "drumnbass",
        "drumandbass",
        "dubstep",
        "electropop",
    },
    "country": {
        "country",
        "countrypop",
        "countryrock",
        "americana",
        "altcountry",
        "folkpop",
        "folkrock",
        "rootsrock",
    },
    "urban": {
        "hiphop",
        "rap",
        "rnb",
        "soul",
        "funk",
        "neosoul",
        "trap",
        "urban",
    },
}

_POSITIVE_TOKENS: set[str] = {
    "vocal",
    "vocals",
    "voice",
    "song",
    "singer",
    "femalevocal",
    "malevocal",
    "anthemic",
    "uplifting",
    "radio",
    "hit",
    "chart",
}

_NEGATIVE_TOKENS: set[str] = {
    "ambient",
    "atmospheric",
    "background",
    "classical",
    "opera",
    "orchestral",
    "symphonic",
    "soundtrack",
    "score",
    "cinematic",
    "jazz",
    "freejazz",
    "bebop",
    "experimental",
    "avantgarde",
    "fieldrecording",
    "spokenword",
    "podcast",
    "speech",
    "meditation",
    "newage",
    "drone",
    "noise",
    "lofi",
    "instrumental",
    "acoustic",
    "solo",
}

_SOURCE_ORDER = {"mtg_jamendo": 0, "fma": 1}


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _token_variants(raw: str) -> set[str]:
    raw = raw.strip().lower()
    if not raw:
        return set()
    variants = {raw, _normalize_token(raw)}
    for part in re.split(r"[^a-z0-9]+", raw):
        part = part.strip()
        if part:
            variants.add(part)
            variants.add(_normalize_token(part))
    return {variant for variant in variants if variant}


@dataclass(frozen=True)
class MusicCandidate:
    source: str
    path: str
    title: str
    artist: str
    license_name: str
    duration_seconds: float
    primary_bucket: str
    score: int
    bucket_scores: dict[str, int] = field(default_factory=dict)
    matched_tokens: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class CuratedLists:
    curated: list[MusicCandidate]
    expanded: list[MusicCandidate]
    by_source: dict[str, list[MusicCandidate]]


@dataclass
class LoadedFmaMetadata:
    rows: list[dict[tuple[str, str], str]]
    genres: dict[int, str]
    audio_dir: Path


@dataclass
class LoadedMtgMetadata:
    rows: list[dict[str, object]]
    audio_dir: Path


class CuratorError(RuntimeError):
    pass


def _flatten_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(_flatten_values(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_flatten_values(item))
        return result
    return [str(value)]


def _literal_list(raw: str) -> object:
    raw = raw.strip()
    if not raw:
        return []
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []


def _score_tokens(tokens: Iterable[str]) -> tuple[int, dict[str, int], tuple[str, ...]]:
    token_set: set[str] = set()
    for token in tokens:
        token_set.update(_token_variants(token))

    bucket_scores: dict[str, int] = {bucket: 0 for bucket in _BUCKET_WEIGHTS}
    matches: set[str] = set()
    for bucket, keywords in _BUCKET_KEYWORDS.items():
        for keyword in keywords:
            for variant in _token_variants(keyword):
                if variant in token_set:
                    bucket_scores[bucket] += 3
                    matches.add(keyword)
                    break

    positive_score = 0
    for token in _POSITIVE_TOKENS:
        for variant in _token_variants(token):
            if variant in token_set:
                positive_score += 1
                matches.add(token)
                break

    negative_score = 0
    for token in _NEGATIVE_TOKENS:
        for variant in _token_variants(token):
            if variant in token_set:
                negative_score += 2
                matches.add(f"-{token}")
                break

    bucket_total = sum(bucket_scores.values())
    total_score = bucket_total + positive_score - negative_score
    return total_score, bucket_scores, tuple(sorted(matches))


def _primary_bucket(bucket_scores: dict[str, int]) -> str | None:
    best_bucket = None
    best_score = 0
    for bucket, score in bucket_scores.items():
        if score > best_score:
            best_bucket = bucket
            best_score = score
    return best_bucket


def _candidate_sort_key(candidate: MusicCandidate) -> tuple[int, int, int, str]:
    primary_score = candidate.bucket_scores.get(candidate.primary_bucket, 0)
    return (-candidate.score, -primary_score, _SOURCE_ORDER.get(candidate.source, 99), candidate.path)


def _read_fma_genres(genres_csv: Path) -> dict[int, str]:
    genres: dict[int, str] = {}
    with genres_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            genre_id_raw = row.get("genre_id") or row.get("") or row.get("id")
            title = row.get("title") or row.get("genre_title") or ""
            if not genre_id_raw or not title:
                continue
            try:
                genres[int(genre_id_raw)] = title
            except ValueError:
                continue
    return genres


def _read_fma_tracks(tracks_csv: Path) -> list[dict[tuple[str, str], str]]:
    with tracks_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header_top = next(reader)
        header_sub = next(reader)
        headers: list[tuple[str, str]] = []
        for idx, top in enumerate(header_top):
            sub = header_sub[idx] if idx < len(header_sub) else ""
            headers.append((top.strip(), sub.strip()))

        rows: list[dict[tuple[str, str], str]] = []
        for row in reader:
            if not row or not row[0].strip():
                continue
            record: dict[tuple[str, str], str] = {}
            for idx in range(1, len(headers)):
                record[headers[idx]] = row[idx] if idx < len(row) else ""
            record[("index", "track_id")] = row[0]
            rows.append(record)
    return rows


def _detect_fma_audio_dir(root: Path) -> Path | None:
    for name in ("fma_full", "fma_large", "fma_medium", "fma_small"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def load_fma_metadata(root: Path) -> LoadedFmaMetadata | None:
    metadata_dir = root / "fma_metadata"
    tracks_csv = metadata_dir / "tracks.csv"
    genres_csv = metadata_dir / "genres.csv"
    audio_dir = _detect_fma_audio_dir(root)
    if not tracks_csv.is_file() or not genres_csv.is_file() or audio_dir is None:
        return None
    return LoadedFmaMetadata(
        rows=_read_fma_tracks(tracks_csv), genres=_read_fma_genres(genres_csv), audio_dir=audio_dir
    )


def _fma_audio_path(audio_dir: Path, track_id: int) -> Path:
    tid = f"{track_id:06d}"
    return audio_dir / tid[:3] / f"{tid}.mp3"


def _extract_strings_from_fma_record(
    record: dict[tuple[str, str], str], genres: dict[int, str]
) -> tuple[list[str], str, str, str, float, str]:
    track_id = int(record[("index", "track_id")])
    title = record.get(("track", "title"), "")
    artist = record.get(("artist", "name"), "")
    license_name = record.get(("track", "license"), "")
    duration_raw = record.get(("track", "duration"), "0")
    try:
        duration_seconds = float(duration_raw)
    except ValueError:
        duration_seconds = 0.0

    tokens: list[str] = []
    for key in (("track", "genre_top"), ("track", "title"), ("album", "title"), ("artist", "name")):
        tokens.extend(_flatten_values(record.get(key, "")))

    for key in (("track", "tags"), ("album", "tags"), ("artist", "tags")):
        tokens.extend(_flatten_values(_literal_list(record.get(key, ""))))

    genre_ids = _literal_list(record.get(("track", "genres_all"), ""))
    if isinstance(genre_ids, list):
        for genre_id in genre_ids:
            try:
                genre_title = genres.get(int(genre_id))
            except (ValueError, TypeError):
                genre_title = None
            if genre_title:
                tokens.append(genre_title)

    return tokens, title, artist, license_name, duration_seconds, str(_fma_audio_path(Path("."), track_id))


def collect_fma_candidates(root: Path) -> list[MusicCandidate]:
    loaded = load_fma_metadata(root)
    if loaded is None:
        return []

    candidates: list[MusicCandidate] = []
    for record in loaded.rows:
        try:
            track_id = int(record[("index", "track_id")])
        except (KeyError, ValueError):
            continue
        audio_path = _fma_audio_path(loaded.audio_dir, track_id).resolve()
        if not audio_path.is_file():
            continue

        tokens, title, artist, license_name, duration_seconds, _ = _extract_strings_from_fma_record(
            record, loaded.genres
        )
        score, bucket_scores, matches = _score_tokens(tokens)
        primary_bucket = _primary_bucket(bucket_scores)
        if primary_bucket is None:
            continue
        if score < 4:
            continue

        candidates.append(
            MusicCandidate(
                source="fma",
                path=str(audio_path),
                title=title,
                artist=artist,
                license_name=license_name,
                duration_seconds=duration_seconds,
                primary_bucket=primary_bucket,
                score=score,
                bucket_scores=bucket_scores,
                matched_tokens=matches,
            )
        )
    return candidates


def _resolve_mtg_data_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
    except IndexError:
        return path
    if (
        first_line.endswith(".tsv")
        and "\t" not in first_line
        and len(path.read_text(encoding="utf-8").splitlines()) == 1
    ):
        candidate = (path.parent / first_line).resolve()
        if candidate.is_file():
            return candidate
    return path


def _detect_mtg_audio_dir(root: Path) -> Path | None:
    candidates = [
        root / "raw_30s" / "audio-low",
        root / "raw_30s" / "audio",
        root / "audio-low",
        root / "audio",
        root,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any((candidate / f"{prefix:02d}").exists() for prefix in range(100)):
            return candidate
    return None


def load_mtg_metadata(root: Path) -> LoadedMtgMetadata | None:
    data_dir = root / "data"
    tags_path = data_dir / "autotagging.tsv"
    raw_meta_path = data_dir / "raw.meta.tsv"
    if not tags_path.exists():
        tags_path = data_dir / "raw_30s_cleantags_50artists.tsv"
    if not tags_path.is_file() or not raw_meta_path.is_file():
        return None

    audio_dir = _detect_mtg_audio_dir(root)
    if audio_dir is None:
        return None

    tags_path = _resolve_mtg_data_file(tags_path)
    meta_by_track: dict[str, dict[str, str]] = {}
    with raw_meta_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            meta_by_track[row.get("TRACK_ID", "")] = row

    rows: list[dict[str, object]] = []
    with tags_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, [])
        for row in reader:
            if len(row) < 6:
                continue
            track_id, artist_id, album_id, rel_path, duration, *tags = row
            meta = meta_by_track.get(track_id, {})
            rows.append(
                {
                    "track_id": track_id,
                    "artist_id": artist_id,
                    "album_id": album_id,
                    "path": rel_path,
                    "duration": duration,
                    "tags": [tag.strip() for tag in tags if tag.strip()],
                    "title": meta.get("TRACK_NAME", ""),
                    "artist": meta.get("ARTIST_NAME", ""),
                    "license_name": "Jamendo / CC track-specific",
                }
            )
    return LoadedMtgMetadata(rows=rows, audio_dir=audio_dir)


def _mtg_audio_path(audio_dir: Path, relative_path: str) -> Path | None:
    base = Path(relative_path)
    candidates = [audio_dir / base]
    if base.suffix == ".mp3":
        candidates.insert(0, audio_dir / base.with_suffix(".low.mp3"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def collect_mtg_candidates(root: Path) -> list[MusicCandidate]:
    loaded = load_mtg_metadata(root)
    if loaded is None:
        return []

    candidates: list[MusicCandidate] = []
    for row in loaded.rows:
        relative_path = str(row["path"])
        audio_path = _mtg_audio_path(loaded.audio_dir, relative_path)
        if audio_path is None:
            continue

        tokens = []
        for tag in row.get("tags", []):
            if "---" in tag:
                _, normalized = tag.split("---", 1)
                tokens.append(normalized)
                tokens.append(tag)
            else:
                tokens.append(tag)
        tokens.extend([str(row.get("title", "")), str(row.get("artist", ""))])

        score, bucket_scores, matches = _score_tokens(tokens)
        primary_bucket = _primary_bucket(bucket_scores)
        if primary_bucket is None:
            continue
        if score < 4:
            continue

        try:
            duration_seconds = float(str(row.get("duration", "0") or 0))
        except ValueError:
            duration_seconds = 0.0

        candidates.append(
            MusicCandidate(
                source="mtg_jamendo",
                path=str(audio_path),
                title=str(row.get("title", "")),
                artist=str(row.get("artist", "")),
                license_name=str(row.get("license_name", "")),
                duration_seconds=duration_seconds,
                primary_bucket=primary_bucket,
                score=score + 1,  # slight preference: closer to modern commercial-pop tagging
                bucket_scores=bucket_scores,
                matched_tokens=matches,
            )
        )
    return candidates


def _round_robin_by_source(candidates: list[MusicCandidate]) -> list[MusicCandidate]:
    grouped: dict[str, list[MusicCandidate]] = defaultdict(list)
    for candidate in sorted(candidates, key=_candidate_sort_key):
        grouped[candidate.source].append(candidate)

    ordered_sources = sorted(grouped, key=lambda src: _SOURCE_ORDER.get(src, 99))
    ordered: list[MusicCandidate] = []
    while True:
        added = False
        for source in ordered_sources:
            if grouped[source]:
                ordered.append(grouped[source].pop(0))
                added = True
        if not added:
            break
    return ordered


def curate_candidates(candidates: list[MusicCandidate], *, target_count: int) -> CuratedLists:
    deduped: dict[str, MusicCandidate] = {}
    for candidate in sorted(candidates, key=_candidate_sort_key):
        deduped.setdefault(candidate.path, candidate)

    expanded = sorted(deduped.values(), key=_candidate_sort_key)
    by_bucket: dict[str, list[MusicCandidate]] = defaultdict(list)
    by_source: dict[str, list[MusicCandidate]] = defaultdict(list)
    for candidate in expanded:
        by_bucket[candidate.primary_bucket].append(candidate)
        by_source[candidate.source].append(candidate)

    selected: list[MusicCandidate] = []
    used_paths: set[str] = set()
    remaining_budget = min(target_count, len(expanded))

    quotas = {
        bucket: min(len(by_bucket[bucket]), int(math.floor(target_count * weight)))
        for bucket, weight in _BUCKET_WEIGHTS.items()
    }
    # distribute leftover quota to keep target_count reachable
    allocated = sum(quotas.values())
    if allocated < remaining_budget:
        bucket_order = sorted(_BUCKET_WEIGHTS, key=lambda bucket: (-len(by_bucket[bucket]), bucket))
        idx = 0
        while allocated < remaining_budget and bucket_order:
            bucket = bucket_order[idx % len(bucket_order)]
            if quotas[bucket] < len(by_bucket[bucket]):
                quotas[bucket] += 1
                allocated += 1
            idx += 1
            if idx > 10000:
                break

    for bucket, quota in quotas.items():
        if quota <= 0:
            continue
        for candidate in _round_robin_by_source(by_bucket[bucket]):
            if candidate.path in used_paths:
                continue
            selected.append(candidate)
            used_paths.add(candidate.path)
            if sum(1 for item in selected if item.primary_bucket == bucket) >= quota:
                break

    if len(selected) < remaining_budget:
        for candidate in expanded:
            if candidate.path in used_paths:
                continue
            selected.append(candidate)
            used_paths.add(candidate.path)
            if len(selected) >= remaining_budget:
                break

    return CuratedLists(curated=selected, expanded=expanded, by_source=dict(by_source))


def _write_list(paths: Iterable[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{path}\n")
    temp.replace(destination)


def _write_catalog(candidates: Iterable[MusicCandidate], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "source",
                "path",
                "bucket",
                "score",
                "duration_seconds",
                "title",
                "artist",
                "license",
                "matched_tokens",
            ]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.source,
                    candidate.path,
                    candidate.primary_bucket,
                    candidate.score,
                    f"{candidate.duration_seconds:.1f}",
                    candidate.title,
                    candidate.artist,
                    candidate.license_name,
                    ",".join(candidate.matched_tokens),
                ]
            )
    temp.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-dir", required=True, help="Directory where background-music lists will be written")
    parser.add_argument("--fma-dir", default=os.environ.get("FMA_DIR"), help="Root directory of extracted FMA data")
    parser.add_argument(
        "--mtg-jamendo-dir",
        default=os.environ.get("MTG_JAMENDO_DIR"),
        help="Root directory of extracted MTG-Jamendo data (metadata + audio)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=int(os.environ.get("BACKGROUND_MUSIC_TARGET_COUNT", "2000")),
        help="Target number of chart-style songs in background_music.txt",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=int(os.environ.get("BACKGROUND_MUSIC_MIN_COUNT", "500")),
        help="Minimum acceptable number of songs before failing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    list_dir = Path(args.list_dir).expanduser().resolve()
    list_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[MusicCandidate] = []

    fma_dir = Path(args.fma_dir).expanduser().resolve() if args.fma_dir else None
    if fma_dir and fma_dir.exists():
        fma_candidates = collect_fma_candidates(fma_dir)
        candidates.extend(fma_candidates)
        print(f"[info] FMA eligible chart-style tracks: {len(fma_candidates):,}")
    else:
        print("[info] FMA corpus not available; skipping", file=sys.stderr)

    mtg_dir = Path(args.mtg_jamendo_dir).expanduser().resolve() if args.mtg_jamendo_dir else None
    if mtg_dir and mtg_dir.exists():
        mtg_candidates = collect_mtg_candidates(mtg_dir)
        candidates.extend(mtg_candidates)
        print(f"[info] MTG-Jamendo eligible chart-style tracks: {len(mtg_candidates):,}")
    else:
        print("[info] MTG-Jamendo corpus not available; skipping", file=sys.stderr)

    curated = curate_candidates(candidates, target_count=args.target_count)
    curated_count = len(curated.curated)
    expanded_count = len(curated.expanded)

    if curated_count < args.min_count:
        raise CuratorError(
            f"Only {curated_count} eligible chart-style background-music tracks were found; need at least {args.min_count}."
        )

    curated_list = list_dir / "background_music.txt"
    expanded_list = list_dir / "background_music_expanded.txt"
    catalog_path = list_dir / "background_music_catalog.tsv"
    fma_list = list_dir / "background_music_fma.txt"
    mtg_list = list_dir / "background_music_mtg_jamendo.txt"

    _write_list((candidate.path for candidate in curated.curated), curated_list)
    _write_list((candidate.path for candidate in curated.expanded), expanded_list)
    _write_catalog(curated.expanded, catalog_path)
    _write_list((candidate.path for candidate in curated.by_source.get("fma", [])), fma_list)
    _write_list((candidate.path for candidate in curated.by_source.get("mtg_jamendo", [])), mtg_list)

    print(f"[ok] wrote {curated_count:,} curated background-music tracks -> {curated_list}")
    print(f"[ok] wrote {expanded_count:,} expanded background-music tracks -> {expanded_list}")
    print(f"[ok] wrote catalog -> {catalog_path}")
    if curated_count < args.target_count:
        print(
            f"[warn] only {curated_count:,} curated tracks available (target was {args.target_count:,}); "
            "using all eligible tracks.",
            file=sys.stderr,
        )

    bucket_counts: dict[str, int] = defaultdict(int)
    for candidate in curated.curated:
        bucket_counts[candidate.primary_bucket] += 1
    bucket_summary = ", ".join(f"{bucket}={bucket_counts.get(bucket, 0):,}" for bucket in _BUCKET_WEIGHTS)
    print(f"[info] curated bucket mix: {bucket_summary}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CuratorError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
