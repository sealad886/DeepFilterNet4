#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

PROGRESS_RE = re.compile(r"(?<!\d)(\d{1,3})%")
STREAM_COMPRESSED_TYPES = {
    "bzip2",
    "deflate",
    "gzip",
    "lz4",
    "lz5",
    "lzma",
    "lzma86",
    "lzo",
    "xz",
    "zstd",
}
TAR_STREAM_SUFFIXES = (
    ".tar.bz2",
    ".tar.gz",
    ".tar.lz4",
    ".tar.lzma",
    ".tar.xz",
    ".tar.zst",
    ".tbz",
    ".tbz2",
    ".tgz",
    ".txz",
    ".tzst",
)


@dataclass(frozen=True)
class ArchiveInfo:
    archive_type: str | None
    first_entry_path: str | None


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )


def find_extractor() -> str:
    for candidate in ("7zz", "7z"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("Neither 7zz nor 7z is available in PATH.")


def inspect_archive(extractor: str, archive: str) -> ArchiveInfo:
    result = run_cmd([extractor, "l", "-slt", archive])
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to inspect archive.\n"
            f"Command: {extractor} l -slt {archive}\n"
            f"Exit code: {result.returncode}\n"
            f"stderr:\n{result.stderr}"
        )

    archive_type: str | None = None
    first_entry_path: str | None = None
    in_entries = False
    for line in result.stdout.splitlines():
        if line.startswith("----------"):
            in_entries = True
            continue
        if not in_entries and archive_type is None and line.startswith("Type = "):
            archive_type = line.split("=", 1)[1].strip().lower()
            continue
        if in_entries and first_entry_path is None and line.startswith("Path = "):
            first_entry_path = line.split("=", 1)[1].strip()
            break

    return ArchiveInfo(archive_type=archive_type, first_entry_path=first_entry_path)


def needs_tar_stream_pipeline(archive: str, info: ArchiveInfo) -> bool:
    lower_archive = archive.lower()
    if lower_archive.endswith(TAR_STREAM_SUFFIXES):
        return True
    return bool(
        info.archive_type in STREAM_COMPRESSED_TYPES
        and info.first_entry_path
        and info.first_entry_path.lower().endswith(".tar")
    )


def normalize_terminal_output(raw: bytes) -> str:
    lines: list[str] = []
    current: list[str] = []

    for ch in raw.decode("utf-8", errors="replace"):
        if ch == "\b":
            if current:
                current.pop()
            continue
        if ch == "\r":
            current.clear()
            continue
        if ch == "\n":
            line = "".join(current).strip()
            if line:
                lines.append(line)
            current.clear()
            continue
        current.append(ch)

    trailing = "".join(current).strip()
    if trailing:
        lines.append(trailing)

    cleaned: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if PROGRESS_RE.fullmatch(line):
            continue
        if line not in seen:
            cleaned.append(line)
            seen.add(line)
    return "\n".join(cleaned)


def remove_existing_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def move_with_strip_components(staging_dir: Path, outdir: Path, strip_components: int) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    moved = 0

    all_paths = sorted(
        staging_dir.rglob("*"), key=lambda path: (len(path.relative_to(staging_dir).parts), path.as_posix())
    )
    for src_dir in [path for path in all_paths if path.is_dir()]:
        rel_parts = src_dir.relative_to(staging_dir).parts
        if len(rel_parts) < strip_components:
            continue
        stripped_parts = rel_parts[strip_components:]
        if stripped_parts:
            outdir.joinpath(*stripped_parts).mkdir(parents=True, exist_ok=True)

    for src_path in [path for path in all_paths if not path.is_dir()]:
        rel_parts = src_path.relative_to(staging_dir).parts
        if len(rel_parts) <= strip_components:
            continue
        dest = outdir.joinpath(*rel_parts[strip_components:])
        dest.parent.mkdir(parents=True, exist_ok=True)
        remove_existing_path(dest)
        shutil.move(str(src_path), str(dest))
        moved += 1

    return moved


def create_extract_target(outdir: Path, strip_components: int) -> tuple[Path, Path | None]:
    if strip_components <= 0:
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir, None

    staging_parent = outdir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{outdir.name or 'extract'}-stage-"
    staging_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=staging_parent))
    return staging_dir, staging_dir


def build_direct_extract_cmd(
    extractor: str,
    archive: str,
    outdir: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [extractor, "x", archive]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(
        [
            f"-o{outdir}",
            "-y",
            "-bb0",
            "-bso0",
            "-bse2",
            "-bsp2",
        ]
    )
    return cmd


def build_tar_stream_pipeline_cmds(
    extractor: str,
    archive: str,
    outdir: str,
    extra_args: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    outer_cmd = [
        extractor,
        "x",
        archive,
        "-so",
        "-bb0",
        "-bso0",
        "-bse2",
        "-bsp2",
    ]
    inner_cmd = [extractor, "x", "-si", "-ttar"]
    if extra_args:
        inner_cmd.extend(extra_args)
    inner_cmd.extend(
        [
            f"-o{outdir}",
            "-y",
            "-bb0",
            "-bso0",
            "-bse2",
            "-bsp0",
        ]
    )
    return outer_cmd, inner_cmd


def monitor_processes(
    processes: dict[str, subprocess.Popen[bytes]],
    *,
    progress_source: str,
    desc: str,
) -> dict[str, bytes]:
    selector = selectors.DefaultSelector()
    captured: dict[str, bytearray] = {}
    progress_tail = ""
    current_percent = 0

    for name, proc in processes.items():
        if proc.stderr is None:
            continue
        selector.register(proc.stderr, selectors.EVENT_READ, data=name)
        captured[name] = bytearray()

    try:
        with tqdm(
            total=100,
            desc=desc,
            unit="%",
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
        ) as pbar:
            while selector.get_map():
                events = selector.select(timeout=0.1)
                if not events:
                    if all(proc.poll() is not None for proc in processes.values()):
                        break
                    continue

                for key, _ in events:
                    stream = key.fileobj
                    chunk = stream.read1(65536)
                    name = key.data
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        continue

                    captured[name].extend(chunk)
                    if name != progress_source:
                        continue

                    progress_tail = (progress_tail + chunk.decode("utf-8", errors="replace"))[-2048:]
                    matches = PROGRESS_RE.findall(progress_tail)
                    if not matches:
                        continue

                    latest_percent = max(0, min(100, int(matches[-1])))
                    if latest_percent > current_percent:
                        pbar.update(latest_percent - current_percent)
                        current_percent = latest_percent

            if current_percent < 100 and all(proc.wait() == 0 for proc in processes.values()):
                pbar.update(100 - current_percent)
    except KeyboardInterrupt:
        for proc in processes.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in processes.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    finally:
        selector.close()

    for proc in processes.values():
        proc.wait()

    return {name: bytes(buffer) for name, buffer in captured.items()}


def raise_process_error(label: str, cmd: list[str], rc: int, stderr: bytes) -> None:
    detail = normalize_terminal_output(stderr)
    message = f"{label} failed.\n" f"Command: {' '.join(cmd)}\n" f"Exit code: {rc}"
    if detail:
        message += f"\nstderr:\n{detail}"
    raise RuntimeError(message)


def extract_direct(
    extractor: str,
    archive: str,
    outdir: str,
    extra_args: list[str] | None = None,
) -> None:
    cmd = build_direct_extract_cmd(extractor, archive, outdir, extra_args)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    captured = monitor_processes({"extract": proc}, progress_source="extract", desc="Extracting")
    if proc.returncode != 0:
        raise_process_error("Extraction", cmd, proc.returncode, captured.get("extract", b""))


def extract_via_tar_stream_pipeline(
    extractor: str,
    archive: str,
    outdir: str,
    extra_args: list[str] | None = None,
) -> None:
    outer_cmd, inner_cmd = build_tar_stream_pipeline_cmds(extractor, archive, outdir, extra_args)

    inner = subprocess.Popen(
        inner_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert inner.stdin is not None

    try:
        outer = subprocess.Popen(
            outer_cmd,
            stdin=subprocess.DEVNULL,
            stdout=inner.stdin,
            stderr=subprocess.PIPE,
        )
    finally:
        inner.stdin.close()

    captured = monitor_processes(
        {"outer": outer, "inner": inner},
        progress_source="outer",
        desc="Extracting",
    )

    if outer.returncode != 0:
        raise_process_error("Decompression", outer_cmd, outer.returncode, captured.get("outer", b""))
    if inner.returncode != 0:
        raise_process_error("Tar extraction", inner_cmd, inner.returncode, captured.get("inner", b""))


def extract_with_progress(
    archive: str,
    outdir: str,
    *,
    strip_components: int = 0,
    extra_args: list[str] | None = None,
) -> int:
    extractor = find_extractor()
    archive_info = inspect_archive(extractor, archive)

    outdir_path = Path(outdir)
    target_dir, staging_dir = create_extract_target(outdir_path, strip_components)

    try:
        if needs_tar_stream_pipeline(archive, archive_info):
            extract_via_tar_stream_pipeline(extractor, archive, str(target_dir), extra_args)
        else:
            extract_direct(extractor, archive, str(target_dir), extra_args)

        if strip_components > 0:
            moved = move_with_strip_components(target_dir, outdir_path, strip_components)
            shutil.rmtree(target_dir, ignore_errors=True)
            return moved
        return 0
    except Exception:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract an archive with 7zz/7z and show bounded real-time progress.",
        allow_abbrev=False,
    )
    parser.add_argument("archive", help="Path to the archive file")
    parser.add_argument(
        "-C",
        "--directory",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--strip-components",
        type=int,
        default=0,
        help="Strip the specified number of leading path components after extraction.",
    )
    return parser


def parse_cli_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = build_parser()
    return parser.parse_known_intermixed_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, unknown_args = parse_cli_args(argv)

    archive = os.path.abspath(os.path.expanduser(args.archive))
    outdir = os.path.abspath(os.path.expanduser(args.directory))

    if args.strip_components < 0:
        print("error: --strip-components must be >= 0", file=sys.stderr)
        return 2

    if not os.path.isfile(archive):
        print(f"error: archive not found: {archive}", file=sys.stderr)
        return 2

    try:
        extracted = extract_with_progress(
            archive,
            outdir,
            strip_components=args.strip_components,
            extra_args=unknown_args,
        )
    except FileNotFoundError:
        print("error: neither 7zz nor 7z was found in PATH", file=sys.stderr)
        return 127
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    if args.strip_components > 0:
        print(f"Done. Relocated {extracted} entries to: {outdir}")
    else:
        print(f"Done. Extracted archive to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
