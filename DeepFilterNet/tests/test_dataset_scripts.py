from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import threading
import wave
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "datasets" / "build_mlx_datastore.sh"
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "datasets" / "download_datasets.sh"
CHAINS_PREPARE_SCRIPT = REPO_ROOT / "scripts" / "datasets" / "prepare_chains_speech.py"


def _write_wav(path: Path, *, sample_rate: int, seconds: float, frequency_hz: float = 440.0) -> None:
    frames = int(sample_rate * seconds)
    amplitude = 12000
    samples = bytearray()
    for i in range(frames):
        value = int(amplitude * math.sin((2.0 * math.pi * frequency_hz * i) / sample_rate))
        samples.extend(int(value).to_bytes(2, byteorder="little", signed=True))

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(samples))


def _write_stereo_wav(
    path: Path,
    *,
    sample_rate: int,
    seconds: float,
    left_frequency_hz: float = 220.0,
    right_frequency_hz: float = 440.0,
) -> None:
    frames = int(sample_rate * seconds)
    amplitude = 12000
    samples = bytearray()
    for i in range(frames):
        left = int(amplitude * math.sin((2.0 * math.pi * left_frequency_hz * i) / sample_rate))
        right = int(amplitude * math.sin((2.0 * math.pi * right_frequency_hz * i) / sample_rate))
        samples.extend(int(left).to_bytes(2, byteorder="little", signed=True))
        samples.extend(int(right).to_bytes(2, byteorder="little", signed=True))

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(samples))


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _build_fake_chains_corpus(root: Path, *, sample_rate: int) -> Path:
    mono_specs = {
        "fast": ("frf01", "frf01_f01_fast.wav"),
        "retell": ("frf01", "frf01_f01_retell.wav"),
        "solo": ("frf01", "frf01_f01_solo.wav"),
        "sync": ("frf01", "frf01_f01_sync_frf02.wav"),
        "whsp": ("frf01", "frf01_f01_whsp.wav"),
    }
    for style, (speaker, filename) in mono_specs.items():
        _write_wav(
            root / style / "data" / style / speaker / filename,
            sample_rate=sample_rate,
            seconds=1.0,
            frequency_hz=330.0,
        )

    _write_stereo_wav(
        root / "rsi" / "data" / "rsi" / "frf01" / "frf01_f01_fs01_rsi_irf05.wav",
        sample_rate=sample_rate,
        seconds=1.0,
    )
    return root


def _write_fake_python_bin(path: Path) -> None:
    script = f"""#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

REAL_PYTHON = {sys.executable!r}
LOG_PATH = Path(os.environ["FAKE_PY_LOG"])
args = sys.argv[1:]
with LOG_PATH.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"args": args}}) + "\\n")

def arg_value(flag: str) -> str:
    return args[args.index(flag) + 1]

if args and args[0].endswith("prepare_chains_speech.py"):
    raise SystemExit(subprocess.call([REAL_PYTHON, *args]))

if args and args[0].endswith("preprocess_clean_speech.py"):
    file_list = Path(arg_value("--file-list"))
    output_root = Path(arg_value("--output-root"))
    base_dir = Path(arg_value("--base-dir")).resolve()
    output_list = Path(arg_value("--output-list"))
    outputs: list[str] = []
    for raw_line in file_list.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        source = Path(line).expanduser().resolve()
        try:
            relative = source.relative_to(base_dir)
        except ValueError:
            relative = Path("_external") / source.name
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake")
        outputs.append(str(target))
    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text("".join(f"{{path}}\\n" for path in outputs), encoding="utf-8")
    raise SystemExit(0)

if len(args) >= 2 and args[0] == "-m" and args[1] == "df_mlx.build_audio_cache":
    output_dir = Path(arg_value("--output-dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text("{{}}\\n", encoding="utf-8")
    index = {{"speech": {{}}, "noise": {{}}, "rir": {{}}}}
    if "--music-list" in args:
        index["music"] = {{}}
    (output_dir / "index.json").write_text(json.dumps(index) + "\\n", encoding="utf-8")
    raise SystemExit(0)

raise SystemExit(subprocess.call([REAL_PYTHON, *args]))
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def test_build_mlx_datastore_help_mentions_preprocess_and_merge_short() -> None:
    result = subprocess.run(
        ["bash", str(BUILD_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--preprocess-clean-speech" in result.stdout
    assert "--preprocess-probe-workers" in result.stdout
    assert "--preprocess-probe-cache" in result.stdout
    assert "DeepFilterNet3-MLX" in result.stdout
    assert "--merge-short" in result.stdout
    assert "--include-chains" in result.stdout
    assert "--chains-dir PATH" in result.stdout
    assert "--music-list PATH" in result.stdout
    assert "Examples:" in result.stdout


def test_prepare_chains_speech_extracts_rsi_subject_and_reuses_outputs(tmp_path: Path) -> None:
    sample_rate = 16_000
    chains_dir = _build_fake_chains_corpus(tmp_path / "CHAINS", sample_rate=sample_rate)
    prepared_root = tmp_path / "prepared"
    output_list = tmp_path / "lists" / "chains_clean.txt"

    first = subprocess.run(
        [
            sys.executable,
            str(CHAINS_PREPARE_SCRIPT),
            "--chains-dir",
            str(chains_dir),
            "--prepared-root",
            str(prepared_root),
            "--output-list",
            str(output_list),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    entries = [line.strip() for line in output_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 6

    raw_rsi = (chains_dir / "rsi" / "data" / "rsi" / "frf01" / "frf01_f01_fs01_rsi_irf05.wav").resolve()
    prepared_rsi = (prepared_root / "rsi_subject" / "frf01" / "frf01_f01_fs01_rsi_irf05.wav").resolve()

    assert str(raw_rsi) not in entries
    assert str(prepared_rsi) in entries
    assert prepared_rsi.exists()
    with wave.open(str(prepared_rsi), "rb") as handle:
        assert handle.getnchannels() == 1

    first_mtime = prepared_rsi.stat().st_mtime_ns
    second = subprocess.run(
        [
            sys.executable,
            str(CHAINS_PREPARE_SCRIPT),
            "--chains-dir",
            str(chains_dir),
            "--prepared-root",
            str(prepared_root),
            "--output-list",
            str(output_list),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert second.returncode == 0, second.stderr
    assert prepared_rsi.stat().st_mtime_ns == first_mtime


def test_build_mlx_datastore_include_chains_builds_combined_cache(tmp_path: Path) -> None:
    sample_rate = 16_000
    data_dir = tmp_path / "data"
    lists_dir = data_dir / "lists"
    cache_dir = tmp_path / "cache"
    chains_dir = _build_fake_chains_corpus(tmp_path / "CHAINS", sample_rate=sample_rate)

    speech_file = tmp_path / "speech" / "speech.wav"
    noise_file = tmp_path / "noise" / "noise.wav"
    rir_file = tmp_path / "rir" / "rir.wav"
    _write_wav(speech_file, sample_rate=sample_rate, seconds=1.2)
    _write_wav(noise_file, sample_rate=sample_rate, seconds=0.6, frequency_hz=220.0)
    _write_wav(rir_file, sample_rate=sample_rate, seconds=0.1, frequency_hz=110.0)

    lists_dir.mkdir(parents=True, exist_ok=True)
    clean_list = lists_dir / "clean_all.txt"
    noise_list = lists_dir / "noise_music.txt"
    music_list = lists_dir / "background_music.txt"
    rir_list = lists_dir / "rir_all.txt"
    clean_list.write_text(f"{speech_file}\n", encoding="utf-8")
    noise_list.write_text(f"{noise_file}\n", encoding="utf-8")
    music_list.write_text(f"{noise_file}\n", encoding="utf-8")
    rir_list.write_text(f"{rir_file}\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(BUILD_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--list-dir",
            str(lists_dir),
            "--output-dir",
            str(cache_dir),
            "--clean-list",
            str(clean_list),
            "--noise-list",
            str(noise_list),
            "--rir-list",
            str(rir_list),
            "--profile",
            "prototype",
            "--include-chains",
            "--chains-dir",
            str(chains_dir),
            "--sample-rate",
            str(sample_rate),
            "--segment-length",
            "1.0",
            "--min-duration",
            "0",
            "--num-workers",
            "1",
            "--shard-size",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    combined_list = lists_dir / "clean_all.with_chains.txt"
    combined_entries = [line.strip() for line in combined_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(combined_entries) == 7

    index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    assert len(index["speech"]) == 7
    assert any("prepared/chains_speech/rsi_subject" in path for path in index["speech"])
    assert (
        str((chains_dir / "rsi" / "data" / "rsi" / "frf01" / "frf01_f01_fs01_rsi_irf05.wav").resolve())
        not in index["speech"]
    )


def test_build_mlx_datastore_include_chains_preprocess_uses_combined_list_and_common_base(tmp_path: Path) -> None:
    sample_rate = 16_000
    data_dir = tmp_path / "data"
    lists_dir = data_dir / "lists"
    cache_dir = tmp_path / "cache"
    chains_dir = _build_fake_chains_corpus(tmp_path / "CHAINS", sample_rate=sample_rate)

    speech_file = tmp_path / "speech" / "speech.wav"
    noise_file = tmp_path / "noise" / "noise.wav"
    rir_file = tmp_path / "rir" / "rir.wav"
    _write_wav(speech_file, sample_rate=sample_rate, seconds=1.2)
    _write_wav(noise_file, sample_rate=sample_rate, seconds=0.6, frequency_hz=220.0)
    _write_wav(rir_file, sample_rate=sample_rate, seconds=0.1, frequency_hz=110.0)

    lists_dir.mkdir(parents=True, exist_ok=True)
    clean_list = lists_dir / "clean_all.txt"
    noise_list = lists_dir / "noise_music.txt"
    music_list = lists_dir / "background_music.txt"
    rir_list = lists_dir / "rir_all.txt"
    clean_list.write_text(f"{speech_file}\n", encoding="utf-8")
    noise_list.write_text(f"{noise_file}\n", encoding="utf-8")
    music_list.write_text(f"{noise_file}\n", encoding="utf-8")
    rir_list.write_text(f"{rir_file}\n", encoding="utf-8")

    fake_python = tmp_path / "fake_python.py"
    fake_log = tmp_path / "fake_python_calls.jsonl"
    _write_fake_python_bin(fake_python)

    env = os.environ.copy()
    env["PYTHON_BIN"] = str(fake_python)
    env["FAKE_PY_LOG"] = str(fake_log)

    result = subprocess.run(
        [
            "bash",
            str(BUILD_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--list-dir",
            str(lists_dir),
            "--output-dir",
            str(cache_dir),
            "--clean-list",
            str(clean_list),
            "--noise-list",
            str(noise_list),
            "--rir-list",
            str(rir_list),
            "--profile",
            "prototype",
            "--include-chains",
            "--chains-dir",
            str(chains_dir),
            "--preprocess-clean-speech",
            "--sample-rate",
            str(sample_rate),
            "--segment-length",
            "1.0",
            "--min-duration",
            "0",
            "--num-workers",
            "1",
            "--shard-size",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in fake_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    preprocess_call = next(
        call for call in calls if call["args"] and call["args"][0].endswith("preprocess_clean_speech.py")
    )
    build_call = next(call for call in calls if call["args"][:2] == ["-m", "df_mlx.build_audio_cache"])

    preprocess_args = preprocess_call["args"]
    assert preprocess_args[preprocess_args.index("--file-list") + 1] == str(lists_dir / "clean_all.with_chains.txt")
    assert Path(preprocess_args[preprocess_args.index("--base-dir") + 1]) == tmp_path.resolve()

    build_args = build_call["args"]
    assert build_args[build_args.index("--speech-list") + 1] == str(lists_dir / "clean_all.preprocessed.txt")
    assert build_args[build_args.index("--music-list") + 1] == str(music_list)


def test_build_mlx_datastore_smoke_prints_cache_dir_override(tmp_path: Path) -> None:
    sample_rate = 16_000
    data_dir = tmp_path / "data"
    lists_dir = data_dir / "lists"
    cache_dir = tmp_path / "cache"

    speech_file = tmp_path / "speech" / "speech.wav"
    noise_file = tmp_path / "noise" / "noise.wav"
    rir_file = tmp_path / "rir" / "rir.wav"

    _write_wav(speech_file, sample_rate=sample_rate, seconds=1.2)
    _write_wav(noise_file, sample_rate=sample_rate, seconds=0.6, frequency_hz=220.0)
    _write_wav(rir_file, sample_rate=sample_rate, seconds=0.1, frequency_hz=110.0)

    lists_dir.mkdir(parents=True, exist_ok=True)
    clean_list = lists_dir / "clean_all.txt"
    noise_list = lists_dir / "noise_music.txt"
    rir_list = lists_dir / "rir_all.txt"
    clean_list.write_text(f"{speech_file}\n", encoding="utf-8")
    noise_list.write_text(f"{noise_file}\n", encoding="utf-8")
    rir_list.write_text(f"{rir_file}\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(BUILD_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--list-dir",
            str(lists_dir),
            "--output-dir",
            str(cache_dir),
            "--clean-list",
            str(clean_list),
            "--noise-list",
            str(noise_list),
            "--rir-list",
            str(rir_list),
            "--profile",
            "prototype",
            "--sample-rate",
            str(sample_rate),
            "--segment-length",
            "1.0",
            "--min-duration",
            "0",
            "--num-workers",
            "1",
            "--shard-size",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (cache_dir / "config.json").exists(), result.stdout
    assert (cache_dir / "index.json").exists(), result.stdout
    assert "df_mlx.validate_audio_cache" in result.stdout
    assert "--cache-dir" in result.stdout
    assert str(cache_dir) in result.stdout


def test_download_datasets_help_mentions_defaults_and_cli_env_flags() -> None:
    result = subprocess.run(
        ["bash", str(DOWNLOAD_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    expected_data_dir = "/Volumes/TrainingData/datasets"
    if not Path(expected_data_dir).exists():
        expected_data_dir = str(REPO_ROOT / "data")

    assert result.returncode == 0, result.stderr
    assert f"default: {expected_data_dir}" in result.stdout
    assert "default: prototype" in result.stdout
    assert "--agree-licenses" in result.stdout
    assert "--keep-archives" in result.stdout
    assert "--verify-cache-file PATH" in result.stdout
    assert "--download-vctk / --no-download-vctk" in result.stdout
    assert "--vctk-dir PATH" in result.stdout
    assert "--librispeech-parts STRING" in result.stdout
    assert "default: 16" in result.stdout
    assert "default: 8" in result.stdout
    assert "default: none" in result.stdout
    assert "Zenodo mirror of VCTK 0.92 zip" in result.stdout


def test_download_datasets_uses_zip_merge_progress_helper() -> None:
    script_text = DOWNLOAD_SCRIPT.read_text(encoding="utf-8")
    assert "zip_merge_progress.py" in script_text
    assert '--download-dir "${DOWNLOAD_DIR}"' in script_text
    assert '--zip-base "${zip_base}"' in script_text


def test_download_datasets_no_download_accepts_cli_overrides(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    lists_dir = tmp_path / "lists"
    downloads_dir = tmp_path / "downloads"
    extract_dir = tmp_path / "raw"

    vctk_dir = tmp_path / "existing" / "VCTK-Corpus-0.92"
    musan_dir = tmp_path / "existing" / "musan"
    air_rir_dir = tmp_path / "existing" / "air"

    _touch(vctk_dir / "wav48_silence_trimmed" / "p001" / "sample.flac")
    _touch(musan_dir / "noise" / "noise.wav")
    _touch(musan_dir / "music" / "music.wav")
    _touch(air_rir_dir / "room.wav")

    result = subprocess.run(
        [
            "bash",
            str(DOWNLOAD_SCRIPT),
            "--no-download",
            "--data-dir",
            str(data_dir),
            "--list-dir",
            str(lists_dir),
            "--download-dir",
            str(downloads_dir),
            "--extract-dir",
            str(extract_dir),
            "--profile",
            "production",
            "--agree-licenses",
            "--keep-archives",
            "--no-resume",
            "--no-aria2",
            "--no-aria2-parallel",
            "--aria2-conn",
            "4",
            "--aria2-split",
            "4",
            "--aria2-min-split",
            "2M",
            "--aria2-max-concurrent",
            "2",
            "--aria2-file-alloc",
            "none",
            "--aria2-user-agent",
            "TestAgent/1.0",
            "--zenodo-referer",
            "https://example.com/zenodo",
            "--no-verify-cache",
            "--verify-cache-file",
            str(tmp_path / "verify.tsv"),
            "--no-gh-auth",
            "--no-audb",
            "--install-audb",
            "--audb-dir",
            str(tmp_path / "audb"),
            "--download-vctk",
            "--no-download-librispeech",
            "--download-musan",
            "--no-download-fsd50k",
            "--download-air",
            "--no-download-openair",
            "--no-download-acousticrooms",
            "--vctk-dir",
            str(vctk_dir),
            "--librispeech-dir",
            str(tmp_path / "missing-librispeech"),
            "--musan-dir",
            str(musan_dir),
            "--fsd50k-dir",
            str(tmp_path / "missing-fsd50k"),
            "--air-rir-dir",
            str(air_rir_dir),
            "--openair-dir",
            str(tmp_path / "missing-openair"),
            "--acousticrooms-dir",
            str(tmp_path / "missing-acousticrooms"),
            "--vctk-url",
            "https://example.com/vctk.zip",
            "--librispeech-parts",
            "dev-clean test-clean",
            "--fsd50k-base-url",
            "https://example.com/fsd50k",
            "--air-version",
            "9.9.9",
            "--openair-version",
            "8.8.8",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"[config] profile=production download=0 data_dir={data_dir}" in result.stdout
    assert f"[config] download_dir={downloads_dir} extract_dir={extract_dir} list_dir={lists_dir}" in result.stdout
    assert (lists_dir / "clean_all.txt").exists(), result.stdout
    assert (lists_dir / "noise_all.txt").exists(), result.stdout
    assert (lists_dir / "background_music.txt").exists(), result.stdout
    assert (lists_dir / "noise_music.txt").exists(), result.stdout
    assert (lists_dir / "rir_all.txt").exists(), result.stdout
    assert str(vctk_dir / "wav48_silence_trimmed" / "p001" / "sample.flac") in (lists_dir / "clean_all.txt").read_text()
    assert str(musan_dir / "noise" / "noise.wav") in (lists_dir / "noise_all.txt").read_text()
    assert str(musan_dir / "music" / "music.wav") not in (lists_dir / "noise_all.txt").read_text()
    assert str(musan_dir / "music" / "music.wav") in (lists_dir / "background_music.txt").read_text()
    combined_noise = (lists_dir / "noise_music.txt").read_text()
    assert str(musan_dir / "noise" / "noise.wav") in combined_noise
    assert str(musan_dir / "music" / "music.wav") in combined_noise
    assert str(air_rir_dir / "room.wav") in (lists_dir / "rir_all.txt").read_text()


def test_download_datasets_skips_completed_processing_for_existing_archive_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    lists_dir = data_dir / "lists"
    downloads_dir = data_dir / "downloads"
    extract_dir = data_dir / "raw"

    vctk_extract_dir = extract_dir / "VCTK-Corpus-0.92"
    musan_dir = tmp_path / "existing" / "musan"
    air_rir_dir = tmp_path / "existing" / "air"

    _touch(vctk_extract_dir / "wav48_silence_trimmed" / "p001" / "sample.flac")
    _touch(vctk_extract_dir / "speaker-info.txt")
    _touch(musan_dir / "noise" / "noise.wav")
    _touch(musan_dir / "music" / "music.wav")
    _touch(air_rir_dir / "room.wav")

    downloads_dir.mkdir(parents=True, exist_ok=True)
    (downloads_dir / "VCTK-Corpus-0.92.zip").write_bytes(b"placeholder archive")

    result = subprocess.run(
        [
            "bash",
            str(DOWNLOAD_SCRIPT),
            "--no-download",
            "--data-dir",
            str(data_dir),
            "--list-dir",
            str(lists_dir),
            "--download-dir",
            str(downloads_dir),
            "--extract-dir",
            str(extract_dir),
            "--vctk-dir",
            str(vctk_extract_dir),
            "--musan-dir",
            str(musan_dir),
            "--air-rir-dir",
            str(air_rir_dir),
            "--no-download-librispeech",
            "--no-download-fsd50k",
            "--no-download-openair",
            "--no-download-acousticrooms",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "unzip:" not in result.stderr.lower()
    assert (lists_dir / "clean_all.txt").exists(), result.stdout
    assert (
        str(vctk_extract_dir / "wav48_silence_trimmed" / "p001" / "sample.flac")
        in (lists_dir / "clean_all.txt").read_text()
    )


def test_download_datasets_zenodo_range_download_bypasses_aria2_and_extracts_vctk(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    lists_dir = data_dir / "lists"
    downloads_dir = data_dir / "downloads"
    extract_dir = data_dir / "raw"
    musan_dir = tmp_path / "existing" / "musan"
    air_rir_dir = tmp_path / "existing" / "air"
    fake_bin_dir = tmp_path / "bin"
    fake_aria2 = fake_bin_dir / "aria2c"
    fake_aria2_log = tmp_path / "fake-aria2.log"

    _touch(musan_dir / "noise" / "noise.wav")
    _touch(musan_dir / "music" / "music.wav")
    _touch(air_rir_dir / "room.wav")
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_aria2.write_text('#!/bin/sh\necho invoked >> "$FAKE_ARIA2_LOG"\nexit 99\n', encoding="utf-8")
    fake_aria2.chmod(0o755)

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("wav48_silence_trimmed/p001/sample.flac", b"fake-flac")
        archive.writestr("speaker-info.txt", "speaker\n")
    archive_bytes = archive_buffer.getvalue()
    range_headers: list[str] = []

    class ZenodoLikeRangeHandler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler naming
            self.send_response(200)
            self.send_header("Content-Length", str(len(archive_bytes)))
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming
            range_header = self.headers.get("Range")
            if range_header:
                range_headers.append(range_header)
            if range_header == "bytes=0-0":
                self.send_response(206)
                self.send_header("Content-Length", "1")
                self.send_header("Content-Range", f"bytes 0-0/{len(archive_bytes)}")
                self.end_headers()
                self.wfile.write(archive_bytes[:1])
                return
            if range_header and range_header.startswith("bytes="):
                start_s, end_s = range_header.removeprefix("bytes=").split("-", 1)
                start = int(start_s)
                end = int(end_s)
                chunk = archive_bytes[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(archive_bytes)}")
                self.end_headers()
                self.wfile.write(chunk)
                return

            self.send_response(200)
            self.send_header("Content-Length", str(len(archive_bytes)))
            self.end_headers()
            self.wfile.write(archive_bytes)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), ZenodoLikeRangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        zenodo_like_url = f"http://127.0.0.1:{server.server_port}/zenodo.org/VCTK-Corpus-0.92.zip?download=1"
        env = os.environ.copy()
        env["PATH"] = f"{fake_bin_dir}{os.pathsep}{env['PATH']}"
        env["FAKE_ARIA2_LOG"] = str(fake_aria2_log)

        result = subprocess.run(
            [
                "bash",
                str(DOWNLOAD_SCRIPT),
                "--data-dir",
                str(data_dir),
                "--list-dir",
                str(lists_dir),
                "--download-dir",
                str(downloads_dir),
                "--extract-dir",
                str(extract_dir),
                "--profile",
                "production",
                "--aria2-conn",
                "4",
                "--aria2-split",
                "4",
                "--aria2-max-concurrent",
                "1",
                "--vctk-url",
                str(zenodo_like_url),
                "--musan-dir",
                str(musan_dir),
                "--air-rir-dir",
                str(air_rir_dir),
                "--no-download-librispeech",
                "--no-download-musan",
                "--no-download-fsd50k",
                "--no-download-air",
                "--no-download-openair",
                "--no-download-acousticrooms",
                "--no-keep-archives",
                "--zenodo-referer",
                "https://example.com/zenodo-test",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode == 0, result.stderr
    assert not fake_aria2_log.exists(), "Zenodo VCTK path should bypass aria2 and use curl range downloads"
    assert "bytes=0-0" in range_headers
    assert any(header != "bytes=0-0" for header in range_headers)
    assert not (downloads_dir / "VCTK-Corpus-0.92.zip").exists()
    clean_entries = (lists_dir / "clean_all.txt").read_text(encoding="utf-8")
    assert str(extract_dir / "VCTK-Corpus-0.92" / "wav48_silence_trimmed" / "p001" / "sample.flac") in clean_entries


def test_download_datasets_zenodo_parallel_range_failure_propagates(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    lists_dir = data_dir / "lists"
    downloads_dir = data_dir / "downloads"
    extract_dir = data_dir / "raw"
    musan_dir = tmp_path / "existing" / "musan"
    air_rir_dir = tmp_path / "existing" / "air"
    fake_bin_dir = tmp_path / "bin"
    fake_aria2 = fake_bin_dir / "aria2c"
    fake_aria2_log = tmp_path / "fake-aria2.log"

    _touch(musan_dir / "noise" / "noise.wav")
    _touch(musan_dir / "music" / "music.wav")
    _touch(air_rir_dir / "room.wav")
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_aria2.write_text('#!/bin/sh\necho invoked >> "$FAKE_ARIA2_LOG"\nexit 99\n', encoding="utf-8")
    fake_aria2.chmod(0o755)

    archive_bytes = b"not-a-real-zip-but-long-enough"
    range_headers: list[str] = []

    class FailingZenodoRangeHandler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler naming
            self.send_response(200)
            self.send_header("Content-Length", str(len(archive_bytes)))
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming
            range_header = self.headers.get("Range")
            if range_header:
                range_headers.append(range_header)
            if range_header == "bytes=0-0":
                self.send_response(206)
                self.send_header("Content-Length", "1")
                self.send_header("Content-Range", f"bytes 0-0/{len(archive_bytes)}")
                self.end_headers()
                self.wfile.write(archive_bytes[:1])
                return
            if range_header and range_header.startswith("bytes="):
                self.send_response(500)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Length", str(len(archive_bytes)))
            self.end_headers()
            self.wfile.write(archive_bytes)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), FailingZenodoRangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        zenodo_like_url = f"http://127.0.0.1:{server.server_port}/zenodo.org/VCTK-Corpus-0.92.zip?download=1"
        env = os.environ.copy()
        env["PATH"] = f"{fake_bin_dir}{os.pathsep}{env['PATH']}"
        env["FAKE_ARIA2_LOG"] = str(fake_aria2_log)

        result = subprocess.run(
            [
                "bash",
                str(DOWNLOAD_SCRIPT),
                "--data-dir",
                str(data_dir),
                "--list-dir",
                str(lists_dir),
                "--download-dir",
                str(downloads_dir),
                "--extract-dir",
                str(extract_dir),
                "--profile",
                "production",
                "--aria2-conn",
                "4",
                "--aria2-split",
                "4",
                "--aria2-max-concurrent",
                "1",
                "--vctk-url",
                str(zenodo_like_url),
                "--musan-dir",
                str(musan_dir),
                "--air-rir-dir",
                str(air_rir_dir),
                "--no-download-librispeech",
                "--no-download-musan",
                "--no-download-fsd50k",
                "--no-download-air",
                "--no-download-openair",
                "--no-download-acousticrooms",
                "--zenodo-referer",
                "https://example.com/zenodo-test",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode != 0
    assert not fake_aria2_log.exists(), "Zenodo VCTK path should fail before invoking aria2"
    assert "bytes=0-0" in range_headers
    assert any(header != "bytes=0-0" for header in range_headers)
    assert "[error] parallel curl range download failed" in result.stderr
