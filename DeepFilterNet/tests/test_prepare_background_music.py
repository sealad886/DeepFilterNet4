from __future__ import annotations

import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "datasets" / "prepare_background_music.py"


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


def _write_rir(path: Path, *, sample_rate: int, seconds: float = 0.2) -> None:
    frames = max(1, int(sample_rate * seconds))
    impulse = np.zeros(frames, dtype=np.float32)
    impulse[0] = 1.0
    if frames > 400:
        impulse[120] = 0.45
        impulse[320] = -0.22
    pcm = np.clip(impulse, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _read_pcm16_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate


def test_prepare_background_music_renders_style_variants_with_consistent_length_and_resumes(tmp_path: Path) -> None:
    sample_rate = 16_000
    base_dir = tmp_path / "raw"
    music_file = base_dir / "music" / "artist" / "track.wav"
    rir_file = tmp_path / "rir" / "room.wav"
    input_list = tmp_path / "lists" / "background_music.txt"
    rir_list = tmp_path / "lists" / "rir_all.txt"
    output_root = tmp_path / "prepared"
    output_list = tmp_path / "lists" / "background_music.prepared.txt"

    _write_wav(music_file, sample_rate=sample_rate, seconds=15001 / sample_rate, frequency_hz=330.0)
    _write_rir(rir_file, sample_rate=sample_rate)
    input_list.parent.mkdir(parents=True, exist_ok=True)
    input_list.write_text(f"{music_file}\n", encoding="utf-8")
    rir_list.write_text(f"{rir_file}\n", encoding="utf-8")

    cmd = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--file-list",
        str(input_list),
        "--output-root",
        str(output_root),
        "--base-dir",
        str(base_dir),
        "--output-list",
        str(output_list),
        "--sample-rate",
        str(sample_rate),
        "--style",
        "phone_room",
        "--rir-list",
        str(rir_list),
        "--variants-per-source",
        "2",
        "--seed",
        "7",
    ]

    first = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    assert "[progress] starting source 1/1:" in first.stderr
    assert "[progress] 2/2 variants" in first.stderr
    prepared_entries = [
        Path(line.strip()) for line in output_list.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(prepared_entries) == 2
    assert all(path.exists() for path in prepared_entries)
    assert all("phone_room_v" in str(path) for path in prepared_entries)

    source_audio, source_sr = _read_pcm16_wav(music_file)
    assert source_sr == sample_rate
    first_mtime = prepared_entries[0].stat().st_mtime_ns

    for prepared_path in prepared_entries:
        prepared_audio, prepared_sr = _read_pcm16_wav(prepared_path)
        assert prepared_sr == sample_rate
        assert prepared_audio.shape == source_audio.shape
        assert np.mean(np.abs(prepared_audio - source_audio)) > 0.01

    second = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert second.returncode == 0, second.stderr
    assert prepared_entries[0].stat().st_mtime_ns == first_mtime
