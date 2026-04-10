from __future__ import annotations

import json
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDITION_SCRIPT = REPO_ROOT / "scripts" / "datasets" / "audition_background_music.py"


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
    if frames > 300:
        impulse[100] = 0.4
        impulse[260] = -0.18
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


def test_audition_background_music_writes_pack_with_compare_clips(tmp_path: Path) -> None:
    sample_rate = 16_000
    source_a = tmp_path / "music" / "artist_a" / "song_a.wav"
    source_b = tmp_path / "music" / "artist_b" / "song_b.wav"
    rir_file = tmp_path / "rir" / "room.wav"
    file_list = tmp_path / "lists" / "background_music.txt"
    rir_list = tmp_path / "lists" / "rir_all.txt"
    output_dir = tmp_path / "audition"

    _write_wav(source_a, sample_rate=sample_rate, seconds=1.4, frequency_hz=330.0)
    _write_wav(source_b, sample_rate=sample_rate, seconds=1.3, frequency_hz=440.0)
    _write_rir(rir_file, sample_rate=sample_rate)
    file_list.parent.mkdir(parents=True, exist_ok=True)
    file_list.write_text(f"{source_a}\n{source_b}\n", encoding="utf-8")
    rir_list.write_text(f"{rir_file}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(AUDITION_SCRIPT),
            "--file-list",
            str(file_list),
            "--output-dir",
            str(output_dir),
            "--sample-rate",
            str(sample_rate),
            "--style",
            "club_live",
            "--rir-list",
            str(rir_list),
            "--num-sources",
            "2",
            "--variants-per-source",
            "1",
            "--clip-seconds",
            "0.5",
            "--seed",
            "5",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "audition_manifest.json").read_text(encoding="utf-8"))
    assert manifest["style"] == "club_live"
    assert "club playback" in manifest["style_description"]
    samples = manifest["samples"]
    assert len(samples) == 2
    readme_text = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "Style preset: `club_live`" in readme_text

    for sample in samples:
        sample_dir = output_dir / sample["sample_dir"]
        original_path = sample_dir / "original.wav"
        prepared_path = sample_dir / "prepared_v00.wav"
        compare_path = sample_dir / "compare_v00.wav"
        assert original_path.exists()
        assert prepared_path.exists()
        assert compare_path.exists()

        original_audio, original_sr = _read_pcm16_wav(original_path)
        prepared_audio, prepared_sr = _read_pcm16_wav(prepared_path)
        compare_audio, compare_sr = _read_pcm16_wav(compare_path)

        assert original_sr == sample_rate
        assert prepared_sr == sample_rate
        assert compare_sr == sample_rate
        assert original_audio.shape == prepared_audio.shape
        assert np.mean(np.abs(prepared_audio - original_audio)) > 0.01
        assert compare_audio.shape[0] > original_audio.shape[0]
