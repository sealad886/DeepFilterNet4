import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_no_silero_fallback_message():
    root = Path(__file__).resolve().parents[1]
    train_script = root / "df_mlx" / "train_dynamic.py"
    text = train_script.read_text()
    assert "falling back to proxy" not in text.lower()


@pytest.mark.skipif(
    os.getenv("DFNET_SILERO_SMOKE", "").lower() not in {"1", "true", "yes"},
    reason="Set DFNET_SILERO_SMOKE=1 to run Silero CLI smoke test",
)
def test_silero_cli_smoke(tmp_path):
    if importlib.util.find_spec("silero_vad") is None:
        pytest.skip("silero_vad not installed")
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnxruntime not installed")
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")

    # Create minimal 5s speech/noise WAVs at 48kHz (segment_length default)
    sr = 48000
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    speech = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    noise = 0.05 * np.random.randn(len(t)).astype(np.float32)

    speech_path = tmp_path / "speech.wav"
    noise_path = tmp_path / "noise.wav"

    try:
        import soundfile as sf

        sf.write(speech_path, speech, sr)
        sf.write(noise_path, noise, sr)
    except Exception:
        from scipy.io import wavfile

        wavfile.write(speech_path, sr, speech)
        wavfile.write(noise_path, sr, noise)

    speech_list = tmp_path / "speech.txt"
    noise_list = tmp_path / "noise.txt"
    speech_list.write_text(str(speech_path))
    noise_list.write_text(str(noise_path))

    root = Path(__file__).resolve().parents[1]
    pkg_root = root / "DeepFilterNet"
    cmd = [
        sys.executable,
        "-m",
        "df_mlx.train_dynamic",
        "--speech-list",
        str(speech_list),
        "--noise-list",
        str(noise_list),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--max-train-batches",
        "1",
        "--max-valid-batches",
        "1",
        "--eval-frequency",
        "1",
        "--vad-eval-mode",
        "silero",
        "--vad-eval-batches",
        "1",
        "--no-mlx-data",
    ]
    env = os.environ.copy()
    env["DFNET_TQDM"] = "0"
    env["PYTHONPATH"] = str(pkg_root)

    result = subprocess.run(cmd, cwd=pkg_root, env=env, capture_output=True, text=True, check=False)
    output = result.stdout + "\n" + result.stderr

    assert result.returncode == 0, output
    assert "vad_eval_ref" in output
    assert "vad_eval_out" in output
    assert "falling back to proxy" not in output.lower()
