"""Silero VAD integration for eval-only metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_DEFAULT_SILERO_SR = 16000


def _require_dependency(module_name: str, install_hint: str) -> None:
    raise RuntimeError(f"Silero VAD requires optional dependency '{module_name}'. " f"Install with: {install_hint}")


def _resolve_model_path(model_path: Optional[str]) -> Path:
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Silero VAD model not found: {path}")
        return path

    try:
        import importlib.resources as resources
    except ImportError as exc:  # pragma: no cover - Python <3.9
        raise RuntimeError("importlib.resources is required to load Silero VAD model") from exc

    try:
        model_file = resources.files("silero_vad.data").joinpath("silero_vad.onnx")
    except Exception as exc:  # pragma: no cover - package layout issues
        raise RuntimeError(
            "Could not locate silero_vad.onnx inside silero_vad package. "
            "Install silero-vad>=6.0.0 or provide --vad-silero-model-path."
        ) from exc

    return Path(model_file)


def _resample_audio(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav

    try:
        from scipy.signal import resample_poly
    except Exception as exc:  # pragma: no cover - scipy should exist in training env
        raise RuntimeError("scipy is required for Silero VAD resampling") from exc

    wav = np.asarray(wav, dtype=np.float32)
    resampled = resample_poly(wav, dst_sr, src_sr, axis=-1)
    return np.asarray(resampled, dtype=np.float32)


@dataclass
class SileroVADConfig:
    sample_rate: int = _DEFAULT_SILERO_SR
    model_path: Optional[str] = None
    max_seconds: Optional[float] = None
    force_cpu: bool = True


class SileroVAD:
    """Thin wrapper around silero-vad ONNX model for mean speech probabilities."""

    def __init__(self, config: SileroVADConfig):
        try:
            import torch
        except Exception:
            _require_dependency("torch", "pip install silero-vad torch onnxruntime")

        try:
            from silero_vad.utils_vad import OnnxWrapper
        except Exception:
            _require_dependency("silero_vad", "pip install silero-vad")

        try:
            import onnxruntime  # noqa: F401
        except Exception:
            _require_dependency("onnxruntime", "pip install onnxruntime")

        torch.set_num_threads(1)
        model_path = _resolve_model_path(config.model_path)
        self._model = OnnxWrapper(str(model_path), force_onnx_cpu=config.force_cpu)
        self.sample_rate = int(config.sample_rate)
        self.max_seconds = config.max_seconds

    def mean_probs(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return mean speech probability per clip (shape: [B])."""
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 1:
            wav = wav[None, :]
        if self.max_seconds and self.max_seconds > 0:
            max_len = int(self.max_seconds * sample_rate)
            if wav.shape[-1] > max_len:
                wav = wav[:, :max_len]

        if sample_rate != self.sample_rate:
            wav = _resample_audio(wav, sample_rate, self.sample_rate)

        import torch

        wav_t = torch.from_numpy(wav)
        probs = self._model.audio_forward(wav_t, self.sample_rate)
        probs = probs.detach().cpu().numpy()
        if probs.ndim == 1:
            probs = probs[None, :]
        return probs.mean(axis=1)
