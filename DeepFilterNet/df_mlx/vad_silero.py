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
            from silero_vad import get_speech_timestamps
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
        self._get_speech_timestamps = get_speech_timestamps
        self._torch = torch
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

    def speech_timestamps(
        self,
        wav: np.ndarray,
        sample_rate: int,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> list[list[dict[str, int]]]:
        """Return speech timestamp segments per clip in source-sample units."""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 1:
            wav = wav[None, :]

        all_segments: list[list[dict[str, int]]] = []
        for clip in wav:
            src_len = int(clip.shape[0])
            clip_for_vad = clip
            if self.max_seconds and self.max_seconds > 0:
                max_len = int(self.max_seconds * sample_rate)
                clip_for_vad = clip_for_vad[:max_len]

            if sample_rate != self.sample_rate:
                clip_for_vad = _resample_audio(clip_for_vad, sample_rate, self.sample_rate)
                resample_ratio = float(sample_rate) / float(self.sample_rate)
            else:
                resample_ratio = 1.0

            clip_tensor = self._torch.from_numpy(np.asarray(clip_for_vad, dtype=np.float32))
            segments = self._get_speech_timestamps(
                clip_tensor,
                self._model,
                threshold=float(threshold),
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=int(min_speech_duration_ms),
                min_silence_duration_ms=int(min_silence_duration_ms),
                speech_pad_ms=int(speech_pad_ms),
                return_seconds=False,
            )

            clip_segments: list[dict[str, int]] = []
            for segment in segments:
                start = int(round(float(segment["start"]) * resample_ratio))
                end = int(round(float(segment["end"]) * resample_ratio))
                start = max(0, min(start, src_len))
                end = max(start, min(end, src_len))
                if end > start:
                    clip_segments.append({"start": start, "end": end})
            all_segments.append(clip_segments)

        return all_segments

    def apply_speech_gain(
        self,
        wav: np.ndarray,
        sample_rate: int,
        gain_db: float,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        ramp_ms: float = 8.0,
        peak_limit: float = 0.99,
    ) -> tuple[np.ndarray, list[list[dict[str, int]]]]:
        """Amplify VAD-detected speech segments and return boosted audio + segments."""
        wav = np.asarray(wav, dtype=np.float32)
        input_1d = wav.ndim == 1
        if input_1d:
            wav = wav[None, :]

        segments_per_clip = self.speech_timestamps(
            wav,
            sample_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        if gain_db <= 0.0:
            return (wav[0] if input_1d else wav), segments_per_clip

        linear_gain = float(10.0 ** (gain_db / 20.0))
        ramp_samples = max(0, int(round(float(ramp_ms) * sample_rate / 1000.0)))

        boosted = np.array(wav, copy=True)
        for clip_idx, segments in enumerate(segments_per_clip):
            clip = boosted[clip_idx]
            envelope = np.ones_like(clip, dtype=np.float32)

            for segment in segments:
                start = int(segment["start"])
                end = int(segment["end"])
                if end <= start:
                    continue

                envelope[start:end] = np.maximum(envelope[start:end], linear_gain)
                if ramp_samples <= 0:
                    continue

                ramp = min(ramp_samples, end - start)
                up_end = start + ramp
                if up_end > start:
                    up = np.linspace(1.0, linear_gain, up_end - start, endpoint=False, dtype=np.float32)
                    envelope[start:up_end] = np.maximum(envelope[start:up_end], up)

                down_start = end - ramp
                if end > down_start:
                    down = np.linspace(linear_gain, 1.0, end - down_start, endpoint=True, dtype=np.float32)
                    envelope[down_start:end] = np.maximum(envelope[down_start:end], down)

            clip *= envelope
            peak = float(np.max(np.abs(clip))) if clip.size else 0.0
            if peak_limit > 0.0 and peak > peak_limit:
                clip *= peak_limit / peak

        return (boosted[0] if input_1d else boosted), segments_per_clip
