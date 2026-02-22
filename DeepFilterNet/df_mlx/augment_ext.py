"""Augmentation extension bridge.

Provides accelerated augmentation functions via the Rust ``libdfaugment``
extension when available, falling back to pure-Python / SciPy implementations.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)

_RUST_AVAILABLE = False
try:
    from libdfaugment import biquad_filter as _rust_biquad
    from libdfaugment import combine_noises as _rust_combine
    from libdfaugment import mix_audio as _rust_mix

    _RUST_AVAILABLE = True
except ImportError:
    pass

if _RUST_AVAILABLE:
    _log.info("Rust augmentation extension (libdfaugment) loaded — using accelerated path")
else:
    _log.debug("libdfaugment not available — using Python/SciPy fallback for augmentations")


def rust_augment_available() -> bool:
    """Return ``True`` if the Rust augmentation extension is loaded."""
    return _RUST_AVAILABLE


def augment_capabilities() -> dict:
    """Return a dict summarizing which augmentation backends are active."""
    return {
        "rust_extension": _RUST_AVAILABLE,
        "biquad_backend": "rust" if _RUST_AVAILABLE else "scipy",
        "mix_backend": "rust" if _RUST_AVAILABLE else "numpy",
        "combine_backend": "rust" if _RUST_AVAILABLE else "numpy",
    }


def biquad_filter(
    audio: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Apply a second-order IIR (biquad) filter.

    Uses the Rust extension when available, otherwise falls back to
    ``scipy.signal.lfilter``.

    Args:
        audio: 1-D float32 audio samples.
        b: Numerator coefficients (length 3).
        a: Denominator coefficients (length 3).

    Returns:
        Filtered audio as float32 ndarray.
    """
    if _RUST_AVAILABLE:
        audio_f32 = np.ascontiguousarray(audio, dtype=np.float32)
        b_f32 = np.ascontiguousarray(b, dtype=np.float32)
        a_f32 = np.ascontiguousarray(a, dtype=np.float32)
        return np.asarray(_rust_biquad(audio_f32, b_f32, a_f32), dtype=np.float32)

    from scipy import signal as scipy_signal

    return np.asarray(scipy_signal.lfilter(b, a, audio), dtype=np.float32)


def mix_audio(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    gain_db: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mix clean speech with noise at a target SNR.

    Uses the Rust extension when available, otherwise falls back to the
    pure-Python implementation.

    Args:
        clean: Clean speech signal (float32).
        noise: Noise signal (float32).
        snr_db: Target SNR in dB.
        gain_db: Gain applied to the clean signal in dB.

    Returns:
        ``(clean_out, noise_scaled, noisy_mixture)``
    """
    if _RUST_AVAILABLE:
        clean_f32 = np.ascontiguousarray(clean, dtype=np.float32)
        noise_f32 = np.ascontiguousarray(noise, dtype=np.float32)
        c, ns, noisy = _rust_mix(clean_f32, noise_f32, float(snr_db), float(gain_db))
        return (
            np.asarray(c, dtype=np.float32),
            np.asarray(ns, dtype=np.float32),
            np.asarray(noisy, dtype=np.float32),
        )

    return _mix_audio_python(clean, noise, snr_db, gain_db)


def combine_noises(
    noises: List[np.ndarray],
    target_len: int,
    gains_db: Optional[List[float]] = None,
) -> np.ndarray:
    """Combine multiple noise signals into one buffer.

    Uses the Rust extension when available, otherwise falls back to the
    pure-Python implementation.

    Args:
        noises: List of 1-D noise arrays.
        target_len: Desired output length in samples.
        gains_db: Per-source gain in dB (default 0 dB for each).

    Returns:
        Combined noise of length *target_len* (float32).
    """
    if not noises:
        return np.zeros(target_len, dtype=np.float32)

    if gains_db is None:
        gains_db = [0.0] * len(noises)

    # Pre-compute random offsets on the Python side (keeps RNG deterministic)
    offsets: List[int] = []
    for noise in noises:
        if len(noise) > target_len:
            offsets.append(random.randint(0, len(noise) - target_len))
        else:
            offsets.append(0)

    if _RUST_AVAILABLE:
        arrs = [np.ascontiguousarray(n, dtype=np.float32) for n in noises]
        return np.asarray(
            _rust_combine(arrs, target_len, list(gains_db), offsets),
            dtype=np.float32,
        )

    return _combine_noises_python(noises, target_len, gains_db, offsets)


# ---------------------------------------------------------------------------
# Pure-Python fallbacks
# ---------------------------------------------------------------------------


def _mix_audio_python(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    gain_db: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gain = 10 ** (gain_db / 20)
    clean_out = clean * gain

    if len(noise) < len(clean_out):
        repeats = int(np.ceil(len(clean_out) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[: len(clean_out)]

    clean_power = np.mean(clean_out**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    mix_factor = np.sqrt(target_noise_power / noise_power)

    noise_scaled = noise * mix_factor
    noisy = clean_out + noise_scaled

    max_val = max(
        np.abs(clean_out).max(),
        np.abs(noise_scaled).max(),
        np.abs(noisy).max(),
    )
    if max_val > 1.0 - 1e-10:
        scale = 1.0 / (max_val + 1e-10)
        clean_out = clean_out * scale
        noise_scaled = noise_scaled * scale
        noisy = noisy * scale

    return (
        np.asarray(clean_out, dtype=np.float32),
        np.asarray(noise_scaled, dtype=np.float32),
        np.asarray(noisy, dtype=np.float32),
    )


def _combine_noises_python(
    noises: List[np.ndarray],
    target_len: int,
    gains_db: List[float],
    offsets: List[int],
) -> np.ndarray:
    combined = np.zeros(target_len, dtype=np.float32)

    for noise, gain_db, offset in zip(noises, gains_db, offsets):
        gain = 10 ** (gain_db / 20)

        if len(noise) < target_len:
            repeats = int(np.ceil(target_len / len(noise)))
            noise = np.tile(noise, repeats)

        if offset > 0:
            noise = noise[offset:]

        noise = noise[:target_len]
        combined += noise * gain

    return combined
