"""Waveform conversion and GAN signal processing utilities."""

from __future__ import annotations

import random
from typing import Callable

import mlx.core as mx

# Cached scalar zero — avoids allocation on early-return paths.
_ZERO = mx.array(0.0)


def specs_to_wavs(
    out_spec: tuple[mx.array, mx.array],
    clean_spec: tuple[mx.array, mx.array],
    *,
    istft_fn: Callable[..., mx.array],
    n_fft: int,
    hop_length: int,
    target_len: int,
    force_fp32: bool = True,
) -> tuple[mx.array, mx.array]:
    """Convert complex specs to waveforms with optional FP32 stabilization."""
    if force_fp32:
        out_spec = (
            out_spec[0].astype(mx.float32) if out_spec[0].dtype != mx.float32 else out_spec[0],
            out_spec[1].astype(mx.float32) if out_spec[1].dtype != mx.float32 else out_spec[1],
        )
        clean_spec = (
            (clean_spec[0].astype(mx.float32) if clean_spec[0].dtype != mx.float32 else clean_spec[0]),
            (clean_spec[1].astype(mx.float32) if clean_spec[1].dtype != mx.float32 else clean_spec[1]),
        )

    clean_wav = istft_fn(
        clean_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        length=target_len,
    )
    out_wav = istft_fn(
        out_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        length=target_len,
    )

    if force_fp32:
        if clean_wav.dtype != mx.float32:
            clean_wav = clean_wav.astype(mx.float32)
        if out_wav.dtype != mx.float32:
            out_wav = out_wav.astype(mx.float32)

    return out_wav, clean_wav


def compute_mrstft_loss(
    out_spec: tuple[mx.array, mx.array],
    clean_spec: tuple[mx.array, mx.array],
    *,
    istft_fn: Callable[..., mx.array],
    loss_fn: Callable[[mx.array, mx.array], mx.array],
    n_fft: int,
    hop_length: int,
    target_len: int,
    force_fp32: bool = True,
) -> mx.array:
    """Compute MRSTFT loss from complex specs with optional FP32 stabilization.

    MRSTFT involves magnitude squaring and power compression, which can overflow
    in FP16 when the model outputs large spectral magnitudes. We optionally cast
    to FP32 for this path to keep losses finite while the rest of the training
    stays in mixed precision.
    """
    if istft_fn is None or loss_fn is None:
        return _ZERO

    out_wav, clean_wav = specs_to_wavs(
        out_spec,
        clean_spec,
        istft_fn=istft_fn,
        n_fft=n_fft,
        hop_length=hop_length,
        target_len=target_len,
        force_fp32=force_fp32,
    )

    return loss_fn(out_wav, clean_wav)


def _gan_waveform_view(wav: mx.array, *, use_fp16: bool) -> mx.array:
    """Return GAN discriminator waveform view in the desired precision.

    GAN discriminator activations are a major memory contributor when adversarial
    training activates. Keeping this path in model precision (BF16 when enabled)
    reduces peak memory while MRSTFT can still run in FP32 for stability.
    """
    if use_fp16 and wav.dtype != mx.bfloat16:
        return wav.astype(mx.bfloat16)
    return wav


def _disc_crop_waveform(wav: mx.array, max_samples: int, crop_start: int | None = None) -> tuple[mx.array, int]:
    """Random-crop waveform along the time axis for discriminator input.

    Waveform-domain discriminators (MPD/MSD) produce enormous activation tensors
    proportional to input length.  Cropping to a shorter segment (e.g. 1 s at
    48 kHz = 48 000 samples) cuts discriminator memory by the ratio
    ``original_len / max_samples`` with negligible quality impact — the
    discriminator only needs to assess local perceptual quality.

    Args:
        wav: Waveform tensor ``(batch, samples)``.
        max_samples: Maximum number of samples to keep (0 = no crop).
        crop_start: If given, reuse this start index (keeps fake/real aligned).

    Returns:
        (cropped_wav, crop_start) so the same offset can be reused for the
        paired waveform.
    """
    if max_samples <= 0 or wav.shape[-1] <= max_samples:
        return wav, 0
    if crop_start is None:
        crop_start = random.randint(0, wav.shape[-1] - max_samples)
    return wav[:, crop_start : crop_start + max_samples], crop_start
