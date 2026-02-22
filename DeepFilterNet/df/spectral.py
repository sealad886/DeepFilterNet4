"""Device-native PyTorch spectral frontend for DeepFilterNet.

Replaces the Rust libDF dependency for STFT/iSTFT and feature extraction,
keeping all computation on the current device (GPU/MPS/CPU) and eliminating
CPU round-trips through pyDF.

Window: Vorbis window  sin(pi/2 * sin^2(pi * (n+0.5) / N))
Normalization: wnorm = 2 * hop_size / fft_size^2  (matches Rust libDF)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def vorbis_window(
    fft_size: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Exact Vorbis window matching Rust libDF implementation.

    Formula per sample *i*:
        w[i] = sin(pi/2 * sin^2(pi * (i + 0.5) / fft_size))
    """
    half = fft_size / 2
    pi = math.pi
    n = torch.arange(fft_size, dtype=torch.float64, device=device)
    inner_sin = torch.sin(0.5 * pi * (n + 0.5) / half)
    window = torch.sin(0.5 * pi * inner_sin * inner_sin)
    return window.to(dtype=dtype)


def freq2erb(freq: float) -> float:
    """Glasberg & Moore ERB-rate scale: frequency (Hz) -> ERB number."""
    return 9.265 * math.log(1.0 + freq / (24.7 * 9.265))


def erb2freq(n_erb: float) -> float:
    """Glasberg & Moore ERB-rate scale: ERB number -> frequency (Hz)."""
    return 24.7 * 9.265 * (math.exp(n_erb / 9.265) - 1.0)


def compute_erb_fb(
    sr: int = 48000,
    fft_size: int = 960,
    nb_bands: int = 32,
    min_nb_freqs: int = 2,
) -> List[int]:
    """ERB filterbank band widths matching Rust ``erb_fb()`` exactly."""
    freq_width = sr / fft_size
    erb_low = freq2erb(0.0)
    erb_high = freq2erb(sr / 2)
    step = (erb_high - erb_low) / nb_bands

    erb: List[int] = [0] * nb_bands
    prev_freq = 0
    freq_over = 0
    for i in range(1, nb_bands + 1):
        f = erb2freq(erb_low + i * step)
        fb = round(f / freq_width)
        nb_freqs = fb - prev_freq - freq_over
        if nb_freqs < min_nb_freqs:
            freq_over = min_nb_freqs - nb_freqs
            nb_freqs = min_nb_freqs
        else:
            freq_over = 0
        erb[i - 1] = nb_freqs
        prev_freq = fb

    erb[nb_bands - 1] += 1  # account for fft_size/2 + 1 bins
    too_large = sum(erb) - (fft_size // 2 + 1)
    if too_large > 0:
        erb[nb_bands - 1] -= too_large

    assert sum(erb) == fft_size // 2 + 1, f"ERB band sum {sum(erb)} != {fft_size // 2 + 1}"
    return erb


def _build_erb_matrix(
    erb_widths: List[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Build dense ERB mapping matrix of shape ``[nb_bands, n_freqs]``.

    Each row contains ``1 / band_size`` for the frequency bins belonging to
    that band, matching the Rust ``compute_band_corr`` scaling.
    """
    n_freqs = sum(erb_widths)
    nb_bands = len(erb_widths)
    matrix = torch.zeros(nb_bands, n_freqs, dtype=dtype, device=device)
    start = 0
    for band_idx, width in enumerate(erb_widths):
        matrix[band_idx, start : start + width] = 1.0 / width
        start += width
    return matrix


class SpectralFrontend(nn.Module):
    """Pure-PyTorch spectral frontend compatible with DeepFilterNet.

    Provides STFT / iSTFT with the Vorbis window, ERB-band energy computation,
    and running-mean / unit normalization — all on-device with no Rust dependency.
    """

    def __init__(
        self,
        sr: int = 48000,
        fft_size: int = 960,
        hop_size: int = 480,
        nb_bands: int = 32,
        min_nb_freqs: int = 2,
    ) -> None:
        super().__init__()
        self._sr = sr
        self._fft_size = fft_size
        self._hop_size = hop_size
        self._nb_bands = nb_bands

        self.register_buffer("window", vorbis_window(fft_size))
        wnorm = 2.0 * hop_size / (fft_size * fft_size)
        self.register_buffer("wnorm", torch.tensor(wnorm))

        widths = compute_erb_fb(sr, fft_size, nb_bands, min_nb_freqs)
        self.register_buffer("erb_widths", torch.tensor(widths, dtype=torch.int32))
        self.register_buffer("_erb_matrix", _build_erb_matrix(widths))

        self._erb_norm_state: Optional[Tensor] = None
        self._unit_norm_state: Optional[Tensor] = None

    # -- properties ----------------------------------------------------------

    @property
    def fft_size_val(self) -> int:
        return self._fft_size

    @property
    def hop_size_val(self) -> int:
        return self._hop_size

    @property
    def sr_val(self) -> int:
        return self._sr

    @property
    def erb_widths_val(self) -> Tensor:
        return self.erb_widths  # type: ignore[return-value]

    # -- core transforms ------------------------------------------------------

    def analysis(self, audio: Tensor) -> Tensor:
        """STFT: ``[..., T]`` -> ``[..., T', F]`` complex."""
        spec = torch.stft(
            audio,
            n_fft=self._fft_size,
            hop_length=self._hop_size,
            window=self.window,  # type: ignore[arg-type]
            normalized=False,
            return_complex=True,
            center=False,
        )
        # torch.stft returns [..., F, T'] — transpose last two dims to [..., T', F]
        spec = spec.transpose(-2, -1)
        spec = spec * self.wnorm  # type: ignore[operator]
        return spec

    def synthesis(self, spec: Tensor) -> Tensor:
        """iSTFT: ``[..., T', F]`` complex -> ``[..., T]``."""
        spec = spec / self.wnorm  # type: ignore[operator]
        spec = spec.transpose(-2, -1)  # [..., T', F] -> [..., F, T']
        audio = torch.istft(
            spec,
            n_fft=self._fft_size,
            hop_length=self._hop_size,
            window=self.window,  # type: ignore[arg-type]
            normalized=False,
            center=False,
        )
        return audio

    # -- ERB features ---------------------------------------------------------

    def erb(self, spec: Tensor, db: bool = True) -> Tensor:
        """ERB-band energies from complex spectrogram.

        Args:
            spec: Complex spectrogram ``[..., T', F]``.
            db: Convert output to decibels.

        Returns:
            ERB energies ``[..., T', nb_bands]``.
        """
        power = (spec * spec.conj()).real  # |X|^2
        erb_out = torch.matmul(power, self._erb_matrix.T)  # type: ignore[arg-type]
        if db:
            erb_out = 10.0 * torch.log10(erb_out + 1e-10)
        return erb_out

    def erb_norm(self, erb_feat: Tensor, alpha: float) -> Tensor:
        """Running-mean normalization over ERB bands (EMA).

        Matches Rust ``band_mean_norm_erb``.  State initialised from
        ``linspace(-60, -90, nb_bands)``.

        Args:
            erb_feat: ``[..., T', E]`` real ERB energies.
            alpha: EMA decay factor.

        Returns:
            Normalised ERB features, same shape.
        """
        *leading, t_steps, n_bands = erb_feat.shape
        flat = erb_feat.reshape(-1, t_steps, n_bands)
        batch = flat.shape[0]

        if self._erb_norm_state is None or self._erb_norm_state.shape[0] != batch:
            self._erb_norm_state = (
                torch.linspace(-60.0, -90.0, n_bands, device=erb_feat.device, dtype=erb_feat.dtype)
                .unsqueeze(0)
                .expand(batch, -1)
                .clone()
            )

        output = torch.empty_like(flat)
        for t in range(t_steps):
            self._erb_norm_state = flat[:, t] * (1.0 - alpha) + self._erb_norm_state * alpha
            output[:, t] = (flat[:, t] - self._erb_norm_state) / 40.0

        return output.reshape_as(erb_feat)

    def unit_norm(self, spec: Tensor, alpha: float) -> Tensor:
        """Running unit normalization over complex spectrogram bins (EMA).

        Matches Rust ``band_unit_norm``.  State initialised from
        ``linspace(0.001, 0.0001, F)``.

        Args:
            spec: ``[..., T', F]`` complex spectrogram (typically first ``nb_df`` bins).
            alpha: EMA decay factor.

        Returns:
            Unit-normalised complex spectrogram, same shape.
        """
        *leading, t_steps, n_freqs = spec.shape
        flat = spec.reshape(-1, t_steps, n_freqs)
        batch = flat.shape[0]

        if self._unit_norm_state is None or self._unit_norm_state.shape[0] != batch:
            self._unit_norm_state = (
                torch.linspace(
                    0.001,
                    0.0001,
                    n_freqs,
                    device=spec.device,
                    dtype=spec.real.dtype,
                )
                .unsqueeze(0)
                .expand(batch, -1)
                .clone()
            )

        output = torch.empty_like(flat)
        for t in range(t_steps):
            mag = flat[:, t].abs()
            self._unit_norm_state = mag * (1.0 - alpha) + self._unit_norm_state * alpha
            output[:, t] = flat[:, t] / self._unit_norm_state.sqrt()

        return output.reshape_as(spec)

    # -- state management -----------------------------------------------------

    def reset(self) -> None:
        """Clear overlap / normalisation states for a new utterance."""
        self._erb_norm_state = None
        self._unit_norm_state = None


def df_features_torch(
    audio: Tensor,
    frontend: SpectralFrontend,
    nb_df: int,
    *,
    norm_alpha: Optional[float] = None,
    device: Optional[torch.device | str] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pure-PyTorch replacement for ``df_features()`` from ``df/enhance.py``.

    Returns ``(spec, erb_feat, spec_feat)`` with shapes matching the original
    Rust-backed implementation.

    Args:
        audio: Time-domain audio ``[C, T]``.
        frontend: A :class:`SpectralFrontend` instance.
        nb_df: Number of deep-filtering frequency bins.
        norm_alpha: EMA decay factor.  When *None*, computed from
            ``DfParams`` via :func:`df.utils.get_norm_alpha`.
        device: Target device for the returned tensors.

    Returns:
        ``(spec, erb_feat, spec_feat)`` — all on *device*.
    """
    frontend.reset()

    if norm_alpha is None:
        from df.utils import get_norm_alpha

        norm_alpha = get_norm_alpha(log=False)

    spec = frontend.analysis(audio)  # [..., T', F] complex

    erb_feat = frontend.erb(spec, db=True)  # [..., T', E]
    erb_feat = frontend.erb_norm(erb_feat, norm_alpha)  # [..., T', E]
    erb_feat = erb_feat.unsqueeze(-3)  # [..., 1, T', E]

    spec_sub = spec[..., :nb_df]  # [..., T', nb_df]
    spec_sub = frontend.unit_norm(spec_sub, norm_alpha)  # [..., T', nb_df]
    spec_feat = torch.stack([spec_sub.real, spec_sub.imag], dim=-1)  # [..., T', nb_df, 2]
    spec_feat = spec_feat.unsqueeze(-4)  # [..., 1, T', nb_df, 2]

    spec_out = torch.stack([spec.real, spec.imag], dim=-1)  # [..., T', F, 2]
    spec_out = spec_out.unsqueeze(-4)  # [..., 1, T', F, 2]

    if device is not None:
        spec_out = spec_out.to(device)
        erb_feat = erb_feat.to(device)
        spec_feat = spec_feat.to(device)

    return spec_out, erb_feat, spec_feat
