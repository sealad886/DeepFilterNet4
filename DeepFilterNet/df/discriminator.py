"""Multi-Period Discriminator for adversarial training in DeepFilterNet4.

Implements HiFi-GAN style multi-period discriminator for improved perceptual quality.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import spectral_norm, weight_norm


class PeriodDiscriminator(nn.Module):
    """Single-period sub-discriminator.

    Reshapes waveform into 2D based on period and applies 2D convolutions
    to capture periodic patterns in speech at different scales.

    Args:
        period: Period for reshaping the waveform
        use_spectral_norm: Use spectral normalization instead of weight normalization
        channels: Channel progression for conv layers
    """

    def __init__(
        self,
        period: int,
        use_spectral_norm: bool = False,
        channels: List[int] = [32, 128, 512, 1024, 1024],
    ):
        super().__init__()
        self.period = period

        norm_f = spectral_norm if use_spectral_norm else weight_norm

        # Build conv layers with progressive channel growth
        self.convs = nn.ModuleList()
        in_ch = 1
        for i, out_ch in enumerate(channels):
            # First 4 layers downsample, last layer doesn't
            stride = (3, 1) if i < len(channels) - 1 else (1, 1)
            self.convs.append(norm_f(nn.Conv2d(in_ch, out_ch, (5, 1), stride, padding=(2, 0))))
            in_ch = out_ch

        self.conv_post = norm_f(nn.Conv2d(channels[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Process waveform through period-based discrimination.

        Args:
            x: Waveform tensor [B, 1, T] or [B, T]

        Returns:
            score: Discriminator score tensor [B, N] where N depends on input length
            fmap: List of intermediate feature maps for feature matching loss
        """
        fmap = []

        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)

        b, c, t = x.shape

        # Pad to make divisible by period
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            t = t + n_pad

        # Reshape to 2D: [B, 1, T] -> [B, 1, T//period, period]
        x = x.view(b, c, t // self.period, self.period)

        # Apply conv layers
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        # Flatten for final score
        x = torch.flatten(x, 1, -1)

        return x, fmap


class ScaleDiscriminator(nn.Module):
    """Multi-scale sub-discriminator.

    Processes waveform at different time scales using 1D convolutions.
    Complements period discriminator for capturing non-periodic patterns.

    Args:
        use_spectral_norm: Use spectral normalization
        channels: Channel progression
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        channels: List[int] = [128, 128, 256, 512, 1024, 1024, 1024],
    ):
        super().__init__()

        norm_f = spectral_norm if use_spectral_norm else weight_norm

        # Build 1D conv layers
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, channels[0], 15, 1, padding=7)),
                norm_f(nn.Conv1d(channels[0], channels[1], 41, 2, groups=4, padding=20)),
                norm_f(nn.Conv1d(channels[1], channels[2], 41, 2, groups=16, padding=20)),
                norm_f(nn.Conv1d(channels[2], channels[3], 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(channels[3], channels[4], 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(channels[4], channels[5], 41, 1, groups=16, padding=20)),
                norm_f(nn.Conv1d(channels[5], channels[6], 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(channels[-1], 1, 3, 1, padding=1))

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Process waveform through scale-based discrimination.

        Args:
            x: Waveform tensor [B, 1, T] or [B, T]

        Returns:
            score: Discriminator score tensor
            fmap: List of feature maps
        """
        fmap = []

        if x.dim() == 2:
            x = x.unsqueeze(1)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator for GAN-based speech enhancement.

    Combines multiple period-based sub-discriminators to capture
    different periodic patterns in speech signals (pitch harmonics, formants).

    Args:
        periods: List of periods for sub-discriminators. Default [2, 3, 5, 7, 11]
            covers prime numbers to minimize overlap
        use_spectral_norm: Use spectral normalization for stability
    """

    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.periods = periods

        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(p, use_spectral_norm=use_spectral_norm) for p in periods]
        )

    def forward(
        self,
        y: Tensor,
    ) -> Tuple[List[Tensor], List[List[Tensor]]]:
        """Run all period discriminators on input.

        Args:
            y: Waveform tensor [B, 1, T] or [B, T]

        Returns:
            scores: List of discriminator scores, one per period
            fmaps: List of feature map lists, one list per discriminator
        """
        if y.dim() == 2:
            y = y.unsqueeze(1)

        scores = []
        fmaps = []

        for d in self.discriminators:
            score, fmap = d(y)
            scores.append(score)
            fmaps.append(fmap)

        return scores, fmaps


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator using average pooling.

    Processes input at multiple time scales using average pooling
    to capture patterns at different temporal resolutions.

    Args:
        num_scales: Number of scales to process
        use_spectral_norm: Use spectral normalization
    """

    def __init__(
        self,
        num_scales: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        self.discriminators = nn.ModuleList(
            [ScaleDiscriminator(use_spectral_norm=use_spectral_norm) for _ in range(num_scales)]
        )

        # Downsampling between scales
        self.pools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2) for _ in range(num_scales - 1)])

    def forward(
        self,
        y: Tensor,
    ) -> Tuple[List[Tensor], List[List[Tensor]]]:
        """Run multi-scale discrimination.

        Args:
            y: Waveform tensor [B, 1, T] or [B, T]

        Returns:
            scores: List of discriminator scores at each scale
            fmaps: List of feature map lists
        """
        if y.dim() == 2:
            y = y.unsqueeze(1)

        scores = []
        fmaps = []

        for i, d in enumerate(self.discriminators):
            if i > 0:
                y = self.pools[i - 1](y)
            score, fmap = d(y)
            scores.append(score)
            fmaps.append(fmap)

        return scores, fmaps


class CombinedDiscriminator(nn.Module):
    """Combined multi-period and multi-scale discriminator.

    Uses both period-based and scale-based discrimination for
    comprehensive coverage of speech characteristics.

    Args:
        periods: Periods for MPD
        num_scales: Number of scales for MSD
        use_spectral_norm: Use spectral normalization
    """

    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        num_scales: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        self.mpd = MultiPeriodDiscriminator(periods=periods, use_spectral_norm=use_spectral_norm)
        self.msd = MultiScaleDiscriminator(
            num_scales=num_scales, use_spectral_norm=use_spectral_norm
        )

    def forward(
        self,
        y: Tensor,
    ) -> Tuple[List[Tensor], List[List[Tensor]], List[Tensor], List[List[Tensor]]]:
        """Run combined discrimination.

        Args:
            y: Waveform tensor [B, 1, T] or [B, T]

        Returns:
            mpd_scores: Multi-period discriminator scores
            mpd_fmaps: Multi-period feature maps
            msd_scores: Multi-scale discriminator scores
            msd_fmaps: Multi-scale feature maps
        """
        mpd_scores, mpd_fmaps = self.mpd(y)
        msd_scores, msd_fmaps = self.msd(y)

        return mpd_scores, mpd_fmaps, msd_scores, msd_fmaps


def discriminator_loss(
    real_scores: List[Tensor],
    fake_scores: List[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Compute discriminator loss for real and fake samples.

    Uses least-squares GAN loss for stability.

    Args:
        real_scores: Discriminator outputs for real samples
        fake_scores: Discriminator outputs for fake/generated samples

    Returns:
        d_loss: Total discriminator loss
        d_real_loss: Loss on real samples (for logging)
    """
    d_loss = 0.0
    d_real_loss = 0.0

    for real_score, fake_score in zip(real_scores, fake_scores):
        # LS-GAN: D wants real->1, fake->0
        r_loss = torch.mean((1 - real_score) ** 2)
        f_loss = torch.mean(fake_score**2)
        d_loss = d_loss + r_loss + f_loss
        d_real_loss = d_real_loss + r_loss

    return d_loss, d_real_loss  # type: ignore[return-value]


def generator_loss(
    fake_scores: List[Tensor],
) -> Tensor:
    """Compute generator loss for adversarial training.

    Generator tries to make discriminator output 1 for fake samples.

    Args:
        fake_scores: Discriminator outputs for generated samples

    Returns:
        g_loss: Generator adversarial loss
    """
    g_loss = 0.0

    for fake_score in fake_scores:
        # LS-GAN: G wants fake->1
        g_loss = g_loss + torch.mean((1 - fake_score) ** 2)

    return g_loss  # type: ignore[return-value]


def feature_matching_loss(
    real_fmaps: List[List[Tensor]],
    fake_fmaps: List[List[Tensor]],
) -> Tensor:
    """Compute feature matching loss between real and fake feature maps.

    Encourages generator to match intermediate discriminator features.

    Args:
        real_fmaps: Feature maps from discriminator on real samples
        fake_fmaps: Feature maps from discriminator on fake samples

    Returns:
        fm_loss: Feature matching loss
    """
    fm_loss = 0.0

    for real_fmap_list, fake_fmap_list in zip(real_fmaps, fake_fmaps):
        for real_fmap, fake_fmap in zip(real_fmap_list, fake_fmap_list):
            fm_loss = fm_loss + torch.mean(torch.abs(real_fmap.detach() - fake_fmap))

    return fm_loss  # type: ignore[return-value]
