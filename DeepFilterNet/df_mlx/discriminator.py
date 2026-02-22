"""GAN Discriminators for MLX DeepFilterNet4 training.

Provides multi-scale and multi-period discriminators for adversarial training,
ported from the PyTorch implementation (df/discriminator.py).

Architecture includes:
- PeriodDiscriminator: Analyzes periodic patterns using 2D convolution
- ScaleDiscriminator: Multi-scale 1D convolution analysis
- MultiPeriodDiscriminator: Ensemble of period discriminators
- MultiScaleDiscriminator: Ensemble of scale discriminators
- CombinedDiscriminator: Combined MPD + MSD

The discriminators output both final scores and intermediate feature maps
for feature matching loss computation.
"""

from typing import List, Literal, Tuple

import mlx.core as mx
import mlx.nn as nn

from .loss import FeatureMatchingLoss, discriminator_loss, generator_loss

# ============================================================================
# Utility Functions
# ============================================================================


def get_padding(
    kernel_size: int | Tuple[int, int],
    dilation: int = 1,
) -> int | Tuple[int, int]:
    """Calculate same padding for convolution.

    Args:
        kernel_size: Convolution kernel size
        dilation: Dilation rate

    Returns:
        Padding to achieve same output size (for stride=1)
    """
    if isinstance(kernel_size, tuple):
        return (
            (kernel_size[0] - 1) * dilation // 2,
            (kernel_size[1] - 1) * dilation // 2,
        )
    return (kernel_size - 1) * dilation // 2


def weight_norm_conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> nn.Conv1d:
    """Create Conv1d with weight normalization.

    Note: MLX doesn't have native weight norm, so we use regular Conv1d.
    Weight normalization can be applied externally if needed.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Input padding
        dilation: Dilation rate
        groups: Number of groups

    Returns:
        Configured Conv1d layer
    """
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def weight_norm_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
) -> nn.Conv2d:
    """Create Conv2d with weight normalization.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Input padding

    Returns:
        Configured Conv2d layer
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


# ============================================================================
# Period Discriminator
# ============================================================================


class PeriodDiscriminator(nn.Module):
    """Period-based discriminator using 2D convolution.

    Reshapes 1D audio into 2D by grouping samples at periodic intervals,
    then applies 2D convolutions to capture periodic patterns.

    Args:
        period: Period for reshaping (e.g., 2, 3, 5, 7, 11)
        channels: Number of channels in conv layers
        kernel_size: Kernel size for convolutions

    Example:
        For period=2, audio [s0, s1, s2, s3, s4, s5, ...] becomes:
        [[s0, s2, s4, ...],
         [s1, s3, s5, ...]]
    """

    def __init__(
        self,
        period: int = 2,
        channels: int = 32,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.period = period

        # Convolutional layers with increasing channels
        self.convs = [
            weight_norm_conv2d(1, channels, (kernel_size, 1), (3, 1), (get_padding(kernel_size), 0)),
            weight_norm_conv2d(channels, channels * 2, (kernel_size, 1), (3, 1), (get_padding(kernel_size), 0)),
            weight_norm_conv2d(channels * 2, channels * 4, (kernel_size, 1), (3, 1), (get_padding(kernel_size), 0)),
            weight_norm_conv2d(channels * 4, channels * 8, (kernel_size, 1), (3, 1), (get_padding(kernel_size), 0)),
            weight_norm_conv2d(channels * 8, channels * 16, (kernel_size, 1), 1, (get_padding(kernel_size), 0)),
        ]

        # Final output layer
        self.output = weight_norm_conv2d(channels * 16, 1, (3, 1), 1, (1, 0))

    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        """Forward pass.

        Args:
            x: Audio waveform (batch, samples) or (batch, channels, samples)

        Returns:
            Tuple of (output, feature_maps) where:
                - output: Discriminator scores
                - feature_maps: List of intermediate features for feature matching
        """
        feature_maps = []

        # Handle input dimensions
        if x.ndim == 2:
            # (batch, samples) -> (batch, 1, samples)
            x = mx.expand_dims(x, axis=1)

        batch, channels, samples = x.shape

        # Pad to be divisible by period
        if samples % self.period != 0:
            pad_amount = self.period - (samples % self.period)
            x = mx.pad(x, [(0, 0), (0, 0), (0, pad_amount)])
            samples = x.shape[2]

        # Reshape: (batch, channels, samples) -> (batch, channels, samples/period, period)
        x = x.reshape(batch, channels, samples // self.period, self.period)
        # Transpose to (batch, channels, period, samples/period) for 2D conv
        # MLX conv2d expects (N, H, W, C), so we need to rearrange
        # From (batch, channels, samples/period, period) to (batch, samples/period, period, channels)
        x = mx.transpose(x, (0, 2, 3, 1))

        # Apply conv layers
        for conv in self.convs:
            x = conv(x)
            x = nn.leaky_relu(x, negative_slope=0.1)
            feature_maps.append(x)

        # Final output
        x = self.output(x)
        feature_maps.append(x)

        # Flatten for output
        x = x.reshape(batch, -1)

        return x, feature_maps


class ScaleDiscriminator(nn.Module):
    """Scale-based discriminator using 1D convolution.

    Applies strided 1D convolutions with grouped convolutions for efficiency.

    Args:
        channels: Base number of channels
        groups: Number of groups for grouped convolutions
        kernel_size: Kernel size for most convolutions
    """

    def __init__(
        self,
        channels: int = 128,
        groups: int = 4,
        kernel_size: int = 41,
    ):
        super().__init__()

        # MLX Conv1d expects (N, L, C) format
        # Progressive downsampling with increasing channels
        self.convs = [
            weight_norm_conv1d(1, channels, 15, 1, padding=7),
            weight_norm_conv1d(channels, channels, kernel_size, 2, padding=kernel_size // 2),
            weight_norm_conv1d(channels, channels * 2, kernel_size, 2, padding=kernel_size // 2),
            weight_norm_conv1d(channels * 2, channels * 4, kernel_size, 4, padding=kernel_size // 2),
            weight_norm_conv1d(channels * 4, channels * 8, kernel_size, 4, padding=kernel_size // 2),
            weight_norm_conv1d(channels * 8, channels * 8, kernel_size, 1, padding=kernel_size // 2),
            weight_norm_conv1d(channels * 8, channels * 8, 5, 1, padding=2),
        ]

        # Final output
        self.output = weight_norm_conv1d(channels * 8, 1, 3, 1, padding=1)

    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        """Forward pass.

        Args:
            x: Audio waveform (batch, samples) or (batch, samples, channels)

        Returns:
            Tuple of (output, feature_maps)
        """
        feature_maps = []

        # MLX Conv1d expects (N, L, C)
        if x.ndim == 2:
            # (batch, samples) -> (batch, samples, 1)
            x = mx.expand_dims(x, axis=-1)
        elif x.ndim == 3 and x.shape[1] == 1:
            # (batch, 1, samples) -> (batch, samples, 1)
            x = mx.transpose(x, (0, 2, 1))

        batch = x.shape[0]

        # Apply conv layers
        for conv in self.convs:
            x = conv(x)
            x = nn.leaky_relu(x, negative_slope=0.1)
            feature_maps.append(x)

        # Final output
        x = self.output(x)
        feature_maps.append(x)

        # Flatten for output
        x = x.reshape(batch, -1)

        return x, feature_maps


# ============================================================================
# Multi-Discriminator Ensembles
# ============================================================================


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator ensemble.

    Uses multiple period discriminators with different periods to capture
    various periodic structures in the audio.

    Args:
        periods: List of periods to use (default: [2, 3, 5, 7, 11])
        channels: Base channels per discriminator
    """

    def __init__(
        self,
        periods: Tuple[int, ...] = (2, 3, 5, 7, 11),
        channels: int = 32,
    ):
        super().__init__()
        self.periods = periods
        self.discriminators = [PeriodDiscriminator(period=p, channels=channels) for p in periods]

    def __call__(self, x: mx.array) -> Tuple[List[mx.array], List[List[mx.array]]]:
        """Forward pass through all period discriminators.

        Args:
            x: Audio waveform (batch, samples)

        Returns:
            Tuple of:
                - outputs: List of discriminator outputs
                - feature_maps: List of feature map lists (one per discriminator)
        """
        outputs = []
        all_features = []

        for disc in self.discriminators:
            out, features = disc(x)
            outputs.append(out)
            all_features.append(features)

        return outputs, all_features


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator ensemble.

    Uses multiple scale discriminators operating at different resolutions
    via average pooling.

    Args:
        num_scales: Number of scales (default: 3)
        pool_factor: Pooling factor between scales (default: 2)
        channels: Base channels per discriminator
    """

    def __init__(
        self,
        num_scales: int = 3,
        pool_factor: int = 2,
        channels: int = 128,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.pool_factor = pool_factor

        self.discriminators = [ScaleDiscriminator(channels=channels) for _ in range(num_scales)]

    def _pool(self, x: mx.array, factor: int) -> mx.array:
        """Average pooling for downsampling.

        Args:
            x: Input (batch, samples)
            factor: Pooling factor

        Returns:
            Downsampled input
        """
        if factor == 1:
            return x

        batch, samples = x.shape[:2]
        # Truncate to multiple of factor
        samples = (samples // factor) * factor
        x = x[:, :samples]
        # Reshape and average
        x = x.reshape(batch, samples // factor, factor)
        return mx.mean(x, axis=-1)

    def __call__(self, x: mx.array) -> Tuple[List[mx.array], List[List[mx.array]]]:
        """Forward pass through all scale discriminators.

        Args:
            x: Audio waveform (batch, samples)

        Returns:
            Tuple of:
                - outputs: List of discriminator outputs
                - feature_maps: List of feature map lists
        """
        outputs = []
        all_features = []

        for i, disc in enumerate(self.discriminators):
            # Downsample input for this scale
            x_scaled = self._pool(x, self.pool_factor**i)
            out, features = disc(x_scaled)
            outputs.append(out)
            all_features.append(features)

        return outputs, all_features


class CombinedDiscriminator(nn.Module):
    """Combined multi-period and multi-scale discriminator.

    Combines MPD and MSD for comprehensive audio analysis.
    This is the recommended discriminator for GAN training.

    Args:
        mpd_periods: Periods for MPD
        mpd_channels: Channels for MPD
        msd_scales: Number of scales for MSD
        msd_channels: Channels for MSD
    """

    def __init__(
        self,
        mpd_periods: Tuple[int, ...] = (2, 3, 5, 7, 11),
        mpd_channels: int = 32,
        msd_scales: int = 3,
        msd_channels: int = 128,
    ):
        super().__init__()

        self.mpd = MultiPeriodDiscriminator(
            periods=mpd_periods,
            channels=mpd_channels,
        )
        self.msd = MultiScaleDiscriminator(
            num_scales=msd_scales,
            channels=msd_channels,
        )

    def __call__(self, x: mx.array) -> Tuple[List[mx.array], List[List[mx.array]]]:
        """Forward pass through both MPD and MSD.

        Args:
            x: Audio waveform (batch, samples)

        Returns:
            Tuple of:
                - outputs: Combined list of all discriminator outputs
                - feature_maps: Combined list of all feature map lists
        """
        mpd_outs, mpd_features = self.mpd(x)
        msd_outs, msd_features = self.msd(x)

        outputs = mpd_outs + msd_outs
        features = mpd_features + msd_features

        return outputs, features


# ============================================================================
# Discriminator Variants
# ============================================================================


class SpectralDiscriminator(nn.Module):
    """Spectrogram-based discriminator.

    Operates on spectrograms instead of waveforms, useful for
    frequency-domain artifacts.

    Args:
        n_fft: FFT size
        hop_length: Hop length
        channels: Base channels
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        channels: int = 32,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 2D convolutions on spectrogram
        self.convs = [
            weight_norm_conv2d(1, channels, (3, 3), (1, 1), (1, 1)),
            weight_norm_conv2d(channels, channels * 2, (3, 3), (2, 2), (1, 1)),
            weight_norm_conv2d(channels * 2, channels * 4, (3, 3), (2, 2), (1, 1)),
            weight_norm_conv2d(channels * 4, channels * 8, (3, 3), (2, 2), (1, 1)),
            weight_norm_conv2d(channels * 8, channels * 8, (3, 3), (1, 1), (1, 1)),
        ]

        self.output = weight_norm_conv2d(channels * 8, 1, (3, 3), (1, 1), (1, 1))

    def _compute_spectrogram(self, x: mx.array) -> mx.array:
        """Compute magnitude spectrogram.

        Args:
            x: Waveform (batch, samples)

        Returns:
            Magnitude spectrogram (batch, time, freq, 1)
        """
        from .ops import stft

        real, imag = stft(x, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = mx.sqrt(real**2 + imag**2 + 1e-8)
        # (batch, freq, time) -> (batch, time, freq, 1)
        mag = mx.transpose(mag, (0, 2, 1))
        return mx.expand_dims(mag, axis=-1)

    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        """Forward pass.

        Args:
            x: Audio waveform (batch, samples)

        Returns:
            Tuple of (output, feature_maps)
        """
        feature_maps = []
        batch = x.shape[0]

        # Compute spectrogram
        x = self._compute_spectrogram(x)

        # Apply conv layers
        for conv in self.convs:
            x = conv(x)
            x = nn.leaky_relu(x, negative_slope=0.1)
            feature_maps.append(x)

        # Output
        x = self.output(x)
        feature_maps.append(x)

        # Flatten
        x = x.reshape(batch, -1)

        return x, feature_maps


# ============================================================================
# GAN Training Utilities
# ============================================================================


def compute_discriminator_loss(
    discriminator: nn.Module,
    real_audio: mx.array,
    fake_audio: mx.array,
) -> Tuple[mx.array, dict]:
    """Compute discriminator loss on real and fake audio.

    Args:
        discriminator: Discriminator module
        real_audio: Real audio samples (batch, samples)
        fake_audio: Generated audio samples (batch, samples)

    Returns:
        Tuple of (loss, loss_dict) where loss_dict contains breakdown
    """
    # Stop gradients on fake audio for discriminator training
    fake_audio = mx.stop_gradient(fake_audio)

    # Forward pass
    real_outs, _ = discriminator(real_audio)
    fake_outs, _ = discriminator(fake_audio)

    # Compute loss
    total_loss, real_loss, fake_loss = discriminator_loss(real_outs, fake_outs)

    return total_loss, {
        "disc_loss": float(total_loss),
        "disc_real": float(real_loss),
        "disc_fake": float(fake_loss),
    }


def compute_generator_loss(
    discriminator: nn.Module,
    real_audio: mx.array,
    fake_audio: mx.array,
    feature_match_factor: float = 2.0,
) -> Tuple[mx.array, dict]:
    """Compute generator loss including adversarial and feature matching.

    Args:
        discriminator: Discriminator module
        real_audio: Real audio samples (batch, samples)
        fake_audio: Generated audio samples (batch, samples)
        feature_match_factor: Weight for feature matching loss

    Returns:
        Tuple of (loss, loss_dict)
    """
    # Forward pass (with gradients on fake)
    real_outs, real_features = discriminator(real_audio)
    fake_outs, fake_features = discriminator(fake_audio)

    # Adversarial loss - generator wants discriminator to output high for fake
    adv_loss = generator_loss(fake_outs)

    # Feature matching loss
    fm_loss_fn = FeatureMatchingLoss(factor=feature_match_factor)
    fm_loss = fm_loss_fn(real_features, fake_features)

    total_loss = adv_loss + fm_loss

    return total_loss, {
        "gen_adv_loss": float(adv_loss),
        "gen_fm_loss": float(fm_loss),
        "gen_total_loss": float(total_loss),
    }


# ============================================================================
# Factory Functions
# ============================================================================


def create_discriminator(
    disc_type: Literal["combined", "mpd", "msd", "spectral"] = "combined",
    **kwargs,
) -> nn.Module:
    """Create discriminator from type string.

    Args:
        disc_type: Type of discriminator
        **kwargs: Arguments passed to discriminator constructor

    Returns:
        Discriminator module
    """
    discriminators = {
        "combined": CombinedDiscriminator,
        "mpd": MultiPeriodDiscriminator,
        "msd": MultiScaleDiscriminator,
        "spectral": SpectralDiscriminator,
    }

    if disc_type not in discriminators:
        raise ValueError(f"Unknown discriminator type: {disc_type}. " f"Available: {list(discriminators.keys())}")

    return discriminators[disc_type](**kwargs)
