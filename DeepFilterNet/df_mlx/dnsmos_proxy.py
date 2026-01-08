"""Differentiable DNSMOS Proxy for perceptual loss in MLX DeepFilterNet4.

Implements a neural network that approximates DNSMOS scores (SIG, BAK, OVL)
in a differentiable manner for use as a training loss.

Based on df/dnsmos_proxy.py but adapted for MLX on Apple Silicon.
"""

from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def hz_to_mel(hz: mx.array) -> mx.array:
    """Convert Hz to mel scale."""
    return 2595 * mx.log10(1 + hz / 700)


def mel_to_hz(mel: mx.array) -> mx.array:
    """Convert mel scale to Hz."""
    return 700 * (mx.power(10.0, mel / 2595) - 1)


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> mx.array:
    """Create mel filterbank matrix.

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        n_mels: Number of mel channels
        f_min: Minimum frequency
        f_max: Maximum frequency (defaults to sample_rate / 2)

    Returns:
        Mel filterbank matrix [n_mels, n_freqs]
    """
    if f_max is None:
        f_max = sample_rate / 2

    # Compute mel points
    mel_min = float(hz_to_mel(mx.array(f_min)))
    mel_max = float(hz_to_mel(mx.array(f_max)))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    # Convert to FFT bins
    n_freqs = n_fft // 2 + 1
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create filterbank
    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        for j in range(left, center):
            if j < n_freqs:
                filterbank[i, j] = (j - left) / max(center - left, 1)
        # Falling slope
        for j in range(center, right):
            if j < n_freqs:
                filterbank[i, j] = (right - j) / max(right - center, 1)

    return mx.array(filterbank.astype(np.float32))


class MelSpectrogram(nn.Module):
    """Mel spectrogram feature extractor.

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT
        n_mels: Number of mel filterbank channels
        f_min: Minimum frequency for mel filterbank
        f_max: Maximum frequency for mel filterbank
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_max = f_max or sample_rate / 2
        self.f_min = f_min

        # Pre-compute mel filterbank
        self._mel_fb = create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, self.f_max)

        # Hann window
        self._window = mx.array(np.hanning(n_fft + 1)[:-1].astype(np.float32))

    def __call__(self, audio: mx.array) -> mx.array:
        """Extract mel spectrogram features.

        Args:
            audio: Audio waveform [B, T] or [B, 1, T]

        Returns:
            mel_spec: Mel spectrogram [B, n_mels, T']
        """
        # Handle 3D input
        if audio.ndim == 3:
            audio = mx.squeeze(audio, axis=1)

        batch_size = audio.shape[0]
        n_samples = audio.shape[1]

        # STFT using strided view
        n_frames = (n_samples - self.n_fft) // self.hop_length + 1

        mel_specs = []
        for b in range(batch_size):
            frames = []
            for i in range(n_frames):
                start = i * self.hop_length
                frame = audio[b, start : start + self.n_fft] * self._window
                frames.append(frame)

            if not frames:
                # Handle edge case of very short audio
                mel_specs.append(mx.zeros((self.n_mels, 1)))
                continue

            frames = mx.stack(frames, axis=0)  # [T', n_fft]

            # FFT
            spec_complex = mx.fft.rfft(frames)  # [T', n_freqs]

            # Power spectrogram
            power = mx.abs(spec_complex) ** 2  # [T', n_freqs]

            # Apply mel filterbank: [n_mels, n_freqs] @ [T', n_freqs].T -> [n_mels, T']
            mel_spec = mx.matmul(self._mel_fb, mx.transpose(power))

            # Log scale with floor to avoid log(0)
            mel_spec = mx.log(mx.maximum(mel_spec, 1e-10))

            mel_specs.append(mel_spec)

        return mx.stack(mel_specs, axis=0)  # [B, n_mels, T']


class ConvBlock(nn.Module):
    """Convolutional block for DNSMOS proxy."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.bn(self.conv(x)))


class DNSMOSProxy(nn.Module):
    """Differentiable proxy model for DNSMOS prediction.

    Predicts SIG (signal quality), BAK (background quality), and OVL (overall quality)
    scores in a differentiable manner for use as a perceptual training loss.

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size for mel spectrogram
        hop_length: Hop length for STFT
        n_mels: Number of mel channels
        hidden_dim: Hidden dimension for conv layers
        fc_dim: Fully connected layer dimension
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 64,
        hidden_dim: int = 64,
        fc_dim: int = 256,
    ):
        super().__init__()

        self.mel_extractor = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # Conv encoder
        self.enc_block1 = ConvBlock(1, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.enc_block2 = ConvBlock(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)
        self.enc_block3 = ConvBlock(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1)
        self.enc_block4 = ConvBlock(hidden_dim * 4, hidden_dim * 4, kernel_size=3, stride=2, padding=1)

        # Score predictors (separate heads for each metric)
        self.sig_fc1 = nn.Linear(hidden_dim * 4, fc_dim)
        self.sig_fc2 = nn.Linear(fc_dim, 1)

        self.bak_fc1 = nn.Linear(hidden_dim * 4, fc_dim)
        self.bak_fc2 = nn.Linear(fc_dim, 1)

        self.ovl_fc1 = nn.Linear(hidden_dim * 4, fc_dim)
        self.ovl_fc2 = nn.Linear(fc_dim, 1)

        self.dropout = nn.Dropout(0.2)

    def __call__(self, audio: mx.array) -> Dict[str, mx.array]:
        """Predict DNSMOS scores for audio.

        Args:
            audio: Audio waveform [B, T] or [B, 1, T]

        Returns:
            Dictionary with 'sig', 'bak', 'ovl' score tensors [B]
        """
        # Extract mel spectrogram
        mel_spec = self.mel_extractor(audio)  # [B, n_mels, T']

        # MLX Conv2d expects NHWC format: [B, H, W, C]
        # Transpose from [B, n_mels, T'] to [B, n_mels, T', 1]
        mel_spec = mx.transpose(mel_spec, (0, 1, 2))  # [B, n_mels, T']
        mel_spec = mx.expand_dims(mel_spec, axis=-1)  # [B, n_mels, T', 1]

        # Encode
        x = self.enc_block1(mel_spec)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        x = self.enc_block4(x)

        # Global pooling over spatial dimensions (H, W), keep batch and channels
        # x is [B, H, W, C] in MLX NHWC format
        features = mx.mean(x, axis=(1, 2))  # [B, C]

        # Predict scores (clamp to valid DNSMOS range [1, 5])
        sig = self.sig_fc2(self.dropout(nn.relu(self.sig_fc1(features))))
        sig = mx.clip(mx.squeeze(sig, axis=-1), 1.0, 5.0)

        bak = self.bak_fc2(self.dropout(nn.relu(self.bak_fc1(features))))
        bak = mx.clip(mx.squeeze(bak, axis=-1), 1.0, 5.0)

        ovl = self.ovl_fc2(self.dropout(nn.relu(self.ovl_fc1(features))))
        ovl = mx.clip(mx.squeeze(ovl, axis=-1), 1.0, 5.0)

        return {
            "sig": sig,
            "bak": bak,
            "ovl": ovl,
        }

    def compute_loss(
        self,
        enhanced_audio: mx.array,
        target_sig: Optional[mx.array] = None,
        target_bak: Optional[mx.array] = None,
        target_ovl: Optional[mx.array] = None,
    ) -> mx.array:
        """Compute DNSMOS proxy loss.

        If targets are provided, computes MSE loss to targets.
        Otherwise, returns negative mean OVL (maximize quality).

        Args:
            enhanced_audio: Enhanced audio [B, T]
            target_sig: Target SIG scores [B] (optional)
            target_bak: Target BAK scores [B] (optional)
            target_ovl: Target OVL scores [B] (optional)

        Returns:
            Loss tensor (scalar)
        """
        scores = self(enhanced_audio)

        if target_ovl is not None:
            # Regression loss to target scores
            loss = mx.mean((scores["ovl"] - target_ovl) ** 2)
            if target_sig is not None:
                loss = loss + mx.mean((scores["sig"] - target_sig) ** 2)
            if target_bak is not None:
                loss = loss + mx.mean((scores["bak"] - target_bak) ** 2)
            return loss
        else:
            # Maximize quality (minimize negative score)
            # Weight OVL highest, then SIG, then BAK
            weighted_score = 0.5 * scores["ovl"] + 0.3 * scores["sig"] + 0.2 * scores["bak"]
            return -mx.mean(weighted_score)


class DNSMOSLoss(nn.Module):
    """DNSMOS-based perceptual loss for training.

    Uses a pre-trained DNSMOSProxy to provide perceptual feedback
    during speech enhancement training.

    Args:
        proxy_path: Path to pre-trained proxy weights (optional)
        freeze_proxy: Whether to freeze proxy weights during training
        target_ovl: Target OVL score to aim for (default 4.5)
        loss_weight: Weight for this loss term
    """

    def __init__(
        self,
        proxy_path: Optional[str] = None,
        freeze_proxy: bool = True,
        target_ovl: float = 4.5,
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.proxy = DNSMOSProxy()
        self.target_ovl = target_ovl
        self.loss_weight = loss_weight
        self.freeze_proxy = freeze_proxy

        # Load pre-trained weights if provided
        if proxy_path is not None:
            weights: Dict[str, mx.array] = mx.load(proxy_path)  # type: ignore[assignment]
            self.proxy.load_weights(list(weights.items()))

        # Freeze proxy if requested
        if freeze_proxy:
            self.proxy.freeze()

    def __call__(
        self,
        enhanced_audio: mx.array,
        clean_audio: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Compute DNSMOS loss.

        Args:
            enhanced_audio: Enhanced audio [B, T]
            clean_audio: Clean reference audio [B, T] (optional, for computing targets)

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary with predicted scores for logging
        """
        # Get scores for enhanced audio
        enhanced_scores = self.proxy(enhanced_audio)

        if clean_audio is not None:
            # Get scores for clean audio as reference
            # Note: In MLX, we don't have explicit no_grad, but frozen proxy doesn't update
            clean_scores = self.proxy(clean_audio)

            # Loss: try to match clean audio scores
            loss = (
                mx.mean((enhanced_scores["ovl"] - clean_scores["ovl"]) ** 2)
                + 0.5 * mx.mean((enhanced_scores["sig"] - clean_scores["sig"]) ** 2)
                + 0.5 * mx.mean((enhanced_scores["bak"] - clean_scores["bak"]) ** 2)
            )
        else:
            # Loss: maximize OVL score toward target
            loss = mx.mean(mx.maximum(self.target_ovl - enhanced_scores["ovl"], 0.0))

        return self.loss_weight * loss, enhanced_scores


class LightweightDNSMOSProxy(nn.Module):
    """Lightweight DNSMOS proxy for faster inference.

    Uses depthwise separable convolutions and smaller hidden dimensions.

    Args:
        sample_rate: Audio sample rate
        n_mels: Number of mel channels
        hidden_dim: Base hidden dimension
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.mel_extractor = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
        )

        # Encoder with standard convolutions (MLX doesn't have grouped conv yet)
        self.conv1 = nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm(hidden_dim * 2)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm(hidden_dim * 4)

        # Single head for all scores
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, 3)  # SIG, BAK, OVL

    def __call__(self, audio: mx.array) -> Dict[str, mx.array]:
        """Predict DNSMOS scores.

        Args:
            audio: Audio waveform [B, T]

        Returns:
            Dictionary with 'sig', 'bak', 'ovl' scores
        """
        mel_spec = self.mel_extractor(audio)  # [B, n_mels, T']
        # MLX expects NHWC: [B, H, W, C]
        mel_spec = mx.expand_dims(mel_spec, axis=-1)  # [B, n_mels, T', 1]

        x = nn.relu(self.bn1(self.conv1(mel_spec)))
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.relu(self.bn3(self.conv3(x)))

        # Global average pool over spatial dims
        features = mx.mean(x, axis=(1, 2))  # [B, C]
        scores = self.fc2(nn.relu(self.fc1(features)))

        return {
            "sig": mx.clip(scores[:, 0], 1.0, 5.0),
            "bak": mx.clip(scores[:, 1], 1.0, 5.0),
            "ovl": mx.clip(scores[:, 2], 1.0, 5.0),
        }


def create_dnsmos_proxy(
    lightweight: bool = False,
    sample_rate: int = 16000,
    pretrained_path: Optional[str] = None,
) -> "DNSMOSProxy | LightweightDNSMOSProxy":
    """Factory function to create DNSMOS proxy.

    Args:
        lightweight: Use lightweight model for faster inference
        sample_rate: Audio sample rate
        pretrained_path: Path to pretrained weights

    Returns:
        DNSMOSProxy or LightweightDNSMOSProxy instance
    """
    if lightweight:
        model: DNSMOSProxy | LightweightDNSMOSProxy = LightweightDNSMOSProxy(sample_rate=sample_rate)
    else:
        model = DNSMOSProxy(sample_rate=sample_rate)

    if pretrained_path is not None:
        weights: Dict[str, mx.array] = mx.load(pretrained_path)  # type: ignore[assignment]
        model.load_weights(list(weights.items()))

    return model
