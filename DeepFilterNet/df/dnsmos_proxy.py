"""Differentiable DNSMOS Proxy for perceptual loss in DeepFilterNet4.

Implements a neural network that approximates DNSMOS scores (SIG, BAK, OVL)
in a differentiable manner for use as a training loss.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
        mel_fb = self._create_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)
        
        # Window for STFT
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)
        
    def _create_mel_filterbank(self) -> Tensor:
        """Create mel filterbank matrix."""
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * torch.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        # Compute mel points
        mel_min = hz_to_mel(torch.tensor(self.f_min))
        mel_max = hz_to_mel(torch.tensor(self.f_max))
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to FFT bins
        n_freqs = self.n_fft // 2 + 1
        bin_points = torch.floor((self.n_fft + 1) * hz_points / self.sample_rate).long()
        
        # Create filterbank
        filterbank = torch.zeros(self.n_mels, n_freqs)
        for i in range(self.n_mels):
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
                    
        return filterbank
    
    def forward(self, audio: Tensor) -> Tensor:
        """Extract mel spectrogram features.
        
        Args:
            audio: Audio waveform [B, T] or [B, 1, T]
            
        Returns:
            mel_spec: Mel spectrogram [B, n_mels, T']
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
            
        # STFT
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        
        # Power spectrogram
        power = spec.abs() ** 2  # [B, F, T]
        
        # Apply mel filterbank
        mel_spec = torch.matmul(self.mel_fb, power)  # [B, n_mels, T]
        
        # Log scale with floor to avoid log(0)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))
        
        return mel_spec


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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


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
        self.encoder = nn.Sequential(
            ConvBlock(1, hidden_dim, kernel_size=3, stride=2, padding=1),
            ConvBlock(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            ConvBlock(hidden_dim * 4, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Score predictors (separate heads for each metric)
        self.sig_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, fc_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_dim, 1),
        )
        
        self.bak_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, fc_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_dim, 1),
        )
        
        self.ovl_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, fc_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_dim, 1),
        )
        
    def forward(self, audio: Tensor) -> Dict[str, Tensor]:
        """Predict DNSMOS scores for audio.
        
        Args:
            audio: Audio waveform [B, T] or [B, 1, T]
            
        Returns:
            Dictionary with 'sig', 'bak', 'ovl' score tensors [B]
        """
        # Extract mel spectrogram
        mel_spec = self.mel_extractor(audio)  # [B, n_mels, T']
        
        # Add channel dimension
        mel_spec = mel_spec.unsqueeze(1)  # [B, 1, n_mels, T']
        
        # Encode
        features = self.encoder(mel_spec)  # [B, C, H, W]
        features = self.pool(features).flatten(1)  # [B, C]
        
        # Predict scores (clamp to valid DNSMOS range [1, 5])
        sig = torch.clamp(self.sig_head(features).squeeze(-1), 1.0, 5.0)
        bak = torch.clamp(self.bak_head(features).squeeze(-1), 1.0, 5.0)
        ovl = torch.clamp(self.ovl_head(features).squeeze(-1), 1.0, 5.0)
        
        return {
            "sig": sig,
            "bak": bak,
            "ovl": ovl,
        }
    
    def compute_loss(
        self,
        enhanced_audio: Tensor,
        target_sig: Optional[Tensor] = None,
        target_bak: Optional[Tensor] = None,
        target_ovl: Optional[Tensor] = None,
    ) -> Tensor:
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
        scores = self.forward(enhanced_audio)
        
        if target_ovl is not None:
            # Regression loss to target scores
            loss = F.mse_loss(scores["ovl"], target_ovl)
            if target_sig is not None:
                loss = loss + F.mse_loss(scores["sig"], target_sig)
            if target_bak is not None:
                loss = loss + F.mse_loss(scores["bak"], target_bak)
            return loss
        else:
            # Maximize quality (minimize negative score)
            # Weight OVL highest, then SIG, then BAK
            return -(0.5 * scores["ovl"] + 0.3 * scores["sig"] + 0.2 * scores["bak"]).mean()


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
        
        # Load pre-trained weights if provided
        if proxy_path is not None:
            state_dict = torch.load(proxy_path, map_location="cpu")
            self.proxy.load_state_dict(state_dict)
            
        # Freeze proxy if requested
        if freeze_proxy:
            for param in self.proxy.parameters():
                param.requires_grad = False
                
    def forward(
        self,
        enhanced_audio: Tensor,
        clean_audio: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
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
            with torch.no_grad():
                clean_scores = self.proxy(clean_audio)
            
            # Loss: try to match clean audio scores
            loss = (
                F.mse_loss(enhanced_scores["ovl"], clean_scores["ovl"]) +
                0.5 * F.mse_loss(enhanced_scores["sig"], clean_scores["sig"]) +
                0.5 * F.mse_loss(enhanced_scores["bak"], clean_scores["bak"])
            )
        else:
            # Loss: maximize OVL score toward target
            loss = F.relu(self.target_ovl - enhanced_scores["ovl"]).mean()
            
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
        
        # Depthwise separable conv encoder
        self.encoder = nn.Sequential(
            # Depthwise
            nn.Conv2d(1, 1, 3, stride=2, padding=1, groups=1),
            nn.Conv2d(1, hidden_dim, 1),  # Pointwise
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, stride=2, padding=1, groups=hidden_dim * 2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Single head for OVL (main metric)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 3),  # SIG, BAK, OVL
        )
        
    def forward(self, audio: Tensor) -> Dict[str, Tensor]:
        """Predict DNSMOS scores.
        
        Args:
            audio: Audio waveform [B, T]
            
        Returns:
            Dictionary with 'sig', 'bak', 'ovl' scores
        """
        mel_spec = self.mel_extractor(audio).unsqueeze(1)
        features = self.pool(self.encoder(mel_spec)).flatten(1)
        scores = self.head(features)
        
        return {
            "sig": torch.clamp(scores[:, 0], 1.0, 5.0),
            "bak": torch.clamp(scores[:, 1], 1.0, 5.0),
            "ovl": torch.clamp(scores[:, 2], 1.0, 5.0),
        }
