"""Hybrid time-frequency encoder for DeepFilterNet4.

This module implements a multi-domain encoder that processes:
- Time-domain waveform features
- Magnitude/ERB spectrum features
- Phase spectrum features

And fuses them using cross-domain attention for improved speech enhancement.

Reference: MH-SENet (ISCA Interspeech 2025)
"""

from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Final

from df.modules import Conv2dNormAct, GroupedLinearEinsum


class WaveformEncoder(nn.Module):
    """Time-domain waveform encoder.

    Processes raw audio waveform with strided convolutions to extract
    time-domain features aligned with STFT frames.

    Args:
        in_channels: Input channels (1 for mono)
        base_channels: Base channel count (doubled each layer)
        num_layers: Number of conv layers
        kernel_sizes: Kernel size per layer
        strides: Stride per layer (should match STFT hop alignment)
        out_dim: Output feature dimension
    """

    in_channels: Final[int]
    out_dim: Final[int]
    total_stride: Final[int]

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_layers: int = 4,
        kernel_sizes: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        out_dim: int = 256,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [7, 5, 5, 3]
        if strides is None:
            strides = [4, 2, 2, 2]  # Total stride = 32, typical for 16kHz audio with 512 hop

        assert len(kernel_sizes) == num_layers
        assert len(strides) == num_layers

        self.in_channels = in_channels
        self.out_dim = out_dim

        layers = []
        ch = in_channels
        for i in range(num_layers):
            out_ch = base_channels * (2**i)
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        ch, out_ch, kernel_sizes[i], strides[i], padding=kernel_sizes[i] // 2
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                )
            )
            ch = out_ch

        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Linear(ch, out_dim)

        # Total stride for alignment calculation
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s

    def forward(self, waveform: Tensor) -> Tensor:
        """Encode waveform to features.

        Args:
            waveform: [B, 1, T_samples] or [B, T_samples]

        Returns:
            features: [B, T_frames, out_dim]
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        x = self.encoder(waveform)  # [B, C, T_frames]
        x = x.transpose(1, 2)  # [B, T_frames, C]
        return self.proj(x)  # [B, T_frames, out_dim]


class PhaseEncoder(nn.Module):
    """Phase spectrum encoder.

    Processes phase information using cos/sin representation.

    Args:
        n_freqs: Number of frequency bins
        conv_ch: Convolutional channels
        out_dim: Output feature dimension
        num_layers: Number of conv layers
    """

    n_freqs: Final[int]
    out_dim: Final[int]

    def __init__(
        self,
        n_freqs: int,
        conv_ch: int = 32,
        out_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.n_freqs = n_freqs
        self.out_dim = out_dim

        # Phase has 2 channels: cos(phase), sin(phase)
        # First layer preserves frequency dimension
        layers = [Conv2dNormAct(2, conv_ch, kernel_size=(3, 3), bias=False, separable=True)]

        # Track dimensions for building layers
        current_ch = conv_ch
        # First conv with kernel_size=3 preserves size due to same-padding
        current_freqs = n_freqs

        # Add downsampling layers with channel growth
        for i in range(num_layers - 1):
            fstride = 2 if current_freqs > 16 else 1
            out_ch = min(current_ch * 2, conv_ch * 4)  # Cap channel growth
            layers.append(
                Conv2dNormAct(
                    current_ch,
                    out_ch,
                    kernel_size=(1, 3),
                    fstride=fstride,
                    bias=False,
                    separable=True,
                )
            )
            current_ch = out_ch
            if fstride > 1:
                # Calculate output size after strided conv
                # Conv with kernel=3, stride=2, padding=1 gives: (n+1)//2
                current_freqs = (current_freqs + 1) // 2

        self.conv_layers = nn.Sequential(*layers)

        # Store final dimensions
        self.final_ch = current_ch
        self.final_freqs = current_freqs

        # Output projection
        self.out_proj = nn.Linear(current_ch * current_freqs, out_dim)

    def forward(self, phase: Tensor) -> Tensor:
        """Encode phase to features.

        Args:
            phase: Phase angle [B, T, F] or [B, 1, T, F] or complex [B, 1, T, F, 2]

        Returns:
            features: [B, T, out_dim]
        """
        # Handle different input formats
        if phase.dim() == 5:
            # Complex format [B, 1, T, F, 2]
            phase_angle = torch.atan2(phase[..., 1], phase[..., 0]).squeeze(1)
        elif phase.dim() == 4:
            # [B, 1, T, F]
            phase_angle = phase.squeeze(1)
        else:
            # [B, T, F]
            phase_angle = phase

        # Convert to cos/sin representation [B, 2, T, F]
        phase_repr = torch.stack([torch.cos(phase_angle), torch.sin(phase_angle)], dim=1)

        x = self.conv_layers(phase_repr)  # [B, C, T, F']
        x = x.permute(0, 2, 1, 3).flatten(2)  # [B, T, C*F']
        return self.out_proj(x)  # [B, T, out_dim]


class CrossDomainAttention(nn.Module):
    """Cross-domain attention for feature fusion.

    Fuses features from different domains (time, magnitude, phase) using
    cross-attention mechanisms.

    Args:
        time_dim: Time-domain feature dimension
        mag_dim: Magnitude feature dimension
        phase_dim: Phase feature dimension
        out_dim: Output fused dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    out_dim: Final[int]

    def __init__(
        self,
        time_dim: int,
        mag_dim: int,
        phase_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_dim = out_dim

        # Project all domains to same dimension
        self.time_proj = nn.Linear(time_dim, out_dim)
        self.mag_proj = nn.Linear(mag_dim, out_dim)
        self.phase_proj = nn.Linear(phase_dim, out_dim)

        # Cross-attention layers
        self.time_mag_attn = nn.MultiheadAttention(
            out_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mag_phase_attn = nn.MultiheadAttention(
            out_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer norms for attention outputs
        self.norm_tm = nn.LayerNorm(out_dim)
        self.norm_mp = nn.LayerNorm(out_dim)

        # Final fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )

        self.norm_out = nn.LayerNorm(out_dim)

    def forward(
        self,
        time_feat: Tensor,
        mag_feat: Tensor,
        phase_feat: Tensor,
    ) -> Tensor:
        """Fuse features from different domains.

        Args:
            time_feat: [B, T, time_dim]
            mag_feat: [B, T, mag_dim]
            phase_feat: [B, T, phase_dim]

        Returns:
            fused: [B, T, out_dim]
        """
        # Project to common dimension
        t = self.time_proj(time_feat)
        m = self.mag_proj(mag_feat)
        p = self.phase_proj(phase_feat)

        # Cross-attention: time attends to magnitude
        t_m, _ = self.time_mag_attn(t, m, m)
        t = self.norm_tm(t + t_m)

        # Cross-attention: magnitude attends to phase
        m_p, _ = self.mag_phase_attn(m, p, p)
        m = self.norm_mp(m + m_p)

        # Concatenate and fuse
        fused = torch.cat([t, m, p], dim=-1)
        return self.norm_out(self.fusion(fused))


class SimpleCrossAttention(nn.Module):
    """Simplified cross-domain attention without full multi-head attention.

    More efficient version using linear attention approximation.

    Args:
        input_dims: List of input dimensions for each domain
        out_dim: Output fused dimension
        temperature: Softmax temperature for attention weights
    """

    def __init__(
        self,
        input_dims: List[int],
        out_dim: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_domains = len(input_dims)
        self.out_dim = out_dim
        self.temperature = temperature

        # Project each domain
        self.projections = nn.ModuleList([nn.Linear(dim, out_dim) for dim in input_dims])

        # Learnable domain weights
        self.domain_weights = nn.Parameter(torch.ones(self.n_domains) / self.n_domains)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, *features: Tensor) -> Tensor:
        """Fuse features with weighted combination.

        Args:
            features: Variable number of feature tensors [B, T, D_i]

        Returns:
            fused: [B, T, out_dim]
        """
        assert len(features) == self.n_domains

        # Project all features
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]

        # Weighted combination
        weights = F.softmax(self.domain_weights / self.temperature, dim=0)
        fused = sum(w * p for w, p in zip(weights, projected))

        return self.out_proj(fused)


class MagnitudeEncoder(nn.Module):
    """Magnitude/ERB spectrum encoder (similar to DFNet3 encoder).

    This encoder processes the ERB-scale magnitude features and complex
    spectrogram to produce embeddings for the decoder.

    Args:
        conv_ch: Convolutional channels
        nb_erb: Number of ERB bands
        nb_df: Number of DF frequency bins
        emb_hidden_dim: Embedding hidden dimension
        enc_lin_groups: Linear groups for efficiency
        conv_kernel_inp: Initial conv kernel size
        conv_kernel: Main conv kernel size
    """

    def __init__(
        self,
        conv_ch: int = 16,
        nb_erb: int = 32,
        nb_df: int = 96,
        emb_hidden_dim: int = 256,
        enc_lin_groups: int = 16,
        conv_kernel_inp: Tuple[int, int] = (3, 3),
        conv_kernel: Tuple[int, int] = (1, 3),
    ):
        super().__init__()

        assert nb_erb % 4 == 0, "nb_erb should be divisible by 4"

        self.nb_erb = nb_erb
        self.nb_df = nb_df

        # ERB pathway
        self.erb_conv0 = Conv2dNormAct(
            1, conv_ch, kernel_size=conv_kernel_inp, bias=False, separable=True
        )
        conv_layer = partial(
            Conv2dNormAct,
            in_ch=conv_ch,
            out_ch=conv_ch,
            kernel_size=conv_kernel,
            bias=False,
            separable=True,
        )
        self.erb_conv1 = conv_layer(fstride=2)
        self.erb_conv2 = conv_layer(fstride=2)
        self.erb_conv3 = conv_layer(fstride=1)

        # DF pathway
        self.df_conv0 = Conv2dNormAct(
            2, conv_ch, kernel_size=conv_kernel_inp, bias=False, separable=True
        )
        self.df_conv1 = conv_layer(fstride=2)

        # Embedding dimensions
        self.emb_in_dim = conv_ch * nb_erb // 4
        self.emb_out_dim = emb_hidden_dim

        # DF to embedding projection
        df_fc_emb = GroupedLinearEinsum(
            conv_ch * nb_df // 2, self.emb_in_dim, groups=enc_lin_groups
        )
        self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU(inplace=True))

    def forward(
        self, feat_erb: Tensor, feat_spec: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode magnitude features.

        Args:
            feat_erb: ERB features [B, 1, T, E]
            feat_spec: Complex spectrogram [B, 2, T, F]

        Returns:
            e0, e1, e2, e3: Encoder intermediate outputs for skip connections
            emb: Embedding [B, T, emb_out_dim]
            c0: DF pathway features for DF decoder
        """
        # ERB pathway
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, E]
        e1 = self.erb_conv1(e0)  # [B, C, T, E/2]
        e2 = self.erb_conv2(e1)  # [B, C, T, E/4]
        e3 = self.erb_conv3(e2)  # [B, C, T, E/4]

        # DF pathway
        c0 = self.df_conv0(feat_spec)  # [B, C, T, F]
        c1 = self.df_conv1(c0)  # [B, C, T, F/2]

        # Combine pathways for embedding
        cemb = c1.permute(0, 2, 3, 1).flatten(2)  # [B, T, C*F/2]
        cemb = self.df_fc_emb(cemb)  # [B, T, emb_in_dim]

        emb = e3.permute(0, 2, 3, 1).flatten(2)  # [B, T, C*E/4]
        emb = emb + cemb  # Add DF pathway contribution

        return e0, e1, e2, e3, emb, c0


class HybridEncoder(nn.Module):
    """Hybrid time-frequency encoder for DFNet4.

    Parallel processing of time-domain, magnitude, and phase information
    with cross-domain attention fusion and Mamba sequence modeling.

    Args:
        conv_ch: Convolutional channels
        nb_erb: Number of ERB bands
        nb_df: Number of DF frequency bins
        fft_size: FFT size for phase encoder
        emb_hidden_dim: Embedding hidden dimension
        emb_num_layers: Number of Mamba/sequence layers
        enc_lin_groups: Linear groups for efficiency
        use_time_branch: Whether to use time-domain branch
        use_phase_branch: Whether to use phase branch
        use_mamba: Whether to use Mamba (True) or GRU (False)
        lsnr_min: Minimum LSNR value
        lsnr_max: Maximum LSNR value
    """

    def __init__(
        self,
        conv_ch: int = 16,
        nb_erb: int = 32,
        nb_df: int = 96,
        fft_size: int = 960,
        emb_hidden_dim: int = 256,
        emb_num_layers: int = 2,
        enc_lin_groups: int = 16,
        use_time_branch: bool = True,
        use_phase_branch: bool = True,
        use_mamba: bool = True,
        lsnr_min: float = -15.0,
        lsnr_max: float = 40.0,
        linear_groups: int = 1,
    ):
        super().__init__()

        self.use_time_branch = use_time_branch
        self.use_phase_branch = use_phase_branch
        self.use_mamba = use_mamba

        # Magnitude encoder (similar to DFNet3)
        self.mag_encoder = MagnitudeEncoder(
            conv_ch=conv_ch,
            nb_erb=nb_erb,
            nb_df=nb_df,
            emb_hidden_dim=emb_hidden_dim,
            enc_lin_groups=enc_lin_groups,
        )

        # Time-domain encoder (optional)
        if use_time_branch:
            self.time_encoder = WaveformEncoder(
                out_dim=emb_hidden_dim,
            )
        else:
            self.time_encoder = None

        # Phase encoder (optional)
        if use_phase_branch:
            # Use nb_df since feat_spec has [B, 2, T, nb_df] shape
            self.phase_encoder = PhaseEncoder(
                n_freqs=nb_df,
                out_dim=emb_hidden_dim,
            )
        else:
            self.phase_encoder = None

        # Cross-domain fusion
        mag_emb_dim = conv_ch * nb_erb // 4
        self.mag_emb_dim = mag_emb_dim
        self.emb_hidden_dim = emb_hidden_dim

        # Projection for mag_emb when used as fallback for time/phase features
        if use_time_branch:
            self.time_fallback_proj = nn.Linear(mag_emb_dim, emb_hidden_dim)
        else:
            self.time_fallback_proj = None
        if use_phase_branch:
            self.phase_fallback_proj = nn.Linear(mag_emb_dim, emb_hidden_dim)
        else:
            self.phase_fallback_proj = None

        self.fusion = CrossDomainAttention(
            time_dim=emb_hidden_dim if use_time_branch else mag_emb_dim,
            mag_dim=mag_emb_dim,
            phase_dim=emb_hidden_dim if use_phase_branch else mag_emb_dim,
            out_dim=emb_hidden_dim,
        )

        # Sequence modeling
        if use_mamba:
            from df.mamba import Mamba

            self.seq_layers = nn.ModuleList([Mamba(emb_hidden_dim) for _ in range(emb_num_layers)])
        else:
            # Fallback to GRU
            from df.modules import SqueezedGRU_S

            self.seq_layers = nn.ModuleList(
                [
                    SqueezedGRU_S(
                        emb_hidden_dim,
                        emb_hidden_dim,
                        num_layers=1,
                        linear_groups=linear_groups,
                    )
                    for _ in range(emb_num_layers)
                ]
            )

        # Output projection to match expected dimension
        self.out_proj = nn.Linear(emb_hidden_dim, conv_ch * nb_erb // 4)

        # LSNR estimation
        self.lsnr_fc = nn.Sequential(
            nn.Linear(conv_ch * nb_erb // 4, 1),
            nn.Sigmoid(),
        )
        self.lsnr_scale = lsnr_max - lsnr_min
        self.lsnr_offset = lsnr_min

    def forward(
        self,
        feat_erb: Tensor,
        feat_spec: Tensor,
        waveform: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode features from multiple domains.

        Args:
            feat_erb: ERB features [B, 1, T, E]
            feat_spec: Complex spectrogram [B, 2, T, F]
            waveform: Raw audio [B, T_samples] (optional)

        Returns:
            e0, e1, e2, e3: Encoder intermediate outputs
            emb: Final embedding [B, T, H]
            c0: DF pathway features
            lsnr: Local SNR estimate [B, T, 1]
        """
        # Magnitude pathway
        e0, e1, e2, e3, mag_emb, c0 = self.mag_encoder(feat_erb, feat_spec)

        # Time-domain features
        if self.time_encoder is not None and waveform is not None:
            time_feat = self.time_encoder(waveform)
            # Align time dimension with mag_emb if needed
            if time_feat.size(1) != mag_emb.size(1):
                time_feat = F.interpolate(
                    time_feat.transpose(1, 2),
                    size=mag_emb.size(1),
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
        elif self.time_fallback_proj is not None:
            # Project mag_emb to match expected time_dim
            time_feat = self.time_fallback_proj(mag_emb)
        else:
            time_feat = mag_emb

        # Phase features
        if self.phase_encoder is not None:
            # feat_spec is [B, 2, T, F] where dim 1 is (real, imag)
            # Convert to phase angle [B, T, F]
            phase_angle = torch.atan2(feat_spec[:, 1], feat_spec[:, 0])  # [B, T, F]
            phase_feat = self.phase_encoder(phase_angle)
            # Align time dimension if needed
            if phase_feat.size(1) != mag_emb.size(1):
                phase_feat = F.interpolate(
                    phase_feat.transpose(1, 2),
                    size=mag_emb.size(1),
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
        elif self.phase_fallback_proj is not None:
            # Project mag_emb to match expected phase_dim
            phase_feat = self.phase_fallback_proj(mag_emb)
        else:
            phase_feat = mag_emb

        # Cross-domain fusion
        emb = self.fusion(time_feat, mag_emb, phase_feat)

        # Sequence modeling
        for layer in self.seq_layers:
            if self.use_mamba:
                emb = layer(emb)
            else:
                emb, _ = layer(emb)

        # Project to expected output dimension
        emb = self.out_proj(emb)

        # LSNR estimation
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset

        return e0, e1, e2, e3, emb, c0, lsnr


class LightweightHybridEncoder(nn.Module):
    """Lightweight hybrid encoder with simplified fusion.

    Uses simple weighted combination instead of full cross-attention
    for efficiency in real-time applications.
    """

    def __init__(
        self,
        conv_ch: int = 16,
        nb_erb: int = 32,
        nb_df: int = 96,
        fft_size: int = 960,
        emb_hidden_dim: int = 256,
        emb_num_layers: int = 2,
        enc_lin_groups: int = 16,
        use_time_branch: bool = False,  # Disabled by default for efficiency
        use_phase_branch: bool = True,
        lsnr_min: float = -15.0,
        lsnr_max: float = 40.0,
    ):
        super().__init__()

        self.use_time_branch = use_time_branch
        self.use_phase_branch = use_phase_branch

        # Magnitude encoder
        self.mag_encoder = MagnitudeEncoder(
            conv_ch=conv_ch,
            nb_erb=nb_erb,
            nb_df=nb_df,
            emb_hidden_dim=emb_hidden_dim,
            enc_lin_groups=enc_lin_groups,
        )

        mag_emb_dim = conv_ch * nb_erb // 4
        input_dims = [mag_emb_dim]

        # Optional phase encoder
        if use_phase_branch:
            # Use nb_df since feat_spec has [B, 2, T, nb_df] shape
            self.phase_encoder = PhaseEncoder(
                n_freqs=nb_df,
                out_dim=mag_emb_dim,  # Match mag dimension
            )
            input_dims.append(mag_emb_dim)
        else:
            self.phase_encoder = None

        # Optional time encoder
        if use_time_branch:
            self.time_encoder = WaveformEncoder(out_dim=mag_emb_dim)
            input_dims.append(mag_emb_dim)
        else:
            self.time_encoder = None

        # Simple fusion
        self.fusion = SimpleCrossAttention(input_dims, mag_emb_dim)

        # Sequence modeling with Mamba
        from df.mamba import Mamba

        self.seq_layers = nn.ModuleList([Mamba(mag_emb_dim) for _ in range(emb_num_layers)])

        # LSNR estimation
        self.lsnr_fc = nn.Sequential(
            nn.Linear(mag_emb_dim, 1),
            nn.Sigmoid(),
        )
        self.lsnr_scale = lsnr_max - lsnr_min
        self.lsnr_offset = lsnr_min

    def forward(
        self,
        feat_erb: Tensor,
        feat_spec: Tensor,
        waveform: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode features with lightweight fusion."""
        # Magnitude pathway
        e0, e1, e2, e3, mag_emb, c0 = self.mag_encoder(feat_erb, feat_spec)

        features = [mag_emb]

        # Phase features
        if self.phase_encoder is not None:
            # feat_spec is [B, 2, T, F] where dim 1 is (real, imag)
            phase_angle = torch.atan2(feat_spec[:, 1], feat_spec[:, 0])
            phase_feat = self.phase_encoder(phase_angle)
            if phase_feat.size(1) != mag_emb.size(1):
                phase_feat = F.interpolate(
                    phase_feat.transpose(1, 2),
                    size=mag_emb.size(1),
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            features.append(phase_feat)

        # Time features
        if self.time_encoder is not None and waveform is not None:
            time_feat = self.time_encoder(waveform)
            if time_feat.size(1) != mag_emb.size(1):
                time_feat = F.interpolate(
                    time_feat.transpose(1, 2),
                    size=mag_emb.size(1),
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            features.append(time_feat)

        # Fusion
        emb = self.fusion(*features)

        # Sequence modeling
        for layer in self.seq_layers:
            emb = layer(emb)

        # LSNR estimation
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset

        return e0, e1, e2, e3, emb, c0, lsnr
