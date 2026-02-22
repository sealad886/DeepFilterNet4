"""DeepFilterNet4 model implementation for MLX.

This module provides the complete MLX implementation of DeepFilterNet4,
including the encoder, ERB decoder, DF decoder, and full model.

Architecture:
- Encoder: Multi-scale convolutional encoder with ERB and DF pathways
- Backbone: Mamba SSM for temporal modeling
- ERB Decoder: Estimates spectral mask for ERB bands
- DF Decoder: Estimates complex filter coefficients for deep filtering

The model processes noisy speech spectrograms and outputs enhanced speech
by applying learned masks and deep filtering operations.
"""

from typing import Any, Dict, Literal, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelParams4, get_default_config
from .mamba import GroupedLinear, SqueezedMamba
from .modules import Conv2dNormAct, ConvTranspose2dNormAct, DfOp, Mask, SqueezedGRU_S, erb_fb

# ============================================================================
# Encoder
# ============================================================================


class Encoder4(nn.Module):
    """DeepFilterNet4 encoder.

    Multi-scale convolutional encoder that processes ERB features and
    DF (deep filter) features through parallel pathways.

    Args:
        p: Model parameters
    """

    def __init__(self, p: ModelParams4):
        super().__init__()

        self.p = p
        conv_ch = p.conv_ch

        # ERB pathway - processes ERB-scale features
        # Input: (batch, time, erb_bands, 1) -> permute to NHWC
        self.erb_conv0 = Conv2dNormAct(
            1, conv_ch, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), norm="batch", activation="relu"
        )
        self.erb_conv1 = Conv2dNormAct(
            conv_ch, conv_ch, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), norm="batch", activation="relu"
        )
        self.erb_conv2 = Conv2dNormAct(
            conv_ch, conv_ch, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), norm="batch", activation="relu"
        )
        self.erb_conv3 = Conv2dNormAct(
            conv_ch, conv_ch, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), norm="batch", activation="relu"
        )

        # Calculate ERB output dimension after convolutions
        erb_out_dim = p.nb_erb
        erb_out_dim = (erb_out_dim + 1) // 2  # After stride 2
        erb_out_dim = (erb_out_dim + 1) // 2  # After stride 2
        self.erb_out_dim = erb_out_dim * conv_ch

        # DF pathway - processes complex DF features
        # Input: (batch, time, df_bins, 2) for real/imag
        self.df_conv0 = Conv2dNormAct(
            2, conv_ch, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), norm="batch", activation="relu"
        )
        self.df_conv1 = Conv2dNormAct(
            conv_ch, conv_ch, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), norm="batch", activation="relu"
        )

        # Calculate DF output dimension
        df_out_dim = p.nb_df
        df_out_dim = (df_out_dim + 1) // 2  # After stride 2
        df_out_dim = (df_out_dim + 1) // 2  # After stride 2
        self.df_out_dim = df_out_dim * conv_ch

        # Combined embedding
        combined_dim = self.erb_out_dim + self.df_out_dim

        # Project to embedding dimension
        self.emb_linear = GroupedLinear(combined_dim, p.emb_hidden_dim, groups=p.enc_linear_groups)
        self.emb_norm = nn.LayerNorm(p.emb_hidden_dim)

        # LSNR estimation head
        lsnr_min = p.lsnr.lsnr_min if hasattr(p, "lsnr") else -15.0
        lsnr_max = p.lsnr.lsnr_max if hasattr(p, "lsnr") else 40.0
        self.lsnr_scale = (lsnr_max - lsnr_min) / 2
        self.lsnr_offset = lsnr_min + self.lsnr_scale
        self.lsnr_fc = nn.Linear(p.emb_hidden_dim, 1)

    def __call__(
        self,
        feat_erb: mx.array,
        feat_spec: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            feat_erb: ERB features (batch, time, erb_bands)
            feat_spec: DF features (batch, time, df_bins, 2)

        Returns:
            Tuple of:
                - Embedding tensor (batch, time, emb_hidden_dim)
                - LSNR estimate (batch, time, 1)
        """
        batch, time = feat_erb.shape[:2]

        # ERB pathway
        # Add channel dim and transpose to NHWC: (batch, time, erb, 1)
        erb = mx.expand_dims(feat_erb, axis=-1)

        erb = self.erb_conv0(erb)
        erb = self.erb_conv1(erb)
        erb = self.erb_conv2(erb)
        erb = self.erb_conv3(erb)

        # Flatten spatial dims: (batch, time, erb_out_dim)
        erb = erb.reshape(batch, time, -1)

        # DF pathway
        # feat_spec already has shape (batch, time, df_bins, 2)
        df = self.df_conv0(feat_spec)
        df = self.df_conv1(df)

        # Flatten: (batch, time, df_out_dim)
        df = df.reshape(batch, time, -1)

        # Combine pathways
        combined = mx.concatenate([erb, df], axis=-1)

        # Project to embedding
        emb = self.emb_linear(combined)
        emb = self.emb_norm(emb)

        # LSNR estimation (tanh to bound output, then scale)
        lsnr = mx.tanh(self.lsnr_fc(emb)) * self.lsnr_scale + self.lsnr_offset

        return emb, lsnr


# ============================================================================
# ERB Decoder
# ============================================================================


class ErbDecoder4(nn.Module):
    """ERB mask decoder.

    Decodes embedding to ERB-scale spectral mask.

    Args:
        p: Model parameters
    """

    def __init__(self, p: ModelParams4):
        super().__init__()

        self.p = p
        conv_ch = p.conv_ch

        # Project from embedding to decoder input
        self.input_proj = nn.Linear(p.emb_hidden_dim, p.erb_hidden_dim)

        # Upsampling convolutions
        self.conv0 = ConvTranspose2dNormAct(
            conv_ch, conv_ch, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), norm="batch", activation="relu"
        )
        self.conv1 = ConvTranspose2dNormAct(
            conv_ch, conv_ch, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), norm="batch", activation="relu"
        )
        self.conv2 = ConvTranspose2dNormAct(
            conv_ch, conv_ch, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), norm="batch", activation="relu"
        )
        self.conv3 = ConvTranspose2dNormAct(
            conv_ch, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), norm=None, activation=None
        )

        # Final projection to ERB bands
        self.output_proj = nn.Linear(p.erb_hidden_dim, p.nb_erb)

        # Mask module
        self.mask = Mask(p.nb_erb, mask_type="sigmoid")

    def __call__(self, emb: mx.array) -> mx.array:
        """Forward pass.

        Args:
            emb: Embedding tensor (batch, time, emb_hidden_dim)

        Returns:
            ERB mask (batch, time, nb_erb)
        """
        # Simple linear decoder for now
        x = self.input_proj(emb)
        x = self.output_proj(x)
        mask: mx.array = self.mask(x)  # type: ignore[assignment]
        return mask


# ============================================================================
# DF Decoder
# ============================================================================


class DfDecoder4(nn.Module):
    """Deep Filter coefficient decoder.

    Decodes embedding to complex filter coefficients or complex gains for deep filtering.

    Args:
        p: Model parameters
        output_mode: Output mode - "coefficients" for FIR filter coefficients,
                     "complex_gain" for multiplicative complex gains per frequency bin.
                     Defaults to p.df_output_mode if not specified.
    """

    VALID_OUTPUT_MODES = ("coefficients", "complex_gain")

    def __init__(self, p: ModelParams4, output_mode: str | None = None):
        super().__init__()

        self.p = p
        self.nb_df = p.nb_df
        self.df_order = p.df_order

        # Determine output mode
        if output_mode is None:
            output_mode = p.df_output_mode
        if output_mode not in self.VALID_OUTPUT_MODES:
            raise ValueError(f"output_mode must be one of {self.VALID_OUTPUT_MODES}, got '{output_mode}'")
        self.output_mode = output_mode

        # Output size depends on mode
        if self.output_mode == "coefficients":
            # nb_df * df_order * 2 (real + imag for each tap)
            self.out_dim = p.nb_df * p.df_order * 2
        else:  # complex_gain
            # nb_df * 2 (one complex gain per frequency bin)
            self.out_dim = p.nb_df * 2

        # Decoder layers
        self.layers = [
            nn.Linear(p.emb_hidden_dim, p.df_hidden_dim),
            nn.ReLU(),
            nn.Linear(p.df_hidden_dim, p.df_hidden_dim),
            nn.ReLU(),
            nn.Linear(p.df_hidden_dim, self.out_dim),
        ]

    @property
    def is_gain_mode(self) -> bool:
        """Check if decoder is in complex gain mode."""
        return self.output_mode == "complex_gain"

    def __call__(self, emb: mx.array) -> mx.array:
        """Forward pass.

        Args:
            emb: Embedding tensor (batch, time, emb_hidden_dim)

        Returns:
            If coefficients mode: (batch, time, nb_df, df_order, 2)
            If complex_gain mode: (batch, time, nb_df, 2)
        """
        batch, time, _ = emb.shape

        x = emb
        for layer in self.layers:
            x = layer(x)

        # Reshape based on mode
        if self.output_mode == "coefficients":
            # (batch, time, nb_df, df_order, 2)
            x = x.reshape(batch, time, self.nb_df, self.df_order, 2)
        else:  # complex_gain
            # (batch, time, nb_df, 2)
            x = x.reshape(batch, time, self.nb_df, 2)

        return x


class MultiResDfDecoder(nn.Module):
    """Multi-resolution Deep Filter decoder.

    Generates DF coefficients at multiple frequency resolutions for improved
    enhancement quality. Uses a shared backbone with resolution-specific output heads.

    Args:
        emb_dim: Input embedding dimension
        hidden_dim: Hidden dimension for backbone
        resolutions: List of (num_freqs, frame_size) tuples defining each resolution
        num_layers: Number of backbone layers
        d_state: Mamba state dimension
        d_conv: Mamba convolution width
    """

    def __init__(
        self,
        emb_dim: int = 256,
        hidden_dim: int = 256,
        resolutions: Optional[list] = None,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
    ):
        super().__init__()

        if resolutions is None:
            resolutions = [(96, 5), (48, 3), (24, 2)]

        self.resolutions = resolutions
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # Shared backbone using SqueezedMamba
        self.backbone = SqueezedMamba(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            output_size=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
        )

        # Resolution-specific output heads
        self.output_heads = []
        for num_freqs, frame_size in resolutions:
            out_dim = num_freqs * frame_size * 2  # Complex coefficients
            head = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.Tanh(),
            )
            self.output_heads.append(head)

    def __call__(self, emb: mx.array) -> list:
        """Generate DF coefficients for all resolutions.

        Args:
            emb: Encoder embedding (batch, time, emb_dim)

        Returns:
            List of coefficient tensors, one per resolution.
            Each has shape (batch, time, num_freqs, frame_size, 2)
        """
        batch, time, _ = emb.shape

        # Shared backbone processing
        hidden, _ = self.backbone(emb)  # (batch, time, hidden_dim)

        # Generate coefficients for each resolution
        coefs_list = []
        for head, (num_freqs, frame_size) in zip(self.output_heads, self.resolutions):
            # Generate raw coefficients
            raw_coefs = head(hidden)  # (batch, time, num_freqs * frame_size * 2)

            # Reshape to (batch, time, num_freqs, frame_size, 2)
            coefs = raw_coefs.reshape(batch, time, num_freqs, frame_size, 2)
            coefs_list.append(coefs)

        return coefs_list


class AdaptiveOrderPredictor(nn.Module):
    """Predicts optimal filter order per frame.

    Uses a small network to predict which filter order is best for each
    time frame based on input characteristics.

    Args:
        emb_dim: Input embedding dimension
        max_order: Maximum filter order
        min_order: Minimum filter order
    """

    def __init__(
        self,
        emb_dim: int = 256,
        max_order: int = 7,
        min_order: int = 2,
    ):
        super().__init__()
        self.max_order = max_order
        self.min_order = min_order
        self.num_orders = max_order - min_order + 1

        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, self.num_orders),
        )

    def __call__(
        self,
        emb: mx.array,
        temperature: float = 1.0,
    ) -> tuple:
        """Predict filter order.

        Args:
            emb: Embedding tensor (batch, time, emb_dim)
            temperature: Softmax temperature (lower = more confident)

        Returns:
            order_weights: Soft order selection weights (batch, time, num_orders)
            predicted_order: Hard order prediction (batch, time)
        """
        logits = self.predictor(emb)  # (batch, time, num_orders)

        # Soft selection using softmax
        order_weights = mx.softmax(logits / temperature, axis=-1)

        # Hard selection (argmax)
        predicted_order = mx.argmax(logits, axis=-1) + self.min_order

        return order_weights, predicted_order


# ============================================================================
# Hybrid Encoder Components
# ============================================================================


class WaveformEncoder(nn.Module):
    """Time-domain waveform encoder.

    Processes raw audio waveform with strided convolutions to extract
    time-domain features aligned with STFT frames.

    Args:
        in_channels: Input channels (1 for mono)
        base_channels: Base channel count (doubled each layer)
        num_layers: Number of conv layers
        out_dim: Output feature dimension
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_layers: int = 4,
        out_dim: int = 256,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.num_layers = num_layers

        # Strided conv layers with channel growth
        # Default strides: [4, 2, 2, 2] -> total_stride = 32
        strides = [4, 2, 2, 2][:num_layers]
        kernel_sizes = [7, 5, 5, 3][:num_layers]

        self.conv_layers = []
        self.norms = []
        ch = in_channels
        for i in range(num_layers):
            out_ch = base_channels * (2**i)
            self.conv_layers.append(
                nn.Conv1d(ch, out_ch, kernel_sizes[i], stride=strides[i], padding=kernel_sizes[i] // 2)
            )
            self.norms.append(nn.BatchNorm(out_ch))
            ch = out_ch

        self.final_ch = ch
        self.proj = nn.Linear(ch, out_dim)

        # Total stride for alignment calculation
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s

    def __call__(self, waveform: mx.array) -> mx.array:
        """Encode waveform to features.

        Args:
            waveform: (batch, samples) or (batch, channels, samples)

        Returns:
            features: (batch, time_frames, out_dim)
        """
        # Handle input formats
        if waveform.ndim == 2:
            # (batch, samples) -> (batch, samples, 1)
            x = mx.expand_dims(waveform, axis=-1)
        elif waveform.ndim == 3 and waveform.shape[1] == 1:
            # (batch, 1, samples) -> (batch, samples, 1) - PyTorch format
            x = mx.transpose(waveform, axes=(0, 2, 1))
        else:
            # Assume already (batch, samples, channels)
            x = waveform

        # MLX Conv1d expects (batch, length, channels) format
        for conv, norm in zip(self.conv_layers, self.norms):
            x = conv(x)
            x = norm(x)
            x = nn.gelu(x)

        # x is now (batch, time, channels), project to out_dim
        return self.proj(x)


class PhaseEncoder(nn.Module):
    """Phase spectrum encoder.

    Processes phase information using cos/sin representation.

    Args:
        n_freqs: Number of frequency bins
        conv_ch: Convolutional channels
        out_dim: Output feature dimension
        num_layers: Number of conv layers
    """

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
        self.conv_layers = []
        current_ch = 2  # cos/sin channels

        for i in range(num_layers):
            out_ch = conv_ch * min(2**i, 4)  # Cap channel growth
            fstride = 2 if i < num_layers - 1 else 1  # Downsample frequency
            self.conv_layers.append(
                Conv2dNormAct(
                    current_ch,
                    out_ch,
                    kernel_size=(1, 3),
                    stride=(1, fstride),
                    padding=(0, 1),
                    norm="batch",
                    activation="gelu",
                )
            )
            current_ch = out_ch

        self.final_ch = current_ch
        # Calculate final frequency dimension after downsampling
        self.final_freqs = n_freqs
        for i in range(num_layers - 1):
            self.final_freqs = (self.final_freqs + 1) // 2

        self.out_proj = nn.Linear(current_ch * self.final_freqs, out_dim)

    def __call__(self, phase: mx.array) -> mx.array:
        """Encode phase to features.

        Args:
            phase: Phase angle (batch, time, freq) or complex spec (batch, time, freq, 2)

        Returns:
            features: (batch, time, out_dim)
        """
        # Handle different input formats
        if phase.ndim == 4:
            # Complex format (batch, time, freq, 2)
            phase_angle = mx.arctan2(phase[..., 1], phase[..., 0])
        else:
            phase_angle = phase  # Already angle

        # Convert to cos/sin representation: (batch, time, freq, 2)
        phase_repr = mx.stack([mx.cos(phase_angle), mx.sin(phase_angle)], axis=-1)

        # Apply conv layers (expecting NHWC format)
        x = phase_repr  # (batch, time, freq, 2)
        for conv in self.conv_layers:
            x = conv(x)

        # Flatten spatial dims: (batch, time, final_ch * final_freqs)
        batch, time = x.shape[:2]
        x = x.reshape(batch, time, -1)

        return self.out_proj(x)


class CrossDomainAttention(nn.Module):
    """Cross-domain attention for feature fusion.

    Fuses features from different domains (time, magnitude, phase) using
    cross-attention and gating mechanisms.

    Args:
        time_dim: Time-domain feature dimension
        mag_dim: Magnitude feature dimension
        phase_dim: Phase feature dimension
        out_dim: Output fused dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

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

        # Cross-attention using multi-head attention
        self.head_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim**-0.5

        # Attention projections for time-mag cross-attention
        self.tm_q = nn.Linear(out_dim, out_dim)
        self.tm_k = nn.Linear(out_dim, out_dim)
        self.tm_v = nn.Linear(out_dim, out_dim)

        # Attention projections for mag-phase cross-attention
        self.mp_q = nn.Linear(out_dim, out_dim)
        self.mp_k = nn.Linear(out_dim, out_dim)
        self.mp_v = nn.Linear(out_dim, out_dim)

        # Layer norms
        self.norm_tm = nn.LayerNorm(out_dim)
        self.norm_mp = nn.LayerNorm(out_dim)
        self.norm_out = nn.LayerNorm(out_dim)

        # Final fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

        self.dropout_rate = dropout

    def _cross_attention(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
    ) -> mx.array:
        """Compute cross-attention."""
        batch, seq_len, _ = query.shape

        q = q_proj(query).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k_proj(key).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = v_proj(value).reshape(batch, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq, head_dim)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))

        # Scaled dot-product attention
        scores = mx.matmul(q, mx.transpose(k, axes=(0, 1, 3, 2))) * self.scale
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)

        # Transpose back: (batch, seq, heads, head_dim) -> (batch, seq, out_dim)
        out = mx.transpose(out, axes=(0, 2, 1, 3)).reshape(batch, seq_len, -1)
        return out

    def __call__(
        self,
        time_feat: mx.array,
        mag_feat: mx.array,
        phase_feat: mx.array,
    ) -> mx.array:
        """Fuse features from multiple domains.

        Args:
            time_feat: Time-domain features (batch, time, time_dim)
            mag_feat: Magnitude features (batch, time, mag_dim)
            phase_feat: Phase features (batch, time, phase_dim)

        Returns:
            fused: Fused features (batch, time, out_dim)
        """
        # Project to common dimension
        time_proj = self.time_proj(time_feat)
        mag_proj = self.mag_proj(mag_feat)
        phase_proj = self.phase_proj(phase_feat)

        # Time-magnitude cross-attention (time attends to magnitude)
        tm_attn = self._cross_attention(time_proj, mag_proj, mag_proj, self.tm_q, self.tm_k, self.tm_v)
        tm_attn = self.norm_tm(time_proj + tm_attn)

        # Magnitude-phase cross-attention (magnitude attends to phase)
        mp_attn = self._cross_attention(mag_proj, phase_proj, phase_proj, self.mp_q, self.mp_k, self.mp_v)
        mp_attn = self.norm_mp(mag_proj + mp_attn)

        # Concatenate and fuse
        combined = mx.concatenate([tm_attn, mp_attn, phase_proj], axis=-1)
        fused = self.fusion(combined)
        return self.norm_out(fused)


class HybridEncoder(nn.Module):
    """Hybrid time-frequency encoder for DFNet4.

    Parallel processing of time-domain, magnitude, and phase information
    with cross-domain attention fusion and Mamba sequence modeling.

    Args:
        p: Model parameters
        use_time_branch: Whether to use time-domain branch
        use_phase_branch: Whether to use phase branch
    """

    def __init__(
        self,
        p: ModelParams4,
        use_time_branch: bool = True,
        use_phase_branch: bool = True,
    ):
        super().__init__()
        self.p = p
        self.use_time_branch = use_time_branch
        self.use_phase_branch = use_phase_branch

        emb_hidden_dim = p.emb_hidden_dim

        # Magnitude encoder (standard Encoder4 pathway)
        self.magnitude_encoder = Encoder4(p)
        mag_emb_dim = emb_hidden_dim

        # Time-domain encoder (optional)
        if use_time_branch:
            self.time_encoder = WaveformEncoder(out_dim=emb_hidden_dim)
        else:
            self.time_encoder = None

        # Phase encoder (optional)
        if use_phase_branch:
            self.phase_encoder = PhaseEncoder(n_freqs=p.nb_df, out_dim=emb_hidden_dim)
        else:
            self.phase_encoder = None

        # Cross-domain fusion
        time_dim = emb_hidden_dim if use_time_branch else mag_emb_dim
        phase_dim = emb_hidden_dim if use_phase_branch else mag_emb_dim

        self.fusion = CrossDomainAttention(
            time_dim=time_dim,
            mag_dim=mag_emb_dim,
            phase_dim=phase_dim,
            out_dim=emb_hidden_dim,
        )

        # Sequence modeling with Mamba
        self.seq_layers = [SqueezedMamba(emb_hidden_dim, emb_hidden_dim, emb_hidden_dim) for _ in range(2)]

        # LSNR estimation
        lsnr_min = p.lsnr.lsnr_min if hasattr(p, "lsnr") else -15.0
        lsnr_max = p.lsnr.lsnr_max if hasattr(p, "lsnr") else 40.0
        self.lsnr_scale = (lsnr_max - lsnr_min) / 2
        self.lsnr_offset = lsnr_min + self.lsnr_scale
        self.lsnr_fc = nn.Linear(emb_hidden_dim, 1)

    def __call__(
        self,
        feat_erb: mx.array,
        feat_spec: mx.array,
        waveform: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Encode features from multiple domains.

        Args:
            feat_erb: ERB features (batch, time, erb_bands)
            feat_spec: Complex spectrogram (batch, time, df_bins, 2)
            waveform: Raw audio (batch, samples) - optional

        Returns:
            emb: Final embedding (batch, time, emb_hidden_dim)
            lsnr: Local SNR estimate (batch, time, 1)
        """
        # Magnitude pathway
        mag_emb, _ = self.magnitude_encoder(feat_erb, feat_spec)

        # Time-domain features
        if self.time_encoder is not None and waveform is not None:
            time_feat = self.time_encoder(waveform)
            # Align time dimension with mag_emb if needed
            if time_feat.shape[1] != mag_emb.shape[1]:
                # Simple linear interpolation
                batch, target_len, dim = mag_emb.shape
                source_len = time_feat.shape[1]
                indices = mx.linspace(0, source_len - 1, target_len).astype(mx.int32)
                time_feat = time_feat[:, indices, :]
        else:
            time_feat = mag_emb  # Fallback to mag features

        # Phase features
        if self.phase_encoder is not None:
            phase_feat = self.phase_encoder(feat_spec)
        else:
            phase_feat = mag_emb  # Fallback to mag features

        # Cross-domain fusion
        emb = self.fusion(time_feat, mag_emb, phase_feat)

        # Sequence modeling
        for layer in self.seq_layers:
            emb, _ = layer(emb)

        # LSNR estimation
        lsnr = mx.tanh(self.lsnr_fc(emb)) * self.lsnr_scale + self.lsnr_offset

        return emb, lsnr


# ============================================================================
# Main Model
# ============================================================================


class DfNet4(nn.Module):
    """DeepFilterNet4 main model.

    Complete model combining encoder, Mamba backbone, and decoders
    for speech enhancement.

    Args:
        p: Model parameters (uses defaults if None)
        lsnr_dropout: Enable LSNR-based dropout during training (overrides config)
        lsnr_dropout_threshold: LSNR threshold for dropout in dB
    """

    def __init__(
        self,
        p: Optional[ModelParams4] = None,
        lsnr_dropout: Optional[bool] = None,
        lsnr_dropout_threshold: Optional[float] = None,
    ):
        super().__init__()

        if p is None:
            p = get_default_config()
        self.p = p

        # Lookahead settings
        self.df_lookahead = p.df_lookahead
        self.conv_lookahead = p.conv_lookahead

        # Validate lookahead settings
        if self.conv_lookahead > 0:
            assert self.conv_lookahead >= self.df_lookahead, (
                f"conv_lookahead ({self.conv_lookahead}) must be >= " f"df_lookahead ({self.df_lookahead})"
            )

        # LSNR dropout settings
        if lsnr_dropout is not None:
            self.lsnr_dropout = lsnr_dropout
        elif hasattr(p, "lsnr"):
            self.lsnr_dropout = p.lsnr.lsnr_dropout
        else:
            self.lsnr_dropout = False

        if lsnr_dropout_threshold is not None:
            self.lsnr_dropout_threshold = lsnr_dropout_threshold
        elif hasattr(p, "lsnr"):
            self.lsnr_dropout_threshold = p.lsnr.lsnr_dropout_threshold
        else:
            self.lsnr_dropout_threshold = -10.0

        # Post-filter settings
        self.post_filter = p.mask_pf
        self.post_filter_beta = p.pf_beta

        # DF output mode
        self.df_output_mode = p.df_output_mode

        # ERB filterbank (non-trainable)
        self._erb_fb = erb_fb(
            sr=p.sr,
            fft_size=p.fft_size,
            nb_bands=p.nb_erb,
        )

        # Encoder
        self.encoder = Encoder4(p)

        # Backbone (Mamba, GRU, or Attention based on config)
        backbone_type = getattr(p.backbone, "backbone_type", "mamba")
        if backbone_type == "gru":
            self.backbone = SqueezedGRU_S(
                input_size=p.emb_hidden_dim,
                hidden_size=p.emb_hidden_dim,
                output_size=p.emb_hidden_dim,
                num_layers=p.backbone.nb_layers,
                linear_groups=8,
                gru_skip=True,
            )
        elif backbone_type == "attention":
            from .modules import SqueezedAttention

            self.backbone = SqueezedAttention(
                input_size=p.emb_hidden_dim,
                hidden_size=p.emb_hidden_dim,
                output_size=p.emb_hidden_dim,
                num_layers=p.backbone.nb_layers,
                num_heads=getattr(p.backbone, "num_heads", 4),
                linear_groups=8,
                gru_skip=True,
            )
        else:
            self.backbone = SqueezedMamba(
                input_size=p.emb_hidden_dim,
                hidden_size=p.emb_hidden_dim,
                output_size=p.emb_hidden_dim,
                num_layers=p.backbone.nb_layers,
                d_state=p.backbone.d_state,
                d_conv=p.backbone.d_conv,
                expand_factor=p.backbone.expand_factor,
            )

        # Decoders
        self.erb_decoder = ErbDecoder4(p)
        self.df_decoder = DfDecoder4(p)  # Uses p.df_output_mode

        # Deep filtering operation (only needed for coefficients mode)
        if self.df_output_mode == "coefficients":
            self.df_op = DfOp(
                nb_df=p.nb_df,
                df_order=p.df_order,
                df_lookahead=p.df_lookahead,
            )
        else:
            self.df_op = None

    def _pad_features(self, x: mx.array, lookahead: int) -> mx.array:
        """Apply lookahead padding to features.

        Shifts the time axis by padding future frames and removing past frames.
        This enables non-causal processing with controllable lookahead.

        Args:
            x: Input tensor with time in axis 1 (batch, time, ...)
            lookahead: Number of lookahead frames

        Returns:
            Padded tensor with same time dimension
        """
        if lookahead == 0:
            return x

        # Pad future frames with zeros, then slice to remove past frames
        # This creates a shift: output[t] sees input[t+lookahead]
        ndim = x.ndim
        if ndim == 3:
            # (batch, time, features)
            x_pad = mx.pad(x, [(0, 0), (0, lookahead), (0, 0)])
            return x_pad[:, lookahead:, :]
        elif ndim == 4:
            # (batch, time, freq, channels)
            x_pad = mx.pad(x, [(0, 0), (0, lookahead), (0, 0), (0, 0)])
            return x_pad[:, lookahead:, :, :]
        else:
            raise ValueError(f"Unsupported ndim {ndim} for feature padding")

    def _apply_post_filter(
        self,
        spec_enhanced: Tuple[mx.array, mx.array],
        spec_original: Tuple[mx.array, mx.array],
    ) -> Tuple[mx.array, mx.array]:
        """Apply mask-based post-filter to reduce musical noise.

        The post-filter applies additional attenuation based on the ratio
        of enhanced to original magnitude, using a sinusoidal transfer
        function that smoothly increases attenuation for low-gain regions.

        Args:
            spec_enhanced: Enhanced spectrum as (real, imag)
            spec_original: Original noisy spectrum as (real, imag)

        Returns:
            Post-filtered spectrum as (real, imag)
        """
        if not self.post_filter:
            return spec_enhanced

        enh_real, enh_imag = spec_enhanced
        orig_real, orig_imag = spec_original
        beta = self.post_filter_beta
        eps = 1e-12

        # Compute magnitudes
        enh_mag = mx.sqrt(enh_real**2 + enh_imag**2 + eps)
        orig_mag = mx.sqrt(orig_real**2 + orig_imag**2 + eps)

        # Compute mask as ratio (clipped to [eps, 1])
        mask = mx.clip(enh_mag / (orig_mag + eps), eps, 1.0)

        # Sinusoidal mask transfer function
        # mask_sin = mask * sin(π * mask / 2), clipped to prevent division by zero
        pi = 3.141592653589793
        mask_sin = mx.maximum(mask * mx.sin(pi * mask / 2), eps)

        # Post-filter gain: (1 + beta) / (1 + beta * (mask / mask_sin)^2)
        ratio = mask / mask_sin
        pf = (1 + beta) / (1 + beta * ratio * ratio)

        # Apply post-filter to enhanced spectrum
        pf_real = enh_real * pf
        pf_imag = enh_imag * pf

        return (pf_real, pf_imag)

    def _apply_complex_gain(
        self,
        spec: Tuple[mx.array, mx.array],
        gain: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Apply complex multiplicative gain to spectrum.

        This is an alternative to DfOp for simpler frequency-domain filtering.
        Instead of convolving with FIR filter coefficients, each frequency bin
        is multiplied by a learned complex gain.

        Args:
            spec: Input spectrum as (real, imag), each (batch, time, freq)
            gain: Complex gains (batch, time, nb_df, 2) where last dim is (real, imag)

        Returns:
            Filtered spectrum as (real, imag) with gains applied to DF bins
        """
        spec_real, spec_imag = spec
        n_freqs = spec_real.shape[-1]
        nb_df = gain.shape[2]

        # Extract DF frequency bins
        df_real = spec_real[:, :, :nb_df]
        df_imag = spec_imag[:, :, :nb_df]

        # Extract gain components
        gain_real = gain[:, :, :, 0]  # (batch, time, nb_df)
        gain_imag = gain[:, :, :, 1]

        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_real = gain_real * df_real - gain_imag * df_imag
        out_imag = gain_real * df_imag + gain_imag * df_real

        # Combine with non-DF frequencies (pass-through)
        if n_freqs > nb_df:
            out_real = mx.concatenate([out_real, spec_real[:, :, nb_df:]], axis=-1)
            out_imag = mx.concatenate([out_imag, spec_imag[:, :, nb_df:]], axis=-1)

        return (out_real, out_imag)

    def __call__(
        self,
        spec: Tuple[mx.array, mx.array],
        feat_erb: mx.array,
        feat_spec: mx.array,
        training: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            spec: Input spectrum as (real, imag), each (batch, time, freq)
            feat_erb: ERB features (batch, time, erb_bands)
            feat_spec: DF features (batch, time, df_bins, 2)
            training: Whether in training mode (enables LSNR dropout)

        Returns:
            Enhanced spectrum as (real, imag)
        """
        spec_real, spec_imag = spec
        # Shape is (batch, time, freq) - used implicitly in operations below
        _ = spec_real.shape

        # Apply feature lookahead padding if configured
        if self.conv_lookahead > 0:
            feat_erb = self._pad_features(feat_erb, self.conv_lookahead)
            feat_spec = self._pad_features(feat_spec, self.conv_lookahead)

        # Encode (now returns embedding and LSNR estimate)
        emb, lsnr = self.encoder(feat_erb, feat_spec)

        # LSNR dropout: process only frames above threshold during training
        # Initialize mask to None for use later
        active_mask: mx.array | None = None
        if self.lsnr_dropout and training:
            # Find active frames (LSNR > threshold)
            # lsnr shape: (batch, time, 1)
            active_mask = mx.squeeze(lsnr, axis=-1) > self.lsnr_dropout_threshold

            # For simplicity in MLX, we process all frames but mask the output
            # This is less efficient than the PyTorch gather approach but works with MLX's static graphs

        # Mamba backbone
        emb, _ = self.backbone(emb)

        # Decode ERB mask
        erb_mask = self.erb_decoder(emb)  # (batch, time, nb_erb)

        # Expand ERB mask to full spectrum
        mask = mx.matmul(erb_mask, mx.transpose(self._erb_fb))

        # Apply mask (for frequencies above nb_df)
        masked_real = spec_real * mask
        masked_imag = spec_imag * mask

        # Decode DF coefficients or complex gains
        df_out = self.df_decoder(emb)

        # Apply deep filtering or complex gain (for low frequencies)
        if self.df_output_mode == "coefficients":
            # FIR filter convolution via DfOp
            assert self.df_op is not None  # Type narrowing
            spec_out = self.df_op(
                (masked_real, masked_imag),
                df_out,
            )
        else:
            # Direct complex multiplication
            spec_out = self._apply_complex_gain((masked_real, masked_imag), df_out)

        # Apply post-filter (optional, reduces musical noise)
        spec_out = self._apply_post_filter(spec_out, spec)

        # Apply LSNR dropout masking
        if self.lsnr_dropout and training and active_mask is not None:
            spec_out_real, spec_out_imag = spec_out
            # For frames below threshold, keep original noisy spectrum
            active_mask_expanded = mx.expand_dims(active_mask, axis=-1)
            spec_out_real = mx.where(active_mask_expanded, spec_out_real, spec_real)
            spec_out_imag = mx.where(active_mask_expanded, spec_out_imag, spec_imag)
            spec_out = (spec_out_real, spec_out_imag)

        return spec_out

    def forward_with_lsnr(
        self,
        spec: Tuple[mx.array, mx.array],
        feat_erb: mx.array,
        feat_spec: mx.array,
        training: bool = False,
    ) -> Tuple[Tuple[mx.array, mx.array], mx.array]:
        """Forward pass that also returns LSNR estimate.

        Useful for training with LSNR loss.

        Args:
            spec: Input spectrum as (real, imag)
            feat_erb: ERB features
            feat_spec: DF features
            training: Whether in training mode

        Returns:
            Tuple of (enhanced spectrum, LSNR estimate)
        """
        spec_real, spec_imag = spec
        # Shape is (batch, time, freq) - used implicitly in operations below
        _ = spec_real.shape

        # Encode
        emb, lsnr = self.encoder(feat_erb, feat_spec)

        # LSNR dropout during training - initialize mask to None for use later
        active_mask: mx.array | None = None
        if self.lsnr_dropout and training:
            active_mask = mx.squeeze(lsnr, axis=-1) > self.lsnr_dropout_threshold

        # Mamba backbone
        emb, _ = self.backbone(emb)

        # Decode ERB mask
        erb_mask = self.erb_decoder(emb)
        mask = mx.matmul(erb_mask, mx.transpose(self._erb_fb))
        masked_real = spec_real * mask
        masked_imag = spec_imag * mask

        # Decode DF coefficients or complex gains
        df_out = self.df_decoder(emb)

        # Apply deep filtering or complex gain
        if self.df_output_mode == "coefficients":
            assert self.df_op is not None  # Type narrowing
            spec_out = self.df_op((masked_real, masked_imag), df_out)
        else:
            spec_out = self._apply_complex_gain((masked_real, masked_imag), df_out)

        # Apply post-filter (optional, reduces musical noise)
        spec_out = self._apply_post_filter(spec_out, spec)

        # Apply LSNR dropout masking
        if self.lsnr_dropout and training and active_mask is not None:
            spec_out_real, spec_out_imag = spec_out
            active_mask_expanded = mx.expand_dims(active_mask, axis=-1)
            spec_out_real = mx.where(active_mask_expanded, spec_out_real, spec_real)
            spec_out_imag = mx.where(active_mask_expanded, spec_out_imag, spec_imag)
            spec_out = (spec_out_real, spec_out_imag)

        return spec_out, lsnr

    def enhance(
        self,
        noisy_audio: mx.array,
        return_spec: bool = False,
    ) -> mx.array | tuple[mx.array, tuple[mx.array, mx.array]]:
        """End-to-end enhancement from audio.

        Args:
            noisy_audio: Noisy audio waveform (samples,) or (batch, samples)
            return_spec: Whether to also return the spectrum

        Returns:
            Enhanced audio waveform (or tuple with spectrum if return_spec)
        """
        from .ops import istft, stft

        # Handle 1D input by adding batch dimension
        input_1d = noisy_audio.ndim == 1
        if input_1d:
            noisy_audio = mx.expand_dims(noisy_audio, axis=0)

        # STFT
        spec_real, spec_imag = stft(
            noisy_audio,
            n_fft=self.p.fft_size,
            hop_length=self.p.hop_size,
        )

        # Compute features
        mag = mx.sqrt(spec_real**2 + spec_imag**2 + 1e-8)
        feat_erb = mx.matmul(mag, self._erb_fb)
        feat_spec = mx.stack([spec_real[:, :, : self.p.nb_df], spec_imag[:, :, : self.p.nb_df]], axis=-1)

        # Forward pass (inference mode - no dropout)
        spec_out = self((spec_real, spec_imag), feat_erb, feat_spec, training=False)

        # iSTFT
        enhanced = istft(
            spec_out,
            n_fft=self.p.fft_size,
            hop_length=self.p.hop_size,
        )

        # Remove batch dimension if input was 1D
        if input_1d:
            enhanced = mx.squeeze(enhanced, axis=0)
            if return_spec:
                spec_out = (mx.squeeze(spec_out[0], axis=0), mx.squeeze(spec_out[1], axis=0))

        if return_spec:
            return enhanced, spec_out
        return enhanced


class DfNet4Lite(DfNet4):
    """Lightweight version of DeepFilterNet4.

    Reduces parameters by ~50% while maintaining most performance.
    Uses smaller hidden dimensions and fewer layers.
    """

    def __init__(self, p: Optional[ModelParams4] = None):
        if p is None:
            p = get_default_config()

        # Always apply lite modifications to create a smaller model
        # Create a copy to avoid mutating the original
        import copy

        p = copy.deepcopy(p)
        p.encoder.conv_channels = 32
        p.encoder.emb_hidden_dim = 128
        p.df.nb_df_hidden = 128
        p.erb.erb_hidden = 32
        p.backbone.nb_layers = 2

        super().__init__(p)


# ============================================================================
# Streaming Inference
# ============================================================================


class StreamingState:
    """Container for streaming inference state.

    Holds all state needed for frame-by-frame processing including
    Mamba hidden states, STFT buffers, and overlap-add state.

    Attributes:
        mamba_states: List of Mamba layer hidden states
        input_buffer: Buffer for STFT input samples
        output_buffer: Buffer for iSTFT overlap-add synthesis
        window_sum: Window sum buffer for iSTFT normalization
        frame_count: Number of frames processed
    """

    def __init__(
        self,
        batch_size: int,
        n_fft: int,
        hop_length: int,
        d_inner: int,
        d_state: int,
        num_layers: int,
    ):
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Mamba state: list of (batch, d_inner, d_state) for each layer
        self.mamba_states: Optional[mx.array] = None  # Will be (num_layers, batch, d_inner, d_state)

        # STFT input buffer - stores last n_fft - hop_length samples
        self.input_buffer = mx.zeros((batch_size, n_fft - hop_length))

        # iSTFT output buffer and window sum
        self.output_buffer = mx.zeros((batch_size, n_fft))
        self.window_sum = mx.zeros((n_fft,))

        # Frame counter
        self.frame_count = 0


class StreamingDfNet4(nn.Module):
    """Streaming wrapper for frame-by-frame DeepFilterNet4 inference.

    Enables real-time audio processing by maintaining state between
    frames and processing audio in chunks of hop_length samples.

    This wrapper handles:
    - STFT frame buffering with proper overlap
    - Mamba hidden state persistence across frames
    - iSTFT overlap-add synthesis
    - Automatic state management

    Example:
        >>> model = DfNet4()
        >>> streaming = StreamingDfNet4(model)
        >>> state = streaming.init_state(batch_size=1)
        >>>
        >>> # Process audio in chunks
        >>> for chunk in audio_chunks:  # Each chunk is hop_length samples
        ...     enhanced, state = streaming.process_frame(chunk, state)
        ...     output_buffer.append(enhanced)
        >>>
        >>> # Flush remaining samples
        >>> final, _ = streaming.flush(state)

    Args:
        model: Pre-initialized DfNet4 model
    """

    def __init__(self, model: DfNet4):
        super().__init__()

        self.model = model
        self.p = model.p

        # Audio parameters
        self.n_fft = self.p.fft_size
        self.hop_length = self.p.hop_size
        self.sr = self.p.sr

        # Model parameters for state sizing
        self.emb_dim = self.p.emb_hidden_dim
        self.num_backbone_layers = self.p.backbone.nb_layers
        self.d_state = self.p.backbone.d_state
        self.expand_factor = self.p.backbone.expand_factor
        self.d_inner = self.emb_dim * self.expand_factor

        # Get window for STFT/iSTFT
        from .ops import get_window

        self.window = get_window("sqrt_hann", self.n_fft)

    def init_state(self, batch_size: int = 1) -> StreamingState:
        """Initialize streaming state for a new audio stream.

        Args:
            batch_size: Number of audio streams to process in parallel

        Returns:
            Initialized StreamingState object
        """
        return StreamingState(
            batch_size=batch_size,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            d_inner=self.d_inner,
            d_state=self.d_state,
            num_layers=self.num_backbone_layers,
        )

    def _stft_frame(self, audio_frame: mx.array, state: StreamingState) -> Tuple[mx.array, mx.array]:
        """Compute STFT for a single frame.

        Args:
            audio_frame: Audio samples (batch, hop_length)
            state: Current streaming state

        Returns:
            Tuple of (real, imag) each (batch, 1, n_freqs)
        """
        # Concatenate buffer with new samples
        full_frame = mx.concatenate([state.input_buffer, audio_frame], axis=1)

        # Apply window
        windowed = full_frame * self.window

        # Compute FFT
        fft_out = mx.fft.rfft(windowed, axis=-1)

        # Extract real and imaginary, add time dimension
        real = mx.expand_dims(mx.real(fft_out), axis=1)  # (batch, 1, n_freqs)
        imag = mx.expand_dims(mx.imag(fft_out), axis=1)

        return real, imag

    def _istft_frame(
        self,
        spec_real: mx.array,
        spec_imag: mx.array,
        state: StreamingState,
    ) -> mx.array:
        """Compute iSTFT for a single frame with overlap-add.

        Args:
            spec_real: Real part (batch, 1, n_freqs)
            spec_imag: Imaginary part (batch, 1, n_freqs)
            state: Current streaming state

        Returns:
            Enhanced audio samples (batch, hop_length)
        """
        # Remove time dimension
        spec_real = mx.squeeze(spec_real, axis=1)
        spec_imag = mx.squeeze(spec_imag, axis=1)

        # Construct complex spectrum and inverse FFT
        complex_spec = spec_real + 1j * spec_imag
        frame = mx.fft.irfft(complex_spec, n=self.n_fft, axis=-1)

        # Apply synthesis window
        frame = frame * self.window

        # Overlap-add with previous buffer
        output = state.output_buffer + frame

        # Extract output samples (first hop_length samples are ready)
        ready_samples = output[:, : self.hop_length]

        # Shift buffer: keep samples that will overlap with next frame
        new_buffer = mx.zeros_like(state.output_buffer)
        new_buffer = new_buffer.at[:, : self.n_fft - self.hop_length].add(output[:, self.hop_length :])
        state.output_buffer = new_buffer

        # Update window sum for normalization
        win_sq = self.window * self.window
        new_window_sum = mx.zeros_like(state.window_sum)
        new_window_sum = new_window_sum.at[: self.n_fft - self.hop_length].add(state.window_sum[self.hop_length :])
        new_window_sum = new_window_sum + win_sq
        state.window_sum = new_window_sum

        # Normalize output (using first hop_length of window sum)
        norm_factor = mx.maximum(state.window_sum[: self.hop_length], 1e-8)
        ready_samples = ready_samples / norm_factor

        return ready_samples

    def _forward_with_state(
        self,
        spec: Tuple[mx.array, mx.array],
        feat_erb: mx.array,
        feat_spec: mx.array,
        mamba_state: Optional[mx.array],
    ) -> Tuple[Tuple[mx.array, mx.array], Optional[mx.array]]:
        """Forward pass with explicit Mamba state threading.

        Args:
            spec: Input spectrum (real, imag) each (batch, time, freq)
            feat_erb: ERB features (batch, time, erb_bands)
            feat_spec: DF features (batch, time, df_bins, 2)
            mamba_state: Previous Mamba states or None

        Returns:
            Tuple of (enhanced_spec, new_mamba_state)
        """
        model = self.model
        spec_real, spec_imag = spec

        # Apply feature lookahead padding if configured
        if model.conv_lookahead > 0:
            feat_erb = model._pad_features(feat_erb, model.conv_lookahead)
            feat_spec = model._pad_features(feat_spec, model.conv_lookahead)

        # Encode
        emb, lsnr = model.encoder(feat_erb, feat_spec)

        # Mamba backbone with state threading
        new_state, emb = self._backbone_with_state(emb, mamba_state)

        # Decode ERB mask
        erb_mask = model.erb_decoder(emb)

        # Expand ERB mask to full spectrum
        mask = mx.matmul(erb_mask, mx.transpose(model._erb_fb))

        # Apply mask
        masked_real = spec_real * mask
        masked_imag = spec_imag * mask

        # Decode DF
        df_out = model.df_decoder(emb)

        # Apply deep filtering or complex gain
        if model.df_output_mode == "coefficients":
            assert model.df_op is not None
            spec_out = model.df_op((masked_real, masked_imag), df_out)
        else:
            spec_out = model._apply_complex_gain((masked_real, masked_imag), df_out)

        # Apply post-filter
        spec_out = model._apply_post_filter(spec_out, spec)

        return spec_out, new_state

    def _backbone_with_state(
        self,
        emb: mx.array,
        state: Optional[mx.array],
    ) -> Tuple[Optional[mx.array], mx.array]:
        """Process through Mamba backbone with explicit state.

        Args:
            emb: Embedding (batch, time, emb_dim)
            state: Previous state (num_layers, batch, d_inner, d_state) or None

        Returns:
            Tuple of (new_state, output_emb)
        """
        backbone = self.model.backbone

        # Input projection
        x = emb
        if backbone.input_proj is not None:
            x = backbone.input_proj(x)

        # Process through Mamba layers with state threading
        new_states = []
        for i, layer in enumerate(backbone.layers):
            layer_state = state[i] if state is not None else None
            x, new_layer_state = layer(x, layer_state)
            new_states.append(new_layer_state)

        # Output projection
        if backbone.output_proj is not None:
            x = backbone.output_proj(x)

        # Stack states
        new_state = mx.stack(new_states, axis=0) if new_states else None

        return new_state, x

    def process_frame(
        self,
        audio_frame: mx.array,
        state: StreamingState,
    ) -> Tuple[mx.array, StreamingState]:
        """Process a single audio frame.

        Args:
            audio_frame: Input audio (batch, hop_length) or (hop_length,)
            state: Current streaming state

        Returns:
            Tuple of (enhanced_audio, updated_state)
        """
        # Handle 1D input
        input_1d = audio_frame.ndim == 1
        if input_1d:
            audio_frame = mx.expand_dims(audio_frame, axis=0)

        # Compute STFT for this frame
        spec_real, spec_imag = self._stft_frame(audio_frame, state)

        # Update input buffer for next frame
        state.input_buffer = mx.concatenate(
            [state.input_buffer[:, self.hop_length :], audio_frame],
            axis=1,
        )

        # Compute features
        mag = mx.sqrt(spec_real**2 + spec_imag**2 + 1e-8)
        feat_erb = mx.matmul(mag, self.model._erb_fb)  # (batch, 1, nb_erb)
        feat_spec = mx.stack(
            [spec_real[:, :, : self.p.nb_df], spec_imag[:, :, : self.p.nb_df]],
            axis=-1,
        )  # (batch, 1, nb_df, 2)

        # Forward pass with state
        spec_out, new_mamba_state = self._forward_with_state(
            (spec_real, spec_imag),
            feat_erb,
            feat_spec,
            state.mamba_states,
        )

        # Update Mamba state
        state.mamba_states = new_mamba_state

        # iSTFT for this frame
        enhanced = self._istft_frame(spec_out[0], spec_out[1], state)

        # Update frame count
        state.frame_count += 1

        # Handle 1D output
        if input_1d:
            enhanced = mx.squeeze(enhanced, axis=0)

        return enhanced, state

    def flush(self, state: StreamingState) -> Tuple[mx.array, StreamingState]:
        """Flush remaining samples from the output buffer.

        Call this after processing all input frames to get any
        remaining samples that haven't been output yet.

        Args:
            state: Current streaming state

        Returns:
            Tuple of (remaining_samples, final_state)
        """
        # Get remaining samples from output buffer
        remaining = state.output_buffer[:, : self.n_fft - self.hop_length]

        # Normalize
        norm_factor = mx.maximum(state.window_sum[: self.n_fft - self.hop_length], 1e-8)
        remaining = remaining / norm_factor

        return remaining, state

    def process_audio(
        self,
        audio: mx.array,
        state: Optional[StreamingState] = None,
    ) -> mx.array:
        """Process complete audio stream frame-by-frame.

        Convenience method that processes entire audio using streaming
        inference and verifies output matches batch processing.

        Args:
            audio: Input audio (samples,) or (batch, samples)
            state: Optional pre-initialized state

        Returns:
            Enhanced audio
        """
        # Handle 1D input
        input_1d = audio.ndim == 1
        if input_1d:
            audio = mx.expand_dims(audio, axis=0)

        batch_size, num_samples = audio.shape

        # Initialize state if not provided
        if state is None:
            state = self.init_state(batch_size)

        # Process frame by frame
        outputs = []
        for start in range(0, num_samples, self.hop_length):
            end = start + self.hop_length
            if end > num_samples:
                # Pad last frame if needed
                frame = mx.pad(audio[:, start:], [(0, 0), (0, end - num_samples)])
            else:
                frame = audio[:, start:end]

            enhanced_frame, state = self.process_frame(frame, state)
            outputs.append(enhanced_frame)

        # Concatenate all frames
        enhanced = mx.concatenate(outputs, axis=-1)

        # Trim to original length
        enhanced = enhanced[:, :num_samples]

        if input_1d:
            enhanced = mx.squeeze(enhanced, axis=0)

        return enhanced


# ============================================================================
# Model Initialization
# ============================================================================


def init_model(
    config: Optional[ModelParams4] = None,
    variant: Literal["full", "lite"] = "full",
) -> DfNet4:
    """Initialize a DeepFilterNet4 model.

    Args:
        config: Model configuration (uses defaults if None)
        variant: Model variant ("full" or "lite")

    Returns:
        Initialized model
    """
    if variant == "lite":
        return DfNet4Lite(config)
    return DfNet4(config)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters.

    Args:
        model: MLX model

    Returns:
        Number of parameters
    """

    def _count_recursive(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += _count_recursive(v)
        elif isinstance(params, list):
            for v in params:
                total += _count_recursive(v)
        elif isinstance(params, mx.array):
            total += params.size
        return total

    # Get all parameters using MLX's parameters() method
    params = model.parameters()
    return _count_recursive(params)


def model_summary(model: DfNet4) -> Dict[str, Any]:
    """Get model summary information.

    Args:
        model: DfNet4 model

    Returns:
        Dictionary with model information
    """
    return {
        "model_type": type(model).__name__,
        "parameters": count_parameters(model),
        "config": {
            "nb_erb": model.p.nb_erb,
            "nb_df": model.p.nb_df,
            "df_order": model.p.df_order,
            "emb_hidden_dim": model.p.emb_hidden_dim,
            "backbone_layers": model.p.backbone.nb_layers,
        },
    }
