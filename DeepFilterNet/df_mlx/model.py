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
from .modules import Conv2dNormAct, ConvTranspose2dNormAct, DfOp, Mask, erb_fb

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

    def __call__(
        self,
        feat_erb: mx.array,
        feat_spec: mx.array,
    ) -> mx.array:
        """Forward pass.

        Args:
            feat_erb: ERB features (batch, time, erb_bands)
            feat_spec: DF features (batch, time, df_bins, 2)

        Returns:
            Embedding tensor (batch, time, emb_hidden_dim)
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

        return emb


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
        mask = self.mask(x)
        return mask


# ============================================================================
# DF Decoder
# ============================================================================


class DfDecoder4(nn.Module):
    """Deep Filter coefficient decoder.

    Decodes embedding to complex filter coefficients for deep filtering.

    Args:
        p: Model parameters
    """

    def __init__(self, p: ModelParams4):
        super().__init__()

        self.p = p
        self.nb_df = p.nb_df
        self.df_order = p.df_order

        # Output size: nb_df * df_order * 2 (real + imag)
        self.out_dim = p.nb_df * p.df_order * 2

        # Decoder layers
        self.layers = [
            nn.Linear(p.emb_hidden_dim, p.df_hidden_dim),
            nn.ReLU(),
            nn.Linear(p.df_hidden_dim, p.df_hidden_dim),
            nn.ReLU(),
            nn.Linear(p.df_hidden_dim, self.out_dim),
        ]

    def __call__(self, emb: mx.array) -> mx.array:
        """Forward pass.

        Args:
            emb: Embedding tensor (batch, time, emb_hidden_dim)

        Returns:
            DF coefficients (batch, time, nb_df, df_order, 2)
        """
        batch, time, _ = emb.shape

        x = emb
        for layer in self.layers:
            x = layer(x)

        # Reshape to (batch, time, nb_df, df_order, 2)
        x = x.reshape(batch, time, self.nb_df, self.df_order, 2)

        return x


# ============================================================================
# Main Model
# ============================================================================


class DfNet4(nn.Module):
    """DeepFilterNet4 main model.

    Complete model combining encoder, Mamba backbone, and decoders
    for speech enhancement.

    Args:
        p: Model parameters (uses defaults if None)
    """

    def __init__(self, p: Optional[ModelParams4] = None):
        super().__init__()

        if p is None:
            p = get_default_config()
        self.p = p

        # ERB filterbank (non-trainable)
        self._erb_fb = erb_fb(
            sr=p.sr,
            fft_size=p.fft_size,
            nb_bands=p.nb_erb,
        )

        # Encoder
        self.encoder = Encoder4(p)

        # Mamba backbone
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
        self.df_decoder = DfDecoder4(p)

        # Deep filtering operation
        self.df_op = DfOp(
            nb_df=p.nb_df,
            df_order=p.df_order,
            df_lookahead=p.df_lookahead,
        )

    def __call__(
        self,
        spec: Tuple[mx.array, mx.array],
        feat_erb: mx.array,
        feat_spec: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            spec: Input spectrum as (real, imag), each (batch, time, freq)
            feat_erb: ERB features (batch, time, erb_bands)
            feat_spec: DF features (batch, time, df_bins, 2)

        Returns:
            Enhanced spectrum as (real, imag)
        """
        spec_real, spec_imag = spec

        # Encode
        emb = self.encoder(feat_erb, feat_spec)

        # Mamba backbone
        emb, _ = self.backbone(emb)

        # Decode ERB mask
        erb_mask = self.erb_decoder(emb)  # (batch, time, nb_erb)

        # Expand ERB mask to full spectrum
        mask = mx.matmul(erb_mask, mx.transpose(self._erb_fb))

        # Apply mask (for frequencies above nb_df)
        masked_real = spec_real * mask
        masked_imag = spec_imag * mask

        # Decode DF coefficients
        df_coef = self.df_decoder(emb)  # (batch, time, nb_df, df_order, 2)

        # Apply deep filtering (for low frequencies)
        spec_out = self.df_op(
            (masked_real, masked_imag),
            df_coef,
        )

        return spec_out

    def enhance(
        self,
        noisy_audio: mx.array,
        return_spec: bool = False,
    ) -> mx.array:
        """End-to-end enhancement from audio.

        Args:
            noisy_audio: Noisy audio waveform (batch, samples)
            return_spec: Whether to also return the spectrum

        Returns:
            Enhanced audio waveform (or tuple with spectrum if return_spec)
        """
        from .ops import istft, stft

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

        # Forward pass
        spec_out = self((spec_real, spec_imag), feat_erb, feat_spec)

        # iSTFT
        enhanced = istft(
            spec_out,
            n_fft=self.p.fft_size,
            hop_length=self.p.hop_size,
        )

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
            # Reduce dimensions for lite version
            p.encoder.conv_channels = 32
            p.encoder.emb_hidden_dim = 128
            p.df.nb_df_hidden = 128
            p.erb.erb_hidden = 32
            p.backbone.nb_layers = 2

        super().__init__(p)


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

    def _count_array(arr):
        if isinstance(arr, mx.array):
            return arr.size
        return 0

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
