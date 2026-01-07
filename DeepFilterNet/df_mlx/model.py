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
        batch, time, freq = spec_real.shape

        # Encode (now returns embedding and LSNR estimate)
        emb, lsnr = self.encoder(feat_erb, feat_spec)

        # LSNR dropout: process only frames above threshold during training
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

        # Decode DF coefficients
        df_coef = self.df_decoder(emb)  # (batch, time, nb_df, df_order, 2)

        # Apply deep filtering (for low frequencies)
        spec_out = self.df_op(
            (masked_real, masked_imag),
            df_coef,
        )

        # Apply LSNR dropout masking
        if self.lsnr_dropout and training:
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
        batch, time, freq = spec_real.shape

        # Encode
        emb, lsnr = self.encoder(feat_erb, feat_spec)

        # LSNR dropout during training
        if self.lsnr_dropout and training:
            active_mask = mx.squeeze(lsnr, axis=-1) > self.lsnr_dropout_threshold

        # Mamba backbone
        emb, _ = self.backbone(emb)

        # Decode ERB mask
        erb_mask = self.erb_decoder(emb)
        mask = mx.matmul(erb_mask, mx.transpose(self._erb_fb))
        masked_real = spec_real * mask
        masked_imag = spec_imag * mask

        # Decode DF coefficients
        df_coef = self.df_decoder(emb)

        # Apply deep filtering
        spec_out = self.df_op((masked_real, masked_imag), df_coef)

        # Apply LSNR dropout masking
        if self.lsnr_dropout and training:
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
    ) -> mx.array:
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
