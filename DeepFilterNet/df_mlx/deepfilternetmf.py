"""DFNetMF (Multi-Frame) MLX Implementation.

DFNetMF uses multi-frame filtering (Wiener filter or MVDR beamformer) instead
of the standard deep filtering approach. It predicts:
- ERB mask for high frequencies
- IFC (inter-frame correlation) vector for speech
- Correlation matrix (noise/noisy) for multi-frame filtering

This model is suitable for scenarios where multi-frame processing provides
better noise reduction than single-frame deep filtering.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from . import multiframe as MF
from .modules import Conv2dNormAct, ConvTranspose2dNormAct, GroupedLinear, SqueezedGRU_S


@dataclass
class ModelParamsMF:
    """Model parameters for DFNetMF."""

    # Audio parameters
    sr: int = 48000
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_df: int = 96

    # DF parameters
    df_order: int = 5
    df_lookahead: int = 0
    df_n_iter: int = 1

    # Convolution parameters
    conv_lookahead: int = 0
    conv_ch: int = 16
    conv_depthwise: bool = True
    convt_depthwise: bool = True
    conv_kernel: List[int] = field(default_factory=lambda: [1, 3])
    conv_kernel_inp: List[int] = field(default_factory=lambda: [3, 3])

    # Embedding parameters
    emb_hidden_dim: int = 256
    emb_num_layers: int = 2
    emb_gru_skip: str = "none"

    # DF pathway parameters
    df_hidden_dim: int = 256
    df_num_layers: int = 3
    df_gru_skip: str = "none"
    df_pathway_kernel_size_t: int = 1

    # Architecture options
    enc_concat: bool = False
    lin_groups: int = 1
    enc_lin_groups: int = 16

    # Multi-frame parameters
    mfop_method: str = "WF"  # "WF" or "MVDR"
    mf_est_inverse: bool = True
    mf_use_cholesky_decomp: bool = False

    # Post-filter
    mask_pf: bool = False

    # SNR estimation
    lsnr_min: float = -15.0
    lsnr_max: float = 35.0


class Add(nn.Module):
    """Add two tensors."""

    def __call__(self, a: mx.array, b: mx.array) -> mx.array:
        return a + b


class Concat(nn.Module):
    """Concatenate two tensors along last dimension."""

    def __call__(self, a: mx.array, b: mx.array) -> mx.array:
        return mx.concatenate([a, b], axis=-1)


class EncoderMF(nn.Module):
    """Encoder for DFNetMF.

    Encodes ERB features and DF spectrogram features into embeddings.
    Uses MLX Conv2dNormAct with NHWC format.
    """

    def __init__(self, p: ModelParamsMF):
        super().__init__()
        assert p.nb_erb % 4 == 0, "nb_erb should be divisible by 4"
        conv_ch = p.conv_ch

        # ERB pathway convolutions (NHWC format for MLX)
        self.erb_conv0 = Conv2dNormAct(
            1,
            conv_ch,
            kernel_size=tuple(p.conv_kernel_inp),
            padding="same",
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.erb_conv1 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=tuple(p.conv_kernel),
            stride=(1, 2),  # Stride 2 on frequency axis
            padding="same",
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.erb_conv2 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=tuple(p.conv_kernel),
            stride=(1, 2),  # Stride 2 on frequency axis
            padding="same",
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.erb_conv3 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=tuple(p.conv_kernel),
            padding="same",
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )

        # DF pathway convolutions
        self.df_conv0 = Conv2dNormAct(
            2,
            conv_ch,
            kernel_size=tuple(p.conv_kernel_inp),
            padding="same",
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.df_conv1 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=tuple(p.conv_kernel),
            stride=(1, 2),  # Stride 2 on frequency axis
            padding="same",
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )

        # Dimensions
        self.erb_bins = p.nb_erb
        self.emb_in_dim = conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = conv_ch * p.nb_erb // 4

        # DF embedding projection
        df_fc_emb = GroupedLinear(conv_ch * p.nb_df // 2, self.emb_in_dim, groups=p.enc_lin_groups)
        self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU())

        # Combine ERB and DF embeddings
        if p.enc_concat:
            self.combine_in_dim = self.emb_in_dim * 2
            self.combine = Concat()
        else:
            self.combine_in_dim = self.emb_in_dim
            self.combine = Add()

        # Embedding GRU
        self.emb_gru = SqueezedGRU_S(
            self.combine_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=1,
            linear_groups=p.lin_groups,
        )

        # LSNR estimation
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def __call__(
        self, feat_erb: mx.array, feat_spec: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Forward pass.

        Args:
            feat_erb: ERB features [B, T, E, 1] (NHWC format)
            feat_spec: DF spectrogram features [B, T, F, 2] (NHWC format)

        Returns:
            e0, e1, e2, e3: Encoder feature maps
            emb: Embedding
            c0: DF conv0 output
            lsnr: Local SNR estimate
        """
        # ERB pathway (NHWC format: [B, T, E, C])
        e0 = self.erb_conv0(feat_erb)  # [B, T, E, C]
        e1 = self.erb_conv1(e0)  # [B, T, E/2, C]
        e2 = self.erb_conv2(e1)  # [B, T, E/4, C]
        e3 = self.erb_conv3(e2)  # [B, T, E/4, C]

        # DF pathway (NHWC format: [B, T, F, C])
        c0 = self.df_conv0(feat_spec)  # [B, T, F, C]
        c1 = self.df_conv1(c0)  # [B, T, F/2, C]

        # Create embeddings
        # In NHWC format, reshape for embedding
        cemb = mx.reshape(c1, (c1.shape[0], c1.shape[1], -1))  # [B, T, F/2*C]
        cemb = self.df_fc_emb(cemb)

        emb = mx.reshape(e3, (e3.shape[0], e3.shape[1], -1))  # [B, T, E/4*C]

        emb = self.combine(emb, cemb)
        emb, _ = self.emb_gru(emb)

        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset

        return e0, e1, e2, e3, emb, c0, lsnr


class ErbDecoderMF(nn.Module):
    """ERB mask decoder for DFNetMF."""

    def __init__(self, p: ModelParamsMF):
        super().__init__()
        conv_ch = p.conv_ch

        # Skip connection convolutions (1x1)
        self.conv3p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, norm="batch", activation="relu")
        self.conv2p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, norm="batch", activation="relu")
        self.conv1p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, norm="batch", activation="relu")
        self.conv0p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, norm="batch", activation="relu")

        # Transpose convolutions for upsampling
        self.convt3 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=tuple(p.conv_kernel),
            padding="same",
            norm="batch",
            activation="relu",
            separable=p.convt_depthwise,
        )
        self.convt2 = ConvTranspose2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=tuple(p.conv_kernel),
            stride=(1, 2),
            padding=(0, p.conv_kernel[1] // 2),
            output_padding=(0, 1),
            norm="batch",
            activation="relu",
            separable=p.convt_depthwise,
        )
        self.convt1 = ConvTranspose2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=tuple(p.conv_kernel),
            stride=(1, 2),
            padding=(0, p.conv_kernel[1] // 2),
            output_padding=(0, 1),
            norm="batch",
            activation="relu",
            separable=p.convt_depthwise,
        )
        self.conv0_out = Conv2dNormAct(
            conv_ch,
            1,
            kernel_size=tuple(p.conv_kernel),
            padding="same",
            norm=None,
            activation="sigmoid",  # Use sigmoid activation directly
        )

        # Store post_filter setting
        self.post_filter = p.mask_pf

    def __call__(
        self,
        emb: mx.array,
        e3: mx.array,
        e2: mx.array,
        e1: mx.array,
        e0: mx.array,
    ) -> mx.array:
        """Decode ERB mask.

        Args:
            emb: Embedding [B, T, emb_out_dim]
            e3, e2, e1, e0: Encoder feature maps [B, T, *, C]

        Returns:
            ERB mask [B, T, E, 1]
        """
        # Reshape embedding to match e3 spatial dimensions
        b, t = emb.shape[:2]
        emb = mx.reshape(emb, (b, t, e3.shape[2], -1))  # [B, T, E/4, C]

        # Decoder with skip connections
        x = self.conv3p(e3) + emb
        x = self.convt3(x)

        x = self.conv2p(e2) + x
        x = self.convt2(x)

        x = self.conv1p(e1) + x
        x = self.convt1(x)

        x = self.conv0p(e0) + x
        m = self.conv0_out(x)  # [B, T, E, 1]

        # Squeeze to (B, T, E)
        return mx.squeeze(m, axis=-1)


class DfDecoderMF(nn.Module):
    """DF decoder for DFNetMF using multi-frame filtering."""

    def __init__(self, p: ModelParamsMF):
        super().__init__()
        conv_ch = p.conv_ch
        self.df_order = p.df_order
        self.nb_df = p.nb_df
        self.n_iter = p.df_n_iter
        self.mfop_method = p.mfop_method

        # Skip connection projection
        self.df_convp = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, norm="batch", activation="relu")

        # Embedding projection to match spatial dims
        self.emb_in_dim = conv_ch * p.nb_erb // 4
        self.df_fc_out = GroupedLinear(self.emb_in_dim, conv_ch * p.nb_df // 2, groups=p.lin_groups)

        # Transpose conv for upsampling
        self.df_convt = ConvTranspose2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=tuple(p.conv_kernel),
            stride=(1, 2),
            padding=(0, p.conv_kernel[1] // 2),
            output_padding=(0, 1),
            norm="batch",
            activation="relu",
            separable=p.convt_depthwise,
        )

        # Output projection convolution
        self.df_conv_out = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=(p.df_pathway_kernel_size_t, 1),
            padding="same",
            norm=None,
            activation=None,
        )

        # DF GRU for temporal modeling
        self.df_gru = SqueezedGRU_S(
            conv_ch * p.nb_df,
            p.df_hidden_dim,
            output_size=conv_ch * p.nb_df,
            num_layers=p.df_num_layers,
            linear_groups=p.lin_groups,
        )

        # Output heads
        # IFC: Inter-frame correlation vector for speech
        ifc_out_dim = p.nb_df * p.df_order * 2  # Complex: real + imag
        self.ifc_out = nn.Linear(conv_ch * p.nb_df, ifc_out_dim)

        # Correlation matrix output (for Wiener filter or MVDR)
        if p.mf_est_inverse:
            # Estimate inverse correlation matrix
            corr_out_dim = p.nb_df * p.df_order * p.df_order * 2  # Complex matrix
        else:
            # Estimate lower triangular Cholesky factor
            n_triu = p.df_order * (p.df_order + 1) // 2
            corr_out_dim = p.nb_df * n_triu * 2

        self.corr_out = nn.Linear(conv_ch * p.nb_df, corr_out_dim)

        # Multi-frame operation
        if p.mfop_method.upper() == "WF":
            self.mfop = MF.MfWf(
                num_freqs=p.nb_df,
                frame_size=p.df_order,
                lookahead=p.df_lookahead,
                cholesky_decomp=p.mf_use_cholesky_decomp,
            )
        elif p.mfop_method.upper() == "MVDR":
            self.mfop = MF.MfMvdr(
                num_freqs=p.nb_df,
                frame_size=p.df_order,
                lookahead=p.df_lookahead,
                cholesky_decomp=p.mf_use_cholesky_decomp,
            )
        else:
            raise ValueError(f"Unknown MF method: {p.mfop_method}")

    def __call__(
        self,
        emb: mx.array,
        c0: mx.array,
        spec: mx.array,
    ) -> mx.array:
        """Apply multi-frame deep filtering.

        Args:
            emb: Embedding [B, T, emb_out_dim]
            c0: DF conv0 output [B, T, F, C]
            spec: Input spectrogram [B, T, F, 2] (complex as real)

        Returns:
            Enhanced DF spectrogram [B, T, nb_df, 2]
        """
        b, t = emb.shape[:2]
        conv_ch = c0.shape[-1]

        # Project embedding and reshape
        emb_proj = self.df_fc_out(emb)  # [B, T, C*F/2]
        emb_proj = mx.reshape(emb_proj, (b, t, -1, conv_ch))  # [B, T, F/2, C]

        # Skip connection with upsampling
        x = self.df_convp(c0[:, :, : self.nb_df // 2, :])  # [B, T, F/2, C]
        x = x + emb_proj
        x = self.df_convt(x)  # [B, T, F, C]

        # Temporal convolution
        x = self.df_conv_out(x)  # [B, T, F, C]

        # Reshape for GRU
        x = mx.reshape(x, (b, t, -1))  # [B, T, F*C]
        x, _ = self.df_gru(x)  # [B, T, F*C]

        # Output heads
        # IFC: [B, T, F*N*2] - flattened as expected by MfWf
        ifc = self.ifc_out(x)  # [B, T, F*N*2]
        ifc = mx.reshape(ifc, (b, t, self.nb_df, -1))  # [B, T, F, N*2]

        # Correlation matrix: [B, T, F*N*N*2] - flattened
        corr = self.corr_out(x)  # [B, T, F*N*N*2]
        corr = mx.reshape(corr, (b, t, self.nb_df, -1))  # [B, T, F, N*N*2]

        # Apply multi-frame operation
        # MfWf expects:
        #   spec: [B, 1, T, F, 2]
        #   ifc: [B, T, F, N*2]
        #   iRxx: [B, T, F, N*N*2]
        spec_df = spec[:, :, : self.nb_df, :]  # [B, T, nb_df, 2]
        spec_df = mx.expand_dims(spec_df, axis=1)  # [B, 1, T, nb_df, 2]

        enhanced = spec_df

        for _ in range(self.n_iter):
            enhanced = self.mfop(enhanced, ifc, corr)

        # Remove channel dim and return [B, T, nb_df, 2]
        enhanced = mx.squeeze(enhanced, axis=1)
        return enhanced


class DFNetMF(nn.Module):
    """DFNetMF: Multi-Frame Deep FilterNet.

    Uses multi-frame filtering (Wiener filter or MVDR beamformer)
    for speech enhancement.
    """

    def __init__(
        self,
        erb_fb: mx.array,
        erb_inv_fb: mx.array,
        run_df: bool = True,
        train_mask: bool = True,
        params: Optional[ModelParamsMF] = None,
    ):
        super().__init__()

        self.p = params or ModelParamsMF()
        self.run_df = run_df
        self.train_mask = train_mask

        # ERB filterbanks
        self.erb_fb = erb_fb
        self.erb_inv_fb = erb_inv_fb

        # Encoder
        self.encoder = EncoderMF(self.p)

        # ERB decoder (always used)
        self.erb_decoder = ErbDecoderMF(self.p)

        # Store inverse ERB filterbank for mask application
        self._erb_inv_fb = erb_inv_fb

        # DF decoder (optional)
        if run_df:
            self.df_decoder = DfDecoderMF(self.p)
        else:
            self.df_decoder = None

    def __call__(
        self,
        spec: mx.array,
        feat_erb: mx.array,
        feat_spec: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Forward pass.

        Args:
            spec: Input spectrogram [B, T, F, 2] (complex as real)
            feat_erb: ERB features [B, T, E, 1]
            feat_spec: DF features [B, T, F_df, 2]

        Returns:
            spec: Enhanced spectrogram [B, T, F, 2]
            m: ERB mask [B, T, E, 1]
            lsnr: Local SNR estimate [B, T, 1]
            df_coefs: DF coefficients (placeholder for MF output)
        """
        # Encode
        e0, e1, e2, e3, emb, c0, lsnr = self.encoder(feat_erb, feat_spec)

        # Decode ERB mask (shape: [B, T, E])
        m = self.erb_decoder(emb, e3, e2, e1, e0)

        # Expand ERB mask to full spectrum using inverse ERB filterbank
        # m: [B, T, E], erb_inv_fb: [E, F]
        mask_full = mx.matmul(m, self._erb_inv_fb)  # [B, T, F]

        # Apply mask to spectrogram [B, T, F, 2]
        spec_real = spec[:, :, :, 0]
        spec_imag = spec[:, :, :, 1]
        spec_masked_real = spec_real * mask_full
        spec_masked_imag = spec_imag * mask_full

        # Apply DF if enabled
        if self.run_df and self.df_decoder is not None:
            # Apply multi-frame filtering to low frequencies
            spec_masked_combined = mx.stack([spec_masked_real, spec_masked_imag], axis=-1)
            spec_df = self.df_decoder(emb, c0, spec_masked_combined)
            # Combine: DF for low freqs, masked for high freqs
            spec_out = mx.concatenate(
                [spec_df, spec_masked_combined[:, :, self.p.nb_df :, :]],
                axis=2,
            )
        else:
            spec_out = mx.stack([spec_masked_real, spec_masked_imag], axis=-1)

        # Return placeholder for df_coefs (MF doesn't use traditional DF coefficients)
        df_coefs = mx.zeros((spec.shape[0], spec.shape[1], self.p.nb_df, self.p.df_order, 2))

        return spec_out, m, lsnr, df_coefs


def create_dfnetmf(
    erb_fb: mx.array,
    erb_inv_fb: mx.array,
    run_df: bool = True,
    train_mask: bool = True,
    **kwargs,
) -> DFNetMF:
    """Create a DFNetMF model.

    Args:
        erb_fb: ERB filterbank matrix
        erb_inv_fb: Inverse ERB filterbank matrix
        run_df: Whether to run DF pathway
        train_mask: Whether mask is trainable
        **kwargs: Additional parameters for ModelParamsMF

    Returns:
        DFNetMF model instance
    """
    params = ModelParamsMF(**kwargs)
    return DFNetMF(erb_fb, erb_inv_fb, run_df, train_mask, params)
