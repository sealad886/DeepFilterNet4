"""DeepFilterNet2 model implementation for MLX.

This module provides the MLX implementation of DeepFilterNet2,
a GRU-based speech enhancement model. DFNet2 supports both
GroupedGRU and SqueezedGRU backbone options.

Architecture:
- Encoder: Parallel ERB and DF pathways with multi-layer GRU
- ERB Decoder: Spectral mask estimation for ERB bands
- DF Decoder: Deep filtering coefficient prediction with alpha blending

Key differences from DFNet3:
- Supports both GroupedGRU and SqueezedGRU backbones
- Multi-layer GRU support (configurable num_layers)
- Alpha output for DF coefficient blending
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .modules import Conv2dNormAct, ConvTranspose2dNormAct, DfOp, GroupedGRU, GroupedLinear, SqueezedGRU, erb_fb


@dataclass
class ModelParams2:
    """DFNet2 model parameters."""

    # Audio parameters
    sr: int = 48000
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_df: int = 96
    df_order: int = 5
    df_lookahead: int = 0

    # Model parameters
    conv_ch: int = 16
    conv_kernel: Tuple[int, int] = (1, 3)
    conv_kernel_inp: Tuple[int, int] = (3, 3)
    convt_kernel: Tuple[int, int] = (1, 3)
    emb_hidden_dim: int = 256
    emb_num_layers: int = 2
    df_hidden_dim: int = 256
    df_num_layers: int = 3

    # GRU configuration
    gru_type: str = "grouped"  # "grouped" or "squeeze"
    gru_groups: int = 1
    linear_groups: int = 1
    group_shuffle: bool = True

    # Encoder options
    enc_concat: bool = False

    # Skip options
    df_gru_skip: str = "none"  # "none", "identity", "groupedlinear"

    # DF pathway
    df_pathway_kernel_size_t: int = 1

    # LSNR
    lsnr_max: float = 30.0
    lsnr_min: float = -15.0

    # Post-filter
    mask_pf: bool = False
    pf_beta: float = 0.02

    # ERB widths
    erb_widths: List[int] = field(
        default_factory=lambda: [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            4,
            4,
            5,
            6,
            7,
            8,
            10,
            12,
            15,
            18,
        ]
    )


class Encoder2(nn.Module):
    """DeepFilterNet2 encoder with parallel ERB and DF pathways."""

    def __init__(self, p: ModelParams2):
        super().__init__()
        self.p = p
        conv_ch = p.conv_ch

        # ERB pathway convolutions (NHWC format for MLX)
        self.erb_conv0 = Conv2dNormAct(
            1, conv_ch, kernel_size=p.conv_kernel_inp, padding="same", norm="batch", activation="relu"
        )
        self.erb_conv1 = Conv2dNormAct(
            conv_ch, conv_ch, kernel_size=p.conv_kernel, stride=(1, 2), padding="same", norm="batch", activation="relu"
        )
        self.erb_conv2 = Conv2dNormAct(
            conv_ch, conv_ch, kernel_size=p.conv_kernel, stride=(1, 2), padding="same", norm="batch", activation="relu"
        )
        self.erb_conv3 = Conv2dNormAct(
            conv_ch, conv_ch, kernel_size=p.conv_kernel, padding="same", norm="batch", activation="relu"
        )

        # DF pathway convolutions
        self.df_conv0 = Conv2dNormAct(
            2, conv_ch, kernel_size=p.conv_kernel_inp, padding="same", norm="batch", activation="relu"
        )
        self.df_conv1 = Conv2dNormAct(
            conv_ch, conv_ch, kernel_size=p.conv_kernel, stride=(1, 2), padding="same", norm="batch", activation="relu"
        )

        # Dimension calculations
        self.emb_in_dim = conv_ch * p.nb_erb // 4
        self.emb_out_dim = p.emb_hidden_dim

        # DF pathway projection
        df_emb_in = conv_ch * p.nb_df // 2
        self.df_fc_emb = nn.Sequential(
            GroupedLinear(df_emb_in, self.emb_in_dim, p.linear_groups),
            nn.ReLU(),
        )

        # GRU input dimension
        gru_input_dim = self.emb_in_dim * 2 if p.enc_concat else self.emb_in_dim

        # Embedding GRU (single layer in encoder)
        if p.gru_type == "grouped":
            self.emb_gru = GroupedGRU(
                input_size=gru_input_dim,
                hidden_size=self.emb_out_dim,
                num_layers=1,
                groups=p.gru_groups,
                shuffle=p.group_shuffle,
            )
        else:  # squeeze
            self.emb_gru = SqueezedGRU(
                input_size=gru_input_dim,
                hidden_size=self.emb_out_dim,
                num_layers=1,
                linear_groups=p.linear_groups,
                linear_act="relu",
            )

        # LSNR estimation
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

        self.enc_concat = p.enc_concat
        self.erb_bins_downsampled = p.nb_erb // 4

    def __call__(
        self, feat_erb: mx.array, feat_spec: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Encode ERB and DF features.

        Args:
            feat_erb: ERB features (batch, time, erb_bins)
            feat_spec: Complex DF features (batch, time, df_bins, 2)

        Returns:
            e0-e3: Encoder skip connections
            emb: Embedding
            c0: DF pathway output
            lsnr: LSNR estimate
        """
        # Expand ERB features: (B, T, F) -> (B, T, F, 1)
        feat_erb = mx.expand_dims(feat_erb, axis=-1)

        # ERB pathway
        e0 = self.erb_conv0(feat_erb)
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)

        # DF pathway
        c0 = self.df_conv0(feat_spec)
        c1 = self.df_conv1(c0)

        # Flatten to embeddings
        b, t = e3.shape[:2]
        emb_erb = e3.reshape(b, t, -1)
        emb_df = c1.reshape(b, t, -1)
        emb_df = self.df_fc_emb(emb_df)

        # Combine
        if self.enc_concat:
            emb = mx.concatenate([emb_erb, emb_df], axis=-1)
        else:
            emb = emb_erb + emb_df

        # GRU
        emb, _ = self.emb_gru(emb)

        # LSNR
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset

        return e0, e1, e2, e3, emb, c0, lsnr


class ErbDecoder2(nn.Module):
    """DeepFilterNet2 ERB mask decoder."""

    def __init__(self, p: ModelParams2):
        super().__init__()
        conv_ch = p.conv_ch
        self.emb_in_dim = conv_ch * p.nb_erb // 4
        self.emb_out_dim = p.emb_hidden_dim

        # Decoder GRU (multi-layer, num_layers - 1 since encoder has 1)
        n_layers = max(1, p.emb_num_layers - 1)
        if p.gru_type == "grouped":
            self.emb_gru = GroupedGRU(
                input_size=self.emb_in_dim,
                hidden_size=self.emb_out_dim,
                num_layers=n_layers,
                groups=p.gru_groups,
                shuffle=p.group_shuffle,
            )
            # Output projection
            self.fc_emb = nn.Sequential(
                GroupedLinear(self.emb_out_dim, self.emb_in_dim, p.linear_groups),
                nn.ReLU(),
            )
        else:  # squeeze
            self.emb_gru = SqueezedGRU(
                input_size=self.emb_out_dim,
                hidden_size=self.emb_out_dim,
                output_size=self.emb_in_dim,
                num_layers=n_layers,
                linear_groups=p.linear_groups,
                gru_skip=True,
                linear_act="relu",
            )
            self.fc_emb = nn.Identity()

        # Decoder convolutions
        self.conv3p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, norm="batch", activation="relu")
        self.convt3 = Conv2dNormAct(
            conv_ch, conv_ch, kernel_size=p.conv_kernel, padding="same", norm="batch", activation="relu"
        )
        self.conv2p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, norm="batch", activation="relu")
        self.convt2 = ConvTranspose2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=p.convt_kernel,
            stride=(1, 2),
            padding=(0, 1),
            output_padding=(0, 1),
            norm="batch",
            activation="relu",
        )
        self.conv1p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, norm="batch", activation="relu")
        self.convt1 = ConvTranspose2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=p.convt_kernel,
            stride=(1, 2),
            padding=(0, 1),
            output_padding=(0, 1),
            norm="batch",
            activation="relu",
        )
        self.conv0p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, norm="batch", activation="relu")
        self.conv0_out = Conv2dNormAct(
            conv_ch, 1, kernel_size=p.conv_kernel, padding="same", norm=None, activation="sigmoid"
        )

        self.erb_bins_downsampled = p.nb_erb // 4

    def __call__(self, emb: mx.array, e3: mx.array, e2: mx.array, e1: mx.array, e0: mx.array) -> mx.array:
        """Decode ERB mask."""
        b, t, _ = emb.shape

        # GRU
        emb, _ = self.emb_gru(emb)
        emb = self.fc_emb(emb)

        # Reshape to spatial
        emb = emb.reshape(b, t, self.erb_bins_downsampled, -1)

        # Decoder with skip connections
        x = self.conv3p(e3) + emb
        x = self.convt3(x)
        x = self.conv2p(e2) + x
        x = self.convt2(x)
        x = self.conv1p(e1) + x
        x = self.convt1(x)
        x = self.conv0p(e0) + x
        m = self.conv0_out(x)

        return mx.squeeze(m, axis=-1)


class DfDecoder2(nn.Module):
    """DeepFilterNet2 DF coefficient decoder with alpha output."""

    def __init__(self, p: ModelParams2):
        super().__init__()
        conv_ch = p.conv_ch
        self.emb_dim = p.emb_hidden_dim
        self.df_hidden = p.df_hidden_dim
        self.df_bins = p.nb_df
        self.df_order = p.df_order
        self.df_out_ch = p.df_order * 2

        # Pathway convolution
        self.df_convp = Conv2dNormAct(
            conv_ch, self.df_out_ch, kernel_size=(p.df_pathway_kernel_size_t, 1), norm="batch", activation="relu"
        )

        # DF GRU (multi-layer)
        if p.gru_type == "grouped":
            self.df_gru = GroupedGRU(
                input_size=p.emb_hidden_dim,
                hidden_size=p.df_hidden_dim,
                num_layers=p.df_num_layers,
                groups=p.gru_groups,
                shuffle=p.group_shuffle,
            )
        else:  # squeeze
            self.df_gru = SqueezedGRU(
                input_size=p.emb_hidden_dim,
                hidden_size=p.df_hidden_dim,
                num_layers=p.df_num_layers,
                linear_groups=p.linear_groups,
                gru_skip=True,
                linear_act="relu",
            )

        # Optional skip
        self.df_skip: Optional[nn.Module] = None
        if p.df_gru_skip == "identity":
            self.df_skip = nn.Identity()
        elif p.df_gru_skip == "groupedlinear":
            self.df_skip = GroupedLinear(p.emb_hidden_dim, p.df_hidden_dim, p.linear_groups)

        # Output layers
        self.df_out = nn.Sequential(
            nn.Linear(self.df_hidden, self.df_bins * self.df_out_ch),
            nn.Tanh(),
        )

        # Alpha for DF blending
        self.df_fc_a = nn.Sequential(nn.Linear(self.df_hidden, 1), nn.Sigmoid())

    def __call__(self, emb: mx.array, c0: mx.array) -> Tuple[mx.array, mx.array]:
        """Decode DF coefficients.

        Returns:
            coef: DF coefficients (batch, time, df_bins, df_order, 2)
            alpha: Blending factor (batch, time, 1)
        """
        b, t, _ = emb.shape

        # GRU
        c, _ = self.df_gru(emb)

        # Skip connection
        if self.df_skip is not None:
            c = c + self.df_skip(emb)

        # Pathway contribution
        c0_proj = self.df_convp(c0)

        # Alpha output
        alpha = self.df_fc_a(c)

        # DF output
        c = self.df_out(c)
        c = c.reshape(b, t, self.df_bins, self.df_out_ch)

        # Add pathway contribution
        c = c + c0_proj

        # Reshape to (B, T, F, df_order, 2)
        c = c.reshape(b, t, self.df_bins, self.df_order, 2)

        return c, alpha


class DFNet2(nn.Module):
    """DeepFilterNet2 complete model.

    This is a multi-layer GRU-based model supporting both GroupedGRU
    and SqueezedGRU backends.

    Args:
        erb_fb_matrix: ERB filterbank matrix (erb_bins, freq_bins)
        erb_inv_fb: Inverse ERB filterbank matrix
        run_df: Whether to run DF stage
        p: Model parameters
    """

    def __init__(
        self,
        erb_fb_matrix: mx.array,
        erb_inv_fb: mx.array,
        run_df: bool = True,
        p: Optional[ModelParams2] = None,
    ):
        super().__init__()

        if p is None:
            p = ModelParams2()

        self.p = p
        self.run_df = run_df
        self.nb_df = p.nb_df
        self.df_order = p.df_order
        self.nb_erb = p.nb_erb
        self.freq_bins = p.fft_size // 2 + 1

        # Store filterbanks
        self._erb_fb = erb_fb_matrix
        self._erb_inv_fb = erb_inv_fb

        # Model components
        self.encoder = Encoder2(p)
        self.erb_decoder = ErbDecoder2(p)
        self.df_decoder = DfDecoder2(p)

        # DF operation
        self.df_op = DfOp(
            nb_df=p.nb_df,
            df_order=p.df_order,
            df_lookahead=p.df_lookahead,
        )

        # Post-filter settings
        self.post_filter = p.mask_pf
        self.pf_beta = p.pf_beta

    def __call__(
        self,
        spec: Tuple[mx.array, mx.array],
        feat_erb: mx.array,
        feat_spec: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            spec: Input spectrum as (real, imag) tuple, each (batch, time, freq)
            feat_erb: ERB features (batch, time, erb_bins)
            feat_spec: DF features (batch, time, df_bins, 2)

        Returns:
            Enhanced spectrum as (real, imag) tuple
        """
        spec_real, spec_imag = spec

        # Encode
        e0, e1, e2, e3, emb, c0, lsnr = self.encoder(feat_erb, feat_spec)

        # Decode ERB mask
        erb_mask = self.erb_decoder(emb, e3, e2, e1, e0)

        # Expand ERB mask to full spectrum
        erb_mask_full = mx.matmul(erb_mask, self._erb_fb)

        # Apply mask
        spec_real = spec_real * erb_mask_full
        spec_imag = spec_imag * erb_mask_full

        # DF stage
        if self.run_df:
            df_coef, alpha = self.df_decoder(emb, c0)

            # Apply DF operation
            spec_real, spec_imag = self.df_op(
                (spec_real, spec_imag),
                df_coef,
            )

        # Optional post-filter
        if self.post_filter:
            spec_real, spec_imag = self._apply_post_filter(spec_real, spec_imag, erb_mask_full)

        return spec_real, spec_imag

    def _apply_post_filter(self, real: mx.array, imag: mx.array, mask: mx.array) -> Tuple[mx.array, mx.array]:
        """Apply post-filter based on ERB mask."""
        beta = self.pf_beta
        pf_gain = (1.0 + beta) * mask / (1.0 + beta * mask)
        return real * pf_gain, imag * pf_gain


def init_model(
    sr: int = 48000,
    fft_size: int = 960,
    hop_size: int = 480,
    nb_bands: int = 32,
    run_df: bool = True,
    gru_type: str = "grouped",
) -> DFNet2:
    """Initialize a DFNet2 model.

    Args:
        sr: Sample rate
        fft_size: FFT size
        hop_size: Hop size
        nb_bands: Number of ERB bands
        run_df: Whether to run DF stage
        gru_type: GRU type ("grouped" or "squeeze")

    Returns:
        Initialized DFNet2 model
    """
    p = ModelParams2()
    p.sr = sr
    p.fft_size = fft_size
    p.hop_size = hop_size
    p.nb_erb = nb_bands
    p.gru_type = gru_type

    # Create filterbanks
    erb_fb_matrix = erb_fb(p.erb_widths, sr, inverse=False)
    erb_inv = erb_fb(p.erb_widths, sr, inverse=True)

    return DFNet2(erb_fb_matrix, erb_inv, run_df=run_df, p=p)


# Test
if __name__ == "__main__":
    print("Testing DFNet2...")

    # Create simple test
    p = ModelParams2()
    p.nb_erb = 32
    p.nb_df = 96
    p.df_order = 5
    p.fft_size = 960
    p.gru_type = "grouped"

    # Dummy filterbanks
    freq_bins = p.fft_size // 2 + 1
    erb_fb_matrix = mx.random.normal((p.nb_erb, freq_bins)) * 0.1
    erb_inv = mx.random.normal((freq_bins, p.nb_erb)) * 0.1

    model = DFNet2(erb_fb_matrix, erb_inv, run_df=True, p=p)

    # Test inputs
    batch, time = 2, 10
    spec_real = mx.random.normal((batch, time, freq_bins))
    spec_imag = mx.random.normal((batch, time, freq_bins))
    feat_erb = mx.random.normal((batch, time, p.nb_erb))
    feat_spec = mx.random.normal((batch, time, p.nb_df, 2))

    # Forward
    out_real, out_imag = model((spec_real, spec_imag), feat_erb, feat_spec)

    print(f"  GRU type: {p.gru_type}")
    print(f"  Input spec: ({spec_real.shape}, {spec_imag.shape})")
    print(f"  Output spec: ({out_real.shape}, {out_imag.shape})")

    # Count parameters
    def count_params(params):
        total = 0
        for v in params.values():
            if isinstance(v, dict):
                total += count_params(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        total += count_params(item)
            elif isinstance(v, mx.array):
                total += v.size
        return total

    n_params = count_params(model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Test with SqueezedGRU type
    print("\nTesting with SqueezedGRU...")
    p.gru_type = "squeeze"
    model_squeeze = DFNet2(erb_fb_matrix, erb_inv, run_df=True, p=p)
    out_real2, out_imag2 = model_squeeze((spec_real, spec_imag), feat_erb, feat_spec)
    print(f"  GRU type: {p.gru_type}")
    print(f"  Output spec: ({out_real2.shape}, {out_imag2.shape})")
    n_params2 = count_params(model_squeeze.parameters())
    print(f"  Parameters: {n_params2:,}")

    print("\n✓ DFNet2 test passed!")
