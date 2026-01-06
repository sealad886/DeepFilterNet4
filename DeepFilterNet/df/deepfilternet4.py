"""DeepFilterNet4 Model Architecture.

This module implements the DeepFilterNet4 architecture, featuring:
- Mamba state-space models for sequence modeling
- Hybrid time-frequency processing
- Multi-resolution deep filtering
- Adaptive filter order selection

The architecture builds upon DFNet3 while incorporating advances from
recent speech enhancement research, including MH-SENet's hybrid processing
and Mamba's efficient long-range modeling.
"""

from functools import partial
from typing import Final, List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn

import df.multiframe as MF
from df.config import Csv, DfParams, config
from df.mamba import SqueezedMamba
from df.modules import (
    Conv2dNormAct,
    ConvTranspose2dNormAct,
    GroupedLinearEinsum,
    Mask,
    SqueezedGRU_S,
    erb_fb,
    get_device,
)
from df.multiframe import AdaptiveOrderPredictor, MultiResolutionDF
from df.utils import as_complex
from libdf import DF

PI = 3.1415926535897932384626433


class ModelParams4(DfParams):
    """Model parameters for DeepFilterNet4.

    Extends DfParams with new configuration options for Mamba backbone,
    hybrid encoder, multi-resolution deep filtering, and adaptive order prediction.

    Configuration Section: [deepfilternet4]

    Standard Parameters (compatible with DFNet3):
        CONV_LOOKAHEAD (int): Convolutional lookahead frames. Default: 0
        CONV_CH (int): Base convolutional channels. Default: 16
        CONV_DEPTHWISE (bool): Use depthwise convolutions. Default: True
        CONVT_DEPTHWISE (bool): Use depthwise transposed convs. Default: True
        CONV_KERNEL (int,int): Encoder conv kernel size. Default: 1,3
        CONVT_KERNEL (int,int): Decoder conv kernel size. Default: 1,3
        CONV_KERNEL_INP (int,int): Input conv kernel size. Default: 3,3
        EMB_HIDDEN_DIM (int): Embedding hidden dimension. Default: 256
        EMB_NUM_LAYERS (int): Number of embedding layers. Default: 2
        EMB_GRU_SKIP_ENC (str): Encoder skip type. Default: "none"
        EMB_GRU_SKIP (str): Skip connection type. Default: "none"
        DF_HIDDEN_DIM (int): DF hidden dimension. Default: 256
        DF_GRU_SKIP (str): DF skip connection type. Default: "none"
        DF_PATHWAY_KERNEL_SIZE_T (int): DF pathway temporal kernel. Default: 1
        ENC_CONCAT (bool): Concatenate encoder outputs. Default: False
        DF_NUM_LAYERS (int): Number of DF layers. Default: 3
        DF_N_ITER (int): DF iterations. Default: 1
        LINEAR_GROUPS (int): Linear layer groups. Default: 1
        ENC_LINEAR_GROUPS (int): Encoder linear groups. Default: 16
        MASK_PF (bool): Use mask post-filter. Default: False
        PF_BETA (float): Post-filter beta parameter. Default: 0.02
        LSNR_DROPOUT (bool): Use LSNR-based dropout. Default: False
        DF_ORDER (int): Deep filter order. Default: 5

    DFNet4-Specific Parameters:
        BACKBONE (str): Sequence model backbone. Options: "mamba", "gru". Default: "mamba"
        MAMBA_D_STATE (int): Mamba SSM state dimension. Default: 16
        MAMBA_D_CONV (int): Mamba local conv width. Default: 4
        MAMBA_EXPAND (int): Mamba expansion factor. Default: 2
        USE_TIME_BRANCH (bool): Enable time-domain encoder branch. Default: False
        USE_PHASE_BRANCH (bool): Enable phase encoder branch. Default: True
        FUSION_TYPE (str): Cross-domain fusion type. Options: "attention", "simple". Default: "simple"
        USE_MULTI_RES_DF (bool): Enable multi-resolution deep filtering. Default: True
        DF_RESOLUTIONS (str): Multi-res DF config as "num_freqs,frame_size;...". Default: "96,5;48,3;24,2"
        ADAPTIVE_ORDER (bool): Enable adaptive filter order selection. Default: False
        MAX_DF_ORDER (int): Maximum DF order when adaptive. Default: 7
        MIN_DF_ORDER (int): Minimum DF order when adaptive. Default: 2
        MODEL_VARIANT (str): Model variant. Options: "full", "lite". Default: "full"

    Example Configuration:
        [deepfilternet4]
        BACKBONE = mamba
        USE_PHASE_BRANCH = true
        USE_MULTI_RES_DF = true
        DF_RESOLUTIONS = 96,5;48,3;24,2
        MODEL_VARIANT = full
    """

    section = "deepfilternet4"

    def __init__(self):
        super().__init__()
        # Standard conv params (compatible with DFNet3)
        self.conv_lookahead: int = config("CONV_LOOKAHEAD", cast=int, default=0, section=self.section)
        self.conv_ch: int = config("CONV_CH", cast=int, default=16, section=self.section)
        self.conv_depthwise: bool = config("CONV_DEPTHWISE", cast=bool, default=True, section=self.section)
        self.convt_depthwise: bool = config("CONVT_DEPTHWISE", cast=bool, default=True, section=self.section)
        self.conv_kernel: List[int] = config("CONV_KERNEL", cast=Csv(int), default=(1, 3), section=self.section)
        self.convt_kernel: List[int] = config("CONVT_KERNEL", cast=Csv(int), default=(1, 3), section=self.section)
        self.conv_kernel_inp: List[int] = config("CONV_KERNEL_INP", cast=Csv(int), default=(3, 3), section=self.section)

        # Embedding params
        self.emb_hidden_dim: int = config("EMB_HIDDEN_DIM", cast=int, default=256, section=self.section)
        self.emb_num_layers: int = config("EMB_NUM_LAYERS", cast=int, default=2, section=self.section)
        self.emb_gru_skip_enc: str = config("EMB_GRU_SKIP_ENC", default="none", section=self.section)
        self.emb_gru_skip: str = config("EMB_GRU_SKIP", default="none", section=self.section)

        # DF params
        self.df_hidden_dim: int = config("DF_HIDDEN_DIM", cast=int, default=256, section=self.section)
        self.df_gru_skip: str = config("DF_GRU_SKIP", default="none", section=self.section)
        self.df_pathway_kernel_size_t: int = config(
            "DF_PATHWAY_KERNEL_SIZE_T", cast=int, default=1, section=self.section
        )
        self.enc_concat: bool = config("ENC_CONCAT", cast=bool, default=False, section=self.section)
        self.df_num_layers: int = config("DF_NUM_LAYERS", cast=int, default=3, section=self.section)
        self.df_n_iter: int = config("DF_N_ITER", cast=int, default=1, section=self.section)
        self.lin_groups: int = config("LINEAR_GROUPS", cast=int, default=1, section=self.section)
        self.enc_lin_groups: int = config("ENC_LINEAR_GROUPS", cast=int, default=16, section=self.section)
        self.mask_pf: bool = config("MASK_PF", cast=bool, default=False, section=self.section)
        self.pf_beta: float = config("PF_BETA", cast=float, default=0.02, section=self.section)
        self.lsnr_dropout: bool = config("LSNR_DROPOUT", cast=bool, default=False, section=self.section)
        self.df_order: int = config("DF_ORDER", cast=int, default=5, section=self.section)

        # === DFNet4 specific params ===

        # Backbone selection
        self.backbone: str = config("BACKBONE", default="mamba", section=self.section)  # "mamba" or "gru"

        # Mamba params
        self.mamba_d_state: int = config("MAMBA_D_STATE", cast=int, default=16, section=self.section)
        self.mamba_d_conv: int = config("MAMBA_D_CONV", cast=int, default=4, section=self.section)
        self.mamba_expand: int = config("MAMBA_EXPAND", cast=int, default=2, section=self.section)

        # Hybrid encoder params
        self.use_time_branch: bool = config("USE_TIME_BRANCH", cast=bool, default=False, section=self.section)
        self.use_phase_branch: bool = config("USE_PHASE_BRANCH", cast=bool, default=True, section=self.section)
        self.fusion_type: str = config("FUSION_TYPE", default="simple", section=self.section)  # "attention" or "simple"

        # Multi-resolution DF params
        self.use_multi_res_df: bool = config("USE_MULTI_RES_DF", cast=bool, default=True, section=self.section)
        self.df_resolutions: str = config(
            "DF_RESOLUTIONS", default="96,5;48,3;24,2", section=self.section
        )  # num_freqs,frame_size pairs

        # Adaptive order params
        self.adaptive_order: bool = config("ADAPTIVE_ORDER", cast=bool, default=False, section=self.section)
        self.max_df_order: int = config("MAX_DF_ORDER", cast=int, default=7, section=self.section)
        self.min_df_order: int = config("MIN_DF_ORDER", cast=int, default=2, section=self.section)

        # Model variant
        self.model_variant: str = config("MODEL_VARIANT", default="full", section=self.section)  # "full" or "lite"

    def get_df_resolutions(self) -> List[Tuple[int, int]]:
        """Parse DF_RESOLUTIONS config string into list of (num_freqs, frame_size) tuples."""
        resolutions = []
        for res_str in self.df_resolutions.split(";"):
            parts = res_str.strip().split(",")
            if len(parts) == 2:
                resolutions.append((int(parts[0]), int(parts[1])))
        return resolutions if resolutions else [(self.nb_df, self.df_order)]

    @classmethod
    def generate_config_template(cls) -> str:
        """Generate a template configuration file for DFNet4.

        Returns:
            str: Template configuration in INI format.
        """
        return """# DeepFilterNet4 Configuration Template
#
# This configuration file defines parameters for DFNet4 speech enhancement.
# Parameters are organized by section.

[df]
# Core audio processing parameters
SR = 48000                    # Sampling rate in Hz
FFT_SIZE = 960               # FFT size in samples
HOP_SIZE = 480               # STFT hop size in samples
NB_ERB = 32                  # Number of ERB bands
NB_DF = 96                   # Number of deep filtering frequency bins
DF_ORDER = 5                 # Deep filter order (frame size)
DF_LOOKAHEAD = 0             # Deep filter lookahead frames

[train]
MODEL = deepfilternet4       # Model architecture to use

[deepfilternet4]
# === Architecture ===
CONV_CH = 16                 # Base convolutional channels
CONV_DEPTHWISE = true        # Use depthwise separable convolutions
CONV_KERNEL = 1,3            # Encoder conv kernel size (time, freq)
CONVT_KERNEL = 1,3           # Decoder conv kernel size (time, freq)
EMB_HIDDEN_DIM = 256         # Embedding hidden dimension
EMB_NUM_LAYERS = 2           # Number of embedding sequence layers
DF_HIDDEN_DIM = 256          # DF pathway hidden dimension
DF_NUM_LAYERS = 3            # Number of DF sequence layers

# === Backbone ===
BACKBONE = mamba             # Sequence model: "mamba" or "gru"
MAMBA_D_STATE = 16           # Mamba SSM state dimension
MAMBA_D_CONV = 4             # Mamba local convolution width
MAMBA_EXPAND = 2             # Mamba expansion factor

# === Hybrid Encoder ===
USE_TIME_BRANCH = false      # Enable time-domain waveform encoder
USE_PHASE_BRANCH = true      # Enable phase spectrum encoder
FUSION_TYPE = simple         # Cross-domain fusion: "simple" or "attention"

# === Multi-Resolution DF ===
USE_MULTI_RES_DF = true      # Enable multi-resolution deep filtering
DF_RESOLUTIONS = 96,5;48,3;24,2  # num_freqs,frame_size pairs

# === Adaptive Order ===
ADAPTIVE_ORDER = false       # Enable adaptive filter order selection
MAX_DF_ORDER = 7             # Maximum filter order when adaptive
MIN_DF_ORDER = 2             # Minimum filter order when adaptive

# === Model Variant ===
MODEL_VARIANT = full         # Model variant: "full" or "lite"
"""


class DfOutputReshape(nn.Module):
    """Reshape DF output coefficients from [B, T, F, O*2] to [B, O, T, F, 2]."""

    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        new_shape = list(coefs.shape)
        new_shape[-1] = -1
        new_shape.append(2)
        coefs = coefs.view(new_shape)
        coefs = coefs.permute(0, 3, 1, 2, 4)
        return coefs


class MultiResDfDecoder(nn.Module):
    """Decoder producing coefficients for multiple frequency resolutions.

    Uses a shared backbone with resolution-specific output heads to efficiently
    generate deep filter coefficients at multiple resolutions.

    Args:
        emb_dim: Input embedding dimension.
        hidden_dim: Hidden dimension for the shared backbone.
        num_layers: Number of layers in the shared backbone.
        resolutions: List of (num_freqs, frame_size) tuples.
        use_mamba: If True, use Mamba for sequence modeling. Otherwise use GRU.
        lin_groups: Number of groups for grouped linear layers.
        mamba_d_state: Mamba state dimension (if use_mamba=True).
        mamba_d_conv: Mamba convolution width (if use_mamba=True).
        conv_ch: Convolutional channels from encoder.
        nb_erb: Number of ERB bands.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        resolutions: List[Tuple[int, int]] = [(96, 5), (48, 3), (24, 2)],
        use_mamba: bool = True,
        lin_groups: int = 1,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        conv_ch: int = 16,
        nb_erb: int = 32,
    ):
        super().__init__()
        self.resolutions = resolutions
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # Input projection from encoder
        self.emb_in_dim = conv_ch * nb_erb // 4

        # Shared backbone
        if use_mamba:
            self.backbone = SqueezedMamba(
                input_size=self.emb_in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
            )
        else:
            self.backbone = SqueezedGRU_S(
                self.emb_in_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                gru_skip_op=None,
                linear_act_layer=partial(nn.ReLU, inplace=True),
            )

        # Resolution-specific output heads
        self.output_heads = nn.ModuleList()
        self.reshape_ops = nn.ModuleList()

        for num_freqs, frame_size in resolutions:
            out_dim = num_freqs * frame_size * 2  # Complex coefficients
            head = nn.Sequential(
                GroupedLinearEinsum(hidden_dim, out_dim, groups=lin_groups),
                nn.Tanh(),
            )
            self.output_heads.append(head)
            self.reshape_ops.append(DfOutputReshape(frame_size, num_freqs))

    def forward(
        self,
        emb: Tensor,
        c0: Tensor,
    ) -> List[Tensor]:
        """Generate DF coefficients for all resolutions.

        Args:
            emb: Encoder embedding [B, T, emb_in_dim]. This is the output from
                the encoder's embedding GRU, already at the right dimension.
            c0: DF pathway features [B, C, T, F] from encoder (currently unused,
                for future skip connection integration).

        Returns:
            coefs_list: List of coefficient tensors, one per resolution.
                Each tensor has shape [B, O, T, F_res, 2] where O is the
                frame_size for that resolution.
        """
        b, t, _ = emb.shape

        # Shared backbone processing using emb (like DFNet3)
        hidden, _ = self.backbone(emb)  # [B, T, hidden_dim]

        # Generate coefficients for each resolution
        coefs_list = []
        for head, reshape, (num_freqs, frame_size) in zip(self.output_heads, self.reshape_ops, self.resolutions):
            # Generate raw coefficients
            raw_coefs = head(hidden)  # [B, T, num_freqs * frame_size * 2]

            # Reshape to [B, T, F, O*2]
            raw_coefs = raw_coefs.view(b, t, num_freqs, frame_size * 2)

            # Reshape to [B, O, T, F, 2]
            coefs = reshape(raw_coefs)
            coefs_list.append(coefs)

        return coefs_list


class SingleResDfDecoder(nn.Module):
    """Standard single-resolution DF decoder (DFNet3 compatible).

    Provides backward compatibility with DFNet3's single-resolution approach.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        df_order: int = 5,
        df_bins: int = 96,
        use_mamba: bool = True,
        lin_groups: int = 1,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        conv_ch: int = 16,
        nb_erb: int = 32,
        df_pathway_kernel_size_t: int = 1,
    ):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins
        self.df_out_ch = df_order * 2

        self.emb_in_dim = conv_ch * nb_erb // 4

        # Conv pathway
        layer_width = conv_ch
        conv_layer = partial(Conv2dNormAct, separable=True, bias=False)
        kt = df_pathway_kernel_size_t
        self.df_convp = conv_layer(layer_width, self.df_out_ch, fstride=1, kernel_size=(kt, 1))

        # Backbone
        if use_mamba:
            self.backbone = SqueezedMamba(
                input_size=self.emb_in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
            )
        else:
            self.backbone = SqueezedGRU_S(
                self.emb_in_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                gru_skip_op=None,
                linear_act_layer=partial(nn.ReLU, inplace=True),
            )

        # Output head
        out_dim = df_bins * self.df_out_ch
        self.df_out = nn.Sequential(
            GroupedLinearEinsum(hidden_dim, out_dim, groups=lin_groups),
            nn.Tanh(),
        )

        self.reshape = DfOutputReshape(df_order, df_bins)

    def forward(
        self,
        emb: Tensor,
        c0: Tensor,
    ) -> Tensor:
        """Generate DF coefficients.

        Args:
            emb: Encoder embedding [B, T, emb_in_dim]. This is the output from
                the encoder's embedding GRU.
            c0: DF pathway features [B, C, T, F] (currently unused, for future
                skip connection like DFNet3's df_convp).

        Returns:
            coefs: Coefficients [B, O, T, F, 2].
        """
        b, t, _ = emb.shape

        hidden, _ = self.backbone(emb)  # [B, T, hidden_dim]

        raw_coefs = self.df_out(hidden)  # [B, T, F*O*2]
        raw_coefs = raw_coefs.view(b, t, self.df_bins, self.df_out_ch)  # [B, T, F, O*2]

        return self.reshape(raw_coefs)


class AdaptiveDfDecoder(nn.Module):
    """DF decoder with adaptive filter order selection.

    Combines multi-resolution DF with adaptive order prediction to select
    the optimal filter order based on input characteristics.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        df_bins: int = 96,
        max_order: int = 7,
        min_order: int = 2,
        use_mamba: bool = True,
        lin_groups: int = 1,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        conv_ch: int = 16,
        nb_erb: int = 32,
    ):
        super().__init__()
        self.df_bins = df_bins
        self.max_order = max_order
        self.min_order = min_order
        self.num_orders = max_order - min_order + 1

        self.emb_in_dim = conv_ch * nb_erb // 4

        # Order predictor
        self.order_predictor = AdaptiveOrderPredictor(
            emb_dim=hidden_dim,
            max_order=max_order,
            min_order=min_order,
        )

        # Backbone
        if use_mamba:
            self.backbone = SqueezedMamba(
                input_size=self.emb_in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
            )
        else:
            self.backbone = SqueezedGRU_S(
                self.emb_in_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                gru_skip_op=None,
                linear_act_layer=partial(nn.ReLU, inplace=True),
            )

        # Output heads for each possible order
        self.output_heads = nn.ModuleList()
        self.reshape_ops = nn.ModuleList()

        for order in range(min_order, max_order + 1):
            out_dim = df_bins * order * 2
            head = nn.Sequential(
                GroupedLinearEinsum(hidden_dim, out_dim, groups=lin_groups),
                nn.Tanh(),
            )
            self.output_heads.append(head)
            self.reshape_ops.append(DfOutputReshape(order, df_bins))

    def forward(
        self,
        emb: Tensor,
        c0: Tensor,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate DF coefficients with adaptive order.

        Args:
            emb: Encoder embedding [B, T, emb_in_dim]. This is the output from
                the encoder's embedding GRU.
            c0: DF pathway features [B, C, T, F] (currently unused, for future
                skip connection integration).
            temperature: Gumbel-Softmax temperature for order selection.

        Returns:
            coefs: Combined coefficients [B, max_order, T, F, 2].
            order_weights: Order selection weights [B, T, num_orders].
            predicted_order: Predicted order per frame [B, T].
        """
        b, t, _ = emb.shape

        hidden, _ = self.backbone(emb)

        # Predict order
        order_weights, predicted_order = self.order_predictor(hidden, temperature)

        # Generate coefficients for each order
        all_coefs = []
        for idx, (head, reshape) in enumerate(zip(self.output_heads, self.reshape_ops)):
            raw = head(hidden)
            order = self.min_order + idx  # Order for this head
            raw = raw.view(b, t, self.df_bins, order * 2)
            coefs = reshape(raw)  # [B, O, T, F, 2]

            # Pad to max_order if needed
            if order < self.max_order:
                pad_size = self.max_order - order
                coefs = F.pad(coefs, (0, 0, 0, 0, 0, 0, 0, pad_size))

            all_coefs.append(coefs)

        # Weighted combination based on order prediction
        # order_weights: [B, T, num_orders]
        # all_coefs: List of [B, max_order, T, F, 2]
        combined = torch.zeros(b, self.max_order, t, self.df_bins, 2, device=emb.device)
        for i, coefs in enumerate(all_coefs):
            weight = order_weights[:, :, i].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            combined = combined + weight * coefs

        return combined, order_weights, predicted_order


def init_model(df_state: Optional[DF] = None, run_df: bool = True, train_mask: bool = True):
    """Initialize DeepFilterNet4 model.

    Args:
        df_state: LibDF state for signal processing parameters.
        run_df: Whether to run deep filtering.
        train_mask: Whether to train with mask loss.

    Returns:
        Initialized DfNet4 model.
    """
    p = ModelParams4()
    if df_state is None:
        df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb = erb_fb(df_state.erb_widths(), p.sr, inverse=False)
    erb_inverse = erb_fb(df_state.erb_widths(), p.sr, inverse=True)

    if p.model_variant == "lite":
        model = DfNet4Lite(erb, erb_inverse, run_df, train_mask)
    else:
        model = DfNet4(erb, erb_inverse, run_df, train_mask)

    return model.to(device=get_device())


class ErbDecoder4(nn.Module):
    """ERB mask decoder for DFNet4.

    Similar to DFNet3's ErbDecoder but supports Mamba backbone.

    Args:
        conv_ch: Convolutional channels
        nb_erb: Number of ERB bands
        emb_hidden_dim: Embedding hidden dimension
        emb_num_layers: Number of layers (minus one used in encoder)
        use_mamba: Use Mamba instead of GRU
        lin_groups: Linear groups for efficiency
        mamba_d_state: Mamba state dimension
        mamba_d_conv: Mamba convolution width
    """

    def __init__(
        self,
        conv_ch: int = 16,
        nb_erb: int = 32,
        emb_hidden_dim: int = 256,
        emb_num_layers: int = 2,
        use_mamba: bool = True,
        lin_groups: int = 1,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
    ):
        super().__init__()

        assert nb_erb % 8 == 0, "nb_erb should be divisible by 8"

        self.emb_in_dim = conv_ch * nb_erb // 4
        self.emb_dim = emb_hidden_dim
        self.emb_out_dim = conv_ch * nb_erb // 4
        self.nb_erb = nb_erb

        # Sequence modeling layer
        if use_mamba:
            self.emb_seq = SqueezedMamba(
                input_size=self.emb_in_dim,
                hidden_size=self.emb_dim,
                output_size=self.emb_out_dim,
                num_layers=max(1, emb_num_layers - 1),
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
            )
        else:
            self.emb_seq = SqueezedGRU_S(
                self.emb_in_dim,
                self.emb_dim,
                output_size=self.emb_out_dim,
                num_layers=max(1, emb_num_layers - 1),
                batch_first=True,
                gru_skip_op=None,
                linear_groups=lin_groups,
                linear_act_layer=partial(nn.ReLU, inplace=True),
            )

        # Transposed convolutions for upsampling
        tconv_layer = partial(
            ConvTranspose2dNormAct,
            kernel_size=(1, 3),
            bias=False,
            separable=True,
        )
        conv_layer = partial(
            Conv2dNormAct,
            bias=False,
            separable=True,
        )

        # Pathway and transpose convolutions
        self.conv3p = conv_layer(conv_ch, conv_ch, kernel_size=1)
        self.convt3 = conv_layer(conv_ch, conv_ch, kernel_size=(1, 3))
        self.conv2p = conv_layer(conv_ch, conv_ch, kernel_size=1)
        self.convt2 = tconv_layer(conv_ch, conv_ch, fstride=2)
        self.conv1p = conv_layer(conv_ch, conv_ch, kernel_size=1)
        self.convt1 = tconv_layer(conv_ch, conv_ch, fstride=2)
        self.conv0p = conv_layer(conv_ch, conv_ch, kernel_size=1)
        self.conv0_out = conv_layer(conv_ch, 1, kernel_size=(1, 3), activation_layer=nn.Sigmoid)

    def forward(self, emb: Tensor, e3: Tensor, e2: Tensor, e1: Tensor, e0: Tensor) -> Tensor:
        """Decode ERB mask.

        Args:
            emb: Embedding from encoder [B, T, emb_in_dim]
            e3, e2, e1, e0: Skip connections from encoder

        Returns:
            mask: ERB mask [B, 1, T, E]
        """
        b, _, t, f8 = e3.shape

        # Sequence processing
        emb, _ = self.emb_seq(emb)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)  # [B, C, T, F/4]

        # Decoder with skip connections
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]

        return m


class Add(nn.Module):
    """Addition module for combining features."""

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return a + b


class Concat(nn.Module):
    """Concatenation module for combining features."""

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.cat((a, b), dim=-1)


class Encoder4(nn.Module):
    """Standard encoder for DFNet4 (non-hybrid variant).

    Similar to DFNet3's encoder but supports Mamba backbone.

    Args:
        conv_ch: Convolutional channels
        nb_erb: Number of ERB bands
        nb_df: Number of DF frequency bins
        emb_hidden_dim: Embedding hidden dimension
        enc_lin_groups: Linear groups for efficiency
        use_mamba: Use Mamba instead of GRU
        mamba_d_state: Mamba state dimension
        mamba_d_conv: Mamba convolution width
        enc_concat: Whether to concatenate ERB and DF embeddings
        lin_groups: Linear groups for output
        lsnr_min: Minimum LSNR value
        lsnr_max: Maximum LSNR value
    """

    def __init__(
        self,
        conv_ch: int = 16,
        nb_erb: int = 32,
        nb_df: int = 96,
        emb_hidden_dim: int = 256,
        enc_lin_groups: int = 16,
        use_mamba: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        enc_concat: bool = False,
        lin_groups: int = 1,
        lsnr_min: float = -15.0,
        lsnr_max: float = 40.0,
        conv_kernel_inp: Tuple[int, int] = (3, 3),
        conv_kernel: Tuple[int, int] = (1, 3),
    ):
        super().__init__()

        assert nb_erb % 4 == 0, "nb_erb should be divisible by 4"

        # ERB pathway
        self.erb_conv0 = Conv2dNormAct(1, conv_ch, kernel_size=conv_kernel_inp, bias=False, separable=True)
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
        self.df_conv0 = Conv2dNormAct(2, conv_ch, kernel_size=conv_kernel_inp, bias=False, separable=True)
        self.df_conv1 = conv_layer(fstride=2)

        # Embedding dimensions
        self.erb_bins = nb_erb
        self.emb_in_dim = conv_ch * nb_erb // 4
        self.emb_dim = emb_hidden_dim
        self.emb_out_dim = conv_ch * nb_erb // 4

        # DF to embedding projection
        df_fc_emb = GroupedLinearEinsum(conv_ch * nb_df // 2, self.emb_in_dim, groups=enc_lin_groups)
        self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU(inplace=True))

        # Combine ERB and DF embeddings
        if enc_concat:
            emb_in_dim_actual = self.emb_in_dim * 2
            self.combine = Concat()
        else:
            emb_in_dim_actual = self.emb_in_dim
            self.combine = Add()

        # Sequence modeling
        if use_mamba:
            self.emb_gru = SqueezedMamba(
                input_size=emb_in_dim_actual,
                hidden_size=self.emb_dim,
                output_size=self.emb_out_dim,
                num_layers=1,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
            )
        else:
            self.emb_gru = SqueezedGRU_S(
                emb_in_dim_actual,
                self.emb_dim,
                output_size=self.emb_out_dim,
                num_layers=1,
                batch_first=True,
                gru_skip_op=None,
                linear_groups=lin_groups,
                linear_act_layer=partial(nn.ReLU, inplace=True),
            )

        # LSNR estimation
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = lsnr_max - lsnr_min
        self.lsnr_offset = lsnr_min

    def forward(
        self, feat_erb: Tensor, feat_spec: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode ERB and spectral features.

        Args:
            feat_erb: ERB features [B, 1, T, E]
            feat_spec: Complex spectrogram [B, 2, T, F]

        Returns:
            e0, e1, e2, e3: Encoder intermediate outputs
            emb: Embedding [B, T, emb_out_dim]
            c0: DF pathway features
            lsnr: LSNR estimate [B, T, 1]
        """
        # ERB pathway
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, E]
        e1 = self.erb_conv1(e0)  # [B, C, T, E/2]
        e2 = self.erb_conv2(e1)  # [B, C, T, E/4]
        e3 = self.erb_conv3(e2)  # [B, C, T, E/4]

        # DF pathway
        c0 = self.df_conv0(feat_spec)  # [B, C, T, F]
        c1 = self.df_conv1(c0)  # [B, C, T, F/2]

        # Combine for embedding
        cemb = c1.permute(0, 2, 3, 1).flatten(2)  # [B, T, C*F/2]
        cemb = self.df_fc_emb(cemb)  # [B, T, emb_in_dim]

        emb = e3.permute(0, 2, 3, 1).flatten(2)  # [B, T, C*E/4]
        emb = self.combine(emb, cemb)

        # Sequence modeling
        emb, _ = self.emb_gru(emb)  # [B, T, emb_out_dim]

        # LSNR estimation
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset

        return e0, e1, e2, e3, emb, c0, lsnr


class DfNet4(nn.Module):
    """Full DeepFilterNet4 model.

    Integrates:
    - Mamba state-space models for sequence modeling
    - Hybrid time-frequency encoder (optional)
    - Multi-resolution deep filtering (optional)
    - ERB mask decoder
    - Post-filtering (optional)

    Args:
        erb_fb: ERB filterbank [E, F]
        erb_inv_fb: Inverse ERB filterbank [F, E]
        run_df: Whether to run deep filtering
        train_mask: Whether to train with mask loss
    """

    run_df: Final[bool]
    run_erb: Final[bool]
    lsnr_dropout: Final[bool]
    post_filter: Final[bool]
    post_filter_beta: Final[float]

    def __init__(
        self,
        erb_fb: Tensor,
        erb_inv_fb: Tensor,
        run_df: bool = True,
        train_mask: bool = True,
    ):
        super().__init__()

        p = ModelParams4()
        self.p = p

        # Core settings
        self.df_lookahead = p.df_lookahead
        self.nb_df = p.nb_df
        self.freq_bins: int = p.fft_size // 2 + 1
        self.erb_bins: int = p.nb_erb
        self.conv_ch = p.conv_ch
        self.use_mamba = p.backbone.lower() == "mamba"
        self.use_hybrid_encoder = p.use_time_branch or p.use_phase_branch
        self.use_multi_res_df = p.use_multi_res_df
        self.adaptive_order = p.adaptive_order

        # Feature padding
        if p.conv_lookahead > 0:
            assert p.conv_lookahead >= p.df_lookahead
            self.pad_feat = nn.ConstantPad2d((0, 0, -p.conv_lookahead, p.conv_lookahead), 0.0)
        else:
            self.pad_feat = nn.Identity()
        if p.df_lookahead > 0:
            self.pad_spec = nn.ConstantPad3d((0, 0, 0, 0, -p.df_lookahead, p.df_lookahead), 0.0)
        else:
            self.pad_spec = nn.Identity()

        # Register ERB filterbanks
        self.register_buffer("erb_fb", erb_fb)
        self.erb_inv_fb = erb_inv_fb

        # Encoder
        if self.use_hybrid_encoder:
            from df.hybrid_encoder import HybridEncoder

            self.enc = HybridEncoder(
                conv_ch=p.conv_ch,
                nb_erb=p.nb_erb,
                nb_df=p.nb_df,
                fft_size=p.fft_size,
                emb_hidden_dim=p.emb_hidden_dim,
                emb_num_layers=p.emb_num_layers,
                enc_lin_groups=p.enc_lin_groups,
                use_time_branch=p.use_time_branch,
                use_phase_branch=p.use_phase_branch,
                use_mamba=self.use_mamba,
                lsnr_min=p.lsnr_min,
                lsnr_max=p.lsnr_max,
                linear_groups=p.lin_groups,
            )
        else:
            # Standard encoder (similar to DFNet3)
            self.enc = Encoder4(
                conv_ch=p.conv_ch,
                nb_erb=p.nb_erb,
                nb_df=p.nb_df,
                emb_hidden_dim=p.emb_hidden_dim,
                enc_lin_groups=p.enc_lin_groups,
                use_mamba=self.use_mamba,
                mamba_d_state=p.mamba_d_state,
                mamba_d_conv=p.mamba_d_conv,
                enc_concat=p.enc_concat,
                lin_groups=p.lin_groups,
                lsnr_min=p.lsnr_min,
                lsnr_max=p.lsnr_max,
            )

        # ERB decoder
        self.erb_dec = ErbDecoder4(
            conv_ch=p.conv_ch,
            nb_erb=p.nb_erb,
            emb_hidden_dim=p.emb_hidden_dim,
            emb_num_layers=p.emb_num_layers,
            use_mamba=self.use_mamba,
            lin_groups=p.lin_groups,
            mamba_d_state=p.mamba_d_state,
            mamba_d_conv=p.mamba_d_conv,
        )

        # ERB mask application
        self.mask = Mask(erb_inv_fb)

        # Post-filter settings
        self.post_filter = p.mask_pf
        self.post_filter_beta = p.pf_beta

        # DF decoder
        if self.use_multi_res_df:
            resolutions = p.get_df_resolutions()
            self.df_dec = MultiResDfDecoder(
                emb_dim=p.emb_hidden_dim,
                hidden_dim=p.df_hidden_dim,
                num_layers=p.df_num_layers,
                resolutions=resolutions,
                use_mamba=self.use_mamba,
                lin_groups=p.lin_groups,
                mamba_d_state=p.mamba_d_state,
                mamba_d_conv=p.mamba_d_conv,
                conv_ch=p.conv_ch,
                nb_erb=p.nb_erb,
            )
            self.df_op = MultiResolutionDF(
                resolutions=resolutions,
                lookahead=self.df_lookahead,
            )
            self.df_order = max(fs for _, fs in resolutions)
        elif self.adaptive_order:
            self.df_dec = AdaptiveDfDecoder(
                emb_dim=p.emb_hidden_dim,
                hidden_dim=p.df_hidden_dim,
                num_layers=p.df_num_layers,
                df_bins=p.nb_df,
                max_order=p.max_df_order,
                min_order=p.min_df_order,
                use_mamba=self.use_mamba,
                lin_groups=p.lin_groups,
                mamba_d_state=p.mamba_d_state,
                mamba_d_conv=p.mamba_d_conv,
                conv_ch=p.conv_ch,
                nb_erb=p.nb_erb,
            )
            self.df_op = MF.DF(
                num_freqs=p.nb_df,
                frame_size=p.max_df_order,
                lookahead=self.df_lookahead,
            )
            self.df_order = p.max_df_order
        else:
            self.df_dec = SingleResDfDecoder(
                emb_dim=p.emb_hidden_dim,
                hidden_dim=p.df_hidden_dim,
                num_layers=p.df_num_layers,
                df_order=p.df_order,
                df_bins=p.nb_df,
                use_mamba=self.use_mamba,
                lin_groups=p.lin_groups,
                mamba_d_state=p.mamba_d_state,
                mamba_d_conv=p.mamba_d_conv,
                conv_ch=p.conv_ch,
                nb_erb=p.nb_erb,
                df_pathway_kernel_size_t=p.df_pathway_kernel_size_t,
            )
            self.df_op = MF.DF(
                num_freqs=p.nb_df,
                frame_size=p.df_order,
                lookahead=self.df_lookahead,
            )
            self.df_order = p.df_order

        # Run flags
        self.run_erb = p.nb_df + 1 < self.freq_bins
        if not self.run_erb:
            logger.warning("Running without ERB stage")
        self.run_df = run_df
        if not run_df:
            logger.warning("Running without DF stage")
        self.train_mask = train_mask
        self.lsnr_dropout = p.lsnr_dropout

    def forward(
        self,
        spec: Tensor,
        feat_erb: Tensor,
        feat_spec: Tensor,
        waveform: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass of DeepFilterNet4.

        Args:
            spec: Spectrum [B, 1, T, F, 2]
            feat_erb: ERB features [B, 1, T, E]
            feat_spec: Complex spectrogram features [B, 1, T, F', 2]
            waveform: Raw audio waveform [B, T_samples] (optional, for hybrid encoder)

        Returns:
            spec_e: Enhanced spectrum [B, 1, T, F, 2]
            m: ERB mask estimate [B, 1, T, E]
            lsnr: Local SNR estimate [B, T, 1]
            df_coefs: DF coefficients (shape depends on decoder type)
        """
        # Get batch and time dimensions
        b, _, t, _, _ = spec.shape

        # Prepare features
        feat_spec_in = feat_spec.squeeze(1).permute(0, 3, 1, 2)  # [B, 2, T, F']

        feat_erb = self.pad_feat(feat_erb)
        feat_spec_in = self.pad_feat(feat_spec_in)

        # Encoder
        if self.use_hybrid_encoder:
            e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec_in, waveform)
        else:
            e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec_in)

        # Initialize m to avoid unbound variable issues
        m: Tensor = torch.zeros((b, 1, t, self.erb_bins), device=spec.device)
        spec_m: Tensor = spec.clone()

        # LSNR dropout for training
        if self.lsnr_dropout and self.training:
            idcs = lsnr.squeeze() > -10.0
            emb_active = emb[:, idcs]
            e0_active = e0[:, :, idcs]
            e1_active = e1[:, :, idcs]
            e2_active = e2[:, :, idcs]
            e3_active = e3[:, :, idcs]
            c0_active = c0[:, :, idcs]
        else:
            idcs = None
            emb_active = emb
            e0_active, e1_active, e2_active, e3_active = e0, e1, e2, e3
            c0_active = c0

        # ERB mask
        if self.run_erb:
            m_active = self.erb_dec(emb_active, e3_active, e2_active, e1_active, e0_active)
            if self.lsnr_dropout and self.training and idcs is not None:
                m[:, :, idcs] = m_active
            else:
                m = m_active
            spec_m = self.mask(spec, m)
        else:
            m = torch.zeros((), device=spec.device)
            spec_m = torch.zeros_like(spec)

        # Deep filtering
        if self.run_df:
            if self.use_multi_res_df:
                coefs_list = self.df_dec(emb_active, c0_active)
                if self.lsnr_dropout and self.training and idcs is not None:
                    # Handle LSNR dropout for multi-res
                    df_coefs_full = [
                        torch.zeros(b, c.shape[1], t, c.shape[3], 2, device=spec.device) for c in coefs_list
                    ]
                    for i, (c_full, c_active) in enumerate(zip(df_coefs_full, coefs_list)):
                        c_full[:, :, idcs] = c_active
                    spec_e = self.df_op(spec.clone(), df_coefs_full)
                else:
                    spec_e = self.df_op(spec.clone(), coefs_list)
                df_coefs = coefs_list[0]  # Return primary resolution for backward compat
            elif self.adaptive_order:
                df_coefs, order_weights, predicted_order = self.df_dec(emb_active, c0_active)
                if self.lsnr_dropout and self.training and idcs is not None:
                    df_coefs_full = torch.zeros(b, self.df_order, t, self.nb_df, 2, device=spec.device)
                    df_coefs_full[:, :, idcs] = df_coefs
                    df_coefs = df_coefs_full
                spec_e = self.df_op(spec.clone(), df_coefs)
            else:
                df_coefs = self.df_dec(emb_active, c0_active)
                if self.lsnr_dropout and self.training and idcs is not None:
                    df_coefs_full = torch.zeros(b, self.df_order, t, self.nb_df, 2, device=spec.device)
                    df_coefs_full[:, :, idcs] = df_coefs
                    df_coefs = df_coefs_full
                spec_e = self.df_op(spec.clone(), df_coefs)

            # Apply ERB mask to frequencies above DF range
            spec_e[..., self.nb_df :, :] = spec_m[..., self.nb_df :, :]
        else:
            df_coefs = torch.zeros((), device=spec.device)
            spec_e = spec_m

        # Post-filter
        if self.post_filter:
            beta = self.post_filter_beta
            eps = 1e-12
            mask = (as_complex(spec_e).abs() / as_complex(spec).abs().add(eps)).clamp(eps, 1)
            mask_sin = mask * torch.sin(PI * mask / 2).clamp_min(eps)
            pf = (1 + beta) / (1 + beta * mask.div(mask_sin).pow(2))
            spec_e = spec_e * pf.unsqueeze(-1)

        return spec_e, m, lsnr, df_coefs


class DfNet4Lite(nn.Module):
    """Lightweight DeepFilterNet4 variant with ~50% parameter reduction.

    Key differences from full DfNet4:
    - Reduced channel count (conv_ch: 16 -> 8)
    - Reduced hidden dimensions (emb_hidden_dim: 256 -> 128)
    - Reduced layers (emb_num_layers: 2 -> 1, df_num_layers: 3 -> 2)
    - Single resolution DF only (no multi-res)
    - No hybrid encoder branches
    - Reduced Mamba state dimension

    Target: ~1.3M parameters (vs ~2.6M for full)

    Args:
        erb_fb: ERB filterbank [E, F]
        erb_inv_fb: Inverse ERB filterbank [F, E]
        run_df: Whether to run deep filtering
        train_mask: Whether to train with mask loss
    """

    run_df: Final[bool]
    run_erb: Final[bool]
    lsnr_dropout: Final[bool]
    post_filter: Final[bool]
    post_filter_beta: Final[float]

    def __init__(
        self,
        erb_fb: Tensor,
        erb_inv_fb: Tensor,
        run_df: bool = True,
        train_mask: bool = True,
    ):
        super().__init__()

        p = ModelParams4()
        self.p = p

        # Lite-specific parameter overrides
        lite_conv_ch = max(8, p.conv_ch // 2)
        lite_emb_hidden_dim = max(128, p.emb_hidden_dim // 2)
        lite_df_hidden_dim = max(128, p.df_hidden_dim // 2)
        lite_emb_num_layers = max(1, p.emb_num_layers - 1)
        lite_df_num_layers = max(2, p.df_num_layers - 1)
        lite_mamba_d_state = max(8, p.mamba_d_state // 2)

        self.nb_df = p.nb_df
        self.df_order = p.df_order

        # Control flags
        self.run_df = run_df
        self.run_erb = train_mask
        self.lsnr_dropout = p.lsnr_dropout and train_mask
        self.lsnr_droput_thresh = 30.0  # LSNR threshold for dropout
        self.post_filter = p.mask_pf
        self.post_filter_beta = p.pf_beta

        # Filterbanks
        self.register_buffer("erb_fb", erb_fb)
        self.register_buffer("erb_inv_fb", erb_inv_fb)

        # Lightweight standard encoder (no hybrid features)
        self.enc = Encoder4Lite(
            conv_ch=lite_conv_ch,
            nb_erb=p.nb_erb,
            nb_df=p.nb_df,
            emb_hidden_dim=lite_emb_hidden_dim,
            enc_lin_groups=p.enc_lin_groups,
            use_mamba=p.backbone.lower() == "mamba",
            mamba_d_state=lite_mamba_d_state,
            mamba_d_conv=p.mamba_d_conv,
            enc_concat=p.enc_concat,
            lin_groups=p.lin_groups,
            lsnr_min=p.lsnr_min,
            lsnr_max=p.lsnr_max,
        )

        # ERB decoder
        if self.run_erb:
            self.erb_dec = ErbDecoder4Lite(
                conv_ch=lite_conv_ch,
                nb_erb=p.nb_erb,
                emb_hidden_dim=lite_emb_hidden_dim,
                emb_num_layers=lite_emb_num_layers,
                use_mamba=p.backbone.lower() == "mamba",
                lin_groups=p.lin_groups,
                mamba_d_state=lite_mamba_d_state,
                mamba_d_conv=p.mamba_d_conv,
            )
            self.mask = Mask(erb_inv_fb, post_filter=False)

        # Single resolution DF decoder (Lite doesn't use multi-res or adaptive)
        if self.run_df:
            self.df_dec = SingleResDfDecoder(
                emb_dim=lite_emb_hidden_dim,
                hidden_dim=lite_df_hidden_dim,
                num_layers=lite_df_num_layers,
                df_bins=p.nb_df,
                df_order=p.df_order,
                use_mamba=p.backbone.lower() == "mamba",
                lin_groups=p.lin_groups,
                mamba_d_state=lite_mamba_d_state,
                mamba_d_conv=p.mamba_d_conv,
                conv_ch=lite_conv_ch,
                nb_erb=p.nb_erb,
            )
            self.df_op = MF.DF(num_freqs=p.nb_df, frame_size=p.df_order, lookahead=p.df_lookahead)

    def forward(
        self,
        spec: Tensor,
        feat_erb: Tensor,
        feat_spec: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            spec: Complex spectrogram [B, 1, T, F, 2]
            feat_erb: ERB features [B, 1, T, E]
            feat_spec: Spectral features [B, 2, T, F]

        Returns:
            spec_e: Enhanced spectrogram [B, 1, T, F, 2]
            m: ERB mask [B, 1, T, E]
            lsnr: Local SNR estimate [B, T, 1]
            df_coefs: DF coefficients [B, O, T, F, 2]
        """
        b, _, t, f, _ = spec.shape

        # Encode
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)

        # Initialize outputs
        m = torch.zeros((b, 1, t, self.p.nb_erb), device=spec.device)

        # LSNR dropout (training only)
        if self.lsnr_dropout and self.training:
            idcs = lsnr.squeeze(-1) > self.lsnr_droput_thresh
            idcs = idcs.any(dim=0)  # [T]
            if idcs.any():
                emb_active = emb[:, idcs]
                e0_active = e0[:, :, idcs]
                e1_active = e1[:, :, idcs]
                e2_active = e2[:, :, idcs]
                e3_active = e3[:, :, idcs]
                c0_active = c0[:, :, idcs]
            else:
                idcs = None
                emb_active = emb
                e0_active, e1_active, e2_active, e3_active = e0, e1, e2, e3
                c0_active = c0
        else:
            idcs = None
            emb_active = emb
            e0_active, e1_active, e2_active, e3_active = e0, e1, e2, e3
            c0_active = c0

        # ERB mask
        if self.run_erb:
            m_active = self.erb_dec(emb_active, e3_active, e2_active, e1_active, e0_active)
            if self.lsnr_dropout and self.training and idcs is not None:
                m[:, :, idcs] = m_active
            else:
                m = m_active
            spec_m = self.mask(spec, m)
        else:
            m = torch.zeros((), device=spec.device)
            spec_m = torch.zeros_like(spec)

        # Deep filtering
        if self.run_df:
            df_coefs = self.df_dec(emb_active, c0_active)
            if self.lsnr_dropout and self.training and idcs is not None:
                df_coefs_full = torch.zeros(b, self.df_order, t, self.nb_df, 2, device=spec.device)
                df_coefs_full[:, :, idcs] = df_coefs
                df_coefs = df_coefs_full
            spec_e = self.df_op(spec.clone(), df_coefs)

            # Apply ERB mask to frequencies above DF range
            spec_e[..., self.nb_df :, :] = spec_m[..., self.nb_df :, :]
        else:
            df_coefs = torch.zeros((), device=spec.device)
            spec_e = spec_m

        # Post-filter
        if self.post_filter:
            beta = self.post_filter_beta
            eps = 1e-12
            mask = (as_complex(spec_e).abs() / as_complex(spec).abs().add(eps)).clamp(eps, 1)
            mask_sin = mask * torch.sin(PI * mask / 2).clamp_min(eps)
            pf = (1 + beta) / (1 + beta * mask.div(mask_sin).pow(2))
            spec_e = spec_e * pf.unsqueeze(-1)

        return spec_e, m, lsnr, df_coefs


class Encoder4Lite(nn.Module):
    """Lightweight encoder for DfNet4Lite.

    Reduced parameters compared to Encoder4:
    - Smaller conv_ch (8 vs 16)
    - Smaller emb_hidden_dim (128 vs 256)
    - Smaller mamba_d_state (8 vs 16)
    """

    def __init__(
        self,
        conv_ch: int = 8,
        nb_erb: int = 32,
        nb_df: int = 96,
        emb_hidden_dim: int = 128,
        enc_lin_groups: int = 16,
        use_mamba: bool = True,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 4,
        enc_concat: bool = False,
        lin_groups: int = 1,
        lsnr_min: float = -15.0,
        lsnr_max: float = 40.0,
        conv_kernel_inp: Tuple[int, int] = (3, 3),
        conv_kernel: Tuple[int, int] = (1, 3),
    ):
        super().__init__()

        assert nb_erb % 4 == 0, "nb_erb should be divisible by 4"

        # ERB pathway
        self.erb_conv0 = Conv2dNormAct(1, conv_ch, kernel_size=conv_kernel_inp, bias=False, separable=True)
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
        self.df_conv0 = Conv2dNormAct(2, conv_ch, kernel_size=conv_kernel_inp, bias=False, separable=True)
        self.df_conv1 = conv_layer(fstride=2)

        # Embedding dimensions
        self.erb_bins = nb_erb
        self.emb_in_dim = conv_ch * nb_erb // 4
        self.emb_dim = emb_hidden_dim
        self.emb_out_dim = conv_ch * nb_erb // 4

        # DF to embedding projection
        df_fc_emb = GroupedLinearEinsum(conv_ch * nb_df // 2, self.emb_in_dim, groups=enc_lin_groups)
        self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU(inplace=True))

        # Combine ERB and DF embeddings
        if enc_concat:
            emb_in_dim_actual = self.emb_in_dim * 2
            self.combine = Concat()
        else:
            emb_in_dim_actual = self.emb_in_dim
            self.combine = Add()

        # Sequence modeling
        if use_mamba:
            self.emb_gru = SqueezedMamba(
                input_size=emb_in_dim_actual,
                hidden_size=self.emb_dim,
                output_size=self.emb_out_dim,
                num_layers=1,  # Lite uses fewer layers
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
            )
        else:
            self.emb_gru = SqueezedGRU_S(
                emb_in_dim_actual,
                self.emb_dim,
                output_size=self.emb_out_dim,
                num_layers=1,  # Lite uses fewer layers
                batch_first=True,
                gru_skip_op=None,
                linear_groups=lin_groups,
                linear_act_layer=partial(nn.ReLU, inplace=True),
            )

        # LSNR estimation
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = lsnr_max - lsnr_min
        self.lsnr_offset = lsnr_min

    def forward(
        self, feat_erb: Tensor, feat_spec: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode ERB and spectral features."""
        # ERB pathway
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, E]
        e1 = self.erb_conv1(e0)  # [B, C, T, E/2]
        e2 = self.erb_conv2(e1)  # [B, C, T, E/4]
        e3 = self.erb_conv3(e2)  # [B, C, T, E/4]

        # DF pathway
        c0 = self.df_conv0(feat_spec)  # [B, C, T, F]
        c1 = self.df_conv1(c0)  # [B, C, T, F/2]

        # Combine for embedding
        cemb = c1.permute(0, 2, 3, 1).flatten(2)  # [B, T, C*F/2]
        cemb = self.df_fc_emb(cemb)  # [B, T, emb_in_dim]

        emb = e3.permute(0, 2, 3, 1).flatten(2)  # [B, T, C*E/4]
        emb = self.combine(emb, cemb)

        # Sequence modeling
        emb, _ = self.emb_gru(emb)  # [B, T, emb_out_dim]

        # LSNR estimation
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset

        return e0, e1, e2, e3, emb, c0, lsnr


class ErbDecoder4Lite(nn.Module):
    """Lightweight ERB mask decoder for DfNet4Lite."""

    def __init__(
        self,
        conv_ch: int = 8,
        nb_erb: int = 32,
        emb_hidden_dim: int = 128,
        emb_num_layers: int = 1,
        use_mamba: bool = True,
        lin_groups: int = 1,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 4,
    ):
        super().__init__()

        assert nb_erb % 8 == 0, "nb_erb should be divisible by 8"

        self.emb_in_dim = conv_ch * nb_erb // 4
        self.emb_dim = emb_hidden_dim
        self.emb_out_dim = conv_ch * nb_erb // 4
        self.nb_erb = nb_erb

        # Sequence modeling layer
        if use_mamba:
            self.emb_seq = SqueezedMamba(
                input_size=self.emb_in_dim,
                hidden_size=self.emb_dim,
                output_size=self.emb_out_dim,
                num_layers=emb_num_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
            )
        else:
            self.emb_seq = SqueezedGRU_S(
                self.emb_in_dim,
                self.emb_dim,
                output_size=self.emb_out_dim,
                num_layers=emb_num_layers,
                batch_first=True,
                gru_skip_op=None,
                linear_groups=lin_groups,
                linear_act_layer=partial(nn.ReLU, inplace=True),
            )

        # Transposed convolutions for upsampling
        tconv_layer = partial(
            ConvTranspose2dNormAct,
            kernel_size=(1, 3),
            bias=False,
            separable=True,
        )
        conv_layer = partial(
            Conv2dNormAct,
            bias=False,
            separable=True,
        )

        # Pathway and transpose convolutions
        self.conv3p = conv_layer(conv_ch, conv_ch, kernel_size=1)
        self.convt3 = conv_layer(conv_ch, conv_ch, kernel_size=(1, 3))
        self.conv2p = conv_layer(conv_ch, conv_ch, kernel_size=1)
        self.convt2 = tconv_layer(conv_ch, conv_ch, fstride=2)
        self.conv1p = conv_layer(conv_ch, conv_ch, kernel_size=1)
        self.convt1 = tconv_layer(conv_ch, conv_ch, fstride=2)
        self.conv0p = conv_layer(conv_ch, conv_ch, kernel_size=1)
        self.conv0_out = conv_layer(conv_ch, 1, kernel_size=(1, 3), activation_layer=nn.Sigmoid)

    def forward(self, emb: Tensor, e3: Tensor, e2: Tensor, e1: Tensor, e0: Tensor) -> Tensor:
        """Decode ERB mask."""
        b, _, t, f8 = e3.shape

        # Sequence processing
        emb, _ = self.emb_seq(emb)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)  # [B, C, T, F/4]

        # Decoder with skip connections
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]

        return m
