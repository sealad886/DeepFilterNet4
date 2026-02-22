"""Configuration management for MLX DeepFilterNet4.

This module provides configuration dataclasses that mirror the PyTorch
implementation's ModelParams4 for compatibility while being optimized
for MLX execution.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class DfParams:
    """Deep Filter parameters."""

    nb_df: int = 96  # Number of DF frequency bins
    df_order: int = 5  # DF filter order
    df_lookahead: int = 0  # DF lookahead frames
    conv_lookahead: int = 0  # Convolutional lookahead frames (encoder/decoder)
    nb_df_hidden: int = 256  # DF hidden dimension
    df_n_layers: int = 3  # Number of DF layers
    df_dec_type: Literal["multi-res", "single-res", "adaptive"] = "multi-res"
    df_resolutions: List[int] = field(default_factory=lambda: [1, 2, 4])
    mask_pf: bool = False  # Enable mask post-filter
    pf_beta: float = 0.02  # Post-filter beta parameter (attenuation strength)
    df_output_mode: Literal["coefficients", "complex_gain"] = "coefficients"


@dataclass
class ErbParams:
    """ERB (Equivalent Rectangular Bandwidth) parameters."""

    nb_erb: int = 32  # Number of ERB bands
    min_erb_width: int = 2  # Minimum ERB filter width
    erb_scale: str = "bark"  # ERB scale type
    erb_hidden: int = 64  # ERB hidden dimension
    nb_erb_hidden: int = 64  # Alias for erb_hidden


@dataclass
class EncoderParams:
    """Encoder parameters."""

    conv_channels: int = 64  # Base convolution channels
    conv_kernel: List[int] = field(default_factory=lambda: [1, 3])
    conv_stride: List[int] = field(default_factory=lambda: [1, 2])
    emb_hidden_dim: int = 256  # Embedding hidden dimension
    enc_linear_groups: int = 16  # Grouped linear groups
    enc_concat_all: bool = True  # Concatenate all encoder outputs
    num_enc_layers: int = 4  # Number of encoder layers


@dataclass
class BackboneParams:
    """Backbone (Mamba/GRU/Attention) parameters."""

    backbone_type: Literal["mamba", "gru", "attention"] = "mamba"
    nb_layers: int = 4  # Number of backbone layers
    hidden_dim: int = 256  # Hidden dimension
    d_state: int = 16  # State dimension for Mamba
    d_conv: int = 4  # Conv kernel size for Mamba
    expand_factor: int = 2  # Expansion factor for Mamba
    bidirectional: bool = False  # Bidirectional processing


@dataclass
class AudioParams:
    """Audio processing parameters."""

    sr: int = 48000  # Sample rate
    fft_size: int = 960  # FFT size
    hop_size: int = 480  # Hop size
    nb_freqs: int = 481  # Number of frequency bins (fft_size // 2 + 1)
    n_freqs: int = 481  # Alias for nb_freqs
    norm: str = "rms"  # Normalization type


@dataclass
class LsnrParams:
    """LSNR (Local SNR) estimation and dropout parameters."""

    lsnr_min: float = -15.0  # Minimum LSNR value (dB)
    lsnr_max: float = 40.0  # Maximum LSNR value (dB)
    lsnr_dropout_threshold: float = -10.0  # LSNR threshold for dropout (dB)
    lsnr_dropout: bool = False  # Enable LSNR dropout during training


@dataclass
class ModelParams4:
    """Complete model parameters for DeepFilterNet4.

    This configuration mirrors the PyTorch ModelParams4 for compatibility
    while being structured for MLX execution.
    """

    # Sub-configurations
    df: DfParams = field(default_factory=DfParams)
    erb: ErbParams = field(default_factory=ErbParams)
    encoder: EncoderParams = field(default_factory=EncoderParams)
    backbone: BackboneParams = field(default_factory=BackboneParams)
    audio: AudioParams = field(default_factory=AudioParams)
    lsnr: LsnrParams = field(default_factory=LsnrParams)

    # Convenience aliases (for compatibility with PyTorch impl)
    @property
    def nb_df(self) -> int:
        return self.df.nb_df

    @property
    def df_order(self) -> int:
        return self.df.df_order

    @property
    def df_lookahead(self) -> int:
        return self.df.df_lookahead

    @property
    def conv_lookahead(self) -> int:
        return self.df.conv_lookahead

    @property
    def nb_erb(self) -> int:
        return self.erb.nb_erb

    @property
    def sr(self) -> int:
        return self.audio.sr

    @property
    def fft_size(self) -> int:
        return self.audio.fft_size

    @property
    def hop_size(self) -> int:
        return self.audio.hop_size

    @property
    def n_freqs(self) -> int:
        return self.audio.n_freqs

    @property
    def conv_ch(self) -> int:
        return self.encoder.conv_channels

    @property
    def emb_hidden_dim(self) -> int:
        return self.encoder.emb_hidden_dim

    @property
    def df_hidden_dim(self) -> int:
        return self.df.nb_df_hidden

    @property
    def erb_hidden_dim(self) -> int:
        return self.erb.erb_hidden

    @property
    def enc_linear_groups(self) -> int:
        return self.encoder.enc_linear_groups

    @property
    def nb_df_layers(self) -> int:
        return self.df.df_n_layers

    @property
    def mask_pf(self) -> bool:
        return self.df.mask_pf

    @property
    def pf_beta(self) -> float:
        return self.df.pf_beta

    @property
    def df_output_mode(self) -> str:
        return self.df.df_output_mode


@dataclass
class LossConfig:
    """Loss function configuration.

    Multi-resolution STFT loss parameters matching PyTorch implementation.
    """

    # Multi-resolution STFT loss
    mrsl_enabled: bool = True  # Enable multi-resolution spectral loss
    mrsl_fft_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    mrsl_hop_sizes: Optional[List[int]] = None  # Defaults to fft_size // 4
    mrsl_gamma: float = 1.0  # Magnitude compression exponent
    mrsl_factor: float = 1.0  # Magnitude loss weight
    mrsl_f_complex: Optional[float] = None  # Complex loss weight (None = disabled)

    # Spectral loss weights
    spectral_mag_weight: float = 0.5  # Weight for magnitude loss
    spectral_complex_weight: float = 0.5  # Weight for complex loss

    # LSNR loss
    lsnr_weight: float = 0.1  # Weight for LSNR prediction loss
    lsnr_min: float = -15.0
    lsnr_max: float = 40.0


@dataclass
class TrainConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    max_steps: int = 100000
    grad_clip: float = 1.0

    # Batch/loader
    batch_size: int = 12
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000
    eval_every: int = 500

    # Mixed precision
    use_amp: bool = False  # MLX handles precision natively

    # Logging
    log_every: int = 100
    wandb_project: Optional[str] = None

    # LSNR dropout
    lsnr_dropout: bool = False
    lsnr_dropout_threshold: float = -10.0


def get_default_config() -> ModelParams4:
    """Get default model configuration."""
    return ModelParams4()


def load_config(path: str) -> ModelParams4:
    """Load configuration from file.

    Supports loading from PyTorch config.ini files for compatibility.

    Args:
        path: Path to configuration file

    Returns:
        ModelParams4 configuration object
    """
    import configparser

    config = configparser.ConfigParser()
    config.read(path)

    params = ModelParams4()

    def _apply_model_like_section(section_name: str) -> None:
        if section_name not in config:
            return
        sec = config[section_name]

        if "sr" in sec:
            params.audio.sr = sec.getint("sr")
        if "fft_size" in sec:
            params.audio.fft_size = sec.getint("fft_size")
        if "hop_size" in sec:
            params.audio.hop_size = sec.getint("hop_size")
        if "nb_erb" in sec:
            params.erb.nb_erb = sec.getint("nb_erb")
        if "nb_df" in sec:
            params.df.nb_df = sec.getint("nb_df")
        if "df_order" in sec:
            params.df.df_order = sec.getint("df_order")
        if "df_lookahead" in sec:
            params.df.df_lookahead = sec.getint("df_lookahead")
        if "conv_lookahead" in sec:
            params.df.conv_lookahead = sec.getint("conv_lookahead")
        if "conv_ch" in sec:
            params.encoder.conv_channels = sec.getint("conv_ch")
        if "emb_hidden_dim" in sec:
            params.encoder.emb_hidden_dim = sec.getint("emb_hidden_dim")
        if "enc_linear_groups" in sec:
            params.encoder.enc_linear_groups = sec.getint("enc_linear_groups")
        if "df_hidden_dim" in sec:
            params.df.nb_df_hidden = sec.getint("df_hidden_dim")
        if "nb_df_hidden" in sec:
            params.df.nb_df_hidden = sec.getint("nb_df_hidden")
        if "df_n_layers" in sec:
            params.df.df_n_layers = sec.getint("df_n_layers")
        if "nb_df_layers" in sec:
            params.df.df_n_layers = sec.getint("nb_df_layers")
        if "mask_pf" in sec:
            params.df.mask_pf = sec.getboolean("mask_pf")
        if "pf_beta" in sec:
            params.df.pf_beta = sec.getfloat("pf_beta")

    # Accept both old/new training config section names.
    for section in ("deepfilternet", "deepfilternet4", "df"):
        _apply_model_like_section(section)

    # [audio] can override transport-level audio params explicitly.
    if "audio" in config:
        _apply_model_like_section("audio")

    # Keep frequency aliases synchronized after any fft_size updates.
    n_freqs = params.audio.fft_size // 2 + 1
    params.audio.nb_freqs = n_freqs
    params.audio.n_freqs = n_freqs

    return params
