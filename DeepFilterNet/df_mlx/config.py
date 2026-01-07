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
    nb_df_hidden: int = 256  # DF hidden dimension
    df_n_layers: int = 3  # Number of DF layers
    df_dec_type: Literal["multi-res", "single-res", "adaptive"] = "multi-res"
    df_resolutions: List[int] = field(default_factory=lambda: [1, 2, 4])


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
    """Backbone (Mamba/GRU) parameters."""

    backbone_type: Literal["mamba", "gru"] = "mamba"
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

    # Parse [deepfilternet] section if present
    if "deepfilternet" in config:
        df_cfg = config["deepfilternet"]

        if "nb_df" in df_cfg:
            params.df.nb_df = int(df_cfg["nb_df"])
        if "df_order" in df_cfg:
            params.df.df_order = int(df_cfg["df_order"])
        if "df_lookahead" in df_cfg:
            params.df.df_lookahead = int(df_cfg["df_lookahead"])
        if "nb_erb" in df_cfg:
            params.erb.nb_erb = int(df_cfg["nb_erb"])
        if "conv_ch" in df_cfg:
            params.encoder.conv_channels = int(df_cfg["conv_ch"])
        if "emb_hidden_dim" in df_cfg:
            params.encoder.emb_hidden_dim = int(df_cfg["emb_hidden_dim"])
        if "df_hidden_dim" in df_cfg:
            params.df.nb_df_hidden = int(df_cfg["df_hidden_dim"])

    # Parse [audio] section if present
    if "audio" in config:
        audio_cfg = config["audio"]

        if "sr" in audio_cfg:
            params.audio.sr = int(audio_cfg["sr"])
        if "fft_size" in audio_cfg:
            params.audio.fft_size = int(audio_cfg["fft_size"])
        if "hop_size" in audio_cfg:
            params.audio.hop_size = int(audio_cfg["hop_size"])

    return params
