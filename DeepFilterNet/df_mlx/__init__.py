"""DeepFilterNet4 MLX Implementation.

This package provides a native MLX implementation of DeepFilterNet4 for
Apple Silicon, offering significantly faster training and inference compared
to PyTorch MPS backend.

Architecture Overview:
- Mamba state-space models for sequence modeling (linear O(n) complexity)
- Hybrid time-frequency processing with ERB filterbank
- Multi-resolution deep filtering for speech enhancement
- Optimized for Apple Silicon unified memory architecture

Usage:
    from df_mlx import DfNet4, init_model
    from df_mlx.train import train, Trainer
    from df_mlx.config import ModelParams4, TrainConfig

    # Initialize model
    model = init_model()

    # Training
    trainer = Trainer(model, TrainConfig())
    trainer.train(train_loader, val_loader)

    # Inference
    enhanced = model(noisy_spec, feat_erb, feat_spec)

    # End-to-end enhancement
    enhanced_audio = model.enhance(noisy_audio)

Note:
    This is a separate MLX implementation that does NOT modify the existing
    PyTorch codebase. Model weights can be converted between PyTorch and MLX
    using the provided conversion utilities:

        from df_mlx.train import load_pytorch_checkpoint
        model = load_pytorch_checkpoint(model, "path/to/checkpoint.pth")

Requirements:
    - Apple Silicon Mac (M1/M2/M3)
    - mlx >= 0.5.0
    - numpy

Optional:
    - soundfile (for audio I/O)
    - resampy (for resampling)
    - pesq (for PESQ metric)
"""

__version__ = "0.1.0"

# Check MLX availability
try:
    import mlx.core as mx  # noqa: F401

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

if MLX_AVAILABLE:
    # Re-export public API - noqa for intentional re-exports
    from .config import ModelParams4, TrainConfig, get_default_config, load_config  # noqa: F401
    from .datastore import DatastoreConfig, MLXDataLoader, MLXDatastoreWriter, StreamingMLXDataLoader  # noqa: F401
    from .mamba import BidirectionalMamba, GroupedLinear, Mamba, MambaBlock, SqueezedMamba  # noqa: F401
    from .model import DfNet4, DfNet4Lite, count_parameters, init_model, model_summary  # noqa: F401
    from .modules import ComplexMask, Conv2dNormAct, ConvTranspose2dNormAct, DfOp, ErbFilterbank  # noqa: F401
    from .modules import GroupedLinear as GroupedLinearModule  # noqa: F401
    from .modules import Mask, erb_fb  # noqa: F401
    from .ops import erb_fb as make_erb_fb  # noqa: F401
    from .ops import erb_transform, istft, stft  # noqa: F401
    from .train import (  # noqa: F401
        Trainer,
        convert_pytorch_weights,
        load_pytorch_checkpoint,
        multi_resolution_stft_loss,
        spectral_loss,
        train,
    )
    from .utils import (  # noqa: F401
        AudioDataset,
        benchmark_model,
        compute_snr,
        create_dataloader,
        extract_features,
        load_audio,
        save_audio,
    )

    __all__ = [
        # Model
        "DfNet4",
        "DfNet4Lite",
        "init_model",
        "count_parameters",
        "model_summary",
        # Mamba
        "MambaBlock",
        "Mamba",
        "SqueezedMamba",
        "BidirectionalMamba",
        # Modules
        "Conv2dNormAct",
        "ConvTranspose2dNormAct",
        "GroupedLinear",
        "DfOp",
        "Mask",
        "ComplexMask",
        "ErbFilterbank",
        "erb_fb",
        # Config
        "ModelParams4",
        "TrainConfig",
        "get_default_config",
        "load_config",
        # Operations
        "stft",
        "istft",
        "make_erb_fb",
        "erb_transform",
        # Training
        "Trainer",
        "train",
        "spectral_loss",
        "multi_resolution_stft_loss",
        "load_pytorch_checkpoint",
        "convert_pytorch_weights",
        # Utils
        "load_audio",
        "save_audio",
        "extract_features",
        "AudioDataset",
        "create_dataloader",
        "compute_snr",
        "benchmark_model",
        # Datastore
        "DatastoreConfig",
        "MLXDataLoader",
        "MLXDatastoreWriter",
        "StreamingMLXDataLoader",
        # Version
        "__version__",
        "MLX_AVAILABLE",
    ]
else:
    # MLX not available - provide stub
    __all__ = ["MLX_AVAILABLE", "__version__"]

    def _mlx_not_available(*args, **kwargs):
        raise ImportError(
            "MLX is not available. MLX only works on Apple Silicon Macs. " "Install with: pip install mlx"
        )

    # Create stubs for common imports
    DfNet4 = _mlx_not_available
    init_model = _mlx_not_available
    train = _mlx_not_available
