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
    from .checkpoint import (  # noqa: F401
        CheckpointManager,
        CheckpointState,
        PatienceState,
        check_patience,
        load_checkpoint,
        read_patience,
        save_checkpoint,
        write_patience,
    )
    from .config import ModelParams4, TrainConfig, get_default_config, load_config  # noqa: F401

    # Checkpoint conversion utilities
    from .convert import convert_and_save  # noqa: F401
    from .convert import convert_dfnet3_checkpoint  # noqa: F401
    from .convert import convert_generic_checkpoint  # noqa: F401
    from .convert import load_mlx_checkpoint  # noqa: F401
    from .convert import save_mlx_checkpoint  # noqa: F401
    from .convert import load_pytorch_checkpoint as load_pytorch_ckpt  # noqa: F401
    from .datastore import DatastoreConfig, MLXDataLoader, MLXDatastoreWriter, StreamingMLXDataLoader  # noqa: F401

    # Legacy model architectures (GRU-based)
    from .deepfilternet import DFNet1, ModelParams1  # noqa: F401
    from .deepfilternet import init_model as init_dfnet1  # noqa: F401
    from .deepfilternet2 import DFNet2, ModelParams2  # noqa: F401
    from .deepfilternet2 import init_model as init_dfnet2  # noqa: F401
    from .deepfilternet3 import DFNet3, ModelParams3, init_dfnet3  # noqa: F401

    # Multi-frame DFNet model
    from .deepfilternetmf import (  # noqa: F401
        DfDecoderMF,
        DFNetMF,
        EncoderMF,
        ErbDecoderMF,
        ModelParamsMF,
        create_dfnetmf,
    )
    from .discriminator import (  # noqa: F401
        CombinedDiscriminator,
        MultiPeriodDiscriminator,
        MultiScaleDiscriminator,
        PeriodDiscriminator,
        ScaleDiscriminator,
        SpectralDiscriminator,
        compute_discriminator_loss,
        compute_generator_loss,
    )
    from .evaluation import (  # noqa: F401
        EvaluationResults,
        Metric,
        MetricResult,
        PESQMetric,
        SegmentalSNRMetric,
        SiSDRMetric,
        SNRMetric,
        STOIMetric,
        ValidationMetrics,
        compare_before_after,
        evaluate_batch,
        evaluate_single,
        get_metric_factory,
        log_improvement,
        quick_eval,
        segmental_snr,
        si_sdr,
        snr,
    )

    # New modules for parity with df/
    from .loss import (  # noqa: F401
        CombinedLoss,
        DfAlphaLoss,
        FeatureMatchingLoss,
        MaskLoss,
        SegmentalSiSdrLoss,
        SiSdrLoss,
        SpectralLoss,
        discriminator_loss,
        generator_loss,
    )
    from .lr import (  # noqa: F401
        CosineScheduler,
        ExponentialDecayScheduler,
        LinearDecayScheduler,
        WarmupScheduler,
        cosine_scheduler,
        create_scheduler,
    )
    from .mamba import BidirectionalMamba, GroupedLinear, Mamba, MambaBlock, SqueezedMamba  # noqa: F401
    from .model import DfNet4, DfNet4Lite, count_parameters, init_model, model_summary  # noqa: F401
    from .modules import (  # noqa: F401
        ComplexMask,
        Conv2dNormAct,
        ConvTranspose2dNormAct,
        DfOp,
        ErbFilterbank,
        GroupedGRU,
    )
    from .modules import GroupedLinear as GroupedLinearModule  # noqa: F401
    from .modules import Mask, SqueezedGRU, SqueezedGRU_S, erb_fb  # noqa: F401

    # Multi-frame processing modules
    from .multiframe import CRM, DF, DFreal, MfMvdr, MfWf, MultiFrameModule, MultiResolutionDF  # noqa: F401
    from .ops import erb_fb as make_erb_fb  # noqa: F401
    from .ops import erb_fb_and_inverse, erb_transform, istft, stft  # noqa: F401
    from .stoi import stoi, stoi_loss, stoi_numpy  # noqa: F401
    from .train import (  # noqa: F401
        Trainer,
        convert_pytorch_weights,
        load_pytorch_checkpoint,
        multi_resolution_stft_loss,
        spectral_loss,
        train,
    )
    from .train_gan import GANConfig, GANTrainer, train_gan  # noqa: F401
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
        # Legacy GRU models
        "DFNet1",
        "DFNet2",
        "DFNet3",
        "ModelParams1",
        "ModelParams2",
        "ModelParams3",
        "init_dfnet1",
        "init_dfnet2",
        "init_dfnet3",
        # GRU modules
        "GroupedGRU",
        "SqueezedGRU",
        "SqueezedGRU_S",
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
        "erb_fb_and_inverse",
        "erb_transform",
        # Training
        "Trainer",
        "train",
        "spectral_loss",
        "multi_resolution_stft_loss",
        "load_pytorch_checkpoint",
        "convert_pytorch_weights",
        # Checkpoint conversion
        "load_pytorch_ckpt",
        "save_mlx_checkpoint",
        "load_mlx_checkpoint",
        "convert_and_save",
        "convert_dfnet3_checkpoint",
        "convert_generic_checkpoint",
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
        # Loss functions (new)
        "SpectralLoss",
        "MaskLoss",
        "SiSdrLoss",
        "SegmentalSiSdrLoss",
        "DfAlphaLoss",
        "FeatureMatchingLoss",
        "CombinedLoss",
        "discriminator_loss",
        "generator_loss",
        # LR Schedulers (new)
        "cosine_scheduler",
        "CosineScheduler",
        "WarmupScheduler",
        "LinearDecayScheduler",
        "ExponentialDecayScheduler",
        "create_scheduler",
        # Checkpoint (new)
        "PatienceState",
        "CheckpointState",
        "check_patience",
        "read_patience",
        "write_patience",
        "save_checkpoint",
        "load_checkpoint",
        "CheckpointManager",
        # Discriminators (new)
        "PeriodDiscriminator",
        "ScaleDiscriminator",
        "MultiPeriodDiscriminator",
        "MultiScaleDiscriminator",
        "CombinedDiscriminator",
        "SpectralDiscriminator",
        "compute_discriminator_loss",
        "compute_generator_loss",
        # STOI (new)
        "stoi",
        "stoi_numpy",
        "stoi_loss",
        # Evaluation (new)
        "si_sdr",
        "snr",
        "segmental_snr",
        "MetricResult",
        "EvaluationResults",
        "Metric",
        "SiSDRMetric",
        "STOIMetric",
        "PESQMetric",
        "SNRMetric",
        "SegmentalSNRMetric",
        "get_metric_factory",
        "evaluate_batch",
        "evaluate_single",
        "ValidationMetrics",
        "quick_eval",
        "compare_before_after",
        "log_improvement",
        # GAN Training (new)
        "GANConfig",
        "GANTrainer",
        "train_gan",
        # Multi-frame processing (new)
        "MultiFrameModule",
        "DF",
        "DFreal",
        "MfWf",
        "MfMvdr",
        "MultiResolutionDF",
        "CRM",
        # Multi-frame DFNet model (new)
        "DFNetMF",
        "ModelParamsMF",
        "create_dfnetmf",
        "EncoderMF",
        "ErbDecoderMF",
        "DfDecoderMF",
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
