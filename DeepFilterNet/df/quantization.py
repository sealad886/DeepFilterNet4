"""Quantization support for DeepFilterNet4.

This module provides tools for:
- Quantization-aware training (QAT) for INT8 deployment
- Post-training quantization (PTQ)
- Model export with quantization

Usage:
    # QAT during training
    from df.quantization import prepare_qat, convert_qat

    model = DfNet4(...)
    qat_model = prepare_qat(model)
    # ... train qat_model ...
    quantized_model = convert_qat(qat_model)

    # Post-training quantization
    from df.quantization import quantize_dynamic

    model = DfNet4(...)
    quantized_model = quantize_dynamic(model)
"""

import copy
from typing import Callable, Dict, List, Optional, Set, Tuple, Type

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

try:
    from torch.ao.quantization import (
        DeQuantStub,
        QConfig,
        QuantStub,
        convert,
        get_default_qat_qconfig,
        get_default_qconfig,
        prepare_qat,
    )

    QUANTIZATION_AVAILABLE = True
    # Note: quantize_fx has Python 3.14 compatibility issues, so we skip it
    # from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
except ImportError:
    QUANTIZATION_AVAILABLE = False
    logger.warning("PyTorch quantization not available. Install torch>=2.0 for quantization support.")
except AttributeError as e:
    # Handle Python 3.14 compatibility issues with quantize_fx
    QUANTIZATION_AVAILABLE = False
    logger.warning(f"PyTorch quantization has compatibility issues (likely Python 3.14): {e}")


# Modules that should be fused for better quantization
FUSABLE_MODULES = [
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU),
    (nn.Conv2d, nn.BatchNorm2d),
    (nn.Conv2d, nn.ReLU),
    (nn.Linear, nn.ReLU),
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU),
    (nn.Conv1d, nn.BatchNorm1d),
]


class QuantizationConfig:
    """Configuration for quantization.

    Attributes:
        backend: Quantization backend ("x86", "fbgemm", "qnnpack", "onednn")
        qat_epochs: Number of QAT fine-tuning epochs
        calibration_batches: Number of batches for PTQ calibration
        per_channel: Use per-channel quantization for weights
        symmetric: Use symmetric quantization
        reduce_range: Reduce quantization range for better accuracy
    """

    def __init__(
        self,
        backend: str = "x86",
        qat_epochs: int = 10,
        calibration_batches: int = 100,
        per_channel: bool = True,
        symmetric: bool = True,
        reduce_range: bool = False,
    ):
        self.backend = backend
        self.qat_epochs = qat_epochs
        self.calibration_batches = calibration_batches
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.reduce_range = reduce_range


def check_quantization_available() -> bool:
    """Check if PyTorch quantization is available."""
    return QUANTIZATION_AVAILABLE


def get_qconfig(
    backend: str = "x86",
    qat: bool = False,
    per_channel: bool = True,
) -> "QConfig":
    """Get quantization config for the specified backend.

    Args:
        backend: Quantization backend
        qat: Whether this is for QAT (vs PTQ)
        per_channel: Use per-channel quantization for weights

    Returns:
        QConfig for the specified settings
    """
    if not QUANTIZATION_AVAILABLE:
        raise RuntimeError("Quantization not available")

    torch.backends.quantized.engine = backend

    if qat:
        qconfig = get_default_qat_qconfig(backend)  # type: ignore[possibly-undefined]
    else:
        qconfig = get_default_qconfig(backend)  # type: ignore[possibly-undefined]

    return qconfig


class QuantizedModelWrapper(nn.Module):
    """Wrapper that adds quant/dequant stubs to a model.

    This wrapper is used to prepare models for static quantization.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = QuantStub() if QUANTIZATION_AVAILABLE else nn.Identity()  # type: ignore[possibly-undefined]
        self.model = model
        self.dequant = DeQuantStub() if QUANTIZATION_AVAILABLE else nn.Identity()  # type: ignore[possibly-undefined]

    def forward(
        self,
        feat_erb: Tensor,
        feat_spec: Tensor,
        waveform: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass with quant/dequant."""
        # Quantize inputs
        feat_erb = self.quant(feat_erb)
        feat_spec = self.quant(feat_spec)

        # Model forward
        output = self.model(feat_erb, feat_spec, waveform)

        # Dequantize outputs
        if isinstance(output, tuple):
            return tuple(self.dequant(o) if isinstance(o, Tensor) else o for o in output)
        return self.dequant(output)


def prepare_model_for_qat(
    model: nn.Module,
    qconfig: Optional["QConfig"] = None,
    backend: str = "x86",
    inplace: bool = False,
) -> nn.Module:
    """Prepare a model for quantization-aware training.

    Args:
        model: The model to prepare for QAT
        qconfig: Quantization config (uses default if None)
        backend: Quantization backend
        inplace: Modify model in place

    Returns:
        Model prepared for QAT
    """
    if not QUANTIZATION_AVAILABLE:
        logger.warning("Quantization not available, returning original model")
        return model

    if not inplace:
        model = copy.deepcopy(model)

    # Set quantization backend
    torch.backends.quantized.engine = backend

    # Use default QAT config if not specified
    if qconfig is None:
        qconfig = get_default_qat_qconfig(backend)  # type: ignore[possibly-undefined]

    # Set qconfig for the model
    model.qconfig = qconfig

    # Fuse modules for better performance
    model = fuse_modules(model)

    # Prepare for QAT
    model.train()
    prepare_qat(model, inplace=True)  # type: ignore[possibly-undefined]

    logger.info(f"Model prepared for QAT with backend={backend}")
    return model


def convert_qat_model(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """Convert a QAT model to a quantized model.

    Args:
        model: QAT-trained model
        inplace: Convert in place

    Returns:
        Quantized model
    """
    if not QUANTIZATION_AVAILABLE:
        logger.warning("Quantization not available, returning original model")
        return model

    if not inplace:
        model = copy.deepcopy(model)

    model.eval()
    quantized_model = convert(model, inplace=True)  # type: ignore[possibly-undefined]

    logger.info("Converted QAT model to quantized model")
    return quantized_model


def fuse_modules(model: nn.Module) -> nn.Module:
    """Fuse eligible modules in the model for quantization.

    Fuses patterns like Conv-BN-ReLU, Conv-BN, Linear-ReLU for better
    quantization efficiency.

    Args:
        model: Model to fuse modules in

    Returns:
        Model with fused modules
    """
    if not QUANTIZATION_AVAILABLE:
        return model

    # Get module names that can be fused
    fuse_patterns = find_fusable_patterns(model)

    if fuse_patterns:
        logger.debug(f"Fusing {len(fuse_patterns)} module patterns")
        model = torch.ao.quantization.fuse_modules(model, fuse_patterns)

    return model


def find_fusable_patterns(model: nn.Module) -> List[List[str]]:
    """Find fusable module patterns in the model.

    Args:
        model: Model to search

    Returns:
        List of module name lists that can be fused
    """
    patterns = []
    modules = dict(model.named_modules())
    module_names = list(modules.keys())

    i = 0
    while i < len(module_names):
        name = module_names[i]
        module = modules[name]

        # Check for Conv-BN-ReLU pattern
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            bn_name = _get_next_module_name(module_names, i, name)
            if bn_name and isinstance(modules.get(bn_name), (nn.BatchNorm1d, nn.BatchNorm2d)):
                relu_name = _get_next_module_name(module_names, i + 1, bn_name)
                if relu_name and isinstance(modules.get(relu_name), nn.ReLU):
                    patterns.append([name, bn_name, relu_name])
                    i += 3
                    continue
                else:
                    patterns.append([name, bn_name])
                    i += 2
                    continue

        # Check for Linear-ReLU pattern
        if isinstance(module, nn.Linear):
            relu_name = _get_next_module_name(module_names, i, name)
            if relu_name and isinstance(modules.get(relu_name), nn.ReLU):
                patterns.append([name, relu_name])
                i += 2
                continue

        i += 1

    return patterns


def _get_next_module_name(
    module_names: List[str],
    current_idx: int,
    current_name: str,
) -> Optional[str]:
    """Get the next sequential module name."""
    if current_idx + 1 >= len(module_names):
        return None
    next_name = module_names[current_idx + 1]
    # Check if it's a direct child
    parent = ".".join(current_name.split(".")[:-1])
    next_parent = ".".join(next_name.split(".")[:-1])
    if parent == next_parent:
        return next_name
    return None


def quantize_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    modules_to_quantize: Optional[Set[Type[nn.Module]]] = None,
) -> nn.Module:
    """Apply dynamic quantization to a model.

    Dynamic quantization quantizes weights statically and activations
    dynamically at runtime. Best for models with large linear layers.

    Args:
        model: Model to quantize
        dtype: Quantization dtype (qint8 or float16)
        modules_to_quantize: Set of module types to quantize

    Returns:
        Dynamically quantized model
    """
    if not QUANTIZATION_AVAILABLE:
        logger.warning("Quantization not available, returning original model")
        return model

    if modules_to_quantize is None:
        modules_to_quantize = {nn.Linear, nn.LSTM, nn.GRU}

    model.eval()
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        modules_to_quantize,
        dtype=dtype,
    )

    logger.info(f"Applied dynamic quantization with dtype={dtype}")
    return quantized


def quantize_static(
    model: nn.Module,
    calibration_fn: Callable[[nn.Module], None],
    backend: str = "x86",
    per_channel: bool = True,
) -> nn.Module:
    """Apply static quantization to a model.

    Static quantization quantizes both weights and activations. Requires
    calibration with representative data.

    Args:
        model: Model to quantize
        calibration_fn: Function that runs calibration data through model
        backend: Quantization backend
        per_channel: Use per-channel weight quantization

    Returns:
        Statically quantized model
    """
    if not QUANTIZATION_AVAILABLE:
        logger.warning("Quantization not available, returning original model")
        return model

    # Wrap model with quant stubs
    model = QuantizedModelWrapper(model)

    # Set backend and qconfig
    torch.backends.quantized.engine = backend
    model.qconfig = get_default_qconfig(backend)  # type: ignore[possibly-undefined]

    # Fuse modules
    model.model = fuse_modules(model.model)

    # Prepare for static quantization
    model.eval()
    torch.ao.quantization.prepare(model, inplace=True)

    # Run calibration
    logger.info("Running calibration...")
    with torch.no_grad():
        calibration_fn(model)

    # Convert to quantized model
    torch.ao.quantization.convert(model, inplace=True)

    logger.info(f"Applied static quantization with backend={backend}")
    return model


def get_model_size_mb(model: nn.Module) -> float:
    """Get the size of a model in megabytes.

    Args:
        model: Model to measure

    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def compare_model_sizes(
    original: nn.Module,
    quantized: nn.Module,
) -> Dict[str, float]:
    """Compare sizes of original and quantized models.

    Args:
        original: Original model
        quantized: Quantized model

    Returns:
        Dictionary with size comparison stats
    """
    original_size = get_model_size_mb(original)
    quantized_size = get_model_size_mb(quantized)
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0

    return {
        "original_mb": original_size,
        "quantized_mb": quantized_size,
        "compression_ratio": compression_ratio,
        "size_reduction_pct": ((1 - quantized_size / original_size) * 100 if original_size > 0 else 0),
    }


def export_quantized_model(
    model: nn.Module,
    export_path: str,
    example_inputs: Optional[Tuple[Tensor, ...]] = None,
    export_format: str = "torchscript",
) -> str:
    """Export a quantized model to a file.

    Args:
        model: Quantized model to export
        export_path: Path to save the exported model
        example_inputs: Example inputs for tracing (required for ONNX)
        export_format: Export format ("torchscript", "onnx", or "state_dict")

    Returns:
        Path to exported model
    """
    model.eval()

    if export_format == "state_dict":
        # Simple state dict save - always works
        torch.save(model.state_dict(), export_path)
        logger.info(f"Exported quantized model state dict to {export_path}")
        return export_path

    if export_format == "torchscript":
        if example_inputs is None:
            # Try scripting instead of tracing if no example inputs
            try:
                scripted = torch.jit.script(model)
                scripted.save(export_path)
                logger.info(f"Exported quantized model to {export_path} (TorchScript scripted)")
                return export_path
            except Exception:
                raise ValueError("example_inputs required for TorchScript tracing")
        # Trace the model
        traced = torch.jit.trace(model, example_inputs)
        traced.save(export_path)  # type: ignore[union-attr]
        logger.info(f"Exported quantized model to {export_path} (TorchScript)")

    elif export_format == "onnx":
        if example_inputs is None:
            raise ValueError("example_inputs required for ONNX export")
        import_path = export_path if export_path.endswith(".onnx") else export_path + ".onnx"
        torch.onnx.export(
            model,
            example_inputs,
            import_path,
            input_names=["feat_erb", "feat_spec"],
            output_names=["spec_out", "m", "lsnr", "alpha"],
            dynamic_axes={
                "feat_erb": {0: "batch", 2: "time"},
                "feat_spec": {0: "batch", 2: "time"},
                "spec_out": {0: "batch", 2: "time"},
            },
            opset_version=14,
        )
        logger.info(f"Exported quantized model to {import_path} (ONNX)")
        export_path = import_path

    else:
        raise ValueError(f"Unknown export format: {export_format}")

    return export_path


class QATCallback:
    """Callback for QAT training integration.

    This callback can be used with the training loop to manage QAT state.

    Usage:
        callback = QATCallback(model, start_epoch=5, freeze_bn_epoch=8)
        for epoch in range(epochs):
            callback.on_epoch_start(epoch)
            # ... train ...
            callback.on_epoch_end(epoch)
    """

    def __init__(
        self,
        model: nn.Module,
        start_epoch: int = 0,
        freeze_bn_epoch: Optional[int] = None,
        backend: str = "x86",
    ):
        """Initialize QAT callback.

        Args:
            model: Model being trained
            start_epoch: Epoch to start QAT (0 = from beginning)
            freeze_bn_epoch: Epoch to freeze BN statistics
            backend: Quantization backend
        """
        self.model = model
        self.start_epoch = start_epoch
        self.freeze_bn_epoch = freeze_bn_epoch
        self.backend = backend
        self.qat_active = False

    @property
    def qat_enabled(self) -> bool:
        """Alias for qat_active for backward compatibility."""
        return self.qat_active

    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        if not QUANTIZATION_AVAILABLE:
            return

        # Enable QAT at start_epoch
        if epoch >= self.start_epoch and not self.qat_active:
            logger.info(f"Enabling QAT at epoch {epoch}")
            prepare_model_for_qat(self.model, backend=self.backend, inplace=True)
            self.qat_active = True

        # Freeze BN at freeze_bn_epoch
        if self.freeze_bn_epoch and epoch >= self.freeze_bn_epoch:
            self._freeze_bn()

    def on_epoch_end(self, epoch: int):
        """Called at the end of each epoch."""

    def _freeze_bn(self):
        """Freeze batch normalization layers."""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def convert(self) -> nn.Module:
        """Convert QAT model to quantized model."""
        if self.qat_active:
            return convert_qat_model(self.model)
        return self.model
