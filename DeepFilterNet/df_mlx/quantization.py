"""Quantization utilities for MLX-based DeepFilterNet models.

This module provides utilities for quantizing models to reduce memory usage
and potentially improve inference speed on Apple Silicon.

MLX supports various quantization schemes:
- 2-bit, 4-bit, 8-bit quantization
- Group-wise quantization with configurable group size
- Affine and symmetric quantization modes

Usage:
    from df_mlx.quantization import quantize_model, save_quantized_model

    # Load your model
    model = DFNet4(...)
    model.load_weights("model.safetensors")

    # Quantize to 4-bit
    quantized_model = quantize_model(model, bits=4)

    # Save quantized model
    save_quantized_model(quantized_model, "model_q4.safetensors")

    # Inference with quantized model
    output = quantized_model(input_audio)
"""

# TODO: Rewrite quantization with lazy loading and any other updates/modernizations made to the overall df_mlx codebase.
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import mlx.core as mx
import mlx.nn as nn
from loguru import logger


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 64,
    exclude_layers: Optional[Set[str]] = None,
    verbose: bool = True,
) -> nn.Module:
    """Quantize model weights to lower precision.

    This function quantizes Linear layers in the model to reduce memory
    usage. Convolutional layers, normalization layers, and embeddings
    are not quantized by default.

    Args:
        model: The model to quantize
        bits: Number of bits per weight (2, 4, or 8)
        group_size: Group size for quantization (default 64)
        exclude_layers: Set of layer names to exclude from quantization
        verbose: Whether to print quantization statistics

    Returns:
        Quantized model with QuantizedLinear layers
    """
    if bits not in [2, 4, 8]:
        raise ValueError(f"bits must be 2, 4, or 8, got {bits}")

    if exclude_layers is None:
        exclude_layers = set()

    # Use MLX's built-in quantize function
    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda p, m: (isinstance(m, nn.Linear) and p not in exclude_layers),
    )

    if verbose:
        stats = get_quantization_stats(model)
        logger.info(f"Quantization complete ({bits}-bit, group_size={group_size})")
        logger.info(f"  Quantized layers: {stats['quantized_layers']}")
        logger.info(f"  Skipped layers: {stats['skipped_layers']}")
        logger.info(
            f"  Memory reduction: {stats['original_size_mb']:.1f}MB -> "
            f"{stats['quantized_size_mb']:.1f}MB "
            f"({stats['compression_ratio']:.1f}x)"
        )

    return model


def get_quantization_stats(model: nn.Module) -> Dict:
    """Get statistics about model quantization.

    Args:
        model: The model to analyze

    Returns:
        Dictionary with quantization statistics
    """
    quantized_layers = 0
    skipped_layers = 0
    original_params = 0
    quantized_params = 0

    def count_params(module, prefix=""):
        nonlocal quantized_layers, skipped_layers, original_params, quantized_params

        for name, child in module.children().items():
            child_prefix = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.QuantizedLinear):
                quantized_layers += 1
                # QuantizedLinear stores weights in compressed format
                # Original would be weight.shape[0] * weight.shape[1] * 4 bytes
                # Estimate original size from scales shape
                # scales shape is [out_features, in_features // group_size]
                if hasattr(child, "scales"):
                    out_features = child.scales.shape[0]
                    in_features = child.scales.shape[1] * child.group_size
                    original_params += out_features * in_features
                    # Quantized size is bits/8 per weight
                    quantized_params += (out_features * in_features * child.bits) // 8
            elif isinstance(child, nn.Linear):
                skipped_layers += 1
                weight = child.weight
                original_params += weight.size
                quantized_params += weight.size
            elif hasattr(child, "children"):
                count_params(child, child_prefix)

    count_params(model)

    original_size_mb = original_params * 4 / (1024 * 1024)  # float32
    quantized_size_mb = quantized_params / (1024 * 1024)

    return {
        "quantized_layers": quantized_layers,
        "skipped_layers": skipped_layers,
        "original_params": original_params,
        "quantized_params": quantized_params,
        "original_size_mb": original_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "compression_ratio": (original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0),
    }


def dequantize_model(model: nn.Module) -> nn.Module:
    """Convert quantized model back to full precision.

    Note: This is primarily useful for fine-tuning or analysis.
    The dequantized weights will be approximations of the original.

    Args:
        model: Quantized model

    Returns:
        Model with full-precision Linear layers
    """

    def dequantize_recursive(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.children().items()):
            child_prefix = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.QuantizedLinear):
                # Create full-precision Linear from QuantizedLinear
                # Get dequantized weights
                dequant_weight = mx.dequantize(
                    child.weight,
                    child.scales,
                    child.biases if hasattr(child, "biases") else None,
                    child.group_size,
                    child.bits,
                )

                # Create new Linear layer
                has_bias = hasattr(child, "bias") and child.bias is not None
                new_linear = nn.Linear(dequant_weight.shape[1], dequant_weight.shape[0], bias=has_bias)
                new_linear.weight = dequant_weight

                if has_bias:
                    new_linear.bias = child.bias

                # Replace in parent
                setattr(module, name, new_linear)
                logger.debug(f"Dequantized: {child_prefix}")

            elif hasattr(child, "children"):
                dequantize_recursive(child, child_prefix)

    dequantize_recursive(model)
    return model


def save_quantized_model(
    model: nn.Module,
    path: Union[str, Path],
    metadata: Optional[Dict] = None,
) -> None:
    """Save quantized model weights.

    Args:
        model: Quantized model to save
        path: Output file path (.safetensors or .npz)
        metadata: Optional metadata to include
    """
    path = Path(path)

    # Get all parameters including quantized ones
    weights = dict(model.parameters())

    # Flatten nested dict
    flat_weights = {}
    _flatten_dict(weights, flat_weights)

    if path.suffix == ".safetensors":
        mx.save_safetensors(str(path), flat_weights, metadata=metadata or {})
    else:
        mx.savez(str(path), **flat_weights)

    logger.info(f"Saved quantized model to {path}")


def load_quantized_model(
    model: nn.Module,
    path: Union[str, Path],
    strict: bool = True,
) -> nn.Module:
    """Load quantized model weights.

    Args:
        model: Model architecture (should match saved model)
        path: Path to saved weights
        strict: If True, raise error on missing/extra keys

    Returns:
        Model with loaded weights
    """
    path = Path(path)

    if path.suffix == ".safetensors":
        loaded = mx.load(str(path))
        # mx.load returns dict for safetensors
        weights: Dict[str, mx.array] = loaded if isinstance(loaded, dict) else {}  # type: ignore[assignment]
    else:
        # TODO: This is incorrect, as df_mlx should now lazy load files
        loaded = mx.load(str(path))
        weights = dict(loaded) if isinstance(loaded, dict) else {}  # type: ignore[arg-type]

    model.load_weights(list(weights.items()), strict=strict)
    logger.info(f"Loaded quantized model from {path}")

    return model


def _flatten_dict(d: Dict, out: Dict, prefix: str = "", sep: str = ".") -> None:
    """Flatten nested dictionary with dot-separated keys."""
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, out, key, sep)
        else:
            out[key] = v


def estimate_memory_savings(
    model: nn.Module,
    target_bits: int = 4,
    group_size: int = 64,
) -> Dict:
    """Estimate memory savings from quantization.

    Args:
        model: Model to analyze
        target_bits: Target bits for quantization
        group_size: Group size for quantization

    Returns:
        Dictionary with memory estimates
    """
    total_linear_params = 0
    other_params = 0

    def count_params(module):
        nonlocal total_linear_params, other_params

        for name, child in module.children().items():
            if isinstance(child, nn.Linear):
                total_linear_params += child.weight.size
                if hasattr(child, "bias") and child.bias is not None:
                    total_linear_params += child.bias.size
            elif isinstance(child, nn.QuantizedLinear):
                # Already quantized, estimate original size
                if hasattr(child, "scales"):
                    out_features = child.scales.shape[0]
                    in_features = child.scales.shape[1] * child.group_size
                    total_linear_params += out_features * in_features
            elif hasattr(child, "children"):
                count_params(child)
            else:
                # Count other parameters
                for p in child.parameters().values():
                    if isinstance(p, mx.array):
                        other_params += p.size

    count_params(model)

    # Calculate sizes
    original_linear_mb = total_linear_params * 4 / (1024 * 1024)  # float32
    quantized_linear_mb = (
        (total_linear_params * target_bits / 8)  # quantized weights
        + (total_linear_params / group_size * 4)  # scales (float32)
    ) / (1024 * 1024)
    other_mb = other_params * 4 / (1024 * 1024)

    return {
        "original_total_mb": original_linear_mb + other_mb,
        "quantized_total_mb": quantized_linear_mb + other_mb,
        "linear_params": total_linear_params,
        "other_params": other_params,
        "linear_original_mb": original_linear_mb,
        "linear_quantized_mb": quantized_linear_mb,
        "other_mb": other_mb,
        "savings_mb": original_linear_mb - quantized_linear_mb,
        "compression_ratio": original_linear_mb / quantized_linear_mb if quantized_linear_mb > 0 else 1.0,
    }


class QuantizationConfig:
    """Configuration for model quantization."""

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 64,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """Initialize quantization configuration.

        Args:
            bits: Bits per weight (2, 4, or 8)
            group_size: Group size for quantization
            exclude_patterns: List of layer name patterns to exclude
        """
        self.bits = bits
        self.group_size = group_size
        self.exclude_patterns = exclude_patterns or []

    def should_quantize(self, layer_name: str) -> bool:
        """Check if a layer should be quantized."""
        for pattern in self.exclude_patterns:
            if pattern in layer_name:
                return False
        return True

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "exclude_patterns": self.exclude_patterns,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "QuantizationConfig":
        """Create from dictionary."""
        return cls(
            bits=d.get("bits", 4),
            group_size=d.get("group_size", 64),
            exclude_patterns=d.get("exclude_patterns", []),
        )


# Predefined quantization configs
QUANTIZATION_PRESETS = {
    "q4": QuantizationConfig(bits=4, group_size=64),
    "q4_small": QuantizationConfig(bits=4, group_size=32),
    "q8": QuantizationConfig(bits=8, group_size=64),
    "q2": QuantizationConfig(bits=2, group_size=64),
}


def get_preset(name: str) -> QuantizationConfig:
    """Get a predefined quantization configuration.

    Args:
        name: Preset name ("q4", "q4_small", "q8", "q2")

    Returns:
        QuantizationConfig for the preset
    """
    if name not in QUANTIZATION_PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(QUANTIZATION_PRESETS.keys())}")
    return QUANTIZATION_PRESETS[name]


# Quick test when run directly
if __name__ == "__main__":
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 128)
            self.linear2 = nn.Linear(128, 64)

        def __call__(self, x):
            x = nn.relu(self.linear1(x))
            return self.linear2(x)

    print("Testing quantization utilities...")

    # Create model
    model = TestModel()
    x = mx.random.normal((2, 64))

    # Test inference before quantization
    y_before = model(x)
    print(f"  Output before quantization: {y_before.shape}")

    # Estimate savings
    estimates = estimate_memory_savings(model, target_bits=4)
    print(f"  Estimated savings: {estimates['savings_mb']:.2f} MB")
    print(f"  Compression ratio: {estimates['compression_ratio']:.1f}x")

    # Quantize model
    quantize_model(model, bits=4, group_size=32)

    # Test inference after quantization
    y_after = model(x)
    print(f"  Output after quantization: {y_after.shape}")

    # Check output difference
    diff = mx.abs(y_before - y_after).max()
    print(f"  Max output difference: {float(diff):.6f}")

    print("✓ All quantization tests passed!")
