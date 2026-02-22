"""Hardware detection and optimization for MLX on Apple Silicon.

This module provides automatic hardware detection and performance tuning
specifically for MLX running on Apple Silicon (M1/M2/M3/M4).

Key Features:
- Apple Silicon chip variant detection
- Memory-aware batch size tuning
- Worker count optimization
- Neural Engine and GPU core detection

Usage:
    from df_mlx.hardware import HardwareConfig

    # Get auto-tuned configuration
    hw_config = HardwareConfig.detect()
    print(f"Chip: {hw_config.profile.chip_variant}")
    print(f"Recommended batch size: {hw_config.batch_size}")
    print(f"Unified memory: {hw_config.profile.total_memory_gb:.1f} GB")

Environment Variables (override auto-detection):
    DF_MLX_BATCH_SIZE: Force batch size
    DF_MLX_NUM_WORKERS: Force worker count
    DF_MLX_DISABLE_AUTOTUNE: Set to 1 to disable auto-tuning
"""

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mlx.core as mx
from loguru import logger

# Apple Silicon chip specifications database
# Format: (variant, gpu_cores, neural_engine_cores, performance_cores, efficiency_cores)
APPLE_CHIP_SPECS = {
    # M1 family
    "M1": (8, 16, 4, 4),
    "M1 Pro": (16, 16, 8, 2),
    "M1 Max": (32, 16, 8, 2),
    "M1 Ultra": (64, 32, 16, 4),
    # M2 family
    "M2": (10, 16, 4, 4),
    "M2 Pro": (19, 16, 8, 4),
    "M2 Max": (38, 16, 8, 4),
    "M2 Ultra": (76, 32, 16, 8),
    # M3 family
    "M3": (10, 16, 4, 4),
    "M3 Pro": (18, 16, 6, 6),
    "M3 Max": (40, 16, 12, 4),
    # M4 family
    "M4": (10, 16, 4, 6),
    "M4 Pro": (20, 16, 10, 4),
    "M4 Max": (40, 16, 14, 4),
}


@dataclass
class HardwareProfile:
    """Hardware profile with detected capabilities."""

    # Device info
    device_name: str = "Apple Silicon"

    # Memory (in GB)
    total_memory_gb: float = 8.0
    available_memory_gb: float = 6.0

    # CPU info
    cpu_count: int = 8
    cpu_name: str = "Unknown"

    # Platform
    os_name: str = "Darwin"
    os_version: str = ""
    macos_version: Tuple[int, int, int] = (0, 0, 0)

    # Apple Silicon specific
    chip_variant: str = "Unknown"
    gpu_cores: int = 8
    neural_engine_cores: int = 16
    performance_cores: int = 4
    efficiency_cores: int = 4

    # MLX capabilities
    supports_fp16: bool = True
    supports_bf16: bool = True
    supports_complex: bool = True


@dataclass
class HardwareConfig:
    """Recommended configuration based on hardware detection."""

    # Core settings
    batch_size: int = 4
    batch_size_eval: int = 8
    num_workers: int = 4

    # Memory management
    prefetch_batches: int = 32

    # Performance tuning
    use_fp16: bool = False  # Mixed precision training

    # Profile info
    profile: Optional[HardwareProfile] = None

    # Override flags
    is_auto_tuned: bool = True

    @classmethod
    def detect(cls, verbose: bool = True) -> "HardwareConfig":
        """Detect hardware and return optimized configuration.

        Args:
            verbose: If True, log detected configuration.

        Returns:
            HardwareConfig with optimized settings for detected hardware.
        """
        # Check for disable flag
        if os.environ.get("DF_MLX_DISABLE_AUTOTUNE", "0") == "1":
            config = cls()
            config.is_auto_tuned = False
            if verbose:
                logger.info("Auto-tuning disabled via DF_MLX_DISABLE_AUTOTUNE")
            return config

        profile = _detect_hardware_profile()
        config = _build_config_for_profile(profile)
        config.profile = profile

        # Apply environment variable overrides
        config = _apply_env_overrides(config)

        if verbose:
            _log_hardware_config(config)

        return config

    @classmethod
    def from_env(cls) -> "HardwareConfig":
        """Create config from environment variables only (no auto-detection)."""
        config = cls()
        config.is_auto_tuned = False
        return _apply_env_overrides(config)

    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        return {
            "batch_size": self.batch_size,
            "batch_size_eval": self.batch_size_eval,
            "num_workers": self.num_workers,
            "prefetch_batches": self.prefetch_batches,
            "use_fp16": self.use_fp16,
            "is_auto_tuned": self.is_auto_tuned,
        }


def _detect_hardware_profile() -> HardwareProfile:
    """Detect hardware capabilities and return profile."""
    profile = HardwareProfile(
        os_name=platform.system(),
        os_version=platform.release(),
        cpu_count=os.cpu_count() or 8,
    )

    # Detect macOS version
    profile.macos_version = _get_macos_version()

    # Detect CPU info and Apple Silicon variant
    profile.cpu_name = _get_cpu_name()
    profile.chip_variant = _parse_chip_variant(profile.cpu_name)
    profile.device_name = f"Apple {profile.chip_variant}"

    # Get chip specifications
    specs = APPLE_CHIP_SPECS.get(profile.chip_variant)
    if specs:
        profile.gpu_cores = specs[0]
        profile.neural_engine_cores = specs[1]
        profile.performance_cores = specs[2]
        profile.efficiency_cores = specs[3]

    # Detect memory
    profile.total_memory_gb = _get_system_memory_gb()
    profile.available_memory_gb = profile.total_memory_gb * 0.7  # Conservative

    return profile


def _build_config_for_profile(profile: HardwareProfile) -> HardwareConfig:
    """Build optimized config based on hardware profile."""
    config = HardwareConfig()

    # Memory-based batch size tuning
    # Rule of thumb: ~1GB per batch item for full training
    mem_gb = profile.available_memory_gb

    if mem_gb >= 64:
        config.batch_size = 32
        config.batch_size_eval = 64
    elif mem_gb >= 32:
        config.batch_size = 16
        config.batch_size_eval = 32
    elif mem_gb >= 16:
        config.batch_size = 8
        config.batch_size_eval = 16
    elif mem_gb >= 8:
        config.batch_size = 4
        config.batch_size_eval = 8
    else:
        config.batch_size = 2
        config.batch_size_eval = 4

    # Worker count based on CPU cores
    # Use performance cores, but not too many to avoid memory pressure
    num_perf_cores = profile.performance_cores
    config.num_workers = max(2, min(num_perf_cores - 1, 8))

    # Prefetch based on memory
    if mem_gb >= 32:
        config.prefetch_batches = 64
    elif mem_gb >= 16:
        config.prefetch_batches = 32
    else:
        config.prefetch_batches = 16

    # FP16 is generally stable on Apple Silicon
    config.use_fp16 = True

    return config


def _apply_env_overrides(config: HardwareConfig) -> HardwareConfig:
    """Apply environment variable overrides to config."""
    if batch_size := os.environ.get("DF_MLX_BATCH_SIZE"):
        config.batch_size = int(batch_size)
        config.is_auto_tuned = False

    if num_workers := os.environ.get("DF_MLX_NUM_WORKERS"):
        config.num_workers = int(num_workers)
        config.is_auto_tuned = False

    return config


def _get_cpu_name() -> str:
    """Get CPU brand string."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "Unknown"


def _get_macos_version() -> Tuple[int, int, int]:
    """Get macOS version as tuple (major, minor, patch)."""
    try:
        version = platform.mac_ver()[0]
        parts = version.split(".")
        return tuple(int(p) for p in parts[:3])  # type: ignore[return-value]
    except Exception:
        return (0, 0, 0)


def _get_system_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024**3)
    except Exception:
        pass

    # Fallback: try /usr/sbin/system_profiler
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Memory:" in line:
                    # Parse "Memory: 32 GB"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "GB" and i > 0:
                            return float(parts[i - 1])
    except Exception:
        pass

    return 8.0  # Default fallback


def _parse_chip_variant(cpu_name: str) -> str:
    """Parse Apple Silicon variant from CPU name."""
    if "Apple" not in cpu_name:
        return "Unknown"

    # Handle different naming patterns
    # "Apple M3 Max", "Apple M2 Pro", "Apple M1", etc.
    parts = cpu_name.replace("Apple", "").strip().split()

    if not parts:
        return "Unknown"

    # Build variant name
    variant = parts[0]  # "M1", "M2", "M3", "M4"
    if len(parts) > 1 and parts[1] in ("Pro", "Max", "Ultra"):
        variant += f" {parts[1]}"

    return variant


def _log_hardware_config(config: HardwareConfig) -> None:
    """Log detected hardware configuration."""
    profile = config.profile
    if not profile:
        logger.info("Hardware auto-tuning: No profile detected")
        return

    logger.info("=" * 50)
    logger.info("MLX Hardware Detection")
    logger.info("=" * 50)
    logger.info(f"  Chip: {profile.device_name}")
    logger.info(f"  CPU: {profile.cpu_name}")
    logger.info(f"  macOS: {'.'.join(str(v) for v in profile.macos_version)}")
    logger.info(f"  Memory: {profile.total_memory_gb:.1f} GB unified")
    logger.info(f"  GPU Cores: {profile.gpu_cores}")
    logger.info(f"  Neural Engine: {profile.neural_engine_cores} cores")
    logger.info(f"  CPU Cores: {profile.performance_cores}P + {profile.efficiency_cores}E")
    logger.info("-" * 50)
    logger.info("Recommended Configuration:")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Eval Batch Size: {config.batch_size_eval}")
    logger.info(f"  Workers: {config.num_workers}")
    logger.info(f"  Prefetch: {config.prefetch_batches}")
    logger.info(f"  FP16: {config.use_fp16}")
    logger.info("=" * 50)


def get_mlx_device_info() -> Dict:
    """Get MLX device information.

    Returns:
        Dictionary with MLX device details.
    """
    return {
        "default_device": str(mx.default_device()),
        "metal_available": mx.metal.is_available() if hasattr(mx, "metal") else False,
    }


def estimate_memory_usage(
    model_params: int,
    batch_size: int,
    sequence_length: int = 48000,
    dtype_bytes: int = 4,
) -> Dict[str, float]:
    """Estimate memory usage for training.

    Args:
        model_params: Number of model parameters
        batch_size: Batch size
        sequence_length: Audio sequence length in samples
        dtype_bytes: Bytes per element (4 for float32, 2 for float16)

    Returns:
        Dictionary with memory estimates in GB.
    """
    # Model parameters
    model_mem = model_params * dtype_bytes / (1024**3)

    # Gradients (same size as model)
    grad_mem = model_mem

    # Optimizer states (AdamW uses 2x model size)
    optimizer_mem = model_mem * 2

    # Activations (rough estimate: 2x model size per batch item)
    activation_mem = model_mem * 2 * batch_size

    # Audio batch memory
    audio_mem = batch_size * sequence_length * dtype_bytes / (1024**3)

    total = model_mem + grad_mem + optimizer_mem + activation_mem + audio_mem

    return {
        "model_gb": model_mem,
        "gradients_gb": grad_mem,
        "optimizer_gb": optimizer_mem,
        "activations_gb": activation_mem,
        "audio_gb": audio_mem,
        "total_gb": total,
    }


def recommend_batch_size(
    model_params: int,
    available_memory_gb: float,
    sequence_length: int = 48000,
    safety_factor: float = 0.8,
) -> int:
    """Recommend batch size based on model and memory.

    Args:
        model_params: Number of model parameters
        available_memory_gb: Available memory in GB
        sequence_length: Audio sequence length
        safety_factor: Fraction of memory to use (default 0.8)

    Returns:
        Recommended batch size.
    """
    target_mem = available_memory_gb * safety_factor

    # Binary search for largest fitting batch size
    for batch_size in [64, 32, 16, 8, 4, 2, 1]:
        estimate = estimate_memory_usage(model_params, batch_size, sequence_length)
        if estimate["total_gb"] <= target_mem:
            return batch_size

    return 1


# Quick test when run directly
if __name__ == "__main__":
    config = HardwareConfig.detect(verbose=True)
    print(f"\nMLX Device Info: {get_mlx_device_info()}")
