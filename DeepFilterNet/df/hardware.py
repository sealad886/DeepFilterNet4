"""Hardware detection and auto-tuning for DeepFilterNet.

This module provides automatic hardware detection and performance tuning,
with special optimizations for Apple Silicon (M1/M2/M3/M4).

Key Features:
- Automatic device detection (CUDA, MPS, CPU)
- Memory-aware batch size tuning
- Worker count optimization
- Apple Silicon specific optimizations
- Manual override support for all settings

Usage:
    from df.hardware import HardwareConfig, auto_tune_config

    # Get auto-tuned configuration
    hw_config = HardwareConfig.detect()
    print(f"Device: {hw_config.device}")
    print(f"Recommended batch size: {hw_config.batch_size}")
    print(f"Recommended workers: {hw_config.num_workers}")

    # Apply to training config
    auto_tune_config(config_path, hw_config)

Environment Variables (override auto-detection):
    DF_DEVICE: Force device (cpu, cuda, cuda:0, mps)
    DF_BATCH_SIZE: Force batch size
    DF_NUM_WORKERS: Force worker count
    DF_DISABLE_AUTOTUNE: Set to 1 to disable auto-tuning
"""

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, cast

import torch
from loguru import logger


@dataclass
class HardwareProfile:
    """Hardware profile with detected capabilities."""

    # Device info
    device_type: str  # "cuda", "mps", "cpu"
    device_name: str  # e.g., "Apple M3 Max", "NVIDIA RTX 4090"
    device_count: int = 1

    # Memory (in GB)
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0

    # CPU info
    cpu_count: int = 1
    cpu_name: str = "Unknown"

    # Platform
    os_name: str = "Unknown"
    os_version: str = ""

    # Apple Silicon specific
    is_apple_silicon: bool = False
    apple_chip_variant: str = ""  # e.g., "M3 Max", "M2 Pro"
    neural_engine_cores: int = 0
    gpu_cores: int = 0
    performance_cores: int = 0
    efficiency_cores: int = 0

    # Capabilities
    supports_fp16: bool = True
    supports_bf16: bool = False
    supports_complex: bool = True

    # MPS specific limitations
    mps_complex_supported: bool = True


@dataclass
class HardwareConfig:
    """Recommended configuration based on hardware detection."""

    # Core settings
    device: str = "cpu"
    batch_size: int = 4
    batch_size_eval: int = 8
    num_workers: int = 4

    # Memory management
    pin_memory: bool = False
    prefetch_batches: int = 32

    # Apple Silicon optimizations
    use_mps_fallback: bool = False
    channels_last: bool = True

    # Performance tuning
    enable_amp: bool = False  # Automatic mixed precision
    amp_dtype: str = "float16"
    torch_compile: bool = False  # torch.compile optimization (experimental on MPS)

    # Profile info (set after detection, not in __init__)
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
        if os.environ.get("DF_DISABLE_AUTOTUNE", "0") == "1":
            config = cls()
            config.is_auto_tuned = False
            if verbose:
                logger.info("Auto-tuning disabled via DF_DISABLE_AUTOTUNE")
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
            "device": self.device,
            "batch_size": self.batch_size,
            "batch_size_eval": self.batch_size_eval,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_batches": self.prefetch_batches,
            "use_mps_fallback": self.use_mps_fallback,
            "channels_last": self.channels_last,
            "enable_amp": self.enable_amp,
            "amp_dtype": self.amp_dtype,
            "is_auto_tuned": self.is_auto_tuned,
        }

    def apply_to_config(self, config_parser) -> None:
        """Apply hardware config to a ConfigParser instance.

        Args:
            config_parser: ConfigParser instance to modify.
        """
        if not config_parser.has_section("train"):
            config_parser.add_section("train")

        config_parser.set("train", "DEVICE", self.device)
        config_parser.set("train", "BATCH_SIZE", str(self.batch_size))
        config_parser.set("train", "BATCH_SIZE_EVAL", str(self.batch_size_eval))
        config_parser.set("train", "NUM_WORKERS", str(self.num_workers))
        config_parser.set("train", "NUM_PREFETCH_BATCHES", str(self.prefetch_batches))


def _detect_hardware_profile() -> HardwareProfile:
    """Detect hardware capabilities and return profile."""
    profile = HardwareProfile(
        device_type="cpu",
        device_name="CPU",
        os_name=platform.system(),
        os_version=platform.release(),
        cpu_count=os.cpu_count() or 1,
    )

    # Detect CPU info
    profile.cpu_name = _get_cpu_name()

    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        profile.is_apple_silicon = True
        profile = _detect_apple_silicon_details(profile)

    # Check for CUDA
    if torch.cuda.is_available():
        profile.device_type = "cuda"
        profile.device_count = torch.cuda.device_count()
        profile.device_name = torch.cuda.get_device_name(0)

        # Get memory info
        props = torch.cuda.get_device_properties(0)
        profile.total_memory_gb = props.total_memory / (1024**3)
        profile.available_memory_gb = profile.total_memory_gb  # Approximate

        # Check capabilities
        profile.supports_bf16 = props.major >= 8  # Ampere and newer
        profile.supports_fp16 = props.major >= 7  # Volta and newer

    # Check for MPS (Apple Metal)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        profile.device_type = "mps"
        profile.device_name = f"Apple {profile.apple_chip_variant}" if profile.apple_chip_variant else "Apple Silicon"

        # Check MPS complex tensor support (requires macOS 14+)
        macos_version = _get_macos_version()
        if macos_version and macos_version[0] < 14:
            profile.mps_complex_supported = False
            profile.supports_complex = False

        # Estimate available memory (unified memory)
        if profile.total_memory_gb == 0:
            profile.total_memory_gb = _get_system_memory_gb()
            profile.available_memory_gb = profile.total_memory_gb * 0.7  # Conservative estimate

    else:
        # CPU fallback
        profile.total_memory_gb = _get_system_memory_gb()
        profile.available_memory_gb = profile.total_memory_gb * 0.8

    return profile


def _detect_apple_silicon_details(profile: HardwareProfile) -> HardwareProfile:
    """Detect Apple Silicon specific details using sysctl."""
    try:
        # Get chip brand string
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            brand = result.stdout.strip()
            profile.cpu_name = brand

            # Parse Apple Silicon variant
            if "Apple" in brand:
                # Extract M1/M2/M3/M4 variant
                import re

                match = re.search(r"Apple (M\d+(?:\s+(?:Pro|Max|Ultra))?)", brand)
                if match:
                    profile.apple_chip_variant = match.group(1)

        # Get physical CPU counts
        result = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 1:
                profile.performance_cores = int(lines[0])
            if len(lines) >= 2:
                profile.efficiency_cores = int(lines[1])

        # Get GPU cores (if available)
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "Total Number of Cores" in result.stdout:
            import re

            match = re.search(r"Total Number of Cores:\s*(\d+)", result.stdout)
            if match:
                profile.gpu_cores = int(match.group(1))

        # Get memory
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            mem_bytes = int(result.stdout.strip())
            profile.total_memory_gb = mem_bytes / (1024**3)
            # Apple Silicon uses unified memory - estimate available
            profile.available_memory_gb = profile.total_memory_gb * 0.7

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        logger.debug(f"Could not detect Apple Silicon details: {e}")

    return profile


def _build_config_for_profile(profile: HardwareProfile) -> HardwareConfig:
    """Build optimized config based on hardware profile."""
    config = HardwareConfig()
    config.device = profile.device_type

    # Base configuration by device type
    if profile.device_type == "cuda":
        config = _tune_for_cuda(config, profile)
    elif profile.device_type == "mps":
        config = _tune_for_mps(config, profile)
    else:
        config = _tune_for_cpu(config, profile)

    return config


def _tune_for_cuda(config: HardwareConfig, profile: HardwareProfile) -> HardwareConfig:
    """Tune configuration for NVIDIA CUDA GPUs."""
    config.pin_memory = True
    config.channels_last = True

    # Batch size based on VRAM
    vram_gb = profile.total_memory_gb
    if vram_gb >= 24:  # RTX 4090, A100, etc.
        config.batch_size = 32
        config.batch_size_eval = 64
        config.prefetch_batches = 64
    elif vram_gb >= 16:  # RTX 4080, A5000, etc.
        config.batch_size = 24
        config.batch_size_eval = 48
        config.prefetch_batches = 48
    elif vram_gb >= 12:  # RTX 4070, etc.
        config.batch_size = 16
        config.batch_size_eval = 32
        config.prefetch_batches = 32
    elif vram_gb >= 8:  # RTX 3070, etc.
        config.batch_size = 8
        config.batch_size_eval = 16
        config.prefetch_batches = 24
    else:  # Older GPUs
        config.batch_size = 4
        config.batch_size_eval = 8
        config.prefetch_batches = 16

    # Worker count based on CPU cores (don't starve GPU)
    config.num_workers = min(profile.cpu_count, 8)

    # Mixed precision
    if profile.supports_fp16:
        config.enable_amp = True
        config.amp_dtype = "bfloat16" if profile.supports_bf16 else "float16"

    return config


def _tune_for_mps(config: HardwareConfig, profile: HardwareProfile) -> HardwareConfig:
    """Tune configuration for Apple Silicon MPS."""
    config.pin_memory = False  # Not needed for unified memory
    config.channels_last = True

    # Check for complex tensor support
    if not profile.mps_complex_supported:
        config.use_mps_fallback = True
        logger.warning(
            "MPS on macOS < 14 has limited complex tensor support. "
            "Setting PYTORCH_ENABLE_MPS_FALLBACK=1 is recommended."
        )
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Batch size tuning for MPS
    # Based on benchmarks:
    #   - Backward pass dominates (~96% of training time on MPS)
    #   - Larger batches improve throughput: batch 12 is 1.29x faster than batch 2
    #   - Memory usage scales linearly with batch size
    #   - Sweet spot is largest batch that fits in memory
    mem_gb = profile.total_memory_gb

    # Apple Silicon variants have different GPU capabilities
    chip = profile.apple_chip_variant.lower()
    is_pro_or_higher = any(x in chip for x in ["pro", "max", "ultra"])

    if is_pro_or_higher:
        if mem_gb >= 96:  # M2/M3 Ultra with 96GB+
            config.batch_size = 16
            config.batch_size_eval = 32
            config.prefetch_batches = 16
        elif mem_gb >= 64:  # M2/M3 Max with 64GB+
            config.batch_size = 12
            config.batch_size_eval = 24
            config.prefetch_batches = 12
        elif mem_gb >= 32:  # M2/M3 Pro with 32GB+
            # Benchmarked: batch 12 works, gives ~0.86 samples/sec
            config.batch_size = 10
            config.batch_size_eval = 16
            config.prefetch_batches = 8
        else:  # M1/M2 Pro with 16GB
            config.batch_size = 4
            config.batch_size_eval = 8
            config.prefetch_batches = 8
    else:
        # Base M1/M2/M3 chips
        if mem_gb >= 24:
            config.batch_size = 6
            config.batch_size_eval = 12
            config.prefetch_batches = 8
        elif mem_gb >= 16:
            config.batch_size = 4
            config.batch_size_eval = 8
            config.prefetch_batches = 6
        else:  # 8GB
            config.batch_size = 2
            config.batch_size_eval = 4
            config.prefetch_batches = 4

    # Worker count: Apple Silicon benefits from fewer workers due to unified memory
    # and efficient I/O. More workers can actually hurt performance.
    config.num_workers = min(profile.performance_cores, 4)

    # MPS autocast has overhead that makes it SLOWER for complex models like DFNet4
    # Benchmarks show ~1.3x slowdown with float16 autocast on MPS
    # Keep disabled by default - users can enable via ENABLE_AMP=true if needed
    config.enable_amp = False
    config.amp_dtype = "float16"

    return config


def _tune_for_cpu(config: HardwareConfig, profile: HardwareProfile) -> HardwareConfig:
    """Tune configuration for CPU-only training."""
    config.pin_memory = False
    config.channels_last = True

    # Batch size based on RAM
    mem_gb = profile.total_memory_gb
    if mem_gb >= 64:
        config.batch_size = 8
        config.batch_size_eval = 16
    elif mem_gb >= 32:
        config.batch_size = 4
        config.batch_size_eval = 8
    elif mem_gb >= 16:
        config.batch_size = 2
        config.batch_size_eval = 4
    else:
        config.batch_size = 1
        config.batch_size_eval = 2

    # Use all CPU cores for workers
    config.num_workers = max(1, profile.cpu_count - 2)
    config.prefetch_batches = config.num_workers * 4

    # No mixed precision on CPU (usually slower)
    config.enable_amp = False

    return config


def _apply_env_overrides(config: HardwareConfig) -> HardwareConfig:
    """Apply environment variable overrides to config."""
    if "DF_DEVICE" in os.environ:
        config.device = os.environ["DF_DEVICE"]
        config.is_auto_tuned = False

    if "DF_BATCH_SIZE" in os.environ:
        config.batch_size = int(os.environ["DF_BATCH_SIZE"])
        config.is_auto_tuned = False

    if "DF_BATCH_SIZE_EVAL" in os.environ:
        config.batch_size_eval = int(os.environ["DF_BATCH_SIZE_EVAL"])
        config.is_auto_tuned = False

    if "DF_NUM_WORKERS" in os.environ:
        config.num_workers = int(os.environ["DF_NUM_WORKERS"])
        config.is_auto_tuned = False

    if "DF_PREFETCH" in os.environ:
        config.prefetch_batches = int(os.environ["DF_PREFETCH"])
        config.is_auto_tuned = False

    return config


def _get_cpu_name() -> str:
    """Get CPU name string."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # sysctl not available; fall back to platform.processor() below
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()
        except FileNotFoundError:
            # /proc/cpuinfo not available; fall back to platform.processor() below
            pass
    return platform.processor() or "Unknown CPU"


def _get_macos_version() -> Optional[Tuple[int, int]]:
    """Get macOS version as (major, minor) tuple."""
    if platform.system() != "Darwin":
        return None
    try:
        version = platform.mac_ver()[0]
        parts = version.split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except (ValueError, IndexError):
        return None


def _get_system_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024**3)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            # sysctl not available or returned invalid value; fall back to default below
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / (1024**2)
        except FileNotFoundError:
            # /proc/meminfo not available; fall back to default below
            pass

    return 8.0  # Conservative default


def _log_hardware_config(config: HardwareConfig) -> None:
    """Log detected hardware configuration."""
    profile = cast(HardwareProfile, config.profile)
    logger.info("=" * 60)
    logger.info("Hardware Auto-Detection Results")
    logger.info("=" * 60)
    logger.info(f"Platform: {profile.os_name} {profile.os_version}")
    logger.info(f"CPU: {profile.cpu_name} ({profile.cpu_count} cores)")

    if profile.is_apple_silicon:
        logger.info(f"Apple Silicon: {profile.apple_chip_variant}")
        if profile.performance_cores:
            logger.info(f"  Performance cores: {profile.performance_cores}")
            logger.info(f"  Efficiency cores: {profile.efficiency_cores}")
        if profile.gpu_cores:
            logger.info(f"  GPU cores: {profile.gpu_cores}")

    logger.info(f"Device: {profile.device_type} ({profile.device_name})")
    logger.info(f"Memory: {profile.total_memory_gb:.1f} GB total, ~{profile.available_memory_gb:.1f} GB available")

    if profile.device_type == "mps" and not profile.mps_complex_supported:
        logger.warning("MPS complex tensor support: LIMITED (macOS < 14)")
    else:
        logger.info(f"Complex tensor support: {'Yes' if profile.supports_complex else 'Limited'}")

    logger.info("-" * 60)
    logger.info("Recommended Settings:")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Batch size (train): {config.batch_size}")
    logger.info(f"  Batch size (eval): {config.batch_size_eval}")
    logger.info(f"  Workers: {config.num_workers}")
    logger.info(f"  Prefetch batches: {config.prefetch_batches}")
    logger.info(f"  Mixed precision: {config.enable_amp} ({config.amp_dtype})")
    logger.info("=" * 60)

    if not config.is_auto_tuned:
        logger.info("Note: Some settings overridden via environment variables")


def auto_tune_training_config(config_path: str, output_path: Optional[str] = None) -> HardwareConfig:
    """Auto-tune an existing training config file for the current hardware.

    Args:
        config_path: Path to existing config.ini file.
        output_path: Path to write tuned config. If None, modifies in place.

    Returns:
        HardwareConfig with applied settings.
    """
    from configparser import ConfigParser

    hw_config = HardwareConfig.detect()

    parser = ConfigParser()
    parser.read(config_path)

    hw_config.apply_to_config(parser)

    output = output_path or config_path
    with open(output, "w") as f:
        parser.write(f)

    logger.info(f"Auto-tuned config written to: {output}")
    return hw_config


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect hardware and show recommended settings")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--apply", type=str, help="Apply settings to config file")
    parser.add_argument("--output", type=str, help="Output path for modified config")
    args = parser.parse_args()

    config = HardwareConfig.detect(verbose=not args.json)

    if args.json:
        import json

        print(json.dumps(config.to_dict(), indent=2))
    elif args.apply:
        auto_tune_training_config(args.apply, args.output)
