from __future__ import annotations

import difflib
import logging
import subprocess
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - covered via tomli in tests
    try:
        import tomli as tomllib  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TOML parser not available. Install 'tomli' for Python <3.11.") from exc


def cfg_field(default: Any, **meta: Any) -> Any:
    return field(default=default, metadata=meta)


# ============================
# Normalizers / Validators
# ============================


def _normalize_optional_str(value: Any, *, none_sentinel: str = "") -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return None if value == none_sentinel else value
    raise TypeError("expected string")


def _normalize_optional_int(
    value: Any,
    *,
    none_sentinel: int = -1,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("expected int")
    if isinstance(value, int):
        if value == none_sentinel:
            return None
        if min_value is not None and value < min_value:
            raise ValueError(f"expected >= {min_value}")
        if max_value is not None and value > max_value:
            raise ValueError(f"expected <= {max_value}")
        return value
    raise TypeError("expected int")


def _normalize_optional_float(
    value: Any,
    *,
    none_sentinel: float = -1.0,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("expected float")
    if isinstance(value, (int, float)):
        if float(value) == none_sentinel:
            return None
        if min_value is not None and float(value) < min_value:
            raise ValueError(f"expected >= {min_value}")
        if max_value is not None and float(value) > max_value:
            raise ValueError(f"expected <= {max_value}")
        return float(value)
    raise TypeError("expected float")


def _normalize_range(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) != 2:
            raise ValueError("expected list/tuple of length 2")
        return (float(value[0]), float(value[1]))
    raise TypeError("expected list/tuple")


def _normalize_probability(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("expected float")
    if isinstance(value, (int, float)):
        val = float(value)
        if not (0.0 <= val <= 1.0):
            raise ValueError("expected in [0, 1]")
        return val
    raise TypeError("expected float")


def _normalize_fp16(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "auto":
            return None
        raise ValueError("expected 'auto', true, or false")
    raise TypeError("expected bool or 'auto'")


def _normalize_resume(value: Any) -> bool | str:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return False if value == "" else value
    raise TypeError("expected bool or string")


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError("expected bool")


def _normalize_int(value: Any, *, min_value: int | None = None) -> int:
    if isinstance(value, bool):
        raise TypeError("expected int")
    if isinstance(value, int):
        if min_value is not None and value < min_value:
            raise ValueError(f"expected >= {min_value}")
        return value
    raise TypeError("expected int")


def _normalize_float(value: Any, *, min_value: float | None = None) -> float:
    if isinstance(value, bool):
        raise TypeError("expected float")
    if isinstance(value, (int, float)):
        val = float(value)
        if min_value is not None and val < min_value:
            raise ValueError(f"expected >= {min_value}")
        return val
    raise TypeError("expected float")


def _normalize_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise TypeError("expected list of ints")


def _normalize_optional_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return [int(v) for v in value]
    raise TypeError("expected list of ints")


def _normalize_table(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise TypeError("expected table")


def _normalize_pipeline_stages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError("expected list of tables")

    stages: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"expected table at index {i}")
        if "start_epoch" not in item:
            raise ValueError(f"pipeline stage at index {i} is missing required key 'start_epoch'")

        stage: dict[str, Any] = {"start_epoch": _normalize_int(item["start_epoch"], min_value=0)}

        if stage["start_epoch"] in seen_epochs:
            raise ValueError(f"duplicate pipeline stage start_epoch={stage['start_epoch']}")
        seen_epochs.add(stage["start_epoch"])

        name = item.get("name")
        if name is not None:
            if not isinstance(name, str):
                raise TypeError(f"pipeline stage name at index {i} must be a string")
            stage["name"] = name

        awesome_loss_weight = item.get("awesome_loss_weight")
        if awesome_loss_weight is not None:
            stage["awesome_loss_weight"] = _normalize_float(awesome_loss_weight, min_value=0.0)

        vad_loss_weight = item.get("vad_loss_weight")
        if vad_loss_weight is not None:
            stage["vad_loss_weight"] = _normalize_float(vad_loss_weight, min_value=0.0)

        vad_speech_loss_weight = item.get("vad_speech_loss_weight")
        if vad_speech_loss_weight is not None:
            stage["vad_speech_loss_weight"] = _normalize_float(vad_speech_loss_weight, min_value=0.0)

        stages.append(stage)

    stages.sort(key=lambda s: int(s["start_epoch"]))
    return stages


# ============================
# Config dataclasses
# ============================


@dataclass
class DatasetRunConfig:
    cache_dir: str | None = cfg_field(
        None,
        help="Path to pre-built audio cache (from build_audio_cache.py)",
        normalize=_normalize_optional_str,
        none_sentinel="",
    )
    speech_list: str | None = cfg_field(
        None,
        help="Path to file containing speech file paths (one per line)",
        normalize=_normalize_optional_str,
        none_sentinel="",
    )
    noise_list: str | None = cfg_field(
        None,
        help="Path to file containing noise file paths (one per line)",
        normalize=_normalize_optional_str,
        none_sentinel="",
    )
    rir_list: str | None = cfg_field(
        None,
        help="Path to file containing RIR file paths (one per line)",
        normalize=_normalize_optional_str,
        none_sentinel="",
    )
    config: str | None = cfg_field(
        None,
        help="Dataset/mixer config JSON path (same as --config)",
        normalize=_normalize_optional_str,
        none_sentinel="",
    )
    snr_range: tuple[float, float] | None = cfg_field(
        None,
        help="Override base SNR range in dB (e.g., [-5, 40])",
        normalize=_normalize_range,
        none_sentinel=[],
    )
    snr_range_extreme: tuple[float, float] | None = cfg_field(
        None,
        help="Override extreme SNR range in dB (e.g., [-20, -5])",
        normalize=_normalize_range,
        none_sentinel=[],
    )
    p_extreme_snr: float | None = cfg_field(
        None,
        help="Probability of sampling from extreme SNR range (0-1)",
        normalize=lambda v: _normalize_optional_float(v, none_sentinel=-1.0, min_value=0.0, max_value=1.0),
        none_sentinel=-1.0,
    )
    snr_range_very_low: tuple[float, float] | None = cfg_field(
        None,
        help="Override very-low SNR range in dB (e.g., [-30, -20])",
        normalize=_normalize_range,
        none_sentinel=[],
    )
    p_very_low_snr: float | None = cfg_field(
        None,
        help="Probability of sampling from very-low SNR range (0-1)",
        normalize=lambda v: _normalize_optional_float(v, none_sentinel=-1.0, min_value=0.0, max_value=1.0),
        none_sentinel=-1.0,
    )
    p_interfer_speech: float | None = cfg_field(
        None,
        help="Probability of adding interfering speaker (0-1)",
        normalize=lambda v: _normalize_optional_float(v, none_sentinel=-1.0, min_value=0.0, max_value=1.0),
        none_sentinel=-1.0,
    )
    speech_gain_range: tuple[float, float] | None = cfg_field(
        None,
        help="Override speech gain range in dB (e.g., [-12, 12])",
        normalize=_normalize_range,
        none_sentinel=[],
    )
    noise_gain_range: tuple[float, float] | None = cfg_field(
        None,
        help="Override noise gain range in dB (e.g., [-12, 12])",
        normalize=_normalize_range,
        none_sentinel=[],
    )


@dataclass
class AugmentationConfig:
    p_reverb: float = cfg_field(
        0.5,
        help="Probability of applying reverb",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
        max=1.0,
    )
    p_clipping: float = cfg_field(
        0.0,
        help="Probability of clipping distortion",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
        max=1.0,
    )


@dataclass
class TrainingConfig:
    epochs: int = cfg_field(100, help="Number of training epochs", normalize=lambda v: _normalize_int(v, min_value=1))
    batch_size: int = cfg_field(8, help="Batch size", normalize=lambda v: _normalize_int(v, min_value=1))
    learning_rate: float = cfg_field(
        1e-4,
        help="Initial learning rate",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    learning_rate_min: float | None = cfg_field(
        None,
        help="Minimum learning rate for cosine schedule (None=1% of base)",
        normalize=lambda v: _normalize_optional_float(v, none_sentinel=-1.0, min_value=0.0),
        none_sentinel=-1.0,
    )
    weight_decay: float = cfg_field(
        0.0,
        help="Weight decay for AdamW",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    warmup_epochs: int = cfg_field(5, help="Warmup epochs", normalize=lambda v: _normalize_int(v, min_value=0))
    curriculum_warmup_epochs: int = cfg_field(
        0,
        help="Curriculum learning warmup epochs (0=disabled). SNR/interferer probabilities ramp from 0 to target.",
        normalize=lambda v: _normalize_int(v, min_value=0),
    )
    patience: int = cfg_field(10, help="Early stopping patience", normalize=lambda v: _normalize_int(v, min_value=0))
    grad_accumulation_steps: int = cfg_field(
        1, help="Gradient accumulation steps", normalize=lambda v: _normalize_int(v, min_value=1)
    )
    max_grad_norm: float = cfg_field(
        1.0,
        help="Maximum gradient norm",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    eval_frequency: int = cfg_field(
        10,
        help="Sync/eval frequency in batches",
        normalize=lambda v: _normalize_int(v, min_value=1),
    )
    fp16: bool | None = cfg_field(
        None,
        help="Enable FP16 (true/false) or 'auto' for hardware default",
        normalize=_normalize_fp16,
        none_sentinel="auto",
        choices=["auto", True, False],
        notes="If set to 'auto', hardware defaults determine FP16 usage.",
    )
    seed: int | None = cfg_field(
        None,
        help="Optional RNG seed override (-1 disables override)",
        normalize=lambda v: _normalize_optional_int(v, none_sentinel=-1, min_value=0),
        none_sentinel=-1,
    )
    train_config: str | None = cfg_field(
        None,
        help="Path to train.py-compatible INI config (optional)",
        normalize=_normalize_optional_str,
        none_sentinel="",
    )


# ============================
# Hardware Tuning Profiles
# ============================

_logger = logging.getLogger(__name__)

HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "entry": {"loader": "prefetch", "num_workers": 2, "prefetch_size": 4},
    "pro": {"loader": "prefetch", "num_workers": 4, "prefetch_size": 8},
    "max": {"loader": "mlx_data", "num_workers": 6, "prefetch_size": 12},
    "ultra": {"loader": "mlx_data", "num_workers": 8, "prefetch_size": 16},
}

_CONSERVATIVE_PROFILE: dict[str, Any] = {
    "loader": "prefetch",
    "num_workers": 2,
    "prefetch_size": 4,
}


def detect_hardware_class() -> str:
    """Return the Apple Silicon hardware class: entry, pro, max, ultra, or unknown."""
    try:
        brand = (
            subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            .lower()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"

    if "ultra" in brand:
        return "ultra"
    if "max" in brand:
        return "max"
    if "pro" in brand:
        return "pro"
    if any(tag in brand for tag in ("m1", "m2", "m3", "m4", "apple")):
        return "entry"
    return "unknown"


def get_hardware_tuning_profile() -> dict[str, Any]:
    """Detect Apple Silicon class and return recommended data pipeline settings.

    Returns a dict with keys ``loader``, ``num_workers``, and ``prefetch_size``.
    Falls back to conservative defaults when the chip cannot be identified.
    """
    hw_class = detect_hardware_class()
    profile = HARDWARE_PROFILES.get(hw_class, _CONSERVATIVE_PROFILE).copy()
    profile["hardware_class"] = hw_class
    return profile


@dataclass
class DataloaderConfig:
    num_workers: int = cfg_field(4, help="Data loader workers", normalize=lambda v: _normalize_int(v, min_value=0))
    prefetch_size: int = cfg_field(8, help="Prefetch size", normalize=lambda v: _normalize_int(v, min_value=1))
    use_mlx_data: bool = cfg_field(True, help="Use MLXDataStream if available", normalize=_normalize_bool)
    auto_tune_dataloader: bool = cfg_field(
        False,
        help="Auto-detect hardware class and apply recommended worker/prefetch settings",
        normalize=_normalize_bool,
    )
    max_train_batches: int | None = cfg_field(
        None,
        help="Limit number of train batches per epoch (-1 disables)",
        normalize=lambda v: _normalize_optional_int(v, none_sentinel=-1, min_value=1),
        none_sentinel=-1,
    )
    max_valid_batches: int | None = cfg_field(
        None,
        help="Limit number of validation batches (-1 disables)",
        normalize=lambda v: _normalize_optional_int(v, none_sentinel=-1, min_value=1),
        none_sentinel=-1,
    )
    shuffle_buffer_size: int = cfg_field(
        0,
        help="Shuffle buffer size (0 = strict ordering, >0 = throughput-optimized shuffle)",
        normalize=lambda v: _normalize_int(v, min_value=0),
    )


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = cfg_field(
        "checkpoints",
        help="Directory for checkpoints",
        normalize=lambda v: _normalize_optional_str(v, none_sentinel="") or "checkpoints",
    )
    save_strategy: str = cfg_field(
        "epoch",
        help="Checkpoint cadence: no | epoch | steps",
        choices=["no", "epoch", "steps"],
        normalize=lambda v: str(v),
        notes="If save_strategy='steps', save_steps must be > 0.",
    )
    save_steps: int = cfg_field(
        500, help="Steps between checkpoints", normalize=lambda v: _normalize_int(v, min_value=1)
    )
    save_total_limit: int | None = cfg_field(
        None,
        help="Max checkpoints to keep (-1 disables pruning)",
        normalize=lambda v: _normalize_optional_int(v, none_sentinel=-1, min_value=1),
        none_sentinel=-1,
    )
    checkpoint_batches: int = cfg_field(
        0,
        help="Save data checkpoint every N batches (0 disables)",
        normalize=lambda v: _normalize_int(v, min_value=0),
    )
    validate_every: int = cfg_field(
        1, help="Validate every N epochs", normalize=lambda v: _normalize_int(v, min_value=1)
    )
    resume: bool | str = cfg_field(
        False,
        help="Resume from checkpoint: true (auto) | false | path",
        normalize=_normalize_resume,
    )
    resume_data: bool | str = cfg_field(
        False,
        help="Resume data state: true (auto) | false | path",
        normalize=_normalize_resume,
    )
    check_chkpts: bool = cfg_field(False, help="Validate checkpoints before start", normalize=_normalize_bool)


@dataclass
class ModelConfig:
    backbone_type: str = cfg_field(
        "mamba",
        help="Backbone type: mamba | gru | attention",
        choices=["mamba", "gru", "attention"],
        normalize=lambda v: str(v),
    )
    variant: str = cfg_field(
        "full",
        help="Model variant: full | lite",
        choices=["full", "lite"],
        normalize=lambda v: str(v),
    )


@dataclass
class AwesomeLossConfig:
    loss_weight: float = cfg_field(
        0.4,
        help="Awesome loss weight",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    mask_sharpness: float = cfg_field(
        6.0,
        help="Mask sharpness",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    warmup_steps: int = cfg_field(
        0, help="Warmup steps for awesome loss", normalize=lambda v: _normalize_int(v, min_value=0)
    )
    proxy_enabled: bool = cfg_field(
        True,
        help="Enable cheap VAD proxy gating",
        normalize=_normalize_bool,
    )


@dataclass
class MultiResSpecLossConfig:
    factor: float = cfg_field(
        0.0,
        help="Multi-res STFT loss weight (0 disables)",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    gamma: float = cfg_field(
        1.0,
        help="Magnitude compression exponent",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    f_complex: float | None = cfg_field(
        None,
        help="Complex loss weight (None disables)",
        normalize=lambda v: _normalize_optional_float(v, none_sentinel=-1.0, min_value=0.0),
        none_sentinel=-1.0,
    )
    fft_sizes: list[int] = field(
        default_factory=lambda: [512, 1024, 2048],
        metadata={"help": "FFT sizes for multi-res loss", "normalize": _normalize_int_list},
    )
    hop_sizes: list[int] | None = cfg_field(
        None,
        help="Hop sizes for multi-res loss (None=fft_size//4)",
        normalize=_normalize_optional_int_list,
        none_sentinel=[],
    )


@dataclass
class LossConfig:
    dynamic_loss: str = cfg_field(
        "baseline",
        help="Dynamic loss: baseline | awesome | pipeline_awesome",
        choices=["baseline", "awesome", "pipeline_awesome"],
        normalize=lambda v: str(v),
        notes="If not 'awesome' or 'pipeline_awesome', the [loss.awesome] block is ignored.",
    )
    pipeline_stages: list[dict[str, Any]] = field(
        default_factory=list,
        metadata={
            "help": (
                "Optional staged loss schedule. Each item supports: "
                "start_epoch (required), name, awesome_loss_weight, "
                "vad_loss_weight, vad_speech_loss_weight."
            ),
            "normalize": _normalize_pipeline_stages,
            "none_sentinel": [],
            "notes": (
                "Example: pipeline_stages = ["
                "{start_epoch=0, name='bootstrap', awesome_loss_weight=0.2}, "
                "{start_epoch=5, name='refine', awesome_loss_weight=0.4, vad_loss_weight=0.05}]"
            ),
        },
    )
    awesome: AwesomeLossConfig = field(default_factory=AwesomeLossConfig)
    mrstft: MultiResSpecLossConfig = field(default_factory=MultiResSpecLossConfig)


@dataclass
class GanConfig:
    enabled: bool = cfg_field(False, help="Enable GAN adversarial training", normalize=_normalize_bool)
    start_epoch: int = cfg_field(
        0,
        help="Epoch to start GAN training (0-based)",
        normalize=lambda v: _normalize_int(v, min_value=0),
    )
    ramp_epochs: int = cfg_field(
        0,
        help="Linearly ramp GAN weights over N epochs (0 disables ramp)",
        normalize=lambda v: _normalize_int(v, min_value=0),
    )
    adv_weight: float = cfg_field(
        0.0,
        help="Generator adversarial loss weight",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    fm_weight: float = cfg_field(
        0.0,
        help="Feature matching loss weight",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    discriminator: str = cfg_field(
        "combined",
        help="Discriminator type: combined | mpd | msd",
        choices=["combined", "mpd", "msd"],
        normalize=lambda v: str(v),
    )
    mpd_periods: list[int] = field(
        default_factory=lambda: [2, 3, 5, 7, 11],
        metadata={"help": "MPD periods", "normalize": _normalize_int_list},
    )
    msd_scales: int = cfg_field(3, help="MSD number of scales", normalize=lambda v: _normalize_int(v, min_value=1))
    disc_lr: float = cfg_field(
        1e-4,
        help="Discriminator learning rate",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    disc_weight_decay: float = cfg_field(
        0.0,
        help="Discriminator weight decay",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    disc_grad_clip: float = cfg_field(
        1.0,
        help="Discriminator gradient clipping",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    disc_update_freq: int = cfg_field(
        1,
        help="Update discriminator every N steps",
        normalize=lambda v: _normalize_int(v, min_value=1),
    )
    disc_max_samples: int = cfg_field(
        48000,
        help="Max waveform samples fed to discriminator to limit memory",
        normalize=lambda v: _normalize_int(v, min_value=0),
    )
    mpd_channels: int = cfg_field(
        32,
        help="Base channel count for MPD period discriminators",
        normalize=lambda v: _normalize_int(v, min_value=8),
    )
    msd_channels: int = cfg_field(
        128,
        help="Base channel count for MSD scale discriminators",
        normalize=lambda v: _normalize_int(v, min_value=16),
    )
    experimental_compile: bool = cfg_field(
        False,
        help="Enable experimental GAN-phase compilation (R&D only, see docs/GAN_COMPILE_EXPERIMENT.md)",
        normalize=_normalize_bool,
    )


@dataclass
class VADEvalConfig:
    mode: str = cfg_field(
        "auto",
        help="VAD eval mode: auto | proxy | silero | off",
        choices=["auto", "proxy", "silero", "off"],
        normalize=lambda v: str(v),
        notes="If mode='silero', install: pip install silero-vad onnxruntime torch.",
    )
    every: int = cfg_field(
        1,
        help="Evaluate VAD metrics every N epochs",
        normalize=lambda v: _normalize_int(v, min_value=1),
    )
    batches: int = cfg_field(
        8, help="Number of batches for VAD eval", normalize=lambda v: _normalize_int(v, min_value=1)
    )
    max_seconds: float = cfg_field(
        0.0,
        help="Max seconds per clip for VAD eval (0 disables)",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    silero_model_path: str | None = cfg_field(
        None,
        help="Path to silero_vad.onnx (optional)",
        normalize=_normalize_optional_str,
        none_sentinel="",
    )
    silero_sample_rate: int = cfg_field(
        16000, help="Silero sample rate (Hz)", normalize=lambda v: _normalize_int(v, min_value=8000)
    )


@dataclass
class VADTrainConfig:
    prob: float = cfg_field(
        0.0,
        help="Probability of VAD regularizer per batch",
        normalize=_normalize_probability,
        min=0.0,
        max=1.0,
    )
    every_steps: int = cfg_field(
        0,
        help="Apply VAD regularizer every N steps (0 disables)",
        normalize=lambda v: _normalize_int(v, min_value=0),
    )


@dataclass
class VADConfig:
    loss_weight: float = cfg_field(
        0.05,
        help="VAD loss weight",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    threshold: float = cfg_field(
        0.6,
        help="VAD probability threshold for speech gating",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
        max=1.0,
    )
    margin: float = cfg_field(
        0.05,
        help="VAD margin",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    speech_loss_weight: float = cfg_field(
        0.0,
        help="VAD speech-structure loss weight",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    warmup_epochs: int = cfg_field(5, help="VAD warmup epochs", normalize=lambda v: _normalize_int(v, min_value=0))
    snr_gate_db: float = cfg_field(-10.0, help="SNR gate threshold (dB)", normalize=_normalize_float)
    snr_gate_width: float = cfg_field(
        6.0, help="SNR gate softness (dB)", normalize=lambda v: _normalize_float(v, min_value=1e-3)
    )
    band_low_hz: float = cfg_field(
        300.0,
        help="Speech band low cutoff (Hz)",
        normalize=lambda v: _normalize_float(v, min_value=1.0),
    )
    band_high_hz: float = cfg_field(
        3400.0,
        help="Speech band high cutoff (Hz)",
        normalize=lambda v: _normalize_float(v, min_value=1.0),
    )
    z_threshold: float = cfg_field(0.0, help="VAD z-score threshold", normalize=_normalize_float)
    z_slope: float = cfg_field(1.0, help="VAD z-score slope", normalize=lambda v: _normalize_float(v, min_value=1e-3))
    eval: VADEvalConfig = field(default_factory=VADEvalConfig)
    train: VADTrainConfig = field(default_factory=VADTrainConfig)


@dataclass
class MetricsConfig:
    eval_sisdr: bool = cfg_field(False, help="Compute SI-SDR during validation", normalize=_normalize_bool)


@dataclass
class DebugConfig:
    verbose: bool = cfg_field(False, help="Verbose timing/logging", normalize=_normalize_bool)
    debug_numerics: bool = cfg_field(False, help="Enable numeric debug mode", normalize=_normalize_bool)
    debug_numerics_fail_fast: bool = cfg_field(True, help="Fail fast on non-finite", normalize=_normalize_bool)
    debug_numerics_every: int = cfg_field(
        1, help="Check every N steps", normalize=lambda v: _normalize_int(v, min_value=1)
    )
    debug_numerics_dump_dir: str | None = cfg_field(
        None,
        help="Directory for numeric debug dumps (empty disables override)",
        normalize=_normalize_optional_str,
        none_sentinel="",
    )
    debug_numerics_dump_arrays: bool = cfg_field(False, help="Dump small tensor slices", normalize=_normalize_bool)
    debug_numerics_max_dumps: int = cfg_field(
        5, help="Max debug dumps", normalize=lambda v: _normalize_int(v, min_value=1)
    )
    nan_skip_batch: bool = cfg_field(False, help="Skip optimizer update on non-finite", normalize=_normalize_bool)
    sync_mode: str = cfg_field(
        "normal",
        help="Sync barrier budget: fast | normal | debug | profile",
        choices=["fast", "normal", "debug", "profile"],
        normalize=lambda v: str(v),
        notes=(
            "Controls eval_frequency default and metric verbosity. "
            "'fast' minimizes syncs for throughput, 'debug' syncs every step."
        ),
    )

    @property
    def sync_mode_enum(self) -> SyncMode:
        return SyncMode(self.sync_mode)


SYNC_MODE_EVAL_FREQUENCY: dict[str, int] = {
    "fast": 50,
    "normal": 10,
    "debug": 1,
    "profile": 5,
}
"""Recommended eval_frequency for each sync_mode."""


class SyncMode(str, Enum):
    """Training sync barrier budget levels.

    Each level defines which diagnostics are active:
    - FAST: minimal syncs, only loss + grad_norm + samples_per_sec
    - NORMAL: per-component loss decomposition, mask/VAD stats
    - DEBUG: per-step grad_norm + loss logging
    - PROFILE: per-step data/fwd/total timing breakdown
    """

    FAST = "fast"
    NORMAL = "normal"
    DEBUG = "debug"
    PROFILE = "profile"

    @property
    def emit_detailed_metrics(self) -> bool:
        return self != SyncMode.FAST

    @property
    def emit_snr_buckets(self) -> bool:
        return self != SyncMode.FAST

    @property
    def emit_mask_stats(self) -> bool:
        return self != SyncMode.FAST


@dataclass
class EnhanceRunConfig:
    speech_boost_db: float = cfg_field(
        0.0,
        help="Boost dB applied only to Silero-detected speech segments (0 disables)",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    speech_boost_threshold: float = cfg_field(
        0.5,
        help="Silero speech probability threshold for segment detection",
        normalize=_normalize_probability,
        min=0.0,
        max=1.0,
    )
    speech_boost_min_speech_ms: int = cfg_field(
        250,
        help="Minimum speech segment length in milliseconds",
        normalize=lambda v: _normalize_int(v, min_value=0),
        min=0,
    )
    speech_boost_min_silence_ms: int = cfg_field(
        100,
        help="Minimum silence length to split speech segments (milliseconds)",
        normalize=lambda v: _normalize_int(v, min_value=0),
        min=0,
    )
    speech_boost_pad_ms: int = cfg_field(
        30,
        help="Padding added around detected speech segments (milliseconds)",
        normalize=lambda v: _normalize_int(v, min_value=0),
        min=0,
    )
    speech_boost_ramp_ms: float = cfg_field(
        8.0,
        help="Fade-in/out ramp around boosted segments (milliseconds)",
        normalize=lambda v: _normalize_float(v, min_value=0.0),
        min=0.0,
    )
    speech_boost_peak_limit: float = cfg_field(
        0.99,
        help="Peak limiter after speech boost (set <=0 to disable)",
        normalize=_normalize_float,
    )
    speech_boost_silero_model_path: str | None = cfg_field(
        None,
        help="Optional path to silero_vad.onnx for speech-segment detection",
        normalize=_normalize_optional_str,
        none_sentinel="",
    )
    speech_boost_silero_sample_rate: int = cfg_field(
        16000,
        help="Silero VAD sample rate used for speech-segment detection",
        normalize=lambda v: _normalize_int(v, min_value=8000),
        min=8000,
    )


@dataclass
class RunConfig:
    dataset: DatasetRunConfig = field(default_factory=DatasetRunConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    gan: GanConfig = field(default_factory=GanConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    enhance: EnhanceRunConfig = field(default_factory=EnhanceRunConfig)
    train_ini: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Embedded train.py INI compatibility tables (optional)",
            "normalize": _normalize_table,
            "notes": (
                "Use [train_ini.<section>] to inline legacy train.ini sections. "
                "Supported sections: df, train, optim, distortion, deepfilternet4, "
                "loss, MultiResSpecLoss, GANLoss, FeatureMatchingLoss."
            ),
        },
    )


# ============================
# Loading / Merging
# ============================


def _format_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _field_map(cfg: Any) -> dict[str, Any]:
    return {f.name: f for f in fields(cfg)}


def _suggest_keys(key: str, choices: list[str]) -> str:
    matches = difflib.get_close_matches(key, choices, n=3)
    if not matches:
        return ""
    return " Did you mean: " + ", ".join(matches)


def apply_run_config_dict(
    cfg: RunConfig,
    data: dict[str, Any],
    *,
    path: str = "",
    _explicitly_set: set[str] | None = None,
) -> None:
    if not isinstance(data, dict):
        raise TypeError(f"Expected a table at '{path or 'root'}'")

    if _explicitly_set is None:
        _explicitly_set = set()

    fmap = _field_map(cfg)
    for key, value in data.items():
        if key not in fmap:
            suggestion = _suggest_keys(key, list(fmap.keys()))
            raise ValueError(f"Unknown key '{_format_path(path, key)}'.{suggestion}")
        field_def = fmap[key]
        current = getattr(cfg, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Expected table for '{_format_path(path, key)}'")
            apply_run_config_dict(current, value, path=_format_path(path, key), _explicitly_set=_explicitly_set)
            continue

        full_key = _format_path(path, key)
        _explicitly_set.add(full_key)

        normalize: Callable[[Any], Any] | None = field_def.metadata.get("normalize")
        if normalize is not None:
            try:
                normalized = normalize(value)
            except Exception as exc:
                raise ValueError(f"Invalid value for '{full_key}': {exc}") from exc
        else:
            normalized = value

        choices = field_def.metadata.get("choices")
        if choices is not None and normalized is not None and normalized not in choices:
            raise ValueError(f"Invalid value for '{full_key}': {normalized}. Allowed: {choices}")

        setattr(cfg, key, normalized)

    # Stash the explicitly-set keys on the top-level RunConfig for resolve_run_config.
    if not path and isinstance(cfg, RunConfig):
        cfg._explicitly_set = _explicitly_set  # type: ignore[attr-defined]


def load_run_config(path: str | Path, *, base: RunConfig | None = None) -> RunConfig:
    cfg = base or RunConfig()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    apply_run_config_dict(cfg, data)
    return cfg


# ============================
# Preset support
# ============================

PRESET_NAMES: list[str] = ["entry", "pro", "max", "ultra", "debug"]
"""Valid preset names corresponding to TOML files in ``schemas/presets/``."""

_PRESETS_DIR: Path | None = None


def _resolve_presets_dir() -> Path:
    """Return the ``schemas/presets/`` directory relative to the repo root."""
    global _PRESETS_DIR
    if _PRESETS_DIR is not None:
        return _PRESETS_DIR

    # Walk up from this file (df_mlx/run_config.py) to the repo root.
    candidate = Path(__file__).resolve().parent.parent.parent / "schemas" / "presets"
    if candidate.is_dir():
        _PRESETS_DIR = candidate
        return _PRESETS_DIR
    raise FileNotFoundError(
        f"Preset directory not found at {candidate}. " "Ensure you are running from the DeepFilterNet repository."
    )


def load_preset_config(name: str, *, base: RunConfig | None = None) -> RunConfig:
    """Load a named preset TOML as a base ``RunConfig``.

    Parameters
    ----------
    name:
        One of :data:`PRESET_NAMES` (e.g. ``"pro"``).
    base:
        Optional existing config to overlay the preset onto.

    Returns
    -------
    RunConfig with preset values applied.

    Raises
    ------
    ValueError
        If *name* is not a recognised preset.
    FileNotFoundError
        If the preset TOML file is missing on disk.
    """
    if name not in PRESET_NAMES:
        raise ValueError(f"Unknown preset '{name}'. Available presets: {', '.join(PRESET_NAMES)}")
    preset_path = _resolve_presets_dir() / f"{name}.toml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_path}")
    return load_run_config(preset_path, base=base)


def set_by_path(cfg: RunConfig, path: str, value: Any) -> None:
    parts = path.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


# ============================
# Validation
# ============================


def validate_run_config(cfg: RunConfig) -> None:
    # Dataset source validation
    if not any(
        [
            cfg.dataset.cache_dir,
            cfg.dataset.config,
            cfg.dataset.speech_list,
        ]
    ):
        raise ValueError(
            "No dataset source provided. Set one of: dataset.cache_dir, dataset.config, dataset.speech_list"
        )

    # Loss-dependent warnings/errors
    if cfg.loss.dynamic_loss != "awesome":
        # Awesome settings are ignored unless dynamic_loss=awesome
        pass

    # VAD eval Silero requirements
    if cfg.vad.eval.mode == "silero":
        try:
            from df_mlx.vad_silero import SileroVADConfig  # noqa: F401
        except Exception as exc:  # pragma: no cover - only in environments without deps
            raise RuntimeError(
                "VAD eval mode 'silero' requires optional dependencies. Install with: "
                "pip install silero-vad onnxruntime torch"
            ) from exc

    if cfg.checkpoint.save_strategy == "steps" and cfg.checkpoint.save_steps <= 0:
        raise ValueError("checkpoint.save_steps must be > 0 when save_strategy='steps'")

    if cfg.vad.eval.mode == "off" and cfg.vad.eval.batches > 0 and cfg.vad.eval.every > 0:
        pass

    if cfg.loss.mrstft.hop_sizes is not None and len(cfg.loss.mrstft.hop_sizes) != len(cfg.loss.mrstft.fft_sizes):
        raise ValueError("loss.mrstft.hop_sizes must match length of loss.mrstft.fft_sizes")

    # Probability checks
    if not (0.0 <= cfg.augmentation.p_reverb <= 1.0):
        raise ValueError("augmentation.p_reverb must be in [0,1]")
    if not (0.0 <= cfg.augmentation.p_clipping <= 1.0):
        raise ValueError("augmentation.p_clipping must be in [0,1]")
    if cfg.vad.train.prob < 0.0 or cfg.vad.train.prob > 1.0:
        raise ValueError("vad.train.prob must be in [0,1]")
    if cfg.gan.enabled and cfg.gan.adv_weight == 0.0 and cfg.gan.fm_weight == 0.0:
        raise ValueError("gan enabled but adv_weight and fm_weight are both 0")
    if cfg.gan.mpd_periods and any(p <= 0 for p in cfg.gan.mpd_periods):
        raise ValueError("gan.mpd_periods must be positive integers")
    if cfg.gan.disc_max_samples < 0:
        raise ValueError("gan.disc_max_samples must be >= 0")
    if cfg.gan.mpd_channels < 8:
        raise ValueError("gan.mpd_channels must be >= 8")
    if cfg.gan.msd_channels < 16:
        raise ValueError("gan.msd_channels must be >= 16")


# ============================
# Resolution
# ============================


def _apply_hardware_auto_tune(cfg: RunConfig) -> None:
    """Apply hardware-detected tuning profile to dataloader settings.

    Only overrides fields that were *not* explicitly set by the user in
    the TOML / CLI layer.  Explicit values always win.
    """
    profile = get_hardware_tuning_profile()
    hw_class = profile["hardware_class"]
    explicitly_set: set[str] = getattr(cfg, "_explicitly_set", set())

    if "dataloader.num_workers" not in explicitly_set:
        cfg.dataloader.num_workers = profile["num_workers"]

    if "dataloader.prefetch_size" not in explicitly_set:
        cfg.dataloader.prefetch_size = profile["prefetch_size"]

    if "dataloader.use_mlx_data" not in explicitly_set:
        cfg.dataloader.use_mlx_data = profile["loader"] == "mlx_data"

    _logger.info(
        "auto_tune_dataloader: hw_class=%s  num_workers=%d  prefetch_size=%d  use_mlx_data=%s",
        hw_class,
        cfg.dataloader.num_workers,
        cfg.dataloader.prefetch_size,
        cfg.dataloader.use_mlx_data,
    )


def resolve_run_config(cfg: RunConfig) -> None:
    """Apply mode-based defaults that depend on cross-field relationships.

    Call after ``apply_run_config_dict`` / ``set_by_path`` and before
    ``validate_run_config``.

    Handles:
    * ``debug.sync_mode`` → ``training.eval_frequency`` override when
      eval_frequency is still at its default value (10).
    * ``dataloader.auto_tune_dataloader`` → apply hardware-detected
      worker/prefetch/loader settings for any field the user did not
      explicitly set.
    """
    default_eval_freq = 10
    if cfg.training.eval_frequency == default_eval_freq:
        recommended = SYNC_MODE_EVAL_FREQUENCY.get(cfg.debug.sync_mode)
        if recommended is not None and recommended != default_eval_freq:
            cfg.training.eval_frequency = recommended

    if cfg.dataloader.auto_tune_dataloader:
        _apply_hardware_auto_tune(cfg)


# ============================
# Example generator
# ============================


def _toml_value(value: Any, meta: dict[str, Any]) -> str:
    if value is None:
        sentinel = meta.get("none_sentinel")
        value = sentinel
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        items = ", ".join(str(v) for v in value)
        return f"[{items}]"
    return f'"{value}"'


def _emit_section(lines: list[str], title: str, obj: Any) -> None:
    lines.append(f"[{title}]")
    for field_def in fields(obj):
        value = getattr(obj, field_def.name)
        meta = field_def.metadata
        if is_dataclass(value):
            continue
        help_text = meta.get("help")
        if help_text:
            lines.append(f"# {help_text}")
        choices = meta.get("choices")
        if choices:
            lines.append(f"# Allowed: {choices}")
        min_val = meta.get("min")
        max_val = meta.get("max")
        if min_val is not None or max_val is not None:
            if min_val is not None and max_val is not None:
                lines.append(f"# Range: [{min_val}, {max_val}]")
            elif min_val is not None:
                lines.append(f"# Min: {min_val}")
            else:
                lines.append(f"# Max: {max_val}")
        notes = meta.get("notes")
        if notes:
            lines.append(f"# Note: {notes}")
        default_val = value
        if default_val is None:
            default_val = meta.get("none_sentinel", None)
        lines.append(f"# Default: {default_val}")
        lines.append(f"{field_def.name} = {_toml_value(value, meta)}")
        lines.append("")


def generate_run_config_example() -> str:
    cfg = RunConfig()
    lines: list[str] = []
    lines.append("# DeepFilterNet4 train_dynamic run-config (TOML)")
    lines.append("# Precedence: defaults < run-config < explicit CLI flags")
    lines.append("# NOTE: --config remains the dataset/mixer config (JSON).")
    lines.append("# Includes all training/runtime CLI settings except meta flags:")
    lines.append("#   --run-config (path to this TOML file)")
    lines.append("#   --print-run-config (prints template and exits)")
    lines.append("")

    _emit_section(lines, "dataset", cfg.dataset)
    _emit_section(lines, "augmentation", cfg.augmentation)
    _emit_section(lines, "training", cfg.training)
    _emit_section(lines, "dataloader", cfg.dataloader)
    _emit_section(lines, "checkpoint", cfg.checkpoint)
    _emit_section(lines, "model", cfg.model)
    _emit_section(lines, "loss", cfg.loss)
    _emit_section(lines, "loss.awesome", cfg.loss.awesome)
    _emit_section(lines, "loss.mrstft", cfg.loss.mrstft)
    _emit_section(lines, "gan", cfg.gan)
    _emit_section(lines, "vad", cfg.vad)
    _emit_section(lines, "vad.eval", cfg.vad.eval)
    _emit_section(lines, "vad.train", cfg.vad.train)
    _emit_section(lines, "metrics", cfg.metrics)
    _emit_section(lines, "debug", cfg.debug)
    _emit_section(lines, "enhance", cfg.enhance)
    lines.append("# Optional: inline train.py INI-compatible sections in TOML")
    lines.append("# This lets you use a single run-config file without --train-config.")
    lines.append("# Example:")
    lines.append("# [train_ini.df]")
    lines.append("# sr = 48000")
    lines.append("# fft_size = 960")
    lines.append("# hop_size = 480")
    lines.append("# nb_erb = 32")
    lines.append("# nb_df = 96")
    lines.append("#")
    lines.append("# [train_ini.train]")
    lines.append("# max_epochs = 100")
    lines.append("# batch_size = 12")
    lines.append("# num_workers = 6")
    lines.append("# num_prefetch_batches = 18")
    lines.append("# max_sample_len_s = 5.0")
    lines.append("#")
    lines.append("# [train_ini.deepfilternet4]")
    lines.append('# backbone = "attention"')
    lines.append('# model_variant = "full"')
    lines.append("# conv_ch = 64")
    lines.append("# conv_kernel = [1, 3]")
    lines.append("# conv_stride = [1, 2]")
    lines.append("# emb_hidden_dim = 256")
    lines.append("# emb_num_layers = 4")
    lines.append("# df_hidden_dim = 256")
    lines.append("# df_num_layers = 3")
    lines.append("#")
    lines.append("# [train_ini.MultiResSpecLoss]")
    lines.append("# factor = 0.6")
    lines.append("# gamma = 0.5")
    lines.append("# factor_complex = 0.25")
    lines.append("# fft_sizes = [512, 1024, 2048]")
    lines.append("# hop_sizes = [128, 256, 512]")

    return "\n".join(lines).rstrip() + "\n"
