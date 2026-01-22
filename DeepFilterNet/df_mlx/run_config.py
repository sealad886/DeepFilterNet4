from __future__ import annotations

import difflib
from dataclasses import dataclass, field, fields, is_dataclass
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
    warmup_epochs: int = cfg_field(5, help="Warmup epochs", normalize=lambda v: _normalize_int(v, min_value=0))
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
        10, help="Sync/eval frequency in batches", normalize=lambda v: _normalize_int(v, min_value=1)
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


@dataclass
class DataloaderConfig:
    num_workers: int = cfg_field(4, help="Data loader workers", normalize=lambda v: _normalize_int(v, min_value=0))
    prefetch_size: int = cfg_field(8, help="Prefetch size", normalize=lambda v: _normalize_int(v, min_value=1))
    use_mlx_data: bool = cfg_field(True, help="Use MLXDataStream if available", normalize=_normalize_bool)
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
        0, help="Save data checkpoint every N batches (0 disables)", normalize=lambda v: _normalize_int(v, min_value=0)
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
class LossConfig:
    dynamic_loss: str = cfg_field(
        "baseline",
        help="Dynamic loss: baseline | awesome",
        choices=["baseline", "awesome"],
        normalize=lambda v: str(v),
        notes="If not 'awesome', the [loss.awesome] block is ignored.",
    )
    awesome: AwesomeLossConfig = field(default_factory=AwesomeLossConfig)


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
        1, help="Evaluate VAD metrics every N epochs", normalize=lambda v: _normalize_int(v, min_value=1)
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
        300.0, help="Speech band low cutoff (Hz)", normalize=lambda v: _normalize_float(v, min_value=1.0)
    )
    band_high_hz: float = cfg_field(
        3400.0, help="Speech band high cutoff (Hz)", normalize=lambda v: _normalize_float(v, min_value=1.0)
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


@dataclass
class RunConfig:
    dataset: DatasetRunConfig = field(default_factory=DatasetRunConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


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


def apply_run_config_dict(cfg: RunConfig, data: dict[str, Any], *, path: str = "") -> None:
    if not isinstance(data, dict):
        raise TypeError(f"Expected a table at '{path or 'root'}'")

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
            apply_run_config_dict(current, value, path=_format_path(path, key))
            continue

        normalize: Callable[[Any], Any] | None = field_def.metadata.get("normalize")
        if normalize is not None:
            try:
                normalized = normalize(value)
            except Exception as exc:
                raise ValueError(f"Invalid value for '{_format_path(path, key)}': {exc}") from exc
        else:
            normalized = value

        choices = field_def.metadata.get("choices")
        if choices is not None and normalized is not None and normalized not in choices:
            raise ValueError(f"Invalid value for '{_format_path(path, key)}': {normalized}. Allowed: {choices}")

        setattr(cfg, key, normalized)


def load_run_config(path: str | Path, *, base: RunConfig | None = None) -> RunConfig:
    cfg = base or RunConfig()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    apply_run_config_dict(cfg, data)
    return cfg


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

    # Probability checks
    if not (0.0 <= cfg.augmentation.p_reverb <= 1.0):
        raise ValueError("augmentation.p_reverb must be in [0,1]")
    if not (0.0 <= cfg.augmentation.p_clipping <= 1.0):
        raise ValueError("augmentation.p_clipping must be in [0,1]")
    if cfg.vad.train.prob < 0.0 or cfg.vad.train.prob > 1.0:
        raise ValueError("vad.train.prob must be in [0,1]")


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
    lines.append("")

    _emit_section(lines, "dataset", cfg.dataset)
    _emit_section(lines, "augmentation", cfg.augmentation)
    _emit_section(lines, "training", cfg.training)
    _emit_section(lines, "dataloader", cfg.dataloader)
    _emit_section(lines, "checkpoint", cfg.checkpoint)
    _emit_section(lines, "model", cfg.model)
    _emit_section(lines, "loss", cfg.loss)
    _emit_section(lines, "loss.awesome", cfg.loss.awesome)
    _emit_section(lines, "vad", cfg.vad)
    _emit_section(lines, "vad.eval", cfg.vad.eval)
    _emit_section(lines, "vad.train", cfg.vad.train)
    _emit_section(lines, "metrics", cfg.metrics)
    _emit_section(lines, "debug", cfg.debug)

    return "\n".join(lines).rstrip() + "\n"
