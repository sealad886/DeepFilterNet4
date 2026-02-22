"""CLI parsing and argument override utilities for dynamic training."""

from __future__ import annotations

import argparse
import json
from typing import Any

from df_mlx.run_config import RunConfig, set_by_path


def _flag_in_argv(flags: list[str], argv: list[str]) -> bool:
    for arg in argv:
        for flag in flags:
            if arg == flag or arg.startswith(f"{flag}="):
                return True
    return False


def _parse_pipeline_stages_cli(raw: str | None) -> list[dict[str, Any]]:
    """Parse --pipeline-stages JSON string into a normalized stage list."""
    if raw is None or raw.strip() == "":
        return []

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--pipeline-stages must be valid JSON") from exc

    if not isinstance(value, list):
        raise ValueError("--pipeline-stages must be a JSON array of stage objects")

    normalized: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    for i, stage in enumerate(value):
        if not isinstance(stage, dict):
            raise ValueError(f"pipeline stage at index {i} must be an object")
        if "start_epoch" not in stage:
            raise ValueError(f"pipeline stage at index {i} is missing required key 'start_epoch'")

        start_epoch = int(stage["start_epoch"])
        if start_epoch < 0:
            raise ValueError("pipeline stage start_epoch must be >= 0")
        if start_epoch in seen_epochs:
            raise ValueError(f"duplicate pipeline stage start_epoch={start_epoch}")
        seen_epochs.add(start_epoch)

        item: dict[str, Any] = {"start_epoch": start_epoch}
        if "name" in stage and stage["name"] is not None:
            item["name"] = str(stage["name"])

        for key in ("awesome_loss_weight", "vad_loss_weight", "vad_speech_loss_weight"):
            if key in stage and stage[key] is not None:
                val = float(stage[key])
                if val < 0.0:
                    raise ValueError(f"pipeline stage {key} must be >= 0")
                item[key] = val

        normalized.append(item)

    normalized.sort(key=lambda x: int(x["start_epoch"]))
    return normalized


def _resolve_pipeline_stage(epoch: int, stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Return active stage metadata for the provided epoch."""
    if not stages:
        return {
            "index": 0,
            "name": "default",
            "start_epoch": 0,
            "awesome_loss_weight": None,
            "vad_loss_weight": None,
            "vad_speech_loss_weight": None,
        }

    active_idx = 0
    for i, stage in enumerate(stages):
        if epoch >= int(stage["start_epoch"]):
            active_idx = i
        else:
            break

    active = stages[active_idx]
    return {
        "index": active_idx,
        "name": str(active.get("name", f"stage_{active_idx}")),
        "start_epoch": int(active["start_epoch"]),
        "awesome_loss_weight": active.get("awesome_loss_weight"),
        "vad_loss_weight": active.get("vad_loss_weight"),
        "vad_speech_loss_weight": active.get("vad_speech_loss_weight"),
    }


def _apply_cli_overrides(cfg: RunConfig, args: argparse.Namespace, argv: list[str]) -> None:
    overrides: list[tuple[list[str], str, Any]] = [
        (["--cache-dir"], "dataset.cache_dir", getattr(args, "cache_dir", None)),
        (["--speech-list"], "dataset.speech_list", getattr(args, "speech_list", None)),
        (["--noise-list"], "dataset.noise_list", getattr(args, "noise_list", None)),
        (["--rir-list"], "dataset.rir_list", getattr(args, "rir_list", None)),
        (["--config"], "dataset.config", getattr(args, "config", None)),
        (["--train-config"], "training.train_config", getattr(args, "train_config", None)),
        (["--snr-range"], "dataset.snr_range", getattr(args, "snr_range", None)),
        (
            ["--snr-range-extreme"],
            "dataset.snr_range_extreme",
            getattr(args, "snr_range_extreme", None),
        ),
        (
            ["--snr-range-very-low"],
            "dataset.snr_range_very_low",
            getattr(args, "snr_range_very_low", None),
        ),
        (["--p-extreme-snr"], "dataset.p_extreme_snr", getattr(args, "p_extreme_snr", None)),
        (["--p-very-low-snr"], "dataset.p_very_low_snr", getattr(args, "p_very_low_snr", None)),
        (
            ["--p-interfer-speech"],
            "dataset.p_interfer_speech",
            getattr(args, "p_interfer_speech", None),
        ),
        (
            ["--curriculum-warmup-epochs"],
            "training.curriculum_warmup_epochs",
            getattr(args, "curriculum_warmup_epochs", None),
        ),
        (
            ["--speech-gain-range"],
            "dataset.speech_gain_range",
            getattr(args, "speech_gain_range", None),
        ),
        (
            ["--noise-gain-range"],
            "dataset.noise_gain_range",
            getattr(args, "noise_gain_range", None),
        ),
        (["--p-reverb"], "augmentation.p_reverb", getattr(args, "p_reverb", None)),
        (["--p-clipping"], "augmentation.p_clipping", getattr(args, "p_clipping", None)),
        (["--epochs"], "training.epochs", getattr(args, "epochs", None)),
        (["--batch-size"], "training.batch_size", getattr(args, "batch_size", None)),
        (["--learning-rate"], "training.learning_rate", getattr(args, "learning_rate", None)),
        (
            ["--learning-rate-min"],
            "training.learning_rate_min",
            getattr(args, "learning_rate_min", None),
        ),
        (["--weight-decay"], "training.weight_decay", getattr(args, "weight_decay", None)),
        (["--warmup-epochs"], "training.warmup_epochs", getattr(args, "warmup_epochs", None)),
        (["--patience"], "training.patience", getattr(args, "patience", None)),
        (
            ["--grad-accumulation-steps"],
            "training.grad_accumulation_steps",
            getattr(args, "grad_accumulation_steps", None),
        ),
        (["--max-grad-norm"], "training.max_grad_norm", getattr(args, "max_grad_norm", None)),
        (["--eval-frequency"], "training.eval_frequency", getattr(args, "eval_frequency", None)),
        (["--seed"], "training.seed", getattr(args, "seed", None)),
        (["--num-workers"], "dataloader.num_workers", getattr(args, "num_workers", None)),
        (["--prefetch-size"], "dataloader.prefetch_size", getattr(args, "prefetch_size", None)),
        (
            ["--max-train-batches"],
            "dataloader.max_train_batches",
            getattr(args, "max_train_batches", None),
        ),
        (
            ["--max-valid-batches"],
            "dataloader.max_valid_batches",
            getattr(args, "max_valid_batches", None),
        ),
        (["--checkpoint-dir"], "checkpoint.checkpoint_dir", getattr(args, "checkpoint_dir", None)),
        (["--save-strategy"], "checkpoint.save_strategy", getattr(args, "save_strategy", None)),
        (["--save-steps"], "checkpoint.save_steps", getattr(args, "save_steps", None)),
        (
            ["--save-total-limit"],
            "checkpoint.save_total_limit",
            getattr(args, "save_total_limit", None),
        ),
        (
            ["--checkpoint-batches"],
            "checkpoint.checkpoint_batches",
            getattr(args, "checkpoint_batches", None),
        ),
        (["--validate-every"], "checkpoint.validate_every", getattr(args, "validate_every", None)),
        (["--resume"], "checkpoint.resume", getattr(args, "resume", None)),
        (["--resume-data"], "checkpoint.resume_data", getattr(args, "resume_data", None)),
        (["--check-chkpts"], "checkpoint.check_chkpts", getattr(args, "check_chkpts", None)),
        (
            ["--backbone", "--backbone-type"],
            "model.backbone_type",
            getattr(args, "backbone_type", None),
        ),
        (["--model-variant"], "model.variant", getattr(args, "model_variant", None)),
        (["--dynamic-loss"], "loss.dynamic_loss", getattr(args, "dynamic_loss", None)),
        (
            ["--awesome-loss-weight"],
            "loss.awesome.loss_weight",
            getattr(args, "awesome_loss_weight", None),
        ),
        (
            ["--awesome-mask-sharpness"],
            "loss.awesome.mask_sharpness",
            getattr(args, "awesome_mask_sharpness", None),
        ),
        (
            ["--awesome-warmup-steps"],
            "loss.awesome.warmup_steps",
            getattr(args, "awesome_warmup_steps", None),
        ),
        (["--mrstft-factor"], "loss.mrstft.factor", getattr(args, "mrstft_factor", None)),
        (["--mrstft-gamma"], "loss.mrstft.gamma", getattr(args, "mrstft_gamma", None)),
        (["--mrstft-f-complex"], "loss.mrstft.f_complex", getattr(args, "mrstft_f_complex", None)),
        (["--mrstft-fft-sizes"], "loss.mrstft.fft_sizes", getattr(args, "mrstft_fft_sizes", None)),
        (["--mrstft-hop-sizes"], "loss.mrstft.hop_sizes", getattr(args, "mrstft_hop_sizes", None)),
        (["--gan-enabled"], "gan.enabled", getattr(args, "gan_enabled", None)),
        (["--gan-start-epoch"], "gan.start_epoch", getattr(args, "gan_start_epoch", None)),
        (["--gan-ramp-epochs"], "gan.ramp_epochs", getattr(args, "gan_ramp_epochs", None)),
        (["--gan-adv-weight"], "gan.adv_weight", getattr(args, "gan_adv_weight", None)),
        (["--gan-fm-weight"], "gan.fm_weight", getattr(args, "gan_fm_weight", None)),
        (["--gan-discriminator"], "gan.discriminator", getattr(args, "gan_discriminator", None)),
        (["--gan-mpd-periods"], "gan.mpd_periods", getattr(args, "gan_mpd_periods", None)),
        (["--gan-msd-scales"], "gan.msd_scales", getattr(args, "gan_msd_scales", None)),
        (["--gan-disc-lr"], "gan.disc_lr", getattr(args, "gan_disc_lr", None)),
        (
            ["--gan-disc-weight-decay"],
            "gan.disc_weight_decay",
            getattr(args, "gan_disc_weight_decay", None),
        ),
        (["--gan-disc-grad-clip"], "gan.disc_grad_clip", getattr(args, "gan_disc_grad_clip", None)),
        (
            ["--gan-disc-update-freq"],
            "gan.disc_update_freq",
            getattr(args, "gan_disc_update_freq", None),
        ),
        (["--vad-loss-weight"], "vad.loss_weight", getattr(args, "vad_loss_weight", None)),
        (["--vad-threshold"], "vad.threshold", getattr(args, "vad_threshold", None)),
        (["--vad-margin"], "vad.margin", getattr(args, "vad_margin", None)),
        (
            ["--vad-speech-loss-weight"],
            "vad.speech_loss_weight",
            getattr(args, "vad_speech_loss_weight", None),
        ),
        (["--vad-warmup-epochs"], "vad.warmup_epochs", getattr(args, "vad_warmup_epochs", None)),
        (["--vad-snr-gate"], "vad.snr_gate_db", getattr(args, "vad_snr_gate", None)),
        (["--vad-snr-gate-width"], "vad.snr_gate_width", getattr(args, "vad_snr_gate_width", None)),
        (["--vad-band-low"], "vad.band_low_hz", getattr(args, "vad_band_low", None)),
        (["--vad-band-high"], "vad.band_high_hz", getattr(args, "vad_band_high", None)),
        (["--vad-z-threshold"], "vad.z_threshold", getattr(args, "vad_z_threshold", None)),
        (["--vad-z-slope"], "vad.z_slope", getattr(args, "vad_z_slope", None)),
        (["--vad-eval-mode"], "vad.eval.mode", getattr(args, "vad_eval_mode", None)),
        (["--vad-eval-every"], "vad.eval.every", getattr(args, "vad_eval_every", None)),
        (["--vad-eval-batches"], "vad.eval.batches", getattr(args, "vad_eval_batches", None)),
        (
            ["--vad-eval-max-seconds"],
            "vad.eval.max_seconds",
            getattr(args, "vad_eval_max_seconds", None),
        ),
        (
            ["--vad-silero-model-path"],
            "vad.eval.silero_model_path",
            getattr(args, "vad_silero_model_path", None),
        ),
        (
            ["--vad-silero-sample-rate"],
            "vad.eval.silero_sample_rate",
            getattr(args, "vad_silero_sample_rate", None),
        ),
        (["--vad-train-prob"], "vad.train.prob", getattr(args, "vad_train_prob", None)),
        (
            ["--vad-train-every-steps"],
            "vad.train.every_steps",
            getattr(args, "vad_train_every_steps", None),
        ),
        (["--eval-sisdr"], "metrics.eval_sisdr", getattr(args, "eval_sisdr", None)),
        (["-v", "--verbose"], "debug.verbose", getattr(args, "verbose", None)),
        (["--debug-numerics"], "debug.debug_numerics", getattr(args, "debug_numerics", None)),
        (
            ["--debug-numerics-no-fail-fast"],
            "debug.debug_numerics_fail_fast",
            not getattr(args, "debug_numerics_no_fail_fast", False),
        ),
        (
            ["--debug-numerics-every"],
            "debug.debug_numerics_every",
            getattr(args, "debug_numerics_every", None),
        ),
        (
            ["--debug-numerics-dump-dir"],
            "debug.debug_numerics_dump_dir",
            getattr(args, "debug_numerics_dump_dir", None),
        ),
        (
            ["--debug-numerics-dump-arrays"],
            "debug.debug_numerics_dump_arrays",
            getattr(args, "debug_numerics_dump_arrays", None),
        ),
        (
            ["--debug-numerics-max-dumps"],
            "debug.debug_numerics_max_dumps",
            getattr(args, "debug_numerics_max_dumps", None),
        ),
        (["--nan-skip-batch"], "debug.nan_skip_batch", getattr(args, "nan_skip_batch", None)),
    ]

    if _flag_in_argv(["--fp16"], argv) and _flag_in_argv(["--no-fp16"], argv):
        raise ValueError("Cannot pass both --fp16 and --no-fp16.")
    if _flag_in_argv(["--fp16"], argv):
        set_by_path(cfg, "training.fp16", True)
    if _flag_in_argv(["--no-fp16"], argv):
        set_by_path(cfg, "training.fp16", False)
    if _flag_in_argv(["--no-mlx-data"], argv):
        set_by_path(cfg, "dataloader.use_mlx_data", False)
    if _flag_in_argv(["--no-vad-proxy"], argv):
        set_by_path(cfg, "loss.awesome.proxy_enabled", False)

    for flags, path, value in overrides:
        if _flag_in_argv(flags, argv):
            set_by_path(cfg, path, value)

    if _flag_in_argv(["--pipeline-stages"], argv):
        parsed_stages = _parse_pipeline_stages_cli(getattr(args, "pipeline_stages", None))
        set_by_path(cfg, "loss.pipeline_stages", parsed_stages)
