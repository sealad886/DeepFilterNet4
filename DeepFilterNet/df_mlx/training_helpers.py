"""Shared helper utilities for dynamic MLX training flows."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def build_setup_panel_line(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dynamic_loss: str,
    gan_enabled: bool,
    vad_enabled: bool,
    checkpoint_dir: str,
    use_fp16: bool,
) -> str:
    """Build single-line setup metadata for the persistent setup panel."""
    return (
        "SETUP │ "
        f"epochs={epochs} "
        f"bs={batch_size} "
        f"lr={learning_rate:.1e} "
        f"loss={dynamic_loss} "
        f"gan={'on' if gan_enabled else 'off'} "
        f"vad={'on' if vad_enabled else 'off'} "
        f"fp16={'on' if use_fp16 else 'off'} "
        f"ckpt={checkpoint_dir}"
    )


def curriculum_schedule(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    target_p_extreme: float,
    target_p_very_low: float,
    target_p_interfer: float,
) -> tuple[float, float, float]:
    """Compute curriculum-scheduled SNR and interferer probabilities."""
    del total_epochs
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return target_p_extreme, target_p_very_low, target_p_interfer

    progress = epoch / warmup_epochs
    return (
        progress * target_p_extreme,
        progress * target_p_very_low,
        progress * target_p_interfer,
    )


def clip_gan_scores(scores: list[mx.array], clip_value: float) -> list[mx.array]:
    """Clamp GAN discriminator logits to a bounded range for stability."""
    if clip_value <= 0:
        return scores
    return [mx.clip(score, -clip_value, clip_value) for score in scores]


def is_vad_train_reg_enabled(
    vad_train_prob: float,
    vad_train_every_steps: int,
    max_stage_vad_weight: float,
) -> bool:
    """Return whether sparse VAD train regularization should be enabled."""
    return (vad_train_prob > 0 or vad_train_every_steps > 0) and max_stage_vad_weight > 0


def _resolve_pipeline_stage_by_index(
    stage_index: int, pipeline_stage_defs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return stage metadata for a fixed stage index."""
    from df_mlx.training_cli import _resolve_pipeline_stage

    if not pipeline_stage_defs:
        return _resolve_pipeline_stage(0, pipeline_stage_defs)

    bounded_index = min(max(int(stage_index), 0), len(pipeline_stage_defs) - 1)
    stage = pipeline_stage_defs[bounded_index]
    return {
        "index": bounded_index,
        "name": str(stage.get("name", f"stage_{bounded_index}")),
        "start_epoch": int(stage.get("start_epoch", 0)),
        "awesome_loss_weight": stage.get("awesome_loss_weight"),
        "vad_loss_weight": stage.get("vad_loss_weight"),
        "vad_speech_loss_weight": stage.get("vad_speech_loss_weight"),
    }
