"""Shared helper utilities and mutable loop-state dataclass for training.

Small, self-contained helpers used across multiple training modules, plus the
``TrainingLoopState`` dataclass that tracks mutable iteration-level state
(counters, best-loss, stage tracking, mode flags) throughout the epoch loop.

Key exports:
    - SCALAR_ZERO: Cached ``mx.array(0.0)`` sentinel for loss placeholders.
    - TrainingLoopState: Mutable dataclass for loop-iteration bookkeeping.
    - build_setup_panel_line: Format a key/value pair for the config panel.
    - curriculum_schedule: Compute curriculum learning schedule values.
    - clip_gan_scores: Clamp discriminator logit lists to a safe range.
    - is_vad_train_reg_enabled: Check whether VAD regularisation is active.
    - _resolve_pipeline_stage_by_index: Look up a pipeline stage definition.
    - print_compiled_step_eligibility: Log which compiled-step variant is used.

Relationship to train_dynamic:
    TrainingLoopState is instantiated at the top of train() and updated every
    batch.  Helper functions are imported by train_dynamic, training_metrics,
    and training_setup.  Not included in the backward-compat re-export block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx

# Cached scalar zero — reused for default loss placeholders in validation and
# accumulated-loss resets.  Avoids repeated micro-allocations.  MLX arrays are
# value-immutable, so sharing a single instance is safe.
SCALAR_ZERO = mx.array(0.0)


@dataclass
class TrainingLoopState:
    """Mutable state that evolves during the training loop.

    Unlike TrainingSession (which holds session-level setup),
    this dataclass tracks variables that change during epoch iteration:
    counters, best-loss tracking, stage management, and mode flags.
    """

    # Counters
    global_step: int = 0
    final_epoch: int = 0
    last_completed_epoch: int = -1

    # Validation / early-stopping
    best_valid_loss: float = float("inf")
    epochs_without_improvement: int = 0
    avg_train_loss: float = float("nan")
    last_valid_loss: float | None = None
    last_valid_epoch: int | None = None

    # Pipeline stage
    active_stage_name: str = ""
    active_stage_index: int = 0

    # Per-epoch weights (from pipeline stage)
    epoch_awesome_loss_weight: float = 0.0
    epoch_vad_loss_weight: float = 0.0
    epoch_vad_speech_loss_weight: float = 0.0

    # Mode flags
    train_mode: Literal["COMPILED", "EAGER"] | None = None
    gan_active: bool = False
    compiled_gan_correctness_verified: bool = False


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


def clip_gan_scores(scores: list[mx.array], clip_value: float = 30.0) -> list[mx.array]:
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


def _resolve_pipeline_stage_by_index(stage_index: int, pipeline_stage_defs: list[dict[str, Any]]) -> dict[str, Any]:
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


def print_compiled_step_eligibility(
    *,
    debug_numerics: bool,
    nan_skip_batch: bool,
    gan_enabled: bool,
    gan_start_epoch: int,
    experimental_compiled_gan: bool,
    grad_accumulation_steps: int,
    batch_size: int,
) -> bool:
    """Determine compiled-step base eligibility and print status diagnostics.

    Returns ``True`` when compiled training steps should be used (base
    eligibility).  Epoch-level mode selection may still choose eager.
    """
    enabled = not (debug_numerics or nan_skip_batch)
    disable_reasons: list[str] = []
    if debug_numerics:
        disable_reasons.append("debug_numerics")
    if nan_skip_batch:
        disable_reasons.append("nan_skip_batch")

    print(f"  Compiled-step base eligibility: {enabled}")
    if enabled:
        if gan_enabled and gan_start_epoch <= 0 and not experimental_compiled_gan:
            print("  GAN starts at epoch 1: training will run eager from the first epoch")
        elif gan_enabled and gan_start_epoch <= 0 and experimental_compiled_gan:
            print("  [EXPERIMENTAL] GAN starts at epoch 1: compiled-GAN experiment keeps compiled mode")
        elif gan_enabled and not experimental_compiled_gan:
            print(
                "  GAN delayed start: training will use compiled mode until GAN activation "
                f"(gan_start_epoch={gan_start_epoch + 1})"
            )
        elif gan_enabled and experimental_compiled_gan:
            print(
                "  [EXPERIMENTAL] GAN delayed start: compiled-GAN experiment will keep compiled "
                f"mode through GAN activation (gan_start_epoch={gan_start_epoch + 1})"
            )
    else:
        joined = ", ".join(disable_reasons) if disable_reasons else "unknown"
        print(f"  Compiled-step disabled by: {joined}")
        if experimental_compiled_gan:
            print(
                "  [EXPERIMENTAL] WARNING: compiled-GAN experiment requested but compiled mode "
                f"is globally disabled ({joined}). Experiment will not activate."
            )
    if grad_accumulation_steps > 1:
        print(
            f"  Gradient accumulation: {grad_accumulation_steps} steps "
            f"(effective batch = {batch_size * grad_accumulation_steps})"
        )
        if enabled:
            print("  Gradient accumulation: compiled forward/backward enabled; optimizer updates remain accumulated")
        else:
            print("  Gradient accumulation: compiled training step disabled")
    if nan_skip_batch:
        print("  nan-skip-batch: enabled (will skip updates on non-finite loss/grads)")

    return enabled
