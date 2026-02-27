"""Tests for extracted dynamic-training helper functions."""

from __future__ import annotations

import mlx.core as mx

from df_mlx import train_dynamic as td
from df_mlx.training_helpers import (
    build_setup_panel_line,
    clip_gan_scores,
    curriculum_schedule,
    is_vad_train_reg_enabled,
)


def test_build_setup_panel_line_format() -> None:
    line = build_setup_panel_line(
        epochs=10,
        batch_size=4,
        learning_rate=1e-4,
        dynamic_loss="baseline",
        gan_enabled=True,
        vad_enabled=False,
        checkpoint_dir="checkpoints",
        use_fp16=True,
    )

    assert line.startswith("SETUP │")
    assert "epochs=10" in line
    assert "bs=4" in line
    assert "lr=1.0e-04" in line
    assert "gan=on" in line
    assert "vad=off" in line
    assert "fp16=on" in line


def test_curriculum_schedule_ramp_and_plateau() -> None:
    early = curriculum_schedule(
        epoch=2,
        total_epochs=10,
        warmup_epochs=4,
        target_p_extreme=0.4,
        target_p_very_low=0.2,
        target_p_interfer=0.1,
    )
    late = curriculum_schedule(
        epoch=5,
        total_epochs=10,
        warmup_epochs=4,
        target_p_extreme=0.4,
        target_p_very_low=0.2,
        target_p_interfer=0.1,
    )

    assert early == (0.2, 0.1, 0.05)
    assert late == (0.4, 0.2, 0.1)


def test_clip_gan_scores_clamps_and_noop() -> None:
    scores = [mx.array([-100.0, -0.5, 0.2, 50.0])]
    clipped = clip_gan_scores(scores, clip_value=10.0)
    unclipped = clip_gan_scores(scores, clip_value=0.0)

    mx.eval(clipped[0], unclipped[0])
    assert float(mx.min(clipped[0])) >= -10.0
    assert float(mx.max(clipped[0])) <= 10.0
    assert float(mx.min(unclipped[0])) == -100.0
    assert float(mx.max(unclipped[0])) == 50.0


def test_is_vad_train_reg_enabled_rules() -> None:
    assert is_vad_train_reg_enabled(vad_train_prob=0.1, vad_train_every_steps=0, max_stage_vad_weight=1.0)
    assert is_vad_train_reg_enabled(vad_train_prob=0.0, vad_train_every_steps=2, max_stage_vad_weight=0.5)
    assert not is_vad_train_reg_enabled(vad_train_prob=0.0, vad_train_every_steps=0, max_stage_vad_weight=1.0)
    assert not is_vad_train_reg_enabled(vad_train_prob=0.2, vad_train_every_steps=0, max_stage_vad_weight=0.0)


def test_train_dynamic_wrapper_equivalence() -> None:
    scores = [mx.array([-5.0, 0.0, 7.0])]

    helper_line = build_setup_panel_line(
        epochs=1,
        batch_size=1,
        learning_rate=1e-4,
        dynamic_loss="baseline",
        gan_enabled=False,
        vad_enabled=False,
        checkpoint_dir="ckpt",
        use_fp16=False,
    )
    wrapper_line = td._build_setup_panel_line(
        epochs=1,
        batch_size=1,
        learning_rate=1e-4,
        dynamic_loss="baseline",
        gan_enabled=False,
        vad_enabled=False,
        checkpoint_dir="ckpt",
        use_fp16=False,
    )

    helper_sched = curriculum_schedule(1, 5, 2, 0.2, 0.1, 0.05)
    wrapper_sched = td.curriculum_schedule(1, 5, 2, 0.2, 0.1, 0.05)

    helper_clip = clip_gan_scores(scores, clip_value=1.0)
    wrapper_clip = td._clip_gan_scores(scores, clip_value=1.0)
    mx.eval(helper_clip[0], wrapper_clip[0])

    assert helper_line == wrapper_line
    assert helper_sched == wrapper_sched
    assert bool(mx.all(helper_clip[0] == wrapper_clip[0]))
    assert is_vad_train_reg_enabled(0.1, 0, 1.0) == td._is_vad_train_reg_enabled(0.1, 0, 1.0)
