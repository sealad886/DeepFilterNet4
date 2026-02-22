import json
import sys
from pathlib import Path

import mlx.nn as nn
import mlx.optimizers as optim

# Ensure the df_mlx package is importable when running tests from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from df_mlx.training_checkpoints import (  # noqa: E402
    _TRAIN_MODE_COMPILED,
    _TRAIN_MODE_EAGER,
    maybe_skip_resume_batches,
    resolve_epoch_train_mode,
    resolve_resume_batch_count,
    save_checkpoint,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def __call__(self, x):
        return self.lin(x)


def test_resolve_resume_batch_count_legacy_checkpoint_conversion():
    state = {
        "kind": "step",
        "batch_idx": 6,
    }
    assert resolve_resume_batch_count(state) == 7


def test_resolve_resume_batch_count_v2_checkpoint():
    state = {
        "kind": "step",
        "counter_semantics_version": 2,
        "micro_batches_completed": 9,
        "batch_idx": 8,
    }
    assert resolve_resume_batch_count(state) == 9


def test_resolve_resume_batch_count_completed_checkpoint_returns_zero():
    state = {
        "kind": "epoch_end",
        "counter_semantics_version": 2,
        "micro_batches_completed": 12,
    }
    assert resolve_resume_batch_count(state) == 0


def test_maybe_skip_resume_batches_uses_processed_count():
    iterator, did_skip = maybe_skip_resume_batches(
        iter(range(10)),
        resume_from="checkpoint.safetensors",
        epoch=3,
        start_epoch=3,
        resume_batch_idx=4,
    )
    assert did_skip is True
    assert list(iterator) == [4, 5, 6, 7, 8, 9]


def test_save_checkpoint_persists_counter_semantics_metadata(tmp_path: Path):
    model = TinyModel()
    optimizer = optim.AdamW(learning_rate=0.001)
    ckpt_path = tmp_path / "step_000042.safetensors"

    ok = save_checkpoint(
        model,
        ckpt_path,
        epoch=2,
        batch_idx=8,
        global_step=42,
        loss=0.1,
        best_valid_loss=0.1,
        config={},
        optimizer=optimizer,
        last_completed_epoch=1,
        kind="step",
    )

    assert ok is True
    state = json.loads(ckpt_path.with_suffix(".state.json").read_text())
    assert state["counter_semantics_version"] == 2
    assert state["batch_idx"] == 8
    assert state["micro_batches_completed"] == 8
    assert state["global_step"] == 42
    assert state["optimizer_steps_completed"] == 42


def test_train_loop_bounds_iterator_to_progress_total():
    source = (Path(__file__).resolve().parents[1] / "df_mlx" / "train_dynamic.py").read_text()
    assert "train_total = max(epoch_target_micro_batches - resume_batches_for_epoch, 0)" in source
    assert "enumerate(islice(data_iterator, train_total))" in source


def test_resolve_epoch_train_mode_before_at_after_gan_start():
    mode = None
    snapshots: list[tuple[str, bool]] = []
    gan_start_epoch = 3

    for epoch in (2, 3, 4):
        gan_active = epoch >= gan_start_epoch
        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=True,
            gan_active=gan_active,
            previous_mode=mode,
        )
        snapshots.append((mode, use_compiled))

    assert snapshots == [
        (_TRAIN_MODE_COMPILED, True),
        (_TRAIN_MODE_EAGER, False),
        (_TRAIN_MODE_EAGER, False),
    ]


def test_resolve_epoch_train_mode_resume_after_switch_stays_eager():
    mode, use_compiled = resolve_epoch_train_mode(
        compiled_step_base_enabled=True,
        gan_enabled=True,
        gan_active=True,
        previous_mode=_TRAIN_MODE_EAGER,
    )
    assert mode == _TRAIN_MODE_EAGER
    assert use_compiled is False


def test_resolve_epoch_train_mode_respects_base_compiled_blockers():
    mode, use_compiled = resolve_epoch_train_mode(
        compiled_step_base_enabled=False,
        gan_enabled=False,
        gan_active=False,
        previous_mode=None,
    )
    assert mode == _TRAIN_MODE_EAGER
    assert use_compiled is False


def test_train_loop_logs_modes_and_guards_against_gan_compiled_mix():
    source = (Path(__file__).resolve().parents[1] / "df_mlx" / "train_dynamic.py").read_text()
    checkpoints_source = (Path(__file__).resolve().parents[1] / "df_mlx" / "training_checkpoints.py").read_text()
    assert '_TRAIN_MODE_COMPILED = "COMPILED"' in checkpoints_source
    assert '_TRAIN_MODE_EAGER = "EAGER"' in checkpoints_source
    assert "TRAIN_MODE={train_mode}" in source
    assert "GAN active epoch cannot run compiled step" in source


def test_compiled_base_eligibility_not_blocked_by_grad_accumulation():
    source = (Path(__file__).resolve().parents[1] / "df_mlx" / "train_dynamic.py").read_text()
    assert "base_compiled_step_enabled = not (debug_numerics or nan_skip_batch)" in source
    assert "grad_accumulation_steps > 1" in source
    assert "compiled forward/backward enabled; optimizer updates remain accumulated" in source


def test_compiled_grad_accumulation_uses_compiled_loss_and_grad_path():
    source = (Path(__file__).resolve().parents[1] / "df_mlx" / "train_dynamic.py").read_text()
    assert "def compiled_loss_and_grad_step(" in source
    assert "if grad_accumulation_steps > 1:" in source
    assert "loss, model_out, cached_out_wav, cached_clean_wav, grads = active_compiled_lag(" in source
