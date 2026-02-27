"""Regression tests for GAN memory-sensitive paths in train_dynamic.py.

These tests are source-level invariants that protect against accidental
reintroduction of known high-memory patterns at GAN activation.
"""

from pathlib import Path

_DF_MLX_DIR = Path(__file__).resolve().parents[1] / "df_mlx"
TRAIN_SOURCE = (_DF_MLX_DIR / "train_dynamic.py").read_text()
WAVEFORM_SOURCE = (_DF_MLX_DIR / "training_waveform.py").read_text()


def test_gan_waveform_view_helper_exists() -> None:
    assert "def _gan_waveform_view(" in WAVEFORM_SOURCE


def test_gan_uses_mrstft_precision_policy_for_istft() -> None:
    # GAN waveform conversion should only force FP32 when MRSTFT needs it.
    assert TRAIN_SOURCE.count("force_fp32=use_mrstft_loss") >= 3


def test_discriminator_update_follows_optimizer_step_cadence() -> None:
    assert "do_disc_update = did_optimizer_update" in TRAIN_SOURCE
    assert "(loop_state.global_step % gan_disc_update_freq) == 0" in TRAIN_SOURCE


def test_generator_fm_real_branch_is_stop_grad() -> None:
    assert "mx.stop_gradient(gan_clean_wav)" in TRAIN_SOURCE
    assert "return_features=_need_feats" in TRAIN_SOURCE


def test_gan_paths_use_precision_view_before_discriminator() -> None:
    assert "_gan_waveform_view(pred_wav, use_fp16=bool(use_fp16))" in TRAIN_SOURCE
    assert "_gan_waveform_view(clean_wav, use_fp16=bool(use_fp16))" in TRAIN_SOURCE
    assert "out_wav = _gan_waveform_view(out_wav, use_fp16=bool(use_fp16))" in TRAIN_SOURCE
