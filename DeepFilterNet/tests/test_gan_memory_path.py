"""Regression tests for GAN memory-sensitive paths in train_dynamic.py.

These tests now include executable coverage for the discriminator-input crop
path used during sync-window metric collection, so GAN memory behavior is
protected by behavior rather than source-text inspection alone.
"""

import sys
from pathlib import Path

import mlx.core as mx

# Ensure the df_mlx package is importable when running tests from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import df_mlx.training_metrics as training_metrics  # noqa: E402

_DF_MLX_DIR = Path(__file__).resolve().parents[1] / "df_mlx"
TRAIN_SOURCE = (_DF_MLX_DIR / "train_dynamic.py").read_text()
WAVEFORM_SOURCE = (_DF_MLX_DIR / "training_waveform.py").read_text()


class _RecordingDiscriminator:
    def __init__(self):
        self.calls: list[tuple[tuple[int, ...], bool | None]] = []

    def __call__(self, wav, return_features=None):
        self.calls.append((tuple(wav.shape), return_features))
        return [mx.zeros((wav.shape[0], 1), dtype=mx.float32)], []


def test_collect_sync_metrics_crops_gan_waveforms_before_discriminator(monkeypatch) -> None:
    discriminator = _RecordingDiscriminator()

    def _fake_specs_to_wavs(*args, **kwargs):
        wav = mx.zeros((2, 240000), dtype=mx.float32)
        return wav, wav

    monkeypatch.setattr(training_metrics, "specs_to_wavs", _fake_specs_to_wavs)

    display = training_metrics.collect_sync_metrics(
        noisy_real=mx.zeros((2, 8, 8), dtype=mx.float32),
        noisy_imag=mx.zeros((2, 8, 8), dtype=mx.float32),
        clean_real=mx.zeros((2, 8, 8), dtype=mx.float32),
        clean_imag=mx.zeros((2, 8, 8), dtype=mx.float32),
        interference_real=mx.zeros((2, 8, 8), dtype=mx.float32),
        interference_imag=mx.zeros((2, 8, 8), dtype=mx.float32),
        snr=mx.zeros((2,), dtype=mx.float32),
        model=None,
        feat_erb=mx.zeros((2, 1, 8, 8), dtype=mx.float32),
        feat_spec=mx.zeros((2, 1, 8, 8), dtype=mx.float32),
        pred_spec_for_logging=(mx.zeros((2, 8, 8), dtype=mx.float32), mx.zeros((2, 8, 8), dtype=mx.float32)),
        loss_val=0.0,
        loss_was_nonfinite=False,
        epoch_eval_frequency=1,
        use_mrstft_loss=False,
        use_vad_loss=False,
        use_awesome_loss=False,
        use_pipeline_awesome_loss=False,
        use_contrastive_awesome_loss=False,
        use_vad_train_reg=False,
        use_fp16=False,
        gan_active=True,
        emit_detailed_metrics=True,
        apply_vad_reg=False,
        debug_numerics=False,
        speech_weight=0.0,
        spectral_loss_fn=lambda *_args, **_kwargs: mx.array(0.0, dtype=mx.float32),
        mrstft_loss_fn=None,
        mrstft_istft=None,
        mrstft_target_len=None,
        discriminator=discriminator,
        feature_match_loss=None,
        gan_loss_fns=(lambda scores: mx.array(1.0, dtype=mx.float32), None),
        gan_istft=object(),
        gan_fm_weight=0.0,
        gan_disc_max_samples=48000,
        gan_target_len=240000,
        config_fft_size=512,
        config_hop_size=256,
        config_sample_rate=48000,
        vad_band_mask=mx.ones((8,), dtype=mx.float32),
        vad_band_bins=8.0,
        vad_threshold=0.6,
        vad_margin=0.05,
        vad_snr_gate_db=-10.0,
        vad_snr_gate_width=6.0,
        vad_z_threshold=0.0,
        vad_z_slope=1.0,
        awesome_mask_sharpness=6.0,
        vad_proxy_enabled=True,
        contrastive_temperature=0.1,
        contrastive_speech_frames_per_sample=32,
        contrastive_interference_frames_per_sample=32,
        contrastive_speech_mask_min=0.7,
        contrastive_interference_mask_max=0.3,
        contrastive_quiet_weight=0.5,
        contrastive_in_batch_negatives=True,
        debugger=None,
        debug_ctx={},
        accums=training_metrics.create_epoch_accums(),
    )

    assert display["gan_g_loss_val"] == 1.0
    assert [shape for shape, _ in discriminator.calls] == [(2, 48000), (2, 48000)]


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
