"""Behavior tests for fast sync-mode metric suppression."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

# Ensure the df_mlx package is importable when running tests from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import df_mlx.training_metrics as training_metrics  # noqa: E402
from df_mlx.training_setup import print_epoch_summary  # noqa: E402


def _unexpected_metric(*_args, **_kwargs):
    raise AssertionError("fast sync mode should skip component-metric recomputation")


def _base_collect_kwargs() -> dict:
    return dict(
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
        loss_val=1.25,
        loss_was_nonfinite=False,
        epoch_eval_frequency=4,
        use_mrstft_loss=False,
        use_vad_loss=True,
        use_awesome_loss=True,
        use_pipeline_awesome_loss=False,
        use_contrastive_awesome_loss=False,
        use_contrastive_silence_loss=True,
        use_vad_train_reg=True,
        use_fp16=False,
        gan_active=False,
        emit_detailed_metrics=False,
        apply_vad_reg=True,
        debug_numerics=False,
        speech_weight=0.5,
        spectral_loss_fn=_unexpected_metric,
        mrstft_loss_fn=None,
        mrstft_istft=None,
        mrstft_target_len=None,
        discriminator=None,
        feature_match_loss=None,
        gan_loss_fns=None,
        gan_istft=None,
        gan_fm_weight=0.0,
        gan_disc_max_samples=None,
        gan_target_len=0,
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
        contrastive_silence_frames_per_sample=32,
        contrastive_silence_mask_max=0.3,
        contrastive_silence_weight=0.8,
        contrastive_silence_asymmetric_penalty=2.5,
        contrastive_silence_transition_blend_low=0.3,
        contrastive_silence_transition_blend_high=0.7,
        contrastive_silence_low_freq_boost=1.5,
        contrastive_silence_high_freq_boost=1.3,
        debugger=None,
        debug_ctx={},
        accums=training_metrics.create_epoch_accums(),
    )


def test_collect_sync_metrics_fast_mode_skips_component_recomputes(monkeypatch) -> None:
    monkeypatch.setattr(training_metrics, "_compute_vad_loss", _unexpected_metric)
    monkeypatch.setattr(training_metrics, "_compute_awesome_losses", _unexpected_metric)
    monkeypatch.setattr(training_metrics, "_compute_contrastive_silence_losses", _unexpected_metric)
    monkeypatch.setattr(training_metrics, "_compute_vad_reg_loss", _unexpected_metric)

    kwargs = _base_collect_kwargs()
    accums = kwargs["accums"]

    display = training_metrics.collect_sync_metrics(**kwargs)

    assert display["spec_loss_val"] == 1.25
    assert display["vad_loss_val"] == 0.0
    assert display["awesome_loss_val"] == 0.0
    assert display["contrastive_loss_val"] == 0.0
    assert display["vad_reg_loss_val"] == 0.0
    assert accums["num_vad_logs"] == 0
    assert accums["num_awesome_logs"] == 0


def test_print_epoch_summary_fast_mode_hides_component_breakdown(capsys) -> None:
    epoch_avgs = {
        "loss": 1.0,
        "spec_loss": 0.9,
        "mrstft_loss": 0.8,
        "gan_g_loss": 0.7,
        "gan_fm_loss": 0.6,
        "gan_d_loss": 0.5,
        "vad_loss": 0.4,
        "speech_loss": 0.3,
        "awesome_loss": 0.2,
        "awesome_speech": 0.1,
        "awesome_noise": 0.05,
        "awesome_smooth": 0.04,
        "contrastive_loss": 0.03,
        "contrastive_speech": 0.02,
        "contrastive_quiet": 0.01,
        "contrastive_pos_sim": 0.9,
        "contrastive_neg_sim": 0.1,
        "music_supp": 0.08,
        "mask_sat": 0.07,
        "vad_reg_loss": 0.06,
        "p_ref": 0.5,
        "p_out": 0.4,
        "gate": 20.0,
        "mask_mean": 0.3,
        "mask_high": 10.0,
        "mask_low": 5.0,
        "proxy": 0.2,
        "speech_ratio": 0.1,
        "music_gate": 0.05,
        "musicness": 0.04,
        "mod": 0.03,
        "energy_boost": 0.02,
        "snr_boost": 0.01,
    }

    print_epoch_summary(
        epoch_avgs,
        epoch=0,
        epochs=1,
        avg_valid_loss=0.9,
        best_valid_loss=0.9,
        samples_processed=128,
        epoch_time=1.0,
        use_vad_loss=True,
        use_awesome_loss=True,
        use_pipeline_awesome_loss=False,
        use_contrastive_awesome_loss=False,
        use_contrastive_silence_loss=True,
        use_mrstft_loss=True,
        use_vad_train_reg=True,
        gan_enabled=True,
        gan_fm_weight=1.0,
        verbose=True,
        debug_numerics=True,
        emit_detailed_metrics=False,
        num_debug_logs=1,
        train_mask_clip_rate=10.0,
        train_eps_clean_rate=5.0,
        train_eps_noise_rate=4.0,
        train_mask_logit_min=-1.0,
        train_mask_logit_max=1.0,
        num_vad_logs=1,
        train_vad_clip_ref=2.0,
        train_vad_clip_out=3.0,
    )
    out = capsys.readouterr().out

    assert "Train: 1.0000" in out
    assert "Spec:" not in out
    assert "VAD stats:" not in out
    assert "Awesome stats:" not in out
    assert "Debug numerics:" not in out
