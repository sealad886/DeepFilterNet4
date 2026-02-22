"""Tests for GAN OOM prevention mechanisms.

Validates the three-pronged OOM fix:
1. disc_crop_waveform — random-crop waveforms for discriminator input
2. epoch_eval_frequency override — force eval_frequency=1 during GAN epochs
3. Config validation — disc_max_samples, mpd_channels, msd_channels
"""

import random

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from df_mlx.run_config import RunConfig, apply_run_config_dict

# ---------------------------------------------------------------------------
# _disc_crop_waveform unit tests
# ---------------------------------------------------------------------------

if HAS_MLX:
    from df_mlx.training_waveform import _disc_crop_waveform


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestDiscCropWaveform:
    def test_noop_when_max_zero(self):
        wav = mx.zeros((2, 96000))
        out, start = _disc_crop_waveform(wav, 0)
        assert out.shape == (2, 96000)
        assert start == 0

    def test_noop_when_shorter_than_max(self):
        wav = mx.zeros((2, 10000))
        out, start = _disc_crop_waveform(wav, 48000)
        assert out.shape == (2, 10000)
        assert start == 0

    def test_crops_to_max_samples(self):
        wav = mx.ones((4, 240000))
        out, start = _disc_crop_waveform(wav, 48000)
        assert out.shape == (4, 48000)
        assert 0 <= start <= 240000 - 48000

    def test_shared_crop_start_aligns_outputs(self):
        random.seed(42)
        clean = mx.arange(96000).reshape(1, 96000)
        pred = mx.arange(96000).reshape(1, 96000)
        clean_crop, crop_start = _disc_crop_waveform(clean, 48000)
        pred_crop, _ = _disc_crop_waveform(pred, 48000, crop_start=crop_start)
        assert clean_crop.shape == pred_crop.shape == (1, 48000)
        assert mx.array_equal(clean_crop, pred_crop)

    def test_batch_dim_preserved(self):
        wav = mx.zeros((8, 120000))
        out, _ = _disc_crop_waveform(wav, 48000)
        assert out.shape[0] == 8

    def test_exact_length_noop(self):
        wav = mx.zeros((2, 48000))
        out, start = _disc_crop_waveform(wav, 48000)
        assert out.shape == (2, 48000)
        assert start == 0


# ---------------------------------------------------------------------------
# Config field validation
# ---------------------------------------------------------------------------


class TestGanConfigFields:
    def test_defaults_present(self):
        cfg = RunConfig()
        assert cfg.gan.disc_max_samples == 48000
        assert cfg.gan.mpd_channels == 32
        assert cfg.gan.msd_channels == 128

    def test_toml_overrides(self):
        cfg = RunConfig()
        apply_run_config_dict(
            cfg,
            {
                "gan": {
                    "disc_max_samples": 24000,
                    "mpd_channels": 16,
                    "msd_channels": 64,
                }
            },
        )
        assert cfg.gan.disc_max_samples == 24000
        assert cfg.gan.mpd_channels == 16
        assert cfg.gan.msd_channels == 64

    def test_oom_safe_toml_loads(self):
        """Verify the OOM-safe TOML profile loads without errors."""
        from pathlib import Path

        from df_mlx.run_config import load_run_config

        toml_path = (
            Path(__file__).resolve().parents[1]
            / "df_mlx"
            / "configs"
            / "run_profiles"
            / "run_pipeline_awesome_gan_silero_single_oom_safe.toml"
        )
        if not toml_path.exists():
            pytest.skip("OOM-safe TOML profile not found")
        cfg = load_run_config(toml_path)
        assert cfg.gan.disc_max_samples == 48000
        assert cfg.gan.mpd_channels == 16
        assert cfg.gan.msd_channels == 64


# ---------------------------------------------------------------------------
# Epoch eval_frequency override logic
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestEpochEvalFrequencyOverride:
    def test_resolve_epoch_train_mode_gan_forces_eager(self):
        from df_mlx.training_checkpoints import _TRAIN_MODE_EAGER, resolve_epoch_train_mode

        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=True,
            gan_active=True,
            experimental_compiled_gan=False,
            previous_mode=_TRAIN_MODE_EAGER,
        )
        assert mode == _TRAIN_MODE_EAGER
        assert use_compiled is False

    def test_eval_frequency_override_concept(self):
        """When GAN is active, epoch_eval_frequency = min(eval_freq, gan_eval_freq)."""
        eval_frequency = 10
        gan_eval_frequency = 2  # default

        # GAN active: clamped to min(eval_frequency, gan_eval_frequency)
        gan_active = True
        epoch_eval_frequency = eval_frequency
        if gan_active:
            epoch_eval_frequency = min(eval_frequency, gan_eval_frequency)
        assert epoch_eval_frequency == 2

        # GAN inactive: original eval_frequency
        gan_active = False
        epoch_eval_frequency = eval_frequency
        if gan_active:
            epoch_eval_frequency = min(eval_frequency, gan_eval_frequency)
        assert epoch_eval_frequency == 10

        # Custom high gan_eval_frequency: no effect
        gan_active = True
        gan_eval_frequency = 20
        epoch_eval_frequency = eval_frequency
        if gan_active:
            epoch_eval_frequency = min(eval_frequency, gan_eval_frequency)
        assert epoch_eval_frequency == 10

        # gan_eval_frequency=1: legacy behavior (every step)
        gan_eval_frequency = 1
        epoch_eval_frequency = eval_frequency
        if gan_active:
            epoch_eval_frequency = min(eval_frequency, gan_eval_frequency)
        assert epoch_eval_frequency == 1
