from df_mlx.config import get_default_config
from df_mlx.run_config import RunConfig
from df_mlx.train_dynamic_config import apply_train_ini_config


def test_apply_train_ini_config_gan(tmp_path):
    ini_path = tmp_path / "train.ini"
    ini_path.write_text("""
[train]
GAN_ENABLED = true
GAN_START_EPOCH = 3
GAN_RAMP_EPOCHS = 2
DISCRIMINATOR_TYPE = mpd
MPD_PERIODS = 2, 3, 5
MSD_SCALES = 2
DISCRIMINATOR_UPDATE_FREQ = 4
DISCRIMINATOR_LR = 1e-4
DISCRIMINATOR_WEIGHT_DECAY = 0.01
DISCRIMINATOR_GRAD_CLIP = 0.5

[GANLoss]
factor = 0.2
type = hinge

[FeatureMatchingLoss]
factor = 1.5
""".strip())

    run_cfg = RunConfig()
    model_cfg = get_default_config()

    apply_train_ini_config(str(ini_path), run_cfg, model_cfg)

    assert run_cfg.gan.enabled is True
    assert run_cfg.gan.start_epoch == 3
    assert run_cfg.gan.ramp_epochs == 2
    assert run_cfg.gan.discriminator == "mpd"
    assert run_cfg.gan.mpd_periods == [2, 3, 5]
    assert run_cfg.gan.msd_scales == 2
    assert run_cfg.gan.disc_update_freq == 4
    assert run_cfg.gan.disc_lr == 1e-4
    assert run_cfg.gan.disc_weight_decay == 0.01
    assert run_cfg.gan.disc_grad_clip == 0.5
    assert run_cfg.gan.adv_weight == 0.2
    assert run_cfg.gan.fm_weight == 1.5
