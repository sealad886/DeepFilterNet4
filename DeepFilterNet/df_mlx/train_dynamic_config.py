"""INI config adapter for MLX train_dynamic.

Maps train.py-style config.ini sections into train_dynamic RunConfig,
MLX model configuration, and dataset overrides.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from typing import Any, Iterable

from df_mlx.config import ModelParams4
from df_mlx.run_config import RunConfig


@dataclass
class TrainIniOverrides:
    dataset_overrides: dict[str, Any]
    warnings: list[str]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _warn_unused(section: str, keys: Iterable[str], used: set[str], warnings: list[str]) -> None:
    for key in keys:
        if key not in used:
            warnings.append(f"train-config: ignoring unsupported {section}.{key}")


def _get_section(parser: configparser.ConfigParser, name: str) -> tuple[configparser.SectionProxy | None, str | None]:
    for section in parser.sections():
        if section.lower() == name.lower():
            return parser[section], section
    return None, None


def _serialize_ini_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def apply_train_ini_config(
    path: str,
    run_cfg: RunConfig,
    model_cfg: ModelParams4,
) -> TrainIniOverrides:
    parser = configparser.ConfigParser()
    parser.optionxform = str.lower
    parser.read(path)
    return _apply_train_ini_parser(parser, run_cfg, model_cfg)


def apply_train_ini_tables(
    tables: dict[str, Any],
    run_cfg: RunConfig,
    model_cfg: ModelParams4,
) -> TrainIniOverrides:
    """Apply train.py INI-compatible tables embedded in run-config TOML.

    Expected shape:
        [train_ini.df]
        sr = 48000
        ...

        [train_ini.train]
        max_epochs = 100
        ...
    """
    parser = configparser.ConfigParser()
    parser.optionxform = str.lower
    for section, values in tables.items():
        if not isinstance(values, dict):
            continue
        parser[section] = {str(k): _serialize_ini_value(v) for k, v in values.items() if v is not None}
    return _apply_train_ini_parser(parser, run_cfg, model_cfg)


def _apply_train_ini_parser(
    parser: configparser.ConfigParser,
    run_cfg: RunConfig,
    model_cfg: ModelParams4,
) -> TrainIniOverrides:
    warnings: list[str] = []
    dataset_overrides: dict[str, Any] = {}

    # [df] section
    sec, sec_name = _get_section(parser, "df")
    if sec is not None:
        used: set[str] = set()
        if "sr" in sec:
            val = sec.getint("sr")
            dataset_overrides["sample_rate"] = val
            model_cfg.audio.sr = val
            used.add("sr")
        if "fft_size" in sec:
            val = sec.getint("fft_size")
            dataset_overrides["fft_size"] = val
            model_cfg.audio.fft_size = val
            used.add("fft_size")
        if "hop_size" in sec:
            val = sec.getint("hop_size")
            dataset_overrides["hop_size"] = val
            model_cfg.audio.hop_size = val
            used.add("hop_size")
        if "nb_erb" in sec:
            val = sec.getint("nb_erb")
            dataset_overrides["nb_erb"] = val
            model_cfg.erb.nb_erb = val
            used.add("nb_erb")
        if "nb_df" in sec:
            val = sec.getint("nb_df")
            dataset_overrides["nb_df"] = val
            model_cfg.df.nb_df = val
            used.add("nb_df")
        if "df_order" in sec:
            model_cfg.df.df_order = sec.getint("df_order")
            used.add("df_order")
        if "df_lookahead" in sec:
            model_cfg.df.df_lookahead = sec.getint("df_lookahead")
            used.add("df_lookahead")
        if "min_nb_erb_freqs" in sec:
            model_cfg.erb.min_erb_width = sec.getint("min_nb_erb_freqs")
            used.add("min_nb_erb_freqs")
        if "lsnr_min" in sec:
            model_cfg.lsnr.lsnr_min = sec.getfloat("lsnr_min")
            used.add("lsnr_min")
        if "lsnr_max" in sec:
            model_cfg.lsnr.lsnr_max = sec.getfloat("lsnr_max")
            used.add("lsnr_max")

        _warn_unused(sec_name or "df", sec.keys(), used, warnings)

    # [train] section
    sec, sec_name = _get_section(parser, "train")
    if sec is not None:
        used = set()
        if "max_epochs" in sec:
            run_cfg.training.epochs = sec.getint("max_epochs")
            used.add("max_epochs")
        if "batch_size" in sec:
            run_cfg.training.batch_size = sec.getint("batch_size")
            used.add("batch_size")
        if "num_workers" in sec:
            run_cfg.dataloader.num_workers = sec.getint("num_workers")
            used.add("num_workers")
        if "num_prefetch_batches" in sec:
            run_cfg.dataloader.prefetch_size = sec.getint("num_prefetch_batches")
            used.add("num_prefetch_batches")
        if "max_sample_len_s" in sec:
            dataset_overrides["segment_length"] = sec.getfloat("max_sample_len_s")
            used.add("max_sample_len_s")
        if "log_timings" in sec:
            run_cfg.debug.verbose = sec.getboolean("log_timings")
            used.add("log_timings")
        if "seed" in sec:
            seed_val = sec.getint("seed")
            run_cfg.training.seed = seed_val
            dataset_overrides["seed"] = seed_val
            used.add("seed")
        if "dataloader_snrs" in sec:
            snrs = _parse_csv_floats(sec.get("dataloader_snrs"))
            if snrs:
                dataset_overrides["snr_range"] = (min(snrs), max(snrs))
            used.add("dataloader_snrs")
        if "dataloader_gains" in sec:
            gains = _parse_csv_floats(sec.get("dataloader_gains"))
            if gains:
                gain_range = (min(gains), max(gains))
                dataset_overrides["speech_gain_range"] = gain_range
                dataset_overrides["noise_gain_range"] = gain_range
            used.add("dataloader_gains")
        if "gan_enabled" in sec:
            run_cfg.gan.enabled = sec.getboolean("gan_enabled")
            used.add("gan_enabled")
        if "gan_start_epoch" in sec:
            run_cfg.gan.start_epoch = sec.getint("gan_start_epoch")
            used.add("gan_start_epoch")
        if "gan_ramp_epochs" in sec:
            run_cfg.gan.ramp_epochs = sec.getint("gan_ramp_epochs")
            used.add("gan_ramp_epochs")
        if "discriminator_type" in sec:
            run_cfg.gan.discriminator = sec.get("discriminator_type").lower()
            used.add("discriminator_type")
        if "mpd_periods" in sec:
            run_cfg.gan.mpd_periods = _parse_csv_ints(sec.get("mpd_periods"))
            used.add("mpd_periods")
        if "msd_scales" in sec:
            run_cfg.gan.msd_scales = sec.getint("msd_scales")
            used.add("msd_scales")
        if "discriminator_update_freq" in sec:
            run_cfg.gan.disc_update_freq = sec.getint("discriminator_update_freq")
            used.add("discriminator_update_freq")
        if "discriminator_lr" in sec:
            run_cfg.gan.disc_lr = sec.getfloat("discriminator_lr")
            used.add("discriminator_lr")
        if "discriminator_weight_decay" in sec:
            run_cfg.gan.disc_weight_decay = sec.getfloat("discriminator_weight_decay")
            used.add("discriminator_weight_decay")
        if "discriminator_grad_clip" in sec:
            run_cfg.gan.disc_grad_clip = sec.getfloat("discriminator_grad_clip")
            used.add("discriminator_grad_clip")

        _warn_unused(sec_name or "train", sec.keys(), used, warnings)

    # [optim] section
    sec, sec_name = _get_section(parser, "optim")
    if sec is not None:
        used = set()
        if "lr" in sec:
            run_cfg.training.learning_rate = sec.getfloat("lr")
            used.add("lr")
        if "lr_min" in sec:
            run_cfg.training.learning_rate_min = sec.getfloat("lr_min")
            used.add("lr_min")
        if "warmup_epochs" in sec:
            run_cfg.training.warmup_epochs = sec.getint("warmup_epochs")
            used.add("warmup_epochs")
        if "weight_decay" in sec:
            run_cfg.training.weight_decay = sec.getfloat("weight_decay")
            used.add("weight_decay")

        _warn_unused(sec_name or "optim", sec.keys(), used, warnings)

    # [distortion] section
    sec, sec_name = _get_section(parser, "distortion")
    if sec is not None:
        used = set()
        if "p_reverb" in sec:
            val = sec.getfloat("p_reverb")
            run_cfg.augmentation.p_reverb = val
            dataset_overrides["p_reverb"] = val
            used.add("p_reverb")
        if "p_clipping" in sec:
            val = sec.getfloat("p_clipping")
            run_cfg.augmentation.p_clipping = val
            dataset_overrides["p_clipping"] = val
            used.add("p_clipping")
        if "p_bandwidth_ext" in sec:
            dataset_overrides["p_bandwidth_ext"] = sec.getfloat("p_bandwidth_ext")
            used.add("p_bandwidth_ext")
        if "p_interfer_sp" in sec:
            val = sec.getfloat("p_interfer_sp")
            run_cfg.dataset.p_interfer_speech = val
            dataset_overrides["p_interfer_speech"] = val
            used.add("p_interfer_sp")
        if "p_interfer_speech" in sec:
            val = sec.getfloat("p_interfer_speech")
            run_cfg.dataset.p_interfer_speech = val
            dataset_overrides["p_interfer_speech"] = val
            used.add("p_interfer_speech")

        _warn_unused(sec_name or "distortion", sec.keys(), used, warnings)

    # [deepfilternet4] section
    sec, sec_name = _get_section(parser, "deepfilternet4")
    if sec is not None:
        used = set()
        if "backbone" in sec:
            backbone = sec.get("backbone").lower()
            if backbone in {"mamba", "gru", "attention"}:
                run_cfg.model.backbone_type = backbone
                model_cfg.backbone.backbone_type = backbone  # type: ignore[assignment]
            else:
                warnings.append(f"train-config: unsupported deepfilternet4.backbone={backbone}")
            used.add("backbone")
        if "model_variant" in sec:
            variant = sec.get("model_variant").lower()
            if variant in {"full", "lite"}:
                run_cfg.model.variant = variant
            else:
                warnings.append(f"train-config: unsupported deepfilternet4.model_variant={variant}")
            used.add("model_variant")
        if "conv_ch" in sec:
            model_cfg.encoder.conv_channels = sec.getint("conv_ch")
            used.add("conv_ch")
        if "conv_kernel" in sec:
            model_cfg.encoder.conv_kernel = _parse_csv_ints(sec.get("conv_kernel"))
            used.add("conv_kernel")
        if "conv_stride" in sec:
            model_cfg.encoder.conv_stride = _parse_csv_ints(sec.get("conv_stride"))
            used.add("conv_stride")
        if "emb_hidden_dim" in sec:
            model_cfg.encoder.emb_hidden_dim = sec.getint("emb_hidden_dim")
            used.add("emb_hidden_dim")
        if "emb_num_layers" in sec:
            model_cfg.encoder.num_enc_layers = sec.getint("emb_num_layers")
            used.add("emb_num_layers")
        if "df_hidden_dim" in sec:
            model_cfg.df.nb_df_hidden = sec.getint("df_hidden_dim")
            used.add("df_hidden_dim")
        if "df_num_layers" in sec:
            model_cfg.df.df_n_layers = sec.getint("df_num_layers")
            used.add("df_num_layers")
        if "df_order" in sec:
            model_cfg.df.df_order = sec.getint("df_order")
            used.add("df_order")
        if "df_lookahead" in sec:
            model_cfg.df.df_lookahead = sec.getint("df_lookahead")
            used.add("df_lookahead")
        if "mask_pf" in sec:
            model_cfg.df.mask_pf = sec.getboolean("mask_pf")
            used.add("mask_pf")
        if "pf_beta" in sec:
            model_cfg.df.pf_beta = sec.getfloat("pf_beta")
            used.add("pf_beta")
        if "lsnr_dropout" in sec:
            model_cfg.lsnr.lsnr_dropout = sec.getboolean("lsnr_dropout")
            used.add("lsnr_dropout")
        if "lsnr_dropout_threshold" in sec:
            model_cfg.lsnr.lsnr_dropout_threshold = sec.getfloat("lsnr_dropout_threshold")
            used.add("lsnr_dropout_threshold")
        if "mamba_d_state" in sec:
            model_cfg.backbone.d_state = sec.getint("mamba_d_state")
            used.add("mamba_d_state")
        if "mamba_d_conv" in sec:
            model_cfg.backbone.d_conv = sec.getint("mamba_d_conv")
            used.add("mamba_d_conv")
        if "mamba_expand" in sec:
            model_cfg.backbone.expand_factor = sec.getint("mamba_expand")
            used.add("mamba_expand")

        _warn_unused(sec_name or "deepfilternet4", sec.keys(), used, warnings)

    # [loss] section (legacy)
    sec, sec_name = _get_section(parser, "loss")
    if sec is not None:
        used = set()
        if "multi_res_stft_f" in sec:
            run_cfg.loss.mrstft.factor = sec.getfloat("multi_res_stft_f")
            used.add("multi_res_stft_f")
        if "multi_res_stft_gamma" in sec:
            run_cfg.loss.mrstft.gamma = sec.getfloat("multi_res_stft_gamma")
            used.add("multi_res_stft_gamma")

        _warn_unused(sec_name or "loss", sec.keys(), used, warnings)

    # [MultiResSpecLoss] section (train.py style)
    sec, sec_name = _get_section(parser, "MultiResSpecLoss")
    if sec is not None:
        used = set()
        if "factor" in sec:
            run_cfg.loss.mrstft.factor = sec.getfloat("factor")
            used.add("factor")
        if "gamma" in sec:
            run_cfg.loss.mrstft.gamma = sec.getfloat("gamma")
            used.add("gamma")
        if "factor_complex" in sec:
            run_cfg.loss.mrstft.f_complex = sec.getfloat("factor_complex")
            used.add("factor_complex")
        if "fft_sizes" in sec:
            run_cfg.loss.mrstft.fft_sizes = _parse_csv_ints(sec.get("fft_sizes"))
            used.add("fft_sizes")
        if "hop_sizes" in sec:
            run_cfg.loss.mrstft.hop_sizes = _parse_csv_ints(sec.get("hop_sizes"))
            used.add("hop_sizes")

        _warn_unused(sec_name or "MultiResSpecLoss", sec.keys(), used, warnings)

    # [GANLoss] section (train.py style)
    sec, sec_name = _get_section(parser, "GANLoss")
    if sec is not None:
        used = set()
        if "factor" in sec:
            run_cfg.gan.adv_weight = sec.getfloat("factor")
            used.add("factor")
        if "type" in sec:
            # MLX GAN uses hinge loss; keep config but warn if mismatched.
            loss_type = sec.get("type").lower()
            if loss_type != "hinge":
                warnings.append(f"train-config: GANLoss.type={loss_type} not supported; using hinge loss.")
            used.add("type")

        _warn_unused(sec_name or "GANLoss", sec.keys(), used, warnings)

    # [FeatureMatchingLoss] section (train.py style)
    sec, sec_name = _get_section(parser, "FeatureMatchingLoss")
    if sec is not None:
        used = set()
        if "factor" in sec:
            run_cfg.gan.fm_weight = sec.getfloat("factor")
            used.add("factor")

        _warn_unused(sec_name or "FeatureMatchingLoss", sec.keys(), used, warnings)

    # Explicitly warn for unsupported advanced losses
    for section in ("asrloss", "speakerloss", "maskloss", "spectralloss"):
        sec, sec_name = _get_section(parser, section)
        if sec is not None:
            warnings.append(
                f"train-config: section [{sec_name}] is not supported in train_dynamic; "
                "use PyTorch train.py where applicable."
            )

    if not run_cfg.gan.enabled and (run_cfg.gan.adv_weight > 0 or run_cfg.gan.fm_weight > 0):
        warnings.append(
            "train-config: GANLoss/FeatureMatchingLoss enabled while GAN_ENABLED=false; " "enabling GAN training."
        )
        run_cfg.gan.enabled = True

    return TrainIniOverrides(dataset_overrides=dataset_overrides, warnings=warnings)
