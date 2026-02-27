"""CLI entry-point for DeepFilterNet4 dynamic training.

Houses the ``main()`` function that builds an ``argparse`` parser with the
full set of training CLI flags, assembles a validated
:class:`~df_mlx.run_config.RunConfig`, and delegates to
:func:`~df_mlx.train_dynamic.train`.  This is the module invoked by
``python -m df_mlx.train_dynamic``.

Key exports:
    - main: CLI entry-point — parse arguments, build RunConfig, call train().

Relationship to train_dynamic:
    Imported by train_dynamic.py and included in the backward-compat re-export
    block.  ``main()`` is the outermost layer that feeds configuration into
    train().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal, cast

from df_mlx.run_config import (
    RunConfig,
    generate_run_config_example,
    load_preset_config,
    load_run_config,
    validate_run_config,
)
from df_mlx.train_dynamic_config import apply_train_ini_config, apply_train_ini_tables
from df_mlx.training_checkpoints import find_latest_checkpoint
from df_mlx.training_cli import _apply_cli_overrides


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train DfNet4 with dynamic on-the-fly mixing. "
            "--config refers to the dataset/mixer JSON config, "
            "--train-config is the train.py-style INI config, "
            "and --run-config refers to CLI/runtime settings (TOML)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data sources (priority: cache_dir > config > file lists)
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Path to pre-built audio cache (from build_audio_cache.py)",
    )
    parser.add_argument(
        "--cache-hf",
        type=str,
        help="HuggingFace dataset repo ID to stream from (e.g., 'my-org/mlx_datastore')",
    )
    parser.add_argument(
        "--speech-list",
        type=str,
        help="Path to file containing speech file paths (one per line)",
    )
    parser.add_argument(
        "--noise-list",
        type=str,
        help="Path to file containing noise file paths (one per line)",
    )
    parser.add_argument(
        "--rir-list",
        type=str,
        help="Path to file containing RIR file paths (one per line)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to dataset/mixer JSON config file (alternative to file lists)",
    )
    parser.add_argument(
        "--run-config",
        type=str,
        help="Path to run-config TOML file (CLI/runtime settings)",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        help="Path to train.py-compatible INI config (model + training settings)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["entry", "pro", "max", "ultra", "debug"],
        default=None,
        help=(
            "Load a named hardware preset as the base config. "
            "Values from --run-config and explicit CLI flags override preset defaults. "
            "See docs/RUN_CONFIG_PRESETS.md for details."
        ),
    )
    parser.add_argument(
        "--print-run-config",
        action="store_true",
        help="Print a commented run-config TOML example and exit",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--learning-rate-min",
        type=float,
        default=None,
        help="Minimum learning rate for cosine schedule (defaults to 1%% of base)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume from checkpoint. If no path given, auto-finds latest in checkpoint-dir",
    )
    parser.add_argument(
        "--resume-data",
        nargs="?",
        const=True,
        default=False,
        help="Resume data loading state. If no path given, uses data_checkpoint.json in checkpoint-dir",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=1,
        help="Validate every N epochs",
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="epoch",
        choices=["no", "epoch", "steps"],
        help=(
            "Checkpoint save strategy for additional checkpoints: "
            "'no' (only best + required epoch_end), "
            "'epoch' (every epoch), "
            "'steps' (every N steps)"
        ),
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (only when --save-strategy=steps)",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep (oldest removed first, best model always kept)",
    )
    parser.add_argument(
        "--checkpoint-batches",
        type=int,
        default=0,
        help="Save data checkpoint every N batches (0=disabled, for resume)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--p-reverb",
        type=float,
        default=0.5,
        help="Probability of applying reverb",
    )
    parser.add_argument(
        "--p-clipping",
        type=float,
        default=0.0,
        help="Probability of clipping distortion",
    )

    # Other parameters
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--prefetch-size",
        type=int,
        default=8,
        help="Number of batches to prefetch (for MLXDataStream)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--no-mlx-data",
        action="store_true",
        help="Disable mlx-data (use PrefetchDataLoader instead)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=None,
        help="Enable mixed-precision training (BF16) for faster performance",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed-precision training (use FP32 for full precision)",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (effective batch = batch_size * grad_accumulation_steps)",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=10,
        help="Sync with GPU every N batches (higher = faster but less responsive logging)",
    )
    parser.add_argument(
        "--backbone",
        "--backbone-type",
        dest="backbone_type",
        type=str,
        choices=["mamba", "gru", "attention"],
        default="mamba",
        help="Backbone type: 'mamba' (parallel scan SSM), 'gru' (recurrent), or 'attention' (fastest backward)",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=["full", "lite"],
        default="full",
        help="Model variant: 'full' or 'lite'",
    )
    parser.add_argument(
        "--snr-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Override base SNR range in dB (e.g., --snr-range -5 40)",
    )
    parser.add_argument(
        "--snr-range-extreme",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Override extreme SNR range in dB (e.g., --snr-range-extreme -20 -5)",
    )
    parser.add_argument(
        "--snr-range-very-low",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Override very-low SNR range in dB (e.g., --snr-range-very-low -30 -20)",
    )
    parser.add_argument(
        "--p-extreme-snr",
        type=float,
        help="Probability of sampling from extreme SNR range (0-1)",
    )
    parser.add_argument(
        "--p-very-low-snr",
        type=float,
        help="Probability of sampling from very-low SNR range (0-1)",
    )
    parser.add_argument(
        "--p-interfer-speech",
        type=float,
        help="Probability of adding interfering speaker (0-1, simulates vocals/competing talker)",
    )
    parser.add_argument(
        "--curriculum-warmup-epochs",
        type=int,
        default=0,
        help="Number of warmup epochs for curriculum learning (0=disabled). "
        "SNR/interferer probabilities ramp linearly from 0 to target values.",
    )
    parser.add_argument(
        "--speech-gain-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Override speech gain range in dB (e.g., --speech-gain-range -12 12)",
    )
    parser.add_argument(
        "--noise-gain-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Override noise gain range in dB (e.g., --noise-gain-range -12 12)",
    )
    parser.add_argument(
        "--dynamic-loss",
        type=str,
        choices=["baseline", "awesome", "pipeline_awesome"],
        default="baseline",
        help="Dynamic loss: 'baseline' (spectral + legacy VAD), 'awesome' (speech-preserving contrastive), or 'pipeline_awesome' (improved speech preservation + music suppression)",
    )
    parser.add_argument(
        "--pipeline-stages",
        type=str,
        default=None,
        help=(
            "JSON array of stage configs with start_epoch and optional overrides. "
            'Example: \'[{"start_epoch":0,"name":"bootstrap","awesome_loss_weight":0.2},'
            '{"start_epoch":5,"name":"refine","awesome_loss_weight":0.4}]\''
        ),
    )
    parser.add_argument(
        "--awesome-loss-weight",
        type=float,
        default=0.4,
        help="Weight for awesome speech-preserving contrastive loss",
    )
    parser.add_argument(
        "--awesome-mask-sharpness",
        type=float,
        default=6.0,
        help="Sharpness for speech/noise dominance mask in awesome loss",
    )
    parser.add_argument(
        "--awesome-warmup-steps",
        type=int,
        default=0,
        help="Warmup steps for ramping awesome loss weight",
    )
    parser.add_argument(
        "--mrstft-factor",
        type=float,
        default=None,
        help="Multi-res STFT loss weight (0 disables)",
    )
    parser.add_argument(
        "--mrstft-gamma",
        type=float,
        default=None,
        help="Multi-res STFT magnitude compression exponent",
    )
    parser.add_argument(
        "--mrstft-f-complex",
        type=float,
        default=None,
        help="Multi-res STFT complex loss weight (None disables)",
    )
    parser.add_argument(
        "--mrstft-fft-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Multi-res STFT FFT sizes (e.g., --mrstft-fft-sizes 512 1024 2048)",
    )
    parser.add_argument(
        "--mrstft-hop-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Multi-res STFT hop sizes (defaults to fft_size//4)",
    )
    parser.add_argument(
        "--gan-enabled",
        action="store_true",
        help="Enable GAN adversarial training",
    )
    parser.add_argument(
        "--gan-start-epoch",
        type=int,
        default=0,
        help="Epoch to start GAN training (0-based)",
    )
    parser.add_argument(
        "--gan-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp GAN weights over N epochs (0 disables ramp)",
    )
    parser.add_argument(
        "--gan-adv-weight",
        type=float,
        default=0.0,
        help="GAN adversarial loss weight",
    )
    parser.add_argument(
        "--gan-fm-weight",
        type=float,
        default=0.0,
        help="GAN feature matching loss weight",
    )
    parser.add_argument(
        "--gan-discriminator",
        type=str,
        default="combined",
        choices=["combined", "mpd", "msd"],
        help="Discriminator type for GAN training",
    )
    parser.add_argument(
        "--gan-mpd-periods",
        type=int,
        nargs="+",
        default=None,
        help="MPD periods for GAN discriminator (e.g., --gan-mpd-periods 2 3 5 7 11)",
    )
    parser.add_argument(
        "--gan-msd-scales",
        type=int,
        default=3,
        help="MSD scales for GAN discriminator",
    )
    parser.add_argument(
        "--gan-disc-lr",
        type=float,
        default=1e-4,
        help="GAN discriminator learning rate",
    )
    parser.add_argument(
        "--gan-disc-weight-decay",
        type=float,
        default=0.0,
        help="GAN discriminator weight decay",
    )
    parser.add_argument(
        "--gan-disc-grad-clip",
        type=float,
        default=1.0,
        help="GAN discriminator gradient clipping",
    )
    parser.add_argument(
        "--gan-disc-update-freq",
        type=int,
        default=1,
        help="Update discriminator every N steps",
    )
    parser.add_argument(
        "--no-vad-proxy",
        action="store_true",
        help="Disable cheap VAD proxy gating in awesome loss",
    )
    parser.add_argument(
        "--vad-loss-weight",
        type=float,
        default=0.05,
        help="Weight for VAD speech-preservation loss (0 disables)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.6,
        help="VAD probability threshold for speech gating",
    )
    parser.add_argument(
        "--vad-margin",
        type=float,
        default=0.05,
        help="Margin for VAD consistency loss",
    )
    parser.add_argument(
        "--vad-speech-loss-weight",
        type=float,
        default=0.0,
        help="Weight for VAD-weighted speech-structure loss",
    )
    parser.add_argument(
        "--vad-warmup-epochs",
        type=int,
        default=5,
        help="Warmup epochs to ramp VAD loss from 0 to target weight",
    )
    parser.add_argument(
        "--vad-snr-gate",
        type=float,
        default=-10.0,
        help="SNR threshold (dB) for VAD gating",
    )
    parser.add_argument(
        "--vad-snr-gate-width",
        type=float,
        default=6.0,
        help="Softness of SNR gating in dB",
    )
    parser.add_argument(
        "--vad-band-low",
        type=float,
        default=300.0,
        help="Low cutoff for speech band in Hz",
    )
    parser.add_argument(
        "--vad-band-high",
        type=float,
        default=3400.0,
        help="High cutoff for speech band in Hz",
    )
    parser.add_argument(
        "--vad-z-threshold",
        type=float,
        default=0.0,
        help="Z-score threshold for VAD sigmoid",
    )
    parser.add_argument(
        "--vad-z-slope",
        type=float,
        default=1.0,
        help="Z-score slope for VAD sigmoid",
    )
    parser.add_argument(
        "--vad-eval-mode",
        type=str,
        choices=["auto", "proxy", "silero", "off"],
        default="auto",
        help="VAD eval mode for periodic metrics (auto enables proxy for awesome loss)",
    )
    parser.add_argument(
        "--vad-eval-every",
        type=int,
        default=1,
        help="Evaluate VAD metrics every N epochs",
    )
    parser.add_argument(
        "--vad-eval-batches",
        type=int,
        default=8,
        help="Number of validation batches used for VAD metrics",
    )
    parser.add_argument(
        "--vad-eval-max-seconds",
        type=float,
        default=0.0,
        help="Max seconds per clip for VAD eval (0 disables)",
    )
    parser.add_argument(
        "--vad-silero-model-path",
        type=str,
        default=None,
        help="Path to silero_vad.onnx (defaults to silero-vad package data)",
    )
    parser.add_argument(
        "--vad-silero-sample-rate",
        type=int,
        default=16000,
        help="Sample rate for Silero VAD evaluation (Hz)",
    )
    parser.add_argument(
        "--vad-train-prob",
        type=float,
        default=0.0,
        help="Probability of applying sparse VAD regularizer per batch (0 disables)",
    )
    parser.add_argument(
        "--vad-train-every-steps",
        type=int,
        default=0,
        help="Apply sparse VAD regularizer every N steps (0 disables)",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit number of training batches per epoch (for fast benchmarking)",
    )
    parser.add_argument(
        "--max-valid-batches",
        type=int,
        default=None,
        help="Limit number of validation batches (for fast benchmarking)",
    )
    parser.add_argument(
        "--eval-sisdr",
        action="store_true",
        help="Compute SI-SDR during validation (slower)",
    )
    parser.add_argument(
        "--check-chkpts",
        action="store_true",
        help="Validate checkpoints and metadata before starting/resuming",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed override (enables deterministic sampling)",
    )
    parser.add_argument(
        "--debug-numerics",
        action="store_true",
        help="Enable numeric debug mode (fail-fast finite checks, short run, deterministic)",
    )
    parser.add_argument(
        "--debug-numerics-no-fail-fast",
        action="store_true",
        help="Disable fail-fast behavior in debug-numerics mode",
    )
    parser.add_argument(
        "--debug-numerics-every",
        type=int,
        default=1,
        help="Check tensors every N steps in debug-numerics mode",
    )
    parser.add_argument(
        "--debug-numerics-dump-dir",
        type=str,
        default=None,
        help="Directory for numeric debug dumps (default: checkpoint_dir/debug_numerics)",
    )
    parser.add_argument(
        "--debug-numerics-dump-arrays",
        action="store_true",
        help="Save small tensor slices alongside numeric debug JSON dumps",
    )
    parser.add_argument(
        "--debug-numerics-max-dumps",
        type=int,
        default=5,
        help="Maximum number of non-finite dumps to write in debug mode",
    )
    parser.add_argument(
        "--nan-skip-batch",
        action="store_true",
        help="Skip optimizer update when loss/grads are non-finite (debug-friendly)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed timing diagnostics and hardware info",
    )

    # Keep parser defaults aligned with RunConfig so --help always reports
    # accurate effective defaults. CLI application still keys off explicit
    # argv presence in _apply_cli_overrides(), so these defaults are display-
    # oriented and do not change precedence semantics.
    default_cfg = RunConfig()
    parser.set_defaults(
        cache_dir=default_cfg.dataset.cache_dir,
        speech_list=default_cfg.dataset.speech_list,
        noise_list=default_cfg.dataset.noise_list,
        rir_list=default_cfg.dataset.rir_list,
        config=default_cfg.dataset.config,
        train_config=default_cfg.training.train_config,
        epochs=default_cfg.training.epochs,
        batch_size=default_cfg.training.batch_size,
        learning_rate=default_cfg.training.learning_rate,
        learning_rate_min=default_cfg.training.learning_rate_min,
        weight_decay=default_cfg.training.weight_decay,
        checkpoint_dir=default_cfg.checkpoint.checkpoint_dir,
        resume=default_cfg.checkpoint.resume,
        resume_data=default_cfg.checkpoint.resume_data,
        validate_every=default_cfg.checkpoint.validate_every,
        save_strategy=default_cfg.checkpoint.save_strategy,
        save_steps=default_cfg.checkpoint.save_steps,
        save_total_limit=default_cfg.checkpoint.save_total_limit,
        checkpoint_batches=default_cfg.checkpoint.checkpoint_batches,
        p_reverb=default_cfg.augmentation.p_reverb,
        p_clipping=default_cfg.augmentation.p_clipping,
        num_workers=default_cfg.dataloader.num_workers,
        prefetch_size=default_cfg.dataloader.prefetch_size,
        max_grad_norm=default_cfg.training.max_grad_norm,
        warmup_epochs=default_cfg.training.warmup_epochs,
        patience=default_cfg.training.patience,
        fp16=default_cfg.training.fp16,
        grad_accumulation_steps=default_cfg.training.grad_accumulation_steps,
        eval_frequency=default_cfg.training.eval_frequency,
        backbone_type=default_cfg.model.backbone_type,
        model_variant=default_cfg.model.variant,
        snr_range=default_cfg.dataset.snr_range,
        snr_range_extreme=default_cfg.dataset.snr_range_extreme,
        snr_range_very_low=default_cfg.dataset.snr_range_very_low,
        p_extreme_snr=default_cfg.dataset.p_extreme_snr,
        p_very_low_snr=default_cfg.dataset.p_very_low_snr,
        p_interfer_speech=default_cfg.dataset.p_interfer_speech,
        curriculum_warmup_epochs=default_cfg.training.curriculum_warmup_epochs,
        speech_gain_range=default_cfg.dataset.speech_gain_range,
        noise_gain_range=default_cfg.dataset.noise_gain_range,
        dynamic_loss=default_cfg.loss.dynamic_loss,
        pipeline_stages=list(default_cfg.loss.pipeline_stages),
        awesome_loss_weight=default_cfg.loss.awesome.loss_weight,
        awesome_mask_sharpness=default_cfg.loss.awesome.mask_sharpness,
        awesome_warmup_steps=default_cfg.loss.awesome.warmup_steps,
        mrstft_factor=default_cfg.loss.mrstft.factor,
        mrstft_gamma=default_cfg.loss.mrstft.gamma,
        mrstft_f_complex=default_cfg.loss.mrstft.f_complex,
        mrstft_fft_sizes=list(default_cfg.loss.mrstft.fft_sizes),
        mrstft_hop_sizes=default_cfg.loss.mrstft.hop_sizes,
        gan_enabled=default_cfg.gan.enabled,
        gan_start_epoch=default_cfg.gan.start_epoch,
        gan_ramp_epochs=default_cfg.gan.ramp_epochs,
        gan_adv_weight=default_cfg.gan.adv_weight,
        gan_fm_weight=default_cfg.gan.fm_weight,
        gan_discriminator=default_cfg.gan.discriminator,
        gan_mpd_periods=list(default_cfg.gan.mpd_periods),
        gan_msd_scales=default_cfg.gan.msd_scales,
        gan_disc_lr=default_cfg.gan.disc_lr,
        gan_disc_weight_decay=default_cfg.gan.disc_weight_decay,
        gan_disc_grad_clip=default_cfg.gan.disc_grad_clip,
        gan_disc_update_freq=default_cfg.gan.disc_update_freq,
        gan_cache_gen_waveforms=default_cfg.gan.cache_gen_waveforms,
        gan_disc_gradient_checkpoint=default_cfg.gan.disc_gradient_checkpoint,
        gan_gen_gradient_checkpoint=default_cfg.gan.gen_gradient_checkpoint,
        gan_eval_frequency=default_cfg.gan.eval_frequency,
        experimental_compiled_gan=default_cfg.gan.experimental_compile,
        vad_loss_weight=default_cfg.vad.loss_weight,
        vad_threshold=default_cfg.vad.threshold,
        vad_margin=default_cfg.vad.margin,
        vad_speech_loss_weight=default_cfg.vad.speech_loss_weight,
        vad_warmup_epochs=default_cfg.vad.warmup_epochs,
        vad_snr_gate=default_cfg.vad.snr_gate_db,
        vad_snr_gate_width=default_cfg.vad.snr_gate_width,
        vad_band_low=default_cfg.vad.band_low_hz,
        vad_band_high=default_cfg.vad.band_high_hz,
        vad_z_threshold=default_cfg.vad.z_threshold,
        vad_z_slope=default_cfg.vad.z_slope,
        vad_eval_mode=default_cfg.vad.eval.mode,
        vad_eval_every=default_cfg.vad.eval.every,
        vad_eval_batches=default_cfg.vad.eval.batches,
        vad_eval_max_seconds=default_cfg.vad.eval.max_seconds,
        vad_silero_model_path=default_cfg.vad.eval.silero_model_path,
        vad_silero_sample_rate=default_cfg.vad.eval.silero_sample_rate,
        vad_train_prob=default_cfg.vad.train.prob,
        vad_train_every_steps=default_cfg.vad.train.every_steps,
        max_train_batches=default_cfg.dataloader.max_train_batches,
        max_valid_batches=default_cfg.dataloader.max_valid_batches,
        eval_sisdr=default_cfg.metrics.eval_sisdr,
        check_chkpts=default_cfg.checkpoint.check_chkpts,
        seed=default_cfg.training.seed,
        verbose=default_cfg.debug.verbose,
        debug_numerics=default_cfg.debug.debug_numerics,
        debug_numerics_fail_fast=default_cfg.debug.debug_numerics_fail_fast,
        debug_numerics_every=default_cfg.debug.debug_numerics_every,
        debug_numerics_dump_dir=default_cfg.debug.debug_numerics_dump_dir,
        debug_numerics_dump_arrays=default_cfg.debug.debug_numerics_dump_arrays,
        debug_numerics_max_dumps=default_cfg.debug.debug_numerics_max_dumps,
        nan_skip_batch=default_cfg.debug.nan_skip_batch,
    )

    args = parser.parse_args()

    if args.print_run_config:
        print(generate_run_config_example(), end="")
        return

    run_cfg = RunConfig()
    if args.preset:
        run_cfg = load_preset_config(args.preset, base=run_cfg)
    if args.run_config:
        run_cfg = load_run_config(args.run_config, base=run_cfg)
    train_config_path = args.train_config or run_cfg.training.train_config
    from df_mlx.config import get_default_config

    model_cfg = get_default_config()
    dataset_overrides: dict[str, Any] = {}
    ini_warnings: list[str] = []
    if train_config_path:
        ini_overrides = apply_train_ini_config(train_config_path, run_cfg, model_cfg)
        dataset_overrides.update(ini_overrides.dataset_overrides)
        ini_warnings.extend(ini_overrides.warnings)
    # Enforce documented precedence: defaults < train-config < run-config < CLI.
    if args.run_config:
        run_cfg = load_run_config(args.run_config, base=run_cfg)
    # Single-file mode: apply INI-compatible sections in run-config.
    # Then re-apply run-config so explicit top-level TOML values win over train_ini.* compatibility tables.
    if run_cfg.train_ini:
        toml_ini_overrides = apply_train_ini_tables(run_cfg.train_ini, run_cfg, model_cfg)
        dataset_overrides.update(toml_ini_overrides.dataset_overrides)
        ini_warnings.extend(toml_ini_overrides.warnings)
        if args.run_config:
            run_cfg = load_run_config(args.run_config, base=run_cfg)
    _apply_cli_overrides(run_cfg, args, sys.argv[1:])
    validate_run_config(run_cfg)
    if ini_warnings:
        print("Train-config compatibility warnings:")
        for warning in ini_warnings:
            print(f"  - {warning}")
    # Ensure backbone override from CLI/run-config wins
    model_cfg.backbone.backbone_type = run_cfg.model.backbone_type  # type: ignore[assignment]

    def _resolve_resume(resume_setting: bool | str, checkpoint_dir: str, label: str) -> str | None:
        if not resume_setting:
            return None
        if isinstance(resume_setting, str):
            return resume_setting
        ckpt_dir = Path(checkpoint_dir)
        if label == "resume":
            latest = find_latest_checkpoint(ckpt_dir)
            if latest:
                resume_path = str(latest)
                print(f"Auto-resuming from: {resume_path}")
                return resume_path
            print(f"Warning: resume requested but no checkpoint found in {ckpt_dir}")
            return None
        data_ckpt = ckpt_dir / "data_checkpoint.json"
        if data_ckpt.exists():
            resume_path = str(data_ckpt)
            print(f"Auto-resuming data from: {resume_path}")
            return resume_path
        print(f"Warning: resume-data requested but {data_ckpt} not found")
        return None

    resume_from = _resolve_resume(run_cfg.checkpoint.resume, run_cfg.checkpoint.checkpoint_dir, "resume")
    resume_data_from = _resolve_resume(
        run_cfg.checkpoint.resume_data,
        run_cfg.checkpoint.checkpoint_dir,
        "resume_data",
    )

    # Deferred import: train() lives in train_dynamic which imports us for re-export.
    from df_mlx.train_dynamic import train

    train(
        cache_dir=run_cfg.dataset.cache_dir,
        speech_list=run_cfg.dataset.speech_list,
        noise_list=run_cfg.dataset.noise_list,
        rir_list=run_cfg.dataset.rir_list,
        config_path=run_cfg.dataset.config,
        epochs=run_cfg.training.epochs,
        batch_size=run_cfg.training.batch_size,
        learning_rate=run_cfg.training.learning_rate,
        learning_rate_min=run_cfg.training.learning_rate_min,
        weight_decay=run_cfg.training.weight_decay,
        checkpoint_dir=run_cfg.checkpoint.checkpoint_dir,
        resume_from=resume_from,
        resume_data_from=resume_data_from,
        validate_every=run_cfg.checkpoint.validate_every,
        save_strategy=cast(Literal["no", "epoch", "steps"], run_cfg.checkpoint.save_strategy),
        save_steps=run_cfg.checkpoint.save_steps,
        save_total_limit=run_cfg.checkpoint.save_total_limit,
        checkpoint_batches=run_cfg.checkpoint.checkpoint_batches,
        max_grad_norm=run_cfg.training.max_grad_norm,
        warmup_epochs=run_cfg.training.warmup_epochs,
        patience=run_cfg.training.patience,
        num_workers=run_cfg.dataloader.num_workers,
        prefetch_size=run_cfg.dataloader.prefetch_size,
        p_reverb=run_cfg.augmentation.p_reverb,
        p_clipping=run_cfg.augmentation.p_clipping,
        use_mlx_data=run_cfg.dataloader.use_mlx_data,
        use_fp16=run_cfg.training.fp16,
        grad_accumulation_steps=run_cfg.training.grad_accumulation_steps,
        eval_frequency=run_cfg.training.eval_frequency,
        backbone_type=cast(Literal["mamba", "gru", "attention"], run_cfg.model.backbone_type),
        model_variant=cast(Literal["full", "lite"], run_cfg.model.variant),
        verbose=run_cfg.debug.verbose,
        snr_range=run_cfg.dataset.snr_range,
        snr_range_extreme=run_cfg.dataset.snr_range_extreme,
        snr_range_very_low=run_cfg.dataset.snr_range_very_low,
        p_extreme_snr=run_cfg.dataset.p_extreme_snr,
        p_very_low_snr=run_cfg.dataset.p_very_low_snr,
        p_interfer_speech=run_cfg.dataset.p_interfer_speech,
        curriculum_warmup_epochs=run_cfg.training.curriculum_warmup_epochs,
        speech_gain_range=run_cfg.dataset.speech_gain_range,
        noise_gain_range=run_cfg.dataset.noise_gain_range,
        dynamic_loss=cast(
            Literal["baseline", "awesome", "pipeline_awesome"],
            run_cfg.loss.dynamic_loss,
        ),
        pipeline_stages=run_cfg.loss.pipeline_stages,
        awesome_loss_weight=run_cfg.loss.awesome.loss_weight,
        awesome_mask_sharpness=run_cfg.loss.awesome.mask_sharpness,
        awesome_warmup_steps=run_cfg.loss.awesome.warmup_steps,
        gan_enabled=run_cfg.gan.enabled,
        gan_start_epoch=run_cfg.gan.start_epoch,
        gan_ramp_epochs=run_cfg.gan.ramp_epochs,
        gan_adv_weight=run_cfg.gan.adv_weight,
        gan_fm_weight=run_cfg.gan.fm_weight,
        gan_disc_type=cast(Literal["combined", "mpd", "msd"], run_cfg.gan.discriminator),
        gan_mpd_periods=(tuple(run_cfg.gan.mpd_periods) if run_cfg.gan.mpd_periods else None),
        gan_msd_scales=run_cfg.gan.msd_scales,
        gan_disc_lr=run_cfg.gan.disc_lr,
        gan_disc_weight_decay=run_cfg.gan.disc_weight_decay,
        gan_disc_grad_clip=run_cfg.gan.disc_grad_clip,
        gan_disc_update_freq=run_cfg.gan.disc_update_freq,
        gan_disc_max_samples=run_cfg.gan.disc_max_samples,
        gan_mpd_channels=run_cfg.gan.mpd_channels,
        gan_msd_channels=run_cfg.gan.msd_channels,
        experimental_compiled_gan=run_cfg.gan.experimental_compile,
        gan_cache_gen_waveforms=run_cfg.gan.cache_gen_waveforms,
        gan_disc_gradient_checkpoint=run_cfg.gan.disc_gradient_checkpoint,
        gan_gen_gradient_checkpoint=run_cfg.gan.gen_gradient_checkpoint,
        gan_eval_frequency=run_cfg.gan.eval_frequency,
        vad_proxy_enabled=run_cfg.loss.awesome.proxy_enabled,
        vad_loss_weight=run_cfg.vad.loss_weight,
        vad_threshold=run_cfg.vad.threshold,
        vad_margin=run_cfg.vad.margin,
        vad_speech_loss_weight=run_cfg.vad.speech_loss_weight,
        vad_warmup_epochs=run_cfg.vad.warmup_epochs,
        vad_snr_gate_db=run_cfg.vad.snr_gate_db,
        vad_snr_gate_width=run_cfg.vad.snr_gate_width,
        vad_band_low_hz=run_cfg.vad.band_low_hz,
        vad_band_high_hz=run_cfg.vad.band_high_hz,
        vad_z_threshold=run_cfg.vad.z_threshold,
        vad_z_slope=run_cfg.vad.z_slope,
        vad_eval_mode=cast(Literal["auto", "proxy", "silero", "off"], run_cfg.vad.eval.mode),
        vad_eval_every=run_cfg.vad.eval.every,
        vad_eval_batches=run_cfg.vad.eval.batches,
        vad_eval_max_seconds=run_cfg.vad.eval.max_seconds,
        vad_silero_model_path=run_cfg.vad.eval.silero_model_path,
        vad_silero_sample_rate=run_cfg.vad.eval.silero_sample_rate,
        vad_train_prob=run_cfg.vad.train.prob,
        vad_train_every_steps=run_cfg.vad.train.every_steps,
        eval_sisdr=run_cfg.metrics.eval_sisdr,
        check_chkpts=run_cfg.checkpoint.check_chkpts,
        max_train_batches=run_cfg.dataloader.max_train_batches,
        max_valid_batches=run_cfg.dataloader.max_valid_batches,
        seed=run_cfg.training.seed,
        debug_numerics=run_cfg.debug.debug_numerics,
        debug_numerics_fail_fast=run_cfg.debug.debug_numerics_fail_fast,
        debug_numerics_every=run_cfg.debug.debug_numerics_every,
        debug_numerics_dump_dir=run_cfg.debug.debug_numerics_dump_dir,
        debug_numerics_dump_arrays=run_cfg.debug.debug_numerics_dump_arrays,
        debug_numerics_max_dumps=run_cfg.debug.debug_numerics_max_dumps,
        nan_skip_batch=run_cfg.debug.nan_skip_batch,
        sync_mode=run_cfg.debug.sync_mode,
        model_config=model_cfg,
        dataset_overrides=dataset_overrides,
        mrstft_config=run_cfg.loss.mrstft,
        train_config_path=train_config_path,
    )


if __name__ == "__main__":
    main()
