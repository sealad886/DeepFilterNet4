import argparse
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from loguru import logger
from torch import Tensor, nn, optim
from torch.amp.autocast_mode import autocast
from torch.autograd.anomaly_mode import set_detect_anomaly
from torch.autograd.grad_mode import set_grad_enabled
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.types import Number

from df.checkpoint import check_patience, load_model, read_cp, write_cp
from df.config import Csv, config
from df.logger import init_logger, log_metrics, log_model_summary
from df.loss import Istft, Loss
from df.lr import cosine_scheduler
from df.model import ModelParams
from df.modules import get_device
from df.utils import (
    as_complex,
    as_real,
    check_finite_module,
    check_manual_seed,
    detach_hidden,
    get_host,
    get_norm_alpha,
    make_np,
)
from libdf import DF
from libdfdata import PytorchDataLoader as DataLoader  # type: ignore[import-not-found]

should_stop = False
debug = False
log_timings = False
state: Optional[DF] = None
istft: Optional[nn.Module]
discriminator: Optional[nn.Module] = None
MAX_NANS = 50


def setup_discriminator() -> Optional[nn.Module]:
    """Initialize discriminator for GAN training if enabled."""
    gan_enabled = config("GAN_ENABLED", False, bool, section="train")
    if not gan_enabled:
        return None

    from df.discriminator import CombinedDiscriminator, MultiPeriodDiscriminator, MultiScaleDiscriminator

    disc_type = config("DISCRIMINATOR_TYPE", "combined", str, section="train").lower()
    use_spectral_norm = config("DISCRIMINATOR_SPECTRAL_NORM", False, bool, section="train")

    if disc_type == "mpd":
        periods = config("MPD_PERIODS", [2, 3, 5, 7, 11], Csv(int), section="train")  # type: ignore[arg-type]
        disc = MultiPeriodDiscriminator(periods=periods, use_spectral_norm=use_spectral_norm)
    elif disc_type == "msd":
        num_scales = config("MSD_SCALES", 3, int, section="train")
        disc = MultiScaleDiscriminator(num_scales=num_scales, use_spectral_norm=use_spectral_norm)
    else:  # combined
        periods = config("MPD_PERIODS", [2, 3, 5, 7, 11], Csv(int), section="train")  # type: ignore[arg-type]
        num_scales = config("MSD_SCALES", 3, int, section="train")
        disc = CombinedDiscriminator(
            periods=periods,
            num_scales=num_scales,
            use_spectral_norm=use_spectral_norm,
        )

    logger.info(f"Initialized {disc_type.upper()} discriminator for GAN training")
    return disc.to(get_device())


@logger.catch
def main():
    global debug, state, log_timings

    parser = argparse.ArgumentParser()
    parser.add_argument("data_config_file", type=str, help="Path to a dataset config file.")
    parser.add_argument("data_dir", type=str, help="Path to the dataset directory containing .hdf5 files.")
    parser.add_argument("base_dir", type=str, help="Directory to store logs, summaries, checkpoints, etc.")
    parser.add_argument(
        "--host-batchsize-config",
        "-b",
        type=str,
        default=None,
        help="Path to a host specific batch size config.",
    )
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logger verbosity. Can be one of (trace, debug, info, error, none)",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-debug", action="store_false", dest="debug")
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Enable quantization-aware training (QAT) for INT8 deployment",
    )
    parser.add_argument(
        "--qat-start-epoch",
        type=int,
        default=0,
        help="Epoch to start QAT (default: 0, train with QAT from start)",
    )
    parser.add_argument(
        "--qat-backend",
        type=str,
        default="x86",
        choices=["x86", "fbgemm", "qnnpack", "onednn"],
        help="Quantization backend (default: x86)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling for first epoch to measure MPS/GPU utilization",
    )
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=50,
        help="Number of iterations to profile (default: 50)",
    )
    args = parser.parse_args()
    if not os.path.isfile(args.data_config_file):
        raise FileNotFoundError("Dataset config not found at {}".format(args.data_config_file))
    if not os.path.isdir(args.data_dir):
        NotADirectoryError("Data directory not found at {}".format(args.data_dir))
    os.makedirs(args.base_dir, exist_ok=True)
    summary_dir = os.path.join(args.base_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    debug = args.debug
    if args.log_level is not None:
        if debug and args.log_level.lower() != "debug":
            raise ValueError("Either specify debug or a manual log level")
        log_level = args.log_level
    else:
        log_level = "DEBUG" if debug else "INFO"
    init_logger(file=os.path.join(args.base_dir, "train.log"), level=log_level, model=args.base_dir)
    config_file = os.path.join(args.base_dir, "config.ini")
    config.load(config_file)

    # Hardware auto-tuning: detect and apply optimal settings if not explicitly configured
    auto_tune = config("AUTO_TUNE", True, bool, section="train")
    if auto_tune:
        try:
            from df.hardware import HardwareConfig

            hw_config = HardwareConfig.detect(verbose=True)

            # Apply auto-tuned settings: override if not set OR if set to invalid/zero values
            # This handles the case where config was saved with bad values from an interrupted session
            current_batch = config("BATCH_SIZE", 0, int, section="train")
            if current_batch <= 0:
                config.set("BATCH_SIZE", hw_config.batch_size, int, section="train")
                logger.info(f"Auto-tuned BATCH_SIZE: {hw_config.batch_size}")

            current_batch_eval = config("BATCH_SIZE_EVAL", 0, int, section="train")
            if current_batch_eval <= 0:
                config.set("BATCH_SIZE_EVAL", hw_config.batch_size_eval, int, section="train")
                logger.info(f"Auto-tuned BATCH_SIZE_EVAL: {hw_config.batch_size_eval}")

            current_workers = config("NUM_WORKERS", 0, int, section="train")
            if current_workers <= 0:
                config.set("NUM_WORKERS", hw_config.num_workers, int, section="train")
                logger.info(f"Auto-tuned NUM_WORKERS: {hw_config.num_workers}")

            current_prefetch = config("NUM_PREFETCH_BATCHES", 0, int, section="train")
            if current_prefetch <= 0:
                config.set("NUM_PREFETCH_BATCHES", hw_config.prefetch_batches, int, section="train")
                logger.info(f"Auto-tuned NUM_PREFETCH_BATCHES: {hw_config.prefetch_batches}")

            current_device = config("DEVICE", "", str, section="train")
            if not current_device:
                config.set("DEVICE", hw_config.device, str, section="train")
                logger.info(f"Auto-tuned DEVICE: {hw_config.device}")

            # Apply mixed precision settings
            current_amp = config("ENABLE_AMP", None, bool, section="train")
            if current_amp is None:
                config.set("ENABLE_AMP", hw_config.enable_amp, bool, section="train")
                logger.info(f"Auto-tuned ENABLE_AMP: {hw_config.enable_amp}")

            current_amp_dtype = config("AMP_DTYPE", "", str, section="train")
            if not current_amp_dtype:
                config.set("AMP_DTYPE", hw_config.amp_dtype, str, section="train")
                logger.info(f"Auto-tuned AMP_DTYPE: {hw_config.amp_dtype}")

            # Apply MPS fallback if needed
            if hw_config.use_mps_fallback:
                os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                logger.info("Enabled PYTORCH_ENABLE_MPS_FALLBACK for MPS complex tensor support")

            logger.info("Hardware auto-tuning applied (set AUTO_TUNE=false to disable)")
        except ImportError as e:
            logger.warning(f"Hardware auto-tuning unavailable: {e}")
        except Exception as e:
            logger.warning(f"Hardware auto-tuning failed, using defaults: {e}")

    seed = config("SEED", 42, int, section="train")
    check_manual_seed(seed)
    logger.info("Running on device {}".format(get_device()))

    # Maybe update batch size
    if args.host_batchsize_config is not None:
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from scripts.set_batch_size import main as set_batch_size  # type: ignore

            key = get_host() + "_" + config.get("model", section="train")
            key += "_" + config.get("fft_size", section="df")
            set_batch_size(config_file, args.host_batchsize_config, host_key=key)
            config.load(config_file, allow_reload=True)  # Load again
        except Exception as e:
            logger.error(f"Could not apply host specific batch size config: {str(e)}")

    signal.signal(signal.SIGUSR1, get_sigusr1_handler(args.base_dir))

    p = ModelParams()
    state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )
    checkpoint_dir = os.path.join(args.base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    mask_only: bool = config("MASK_ONLY", False, bool, section="train")
    train_df_only: bool = config("DF_ONLY", False, bool, section="train")
    jit = config("JIT", False, cast=bool, section="train")
    model, epoch = load_model(
        checkpoint_dir if args.resume else None,
        state,
        jit=False,
        mask_only=mask_only,
        train_df_only=train_df_only,
    )

    # Optional torch.compile for faster training (experimental on MPS)
    use_compile: bool = config("TORCH_COMPILE", False, bool, section="train")
    if use_compile:
        try:
            dev = get_device()
            # torch.compile works best with CUDA, has limited MPS support
            if dev.type == "cuda":
                logger.info("Compiling model with torch.compile (mode=reduce-overhead)")
                model = torch.compile(model, mode="reduce-overhead")  # type: ignore
            elif dev.type == "mps":
                logger.info("Compiling model with torch.compile (mode=default, MPS backend)")
                model = torch.compile(model, mode="default")  # type: ignore
            else:
                logger.warning("torch.compile not supported on CPU, skipping")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without: {e}")

    bs: int = config("BATCH_SIZE", 1, int, section="train")
    bs_eval: int = config("BATCH_SIZE_EVAL", 0, int, section="train")
    bs_eval = bs_eval if bs_eval > 0 else bs
    overfit = config("OVERFIT", False, bool, section="train")
    log_timings = config("LOG_TIMINGS", False, bool, section="train", save=False)
    dataloader = DataLoader(
        ds_dir=args.data_dir,
        ds_config=args.data_config_file,
        sr=p.sr,
        batch_size=bs,
        batch_size_eval=bs_eval,
        num_workers=config("NUM_WORKERS", 4, int, section="train"),
        pin_memory=get_device().type == "cuda",
        max_len_s=config("MAX_SAMPLE_LEN_S", 5.0, float, section="train"),
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_erb=p.nb_erb,
        nb_spec=p.nb_df,
        norm_alpha=get_norm_alpha(log=False),
        p_reverb=config("p_reverb", 0.2, float, section="distortion"),
        p_bw_ext=config("p_bandwidth_ext", 0.0, float, section="distortion"),
        p_clipping=config("p_clipping", 0.0, float, section="distortion"),
        p_zeroing=config("p_zeroing", 0.0, float, section="distortion"),
        p_air_absorption=config("p_air_absorption", 0.0, float, section="distortion"),
        p_interfer_sp=config("p_interfer_sp", 0.0, float, section="distortion"),
        prefetch=config("NUM_PREFETCH_BATCHES", 32, int, section="train"),
        overfit=overfit,
        seed=seed,
        min_nb_erb_freqs=p.min_nb_freqs,
        log_timings=log_timings,
        global_sampling_factor=config("GLOBAL_DS_SAMPLING_F", 1.0, float, section="train"),
        snrs=config("DATALOADER_SNRS", [-5, 0, 5, 10, 20, 40], Csv(int), section="train"),  # type: ignore
        gains=config("DATALOADER_GAINS", [-6, 0, 6], Csv(int), section="train"),  # type: ignore
        log_level=log_level,
    )

    # Batch size scheduling limits the batch size for the first epochs. It will increase the batch
    # size during training as specified. Used format is a comma separated list containing
    # epoch/batch size tuples where each tuple is separated via '/':
    # '<epoch>/<batch_size>,<epoch>/<batch_size>,<epoch>/<batch_size>'
    # The first epoch has to be 0, later epoch may modify the batch size as specified.
    # This only applies to training batch size.
    batch_size_scheduling_raw: List[str] = config("BATCH_SIZE_SCHEDULING", [], Csv(str), section="train")  # type: ignore
    batch_size_scheduling: List[Tuple[int, int]] = []
    scheduling_bs = bs
    prev_scheduling_bs = bs
    if len(batch_size_scheduling_raw) > 0:
        batch_size_scheduling = [
            (int(item.split("/")[0]), int(item.split("/")[1])) for item in batch_size_scheduling_raw
        ]
        assert batch_size_scheduling[0][0] == 0  # First epoch must be 0
        logger.info("Running with batch size scheduling")

    max_epochs = config("MAX_EPOCHS", 10, int, section="train")
    assert epoch >= 0
    opt = load_opt(
        checkpoint_dir if args.resume else None,
        model,
        mask_only,
        train_df_only,
    )
    lrs = setup_lrs(len(dataloader))
    wds = setup_wds(len(dataloader))
    if not args.resume and os.path.isfile(os.path.join(checkpoint_dir, ".patience")):
        os.remove(os.path.join(checkpoint_dir, ".patience"))
    try:
        log_model_summary(model, verbose=args.debug)
    except Exception as e:
        logger.warning(f"Failed to print model summary: {e}")

    # Quantization-aware training (QAT) setup
    qat_callback = None
    if args.qat:
        try:
            from df.quantization import QATCallback, check_quantization_available

            if check_quantization_available():
                qat_callback = QATCallback(
                    model,
                    start_epoch=args.qat_start_epoch,
                    freeze_bn_epoch=max_epochs - 2 if max_epochs > 2 else None,
                    backend=args.qat_backend,
                )
                logger.info(f"QAT enabled with backend={args.qat_backend}, start_epoch={args.qat_start_epoch}")
            else:
                logger.warning("QAT requested but quantization not available (requires torch>=2.0)")
        except ImportError as e:
            logger.warning(f"Could not import quantization module: {e}")

    if jit:
        # Load as jit after log_model_summary
        model = torch.jit.script(model)

    # Validation optimization target. Used for early stopping and selecting best checkpoint
    val_criteria = []
    val_criteria_type = config("VALIDATION_CRITERIA", "loss", section="train")  # must be in metrics
    val_criteria_rule = config("VALIDATION_CRITERIA_RULE", "min", section="train")
    val_criteria_rule = val_criteria_rule.replace("less", "min").replace("more", "max")
    patience = config("EARLY_STOPPING_PATIENCE", 5, int, section="train")

    losses = setup_losses()

    # GAN training setup
    gan_enabled = config("GAN_ENABLED", False, bool, section="train")
    opt_disc = None
    disc_lrs = None
    gan_start_epoch = 0
    if gan_enabled and discriminator is not None:
        opt_disc = load_discriminator_opt(checkpoint_dir if args.resume else None, discriminator)
        disc_lrs = setup_discriminator_lrs(len(dataloader))
        # Progressive GAN training: start GAN training after warmup epochs
        gan_start_epoch = config("GAN_START_EPOCH", 0, int, section="train")
        logger.info(f"GAN training enabled, starting at epoch {gan_start_epoch}")

    if config("START_EVAL", False, cast=bool, section="train"):
        val_loss = run_epoch(
            model=model,
            epoch=epoch - 1,
            loader=dataloader,
            split="valid",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
        )
        metrics = {"loss": val_loss}
        metrics.update({n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()})
        log_metrics(f"[{epoch - 1}] [valid]", metrics)
    losses.reset_summaries()
    # Save default values to disk
    config.save(os.path.join(args.base_dir, "config.ini"))
    for epoch in range(epoch, max_epochs):
        # QAT epoch start callback
        if qat_callback is not None:
            qat_callback.on_epoch_start(epoch)

        if len(batch_size_scheduling) > 0:
            # Get current batch size
            for e, b in batch_size_scheduling:
                if e <= epoch:
                    # Update bs, but don't go higher than the batch size specified in the config
                    scheduling_bs = min(b, bs)
            if prev_scheduling_bs != scheduling_bs:
                logger.info(f"Batch scheduling | Setting batch size to {scheduling_bs}")
                dataloader.set_batch_size(scheduling_bs, "train")
                # Update lr/wd scheduling since dataloader len changed
                lrs = setup_lrs(len(dataloader))
                wds = setup_wds(len(dataloader))
                if gan_enabled:
                    disc_lrs = setup_discriminator_lrs(len(dataloader))
                prev_scheduling_bs = scheduling_bs
        train_loss = run_epoch(
            model=model,
            epoch=epoch,
            loader=dataloader,
            split="train",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
            lr_scheduler_values=lrs,
            wd_scheduler_values=wds,
            opt_disc=opt_disc,
            disc_lr_values=disc_lrs,
            gan_start_epoch=gan_start_epoch if gan_enabled else 0,
            profile_iterations=args.profile_iterations if args.profile else 0,
        )
        metrics = {"loss": train_loss}
        try:
            metrics["lr"] = opt.param_groups[0]["lr"]
        except AttributeError:
            pass
        if debug:
            metrics.update({n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()})
        log_metrics(f"[{epoch}] [train]", metrics)
        write_cp(model, "model", checkpoint_dir, epoch + 1)
        write_cp(opt, "opt", checkpoint_dir, epoch + 1)
        # Save discriminator checkpoint if GAN training is enabled
        if gan_enabled and discriminator is not None and opt_disc is not None:
            write_cp(discriminator, "disc", checkpoint_dir, epoch + 1)
            write_cp(opt_disc, "opt_disc", checkpoint_dir, epoch + 1)
        losses.reset_summaries()
        val_loss = run_epoch(
            model=model,
            epoch=epoch,
            loader=dataloader,
            split="valid",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
        )
        metrics = {"loss": val_loss}
        metrics.update({n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()})
        val_criteria = metrics[val_criteria_type]
        write_cp(model, "model", checkpoint_dir, epoch + 1, metric=val_criteria, cmp=val_criteria_rule)
        log_metrics(f"[{epoch}] [valid]", metrics)
        if not check_patience(
            checkpoint_dir,
            max_patience=patience,
            new_metric=val_criteria,
            cmp=val_criteria_rule,
            raise_=False,
        ):
            break
        if should_stop:
            logger.info("Stopping training due to timeout")
            exit(0)
        losses.reset_summaries()

        # QAT epoch end callback
        if qat_callback is not None:
            qat_callback.on_epoch_end(epoch)

    # Export quantized model if QAT was enabled
    if qat_callback is not None and qat_callback.qat_active:
        try:
            from df.quantization import convert_qat_model, export_quantized_model

            logger.info("Converting QAT model to quantized model...")
            quantized_model = convert_qat_model(model)
            qat_export_path = os.path.join(checkpoint_dir, "model_quantized.pt")
            export_quantized_model(quantized_model, qat_export_path, export_format="state_dict")
            logger.info(f"Quantized model exported to {qat_export_path}")
        except Exception as e:
            logger.warning(f"Failed to export quantized model: {e}")

    model, epoch = load_model(
        checkpoint_dir,
        state,
        jit=jit,
        epoch="best",
        mask_only=mask_only,
        train_df_only=train_df_only,
    )
    test_loss = run_epoch(
        model=model,
        epoch=epoch,
        loader=dataloader,
        split="test",
        opt=opt,
        losses=losses,
        summary_dir=summary_dir,
    )
    metrics: Dict[str, Number] = {"loss": test_loss}
    metrics.update({n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()})
    log_metrics(f"[{epoch}] [test]", metrics)
    logger.info("Finished training")


def synthesize_waveform(spec: Tensor, df_state: DF) -> Tensor:
    """Synthesize waveform from complex spectrogram using DF state.

    Args:
        spec: Complex spectrogram [B, 1, T, F, 2] or [B, T, F, 2]
        df_state: DF state for ISTFT

    Returns:
        Waveform tensor [B, T_samples]
    """
    global istft
    if istft is None:
        raise RuntimeError("ISTFT not initialized. Call setup_losses() first.")

    # Handle different input shapes
    if spec.dim() == 5:
        spec = spec.squeeze(1)  # [B, T, F, 2]

    # Convert to complex
    spec_complex = as_complex(spec)  # [B, T, F]

    # Transpose for ISTFT: [B, F, T]
    spec_complex = spec_complex.transpose(1, 2)

    # Synthesize using ISTFT
    waveform = istft(spec_complex)  # [B, T_samples]

    return waveform


def train_discriminator_step(
    disc: nn.Module,
    opt_disc: optim.Optimizer,
    losses: Loss,
    real_wav: Tensor,
    fake_wav: Tensor,
) -> Tensor:
    """Perform one discriminator training step.

    Args:
        disc: Discriminator module
        opt_disc: Discriminator optimizer
        losses: Loss module with compute_d_loss_with_disc method
        real_wav: Real (clean) waveform [B, T]
        fake_wav: Fake (enhanced) waveform [B, T], should be detached

    Returns:
        Discriminator loss value
    """
    opt_disc.zero_grad()

    # Compute discriminator loss using new wrapper method
    d_loss = losses.compute_d_loss_with_disc(disc, real_wav, fake_wav)

    # Backward and step
    d_loss.backward()
    clip_grad_norm_(disc.parameters(), 1.0, error_if_nonfinite=True)
    opt_disc.step()

    return d_loss


def run_epoch(
    model: nn.Module,
    epoch: int,
    loader: DataLoader,
    split: str,
    opt: optim.Optimizer,
    losses: Loss,
    summary_dir: str,
    lr_scheduler_values: Optional[np.ndarray] = None,
    wd_scheduler_values: Optional[np.ndarray] = None,
    opt_disc: Optional[optim.Optimizer] = None,
    disc_lr_values: Optional[np.ndarray] = None,
    gan_start_epoch: int = 0,
    profile_iterations: int = 0,
) -> float:
    log_freq = config("LOG_FREQ", cast=int, default=100, section="train")
    bs = loader.get_batch_size(split)
    logger.info("Start {} epoch {} with batch size {}".format(split, epoch, bs))

    detect_anomaly: bool = config("DETECT_ANOMALY", False, bool, section="train")
    if detect_anomaly:
        logger.info("Running with autograd profiling")
    dev = get_device()

    # Mixed precision setup
    enable_amp: bool = config("ENABLE_AMP", False, bool, section="train")
    amp_dtype_str: str = config("AMP_DTYPE", "float16", str, section="train")
    amp_dtype = torch.float16 if amp_dtype_str == "float16" else torch.bfloat16

    # Determine autocast device type
    if dev.type == "mps":
        amp_device = "mps"
    elif dev.type == "cuda":
        amp_device = "cuda"
    else:
        amp_device = "cpu"
        enable_amp = False  # AMP not beneficial on CPU

    if enable_amp:
        logger.info(f"Mixed precision enabled: {amp_dtype_str} on {amp_device}")
        amp_context = autocast(device_type=amp_device, dtype=amp_dtype)
    else:
        amp_context = nullcontext()

    l_mem = []
    l_gan_g_mem = []
    l_gan_d_mem = []
    is_train = split == "train"
    model.train(mode=is_train)
    if discriminator is not None:
        discriminator.train(mode=is_train)
    losses.store_losses = debug or not is_train
    max_steps = loader.len(split) - 1
    seed = epoch if is_train else 42
    n_nans = 0
    start_steps = epoch * loader.len(split)

    # Profiling setup
    do_profile = profile_iterations > 0 and is_train and epoch == 0
    profile_timings: Dict[str, List[float]] = {
        "transfer": [],
        "forward": [],
        "loss": [],
        "backward": [],
        "optimizer": [],
        "total": [],
    }

    def sync_device():
        """Synchronize device for accurate timing."""
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()

    # GAN training settings
    gan_active = is_train and discriminator is not None and opt_disc is not None and epoch >= gan_start_epoch
    disc_update_freq = config("DISCRIMINATOR_UPDATE_FREQ", 1, int, section="train") if gan_active else 1

    for i, batch in enumerate(loader.iter_epoch(split, seed)):
        iter_start = time.perf_counter() if do_profile and i < profile_iterations else 0.0

        opt.zero_grad()
        if opt_disc is not None:
            opt_disc.zero_grad()
        it = start_steps + i  # global training iteration

        # Update generator learning rate
        if lr_scheduler_values is not None or wd_scheduler_values is not None:
            for param_group in opt.param_groups:
                if lr_scheduler_values is not None:
                    param_group["lr"] = lr_scheduler_values[it] * param_group.get("lr_scale", 1)
                if wd_scheduler_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_scheduler_values[it]

        # Update discriminator learning rate
        if gan_active and disc_lr_values is not None and opt_disc is not None:
            for param_group in opt_disc.param_groups:
                param_group["lr"] = disc_lr_values[it] * param_group.get("lr_scale", 1)

        assert batch.feat_spec is not None
        assert batch.feat_erb is not None

        # Profile data transfer
        if do_profile and i < profile_iterations:
            transfer_start = time.perf_counter()

        feat_erb = batch.feat_erb.to(dev, non_blocking=True)
        feat_spec = as_real(batch.feat_spec.to(dev, non_blocking=True))
        noisy = batch.noisy.to(dev, non_blocking=True)
        clean = batch.speech.to(dev, non_blocking=True)
        snrs = batch.snr.to(dev, non_blocking=True)

        if do_profile and i < profile_iterations:
            sync_device()
            profile_timings["transfer"].append((time.perf_counter() - transfer_start) * 1000)

        with set_detect_anomaly(detect_anomaly and is_train), set_grad_enabled(is_train), amp_context:
            if not is_train:
                input = as_real(noisy).clone()
            else:
                input = as_real(noisy)

            # Profile forward pass
            if do_profile and i < profile_iterations:
                forward_start = time.perf_counter()

            enh, m, lsnr, other = model.forward(
                spec=input,
                feat_erb=feat_erb,
                feat_spec=feat_spec,
            )

            if do_profile and i < profile_iterations:
                sync_device()
                profile_timings["forward"].append((time.perf_counter() - forward_start) * 1000)

            # Profile loss computation
            if do_profile and i < profile_iterations:
                loss_start = time.perf_counter()

            try:
                err = losses.forward(clean, noisy, enh, m, lsnr, snrs=snrs)
            except Exception as e:
                if "nan" in str(e).lower() or "finite" in str(e).lower():
                    logger.warning("NaN in loss computation: {}. Skipping backward.".format(str(e)))
                    check_finite_module(model)
                    n_nans += 1
                    if n_nans > MAX_NANS:
                        raise e
                    continue
                raise e

            # GAN training
            gan_g_loss = torch.tensor(0.0, device=dev)
            gan_d_loss = torch.tensor(0.0, device=dev)

            if gan_active:
                # These assertions satisfy the type checker - gan_active implies all are not None
                assert state is not None
                assert discriminator is not None
                assert opt_disc is not None

                # Synthesize waveforms for discriminator
                clean_wav = synthesize_waveform(clean, state)
                enh_wav = synthesize_waveform(enh, state)

                # Discriminator update (every disc_update_freq steps)
                if i % disc_update_freq == 0:
                    gan_d_loss = train_discriminator_step(discriminator, opt_disc, losses, clean_wav, enh_wav.detach())
                    l_gan_d_mem.append(gan_d_loss.detach())

                # Generator GAN loss using new wrapper method
                gan_g_loss = losses.compute_g_loss_with_disc(discriminator, clean_wav, enh_wav)
                err = err + gan_g_loss
                l_gan_g_mem.append(gan_g_loss.detach())

            if do_profile and i < profile_iterations:
                sync_device()
                profile_timings["loss"].append((time.perf_counter() - loss_start) * 1000)

            if is_train:
                # Profile backward pass
                if do_profile and i < profile_iterations:
                    backward_start = time.perf_counter()

                try:
                    err.backward()
                    clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                except RuntimeError as e:
                    e_str = str(e)
                    if "nan" in e_str.lower() or "non-finite" in e_str:
                        check_finite_module(model)
                        logger.error(e_str)
                        os.makedirs(os.path.join(summary_dir, "nan"), exist_ok=True)
                        for batch_idx in range(clean.shape[0]):
                            clean_idx = batch.ids[batch_idx].item()
                            summary_write(
                                clean.detach(),
                                noisy.detach(),
                                enh.detach(),
                                batch.snr.detach(),
                                lsnr.detach().float(),
                                os.path.join(summary_dir, "nan"),
                                prefix=split + f"_e{epoch}_i{i}_b{batch_idx}_ds{clean_idx}",
                                idx=batch_idx,
                            )
                        cleanup(err, noisy, clean, enh, m, feat_erb, feat_spec, batch)
                        n_nans += 1
                        if n_nans > MAX_NANS:
                            raise e
                        continue
                    else:
                        raise e

                if do_profile and i < profile_iterations:
                    sync_device()
                    profile_timings["backward"].append((time.perf_counter() - backward_start) * 1000)
                    opt_start = time.perf_counter()

                opt.step()

                if do_profile and i < profile_iterations:
                    sync_device()
                    profile_timings["optimizer"].append((time.perf_counter() - opt_start) * 1000)

            detach_hidden(model)
        l_mem.append(err.detach())

        # Record total iteration time
        if do_profile and i < profile_iterations:
            profile_timings["total"].append((time.perf_counter() - iter_start) * 1000)

        # Print profiling report after collecting all samples
        if do_profile and i == profile_iterations - 1:
            logger.info("\n" + "=" * 80)
            logger.info("MPS/GPU PROFILING REPORT")
            logger.info("=" * 80)
            logger.info(f"Device: {dev}")
            logger.info(f"Profiled iterations: {profile_iterations}")
            logger.info("-" * 80)

            total_avg = sum(profile_timings["total"]) / len(profile_timings["total"])
            for stage, times in profile_timings.items():
                if times:
                    avg = sum(times) / len(times)
                    pct = (avg / total_avg * 100) if total_avg > 0 else 0
                    bar = "█" * int(pct / 2)
                    logger.info(f"{stage:12s}: {avg:8.2f} ms ({pct:5.1f}%) {bar}")

            # MPS utilization estimate
            if dev.type == "mps":
                gpu_time = (
                    sum(profile_timings["forward"]) / len(profile_timings["forward"])
                    + sum(profile_timings["backward"]) / len(profile_timings["backward"])
                    + sum(profile_timings["loss"]) / len(profile_timings["loss"])
                )
                logger.info("-" * 80)
                logger.info(f"Estimated MPS GPU compute time: {gpu_time:.2f} ms/iter")
                logger.info(f"Estimated MPS utilization: {gpu_time / total_avg * 100:.1f}%")

                # Memory stats
                logger.info("-" * 80)
                logger.info(f"MPS allocated: {torch.mps.current_allocated_memory() / 1e9:.3f} GB")
                logger.info(f"MPS driver allocated: {torch.mps.driver_allocated_memory() / 1e9:.3f} GB")

            logger.info("=" * 80 + "\n")

        if i % log_freq == 0:
            l_mean = torch.stack(l_mem[-100:]).mean().cpu()
            if torch.isnan(l_mean):
                check_finite_module(model)
            l_dict = {"loss": l_mean.item()}
            if lr_scheduler_values is not None:
                l_dict["lr"] = opt.param_groups[0]["lr"]
            if wd_scheduler_values is not None:
                l_dict["wd"] = opt.param_groups[0]["weight_decay"]
            if gan_active and len(l_gan_g_mem) > 0:
                l_dict["gan_g"] = torch.stack(l_gan_g_mem[-100:]).mean().cpu().item()
            if gan_active and len(l_gan_d_mem) > 0:
                l_dict["gan_d"] = torch.stack(l_gan_d_mem[-100:]).mean().cpu().item()
            if log_timings:
                l_dict["t_sample"] = batch.timings[:-1].sum()
                l_dict["t_batch"] = batch.timings[-1].mean()  # last is for whole batch
            if debug:
                l_dict.update({n: torch.mean(torch.stack(vals[-bs:])).item() for n, vals in losses.get_summaries()})
            step = str(i).rjust(len(str(max_steps)))
            log_metrics(f"[{epoch}] [{step}/{max_steps}]", l_dict)
            summary_write(
                clean.detach(),
                noisy.detach(),
                enh.detach(),
                batch.snr.detach(),
                lsnr.detach().float(),
                summary_dir,
                prefix=split,
            )
    try:
        cleanup(err, noisy, clean, enh, m, feat_erb, feat_spec, batch)
    except UnboundLocalError as err:
        logger.error(str(err))
    return torch.stack(l_mem).mean().cpu().item()


def setup_losses() -> Loss:
    global istft, discriminator
    assert state is not None

    p = ModelParams()

    istft = Istft(p.fft_size, p.hop_size, torch.as_tensor(state.fft_window().copy())).to(get_device())

    # Initialize discriminator if GAN training is enabled
    discriminator = setup_discriminator()

    loss = Loss(state, istft, discriminator=discriminator).to(get_device())
    # loss = torch.jit.script(loss)
    return loss


def load_discriminator_opt(
    cp_dir: Optional[str],
    disc: nn.Module,
) -> optim.Optimizer:
    """Load optimizer for discriminator."""
    lr = config("DISCRIMINATOR_LR", 2e-4, float, section="optim")
    decay = config("DISCRIMINATOR_WEIGHT_DECAY", 0.0, float, section="optim")
    betas: Tuple[float, float] = config(
        "DISCRIMINATOR_BETAS", [0.8, 0.99], Csv(float), section="optim", save=False  # type: ignore
    )

    opt = optim.AdamW(disc.parameters(), lr=lr, weight_decay=decay, betas=betas)
    logger.debug(f"Training discriminator with optimizer {opt}")

    if cp_dir is not None:
        try:
            read_cp(opt, "opt_disc", cp_dir, log=False)
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Could not load discriminator optimizer state: {e}")

    for group in opt.param_groups:
        group.setdefault("initial_lr", lr)

    return opt


def load_opt(
    cp_dir: Optional[str], model: nn.Module, mask_only: bool = False, df_only: bool = False
) -> optim.Optimizer:
    lr = config("LR", 5e-4, float, section="optim")
    momentum = config("momentum", 0, float, section="optim")  # For sgd, rmsprop
    decay = config("weight_decay", 0.05, float, section="optim")
    optimizer = config("optimizer", "adamw", str, section="optim").lower()
    betas: Tuple[int, int] = config("opt_betas", [0.9, 0.999], Csv(float), section="optim", save=False)  # type: ignore
    if mask_only:
        params = []
        for n, p in model.named_parameters():
            if not ("dfrnn" in n or "df_dec" in n):
                params.append(p)
    elif df_only:
        params = (p for n, p in model.named_parameters() if "df" in n.lower())
    else:
        params = model.parameters()
    supported = {
        "adam": lambda p: optim.Adam(p, lr=lr, weight_decay=decay, betas=betas, amsgrad=True),
        "adamw": lambda p: optim.AdamW(p, lr=lr, weight_decay=decay, betas=betas, amsgrad=True),
        "sgd": lambda p: optim.SGD(p, lr=lr, momentum=momentum, nesterov=True, weight_decay=decay),
        "rmsprop": lambda p: optim.RMSprop(p, lr=lr, momentum=momentum, weight_decay=decay),
    }
    if optimizer not in supported:
        raise ValueError(f"Unsupported optimizer: {optimizer}. Must be one of {list(supported.keys())}")
    opt = supported[optimizer](params)
    logger.debug(f"Training with optimizer {opt}")
    if cp_dir is not None:
        try:
            read_cp(opt, "opt", cp_dir, log=False)
        except ValueError as e:
            logger.error(f"Could not load optimizer state: {e}")
    for group in opt.param_groups:
        group.setdefault("initial_lr", lr)
    return opt


def setup_lrs(steps_per_epoch: int) -> np.ndarray:
    lr = config.get("lr", float, "optim")
    num_epochs = config.get("max_epochs", int, "train")
    lr_min = config("lr_min", 1e-6, float, section="optim")
    lr_warmup = config("lr_warmup", 1e-4, float, section="optim")
    assert lr_warmup < lr
    warmup_epochs = config("warmup_epochs", 3, int, section="optim")
    lr_cycle_mul = config("lr_cycle_mul", 1.0, float, section="optim")
    lr_cycle_decay = config("lr_cycle_decay", 0.5, float, section="optim")
    lr_cycle_epochs = config("lr_cycle_epochs", -1, int, section="optim")
    lr_values = cosine_scheduler(
        lr,
        lr_min,
        epochs=num_epochs,
        niter_per_ep=steps_per_epoch,
        warmup_epochs=warmup_epochs,
        start_warmup_value=lr_warmup,
        initial_ep_per_cycle=lr_cycle_epochs,
        cycle_decay=lr_cycle_decay,
        cycle_mul=lr_cycle_mul,
    )
    return lr_values


def setup_discriminator_lrs(steps_per_epoch: int) -> np.ndarray:
    """Setup learning rate schedule for discriminator."""
    lr = config("DISCRIMINATOR_LR", 2e-4, float, section="optim")
    num_epochs = config.get("max_epochs", int, "train")
    lr_min = config("DISCRIMINATOR_LR_MIN", 1e-6, float, section="optim")
    warmup_epochs = config("DISCRIMINATOR_WARMUP_EPOCHS", 0, int, section="optim")

    lr_values = cosine_scheduler(
        lr,
        lr_min,
        epochs=num_epochs,
        niter_per_ep=steps_per_epoch,
        warmup_epochs=warmup_epochs,
        start_warmup_value=lr_min,
    )
    return lr_values


def setup_wds(steps_per_epoch: int) -> Optional[np.ndarray]:
    decay = config("weight_decay", 0.05, float, section="optim")
    decay_end = config("weight_decay_end", -1, float, section="optim")
    if decay_end == -1:
        return None
    if decay == 0.0:
        decay = 1e-12
        logger.warning("Got 'weight_decay_end' value > 0, but weight_decay is disabled.")
        logger.warning(f"Setting initial weight decay to {decay}.")
        config.overwrite("optim", "weight_decay", decay)
    num_epochs = config.get("max_epochs", int, "train")
    decay_values = cosine_scheduler(decay, decay_end, niter_per_ep=steps_per_epoch, epochs=num_epochs)
    return decay_values


@torch.no_grad()
def summary_write(
    clean: Tensor,
    noisy: Tensor,
    enh: Tensor,
    snrs: Tensor,
    lsnr: Tensor,
    summary_dir: str,
    prefix="train",
    idx: Optional[int] = None,
):
    assert state is not None
    _state = state  # Capture in local variable for closure

    p = ModelParams()
    bs = snrs.shape[0]
    if idx is None:
        idx = random.randrange(bs)
    snr = snrs[idx].detach().cpu().item()

    def synthesis(x: Tensor) -> Tensor:
        return torch.as_tensor(_state.synthesis(make_np(as_complex(x.detach()))))

    torchaudio.save(os.path.join(summary_dir, f"{prefix}_clean_snr{snr}.wav"), synthesis(clean[idx]), p.sr)
    torchaudio.save(os.path.join(summary_dir, f"{prefix}_noisy_snr{snr}.wav"), synthesis(noisy[idx]), p.sr)
    torchaudio.save(os.path.join(summary_dir, f"{prefix}_enh_snr{snr}.wav"), synthesis(enh[idx]), p.sr)
    np.savetxt(
        os.path.join(summary_dir, f"{prefix}_lsnr_snr{snr}.txt"),
        lsnr[idx].detach().cpu().numpy(),
        fmt="%.3f",
    )


def summary_noop(*__args, **__kwargs):  # type: ignore
    pass


def get_sigusr1_handler(base_dir):
    def h(*__args):  # type: ignore
        global should_stop
        logger.warning("Received timeout signal. Stopping after current epoch")
        should_stop = True
        continue_file = os.path.join(base_dir, "continue")
        logger.warning(f"Writing {continue_file}")
        open(continue_file, "w").close()

    return h


def cleanup(*args):
    import gc

    for arg in args:
        del arg
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        from icecream import ic
        from icecream.builtins import install

        ic.includeContext = True
        install()
    except ImportError:
        pass  # icecream is optional for debugging
    main()
