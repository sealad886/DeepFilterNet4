"""Validation loop for DeepFilterNet4 dynamic training.

Runs the full validation pass over a held-out dataset at the end of each epoch
(or at sync boundaries when configured).  Computes all loss components
(spectral, MRSTFT, VAD, awesome, pipeline) and aggregates per-SNR-bucket
metrics for detailed logging.

Key exports:
    - ValidationContext: Bundles immutable configuration needed by run_validation.
    - run_validation: Execute a complete validation pass and return a metrics dict.

Relationship to train_dynamic:
    ValidationContext is constructed once during train() setup.  run_validation
    is called at epoch end (and optionally at sync boundaries).  Not included
    in the backward-compat re-export block; imported directly by train_dynamic.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm.auto import tqdm

from df_mlx.training_losses import (
    _compute_awesome_losses,
    _compute_pipeline_awesome_losses,
    _compute_speech_band_logmag_loss,
    _compute_vad_eval_metrics,
    _compute_vad_loss,
    _compute_vad_probs,
    _compute_vad_reg_loss,
    _snr_bucket_name,
)
from df_mlx.training_ops import _batch_to_float
from df_mlx.training_waveform import compute_mrstft_loss

if TYPE_CHECKING:
    from df_mlx.training_ops import NumericDebugger

from df_mlx.training_helpers import SCALAR_ZERO


@dataclass
class ValidationContext:
    """Immutable context for validation runs.

    Groups all configuration and objects needed by run_validation() that
    remain constant throughout training.  Created once during setup and
    passed to every validation call.
    """

    model: nn.Module
    dataset: Any  # DynamicDataset
    batch_size: int
    fft_size: int
    hop_size: int
    sample_rate: int
    spectral_loss_fn: Callable
    # Loss flags
    use_awesome_loss: bool
    use_pipeline_awesome_loss: bool
    use_vad_loss: bool
    use_vad_train_reg: bool
    use_mrstft_loss: bool
    mrstft_loss_fn: Any
    mrstft_istft: Any
    mrstft_target_len: int | None
    # Awesome config
    awesome_mask_sharpness: float
    awesome_warmup_steps: int
    # VAD config
    vad_band_mask: Any  # mx.array
    vad_band_bins: float
    vad_z_threshold: float
    vad_z_slope: float
    vad_snr_gate_db: float
    vad_snr_gate_width: float
    vad_proxy_enabled: bool
    vad_threshold: float
    vad_margin: float
    # VAD eval
    vad_eval_mode: str
    vad_eval_batches: int
    silero_vad: Any
    # Debugging
    debugger: NumericDebugger | None
    # Eval config
    eval_sisdr: bool
    emit_detailed_metrics: bool
    max_valid_batches: int | None
    # Data loading
    use_mlx_stream: bool
    prefetch_size: int
    num_workers: int
    # Display / checkpointing
    ckpt_dir: Path
    dynamic_loss: str
    tqdm_valid_position: int
    tqdm_panels: bool
    tqdm_kwargs: dict


def run_validation(
    ctx: ValidationContext,
    *,
    epoch: int,
    global_step: int,
    epoch_awesome_loss_weight: float,
    epoch_vad_loss_weight: float,
    epoch_vad_speech_loss_weight: float,
    active_stage_index: int,
    active_stage_name: str,
    train_mode: str,
    label: str = "  Validating",
    do_vad_eval: bool = False,
) -> float:
    """Run validation on the fixed validation split and return average loss."""
    from df_mlx.dynamic_dataset import MLXDataStream, PrefetchDataLoader

    ctx.model.eval()

    ctx.dataset.set_split("valid")
    ctx.dataset.set_epoch(0)  # Fixed epoch for reproducible validation

    if len(ctx.dataset) == 0:
        return float("inf")

    valid_loss = 0.0
    valid_spec_loss = 0.0
    valid_mrstft_loss = 0.0
    valid_vad_loss = 0.0
    valid_speech_loss = 0.0
    valid_awesome_loss = 0.0
    valid_awesome_speech = 0.0
    valid_awesome_noise = 0.0
    valid_awesome_smooth = 0.0
    valid_music_supp_loss = 0.0
    valid_mask_sat_loss = 0.0
    valid_mask_mean = 0.0
    valid_mask_high = 0.0
    valid_mask_low = 0.0
    valid_proxy_mean = 0.0
    valid_speech_ratio = 0.0
    valid_music_gate = 0.0
    valid_musicness = 0.0
    valid_mod_energy = 0.0
    valid_energy_boost = 0.0
    valid_snr_boost = 0.0
    valid_vad_reg_loss = 0.0
    valid_p_ref = 0.0
    valid_p_out = 0.0
    valid_gate_pct = 0.0
    valid_residual = 0.0
    valid_sisdr = 0.0
    bucket_metrics: dict[str, dict[str, float]] = {}
    vad_eval_p_ref = 0.0
    vad_eval_p_out = 0.0
    vad_eval_delta = 0.0
    vad_eval_batches_done = 0
    vad_eval_seconds = 0.0
    vad_eval_clips = 0
    num_valid_batches = 0
    valid_steps = len(ctx.dataset) // ctx.batch_size
    if ctx.max_valid_batches is not None:
        valid_steps = min(valid_steps, ctx.max_valid_batches)

    if ctx.use_mlx_stream:
        valid_loader = MLXDataStream(
            dataset=ctx.dataset,
            batch_size=ctx.batch_size,
            prefetch_size=max(1, ctx.prefetch_size // 2),
            num_workers=max(1, min(ctx.num_workers, 4)),
        )
        valid_loader.set_split("valid")
        valid_loader.set_epoch(0)
    else:
        valid_loader = PrefetchDataLoader(
            ctx.dataset,
            batch_size=ctx.batch_size,
            num_workers=max(1, ctx.num_workers),
            prefetch_factor=2,
        )

    valid_tqdm_kwargs = dict(ctx.tqdm_kwargs)
    if ctx.tqdm_panels:
        valid_tqdm_kwargs["position"] = ctx.tqdm_valid_position

    valid_pbar = tqdm(
        valid_loader,
        total=valid_steps,
        desc=label,
        unit="batch",
        leave=False,
        **valid_tqdm_kwargs,
    )

    sisdr_fn = None
    if ctx.eval_sisdr:
        from df_mlx.loss import si_sdr
        from df_mlx.ops import istft

        sisdr_fn = (si_sdr, istft)

    silero_istft = None
    if do_vad_eval and ctx.vad_eval_mode == "silero":
        from df_mlx.ops import istft

        silero_istft = istft

    for batch_idx, batch in enumerate(valid_pbar):
        noisy_real = batch["noisy_real"]
        noisy_imag = batch["noisy_imag"]
        clean_real = batch["clean_real"]
        clean_imag = batch["clean_imag"]
        feat_erb = batch["feat_erb"]
        feat_spec = batch["feat_spec"]
        snr = batch["snr"]
        debug_ctx = {
            "phase": "valid",
            "epoch": epoch,
            "batch": batch_idx,
            "global_step": global_step,
        }
        if ctx.debugger is not None:
            ctx.debugger.check("batch.noisy_real", noisy_real, debug_ctx)
            ctx.debugger.check("batch.noisy_imag", noisy_imag, debug_ctx)
            ctx.debugger.check("batch.clean_real", clean_real, debug_ctx)
            ctx.debugger.check("batch.clean_imag", clean_imag, debug_ctx)
            ctx.debugger.check("batch.feat_erb", feat_erb, debug_ctx)
            ctx.debugger.check("batch.feat_spec", feat_spec, debug_ctx)
            ctx.debugger.check("batch.snr", snr, debug_ctx)

        # Model expects spec as tuple (real, imag)
        noisy_spec = (noisy_real, noisy_imag)
        target_spec = (clean_real, clean_imag)

        out = ctx.model(noisy_spec, feat_erb, feat_spec, return_vad=True)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], tuple):
            spec_out, vad_logits = out
        else:
            spec_out = out
            vad_logits = None

        if ctx.debugger is not None:
            ctx.debugger.check("model.out_real", spec_out[0], debug_ctx)
            ctx.debugger.check("model.out_imag", spec_out[1], debug_ctx)
            if vad_logits is not None:
                ctx.debugger.check("model.vad_logits", vad_logits, debug_ctx)
        spec_loss = ctx.spectral_loss_fn(spec_out, target_spec)
        mrstft_loss = SCALAR_ZERO
        if ctx.use_mrstft_loss and ctx.mrstft_loss_fn is not None and ctx.mrstft_istft is not None:
            mrstft_loss = compute_mrstft_loss(
                spec_out,
                target_spec,
                istft_fn=ctx.mrstft_istft,
                loss_fn=ctx.mrstft_loss_fn,
                n_fft=ctx.fft_size,
                hop_length=ctx.hop_size,
                target_len=ctx.mrstft_target_len,
                force_fp32=True,
            )

        awesome_loss = SCALAR_ZERO
        awesome_speech = SCALAR_ZERO
        awesome_noise = SCALAR_ZERO
        awesome_smooth = SCALAR_ZERO
        music_suppression_loss = SCALAR_ZERO
        mask_saturation_loss = SCALAR_ZERO
        mask = SCALAR_ZERO
        proxy_frame = SCALAR_ZERO
        speech_ratio = SCALAR_ZERO
        music_gate = SCALAR_ZERO
        musicness = SCALAR_ZERO
        mod_energy = SCALAR_ZERO
        energy_boost = SCALAR_ZERO
        snr_boost = SCALAR_ZERO

        if ctx.use_awesome_loss:
            (
                awesome_loss,
                awesome_speech,
                awesome_noise,
                awesome_smooth,
                mask,
                proxy_frame,
                speech_ratio,
                music_gate,
                musicness,
                mod_energy,
                energy_boost,
                snr_boost,
            ) = _compute_awesome_losses(
                noisy_real,
                noisy_imag,
                clean_real,
                clean_imag,
                spec_out[0],
                spec_out[1],
                snr,
                ctx.vad_band_mask,
                ctx.vad_band_bins,
                ctx.awesome_mask_sharpness,
                ctx.vad_z_threshold,
                ctx.vad_z_slope,
                ctx.vad_snr_gate_db,
                ctx.vad_snr_gate_width,
                ctx.vad_proxy_enabled,
                debug=ctx.debugger,
                debug_ctx=debug_ctx,
            )

        if ctx.use_pipeline_awesome_loss:
            (
                awesome_loss,
                awesome_speech,
                awesome_noise,
                awesome_smooth,
                music_suppression_loss,
                mask_saturation_loss,
                mask,
                proxy_frame,
                speech_ratio,
                music_gate,
                musicness,
                _,  # vocal_gate
                _,  # instrument_gate
                mod_energy,
                energy_boost,
                snr_boost,
            ) = _compute_pipeline_awesome_losses(
                noisy_real,
                noisy_imag,
                clean_real,
                clean_imag,
                spec_out[0],
                spec_out[1],
                snr,
                ctx.vad_band_mask,
                ctx.vad_band_bins,
                ctx.awesome_mask_sharpness,
                ctx.vad_z_threshold,
                ctx.vad_z_slope,
                ctx.vad_snr_gate_db,
                ctx.vad_snr_gate_width,
                ctx.vad_proxy_enabled,
                debug=ctx.debugger,
                debug_ctx=debug_ctx,
            )

        if ctx.use_vad_loss:
            vad_loss, p_ref, p_out, gate = _compute_vad_loss(
                clean_real,
                clean_imag,
                spec_out[0],
                spec_out[1],
                snr,
                ctx.vad_band_mask,
                ctx.vad_band_bins,
                ctx.vad_threshold,
                ctx.vad_margin,
                ctx.vad_snr_gate_db,
                ctx.vad_snr_gate_width,
                ctx.vad_z_threshold,
                ctx.vad_z_slope,
                debug=ctx.debugger,
                debug_ctx=debug_ctx,
            )
            speech_loss = SCALAR_ZERO
            if epoch_vad_speech_loss_weight > 0:
                speech_loss = _compute_speech_band_logmag_loss(
                    clean_real,
                    clean_imag,
                    spec_out[0],
                    spec_out[1],
                    ctx.vad_band_mask,
                    ctx.vad_band_bins,
                    gate,
                    debug=ctx.debugger,
                    debug_ctx=debug_ctx,
                )
        else:
            vad_loss = SCALAR_ZERO
            speech_loss = SCALAR_ZERO
            p_ref = SCALAR_ZERO
            p_out = SCALAR_ZERO
            gate = SCALAR_ZERO

        vad_reg_loss = SCALAR_ZERO
        if ctx.use_vad_train_reg:
            vad_reg_loss, _, _, _, _, _, _ = _compute_vad_reg_loss(
                clean_real,
                clean_imag,
                noisy_real,
                noisy_imag,
                spec_out[0],
                spec_out[1],
                snr,
                ctx.vad_band_mask,
                ctx.vad_band_bins,
                ctx.vad_threshold,
                ctx.vad_margin,
                ctx.vad_z_threshold,
                ctx.vad_z_slope,
                ctx.vad_snr_gate_db,
                ctx.vad_snr_gate_width,
                debug=ctx.debugger,
                debug_ctx=debug_ctx,
            )

        awesome_weight_val = epoch_awesome_loss_weight
        if (ctx.use_awesome_loss or ctx.use_pipeline_awesome_loss) and ctx.awesome_warmup_steps > 0:
            awesome_weight_val = epoch_awesome_loss_weight * min(1.0, global_step / max(ctx.awesome_warmup_steps, 1))

        loss = spec_loss
        if ctx.use_mrstft_loss:
            loss = loss + mrstft_loss
        if ctx.use_awesome_loss or ctx.use_pipeline_awesome_loss:
            loss = loss + awesome_weight_val * awesome_loss
        if ctx.use_vad_loss:
            loss = loss + epoch_vad_loss_weight * vad_loss + epoch_vad_speech_loss_weight * speech_loss

            # Option C: Multi-task VAD head BCE loss (logits path)
            if vad_logits is not None:
                p_ref_expanded = mx.expand_dims(p_ref, axis=-1)
                vad_head_loss = nn.losses.binary_cross_entropy(
                    vad_logits, p_ref_expanded, with_logits=True, reduction="mean"
                )
                loss = loss + epoch_vad_loss_weight * vad_head_loss

        residual = mx.mean((spec_out[0] - clean_real) ** 2 + (spec_out[1] - clean_imag) ** 2)
        residual_by_sample = mx.mean((spec_out[0] - clean_real) ** 2 + (spec_out[1] - clean_imag) ** 2, axis=(1, 2))

        (
            loss_val,
            spec_loss_val,
            mrstft_loss_val,
            vad_loss_val,
            speech_loss_val,
            awesome_loss_val,
            awesome_speech_val,
            awesome_noise_val,
            awesome_smooth_val,
            music_suppression_loss_val,
            mask_saturation_loss_val,
            vad_reg_loss_val,
            residual_val,
        ) = _batch_to_float(
            loss,
            spec_loss,
            mrstft_loss,
            vad_loss,
            speech_loss,
            awesome_loss,
            awesome_speech,
            awesome_noise,
            awesome_smooth,
            music_suppression_loss,
            mask_saturation_loss,
            vad_reg_loss,
            residual,
        )

        valid_loss += loss_val
        valid_spec_loss += spec_loss_val
        valid_mrstft_loss += mrstft_loss_val
        valid_vad_loss += vad_loss_val
        valid_speech_loss += speech_loss_val
        valid_awesome_loss += awesome_loss_val
        valid_awesome_speech += awesome_speech_val
        valid_awesome_noise += awesome_noise_val
        valid_awesome_smooth += awesome_smooth_val
        valid_music_supp_loss += music_suppression_loss_val
        valid_mask_sat_loss += mask_saturation_loss_val
        valid_vad_reg_loss += vad_reg_loss_val
        valid_residual += residual_val
        num_valid_batches += 1

        if ctx.use_vad_loss:
            _p_ref_m = mx.mean(p_ref)
            _p_out_m = mx.mean(p_out)
            _gate_m = mx.mean(mx.where(gate > 0.0, 1.0, 0.0))
            _p_ref_f, _p_out_f, _gate_f = _batch_to_float(_p_ref_m, _p_out_m, _gate_m)
            valid_p_ref += _p_ref_f
            valid_p_out += _p_out_f
            valid_gate_pct += 100.0 * _gate_f

        if ctx.emit_detailed_metrics:
            snr_np = np.asarray(snr, dtype=np.float32).reshape(-1)
            residual_np = np.asarray(residual_by_sample, dtype=np.float32).reshape(-1)
            if ctx.use_vad_loss:
                vad_delta_np = np.asarray(
                    mx.mean(mx.maximum(p_ref - p_out - ctx.vad_margin, 0.0), axis=1),
                    dtype=np.float32,
                )
            else:
                vad_delta_np = np.zeros_like(snr_np, dtype=np.float32)
            if ctx.use_awesome_loss or ctx.use_pipeline_awesome_loss:
                if isinstance(musicness, mx.array):
                    musicness_np = np.asarray(musicness, dtype=np.float32).reshape(-1)
                else:
                    musicness_np = np.zeros_like(snr_np, dtype=np.float32)
                if musicness_np.shape[0] != snr_np.shape[0]:
                    musicness_np = np.full_like(snr_np, float(np.mean(musicness_np)), dtype=np.float32)
            else:
                musicness_np = np.zeros_like(snr_np, dtype=np.float32)

            for i, snr_val in enumerate(snr_np):
                bucket = _snr_bucket_name(float(snr_val))
                metric = bucket_metrics.setdefault(
                    bucket,
                    {
                        "count": 0.0,
                        "residual_sum": 0.0,
                        "vad_delta_sum": 0.0,
                        "musicness_sum": 0.0,
                    },
                )
                metric["count"] += 1.0
                metric["residual_sum"] += float(residual_np[i])
                metric["vad_delta_sum"] += float(vad_delta_np[i])
                metric["musicness_sum"] += float(musicness_np[i])

        if ctx.use_awesome_loss and ctx.emit_detailed_metrics:
            _mask_m = mx.mean(mask)
            _mask_hi = mx.mean(mx.where(mask > 0.8, 1.0, 0.0))
            _mask_lo = mx.mean(mx.where(mask < 0.2, 1.0, 0.0))
            _proxy_m = mx.mean(proxy_frame)
            _sr_m = mx.mean(speech_ratio)
            _mg_m = mx.mean(music_gate)
            _mu_m = mx.mean(musicness)
            _me_m = mx.mean(mod_energy)
            _eb_m = mx.mean(energy_boost)
            _sb_m = mx.mean(snr_boost)
            (
                mask_mean,
                mask_high,
                mask_low,
                proxy_mean,
                speech_ratio_mean,
                music_gate_mean,
                musicness_mean,
                mod_energy_mean,
                energy_boost_mean,
                snr_boost_mean,
            ) = _batch_to_float(
                _mask_m,
                _mask_hi,
                _mask_lo,
                _proxy_m,
                _sr_m,
                _mg_m,
                _mu_m,
                _me_m,
                _eb_m,
                _sb_m,
            )
            mask_high *= 100.0
            mask_low *= 100.0

            valid_mask_mean += mask_mean
            valid_mask_high += mask_high
            valid_mask_low += mask_low
            valid_proxy_mean += proxy_mean
            valid_speech_ratio += speech_ratio_mean
            valid_music_gate += music_gate_mean
            valid_musicness += musicness_mean
            valid_mod_energy += mod_energy_mean
            valid_energy_boost += energy_boost_mean
            valid_snr_boost += snr_boost_mean

        if do_vad_eval and vad_eval_batches_done < ctx.vad_eval_batches:
            if ctx.vad_eval_mode == "proxy":
                p_ref_eval, p_out_eval = _compute_vad_probs(
                    clean_real.astype(mx.float32),
                    clean_imag.astype(mx.float32),
                    spec_out[0].astype(mx.float32),
                    spec_out[1].astype(mx.float32),
                    ctx.vad_band_mask,
                    ctx.vad_band_bins,
                    ctx.vad_z_threshold,
                    ctx.vad_z_slope,
                )
                p_ref_mean, p_out_mean, vad_dec = _compute_vad_eval_metrics(
                    p_ref_eval,
                    p_out_eval,
                    ctx.vad_margin,
                )
                vad_eval_p_ref += float(p_ref_mean)
                vad_eval_p_out += float(p_out_mean)
                vad_eval_delta += float(vad_dec)
                vad_eval_batches_done += 1
            elif ctx.vad_eval_mode == "silero":
                if ctx.silero_vad is None or silero_istft is None:
                    raise RuntimeError("Silero VAD requested but not initialized")
                vad_start = time.perf_counter()
                clean_wav = silero_istft(target_spec, n_fft=ctx.fft_size, hop_length=ctx.hop_size)
                out_wav = silero_istft(spec_out, n_fft=ctx.fft_size, hop_length=ctx.hop_size)
                mx.eval(clean_wav, out_wav)
                clean_np = np.asarray(clean_wav, dtype=np.float32)
                out_np = np.asarray(out_wav, dtype=np.float32)
                p_ref_batch = ctx.silero_vad.mean_probs(clean_np, ctx.sample_rate)
                p_out_batch = ctx.silero_vad.mean_probs(out_np, ctx.sample_rate)
                vad_eval_p_ref += float(np.mean(p_ref_batch))
                vad_eval_p_out += float(np.mean(p_out_batch))
                vad_eval_delta += float(np.mean(np.maximum(p_ref_batch - p_out_batch - ctx.vad_margin, 0.0)))
                vad_eval_batches_done += 1
                vad_eval_clips += int(len(p_ref_batch))
                vad_eval_seconds += time.perf_counter() - vad_start

        if sisdr_fn is not None:
            si_sdr_fn, istft_fn = sisdr_fn
            clean_wav = istft_fn(target_spec, n_fft=ctx.fft_size, hop_length=ctx.hop_size)
            out_wav = istft_fn(spec_out, n_fft=ctx.fft_size, hop_length=ctx.hop_size)
            sisdr_val = float(si_sdr_fn(out_wav, clean_wav))
            if math.isfinite(sisdr_val):
                valid_sisdr += sisdr_val
            else:
                print("⚠️  SI-SDR non-finite; skipping metric for this batch")

        valid_pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            avg=f"{valid_loss / num_valid_batches:.4f}",
        )

        if ctx.max_valid_batches is not None and (batch_idx + 1) >= ctx.max_valid_batches:
            break

    valid_pbar.close()

    if num_valid_batches > 0:
        avg_spec = valid_spec_loss / num_valid_batches
        avg_mrstft = valid_mrstft_loss / num_valid_batches
        avg_vad = valid_vad_loss / num_valid_batches
        avg_speech = valid_speech_loss / num_valid_batches
        avg_awesome = valid_awesome_loss / num_valid_batches
        avg_awesome_speech = valid_awesome_speech / num_valid_batches
        avg_awesome_noise = valid_awesome_noise / num_valid_batches
        avg_awesome_smooth = valid_awesome_smooth / num_valid_batches
        avg_music_supp = valid_music_supp_loss / num_valid_batches
        avg_mask_sat = valid_mask_sat_loss / num_valid_batches
        avg_vad_reg = valid_vad_reg_loss / num_valid_batches
        avg_residual = valid_residual / num_valid_batches
        avg_p_ref = valid_p_ref / num_valid_batches if ctx.use_vad_loss else 0.0
        avg_p_out = valid_p_out / num_valid_batches if ctx.use_vad_loss else 0.0
        avg_gate = valid_gate_pct / num_valid_batches if ctx.use_vad_loss else 0.0
        avg_sisdr = valid_sisdr / num_valid_batches if ctx.eval_sisdr else None
        use_awesome_metrics = ctx.use_awesome_loss or ctx.use_pipeline_awesome_loss
        avg_mask_mean = valid_mask_mean / num_valid_batches if use_awesome_metrics else 0.0
        avg_mask_high = valid_mask_high / num_valid_batches if use_awesome_metrics else 0.0
        avg_mask_low = valid_mask_low / num_valid_batches if use_awesome_metrics else 0.0
        avg_proxy = valid_proxy_mean / num_valid_batches if use_awesome_metrics else 0.0
        avg_speech_ratio = valid_speech_ratio / num_valid_batches if use_awesome_metrics else 0.0
        avg_music_gate = valid_music_gate / num_valid_batches if use_awesome_metrics else 0.0
        avg_musicness = valid_musicness / num_valid_batches if use_awesome_metrics else 0.0
        avg_mod = valid_mod_energy / num_valid_batches if use_awesome_metrics else 0.0
        avg_energy_boost = valid_energy_boost / num_valid_batches if use_awesome_metrics else 0.0
        avg_snr_boost = valid_snr_boost / num_valid_batches if use_awesome_metrics else 0.0
        avg_vad_eval_p_ref = (
            vad_eval_p_ref / vad_eval_batches_done if do_vad_eval and vad_eval_batches_done > 0 else 0.0
        )
        avg_vad_eval_p_out = (
            vad_eval_p_out / vad_eval_batches_done if do_vad_eval and vad_eval_batches_done > 0 else 0.0
        )
        avg_vad_eval_delta = (
            vad_eval_delta / vad_eval_batches_done if do_vad_eval and vad_eval_batches_done > 0 else 0.0
        )
        vad_eval_time = vad_eval_seconds
        vad_eval_clips_total = vad_eval_clips

        if (
            ctx.use_vad_loss
            or ctx.eval_sisdr
            or ctx.use_awesome_loss
            or ctx.use_pipeline_awesome_loss
            or ctx.use_vad_train_reg
            or do_vad_eval
            or ctx.use_mrstft_loss
        ):
            extras = [f"spec={avg_spec:.4f}", f"resid={avg_residual:.4f}"]
            if ctx.use_mrstft_loss:
                extras.append(f"mrstft={avg_mrstft:.4f}")
            if ctx.use_vad_loss:
                extras.extend([f"vad={avg_vad:.4f}", f"speech={avg_speech:.4f}"])
            if use_awesome_metrics:
                extras.extend(
                    [
                        f"awesome={avg_awesome:.4f}",
                        f"aw_s={avg_awesome_speech:.4f}",
                        f"aw_n={avg_awesome_noise:.4f}",
                        f"aw_sm={avg_awesome_smooth:.4f}",
                    ]
                )
            if ctx.use_pipeline_awesome_loss:
                extras.extend(
                    [
                        f"mus_sup={avg_music_supp:.4f}",
                        f"mask_sat={avg_mask_sat:.4f}",
                    ]
                )
            if ctx.use_vad_train_reg:
                extras.append(f"vad_reg={avg_vad_reg:.4f}")
            if ctx.use_vad_loss:
                extras.append(f"p_ref={avg_p_ref:.2f}")
                extras.append(f"p_out={avg_p_out:.2f}")
                extras.append(f"gate={avg_gate:.0f}%")
            if use_awesome_metrics:
                extras.extend(
                    [
                        f"mask={avg_mask_mean:.2f}",
                        f"mask_hi={avg_mask_high:.0f}%",
                        f"mask_lo={avg_mask_low:.0f}%",
                        f"proxy={avg_proxy:.2f}",
                        f"ratio={avg_speech_ratio:.2f}",
                        f"music_gate={avg_music_gate:.2f}",
                        f"music={avg_musicness:.2f}",
                        f"mod={avg_mod:.2f}",
                        f"e_boost={avg_energy_boost:.2f}",
                        f"snr_boost={avg_snr_boost:.2f}",
                    ]
                )
            if do_vad_eval and vad_eval_batches_done > 0:
                extras.append(f"vad_eval_ref={avg_vad_eval_p_ref:.2f}")
                extras.append(f"vad_eval_out={avg_vad_eval_p_out:.2f}")
                extras.append(f"vad_eval_dec={avg_vad_eval_delta:.2f}")
                if ctx.vad_eval_mode == "silero":
                    extras.append(f"vad_eval_s={vad_eval_time:.1f}")
                    extras.append(f"vad_eval_clips={vad_eval_clips_total}")
            if avg_sisdr is not None:
                extras.append(f"si-sdr={avg_sisdr:.2f}dB")
            print(f"{label} metrics: " + " | ".join(extras))

        if bucket_metrics:
            bucket_parts = []
            bucket_summary: dict[str, dict[str, float]] = {}
            for bucket_name in sorted(bucket_metrics.keys()):
                bm = bucket_metrics[bucket_name]
                count = max(bm["count"], 1.0)
                residual_mean = bm["residual_sum"] / count
                vad_delta_mean = bm["vad_delta_sum"] / count
                musicness_mean = bm["musicness_sum"] / count
                bucket_summary[bucket_name] = {
                    "count": float(count),
                    "residual": float(residual_mean),
                    "vad_delta": float(vad_delta_mean),
                    "musicness": float(musicness_mean),
                }
                bucket_parts.append(
                    f"{bucket_name}:n={int(count)} resid={residual_mean:.4f} vadΔ={vad_delta_mean:.4f} mus={musicness_mean:.3f}"
                )
            print(f"{label} buckets: " + " | ".join(bucket_parts))

            ablation_row = {
                "epoch": int(epoch + 1),
                "stage_index": int(active_stage_index),
                "stage_name": active_stage_name,
                "dynamic_loss": ctx.dynamic_loss,
                "train_mode": train_mode,
                "valid_loss": float(valid_loss / max(num_valid_batches, 1)),
                "awesome": {
                    "music_suppression": float(avg_music_supp),
                    "mask_saturation": float(avg_mask_sat),
                },
                "buckets": bucket_summary,
            }
            ablation_path = ctx.ckpt_dir / "ablation_metrics.jsonl"
            try:
                with open(ablation_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ablation_row) + "\n")
            except OSError as exc:
                tqdm.write(f"\u26a0\ufe0f  Failed to write ablation metrics: {exc}")

    return valid_loss / max(num_valid_batches, 1)
