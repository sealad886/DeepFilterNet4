"""Sync-window metric collection and progress-bar formatting.

At each sync boundary (every ``eval_frequency`` batches) the training loop
needs to decompose the composite loss into its individual components, accumulate
per-epoch totals, and refresh the tqdm progress bar.  This module encapsulates
that logic so the batch loop in ``train()`` remains compact.

Key exports:
    - create_epoch_accums: Build the dict of zero-initialised epoch accumulators.
    - compute_epoch_averages: Divide accumulated sums by sync count for logging.
    - collect_sync_metrics: Decompose loss and accumulate at a sync boundary.
    - update_progress_bar: Format accumulated values into tqdm postfix dict.

Relationship to train_dynamic:
    Called every ``eval_frequency`` batches inside the inner batch loop of
    train().  Not included in the backward-compat re-export block; imported
    directly by train_dynamic.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

from df_mlx.training_helpers import SCALAR_ZERO
from df_mlx.training_helpers import clip_gan_scores
from df_mlx.training_losses import (
    _AWESOME_MASK_LOGIT_CLAMP,
    _EPS,
    _VAD_LOGIT_CLAMP,
    _compute_awesome_losses,
    _compute_pipeline_awesome_losses,
    _compute_speech_band_logmag_loss,
    _compute_vad_loss,
    _compute_vad_reg_loss,
    _log1p_mag,
)
from df_mlx.training_ops import _batch_to_float
from df_mlx.training_waveform import (
    _gan_waveform_view,
    compute_mrstft_loss,
    specs_to_wavs,
)


def create_epoch_accums() -> dict[str, Any]:
    """Return a fresh accumulators dict for one training epoch.

    Keys mirror the ``train_*`` local variables that were previously scattered
    across the epoch-setup block in ``train()``.
    """
    return {
        "spec_loss": 0.0,
        "mrstft_loss": 0.0,
        "gan_g_loss": 0.0,
        "gan_fm_loss": 0.0,
        "vad_loss": 0.0,
        "speech_loss": 0.0,
        "awesome_loss": 0.0,
        "awesome_speech": 0.0,
        "awesome_noise": 0.0,
        "awesome_smooth": 0.0,
        "music_supp_loss": 0.0,
        "mask_sat_loss": 0.0,
        "vad_reg_loss": 0.0,
        "p_ref": 0.0,
        "p_out": 0.0,
        "gate_pct": 0.0,
        "mask_mean": 0.0,
        "mask_high": 0.0,
        "mask_low": 0.0,
        "proxy_mean": 0.0,
        "speech_ratio": 0.0,
        "music_gate": 0.0,
        "musicness": 0.0,
        "mod_energy": 0.0,
        "energy_boost": 0.0,
        "snr_boost": 0.0,
        "num_vad_logs": 0,
        "num_awesome_logs": 0,
        # Debug accumulators
        "vad_clip_ref": 0.0,
        "vad_clip_out": 0.0,
        "mask_logit_min": float("inf"),
        "mask_logit_max": float("-inf"),
        "mask_clip_rate": 0.0,
        "eps_clean_rate": 0.0,
        "eps_noise_rate": 0.0,
        "num_debug_logs": 0,
    }


def compute_epoch_averages(
    accums: dict[str, Any],
    *,
    train_loss: float,
    num_train_batches: int,
    train_gan_d_loss: float,
    train_gan_d_updates: int,
) -> dict[str, float]:
    """Compute per-epoch average metrics from raw accumulators.

    Returns a dict of human-readable metric averages suitable for
    ``print_epoch_summary`` and logging.
    """
    _n = max(num_train_batches, 1)
    _n_d = max(train_gan_d_updates, 1)
    _n_v = max(accums["num_vad_logs"], 1)
    _n_a = max(accums["num_awesome_logs"], 1)
    return {
        "loss": train_loss / _n,
        "spec_loss": accums["spec_loss"] / _n,
        "mrstft_loss": accums["mrstft_loss"] / _n,
        "gan_g_loss": accums["gan_g_loss"] / _n,
        "gan_fm_loss": accums["gan_fm_loss"] / _n,
        "gan_d_loss": train_gan_d_loss / _n_d,
        "vad_loss": accums["vad_loss"] / _n,
        "speech_loss": accums["speech_loss"] / _n,
        "awesome_loss": accums["awesome_loss"] / _n,
        "awesome_speech": accums["awesome_speech"] / _n,
        "awesome_noise": accums["awesome_noise"] / _n,
        "awesome_smooth": accums["awesome_smooth"] / _n,
        "music_supp": accums["music_supp_loss"] / _n,
        "mask_sat": accums["mask_sat_loss"] / _n,
        "vad_reg_loss": accums["vad_reg_loss"] / _n,
        "p_ref": accums["p_ref"] / _n_v,
        "p_out": accums["p_out"] / _n_v,
        "gate": accums["gate_pct"] / _n_v,
        "mask_mean": accums["mask_mean"] / _n_a,
        "mask_high": accums["mask_high"] / _n_a,
        "mask_low": accums["mask_low"] / _n_a,
        "proxy": accums["proxy_mean"] / _n_a,
        "speech_ratio": accums["speech_ratio"] / _n_a,
        "music_gate": accums["music_gate"] / _n_a,
        "musicness": accums["musicness"] / _n_a,
        "mod": accums["mod_energy"] / _n_a,
        "energy_boost": accums["energy_boost"] / _n_a,
        "snr_boost": accums["snr_boost"] / _n_a,
    }


def collect_sync_metrics(
    *,
    # Batch data
    noisy_real: mx.array,
    noisy_imag: mx.array,
    clean_real: mx.array,
    clean_imag: mx.array,
    snr: mx.array,
    # Model and output
    model: Any,
    feat_erb: mx.array,
    feat_spec: mx.array,
    pred_spec_for_logging: tuple[mx.array, mx.array] | None,
    # Loss/sync state
    loss_val: float,
    loss_was_nonfinite: bool,
    epoch_eval_frequency: int,
    # Config flags
    use_mrstft_loss: bool,
    use_vad_loss: bool,
    use_awesome_loss: bool,
    use_pipeline_awesome_loss: bool,
    use_vad_train_reg: bool,
    use_fp16: bool,
    gan_active: bool,
    emit_detailed_metrics: bool,
    apply_vad_reg: bool,
    debug_numerics: bool,
    speech_weight: float,
    # Loss functions and objects
    spectral_loss_fn: Any,
    mrstft_loss_fn: Any | None,
    mrstft_istft: Any | None,
    mrstft_target_len: int | None,
    discriminator: Any | None,
    feature_match_loss: Any | None,
    gan_loss_fns: tuple | None,
    gan_istft: Any | None,
    gan_fm_weight: float,
    gan_disc_max_samples: int | None,
    gan_target_len: int,
    # Config values for computations
    config_fft_size: int,
    config_hop_size: int,
    config_sample_rate: int,
    # VAD params
    vad_band_mask: mx.array,
    vad_band_bins: float,
    vad_threshold: float,
    vad_margin: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    # Awesome params
    awesome_mask_sharpness: float,
    vad_proxy_enabled: bool,
    # Debug
    debugger: Any | None,
    debug_ctx: dict[str, Any],
    # Accumulators (mutated in-place)
    accums: dict[str, Any],
) -> dict[str, float]:
    """Collect detailed metrics at a sync boundary.

    Mutates *accums* in-place (epoch-level totals) and returns a dict of
    per-step display values suitable for ``update_progress_bar``.
    """
    # ------------------------------------------------------------------
    # Defaults for display values
    # ------------------------------------------------------------------
    spec_loss_val = loss_val
    mrstft_loss_val = 0.0
    gan_g_loss_val = 0.0
    gan_fm_loss_val = 0.0
    vad_loss_val = 0.0
    speech_loss_val = 0.0
    p_ref_mean = 0.0
    p_out_mean = 0.0
    gate_pct = 0.0
    awesome_loss_val = 0.0
    awesome_speech_val = 0.0
    awesome_noise_val = 0.0
    awesome_smooth_val = 0.0
    mask_mean = 0.0
    mask_high = 0.0
    mask_low = 0.0
    proxy_mean = 0.0
    speech_ratio_mean = 0.0
    music_gate_mean = 0.0
    musicness_mean = 0.0
    mod_energy_mean = 0.0
    energy_boost_mean = 0.0
    snr_boost_mean = 0.0
    vad_reg_loss_val = 0.0

    # ------------------------------------------------------------------
    # Compute model output for any metric block that needs it.
    # ------------------------------------------------------------------
    needs_model_out = not loss_was_nonfinite and (
        use_vad_loss
        or use_awesome_loss
        or use_pipeline_awesome_loss
        or use_vad_train_reg
        or (emit_detailed_metrics and (use_mrstft_loss or gan_active))
    )
    out: tuple[mx.array, mx.array] | None = None
    spec_out: tuple[mx.array, mx.array] | None = None
    if needs_model_out:
        out = pred_spec_for_logging
        if out is None:
            raw = model((noisy_real, noisy_imag), feat_erb, feat_spec, return_vad=True)
            if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], tuple):
                spec_out, _vad_logits = raw
            else:
                spec_out = raw
            out = (
                mx.stop_gradient(spec_out[0]),
                mx.stop_gradient(spec_out[1]),
            )
        else:
            spec_out = out
        if debugger is not None:
            debugger.check("model.out_real", spec_out[0], debug_ctx)
            debugger.check("model.out_imag", spec_out[1], debug_ctx)

    # ------------------------------------------------------------------
    # Detailed spectral / MRSTFT / GAN metrics (only in detailed sync mode)
    # Batched into a single _batch_to_float sync to reduce GPU stalls.
    # ------------------------------------------------------------------
    if emit_detailed_metrics and needs_model_out:
        # Collect lazy MLX arrays, then extract all floats in one sync.
        _detail_arrays: list[mx.array] = []
        _detail_keys: list[str] = []

        spec_loss = spectral_loss_fn(spec_out, (clean_real, clean_imag))
        _detail_arrays.append(spec_loss)
        _detail_keys.append("spec")

        mrstft_arr: mx.array | None = None
        if use_mrstft_loss and mrstft_loss_fn is not None and mrstft_istft is not None:
            mrstft_arr = compute_mrstft_loss(
                spec_out,
                (clean_real, clean_imag),
                istft_fn=mrstft_istft,
                loss_fn=mrstft_loss_fn,
                n_fft=config_fft_size,
                hop_length=config_hop_size,
                target_len=mrstft_target_len,
                force_fp32=True,
            )
            _detail_arrays.append(mrstft_arr)
            _detail_keys.append("mrstft")

        gan_g_arr: mx.array | None = None
        gan_fm_arr: mx.array | None = None
        if gan_active and gan_loss_fns is not None and discriminator is not None and gan_istft is not None:
            out_wav, clean_wav = specs_to_wavs(
                spec_out,
                (clean_real, clean_imag),
                istft_fn=gan_istft,
                n_fft=config_fft_size,
                hop_length=config_hop_size,
                target_len=gan_target_len,
                force_fp32=use_mrstft_loss,
            )
            out_wav = _gan_waveform_view(out_wav, use_fp16=bool(use_fp16))
            clean_wav = _gan_waveform_view(clean_wav, use_fp16=bool(use_fp16))
            gen_loss_fn, _ = gan_loss_fns
            disc_fake, fake_feats = discriminator(out_wav)
            disc_real, real_feats = discriminator(clean_wav)
            disc_fake = clip_gan_scores(disc_fake)
            gan_g_arr = gen_loss_fn(disc_fake)
            _detail_arrays.append(gan_g_arr)
            _detail_keys.append("gan_g")
            if feature_match_loss is not None and gan_fm_weight > 0:
                gan_fm_arr = feature_match_loss(real_feats, fake_feats)
                _detail_arrays.append(gan_fm_arr)
                _detail_keys.append("gan_fm")

        # Single sync barrier for all detailed metrics
        _detail_vals = _batch_to_float(*_detail_arrays)
        _detail_map = dict(zip(_detail_keys, _detail_vals))

        spec_loss_val = _detail_map["spec"]
        accums["spec_loss"] += spec_loss_val * epoch_eval_frequency
        if "mrstft" in _detail_map:
            mrstft_loss_val = _detail_map["mrstft"]
            accums["mrstft_loss"] += mrstft_loss_val * epoch_eval_frequency
        if "gan_g" in _detail_map:
            gan_g_loss_val = _detail_map["gan_g"]
            accums["gan_g_loss"] += gan_g_loss_val * epoch_eval_frequency
        if "gan_fm" in _detail_map:
            gan_fm_loss_val = _detail_map["gan_fm"]
            accums["gan_fm_loss"] += gan_fm_loss_val * epoch_eval_frequency

    # ------------------------------------------------------------------
    # VAD loss metrics
    # ------------------------------------------------------------------
    if use_vad_loss and needs_model_out:
        vad_loss, p_ref, p_out, gate = _compute_vad_loss(
            clean_real,
            clean_imag,
            spec_out[0],
            spec_out[1],
            snr,
            vad_band_mask,
            vad_band_bins,
            vad_threshold,
            vad_margin,
            vad_snr_gate_db,
            vad_snr_gate_width,
            vad_z_threshold,
            vad_z_slope,
            debug=debugger,
            debug_ctx=debug_ctx,
        )
        speech_loss = SCALAR_ZERO
        if speech_weight > 0:
            speech_loss = _compute_speech_band_logmag_loss(
                clean_real,
                clean_imag,
                spec_out[0],
                spec_out[1],
                vad_band_mask,
                vad_band_bins,
                gate,
                debug=debugger,
                debug_ctx=debug_ctx,
            )
        _p_ref_m = mx.mean(p_ref)
        _p_out_m = mx.mean(p_out)
        _gate_m = mx.mean(mx.where(gate > 0.0, 1.0, 0.0))
        (
            vad_loss_val,
            speech_loss_val,
            p_ref_mean,
            p_out_mean,
            _gate_f,
        ) = _batch_to_float(vad_loss, speech_loss, _p_ref_m, _p_out_m, _gate_m)
        gate_pct = 100.0 * _gate_f

        accums["vad_loss"] += vad_loss_val * epoch_eval_frequency
        accums["speech_loss"] += speech_loss_val * epoch_eval_frequency
        accums["p_ref"] += p_ref_mean
        accums["p_out"] += p_out_mean
        accums["gate_pct"] += gate_pct
        accums["num_vad_logs"] += 1

        if debug_numerics:
            clean_power_dbg = clean_real.astype(mx.float32) ** 2 + clean_imag.astype(mx.float32) ** 2
            out_power_dbg = spec_out[0].astype(mx.float32) ** 2 + spec_out[1].astype(mx.float32) ** 2
            clean_band_dbg = mx.sum(clean_power_dbg * vad_band_mask, axis=-1) / (vad_band_bins + _EPS)
            out_band_dbg = mx.sum(out_power_dbg * vad_band_mask, axis=-1) / (vad_band_bins + _EPS)
            log_clean_dbg = mx.log10(clean_band_dbg + _EPS)
            mu_dbg = mx.mean(log_clean_dbg, axis=1, keepdims=True)
            sigma_dbg = mx.sqrt(mx.mean((log_clean_dbg - mu_dbg) ** 2, axis=1, keepdims=True) + _EPS)
            z_ref_dbg = (log_clean_dbg - mu_dbg) / (sigma_dbg + _EPS)
            z_out_dbg = (mx.log10(out_band_dbg + _EPS) - mu_dbg) / (sigma_dbg + _EPS)
            # Batched sync: 2 float extractions in one barrier
            _clip_ref_arr = mx.mean(mx.where(mx.abs(z_ref_dbg) > _VAD_LOGIT_CLAMP, 1.0, 0.0))
            _clip_out_arr = mx.mean(mx.where(mx.abs(z_out_dbg) > _VAD_LOGIT_CLAMP, 1.0, 0.0))
            _cr, _co = _batch_to_float(_clip_ref_arr, _clip_out_arr)
            clip_ref = 100.0 * _cr
            clip_out = 100.0 * _co
            accums["vad_clip_ref"] += clip_ref
            accums["vad_clip_out"] += clip_out

    # ------------------------------------------------------------------
    # Awesome loss metrics
    # ------------------------------------------------------------------
    if use_awesome_loss and needs_model_out:
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
            vad_band_mask,
            vad_band_bins,
            awesome_mask_sharpness,
            vad_z_threshold,
            vad_z_slope,
            vad_snr_gate_db,
            vad_snr_gate_width,
            vad_proxy_enabled,
            debug=debugger,
            debug_ctx=debug_ctx,
        )
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
            awesome_loss_val,
            awesome_speech_val,
            awesome_noise_val,
            awesome_smooth_val,
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
            awesome_loss,
            awesome_speech,
            awesome_noise,
            awesome_smooth,
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

        accums["awesome_loss"] += awesome_loss_val * epoch_eval_frequency
        accums["awesome_speech"] += awesome_speech_val * epoch_eval_frequency
        accums["awesome_noise"] += awesome_noise_val * epoch_eval_frequency
        accums["awesome_smooth"] += awesome_smooth_val * epoch_eval_frequency
        accums["mask_mean"] += mask_mean
        accums["mask_high"] += mask_high
        accums["mask_low"] += mask_low
        accums["proxy_mean"] += proxy_mean
        accums["speech_ratio"] += speech_ratio_mean
        accums["music_gate"] += music_gate_mean
        accums["musicness"] += musicness_mean
        accums["mod_energy"] += mod_energy_mean
        accums["energy_boost"] += energy_boost_mean
        accums["snr_boost"] += snr_boost_mean
        accums["num_awesome_logs"] += 1

        if debug_numerics:
            clean_power_dbg = clean_real.astype(mx.float32) ** 2 + clean_imag.astype(mx.float32) ** 2
            noise_real_dbg = noisy_real.astype(mx.float32) - clean_real.astype(mx.float32)
            noise_imag_dbg = noisy_imag.astype(mx.float32) - clean_imag.astype(mx.float32)
            noise_power_dbg = noise_real_dbg**2 + noise_imag_dbg**2
            clean_band_dbg = mx.sum(clean_power_dbg * vad_band_mask, axis=-1) / (vad_band_bins + _EPS)
            noise_band_dbg = mx.sum(noise_power_dbg * vad_band_mask, axis=-1) / (vad_band_bins + _EPS)
            mask_logits_raw = awesome_mask_sharpness * (
                _log1p_mag(clean_real, clean_imag) - _log1p_mag(noise_real_dbg, noise_imag_dbg)
            )
            # Batched sync: 5 float extractions in one barrier
            _ml_min = mx.min(mask_logits_raw)
            _ml_max = mx.max(mask_logits_raw)
            _mc_rate = mx.mean(mx.where(mx.abs(mask_logits_raw) > _AWESOME_MASK_LOGIT_CLAMP, 1.0, 0.0))
            _ce_rate = mx.mean(mx.where(clean_band_dbg <= _EPS, 1.0, 0.0))
            _ne_rate = mx.mean(mx.where(noise_band_dbg <= _EPS, 1.0, 0.0))
            _ml_min_f, _ml_max_f, _mc_f, _ce_f, _ne_f = _batch_to_float(_ml_min, _ml_max, _mc_rate, _ce_rate, _ne_rate)
            mask_logit_min = _ml_min_f
            mask_logit_max = _ml_max_f
            mask_clip_rate = 100.0 * _mc_f
            clean_eps_rate = 100.0 * _ce_f
            noise_eps_rate = 100.0 * _ne_f
            accums["mask_logit_min"] = min(accums["mask_logit_min"], mask_logit_min)
            accums["mask_logit_max"] = max(accums["mask_logit_max"], mask_logit_max)
            accums["mask_clip_rate"] += mask_clip_rate
            accums["eps_clean_rate"] += clean_eps_rate
            accums["eps_noise_rate"] += noise_eps_rate
            accums["num_debug_logs"] += 1

    # ------------------------------------------------------------------
    # Pipeline awesome loss metrics
    # ------------------------------------------------------------------
    if use_pipeline_awesome_loss and needs_model_out:
        (
            awesome_loss,
            awesome_speech,
            awesome_noise,
            awesome_smooth,
            music_supp_loss,
            mask_sat_loss,
            mask,
            proxy_frame,
            speech_ratio,
            music_gate,
            musicness,
            vocal_gate,
            instrument_gate,
            mod_energy,
            energy_boost,
            snr_boost,
        ) = _compute_pipeline_awesome_losses(
            noisy_real,
            noisy_imag,
            clean_real,
            clean_imag,
            out[0],
            out[1],
            snr,
            vad_band_mask,
            vad_band_bins,
            awesome_mask_sharpness,
            vad_z_threshold,
            vad_z_slope,
            vad_snr_gate_db,
            vad_snr_gate_width,
            vad_proxy_enabled,
            debug=debugger,
            debug_ctx=debug_ctx,
        )
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
            awesome_loss_val,
            awesome_speech_val,
            awesome_noise_val,
            awesome_smooth_val,
            music_supp_loss_val,
            mask_sat_loss_val,
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
            awesome_loss,
            awesome_speech,
            awesome_noise,
            awesome_smooth,
            music_supp_loss,
            mask_sat_loss,
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

        accums["awesome_loss"] += awesome_loss_val * epoch_eval_frequency
        accums["awesome_speech"] += awesome_speech_val * epoch_eval_frequency
        accums["awesome_noise"] += awesome_noise_val * epoch_eval_frequency
        accums["awesome_smooth"] += awesome_smooth_val * epoch_eval_frequency
        accums["music_supp_loss"] += music_supp_loss_val * epoch_eval_frequency
        accums["mask_sat_loss"] += mask_sat_loss_val * epoch_eval_frequency
        accums["mask_mean"] += mask_mean
        accums["mask_high"] += mask_high
        accums["mask_low"] += mask_low
        accums["proxy_mean"] += proxy_mean
        accums["speech_ratio"] += speech_ratio_mean
        accums["music_gate"] += music_gate_mean
        accums["musicness"] += musicness_mean
        accums["mod_energy"] += mod_energy_mean
        accums["energy_boost"] += energy_boost_mean
        accums["snr_boost"] += snr_boost_mean
        accums["num_awesome_logs"] += 1

    # ------------------------------------------------------------------
    # VAD train regularization metrics
    # ------------------------------------------------------------------
    if use_vad_train_reg and apply_vad_reg and needs_model_out:
        vad_reg_loss, vad_dec, gate, _, _, _, _ = _compute_vad_reg_loss(
            clean_real,
            clean_imag,
            noisy_real,
            noisy_imag,
            out[0],
            out[1],
            snr,
            vad_band_mask,
            vad_band_bins,
            vad_threshold,
            vad_margin,
            vad_z_threshold,
            vad_z_slope,
            vad_snr_gate_db,
            vad_snr_gate_width,
            debug=debugger,
            debug_ctx=debug_ctx,
        )
        vad_reg_loss_val = float(vad_reg_loss)
        accums["vad_reg_loss"] += vad_reg_loss_val * epoch_eval_frequency

    # ------------------------------------------------------------------
    # Build display dict
    # ------------------------------------------------------------------
    return {
        "spec_loss_val": spec_loss_val,
        "mrstft_loss_val": mrstft_loss_val,
        "gan_g_loss_val": gan_g_loss_val,
        "gan_fm_loss_val": gan_fm_loss_val,
        "vad_loss_val": vad_loss_val,
        "speech_loss_val": speech_loss_val,
        "p_ref_mean": p_ref_mean,
        "p_out_mean": p_out_mean,
        "gate_pct": gate_pct,
        "awesome_loss_val": awesome_loss_val,
        "awesome_speech_val": awesome_speech_val,
        "awesome_noise_val": awesome_noise_val,
        "awesome_smooth_val": awesome_smooth_val,
        "mask_mean": mask_mean,
        "mask_high": mask_high,
        "mask_low": mask_low,
        "proxy_mean": proxy_mean,
        "speech_ratio_mean": speech_ratio_mean,
        "music_gate_mean": music_gate_mean,
        "musicness_mean": musicness_mean,
        "mod_energy_mean": mod_energy_mean,
        "energy_boost_mean": energy_boost_mean,
        "snr_boost_mean": snr_boost_mean,
        "vad_reg_loss_val": vad_reg_loss_val,
    }


def update_progress_bar(
    train_pbar: Any,
    display: dict[str, float],
    *,
    loss_val: float,
    train_loss: float,
    num_train_batches: int,
    gan_d_loss_val: float,
    lr: float,
    grad_norm: float,
    samples_per_sec: float,
    data_time: float,
    fwd_time: float,
    global_step: int,
    verbose: bool,
    use_mrstft_loss: bool,
    use_vad_loss: bool,
    use_awesome_loss: bool,
    use_pipeline_awesome_loss: bool,
    use_vad_train_reg: bool,
    gan_active: bool,
) -> None:
    """Update the tqdm progress bar with current metrics."""
    spec_loss_val = display["spec_loss_val"]
    mrstft_loss_val = display["mrstft_loss_val"]
    gan_g_loss_val = display["gan_g_loss_val"]
    gan_fm_loss_val = display["gan_fm_loss_val"]
    vad_loss_val = display["vad_loss_val"]
    speech_loss_val = display["speech_loss_val"]
    awesome_loss_val = display["awesome_loss_val"]
    mask_mean = display["mask_mean"]
    p_ref_mean = display["p_ref_mean"]
    p_out_mean = display["p_out_mean"]
    gate_pct = display["gate_pct"]
    vad_reg_loss_val = display["vad_reg_loss_val"]

    if verbose:
        train_pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            spec=(
                f"{spec_loss_val:.4f}"
                if (use_vad_loss or use_awesome_loss or use_pipeline_awesome_loss or use_vad_train_reg)
                else f"{loss_val:.4f}"
            ),
            mrstft=f"{mrstft_loss_val:.4f}" if use_mrstft_loss else "0.0000",
            gan_g=f"{gan_g_loss_val:.4f}" if gan_active else "0.0000",
            gan_d=f"{gan_d_loss_val:.4f}" if gan_active else "0.0000",
            fm=f"{gan_fm_loss_val:.4f}" if gan_active else "0.0000",
            vad=f"{vad_loss_val:.4f}" if use_vad_loss else "0.0000",
            speech=f"{speech_loss_val:.4f}" if use_vad_loss else "0.0000",
            awesome=(f"{awesome_loss_val:.4f}" if (use_awesome_loss or use_pipeline_awesome_loss) else "0.0000"),
            mask=(f"{mask_mean:.2f}" if (use_awesome_loss or use_pipeline_awesome_loss) else "0.00"),
            lr=f"{lr:.1e}",
            data=f"{data_time * 1000:.0f}ms",
            fwd=f"{fwd_time * 1000:.0f}ms",
            spd=f"{samples_per_sec:.0f}/s",
            gstep=global_step,
        )
    else:
        grad_display = f"{grad_norm:.2f}" if math.isfinite(grad_norm) else "n/a"
        train_pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            avg=f"{train_loss / num_train_batches:.4f}",
            gan_g=f"{gan_g_loss_val:.4f}" if gan_active else "0.0000",
            gan_d=f"{gan_d_loss_val:.4f}" if gan_active else "0.0000",
            fm=f"{gan_fm_loss_val:.4f}" if gan_active else "0.0000",
            vad=f"{vad_loss_val:.4f}" if use_vad_loss else "0.0000",
            speech=f"{speech_loss_val:.4f}" if use_vad_loss else "0.0000",
            awesome=(f"{awesome_loss_val:.4f}" if (use_awesome_loss or use_pipeline_awesome_loss) else "0.0000"),
            mask=(f"{mask_mean:.2f}" if (use_awesome_loss or use_pipeline_awesome_loss) else "0.00"),
            p_ref=f"{p_ref_mean:.2f}" if use_vad_loss else "0.00",
            p_out=f"{p_out_mean:.2f}" if use_vad_loss else "0.00",
            gate=f"{gate_pct:.0f}%" if use_vad_loss else "0%",
            vad_reg=f"{vad_reg_loss_val:.4f}" if use_vad_train_reg else "0.0000",
            lr=f"{lr:.1e}",
            grad=grad_display,
            spd=f"{samples_per_sec:.0f}/s",
            gstep=global_step,
        )
