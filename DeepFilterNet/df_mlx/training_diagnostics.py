"""Numeric diagnostic utilities for training.

Provides the ``diagnose_nonfinite`` function and its companion
``DiagnosticContext`` dataclass.  When a non-finite loss is detected during
training, this module performs a layer-by-layer forward-pass analysis to
identify which operation introduced NaN/Inf values.

Key exports:
    - DiagnosticContext: Bundles immutable config needed by the diagnostic pass.
    - diagnose_nonfinite: Layer-by-layer forward analysis for NaN/Inf root-cause.

Relationship to train_dynamic:
    DiagnosticContext is constructed once during train() setup.  diagnose_nonfinite
    is called from the batch loop when a non-finite loss is detected.  Not
    re-exported; imported directly by train_dynamic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import mlx.core as mx
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import mlx.nn as nn

    from df_mlx.training_ops import NumericDebugger


@dataclass
class DiagnosticContext:
    """Bundles all immutable configuration needed by _diagnose_nonfinite.

    This avoids passing 30+ individual parameters and makes the diagnostic
    function callable from outside the train() closure.
    """

    model: nn.Module
    debugger: NumericDebugger | None
    spectral_loss_fn: Callable
    use_mrstft_loss: bool
    mrstft_loss_fn: Any
    mrstft_istft: Any
    fft_size: int
    hop_size: int
    mrstft_target_len: int | None
    gan_loss_fns: Any
    discriminator: Any
    gan_istft: Any
    gan_target_len: int
    feature_match_loss: Any
    gan_fm_weight: float
    clip_gan_scores_fn: Callable
    use_awesome_loss: bool
    use_pipeline_awesome_loss: bool
    vad_band_mask: mx.array
    vad_band_bins: float
    awesome_mask_sharpness: float
    vad_z_threshold: float
    vad_z_slope: float
    vad_snr_gate_db: float
    vad_snr_gate_width: float
    vad_proxy_enabled: bool
    use_vad_loss: bool
    vad_threshold: float
    vad_margin: float
    vad_speech_loss_weight: float
    use_vad_train_reg: bool


def diagnose_nonfinite(
    ctx: DiagnosticContext,
    noisy_real: mx.array,
    noisy_imag: mx.array,
    feat_erb: mx.array,
    feat_spec: mx.array,
    clean_real: mx.array,
    clean_imag: mx.array,
    snr: mx.array,
    debug_ctx: dict[str, Any],
    *,
    gan_active: bool = False,
) -> None:
    """Run a diagnostic forward pass with detailed finite checks.

    Uses a non-fail-fast debugger so all components are checked and
    logged even when multiple contain non-finite values.
    """
    if ctx.debugger is None:
        return

    from dataclasses import replace as _dc_replace

    from df_mlx.training_losses import (
        _compute_awesome_losses,
        _compute_pipeline_awesome_losses,
        _compute_speech_band_logmag_loss,
        _compute_vad_loss,
        _compute_vad_reg_loss,
    )
    from df_mlx.training_ops import NumericDebugger
    from df_mlx.training_waveform import compute_mrstft_loss, specs_to_wavs

    diag_cfg = _dc_replace(ctx.debugger.config, fail_fast=False)
    diag = NumericDebugger(diag_cfg)
    tqdm.write("  [diagnose] Running non-finite diagnostic pass...")
    findings: list[str] = []

    def _diag_check(name: str, tensor: mx.array) -> None:
        if not diag.check(name, tensor, debug_ctx):
            findings.append(name)

    out = ctx.model((noisy_real, noisy_imag), feat_erb, feat_spec, return_vad=True)
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], tuple):
        spec_out, vad_logits = out
    else:
        spec_out = out
        vad_logits = None

    _diag_check("model.out_real", spec_out[0])
    _diag_check("model.out_imag", spec_out[1])
    if vad_logits is not None:
        _diag_check("model.vad_logits", vad_logits)

    spec_loss = ctx.spectral_loss_fn(spec_out, (clean_real, clean_imag))
    _diag_check("spec_loss", spec_loss)

    if ctx.use_mrstft_loss and ctx.mrstft_loss_fn is not None and ctx.mrstft_istft is not None:
        mrstft_loss = compute_mrstft_loss(
            spec_out,
            (clean_real, clean_imag),
            istft_fn=ctx.mrstft_istft,
            loss_fn=ctx.mrstft_loss_fn,
            n_fft=ctx.fft_size,
            hop_length=ctx.hop_size,
            target_len=ctx.mrstft_target_len,
            force_fp32=True,
        )
        _diag_check("mrstft_loss", mrstft_loss)

    if gan_active and ctx.gan_loss_fns is not None and ctx.discriminator is not None and ctx.gan_istft is not None:
        out_wav, clean_wav = specs_to_wavs(
            spec_out,
            (clean_real, clean_imag),
            istft_fn=ctx.gan_istft,
            n_fft=ctx.fft_size,
            hop_length=ctx.hop_size,
            target_len=ctx.gan_target_len,
            force_fp32=True,
        )
        gen_loss_fn, _ = ctx.gan_loss_fns
        disc_fake, fake_feats = ctx.discriminator(out_wav)
        disc_real, real_feats = ctx.discriminator(clean_wav)
        disc_fake = ctx.clip_gan_scores_fn(disc_fake)
        gan_g_loss = gen_loss_fn(disc_fake)
        _diag_check("gan_g_loss", gan_g_loss)
        if ctx.feature_match_loss is not None and ctx.gan_fm_weight > 0:
            fm_loss = ctx.feature_match_loss(real_feats, fake_feats)
            _diag_check("gan_fm_loss", fm_loss)

    if ctx.use_awesome_loss:
        _compute_awesome_losses(
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
            debug=diag,
            debug_ctx=debug_ctx,
        )

    if ctx.use_pipeline_awesome_loss:
        _compute_pipeline_awesome_losses(
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
            debug=diag,
            debug_ctx=debug_ctx,
        )

    if ctx.use_vad_loss:
        _compute_vad_loss(
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
            debug=diag,
            debug_ctx=debug_ctx,
        )
        if ctx.vad_speech_loss_weight > 0:
            gate = mx.ones((clean_real.shape[0], clean_real.shape[1]))
            _compute_speech_band_logmag_loss(
                clean_real,
                clean_imag,
                spec_out[0],
                spec_out[1],
                ctx.vad_band_mask,
                ctx.vad_band_bins,
                gate,
                debug=diag,
                debug_ctx=debug_ctx,
            )

    if ctx.use_vad_train_reg:
        _compute_vad_reg_loss(
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
            debug=diag,
            debug_ctx=debug_ctx,
        )

    if findings:
        tqdm.write(f"  [diagnose] Non-finite in: {', '.join(findings)}")
    else:
        tqdm.write("  [diagnose] All individual components finite — NaN likely in backward pass")
