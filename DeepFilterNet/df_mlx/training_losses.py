"""Loss computation functions for MLX DeepFilterNet4 training.

Contains all auxiliary loss helpers consumed by the loss closures inside
``train()``: VAD probability estimation, awesome (speech-preserving
contrastive) loss, pipeline awesome loss (improved speech preservation and
music suppression), speech-band analysis, and SNR bucketing.  Constants
governing loss behaviour thresholds are also defined here.

Key exports:
    - _compute_vad_probs / _compute_vad_loss / _compute_vad_reg_loss:
      Voice-activity detection probability and loss computation.
    - _compute_awesome_losses: Speech-preserving contrastive loss.
    - _compute_pipeline_awesome_losses: Improved pipeline-aware awesome loss.
    - _compute_speech_band_logmag_loss: Speech-band log-magnitude loss.
    - _compute_vad_eval_metrics: VAD metrics for validation.
    - _snr_bucket_name: Bin an SNR value into a named bucket.
    - _log1p_mag: Numerically stable log1p magnitude helper.
    - Various threshold/scale constants (_AWESOME_*, _PIPELINE_*, _VAD_*).

Relationship to train_dynamic:
    All public symbols (functions and constants) are re-exported via
    train_dynamic.py for backward compatibility.  Functions are called inside
    the ``loss_fn`` / ``loss_fn_gan`` closures and by training_metrics and
    training_validation during sync-window and validation metric collection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

from df_mlx.model import ContrastiveFrameProjector

if TYPE_CHECKING:
    from df_mlx.training_ops import NumericDebugger

__all__ = [
    "ContrastiveFrameProjector",
    "_AWESOME_ENERGY_BOOST_DB",
    "_AWESOME_ENERGY_BOOST_WIDTH",
    "_AWESOME_LOW_ENERGY_WEIGHT",
    "_AWESOME_LOW_SNR_WEIGHT",
    "_AWESOME_MASK_LOGIT_CLAMP",
    "_AWESOME_MOD_THRESHOLD",
    "_AWESOME_MOD_WIDTH",
    "_AWESOME_MUSIC_FLUX_THR",
    "_AWESOME_MUSIC_FLUX_WIDTH",
    "_AWESOME_MUSICNESS_THR",
    "_AWESOME_MUSICNESS_WIDTH",
    "_AWESOME_PROXY_RATIO_FLOOR",
    "_AWESOME_PROXY_RATIO_SCALE",
    "_AWESOME_SMOOTH_WEIGHT",
    "_EPS",
    "_PIPELINE_ARTIFACT_SMOOTH_WEIGHT",
    "_PIPELINE_LOW_ENERGY_ADDITIVE",
    "_PIPELINE_LOW_SNR_ADDITIVE",
    "_PIPELINE_MIN_MASK_FLOOR",
    "_PIPELINE_MUSIC_SUPPRESSION_WEIGHT",
    "_PIPELINE_PITCH_STABILITY_THR",
    "_PIPELINE_PROXY_FLOOR",
    "_PIPELINE_VOCAL_HARMONIC_THR",
    "_VAD_LOGIT_CLAMP",
    "_build_speech_band_mask",
    "_compute_awesome_losses",
    "_compute_contrastive_awesome_losses",
    "_compute_harmonic_ratio",
    "_compute_improved_musicness",
    "_compute_musicness",
    "_compute_pipeline_awesome_losses",
    "_compute_pitch_stability",
    "_compute_proxy_gates",
    "_compute_speech_band_logmag_loss",
    "_compute_vad_eval_metrics",
    "_compute_vad_loss",
    "_compute_vad_probs",
    "_compute_vad_reg_loss",
    "_log1p_mag",
    "_snr_bucket_name",
]

# =============================================================================
# VAD-based speech preservation helpers
# =============================================================================

_EPS = 1e-8
_MIN_VARIANCE = 1e-4

# =============================================================================
# Awesome loss (speech-preserving contrastive) + proxy VAD constants
# =============================================================================
_AWESOME_PROXY_RATIO_FLOOR = 0.3
_AWESOME_PROXY_RATIO_SCALE = 0.7
_AWESOME_LOW_ENERGY_WEIGHT = 0.7
_AWESOME_LOW_SNR_WEIGHT = 0.7
_AWESOME_MOD_THRESHOLD = 0.25
_AWESOME_MOD_WIDTH = 0.15
_AWESOME_ENERGY_BOOST_DB = -3.5
_AWESOME_ENERGY_BOOST_WIDTH = 1.5
_AWESOME_SMOOTH_WEIGHT = 0.2
_AWESOME_MUSICNESS_THR = 0.55
_AWESOME_MUSICNESS_WIDTH = 0.15
_AWESOME_MUSIC_FLUX_THR = 0.08
_AWESOME_MUSIC_FLUX_WIDTH = 0.05
_AWESOME_MASK_LOGIT_CLAMP = 30.0
_VAD_LOGIT_CLAMP = 20.0

# =============================================================================
# Pipeline Awesome loss constants (improved speech preservation + music suppression)
# =============================================================================
_PIPELINE_MIN_MASK_FLOOR = 0.08  # Prevent complete suppression
_PIPELINE_LOW_ENERGY_ADDITIVE = 0.25  # Additive boost for quiet speech
_PIPELINE_LOW_SNR_ADDITIVE = 0.25  # Additive boost for low-SNR
_PIPELINE_PROXY_FLOOR = 0.15  # Higher minimum proxy weight
_PIPELINE_MUSIC_SUPPRESSION_WEIGHT = 1.5  # Music suppression strength
_PIPELINE_VOCAL_HARMONIC_THR = 0.4  # Threshold for vocal harmonic detection
_PIPELINE_PITCH_STABILITY_THR = 0.3  # Threshold for pitch stability (vocals)
_PIPELINE_ARTIFACT_SMOOTH_WEIGHT = 0.3  # Temporal smoothing for artifact control


def _build_speech_band_mask(
    n_freqs: int,
    sample_rate: int,
    band_low_hz: float,
    band_high_hz: float,
) -> tuple[mx.array, float]:
    """Build a fixed speech-band mask for STFT bins."""
    freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float32)
    mask = ((freqs >= band_low_hz) & (freqs <= band_high_hz)).astype(np.float32)
    band_bins = float(mask.sum())
    if band_bins < 1:
        raise ValueError(
            f"Speech band [{band_low_hz}, {band_high_hz}] Hz has no bins for " f"n_freqs={n_freqs}, sr={sample_rate}."
        )
    return mx.array(mask), band_bins


def _z_score_clean_energy(
    clean_real: mx.array,
    clean_imag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    eps: float = _EPS,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Shared z-scored clean energy computation used by VAD and proxy gates.

    Returns:
        clean_power: (B, T, F) power spectrum
        clean_band: (B, T) band energy
        log_clean: (B, T) log10 band energy
        z_ref_raw: (B, T) raw z-scored energy (before clipping)
        z_ref: (B, T) clipped z-scored energy
        p_ref: (B, T) sigmoid VAD probability
    """
    clean_power = clean_real**2 + clean_imag**2
    clean_band = mx.sum(clean_power * band_mask, axis=-1) / (band_bins + eps)
    log_clean = mx.log10(clean_band + eps)
    mu = mx.mean(log_clean, axis=1, keepdims=True)
    variance = mx.mean((log_clean - mu) ** 2, axis=1, keepdims=True)
    sigma = mx.sqrt(mx.maximum(variance, _MIN_VARIANCE) + eps)
    z_ref_raw = (log_clean - mu) / (sigma + eps)
    z_ref = mx.clip(z_ref_raw, -_VAD_LOGIT_CLAMP, _VAD_LOGIT_CLAMP)
    z_slope = max(vad_z_slope, 1e-3)
    p_ref = mx.sigmoid((z_ref - vad_z_threshold) / z_slope)
    return clean_power, clean_band, log_clean, z_ref_raw, z_ref, p_ref


def _compute_vad_probs(
    clean_real: mx.array,
    clean_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
    _assume_float32: bool = False,
    _precomputed_z: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array] | None = None,
) -> tuple[mx.array, mx.array]:
    """Compute soft VAD probabilities from log-band energy (z-scored per utterance).

    Args:
        _precomputed_z: Optional tuple from ``_z_score_clean_energy`` to skip
            redundant clean-signal z-scoring.
    """
    if not _assume_float32:
        if clean_real.dtype != mx.float32:
            clean_real = clean_real.astype(mx.float32)
        if clean_imag.dtype != mx.float32:
            clean_imag = clean_imag.astype(mx.float32)
        if out_real.dtype != mx.float32:
            out_real = out_real.astype(mx.float32)
        if out_imag.dtype != mx.float32:
            out_imag = out_imag.astype(mx.float32)

    if _precomputed_z is not None:
        _, clean_band, log_clean, z_ref_raw, z_ref, p_ref = _precomputed_z
    else:
        _, clean_band, log_clean, z_ref_raw, z_ref, p_ref = _z_score_clean_energy(
            clean_real, clean_imag, band_mask, band_bins, vad_z_threshold, vad_z_slope, eps
        )

    out_power = out_real**2 + out_imag**2
    out_band = mx.sum(out_power * band_mask, axis=-1) / (band_bins + eps)
    mu = mx.mean(log_clean, axis=1, keepdims=True)
    sigma = mx.sqrt(mx.maximum(mx.mean((log_clean - mu) ** 2, axis=1, keepdims=True), _MIN_VARIANCE) + eps)
    z_out_raw = (mx.log10(out_band + eps) - mu) / (sigma + eps)
    z_out = mx.clip(z_out_raw, -_VAD_LOGIT_CLAMP, _VAD_LOGIT_CLAMP)
    z_slope = max(vad_z_slope, 1e-3)
    p_out = mx.sigmoid((z_out - vad_z_threshold) / z_slope)

    if debug is not None:
        debug.check("vad.clean_band", clean_band, debug_ctx)
        debug.check("vad.out_band", out_band, debug_ctx)
        debug.check("vad.log_clean", log_clean, debug_ctx)
        debug.check("vad.sigma", sigma, debug_ctx)
        debug.check("vad.z_ref_raw", z_ref_raw, debug_ctx)
        debug.check("vad.z_out_raw", z_out_raw, debug_ctx)
        debug.check("vad.z_ref", z_ref, debug_ctx)
        debug.check("vad.z_out", z_out, debug_ctx)
        debug.check("vad.p_ref", p_ref, debug_ctx)
        debug.check("vad.p_out", p_out, debug_ctx)
    return p_ref, p_out


def _compute_vad_loss(
    clean_real: mx.array,
    clean_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    snr: mx.array,
    band_mask: mx.array,
    band_bins: float,
    vad_threshold: float,
    vad_margin: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute soft VAD loss and diagnostics.

    Penalizes decreases in VAD probability relative to reference speech.
    """
    p_ref, p_out = _compute_vad_probs(
        clean_real,
        clean_imag,
        out_real,
        out_imag,
        band_mask,
        band_bins,
        vad_z_threshold,
        vad_z_slope,
        debug=debug,
        debug_ctx=debug_ctx,
    )

    speech_gate = mx.clip((p_ref - vad_threshold) / (1.0 - vad_threshold + _EPS), 0.0, 1.0)
    snr_scale = max(vad_snr_gate_width, 1e-3)
    snr_gate = mx.sigmoid((snr[:, None] - vad_snr_gate_db) / snr_scale)
    gate = mx.stop_gradient(speech_gate * snr_gate)

    vad_loss = mx.mean(mx.maximum(p_ref - p_out - vad_margin, 0.0) * gate)
    if debug is not None:
        debug.check("vad.speech_gate", speech_gate, debug_ctx)
        debug.check("vad.snr_gate", snr_gate, debug_ctx)
        debug.check("vad.gate", gate, debug_ctx)
        debug.check("vad.loss", vad_loss, debug_ctx)
    return vad_loss, p_ref, p_out, gate


def _compute_speech_band_logmag_loss(
    clean_real: mx.array,
    clean_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    gate: mx.array,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
    _assume_float32: bool = False,
) -> mx.array:
    """Compute speech-band log-magnitude L1 loss weighted by VAD gate."""
    if not _assume_float32:
        if clean_real.dtype != mx.float32:
            clean_real = clean_real.astype(mx.float32)
        if clean_imag.dtype != mx.float32:
            clean_imag = clean_imag.astype(mx.float32)
        if out_real.dtype != mx.float32:
            out_real = out_real.astype(mx.float32)
        if out_imag.dtype != mx.float32:
            out_imag = out_imag.astype(mx.float32)
    clean_mag = mx.sqrt(clean_real**2 + clean_imag**2 + eps)
    out_mag = mx.sqrt(out_real**2 + out_imag**2 + eps)

    clean_log = mx.log10(clean_mag + eps)
    out_log = mx.log10(out_mag + eps)

    clean_band = mx.sum(clean_log * band_mask, axis=-1) / (band_bins + eps)
    out_band = mx.sum(out_log * band_mask, axis=-1) / (band_bins + eps)

    loss = mx.mean(mx.abs(out_band - clean_band) * gate)
    if debug is not None:
        debug.check("speech_band.clean_band", clean_band, debug_ctx)
        debug.check("speech_band.out_band", out_band, debug_ctx)
        debug.check("speech_band.loss", loss, debug_ctx)
    return loss


def _log1p_mag(
    real: mx.array,
    imag: mx.array,
    eps: float = _EPS,
    _assume_float32: bool = False,
) -> mx.array:
    """Compute log1p magnitude for complex STFT."""
    if not _assume_float32:
        if real.dtype != mx.float32:
            real = real.astype(mx.float32)
        if imag.dtype != mx.float32:
            imag = imag.astype(mx.float32)
    mag = mx.sqrt(real**2 + imag**2 + eps)
    return mx.log1p(mag)


def _compute_musicness(
    mag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
    _assume_float32: bool = False,
) -> tuple[mx.array, mx.array]:
    """Compute a cheap musicness score and its inverse gate.

    Uses spectral flatness (tonalness) and temporal flux stability.
    Returns per-sample musicness and a [0,1] gate (1 = keep speech bias).
    """
    # Spectral flatness over speech band
    if not _assume_float32 and mag.dtype != mx.float32:
        mag = mag.astype(mx.float32)
    log_mag = mx.log(mag + eps)
    mean_log = mx.sum(log_mag * band_mask, axis=-1) / (band_bins + eps)
    geom_mean = mx.exp(mean_log)
    arith_mean = mx.sum(mag * band_mask, axis=-1) / (band_bins + eps)
    flatness = geom_mean / (arith_mean + eps)
    tonal = 1.0 - mx.clip(flatness, 0.0, 1.0)
    tonal_mean = mx.mean(tonal, axis=1, keepdims=True)

    # Temporal flux (lower flux => more music-like)
    # Edge case: with single frame, no flux can be computed - assume speech-like
    band_mag = mag * band_mask
    if mag.shape[1] > 1:
        flux = mx.sum(mx.abs(band_mag[:, 1:, :] - band_mag[:, :-1, :]), axis=-1) / (band_bins + eps)
        flux = mx.mean(flux, axis=1, keepdims=True)
    else:
        flux = mx.zeros((mag.shape[0], 1))
    flux_gate = mx.sigmoid((_AWESOME_MUSIC_FLUX_THR - flux) / _AWESOME_MUSIC_FLUX_WIDTH)

    musicness = mx.clip(tonal_mean * flux_gate, 0.0, 1.0)
    music_gate = 1.0 - mx.sigmoid((musicness - _AWESOME_MUSICNESS_THR) / _AWESOME_MUSICNESS_WIDTH)
    musicness = musicness.squeeze(-1)
    music_gate = music_gate.squeeze(-1)
    if debug is not None:
        debug.check("musicness.score", musicness, debug_ctx)
        debug.check("musicness.gate", music_gate, debug_ctx)
    return musicness, music_gate


def _compute_proxy_gates(
    clean_real: mx.array,
    clean_imag: mx.array,
    noisy_real: mx.array,
    noisy_imag: mx.array,
    snr: mx.array,
    band_mask: mx.array,
    band_bins: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    proxy_enabled: bool,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
    noise_real: mx.array | None = None,
    noise_imag: mx.array | None = None,
    _assume_float32: bool = False,
    _precomputed_z: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Compute proxy VAD gates and statistics.

    Args:
        noise_real: Pre-computed ``noisy_real - clean_real`` (avoids duplicate
            subtraction when the caller already has this value).
        noise_imag: Pre-computed ``noisy_imag - clean_imag``.
        _assume_float32: When True, skip dtype checks — caller guarantees FP32 inputs.
        _precomputed_z: Optional tuple from ``_z_score_clean_energy`` to skip
            redundant clean-signal z-scoring.

    Returns:
        proxy_frame: (B, T) speech presence proxy
        speech_ratio: (B, T) speech energy ratio in speech band
        music_gate: (B,) gate to downweight music-like frames
        musicness: (B,) musicness score
        mod_energy: (B, 1) modulation energy proxy
        energy_boost: (B, 1) low-energy boost
        snr_boost: (B, 1) low-SNR boost
    """
    if not _assume_float32:
        if clean_real.dtype != mx.float32:
            clean_real = clean_real.astype(mx.float32)
        if clean_imag.dtype != mx.float32:
            clean_imag = clean_imag.astype(mx.float32)
        if noisy_real.dtype != mx.float32:
            noisy_real = noisy_real.astype(mx.float32)
        if noisy_imag.dtype != mx.float32:
            noisy_imag = noisy_imag.astype(mx.float32)

    if _precomputed_z is not None:
        clean_power, clean_band, log_clean, z_ref_raw, z_ref, p_ref = _precomputed_z
    else:
        clean_power = clean_real**2 + clean_imag**2
        clean_band = mx.sum(clean_power * band_mask, axis=-1) / (band_bins + eps)
        log_clean = mx.log10(clean_band + eps)
        mu = mx.mean(log_clean, axis=1, keepdims=True)
        variance = mx.mean((log_clean - mu) ** 2, axis=1, keepdims=True)
        sigma = mx.sqrt(mx.maximum(variance, _MIN_VARIANCE) + eps)
        z_ref_raw = (log_clean - mu) / (sigma + eps)
        z_ref = mx.clip(z_ref_raw, -_VAD_LOGIT_CLAMP, _VAD_LOGIT_CLAMP)
        z_slope = max(vad_z_slope, 1e-3)
        p_ref = mx.sigmoid((z_ref - vad_z_threshold) / z_slope)

    if noise_real is None:
        noise_real = noisy_real - clean_real
    if noise_imag is None:
        noise_imag = noisy_imag - clean_imag
    noise_power = noise_real**2 + noise_imag**2

    noise_band = mx.sum(noise_power * band_mask, axis=-1) / (band_bins + eps)
    speech_ratio = clean_band / (clean_band + noise_band + eps)

    # Modulation proxy from z-scored energy trajectory
    # Edge case: if only 1 frame, no modulation can be computed
    if z_ref.shape[1] > 1:
        mod_energy = mx.mean(mx.abs(z_ref[:, 1:] - z_ref[:, :-1]), axis=1, keepdims=True)
    else:
        mod_energy = mx.zeros((z_ref.shape[0], 1))
    mod_gate = mx.sigmoid((mod_energy - _AWESOME_MOD_THRESHOLD) / _AWESOME_MOD_WIDTH)

    mean_log = mx.mean(log_clean, axis=1, keepdims=True)
    energy_boost = mx.sigmoid((_AWESOME_ENERGY_BOOST_DB - mean_log) / _AWESOME_ENERGY_BOOST_WIDTH)

    snr_scale = max(vad_snr_gate_width, 1e-3)
    snr_boost = mx.sigmoid((vad_snr_gate_db - snr[:, None]) / snr_scale)

    # Musicness gate from noisy magnitude
    noisy_mag = mx.sqrt(noisy_real**2 + noisy_imag**2 + eps)
    musicness, music_gate = _compute_musicness(
        noisy_mag,
        band_mask,
        band_bins,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
        _assume_float32=_assume_float32,
    )

    if not proxy_enabled:
        proxy_frame = mx.ones_like(clean_band)
    else:
        proxy_frame = p_ref * (_AWESOME_PROXY_RATIO_FLOOR + _AWESOME_PROXY_RATIO_SCALE * speech_ratio)
        proxy_frame = proxy_frame * mod_gate * music_gate[:, None]
        proxy_frame = proxy_frame * (
            1.0 + _AWESOME_LOW_ENERGY_WEIGHT * energy_boost + _AWESOME_LOW_SNR_WEIGHT * snr_boost
        )
        proxy_frame = mx.clip(proxy_frame, 0.0, 5.0)

    proxy_frame = mx.stop_gradient(proxy_frame)
    if debug is not None:
        debug.check("proxy.z_ref_raw", z_ref_raw, debug_ctx)
        debug.check("proxy.z_ref", z_ref, debug_ctx)
        debug.check("proxy.speech_ratio", speech_ratio, debug_ctx)
        debug.check("proxy.p_ref", p_ref, debug_ctx)
        debug.check("proxy.mod_energy", mod_energy, debug_ctx)
        debug.check("proxy.energy_boost", energy_boost, debug_ctx)
        debug.check("proxy.snr_boost", snr_boost, debug_ctx)
        debug.check("proxy.frame", proxy_frame, debug_ctx)
    return proxy_frame, speech_ratio, music_gate, musicness, mod_energy, energy_boost, snr_boost


def _compute_awesome_teacher_signals(
    noisy_real: mx.array,
    noisy_imag: mx.array,
    clean_real: mx.array,
    clean_imag: mx.array,
    snr: mx.array,
    band_mask: mx.array,
    band_bins: float,
    mask_sharpness: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    proxy_enabled: bool,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Return the stop-gradient teacher signals used by awesome-family losses."""
    clean_real_f32 = clean_real.astype(mx.float32) if clean_real.dtype != mx.float32 else clean_real
    clean_imag_f32 = clean_imag.astype(mx.float32) if clean_imag.dtype != mx.float32 else clean_imag
    noisy_real_f32 = noisy_real.astype(mx.float32) if noisy_real.dtype != mx.float32 else noisy_real
    noisy_imag_f32 = noisy_imag.astype(mx.float32) if noisy_imag.dtype != mx.float32 else noisy_imag

    clean_log = _log1p_mag(clean_real_f32, clean_imag_f32, eps=eps, _assume_float32=True)
    noise_real = noisy_real_f32 - clean_real_f32
    noise_imag = noisy_imag_f32 - clean_imag_f32
    noise_log = _log1p_mag(noise_real, noise_imag, eps=eps, _assume_float32=True)

    mask_logits = mx.clip(
        mask_sharpness * (clean_log - noise_log),
        -_AWESOME_MASK_LOGIT_CLAMP,
        _AWESOME_MASK_LOGIT_CLAMP,
    )
    mask = mx.stop_gradient(mx.sigmoid(mask_logits))
    if debug is not None:
        debug.check("awesome.clean_log", clean_log, debug_ctx)
        debug.check("awesome.noise_log", noise_log, debug_ctx)
        debug.check("awesome.mask_logits", mask_logits, debug_ctx)
        debug.check("awesome.mask", mask, debug_ctx)

    (
        proxy_frame,
        speech_ratio,
        music_gate,
        musicness,
        mod_energy,
        energy_boost,
        snr_boost,
    ) = _compute_proxy_gates(
        clean_real_f32,
        clean_imag_f32,
        noisy_real_f32,
        noisy_imag_f32,
        snr,
        band_mask,
        band_bins,
        vad_z_threshold,
        vad_z_slope,
        vad_snr_gate_db,
        vad_snr_gate_width,
        proxy_enabled,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
        noise_real=noise_real,
        noise_imag=noise_imag,
        _assume_float32=True,
    )

    return (
        clean_log,
        noise_log,
        mask,
        proxy_frame,
        speech_ratio,
        music_gate,
        musicness,
        mod_energy,
        energy_boost,
        snr_boost,
    )


def _compute_awesome_losses(
    noisy_real: mx.array,
    noisy_imag: mx.array,
    clean_real: mx.array,
    clean_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    snr: mx.array,
    band_mask: mx.array,
    band_bins: float,
    mask_sharpness: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    proxy_enabled: bool,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    """Compute awesome loss components and diagnostic gates."""
    # Cast all inputs to FP32 once at function entry (avoids redundant casts downstream)
    out_real_f32 = out_real.astype(mx.float32) if out_real.dtype != mx.float32 else out_real
    out_imag_f32 = out_imag.astype(mx.float32) if out_imag.dtype != mx.float32 else out_imag

    out_log = _log1p_mag(out_real_f32, out_imag_f32, eps=eps, _assume_float32=True)
    (
        clean_log,
        _noise_log,
        mask,
        proxy_frame,
        speech_ratio,
        music_gate,
        musicness,
        mod_energy,
        energy_boost,
        snr_boost,
    ) = _compute_awesome_teacher_signals(
        noisy_real,
        noisy_imag,
        clean_real,
        clean_imag,
        snr,
        band_mask,
        band_bins,
        mask_sharpness,
        vad_z_threshold,
        vad_z_slope,
        vad_snr_gate_db,
        vad_snr_gate_width,
        proxy_enabled,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
    )

    proxy_frame = proxy_frame[:, :, None]
    speech_loss = mx.mean(mx.abs(out_log - clean_log) * mask * proxy_frame)
    noise_loss = mx.mean(mx.abs(out_log) * (1.0 - mask))

    if out_log.shape[1] > 1:
        smooth_mask = 1.0 - mask[:, 1:, :]
        smooth_loss = mx.mean(mx.abs(out_log[:, 1:, :] - out_log[:, :-1, :]) * smooth_mask)
    else:
        smooth_loss = mx.array(0.0)

    awesome_loss = speech_loss + noise_loss + _AWESOME_SMOOTH_WEIGHT * smooth_loss
    if debug is not None:
        debug.check("awesome.speech_loss", speech_loss, debug_ctx)
        debug.check("awesome.noise_loss", noise_loss, debug_ctx)
        debug.check("awesome.smooth_loss", smooth_loss, debug_ctx)
        debug.check("awesome.loss", awesome_loss, debug_ctx)

    return (
        awesome_loss,
        speech_loss,
        noise_loss,
        smooth_loss,
        mask,
        proxy_frame.squeeze(-1),
        speech_ratio,
        music_gate,
        musicness,
        mod_energy,
        energy_boost,
        snr_boost,
    )


def _build_contrastive_frame_features(
    real: mx.array,
    imag: mx.array,
    eps: float = _EPS,
) -> mx.array:
    """Build per-frame contrastive features from complex STFT components."""
    real_f32 = real.astype(mx.float32) if real.dtype != mx.float32 else real
    imag_f32 = imag.astype(mx.float32) if imag.dtype != mx.float32 else imag
    log_mag = _log1p_mag(real_f32, imag_f32, eps=eps, _assume_float32=True)
    return mx.concatenate([real_f32, imag_f32, log_mag], axis=-1)


def _gather_time_frames(frames: mx.array, indices: mx.array) -> mx.array:
    """Gather ``(B, K)`` time indices from a ``(B, T, C)`` frame tensor."""
    feature_dim = frames.shape[-1]
    gather_idx = mx.expand_dims(indices, axis=-1)
    gather_idx = mx.broadcast_to(gather_idx, (*indices.shape, feature_dim))
    return mx.take_along_axis(frames, gather_idx, axis=1)


def _select_topk_frames(
    scores: mx.array,
    valid_mask: mx.array,
    k: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Select deterministic top-k frames and return indices, weights, validity."""
    batch, time = scores.shape
    if k <= 0:
        empty_idx = mx.zeros((batch, 0), dtype=mx.int32)
        empty_val = mx.zeros((batch, 0), dtype=mx.float32)
        return empty_idx, empty_val, empty_val

    k = min(int(k), int(time))
    neg_inf = mx.full(scores.shape, -1e9, dtype=mx.float32)
    valid_scores = mx.where(valid_mask > 0.0, scores, neg_inf)
    sorted_idx = mx.argsort(valid_scores, axis=1)
    top_idx = sorted_idx[:, -k:]

    gathered_scores = mx.take_along_axis(scores, top_idx, axis=1)
    gathered_valid = mx.take_along_axis(valid_mask, top_idx, axis=1)
    gathered_scores = gathered_scores * gathered_valid
    return top_idx, gathered_scores, gathered_valid


def _reduce_weighted_rows(values: mx.array, row_weights: mx.array, eps: float = _EPS) -> mx.array:
    """Weighted mean over flattened query rows with a safe zero-valid fallback."""
    denom = mx.sum(row_weights)
    weighted = mx.sum(values * row_weights)
    return mx.where(denom > eps, weighted / (denom + eps), mx.array(0.0, dtype=mx.float32))


def _contrastive_info_nce(
    query: mx.array,
    positive: mx.array,
    noisy_negative: mx.array,
    interference_negative: mx.array,
    row_weights: mx.array,
    row_valid: mx.array,
    temperature: float,
    *,
    in_batch_negatives: bool,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute weighted InfoNCE loss and similarity diagnostics for selected rows."""
    if query.shape[1] == 0:
        zero = mx.array(0.0, dtype=mx.float32)
        return zero, zero, zero

    tau = max(float(temperature), 1e-4)
    q = query.reshape(-1, query.shape[-1])
    p = positive.reshape(-1, positive.shape[-1])
    n_mix = noisy_negative.reshape(-1, noisy_negative.shape[-1])
    n_int = interference_negative.reshape(-1, interference_negative.shape[-1])
    weights = (row_weights * row_valid).reshape(-1)

    pos_sim = mx.sum(q * p, axis=-1)
    noisy_sim = mx.sum(q * n_mix, axis=-1)
    interference_sim = mx.sum(q * n_int, axis=-1)

    pos_logits = pos_sim / tau
    neg_logits = mx.stack([noisy_sim / tau, interference_sim / tau], axis=1)
    all_logits = mx.concatenate([mx.expand_dims(pos_logits, axis=1), neg_logits], axis=1)

    if in_batch_negatives and query.shape[0] > 1:
        batch_size, k = query.shape[:2]
        sample_ids = mx.repeat(mx.arange(batch_size, dtype=mx.int32), k)
        other_clean_logits = (q @ mx.transpose(p)) / tau
        other_sample = mx.expand_dims(sample_ids, axis=1) != mx.expand_dims(sample_ids, axis=0)
        valid_cols = mx.expand_dims((weights > 0.0).astype(mx.float32), axis=0)
        neg_inf = mx.full(other_clean_logits.shape, -1e9, dtype=mx.float32)
        other_clean_logits = mx.where(other_sample & (valid_cols > 0.0), other_clean_logits, neg_inf)
        all_logits = mx.concatenate([all_logits, other_clean_logits], axis=1)

    per_row = -pos_logits + mx.logsumexp(all_logits, axis=1)
    avg_neg_sim = 0.5 * (noisy_sim + interference_sim)
    return (
        _reduce_weighted_rows(per_row, weights, eps=_EPS),
        _reduce_weighted_rows(pos_sim, weights, eps=_EPS),
        _reduce_weighted_rows(avg_neg_sim, weights, eps=_EPS),
    )


def _compute_contrastive_awesome_losses(
    noisy_real: mx.array,
    noisy_imag: mx.array,
    clean_real: mx.array,
    clean_imag: mx.array,
    interference_real: mx.array,
    interference_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    snr: mx.array,
    band_mask: mx.array,
    band_bins: float,
    mask_sharpness: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    proxy_enabled: bool,
    projector: Any,
    *,
    temperature: float = 0.1,
    speech_frames_per_sample: int = 32,
    interference_frames_per_sample: int = 32,
    speech_mask_min: float = 0.7,
    interference_mask_max: float = 0.3,
    quiet_weight: float = 0.5,
    in_batch_negatives: bool = True,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    """Compute local InfoNCE-style contrastive AWESOME loss."""
    if projector is None:
        raise ValueError("contrastive_awesome requires an attached ContrastiveFrameProjector")

    clean_real_f32 = clean_real.astype(mx.float32) if clean_real.dtype != mx.float32 else clean_real
    clean_imag_f32 = clean_imag.astype(mx.float32) if clean_imag.dtype != mx.float32 else clean_imag
    noisy_real_f32 = noisy_real.astype(mx.float32) if noisy_real.dtype != mx.float32 else noisy_real
    noisy_imag_f32 = noisy_imag.astype(mx.float32) if noisy_imag.dtype != mx.float32 else noisy_imag
    interference_real_f32 = (
        interference_real.astype(mx.float32) if interference_real.dtype != mx.float32 else interference_real
    )
    interference_imag_f32 = (
        interference_imag.astype(mx.float32) if interference_imag.dtype != mx.float32 else interference_imag
    )
    out_real_f32 = out_real.astype(mx.float32) if out_real.dtype != mx.float32 else out_real
    out_imag_f32 = out_imag.astype(mx.float32) if out_imag.dtype != mx.float32 else out_imag

    clean_log = _log1p_mag(clean_real_f32, clean_imag_f32, eps=eps, _assume_float32=True)
    noise_log = _log1p_mag(interference_real_f32, interference_imag_f32, eps=eps, _assume_float32=True)
    mask_logits = mx.clip(
        mask_sharpness * (clean_log - noise_log),
        -_AWESOME_MASK_LOGIT_CLAMP,
        _AWESOME_MASK_LOGIT_CLAMP,
    )
    mask = mx.stop_gradient(mx.sigmoid(mask_logits))

    (
        proxy_frame,
        speech_ratio,
        music_gate,
        musicness,
        mod_energy,
        energy_boost,
        snr_boost,
    ) = _compute_proxy_gates(
        clean_real_f32,
        clean_imag_f32,
        noisy_real_f32,
        noisy_imag_f32,
        snr,
        band_mask,
        band_bins,
        vad_z_threshold,
        vad_z_slope,
        vad_snr_gate_db,
        vad_snr_gate_width,
        proxy_enabled,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
        noise_real=interference_real_f32,
        noise_imag=interference_imag_f32,
        _assume_float32=True,
    )

    mask_mean = mx.mean(mask, axis=-1)
    speech_valid = mx.stop_gradient((mask_mean >= float(speech_mask_min)).astype(mx.float32))
    interference_valid = mx.stop_gradient((mask_mean <= float(interference_mask_max)).astype(mx.float32))
    speech_scores = mx.stop_gradient(mask_mean * proxy_frame) * speech_valid
    interference_scores = mx.stop_gradient(mx.mean(1.0 - mask, axis=-1)) * interference_valid

    speech_idx, speech_weights, speech_valid_rows = _select_topk_frames(
        speech_scores,
        speech_valid,
        speech_frames_per_sample,
    )
    quiet_idx, quiet_scores, quiet_valid_rows = _select_topk_frames(
        interference_scores,
        interference_valid,
        interference_frames_per_sample,
    )

    out_frames = _build_contrastive_frame_features(out_real_f32, out_imag_f32, eps=eps)
    clean_frames = _build_contrastive_frame_features(clean_real_f32, clean_imag_f32, eps=eps)
    noisy_frames = _build_contrastive_frame_features(noisy_real_f32, noisy_imag_f32, eps=eps)
    interference_frames = _build_contrastive_frame_features(interference_real_f32, interference_imag_f32, eps=eps)

    speech_q = projector(_gather_time_frames(out_frames, speech_idx))
    speech_p = projector(_gather_time_frames(clean_frames, speech_idx))
    speech_n_mix = projector(_gather_time_frames(noisy_frames, speech_idx))
    speech_n_int = projector(_gather_time_frames(interference_frames, speech_idx))

    quiet_q = projector(_gather_time_frames(out_frames, quiet_idx))
    quiet_p = projector(_gather_time_frames(clean_frames, quiet_idx))
    quiet_n_mix = projector(_gather_time_frames(noisy_frames, quiet_idx))
    quiet_n_int = projector(_gather_time_frames(interference_frames, quiet_idx))

    speech_ctr_loss, speech_pos_sim, speech_neg_sim = _contrastive_info_nce(
        speech_q,
        speech_p,
        speech_n_mix,
        speech_n_int,
        speech_weights,
        speech_valid_rows,
        temperature,
        in_batch_negatives=in_batch_negatives,
    )
    quiet_ctr_loss, quiet_pos_sim, quiet_neg_sim = _contrastive_info_nce(
        quiet_q,
        quiet_p,
        quiet_n_mix,
        quiet_n_int,
        quiet_scores,
        quiet_valid_rows,
        temperature,
        in_batch_negatives=in_batch_negatives,
    )

    total_loss = speech_ctr_loss + max(float(quiet_weight), 0.0) * quiet_ctr_loss
    speech_valid_count = mx.sum(speech_valid_rows)
    interference_valid_count = mx.sum(quiet_valid_rows)
    speech_weight_mass = mx.sum(speech_weights * speech_valid_rows)
    quiet_weight_mass = mx.sum(quiet_scores * quiet_valid_rows)
    contrastive_pos_sim = _reduce_weighted_rows(
        mx.stack([speech_pos_sim, quiet_pos_sim]),
        mx.stack([speech_weight_mass, quiet_weight_mass]),
        eps=eps,
    )
    contrastive_neg_sim = _reduce_weighted_rows(
        mx.stack([speech_neg_sim, quiet_neg_sim]),
        mx.stack([speech_weight_mass, quiet_weight_mass]),
        eps=eps,
    )

    if debug is not None:
        debug.check("contrastive.mask", mask, debug_ctx)
        debug.check("contrastive.proxy_frame", proxy_frame, debug_ctx)
        debug.check("contrastive.speech_scores", speech_scores, debug_ctx)
        debug.check("contrastive.interference_scores", interference_scores, debug_ctx)
        debug.check("contrastive.total_loss", total_loss, debug_ctx)
        debug.check("contrastive.speech_loss", speech_ctr_loss, debug_ctx)
        debug.check("contrastive.quiet_loss", quiet_ctr_loss, debug_ctx)

    return (
        total_loss,
        speech_ctr_loss,
        quiet_ctr_loss,
        contrastive_pos_sim,
        contrastive_neg_sim,
        mask,
        proxy_frame,
        speech_ratio,
        music_gate,
        musicness,
        mod_energy,
        energy_boost,
        snr_boost,
        speech_valid_count,
        interference_valid_count,
    )


def _compute_pitch_stability(
    mag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    eps: float = _EPS,
    _assume_float32: bool = False,
) -> mx.array:
    """Compute pitch stability metric to detect sustained vocals vs speech.

    Vocals tend to have more stable pitch (lower frame-to-frame variation)
    while speech has more dynamic pitch contours.

    Returns per-sample pitch stability in [0, 1], where 1 = very stable (vocal-like).
    """
    if not _assume_float32 and mag.dtype != mx.float32:
        mag = mag.astype(mx.float32)
    band_mag = mag * band_mask

    # Compute spectral centroid per frame
    freq_weights = mx.arange(band_mag.shape[-1], dtype=mx.float32)
    centroid = mx.sum(band_mag * freq_weights, axis=-1) / (mx.sum(band_mag, axis=-1) + eps)

    # Pitch stability = inverse of centroid variation
    if centroid.shape[1] > 1:
        centroid_diff = mx.abs(centroid[:, 1:] - centroid[:, :-1])
        centroid_var = mx.mean(centroid_diff, axis=1, keepdims=True)
        # Normalize and invert: low variation = high stability
        stability = mx.exp(-centroid_var / 10.0)
    else:
        stability = mx.ones((mag.shape[0], 1))

    return mx.clip(stability, 0.0, 1.0).squeeze(-1)


def _compute_harmonic_ratio(
    mag: mx.array,
    eps: float = _EPS,
    _assume_float32: bool = False,
) -> mx.array:
    """Compute harmonic-to-noise ratio to detect tonal content (vocals/music).

    Uses autocorrelation proxy: high HNR = more harmonic/tonal content.
    Returns per-sample HNR score in [0, 1].
    """
    if not _assume_float32 and mag.dtype != mx.float32:
        mag = mag.astype(mx.float32)

    # Simple proxy: ratio of peak to mean energy in low-mid frequencies
    # Harmonic content creates spectral peaks
    low_mid_mag = mag[:, :, : mag.shape[-1] // 2]  # Lower half of spectrum
    peak_energy = mx.max(low_mid_mag, axis=-1)
    mean_energy = mx.mean(low_mid_mag, axis=-1) + eps

    hnr_proxy = peak_energy / mean_energy
    # Normalize to [0, 1] using sigmoid
    hnr_score = mx.sigmoid((hnr_proxy - 3.0) / 1.0)  # Center at ratio=3

    return mx.mean(hnr_score, axis=1)


def _compute_improved_musicness(
    mag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    snr: mx.array,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
    _assume_float32: bool = False,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute improved musicness score with vocal detection.

    Returns:
        musicness: (B,) overall musicness score
        vocal_gate: (B,) gate for vocal content (1 = protect as speech)
        instrument_gate: (B,) gate for instrumental content (1 = suppress)
    """
    if not _assume_float32 and mag.dtype != mx.float32:
        mag = mag.astype(mx.float32)

    # Original spectral flatness
    log_mag = mx.log(mag + eps)
    mean_log = mx.sum(log_mag * band_mask, axis=-1) / (band_bins + eps)
    geom_mean = mx.exp(mean_log)
    arith_mean = mx.sum(mag * band_mask, axis=-1) / (band_bins + eps)
    flatness = geom_mean / (arith_mean + eps)
    tonal = 1.0 - mx.clip(flatness, 0.0, 1.0)
    tonal_mean = mx.mean(tonal, axis=1, keepdims=True)

    # Temporal flux
    # Edge case: with single frame, no flux can be computed - assume speech-like
    band_mag = mag * band_mask
    if mag.shape[1] > 1:
        flux = mx.sum(mx.abs(band_mag[:, 1:, :] - band_mag[:, :-1, :]), axis=-1) / (band_bins + eps)
        flux_mean = mx.mean(flux, axis=1, keepdims=True)
    else:
        flux_mean = mx.zeros((mag.shape[0], 1))
    flux_gate = mx.sigmoid((_AWESOME_MUSIC_FLUX_THR - flux_mean) / _AWESOME_MUSIC_FLUX_WIDTH)

    # Pitch stability (vocals = less stable than instruments)
    pitch_stability = _compute_pitch_stability(mag, band_mask, band_bins, eps, _assume_float32=_assume_float32)

    # Harmonic ratio
    harmonic_ratio = _compute_harmonic_ratio(mag, eps, _assume_float32=_assume_float32)

    # Musicness from original features
    musicness_base = mx.clip(tonal_mean.squeeze(-1) * flux_gate.squeeze(-1), 0.0, 1.0)

    # Vocal detection: high tonality + moderate pitch stability + present in speech band
    # Vocals: tonal but with more pitch variation than instruments
    vocal_indicator = tonal_mean.squeeze(-1) * (1.0 - pitch_stability) * harmonic_ratio
    vocal_gate = mx.sigmoid((vocal_indicator - _PIPELINE_VOCAL_HARMONIC_THR) / 0.15)

    # Instrumental: high tonality + high pitch stability (sustained notes)
    instrument_indicator = tonal_mean.squeeze(-1) * pitch_stability * flux_gate.squeeze(-1)
    instrument_gate = mx.sigmoid((instrument_indicator - _PIPELINE_PITCH_STABILITY_THR) / 0.15)

    # Adjust musicness: reduce for vocals (they should be preserved as speech-like)
    musicness = musicness_base * (1.0 - 0.5 * vocal_gate)

    if debug is not None:
        debug.check("improved_musicness.tonal", tonal_mean, debug_ctx)
        debug.check("improved_musicness.flux", flux_mean, debug_ctx)
        debug.check("improved_musicness.pitch_stab", pitch_stability, debug_ctx)
        debug.check("improved_musicness.harmonic", harmonic_ratio, debug_ctx)
        debug.check("improved_musicness.vocal_gate", vocal_gate, debug_ctx)
        debug.check("improved_musicness.instrument_gate", instrument_gate, debug_ctx)

    return musicness, vocal_gate, instrument_gate


def _compute_pipeline_awesome_losses(
    noisy_real: mx.array,
    noisy_imag: mx.array,
    clean_real: mx.array,
    clean_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    snr: mx.array,
    band_mask: mx.array,
    band_bins: float,
    mask_sharpness: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    proxy_enabled: bool,
    min_mask_floor: float = _PIPELINE_MIN_MASK_FLOOR,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[
    mx.array,  # total loss
    mx.array,  # speech loss
    mx.array,  # noise loss
    mx.array,  # smooth loss
    mx.array,  # music suppression loss
    mx.array,  # mask saturation loss
    mx.array,  # mask
    mx.array,  # proxy_frame
    mx.array,  # speech_ratio
    mx.array,  # music_gate
    mx.array,  # musicness
    mx.array,  # vocal_gate
    mx.array,  # instrument_gate
    mx.array,  # mod_energy
    mx.array,  # energy_boost
    mx.array,  # snr_boost
]:
    """Compute pipeline_awesome loss with improved speech preservation and music suppression.

    Key improvements over basic awesome loss:
    1. Minimum mask floor to prevent complete speech suppression
    2. Additive (not multiplicative) boosts for low-energy and low-SNR speech
    3. Improved musicness detection with vocal/instrument separation
    4. Speech-band weighted loss
    5. Mask saturation penalty to encourage confident predictions
    6. Explicit music suppression loss

    Note: The mask saturation penalty uses mask entropy: mask*(1-mask).
    This is minimized when mask is near 0 or 1 (confident), and maximized
    at 0.5 (uncertain). We want to PENALIZE uncertainty, so we use it directly.
    """
    # Cast all inputs to FP32 once at function entry (avoids redundant casts downstream)
    clean_real_f32 = clean_real.astype(mx.float32) if clean_real.dtype != mx.float32 else clean_real
    clean_imag_f32 = clean_imag.astype(mx.float32) if clean_imag.dtype != mx.float32 else clean_imag
    noisy_real_f32 = noisy_real.astype(mx.float32) if noisy_real.dtype != mx.float32 else noisy_real
    noisy_imag_f32 = noisy_imag.astype(mx.float32) if noisy_imag.dtype != mx.float32 else noisy_imag
    out_real_f32 = out_real.astype(mx.float32) if out_real.dtype != mx.float32 else out_real
    out_imag_f32 = out_imag.astype(mx.float32) if out_imag.dtype != mx.float32 else out_imag

    # Compute log magnitudes using pre-cast FP32 values (no redundant casts in _log1p_mag)
    clean_log = _log1p_mag(clean_real_f32, clean_imag_f32, eps=eps, _assume_float32=True)
    out_log = _log1p_mag(out_real_f32, out_imag_f32, eps=eps, _assume_float32=True)

    noise_real = noisy_real_f32 - clean_real_f32
    noise_imag = noisy_imag_f32 - clean_imag_f32
    noise_log = _log1p_mag(noise_real, noise_imag, eps=eps, _assume_float32=True)

    # Compute speech/noise dominance mask with floor
    mask_logits = mx.clip(
        mask_sharpness * (clean_log - noise_log),
        -_AWESOME_MASK_LOGIT_CLAMP,
        _AWESOME_MASK_LOGIT_CLAMP,
    )
    raw_mask = mx.sigmoid(mask_logits)
    # Apply minimum floor to prevent complete suppression
    mask = mx.maximum(raw_mask, min_mask_floor)
    mask = mx.stop_gradient(mask)

    if debug is not None:
        debug.check("pipeline.clean_log", clean_log, debug_ctx)
        debug.check("pipeline.noise_log", noise_log, debug_ctx)
        debug.check("pipeline.mask_logits", mask_logits, debug_ctx)
        debug.check("pipeline.raw_mask", raw_mask, debug_ctx)
        debug.check("pipeline.mask", mask, debug_ctx)

    # Reuse pre-cast FP32 values for proxy gates (no duplicate casts)
    clean_power = clean_real_f32**2 + clean_imag_f32**2
    # Reuse noise_real/noise_imag computed above (avoids duplicate subtraction)
    noise_power = noise_real**2 + noise_imag**2

    clean_band = mx.sum(clean_power * band_mask, axis=-1) / (band_bins + eps)
    noise_band = mx.sum(noise_power * band_mask, axis=-1) / (band_bins + eps)
    speech_ratio = clean_band / (clean_band + noise_band + eps)

    # Z-scored log energy for VAD proxy
    # Edge case handling: if variance is near-zero (silence), use neutral z-scores
    log_clean = mx.log10(clean_band + eps)
    mu = mx.mean(log_clean, axis=1, keepdims=True)
    variance = mx.mean((log_clean - mu) ** 2, axis=1, keepdims=True)
    # Use a minimum variance threshold to avoid division instability on silence
    sigma = mx.sqrt(mx.maximum(variance, _MIN_VARIANCE) + eps)
    # When variance is too low, z-scores become unreliable; clamp them
    z_ref_raw = (log_clean - mu) / (sigma + eps)
    z_ref = mx.clip(z_ref_raw, -_VAD_LOGIT_CLAMP, _VAD_LOGIT_CLAMP)

    z_slope = max(vad_z_slope, 1e-3)
    p_ref = mx.sigmoid((z_ref - vad_z_threshold) / z_slope)

    # Modulation proxy
    # Edge case: with single frame, no modulation can be computed
    if z_ref.shape[1] > 1:
        mod_energy = mx.mean(mx.abs(z_ref[:, 1:] - z_ref[:, :-1]), axis=1, keepdims=True)
    else:
        mod_energy = mx.zeros((z_ref.shape[0], 1))
    mod_gate = mx.sigmoid((mod_energy - _AWESOME_MOD_THRESHOLD) / _AWESOME_MOD_WIDTH)

    # Energy and SNR boosts (ADDITIVE, not multiplicative)
    mean_log = mx.mean(log_clean, axis=1, keepdims=True)
    energy_boost = mx.sigmoid((_AWESOME_ENERGY_BOOST_DB - mean_log) / _AWESOME_ENERGY_BOOST_WIDTH)

    snr_scale = max(vad_snr_gate_width, 1e-3)
    snr_boost = mx.sigmoid((vad_snr_gate_db - snr[:, None]) / snr_scale)

    # Improved musicness detection
    noisy_mag = mx.sqrt(noisy_real_f32**2 + noisy_imag_f32**2 + eps)
    musicness, vocal_gate, instrument_gate = _compute_improved_musicness(
        noisy_mag,
        band_mask,
        band_bins,
        snr,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
        _assume_float32=True,
    )

    # Music gate: downweight for instrumental, but preserve vocal-like content
    music_gate = 1.0 - mx.sigmoid((musicness - _AWESOME_MUSICNESS_THR) / _AWESOME_MUSICNESS_WIDTH)
    # Boost back for vocals (they should be preserved)
    music_gate = music_gate + 0.5 * vocal_gate * (1.0 - music_gate)

    if not proxy_enabled:
        proxy_frame = mx.ones_like(clean_band)
    else:
        # Base proxy from VAD and speech ratio (with higher floor)
        base_proxy = p_ref * (_PIPELINE_PROXY_FLOOR + (1.0 - _PIPELINE_PROXY_FLOOR) * speech_ratio)
        base_proxy = base_proxy * mod_gate * music_gate[:, None]

        # ADDITIVE boosts (key improvement for low-signal speech)
        proxy_frame = base_proxy + _PIPELINE_LOW_ENERGY_ADDITIVE * energy_boost + _PIPELINE_LOW_SNR_ADDITIVE * snr_boost
        proxy_frame = mx.clip(proxy_frame, _PIPELINE_PROXY_FLOOR, 5.0)

    proxy_frame = mx.stop_gradient(proxy_frame)

    if debug is not None:
        debug.check("pipeline.z_ref", z_ref, debug_ctx)
        debug.check("pipeline.p_ref", p_ref, debug_ctx)
        debug.check("pipeline.speech_ratio", speech_ratio, debug_ctx)
        debug.check("pipeline.energy_boost", energy_boost, debug_ctx)
        debug.check("pipeline.snr_boost", snr_boost, debug_ctx)
        debug.check("pipeline.music_gate", music_gate, debug_ctx)
        debug.check("pipeline.proxy_frame", proxy_frame, debug_ctx)

    # ========== Loss components ==========

    # 1. Speech preservation loss (weighted by proxy)
    proxy_frame_3d = proxy_frame[:, :, None]
    speech_loss = mx.mean(mx.abs(out_log - clean_log) * mask * proxy_frame_3d)

    # 2. Noise suppression loss
    noise_loss = mx.mean(mx.abs(out_log) * (1.0 - mask))

    # 3. Temporal smoothness for artifact control (stronger than base awesome)
    if out_log.shape[1] > 1:
        smooth_mask = 1.0 - mask[:, 1:, :]
        smooth_loss = mx.mean(mx.abs(out_log[:, 1:, :] - out_log[:, :-1, :]) * smooth_mask)
    else:
        smooth_loss = mx.array(0.0)

    # 4. Music suppression loss: penalize output energy where instrumental music detected
    instrument_weight = instrument_gate[:, None, None] * (1.0 - mask)  # Only where noise dominant
    music_suppression_loss = mx.mean(mx.abs(out_log) * instrument_weight)

    # 5. Mask saturation metric: measures confidence of ground-truth mask.
    # NOTE: raw_mask depends only on ground-truth signals (clean_log, noise_log),
    # NOT model parameters, so its gradient w.r.t. model params is always zero.
    # We compute it as a diagnostic metric but exclude it from the training loss
    # to avoid inflating the loss with a gradient-free constant.
    mask_entropy = mx.mean(raw_mask * (1.0 - raw_mask))
    mask_saturation_loss = 4.0 * mask_entropy

    # Total loss (mask_saturation excluded — zero gradient w.r.t. model params)
    total_loss = (
        speech_loss
        + noise_loss
        + _PIPELINE_ARTIFACT_SMOOTH_WEIGHT * smooth_loss
        + _PIPELINE_MUSIC_SUPPRESSION_WEIGHT * music_suppression_loss
    )

    if debug is not None:
        debug.check("pipeline.speech_loss", speech_loss, debug_ctx)
        debug.check("pipeline.noise_loss", noise_loss, debug_ctx)
        debug.check("pipeline.smooth_loss", smooth_loss, debug_ctx)
        debug.check("pipeline.music_suppression_loss", music_suppression_loss, debug_ctx)
        debug.check("pipeline.mask_saturation_loss", mask_saturation_loss, debug_ctx)
        debug.check("pipeline.total_loss", total_loss, debug_ctx)

    return (
        total_loss,
        speech_loss,
        noise_loss,
        smooth_loss,
        music_suppression_loss,
        mask_saturation_loss,
        mask,
        proxy_frame,
        speech_ratio,
        music_gate,
        musicness,
        vocal_gate,
        instrument_gate,
        mod_energy.squeeze(-1),
        energy_boost.squeeze(-1),
        snr_boost.squeeze(-1),
    )


def _compute_vad_reg_loss(
    clean_real: mx.array,
    clean_imag: mx.array,
    noisy_real: mx.array,
    noisy_imag: mx.array,
    out_real: mx.array,
    out_imag: mx.array,
    snr: mx.array,
    band_mask: mx.array,
    band_bins: float,
    vad_threshold: float,
    vad_margin: float,
    vad_z_threshold: float,
    vad_z_slope: float,
    vad_snr_gate_db: float,
    vad_snr_gate_width: float,
    eps: float = _EPS,
    debug: NumericDebugger | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Compute sparse VAD regularizer loss gated by speech ratio and musicness.

    Uses VAD probabilities only as stop-grad weights (non-differentiable).
    Computes clean-signal z-scoring once and shares it between VAD probs
    and proxy gates to avoid redundant computation.
    """
    # Compute z-scored clean energy ONCE (shared by vad_probs and proxy_gates)
    precomputed_z = _z_score_clean_energy(
        clean_real, clean_imag, band_mask, band_bins, vad_z_threshold, vad_z_slope, eps
    )

    p_ref, p_out = _compute_vad_probs(
        clean_real,
        clean_imag,
        out_real,
        out_imag,
        band_mask,
        band_bins,
        vad_z_threshold,
        vad_z_slope,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
        _precomputed_z=precomputed_z,
    )

    vad_decrease = mx.maximum(p_ref - p_out - vad_margin, 0.0)

    proxy_frame, speech_ratio, music_gate, musicness, _, _, _ = _compute_proxy_gates(
        clean_real,
        clean_imag,
        noisy_real,
        noisy_imag,
        snr,
        band_mask,
        band_bins,
        vad_z_threshold,
        vad_z_slope,
        vad_snr_gate_db,
        vad_snr_gate_width,
        proxy_enabled=True,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
        _precomputed_z=precomputed_z,
    )

    ratio_gate = mx.sigmoid((speech_ratio - vad_threshold) / 0.1)
    gate = mx.stop_gradient(vad_decrease * ratio_gate * music_gate[:, None])

    speech_loss = _compute_speech_band_logmag_loss(
        clean_real,
        clean_imag,
        out_real,
        out_imag,
        band_mask,
        band_bins,
        gate,
        eps=eps,
        debug=debug,
        debug_ctx=debug_ctx,
    )

    return (
        speech_loss,
        vad_decrease,
        gate,
        p_ref,
        p_out,
        speech_ratio,
        musicness,
    )


def _compute_vad_eval_metrics(
    p_ref: mx.array,
    p_out: mx.array,
    vad_margin: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute VAD evaluation metrics (mean p_ref/p_out and decrease)."""
    p_ref_mean = mx.mean(p_ref)
    p_out_mean = mx.mean(p_out)
    vad_decrease = mx.mean(mx.maximum(p_ref - p_out - vad_margin, 0.0))
    return p_ref_mean, p_out_mean, vad_decrease


def _snr_bucket_name(snr_db: float) -> str:
    """Map SNR to a stable scenario bucket label."""
    if snr_db <= -20.0:
        return "very_low"
    if snr_db <= -5.0:
        return "extreme"
    if snr_db <= 5.0:
        return "low"
    if snr_db <= 20.0:
        return "mid"
    return "high"
