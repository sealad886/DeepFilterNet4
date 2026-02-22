"""Loss functions for MLX DeepFilterNet4 training.

This module provides comprehensive loss functions ported from the PyTorch
implementation, including:
- Multi-resolution spectral losses
- Mask losses (ERB-based, spectral)
- SI-SDR and SDR losses
- DF Alpha loss with LSNR-based scheduling
- Feature matching loss for GAN training

All losses are implemented using MLX primitives for efficient Apple Silicon
execution.
"""

from typing import Dict, Literal, Optional, Tuple

import mlx.core as mx

from .ops import get_window, stft

# ============================================================================
# Constants and Utilities
# ============================================================================

EPS = 1e-10

_ZERO = mx.array(0.0)


def as_complex(x: mx.array) -> Tuple[mx.array, mx.array]:
    """Convert real tensor with last dim 2 to (real, imag) tuple."""
    if x.ndim >= 1 and x.shape[-1] == 2:
        return x[..., 0], x[..., 1]
    raise ValueError(f"Expected last dim to be 2, got shape {x.shape}")


def complex_norm(real: mx.array, imag: mx.array, eps: float = EPS) -> mx.array:
    """Compute complex magnitude."""
    return mx.sqrt(real**2 + imag**2 + eps)


# ============================================================================
# Mask Functions
# ============================================================================


def wg(clean: mx.array, noisy: mx.array, eps: float = EPS) -> mx.array:
    """Wiener Gain mask.

    WG = |S|^2 / (|S|^2 + |N|^2)

    Args:
        clean: Clean signal power spectrum
        noisy: Noisy signal power spectrum

    Returns:
        Wiener gain mask
    """
    clean_pow = clean**2
    noise_pow = mx.maximum(noisy**2 - clean_pow, 0)
    return clean_pow / (clean_pow + noise_pow + eps)


def irm(clean: mx.array, noisy: mx.array, eps: float = EPS) -> mx.array:
    """Ideal Ratio Mask.

    IRM = |S| / (|S| + |N|)

    Args:
        clean: Clean signal magnitude
        noisy: Noisy signal magnitude

    Returns:
        Ideal ratio mask
    """
    noise = mx.maximum(noisy - clean, eps)
    return clean / (clean + noise + eps)


def iam(clean: mx.array, noisy: mx.array, eps: float = EPS) -> mx.array:
    """Ideal Amplitude Mask.

    IAM = |S| / |X|

    Args:
        clean: Clean signal magnitude
        noisy: Noisy signal magnitude

    Returns:
        Ideal amplitude mask
    """
    return clean / (noisy + eps)


# ============================================================================
# Spectral Losses
# ============================================================================


class SpectralLoss:
    """Multi-resolution spectral loss with gamma compression.

    Computes loss on magnitude spectra across multiple FFT resolutions.
    Supports:
    - Magnitude loss with optional gamma compression
    - Complex loss on real/imag components
    - Configurable weights per component

    Args:
        fft_sizes: Tuple of FFT sizes to use
        hop_sizes: Tuple of hop sizes (defaults to fft_size // 4)
        gamma: Magnitude compression exponent (1.0 = no compression)
        factor: Weight for magnitude loss
        factor_complex: Weight for complex loss (None to disable)
        eps: Numerical stability constant

    Example:
        >>> loss_fn = SpectralLoss(
        ...     fft_sizes=(512, 1024, 2048),
        ...     gamma=0.3,  # Strong compression
        ...     factor=1.0,
        ...     factor_complex=0.5,
        ... )
        >>> loss = loss_fn(pred_waveform, target_waveform)
    """

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Optional[Tuple[int, ...]] = None,
        gamma: float = 1.0,
        factor: float = 1.0,
        factor_complex: Optional[float] = None,
        eps: float = EPS,
    ):
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes or tuple(fft // 4 for fft in fft_sizes)
        self.gamma = gamma
        self.factor = factor
        self.factor_complex = factor_complex
        self.eps = eps

    def __call__(self, pred: mx.array, target: mx.array) -> mx.array:
        """Compute multi-resolution spectral loss.

        Args:
            pred: Predicted waveform (batch, samples) or (samples,)
            target: Target waveform (batch, samples) or (samples,)

        Returns:
            Scalar loss value
        """
        if pred.ndim == 1:
            pred = mx.expand_dims(pred, axis=0)
        if target.ndim == 1:
            target = mx.expand_dims(target, axis=0)

        total_loss = _ZERO

        for fft_size, hop_size in zip(self.fft_sizes, self.hop_sizes):
            pred_real, pred_imag = stft(pred, n_fft=fft_size, hop_length=hop_size)
            target_real, target_imag = stft(target, n_fft=fft_size, hop_length=hop_size)

            pred_mag = complex_norm(pred_real, pred_imag, self.eps)
            target_mag = complex_norm(target_real, target_imag, self.eps)

            # Apply gamma compression
            if self.gamma != 1.0:
                pred_mag_c = mx.power(pred_mag, self.gamma)
                target_mag_c = mx.power(target_mag, self.gamma)
            else:
                pred_mag_c = pred_mag
                target_mag_c = target_mag

            # Magnitude loss
            mag_loss = mx.mean((pred_mag_c - target_mag_c) ** 2) * self.factor
            total_loss = total_loss + mag_loss

            # Complex loss
            if self.factor_complex is not None and self.factor_complex > 0:
                if self.gamma != 1.0:
                    # Compressed complex: mag_c * (real/mag, imag/mag)
                    # Avoids arctan2 whose gradient explodes as O(1/eps)
                    # for near-silent bins. Direct division has stable
                    # gradients of O(1/sqrt(eps)) since mag >= sqrt(eps).
                    pred_phase = pred_mag_c / pred_mag
                    target_phase = target_mag_c / target_mag

                    pred_real_c = pred_phase * pred_real
                    pred_imag_c = pred_phase * pred_imag
                    target_real_c = target_phase * target_real
                    target_imag_c = target_phase * target_imag
                else:
                    pred_real_c = pred_real
                    pred_imag_c = pred_imag
                    target_real_c = target_real
                    target_imag_c = target_imag

                complex_loss = (
                    mx.mean((pred_real_c - target_real_c) ** 2) + mx.mean((pred_imag_c - target_imag_c) ** 2)
                ) * self.factor_complex
                total_loss = total_loss + complex_loss

        return total_loss / len(self.fft_sizes)


class FusedSpectralLoss:
    """Multi-resolution spectral loss with fused STFT frontend.

    Reduces kernel launches by:
    1. Pre-computing windows at init time (not per-call)
    2. Using mx.compile on the inner loss computation
    3. Computing all resolutions' STFTs before loss accumulation
       (allows MLX graph-level fusion of independent operations)

    Args:
        fft_sizes: Tuple of FFT sizes to use
        hop_sizes: Tuple of hop sizes (defaults to fft_size // 4)
        gamma: Magnitude compression exponent (1.0 = no compression)
        factor: Weight for magnitude loss
        factor_complex: Weight for complex loss (None to disable)
        eps: Numerical stability constant

    Example:
        >>> loss_fn = FusedSpectralLoss(
        ...     fft_sizes=(512, 1024, 2048),
        ...     gamma=0.3,
        ...     factor=1.0,
        ...     factor_complex=0.5,
        ... )
        >>> loss = loss_fn(pred_waveform, target_waveform)
    """

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Optional[Tuple[int, ...]] = None,
        gamma: float = 1.0,
        factor: float = 1.0,
        factor_complex: Optional[float] = None,
        eps: float = EPS,
    ):
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes or tuple(fft // 4 for fft in fft_sizes)
        self.gamma = gamma
        self.factor = factor
        self.factor_complex = factor_complex
        self.eps = eps

        self._windows = [get_window("sqrt_hann", fft_size) for fft_size in fft_sizes]

        self._compiled_loss = mx.compile(self._compute_loss)

    @staticmethod
    def _stft_inline(x: mx.array, n_fft: int, hop_length: int, window: mx.array) -> Tuple[mx.array, mx.array]:
        """Compute STFT inline without function-call overhead."""
        pad_amount = n_fft // 2
        x_padded = mx.pad(x, [(0, 0), (pad_amount, pad_amount)])
        num_samples = x_padded.shape[1]
        num_frames = (num_samples - n_fft) // hop_length + 1
        frame_starts = mx.arange(num_frames) * hop_length
        offsets = mx.arange(n_fft)
        indices = frame_starts[:, None] + offsets[None, :]
        frames = mx.take(x_padded, indices.flatten(), axis=1).reshape(x_padded.shape[0], num_frames, n_fft)
        frames = frames * window
        fft_out = mx.fft.rfft(frames, axis=-1)
        return mx.real(fft_out), mx.imag(fft_out)

    def _compute_loss(self, pred: mx.array, target: mx.array) -> mx.array:
        """Core loss computation — designed for mx.compile."""
        if pred.ndim == 1:
            pred = mx.expand_dims(pred, axis=0)
        if target.ndim == 1:
            target = mx.expand_dims(target, axis=0)

        total_loss = _ZERO

        for i, (fft_size, hop_size) in enumerate(zip(self.fft_sizes, self.hop_sizes)):
            window = self._windows[i]

            pred_real, pred_imag = self._stft_inline(pred, fft_size, hop_size, window)
            target_real, target_imag = self._stft_inline(target, fft_size, hop_size, window)

            pred_mag = mx.sqrt(pred_real**2 + pred_imag**2 + self.eps)
            target_mag = mx.sqrt(target_real**2 + target_imag**2 + self.eps)

            if self.gamma != 1.0:
                pred_mag_c = mx.power(pred_mag, self.gamma)
                target_mag_c = mx.power(target_mag, self.gamma)
            else:
                pred_mag_c = pred_mag
                target_mag_c = target_mag

            mag_loss = mx.mean((pred_mag_c - target_mag_c) ** 2) * self.factor
            total_loss = total_loss + mag_loss

            if self.factor_complex is not None and self.factor_complex > 0:
                if self.gamma != 1.0:
                    # Compressed complex: mag_c * (real/mag, imag/mag)
                    # Avoids arctan2 whose gradient explodes as O(1/eps)
                    # for near-silent bins. Direct division has stable
                    # gradients of O(1/sqrt(eps)) since mag >= sqrt(eps).
                    pred_phase = pred_mag_c / pred_mag
                    target_phase = target_mag_c / target_mag

                    pred_real_c = pred_phase * pred_real
                    pred_imag_c = pred_phase * pred_imag
                    target_real_c = target_phase * target_real
                    target_imag_c = target_phase * target_imag
                else:
                    pred_real_c = pred_real
                    pred_imag_c = pred_imag
                    target_real_c = target_real
                    target_imag_c = target_imag

                complex_loss = (
                    mx.mean((pred_real_c - target_real_c) ** 2) + mx.mean((pred_imag_c - target_imag_c) ** 2)
                ) * self.factor_complex
                total_loss = total_loss + complex_loss

        return total_loss / len(self.fft_sizes)

    def __call__(self, pred: mx.array, target: mx.array) -> mx.array:
        """Compute multi-resolution spectral loss (compiled path).

        Args:
            pred: Predicted waveform (batch, samples) or (samples,)
            target: Target waveform (batch, samples) or (samples,)

        Returns:
            Scalar loss value
        """
        return self._compiled_loss(pred, target)


class MaskLoss:
    """ERB-based mask loss.

    Computes loss between predicted and target masks in the ERB domain.
    Supports different mask types (WG, IRM, IAM) and optional temporal
    smoothing loss.

    Args:
        erb_fb: ERB filterbank matrix (n_erb, n_freq)
        mask_type: Target mask type ("wg", "irm", "iam")
        factor: Loss weight
        f_temporal: Factor for temporal smoothing loss (None to disable)
        gamma: Mask compression exponent

    Example:
        >>> erb_fb = erb_filterbank(sr=48000, n_fft=960, n_erb=32)
        >>> loss_fn = MaskLoss(erb_fb, mask_type="wg", factor=1.0)
        >>> loss = loss_fn(pred_mask, clean_erb, noisy_erb)
    """

    MASK_FNS = {"wg": wg, "irm": irm, "iam": iam}

    def __init__(
        self,
        erb_fb: mx.array,
        mask_type: Literal["wg", "irm", "iam"] = "wg",
        factor: float = 1.0,
        f_temporal: Optional[float] = None,
        gamma: float = 0.6,
    ):
        self.erb_fb = erb_fb
        self.mask_fn = self.MASK_FNS[mask_type]
        self.factor = factor
        self.f_temporal = f_temporal
        self.gamma = gamma

    def __call__(
        self,
        pred_mask: mx.array,
        clean_erb: mx.array,
        noisy_erb: mx.array,
    ) -> mx.array:
        """Compute mask loss.

        Args:
            pred_mask: Predicted ERB mask (batch, time, n_erb)
            clean_erb: Clean ERB features (batch, time, n_erb)
            noisy_erb: Noisy ERB features (batch, time, n_erb)

        Returns:
            Scalar loss value
        """
        # Compute target mask
        target_mask = self.mask_fn(clean_erb, noisy_erb)

        # Apply gamma compression to both
        if self.gamma != 1.0:
            pred_mask = mx.power(mx.maximum(pred_mask, EPS), self.gamma)
            target_mask = mx.power(mx.maximum(target_mask, EPS), self.gamma)

        # Main mask loss (L2)
        loss = mx.mean((pred_mask - target_mask) ** 2) * self.factor

        # Optional temporal smoothing loss
        if self.f_temporal is not None and self.f_temporal > 0:
            pred_diff = pred_mask[:, 1:] - pred_mask[:, :-1]
            target_diff = target_mask[:, 1:] - target_mask[:, :-1]
            temporal_loss = mx.mean((pred_diff - target_diff) ** 2) * self.f_temporal
            loss = loss + temporal_loss

        return loss


# ============================================================================
# SDR Losses
# ============================================================================


def si_sdr(pred: mx.array, target: mx.array, eps: float = EPS) -> mx.array:
    """Scale-Invariant Signal-to-Distortion Ratio.

    SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)

    where:
        s_target = (<pred, target> / ||target||^2) * target
        e_noise = pred - s_target

    Args:
        pred: Predicted signal (batch, samples) or (samples,)
        target: Target signal (same shape)
        eps: Numerical stability constant

    Returns:
        SI-SDR in dB (higher is better)
    """
    if pred.dtype != mx.float32:
        pred = pred.astype(mx.float32)
    if target.dtype != mx.float32:
        target = target.astype(mx.float32)

    if pred.ndim == 1:
        pred = mx.expand_dims(pred, axis=0)
    if target.ndim == 1:
        target = mx.expand_dims(target, axis=0)

    # Remove mean
    pred = pred - mx.mean(pred, axis=-1, keepdims=True)
    target = target - mx.mean(target, axis=-1, keepdims=True)

    # Compute scaling factor
    dot = mx.sum(pred * target, axis=-1, keepdims=True)
    s_target_norm = mx.sum(target**2, axis=-1, keepdims=True) + eps
    s_target = (dot / s_target_norm) * target

    # Compute noise
    e_noise = pred - s_target

    # SI-SDR in dB
    s_target_pow = mx.sum(s_target**2, axis=-1) + eps
    e_noise_pow = mx.sum(e_noise**2, axis=-1) + eps

    si_sdr_val = 10 * mx.log10(s_target_pow / e_noise_pow)

    return mx.mean(si_sdr_val)


class SiSdrLoss:
    """Negative SI-SDR loss for training.

    Returns negative SI-SDR so that minimizing the loss maximizes SI-SDR.

    Args:
        factor: Loss weight

    Example:
        >>> loss_fn = SiSdrLoss(factor=1.0)
        >>> loss = loss_fn(pred_waveform, target_waveform)
    """

    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def __call__(self, pred: mx.array, target: mx.array) -> mx.array:
        """Compute negative SI-SDR loss."""
        return -si_sdr(pred, target) * self.factor


class SegmentalSiSdrLoss:
    """Segmental SI-SDR loss.

    Computes SI-SDR over overlapping segments and averages.
    Better captures local quality variations.

    Args:
        segment_size: Size of each segment in samples
        overlap: Overlap ratio between segments
        factor: Loss weight
    """

    def __init__(
        self,
        segment_size: int = 960,
        overlap: float = 0.5,
        factor: float = 1.0,
    ):
        self.segment_size = segment_size
        self.hop_size = int(segment_size * (1 - overlap))
        self.factor = factor

    def __call__(self, pred: mx.array, target: mx.array) -> mx.array:
        """Compute segmental SI-SDR loss using vectorized segmentation."""
        if pred.dtype != mx.float32:
            pred = pred.astype(mx.float32)
        if target.dtype != mx.float32:
            target = target.astype(mx.float32)

        if pred.ndim == 1:
            pred = mx.expand_dims(pred, axis=0)
        if target.ndim == 1:
            target = mx.expand_dims(target, axis=0)

        batch, samples = pred.shape

        # Fall back to global SI-SDR when signal is shorter than one segment
        if samples < self.segment_size:
            return -si_sdr(pred, target) * self.factor

        n_segments = max(1, (samples - self.segment_size) // self.hop_size + 1)

        # Vectorized segmentation using mx.take
        segment_starts = mx.arange(n_segments) * self.hop_size  # [n_segments]
        offsets = mx.arange(self.segment_size)  # [segment_size]
        indices = segment_starts[:, None] + offsets[None, :]  # [n_segments, segment_size]

        # Clip indices to valid range (handles edge cases)
        max_idx = samples - 1
        indices = mx.clip(indices, 0, max_idx)
        flat_indices = indices.flatten()

        # Extract all segments at once: [batch, n_segments * segment_size]
        seg_pred_flat = mx.take(pred, flat_indices, axis=1)
        seg_target_flat = mx.take(target, flat_indices, axis=1)

        # Reshape to [batch, n_segments, segment_size]
        seg_pred = seg_pred_flat.reshape(batch, n_segments, self.segment_size)
        seg_target = seg_target_flat.reshape(batch, n_segments, self.segment_size)

        # Per-segment mean removal (required for true SI-SDR)
        seg_pred = seg_pred - mx.mean(seg_pred, axis=-1, keepdims=True)
        seg_target = seg_target - mx.mean(seg_target, axis=-1, keepdims=True)

        # Compute SI-SDR for all segments at once
        # s_target projection: <s_hat, s_target> / ||s_target||^2 * s_target
        dot = mx.sum(seg_pred * seg_target, axis=-1, keepdims=True)  # [batch, n_segments, 1]
        s_target_pow = mx.sum(seg_target**2, axis=-1, keepdims=True) + EPS
        s_target_proj = (dot / s_target_pow) * seg_target

        # Noise
        noise = seg_pred - s_target_proj

        # SI-SDR per segment: [batch, n_segments]
        # eps in both numerator and denominator for consistent silence handling
        signal_pow = mx.sum(s_target_proj**2, axis=-1) + EPS
        noise_pow = mx.sum(noise**2, axis=-1) + EPS
        sisdr = 10 * mx.log10(signal_pow / noise_pow)

        # Mean over segments and batch
        return -mx.mean(sisdr) * self.factor


class SdrLoss:
    """SDR (Signal-to-Distortion Ratio) loss.

    Standard SDR without scale invariance.

    Args:
        factor: Loss weight
    """

    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def __call__(self, pred: mx.array, target: mx.array) -> mx.array:
        """Compute negative SDR loss."""
        if pred.dtype != mx.float32:
            pred = pred.astype(mx.float32)
        if target.dtype != mx.float32:
            target = target.astype(mx.float32)

        if pred.ndim == 1:
            pred = mx.expand_dims(pred, axis=0)
        if target.ndim == 1:
            target = mx.expand_dims(target, axis=0)

        noise = pred - target
        signal_pow = mx.sum(target**2, axis=-1) + EPS
        noise_pow = mx.sum(noise**2, axis=-1) + EPS

        sdr = 10 * mx.log10(signal_pow / noise_pow)
        return -mx.mean(sdr) * self.factor


# ============================================================================
# DF Alpha Loss
# ============================================================================


class DfAlphaLoss:
    """DF Alpha scheduling loss.

    Controls the deep filtering alpha parameter based on local SNR.
    High LSNR -> high alpha (more DF), Low LSNR -> low alpha (less DF)

    The alpha schedule is:
        - lsnr < lsnr_thresh_low: alpha = 0
        - lsnr > lsnr_thresh_high: alpha = 1
        - otherwise: linear interpolation

    Args:
        factor: Loss weight for DF alpha prediction
        lsnr_thresh: Tuple of (low, high) LSNR thresholds in dB
            Default (-10, -7.5) means:
            - Below -10 dB: fully suppress DF
            - Above -7.5 dB: fully enable DF
    """

    def __init__(
        self,
        factor: float = 1.0,
        lsnr_thresh: Tuple[float, float] = (-10.0, -7.5),
    ):
        self.factor = factor
        self.lsnr_lo = lsnr_thresh[0]
        self.lsnr_hi = lsnr_thresh[1]

    def compute_target_alpha(self, lsnr: mx.array) -> mx.array:
        """Compute target alpha from LSNR.

        Args:
            lsnr: Local SNR values in dB (batch, time) or (batch, time, 1)

        Returns:
            Target alpha values in [0, 1]
        """
        if lsnr.ndim == 3:
            lsnr = mx.squeeze(lsnr, axis=-1)

        # Linear interpolation between thresholds
        alpha = (lsnr - self.lsnr_lo) / (self.lsnr_hi - self.lsnr_lo + EPS)
        alpha = mx.clip(alpha, 0.0, 1.0)

        return alpha

    def __call__(
        self,
        pred_alpha: mx.array,
        target_lsnr: mx.array,
    ) -> mx.array:
        """Compute DF alpha loss.

        Args:
            pred_alpha: Predicted alpha values (batch, time) or (batch, time, 1)
            target_lsnr: Target LSNR values in dB

        Returns:
            Scalar loss value
        """
        if pred_alpha.ndim == 3:
            pred_alpha = mx.squeeze(pred_alpha, axis=-1)

        target_alpha = self.compute_target_alpha(target_lsnr)

        # MSE loss on alpha
        loss = mx.mean((pred_alpha - target_alpha) ** 2) * self.factor
        return loss


# ============================================================================
# Feature Matching Loss (for GAN)
# ============================================================================


class FeatureMatchingLoss:
    """Feature matching loss for GAN training stability.

    Computes L1 distance between discriminator feature maps from real
    and generated samples. Provides stable gradients to the generator.

    Args:
        factor: Loss weight

    Example:
        >>> loss_fn = FeatureMatchingLoss(factor=2.0)
        >>> loss = loss_fn(real_fmaps, fake_fmaps)
    """

    def __init__(self, factor: float = 2.0):
        self.factor = factor

    def __call__(
        self,
        real_fmaps: list[list[mx.array]],
        fake_fmaps: list[list[mx.array]],
    ) -> mx.array:
        """Compute feature matching loss.

        Args:
            real_fmaps: Feature maps from discriminator on real samples
                List of lists: [[disc1_layer1, disc1_layer2, ...], ...]
            fake_fmaps: Feature maps from discriminator on fake samples

        Returns:
            Feature matching loss (scalar)
        """
        # Collect per-layer means instead of sequential scalar accumulation.
        # This breaks the chain dependency in the compiled graph, allowing
        # independent mx.mean(mx.abs(...)) ops to be scheduled in parallel.
        means: list[mx.array] = []
        for real_disc, fake_disc in zip(real_fmaps, fake_fmaps):
            for real_feat, fake_feat in zip(real_disc, fake_disc):
                means.append(mx.mean(mx.abs(real_feat - fake_feat)))

        if not means:
            return _ZERO
        return mx.mean(mx.stack(means)) * self.factor


# ============================================================================
# GAN Losses
# ============================================================================


def discriminator_loss(
    real_outputs: list[mx.array],
    fake_outputs: list[mx.array],
) -> Tuple[mx.array, mx.array, mx.array]:
    """Discriminator loss for GAN training.

    Uses hinge loss for stable training.

    Args:
        real_outputs: Discriminator outputs on real samples
        fake_outputs: Discriminator outputs on fake/generated samples

    Returns:
        Tuple of (total_loss, real_loss, fake_loss)
    """
    n_disc = len(real_outputs)
    if n_disc == 0:
        return _ZERO, _ZERO, _ZERO
    # Collect per-discriminator losses, then stack+mean to break chain dependency
    real_means = [mx.mean(mx.maximum(1 - r, 0)) for r in real_outputs]
    fake_means = [mx.mean(mx.maximum(1 + f, 0)) for f in fake_outputs]
    real_loss = mx.mean(mx.stack(real_means))
    fake_loss = mx.mean(mx.stack(fake_means))
    total_loss = real_loss + fake_loss

    return total_loss, real_loss, fake_loss


def generator_loss(fake_outputs: list[mx.array]) -> mx.array:
    """Generator adversarial loss.

    Uses hinge loss variant where generator wants discriminator to
    output high values (close to real).

    Args:
        fake_outputs: Discriminator outputs on generated samples

    Returns:
        Generator loss (scalar)
    """
    n = len(fake_outputs)
    if n == 0:
        return _ZERO
    # Collect per-discriminator losses, then stack+mean to break chain dependency
    means = [mx.mean(-f) for f in fake_outputs]
    return mx.mean(mx.stack(means))


# ============================================================================
# Combined Loss Functions
# ============================================================================


class CombinedLoss:
    """Combined loss function for DeepFilterNet4 training.

    Combines multiple loss components with configurable weights:
    - Multi-resolution spectral loss
    - SI-SDR loss
    - Mask loss (optional)
    - DF Alpha loss (optional)

    Args:
        spectral_loss: SpectralLoss instance or config
        sisdr_factor: Weight for SI-SDR loss
        mask_loss: Optional MaskLoss instance
        alpha_loss: Optional DfAlphaLoss instance

    Example:
        >>> loss_fn = CombinedLoss(
        ...     spectral_loss=SpectralLoss(gamma=0.3),
        ...     sisdr_factor=0.5,
        ... )
        >>> loss = loss_fn(pred_wav, target_wav)
    """

    def __init__(
        self,
        spectral_loss: Optional[SpectralLoss] = None,
        sisdr_factor: float = 0.5,
        mask_loss: Optional[MaskLoss] = None,
        alpha_loss: Optional[DfAlphaLoss] = None,
    ):
        self.spectral_loss = spectral_loss or SpectralLoss()
        self.sisdr_loss = SiSdrLoss(factor=sisdr_factor) if sisdr_factor > 0 else None
        self.mask_loss = mask_loss
        self.alpha_loss = alpha_loss

    def __call__(
        self,
        pred_wav: mx.array,
        target_wav: mx.array,
        pred_mask: Optional[mx.array] = None,
        clean_erb: Optional[mx.array] = None,
        noisy_erb: Optional[mx.array] = None,
        pred_alpha: Optional[mx.array] = None,
        target_lsnr: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Compute combined loss with breakdown.

        Args:
            pred_wav: Predicted waveform
            target_wav: Target waveform
            pred_mask: Optional predicted ERB mask
            clean_erb: Optional clean ERB features
            noisy_erb: Optional noisy ERB features
            pred_alpha: Optional predicted DF alpha
            target_lsnr: Optional target LSNR

        Returns:
            Tuple of (total_loss, loss_breakdown_dict) where dict values are lazy mx.array
        """
        losses: Dict[str, mx.array] = {}
        total_loss = _ZERO

        # Spectral loss
        spec_loss = self.spectral_loss(pred_wav, target_wav)
        losses["spectral"] = spec_loss
        total_loss = total_loss + spec_loss

        # SI-SDR loss
        if self.sisdr_loss is not None:
            sisdr = self.sisdr_loss(pred_wav, target_wav)
            losses["sisdr"] = sisdr
            total_loss = total_loss + sisdr

        # Mask loss
        if self.mask_loss is not None and pred_mask is not None:
            if clean_erb is not None and noisy_erb is not None:
                mask_l = self.mask_loss(pred_mask, clean_erb, noisy_erb)
                losses["mask"] = mask_l
                total_loss = total_loss + mask_l

        # Alpha loss
        if self.alpha_loss is not None and pred_alpha is not None:
            if target_lsnr is not None:
                alpha_l = self.alpha_loss(pred_alpha, target_lsnr)
                losses["alpha"] = alpha_l
                total_loss = total_loss + alpha_l

        losses["total"] = total_loss
        return total_loss, losses
