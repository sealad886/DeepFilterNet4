"""Numeric stability constants and pure-MLX helpers for complex-spectral ops.

Provides the split epsilon policy for the training pipeline:

- ``_EPS_F`` (1e-10): used for complex-magnitude and log-magnitude paths
  where a lower floor reduces artificial bias on quiet frequency bins.
- ``_BAND_ENERGY_EPS_F`` (1e-8): used for band-energy reductions and
  denominator protection where a slightly higher floor is appropriate.

Helper functions implement the same math that was formerly dispatched
through custom Metal kernels but using standard MLX ops, which
``mx.compile`` fuses efficiently.
"""

from __future__ import annotations

import mlx.core as mx

# Magnitude/log-magnitude stability constant.
# Tuned via adversarial quiet-bin bias analysis: 1e-10 materially reduces
# artificial floors for very low-energy bins while remaining numerically safe
# for the complex-magnitude and log1p-magnitude paths.
_EPS_F = 1e-10
# Reduction/statistics epsilon used by band-energy and log-energy style
# reductions. This remains higher because it protects denominators and log10
# energy reductions rather than directly flooring complex magnitudes.
_BAND_ENERGY_EPS_F = 1e-8


def complex_mag(real: mx.array, imag: mx.array, eps: float = _EPS_F) -> mx.array:
    """Return elementwise complex magnitude with a stability floor.

    Args:
        real: Real component of a complex-valued tensor.
        imag: Imaginary component broadcast-compatible with ``real``.
        eps: Non-negative stability term added inside the square root.

    Returns:
        Elementwise ``sqrt(real² + imag² + eps)`` with the broadcasted input
        shape. The output dtype follows MLX elementwise promotion rules, so
        reduced-precision inputs remain reduced precision unless callers cast
        first.
    """
    return mx.sqrt(real * real + imag * imag + eps)


def log1p_mag(real: mx.array, imag: mx.array, eps: float = _EPS_F) -> mx.array:
    """Return ``log1p`` of the elementwise complex magnitude.

    Args:
        real: Real component of a complex-valued tensor.
        imag: Imaginary component broadcast-compatible with ``real``.
        eps: Non-negative stability term forwarded to :func:`complex_mag`.

    Returns:
        Elementwise ``log1p(sqrt(real² + imag² + eps))`` with the broadcasted
        input shape. The output dtype matches the dtype produced by
        :func:`complex_mag`, so callers that need float32 numerics should cast
        before calling this helper.
    """
    return mx.log1p(complex_mag(real, imag, eps))


def band_energy(
    real: mx.array,
    imag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    eps: float = _BAND_ENERGY_EPS_F,
) -> tuple[mx.array, mx.array]:
    """Compute masked band energy and log10 band energy.

    Args:
        real: Complex real part, shape (B, T, F).
        imag: Complex imaginary part, shape (B, T, F).
        band_mask: Float mask broadcast-compatible with ``real``/``imag`` over
            the trailing frequency dimension.
        band_bins: Positive number of active bins represented by
            ``band_mask``.
        eps: Numerical stability constant.

    Returns:
        (band, log_band) where band = sum_f(power*mask)/band_bins
        and log_band = log10(band + eps), both shape ``real.shape[:-1]``. The
        reduction is accumulated in float32 for stable mixed-precision use.

    Raises:
        ValueError: If ``band_bins`` is not strictly positive.
    """
    if band_bins <= 0:
        raise ValueError(f"band_bins must be positive, got {band_bins!r}")

    real_f32 = real.astype(mx.float32) if real.dtype != mx.float32 else real
    imag_f32 = imag.astype(mx.float32) if imag.dtype != mx.float32 else imag
    mask_f32 = band_mask.astype(mx.float32) if band_mask.dtype != mx.float32 else band_mask
    power = real_f32 * real_f32 + imag_f32 * imag_f32
    band = mx.sum(power * mask_f32, axis=-1) / (band_bins + eps)
    log_band = mx.log10(band + eps)
    return band, log_band
