"""Fused Metal GPU kernels for training-hot-path operations.

Provides custom ``mx.fast.metal_kernel`` primitives that fuse multi-op
chains into single GPU dispatches, eliminating intermediate buffer
materializations that ``mx.compile`` cannot elide across reduction
boundaries.

Kernels
-------
fused_log1p_mag
    ``log1p(sqrt(r² + i² + eps))`` — forward + VJP via ``mx.custom_function``.
    Replaces the elementwise chain in ``_log1p_mag`` (training_losses.py).

fused_complex_mag
    ``sqrt(r² + i² + eps)`` — forward + VJP via ``mx.custom_function``.
    Replaces ``complex_norm`` / inline magnitude in spectral_loss.

fused_band_energy
    ``(band, log10_band)`` from ``sum_f((r²+i²)*mask[f]) / band_bins``
    — fuses power computation, masked reduction over frequency,
    and log10 into a single kernel.  Eliminates the (B,T,F)
    intermediate that standard MLX ops must materialize.
    Stop-gradient only (no VJP needed).
"""

from __future__ import annotations

import mlx.core as mx

# Numerical stability constant (matches training_losses._EPS)
_EPS_F = 1e-10

# ====================================================================
# Kernel 1: fused_log1p_mag  —  log1p(sqrt(r² + i² + eps))
# ====================================================================

_LOG1P_MAG_FWD_SRC = """
    uint elem = thread_position_in_grid.x;
    T r = real[elem];
    T i = imag[elem];
    T mag = metal::sqrt(r * r + i * i + T(1e-10));
    out[elem] = metal::log(T(1) + mag);
"""

_LOG1P_MAG_BWD_SRC = """
    uint elem = thread_position_in_grid.x;
    T g = grad[elem];
    T r = real[elem];
    T i = imag[elem];
    T mag = metal::sqrt(r * r + i * i + T(1e-10));
    T inv = g / (mag * (T(1) + mag));
    grad_real[elem] = inv * r;
    grad_imag[elem] = inv * i;
"""

_log1p_mag_fwd_kernel = mx.fast.metal_kernel(
    name="fused_log1p_mag_fwd",
    input_names=["real", "imag"],
    output_names=["out"],
    source=_LOG1P_MAG_FWD_SRC,
)

_log1p_mag_bwd_kernel = mx.fast.metal_kernel(
    name="fused_log1p_mag_bwd",
    input_names=["grad", "real", "imag"],
    output_names=["grad_real", "grad_imag"],
    source=_LOG1P_MAG_BWD_SRC,
)


@mx.custom_function
def fused_log1p_mag(real: mx.array, imag: mx.array) -> mx.array:
    """Compute ``log1p(sqrt(real² + imag² + eps))`` in a single GPU kernel.

    Supports reverse-mode autodiff via a fused backward kernel.
    """
    shape = real.shape
    n = real.size
    if n == 0:
        return mx.zeros(shape, dtype=real.dtype)
    out = _log1p_mag_fwd_kernel(
        inputs=[real.reshape(-1), imag.reshape(-1)],
        template=[("T", real.dtype)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[real.dtype],
    )[0]
    return out.reshape(shape)


@fused_log1p_mag.vjp
def _fused_log1p_mag_vjp(primals, cotangent, output):
    real, imag = primals
    shape = real.shape
    n = real.size
    if n == 0:
        return mx.zeros(shape, dtype=real.dtype), mx.zeros(shape, dtype=real.dtype)
    grad_r, grad_i = _log1p_mag_bwd_kernel(
        inputs=[cotangent.reshape(-1), real.reshape(-1), imag.reshape(-1)],
        template=[("T", real.dtype)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(n,), (n,)],
        output_dtypes=[real.dtype, real.dtype],
    )
    return grad_r.reshape(shape), grad_i.reshape(shape)


# ====================================================================
# Kernel 2: fused_complex_mag  —  sqrt(r² + i² + eps)
# ====================================================================

_COMPLEX_MAG_FWD_SRC = """
    uint elem = thread_position_in_grid.x;
    T r = real[elem];
    T i = imag[elem];
    out[elem] = metal::sqrt(r * r + i * i + T(1e-10));
"""

_COMPLEX_MAG_BWD_SRC = """
    uint elem = thread_position_in_grid.x;
    T g = grad[elem];
    T r = real[elem];
    T i = imag[elem];
    T mag = metal::sqrt(r * r + i * i + T(1e-10));
    grad_real[elem] = g * r / mag;
    grad_imag[elem] = g * i / mag;
"""

_complex_mag_fwd_kernel = mx.fast.metal_kernel(
    name="fused_complex_mag_fwd",
    input_names=["real", "imag"],
    output_names=["out"],
    source=_COMPLEX_MAG_FWD_SRC,
)

_complex_mag_bwd_kernel = mx.fast.metal_kernel(
    name="fused_complex_mag_bwd",
    input_names=["grad", "real", "imag"],
    output_names=["grad_real", "grad_imag"],
    source=_COMPLEX_MAG_BWD_SRC,
)


@mx.custom_function
def fused_complex_mag(real: mx.array, imag: mx.array) -> mx.array:
    """Compute ``sqrt(real² + imag² + eps)`` in a single GPU kernel.

    Supports reverse-mode autodiff via a fused backward kernel.
    """
    shape = real.shape
    n = real.size
    if n == 0:
        return mx.zeros(shape, dtype=real.dtype)
    out = _complex_mag_fwd_kernel(
        inputs=[real.reshape(-1), imag.reshape(-1)],
        template=[("T", real.dtype)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[real.dtype],
    )[0]
    return out.reshape(shape)


@fused_complex_mag.vjp
def _fused_complex_mag_vjp(primals, cotangent, output):
    real, imag = primals
    shape = real.shape
    n = real.size
    if n == 0:
        return mx.zeros(shape, dtype=real.dtype), mx.zeros(shape, dtype=real.dtype)
    grad_r, grad_i = _complex_mag_bwd_kernel(
        inputs=[cotangent.reshape(-1), real.reshape(-1), imag.reshape(-1)],
        template=[("T", real.dtype)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(n,), (n,)],
        output_dtypes=[real.dtype, real.dtype],
    )
    return grad_r.reshape(shape), grad_i.reshape(shape)


# ====================================================================
# Kernel 3: fused_band_energy
# Fuses: power(B,T,F) → masked_sum(B,T) → log10(B,T)
# Eliminates the (B,T,F) intermediate that standard ops materialize.
# Forward-only (used exclusively in stop-gradient paths).
# ====================================================================

_BAND_ENERGY_SRC = """
    uint elem = thread_position_in_grid.x;
    int F = real_shape[2];
    int offset = elem * F;

    T acc = T(0);
    for (int f = 0; f < F; f++) {
        T r = real[offset + f];
        T i = imag[offset + f];
        acc += (r * r + i * i) * mask[f];
    }
    T bins = params[0];
    T eps_val = params[1];
    T band = acc / (bins + eps_val);
    band_out[elem] = band;
    log_out[elem] = metal::log10(band + eps_val);
"""

_band_energy_kernel = mx.fast.metal_kernel(
    name="fused_band_energy",
    input_names=["real", "imag", "mask", "params"],
    output_names=["band_out", "log_out"],
    source=_BAND_ENERGY_SRC,
)


def fused_band_energy(
    real: mx.array,
    imag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    eps: float = _EPS_F,
) -> tuple[mx.array, mx.array]:
    """Fused power → masked-sum-over-F → log10 in a single GPU kernel.

    Eliminates the (B,T,F) power intermediate that standard MLX ops
    must materialize before the masked reduction.

    Args:
        real: Complex real part, shape (B, T, F).
        imag: Complex imaginary part, shape (B, T, F).
        band_mask: Frequency-bin mask, shape (F,).
        band_bins: Number of active bins in the mask.
        eps: Numerical stability constant.

    Returns:
        (band, log_band) where band = sum_f(power*mask)/band_bins
        and log_band = log10(band + eps), both shape (B, T).
    """
    B, T, F = real.shape
    n_out = B * T
    if n_out == 0:
        empty = mx.zeros((B, T), dtype=real.dtype)
        return empty, empty
    params = mx.array([band_bins, eps], dtype=real.dtype)
    band_flat, log_flat = _band_energy_kernel(
        inputs=[real, imag, band_mask.reshape(-1), params],
        template=[("T", real.dtype)],
        grid=(n_out, 1, 1),
        threadgroup=(min(n_out, 256), 1, 1),
        output_shapes=[(n_out,), (n_out,)],
        output_dtypes=[real.dtype, real.dtype],
    )
    return band_flat.reshape(B, T), log_flat.reshape(B, T)


# ====================================================================
# Reference implementations for correctness verification
# ====================================================================


def _ref_log1p_mag(real: mx.array, imag: mx.array) -> mx.array:
    """Reference (standard MLX ops) for fused_log1p_mag."""
    mag = mx.sqrt(real * real + imag * imag + _EPS_F)
    return mx.log1p(mag)


def _ref_complex_mag(real: mx.array, imag: mx.array) -> mx.array:
    """Reference (standard MLX ops) for fused_complex_mag."""
    return mx.sqrt(real * real + imag * imag + _EPS_F)


def _ref_band_energy(
    real: mx.array,
    imag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    eps: float = _EPS_F,
) -> tuple[mx.array, mx.array]:
    """Reference (standard MLX ops) for fused_band_energy."""
    power = real * real + imag * imag
    band = mx.sum(power * band_mask, axis=-1) / (band_bins + eps)
    log_band = mx.log10(band + eps)
    return band, log_band
