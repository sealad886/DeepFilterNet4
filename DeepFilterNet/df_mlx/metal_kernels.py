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

from functools import lru_cache

import mlx.core as mx

# Numerical stability constant (matches training_losses._EPS and train.spectral_loss)
_EPS_F = 1e-8
_DEFAULT_LOG1P_MAG_THREADGROUP = 512
_DEFAULT_COMPLEX_MAG_THREADGROUP = 512
_COMPLEX_MAG_LARGE_WORKLOAD_THRESHOLD = 1_000_000
_BAND_ENERGY_FUSED_BT_THRESHOLD = 1800

# ====================================================================
# Kernel 1: fused_log1p_mag  —  log1p(sqrt(r² + i² + eps))
# ====================================================================

_LOG1P_MAG_FWD_SRC = """
    uint elem = thread_position_in_grid.x;
    T r = real[elem];
    T i = imag[elem];
    T eps_val = params[0];
    T mag = metal::sqrt(r * r + i * i + eps_val);
    out[elem] = metal::log(T(1) + mag);
"""

_LOG1P_MAG_BWD_SRC = """
    uint elem = thread_position_in_grid.x;
    T g = grad[elem];
    T r = real[elem];
    T i = imag[elem];
    T eps_val = params[0];
    T mag = metal::sqrt(r * r + i * i + eps_val);
    T inv = g / (mag * (T(1) + mag));
    grad_real[elem] = inv * r;
    grad_imag[elem] = inv * i;
"""

_log1p_mag_fwd_kernel = mx.fast.metal_kernel(
    name="fused_log1p_mag_fwd",
    input_names=["real", "imag", "params"],
    output_names=["out"],
    source=_LOG1P_MAG_FWD_SRC,
)

_log1p_mag_bwd_kernel = mx.fast.metal_kernel(
    name="fused_log1p_mag_bwd",
    input_names=["grad", "real", "imag", "params"],
    output_names=["grad_real", "grad_imag"],
    source=_LOG1P_MAG_BWD_SRC,
)


def _resolve_threadgroup_size(n: int, preferred: int = _DEFAULT_LOG1P_MAG_THREADGROUP) -> int:
    """Return a safe 1D threadgroup size for flat elementwise kernels."""
    if n <= 0:
        return 1
    return max(1, min(n, preferred))


def _make_params(real: mx.array, eps: float) -> mx.array:
    """Build a dtype-aligned scalar parameter buffer for Metal kernels."""
    return mx.array([eps], dtype=real.dtype)


def _native_complex_mag(real: mx.array, imag: mx.array, eps: float = _EPS_F) -> mx.array:
    """Native MLX reference for magnitude, kept internal for adaptive validation."""
    return mx.sqrt(real * real + imag * imag + eps)


def _native_log1p_mag(real: mx.array, imag: mx.array, eps: float = _EPS_F) -> mx.array:
    """Native MLX reference for log1p magnitude."""
    return mx.log1p(_native_complex_mag(real, imag, eps))


def _select_log1p_mag_threadgroup(real: mx.array) -> int:
    """Choose the threadgroup size for fused_log1p_mag.

    A dedicated sweep over representative DF training shapes showed the 512-wide
    launch to be the most robust default for this kernel, with no evidence that
    a native fallback is beneficial.
    """
    return _DEFAULT_LOG1P_MAG_THREADGROUP


def _select_complex_mag_threadgroup(real: mx.array) -> int:
    """Choose a threadgroup size based on measured flat-workload crossover.

    Benchmarks on representative DF training spectra showed that 512 threads are
    best below about 1M output elements, while 256 threads are more robust for
    larger workloads.
    """
    return 256 if real.size >= _COMPLEX_MAG_LARGE_WORKLOAD_THRESHOLD else _DEFAULT_COMPLEX_MAG_THREADGROUP


def _dispatch_log1p_mag_forward(
    real: mx.array,
    imag: mx.array,
    params: mx.array,
    *,
    threadgroup_size: int = _DEFAULT_LOG1P_MAG_THREADGROUP,
) -> mx.array:
    """Raw forward dispatch for fused_log1p_mag with configurable threadgroup size."""
    shape = real.shape
    n = real.size
    if n == 0:
        return mx.zeros(shape, dtype=real.dtype)
    out = _log1p_mag_fwd_kernel(
        inputs=[real.reshape(-1), imag.reshape(-1), params],
        template=[("T", real.dtype)],
        grid=(n, 1, 1),
        threadgroup=(_resolve_threadgroup_size(n, threadgroup_size), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[real.dtype],
    )[0]
    return out.reshape(shape)


def _dispatch_log1p_mag_backward(
    cotangent: mx.array,
    real: mx.array,
    imag: mx.array,
    params: mx.array,
    *,
    threadgroup_size: int = _DEFAULT_LOG1P_MAG_THREADGROUP,
) -> tuple[mx.array, mx.array]:
    """Raw backward dispatch for fused_log1p_mag with configurable threadgroup size."""
    shape = real.shape
    n = real.size
    if n == 0:
        zeros = mx.zeros(shape, dtype=real.dtype)
        return zeros, zeros
    grad_r, grad_i = _log1p_mag_bwd_kernel(
        inputs=[cotangent.reshape(-1), real.reshape(-1), imag.reshape(-1), params],
        template=[("T", real.dtype)],
        grid=(n, 1, 1),
        threadgroup=(_resolve_threadgroup_size(n, threadgroup_size), 1, 1),
        output_shapes=[(n,), (n,)],
        output_dtypes=[real.dtype, real.dtype],
    )
    return grad_r.reshape(shape), grad_i.reshape(shape)


@lru_cache(maxsize=8)
def _get_log1p_mag_impl(threadgroup_size: int):
    @mx.custom_function
    def _impl(real: mx.array, imag: mx.array, params: mx.array) -> mx.array:
        return _dispatch_log1p_mag_forward(real, imag, params, threadgroup_size=threadgroup_size)

    @_impl.vjp
    def _impl_vjp(primals, cotangent, output):
        real, imag, params = primals
        grad_r, grad_i = _dispatch_log1p_mag_backward(
            cotangent,
            real,
            imag,
            params,
            threadgroup_size=threadgroup_size,
        )
        return grad_r, grad_i, mx.zeros_like(params)

    return _impl


def fused_log1p_mag(real: mx.array, imag: mx.array, eps: float = _EPS_F) -> mx.array:
    """Compute ``log1p(sqrt(real² + imag² + eps))`` in a single GPU kernel.

    Supports reverse-mode autodiff via a fused backward kernel.
    """
    params = _make_params(real, eps)
    threadgroup_size = _select_log1p_mag_threadgroup(real)
    return _get_log1p_mag_impl(threadgroup_size)(real, imag, params)


# ====================================================================
# Kernel 2: fused_complex_mag  —  sqrt(r² + i² + eps)
# ====================================================================

_COMPLEX_MAG_FWD_SRC = """
    uint elem = thread_position_in_grid.x;
    T r = real[elem];
    T i = imag[elem];
    T eps_val = params[0];
    out[elem] = metal::sqrt(r * r + i * i + eps_val);
"""

_COMPLEX_MAG_BWD_SRC = """
    uint elem = thread_position_in_grid.x;
    T g = grad[elem];
    T r = real[elem];
    T i = imag[elem];
    T eps_val = params[0];
    T mag = metal::sqrt(r * r + i * i + eps_val);
    grad_real[elem] = g * r / mag;
    grad_imag[elem] = g * i / mag;
"""

_complex_mag_fwd_kernel = mx.fast.metal_kernel(
    name="fused_complex_mag_fwd",
    input_names=["real", "imag", "params"],
    output_names=["out"],
    source=_COMPLEX_MAG_FWD_SRC,
)

_complex_mag_bwd_kernel = mx.fast.metal_kernel(
    name="fused_complex_mag_bwd",
    input_names=["grad", "real", "imag", "params"],
    output_names=["grad_real", "grad_imag"],
    source=_COMPLEX_MAG_BWD_SRC,
)


def _dispatch_complex_mag_forward(
    real: mx.array,
    imag: mx.array,
    params: mx.array,
    *,
    threadgroup_size: int = _DEFAULT_COMPLEX_MAG_THREADGROUP,
) -> mx.array:
    """Raw forward dispatch for fused_complex_mag with configurable threadgroup size."""
    shape = real.shape
    n = real.size
    if n == 0:
        return mx.zeros(shape, dtype=real.dtype)
    out = _complex_mag_fwd_kernel(
        inputs=[real.reshape(-1), imag.reshape(-1), params],
        template=[("T", real.dtype)],
        grid=(n, 1, 1),
        threadgroup=(_resolve_threadgroup_size(n, threadgroup_size), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[real.dtype],
    )[0]
    return out.reshape(shape)


def _dispatch_complex_mag_backward(
    cotangent: mx.array,
    real: mx.array,
    imag: mx.array,
    params: mx.array,
    *,
    threadgroup_size: int = _DEFAULT_COMPLEX_MAG_THREADGROUP,
) -> tuple[mx.array, mx.array]:
    """Raw backward dispatch for fused_complex_mag with configurable threadgroup size."""
    shape = real.shape
    n = real.size
    if n == 0:
        zeros = mx.zeros(shape, dtype=real.dtype)
        return zeros, zeros
    grad_r, grad_i = _complex_mag_bwd_kernel(
        inputs=[cotangent.reshape(-1), real.reshape(-1), imag.reshape(-1), params],
        template=[("T", real.dtype)],
        grid=(n, 1, 1),
        threadgroup=(_resolve_threadgroup_size(n, threadgroup_size), 1, 1),
        output_shapes=[(n,), (n,)],
        output_dtypes=[real.dtype, real.dtype],
    )
    return grad_r.reshape(shape), grad_i.reshape(shape)


@lru_cache(maxsize=8)
def _get_complex_mag_impl(threadgroup_size: int):
    @mx.custom_function
    def _impl(real: mx.array, imag: mx.array, params: mx.array) -> mx.array:
        return _dispatch_complex_mag_forward(real, imag, params, threadgroup_size=threadgroup_size)

    @_impl.vjp
    def _impl_vjp(primals, cotangent, output):
        real, imag, params = primals
        grad_r, grad_i = _dispatch_complex_mag_backward(
            cotangent,
            real,
            imag,
            params,
            threadgroup_size=threadgroup_size,
        )
        return grad_r, grad_i, mx.zeros_like(params)

    return _impl


def fused_complex_mag(real: mx.array, imag: mx.array, eps: float = _EPS_F) -> mx.array:
    """Compute ``sqrt(real² + imag² + eps)`` in a single GPU kernel.

    Supports reverse-mode autodiff via a fused backward kernel.
    """
    params = _make_params(real, eps)
    threadgroup_size = _select_complex_mag_threadgroup(real)
    return _get_complex_mag_impl(threadgroup_size)(real, imag, params)


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

    float acc = 0.0f;
    for (int f = 0; f < F; f++) {
        float r = float(real[offset + f]);
        float i = float(imag[offset + f]);
        float m = float(mask[f]);
        acc += (r * r + i * i) * m;
    }
    float bins = params[0];
    float eps_val = params[1];
    float band = acc / (bins + eps_val);
    band_out[elem] = band;
    log_out[elem] = metal::log10(band + eps_val);
"""

_band_energy_kernel = mx.fast.metal_kernel(
    name="fused_band_energy",
    input_names=["real", "imag", "mask", "params"],
    output_names=["band_out", "log_out"],
    source=_BAND_ENERGY_SRC,
)


def _native_band_energy(
    real: mx.array,
    imag: mx.array,
    band_mask: mx.array,
    band_bins: float,
    eps: float = _EPS_F,
) -> tuple[mx.array, mx.array]:
    """Numerically stable native MLX implementation used for tiny workloads."""
    real_f32 = real.astype(mx.float32) if real.dtype != mx.float32 else real
    imag_f32 = imag.astype(mx.float32) if imag.dtype != mx.float32 else imag
    mask_f32 = band_mask.astype(mx.float32) if band_mask.dtype != mx.float32 else band_mask
    return _ref_band_energy(real_f32, imag_f32, mask_f32, band_bins, eps)


def _should_use_fused_band_energy(real: mx.array) -> bool:
    """Return True when the fused band-energy kernel amortizes launch overhead.

    Benchmarks show the native path wins for tiny workloads, while fused wins
    reliably once the batched frame count reaches roughly 1800.
    """
    if real.ndim < 3:
        return True
    return (real.shape[0] * real.shape[1]) >= _BAND_ENERGY_FUSED_BT_THRESHOLD


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

        Outputs are always float32 for numerical stability and to support
        reduced-precision inputs such as bfloat16 without Metal type-mixing
        compilation errors during accumulation.
    """
    B, T, F = real.shape
    if not _should_use_fused_band_energy(real):
        return _native_band_energy(real, imag, band_mask, band_bins, eps)
    n_out = B * T
    if n_out == 0:
        empty = mx.zeros((B, T), dtype=mx.float32)
        return empty, empty
    params = mx.array([band_bins, eps], dtype=mx.float32)
    band_flat, log_flat = _band_energy_kernel(
        inputs=[real, imag, band_mask.reshape(-1), params],
        template=[("T", real.dtype)],
        grid=(n_out, 1, 1),
        threadgroup=(min(n_out, 256), 1, 1),
        output_shapes=[(n_out,), (n_out,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return band_flat.reshape(B, T), log_flat.reshape(B, T)


# ====================================================================
# Reference implementations for correctness verification
# ====================================================================


def _ref_log1p_mag(real: mx.array, imag: mx.array, eps: float = _EPS_F) -> mx.array:
    """Reference (standard MLX ops) for fused_log1p_mag."""
    return _native_log1p_mag(real, imag, eps)


def _ref_complex_mag(real: mx.array, imag: mx.array, eps: float = _EPS_F) -> mx.array:
    """Reference (standard MLX ops) for fused_complex_mag."""
    return _native_complex_mag(real, imag, eps)


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
