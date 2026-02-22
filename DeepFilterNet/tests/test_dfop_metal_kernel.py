"""Tests for DfOp custom Metal kernel vs pure-MLX fallback."""

import mlx.core as mx
import pytest

from df_mlx.kernels import df_op_kernel, metal_kernels_available
from df_mlx.modules import DfOp

RTOL = 1e-4
ATOL = 1e-5


def _random_inputs(
    batch: int,
    time: int,
    n_freqs: int,
    nb_df: int,
    df_order: int,
    dtype: mx.Dtype = mx.float32,
) -> tuple[tuple[mx.array, mx.array], mx.array]:
    """Build random (spec, coef) inputs for DfOp."""
    spec_real = mx.random.normal(shape=(batch, time, n_freqs)).astype(dtype)
    spec_imag = mx.random.normal(shape=(batch, time, n_freqs)).astype(dtype)
    coef = mx.random.normal(shape=(batch, time, nb_df, df_order, 2)).astype(dtype)
    return (spec_real, spec_imag), coef


# ------------------------------------------------------------------
# Parity: Metal kernel vs fallback
# ------------------------------------------------------------------

_SHAPES = [
    # (batch, time, n_freqs, nb_df, df_order, lookahead)
    (2, 10, 96, 96, 5, 0),
    (1, 1, 48, 32, 1, 0),
    (4, 20, 128, 64, 3, 0),
    (2, 10, 96, 96, 5, 2),
    (1, 5, 64, 32, 7, 1),
    (3, 8, 96, 96, 5, 0),
]


@pytest.mark.parametrize(
    "batch,time,n_freqs,nb_df,df_order,lookahead",
    _SHAPES,
    ids=[
        "typical",
        "minimal",
        "large_batch",
        "with_lookahead",
        "odd_order_lookahead",
        "batch3",
    ],
)
def test_metal_vs_fallback_parity(
    batch: int,
    time: int,
    n_freqs: int,
    nb_df: int,
    df_order: int,
    lookahead: int,
) -> None:
    """Metal kernel output must match the pure-MLX fallback within tolerance."""
    spec, coef = _random_inputs(batch, time, n_freqs, nb_df, df_order)

    metal_op = DfOp(nb_df=nb_df, df_order=df_order, df_lookahead=lookahead, use_metal_kernel=True)
    fallback_op = DfOp(nb_df=nb_df, df_order=df_order, df_lookahead=lookahead, use_metal_kernel=False)

    if not metal_op.use_metal_kernel:
        pytest.skip("Metal kernels not available on this platform")

    out_metal = metal_op(spec, coef)
    out_fallback = fallback_op(spec, coef)
    mx.eval(out_metal[0], out_metal[1], out_fallback[0], out_fallback[1])

    assert mx.allclose(
        out_metal[0], out_fallback[0], rtol=RTOL, atol=ATOL
    ).item(), f"Real part mismatch: max diff = {mx.max(mx.abs(out_metal[0] - out_fallback[0])).item()}"
    assert mx.allclose(
        out_metal[1], out_fallback[1], rtol=RTOL, atol=ATOL
    ).item(), f"Imag part mismatch: max diff = {mx.max(mx.abs(out_metal[1] - out_fallback[1])).item()}"


# ------------------------------------------------------------------
# Fallback path works when Metal disabled
# ------------------------------------------------------------------


def test_fallback_path() -> None:
    """DfOp with use_metal_kernel=False must produce valid output."""
    spec, coef = _random_inputs(2, 10, 96, 96, 5)
    op = DfOp(nb_df=96, df_order=5, df_lookahead=0, use_metal_kernel=False)
    out = op(spec, coef)
    mx.eval(out[0], out[1])

    assert out[0].shape == (2, 10, 96)
    assert out[1].shape == (2, 10, 96)


# ------------------------------------------------------------------
# Output shape and passthrough frequencies
# ------------------------------------------------------------------


def test_output_shape_with_passthrough() -> None:
    """Non-DF frequencies pass through untouched."""
    n_freqs, nb_df = 128, 64
    spec, coef = _random_inputs(2, 10, n_freqs, nb_df, 5)

    for use_kernel in (True, False):
        op = DfOp(nb_df=nb_df, df_order=5, df_lookahead=0, use_metal_kernel=use_kernel)
        if use_kernel and not op.use_metal_kernel:
            continue
        out_real, out_imag = op(spec, coef)
        mx.eval(out_real, out_imag)

        assert out_real.shape == (2, 10, n_freqs)
        assert out_imag.shape == (2, 10, n_freqs)

        # Passthrough frequencies should equal the input
        assert mx.allclose(out_real[:, :, nb_df:], spec[0][:, :, nb_df:], rtol=1e-6, atol=1e-7).item()
        assert mx.allclose(out_imag[:, :, nb_df:], spec[1][:, :, nb_df:], rtol=1e-6, atol=1e-7).item()


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_single_batch() -> None:
    """Works with batch_size=1."""
    spec, coef = _random_inputs(1, 10, 96, 96, 5)
    for use_kernel in (True, False):
        op = DfOp(nb_df=96, df_order=5, use_metal_kernel=use_kernel)
        if use_kernel and not op.use_metal_kernel:
            continue
        out = op(spec, coef)
        mx.eval(out[0], out[1])
        assert out[0].shape == (1, 10, 96)


def test_single_time_step() -> None:
    """Works with a single time step."""
    spec, coef = _random_inputs(2, 1, 48, 32, 3)
    for use_kernel in (True, False):
        op = DfOp(nb_df=32, df_order=3, use_metal_kernel=use_kernel)
        if use_kernel and not op.use_metal_kernel:
            continue
        out = op(spec, coef)
        mx.eval(out[0], out[1])
        assert out[0].shape == (2, 1, 48)


def test_df_order_one() -> None:
    """df_order=1 degenerates to pointwise complex multiply."""
    spec, coef = _random_inputs(2, 5, 64, 32, 1)
    for use_kernel in (True, False):
        op = DfOp(nb_df=32, df_order=1, use_metal_kernel=use_kernel)
        if use_kernel and not op.use_metal_kernel:
            continue
        out = op(spec, coef)
        mx.eval(out[0], out[1])
        assert out[0].shape == (2, 5, 64)


# ------------------------------------------------------------------
# Low-level kernel function
# ------------------------------------------------------------------


@pytest.mark.skipif(not metal_kernels_available(), reason="Metal kernels not available")
def test_df_op_kernel_direct() -> None:
    """Directly call df_op_kernel and validate output shapes and values."""
    batch, time, nb_df, df_order = 2, 8, 48, 5
    pad = df_order - 1

    spec_r = mx.random.normal(shape=(batch, time + pad, nb_df))
    spec_i = mx.random.normal(shape=(batch, time + pad, nb_df))
    coef_r = mx.random.normal(shape=(batch, time, nb_df, df_order))
    coef_i = mx.random.normal(shape=(batch, time, nb_df, df_order))

    out_r, out_i = df_op_kernel(spec_r, spec_i, coef_r, coef_i, time, nb_df, df_order, batch)
    mx.eval(out_r, out_i)

    assert out_r.shape == (batch, time, nb_df)
    assert out_i.shape == (batch, time, nb_df)
    assert out_r.dtype == mx.float32
    assert out_i.dtype == mx.float32


# ------------------------------------------------------------------
# Availability flag
# ------------------------------------------------------------------


def test_metal_kernels_available_is_bool() -> None:
    """metal_kernels_available() returns a Python bool."""
    result = metal_kernels_available()
    assert isinstance(result, bool)
