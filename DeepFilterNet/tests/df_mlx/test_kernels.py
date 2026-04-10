"""Tests for df_mlx kernels VJP vectorization."""

import mlx.core as mx
import numpy as np
import pytest


def numpy_dfop_forward(spec_real_pad, spec_imag_pad, coef_real, coef_imag):
    """NumPy reference implementation of DfOp forward."""
    batch_size, pad_time, nb_df = spec_real_pad.shape
    output_time, df_order = coef_real.shape[1], coef_real.shape[3]

    out_real = np.zeros((batch_size, output_time, nb_df))
    out_imag = np.zeros((batch_size, output_time, nb_df))

    for b in range(batch_size):
        for t in range(output_time):
            for f in range(nb_df):
                for k in range(df_order):
                    spec_idx = t + k
                    if spec_idx < pad_time:
                        sr = spec_real_pad[b, spec_idx, f]
                        si = spec_imag_pad[b, spec_idx, f]
                        cr = coef_real[b, t, f, k]
                        ci = coef_imag[b, t, f, k]
                        out_real[b, t, f] += cr * sr - ci * si
                        out_imag[b, t, f] += cr * si + ci * sr
    return out_real, out_imag


def numpy_dfop_vjp(spec_real_pad, spec_imag_pad, coef_real, coef_imag, d_out_real, d_out_imag):
    """NumPy reference implementation of DfOp VJP (backward pass)."""
    batch_size, pad_time, nb_df = spec_real_pad.shape
    output_time, df_order = coef_real.shape[1], coef_real.shape[3]

    d_spec_real_pad = np.zeros_like(spec_real_pad)
    d_spec_imag_pad = np.zeros_like(spec_imag_pad)
    d_coef_real = np.zeros_like(coef_real)
    d_coef_imag = np.zeros_like(coef_imag)

    for b in range(batch_size):
        for t in range(output_time):
            for f in range(nb_df):
                for k in range(df_order):
                    spec_idx = t + k
                    if spec_idx < pad_time:
                        cr = coef_real[b, t, f, k]
                        ci = coef_imag[b, t, f, k]
                        dr = d_out_real[b, t, f]
                        di = d_out_imag[b, t, f]

                        d_spec_real_pad[b, spec_idx, f] += cr * dr + ci * di
                        d_spec_imag_pad[b, spec_idx, f] += cr * di - ci * dr

                        d_coef_real[b, t, f, k] = (
                            spec_real_pad[b, spec_idx, f] * dr + spec_imag_pad[b, spec_idx, f] * di
                        )
                        d_coef_imag[b, t, f, k] = (
                            spec_real_pad[b, spec_idx, f] * di - spec_imag_pad[b, spec_idx, f] * dr
                        )

    return d_spec_real_pad, d_spec_imag_pad, d_coef_real, d_coef_imag


class TestDfOpVJP:
    """Tests for the DfOp VJP using MLX's automatic differentiation."""

    @pytest.fixture
    def sample_inputs(self):
        """Common input shapes for testing."""
        batch_size = 2
        pad_time = 10
        output_time = 8
        nb_df = 4
        df_order = 3

        spec_real_pad = np.random.randn(batch_size, pad_time, nb_df).astype(np.float32)
        spec_imag_pad = np.random.randn(batch_size, pad_time, nb_df).astype(np.float32)
        coef_real = np.random.randn(batch_size, output_time, nb_df, df_order).astype(np.float32)
        coef_imag = np.random.randn(batch_size, output_time, nb_df, df_order).astype(np.float32)

        return {
            "spec_real_pad": spec_real_pad,
            "spec_imag_pad": spec_imag_pad,
            "coef_real": coef_real,
            "coef_imag": coef_imag,
        }

    def test_gradient_shapes(self, sample_inputs):
        """Test that gradients returned have correct shapes."""
        from df_mlx import kernels

        spec_real_pad = mx.array(sample_inputs["spec_real_pad"])
        spec_imag_pad = mx.array(sample_inputs["spec_imag_pad"])
        coef_real = mx.array(sample_inputs["coef_real"])
        coef_imag = mx.array(sample_inputs["coef_imag"])

        def loss_fn(sr, si, cr, ci):
            out_r, out_i = kernels._dfop_custom(sr, si, cr, ci)
            return mx.sum(out_r * out_r + out_i * out_i)

        grads = mx.grad(loss_fn, argnums=[0, 1, 2, 3])(spec_real_pad, spec_imag_pad, coef_real, coef_imag)

        d_spec_r, d_spec_i, d_coef_r, d_coef_i = grads

        assert d_spec_r.shape == spec_real_pad.shape
        assert d_spec_i.shape == spec_imag_pad.shape
        assert d_coef_r.shape == coef_real.shape
        assert d_coef_i.shape == coef_imag.shape

    def test_gradient_correctness(self, sample_inputs):
        """Test gradient correctness against numerical gradient."""
        from df_mlx import kernels

        spec_real_pad = mx.array(sample_inputs["spec_real_pad"])
        spec_imag_pad = mx.array(sample_inputs["spec_imag_pad"])
        coef_real = mx.array(sample_inputs["coef_real"])
        coef_imag = mx.array(sample_inputs["coef_imag"])

        d_out_real = np.random.randn(*coef_real.shape[:3]).astype(np.float32)
        d_out_imag = np.random.randn(*coef_real.shape[:3]).astype(np.float32)

        np_spec_r = sample_inputs["spec_real_pad"]
        np_spec_i = sample_inputs["spec_imag_pad"]
        np_coef_r = sample_inputs["coef_real"]
        np_coef_i = sample_inputs["coef_imag"]

        def loss_fn(sr, si, cr, ci):
            out_r, out_i = kernels._dfop_custom(sr, si, cr, ci)
            return mx.sum(out_r * d_out_real + out_i * d_out_imag)

        grads = mx.grad(loss_fn, argnums=[0, 1, 2, 3])(spec_real_pad, spec_imag_pad, coef_real, coef_imag)

        d_spec_r_mlx, d_spec_i_mlx, d_coef_r_mlx, d_coef_i_mlx = grads

        np_result = numpy_dfop_vjp(np_spec_r, np_spec_i, np_coef_r, np_coef_i, d_out_real, d_out_imag)

        np.testing.assert_allclose(d_spec_r_mlx, np_result[0], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(d_spec_i_mlx, np_result[1], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(d_coef_r_mlx, np_result[2], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(d_coef_i_mlx, np_result[3], rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("df_order", [1, 3, 5, 7])
    def test_various_df_order(self, df_order):
        """Test VJP with various df_order values."""
        from df_mlx import kernels

        batch_size = 2
        pad_time = 15
        output_time = 10
        nb_df = 4

        spec_real_pad = mx.array(np.random.randn(batch_size, pad_time, nb_df).astype(np.float32))
        spec_imag_pad = mx.array(np.random.randn(batch_size, pad_time, nb_df).astype(np.float32))
        coef_real = mx.array(np.random.randn(batch_size, output_time, nb_df, df_order).astype(np.float32))
        coef_imag = mx.array(np.random.randn(batch_size, output_time, nb_df, df_order).astype(np.float32))

        def loss_fn(sr, si, cr, ci):
            out_r, out_i = kernels._dfop_custom(sr, si, cr, ci)
            return mx.sum(out_r * out_r + out_i * out_i)

        grads = mx.grad(loss_fn, argnums=[0, 1, 2, 3])(spec_real_pad, spec_imag_pad, coef_real, coef_imag)

        d_spec_r, d_spec_i, d_coef_r, d_coef_i = grads

        assert d_spec_r.shape == (batch_size, pad_time, nb_df)
        assert d_coef_r.shape == (batch_size, output_time, nb_df, df_order)

    def test_vjp_matches_forward_indexing(self, sample_inputs):
        """Test that VJP uses same indexing as forward pass."""
        from df_mlx import kernels

        spec_real_pad = mx.array(sample_inputs["spec_real_pad"])
        spec_imag_pad = mx.array(sample_inputs["spec_imag_pad"])
        coef_real = mx.array(sample_inputs["coef_real"])
        coef_imag = mx.array(sample_inputs["coef_imag"])

        d_out_real = np.random.randn(*coef_real.shape[:3]).astype(np.float32)
        d_out_imag = np.random.randn(*coef_real.shape[:3]).astype(np.float32)

        np_spec_r = sample_inputs["spec_real_pad"]
        np_spec_i = sample_inputs["spec_imag_pad"]
        np_coef_r = sample_inputs["coef_real"]
        np_coef_i = sample_inputs["coef_imag"]

        def loss_fn(sr, si, cr, ci):
            out_r, out_i = kernels._dfop_custom(sr, si, cr, ci)
            return mx.sum(out_r * d_out_real + out_i * d_out_imag)

        grads = mx.grad(loss_fn, argnums=[0, 1, 2, 3])(spec_real_pad, spec_imag_pad, coef_real, coef_imag)

        d_spec_r, d_spec_i, _, _ = grads

        np_result = numpy_dfop_vjp(np_spec_r, np_spec_i, np_coef_r, np_coef_i, d_out_real, d_out_imag)

        np.testing.assert_allclose(d_spec_r, np_result[0], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(d_spec_i, np_result[1], rtol=1e-4, atol=1e-5)
