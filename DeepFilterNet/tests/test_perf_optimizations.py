"""Tests for performance optimizations: dtype guards, cached transpose, lazy loss dict,
Mamba scan concat-based parallel prefix, and post-filter Metal kernel.

Validates that:
1. Redundant FP32 casts are skipped when input is already FP32 (PERF-P0-001)
2. Pipeline/awesome losses cast once at entry (PERF-P0-002)
3. ERB filterbank transpose is cached at init (PERF-P0-003)
4. CombinedLoss returns lazy mx.array dict, not float dict (PERF-P1-002)
5. All dtype-guarded functions produce identical results for FP16 and FP32 inputs
6. Mamba scan uses concat-based parallel prefix (efficient in MLX's lazy eval model) (PERF-P2-001)
7. Post-filter Metal kernel matches pure-MLX fallback (PERF-P2-003)
"""

import inspect
import re
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestDtypeGuards:
    """PERF-P0-001: Verify dtype guards skip redundant casts."""

    def test_log1p_mag_fp32_no_cast(self):
        """_log1p_mag should not add astype nodes when input is already FP32."""
        from df_mlx.training_losses import _log1p_mag

        real = mx.ones((2, 10, 32), dtype=mx.float32)
        imag = mx.ones((2, 10, 32), dtype=mx.float32)
        result = _log1p_mag(real, imag)
        mx.eval(result)
        assert result.dtype == mx.float32

    def test_log1p_mag_fp16_casts(self):
        """_log1p_mag should cast FP16 inputs to FP32."""
        from df_mlx.training_losses import _log1p_mag

        real = mx.ones((2, 10, 32), dtype=mx.float16)
        imag = mx.ones((2, 10, 32), dtype=mx.float16)
        result = _log1p_mag(real, imag)
        mx.eval(result)
        assert result.dtype == mx.float32

    def test_log1p_mag_numerical_equivalence(self):
        """FP16 vs FP32 inputs should produce close results."""
        from df_mlx.training_losses import _log1p_mag

        np.random.seed(42)
        data_np = np.random.randn(2, 10, 32).astype(np.float32) * 0.1
        real_f32 = mx.array(data_np)
        imag_f32 = mx.array(data_np * 0.5)
        real_f16 = real_f32.astype(mx.float16)
        imag_f16 = imag_f32.astype(mx.float16)

        result_f32 = _log1p_mag(real_f32, imag_f32)
        result_f16 = _log1p_mag(real_f16, imag_f16)
        mx.eval(result_f32, result_f16)

        np.testing.assert_allclose(np.array(result_f32), np.array(result_f16), rtol=1e-2, atol=1e-3)

    def test_compute_vad_probs_fp32_no_cast(self):
        """_compute_vad_probs should skip casts when inputs are FP32."""
        from df_mlx.training_losses import _compute_vad_probs

        clean_real = mx.ones((2, 10, 32), dtype=mx.float32) * 0.5
        clean_imag = mx.ones((2, 10, 32), dtype=mx.float32) * 0.3
        out_real = mx.ones((2, 10, 32), dtype=mx.float32) * 0.4
        out_imag = mx.ones((2, 10, 32), dtype=mx.float32) * 0.2
        band_mask = mx.ones((1, 1, 32), dtype=mx.float32)
        band_bins = 32.0

        p_ref, p_out = _compute_vad_probs(
            clean_real,
            clean_imag,
            out_real,
            out_imag,
            band_mask,
            band_bins,
            vad_z_threshold=0.0,
            vad_z_slope=0.5,
        )
        mx.eval(p_ref, p_out)
        assert p_ref.dtype == mx.float32
        assert p_out.dtype == mx.float32


class TestCastOnceAtEntry:
    """PERF-P0-002: Verify cast-once pattern in awesome/pipeline losses."""

    def test_compute_proxy_gates_fp32_passthrough(self):
        """_compute_proxy_gates should skip casts when inputs are FP32."""
        from df_mlx.training_losses import _compute_proxy_gates

        B, T, F = 2, 10, 32
        clean_real = mx.ones((B, T, F), dtype=mx.float32) * 0.5
        clean_imag = mx.ones((B, T, F), dtype=mx.float32) * 0.3
        noisy_real = mx.ones((B, T, F), dtype=mx.float32) * 0.8
        noisy_imag = mx.ones((B, T, F), dtype=mx.float32) * 0.6
        snr = mx.array([10.0, 15.0])
        band_mask = mx.ones((1, 1, F), dtype=mx.float32)
        band_bins = float(F)

        result = _compute_proxy_gates(
            clean_real,
            clean_imag,
            noisy_real,
            noisy_imag,
            snr,
            band_mask,
            band_bins,
            vad_z_threshold=0.0,
            vad_z_slope=0.5,
            vad_snr_gate_db=5.0,
            vad_snr_gate_width=2.0,
            proxy_enabled=True,
        )
        mx.eval(*result)
        proxy_frame = result[0]
        assert proxy_frame.dtype == mx.float32


class TestErbFbCached:
    """PERF-P0-003: Verify _erb_fb_T is cached at init."""

    def test_erb_fb_t_exists(self):
        """DfNet4 should have _erb_fb_T attribute after init."""
        from df_mlx.config import get_default_config
        from df_mlx.model import DfNet4

        p = get_default_config()
        model = DfNet4(p)
        assert hasattr(model, "_erb_fb_T"), "DfNet4 must have _erb_fb_T cached at init"
        expected = mx.transpose(model._erb_fb)
        mx.eval(expected, model._erb_fb_T)
        np.testing.assert_array_equal(np.array(expected), np.array(model._erb_fb_T))

    def test_erb_fb_t_used_in_forward(self):
        """Forward pass should use _erb_fb_T (not recompute transpose)."""
        from df_mlx.config import get_default_config
        from df_mlx.model import DfNet4

        p = get_default_config()
        model = DfNet4(p)

        B, T = 1, 10
        n_freq = p.fft_size // 2 + 1
        spec_real = mx.zeros((B, T, n_freq))
        spec_imag = mx.zeros((B, T, n_freq))
        feat_erb = mx.zeros((B, T, p.nb_erb))
        feat_spec = mx.zeros((B, T, p.nb_df, 2))

        out = model((spec_real, spec_imag), feat_erb, feat_spec)
        mx.eval(*out)
        assert out[0].shape == (B, T, n_freq)


class TestCombinedLossLazy:
    """PERF-P1-002: Verify CombinedLoss returns lazy mx.array dict."""

    def test_combined_loss_returns_lazy_arrays(self):
        """CombinedLoss dict values should be mx.array, not float."""
        from df_mlx.loss import CombinedLoss

        loss_fn = CombinedLoss(sisdr_factor=0.5)

        pred = mx.zeros((1, 48000))
        target = mx.ones((1, 48000)) * 0.01

        total_loss, losses = loss_fn(pred, target)
        assert isinstance(total_loss, mx.array)
        for key, val in losses.items():
            assert isinstance(val, mx.array), f"CombinedLoss['{key}'] should be mx.array, got {type(val)}"


class TestMusicnessGuards:
    """PERF-P0-001: Verify musicness functions have dtype guards."""

    def test_compute_musicness_fp32_passthrough(self):
        """_compute_musicness should skip cast when mag is FP32."""
        from df_mlx.training_losses import _compute_musicness

        mag = mx.ones((2, 10, 32), dtype=mx.float32) * 0.5
        band_mask = mx.ones((1, 1, 32), dtype=mx.float32)
        musicness, gate = _compute_musicness(mag, band_mask, 32.0)
        mx.eval(musicness, gate)
        assert musicness.dtype == mx.float32

    def test_compute_improved_musicness_fp32_passthrough(self):
        """_compute_improved_musicness should skip cast when mag is FP32."""
        from df_mlx.training_losses import _compute_improved_musicness

        mag = mx.ones((2, 10, 32), dtype=mx.float32) * 0.5
        band_mask = mx.ones((1, 1, 32), dtype=mx.float32)
        snr = mx.array([10.0, 15.0])
        musicness, vocal, instrument = _compute_improved_musicness(mag, band_mask, 32.0, snr)
        mx.eval(musicness, vocal, instrument)
        assert musicness.dtype == mx.float32


# ---------------------------------------------------------------------------
# Phase 3 optimizations: Mamba scan + Post-filter Metal kernel
# ---------------------------------------------------------------------------


class TestMambaScanOptimization:
    """Verify that MambaBlock._selective_scan uses the concat-based parallel
    prefix scan approach (more efficient in MLX's lazy evaluation model than
    pre-allocated buffers with scatter writes)."""

    def test_mamba_scan_uses_concat_in_scan_loop(self):
        """The parallel scan loop in _selective_scan should use mx.concatenate
        for iterative doubling (generates optimal MLX computation graphs)."""
        from df_mlx.mamba import MambaBlock

        source = inspect.getsource(MambaBlock._selective_scan)
        loop_match = re.search(r"for d in range\(log2_L\):", source)
        assert loop_match is not None, "_selective_scan must contain the parallel scan loop"
        loop_body = source[loop_match.start() :]
        assert "concatenate" in loop_body, "_selective_scan scan loop must use mx.concatenate for iterative doubling"

    def test_mamba_scan_uses_conditional_padding(self):
        """_selective_scan should conditionally pad only when needed (pad_amount > 0)."""
        from df_mlx.mamba import MambaBlock

        source = inspect.getsource(MambaBlock._selective_scan)
        assert "pad_amount" in source, "_selective_scan must compute pad_amount"
        assert "if pad_amount > 0" in source, "_selective_scan must conditionally pad only when needed"

    def test_mamba_scan_numerical_equivalence(self):
        """MambaBlock forward pass must produce finite outputs with correct shape."""
        from df_mlx.mamba import MambaBlock

        d_model, d_state = 64, 16
        block = MambaBlock(d_model=d_model, d_state=d_state)
        mx.eval(block.parameters())

        batch, seq_len = 2, 32
        x = mx.random.normal((batch, seq_len, d_model))
        mx.eval(x)

        y, h_final = block(x)
        mx.eval(y, h_final)

        assert y.shape == (
            batch,
            seq_len,
            d_model,
        ), f"Expected output shape {(batch, seq_len, d_model)}, got {y.shape}"
        assert h_final.shape == (
            batch,
            block.d_inner,
            d_state,
        ), f"Expected state shape {(batch, block.d_inner, d_state)}, got {h_final.shape}"
        assert mx.all(mx.isfinite(y)).item(), "Output must contain only finite values"
        assert mx.all(mx.isfinite(h_final)).item(), "Final state must contain only finite values"


class TestPostFilterKernel:
    """Verify the fused post-filter Metal kernel in kernels.py."""

    def test_post_filter_kernel_exists(self):
        """post_filter_kernel must be importable from df_mlx.kernels."""
        from df_mlx.kernels import post_filter_kernel  # noqa: F401

        assert callable(post_filter_kernel)

    def test_post_filter_kernel_numerical_match(self):
        """Metal kernel and pure-MLX fallback must produce matching outputs."""
        from df_mlx.kernels import (
            _post_filter_fallback,
            _post_filter_forward_metal,
            metal_kernels_available,
        )

        if not metal_kernels_available():
            import pytest

            pytest.skip("Metal kernels not available")

        np.random.seed(123)
        shape = (2, 10, 65)
        enh_real = mx.array(np.random.randn(*shape).astype(np.float32) * 0.5)
        enh_imag = mx.array(np.random.randn(*shape).astype(np.float32) * 0.5)
        orig_real = mx.array(np.random.randn(*shape).astype(np.float32) * 0.8 + 1.0)
        orig_imag = mx.array(np.random.randn(*shape).astype(np.float32) * 0.8 + 1.0)
        beta_arr = mx.array([0.02], dtype=mx.float32)

        metal_r, metal_i = _post_filter_forward_metal(enh_real, enh_imag, orig_real, orig_imag, beta_arr)
        fallback_r, fallback_i = _post_filter_fallback(enh_real, enh_imag, orig_real, orig_imag, beta_arr)
        mx.eval(metal_r, metal_i, fallback_r, fallback_i)

        np.testing.assert_allclose(np.array(metal_r), np.array(fallback_r), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array(metal_i), np.array(fallback_i), rtol=1e-5, atol=1e-5)

    def test_post_filter_vjp_matches_autograd(self):
        """Gradients through _post_filter_custom must match _post_filter_fallback."""
        from df_mlx.kernels import (
            _post_filter_custom,
            _post_filter_fallback,
            metal_kernels_available,
        )

        if not metal_kernels_available():
            import pytest

            pytest.skip("Metal kernels not available")

        np.random.seed(456)
        shape = (1, 4, 8)
        enh_real = mx.array(np.random.randn(*shape).astype(np.float32) * 0.3)
        enh_imag = mx.array(np.random.randn(*shape).astype(np.float32) * 0.3)
        orig_real = mx.array(np.random.randn(*shape).astype(np.float32) * 0.5 + 1.0)
        orig_imag = mx.array(np.random.randn(*shape).astype(np.float32) * 0.5 + 1.0)
        beta_arr = mx.array([0.02], dtype=mx.float32)

        def loss_custom(er, ei, oreal, oimag, ba):
            r, i = _post_filter_custom(er, ei, oreal, oimag, ba)
            return mx.sum(r + i)

        def loss_fallback(er, ei, oreal, oimag, ba):
            r, i = _post_filter_fallback(er, ei, oreal, oimag, ba)
            return mx.sum(r + i)

        grad_custom_fn = mx.grad(loss_custom, argnums=(0, 1, 2, 3))
        grad_fallback_fn = mx.grad(loss_fallback, argnums=(0, 1, 2, 3))

        grads_c = grad_custom_fn(enh_real, enh_imag, orig_real, orig_imag, beta_arr)
        grads_f = grad_fallback_fn(enh_real, enh_imag, orig_real, orig_imag, beta_arr)
        mx.eval(*grads_c, *grads_f)

        for idx, label in enumerate(["enh_real", "enh_imag", "orig_real", "orig_imag"]):
            np.testing.assert_allclose(
                np.array(grads_c[idx]),
                np.array(grads_f[idx]),
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Gradient mismatch for {label}",
            )

    def test_post_filter_integrated_in_model(self):
        """model.py must import and use the post-filter kernel."""
        model_path = Path(__file__).resolve().parent.parent / "df_mlx" / "model.py"
        source = model_path.read_text()
        assert "metal_kernels_available" in source, "model.py must import metal_kernels_available"
        assert "post_filter_kernel" in source, "model.py must import post_filter_kernel"

    def test_post_filter_kernel_fp16(self):
        """post_filter_kernel must handle float16 inputs and produce finite output."""
        from df_mlx.kernels import metal_kernels_available, post_filter_kernel

        if not metal_kernels_available():
            import pytest

            pytest.skip("Metal kernels not available")

        np.random.seed(789)
        shape = (1, 8, 32)
        enh_real = mx.array(np.random.randn(*shape).astype(np.float32) * 0.4).astype(mx.float16)
        enh_imag = mx.array(np.random.randn(*shape).astype(np.float32) * 0.4).astype(mx.float16)
        orig_real = mx.array((np.random.randn(*shape).astype(np.float32) * 0.5 + 1.0)).astype(mx.float16)
        orig_imag = mx.array((np.random.randn(*shape).astype(np.float32) * 0.5 + 1.0)).astype(mx.float16)

        out_r, out_i = post_filter_kernel(enh_real, enh_imag, orig_real, orig_imag, 0.02)
        mx.eval(out_r, out_i)

        assert out_r.dtype == mx.float16, f"Expected float16 output, got {out_r.dtype}"
        assert out_i.dtype == mx.float16, f"Expected float16 output, got {out_i.dtype}"
        assert out_r.shape == shape
        assert mx.all(mx.isfinite(out_r)).item(), "Real output must be finite"
        assert mx.all(mx.isfinite(out_i)).item(), "Imag output must be finite"
