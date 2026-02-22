"""Tests for Metal kernel VJP (backward pass) correctness.

Each Metal kernel is wrapped with ``mx.custom_function`` and has a custom
VJP that uses pure-MLX ops for the backward pass.  These tests verify:

  1. Forward: Metal kernel matches the pure-MLX fallback numerically.
  2. Backward: Gradients through the Metal kernel match finite-difference
     approximations and/or gradients through the pure-MLX fallback.
  3. Integration: ``nn.value_and_grad`` succeeds through DfOp and
     MelSpectrogram in training mode with Metal kernels enabled.
  4. iSTFT: Metal kernel path produces correct output and gradients.
"""

from __future__ import annotations

import platform

import mlx.core as mx
import mlx.nn as nn
import pytest


def _on_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


pytestmark = pytest.mark.skipif(
    not _on_apple_silicon(),
    reason="Metal kernels require Apple Silicon",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finite_diff_grad(fn, inputs, idx, eps=1e-3):
    """Approximate gradient of scalar fn w.r.t. inputs[idx] via central differences."""
    x = inputs[idx]
    grad = mx.zeros_like(x)
    flat = x.reshape(-1)
    n = min(flat.size, 8)  # sample a few elements to keep it fast
    for i in range(n):
        x_plus = flat.at[i].add(eps)
        x_minus = flat.at[i].add(-eps)
        inp_plus = list(inputs)
        inp_minus = list(inputs)
        inp_plus[idx] = x_plus.reshape(x.shape)
        inp_minus[idx] = x_minus.reshape(x.shape)
        f_plus = fn(*inp_plus)
        f_minus = fn(*inp_minus)
        mx.eval(f_plus, f_minus)
        grad_val = (f_plus.item() - f_minus.item()) / (2 * eps)
        grad = grad.reshape(-1).at[i].add(grad_val).reshape(x.shape)
    return grad


# ---------------------------------------------------------------------------
# 1. DfOp kernel: forward + VJP
# ---------------------------------------------------------------------------


class TestDfOpKernelVJP:
    """DfOp Metal kernel produces correct forward output and gradients."""

    @staticmethod
    def _make_inputs(batch=2, time=4, nb_df=8, df_order=3):
        spec_real_pad = mx.random.normal((batch, time + df_order - 1, nb_df))
        spec_imag_pad = mx.random.normal((batch, time + df_order - 1, nb_df))
        coef_real = mx.random.normal((batch, time, nb_df, df_order))
        coef_imag = mx.random.normal((batch, time, nb_df, df_order))
        return spec_real_pad, spec_imag_pad, coef_real, coef_imag

    def test_forward_matches_fallback(self):
        """Metal kernel forward must match pure-MLX fallback."""
        from df_mlx.kernels import _dfop_fallback, df_op_kernel

        sr, si, cr, ci = self._make_inputs()
        B, T, nb_df, df_order = cr.shape

        kernel_r, kernel_i = df_op_kernel(sr, si, cr, ci, T, nb_df, df_order, B)
        fallback_r, fallback_i = _dfop_fallback(sr, si, cr, ci)
        mx.eval(kernel_r, kernel_i, fallback_r, fallback_i)

        assert mx.allclose(kernel_r, fallback_r, atol=1e-4).item()
        assert mx.allclose(kernel_i, fallback_i, atol=1e-4).item()

    def test_vjp_coef_matches_fallback(self):
        """Gradient w.r.t. coef through Metal kernel matches fallback auto-diff."""
        from df_mlx.kernels import _dfop_fallback, df_op_kernel

        sr, si, cr, ci = self._make_inputs()
        B, T, nb_df, df_order = cr.shape

        def loss_kernel(coef_r, coef_i):
            kr, ki = df_op_kernel(sr, si, coef_r, coef_i, T, nb_df, df_order, B)
            return mx.mean(kr**2 + ki**2)

        def loss_fallback(coef_r, coef_i):
            fr, fi = _dfop_fallback(sr, si, coef_r, coef_i)
            return mx.mean(fr**2 + fi**2)

        grad_kernel = mx.grad(loss_kernel, argnums=[0, 1])
        grad_fallback = mx.grad(loss_fallback, argnums=[0, 1])

        gk = grad_kernel(cr, ci)
        gf = grad_fallback(cr, ci)
        mx.eval(*gk, *gf)

        assert mx.allclose(gk[0], gf[0], atol=1e-3).item(), "d_coef_real mismatch"
        assert mx.allclose(gk[1], gf[1], atol=1e-3).item(), "d_coef_imag mismatch"

    def test_vjp_spec_matches_fallback(self):
        """Gradient w.r.t. spec through Metal kernel matches fallback auto-diff."""
        from df_mlx.kernels import _dfop_fallback, df_op_kernel

        sr, si, cr, ci = self._make_inputs()
        B, T, nb_df, df_order = cr.shape

        def loss_kernel(spec_r, spec_i):
            kr, ki = df_op_kernel(spec_r, spec_i, cr, ci, T, nb_df, df_order, B)
            return mx.mean(kr**2 + ki**2)

        def loss_fallback(spec_r, spec_i):
            fr, fi = _dfop_fallback(spec_r, spec_i, cr, ci)
            return mx.mean(fr**2 + fi**2)

        grad_kernel = mx.grad(loss_kernel, argnums=[0, 1])
        grad_fallback = mx.grad(loss_fallback, argnums=[0, 1])

        gk = grad_kernel(sr, si)
        gf = grad_fallback(sr, si)
        mx.eval(*gk, *gf)

        assert mx.allclose(gk[0], gf[0], atol=1e-3).item(), "d_spec_real mismatch"
        assert mx.allclose(gk[1], gf[1], atol=1e-3).item(), "d_spec_imag mismatch"

    def test_value_and_grad_through_dfop_module(self):
        """nn.value_and_grad succeeds through DfOp with Metal kernel in training mode."""
        from df_mlx.modules import DfOp

        dfop = DfOp(nb_df=8, df_order=3, df_lookahead=0, use_metal_kernel=True)

        class Wrapper(nn.Module):
            def __init__(self, op):
                super().__init__()
                self.op = op
                self.scale = mx.ones((1,))

            def __call__(self, spec_real, spec_imag, coef):
                out_r, out_i = self.op((spec_real, spec_imag), coef)
                return mx.mean(self.scale * (out_r**2 + out_i**2))

        wrapper = Wrapper(dfop)
        wrapper.train()

        spec_real = mx.random.normal((2, 4, 16))
        spec_imag = mx.random.normal((2, 4, 16))
        coef = mx.random.normal((2, 4, 8, 3, 2))

        loss_fn = nn.value_and_grad(wrapper, wrapper)
        loss, grads = loss_fn(spec_real, spec_imag, coef)
        mx.eval(loss, grads)
        assert loss.ndim == 0

    def test_train_eval_output_consistency(self):
        """Training and eval modes produce the same numerical output."""
        from df_mlx.modules import DfOp

        dfop = DfOp(nb_df=8, df_order=3, df_lookahead=0, use_metal_kernel=True)

        spec_real = mx.random.normal((2, 4, 16))
        spec_imag = mx.random.normal((2, 4, 16))
        coef = mx.random.normal((2, 4, 8, 3, 2))

        dfop.train()
        train_r, train_i = dfop((spec_real, spec_imag), coef)
        mx.eval(train_r, train_i)

        dfop.eval()
        eval_r, eval_i = dfop((spec_real, spec_imag), coef)
        mx.eval(eval_r, eval_i)

        assert mx.allclose(train_r, eval_r, atol=1e-4).item()
        assert mx.allclose(train_i, eval_i, atol=1e-4).item()


# ---------------------------------------------------------------------------
# 2. Mel spectrogram kernel: forward + VJP
# ---------------------------------------------------------------------------


class TestMelKernelVJP:
    """Mel power+log Metal kernel produces correct forward output and gradients."""

    @pytest.fixture()
    def mel(self):
        from df_mlx.dnsmos_proxy import MelSpectrogram

        return MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=256,
            n_mels=64,
            use_metal_kernel=True,
        )

    def test_forward_matches_fallback(self):
        """Metal kernel forward matches pure-MLX power+mel+log."""
        from df_mlx.kernels import _mel_forward_metal

        B, T, n_freqs, n_mels = 2, 10, 257, 64
        spec_real = mx.random.normal((B, T, n_freqs))
        spec_imag = mx.random.normal((B, T, n_freqs))
        mel_fb = mx.abs(mx.random.normal((n_mels, n_freqs))) + 0.01

        kernel_out = _mel_forward_metal(spec_real, spec_imag, mel_fb)

        power = spec_real**2 + spec_imag**2
        fallback_out = mx.log(mx.maximum(mx.matmul(power, mx.transpose(mel_fb)), 1e-10))

        mx.eval(kernel_out, fallback_out)
        assert mx.allclose(kernel_out, fallback_out, atol=1e-3).item()

    def test_vjp_spec_matches_fallback(self):
        """Gradient w.r.t. spec through Metal kernel matches pure-MLX auto-diff."""
        from df_mlx.kernels import mel_power_log_kernel

        B, T, n_freqs, n_mels = 2, 10, 257, 64
        spec_real = mx.random.normal((B, T, n_freqs))
        spec_imag = mx.random.normal((B, T, n_freqs))
        mel_fb = mx.abs(mx.random.normal((n_mels, n_freqs))) + 0.01

        def loss_kernel(sr, si):
            return mx.mean(mel_power_log_kernel(sr, si, mel_fb, B, T, n_mels))

        def loss_fallback(sr, si):
            power = sr**2 + si**2
            return mx.mean(mx.log(mx.maximum(mx.matmul(power, mx.transpose(mel_fb)), 1e-10)))

        gk = mx.grad(loss_kernel, argnums=[0, 1])(spec_real, spec_imag)
        gf = mx.grad(loss_fallback, argnums=[0, 1])(spec_real, spec_imag)
        mx.eval(*gk, *gf)

        assert mx.allclose(gk[0], gf[0], atol=1e-3).item(), "d_spec_real mismatch"
        assert mx.allclose(gk[1], gf[1], atol=1e-3).item(), "d_spec_imag mismatch"

    def test_value_and_grad_through_mel_module(self, mel):
        """nn.value_and_grad succeeds through MelSpectrogram in training mode."""

        class Wrapper(nn.Module):
            def __init__(self, mel_mod):
                super().__init__()
                self.mel = mel_mod
                self.scale = mx.ones((1,))

            def __call__(self, audio):
                return mx.mean(self.scale * self.mel(audio))

        wrapper = Wrapper(mel)
        wrapper.train()

        audio = mx.random.normal((2, 16000))
        loss_fn = nn.value_and_grad(wrapper, wrapper)
        loss, grads = loss_fn(audio)
        mx.eval(loss, grads)
        assert loss.ndim == 0

    def test_train_eval_output_consistency(self, mel):
        """Training and eval modes produce the same numerical output."""
        audio = mx.random.normal((2, 16000))

        mel.train()
        train_out = mel(audio)
        mx.eval(train_out)

        mel.eval()
        eval_out = mel(audio)
        mx.eval(eval_out)

        assert mx.allclose(train_out, eval_out, atol=1e-4).item()


# ---------------------------------------------------------------------------
# 3. iSTFT kernel: forward + VJP
# ---------------------------------------------------------------------------


class TestIstftKernelVJP:
    """iSTFT overlap-add Metal kernel produces correct output and gradients."""

    def test_forward_kernel_matches_fallback(self):
        """Metal kernel iSTFT matches pure-MLX vectorized path."""
        from df_mlx.ops import istft

        n_fft, hop = 960, 480
        frames = 10
        freq = n_fft // 2 + 1
        spec_real = mx.random.normal((1, frames, freq))
        spec_imag = mx.random.normal((1, frames, freq))

        wav_kernel = istft((spec_real, spec_imag), n_fft=n_fft, hop_length=hop, use_metal_kernel=True)
        wav_fallback = istft((spec_real, spec_imag), n_fft=n_fft, hop_length=hop, use_metal_kernel=False)
        mx.eval(wav_kernel, wav_fallback)

        assert mx.allclose(wav_kernel, wav_fallback, atol=1e-3).item()

    def test_istft_kernel_gradient(self):
        """Gradient through iSTFT Metal kernel matches fallback gradient."""
        from df_mlx.ops import istft

        n_fft, hop = 480, 240
        frames = 8
        freq = n_fft // 2 + 1
        spec_real = mx.random.normal((2, frames, freq))
        spec_imag = mx.random.normal((2, frames, freq))

        def loss_kernel(sr, si):
            wav = istft((sr, si), n_fft=n_fft, hop_length=hop, use_metal_kernel=True)
            return mx.mean(wav**2)

        def loss_fallback(sr, si):
            wav = istft((sr, si), n_fft=n_fft, hop_length=hop, use_metal_kernel=False)
            return mx.mean(wav**2)

        gk = mx.grad(loss_kernel, argnums=[0, 1])(spec_real, spec_imag)
        gf = mx.grad(loss_fallback, argnums=[0, 1])(spec_real, spec_imag)
        mx.eval(*gk, *gf)

        assert mx.allclose(gk[0], gf[0], atol=1e-2).item(), "d_spec_real mismatch"
        assert mx.allclose(gk[1], gf[1], atol=1e-2).item(), "d_spec_imag mismatch"

    def test_istft_no_kernel_still_works(self):
        """istft with use_metal_kernel=False still works correctly."""
        from df_mlx.ops import istft

        n_fft, hop = 960, 480
        frames = 10
        freq = n_fft // 2 + 1
        spec_real = mx.random.normal((1, frames, freq))
        spec_imag = mx.random.normal((1, frames, freq))

        wav = istft((spec_real, spec_imag), n_fft=n_fft, hop_length=hop, use_metal_kernel=False)
        mx.eval(wav)
        assert wav.ndim == 1 or wav.ndim == 2
