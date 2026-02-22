"""Tests for the pure-PyTorch spectral frontend (df.spectral).

Validates numerical parity with the Rust libDF implementation where
available, and correctness of the standalone PyTorch path.
"""

import math

import numpy as np
import pytest
import torch

from df.spectral import (
    SpectralFrontend,
    _build_erb_matrix,
    compute_erb_fb,
    df_features_torch,
    erb2freq,
    freq2erb,
    vorbis_window,
)

# ---------------------------------------------------------------------------
# Vorbis window
# ---------------------------------------------------------------------------


class TestVorbisWindow:
    def test_shape_and_dtype(self):
        w = vorbis_window(960)
        assert w.shape == (960,)
        assert w.dtype == torch.float32

    def test_symmetry(self):
        w = vorbis_window(960)
        assert torch.allclose(w, w.flip(0), atol=1e-6)

    def test_peak_at_center(self):
        w = vorbis_window(960)
        assert w[479].item() == pytest.approx(1.0, abs=1e-6)
        assert w[480].item() == pytest.approx(1.0, abs=1e-6)

    def test_edges_near_zero(self):
        w = vorbis_window(960)
        assert w[0].item() < 1e-3
        assert w[-1].item() < 1e-3

    def test_matches_rust_formula(self):
        fft_size = 960
        half = fft_size // 2
        pi = math.pi
        expected = []
        for i in range(fft_size):
            s = math.sin(0.5 * pi * (i + 0.5) / half)
            expected.append(math.sin(0.5 * pi * s * s))
        expected_t = torch.tensor(expected, dtype=torch.float32)
        w = vorbis_window(fft_size)
        assert torch.allclose(w, expected_t, atol=1e-6)

    def test_custom_device_dtype(self):
        w = vorbis_window(960, dtype=torch.float64, device="cpu")
        assert w.dtype == torch.float64

    @pytest.mark.skipif(not hasattr(torch, "__version__"), reason="sanity")
    def test_parity_with_rust(self):
        try:
            from libdf import DF

            df = DF(48000, 960, 480, 32, 2)
            rust_win = df.fft_window()
            torch_win = vorbis_window(960).numpy()
            np.testing.assert_allclose(torch_win, rust_win, atol=1e-6)
        except ImportError:
            pytest.skip("libdf not available")


# ---------------------------------------------------------------------------
# ERB filterbank
# ---------------------------------------------------------------------------


class TestERBFilterbank:
    def test_default_band_sum(self):
        widths = compute_erb_fb(48000, 960, 32, 2)
        assert sum(widths) == 481  # fft_size // 2 + 1

    def test_all_positive(self):
        widths = compute_erb_fb(48000, 960, 32, 2)
        assert all(w > 0 for w in widths)

    def test_min_nb_freqs_respected(self):
        widths = compute_erb_fb(48000, 960, 32, 2)
        assert all(w >= 2 for w in widths)

    def test_parity_with_rust(self):
        try:
            from libdf import DF

            df = DF(48000, 960, 480, 32, 2)
            rust_erb = list(df.erb_widths())
            torch_erb = compute_erb_fb(48000, 960, 32, 2)
            assert torch_erb == rust_erb
        except ImportError:
            pytest.skip("libdf not available")

    def test_erb_matrix_shape(self):
        widths = compute_erb_fb(48000, 960, 32, 2)
        mat = _build_erb_matrix(widths)
        assert mat.shape == (32, 481)

    def test_erb_matrix_row_sums(self):
        widths = compute_erb_fb(48000, 960, 32, 2)
        mat = _build_erb_matrix(widths)
        row_sums = mat.sum(dim=1)
        expected = torch.ones(32)
        assert torch.allclose(row_sums, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# ERB scale conversions
# ---------------------------------------------------------------------------


class TestERBScale:
    def test_roundtrip(self):
        for f in [0.0, 100.0, 1000.0, 8000.0, 24000.0]:
            assert erb2freq(freq2erb(f)) == pytest.approx(f, rel=1e-5)

    def test_monotonic(self):
        freqs = [100 * i for i in range(1, 50)]
        erbs = [freq2erb(f) for f in freqs]
        assert all(erbs[i] < erbs[i + 1] for i in range(len(erbs) - 1))


# ---------------------------------------------------------------------------
# SpectralFrontend — STFT / iSTFT
# ---------------------------------------------------------------------------


class TestSpectralFrontend:
    @pytest.fixture
    def frontend(self):
        return SpectralFrontend(48000, 960, 480, 32, 2)

    def test_analysis_shape(self, frontend):
        audio = torch.randn(1, 48000)
        spec = frontend.analysis(audio)
        assert spec.shape == (1, 99, 481)
        assert spec.is_complex()

    def test_analysis_2d(self, frontend):
        audio = torch.randn(1, 48000)
        spec = frontend.analysis(audio)
        assert spec.ndim == 2 or spec.shape[0] == 1

    def test_synthesis_roundtrip(self, frontend):
        audio = torch.randn(1, 48000)
        spec = frontend.analysis(audio)
        recon = frontend.synthesis(spec)
        assert recon.shape == audio.shape
        # Exclude boundary region (fft_size - hop_size = 480 samples each side)
        # where overlap-add is incomplete with center=False
        d = frontend.fft_size_val - frontend.hop_size_val
        interior = slice(d, -d) if d > 0 else slice(None)
        assert torch.allclose(audio[..., interior], recon[..., interior], atol=1e-4)

    def test_synthesis_roundtrip_multichannel(self, frontend):
        audio = torch.randn(2, 48000)
        spec = frontend.analysis(audio)
        recon = frontend.synthesis(spec)
        assert recon.shape == audio.shape
        d = frontend.fft_size_val - frontend.hop_size_val
        interior = slice(d, -d) if d > 0 else slice(None)
        assert torch.allclose(audio[..., interior], recon[..., interior], atol=1e-4)

    def test_erb_output_shape(self, frontend):
        spec = frontend.analysis(torch.randn(1, 48000))
        erb = frontend.erb(spec, db=True)
        assert erb.shape == (1, 99, 32)

    def test_erb_output_db_range(self, frontend):
        spec = frontend.analysis(torch.randn(1, 48000))
        erb = frontend.erb(spec, db=True)
        # dB values should be finite
        assert torch.isfinite(erb).all()

    def test_erb_no_db(self, frontend):
        spec = frontend.analysis(torch.randn(1, 48000))
        erb = frontend.erb(spec, db=False)
        # Linear power should be non-negative
        assert (erb >= 0).all()

    def test_erb_norm_shape(self, frontend):
        spec = frontend.analysis(torch.randn(1, 48000))
        erb = frontend.erb(spec, db=True)
        normed = frontend.erb_norm(erb, alpha=0.99)
        assert normed.shape == erb.shape

    def test_unit_norm_shape(self, frontend):
        spec = frontend.analysis(torch.randn(1, 48000))
        spec_sub = spec[..., :96]
        normed = frontend.unit_norm(spec_sub, alpha=0.99)
        assert normed.shape == spec_sub.shape
        assert normed.is_complex()

    def test_reset_clears_state(self, frontend):
        spec = frontend.analysis(torch.randn(1, 48000))
        erb = frontend.erb(spec, db=True)
        frontend.erb_norm(erb, alpha=0.99)
        assert frontend._erb_norm_state is not None
        frontend.reset()
        assert frontend._erb_norm_state is None
        assert frontend._unit_norm_state is None

    def test_properties(self, frontend):
        assert frontend.fft_size_val == 960
        assert frontend.hop_size_val == 480
        assert frontend.sr_val == 48000


# ---------------------------------------------------------------------------
# df_features_torch
# ---------------------------------------------------------------------------


class TestDfFeaturesTorch:
    @pytest.fixture
    def frontend(self):
        return SpectralFrontend(48000, 960, 480, 32, 2)

    def test_output_shapes(self, frontend):
        audio = torch.randn(1, 48000)
        spec, erb_feat, spec_feat = df_features_torch(audio, frontend, nb_df=96, norm_alpha=0.99)
        assert spec.shape == (1, 1, 99, 481, 2)  # [C, 1, T', F, 2]
        assert erb_feat.shape == (1, 1, 99, 32)  # [C, 1, T', E]
        assert spec_feat.shape == (1, 1, 99, 96, 2)  # [C, 1, T', nb_df, 2]

    def test_outputs_finite(self, frontend):
        audio = torch.randn(1, 48000)
        spec, erb_feat, spec_feat = df_features_torch(audio, frontend, nb_df=96, norm_alpha=0.99)
        assert torch.isfinite(spec).all()
        assert torch.isfinite(erb_feat).all()
        assert torch.isfinite(spec_feat).all()

    def test_device_transfer(self, frontend):
        audio = torch.randn(1, 48000)
        spec, erb_feat, spec_feat = df_features_torch(audio, frontend, nb_df=96, norm_alpha=0.99, device="cpu")
        assert spec.device.type == "cpu"


# ---------------------------------------------------------------------------
# Parity with Rust libDF
# ---------------------------------------------------------------------------


class TestRustParity:
    """Cross-validate PyTorch frontend against Rust libDF."""

    @pytest.fixture
    def both(self):
        try:
            from libdf import DF

            df = DF(48000, 960, 480, 32, 2)
            fe = SpectralFrontend(48000, 960, 480, 32, 2)
            return df, fe
        except ImportError:
            pytest.skip("libdf not available")

    def test_stft_parity(self, both):
        df, fe = both
        np.random.seed(42)
        audio_np = np.random.randn(1, 48000).astype(np.float32)
        audio_t = torch.from_numpy(audio_np.copy())

        rust_spec = df.analysis(audio_np)
        torch_spec = fe.analysis(audio_t).detach().numpy()

        min_t = min(rust_spec.shape[1], torch_spec.shape[1])
        # Skip first 2 frames: Rust streaming STFT has zero-padded analysis
        # memory warmup that batch torch.stft doesn't replicate at boundaries.
        # Interior frames should be very close.
        diff = np.abs(rust_spec[:, 2:min_t] - torch_spec[:, 2:min_t])
        assert diff.mean() < 0.05, f"STFT mean diff {diff.mean():.4e} too large"

    def test_erb_widths_parity(self, both):
        df, fe = both
        rust_erb = list(df.erb_widths())
        torch_erb = compute_erb_fb(48000, 960, 32, 2)
        assert rust_erb == torch_erb

    def test_erb_features_parity(self, both):
        from libdf import erb as rust_erb_fn

        df, fe = both
        np.random.seed(42)
        audio_np = np.random.randn(1, 48000).astype(np.float32)
        audio_t = torch.from_numpy(audio_np.copy())

        rust_spec = df.analysis(audio_np)
        torch_spec = fe.analysis(audio_t)

        rust_erb = rust_erb_fn(rust_spec, df.erb_widths())
        torch_erb = fe.erb(torch_spec, db=True).detach().numpy()

        min_t = min(rust_erb.shape[1], torch_erb.shape[1])
        # ERB dB differences compound from STFT boundary mismatch.
        # Skip warmup frames and check interior agreement.
        diff = np.abs(rust_erb[:, 2:min_t] - torch_erb[:, 2:min_t])
        assert diff.mean() < 3.0, f"ERB mean diff {diff.mean():.4e} too large"
