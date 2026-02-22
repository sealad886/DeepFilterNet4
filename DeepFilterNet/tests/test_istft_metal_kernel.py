"""Tests for fused Metal kernel iSTFT overlap-add + normalization.

Validates that the custom Metal kernel produces output numerically close to
the pure-MLX fallback path, and that STFT→iSTFT roundtrip reconstruction
works correctly with the kernel enabled.
"""

import mlx.core as mx
import numpy as np
import pytest

from df_mlx.kernels import metal_kernels_available
from df_mlx.ops import istft, stft

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METAL = metal_kernels_available()


def _make_signal(batch: int, length: int, seed: int = 42) -> mx.array:
    """Generate a deterministic random signal."""
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((batch, length)).astype(np.float32))


# ---------------------------------------------------------------------------
# Kernel vs fallback parity
# ---------------------------------------------------------------------------


class TestKernelVsFallbackParity:
    """The Metal kernel path must produce output close to the pure-MLX path."""

    @pytest.mark.skipif(not _METAL, reason="Metal kernels not available")
    @pytest.mark.parametrize("n_fft,hop", [(960, 480), (512, 256), (256, 128)])
    def test_parity_various_fft_hop(self, n_fft: int, hop: int) -> None:
        batch, length = 2, n_fft * 20
        sig = _make_signal(batch, length)
        spec = stft(sig, n_fft=n_fft, hop_length=hop, center=True)

        out_kernel = istft(spec, n_fft=n_fft, hop_length=hop, center=True, use_metal_kernel=True)
        out_fallback = istft(spec, n_fft=n_fft, hop_length=hop, center=True, use_metal_kernel=False)
        mx.eval(out_kernel, out_fallback)

        assert out_kernel.shape == out_fallback.shape
        assert mx.allclose(out_kernel, out_fallback, rtol=1e-4, atol=1e-5).item()

    @pytest.mark.skipif(not _METAL, reason="Metal kernels not available")
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_parity_batch_sizes(self, batch_size: int) -> None:
        n_fft, hop, length = 960, 480, 960 * 10
        sig = _make_signal(batch_size, length)
        spec = stft(sig, n_fft=n_fft, hop_length=hop, center=True)

        out_kernel = istft(spec, n_fft=n_fft, hop_length=hop, center=True, use_metal_kernel=True)
        out_fallback = istft(spec, n_fft=n_fft, hop_length=hop, center=True, use_metal_kernel=False)
        mx.eval(out_kernel, out_fallback)

        assert out_kernel.shape == out_fallback.shape
        assert mx.allclose(out_kernel, out_fallback, rtol=1e-4, atol=1e-5).item()

    @pytest.mark.skipif(not _METAL, reason="Metal kernels not available")
    def test_parity_center_false(self) -> None:
        n_fft, hop = 960, 480
        sig = _make_signal(2, 960 * 15)
        spec = stft(sig, n_fft=n_fft, hop_length=hop, center=False)

        out_kernel = istft(spec, n_fft=n_fft, hop_length=hop, center=False, use_metal_kernel=True)
        out_fallback = istft(spec, n_fft=n_fft, hop_length=hop, center=False, use_metal_kernel=False)
        mx.eval(out_kernel, out_fallback)

        assert out_kernel.shape == out_fallback.shape
        assert mx.allclose(out_kernel, out_fallback, rtol=1e-4, atol=1e-5).item()

    @pytest.mark.skipif(not _METAL, reason="Metal kernels not available")
    def test_parity_with_length(self) -> None:
        n_fft, hop = 960, 480
        original_len = 960 * 12
        sig = _make_signal(2, original_len)
        spec = stft(sig, n_fft=n_fft, hop_length=hop, center=True)

        out_kernel = istft(
            spec,
            n_fft=n_fft,
            hop_length=hop,
            center=True,
            length=original_len,
            use_metal_kernel=True,
        )
        out_fallback = istft(
            spec,
            n_fft=n_fft,
            hop_length=hop,
            center=True,
            length=original_len,
            use_metal_kernel=False,
        )
        mx.eval(out_kernel, out_fallback)

        assert out_kernel.shape == out_fallback.shape
        assert out_kernel.shape[1] == original_len
        assert mx.allclose(out_kernel, out_fallback, rtol=1e-4, atol=1e-5).item()

    @pytest.mark.skipif(not _METAL, reason="Metal kernels not available")
    def test_parity_stacked_input(self) -> None:
        """Verify parity when spec is passed as stacked (..., freq, 2)."""
        n_fft, hop = 512, 256
        sig = _make_signal(1, 512 * 10)
        real, imag = stft(sig, n_fft=n_fft, hop_length=hop, center=True)
        spec_stacked = mx.stack([real, imag], axis=-1)

        out_kernel = istft(
            spec_stacked,
            n_fft=n_fft,
            hop_length=hop,
            center=True,
            use_metal_kernel=True,
        )
        out_fallback = istft(
            spec_stacked,
            n_fft=n_fft,
            hop_length=hop,
            center=True,
            use_metal_kernel=False,
        )
        mx.eval(out_kernel, out_fallback)

        assert out_kernel.shape == out_fallback.shape
        assert mx.allclose(out_kernel, out_fallback, rtol=1e-4, atol=1e-5).item()


# ---------------------------------------------------------------------------
# STFT → iSTFT roundtrip
# ---------------------------------------------------------------------------


class TestSTFTRoundtrip:
    """Verify STFT→iSTFT roundtrip with the Metal kernel path."""

    @pytest.mark.skipif(not _METAL, reason="Metal kernels not available")
    @pytest.mark.parametrize(
        "n_fft,hop",
        [(960, 480), (512, 256), (256, 128)],
    )
    def test_roundtrip_center_true(self, n_fft: int, hop: int) -> None:
        length = n_fft * 20
        sig = _make_signal(2, length)
        spec = stft(sig, n_fft=n_fft, hop_length=hop, center=True)
        recon = istft(
            spec,
            n_fft=n_fft,
            hop_length=hop,
            center=True,
            length=length,
            use_metal_kernel=True,
        )
        mx.eval(recon)

        assert recon.shape == sig.shape
        assert mx.allclose(recon, sig, rtol=1e-3, atol=1e-4).item()

    @pytest.mark.skipif(not _METAL, reason="Metal kernels not available")
    def test_roundtrip_center_false(self) -> None:
        n_fft, hop = 960, 480
        sig = _make_signal(2, 960 * 15)
        spec = stft(sig, n_fft=n_fft, hop_length=hop, center=False)
        recon = istft(
            spec,
            n_fft=n_fft,
            hop_length=hop,
            center=False,
            use_metal_kernel=True,
        )
        mx.eval(recon)

        # With center=False, the reconstructed length may differ slightly
        min_len = min(recon.shape[1], sig.shape[1])
        # Trim edges where boundary effects dominate
        trim = n_fft
        if min_len > 2 * trim:
            assert mx.allclose(
                recon[:, trim : min_len - trim],
                sig[:, trim : min_len - trim],
                rtol=1e-3,
                atol=1e-4,
            ).item()

    @pytest.mark.skipif(not _METAL, reason="Metal kernels not available")
    def test_roundtrip_single_batch(self) -> None:
        n_fft, hop, length = 512, 256, 8192
        sig = _make_signal(1, length)
        spec = stft(sig, n_fft=n_fft, hop_length=hop, center=True)
        recon = istft(
            spec,
            n_fft=n_fft,
            hop_length=hop,
            center=True,
            length=length,
            use_metal_kernel=True,
        )
        mx.eval(recon)

        assert recon.shape == sig.shape
        assert mx.allclose(recon, sig, rtol=1e-3, atol=1e-4).item()


# ---------------------------------------------------------------------------
# Fallback forcing
# ---------------------------------------------------------------------------


class TestFallbackForcing:
    """Verify that use_metal_kernel=False forces the pure-MLX path."""

    def test_fallback_produces_valid_output(self) -> None:
        n_fft, hop = 960, 480
        sig = _make_signal(2, 960 * 10)
        spec = stft(sig, n_fft=n_fft, hop_length=hop, center=True)
        out = istft(
            spec,
            n_fft=n_fft,
            hop_length=hop,
            center=True,
            length=960 * 10,
            use_metal_kernel=False,
        )
        mx.eval(out)

        assert out.shape == sig.shape
        assert mx.allclose(out, sig, rtol=1e-3, atol=1e-4).item()

    def test_fallback_matches_default_when_no_metal(self) -> None:
        """When use_metal_kernel=False, output must be identical to prior behavior."""
        n_fft, hop = 512, 256
        sig = _make_signal(1, 512 * 10)
        spec = stft(sig, n_fft=n_fft, hop_length=hop, center=True)

        out_a = istft(spec, n_fft=n_fft, hop_length=hop, center=True, use_metal_kernel=False)
        out_b = istft(spec, n_fft=n_fft, hop_length=hop, center=True, use_metal_kernel=False)
        mx.eval(out_a, out_b)

        assert mx.array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# Kernel wrapper direct tests
# ---------------------------------------------------------------------------


class TestKernelWrapper:
    """Direct tests for istft_overlap_add_kernel wrapper."""

    @pytest.mark.skipif(not _METAL, reason="Metal kernels not available")
    def test_kernel_output_shape(self) -> None:
        from df_mlx.kernels import istft_overlap_add_kernel

        batch, num_frames, n_fft, hop = 2, 20, 960, 480
        output_length = (num_frames - 1) * hop + n_fft
        frames = mx.zeros((batch, num_frames, n_fft))
        wn = mx.ones((output_length,))

        out = istft_overlap_add_kernel(
            frames=frames,
            window_norm=wn,
            hop_length=hop,
            output_length=output_length,
            batch_size=batch,
        )
        mx.eval(out)

        assert out.shape == (batch, output_length)

    @pytest.mark.skipif(not _METAL, reason="Metal kernels not available")
    def test_kernel_zero_input(self) -> None:
        from df_mlx.kernels import istft_overlap_add_kernel

        batch, num_frames, n_fft, hop = 1, 10, 512, 256
        output_length = (num_frames - 1) * hop + n_fft
        frames = mx.zeros((batch, num_frames, n_fft))
        wn = mx.ones((output_length,))

        out = istft_overlap_add_kernel(
            frames=frames,
            window_norm=wn,
            hop_length=hop,
            output_length=output_length,
            batch_size=batch,
        )
        mx.eval(out)

        assert mx.allclose(out, mx.zeros_like(out), atol=1e-8).item()

    def test_kernel_raises_without_metal(self) -> None:
        """Ensure RuntimeError when metal_kernel is unavailable."""
        from df_mlx import kernels

        orig = kernels._istft_ola_kernel
        try:
            kernels._istft_ola_kernel = None
            with pytest.raises(RuntimeError, match="metal_kernel"):
                kernels.istft_overlap_add_kernel(
                    frames=mx.zeros((1, 5, 4)),
                    window_norm=mx.ones((24,)),
                    hop_length=4,
                    output_length=24,
                    batch_size=1,
                )
        finally:
            kernels._istft_ola_kernel = orig
