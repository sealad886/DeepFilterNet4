"""Tests for hybrid encoder components.

Tests cover:
- WaveformEncoder shape and behavior
- PhaseEncoder with different input formats
- CrossDomainAttention fusion
- HybridEncoder full integration
- LightweightHybridEncoder variant
"""

import pytest
import torch

from df.hybrid_encoder import (
    CrossDomainAttention,
    HybridEncoder,
    LightweightHybridEncoder,
    MagnitudeEncoder,
    PhaseEncoder,
    SimpleCrossAttention,
    WaveformEncoder,
)

# ============================================================================
# WaveformEncoder Tests
# ============================================================================


class TestWaveformEncoder:
    """Tests for WaveformEncoder."""

    @pytest.fixture
    def encoder(self):
        return WaveformEncoder(
            in_channels=1,
            base_channels=16,
            num_layers=4,
            out_dim=128,
        )

    def test_output_shape(self, encoder):
        """Test output shape matches expected dimensions."""
        batch_size = 2
        samples = 16000  # 1 second at 16kHz

        x = torch.randn(batch_size, 1, samples)
        out = encoder(x)

        expected_frames = samples // encoder.total_stride
        assert out.shape == (batch_size, expected_frames, 128)

    def test_2d_input(self, encoder):
        """Test encoder handles [B, T] input."""
        x = torch.randn(2, 16000)
        out = encoder(x)

        assert out.dim() == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 128

    def test_total_stride(self, encoder):
        """Verify total stride calculation."""
        # Default strides: [4, 2, 2, 2] -> 32
        assert encoder.total_stride == 32

    def test_custom_strides(self):
        """Test encoder with custom strides."""
        encoder = WaveformEncoder(
            kernel_sizes=[5, 3, 3],
            strides=[2, 2, 2],
            num_layers=3,
            out_dim=64,
        )

        assert encoder.total_stride == 8

        x = torch.randn(1, 1, 8000)
        out = encoder(x)
        assert out.shape[1] == 8000 // 8

    def test_gradient_flow(self, encoder):
        """Verify gradients flow through encoder."""
        x = torch.randn(2, 16000, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


# ============================================================================
# PhaseEncoder Tests
# ============================================================================


class TestPhaseEncoder:
    """Tests for PhaseEncoder."""

    @pytest.fixture
    def encoder(self):
        return PhaseEncoder(
            n_freqs=257,  # FFT=512
            conv_ch=16,
            out_dim=128,
        )

    def test_output_shape_angle(self, encoder):
        """Test with angle input [B, T, F]."""
        batch_size = 2
        time_frames = 50

        phase = torch.randn(batch_size, time_frames, 257) * torch.pi
        out = encoder(phase)

        assert out.shape == (batch_size, time_frames, 128)

    def test_output_shape_complex(self, encoder):
        """Test with complex input [B, 1, T, F, 2]."""
        batch_size = 2
        time_frames = 50

        phase = torch.randn(batch_size, 1, time_frames, 257, 2)
        out = encoder(phase)

        assert out.shape == (batch_size, time_frames, 128)

    def test_output_shape_4d(self, encoder):
        """Test with [B, 1, T, F] input."""
        phase = torch.randn(2, 1, 50, 257)
        out = encoder(phase)

        assert out.shape == (2, 50, 128)

    def test_gradient_flow(self, encoder):
        """Verify gradients flow through encoder."""
        phase = torch.randn(2, 50, 257, requires_grad=True)
        out = encoder(phase)
        loss = out.sum()
        loss.backward()

        assert phase.grad is not None

    def test_phase_wrapping(self, encoder):
        """Test encoder handles wrapped phase values."""
        # Values outside [-pi, pi]
        phase = torch.randn(2, 50, 257) * 2 * torch.pi
        out = encoder(phase)

        # Should still produce valid output
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ============================================================================
# CrossDomainAttention Tests
# ============================================================================


class TestCrossDomainAttention:
    """Tests for CrossDomainAttention."""

    @pytest.fixture
    def attention(self):
        return CrossDomainAttention(
            time_dim=128,
            mag_dim=256,
            phase_dim=128,
            out_dim=256,
            num_heads=8,
        )

    def test_output_shape(self, attention):
        """Test output shape."""
        batch_size = 2
        seq_len = 50

        time_feat = torch.randn(batch_size, seq_len, 128)
        mag_feat = torch.randn(batch_size, seq_len, 256)
        phase_feat = torch.randn(batch_size, seq_len, 128)

        out = attention(time_feat, mag_feat, phase_feat)

        assert out.shape == (batch_size, seq_len, 256)

    def test_gradient_flow(self, attention):
        """Verify gradients flow to all inputs."""
        time_feat = torch.randn(2, 50, 128, requires_grad=True)
        mag_feat = torch.randn(2, 50, 256, requires_grad=True)
        phase_feat = torch.randn(2, 50, 128, requires_grad=True)

        out = attention(time_feat, mag_feat, phase_feat)
        loss = out.sum()
        loss.backward()

        assert time_feat.grad is not None
        assert mag_feat.grad is not None
        assert phase_feat.grad is not None

    def test_different_batch_sizes(self, attention):
        """Test with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            time_feat = torch.randn(batch_size, 50, 128)
            mag_feat = torch.randn(batch_size, 50, 256)
            phase_feat = torch.randn(batch_size, 50, 128)

            out = attention(time_feat, mag_feat, phase_feat)
            assert out.shape[0] == batch_size


class TestSimpleCrossAttention:
    """Tests for SimpleCrossAttention."""

    def test_two_domains(self):
        """Test fusion of two domains."""
        attention = SimpleCrossAttention(
            input_dims=[128, 256],
            out_dim=256,
        )

        feat1 = torch.randn(2, 50, 128)
        feat2 = torch.randn(2, 50, 256)

        out = attention(feat1, feat2)
        assert out.shape == (2, 50, 256)

    def test_three_domains(self):
        """Test fusion of three domains."""
        attention = SimpleCrossAttention(
            input_dims=[64, 128, 256],
            out_dim=128,
        )

        feat1 = torch.randn(2, 50, 64)
        feat2 = torch.randn(2, 50, 128)
        feat3 = torch.randn(2, 50, 256)

        out = attention(feat1, feat2, feat3)
        assert out.shape == (2, 50, 128)


# ============================================================================
# MagnitudeEncoder Tests
# ============================================================================


class TestMagnitudeEncoder:
    """Tests for MagnitudeEncoder."""

    @pytest.fixture
    def encoder(self):
        return MagnitudeEncoder(
            conv_ch=16,
            nb_erb=32,
            nb_df=96,
            emb_hidden_dim=256,
        )

    def test_output_shapes(self, encoder):
        """Test all output shapes."""
        batch_size = 2
        time_frames = 50

        feat_erb = torch.randn(batch_size, 1, time_frames, 32)
        feat_spec = torch.randn(batch_size, 2, time_frames, 96)

        e0, e1, e2, e3, emb, c0 = encoder(feat_erb, feat_spec)

        # Check intermediate outputs
        assert e0.shape == (batch_size, 16, time_frames, 32)
        assert e1.shape == (batch_size, 16, time_frames, 16)
        assert e2.shape == (batch_size, 16, time_frames, 8)
        assert e3.shape == (batch_size, 16, time_frames, 8)

        # Check embedding
        assert emb.shape == (batch_size, time_frames, 16 * 32 // 4)

        # Check DF pathway
        assert c0.shape == (batch_size, 16, time_frames, 96)

    def test_gradient_flow(self, encoder):
        """Verify gradients flow through all outputs."""
        feat_erb = torch.randn(2, 1, 50, 32, requires_grad=True)
        feat_spec = torch.randn(2, 2, 50, 96, requires_grad=True)

        e0, e1, e2, e3, emb, c0 = encoder(feat_erb, feat_spec)

        loss = e0.sum() + e1.sum() + e2.sum() + e3.sum() + emb.sum() + c0.sum()
        loss.backward()

        assert feat_erb.grad is not None
        assert feat_spec.grad is not None


# ============================================================================
# HybridEncoder Tests
# ============================================================================


class TestHybridEncoder:
    """Tests for HybridEncoder."""

    @pytest.fixture
    def encoder(self):
        return HybridEncoder(
            conv_ch=16,
            nb_erb=32,
            nb_df=96,
            fft_size=512,
            emb_hidden_dim=256,
            emb_num_layers=2,
            use_time_branch=False,  # Disabled to simplify tests
            use_phase_branch=True,
            use_mamba=True,
        )

    @pytest.fixture
    def encoder_no_mamba(self):
        return HybridEncoder(
            conv_ch=16,
            nb_erb=32,
            nb_df=96,
            fft_size=512,
            emb_hidden_dim=256,
            emb_num_layers=2,
            use_time_branch=False,
            use_phase_branch=False,
            use_mamba=False,
        )

    def test_output_shapes(self, encoder):
        """Test all output shapes with full config."""
        batch_size = 2
        time_frames = 50

        feat_erb = torch.randn(batch_size, 1, time_frames, 32)
        feat_spec = torch.randn(batch_size, 2, time_frames, 96)

        e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)

        # Check skip connection outputs
        assert e0.shape[0] == batch_size
        assert e1.shape[0] == batch_size
        assert e2.shape[0] == batch_size
        assert e3.shape[0] == batch_size

        # Check embedding matches expected dimension
        assert emb.shape == (batch_size, time_frames, 16 * 32 // 4)

        # Check LSNR
        assert lsnr.shape == (batch_size, time_frames, 1)

    def test_without_waveform(self, encoder):
        """Test encoder works without waveform input."""
        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 2, 50, 96)

        # Should work without waveform
        e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)

        assert emb.shape[0] == 2
        assert lsnr.shape == (2, 50, 1)

    def test_gru_fallback(self, encoder_no_mamba):
        """Test encoder with GRU fallback."""
        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 2, 50, 96)

        e0, e1, e2, e3, emb, c0, lsnr = encoder_no_mamba(feat_erb, feat_spec)

        assert emb.shape[0] == 2

    def test_lsnr_range(self, encoder):
        """Test LSNR output is in expected range."""
        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 2, 50, 96)

        _, _, _, _, _, _, lsnr = encoder(feat_erb, feat_spec)

        # LSNR should be in [lsnr_min, lsnr_max] range
        assert lsnr.min() >= -15.0
        assert lsnr.max() <= 40.0

    def test_gradient_flow(self, encoder):
        """Verify gradients flow through encoder."""
        feat_erb = torch.randn(2, 1, 50, 32, requires_grad=True)
        feat_spec = torch.randn(2, 2, 50, 96, requires_grad=True)

        e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)

        loss = emb.sum() + lsnr.sum()
        loss.backward()

        assert feat_erb.grad is not None
        assert feat_spec.grad is not None

    def test_batch_size_one(self, encoder):
        """Test with batch size 1."""
        feat_erb = torch.randn(1, 1, 50, 32)
        feat_spec = torch.randn(1, 2, 50, 96)

        e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)

        assert emb.shape[0] == 1


# ============================================================================
# LightweightHybridEncoder Tests
# ============================================================================


class TestLightweightHybridEncoder:
    """Tests for LightweightHybridEncoder."""

    @pytest.fixture
    def encoder(self):
        return LightweightHybridEncoder(
            conv_ch=16,
            nb_erb=32,
            nb_df=96,
            fft_size=512,
            emb_hidden_dim=256,
            emb_num_layers=2,
            use_time_branch=False,
            use_phase_branch=True,
        )

    def test_output_shapes(self, encoder):
        """Test output shapes."""
        batch_size = 2
        time_frames = 50

        feat_erb = torch.randn(batch_size, 1, time_frames, 32)
        feat_spec = torch.randn(batch_size, 2, time_frames, 96)

        e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)

        assert emb.shape == (batch_size, time_frames, 16 * 32 // 4)
        assert lsnr.shape == (batch_size, time_frames, 1)

    def test_fewer_parameters(self):
        """Verify lightweight version has fewer parameters."""
        full = HybridEncoder(
            use_time_branch=True,
            use_phase_branch=True,
        )
        lightweight = LightweightHybridEncoder(
            use_time_branch=False,
            use_phase_branch=True,
        )

        full_params = sum(p.numel() for p in full.parameters())
        light_params = sum(p.numel() for p in lightweight.parameters())

        assert light_params < full_params


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for hybrid encoder with other components."""

    def test_encoder_decoder_compatibility(self):
        """Test encoder output is compatible with expected decoder input."""
        encoder = HybridEncoder(
            conv_ch=16,
            nb_erb=32,
            nb_df=96,
            fft_size=512,
            use_time_branch=False,
            use_phase_branch=True,
        )

        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 2, 50, 96)

        e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)

        # Check skip connection dimensions for decoder
        assert e0.shape[1] == 16  # conv_ch
        assert e0.shape[3] == 32  # nb_erb
        assert e1.shape[3] == 16  # nb_erb / 2
        assert e2.shape[3] == 8  # nb_erb / 4
        assert e3.shape[3] == 8  # nb_erb / 4

        # DF pathway
        assert c0.shape[1] == 16  # conv_ch
        assert c0.shape[3] == 96  # nb_df

    def test_different_fft_sizes(self):
        """Test encoder works with different FFT sizes."""
        for fft_size in [256, 512, 960]:
            encoder = HybridEncoder(
                fft_size=fft_size,
                nb_df=96,  # Keep nb_df constant for testing
                use_time_branch=False,
                use_phase_branch=True,
            )

            fft_size // 2 + 1
            feat_erb = torch.randn(2, 1, 50, 32)
            feat_spec = torch.randn(2, 2, 50, 96)

            # Should not raise
            e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)


# ============================================================================
# Device Tests
# ============================================================================


class TestDevices:
    """Test encoder on different devices."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test encoder on CUDA."""
        encoder = HybridEncoder(
            use_time_branch=False,
            use_phase_branch=True,
        ).cuda()

        feat_erb = torch.randn(2, 1, 50, 32).cuda()
        feat_spec = torch.randn(2, 2, 50, 96).cuda()

        e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)

        assert emb.device.type == "cuda"

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_mps(self):
        """Test encoder on MPS (Apple Silicon)."""
        device = torch.device("mps")
        encoder = HybridEncoder(
            use_time_branch=False,  # Simpler for MPS test
            use_phase_branch=True,
        ).to(device)

        feat_erb = torch.randn(2, 1, 50, 32).to(device)
        feat_spec = torch.randn(2, 2, 50, 96).to(device)

        e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)

        assert emb.device.type == "mps"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_short_sequence(self):
        """Test with very short sequences."""
        encoder = HybridEncoder(
            use_time_branch=False,
            use_phase_branch=False,
        )

        feat_erb = torch.randn(1, 1, 5, 32)
        feat_spec = torch.randn(1, 2, 5, 96)

        # Should handle short sequences
        e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)
        assert emb.shape[1] == 5

    def test_long_sequence(self):
        """Test with long sequences."""
        encoder = HybridEncoder(
            use_time_branch=False,
            use_phase_branch=False,
        )

        feat_erb = torch.randn(1, 1, 500, 32)
        feat_spec = torch.randn(1, 2, 500, 96)

        e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)
        assert emb.shape[1] == 500

    def test_eval_mode(self):
        """Test encoder in eval mode."""
        encoder = HybridEncoder(
            use_time_branch=False,
            use_phase_branch=True,
        )
        encoder.eval()

        feat_erb = torch.randn(2, 1, 50, 32)
        feat_spec = torch.randn(2, 2, 50, 96)

        with torch.no_grad():
            e0, e1, e2, e3, emb, c0, lsnr = encoder(feat_erb, feat_spec)

        assert emb.shape[0] == 2
