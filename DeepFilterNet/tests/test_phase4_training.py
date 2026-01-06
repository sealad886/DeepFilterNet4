"""Tests for Phase 4: Training Enhancements (GAN, DNSMOS, Speaker loss)."""

import pytest
import torch

# Import modules under test
from df.discriminator import (
    CombinedDiscriminator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    PeriodDiscriminator,
    ScaleDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
)
from df.dnsmos_proxy import DNSMOSLoss, DNSMOSProxy, LightweightDNSMOSProxy, MelSpectrogram
from df.loss import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, SpeakerContrastiveLoss

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_length():
    return 16000  # 1 second at 16kHz


@pytest.fixture
def waveform(batch_size, seq_length):
    """Generate random waveform."""
    return torch.randn(batch_size, seq_length)


@pytest.fixture
def waveform_3d(batch_size, seq_length):
    """Generate random waveform with channel dimension."""
    return torch.randn(batch_size, 1, seq_length)


# ============================================================================
# PeriodDiscriminator Tests
# ============================================================================


class TestPeriodDiscriminator:
    def test_output_shape(self, waveform_3d):
        """Test PeriodDiscriminator produces correct output shapes."""
        disc = PeriodDiscriminator(period=2)
        score, fmaps = disc(waveform_3d)

        # Score should be flattened
        assert score.dim() == 2
        assert score.shape[0] == waveform_3d.shape[0]

        # Should have feature maps from each conv layer
        assert len(fmaps) == 6  # 5 conv + 1 post conv

    def test_different_periods(self, waveform_3d):
        """Test different period values."""
        for period in [2, 3, 5, 7, 11]:
            disc = PeriodDiscriminator(period=period)
            score, fmaps = disc(waveform_3d)
            assert score.shape[0] == waveform_3d.shape[0]

    def test_spectral_norm(self, waveform_3d):
        """Test with spectral normalization."""
        disc = PeriodDiscriminator(period=3, use_spectral_norm=True)
        score, fmaps = disc(waveform_3d)
        assert score.shape[0] == waveform_3d.shape[0]

    def test_2d_input(self, waveform):
        """Test with 2D input [B, T]."""
        disc = PeriodDiscriminator(period=5)
        score, fmaps = disc(waveform)
        assert score.shape[0] == waveform.shape[0]


# ============================================================================
# ScaleDiscriminator Tests
# ============================================================================


class TestScaleDiscriminator:
    def test_output_shape(self, waveform_3d):
        """Test ScaleDiscriminator produces correct output shapes."""
        disc = ScaleDiscriminator()
        score, fmaps = disc(waveform_3d)

        assert score.dim() == 2
        assert score.shape[0] == waveform_3d.shape[0]
        assert len(fmaps) == 8  # 7 conv + 1 post conv

    def test_spectral_norm(self, waveform_3d):
        """Test with spectral normalization."""
        disc = ScaleDiscriminator(use_spectral_norm=True)
        score, fmaps = disc(waveform_3d)
        assert score.shape[0] == waveform_3d.shape[0]


# ============================================================================
# MultiPeriodDiscriminator Tests
# ============================================================================


class TestMultiPeriodDiscriminator:
    def test_default_periods(self, waveform):
        """Test MPD with default periods."""
        mpd = MultiPeriodDiscriminator()
        scores, fmaps = mpd(waveform)

        # 5 sub-discriminators by default
        assert len(scores) == 5
        assert len(fmaps) == 5

    def test_custom_periods(self, waveform):
        """Test MPD with custom periods."""
        periods = [2, 3, 5]
        mpd = MultiPeriodDiscriminator(periods=periods)
        scores, fmaps = mpd(waveform)

        assert len(scores) == len(periods)
        assert len(fmaps) == len(periods)

    def test_batch_consistency(self, batch_size, seq_length):
        """Test batch processing is consistent."""
        mpd = MultiPeriodDiscriminator()

        x1 = torch.randn(1, seq_length)
        x2 = torch.randn(1, seq_length)

        scores_1, _ = mpd(x1)
        scores_2, _ = mpd(x2)
        scores_batch, _ = mpd(torch.cat([x1, x2], dim=0))

        for s1, s2, sb in zip(scores_1, scores_2, scores_batch):
            assert torch.allclose(s1[0], sb[0], atol=1e-5)
            assert torch.allclose(s2[0], sb[1], atol=1e-5)


# ============================================================================
# MultiScaleDiscriminator Tests
# ============================================================================


class TestMultiScaleDiscriminator:
    def test_default_scales(self, waveform):
        """Test MSD with default scales."""
        msd = MultiScaleDiscriminator()
        scores, fmaps = msd(waveform)

        # 3 scales by default
        assert len(scores) == 3
        assert len(fmaps) == 3

    def test_custom_scales(self, waveform):
        """Test MSD with custom number of scales."""
        msd = MultiScaleDiscriminator(num_scales=2)
        scores, fmaps = msd(waveform)

        assert len(scores) == 2
        assert len(fmaps) == 2


# ============================================================================
# CombinedDiscriminator Tests
# ============================================================================


class TestCombinedDiscriminator:
    def test_combined_output(self, waveform):
        """Test combined discriminator output."""
        disc = CombinedDiscriminator()
        mpd_scores, mpd_fmaps, msd_scores, msd_fmaps = disc(waveform)

        assert len(mpd_scores) == 5
        assert len(msd_scores) == 3


# ============================================================================
# Loss Function Tests
# ============================================================================


class TestDiscriminatorLossFunctions:
    def test_discriminator_loss(self):
        """Test discriminator_loss function."""
        real_scores = [torch.randn(4, 100) for _ in range(3)]
        fake_scores = [torch.randn(4, 100) for _ in range(3)]

        d_loss, d_real = discriminator_loss(real_scores, fake_scores)

        assert d_loss.dim() == 0  # Scalar
        assert d_real.dim() == 0
        assert d_loss >= 0  # LS-GAN loss is always non-negative

    def test_generator_loss(self):
        """Test generator_loss function."""
        fake_scores = [torch.randn(4, 100) for _ in range(3)]

        g_loss = generator_loss(fake_scores)

        assert g_loss.dim() == 0

    def test_feature_matching_loss(self):
        """Test feature_matching_loss function."""
        # Create fake feature maps
        real_fmaps = [[torch.randn(4, 32, 10, 10) for _ in range(3)] for _ in range(2)]
        fake_fmaps = [[torch.randn(4, 32, 10, 10) for _ in range(3)] for _ in range(2)]

        fm_loss = feature_matching_loss(real_fmaps, fake_fmaps)

        assert fm_loss.dim() == 0
        assert fm_loss >= 0


# ============================================================================
# FeatureMatchingLoss Module Tests
# ============================================================================


class TestFeatureMatchingLossModule:
    def test_forward(self):
        """Test FeatureMatchingLoss module."""
        loss = FeatureMatchingLoss(factor=2.0)

        real_fmaps = [[torch.randn(4, 32, 10, 10) for _ in range(3)] for _ in range(2)]
        fake_fmaps = [[torch.randn(4, 32, 10, 10) for _ in range(3)] for _ in range(2)]

        fm_loss = loss(real_fmaps, fake_fmaps)

        assert fm_loss.dim() == 0
        assert fm_loss >= 0

    def test_gradient_flow(self):
        """Test gradients flow through fake features."""
        loss = FeatureMatchingLoss(factor=1.0)

        real_fmaps = [[torch.randn(2, 16, 5, 5) for _ in range(2)] for _ in range(2)]
        fake_fmaps = [
            [torch.randn(2, 16, 5, 5, requires_grad=True) for _ in range(2)] for _ in range(2)
        ]

        fm_loss = loss(real_fmaps, fake_fmaps)
        fm_loss.backward()

        # Gradients should flow to fake features
        for fmap_list in fake_fmaps:
            for fmap in fmap_list:
                assert fmap.grad is not None


# ============================================================================
# GeneratorLoss Module Tests
# ============================================================================


class TestGeneratorLossModule:
    @pytest.mark.parametrize("loss_type", ["lsgan", "vanilla", "hinge"])
    def test_loss_types(self, loss_type):
        """Test different GAN loss types."""
        loss = GeneratorLoss(loss_type=loss_type, factor=1.0)

        fake_scores = [torch.randn(4, 100) for _ in range(3)]
        g_loss = loss(fake_scores)

        assert g_loss.dim() == 0

    def test_factor_scaling(self):
        """Test factor correctly scales loss."""
        loss1 = GeneratorLoss(factor=1.0)
        loss2 = GeneratorLoss(factor=2.0)

        fake_scores = [torch.randn(4, 100)]

        g1 = loss1(fake_scores)
        g2 = loss2(fake_scores)

        assert torch.allclose(g2, g1 * 2, atol=1e-5)


# ============================================================================
# DiscriminatorLoss Module Tests
# ============================================================================


class TestDiscriminatorLossModule:
    @pytest.mark.parametrize("loss_type", ["lsgan", "vanilla", "hinge"])
    def test_loss_types(self, loss_type):
        """Test different GAN loss types."""
        loss = DiscriminatorLoss(loss_type=loss_type, factor=1.0)

        real_scores = [torch.randn(4, 100) for _ in range(3)]
        fake_scores = [torch.randn(4, 100) for _ in range(3)]

        d_loss, d_real = loss(real_scores, fake_scores)

        assert d_loss.dim() == 0
        assert d_real.dim() == 0


# ============================================================================
# MelSpectrogram Tests
# ============================================================================


class TestMelSpectrogram:
    def test_output_shape(self, waveform):
        """Test mel spectrogram output shape."""
        mel = MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=64)
        mel_spec = mel(waveform)

        # Expected time frames
        waveform.shape[-1] // 256 + 1

        assert mel_spec.dim() == 3
        assert mel_spec.shape[0] == waveform.shape[0]
        assert mel_spec.shape[1] == 64  # n_mels

    def test_3d_input(self, waveform_3d):
        """Test with 3D input [B, 1, T]."""
        mel = MelSpectrogram()
        mel_spec = mel(waveform_3d)

        assert mel_spec.dim() == 3
        assert mel_spec.shape[0] == waveform_3d.shape[0]


# ============================================================================
# DNSMOSProxy Tests
# ============================================================================


class TestDNSMOSProxy:
    def test_output_scores(self, waveform):
        """Test DNSMOSProxy produces expected scores."""
        proxy = DNSMOSProxy()
        scores = proxy(waveform)

        assert "sig" in scores
        assert "bak" in scores
        assert "ovl" in scores

        # Scores should be in [1, 5] range
        for key in ["sig", "bak", "ovl"]:
            assert torch.all(scores[key] >= 1.0)
            assert torch.all(scores[key] <= 5.0)

    def test_batch_processing(self, batch_size, seq_length):
        """Test batch processing."""
        proxy = DNSMOSProxy()
        waveform = torch.randn(batch_size, seq_length)

        scores = proxy(waveform)

        assert scores["ovl"].shape == (batch_size,)

    def test_compute_loss_no_target(self, waveform):
        """Test compute_loss without targets (maximize quality)."""
        proxy = DNSMOSProxy()
        loss = proxy.compute_loss(waveform)

        assert loss.dim() == 0

    def test_compute_loss_with_target(self, waveform):
        """Test compute_loss with target scores."""
        proxy = DNSMOSProxy()
        target_ovl = torch.tensor([4.0, 4.0])

        loss = proxy.compute_loss(waveform, target_ovl=target_ovl)

        assert loss.dim() == 0

    def test_gradient_flow(self):
        """Test gradients flow through proxy."""
        proxy = DNSMOSProxy()
        waveform = torch.randn(2, 8000, requires_grad=True)

        scores = proxy(waveform)
        loss = -scores["ovl"].mean()
        loss.backward()

        assert waveform.grad is not None


# ============================================================================
# LightweightDNSMOSProxy Tests
# ============================================================================


class TestLightweightDNSMOSProxy:
    def test_output_scores(self, waveform):
        """Test lightweight proxy produces expected scores."""
        proxy = LightweightDNSMOSProxy()
        scores = proxy(waveform)

        assert "sig" in scores
        assert "bak" in scores
        assert "ovl" in scores

    def test_parameter_count(self):
        """Test lightweight proxy has fewer parameters."""
        full = DNSMOSProxy()
        lite = LightweightDNSMOSProxy()

        full_params = sum(p.numel() for p in full.parameters())
        lite_params = sum(p.numel() for p in lite.parameters())

        assert lite_params < full_params


# ============================================================================
# DNSMOSLoss Tests
# ============================================================================


class TestDNSMOSLoss:
    def test_forward_without_clean(self, waveform):
        """Test DNSMOSLoss forward without clean reference."""
        loss_fn = DNSMOSLoss(freeze_proxy=True, target_ovl=4.5)
        loss, scores = loss_fn(waveform)

        assert loss.dim() == 0
        assert "ovl" in scores

    def test_forward_with_clean(self, waveform):
        """Test DNSMOSLoss forward with clean reference."""
        loss_fn = DNSMOSLoss(freeze_proxy=True)
        clean = torch.randn_like(waveform)

        loss, scores = loss_fn(waveform, clean_audio=clean)

        assert loss.dim() == 0

    def test_frozen_proxy(self, waveform):
        """Test proxy is frozen when freeze_proxy=True."""
        loss_fn = DNSMOSLoss(freeze_proxy=True)

        for param in loss_fn.proxy.parameters():
            assert not param.requires_grad


# ============================================================================
# SpeakerContrastiveLoss Tests
# ============================================================================


class TestSpeakerContrastiveLoss:
    def test_initialization(self):
        """Test SpeakerContrastiveLoss initialization."""
        loss = SpeakerContrastiveLoss(factor=0.1)

        assert loss.factor == 0.1
        assert loss.speaker_encoder is None  # Lazy loaded

    def test_forward_without_encoder(self):
        """Test forward returns zero when encoder not available."""
        loss = SpeakerContrastiveLoss(factor=0.1)

        # Force encoder to None (simulating missing resemblyzer)
        loss._encoder_loaded = True
        loss.speaker_encoder = None

        enhanced = torch.randn(2, 16000)
        clean = torch.randn(2, 16000)

        result = loss(enhanced, clean)

        # Should return zero when encoder not available
        assert result.item() == 0.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestGANTrainingIntegration:
    def test_full_gan_step(self, waveform):
        """Test full GAN training step with all components."""
        # Create discriminator
        mpd = MultiPeriodDiscriminator()

        # Create losses
        g_loss_fn = GeneratorLoss(factor=1.0)
        d_loss_fn = DiscriminatorLoss(factor=1.0)
        fm_loss_fn = FeatureMatchingLoss(factor=2.0)

        # Simulate clean and enhanced audio
        clean = waveform
        enhanced = waveform + torch.randn_like(waveform) * 0.1

        # Discriminator forward
        real_scores, real_fmaps = mpd(clean)
        fake_scores, fake_fmaps = mpd(enhanced)

        # Compute discriminator loss
        d_loss, d_real = d_loss_fn(real_scores, fake_scores)

        # Compute generator losses
        g_adv_loss = g_loss_fn(fake_scores)
        fm_loss = fm_loss_fn(real_fmaps, fake_fmaps)

        # Total generator loss
        g_total = g_adv_loss + fm_loss

        # All losses should be scalars
        assert d_loss.dim() == 0
        assert g_adv_loss.dim() == 0
        assert fm_loss.dim() == 0
        assert g_total.dim() == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gan_on_cuda(self):
        """Test GAN components on CUDA."""
        device = torch.device("cuda")

        mpd = MultiPeriodDiscriminator().to(device)
        waveform = torch.randn(2, 16000, device=device)

        scores, fmaps = mpd(waveform)

        assert all(s.device.type == "cuda" for s in scores)

    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available",
    )
    def test_gan_on_mps(self):
        """Test GAN components on MPS (Apple Silicon)."""
        device = torch.device("mps")

        mpd = MultiPeriodDiscriminator().to(device)
        waveform = torch.randn(2, 16000, device=device)

        scores, fmaps = mpd(waveform)

        assert all(s.device.type == "mps" for s in scores)


class TestDNSMOSIntegration:
    def test_full_dnsmos_training_step(self):
        """Test DNSMOS proxy loss in training context."""
        proxy = DNSMOSProxy()

        # Simulate enhanced audio that requires gradients
        enhanced = torch.randn(2, 16000, requires_grad=True)

        # Use compute_loss method which is designed for training
        # This maximizes quality (minimizes negative score)
        loss = proxy.compute_loss(enhanced)

        # Backward pass
        loss.backward()

        # Gradients should flow
        assert enhanced.grad is not None
        # Note: gradients may be zero if scores hit clamp boundaries
        # This is expected behavior - we just verify the graph is connected
        assert enhanced.grad.shape == enhanced.shape

    def test_dnsmos_gradient_flow_with_targets(self):
        """Test DNSMOS proxy gradient flow with regression targets."""
        proxy = DNSMOSProxy()

        # Simulate enhanced audio
        enhanced = torch.randn(2, 16000, requires_grad=True)

        # Use targets in the middle of the range so gradients flow
        target_ovl = torch.tensor([3.5, 3.5])

        # Compute loss with targets
        loss = proxy.compute_loss(enhanced, target_ovl=target_ovl)

        # Backward pass
        loss.backward()

        # With targets in valid range, gradients should flow
        assert enhanced.grad is not None
        assert enhanced.grad.shape == enhanced.shape


class TestLossGANIntegration:
    """Test GAN integration methods in Loss class."""

    def test_loss_accepts_discriminator(self, df_config):
        """Test Loss class accepts discriminator parameter."""
        from df.loss import Loss
        from libdf import DF

        # Create minimal DF state
        df_state = DF(sr=48000, fft_size=960, hop_size=480, nb_bands=32)

        # Create discriminator
        mpd = MultiPeriodDiscriminator(periods=[2, 3])

        # Loss should accept discriminator
        loss = Loss(df_state, istft=None, discriminator=mpd)
        assert loss.discriminator is mpd

    def test_loss_run_discriminator(self, df_config):
        """Test Loss.run_discriminator method."""
        from df.loss import Loss
        from libdf import DF

        df_state = DF(sr=48000, fft_size=960, hop_size=480, nb_bands=32)
        mpd = MultiPeriodDiscriminator(periods=[2, 3])
        loss = Loss(df_state, istft=None, discriminator=mpd)

        # Test with 2D input
        waveform = torch.randn(2, 16000)
        scores, fmaps = loss.run_discriminator(mpd, waveform)

        assert len(scores) == 2  # 2 periods
        assert len(fmaps) == 2
        assert all(s.shape[0] == 2 for s in scores)

        # Test with 3D input
        waveform_3d = torch.randn(2, 1, 16000)
        scores, fmaps = loss.run_discriminator(mpd, waveform_3d)
        assert len(scores) == 2

    def test_compute_d_loss_with_disc(self, df_config):
        """Test Loss.compute_d_loss_with_disc method."""
        from df.loss import DiscriminatorLoss, Loss
        from libdf import DF

        df_state = DF(sr=48000, fft_size=960, hop_size=480, nb_bands=32)
        mpd = MultiPeriodDiscriminator(periods=[2, 3])
        loss = Loss(df_state, istft=None, discriminator=mpd)

        # Manually set disc_loss for testing
        loss.disc_loss = DiscriminatorLoss(loss_type="lsgan", factor=1.0)

        # Real and fake waveforms
        real_wav = torch.randn(2, 16000)
        fake_wav = torch.randn(2, 16000)

        # Compute loss
        d_loss = loss.compute_d_loss_with_disc(mpd, real_wav, fake_wav)

        # Should be a scalar tensor
        assert d_loss.dim() == 0
        assert d_loss.item() >= 0

    def test_compute_g_loss_with_disc(self, df_config):
        """Test Loss.compute_g_loss_with_disc method."""
        from df.loss import FeatureMatchingLoss, GeneratorLoss, Loss
        from libdf import DF

        df_state = DF(sr=48000, fft_size=960, hop_size=480, nb_bands=32)
        mpd = MultiPeriodDiscriminator(periods=[2, 3])
        loss = Loss(df_state, istft=None, discriminator=mpd)

        # Set up generator losses
        loss.gen_loss = GeneratorLoss(loss_type="lsgan", factor=1.0)
        loss.fm_loss = FeatureMatchingLoss(factor=2.0)

        # Real and fake waveforms
        real_wav = torch.randn(2, 16000)
        fake_wav = torch.randn(2, 16000, requires_grad=True)

        # Compute loss
        g_loss = loss.compute_g_loss_with_disc(mpd, real_wav, fake_wav)

        # Should be a scalar tensor
        assert g_loss.dim() == 0

        # Should be differentiable
        g_loss.backward()
        assert fake_wav.grad is not None

    def test_gan_training_step_simulation(self, df_config):
        """Simulate a full GAN training step like in train.py."""
        from df.loss import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, Loss
        from libdf import DF

        # Setup
        df_state = DF(sr=48000, fft_size=960, hop_size=480, nb_bands=32)
        mpd = MultiPeriodDiscriminator(periods=[2, 3])
        loss = Loss(df_state, istft=None, discriminator=mpd)

        # Set up losses
        loss.gen_loss = GeneratorLoss(loss_type="lsgan", factor=1.0)
        loss.disc_loss = DiscriminatorLoss(loss_type="lsgan", factor=1.0)
        loss.fm_loss = FeatureMatchingLoss(factor=2.0)

        opt_disc = torch.optim.Adam(mpd.parameters(), lr=1e-4)

        # Simulated waveforms
        clean_wav = torch.randn(2, 16000)
        enhanced_wav = torch.randn(2, 16000, requires_grad=True)

        # === Discriminator step ===
        opt_disc.zero_grad()
        d_loss = loss.compute_d_loss_with_disc(mpd, clean_wav, enhanced_wav.detach())
        d_loss.backward()
        opt_disc.step()

        # === Generator step ===
        g_loss = loss.compute_g_loss_with_disc(mpd, clean_wav, enhanced_wav)

        # Backward should work
        g_loss.backward()

        # Both losses should be reasonable values
        assert d_loss.item() >= 0
        assert not torch.isnan(d_loss)
        assert not torch.isnan(g_loss)

        # Generator gradients should flow
        assert enhanced_wav.grad is not None
