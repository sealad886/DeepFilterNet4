"""Loss function audit tests — validates fixes from the loss correctness audit.

Tests cover:
- arctan2 gradient explosion fix (complex spectral loss)
- SegmentalSiSdrLoss silence handling, mean removal, float16, short signals
- SdrLoss float16 promotion
- Mask saturation penalty removed from training loss (zero-gradient term)
- CombinedLoss additive composition
- GAN loss mathematical correctness
"""

import math

import mlx.core as mx

# ─────────────────────────────────────────────────────────────────────────────
# Complex spectral loss — arctan2 gradient fix
# ─────────────────────────────────────────────────────────────────────────────


class TestComplexSpectralGradientStability:
    """Verify arctan2 replacement produces stable gradients on quiet signals."""

    def test_loss_py_complex_gradient_finite_on_silence(self):
        """SpectralLoss (loss.py) complex path should not explode on silence."""
        from df_mlx.loss import SpectralLoss

        loss_fn = SpectralLoss(fft_sizes=(512,), gamma=0.3, factor=1.0, factor_complex=0.5)
        target = mx.zeros((1, 4096)) + 1e-10

        def compute(pred):
            return loss_fn(pred, target)

        pred = mx.zeros((1, 4096)) + 1e-10
        loss, grad = mx.value_and_grad(compute)(pred)
        mx.eval(loss, grad)
        grad_norm = float(mx.sqrt(mx.sum(grad**2)))
        assert math.isfinite(float(loss)), f"Loss should be finite, got {float(loss)}"
        assert math.isfinite(grad_norm), f"Gradient norm should be finite, got {grad_norm}"
        assert grad_norm < 1e6, f"Gradient should not explode, norm={grad_norm}"

    def test_train_py_complex_gradient_finite_on_silence(self):
        """MultiResolutionSTFTLoss (train.py) complex path: stable on quiet signals."""
        from df_mlx.train import MultiResolutionSTFTLoss

        loss_fn = MultiResolutionSTFTLoss(fft_sizes=(512,), gamma=0.3, factor=1.0, f_complex=0.15)
        target = mx.random.normal((1, 4096)) * 0.001

        def compute(pred):
            return loss_fn(pred, target)

        pred = mx.random.normal((1, 4096)) * 0.001
        loss, grad = mx.value_and_grad(compute)(pred)
        mx.eval(loss, grad)
        grad_norm = float(mx.sqrt(mx.sum(grad**2)))
        assert math.isfinite(float(loss))
        assert math.isfinite(grad_norm), f"Gradient norm should be finite, got {grad_norm}"

    def test_complex_path_mathematically_equivalent(self):
        """mag_c * (real/mag, imag/mag) should equal mag_c * (cos(angle), sin(angle))."""
        real = mx.random.normal((4, 10, 257))
        imag = mx.random.normal((4, 10, 257))
        eps = 1e-7
        gamma = 0.3

        mag = mx.sqrt(real**2 + imag**2 + eps)
        mag_c = mx.power(mx.maximum(mag, eps), gamma)

        # New method: direct division
        phase = mag_c / mag
        real_c_new = phase * real
        imag_c_new = phase * imag

        # Old method: arctan2
        angle = mx.arctan2(imag, real + eps)
        real_c_old = mag_c * mx.cos(angle)
        imag_c_old = mag_c * mx.sin(angle)

        mx.eval(real_c_new, imag_c_new, real_c_old, imag_c_old)

        # They should be very close for non-silent bins
        mask = mag > 0.01  # Skip near-silent bins where arctan2 is unstable
        if float(mx.sum(mask)) > 0:
            diff_real = float(mx.max(mx.abs(real_c_new - real_c_old) * mask))
            diff_imag = float(mx.max(mx.abs(imag_c_new - imag_c_old) * mask))
            assert diff_real < 1e-4, f"Real part differs by {diff_real}"
            assert diff_imag < 1e-4, f"Imag part differs by {diff_imag}"


# ─────────────────────────────────────────────────────────────────────────────
# SegmentalSiSdrLoss fixes
# ─────────────────────────────────────────────────────────────────────────────


class TestSegmentalSiSdrLossFixes:
    """Validates all fixes to SegmentalSiSdrLoss."""

    def test_silence_eps_consistency(self):
        """Silence should produce near-zero loss, not -100 (eps fix)."""
        from df_mlx.loss import SegmentalSiSdrLoss

        loss_fn = SegmentalSiSdrLoss(segment_size=960, factor=1.0)
        x = mx.zeros((1, 4800))
        result = float(loss_fn(x, x))
        assert abs(result) < 5.0, f"Silence should give near-zero loss (eps consistency), got {result}"

    def test_mean_removal_applied(self):
        """Per-segment mean removal: DC offset should not affect SI-SDR."""
        from df_mlx.loss import SegmentalSiSdrLoss

        loss_fn = SegmentalSiSdrLoss(segment_size=960, factor=1.0)
        target = mx.random.normal((1, 4800))
        pred_with_dc = target + 50.0
        result = float(loss_fn(pred_with_dc, target))
        assert result < -30, f"DC offset should be removed by mean subtraction, got {result}"

    def test_float16_promoted(self):
        """Float16 inputs should be promoted to float32 for numerical safety."""
        from df_mlx.loss import SegmentalSiSdrLoss

        loss_fn = SegmentalSiSdrLoss(segment_size=480, factor=1.0)
        x = mx.random.normal((1, 4800)).astype(mx.float16)
        result = float(loss_fn(x, x))
        assert math.isfinite(result), f"Float16 input should be finite, got {result}"

    def test_short_signal_fallback(self):
        """Signals shorter than segment_size should fall back to global SI-SDR."""
        from df_mlx.loss import SegmentalSiSdrLoss, si_sdr

        loss_fn = SegmentalSiSdrLoss(segment_size=960, factor=1.0)
        x = mx.random.normal((1, 500))
        y = mx.random.normal((1, 500))

        segmental_result = float(loss_fn(x, y))
        global_result = float(-si_sdr(x, y) * 1.0)
        mx.eval(mx.array(segmental_result), mx.array(global_result))
        assert abs(segmental_result - global_result) < 1e-4, (
            f"Short signal should use global SI-SDR: segmental={segmental_result}, " f"global={global_result}"
        )

    def test_gradient_flows_through(self):
        """Gradient should flow through the loss to the prediction."""
        from df_mlx.loss import SegmentalSiSdrLoss

        loss_fn = SegmentalSiSdrLoss(segment_size=480, factor=1.0)
        target = mx.random.normal((1, 4800))

        def compute(pred):
            return loss_fn(pred, target)

        pred = mx.random.normal((1, 4800))
        loss, grad = mx.value_and_grad(compute)(pred)
        mx.eval(loss, grad)
        assert float(mx.sum(mx.abs(grad))) > 0, "Gradient should be non-zero"


# ─────────────────────────────────────────────────────────────────────────────
# SdrLoss float32 promotion
# ─────────────────────────────────────────────────────────────────────────────


class TestSdrLossFloat32Promotion:
    """SdrLoss should promote float16 inputs to float32."""

    def test_float16_safe(self):
        from df_mlx.loss import SdrLoss

        loss_fn = SdrLoss(factor=1.0)
        x = mx.random.normal((1, 4000)).astype(mx.float16)
        result = float(loss_fn(x, x))
        assert math.isfinite(result), f"Float16 input should be finite, got {result}"


# ─────────────────────────────────────────────────────────────────────────────
# Mask saturation — removed from training loss
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskSaturationRemoved:
    """Mask saturation is now a diagnostic metric, not in the training loss."""

    def test_mask_saturation_not_in_total_loss(self):
        """Verify mask saturation loss is computed but not in total_loss."""
        from df_mlx.training_losses import _compute_pipeline_awesome_losses

        batch, time, freq = 2, 50, 257
        clean_real = mx.random.normal((batch, time, freq))
        clean_imag = mx.random.normal((batch, time, freq))
        noisy_real = clean_real + 0.1 * mx.random.normal((batch, time, freq))
        noisy_imag = clean_imag + 0.1 * mx.random.normal((batch, time, freq))
        out_real = clean_real + 0.05 * mx.random.normal((batch, time, freq))
        out_imag = clean_imag + 0.05 * mx.random.normal((batch, time, freq))
        snr = mx.array([10.0, 15.0])
        band_mask = mx.zeros((freq,))
        band_mask = band_mask.at[10:100].add(1.0)
        band_bins = float(mx.sum(band_mask))

        result = _compute_pipeline_awesome_losses(
            noisy_real,
            noisy_imag,
            clean_real,
            clean_imag,
            out_real,
            out_imag,
            snr,
            band_mask,
            band_bins,
            mask_sharpness=5.0,
            vad_z_threshold=0.0,
            vad_z_slope=1.0,
            vad_snr_gate_db=5.0,
            vad_snr_gate_width=3.0,
            proxy_enabled=True,
        )

        total_loss = result[0]
        speech_loss = result[1]
        noise_loss = result[2]
        smooth_loss = result[3]
        music_suppression_loss = result[4]
        mask_saturation_loss = result[5]

        mx.eval(
            total_loss,
            speech_loss,
            noise_loss,
            smooth_loss,
            music_suppression_loss,
            mask_saturation_loss,
        )

        # mask_saturation_loss should be computed (non-zero for typical inputs)
        assert float(mask_saturation_loss) >= 0, "Metric should be non-negative"

        # But it should NOT be in the total loss
        from df_mlx.training_losses import (
            _PIPELINE_ARTIFACT_SMOOTH_WEIGHT,
            _PIPELINE_MUSIC_SUPPRESSION_WEIGHT,
        )

        expected_total = (
            float(speech_loss)
            + float(noise_loss)
            + _PIPELINE_ARTIFACT_SMOOTH_WEIGHT * float(smooth_loss)
            + _PIPELINE_MUSIC_SUPPRESSION_WEIGHT * float(music_suppression_loss)
        )
        actual_total = float(total_loss)
        assert abs(actual_total - expected_total) < 1e-5, (
            f"Total should NOT include mask saturation: " f"actual={actual_total}, expected={expected_total}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SI-SDR core function
# ─────────────────────────────────────────────────────────────────────────────


class TestSiSdrCore:
    """Core SI-SDR mathematical correctness."""

    def test_scale_invariance(self):
        """SI-SDR should be invariant to scaling of the prediction."""
        from df_mlx.loss import si_sdr

        target = mx.random.normal((1, 4000))
        pred = target * 3.0
        result = float(si_sdr(pred, target))
        assert result > 80, f"Scaled copy should give very high SI-SDR, got {result}"

    def test_zero_mean_removal(self):
        """SI-SDR should remove mean from both signals."""
        from df_mlx.loss import si_sdr

        target = mx.random.normal((1, 4000))
        pred = target + 100.0  # Large DC offset
        result = float(si_sdr(pred, target))
        assert result > 80, f"DC-shifted copy should give high SI-SDR, got {result}"

    def test_silence_sentinel(self):
        """Silence should yield near-zero SI-SDR (0/0 -> eps/eps -> 0 dB)."""
        from df_mlx.loss import si_sdr

        x = mx.zeros((1, 1000))
        result = float(si_sdr(x, x))
        assert abs(result) < 1.0, f"Silence should give ~0 dB, got {result}"


# ─────────────────────────────────────────────────────────────────────────────
# CombinedLoss additive composition
# ─────────────────────────────────────────────────────────────────────────────


class TestCombinedLossComposition:
    """CombinedLoss should additively compose all sub-losses."""

    def test_additive_not_multiplicative(self):
        from df_mlx.loss import CombinedLoss

        loss_fn = CombinedLoss(sisdr_factor=0.5)
        x = mx.random.normal((1, 4096))
        y = mx.random.normal((1, 4096))
        total, breakdown = loss_fn(x, y)
        mx.eval(total, *breakdown.values())
        expected = float(breakdown["spectral"]) + float(breakdown["sisdr"])
        actual = float(total)
        assert abs(actual - expected) < 1e-4, f"Total should be sum of components: {actual} vs {expected}"


# ─────────────────────────────────────────────────────────────────────────────
# GAN losses
# ─────────────────────────────────────────────────────────────────────────────


class TestGANLossCorrectness:
    """Mathematical correctness of GAN loss functions."""

    def test_disc_hinge_loss_perfect_discrimination(self):
        from df_mlx.loss import discriminator_loss

        real = [mx.array([2.0, 3.0])]
        fake = [mx.array([-2.0, -3.0])]
        total, real_l, fake_l = discriminator_loss(real, fake)
        mx.eval(total, real_l, fake_l)
        assert float(total) < 1e-6

    def test_generator_loss_sign(self):
        from df_mlx.loss import generator_loss

        # Disc outputs negative for fake → generator loss should be positive
        fake = [mx.array([-1.0, -2.0])]
        result = float(generator_loss(fake))
        assert result > 0, f"Gen loss should be positive, got {result}"

    def test_feature_matching_identical_zero(self):
        from df_mlx.loss import FeatureMatchingLoss

        fm = FeatureMatchingLoss(factor=2.0)
        feats = [[mx.random.normal((4, 32))]]
        result = float(fm(feats, feats))
        assert result < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Dead constant removal
# ─────────────────────────────────────────────────────────────────────────────


class TestDeadConstantRemoved:
    """Verify dead constants were cleaned up."""

    def test_no_pipeline_speech_band_weight(self):
        """_PIPELINE_SPEECH_BAND_WEIGHT was unused and should be removed."""
        import df_mlx.train_dynamic as td

        assert not hasattr(
            td, "_PIPELINE_SPEECH_BAND_WEIGHT"
        ), "_PIPELINE_SPEECH_BAND_WEIGHT should be removed (unused)"

    def test_no_pipeline_mask_saturation_penalty(self):
        """_PIPELINE_MASK_SATURATION_PENALTY was removed (zero-gradient)."""
        import df_mlx.train_dynamic as td

        assert not hasattr(
            td, "_PIPELINE_MASK_SATURATION_PENALTY"
        ), "_PIPELINE_MASK_SATURATION_PENALTY should be removed"
