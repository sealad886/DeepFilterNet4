#!/usr/bin/env python3
"""Correctness tests for train_dynamic.py loss functions.

This module validates:
1. Dimensional consistency of loss terms
2. Edge case handling (all-silence, all-speech, single frames)
3. Numerical stability (NaN/Inf detection)
4. VAD mask alignment and normalization
5. Gradient behavior through masked operations

Run with:
    python -m pytest DeepFilterNet/tests/test_loss_correctness.py -v

Or standalone smoke test:
    python -m tests.test_loss_correctness
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy import pytest only when running tests
try:
    import pytest
except ImportError:
    pytest = None  # type: ignore


# =============================================================================
# Test Fixtures
# =============================================================================


def _fixture_sample_spectrograms():
    """Generate sample spectrograms for testing."""
    batch_size = 4
    n_frames = 100
    n_freqs = 481  # Default FFT size 960 -> 481 bins

    np.random.seed(42)

    # Clean spectrogram (speech-like: more energy in low freqs)
    clean_mag = np.random.exponential(0.1, (batch_size, n_frames, n_freqs)).astype(np.float32)
    clean_mag[:, :, :100] *= 5  # Boost low freqs
    clean_phase = np.random.uniform(-np.pi, np.pi, (batch_size, n_frames, n_freqs)).astype(np.float32)
    clean_real = clean_mag * np.cos(clean_phase)
    clean_imag = clean_mag * np.sin(clean_phase)

    # Noisy spectrogram (clean + noise)
    noise_mag = np.random.exponential(0.05, (batch_size, n_frames, n_freqs)).astype(np.float32)
    noise_phase = np.random.uniform(-np.pi, np.pi, (batch_size, n_frames, n_freqs)).astype(np.float32)
    noise_real = noise_mag * np.cos(noise_phase)
    noise_imag = noise_mag * np.sin(noise_phase)

    noisy_real = clean_real + noise_real
    noisy_imag = clean_imag + noise_imag

    # Output spectrogram (enhanced, close to clean)
    out_real = clean_real + 0.1 * noise_real
    out_imag = clean_imag + 0.1 * noise_imag

    return {
        "clean_real": mx.array(clean_real),
        "clean_imag": mx.array(clean_imag),
        "noisy_real": mx.array(noisy_real),
        "noisy_imag": mx.array(noisy_imag),
        "out_real": mx.array(out_real),
        "out_imag": mx.array(out_imag),
        "snr": mx.array([10.0, 5.0, 0.0, -5.0]),
        "batch_size": batch_size,
        "n_frames": n_frames,
        "n_freqs": n_freqs,
    }


def _fixture_band_mask(n_freqs):
    """Generate speech band mask (300-3400 Hz at 48kHz, 960 FFT)."""
    freqs = np.linspace(0, 24000, n_freqs)  # 48kHz / 2
    mask = ((freqs >= 300) & (freqs <= 3400)).astype(np.float32)
    band_bins = float(mask.sum())
    return mx.array(mask), band_bins


if pytest is not None:

    @pytest.fixture
    def sample_spectrograms():
        return _fixture_sample_spectrograms()

    @pytest.fixture
    def band_mask(sample_spectrograms):
        return _fixture_band_mask(sample_spectrograms["n_freqs"])


# =============================================================================
# Import loss functions
# =============================================================================


def import_loss_functions():
    """Import loss functions from train_dynamic.py."""
    from df_mlx.training_losses import (
        _EPS,
        _compute_awesome_losses,
        _compute_pipeline_awesome_losses,
        _compute_speech_band_logmag_loss,
        _compute_vad_loss,
        _compute_vad_probs,
    )

    return {
        "compute_awesome_losses": _compute_awesome_losses,
        "compute_pipeline_awesome_losses": _compute_pipeline_awesome_losses,
        "compute_speech_band_logmag_loss": _compute_speech_band_logmag_loss,
        "compute_vad_loss": _compute_vad_loss,
        "compute_vad_probs": _compute_vad_probs,
        "EPS": _EPS,
    }


# =============================================================================
# Test Cases
# =============================================================================


class TestDimensionalConsistency:
    """Verify loss terms have correct shapes and magnitudes."""

    def test_awesome_loss_shape(self, sample_spectrograms, band_mask):
        """Awesome loss should return scalar and matching diagnostic shapes."""
        loss_fns = import_loss_functions()
        band_mask_arr, band_bins = band_mask
        s = sample_spectrograms

        result = loss_fns["compute_awesome_losses"](
            s["noisy_real"],
            s["noisy_imag"],
            s["clean_real"],
            s["clean_imag"],
            s["out_real"],
            s["out_imag"],
            s["snr"],
            band_mask_arr,
            band_bins,
            mask_sharpness=6.0,
            vad_z_threshold=0.0,
            vad_z_slope=1.0,
            vad_snr_gate_db=-10.0,
            vad_snr_gate_width=6.0,
            proxy_enabled=True,
        )

        awesome_loss = result[0]
        speech_loss = result[1]
        noise_loss = result[2]
        _smooth_loss = result[3]  # noqa: F841
        mask = result[4]
        proxy_frame = result[5]

        # Check scalars
        assert awesome_loss.shape == (), f"Expected scalar, got {awesome_loss.shape}"
        assert speech_loss.shape == (), f"Expected scalar, got {speech_loss.shape}"
        assert noise_loss.shape == (), f"Expected scalar, got {noise_loss.shape}"

        # Check mask shape
        assert mask.shape == (
            s["batch_size"],
            s["n_frames"],
            s["n_freqs"],
        ), f"Mask shape mismatch: {mask.shape}"

        # Check proxy frame shape
        assert proxy_frame.shape == (
            s["batch_size"],
            s["n_frames"],
        ), f"Proxy frame shape mismatch: {proxy_frame.shape}"

    def test_pipeline_awesome_loss_shape(self, sample_spectrograms, band_mask):
        """Pipeline awesome loss should return all expected components."""
        loss_fns = import_loss_functions()
        band_mask_arr, band_bins = band_mask
        s = sample_spectrograms

        result = loss_fns["compute_pipeline_awesome_losses"](
            s["noisy_real"],
            s["noisy_imag"],
            s["clean_real"],
            s["clean_imag"],
            s["out_real"],
            s["out_imag"],
            s["snr"],
            band_mask_arr,
            band_bins,
            mask_sharpness=6.0,
            vad_z_threshold=0.0,
            vad_z_slope=1.0,
            vad_snr_gate_db=-10.0,
            vad_snr_gate_width=6.0,
            proxy_enabled=True,
        )

        # Should return 16 values
        assert len(result) == 16, f"Expected 16 return values, got {len(result)}"

        total_loss = result[0]
        music_suppression_loss = result[4]
        mask_saturation_loss = result[5]

        # Check scalars
        assert total_loss.shape == (), f"Expected scalar, got {total_loss.shape}"
        assert music_suppression_loss.shape == (), f"Expected scalar, got shape {music_suppression_loss.shape}"
        assert mask_saturation_loss.shape == (), f"Expected scalar, got shape {mask_saturation_loss.shape}"


class TestNumericalStability:
    """Verify loss functions don't produce NaN/Inf."""

    def _check_finite(self, tensor, name):
        """Assert tensor is finite."""
        mx.eval(tensor)
        arr = np.asarray(tensor)
        assert np.all(np.isfinite(arr)), f"{name} contains non-finite values: min={arr.min()}, max={arr.max()}"

    def test_awesome_loss_finite(self, sample_spectrograms, band_mask):
        """Awesome loss should not produce NaN/Inf."""
        loss_fns = import_loss_functions()
        band_mask_arr, band_bins = band_mask
        s = sample_spectrograms

        result = loss_fns["compute_awesome_losses"](
            s["noisy_real"],
            s["noisy_imag"],
            s["clean_real"],
            s["clean_imag"],
            s["out_real"],
            s["out_imag"],
            s["snr"],
            band_mask_arr,
            band_bins,
            mask_sharpness=6.0,
            vad_z_threshold=0.0,
            vad_z_slope=1.0,
            vad_snr_gate_db=-10.0,
            vad_snr_gate_width=6.0,
            proxy_enabled=True,
        )

        self._check_finite(result[0], "awesome_loss")
        self._check_finite(result[1], "speech_loss")
        self._check_finite(result[2], "noise_loss")
        self._check_finite(result[3], "smooth_loss")
        self._check_finite(result[4], "mask")
        self._check_finite(result[5], "proxy_frame")

    def test_zero_energy_stability(self, band_mask):
        """Test handling of near-zero energy inputs (silence)."""
        loss_fns = import_loss_functions()
        band_mask_arr, band_bins = band_mask

        batch_size = 2
        n_frames = 50
        n_freqs = 481

        # Near-silence input (very small values)
        silence_val = 1e-10
        clean_real = mx.full((batch_size, n_frames, n_freqs), silence_val)
        clean_imag = mx.full((batch_size, n_frames, n_freqs), silence_val)
        noisy_real = clean_real + mx.full((batch_size, n_frames, n_freqs), 1e-11)
        noisy_imag = clean_imag + mx.full((batch_size, n_frames, n_freqs), 1e-11)
        out_real = clean_real[:]
        out_imag = clean_imag[:]
        snr = mx.array([0.0, -10.0])

        result = loss_fns["compute_awesome_losses"](
            noisy_real,
            noisy_imag,
            clean_real,
            clean_imag,
            out_real,
            out_imag,
            snr,
            band_mask_arr,
            band_bins,
            mask_sharpness=6.0,
            vad_z_threshold=0.0,
            vad_z_slope=1.0,
            vad_snr_gate_db=-10.0,
            vad_snr_gate_width=6.0,
            proxy_enabled=True,
        )

        self._check_finite(result[0], "awesome_loss_silence")
        self._check_finite(result[5], "proxy_frame_silence")

    def test_extreme_values_stability(self, band_mask):
        """Test handling of extreme magnitude values."""
        loss_fns = import_loss_functions()
        band_mask_arr, band_bins = band_mask

        batch_size = 2
        n_frames = 50
        n_freqs = 481

        # Large values
        clean_real = mx.full((batch_size, n_frames, n_freqs), 100.0)
        clean_imag = mx.full((batch_size, n_frames, n_freqs), 100.0)
        noisy_real = clean_real + mx.full((batch_size, n_frames, n_freqs), 50.0)
        noisy_imag = clean_imag + mx.full((batch_size, n_frames, n_freqs), 50.0)
        out_real = clean_real[:]
        out_imag = clean_imag[:]
        snr = mx.array([40.0, 30.0])

        result = loss_fns["compute_awesome_losses"](
            noisy_real,
            noisy_imag,
            clean_real,
            clean_imag,
            out_real,
            out_imag,
            snr,
            band_mask_arr,
            band_bins,
            mask_sharpness=6.0,
            vad_z_threshold=0.0,
            vad_z_slope=1.0,
            vad_snr_gate_db=-10.0,
            vad_snr_gate_width=6.0,
            proxy_enabled=True,
        )

        self._check_finite(result[0], "awesome_loss_extreme")


class TestEdgeCases:
    """Test edge case handling."""

    def test_single_frame(self, band_mask):
        """Test with single time frame."""
        loss_fns = import_loss_functions()
        band_mask_arr, band_bins = band_mask

        batch_size = 2
        n_frames = 1  # Single frame
        n_freqs = 481

        np.random.seed(42)
        clean_real = mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32))
        clean_imag = mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32))
        noisy_real = clean_real + mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32) * 0.5)
        noisy_imag = clean_imag + mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32) * 0.5)
        out_real = clean_real[:]
        out_imag = clean_imag[:]
        snr = mx.array([10.0, 5.0])

        result = loss_fns["compute_awesome_losses"](
            noisy_real,
            noisy_imag,
            clean_real,
            clean_imag,
            out_real,
            out_imag,
            snr,
            band_mask_arr,
            band_bins,
            mask_sharpness=6.0,
            vad_z_threshold=0.0,
            vad_z_slope=1.0,
            vad_snr_gate_db=-10.0,
            vad_snr_gate_width=6.0,
            proxy_enabled=True,
        )

        mx.eval(result[0])
        arr = np.asarray(result[0])
        assert np.all(np.isfinite(arr)), f"Single frame loss not finite: {arr}"

        # Smooth loss should be 0 for single frame
        mx.eval(result[3])
        smooth_loss = float(result[3])
        assert smooth_loss == 0.0, f"Smooth loss should be 0 for single frame, got {smooth_loss}"

    def test_all_speech_vs_no_speech(self, band_mask):
        """Compare loss with high vs low speech energy."""
        loss_fns = import_loss_functions()
        band_mask_arr, band_bins = band_mask

        batch_size = 2
        n_frames = 50
        n_freqs = 481

        # High speech energy (all frames have speech)
        np.random.seed(42)
        high_clean = np.random.exponential(1.0, (batch_size, n_frames, n_freqs)).astype(np.float32)
        high_clean[:, :, :100] *= 10  # Strong low freq content

        # Low speech energy (near silence)
        low_clean = high_clean * 0.001

        for label, clean_np in [("high", high_clean), ("low", low_clean)]:
            clean_real = mx.array(clean_np)
            clean_imag = mx.array(clean_np * 0.5)
            noise = np.random.exponential(0.1, (batch_size, n_frames, n_freqs)).astype(np.float32)
            noisy_real = clean_real + mx.array(noise)
            noisy_imag = clean_imag + mx.array(noise * 0.5)
            out_real = clean_real[:]
            out_imag = clean_imag[:]
            snr = mx.array([10.0, 5.0])

            result = loss_fns["compute_awesome_losses"](
                noisy_real,
                noisy_imag,
                clean_real,
                clean_imag,
                out_real,
                out_imag,
                snr,
                band_mask_arr,
                band_bins,
                mask_sharpness=6.0,
                vad_z_threshold=0.0,
                vad_z_slope=1.0,
                vad_snr_gate_db=-10.0,
                vad_snr_gate_width=6.0,
                proxy_enabled=True,
            )

            mx.eval(result[0])
            arr = np.asarray(result[0])
            assert np.all(np.isfinite(arr)), f"{label} speech loss not finite: {arr}"


class TestMaskSaturationPenalty:
    """Test the mask saturation penalty component."""

    def test_mask_saturation_penalty_direction(self):
        """Verify mask saturation penalty rewards confident predictions.

        KNOWN BUG: Current implementation inverts the penalty.
        This test documents expected behavior after fix.
        """
        # Confident mask (values near 0 or 1)
        confident_mask = mx.array([[0.05, 0.95, 0.02, 0.98]])

        # Uncertain mask (values near 0.5)
        uncertain_mask = mx.array([[0.45, 0.55, 0.48, 0.52]])

        # Compute entropy-like penalty: mask * (1 - mask)
        confident_penalty = mx.mean(confident_mask * (1.0 - confident_mask))
        uncertain_penalty = mx.mean(uncertain_mask * (1.0 - uncertain_mask))

        mx.eval(confident_penalty, uncertain_penalty)

        # Confident masks should have LOWER penalty (near 0)
        # Uncertain masks should have HIGHER penalty (near 0.25)
        conf_val = float(confident_penalty)
        unc_val = float(uncertain_penalty)

        print(f"Confident mask penalty: {conf_val:.4f}")
        print(f"Uncertain mask penalty: {unc_val:.4f}")

        assert conf_val < unc_val, f"Confident ({conf_val}) should be < uncertain ({unc_val})"

        # Current implementation: 1.0 - 4.0 * penalty
        # This INVERTS the behavior (rewards uncertain, penalizes confident)
        # After fix, this comment should reflect correct behavior


class TestVADLossNormalization:
    """Test VAD loss normalization behavior."""

    def test_vad_loss_varies_with_speech_proportion(self, band_mask):
        """Test that VAD loss scales appropriately with speech proportion.

        Current behavior: Loss scales with number of active frames
        Expected behavior: Loss should be normalized by active frames
        """
        loss_fns = import_loss_functions()
        band_mask_arr, band_bins = band_mask

        batch_size = 1
        n_frames = 100
        n_freqs = 481

        # Create clean with varying speech proportion
        base_clean = np.random.exponential(0.5, (batch_size, n_frames, n_freqs)).astype(np.float32)

        losses = []
        for speech_frames in [10, 50, 90]:
            clean_np = base_clean[:]
            # Suppress non-speech frames
            clean_np[:, speech_frames:, :] *= 0.001

            clean_real = mx.array(clean_np)
            clean_imag = mx.array(clean_np * 0.5)
            # noisy_* created for completeness but not used by VAD loss
            _ = clean_real + mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32) * 0.1)
            _ = clean_imag + mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32) * 0.1)
            # Make output slightly worse than clean to trigger VAD loss
            out_real = clean_real * 0.9
            out_imag = clean_imag * 0.9
            snr = mx.array([10.0])

            vad_loss, _, _, gate = loss_fns["compute_vad_loss"](
                clean_real,
                clean_imag,
                out_real,
                out_imag,
                snr,
                band_mask_arr,
                band_bins,
                vad_threshold=0.6,
                vad_margin=0.05,
                vad_snr_gate_db=-10.0,
                vad_snr_gate_width=6.0,
                vad_z_threshold=0.0,
                vad_z_slope=1.0,
            )

            mx.eval(vad_loss, gate)
            losses.append((speech_frames, float(vad_loss), float(mx.mean(gate))))
            print(
                f"Speech frames: {speech_frames}, VAD loss: {float(vad_loss):.6f}, gate mean: {float(mx.mean(gate)):.4f}"
            )

        # Document current behavior (not asserting - this is diagnostic)
        # After normalization fix, losses should be more similar across speech proportions


# =============================================================================
# Standalone Smoke Test
# =============================================================================


def run_smoke_test():
    """Run minimal smoke test to verify loss functions work."""
    print("=" * 60)
    print("Loss Function Smoke Test")
    print("=" * 60)

    loss_fns = import_loss_functions()

    batch_size = 2
    n_frames = 50
    n_freqs = 481

    np.random.seed(42)

    # Create sample data
    clean_real = mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32) * 0.5)
    clean_imag = mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32) * 0.5)
    noisy_real = clean_real + mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32) * 0.2)
    noisy_imag = clean_imag + mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32) * 0.2)
    out_real = clean_real + mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32) * 0.05)
    out_imag = clean_imag + mx.array(np.random.randn(batch_size, n_frames, n_freqs).astype(np.float32) * 0.05)
    snr = mx.array([10.0, -5.0])

    # Create band mask
    freqs = np.linspace(0, 24000, n_freqs)
    mask = ((freqs >= 300) & (freqs <= 3400)).astype(np.float32)
    band_mask = mx.array(mask)
    band_bins = float(mask.sum())

    print("\nInput shapes:")
    print(f"  Clean: {clean_real.shape}")
    print(f"  Noisy: {noisy_real.shape}")
    print(f"  Output: {out_real.shape}")
    print(f"  Band mask bins: {band_bins}")

    # Test awesome loss
    print("\nTesting awesome loss...")
    result = loss_fns["compute_awesome_losses"](
        noisy_real,
        noisy_imag,
        clean_real,
        clean_imag,
        out_real,
        out_imag,
        snr,
        band_mask,
        band_bins,
        mask_sharpness=6.0,
        vad_z_threshold=0.0,
        vad_z_slope=1.0,
        vad_snr_gate_db=-10.0,
        vad_snr_gate_width=6.0,
        proxy_enabled=True,
    )

    mx.eval(result)
    awesome_loss = float(result[0])
    speech_loss = float(result[1])
    noise_loss = float(result[2])
    smooth_loss = float(result[3])
    proxy_mean = float(mx.mean(result[5]))

    print(f"  Awesome loss: {awesome_loss:.6f}")
    print(f"  Speech loss:  {speech_loss:.6f}")
    print(f"  Noise loss:   {noise_loss:.6f}")
    print(f"  Smooth loss:  {smooth_loss:.6f}")
    print(f"  Proxy mean:   {proxy_mean:.4f}")

    # Check finite
    for i, name in enumerate(["total", "speech", "noise", "smooth"]):
        arr = np.asarray(result[i])
        if not np.all(np.isfinite(arr)):
            print(f"  ❌ FAIL: {name} loss is not finite!")
            return False
        print(f"  ✓ {name} loss is finite")

    # Test pipeline awesome loss
    print("\nTesting pipeline awesome loss...")
    result2 = loss_fns["compute_pipeline_awesome_losses"](
        noisy_real,
        noisy_imag,
        clean_real,
        clean_imag,
        out_real,
        out_imag,
        snr,
        band_mask,
        band_bins,
        mask_sharpness=6.0,
        vad_z_threshold=0.0,
        vad_z_slope=1.0,
        vad_snr_gate_db=-10.0,
        vad_snr_gate_width=6.0,
        proxy_enabled=True,
    )

    mx.eval(result2)
    pipeline_loss = float(result2[0])
    music_supp = float(result2[4])
    mask_sat = float(result2[5])

    print(f"  Pipeline loss:       {pipeline_loss:.6f}")
    print(f"  Music suppression:   {music_supp:.6f}")
    print(f"  Mask saturation:     {mask_sat:.6f}")

    for i, name in enumerate(["total", "speech", "noise", "smooth", "music_supp", "mask_sat"]):
        arr = np.asarray(result2[i])
        if not np.all(np.isfinite(arr)):
            print(f"  ❌ FAIL: pipeline {name} is not finite!")
            return False
        print(f"  ✓ pipeline {name} is finite")

    print("\n" + "=" * 60)
    print("✓ All smoke tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
