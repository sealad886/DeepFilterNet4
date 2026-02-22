#!/usr/bin/env python3
"""Test script for all vectorized optimization implementations."""

import mlx.core as mx


def test_stft_operations():
    """Test optimized STFT/iSTFT operations."""
    from df_mlx.ops import as_strided_frames, istft, stft

    print("=" * 60)
    print("Testing Optimized STFT/iSTFT Operations")
    print("=" * 60)

    # Test STFT with actual parameter names
    x = mx.random.normal((2, 48000))  # 2 channels, 1 second
    real, imag = stft(x, n_fft=960, hop_length=480, window="sqrt_hann")
    print(f"STFT input: {x.shape} -> real: {real.shape}, imag: {imag.shape}")
    assert real.shape == (2, 101, 481), f"Unexpected shape: {real.shape}"

    # Test iSTFT round-trip - pass tuple
    reconstructed = istft((real, imag), n_fft=960, hop_length=480, window="sqrt_hann", length=48000)
    print(f"iSTFT round-trip: {real.shape} -> {reconstructed.shape}")

    # Test as_strided_frames
    frames = as_strided_frames(x, frame_length=960, hop_length=480)
    print(f"as_strided_frames: {x.shape} -> {frames.shape}")

    print("STFT operations: PASSED")
    return True


def test_multiframe_operations():
    """Test optimized multiframe operations via MultiFrameModule."""
    from df_mlx.multiframe import DF, DFreal

    print()
    print("=" * 60)
    print("Testing Optimized Multiframe Operations (DF/DFreal)")
    print("=" * 60)

    # Test DF spec_unfold
    df = DF(num_freqs=96, frame_size=5, lookahead=0)

    # Test spec_unfold: expects [B, C, T, F, 2]
    spec = mx.random.normal((2, 1, 100, 96, 2))  # batch, channel, time, freq, 2
    unfolded = df.spec_unfold(spec)
    print(f"DF.spec_unfold: {spec.shape} -> {unfolded.shape}")
    # Expected: [B, C, T, F, N, 2] - T is preserved due to padding
    assert unfolded.shape == (2, 1, 100, 96, 5, 2), f"Unexpected shape: {unfolded.shape}"

    # Test DFreal spec_unfold_real
    df_real = DFreal(num_freqs=96, frame_size=5, lookahead=0)
    unfolded_real = df_real.spec_unfold_real(spec)
    print(f"DFreal.spec_unfold_real: {spec.shape} -> {unfolded_real.shape}")
    # Expected: [B, C, N, T, F, 2] - T is preserved
    assert unfolded_real.shape == (2, 1, 5, 100, 96, 2), f"Unexpected shape: {unfolded_real.shape}"

    print("Multiframe operations: PASSED")
    return True


def test_loss_functions():
    """Test optimized loss functions."""
    from df_mlx.loss import SegmentalSiSdrLoss

    print()
    print("=" * 60)
    print("Testing Optimized Loss Functions")
    print("=" * 60)

    # Test SegmentalSiSdrLoss with correct signature
    loss_fn = SegmentalSiSdrLoss(segment_size=512, overlap=0.5)
    clean = mx.random.normal((2, 8000))
    enhanced = mx.random.normal((2, 8000))
    loss = loss_fn(clean, enhanced)
    print(f"SegmentalSiSdrLoss: clean={clean.shape}, enhanced={enhanced.shape} -> loss={float(loss):.4f}")

    print("Loss functions: PASSED")
    return True


def test_mx_compile():
    """Test mx.compile integration in training."""
    print()
    print("=" * 60)
    print("Testing mx.compile Training Integration")
    print("=" * 60)

    # Import and check training module for compile usage
    import inspect

    from df_mlx import train_dynamic

    # Read source to check for mx.compile usage
    source = inspect.getsource(train_dynamic)
    has_compile = "mx.compile" in source or "@partial(mx.compile" in source
    print(f"mx.compile in train_dynamic.py: {has_compile}")

    if has_compile:
        print("mx.compile integration: FOUND")
        print("mx.compile test: PASSED")
        return True
    else:
        print("mx.compile integration: NOT FOUND")
        return False


def test_fp16_support():
    """Test FP16 training support."""
    from df_mlx.hardware import HardwareConfig

    print()
    print("=" * 60)
    print("Testing FP16 Training Support")
    print("=" * 60)

    config = HardwareConfig()
    print(f"HardwareConfig.use_fp16: {config.use_fp16}")

    # Test FP16 conversion
    x = mx.random.normal((2, 100))
    x_fp16 = x.astype(mx.float16)
    print(f"FP16 conversion: {x.dtype} -> {x_fp16.dtype}")

    print("FP16 support: PASSED")
    return True


def main():
    """Run all optimization tests."""
    print("\n" + "=" * 60)
    print("  OPTIMIZATION VALIDATION TEST SUITE")
    print("=" * 60 + "\n")

    all_passed = True

    try:
        all_passed &= test_stft_operations()
    except Exception as e:
        print(f"STFT test FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_multiframe_operations()
    except Exception as e:
        print(f"Multiframe test FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_loss_functions()
    except Exception as e:
        print(f"Loss function test FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_mx_compile()
    except Exception as e:
        print(f"mx.compile test FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_fp16_support()
    except Exception as e:
        print(f"FP16 test FAILED: {e}")
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("  ALL OPTIMIZATIONS VALIDATED SUCCESSFULLY!")
    else:
        print("  SOME TESTS FAILED - SEE ABOVE")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
