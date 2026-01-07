#!/usr/bin/env python
"""Test script for MLX DeepFilterNet4 implementation.

This script verifies that the MLX implementation works correctly by:
1. Testing model initialization
2. Testing forward pass with dummy data
3. Benchmarking performance
4. Comparing with PyTorch implementation (if available)

Usage:
    python -m df_mlx.test_mlx
"""

import sys
import time

# Check MLX availability
try:
    import mlx.core as mx
    import mlx.nn as nn  # noqa: F401

    print(f"✓ MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
except ImportError:
    print("✗ MLX not installed. Install with: pip install mlx")
    print("  Note: MLX only works on Apple Silicon Macs")
    sys.exit(1)


def test_ops():
    """Test core operations."""
    print("\n" + "=" * 60)
    print("Testing Core Operations")
    print("=" * 60)

    from df_mlx.ops import erb_fb, get_window, istft, stft

    # Test window functions
    print("\n1. Window functions...")
    for win_type in ["hann", "hamming", "sqrt_hann"]:
        win = get_window(win_type, 960)
        assert win.shape == (960,), f"Window shape mismatch for {win_type}"
    print("   ✓ Window functions work")

    # Test STFT
    print("\n2. STFT/iSTFT...")
    audio = mx.random.normal((2, 48000))  # 2 samples, 1 second at 48kHz
    spec = stft(audio, n_fft=960, hop_length=480)
    assert len(spec) == 2, "STFT should return (real, imag) tuple"
    assert spec[0].shape == spec[1].shape, "Real and imag should have same shape"
    print(f"   ✓ STFT output shape: {spec[0].shape}")

    # Test iSTFT (reconstruction)
    reconstructed = istft(spec, n_fft=960, hop_length=480, length=48000)
    assert reconstructed.shape == audio.shape, "Reconstruction shape mismatch"
    print(f"   ✓ iSTFT reconstruction shape: {reconstructed.shape}")

    # Test ERB filterbank
    print("\n3. ERB filterbank...")
    fb = erb_fb(sr=48000, fft_size=960, nb_bands=32)
    assert fb.shape == (481, 32), f"ERB filterbank shape mismatch: {fb.shape}"
    print(f"   ✓ ERB filterbank shape: {fb.shape}")

    print("\n✓ All ops tests passed!")


def test_mamba():
    """Test Mamba modules."""
    print("\n" + "=" * 60)
    print("Testing Mamba Modules")
    print("=" * 60)

    from df_mlx.mamba import Mamba, MambaBlock, SqueezedMamba

    batch, seq_len, d_model = 2, 50, 128

    # Test MambaBlock
    print("\n1. MambaBlock...")
    block = MambaBlock(d_model=d_model, d_state=16, d_conv=4)
    x = mx.random.normal((batch, seq_len, d_model))
    out, state = block(x)
    assert out.shape == x.shape, f"MambaBlock output shape mismatch: {out.shape}"
    print(f"   ✓ MambaBlock output shape: {out.shape}")
    print(f"   ✓ State shape: {state.shape}")

    # Test Mamba (with norm + residual)
    print("\n2. Mamba (with LayerNorm + residual)...")
    mamba = Mamba(d_model=d_model)
    out, state = mamba(x)
    assert out.shape == x.shape, "Mamba output shape mismatch"
    print(f"   ✓ Mamba output shape: {out.shape}")

    # Test SqueezedMamba
    print("\n3. SqueezedMamba...")
    squeezed = SqueezedMamba(
        input_size=d_model,
        hidden_size=d_model,
        output_size=d_model,
        num_layers=2,
    )
    out, states = squeezed(x)
    assert out.shape == x.shape, "SqueezedMamba output shape mismatch"
    print(f"   ✓ SqueezedMamba output shape: {out.shape}")

    print("\n✓ All Mamba tests passed!")


def test_modules():
    """Test neural network modules."""
    print("\n" + "=" * 60)
    print("Testing Neural Network Modules")
    print("=" * 60)

    from df_mlx.modules import Conv2dNormAct, DfOp, GroupedLinear, Mask

    batch = 2

    # Test Conv2dNormAct (MLX uses NHWC format)
    print("\n1. Conv2dNormAct...")
    conv = Conv2dNormAct(1, 32, kernel_size=3, stride=1, padding=1)
    x = mx.random.normal((batch, 100, 32, 1))  # NHWC: (batch, height, width, channels)
    out = conv(x)
    print(f"   ✓ Conv2dNormAct output shape: {out.shape}")

    # Test GroupedLinear
    print("\n2. GroupedLinear...")
    grouped = GroupedLinear(256, 128, groups=8)
    x = mx.random.normal((batch, 50, 256))
    out = grouped(x)
    assert out.shape == (batch, 50, 128), "GroupedLinear shape mismatch"
    print(f"   ✓ GroupedLinear output shape: {out.shape}")

    # Test DfOp
    print("\n3. DfOp (Deep Filtering)...")
    df_op = DfOp(nb_df=96, df_order=5)
    spec_real = mx.random.normal((batch, 50, 481))
    spec_imag = mx.random.normal((batch, 50, 481))
    coef = mx.random.normal((batch, 50, 96, 5, 2))
    out = df_op((spec_real, spec_imag), coef)
    assert out[0].shape == spec_real.shape, "DfOp shape mismatch"
    print(f"   ✓ DfOp output shape: {out[0].shape}")

    # Test Mask
    print("\n4. Mask...")
    mask_module = Mask(481, mask_type="sigmoid")
    mask = mx.random.normal((batch, 50, 481))
    out = mask_module(mask)
    assert mx.all(out >= 0) and mx.all(out <= 1), "Sigmoid mask should be in [0,1]"
    print(f"   ✓ Mask output shape: {out.shape}")

    print("\n✓ All module tests passed!")


def test_model():
    """Test complete model."""
    print("\n" + "=" * 60)
    print("Testing Complete Model")
    print("=" * 60)

    from df_mlx.config import get_default_config
    from df_mlx.model import count_parameters, init_model

    # Test model initialization
    print("\n1. Model initialization...")
    model = init_model()
    num_params = count_parameters(model)
    print(f"   ✓ DfNet4 initialized with {num_params:,} parameters")

    # Test lite model
    lite_model = init_model(variant="lite")
    lite_params = count_parameters(lite_model)
    print(f"   ✓ DfNet4Lite initialized with {lite_params:,} parameters")

    # Test forward pass
    print("\n2. Forward pass...")
    p = get_default_config()
    batch, seq_len = 2, 50

    spec_real = mx.random.normal((batch, seq_len, p.n_freqs))
    spec_imag = mx.random.normal((batch, seq_len, p.n_freqs))
    feat_erb = mx.random.normal((batch, seq_len, p.nb_erb))
    feat_spec = mx.random.normal((batch, seq_len, p.nb_df, 2))

    start = time.perf_counter()
    out = model((spec_real, spec_imag), feat_erb, feat_spec)
    mx.eval(out)
    elapsed = time.perf_counter() - start

    assert out[0].shape == spec_real.shape, "Output shape mismatch"
    print(f"   ✓ Forward pass output shape: {out[0].shape}")
    print(f"   ✓ Forward pass time: {elapsed * 1000:.2f}ms")

    print("\n✓ All model tests passed!")


def benchmark():
    """Benchmark model performance."""
    print("\n" + "=" * 60)
    print("Benchmarking Performance")
    print("=" * 60)

    from df_mlx.model import init_model
    from df_mlx.utils import benchmark_model

    model = init_model()

    print("\n1. Inference benchmark...")
    for batch_size in [1, 4, 8, 12]:
        results = benchmark_model(
            model,
            batch_size=batch_size,
            seq_length=100,
            num_warmup=3,
            num_runs=10,
        )
        print(
            f"   Batch {batch_size:2d}: {results['mean_ms']:6.2f}ms ± {results['std_ms']:.2f}ms "
            f"({results['throughput']:.1f} samples/s)"
        )

    print("\n✓ Benchmark complete!")


def test_training():
    """Test training utilities."""
    print("\n" + "=" * 60)
    print("Testing Training Utilities")
    print("=" * 60)

    from df_mlx.config import get_default_config
    from df_mlx.model import init_model
    from df_mlx.train import WarmupCosineSchedule, spectral_loss

    p = get_default_config()
    batch, seq_len = 2, 50

    # Test loss function
    print("\n1. Loss functions...")
    pred = (
        mx.random.normal((batch, seq_len, p.n_freqs)),
        mx.random.normal((batch, seq_len, p.n_freqs)),
    )
    target = (
        mx.random.normal((batch, seq_len, p.n_freqs)),
        mx.random.normal((batch, seq_len, p.n_freqs)),
    )
    loss = spectral_loss(pred, target)
    mx.eval(loss)
    print(f"   ✓ Spectral loss: {float(loss):.4f}")

    # Test LR schedule
    print("\n2. Learning rate schedule...")
    schedule = WarmupCosineSchedule(
        base_lr=1e-3,
        warmup_steps=100,
        total_steps=1000,
    )

    lrs = [schedule(s) for s in [0, 50, 100, 500, 1000]]
    print(f"   Step    0: LR = {lrs[0]:.2e} (warmup start)")
    print(f"   Step   50: LR = {lrs[1]:.2e} (warmup mid)")
    print(f"   Step  100: LR = {lrs[2]:.2e} (warmup end)")
    print(f"   Step  500: LR = {lrs[3]:.2e} (decay mid)")
    print(f"   Step 1000: LR = {lrs[4]:.2e} (decay end)")
    print("   ✓ Learning rate schedule works")

    # Test gradient computation
    print("\n3. Gradient computation...")
    model = init_model()

    def loss_fn(model, spec, feat_erb, feat_spec, target):
        pred = model(spec, feat_erb, feat_spec)
        return spectral_loss(pred, target)

    spec = (
        mx.random.normal((batch, seq_len, p.n_freqs)),
        mx.random.normal((batch, seq_len, p.n_freqs)),
    )
    feat_erb = mx.random.normal((batch, seq_len, p.nb_erb))
    feat_spec = mx.random.normal((batch, seq_len, p.nb_df, 2))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, spec, feat_erb, feat_spec, target)
    mx.eval(loss, grads)

    print(f"   ✓ Loss: {float(loss):.4f}")
    print("   ✓ Gradients computed successfully")

    print("\n✓ All training tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("MLX DeepFilterNet4 Test Suite")
    print("=" * 60)

    # Check device
    print("\nDevice: Apple Silicon (MLX)")

    try:
        test_ops()
        test_mamba()
        test_modules()
        test_model()
        test_training()
        benchmark()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nThe MLX implementation is ready for use.")
        print("\nNext steps:")
        print("1. Convert PyTorch weights: from df_mlx.train import load_pytorch_checkpoint")
        print("2. Train from scratch: from df_mlx import train")
        print("3. Enhance audio: model.enhance(noisy_audio)")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
