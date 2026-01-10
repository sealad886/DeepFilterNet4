#!/usr/bin/env python3
"""Test and evaluate DFNetMF (Multi-Frame filtering model).

This script tests the DFNetMF implementation and provides guidance on its
suitability for different noise scenarios.

DFNetMF uses multi-frame filtering (Wiener Filter or MVDR beamformer) instead
of the standard deep filtering approach used in DFNet4.

Usage:
    # Basic functionality test
    python test_dfnetmf.py --test-basic

    # Test with audio file
    python test_dfnetmf.py --test-audio /path/to/noisy.wav

    # Compare MF methods (WF vs MVDR)
    python test_dfnetmf.py --compare-methods

    # Benchmark against DFNet4
    python test_dfnetmf.py --benchmark
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import numpy as np


def get_erb_filterbanks(
    sr: int = 48000,
    fft_size: int = 960,
    nb_erb: int = 32,
    nb_df: int = 96,
) -> Tuple[mx.array, mx.array]:
    """Create ERB filterbank matrices.

    Returns:
        erb_fb: ERB filterbank [F, E]
        erb_inv_fb: Inverse ERB filterbank [E, F]
    """
    from df_mlx.ops import erb_fb_and_inverse

    erb_fb, erb_inv = erb_fb_and_inverse(
        sr=sr,
        fft_size=fft_size,
        nb_bands=nb_erb,
    )
    return erb_fb, erb_inv


def test_basic_forward():
    """Test basic forward pass of DFNetMF."""
    print("=" * 60)
    print("TEST: Basic DFNetMF Forward Pass")
    print("=" * 60)

    from df_mlx.deepfilternetmf import DFNetMF, ModelParamsMF

    # Create model parameters
    p = ModelParamsMF(
        sr=48000,
        fft_size=960,
        hop_size=480,
        nb_erb=32,
        nb_df=96,
        df_order=5,
        mfop_method="WF",  # Wiener Filter
    )

    # Get filterbanks
    erb_fb, erb_inv = get_erb_filterbanks(
        sr=p.sr,
        fft_size=p.fft_size,
        nb_erb=p.nb_erb,
        nb_df=p.nb_df,
    )

    # Create model
    print(f"\nCreating DFNetMF with method: {p.mfop_method}")
    model = DFNetMF(erb_fb, erb_inv, run_df=True, train_mask=True, params=p)

    # Count parameters
    def count_params(m):
        from mlx.utils import tree_flatten

        flat_params = tree_flatten(m.parameters())
        return sum(p.size for _, p in flat_params)

    num_params = count_params(model)
    print(f"Parameters: {num_params:,}")

    # Create dummy inputs
    batch_size = 2
    time_frames = 100
    n_freqs = p.fft_size // 2 + 1

    spec = mx.random.normal((batch_size, time_frames, n_freqs, 2))
    feat_erb = mx.random.normal((batch_size, time_frames, p.nb_erb, 1))
    feat_spec = mx.random.normal((batch_size, time_frames, p.nb_df, 2))

    print("\nInput shapes:")
    print(f"  spec:      {spec.shape}")
    print(f"  feat_erb:  {feat_erb.shape}")
    print(f"  feat_spec: {feat_spec.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    start = time.time()
    spec_out, mask, lsnr, df_coefs = model(spec, feat_erb, feat_spec)
    mx.eval(spec_out, mask, lsnr, df_coefs)
    elapsed = time.time() - start

    print("\nOutput shapes:")
    print(f"  spec_out:  {spec_out.shape}")
    print(f"  mask:      {mask.shape}")
    print(f"  lsnr:      {lsnr.shape}")
    print(f"  df_coefs:  {df_coefs.shape}")
    print(f"\nForward pass time: {elapsed * 1000:.2f}ms")

    # Verify outputs
    assert spec_out.shape == spec.shape, f"Output shape mismatch: {spec_out.shape} vs {spec.shape}"
    assert mask.shape == (batch_size, time_frames, p.nb_erb), f"Mask shape unexpected: {mask.shape}"
    assert lsnr.shape == (batch_size, time_frames, 1), f"LSNR shape unexpected: {lsnr.shape}"

    print("\n✅ Basic forward pass test PASSED")
    return True


def test_mvdr_method():
    """Test MVDR beamformer method."""
    print("\n" + "=" * 60)
    print("TEST: MVDR Method")
    print("=" * 60)

    from df_mlx.deepfilternetmf import DFNetMF, ModelParamsMF

    p = ModelParamsMF(mfop_method="MVDR")

    erb_fb, erb_inv = get_erb_filterbanks(
        sr=p.sr,
        fft_size=p.fft_size,
        nb_erb=p.nb_erb,
        nb_df=p.nb_df,
    )

    print("Creating DFNetMF with method: {p.mfop_method}")
    model = DFNetMF(erb_fb, erb_inv, run_df=True, params=p)

    # Quick forward test
    spec = mx.random.normal((1, 50, 481, 2))
    feat_erb = mx.random.normal((1, 50, p.nb_erb, 1))
    feat_spec = mx.random.normal((1, 50, p.nb_df, 2))

    spec_out, mask, lsnr, _ = model(spec, feat_erb, feat_spec)
    mx.eval(spec_out)

    print(f"Output shape: {spec_out.shape}")
    print("\n✅ MVDR method test PASSED")
    return True


def compare_mf_methods():
    """Compare Wiener Filter vs MVDR methods."""
    print("\n" + "=" * 60)
    print("COMPARISON: Wiener Filter vs MVDR")
    print("=" * 60)

    from df_mlx.deepfilternetmf import DFNetMF, ModelParamsMF

    erb_fb, erb_inv = get_erb_filterbanks()

    results = {}

    for method in ["WF", "MVDR"]:
        p = ModelParamsMF(mfop_method=method)
        model = DFNetMF(erb_fb, erb_inv, run_df=True, params=p)

        # Count params
        def count_params(m):
            from mlx.utils import tree_flatten

            flat_params = tree_flatten(m.parameters())
            return sum(p.size for _, p in flat_params)

        num_params = count_params(model)

        # Benchmark
        spec = mx.random.normal((1, 100, 481, 2))
        feat_erb = mx.random.normal((1, 100, 32, 1))
        feat_spec = mx.random.normal((1, 100, 96, 2))

        # Warmup
        for _ in range(3):
            out, _, _, _ = model(spec, feat_erb, feat_spec)
            mx.eval(out)

        # Timed runs
        times = []
        for _ in range(10):
            start = time.time()
            out, _, _, _ = model(spec, feat_erb, feat_spec)
            mx.eval(out)
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000

        results[method] = {
            "params": num_params,
            "time_ms": avg_time,
            "std_ms": std_time,
        }

        print(f"\n{method}:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Forward time: {avg_time:.2f} ± {std_time:.2f} ms")

    print("\n" + "-" * 40)
    print("Method comparison (100 frames @ 48kHz):")
    print("-" * 40)
    print(f"{'Method':<10} {'Params':>12} {'Time (ms)':>15}")
    print("-" * 40)
    for method, data in results.items():
        print(f"{method:<10} {data['params']:>12,} {data['time_ms']:>10.2f} ± {data['std_ms']:.2f}")

    return results


def benchmark_vs_dfnet4():
    """Benchmark DFNetMF against DFNet4."""
    print("\n" + "=" * 60)
    print("BENCHMARK: DFNetMF vs DFNet4")
    print("=" * 60)

    from df_mlx.config import ModelParams4
    from df_mlx.deepfilternetmf import DFNetMF, ModelParamsMF
    from df_mlx.model import DfNet4

    results = {}

    # Test DFNet4
    print("\n[DFNet4]")
    config = ModelParams4()
    model_df4 = DfNet4(config)

    def count_params(m):
        from mlx.utils import tree_flatten

        flat_params = tree_flatten(m.parameters())
        return sum(p.size for _, p in flat_params)

    results["DFNet4"] = {"params": count_params(model_df4)}

    # Create inputs for DFNet4 - note: spec is (real, imag) tuple
    spec_real = mx.random.normal((1, 100, 481))
    spec_imag = mx.random.normal((1, 100, 481))
    spec_tuple = (spec_real, spec_imag)
    feat_erb = mx.random.normal((1, 100, 32))
    feat_spec = mx.random.normal((1, 100, 96, 2))

    # Warmup and benchmark DFNet4
    for _ in range(3):
        out = model_df4(spec_tuple, feat_erb, feat_spec)
        mx.eval(out[0])

    times = []
    for _ in range(10):
        start = time.time()
        out = model_df4(spec_tuple, feat_erb, feat_spec)
        mx.eval(out[0])
        times.append(time.time() - start)

    results["DFNet4"]["time_ms"] = np.mean(times) * 1000
    results["DFNet4"]["std_ms"] = np.std(times) * 1000
    print(f"  Parameters: {results['DFNet4']['params']:,}")
    print(f"  Forward time: {results['DFNet4']['time_ms']:.2f} ± {results['DFNet4']['std_ms']:.2f} ms")

    # Test DFNetMF (WF)
    print("\n[DFNetMF-WF]")
    erb_fb, erb_inv = get_erb_filterbanks()
    p_mf = ModelParamsMF(mfop_method="WF")
    model_mf = DFNetMF(erb_fb, erb_inv, run_df=True, params=p_mf)

    results["DFNetMF-WF"] = {"params": count_params(model_mf)}

    # Input format differs slightly for MF - need combined spec format
    spec_mf = mx.random.normal((1, 100, 481, 2))
    feat_erb_mf = mx.expand_dims(feat_erb, axis=-1)

    for _ in range(3):
        out, _, _, _ = model_mf(spec_mf, feat_erb_mf, feat_spec)
        mx.eval(out)

    times = []
    for _ in range(10):
        start = time.time()
        out, _, _, _ = model_mf(spec_mf, feat_erb_mf, feat_spec)
        mx.eval(out)
        times.append(time.time() - start)

    results["DFNetMF-WF"]["time_ms"] = np.mean(times) * 1000
    results["DFNetMF-WF"]["std_ms"] = np.std(times) * 1000
    print(f"  Parameters: {results['DFNetMF-WF']['params']:,}")
    print(f"  Forward time: {results['DFNetMF-WF']['time_ms']:.2f} ± {results['DFNetMF-WF']['std_ms']:.2f} ms")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Model Comparison")
    print("=" * 60)
    print(f"{'Model':<15} {'Params':>12} {'Time (ms)':>15}")
    print("-" * 45)
    for model_name, data in results.items():
        print(f"{model_name:<15} {data['params']:>12,} {data['time_ms']:>10.2f} ± {data['std_ms']:.2f}")

    return results


def test_with_audio(audio_path: str):
    """Test DFNetMF with actual audio file."""
    print("\n" + "=" * 60)
    print("TEST: Audio Processing with DFNetMF")
    print(f"File: {audio_path}")
    print("=" * 60)

    import soundfile as sf

    from df_mlx.deepfilternetmf import DFNetMF, ModelParamsMF
    from df_mlx.ops import istft, stft

    # Load audio
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Mono

    # Resample if needed
    if sr != 48000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        sr = 48000

    print(f"Audio: {len(audio) / sr:.2f}s @ {sr}Hz")

    # Create model
    p = ModelParamsMF(
        sr=sr,
        fft_size=960,
        hop_size=480,
        nb_erb=32,
        nb_df=96,
        mfop_method="WF",
    )

    erb_fb, erb_inv = get_erb_filterbanks(
        sr=p.sr,
        fft_size=p.fft_size,
        nb_erb=p.nb_erb,
        nb_df=p.nb_df,
    )

    model = DFNetMF(erb_fb, erb_inv, run_df=True, params=p)

    # Convert to MLX
    audio_mx = mx.array(audio.astype(np.float32))
    audio_mx = mx.expand_dims(audio_mx, axis=0)  # Add batch dim

    # STFT
    spec = stft(audio_mx, p.fft_size, p.hop_size)
    print(f"Spectrogram shape: {spec.shape}")

    # Create features (simplified - in practice these come from proper feature extraction)
    # ERB features: magnitude in ERB bands
    spec_mag = mx.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2)
    feat_erb = mx.matmul(spec_mag, mx.transpose(erb_fb))  # [B, T, E]
    feat_erb = mx.expand_dims(feat_erb, axis=-1)  # [B, T, E, 1]
    feat_erb = mx.log(feat_erb + 1e-8)  # Log scale

    # DF features: low-freq complex spec
    feat_spec = spec[:, :, : p.nb_df, :]

    print(f"Feature shapes: erb={feat_erb.shape}, spec={feat_spec.shape}")

    # Forward pass (untrained model - just checking it runs)
    print("\nRunning forward pass...")
    start = time.time()
    spec_out, mask, lsnr, _ = model(spec, feat_erb, feat_spec)
    mx.eval(spec_out, mask, lsnr)
    elapsed = time.time() - start

    print(f"Processing time: {elapsed * 1000:.2f}ms")
    print(f"Real-time factor: {elapsed / (len(audio) / sr):.4f}x")

    # iSTFT to get audio back
    audio_out = istft(spec_out, p.fft_size, p.hop_size)
    mx.eval(audio_out)

    print(f"Output audio shape: {audio_out.shape}")
    print("\n⚠️  Note: Model is untrained - output quality will be poor")
    print("    This test only verifies the processing pipeline works")

    # Save output
    output_path = Path(audio_path).stem + "_dfnetmf_test.wav"
    audio_np = np.array(audio_out[0])
    sf.write(output_path, audio_np, sr)
    print(f"\nSaved output to: {output_path}")

    print("\n✅ Audio processing test PASSED")
    return True


def print_suitability_guide():
    """Print guidance on when to use MF vs DF."""
    print("\n" + "=" * 60)
    print("SUITABILITY GUIDE: Multi-Frame (MF) vs Deep Filtering (DF)")
    print("=" * 60)

    guide = """
┌─────────────────────────────────────────────────────────────────────────┐
│                     WHEN TO USE EACH APPROACH                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MULTI-FRAME FILTERING (DFNetMF)                                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                        │
│  ✓ Stationary noise (HVAC, fan noise, consistent hum)                   │
│  ✓ When statistical assumptions hold well                               │
│  ✓ Scenarios requiring mathematical guarantees                          │
│  ✓ Predictable, slowly-varying noise characteristics                    │
│  ✓ Industrial/machinery noise suppression                               │
│                                                                         │
│  Methods:                                                               │
│  • Wiener Filter (WF): Classic MMSE estimator, good general choice      │
│  • MVDR: Better speech distortion control, good for low SNR             │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DEEP FILTERING (DFNet4)                                                │
│  ━━━━━━━━━━━━━━━━━━━━━━━                                                 │
│  ✓ Non-stationary noise (traffic, babble, music)                        │
│  ✓ Complex acoustic scenarios                                           │
│  ✓ Mixed noise types                                                    │
│  ✓ Generally better for real-world recordings                           │
│  ✓ State-of-the-art performance on benchmarks                           │
│                                                                         │
│  Advantages:                                                            │
│  • Learns complex nonlinear transformations                             │
│  • Adapts to diverse noise patterns                                     │
│  • Better temporal dynamics modeling                                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  RECOMMENDATION FOR YOUR DATASET                                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                        │
│                                                                         │
│  If your noise is primarily:                                            │
│                                                                         │
│  • Fan/HVAC noise, electrical hum     → Consider DFNetMF (WF)           │
│  • Mixed real-world noise             → Use DFNet4 (default)            │
│  • Speech babble/crosstalk            → Use DFNet4                      │
│  • Music interference                 → Use DFNet4                      │
│  • Industrial machinery               → Try both, compare               │
│                                                                         │
│  For most use cases, DFNet4 will outperform DFNetMF.                    │
│  DFNetMF is primarily for specialized scenarios where                   │
│  classical signal processing assumptions are well-matched.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

INTEGRATION STATUS:
━━━━━━━━━━━━━━━━━━━

Current state: DFNetMF is IMPLEMENTED but NOT integrated into training pipeline.

To use DFNetMF, you would need to:
1. Create a training script that uses DFNetMF instead of DfNet4
2. Adapt loss functions for MF output format
3. Train from scratch (no pretrained checkpoints available)

This is experimental code - suitable for research/prototyping, not production.
"""
    print(guide)


def main():
    parser = argparse.ArgumentParser(description="Test DFNetMF model")
    parser.add_argument("--test-basic", action="store_true", help="Run basic forward pass test")
    parser.add_argument("--test-mvdr", action="store_true", help="Test MVDR method")
    parser.add_argument("--compare-methods", action="store_true", help="Compare WF vs MVDR")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark vs DFNet4")
    parser.add_argument("--test-audio", type=str, help="Test with audio file")
    parser.add_argument("--guide", action="store_true", help="Print suitability guide")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    # If no args, show help and run basic test
    if not any(
        [
            args.test_basic,
            args.test_mvdr,
            args.compare_methods,
            args.benchmark,
            args.test_audio,
            args.guide,
            args.all,
        ]
    ):
        parser.print_help()
        print("\n" + "-" * 60)
        print("Running basic test by default...")
        print("-" * 60)
        args.test_basic = True
        args.guide = True

    results = {"passed": 0, "failed": 0}

    try:
        if args.all or args.test_basic:
            if test_basic_forward():
                results["passed"] += 1
            else:
                results["failed"] += 1

        if args.all or args.test_mvdr:
            if test_mvdr_method():
                results["passed"] += 1
            else:
                results["failed"] += 1

        if args.all or args.compare_methods:
            compare_mf_methods()

        if args.all or args.benchmark:
            benchmark_vs_dfnet4()

        if args.test_audio:
            if test_with_audio(args.test_audio):
                results["passed"] += 1
            else:
                results["failed"] += 1

        if args.all or args.guide:
            print_suitability_guide()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        results["failed"] += 1

    # Summary
    if results["passed"] > 0 or results["failed"] > 0:
        print("\n" + "=" * 60)
        print(f"RESULTS: {results['passed']} passed, {results['failed']} failed")
        print("=" * 60)

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
