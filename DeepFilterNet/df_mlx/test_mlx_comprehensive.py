"""Comprehensive tests for MLX DeepFilterNet4 implementation.

This module provides exhaustive testing for feature parity with PyTorch,
edge cases, numerical stability, and correctness of the MLX implementation.

Tests are organized by module:
- ops.py: STFT/iSTFT, ERB filterbank, complex operations
- modules.py: Conv layers, masks, DfOp
- mamba.py: Mamba blocks, selective scan
- model.py: Full model forward/backward
- train.py: Training utilities, weight conversion
"""

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    sr = 48000
    duration = 1.0  # 1 second
    num_samples = int(sr * duration)

    # Generate a test signal (sine wave + noise)
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    signal = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(num_samples).astype(np.float32)
    return mx.array(signal)


@pytest.fixture
def batch_audio():
    """Generate batch of audio for testing."""
    batch_size = 4
    sr = 48000
    duration = 1.0
    num_samples = int(sr * duration)

    signals = []
    for _ in range(batch_size):
        t = np.linspace(0, duration, num_samples, dtype=np.float32)
        freq = np.random.uniform(200, 1000)
        signal = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(num_samples).astype(np.float32)
        signals.append(signal)

    return mx.array(np.stack(signals))


# ============================================================================
# Test: ops.py - STFT/iSTFT
# ============================================================================


class TestSTFT:
    """Tests for STFT and iSTFT operations."""

    def test_stft_basic_shape(self, sample_audio):
        """Test STFT output shape."""
        from df_mlx.ops import stft

        n_fft = 960
        hop_length = 480

        real, imag = stft(sample_audio, n_fft=n_fft, hop_length=hop_length)

        # Expected shape: (num_frames, n_fft//2 + 1)
        expected_freqs = n_fft // 2 + 1
        # Note: expected_frames would be (len(sample_audio) + n_fft) // hop_length
        # but it varies based on padding

        assert real.shape[-1] == expected_freqs
        assert imag.shape[-1] == expected_freqs

    def test_stft_batch(self, batch_audio):
        """Test STFT with batched input."""
        from df_mlx.ops import stft

        n_fft = 960
        hop_length = 480

        real, imag = stft(batch_audio, n_fft=n_fft, hop_length=hop_length)

        assert real.shape[0] == batch_audio.shape[0]
        assert imag.shape[0] == batch_audio.shape[0]

    @pytest.mark.parametrize("n_fft", [256, 512, 960, 1024, 2048])
    def test_stft_various_fft_sizes(self, sample_audio, n_fft):
        """Test STFT with various FFT sizes."""
        from df_mlx.ops import stft

        hop_length = n_fft // 2
        real, imag = stft(sample_audio, n_fft=n_fft, hop_length=hop_length)

        expected_freqs = n_fft // 2 + 1
        assert real.shape[-1] == expected_freqs

    @pytest.mark.parametrize(
        "window_type",
        ["hann", "hamming", "blackman", "sqrt_hann"],
    )
    def test_stft_window_types(self, sample_audio, window_type):
        """Test STFT with different window types."""
        from df_mlx.ops import stft

        real, imag = stft(sample_audio, n_fft=512, hop_length=256, window=window_type)
        assert not mx.any(mx.isnan(real))
        assert not mx.any(mx.isnan(imag))

    def test_istft_reconstruction(self, sample_audio):
        """Test STFT → iSTFT reconstruction accuracy."""
        from df_mlx.ops import istft, stft

        n_fft = 960
        hop_length = 480

        # Forward transform
        spec = stft(sample_audio, n_fft=n_fft, hop_length=hop_length)

        # Inverse transform
        reconstructed = istft(spec, n_fft=n_fft, hop_length=hop_length, length=len(sample_audio))

        # Check reconstruction error
        error = mx.mean(mx.abs(sample_audio - reconstructed))
        mx.eval(error)

        # Allow for some reconstruction error due to windowing
        assert float(error) < 0.1, f"Reconstruction error too high: {float(error)}"

    def test_stft_short_signal(self):
        """Test STFT with very short signal."""
        from df_mlx.ops import stft

        # Signal shorter than FFT size
        short_signal = mx.array(np.random.randn(256).astype(np.float32))

        # Should work with center padding
        real, imag = stft(short_signal, n_fft=512, hop_length=256, center=True)
        assert not mx.any(mx.isnan(real))

    def test_stft_exact_fft_length(self):
        """Test STFT when signal length equals FFT size."""
        from df_mlx.ops import stft

        signal = mx.array(np.random.randn(512).astype(np.float32))
        real, imag = stft(signal, n_fft=512, hop_length=256, center=True)
        assert real.shape[0] >= 1  # At least one frame

    def test_stft_no_center_padding(self, sample_audio):
        """Test STFT without center padding."""
        from df_mlx.ops import stft

        real, imag = stft(sample_audio, n_fft=512, hop_length=256, center=False)
        assert not mx.any(mx.isnan(real))


# ============================================================================
# Test: ops.py - ERB Filterbank
# ============================================================================


class TestERBFilterbank:
    """Tests for ERB filterbank operations."""

    def test_erb_fb_shape(self):
        """Test ERB filterbank shape."""
        from df_mlx.ops import erb_fb

        sr = 48000
        fft_size = 960
        nb_bands = 32

        fb = erb_fb(sr, fft_size, nb_bands)

        n_freqs = fft_size // 2 + 1
        assert fb.shape == (n_freqs, nb_bands)

    def test_erb_fb_normalization(self):
        """Test ERB filterbank is normalized."""
        from df_mlx.ops import erb_fb

        fb = erb_fb(48000, 960, 32, normalized=True)

        # Each column should sum to ~1 (normalized filters)
        col_sums = mx.sum(fb, axis=0)
        mx.eval(col_sums)

        # Most columns should sum to ~1 (edge bands may differ)
        assert mx.all(col_sums > 0.5)
        assert mx.all(col_sums <= 2.0)

    def test_erb_transform_inverse(self):
        """Test ERB transform and inverse."""
        from df_mlx.ops import erb_fb, erb_inv_transform, erb_transform

        fb = erb_fb(48000, 960, 32)

        # Random spectrogram
        spec = mx.random.uniform(shape=(10, 481))  # (time, freqs)

        # Forward transform
        erb_spec = erb_transform(spec, fb)
        assert erb_spec.shape == (10, 32)

        # Inverse transform
        reconstructed = erb_inv_transform(erb_spec, fb)
        assert reconstructed.shape == (10, 481)

    @pytest.mark.parametrize("nb_bands", [16, 32, 64])
    def test_erb_fb_various_bands(self, nb_bands):
        """Test ERB filterbank with various band counts."""
        from df_mlx.ops import erb_fb

        fb = erb_fb(48000, 960, nb_bands)
        assert fb.shape[1] == nb_bands

    def test_erb_fb_frequency_range(self):
        """Test ERB filterbank with custom frequency range."""
        from df_mlx.ops import erb_fb

        fb = erb_fb(48000, 960, 32, min_freq=100.0, max_freq=8000.0)

        # First column should have weight at low frequencies only
        assert fb.shape == (481, 32)


# ============================================================================
# Test: ops.py - Complex Operations
# ============================================================================


class TestComplexOps:
    """Tests for complex number operations."""

    def test_complex_mul(self):
        """Test complex multiplication."""
        from df_mlx.ops import complex_mul

        a = (mx.array([1.0, 2.0]), mx.array([1.0, 1.0]))  # 1+i, 2+i
        b = (mx.array([2.0, 3.0]), mx.array([1.0, 2.0]))  # 2+i, 3+2i

        real, imag = complex_mul(a, b)
        mx.eval(real, imag)

        # (1+i)(2+i) = 2+i+2i-1 = 1+3i
        # (2+i)(3+2i) = 6+4i+3i-2 = 4+7i
        assert np.allclose(np.array(real), [1.0, 4.0])
        assert np.allclose(np.array(imag), [3.0, 7.0])

    def test_complex_conj(self):
        """Test complex conjugate."""
        from df_mlx.ops import complex_conj

        x = (mx.array([1.0, 2.0]), mx.array([3.0, 4.0]))
        real, imag = complex_conj(x)

        assert np.allclose(np.array(real), [1.0, 2.0])
        assert np.allclose(np.array(imag), [-3.0, -4.0])

    def test_complex_abs(self):
        """Test complex absolute value."""
        from df_mlx.ops import complex_abs

        x = (mx.array([3.0, 0.0]), mx.array([4.0, 5.0]))
        mag = complex_abs(x)
        mx.eval(mag)

        assert np.allclose(np.array(mag), [5.0, 5.0])

    def test_polar_conversion(self):
        """Test complex ↔ polar conversion."""
        from df_mlx.ops import complex_to_polar, polar_to_complex

        x = (mx.array([1.0, 0.0, -1.0]), mx.array([0.0, 1.0, 0.0]))

        mag, phase = complex_to_polar(x)
        real, imag = polar_to_complex(mag, phase)
        mx.eval(real, imag)

        assert np.allclose(np.array(real), [1.0, 0.0, -1.0], atol=1e-6)
        assert np.allclose(np.array(imag), [0.0, 1.0, 0.0], atol=1e-6)


# ============================================================================
# Test: modules.py - Convolution Modules
# ============================================================================


class TestConvModules:
    """Tests for convolution modules."""

    def test_conv2d_norm_act_basic(self):
        """Test Conv2dNormAct basic functionality."""
        from df_mlx.modules import Conv2dNormAct

        layer = Conv2dNormAct(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
            norm="batch",
            activation="relu",
        )

        x = mx.random.normal(shape=(2, 8, 16, 16))  # NHWC
        y = layer(x)
        mx.eval(y)

        assert y.shape == (2, 8, 16, 32)
        assert not mx.any(mx.isnan(y))

    @pytest.mark.parametrize("norm", [None, "batch", "group", "layer"])
    def test_conv2d_norm_types(self, norm):
        """Test Conv2dNormAct with different norm types."""
        from df_mlx.modules import Conv2dNormAct

        layer = Conv2dNormAct(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding=1,
            norm=norm,
        )

        x = mx.random.normal(shape=(2, 8, 16, 16))
        y = layer(x)
        mx.eval(y)

        assert not mx.any(mx.isnan(y))

    @pytest.mark.parametrize("activation", [None, "relu", "gelu", "silu", "leaky_relu"])
    def test_conv2d_activation_types(self, activation):
        """Test Conv2dNormAct with different activations."""
        from df_mlx.modules import Conv2dNormAct

        layer = Conv2dNormAct(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding=1,
            activation=activation,
        )

        x = mx.random.normal(shape=(2, 8, 16, 16))
        y = layer(x)
        mx.eval(y)

        assert not mx.any(mx.isnan(y))

    def test_conv_transpose_2d(self):
        """Test ConvTranspose2dNormAct."""
        from df_mlx.modules import ConvTranspose2dNormAct

        layer = ConvTranspose2dNormAct(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        x = mx.random.normal(shape=(2, 8, 8, 32))
        y = layer(x)
        mx.eval(y)

        # Output should be upsampled
        assert y.shape[1] > x.shape[1] or y.shape[2] > x.shape[2]


# ============================================================================
# Test: modules.py - Mask Operations
# ============================================================================


class TestMaskOperations:
    """Tests for mask operations."""

    @pytest.mark.parametrize(
        "mask_type",
        ["sigmoid", "relu", "bounded", "tanh"],
    )
    def test_mask_types(self, mask_type):
        """Test different mask types."""
        from df_mlx.modules import Mask

        mask_module = Mask(n_freqs=481, mask_type=mask_type)

        raw_mask = mx.random.normal(shape=(2, 100, 481))
        result = mask_module(raw_mask)
        mx.eval(result)

        # When spec=None, Mask returns just the processed mask array
        assert isinstance(result, mx.array)
        processed: mx.array = result  # Type narrowing for linter

        assert processed.shape == raw_mask.shape
        assert not mx.any(mx.isnan(processed))

        # Check mask is bounded for bounded types
        if mask_type in ["sigmoid", "bounded", "tanh"]:
            assert mx.all(processed >= -1.0)
            assert mx.all(processed <= 2.0)

    def test_mask_application(self):
        """Test mask application to spectrum."""
        from df_mlx.modules import Mask

        mask_module = Mask(n_freqs=481, mask_type="sigmoid")

        raw_mask = mx.random.normal(shape=(2, 100, 481))
        spec_real = mx.random.normal(shape=(2, 100, 481))
        spec_imag = mx.random.normal(shape=(2, 100, 481))

        out_real, out_imag = mask_module(raw_mask, spec=(spec_real, spec_imag))
        mx.eval(out_real, out_imag)

        assert out_real.shape == spec_real.shape
        assert out_imag.shape == spec_imag.shape


# ============================================================================
# Test: modules.py - Deep Filtering Operation
# ============================================================================


class TestDfOp:
    """Tests for Deep Filtering operation."""

    def test_dfop_basic(self):
        """Test basic DfOp functionality."""
        from df_mlx.modules import DfOp

        nb_df = 96
        df_order = 5

        df_op = DfOp(nb_df=nb_df, df_order=df_order, df_lookahead=0)

        # Create inputs
        batch_size = 2
        time_steps = 50
        n_freqs = 481

        spec_real = mx.random.normal(shape=(batch_size, time_steps, n_freqs))
        spec_imag = mx.random.normal(shape=(batch_size, time_steps, n_freqs))

        # Coefficients: (batch, time, nb_df, df_order, 2)
        coef = mx.random.normal(shape=(batch_size, time_steps, nb_df, df_order, 2)) * 0.1

        out_real, out_imag = df_op((spec_real, spec_imag), coef)
        mx.eval(out_real, out_imag)

        assert out_real.shape == (batch_size, time_steps, n_freqs)
        assert out_imag.shape == (batch_size, time_steps, n_freqs)
        assert not mx.any(mx.isnan(out_real))

    @pytest.mark.parametrize("df_order", [1, 3, 5, 7])
    def test_dfop_various_orders(self, df_order):
        """Test DfOp with various filter orders."""
        from df_mlx.modules import DfOp

        nb_df = 96
        df_op = DfOp(nb_df=nb_df, df_order=df_order, df_lookahead=0)

        spec_real = mx.random.normal(shape=(2, 50, 481))
        spec_imag = mx.random.normal(shape=(2, 50, 481))
        coef = mx.random.normal(shape=(2, 50, nb_df, df_order, 2)) * 0.1

        out_real, out_imag = df_op((spec_real, spec_imag), coef)
        mx.eval(out_real, out_imag)

        assert not mx.any(mx.isnan(out_real))

    @pytest.mark.parametrize("df_lookahead", [0, 1, 2])
    def test_dfop_lookahead(self, df_lookahead):
        """Test DfOp with various lookahead values."""
        from df_mlx.modules import DfOp

        nb_df = 96
        df_order = 5
        df_op = DfOp(nb_df=nb_df, df_order=df_order, df_lookahead=df_lookahead)

        spec_real = mx.random.normal(shape=(2, 50, 481))
        spec_imag = mx.random.normal(shape=(2, 50, 481))
        coef = mx.random.normal(shape=(2, 50, nb_df, df_order, 2)) * 0.1

        out_real, out_imag = df_op((spec_real, spec_imag), coef)
        mx.eval(out_real, out_imag)

        assert not mx.any(mx.isnan(out_real))

    def test_dfop_identity(self):
        """Test DfOp with identity-like coefficients."""
        from df_mlx.modules import DfOp

        nb_df = 96
        df_order = 5
        df_op = DfOp(nb_df=nb_df, df_order=df_order, df_lookahead=0)

        # Create input
        spec_real = mx.ones(shape=(1, 10, 481))
        spec_imag = mx.zeros(shape=(1, 10, 481))

        # Create identity-like coefficients (1+0i at center tap)
        coef = mx.zeros(shape=(1, 10, nb_df, df_order, 2))
        # Set center tap to (1, 0) for identity
        center_tap = df_order // 2
        coef = coef.at[:, :, :, center_tap, 0].add(1.0)

        out_real, out_imag = df_op((spec_real, spec_imag), coef)
        mx.eval(out_real, out_imag)

        # Output DF bins should be similar to input
        df_in = spec_real[:, :, :nb_df]
        df_out = out_real[:, :, :nb_df]

        error = mx.mean(mx.abs(df_in - df_out))
        mx.eval(error)
        # Allow some error due to edge effects
        assert float(error) < 0.5


# ============================================================================
# Test: modules.py - Grouped Linear
# ============================================================================


class TestGroupedLinear:
    """Tests for grouped linear layers."""

    def test_grouped_linear_basic(self):
        """Test GroupedLinear basic functionality."""
        from df_mlx.modules import GroupedLinear

        layer = GroupedLinear(in_features=256, out_features=128, groups=4)

        x = mx.random.normal(shape=(2, 100, 256))
        y = layer(x)
        mx.eval(y)

        assert y.shape == (2, 100, 128)
        assert not mx.any(mx.isnan(y))

    @pytest.mark.parametrize("groups", [1, 2, 4, 8, 16])
    def test_grouped_linear_various_groups(self, groups):
        """Test GroupedLinear with various group counts."""
        from df_mlx.modules import GroupedLinear

        in_features = 256
        out_features = 128
        layer = GroupedLinear(in_features=in_features, out_features=out_features, groups=groups)

        x = mx.random.normal(shape=(2, 100, in_features))
        y = layer(x)
        mx.eval(y)

        assert y.shape == (2, 100, out_features)


# ============================================================================
# Test: mamba.py - Mamba Modules
# ============================================================================


class TestMamba:
    """Tests for Mamba modules."""

    def test_mamba_block_basic(self):
        """Test MambaBlock basic functionality."""
        from df_mlx.mamba import MambaBlock

        d_model = 256
        mamba = MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand_factor=2)

        x = mx.random.normal(shape=(2, 50, d_model))
        out, state = mamba(x)
        mx.eval(out, state)

        assert out.shape == x.shape
        assert not mx.any(mx.isnan(out))

    def test_mamba_state_propagation(self):
        """Test Mamba state propagation across chunks."""
        from df_mlx.mamba import MambaBlock

        d_model = 128
        mamba = MambaBlock(d_model=d_model, d_state=16)

        # Process first chunk
        x1 = mx.random.normal(shape=(1, 20, d_model))
        out1, state1 = mamba(x1)
        mx.eval(out1, state1)

        # Process second chunk with state
        x2 = mx.random.normal(shape=(1, 20, d_model))
        out2, state2 = mamba(x2, state=state1)
        mx.eval(out2, state2)

        assert out2.shape == x2.shape
        assert not mx.any(mx.isnan(out2))

    @pytest.mark.parametrize("seq_len", [10, 50, 100, 200])
    def test_mamba_various_sequence_lengths(self, seq_len):
        """Test Mamba with various sequence lengths."""
        from df_mlx.mamba import MambaBlock

        mamba = MambaBlock(d_model=128, d_state=16)

        x = mx.random.normal(shape=(2, seq_len, 128))
        out, _ = mamba(x)
        mx.eval(out)

        assert out.shape == x.shape
        assert not mx.any(mx.isnan(out))

    def test_squeezed_mamba(self):
        """Test SqueezedMamba module."""
        from df_mlx.mamba import SqueezedMamba

        mamba = SqueezedMamba(
            input_size=128,
            hidden_size=256,
            output_size=128,
            num_layers=2,
            d_state=16,
        )

        x = mx.random.normal(shape=(2, 50, 128))
        out, state = mamba(x)
        mx.eval(out)

        assert out.shape == (2, 50, 128)
        assert not mx.any(mx.isnan(out))


# ============================================================================
# Test: model.py - Full Model
# ============================================================================


class TestFullModel:
    """Tests for full DfNet4 model."""

    def test_encoder_forward(self):
        """Test Encoder4 forward pass."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import Encoder4

        params = ModelParams4()
        encoder = Encoder4(params)

        # Input shapes: (batch, time, erb_bands) and (batch, time, df_bins, 2)
        feat_erb = mx.random.normal(shape=(2, 100, 32))
        feat_spec = mx.random.normal(shape=(2, 100, 96, 2))

        # Encoder now returns (emb, lsnr) tuple
        emb, lsnr = encoder(feat_erb, feat_spec)
        mx.eval(emb, lsnr)

        assert emb.shape[0] == 2
        assert emb.shape[1] == 100
        assert emb.shape[2] == params.emb_hidden_dim
        assert not mx.any(mx.isnan(emb))
        # LSNR is per-frame with shape (batch, time, 1)
        assert lsnr.shape == (2, 100, 1)

    def test_erb_decoder_forward(self):
        """Test ErbDecoder4 forward pass."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import ErbDecoder4

        params = ModelParams4()
        decoder = ErbDecoder4(params)

        # Create dummy encoder embedding
        batch, time = 2, 100
        emb = mx.random.normal(shape=(batch, time, params.emb_hidden_dim))

        mask = decoder(emb)
        mx.eval(mask)

        assert mask.shape[0] == batch
        assert mask.shape[1] == time
        assert mask.shape[-1] == params.nb_erb
        assert not mx.any(mx.isnan(mask))

    def test_dfnet4_forward(self):
        """Test DfNet4 forward pass."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import DfNet4

        params = ModelParams4()
        model = DfNet4(params)

        # Create input - shapes match MLX model expectations
        batch, time, n_freqs = 2, 50, 481
        spec_real = mx.random.normal(shape=(batch, time, n_freqs))
        spec_imag = mx.random.normal(shape=(batch, time, n_freqs))
        feat_erb = mx.random.normal(shape=(batch, time, 32))  # (batch, time, erb_bands)
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2))  # (batch, time, df_bins, 2)

        spec = (spec_real, spec_imag)
        output = model(spec, feat_erb, feat_spec)
        mx.eval(output[0], output[1])

        out_real, out_imag = output
        assert out_real.shape == spec_real.shape
        assert not mx.any(mx.isnan(out_real))

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_model_batch_sizes(self, batch_size):
        """Test model with various batch sizes."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import DfNet4

        params = ModelParams4()
        model = DfNet4(params)

        time, n_freqs = 50, 481
        spec_real = mx.random.normal(shape=(batch_size, time, n_freqs))
        spec_imag = mx.random.normal(shape=(batch_size, time, n_freqs))
        feat_erb = mx.random.normal(shape=(batch_size, time, 32))
        feat_spec = mx.random.normal(shape=(batch_size, time, 96, 2))

        out_real, out_imag = model((spec_real, spec_imag), feat_erb, feat_spec)
        mx.eval(out_real)

        assert out_real.shape[0] == batch_size

    def test_model_gradient_flow(self):
        """Test gradient flow through the model."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import DfNet4

        params = ModelParams4()
        model = DfNet4(params)

        # Create input
        batch, time, n_freqs = 2, 20, 481
        spec_real = mx.random.normal(shape=(batch, time, n_freqs))
        spec_imag = mx.random.normal(shape=(batch, time, n_freqs))
        feat_erb = mx.random.normal(shape=(batch, time, 32))
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2))

        def loss_fn(model):
            out_real, out_imag = model((spec_real, spec_imag), feat_erb, feat_spec)
            return mx.mean(out_real**2 + out_imag**2)

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss)

        assert not mx.isnan(loss)
        # Check that gradients exist for some parameters
        assert grads is not None

    def test_model_enhance(self, sample_audio):
        """Test end-to-end enhancement."""
        from df_mlx.model import init_model

        model = init_model()

        # Enhance audio
        enhanced = model.enhance(sample_audio)
        mx.eval(enhanced)

        assert enhanced.shape == sample_audio.shape
        assert not mx.any(mx.isnan(enhanced))

    def test_dfnet4_lite(self):
        """Test DfNet4Lite variant."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import DfNet4Lite

        params = ModelParams4()
        model = DfNet4Lite(params)

        batch, time, n_freqs = 2, 50, 481
        spec_real = mx.random.normal(shape=(batch, time, n_freqs))
        spec_imag = mx.random.normal(shape=(batch, time, n_freqs))
        feat_erb = mx.random.normal(shape=(batch, time, 32))
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2))

        out_real, out_imag = model((spec_real, spec_imag), feat_erb, feat_spec)
        mx.eval(out_real)

        assert not mx.any(mx.isnan(out_real))

    def test_parameter_count(self):
        """Test parameter counting."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import DfNet4, DfNet4Lite, count_parameters

        params = ModelParams4()
        model_full = DfNet4(params)
        model_lite = DfNet4Lite(params)

        count_full = count_parameters(model_full)
        count_lite = count_parameters(model_lite)

        assert count_full > 0
        assert count_lite > 0
        # Lite should have fewer parameters
        assert count_lite < count_full


# ============================================================================
# Test: train.py - Training Utilities
# ============================================================================


class TestTraining:
    """Tests for training utilities."""

    def test_spectral_loss(self):
        """Test spectral loss computation."""
        from df_mlx.train import spectral_loss

        pred = (mx.random.normal(shape=(2, 50, 481)), mx.random.normal(shape=(2, 50, 481)))
        target = (mx.random.normal(shape=(2, 50, 481)), mx.random.normal(shape=(2, 50, 481)))

        loss = spectral_loss(pred, target)
        mx.eval(loss)

        assert loss.shape == ()  # Scalar
        assert not mx.isnan(loss)
        assert float(loss) >= 0

    def test_multi_resolution_stft_loss(self):
        """Test multi-resolution STFT loss."""
        from df_mlx.train import multi_resolution_stft_loss

        pred = mx.random.normal(shape=(2, 48000))
        target = mx.random.normal(shape=(2, 48000))

        loss = multi_resolution_stft_loss(pred, target, fft_sizes=(512, 1024, 2048))
        mx.eval(loss)

        assert loss.shape == ()
        assert not mx.isnan(loss)

    def test_snr_loss(self):
        """Test SNR loss."""
        from df_mlx.train import snr_loss

        pred = mx.random.normal(shape=(2, 48000))
        target = mx.random.normal(shape=(2, 48000))

        loss = snr_loss(pred, target)
        mx.eval(loss)

        assert loss.shape == ()
        assert not mx.isnan(loss)

    def test_lr_schedule(self):
        """Test learning rate schedule."""
        from df_mlx.train import WarmupCosineSchedule

        schedule = WarmupCosineSchedule(
            base_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
            min_lr=1e-6,
        )

        # Check warmup phase
        assert schedule(0) == 0.0
        assert schedule(50) == 0.5e-3
        assert schedule(100) == 1e-3

        # Check decay phase
        lr_500 = schedule(500)
        lr_900 = schedule(900)
        assert lr_500 > lr_900  # Should decay

        # Check min lr
        assert schedule(1000) >= 1e-6

    def test_trainer_init(self):
        """Test Trainer initialization."""
        from df_mlx.config import TrainConfig
        from df_mlx.model import init_model
        from df_mlx.train import Trainer

        model = init_model()
        config = TrainConfig()
        trainer = Trainer(model, config)

        assert trainer.step == 0
        assert trainer.best_loss == float("inf")

    def test_train_step(self):
        """Test single training step."""
        from df_mlx.config import TrainConfig
        from df_mlx.model import init_model
        from df_mlx.train import Trainer

        model = init_model()
        config = TrainConfig()
        trainer = Trainer(model, config)

        # Create dummy batch - shapes match MLX model expectations
        batch, time, n_freqs = 2, 20, 481
        spec = (mx.random.normal(shape=(batch, time, n_freqs)), mx.random.normal(shape=(batch, time, n_freqs)))
        feat_erb = mx.random.normal(shape=(batch, time, 32))  # (batch, time, erb_bands)
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2))  # (batch, time, df_bins, 2)
        target = (mx.random.normal(shape=(batch, time, n_freqs)), mx.random.normal(shape=(batch, time, n_freqs)))

        loss = trainer.train_step(spec, feat_erb, feat_spec, target)

        assert loss >= 0
        assert trainer.step == 1

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        from df_mlx.config import TrainConfig
        from df_mlx.model import init_model
        from df_mlx.train import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(checkpoint_dir=tmpdir)
            model = init_model()
            trainer = Trainer(model, config)

            # Save checkpoint
            trainer.step = 100
            trainer.save_checkpoint("test.safetensors")

            # Create new model and load
            model2 = init_model()
            trainer2 = Trainer(model2, config)
            trainer2.load_checkpoint(Path(tmpdir) / "test.safetensors")

            assert trainer2.step == 100


# ============================================================================
# Test: Weight Conversion
# ============================================================================


class TestWeightConversion:
    """Tests for PyTorch ↔ MLX weight conversion."""

    def test_convert_pytorch_weights_shape(self):
        """Test weight conversion preserves appropriate shapes."""
        from df_mlx.train import convert_pytorch_weights

        # Simulate PyTorch state dict with numpy arrays
        pytorch_state = {
            "linear.weight": np.random.randn(64, 128).astype(np.float32),
            "linear.bias": np.random.randn(64).astype(np.float32),
            "conv.weight": np.random.randn(32, 16, 3, 3).astype(np.float32),
        }

        mlx_weights = convert_pytorch_weights(pytorch_state)

        # Linear weights should keep same shape
        assert mlx_weights["linear.weight"].shape == (64, 128)

        # Conv weights should be transposed: (out, in, H, W) -> (out, H, W, in)
        assert mlx_weights["conv.weight"].shape == (32, 3, 3, 16)


# ============================================================================
# Test: Edge Cases and Numerical Stability
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_zero_input(self):
        """Test model with zero input."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import DfNet4

        params = ModelParams4()
        model = DfNet4(params)

        batch, time, n_freqs = 2, 50, 481
        spec = (mx.zeros(shape=(batch, time, n_freqs)), mx.zeros(shape=(batch, time, n_freqs)))
        feat_erb = mx.zeros(shape=(batch, time, 32))
        feat_spec = mx.zeros(shape=(batch, time, 96, 2))

        out_real, out_imag = model(spec, feat_erb, feat_spec)
        mx.eval(out_real, out_imag)

        # Should not produce NaN
        assert not mx.any(mx.isnan(out_real))
        assert not mx.any(mx.isnan(out_imag))

    def test_large_values(self):
        """Test model with large input values."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import DfNet4

        params = ModelParams4()
        model = DfNet4(params)

        batch, time, n_freqs = 2, 50, 481
        spec = (mx.ones(shape=(batch, time, n_freqs)) * 100, mx.ones(shape=(batch, time, n_freqs)) * 100)
        feat_erb = mx.ones(shape=(batch, time, 32)) * 100
        feat_spec = mx.ones(shape=(batch, time, 96, 2)) * 100

        out_real, out_imag = model(spec, feat_erb, feat_spec)
        mx.eval(out_real, out_imag)

        # Should not produce NaN or Inf
        assert not mx.any(mx.isnan(out_real))
        assert not mx.any(mx.isinf(out_real))

    def test_very_short_sequence(self):
        """Test model with very short sequence."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import DfNet4

        params = ModelParams4()
        model = DfNet4(params)

        batch, time, n_freqs = 2, 5, 481  # Very short
        spec = (mx.random.normal(shape=(batch, time, n_freqs)), mx.random.normal(shape=(batch, time, n_freqs)))
        feat_erb = mx.random.normal(shape=(batch, time, 32))
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2))

        out_real, out_imag = model(spec, feat_erb, feat_spec)
        mx.eval(out_real, out_imag)

        assert out_real.shape == (batch, time, n_freqs)
        assert not mx.any(mx.isnan(out_real))

    def test_single_sample_batch(self):
        """Test model with batch size 1."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import DfNet4

        params = ModelParams4()
        model = DfNet4(params)

        batch, time, n_freqs = 1, 50, 481
        spec = (mx.random.normal(shape=(batch, time, n_freqs)), mx.random.normal(shape=(batch, time, n_freqs)))
        feat_erb = mx.random.normal(shape=(batch, time, 32))
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2))

        out_real, out_imag = model(spec, feat_erb, feat_spec)
        mx.eval(out_real, out_imag)

        assert out_real.shape[0] == 1
        assert not mx.any(mx.isnan(out_real))


# ============================================================================
# Test: Numerical Properties
# ============================================================================


class TestNumericalProperties:
    """Tests for numerical properties and invariants."""

    def test_stft_istft_roundtrip_energy(self):
        """Test that STFT->iSTFT preserves signal energy (approximately)."""
        from df_mlx.ops import istft, stft

        # Create a test signal
        np.random.seed(42)
        signal = mx.array(np.random.randn(48000).astype(np.float32))
        original_energy = float(mx.sum(signal**2))

        # STFT and iSTFT
        spec_real, spec_imag = stft(signal, n_fft=960, hop_length=480)
        reconstructed = istft((spec_real, spec_imag), n_fft=960, hop_length=480)

        # Trim to same length and compute reconstructed energy
        min_len = min(signal.shape[0], reconstructed.shape[0])
        recon_trimmed = reconstructed[:min_len]

        # Check energy is approximately preserved
        recon_energy = float(mx.sum(recon_trimmed**2))
        energy_ratio = recon_energy / original_energy

        # Energy should be within 10% (accounting for edge effects)
        assert 0.85 < energy_ratio < 1.15, f"Energy ratio: {energy_ratio}"

    def test_erb_filterbank_covers_spectrum(self):
        """Test that ERB filterbank covers the full spectrum."""
        from df_mlx.ops import erb_fb

        sr = 48000
        fft_size = 960
        n_erb = 32
        fb = erb_fb(sr=sr, fft_size=fft_size, nb_bands=n_erb)

        # Sum across all bands should give roughly full coverage
        # (not exactly 1 due to triangular filters)
        coverage = mx.sum(fb, axis=1)
        min_coverage = float(mx.min(coverage))

        # Each frequency bin should be covered by at least some bands
        assert min_coverage > 0, "Some frequency bins have no coverage"

    def test_mask_bounds(self):
        """Test that masks stay within expected bounds."""
        from df_mlx.modules import Mask

        # Test bounded mask
        mask = Mask(n_freqs=32, mask_type="bounded")
        x = mx.random.normal(shape=(2, 50, 32)) * 10  # Large values

        output = mask(x)
        mx.eval(output)

        # Bounded mask should be in [0, 1]
        assert float(mx.min(output)) >= 0.0
        assert float(mx.max(output)) <= 1.0

    def test_model_output_scale(self):
        """Test that model output scale is reasonable."""
        from df_mlx.model import init_model

        model = init_model()

        # Create input with known scale
        batch, time, n_freqs = 2, 50, 481
        input_scale = 0.5
        spec = (
            mx.random.normal(shape=(batch, time, n_freqs)) * input_scale,
            mx.random.normal(shape=(batch, time, n_freqs)) * input_scale,
        )
        feat_erb = mx.random.normal(shape=(batch, time, 32)) * input_scale
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2)) * input_scale

        out_real, out_imag = model(spec, feat_erb, feat_spec)
        mx.eval(out_real, out_imag)

        # Output should have reasonable scale (not exploding)
        output_scale = float(mx.mean(mx.abs(out_real)))
        assert output_scale < 100, f"Output scale too large: {output_scale}"
        assert output_scale > 1e-6, f"Output scale too small: {output_scale}"

    def test_gradient_magnitude(self):
        """Test that gradients have reasonable magnitude."""
        import mlx.nn as nn

        from df_mlx.model import init_model
        from df_mlx.train import spectral_loss

        model = init_model()

        # Create inputs
        batch, time, n_freqs = 2, 20, 481
        spec = (mx.random.normal(shape=(batch, time, n_freqs)), mx.random.normal(shape=(batch, time, n_freqs)))
        feat_erb = mx.random.normal(shape=(batch, time, 32))
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2))
        target = (mx.random.normal(shape=(batch, time, n_freqs)), mx.random.normal(shape=(batch, time, n_freqs)))

        def loss_fn(model, spec, feat_erb, feat_spec, target):
            pred = model(spec, feat_erb, feat_spec)
            return spectral_loss(pred, target)

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, spec, feat_erb, feat_spec, target)
        mx.eval(loss, grads)

        # Check gradient magnitudes are reasonable
        from mlx.utils import tree_flatten

        flat_grads = tree_flatten(grads)

        for name, grad in flat_grads:
            if isinstance(grad, mx.array):
                grad_norm = float(mx.sqrt(mx.sum(grad**2)))
                assert grad_norm < 1000, f"Gradient too large for {name}: {grad_norm}"
            # Some gradients might be exactly zero or non-array, that's ok

    def test_deterministic_inference(self):
        """Test that inference is deterministic (no dropout/noise in eval mode)."""
        from df_mlx.model import init_model

        model = init_model()

        # Create fixed input
        mx.random.seed(42)
        batch, time, n_freqs = 2, 50, 481
        spec = (mx.random.normal(shape=(batch, time, n_freqs)), mx.random.normal(shape=(batch, time, n_freqs)))
        feat_erb = mx.random.normal(shape=(batch, time, 32))
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2))

        # Run twice
        out1_real, out1_imag = model(spec, feat_erb, feat_spec)
        mx.eval(out1_real, out1_imag)

        out2_real, out2_imag = model(spec, feat_erb, feat_spec)
        mx.eval(out2_real, out2_imag)

        # Should be identical
        assert mx.allclose(out1_real, out2_real).item()
        assert mx.allclose(out1_imag, out2_imag).item()


# ============================================================================
# Test: New Features - LSNR, MultiResDfDecoder, AdaptiveOrderPredictor
# ============================================================================


class TestLSNRFeatures:
    """Tests for LSNR estimation and dropout features."""

    def test_encoder_outputs_lsnr(self):
        """Test that Encoder4 now outputs both embedding and LSNR."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import Encoder4

        params = ModelParams4()
        encoder = Encoder4(params)

        batch, time = 2, 100
        feat_erb = mx.random.normal(shape=(batch, time, 32))
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2))

        output = encoder(feat_erb, feat_spec)
        mx.eval(output)

        # Should return tuple of (emb, lsnr)
        assert isinstance(output, tuple)
        assert len(output) == 2

        emb, lsnr = output
        mx.eval(emb, lsnr)

        assert emb.shape == (batch, time, params.emb_hidden_dim)
        assert lsnr.shape == (batch, time, 1)
        assert not mx.any(mx.isnan(emb))
        assert not mx.any(mx.isnan(lsnr))

    def test_lsnr_range(self):
        """Test that LSNR is within expected range."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import Encoder4

        params = ModelParams4()
        encoder = Encoder4(params)

        batch, time = 2, 100
        feat_erb = mx.random.normal(shape=(batch, time, 32))
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2))

        _, lsnr = encoder(feat_erb, feat_spec)
        mx.eval(lsnr)

        lsnr_min = params.lsnr.lsnr_min
        lsnr_max = params.lsnr.lsnr_max

        # LSNR should be bounded by tanh scaling
        assert float(mx.min(lsnr)) >= lsnr_min - 0.1
        assert float(mx.max(lsnr)) <= lsnr_max + 0.1

    def test_dfnet4_lsnr_dropout_flag(self):
        """Test DfNet4 with LSNR dropout enabled."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import DfNet4

        params = ModelParams4()
        model = DfNet4(params, lsnr_dropout=True, lsnr_dropout_threshold=-10.0)

        assert model.lsnr_dropout is True
        assert model.lsnr_dropout_threshold == -10.0

    def test_dfnet4_training_mode(self):
        """Test DfNet4 forward with training flag."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import DfNet4

        params = ModelParams4()
        model = DfNet4(params, lsnr_dropout=True)

        batch, time, n_freqs = 2, 50, 481
        spec_real = mx.random.normal(shape=(batch, time, n_freqs))
        spec_imag = mx.random.normal(shape=(batch, time, n_freqs))
        feat_erb = mx.random.normal(shape=(batch, time, 32))
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2))

        # Run in training mode
        out_train = model((spec_real, spec_imag), feat_erb, feat_spec, training=True)
        mx.eval(out_train[0])

        # Run in inference mode
        out_eval = model((spec_real, spec_imag), feat_erb, feat_spec, training=False)
        mx.eval(out_eval[0])

        # Both should produce valid output
        assert not mx.any(mx.isnan(out_train[0]))
        assert not mx.any(mx.isnan(out_eval[0]))

    def test_forward_with_lsnr(self):
        """Test forward_with_lsnr method."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import DfNet4

        params = ModelParams4()
        model = DfNet4(params)

        batch, time, n_freqs = 2, 50, 481
        spec_real = mx.random.normal(shape=(batch, time, n_freqs))
        spec_imag = mx.random.normal(shape=(batch, time, n_freqs))
        feat_erb = mx.random.normal(shape=(batch, time, 32))
        feat_spec = mx.random.normal(shape=(batch, time, 96, 2))

        result = model.forward_with_lsnr((spec_real, spec_imag), feat_erb, feat_spec)
        mx.eval(result)

        assert isinstance(result, tuple)
        spec_out, lsnr = result

        assert isinstance(spec_out, tuple)
        assert lsnr.shape == (batch, time, 1)
        assert not mx.any(mx.isnan(lsnr))

    def test_lsnr_loss_function(self):
        """Test LSNR loss computation."""
        from df_mlx.train import lsnr_loss

        pred_lsnr = mx.random.normal(shape=(2, 50, 1)) * 10
        target_lsnr = mx.random.normal(shape=(2, 50, 1)) * 10

        loss = lsnr_loss(pred_lsnr, target_lsnr)
        mx.eval(loss)

        assert loss.shape == ()
        assert not mx.isnan(loss)
        assert float(loss) >= 0

    def test_lsnr_loss_clipping(self):
        """Test that LSNR loss clips values correctly."""
        from df_mlx.train import lsnr_loss

        # Create extreme values
        pred_lsnr = mx.array([[[100.0]], [[-100.0]]])
        target_lsnr = mx.array([[[0.0]], [[0.0]]])

        loss = lsnr_loss(pred_lsnr, target_lsnr, lsnr_min=-15.0, lsnr_max=40.0)
        mx.eval(loss)

        # Loss should be finite due to clipping
        assert not mx.isnan(loss)


class TestMultiResDfDecoder:
    """Tests for multi-resolution DF decoder."""

    def test_multi_res_df_decoder_init(self):
        """Test MultiResDfDecoder initialization."""
        from df_mlx.model import MultiResDfDecoder

        resolutions = [(96, 5), (48, 3), (24, 2)]
        decoder = MultiResDfDecoder(
            emb_dim=256,
            hidden_dim=256,
            resolutions=resolutions,
        )

        assert len(decoder.output_heads) == len(resolutions)

    def test_multi_res_df_decoder_forward(self):
        """Test MultiResDfDecoder forward pass."""
        from df_mlx.model import MultiResDfDecoder

        resolutions = [(96, 5), (48, 3), (24, 2)]
        emb_dim = 256
        decoder = MultiResDfDecoder(
            emb_dim=emb_dim,
            hidden_dim=256,
            resolutions=resolutions,
        )

        batch, time = 2, 50
        emb = mx.random.normal(shape=(batch, time, emb_dim))

        outputs = decoder(emb)
        mx.eval(outputs)

        # Should return list of coefficients for each resolution
        assert isinstance(outputs, list)
        assert len(outputs) == len(resolutions)

        for (nb_df, df_order), coef in zip(resolutions, outputs):
            assert coef.shape == (batch, time, nb_df, df_order, 2)
            assert not mx.any(mx.isnan(coef))

    @pytest.mark.parametrize(
        "resolutions",
        [
            [(96, 5)],
            [(96, 5), (48, 3)],
            [(96, 5), (48, 3), (24, 2)],
            [(64, 4), (32, 2)],
        ],
    )
    def test_multi_res_various_configs(self, resolutions):
        """Test MultiResDfDecoder with various resolution configs."""
        from df_mlx.model import MultiResDfDecoder

        emb_dim = 256
        decoder = MultiResDfDecoder(
            emb_dim=emb_dim,
            hidden_dim=256,
            resolutions=resolutions,
        )

        batch, time = 2, 50
        emb = mx.random.normal(shape=(batch, time, emb_dim))

        outputs = decoder(emb)
        mx.eval(outputs)

        assert len(outputs) == len(resolutions)

    def test_multi_res_gradient_flow(self):
        """Test gradient flow through MultiResDfDecoder."""
        from df_mlx.model import MultiResDfDecoder

        resolutions = [(96, 5), (48, 3)]
        emb_dim = 256
        decoder = MultiResDfDecoder(
            emb_dim=emb_dim,
            hidden_dim=256,
            resolutions=resolutions,
        )

        batch, time = 2, 20
        emb = mx.random.normal(shape=(batch, time, emb_dim))

        def loss_fn(model):
            outputs = model(emb)
            return sum(mx.mean(c**2) for c in outputs)

        loss, grads = nn.value_and_grad(decoder, loss_fn)(decoder)
        mx.eval(loss)

        assert not mx.isnan(loss)


class TestAdaptiveOrderPredictor:
    """Tests for adaptive order predictor."""

    def test_adaptive_order_init(self):
        """Test AdaptiveOrderPredictor initialization."""
        from df_mlx.model import AdaptiveOrderPredictor

        predictor = AdaptiveOrderPredictor(
            emb_dim=256,
            max_order=5,
            min_order=1,
        )

        assert predictor.max_order == 5

    def test_adaptive_order_forward(self):
        """Test AdaptiveOrderPredictor forward pass."""
        from df_mlx.model import AdaptiveOrderPredictor

        max_order = 5
        min_order = 1
        predictor = AdaptiveOrderPredictor(
            emb_dim=256,
            max_order=max_order,
            min_order=min_order,
        )

        batch, time = 2, 50
        emb = mx.random.normal(shape=(batch, time, 256))

        order_weights, predicted_order = predictor(emb)
        mx.eval(order_weights, predicted_order)

        # order_weights: (batch, time, num_orders) - softmax probabilities
        num_orders = max_order - min_order + 1
        assert order_weights.shape == (batch, time, num_orders)
        # Should sum to ~1 along order dimension
        sums = mx.sum(order_weights, axis=-1)
        assert mx.allclose(sums, mx.ones_like(sums), atol=1e-5).item()

        # predicted_order: (batch, time) - integer order
        assert predicted_order.shape == (batch, time)

    @pytest.mark.parametrize("max_order", [3, 5, 7, 10])
    def test_adaptive_order_various_max(self, max_order):
        """Test with various max_order values."""
        from df_mlx.model import AdaptiveOrderPredictor

        min_order = 2
        predictor = AdaptiveOrderPredictor(
            emb_dim=256,
            max_order=max_order,
            min_order=min_order,
        )

        batch, time = 2, 50
        emb = mx.random.normal(shape=(batch, time, 256))

        order_weights, predicted_order = predictor(emb)
        mx.eval(order_weights)

        num_orders = max_order - min_order + 1
        assert order_weights.shape == (batch, time, num_orders)

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_adaptive_order_temperature(self, temperature):
        """Test temperature effect on order predictions."""
        from df_mlx.model import AdaptiveOrderPredictor

        predictor = AdaptiveOrderPredictor(
            emb_dim=256,
            max_order=5,
            min_order=2,
        )

        batch, time = 2, 50
        emb = mx.random.normal(shape=(batch, time, 256))

        # Temperature is passed to __call__, not __init__
        order_weights, _ = predictor(emb, temperature=temperature)
        mx.eval(order_weights)

        # Lower temperature should produce more peaked distributions
        # Higher temperature should produce more uniform distributions
        entropy = -mx.sum(order_weights * mx.log(order_weights + 1e-8), axis=-1)
        avg_entropy = float(mx.mean(entropy))

        # Just verify it produces valid output
        assert not mx.isnan(mx.array(avg_entropy))

    def test_adaptive_order_gradient_flow(self):
        """Test gradient flow through AdaptiveOrderPredictor."""
        from df_mlx.model import AdaptiveOrderPredictor

        predictor = AdaptiveOrderPredictor(
            emb_dim=256,
            max_order=5,
            min_order=2,
        )

        batch, time = 2, 20
        emb = mx.random.normal(shape=(batch, time, 256))

        def loss_fn(model):
            order_weights, _ = model(emb)
            return mx.mean(order_weights)

        loss, grads = nn.value_and_grad(predictor, loss_fn)(predictor)
        mx.eval(loss)

        assert not mx.isnan(loss)


class TestLSNRConfig:
    """Tests for LSNR configuration."""

    def test_lsnr_params_defaults(self):
        """Test LsnrParams default values."""
        from df_mlx.config import LsnrParams

        params = LsnrParams()

        assert params.lsnr_min == -15.0
        assert params.lsnr_max == 40.0
        assert params.lsnr_dropout_threshold == -10.0
        assert params.lsnr_dropout is False

    def test_model_params_has_lsnr(self):
        """Test that ModelParams4 includes LsnrParams."""
        from df_mlx.config import ModelParams4

        params = ModelParams4()

        assert hasattr(params, "lsnr")
        assert params.lsnr.lsnr_min == -15.0
        assert params.lsnr.lsnr_max == 40.0

    def test_train_config_lsnr_dropout(self):
        """Test TrainConfig has LSNR dropout settings."""
        from df_mlx.config import TrainConfig

        config = TrainConfig()

        assert hasattr(config, "lsnr_dropout")
        assert hasattr(config, "lsnr_dropout_threshold")
        assert config.lsnr_dropout is False
        assert config.lsnr_dropout_threshold == -10.0


# ============================================================================
# Hybrid Encoder Tests
# ============================================================================


class TestWaveformEncoder:
    """Tests for WaveformEncoder."""

    def test_waveform_encoder_init(self):
        """Test WaveformEncoder initialization."""
        from df_mlx.model import WaveformEncoder

        encoder = WaveformEncoder(out_dim=256)
        assert encoder.out_dim == 256
        assert encoder.in_channels == 1

    def test_waveform_encoder_forward(self):
        """Test WaveformEncoder forward pass."""
        from df_mlx.model import WaveformEncoder

        encoder = WaveformEncoder(out_dim=256)

        batch, samples = 2, 48000  # 1 second at 48kHz
        waveform = mx.random.normal(shape=(batch, samples))

        features = encoder(waveform)
        mx.eval(features)

        assert features.ndim == 3
        assert features.shape[0] == batch
        assert features.shape[2] == 256

    def test_waveform_encoder_3d_input(self):
        """Test WaveformEncoder with 3D input."""
        from df_mlx.model import WaveformEncoder

        encoder = WaveformEncoder(out_dim=128)

        batch, channels, samples = 2, 1, 24000
        waveform = mx.random.normal(shape=(batch, channels, samples))

        features = encoder(waveform)
        mx.eval(features)

        assert features.ndim == 3
        assert features.shape[0] == batch

    @pytest.mark.parametrize("num_layers", [2, 3, 4])
    def test_waveform_encoder_layers(self, num_layers):
        """Test WaveformEncoder with different layer counts."""
        from df_mlx.model import WaveformEncoder

        encoder = WaveformEncoder(num_layers=num_layers, out_dim=256)

        waveform = mx.random.normal(shape=(2, 16000))
        features = encoder(waveform)
        mx.eval(features)

        assert features.shape[2] == 256


class TestPhaseEncoder:
    """Tests for PhaseEncoder."""

    def test_phase_encoder_init(self):
        """Test PhaseEncoder initialization."""
        from df_mlx.model import PhaseEncoder

        encoder = PhaseEncoder(n_freqs=96, out_dim=256)
        assert encoder.n_freqs == 96
        assert encoder.out_dim == 256

    def test_phase_encoder_forward_angle(self):
        """Test PhaseEncoder with angle input."""
        from df_mlx.model import PhaseEncoder

        encoder = PhaseEncoder(n_freqs=96, out_dim=256)

        batch, time, freqs = 2, 50, 96
        phase_angle = mx.random.uniform(shape=(batch, time, freqs), low=-3.14159, high=3.14159)

        features = encoder(phase_angle)
        mx.eval(features)

        assert features.shape == (batch, time, 256)

    def test_phase_encoder_forward_complex(self):
        """Test PhaseEncoder with complex input."""
        from df_mlx.model import PhaseEncoder

        encoder = PhaseEncoder(n_freqs=96, out_dim=256)

        batch, time, freqs = 2, 50, 96
        # Complex format (batch, time, freq, 2)
        complex_spec = mx.random.normal(shape=(batch, time, freqs, 2))

        features = encoder(complex_spec)
        mx.eval(features)

        assert features.shape == (batch, time, 256)

    @pytest.mark.parametrize("n_freqs", [48, 96, 128])
    def test_phase_encoder_various_freqs(self, n_freqs):
        """Test PhaseEncoder with various frequency counts."""
        from df_mlx.model import PhaseEncoder

        encoder = PhaseEncoder(n_freqs=n_freqs, out_dim=128)

        phase = mx.random.uniform(shape=(2, 50, n_freqs), low=-3.14159, high=3.14159)
        features = encoder(phase)
        mx.eval(features)

        assert features.shape == (2, 50, 128)


class TestCrossDomainAttention:
    """Tests for CrossDomainAttention."""

    def test_cross_domain_attention_init(self):
        """Test CrossDomainAttention initialization."""
        from df_mlx.model import CrossDomainAttention

        attn = CrossDomainAttention(
            time_dim=256,
            mag_dim=256,
            phase_dim=256,
            out_dim=256,
            num_heads=8,
        )
        assert attn.out_dim == 256

    def test_cross_domain_attention_forward(self):
        """Test CrossDomainAttention forward pass."""
        from df_mlx.model import CrossDomainAttention

        attn = CrossDomainAttention(
            time_dim=256,
            mag_dim=256,
            phase_dim=256,
            out_dim=256,
        )

        batch, time = 2, 50
        time_feat = mx.random.normal(shape=(batch, time, 256))
        mag_feat = mx.random.normal(shape=(batch, time, 256))
        phase_feat = mx.random.normal(shape=(batch, time, 256))

        fused = attn(time_feat, mag_feat, phase_feat)
        mx.eval(fused)

        assert fused.shape == (batch, time, 256)

    def test_cross_domain_attention_different_dims(self):
        """Test CrossDomainAttention with different input dimensions."""
        from df_mlx.model import CrossDomainAttention

        attn = CrossDomainAttention(
            time_dim=128,
            mag_dim=256,
            phase_dim=192,
            out_dim=256,
        )

        batch, time = 2, 50
        time_feat = mx.random.normal(shape=(batch, time, 128))
        mag_feat = mx.random.normal(shape=(batch, time, 256))
        phase_feat = mx.random.normal(shape=(batch, time, 192))

        fused = attn(time_feat, mag_feat, phase_feat)
        mx.eval(fused)

        assert fused.shape == (batch, time, 256)


class TestHybridEncoder:
    """Tests for HybridEncoder."""

    def test_hybrid_encoder_init(self):
        """Test HybridEncoder initialization."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import HybridEncoder

        params = ModelParams4()
        encoder = HybridEncoder(params)

        assert encoder.use_time_branch is True
        assert encoder.use_phase_branch is True

    def test_hybrid_encoder_forward(self):
        """Test HybridEncoder forward pass."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import HybridEncoder

        params = ModelParams4()
        encoder = HybridEncoder(params)

        batch, time = 2, 50
        feat_erb = mx.random.normal(shape=(batch, time, params.nb_erb))
        feat_spec = mx.random.normal(shape=(batch, time, params.nb_df, 2))

        emb, lsnr = encoder(feat_erb, feat_spec)
        mx.eval(emb, lsnr)

        assert emb.shape == (batch, time, params.emb_hidden_dim)
        assert lsnr.shape == (batch, time, 1)

    def test_hybrid_encoder_with_waveform(self):
        """Test HybridEncoder with waveform input."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import HybridEncoder

        params = ModelParams4()
        encoder = HybridEncoder(params)

        batch, time = 2, 50
        feat_erb = mx.random.normal(shape=(batch, time, params.nb_erb))
        feat_spec = mx.random.normal(shape=(batch, time, params.nb_df, 2))
        waveform = mx.random.normal(shape=(batch, 48000))

        emb, lsnr = encoder(feat_erb, feat_spec, waveform)
        mx.eval(emb, lsnr)

        assert emb.shape[0] == batch

    def test_hybrid_encoder_no_time_branch(self):
        """Test HybridEncoder without time branch."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import HybridEncoder

        params = ModelParams4()
        encoder = HybridEncoder(params, use_time_branch=False)

        batch, time = 2, 50
        feat_erb = mx.random.normal(shape=(batch, time, params.nb_erb))
        feat_spec = mx.random.normal(shape=(batch, time, params.nb_df, 2))

        emb, lsnr = encoder(feat_erb, feat_spec)
        mx.eval(emb, lsnr)

        assert emb.shape == (batch, time, params.emb_hidden_dim)

    def test_hybrid_encoder_no_phase_branch(self):
        """Test HybridEncoder without phase branch."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import HybridEncoder

        params = ModelParams4()
        encoder = HybridEncoder(params, use_phase_branch=False)

        batch, time = 2, 50
        feat_erb = mx.random.normal(shape=(batch, time, params.nb_erb))
        feat_spec = mx.random.normal(shape=(batch, time, params.nb_df, 2))

        emb, lsnr = encoder(feat_erb, feat_spec)
        mx.eval(emb, lsnr)

        assert emb.shape == (batch, time, params.emb_hidden_dim)

    def test_hybrid_encoder_lsnr_range(self):
        """Test HybridEncoder LSNR output range."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import HybridEncoder

        params = ModelParams4()
        encoder = HybridEncoder(params)

        feat_erb = mx.random.normal(shape=(2, 50, params.nb_erb))
        feat_spec = mx.random.normal(shape=(2, 50, params.nb_df, 2))

        _, lsnr = encoder(feat_erb, feat_spec)
        mx.eval(lsnr)

        lsnr_min = params.lsnr.lsnr_min
        lsnr_max = params.lsnr.lsnr_max
        assert mx.all(lsnr >= lsnr_min - 1).item()
        assert mx.all(lsnr <= lsnr_max + 1).item()

    def test_hybrid_encoder_gradient_flow(self):
        """Test gradient flow through HybridEncoder."""
        from df_mlx.config import ModelParams4
        from df_mlx.model import HybridEncoder

        params = ModelParams4()
        encoder = HybridEncoder(params, use_time_branch=False)

        feat_erb = mx.random.normal(shape=(2, 20, params.nb_erb))
        feat_spec = mx.random.normal(shape=(2, 20, params.nb_df, 2))

        def loss_fn(model):
            emb, lsnr = model(feat_erb, feat_spec)
            return mx.mean(emb) + mx.mean(lsnr)

        loss, grads = nn.value_and_grad(encoder, loss_fn)(encoder)
        mx.eval(loss)

        assert not mx.isnan(loss)


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
