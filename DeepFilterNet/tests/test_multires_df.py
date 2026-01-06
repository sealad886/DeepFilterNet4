"""Tests for Multi-Resolution Deep Filtering components.

Tests the MultiResolutionDF, AdaptiveOrderPredictor, and related decoder
components introduced in DeepFilterNet4 Phase 3.
"""

import pytest
import torch

from df.deepfilternet4 import AdaptiveDfDecoder, DfOutputReshape, ModelParams4, MultiResDfDecoder, SingleResDfDecoder
from df.multiframe import DF, AdaptiveOrderPredictor, DFreal, MultiResolutionDF

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 100


@pytest.fixture
def num_freqs():
    return 96


@pytest.fixture
def default_resolutions():
    """Default multi-resolution configuration."""
    return [(96, 5), (48, 3), (24, 2)]


@pytest.fixture
def spec_tensor(batch_size, seq_len, num_freqs):
    """Complex spectrogram tensor [B, C, T, F, 2]."""
    return torch.randn(batch_size, 1, seq_len, num_freqs, 2)


@pytest.fixture
def embedding_tensor(batch_size, seq_len):
    """Embedding tensor [B, T, emb_in_dim].

    For decoders, emb_in_dim = conv_ch * nb_erb // 4 = 16 * 32 // 4 = 128.
    """
    return torch.randn(batch_size, seq_len, 128)


@pytest.fixture
def c0_tensor(batch_size, seq_len):
    """DF pathway features [B, C, T, F]."""
    return torch.randn(batch_size, 16, seq_len, 48)


# ============================================================================
# Test MultiResolutionDF
# ============================================================================


class TestMultiResolutionDF:
    """Tests for MultiResolutionDF module."""

    def test_init_default(self):
        """Test default initialization."""
        mr_df = MultiResolutionDF()

        assert len(mr_df.df_ops) == 3
        assert len(mr_df.resolutions) == 3
        assert mr_df.resolutions == [(96, 5), (48, 3), (24, 2)]
        assert mr_df.resolution_weights.shape == (3,)

    def test_init_custom_resolutions(self):
        """Test custom resolution configuration."""
        resolutions = [(64, 7), (32, 5)]
        mr_df = MultiResolutionDF(resolutions=resolutions)

        assert len(mr_df.df_ops) == 2
        assert mr_df.resolutions == resolutions

    def test_init_fixed_weights(self):
        """Test initialization with fixed (non-learnable) weights."""
        mr_df = MultiResolutionDF(learnable_weights=False)

        assert not mr_df.resolution_weights.requires_grad

    def test_init_real_mode(self):
        """Test initialization with real-valued DF operations."""
        mr_df = MultiResolutionDF(use_real=True)

        for df_op in mr_df.df_ops:
            assert isinstance(df_op, DFreal)

    def test_forward_shape(self, spec_tensor, default_resolutions):
        """Test forward pass output shape."""
        mr_df = MultiResolutionDF(resolutions=default_resolutions)

        # Generate coefficients for each resolution in [B, O, T, F, 2] format
        coefs_list = []
        for num_freqs, frame_size in default_resolutions:
            # [B, O, T, F, 2] where O is frame_size
            coefs = torch.randn(spec_tensor.shape[0], frame_size, spec_tensor.shape[2], num_freqs, 2)
            coefs_list.append(coefs)

        output = mr_df(spec_tensor, coefs_list)

        assert output.shape == spec_tensor.shape

    def test_forward_weights_sum_to_one(self, spec_tensor, default_resolutions):
        """Test that resolution weights sum to 1 after softmax."""
        mr_df = MultiResolutionDF(resolutions=default_resolutions)

        weights = mr_df.get_resolution_weights()

        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_forward_coefficient_mismatch_error(self, spec_tensor, default_resolutions):
        """Test that mismatched coefficient count raises error."""
        mr_df = MultiResolutionDF(resolutions=default_resolutions)

        # Provide wrong number of coefficients
        coefs_list = [torch.randn(2, 1, 100, 96, 10)]  # Only 1 instead of 3

        with pytest.raises(ValueError):
            mr_df(spec_tensor, coefs_list)

    def test_gradient_flow(self, spec_tensor, default_resolutions):
        """Test that gradients flow through the module."""
        mr_df = MultiResolutionDF(resolutions=default_resolutions)

        coefs_list = []
        for num_freqs, frame_size in default_resolutions:
            # [B, O, T, F, 2] format
            coefs = torch.randn(
                spec_tensor.shape[0],
                frame_size,
                spec_tensor.shape[2],
                num_freqs,
                2,
                requires_grad=True,
            )
            coefs_list.append(coefs)

        output = mr_df(spec_tensor, coefs_list)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert mr_df.resolution_weights.grad is not None
        for coefs in coefs_list:
            assert coefs.grad is not None


# ============================================================================
# Test AdaptiveOrderPredictor
# ============================================================================


class TestAdaptiveOrderPredictor:
    """Tests for AdaptiveOrderPredictor module."""

    @pytest.fixture
    def predictor_embedding(self, batch_size, seq_len):
        """Embedding tensor with dimension matching predictor default (256)."""
        return torch.randn(batch_size, seq_len, 256)

    def test_init_default(self):
        """Test default initialization."""
        predictor = AdaptiveOrderPredictor(emb_dim=256)

        assert predictor.max_order == 7
        assert predictor.min_order == 2
        assert predictor.num_orders == 6

    def test_init_custom_orders(self):
        """Test custom order range."""
        predictor = AdaptiveOrderPredictor(emb_dim=128, max_order=10, min_order=3)

        assert predictor.num_orders == 8

    def test_init_invalid_orders(self):
        """Test that invalid order range raises error."""
        with pytest.raises(ValueError):
            AdaptiveOrderPredictor(emb_dim=256, max_order=2, min_order=5)

    def test_forward_shape_training(self, predictor_embedding):
        """Test forward output shapes during training."""
        predictor = AdaptiveOrderPredictor(emb_dim=256)
        predictor.train()

        order_weights, predicted_order = predictor(predictor_embedding)

        # order_weights: [B, T, num_orders]
        assert order_weights.shape == (
            predictor_embedding.shape[0],
            predictor_embedding.shape[1],
            predictor.num_orders,
        )
        # predicted_order: [B, T]
        assert predicted_order.shape == (
            predictor_embedding.shape[0],
            predictor_embedding.shape[1],
        )

    def test_forward_shape_eval(self, predictor_embedding):
        """Test forward output shapes during evaluation."""
        predictor = AdaptiveOrderPredictor(emb_dim=256)
        predictor.eval()

        order_weights, predicted_order = predictor(predictor_embedding)

        assert order_weights.shape == (
            predictor_embedding.shape[0],
            predictor_embedding.shape[1],
            predictor.num_orders,
        )

    def test_eval_produces_one_hot(self, predictor_embedding):
        """Test that eval mode produces one-hot weights."""
        predictor = AdaptiveOrderPredictor(emb_dim=256)
        predictor.eval()

        order_weights, _ = predictor(predictor_embedding)

        # Each weight should be 0 or 1
        assert torch.allclose(order_weights.sum(dim=-1), torch.ones_like(order_weights.sum(dim=-1)))
        # Should be one-hot (max should be 1)
        assert order_weights.max() == 1.0

    def test_predicted_order_range(self, predictor_embedding):
        """Test that predicted orders are in valid range."""
        predictor = AdaptiveOrderPredictor(emb_dim=256, max_order=7, min_order=2)

        _, predicted_order = predictor(predictor_embedding)

        assert predicted_order.min() >= 2
        assert predicted_order.max() <= 7

    def test_temperature_effect(self, predictor_embedding):
        """Test that temperature affects softness of selection."""
        predictor = AdaptiveOrderPredictor(emb_dim=256)
        predictor.train()

        # High temperature -> softer distribution
        weights_high, _ = predictor(predictor_embedding, temperature=10.0)
        # Low temperature -> sharper distribution
        weights_low, _ = predictor(predictor_embedding, temperature=0.1)

        # Low temperature should have higher max values (sharper)
        assert weights_low.max() > weights_high.max()

    def test_gradient_flow(self, predictor_embedding):
        """Test gradient flow through predictor."""
        predictor = AdaptiveOrderPredictor(emb_dim=256)
        predictor.train()

        emb = predictor_embedding.requires_grad_(True)
        order_weights, _ = predictor(emb)
        loss = order_weights.sum()
        loss.backward()

        assert emb.grad is not None

    def test_get_order_distribution(self, predictor_embedding):
        """Test order distribution method."""
        predictor = AdaptiveOrderPredictor(emb_dim=256)

        probs = predictor.get_order_distribution(predictor_embedding)

        # Should be valid probability distribution
        assert probs.shape[-1] == predictor.num_orders
        assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))
        assert (probs >= 0).all()


# ============================================================================
# Test DfOutputReshape
# ============================================================================


class TestDfOutputReshape:
    """Tests for DfOutputReshape module."""

    def test_reshape_correctness(self):
        """Test output shape is correct."""
        reshape = DfOutputReshape(df_order=5, df_bins=96)

        # Input: [B, T, F, O*2]
        x = torch.randn(2, 100, 96, 10)  # O*2 = 5*2 = 10

        # Output: [B, O, T, F, 2]
        y = reshape(x)

        assert y.shape == (2, 5, 100, 96, 2)

    def test_reshape_different_orders(self):
        """Test reshape with different filter orders."""
        for order in [2, 3, 5, 7]:
            reshape = DfOutputReshape(df_order=order, df_bins=48)
            x = torch.randn(2, 50, 48, order * 2)
            y = reshape(x)
            assert y.shape == (2, order, 50, 48, 2)


# ============================================================================
# Test MultiResDfDecoder
# ============================================================================


class TestMultiResDfDecoder:
    """Tests for MultiResDfDecoder module."""

    def test_init_default(self):
        """Test default initialization."""
        decoder = MultiResDfDecoder()

        assert len(decoder.output_heads) == 3
        assert len(decoder.reshape_ops) == 3

    def test_init_with_mamba(self):
        """Test initialization with Mamba backbone."""
        decoder = MultiResDfDecoder(use_mamba=True)

        from df.mamba import SqueezedMamba

        assert isinstance(decoder.backbone, SqueezedMamba)

    def test_init_with_gru(self):
        """Test initialization with GRU backbone."""
        decoder = MultiResDfDecoder(use_mamba=False)

        from df.modules import SqueezedGRU_S

        assert isinstance(decoder.backbone, SqueezedGRU_S)

    def test_forward_output_shapes(self, embedding_tensor, c0_tensor, default_resolutions):
        """Test forward pass output shapes."""
        decoder = MultiResDfDecoder(resolutions=default_resolutions)

        coefs_list = decoder(embedding_tensor, c0_tensor)

        assert len(coefs_list) == 3

        # Check each output shape
        for coefs, (num_freqs, frame_size) in zip(coefs_list, default_resolutions):
            assert coefs.shape == (
                c0_tensor.shape[0],  # B
                frame_size,  # O
                c0_tensor.shape[2],  # T
                num_freqs,  # F
                2,  # complex
            )

    def test_gradient_flow(self, embedding_tensor, c0_tensor):
        """Test gradient flow through decoder."""
        decoder = MultiResDfDecoder()

        emb = embedding_tensor.requires_grad_(True)

        coefs_list = decoder(emb, c0_tensor)
        loss = sum(c.sum() for c in coefs_list)
        loss.backward()

        assert emb.grad is not None
        # Note: c0 is not used in current implementation


# ============================================================================
# Test SingleResDfDecoder
# ============================================================================


class TestSingleResDfDecoder:
    """Tests for SingleResDfDecoder module."""

    def test_init_default(self):
        """Test default initialization."""
        decoder = SingleResDfDecoder()

        assert decoder.df_order == 5
        assert decoder.df_bins == 96

    def test_forward_output_shape(self, embedding_tensor, c0_tensor):
        """Test forward pass output shape."""
        decoder = SingleResDfDecoder(df_order=5, df_bins=96)

        coefs = decoder(embedding_tensor, c0_tensor)

        # [B, O, T, F, 2]
        assert coefs.shape == (
            c0_tensor.shape[0],  # B
            5,  # O
            c0_tensor.shape[2],  # T
            96,  # F
            2,  # complex
        )

    def test_backward_compatibility_shape(self, embedding_tensor, c0_tensor):
        """Test that output is compatible with standard DF operations."""
        decoder = SingleResDfDecoder(df_order=5, df_bins=96)

        coefs = decoder(embedding_tensor, c0_tensor)

        # Should be usable with MF.DF
        df_op = DF(num_freqs=96, frame_size=5)
        spec = torch.randn(c0_tensor.shape[0], 1, c0_tensor.shape[2], 96, 2)

        # This should not raise
        _ = df_op(spec, coefs)


# ============================================================================
# Test AdaptiveDfDecoder
# ============================================================================


class TestAdaptiveDfDecoder:
    """Tests for AdaptiveDfDecoder module."""

    def test_init_default(self):
        """Test default initialization."""
        decoder = AdaptiveDfDecoder()

        assert decoder.max_order == 7
        assert decoder.min_order == 2
        assert decoder.num_orders == 6

    def test_forward_output_shapes(self, embedding_tensor, c0_tensor):
        """Test forward pass output shapes."""
        decoder = AdaptiveDfDecoder(max_order=7, min_order=2)

        coefs, order_weights, predicted_order = decoder(embedding_tensor, c0_tensor)

        # coefs: [B, max_order, T, F, 2]
        assert coefs.shape == (c0_tensor.shape[0], 7, c0_tensor.shape[2], decoder.df_bins, 2)

        # order_weights: [B, T, num_orders]
        assert order_weights.shape == (c0_tensor.shape[0], c0_tensor.shape[2], decoder.num_orders)

        # predicted_order: [B, T]
        assert predicted_order.shape == (c0_tensor.shape[0], c0_tensor.shape[2])

    def test_temperature_passed_to_predictor(self, embedding_tensor, c0_tensor):
        """Test that temperature affects order selection."""
        decoder = AdaptiveDfDecoder()
        decoder.train()

        # Different temperatures should produce different distributions
        _, weights_high, _ = decoder(embedding_tensor, c0_tensor, temperature=10.0)
        _, weights_low, _ = decoder(embedding_tensor, c0_tensor, temperature=0.1)

        # Low temperature should have sharper weights
        assert weights_low.max() > weights_high.max()

    def test_gradient_flow(self, embedding_tensor, c0_tensor):
        """Test gradient flow through adaptive decoder."""
        decoder = AdaptiveDfDecoder()
        decoder.train()

        emb = embedding_tensor.requires_grad_(True)

        coefs, _, _ = decoder(emb, c0_tensor)
        loss = coefs.sum()
        loss.backward()

        assert emb.grad is not None
        # Note: c0 is not used in current implementation


# ============================================================================
# Test ModelParams4
# ============================================================================


class TestModelParams4:
    """Tests for ModelParams4 configuration."""

    def test_default_values(self):
        """Test default parameter values."""
        # Skip if config not available
        try:
            p = ModelParams4()
        except Exception:
            pytest.skip("Config not initialized")

        assert p.backbone in ("mamba", "gru")
        assert p.mamba_d_state == 16
        assert p.mamba_d_conv == 4

    def test_get_df_resolutions(self):
        """Test DF resolution parsing."""
        try:
            p = ModelParams4()
        except Exception:
            pytest.skip("Config not initialized")

        resolutions = p.get_df_resolutions()

        assert isinstance(resolutions, list)
        assert len(resolutions) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in resolutions)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for multi-resolution DF components."""

    def test_multires_df_with_decoder(self, spec_tensor, embedding_tensor, c0_tensor, default_resolutions):
        """Test MultiResolutionDF with MultiResDfDecoder."""
        decoder = MultiResDfDecoder(resolutions=default_resolutions)
        mr_df = MultiResolutionDF(resolutions=default_resolutions)

        # Generate coefficients
        coefs_list = decoder(embedding_tensor, c0_tensor)

        # Apply multi-resolution filtering
        enhanced = mr_df(spec_tensor, coefs_list)

        assert enhanced.shape == spec_tensor.shape

    def test_adaptive_order_end_to_end(self, spec_tensor, embedding_tensor, c0_tensor):
        """Test AdaptiveDfDecoder with DF operation."""
        decoder = AdaptiveDfDecoder(max_order=7, min_order=2)

        coefs, order_weights, predicted_order = decoder(embedding_tensor, c0_tensor)

        # Use maximum order DF for the combined coefficients
        df_op = DF(num_freqs=96, frame_size=7)
        enhanced = df_op(spec_tensor, coefs)

        assert enhanced.shape == spec_tensor.shape


# ============================================================================
# Device Tests
# ============================================================================


class TestDevices:
    """Tests for device compatibility."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multires_df_cuda(self, spec_tensor, default_resolutions):
        """Test MultiResolutionDF on CUDA."""
        mr_df = MultiResolutionDF(resolutions=default_resolutions).cuda()
        spec = spec_tensor.cuda()

        coefs_list = []
        for num_freqs, frame_size in default_resolutions:
            # [B, O, T, F, 2] format
            coefs = torch.randn(spec.shape[0], frame_size, spec.shape[2], num_freqs, 2, device="cuda")
            coefs_list.append(coefs)

        output = mr_df(spec, coefs_list)

        assert output.device.type == "cuda"

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_multires_df_mps(self, spec_tensor, default_resolutions):
        """Test MultiResolutionDF on MPS."""
        mr_df = MultiResolutionDF(resolutions=default_resolutions).to("mps")
        spec = spec_tensor.to("mps")

        coefs_list = []
        for num_freqs, frame_size in default_resolutions:
            # [B, O, T, F, 2] format
            coefs = torch.randn(spec.shape[0], frame_size, spec.shape[2], num_freqs, 2, device="mps")
            coefs_list.append(coefs)

        output = mr_df(spec, coefs_list)

        assert output.device.type == "mps"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_resolution(self):
        """Test with only one resolution."""
        mr_df = MultiResolutionDF(resolutions=[(96, 5)])

        spec = torch.randn(2, 1, 50, 96, 2)
        # [B, O, T, F, 2] format where O=5
        coefs_list = [torch.randn(2, 5, 50, 96, 2)]

        output = mr_df(spec, coefs_list)

        assert output.shape == spec.shape

    def test_order_predictor_single_order(self):
        """Test predictor with single order option."""
        predictor = AdaptiveOrderPredictor(emb_dim=256, max_order=5, min_order=5)

        emb = torch.randn(2, 100, 256)
        order_weights, predicted_order = predictor(emb)

        # With only one option, should always predict that order
        assert (predicted_order == 5).all()

    def test_batch_size_one(self, default_resolutions):
        """Test with batch size of 1."""
        mr_df = MultiResolutionDF(resolutions=default_resolutions)

        spec = torch.randn(1, 1, 50, 96, 2)
        coefs_list = []
        for num_freqs, frame_size in default_resolutions:
            # [B, O, T, F, 2] format
            coefs = torch.randn(1, frame_size, 50, num_freqs, 2)
            coefs_list.append(coefs)

        output = mr_df(spec, coefs_list)

        assert output.shape == spec.shape

    def test_very_short_sequence(self, default_resolutions):
        """Test with very short sequence."""
        mr_df = MultiResolutionDF(resolutions=default_resolutions)

        spec = torch.randn(2, 1, 5, 96, 2)  # Only 5 time steps
        coefs_list = []
        for num_freqs, frame_size in default_resolutions:
            # [B, O, T, F, 2] format
            coefs = torch.randn(2, frame_size, 5, num_freqs, 2)
            coefs_list.append(coefs)

        output = mr_df(spec, coefs_list)

        assert output.shape == spec.shape
