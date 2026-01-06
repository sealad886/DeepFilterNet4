"""Tests for Mamba state space model modules."""

import pytest
import torch
import torch.nn as nn

from df.mamba import BidirectionalMamba, Mamba, MambaBlock, SqueezedMamba


class TestMambaBlock:
    """Test suite for MambaBlock class."""

    @pytest.fixture
    def mamba_block(self):
        """Create a MambaBlock instance for testing."""
        return MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2)

    @pytest.fixture
    def small_mamba_block(self):
        """Create a small MambaBlock for faster tests."""
        return MambaBlock(d_model=32, d_state=8, d_conv=2, expand=1)

    def test_output_shape(self, mamba_block):
        """Verify output shape matches input shape."""
        x = torch.randn(2, 100, 64)
        y = mamba_block(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_output_shape_various_batch_sizes(self, small_mamba_block):
        """Test with various batch sizes including 1."""
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 50, 32)
            y = small_mamba_block(x)
            assert y.shape == x.shape

    def test_output_shape_various_lengths(self, small_mamba_block):
        """Test with various sequence lengths."""
        for length in [1, 10, 50, 100, 200]:
            x = torch.randn(2, length, 32)
            y = small_mamba_block(x)
            assert y.shape == x.shape

    def test_causality(self, mamba_block):
        """Verify output at time t doesn't depend on input at time > t."""
        torch.manual_seed(42)
        x = torch.randn(1, 50, 64)
        y1 = mamba_block(x)

        # Modify future input
        x2 = x.clone()
        x2[:, 25:, :] = torch.randn(1, 25, 64)
        y2 = mamba_block(x2)

        # First 25 timesteps should be identical
        assert torch.allclose(
            y1[:, :25], y2[:, :25], atol=1e-5
        ), "Causality violated: output depends on future inputs"

    def test_batch_independence(self, mamba_block):
        """Verify batches are processed independently."""
        torch.manual_seed(42)
        x1 = torch.randn(1, 50, 64)
        x2 = torch.randn(1, 50, 64)

        y_separate = torch.cat([mamba_block(x1), mamba_block(x2)], dim=0)
        y_batched = mamba_block(torch.cat([x1, x2], dim=0))

        assert torch.allclose(
            y_separate, y_batched, atol=1e-5
        ), "Batch processing produces different results than separate processing"

    def test_gradient_flow(self, mamba_block):
        """Verify gradients flow through the block."""
        x = torch.randn(2, 50, 64, requires_grad=True)
        y = mamba_block(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "No gradients computed for input"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"
        assert not torch.isinf(x.grad).any(), "Inf in gradients"

    def test_deterministic(self, mamba_block):
        """Verify deterministic output in eval mode."""
        mamba_block.eval()
        x = torch.randn(2, 50, 64)

        y1 = mamba_block(x)
        y2 = mamba_block(x)

        assert torch.allclose(y1, y2), "Non-deterministic output in eval mode"

    def test_different_configurations(self):
        """Test various Mamba configurations."""
        configs = [
            {"d_model": 32, "d_state": 8, "d_conv": 2, "expand": 1},
            {"d_model": 64, "d_state": 16, "d_conv": 4, "expand": 2},
            {"d_model": 128, "d_state": 32, "d_conv": 4, "expand": 2},
            {"d_model": 64, "d_state": 16, "d_conv": 8, "expand": 1},
        ]

        for config in configs:
            block = MambaBlock(**config)
            x = torch.randn(2, 20, config["d_model"])
            y = block(x)
            assert y.shape == x.shape, f"Failed for config {config}"


class TestMamba:
    """Test suite for Mamba layer with residual and norm."""

    @pytest.fixture
    def mamba_layer(self):
        """Create a Mamba layer for testing."""
        return Mamba(d_model=64, d_state=16, d_conv=4, expand=2)

    def test_output_shape(self, mamba_layer):
        """Verify output shape matches input shape."""
        x = torch.randn(2, 100, 64)
        y = mamba_layer(x)
        assert y.shape == x.shape

    def test_residual_connection(self, mamba_layer):
        """Verify residual connection is working."""
        x = torch.randn(2, 50, 64)
        y = mamba_layer(x)

        # Output should not be identical to input (transformation applied)
        assert not torch.allclose(
            x, y, atol=1e-3
        ), "Output identical to input - residual might be all zeros"

    def test_stacking(self):
        """Test stacking multiple Mamba layers."""
        layers = nn.Sequential(*[Mamba(64) for _ in range(3)])
        x = torch.randn(2, 50, 64)
        y = layers(x)
        assert y.shape == x.shape


class TestSqueezedMamba:
    """Test suite for SqueezedMamba - GRU drop-in replacement."""

    @pytest.fixture
    def squeezed_mamba(self):
        """Create a SqueezedMamba instance."""
        return SqueezedMamba(
            input_size=128,
            hidden_size=256,
            output_size=128,
            num_layers=2,
        )

    def test_gru_api_compatibility(self, squeezed_mamba):
        """Verify API matches SqueezedGRU_S."""
        x = torch.randn(4, 100, 128)
        output, h_n = squeezed_mamba(x)

        assert output.shape == (4, 100, 128), f"Expected (4, 100, 128), got {output.shape}"
        assert h_n.shape[0] == 1, "Hidden state should have 1 in first dimension"

    def test_hidden_state_compatibility(self, squeezed_mamba):
        """Test that hidden state input is accepted (for API compat)."""
        x = torch.randn(4, 100, 128)
        h = torch.randn(1, 4, 256)  # Dummy hidden state

        # Should not raise an error
        output, h_n = squeezed_mamba(x, h)
        assert output.shape == (4, 100, 128)

    def test_skip_connection(self):
        """Test with skip connection."""
        mamba = SqueezedMamba(
            input_size=64,
            hidden_size=64,
            gru_skip_op=lambda: nn.Identity(),
        )

        x = torch.randn(2, 50, 64)
        output, _ = mamba(x)
        assert output.shape == (2, 50, 64)

    def test_batch_first_false(self):
        """Test with batch_first=False."""
        mamba = SqueezedMamba(
            input_size=64,
            hidden_size=128,
            output_size=64,
            batch_first=False,
        )

        x = torch.randn(50, 4, 64)  # [T, B, F]
        output, _ = mamba(x)
        assert output.shape == (50, 4, 64)

    def test_linear_groups(self):
        """Test with grouped linear projections."""
        mamba = SqueezedMamba(
            input_size=64,
            hidden_size=64,
            output_size=64,
            linear_groups=8,
        )

        x = torch.randn(2, 50, 64)
        output, _ = mamba(x)
        assert output.shape == (2, 50, 64)

    def test_no_output_size(self):
        """Test when output_size is not specified."""
        mamba = SqueezedMamba(
            input_size=64,
            hidden_size=128,
        )

        x = torch.randn(2, 50, 64)
        output, _ = mamba(x)
        # Should output hidden_size when output_size not specified
        assert output.shape == (2, 50, 128)

    def test_gradient_flow(self, squeezed_mamba):
        """Verify gradients flow through SqueezedMamba."""
        x = torch.randn(2, 50, 128, requires_grad=True)
        output, _ = squeezed_mamba(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestBidirectionalMamba:
    """Test suite for BidirectionalMamba."""

    def test_concat_merge(self):
        """Test concatenation merge mode."""
        bi_mamba = BidirectionalMamba(d_model=64, merge="concat")
        x = torch.randn(2, 50, 64)
        y = bi_mamba(x)
        assert y.shape == (2, 50, 128)  # Doubled due to concat

    def test_add_merge(self):
        """Test addition merge mode."""
        bi_mamba = BidirectionalMamba(d_model=64, merge="add")
        x = torch.randn(2, 50, 64)
        y = bi_mamba(x)
        assert y.shape == (2, 50, 64)  # Same dimension

    def test_proj_merge(self):
        """Test projection merge mode."""
        bi_mamba = BidirectionalMamba(d_model=64, merge="proj")
        x = torch.randn(2, 50, 64)
        y = bi_mamba(x)
        assert y.shape == (2, 50, 64)  # Projected back to d_model


class TestMPSCompatibility:
    """Test MPS (Apple Silicon) compatibility if available."""

    @pytest.fixture
    def device(self):
        """Get MPS device if available, skip otherwise."""
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available")
        return torch.device("mps")

    def test_mamba_block_on_mps(self, device):
        """Test MambaBlock runs on MPS."""
        block = MambaBlock(d_model=64).to(device)
        x = torch.randn(2, 50, 64, device=device)
        y = block(x)
        assert y.device.type == "mps"
        assert y.shape == x.shape

    def test_squeezed_mamba_on_mps(self, device):
        """Test SqueezedMamba runs on MPS."""
        mamba = SqueezedMamba(input_size=64, hidden_size=128, output_size=64).to(device)
        x = torch.randn(2, 50, 64, device=device)
        output, _ = mamba(x)
        assert output.device.type == "mps"
        assert output.shape == (2, 50, 64)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_timestep(self):
        """Test with single timestep input."""
        block = MambaBlock(d_model=64)
        x = torch.randn(2, 1, 64)
        y = block(x)
        assert y.shape == x.shape

    def test_very_long_sequence(self):
        """Test with long sequence (memory efficiency)."""
        block = MambaBlock(d_model=32, d_state=8, expand=1)
        x = torch.randn(1, 1000, 32)
        y = block(x)
        assert y.shape == x.shape

    def test_empty_batch(self):
        """Test behavior with empty batch."""
        block = MambaBlock(d_model=64)
        x = torch.randn(0, 50, 64)
        y = block(x)
        assert y.shape == (0, 50, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
