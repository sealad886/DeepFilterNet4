#!/usr/bin/env python3
"""Profile attention-based backbone vs GRU/Mamba.

This tests whether causal self-attention could be faster than RNN-based
backbones for the backward pass.
"""

import sys
import time
from pathlib import Path

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))


class CausalSelfAttention(nn.Module):
    """Causal self-attention block for sequence modeling.

    Uses MLX's optimized attention implementation which should have
    much faster backward pass than sequential RNN operations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiHeadAttention(
            dims=hidden_dim,
            num_heads=num_heads,
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with causal masking.

        Args:
            x: Input (batch, time, input_dim)

        Returns:
            Output (batch, time, hidden_dim)
        """
        # Project to hidden dim
        h = self.input_proj(x)

        # Create causal mask
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Self-attention with residual
        attn_out = self.attention(h, h, h, mask=mask)
        h = self.norm1(h + attn_out)

        # FFN with residual
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)

        return h


class AttentionBackbone(nn.Module):
    """Multi-layer attention backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()

        self.layers = [
            CausalSelfAttention(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                num_heads=num_heads,
            )
            for i in range(num_layers)
        ]
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)


def profile_backbone(name: str, model: nn.Module, x: mx.array, n_iter: int = 5):
    """Profile forward and backward pass of a backbone."""
    print(f"\n{'=' * 50}")
    print(f"{name}")
    print("=" * 50)

    mx.eval(model.parameters())

    # Warmup
    for _ in range(2):
        y = model(x)
        mx.eval(y)

    # Forward
    times = []
    for _ in range(n_iter):
        start = time.time()
        y = model(x)
        mx.eval(y)
        times.append((time.time() - start) * 1000)
    fwd_time = sum(times) / len(times)
    print(f"  Forward:  {fwd_time:.1f}ms")

    # Backward
    def loss_fn(model, x):
        return model(x).sum()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Warmup backward
    for _ in range(2):
        loss, grads = loss_and_grad(model, x)
        mx.eval(loss, grads)

    # Time backward
    times = []
    for _ in range(n_iter):
        start = time.time()
        loss, grads = loss_and_grad(model, x)
        mx.eval(loss, grads)
        times.append((time.time() - start) * 1000)
    fwd_bwd_time = sum(times) / len(times)
    bwd_time = fwd_bwd_time - fwd_time
    print(f"  Forward + backward: {fwd_bwd_time:.1f}ms")
    print(f"  Backward only: {bwd_time:.1f}ms")
    print(f"  Ratio (bwd/fwd): {bwd_time / fwd_time:.1f}x")

    return fwd_time, bwd_time


def main():
    print("=" * 60)
    print("ATTENTION vs GRU BACKBONE COMPARISON")
    print("=" * 60)

    batch_size = 8
    seq_len = 500
    input_dim = 256
    hidden_dim = 256

    # Create input
    x = mx.random.normal((batch_size, seq_len, input_dim))
    mx.eval(x)

    print(f"\nBatch: {batch_size}, Seq len: {seq_len}, Dim: {input_dim}")

    # Test GRU
    gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim)
    fwd_gru, bwd_gru = profile_backbone("MLX nn.GRU", gru, x)

    # Test Attention (1 layer)
    attn1 = CausalSelfAttention(input_dim, hidden_dim, num_heads=4)
    fwd_attn1, bwd_attn1 = profile_backbone("Attention (1 layer, 4 heads)", attn1, x)

    # Test Attention (2 layers)
    attn2 = AttentionBackbone(input_dim, hidden_dim, num_layers=2, num_heads=4)
    fwd_attn2, bwd_attn2 = profile_backbone("Attention (2 layers, 4 heads)", attn2, x)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"GRU:           Forward {fwd_gru:.0f}ms, Backward {bwd_gru:.0f}ms")
    print(f"Attention (1): Forward {fwd_attn1:.0f}ms, Backward {bwd_attn1:.0f}ms")
    print(f"Attention (2): Forward {fwd_attn2:.0f}ms, Backward {bwd_attn2:.0f}ms")

    if bwd_gru > 0:
        print(f"\nAttention (1) backward is {bwd_gru / bwd_attn1:.1f}x faster than GRU")
        print(f"Attention (2) backward is {bwd_gru / bwd_attn2:.1f}x faster than GRU")


if __name__ == "__main__":
    main()
