"""Mamba State Space Model implementation for MLX.

This module provides MLX-native implementations of the Mamba architecture
used in DeepFilterNet4 for efficient sequence modeling with linear complexity.

The key innovation of Mamba is the selective scan mechanism which allows
input-dependent state transitions, enabling the model to filter relevant
information from the input sequence.

Architecture:
- MambaBlock: Core Mamba block with selective SSM
- Mamba: MambaBlock with LayerNorm and residual connection
- SqueezedMamba: Drop-in replacement for SqueezedGRU_S
- BidirectionalMamba: Processes sequences in both directions
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def flip_along_axis(x: mx.array, axis: int) -> mx.array:
    """Reverse an array along a given axis.

    MLX doesn't have a native flip function, so we use slice notation.

    Args:
        x: Input array
        axis: Axis along which to reverse

    Returns:
        Reversed array
    """
    # Build slice tuple with ::-1 at the specified axis
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(None, None, -1)
    return x[tuple(slices)]


class MambaBlock(nn.Module):
    """Core Mamba block implementing selective state space model.

    The selective scan mechanism allows the model to selectively propagate
    or forget information along the sequence based on the input content.

    Args:
        d_model: Input/output dimension
        d_state: State dimension (N in paper)
        d_conv: Local convolution width
        expand_factor: Expansion factor for inner dimension
        dt_rank: Rank of delta (time step) projection
        dt_min: Minimum delta value
        dt_max: Maximum delta value
        dt_init: Delta initialization method
        dt_scale: Delta scaling factor
        dt_init_floor: Minimum delta initialization
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = d_model * expand_factor

        if dt_rank is None:
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        # Input projection: projects to 2 * d_inner (for x and z paths)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Conv1d for local context (depthwise)
        # MLX doesn't have Conv1d, so we use Conv2d with height=1
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Depthwise
        )

        # SSM parameters
        # x_proj: projects x to (delta, B, C)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)

        # dt_proj: projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj bias for proper delta range
        if dt_init == "constant":
            self.dt_proj.bias = mx.ones((self.d_inner,)) * math.log(dt_min)
        elif dt_init == "random":
            # Uniform in [dt_min, dt_max]
            self.dt_proj.bias = mx.random.uniform(low=math.log(dt_min), high=math.log(dt_max), shape=(self.d_inner,))

        # A parameter (structured as log for stability)
        # A is (d_inner, d_state) - diagonal state matrix
        A = mx.repeat(mx.arange(1, d_state + 1, dtype=mx.float32)[None, :], self.d_inner, axis=0)
        self.A_log = mx.log(A)  # Store as log for stability

        # D parameter (skip connection)
        self.D = mx.ones((self.d_inner,))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def __call__(self, x: mx.array, state: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """Forward pass of Mamba block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            state: Optional initial state of shape (batch, d_inner, d_state)

        Returns:
            Tuple of (output, final_state)
        """
        batch, seq_len, _ = x.shape

        # Input projection
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x_path, z = mx.split(xz, 2, axis=-1)  # Each (batch, seq_len, d_inner)

        # Convolution path
        # MLX Conv1d expects (batch, seq_len, channels) - channels last
        x_path = self.conv1d(x_path)
        # Truncate to original length (causal padding)
        x_path = x_path[:, :seq_len, :]

        # Activation
        x_path = nn.silu(x_path)

        # SSM projection
        x_dbl = self.x_proj(x_path)  # (batch, seq_len, dt_rank + 2*d_state)

        # Split into delta, B, C
        delta = x_dbl[:, :, : self.dt_rank]
        B = x_dbl[:, :, self.dt_rank : self.dt_rank + self.d_state]
        C = x_dbl[:, :, self.dt_rank + self.d_state :]

        # Project and softplus delta
        delta = self.dt_proj(delta)  # (batch, seq_len, d_inner)
        delta = nn.softplus(delta)

        # Get A from log representation
        A = -mx.exp(self.A_log)  # (d_inner, d_state), negative for stability

        # Selective scan
        y, final_state = self._selective_scan(x_path, delta, A, B, C, self.D, state)

        # Gating with z path
        z = nn.silu(z)
        y = y * z

        # Output projection
        output = self.out_proj(y)

        return output, final_state

    def _selective_scan(
        self,
        u: mx.array,  # (batch, seq_len, d_inner)
        delta: mx.array,  # (batch, seq_len, d_inner)
        A: mx.array,  # (d_inner, d_state)
        B: mx.array,  # (batch, seq_len, d_state)
        C: mx.array,  # (batch, seq_len, d_state)
        D: mx.array,  # (d_inner,)
        state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Selective scan (S6) algorithm.

        This is the core of Mamba - a selective state space model that can
        filter information based on input content.

        The discretized state space equations are:
            h_t = A_bar * h_{t-1} + B_bar * x_t
            y_t = C * h_t + D * x_t

        Where A_bar, B_bar depend on delta (the input-dependent time step).

        Args:
            u: Input tensor (batch, seq_len, d_inner)
            delta: Time step tensor (batch, seq_len, d_inner)
            A: State transition matrix (d_inner, d_state)
            B: Input matrix (batch, seq_len, d_state)
            C: Output matrix (batch, seq_len, d_state)
            D: Skip connection (d_inner,)
            state: Optional initial state (batch, d_inner, d_state)

        Returns:
            Tuple of (output, final_state)
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        # Initialize state if not provided
        if state is None:
            state = mx.zeros((batch, d_inner, d_state))

        # Discretize A and B
        # A_bar = exp(delta * A) for each position
        # delta: (batch, seq_len, d_inner)
        # A: (d_inner, d_state)
        # A_bar: (batch, seq_len, d_inner, d_state)
        deltaA = mx.exp(delta[:, :, :, None] * A[None, None, :, :])

        # B_bar = delta * B (simplified Euler discretization)
        # B: (batch, seq_len, d_state)
        # B_bar: (batch, seq_len, d_inner, d_state)
        deltaB_u = delta[:, :, :, None] * B[:, :, None, :] * u[:, :, :, None]

        # Sequential scan (this is the bottleneck - could be optimized)
        outputs = []
        h = state

        for t in range(seq_len):
            # State update: h = A_bar * h + B_bar * u
            h = deltaA[:, t, :, :] * h + deltaB_u[:, t, :, :]

            # Output: y = C * h
            # C: (batch, d_state) at time t
            # h: (batch, d_inner, d_state)
            y_t = mx.sum(h * C[:, t, None, :], axis=-1)  # (batch, d_inner)
            outputs.append(y_t)

        # Stack outputs
        y = mx.stack(outputs, axis=1)  # (batch, seq_len, d_inner)

        # Add skip connection
        y = y + D[None, None, :] * u

        return y, h


class Mamba(nn.Module):
    """Mamba block with LayerNorm and residual connection.

    This wraps MambaBlock with:
    - Pre-norm LayerNorm
    - Residual connection

    Args:
        d_model: Input/output dimension
        d_state: State dimension
        d_conv: Local convolution width
        expand_factor: Expansion factor
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )

    def __call__(self, x: mx.array, state: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """Forward pass with residual.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            state: Optional initial state

        Returns:
            Tuple of (output, final_state)
        """
        residual = x
        x = self.norm(x)
        out, state = self.mamba(x, state)
        return out + residual, state


class SqueezedMamba(nn.Module):
    """Mamba variant compatible with SqueezedGRU_S interface.

    This provides a drop-in replacement for SqueezedGRU_S used in
    DeepFilterNet, with the same interface but using Mamba internally.

    The "squeezed" aspect refers to processing a squeezed representation
    where frequency bins are collapsed before temporal modeling.

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        output_size: Output dimension (if None, equals hidden_size)
        num_layers: Number of stacked Mamba layers
        d_state: Mamba state dimension
        d_conv: Mamba conv kernel size
        expand_factor: Mamba expansion factor
        linear_groups: Groups for output projection
        batch_first: Whether input is (batch, seq, feat)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        linear_groups: int = 1,
        batch_first: bool = True,
    ):
        super().__init__()

        if output_size is None:
            output_size = hidden_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Input projection if sizes don't match
        self.input_proj = None
        if input_size != hidden_size:
            self.input_proj = nn.Linear(input_size, hidden_size)

        # Stack of Mamba layers
        self.layers = [
            Mamba(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
            )
            for _ in range(num_layers)
        ]

        # Output projection
        self.output_proj = None
        if hidden_size != output_size:
            if linear_groups > 1:
                self.output_proj = GroupedLinear(hidden_size, output_size, groups=linear_groups)
            else:
                self.output_proj = nn.Linear(hidden_size, output_size)

    def __call__(
        self,
        x: mx.array,
        state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Input tensor
               - If batch_first: (batch, seq_len, input_size)
               - Else: (seq_len, batch, input_size)
            state: Optional initial states for each layer

        Returns:
            Tuple of (output, final_states)
        """
        # Handle batch dimension
        if not self.batch_first:
            x = mx.transpose(x, (1, 0, 2))

        # Input projection
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Process through Mamba layers
        states = []
        for i, layer in enumerate(self.layers):
            layer_state = state[i] if state is not None else None
            x, new_state = layer(x, layer_state)
            states.append(new_state)

        # Output projection
        if self.output_proj is not None:
            x = self.output_proj(x)

        # Handle batch dimension for output
        if not self.batch_first:
            x = mx.transpose(x, (1, 0, 2))

        return x, mx.stack(states, axis=0) if states else None


class BidirectionalMamba(nn.Module):
    """Bidirectional Mamba processing.

    Processes the sequence in both forward and backward directions,
    then combines the outputs.

    Args:
        d_model: Input/output dimension
        d_state: State dimension
        d_conv: Conv kernel size
        expand_factor: Expansion factor
        merge: How to merge directions ("concat", "add", "gate")
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        merge: str = "concat",
    ):
        super().__init__()

        self.merge = merge

        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )

        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )

        # Output projection for concat merge
        if merge == "concat":
            self.out_proj = nn.Linear(2 * d_model, d_model)
        elif merge == "gate":
            self.gate = nn.Linear(2 * d_model, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Forward direction
        fwd_out, _ = self.forward_mamba(x)

        # Backward direction (flip, process, flip back)
        x_rev = flip_along_axis(x, axis=1)
        bwd_out, _ = self.backward_mamba(x_rev)
        bwd_out = flip_along_axis(bwd_out, axis=1)

        # Merge
        if self.merge == "concat":
            combined = mx.concatenate([fwd_out, bwd_out], axis=-1)
            return self.out_proj(combined)
        elif self.merge == "add":
            return fwd_out + bwd_out
        elif self.merge == "gate":
            combined = mx.concatenate([fwd_out, bwd_out], axis=-1)
            gate = mx.sigmoid(self.gate(combined))
            return gate * fwd_out + (1 - gate) * bwd_out
        else:
            raise ValueError(f"Unknown merge method: {self.merge}")


class GroupedLinear(nn.Module):
    """Grouped linear layer for efficient computation.

    Splits input into groups and applies separate linear transformations.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        groups: Number of groups
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        assert in_features % groups == 0, "in_features must be divisible by groups"
        assert out_features % groups == 0, "out_features must be divisible by groups"

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups

        self.group_in = in_features // groups
        self.group_out = out_features // groups

        # Weight: (groups, group_in, group_out)
        self.weight = (
            mx.random.normal(shape=(groups, self.group_in, self.group_out)) * (2 / (in_features + out_features)) ** 0.5
        )

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Output tensor (..., out_features)
        """
        # Get batch dimensions
        batch_shape = x.shape[:-1]

        # Reshape for grouped operation
        x = x.reshape(*batch_shape, self.groups, self.group_in)

        # Apply grouped linear: einsum("...gi,gio->...go", x, self.weight)
        out = mx.einsum("...gi,gio->...go", x, self.weight)

        # Reshape output
        out = out.reshape(*batch_shape, self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return out
