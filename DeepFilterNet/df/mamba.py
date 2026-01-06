"""Mamba State Space Model implementation for DeepFilterNet4.

This module provides Mamba blocks as drop-in replacements for GRU/LSTM layers,
offering linear O(n) complexity vs O(n²) for attention mechanisms.

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
           Gu, A., & Dao, T. (2023)
"""

import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Final


def _init_A_log(d_inner: int, d_state: int) -> Tensor:
    """Initialize A in log-space for numerical stability."""
    A = torch.arange(1, d_state + 1, dtype=torch.float32)
    A = A.unsqueeze(0).expand(d_inner, -1)
    return torch.log(A)


def _init_dt_bias(d_inner: int, dt_min: float = 0.001, dt_max: float = 0.1) -> Tensor:
    """Initialize delta bias to map to [dt_min, dt_max] range."""
    dt = torch.exp(
        torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    ).clamp(min=1e-4)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    return inv_dt


class MambaBlock(nn.Module):
    """Selective State Space Model block.

    Implements the core Mamba computation:
    1. Input projection to (x, z)
    2. Causal 1D convolution on x
    3. Selective scan with input-dependent (Δ, B, C)
    4. Gated output: y * silu(z)
    5. Output projection

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (default: 16)
        d_conv: Local convolution width (default: 4)
        expand: Expansion factor for inner dimension (default: 2)
        dt_rank: Rank for delta projection (default: "auto")
        dt_min: Minimum delta value (default: 0.001)
        dt_max: Maximum delta value (default: 0.1)
        dt_init: Delta initialization method (default: "random")
        dt_scale: Delta scale factor (default: 1.0)
        bias: Use bias in projections (default: False)
        conv_bias: Use bias in conv layer (default: True)
    """

    d_model: Final[int]
    d_state: Final[int]
    d_conv: Final[int]
    expand: Final[int]
    d_inner: Final[int]
    dt_rank: Final[int]

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        # Input projection: d_model -> 2 * d_inner (for x and z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Causal 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # Causal padding
            groups=self.d_inner,
            bias=conv_bias,
        )

        # SSM parameters projection: d_inner -> dt_rank + 2*d_state
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        # Delta projection: dt_rank -> d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt weights
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias
        with torch.no_grad():
            self.dt_proj.bias.copy_(_init_dt_bias(self.d_inner, dt_min, dt_max))

        # A parameter (log-space for stability)
        self.A_log = nn.Parameter(_init_A_log(self.d_inner, d_state))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Mamba block.

        Args:
            x: Input tensor of shape [B, L, D]

        Returns:
            Output tensor of shape [B, L, D]
        """
        batch, length, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, L, d_inner]

        # Causal convolution
        x_conv = x_proj.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :length]  # Truncate for causality
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_inner]
        x_conv = F.silu(x_conv)

        # SSM parameters from input
        ssm_params = self.x_proj(x_conv)  # [B, L, dt_rank + 2*d_state]
        delta, B, C = torch.split(ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Delta projection and softplus
        delta = F.softplus(self.dt_proj(delta))  # [B, L, d_inner]

        # Recover A from log-space
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # Selective scan
        y = self._selective_scan(x_conv, delta, A, B, C)

        # Skip connection with D
        y = y + x_conv * self.D

        # Gated output
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: Tensor,  # [B, L, D]
        delta: Tensor,  # [B, L, D]
        A: Tensor,  # [D, N]
        B: Tensor,  # [B, L, N]
        C: Tensor,  # [B, L, N]
    ) -> Tensor:
        """Selective scan operation.

        This is a sequential implementation for correctness and compatibility.
        Can be optimized with parallel associative scan for production use.
        """
        batch, length, d_inner = x.shape
        d_state = A.shape[1]
        device = x.device
        dtype = x.dtype

        # Discretize A and B
        # A_bar = exp(delta * A)
        # B_bar = delta * B
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [B, L, D, N]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, D, N]

        # Sequential scan
        h = torch.zeros(batch, d_inner, d_state, device=device, dtype=dtype)
        ys = []

        for i in range(length):
            h = deltaA[:, i] * h + deltaB[:, i] * x[:, i, :, None]
            y = (h * C[:, i, None, :]).sum(dim=-1)
            ys.append(y)

        return torch.stack(ys, dim=1)


class Mamba(nn.Module):
    """Mamba layer with residual connection and layer normalization.

    This is the standard Mamba layer that can be stacked.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (default: 16)
        d_conv: Local convolution width (default: 4)
        expand: Expansion factor for inner dimension (default: 2)
        **kwargs: Additional arguments passed to MambaBlock
    """

    d_model: Final[int]

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output tensor [B, L, D]
        """
        return x + self.mamba(self.norm(x))


class SqueezedMamba(nn.Module):
    """Mamba-based sequence model with squeeze/expand projections.

    Drop-in replacement for SqueezedGRU_S with same API.

    Args:
        input_size: Input feature dimension
        hidden_size: Mamba hidden dimension
        output_size: Output feature dimension (default: same as hidden_size)
        num_layers: Number of stacked Mamba layers
        batch_first: If True, input is [B, T, F] (default: True)
        gru_skip_op: Optional skip connection factory (renamed from mamba_skip_op for API compat)
        linear_groups: Groups for input/output projections
        linear_act_layer: Activation after linear projections
        d_state: SSM state dimension (default: 16)
        d_conv: Local convolution width (default: 4)
        expand: Expansion factor (default: 2)
    """

    input_size: Final[int]
    hidden_size: Final[int]
    output_size: Final[int]
    batch_first: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        batch_first: bool = True,
        gru_skip_op: Optional[Callable[..., nn.Module]] = None,
        linear_groups: int = 1,
        linear_act_layer: Callable[..., nn.Module] = nn.Identity,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else hidden_size
        self.batch_first = batch_first

        # Input projection (squeeze)
        if linear_groups > 1:
            from df.modules import GroupedLinearEinsum

            self.linear_in = nn.Sequential(
                GroupedLinearEinsum(input_size, hidden_size, linear_groups), linear_act_layer()
            )
        else:
            self.linear_in = nn.Sequential(nn.Linear(input_size, hidden_size), linear_act_layer())

        # Mamba layers
        self.mamba_layers = nn.ModuleList(
            [Mamba(hidden_size, d_state, d_conv, expand) for _ in range(num_layers)]
        )

        # Skip connection (same API as SqueezedGRU_S)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None

        # Output projection (expand)
        if output_size is not None:
            if linear_groups > 1:
                from df.modules import GroupedLinearEinsum

                self.linear_out = nn.Sequential(
                    GroupedLinearEinsum(hidden_size, self.output_size, linear_groups),
                    linear_act_layer(),
                )
            else:
                self.linear_out = nn.Sequential(
                    nn.Linear(hidden_size, self.output_size), linear_act_layer()
                )
        else:
            self.linear_out = nn.Identity()

    def forward(
        self,
        input: Tensor,
        h: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with SqueezedGRU_S compatible API.

        Args:
            input: Input [B, T, input_size] if batch_first else [T, B, input_size]
            h: Ignored (for API compatibility with GRU)

        Returns:
            output: [B, T, output_size] or [T, B, output_size]
            h_n: Dummy hidden state (zeros) for API compatibility
        """
        if not self.batch_first:
            input = input.transpose(0, 1)

        batch_size = input.size(0)

        # Squeeze
        x = self.linear_in(input)

        # Mamba layers
        for layer in self.mamba_layers:
            x = layer(x)

        # Expand
        x = self.linear_out(x)

        # Skip connection (applied after expansion like SqueezedGRU_S)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)

        if not self.batch_first:
            x = x.transpose(0, 1)

        # Dummy hidden state for compatibility with GRU API
        h_n = torch.zeros(1, batch_size, self.hidden_size, device=input.device, dtype=input.dtype)

        return x, h_n


class BidirectionalMamba(nn.Module):
    """Bidirectional Mamba layer.

    Processes sequence in both directions and concatenates outputs.

    Args:
        d_model: Model dimension (output will be 2*d_model if not using merge)
        merge: How to merge forward/backward outputs ("concat", "add", "proj")
        **kwargs: Arguments passed to Mamba
    """

    def __init__(
        self,
        d_model: int,
        merge: str = "concat",
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.merge = merge

        self.forward_mamba = Mamba(d_model, **kwargs)
        self.backward_mamba = Mamba(d_model, **kwargs)

        if merge == "proj":
            self.out_proj = nn.Linear(d_model * 2, d_model)
        else:
            self.out_proj = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input [B, L, D]

        Returns:
            Output [B, L, D] or [B, L, 2D] depending on merge mode
        """
        # Forward direction
        y_fwd = self.forward_mamba(x)

        # Backward direction (flip, process, flip back)
        x_bwd = torch.flip(x, dims=[1])
        y_bwd = self.backward_mamba(x_bwd)
        y_bwd = torch.flip(y_bwd, dims=[1])

        if self.merge == "concat":
            return torch.cat([y_fwd, y_bwd], dim=-1)
        elif self.merge == "add":
            return y_fwd + y_bwd
        elif self.merge == "proj":
            assert self.out_proj is not None
            return self.out_proj(torch.cat([y_fwd, y_bwd], dim=-1))
        else:
            raise ValueError(f"Unknown merge mode: {self.merge}")
