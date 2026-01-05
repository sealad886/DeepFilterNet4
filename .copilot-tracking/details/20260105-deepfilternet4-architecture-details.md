<!-- markdownlint-disable-file -->
# Task Details: DeepFilterNet4 Architecture Implementation

## Research Reference

**Source Research**: #file:../research/20260105-deepfilternet4-architecture-research.md

---

## Phase 1: Core Mamba Integration

### Task 1.1: Mamba module implementation

Create the foundational Mamba/S4 state space model implementation as the core building block for DFNet4.

- **Files**:
  - `DeepFilterNet/df/mamba.py` - New file with Mamba implementation
  - `DeepFilterNet/df/__init__.py` - Add exports for new modules
- **Success**:
  - `MambaBlock` class implements selective state space model
  - `Mamba` class wraps block with normalization and residual
  - Forward pass produces correct output shapes
  - Causal behavior verified (no future information leakage)
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 180-230) - Mamba block specification
- **Dependencies**:
  - None (foundational task)

**Implementation Specification:**

```python
# DeepFilterNet/df/mamba.py

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

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
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        
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
        
        # Initialize dt bias for proper range
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            
        # Initialize dt bias to map to [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # A parameter (log-space for stability)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape [B, L, D]
            
        Returns:
            Output tensor of shape [B, L, D]
        """
        batch, length, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)  # Each [B, L, d_inner]
        
        # Causal convolution
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :length]  # Truncate to maintain causality
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)
        
        # SSM parameters from input
        x_proj = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]
        delta, B, C = torch.split(
            x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # Delta projection and softplus
        delta = F.softplus(self.dt_proj(delta))  # [B, L, d_inner]
        
        # Recover A from log-space
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Selective scan
        y = self.selective_scan(x, delta, A, B, C)
        
        # Skip connection with D
        y = y + x * self.D
        
        # Gated output
        y = y * F.silu(z)
        
        # Output projection
        return self.out_proj(y)
    
    def selective_scan(
        self,
        x: Tensor,      # [B, L, D]
        delta: Tensor,  # [B, L, D]
        A: Tensor,      # [D, N]
        B: Tensor,      # [B, L, N]
        C: Tensor,      # [B, L, N]
    ) -> Tensor:
        """Selective scan operation (sequential for reference, parallel in practice)."""
        batch, length, d_inner = x.shape
        d_state = A.shape[1]
        
        # Discretize A and B
        # A_bar = exp(delta * A)
        # B_bar = delta * B
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [B, L, D, N]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, D, N]
        
        # Sequential scan (can be parallelized with associative scan)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(length):
            h = deltaA[:, i] * h + deltaB[:, i] * x[:, i, :, None]
            y = (h * C[:, i, None, :]).sum(dim=-1)
            ys.append(y)
            
        return torch.stack(ys, dim=1)


class Mamba(nn.Module):
    """Mamba layer with residual connection and layer normalization."""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        return x + self.mamba(self.norm(x))
```

---

### Task 1.2: SqueezedMamba layer

Create a drop-in replacement for `SqueezedGRU_S` using Mamba blocks.

- **Files**:
  - `DeepFilterNet/df/mamba.py` - Add `SqueezedMamba` class
  - `DeepFilterNet/df/modules.py` - Import and expose `SqueezedMamba`
- **Success**:
  - API-compatible with `SqueezedGRU_S`
  - Same input/output shapes
  - Supports skip connections like GRU variant
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 81-130) - SqueezedMamba spec
  - `DeepFilterNet/df/modules.py` (Lines 650-750) - SqueezedGRU_S reference
- **Dependencies**:
  - Task 1.1 completion

**Implementation Specification:**

```python
class SqueezedMamba(nn.Module):
    """Mamba-based sequence model with squeeze/expand projections.
    
    Drop-in replacement for SqueezedGRU_S with same API.
    
    Args:
        input_size: Input feature dimension
        hidden_size: Mamba hidden dimension
        output_size: Output feature dimension (default: same as hidden_size)
        num_layers: Number of stacked Mamba layers
        batch_first: If True, input is [B, T, F] (default: True)
        mamba_skip_op: Optional skip connection factory
        linear_groups: Groups for input/output projections
        linear_act_layer: Activation after linear projections
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        batch_first: bool = True,
        mamba_skip_op: Optional[Callable] = None,
        linear_groups: int = 1,
        linear_act_layer: Optional[Callable] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.batch_first = batch_first
        
        # Input projection (squeeze)
        if linear_groups > 1:
            self.input_proj = GroupedLinearEinsum(
                input_size, hidden_size, groups=linear_groups
            )
        else:
            self.input_proj = nn.Linear(input_size, hidden_size)
            
        if linear_act_layer is not None:
            self.input_act = linear_act_layer()
        else:
            self.input_act = nn.Identity()
        
        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(hidden_size, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])
        
        # Skip connection
        self.skip = mamba_skip_op() if mamba_skip_op is not None else None
        
        # Output projection (expand)
        if linear_groups > 1:
            self.output_proj = GroupedLinearEinsum(
                hidden_size, self.output_size, groups=linear_groups
            )
        else:
            self.output_proj = nn.Linear(hidden_size, self.output_size)
            
    def forward(
        self, 
        x: Tensor, 
        h: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input [B, T, input_size] if batch_first else [T, B, input_size]
            h: Ignored (for API compatibility with GRU)
            
        Returns:
            output: [B, T, output_size]
            h_n: Dummy hidden state (zeros) for API compatibility
        """
        if not self.batch_first:
            x = x.transpose(0, 1)
            
        # Squeeze
        x = self.input_act(self.input_proj(x))
        
        # Store for skip connection
        x_in = x
        
        # Mamba layers
        for layer in self.mamba_layers:
            x = layer(x)
            
        # Skip connection
        if self.skip is not None:
            x = x + self.skip(x_in)
            
        # Expand
        output = self.output_proj(x)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
            
        # Dummy hidden state for compatibility
        h_n = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        
        return output, h_n
```

---

### Task 1.3: Mamba unit tests

Comprehensive test suite for Mamba modules.

- **Files**:
  - `DeepFilterNet/tests/test_mamba.py` - New test file
- **Success**:
  - All tests pass
  - Coverage > 90% for mamba.py
  - Tests cover edge cases (batch_size=1, varying lengths)
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 131-170)
- **Dependencies**:
  - Tasks 1.1, 1.2 completion

**Test Specification:**

```python
# DeepFilterNet/tests/test_mamba.py

import pytest
import torch
from df.mamba import MambaBlock, Mamba, SqueezedMamba


class TestMambaBlock:
    @pytest.fixture
    def mamba_block(self):
        return MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2)
    
    def test_output_shape(self, mamba_block):
        x = torch.randn(2, 100, 64)
        y = mamba_block(x)
        assert y.shape == x.shape
        
    def test_causality(self, mamba_block):
        """Verify output at time t doesn't depend on input at time > t."""
        x = torch.randn(1, 50, 64)
        y1 = mamba_block(x)
        
        # Modify future input
        x2 = x.clone()
        x2[:, 25:, :] = torch.randn(1, 25, 64)
        y2 = mamba_block(x2)
        
        # First 25 timesteps should be identical
        assert torch.allclose(y1[:, :25], y2[:, :25], atol=1e-5)
        
    def test_batch_independence(self, mamba_block):
        """Verify batches are processed independently."""
        x1 = torch.randn(1, 50, 64)
        x2 = torch.randn(1, 50, 64)
        
        y_separate = torch.cat([mamba_block(x1), mamba_block(x2)], dim=0)
        y_batched = mamba_block(torch.cat([x1, x2], dim=0))
        
        assert torch.allclose(y_separate, y_batched, atol=1e-5)


class TestSqueezedMamba:
    def test_gru_api_compatibility(self):
        """Verify API matches SqueezedGRU_S."""
        mamba = SqueezedMamba(
            input_size=128,
            hidden_size=256,
            output_size=128,
            num_layers=2,
        )
        
        x = torch.randn(4, 100, 128)
        output, h_n = mamba(x)
        
        assert output.shape == (4, 100, 128)
        assert h_n.shape[0] == 1  # num_layers direction
        
    def test_skip_connection(self):
        """Test with skip connection."""
        mamba = SqueezedMamba(
            input_size=64,
            hidden_size=64,
            mamba_skip_op=lambda: nn.Identity(),
        )
        
        x = torch.randn(2, 50, 64)
        output, _ = mamba(x)
        assert output.shape == (2, 50, 64)
```

---

### Task 1.4: Mamba benchmark script

Performance comparison script for Mamba vs GRU.

- **Files**:
  - `DeepFilterNet/df/scripts/benchmark_mamba.py` - New benchmark script
- **Success**:
  - Measures inference latency, throughput, memory usage
  - Compares Mamba vs GRU on same configurations
  - Outputs markdown-formatted results
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 171-200)
- **Dependencies**:
  - Tasks 1.1, 1.2 completion

---

### Task 1.5: MPS compatibility for Mamba

Ensure Mamba modules work correctly on Apple Silicon MPS backend.

- **Files**:
  - `DeepFilterNet/df/mamba.py` - Add MPS compatibility checks
  - `DeepFilterNet/tests/test_mamba.py` - Add MPS-specific tests
- **Success**:
  - All Mamba operations work on MPS
  - Fallback to CPU for unsupported ops
  - No silent numerical errors
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 201-230)
  - Recent MPS backend work in utils.py
- **Dependencies**:
  - Tasks 1.1-1.3 completion

---

## Phase 2: Hybrid Encoder Architecture

### Task 2.1: Time-domain encoder branch

Create waveform encoder for time-domain processing.

- **Files**:
  - `DeepFilterNet/df/hybrid_encoder.py` - New file with `WaveformEncoder`
- **Success**:
  - Encodes raw waveform to feature representation
  - Maintains temporal alignment with frequency features
  - Configurable depth and width
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 231-290)
- **Dependencies**:
  - Phase 1 completion

**Implementation Specification:**

```python
# DeepFilterNet/df/hybrid_encoder.py

class WaveformEncoder(nn.Module):
    """Time-domain waveform encoder.
    
    Processes raw audio waveform with strided convolutions to extract
    time-domain features aligned with STFT frames.
    
    Args:
        in_channels: Input channels (1 for mono)
        base_channels: Base channel count (doubled each layer)
        num_layers: Number of conv layers
        kernel_sizes: Kernel size per layer
        strides: Stride per layer (should match STFT hop alignment)
        out_dim: Output feature dimension
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_layers: int = 4,
        kernel_sizes: List[int] = [7, 5, 5, 3],
        strides: List[int] = [4, 2, 2, 2],
        out_dim: int = 256,
    ):
        super().__init__()
        
        layers = []
        ch = in_channels
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            layers.append(
                nn.Sequential(
                    nn.Conv1d(ch, out_ch, kernel_sizes[i], strides[i], 
                              padding=kernel_sizes[i] // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                )
            )
            ch = out_ch
            
        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Linear(ch, out_dim)
        
        # Total stride for alignment calculation
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s
            
    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform: [B, 1, T_samples] or [B, T_samples]
            
        Returns:
            features: [B, T_frames, out_dim]
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
            
        x = self.encoder(waveform)  # [B, C, T_frames]
        x = x.transpose(1, 2)        # [B, T_frames, C]
        return self.proj(x)          # [B, T_frames, out_dim]
```

---

### Task 2.2: Phase encoder branch

Create phase spectrum encoder.

- **Files**:
  - `DeepFilterNet/df/hybrid_encoder.py` - Add `PhaseEncoder`
- **Success**:
  - Processes unwrapped or instantaneous phase
  - Handles phase wrapping appropriately
  - Outputs features aligned with magnitude
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 291-340)
- **Dependencies**:
  - Task 2.1 completion

**Implementation Specification:**

```python
class PhaseEncoder(nn.Module):
    """Phase spectrum encoder.
    
    Processes phase information using group delay and instantaneous frequency.
    
    Args:
        n_freqs: Number of frequency bins
        conv_ch: Convolutional channels
        out_dim: Output feature dimension
    """
    
    def __init__(
        self,
        n_freqs: int,
        conv_ch: int = 32,
        out_dim: int = 256,
    ):
        super().__init__()
        
        # Phase has 2 channels: cos(phase), sin(phase)
        self.conv_layers = nn.Sequential(
            Conv2dNormAct(2, conv_ch, kernel_size=(3, 3), separable=True),
            Conv2dNormAct(conv_ch, conv_ch * 2, kernel_size=(1, 3), 
                          fstride=2, separable=True),
            Conv2dNormAct(conv_ch * 2, conv_ch * 2, kernel_size=(1, 3),
                          fstride=2, separable=True),
        )
        
        self.out_proj = nn.Linear(conv_ch * 2 * n_freqs // 4, out_dim)
        
    def forward(self, phase: Tensor) -> Tensor:
        """
        Args:
            phase: Complex phase tensor [B, 1, T, F, 2] or angle [B, 1, T, F]
            
        Returns:
            features: [B, T, out_dim]
        """
        if phase.dim() == 4:
            # Convert angle to cos/sin representation
            phase = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)
            
        # Reshape: [B, 1, T, F, 2] -> [B, 2, T, F]
        phase = phase.squeeze(1).permute(0, 3, 1, 2)
        
        x = self.conv_layers(phase)  # [B, C, T, F']
        x = x.permute(0, 2, 1, 3).flatten(2)  # [B, T, C*F']
        return self.out_proj(x)
```

---

### Task 2.3: Cross-domain attention fusion

Create attention-based feature fusion module.

- **Files**:
  - `DeepFilterNet/df/hybrid_encoder.py` - Add `CrossDomainAttention`
- **Success**:
  - Fuses time, magnitude, and phase features
  - Uses cross-attention for domain interaction
  - Efficient implementation (linear complexity)
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 341-400)
- **Dependencies**:
  - Tasks 2.1, 2.2 completion

**Implementation Specification:**

```python
class CrossDomainAttention(nn.Module):
    """Cross-domain attention for feature fusion.
    
    Uses efficient linear attention to fuse features from different domains.
    
    Args:
        time_dim: Time-domain feature dimension
        mag_dim: Magnitude feature dimension
        phase_dim: Phase feature dimension
        out_dim: Output fused dimension
        num_heads: Number of attention heads
    """
    
    def __init__(
        self,
        time_dim: int,
        mag_dim: int,
        phase_dim: int,
        out_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        
        # Project all domains to same dimension
        self.time_proj = nn.Linear(time_dim, out_dim)
        self.mag_proj = nn.Linear(mag_dim, out_dim)
        self.phase_proj = nn.Linear(phase_dim, out_dim)
        
        # Cross-attention layers
        self.time_mag_attn = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)
        self.mag_phase_attn = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(
        self,
        time_feat: Tensor,
        mag_feat: Tensor,
        phase_feat: Tensor,
    ) -> Tensor:
        """
        Args:
            time_feat: [B, T, time_dim]
            mag_feat: [B, T, mag_dim]
            phase_feat: [B, T, phase_dim]
            
        Returns:
            fused: [B, T, out_dim]
        """
        # Project to common dimension
        t = self.time_proj(time_feat)
        m = self.mag_proj(mag_feat)
        p = self.phase_proj(phase_feat)
        
        # Cross-attention: time attends to magnitude
        t_m, _ = self.time_mag_attn(t, m, m)
        t = t + t_m
        
        # Cross-attention: magnitude attends to phase
        m_p, _ = self.mag_phase_attn(m, p, p)
        m = m + m_p
        
        # Concatenate and fuse
        fused = torch.cat([t, m, p], dim=-1)
        return self.norm(self.fusion(fused))
```

---

### Task 2.4: HybridEncoder integration

Complete hybrid encoder integrating all branches.

- **Files**:
  - `DeepFilterNet/df/hybrid_encoder.py` - Add `HybridEncoder` class
- **Success**:
  - Combines time, magnitude, phase encoders
  - Uses Mamba for sequence modeling
  - Outputs embeddings compatible with decoders
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 401-460)
- **Dependencies**:
  - Tasks 2.1-2.3 completion

**Implementation Specification:**

```python
class HybridEncoder(nn.Module):
    """Hybrid time-frequency encoder for DFNet4.
    
    Parallel processing of time-domain, magnitude, and phase information
    with cross-domain attention fusion and Mamba sequence modeling.
    """
    
    def __init__(self, params: "ModelParams4"):
        super().__init__()
        self.params = params
        
        # Domain-specific encoders
        self.time_encoder = WaveformEncoder(
            out_dim=params.emb_hidden_dim,
        ) if params.use_time_branch else None
        
        # Magnitude encoder (similar to DFNet3 ERB encoder)
        self.mag_encoder = MagnitudeEncoder(params)
        
        self.phase_encoder = PhaseEncoder(
            n_freqs=params.fft_size // 2 + 1,
            out_dim=params.emb_hidden_dim,
        ) if params.use_phase_branch else None
        
        # Cross-domain fusion
        time_dim = params.emb_hidden_dim if params.use_time_branch else 0
        phase_dim = params.emb_hidden_dim if params.use_phase_branch else 0
        
        self.fusion = CrossDomainAttention(
            time_dim=time_dim or params.emb_hidden_dim,
            mag_dim=params.emb_hidden_dim,
            phase_dim=phase_dim or params.emb_hidden_dim,
            out_dim=params.emb_hidden_dim,
        )
        
        # Sequence modeling (Mamba replaces GRU)
        self.mamba_layers = nn.ModuleList([
            Mamba(params.emb_hidden_dim)
            for _ in range(params.emb_num_layers)
        ])
        
        # LSNR estimation
        self.lsnr_fc = nn.Sequential(
            nn.Linear(params.emb_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.lsnr_scale = params.lsnr_max - params.lsnr_min
        self.lsnr_offset = params.lsnr_min
        
    def forward(
        self,
        waveform: Optional[Tensor],
        feat_erb: Tensor,
        feat_spec: Tensor,
    ) -> Tuple[Tensor, ...]:
        """
        Args:
            waveform: Raw audio [B, T_samples] (optional)
            feat_erb: ERB features [B, 1, T, E]
            feat_spec: Complex spectrogram [B, 1, T, F', 2]
            
        Returns:
            e0, e1, e2, e3: Encoder intermediate outputs
            emb: Final embedding [B, T, H]
            c0: DF pathway features
            lsnr: Local SNR estimate [B, T, 1]
        """
        # Magnitude pathway (similar to DFNet3)
        e0, e1, e2, e3, mag_emb, c0 = self.mag_encoder(feat_erb, feat_spec)
        
        # Time-domain features
        if self.time_encoder is not None and waveform is not None:
            time_feat = self.time_encoder(waveform)
        else:
            time_feat = mag_emb  # Fallback
            
        # Phase features
        if self.phase_encoder is not None:
            phase = torch.atan2(feat_spec[..., 1], feat_spec[..., 0])
            phase_feat = self.phase_encoder(phase)
        else:
            phase_feat = mag_emb  # Fallback
            
        # Cross-domain fusion
        emb = self.fusion(time_feat, mag_emb, phase_feat)
        
        # Mamba sequence modeling
        for layer in self.mamba_layers:
            emb = layer(emb)
            
        # LSNR estimation
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset
        
        return e0, e1, e2, e3, emb, c0, lsnr
```

---

### Task 2.5: Hybrid encoder tests

Unit tests for hybrid encoder components.

- **Files**:
  - `DeepFilterNet/tests/test_hybrid_encoder.py` - New test file
- **Success**:
  - All encoder components tested individually
  - Integration test for full HybridEncoder
  - Shape consistency verified
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 461-500)
- **Dependencies**:
  - Tasks 2.1-2.4 completion

---

## Phase 3: Multi-Resolution Deep Filtering

### Task 3.1: MultiResolutionDF module

Multi-resolution deep filtering with learnable weights.

- **Files**:
  - `DeepFilterNet/df/multiframe.py` - Add `MultiResolutionDF` class
- **Success**:
  - Applies DF at multiple resolutions
  - Learnable resolution weights
  - Compatible with existing DF infrastructure
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 501-560)
- **Dependencies**:
  - Phase 2 completion

**Implementation Specification:**

```python
class MultiResolutionDF(nn.Module):
    """Apply deep filtering at multiple frequency resolutions.
    
    Args:
        resolutions: List of (num_freqs, frame_size) tuples
        lookahead: Lookahead frames for DF
        learnable_weights: Whether to learn resolution weights
    """
    
    def __init__(
        self,
        resolutions: List[Tuple[int, int]] = [(96, 5), (48, 3), (24, 2)],
        lookahead: int = 0,
        learnable_weights: bool = True,
    ):
        super().__init__()
        
        self.df_ops = nn.ModuleList([
            DF(num_freqs=nf, frame_size=fs, lookahead=lookahead)
            for nf, fs in resolutions
        ])
        
        self.resolutions = resolutions
        
        if learnable_weights:
            self.resolution_weights = nn.Parameter(
                torch.ones(len(resolutions)) / len(resolutions)
            )
        else:
            self.register_buffer(
                "resolution_weights",
                torch.ones(len(resolutions)) / len(resolutions)
            )
            
    def forward(
        self,
        spec: Tensor,
        coefs_list: List[Tensor],
    ) -> Tensor:
        """
        Args:
            spec: Input spectrum [B, 1, T, F, 2]
            coefs_list: List of coefficients for each resolution
            
        Returns:
            Enhanced spectrum [B, 1, T, F, 2]
        """
        weights = F.softmax(self.resolution_weights, dim=0)
        
        outputs = []
        for df_op, coefs, (nf, _) in zip(self.df_ops, coefs_list, self.resolutions):
            # Apply DF at this resolution
            spec_res = spec[..., :nf, :].clone()
            out_res = df_op(spec_res, coefs)
            
            # Pad back to full frequency range
            if nf < spec.shape[-2]:
                out_res = F.pad(out_res, (0, 0, 0, spec.shape[-2] - nf))
                
            outputs.append(out_res)
            
        # Weighted combination
        result = sum(w * out for w, out in zip(weights, outputs))
        return result
```

---

### Task 3.2: AdaptiveOrderPredictor

SNR-based filter order selection module.

- **Files**:
  - `DeepFilterNet/df/multiframe.py` - Add `AdaptiveOrderPredictor`
- **Success**:
  - Predicts optimal filter order from embeddings
  - Soft selection during training
  - Hard selection during inference
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 561-610)
- **Dependencies**:
  - Task 3.1 completion

**Implementation Specification:**

```python
class AdaptiveOrderPredictor(nn.Module):
    """Predict optimal filter order based on input characteristics.
    
    Args:
        emb_dim: Input embedding dimension
        max_order: Maximum filter order
        min_order: Minimum filter order
    """
    
    def __init__(
        self,
        emb_dim: int,
        max_order: int = 7,
        min_order: int = 2,
    ):
        super().__init__()
        self.max_order = max_order
        self.min_order = min_order
        self.num_orders = max_order - min_order + 1
        
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.num_orders),
        )
        
    def forward(
        self,
        emb: Tensor,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            emb: Input embedding [B, T, emb_dim]
            temperature: Softmax temperature (lower = sharper)
            
        Returns:
            order_weights: Soft weights [B, T, num_orders]
            predicted_order: Hard order selection [B, T]
        """
        logits = self.predictor(emb)  # [B, T, num_orders]
        
        if self.training:
            # Soft selection with Gumbel-Softmax
            order_weights = F.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            # Hard selection
            order_weights = F.one_hot(
                logits.argmax(-1), self.num_orders
            ).float()
            
        predicted_order = logits.argmax(-1) + self.min_order
        
        return order_weights, predicted_order
```

---

### Task 3.3: Multi-res DF decoder

Decoder producing coefficients for multiple resolutions.

- **Files**:
  - `DeepFilterNet/df/deepfilternet4.py` - Add `MultiResDfDecoder`
- **Success**:
  - Outputs coefficients for each resolution
  - Efficient shared backbone
  - Resolution-specific output heads
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 611-670)
- **Dependencies**:
  - Tasks 3.1, 3.2 completion

---

### Task 3.4: Multi-resolution DF tests

Integration tests for multi-resolution DF.

- **Files**:
  - `DeepFilterNet/tests/test_multires_df.py` - New test file
- **Success**:
  - All components tested
  - Output shapes verified
  - Gradient flow confirmed
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 671-710)
- **Dependencies**:
  - Tasks 3.1-3.3 completion

---

## Phase 4: Training Enhancements

### Task 4.1: MultiPeriodDiscriminator

Implement multi-period discriminator for adversarial training.

- **Files**:
  - `DeepFilterNet/df/discriminator.py` - New file with discriminator
- **Success**:
  - `PeriodDiscriminator` for single period
  - `MultiPeriodDiscriminator` combining multiple periods
  - Returns scores and feature maps for loss computation
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 711-780)
- **Dependencies**:
  - Phase 3 completion

**Implementation Specification:**

```python
# DeepFilterNet/df/discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class PeriodDiscriminator(nn.Module):
    """Single-period sub-discriminator.
    
    Reshapes waveform into 2D based on period and applies 2D convolutions.
    
    Args:
        period: Period for reshaping
        use_spectral_norm: Use spectral normalization
    """
    
    def __init__(self, period: int, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Waveform [B, 1, T]
            
        Returns:
            score: Discriminator score [B, ...]
            fmap: List of feature maps for feature matching loss
        """
        fmap = []
        
        # Reshape to 2D based on period
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
            
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator.
    
    Combines multiple period-based sub-discriminators.
    
    Args:
        periods: List of periods to use
    """
    
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])
        
    def forward(
        self, 
        y: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            y: Waveform [B, 1, T] or [B, T]
            
        Returns:
            scores: List of discriminator scores
            fmaps: List of feature map lists
        """
        if y.dim() == 2:
            y = y.unsqueeze(1)
            
        scores = []
        fmaps = []
        
        for d in self.discriminators:
            score, fmap = d(y)
            scores.append(score)
            fmaps.append(fmap)
            
        return scores, fmaps
```

---

### Task 4.2: Feature matching loss

Discriminator feature matching loss for training stability.

- **Files**:
  - `DeepFilterNet/df/loss.py` - Add `FeatureMatchingLoss`
- **Success**:
  - Computes L1 distance between feature maps
  - Weighted combination across discriminators
  - Integrated with existing loss infrastructure
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 781-820)
- **Dependencies**:
  - Task 4.1 completion

---

### Task 4.3: DNSMOSProxy model

Differentiable DNSMOS approximation model.

- **Files**:
  - `DeepFilterNet/df/dnsmos_proxy.py` - New file
- **Success**:
  - Predicts SIG, BAK, OVL scores
  - Pre-trained weights loadable
  - Training script for proxy model
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 821-890)
- **Dependencies**:
  - Task 4.2 completion

---

### Task 4.4: SpeakerContrastiveLoss

Speaker embedding preservation loss.

- **Files**:
  - `DeepFilterNet/df/loss.py` - Add `SpeakerContrastiveLoss`
- **Success**:
  - Uses frozen speaker encoder
  - Computes cosine similarity loss
  - Configurable positive/negative weighting
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 891-940)
- **Dependencies**:
  - Task 4.3 completion

---

### Task 4.5: Loss class updates

Integrate new losses into training.

- **Files**:
  - `DeepFilterNet/df/loss.py` - Update `Loss` class
- **Success**:
  - `Loss` class supports discriminator losses
  - Configurable loss weights
  - Backward compatible with DFNet3 training
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 941-1000)
- **Dependencies**:
  - Tasks 4.1-4.4 completion

---

### Task 4.6: GAN training loop

Update training for adversarial learning.

- **Files**:
  - `DeepFilterNet/df/train.py` - Add GAN training support
- **Success**:
  - Generator/discriminator alternating updates
  - Gradient penalty support
  - Progressive training schedule
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 1001-1060)
- **Dependencies**:
  - Task 4.5 completion

---

## Phase 5: Model Variants & Optimization

### Task 5.1: DFNet4 full model

Complete DeepFilterNet4 model integration.

- **Files**:
  - `DeepFilterNet/df/deepfilternet4.py` - Complete `DfNet4` class
  - `DeepFilterNet/df/model.py` - Update `init_model` for DFNet4
- **Success**:
  - Full model with all new components
  - Backward compatible config loading
  - `init_model` returns DFNet4 when configured
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 1061-1150)
- **Dependencies**:
  - Phases 1-4 completion

---

### Task 5.2: DFNet4Lite variant

50% parameter reduction variant.

- **Files**:
  - `DeepFilterNet/df/deepfilternet4.py` - Add `DfNet4Lite` class
- **Success**:
  - ~1M parameters (vs ~2M for full)
  - Maintains core architecture
  - Configurable via `MODEL_VARIANT` param
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 1151-1200)
- **Dependencies**:
  - Task 5.1 completion

---

### Task 5.3: Config parameters

Configuration for DFNet4.

- **Files**:
  - `DeepFilterNet/df/config.py` - Add DFNet4 config section
  - `DeepFilterNet/df/deepfilternet4.py` - Add `ModelParams4`
- **Success**:
  - All new parameters documented
  - Default values match research spec
  - Backward compatible with DFNet3 configs
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 1201-1260)
- **Dependencies**:
  - Task 5.2 completion

---

### Task 5.4: Quantization-aware training

QAT support for INT8 deployment.

- **Files**:
  - `DeepFilterNet/df/quantization.py` - New file
  - `DeepFilterNet/df/train.py` - Add QAT flag
- **Success**:
  - QAT training mode
  - Post-training quantization script
  - INT8 model export
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 1261-1310)
- **Dependencies**:
  - Task 5.3 completion

---

### Task 5.5: Knowledge distillation

Training script for Lite variant.

- **Files**:
  - `DeepFilterNet/df/scripts/distill.py` - New script
- **Success**:
  - Teacher-student training setup
  - Configurable distillation temperature
  - Logs student/teacher comparison
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 1311-1360)
- **Dependencies**:
  - Task 5.4 completion

---

## Phase 6: Integration & Testing

### Task 6.1: Full integration tests

End-to-end model tests.

- **Files**:
  - `DeepFilterNet/tests/test_deepfilternet4.py` - New test file
- **Success**:
  - Full forward pass tested
  - Training step tested
  - All model variants tested
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 1361-1410)
- **Dependencies**:
  - Phase 5 completion

---

### Task 6.2: Performance benchmarks

PESQ/STOI/DNSMOS evaluation scripts.

- **Files**:
  - `DeepFilterNet/df/scripts/benchmark_dfnet4.py` - New script
- **Success**:
  - Evaluates on VoiceBank-DEMAND
  - Reports PESQ, STOI, DNSMOS
  - Comparison with DFNet3 baseline
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 1411-1460)
- **Dependencies**:
  - Task 6.1 completion

---

### Task 6.3: Model export

ONNX export with validation.

- **Files**:
  - `DeepFilterNet/df/scripts/export_onnx.py` - Update for DFNet4
- **Success**:
  - ONNX export works
  - Output matches PyTorch within tolerance
  - Dynamic axes for variable length
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 1461-1500)
- **Dependencies**:
  - Task 6.2 completion

---

### Task 6.4: Documentation

README updates and architecture docs.

- **Files**:
  - `README.md` - Add DFNet4 section
  - `DeepFilterNet/README.md` - Update package docs
  - `docs/ARCHITECTURE.md` - New architecture documentation
- **Success**:
  - DFNet4 features documented
  - Installation instructions updated
  - Architecture diagrams included
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 1501-1550)
- **Dependencies**:
  - Task 6.3 completion

---

### Task 6.5: Migration guide

Guide for DFNet3 → DFNet4 users.

- **Files**:
  - `docs/MIGRATION.md` - New migration guide
- **Success**:
  - Configuration changes documented
  - API differences explained
  - Common issues and solutions
- **Research References**:
  - #file:../research/20260105-deepfilternet4-architecture-research.md (Lines 1551-1600)
- **Dependencies**:
  - Task 6.4 completion

---

## Dependencies Summary

### Python Packages
- `torch>=2.0` - PyTorch with MPS support
- `einops>=0.7.0` - Tensor operations
- `mamba-ssm>=1.2.0` - Mamba implementation (optional, can use pure PyTorch)
- `resemblyzer>=0.1.3` - Speaker embeddings (optional)

### Hardware
- Training: GPU with 16GB+ VRAM
- Inference: CPU, CUDA, or MPS

---

## Success Criteria

### Performance Targets
- PESQ: >= 3.45 on VoiceBank-DEMAND
- STOI: >= 0.96
- DNSMOS-OVL: >= 3.50
- RTF (CPU): < 0.25
- RTF (MPS): < 0.10

### Code Quality
- Test coverage > 90% for new code
- No regressions in existing tests
- Documentation complete

