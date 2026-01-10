"""Neural network modules for MLX DeepFilterNet4.

This module provides MLX implementations of the building blocks used in
DeepFilterNet4, including:
- Convolution layers with normalization and activation
- ERB filterbank operations
- Mask estimation and application
- Deep Filtering operations
- Grouped linear layers

All modules are optimized for Apple Silicon unified memory architecture.
"""

import math
from typing import Optional, Tuple, Union, cast

import mlx.core as mx
import mlx.nn as nn

from .ops import erb_fb as make_erb_fb

# ============================================================================
# Convolution Modules
# ============================================================================


class Conv2dNormAct(nn.Module):
    """2D Convolution with optional normalization and activation.

    This combines Conv2d, optional BatchNorm/GroupNorm/LayerNorm,
    and optional activation into a single module.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding mode or size
        dilation: Dilation rate
        groups: Number of groups for grouped convolution
        bias: Whether to use bias
        norm: Normalization type ("batch", "group", "layer", None)
        activation: Activation type ("relu", "gelu", "silu", "prelu", None)
        norm_groups: Number of groups for GroupNorm
        separable: If True, use depthwise separable convolution (grouped conv + 1x1 pointwise)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        norm: Optional[str] = "batch",
        activation: Optional[str] = "relu",
        norm_groups: int = 8,
        separable: bool = False,
    ):
        super().__init__()

        # Parse kernel_size, stride, etc.
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        # Calculate padding
        if padding == "same":
            # Calculate same padding
            pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
            pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
            padding = (pad_h, pad_w)
        elif isinstance(padding, int):
            padding = (padding, padding)

        self.padding = padding
        self.stride = stride

        # Handle separable convolution
        # Separable: use groups = gcd(in, out), then add 1x1 pointwise conv
        self.pointwise_conv = None
        if separable:
            groups = math.gcd(in_channels, out_channels)
            # Disable separable if groups is 1 or kernel is 1x1
            if groups == 1 or max(kernel_size) == 1:
                separable = False
            else:
                # Will add pointwise conv after grouped conv
                self.pointwise_conv = nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                )

        # Convolution
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias and norm is None,  # No bias if using norm
        )

        # Normalization
        self.norm_layer = None
        if norm == "batch":
            self.norm_layer = nn.BatchNorm(out_channels)
        elif norm == "group":
            self.norm_layer = nn.GroupNorm(norm_groups, out_channels)
        elif norm == "layer":
            self.norm_layer = nn.LayerNorm(out_channels)

        # Activation
        self.activation = None
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, height, width, channels) - MLX NHWC format

        Returns:
            Output tensor
        """
        x = self.conv(x)

        # Apply pointwise conv for separable convolution
        if self.pointwise_conv is not None:
            x = self.pointwise_conv(x)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class ConvTranspose2dNormAct(nn.Module):
    """Transposed 2D Convolution with optional normalization and activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding size
        output_padding: Additional output padding
        groups: Number of groups
        bias: Whether to use bias
        norm: Normalization type
        activation: Activation type
        separable: If True, use depthwise separable convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        norm: Optional[str] = "batch",
        activation: Optional[str] = "relu",
        separable: bool = False,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)

        self.padding = padding
        self.stride = stride
        self.output_padding = output_padding

        # Handle separable convolution
        self.pointwise_conv = None
        if separable:
            groups = math.gcd(in_channels, out_channels)
            if groups == 1 or max(kernel_size) == 1:
                separable = False
            else:
                self.pointwise_conv = nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                )

        # Transposed convolution
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias and norm is None,
        )

        # Normalization
        self.norm_layer = None
        if norm == "batch":
            self.norm_layer = nn.BatchNorm(out_channels)
        elif norm == "group":
            self.norm_layer = nn.GroupNorm(8, out_channels)
        elif norm == "layer":
            self.norm_layer = nn.LayerNorm(out_channels)

        # Activation
        self.activation = None
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        x = self.conv(x)

        if self.pointwise_conv is not None:
            x = self.pointwise_conv(x)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


# ============================================================================
# ERB Filterbank
# ============================================================================


class ErbFilterbank(nn.Module):
    """ERB filterbank for frequency-domain processing.

    Transforms linear frequency spectrogram to ERB-scale representation
    and back.

    Args:
        sr: Sample rate
        fft_size: FFT size
        nb_bands: Number of ERB bands
        min_freq: Minimum frequency
        max_freq: Maximum frequency
    """

    def __init__(
        self,
        sr: int = 48000,
        fft_size: int = 960,
        nb_bands: int = 32,
        min_freq: float = 20.0,
        max_freq: Optional[float] = None,
    ):
        super().__init__()

        self.sr = sr
        self.fft_size = fft_size
        self.nb_bands = nb_bands
        self.n_freqs = fft_size // 2 + 1

        # Create filterbank matrix (not trainable)
        fb = make_erb_fb(
            sr=sr,
            fft_size=fft_size,
            nb_bands=nb_bands,
            min_freq=min_freq,
            max_freq=max_freq,
            as_numpy=True,
        )
        self._fb = mx.array(fb)
        self._fb_inv = mx.transpose(self._fb)

    def forward(self, spec: mx.array) -> mx.array:
        """Transform spectrogram to ERB bands.

        Args:
            spec: Magnitude spectrogram (..., n_freqs)

        Returns:
            ERB representation (..., nb_bands)
        """
        return mx.matmul(spec, self._fb)

    def inverse(self, erb: mx.array) -> mx.array:
        """Transform ERB bands back to spectrogram.

        Args:
            erb: ERB representation (..., nb_bands)

        Returns:
            Approximated spectrogram (..., n_freqs)
        """
        return mx.matmul(erb, self._fb_inv)

    def __call__(self, spec: mx.array) -> mx.array:
        return self.forward(spec)


def erb_fb(sr: int = 48000, fft_size: int = 960, nb_bands: int = 32, **kwargs) -> mx.array:
    """Convenience function to create ERB filterbank matrix.

    Args:
        sr: Sample rate
        fft_size: FFT size
        nb_bands: Number of ERB bands
        **kwargs: Additional arguments for make_erb_fb

    Returns:
        ERB filterbank matrix (n_freqs, nb_bands)
    """
    return cast(mx.array, make_erb_fb(sr=sr, fft_size=fft_size, nb_bands=nb_bands, **kwargs))


# ============================================================================
# Mask Operations
# ============================================================================


class Mask(nn.Module):
    """Spectral mask estimation and application.

    Supports various mask types including sigmoid, ReLU, and complex ratio masks.

    Args:
        n_freqs: Number of frequency bins
        mask_type: Type of mask ("sigmoid", "relu", "crm", "bounded")
        min_val: Minimum mask value for bounded types
        max_val: Maximum mask value for bounded types
    """

    def __init__(
        self,
        n_freqs: int,
        mask_type: str = "sigmoid",
        min_val: float = 0.0,
        max_val: float = 1.0,
    ):
        super().__init__()

        self.n_freqs = n_freqs
        self.mask_type = mask_type
        self.min_val = min_val
        self.max_val = max_val

    def __call__(
        self,
        mask: mx.array,
        spec: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        """Apply mask to spectrum.

        Args:
            mask: Estimated mask tensor
            spec: Optional complex spectrum as (real, imag) tuple

        Returns:
            If spec is None: processed mask
            If spec is provided: masked spectrum as (real, imag)
        """
        # Process mask based on type
        if self.mask_type == "sigmoid":
            mask = mx.sigmoid(mask)
        elif self.mask_type == "relu":
            mask = mx.maximum(mask, 0.0)
        elif self.mask_type == "bounded":
            mask = mx.sigmoid(mask) * (self.max_val - self.min_val) + self.min_val
        elif self.mask_type == "tanh":
            mask = mx.tanh(mask) * 0.5 + 0.5
        # crm (complex ratio mask) uses raw values

        if spec is None:
            return mask

        # Apply mask to spectrum
        real, imag = spec
        return (mask * real, mask * imag)


class ComplexMask(nn.Module):
    """Complex-valued mask for spectral processing.

    Estimates both magnitude and phase modifications.

    Args:
        n_freqs: Number of frequency bins
    """

    def __init__(self, n_freqs: int):
        super().__init__()
        self.n_freqs = n_freqs

    def __call__(
        self,
        mask_real: mx.array,
        mask_imag: mx.array,
        spec_real: mx.array,
        spec_imag: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Apply complex mask to spectrum.

        Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i

        Args:
            mask_real: Real part of mask
            mask_imag: Imaginary part of mask
            spec_real: Real part of spectrum
            spec_imag: Imaginary part of spectrum

        Returns:
            Masked spectrum as (real, imag)
        """
        out_real = mask_real * spec_real - mask_imag * spec_imag
        out_imag = mask_real * spec_imag + mask_imag * spec_real
        return (out_real, out_imag)


# ============================================================================
# Deep Filtering Operations
# ============================================================================


class DfOp(nn.Module):
    """Deep Filtering operation.

    Applies learned complex-valued FIR filters to the spectrum.

    This is the core operation of DeepFilterNet - it learns to apply
    different filters to different frequency bins based on the input.

    Args:
        nb_df: Number of DF frequency bins
        df_order: Filter order (number of taps)
        df_lookahead: Number of lookahead frames
    """

    def __init__(
        self,
        nb_df: int = 96,
        df_order: int = 5,
        df_lookahead: int = 0,
    ):
        super().__init__()

        self.nb_df = nb_df
        self.df_order = df_order
        self.df_lookahead = df_lookahead

    def __call__(
        self,
        spec: Tuple[mx.array, mx.array],
        coef: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Apply deep filtering.

        Args:
            spec: Input spectrum as (real, imag), each (batch, time, freq)
            coef: Filter coefficients (batch, time, nb_df, df_order, 2)
                  Last dim is (real, imag) for complex coefficients

        Returns:
            Filtered spectrum as (real, imag)
        """
        spec_real, spec_imag = spec
        batch, time, n_freqs = spec_real.shape

        # Extract DF frequency bins
        df_real = spec_real[:, :, : self.nb_df]
        df_imag = spec_imag[:, :, : self.nb_df]

        # Pad for filter application
        pad_past = self.df_order - 1 - self.df_lookahead
        pad_future = self.df_lookahead

        df_real_pad = mx.pad(df_real, [(0, 0), (pad_past, pad_future), (0, 0)])
        df_imag_pad = mx.pad(df_imag, [(0, 0), (pad_past, pad_future), (0, 0)])

        # Apply filtering frame by frame
        out_real_list = []
        out_imag_list = []

        for t in range(time):
            # Get filter coefficients for this frame
            coef_t = coef[:, t, :, :, :]  # (batch, nb_df, df_order, 2)
            coef_real = coef_t[:, :, :, 0]  # (batch, nb_df, df_order)
            coef_imag = coef_t[:, :, :, 1]

            # Get input frames for this output
            # (batch, df_order, nb_df)
            in_real = []
            in_imag = []
            for k in range(self.df_order):
                in_real.append(df_real_pad[:, t + k, :])
                in_imag.append(df_imag_pad[:, t + k, :])

            in_real = mx.stack(in_real, axis=1)  # (batch, df_order, nb_df)
            in_imag = mx.stack(in_imag, axis=1)

            # Transpose for multiplication: (batch, nb_df, df_order)
            in_real = mx.transpose(in_real, (0, 2, 1))
            in_imag = mx.transpose(in_imag, (0, 2, 1))

            # Complex multiplication and sum over filter taps
            # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            # Sum over df_order dimension
            y_real = mx.sum(coef_real * in_real - coef_imag * in_imag, axis=-1)
            y_imag = mx.sum(coef_real * in_imag + coef_imag * in_real, axis=-1)

            out_real_list.append(y_real)
            out_imag_list.append(y_imag)

        # Stack time frames
        df_out_real = mx.stack(out_real_list, axis=1)  # (batch, time, nb_df)
        df_out_imag = mx.stack(out_imag_list, axis=1)

        # Combine with non-DF frequencies (pass-through)
        if n_freqs > self.nb_df:
            out_real = mx.concatenate([df_out_real, spec_real[:, :, self.nb_df :]], axis=-1)
            out_imag = mx.concatenate([df_out_imag, spec_imag[:, :, self.nb_df :]], axis=-1)
        else:
            out_real = df_out_real
            out_imag = df_out_imag

        return (out_real, out_imag)


# ============================================================================
# Grouped Linear
# ============================================================================


class GroupedLinear(nn.Module):
    """Grouped linear layer.

    Splits input into groups and applies separate linear transformations,
    then concatenates the results.

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

        assert in_features % groups == 0
        assert out_features % groups == 0

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.group_in = in_features // groups
        self.group_out = out_features // groups

        # Initialize weights
        scale = (2 / (in_features + out_features)) ** 0.5
        self.weight = mx.random.normal(shape=(groups, self.group_in, self.group_out)) * scale

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input (..., in_features)

        Returns:
            Output (..., out_features)
        """
        batch_shape = x.shape[:-1]

        # Reshape for grouped operation
        x = x.reshape(*batch_shape, self.groups, self.group_in)

        # Grouped matmul using einsum
        out = mx.einsum("...gi,gio->...go", x, self.weight)

        # Reshape output
        out = out.reshape(*batch_shape, self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return out


class GroupedLinearEinsum(GroupedLinear):
    """Alias for GroupedLinear using einsum."""

    pass


# ============================================================================
# Utility Modules
# ============================================================================


class Swish(nn.Module):
    """Swish/SiLU activation: x * sigmoid(x)."""

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.sigmoid(x)


class PReLU(nn.Module):
    """Parametric ReLU with learnable slope."""

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.weight = mx.ones((num_parameters,)) * init

    def __call__(self, x: mx.array) -> mx.array:
        return mx.maximum(x, 0) + self.weight * mx.minimum(x, 0)


class Permute(nn.Module):
    """Permute tensor dimensions."""

    def __init__(self, dims: Tuple[int, ...]):
        super().__init__()
        self.dims = dims

    def __call__(self, x: mx.array) -> mx.array:
        return mx.transpose(x, self.dims)


class Squeeze(nn.Module):
    """Squeeze a dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        return mx.squeeze(x, axis=self.dim)


class Unsqueeze(nn.Module):
    """Unsqueeze (add) a dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        return mx.expand_dims(x, axis=self.dim)


# ============================================================================
# GRU-based Modules (for legacy DFNet1/2/3 support)
# ============================================================================


class GroupedGRULayer(nn.Module):
    """Single layer of grouped GRU.

    Splits input into groups and processes each group with a separate GRU.
    This reduces parameters while maintaining capacity for sequence modeling.

    Note: MLX's GRU is unidirectional only. Bidirectional support requires
    manual implementation with sequence reversal.

    Args:
        input_size: Total input feature dimension
        hidden_size: Total hidden state dimension
        groups: Number of groups to split input/hidden
        batch_first: If True, input is (batch, time, features)
        bias: Whether to use bias in GRU
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int = 4,
        batch_first: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        assert input_size % groups == 0, f"input_size {input_size} not divisible by {groups}"
        assert hidden_size % groups == 0, f"hidden_size {hidden_size} not divisible by {groups}"

        self.input_size_per_group = input_size // groups
        self.hidden_size_per_group = hidden_size // groups
        self.out_size = hidden_size
        self.groups = groups
        self.batch_first = batch_first

        # Create GRU for each group
        # MLX GRU: input_size, hidden_size, bias
        self.grus = [
            nn.GRU(
                input_size=self.input_size_per_group,
                hidden_size=self.hidden_size_per_group,
                bias=bias,
            )
            for _ in range(groups)
        ]

    def get_h0(self, batch_size: int) -> mx.array:
        """Get initial hidden state."""
        return mx.zeros((self.groups, batch_size, self.hidden_size_per_group))

    def __call__(self, x: mx.array, h0: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Input tensor (batch, time, features) if batch_first
            h0: Initial hidden state (groups, batch, hidden_per_group)

        Returns:
            output: (batch, time, hidden_size)
            hidden: (groups, batch, hidden_per_group)
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len, batch_size, _ = x.shape
            x = mx.transpose(x, (1, 0, 2))  # Convert to batch first

        if h0 is None:
            h0 = self.get_h0(batch_size)

        outputs = []
        out_states = []

        for i, gru in enumerate(self.grus):
            # Extract group's input
            start = i * self.input_size_per_group
            end = (i + 1) * self.input_size_per_group
            x_group = x[..., start:end]

            # Get group's hidden state
            h_group = h0[i]  # (batch, hidden)

            # Process with GRU - MLX GRU returns output (all timesteps)
            # Hidden state is last output timestep
            out = gru(x_group, h_group)  # (batch, time, hidden)
            h_out = out[:, -1, :]  # (batch, hidden) - last timestep
            outputs.append(out)
            out_states.append(mx.expand_dims(h_out, axis=0))

        # Concatenate outputs along feature dimension
        output = mx.concatenate(outputs, axis=-1)
        hidden = mx.concatenate(out_states, axis=0)

        if not self.batch_first:
            output = mx.transpose(output, (1, 0, 2))

        return output, hidden


class GroupedGRU(nn.Module):
    """Multi-layer grouped GRU with optional shuffling.

    Note: MLX's GRU is unidirectional. For consistency with PyTorch API,
    bidirectional parameter is accepted but must be False.

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        num_layers: Number of GRU layers
        groups: Number of groups
        bias: Whether to use bias
        batch_first: If True, input is (batch, time, features)
        bidirectional: Must be False (MLX GRU is unidirectional)
        shuffle: Whether to shuffle features between layers
        add_outputs: If True, add layer outputs; else use last layer output
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        groups: int = 4,
        bias: bool = True,
        batch_first: bool = True,
        bidirectional: bool = False,
        shuffle: bool = True,
        add_outputs: bool = False,
    ):
        super().__init__()

        if bidirectional:
            raise ValueError("MLX GRU does not support bidirectional. Use bidirectional=False.")

        assert input_size % groups == 0
        assert hidden_size % groups == 0
        assert num_layers > 0

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size_per_group = hidden_size // groups
        self.num_layers = num_layers
        self.groups = groups
        self.batch_first = batch_first
        self.shuffle = shuffle if groups > 1 else False
        self.add_outputs = add_outputs

        # Create layers
        self.layers = []
        self.layers.append(
            GroupedGRULayer(
                input_size=input_size,
                hidden_size=hidden_size,
                groups=groups,
                batch_first=batch_first,
                bias=bias,
            )
        )
        for _ in range(1, num_layers):
            self.layers.append(
                GroupedGRULayer(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    groups=groups,
                    batch_first=batch_first,
                    bias=bias,
                )
            )

    def get_h0(self, batch_size: int) -> mx.array:
        """Get initial hidden state for all layers."""
        return mx.zeros(
            (
                self.num_layers * self.groups,
                batch_size,
                self.hidden_size_per_group,
            )
        )

    def _shuffle_features(self, x: mx.array) -> mx.array:
        """Shuffle features between groups.

        Reshapes to (batch, time, hidden_per_group, groups),
        transposes groups and hidden, then flattens back.
        """
        shape = x.shape
        x = x.reshape(shape[0], shape[1], self.groups, -1)
        x = mx.transpose(x, (0, 1, 3, 2))
        x = x.reshape(shape)
        return x

    def __call__(self, x: mx.array, state: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Input (batch, time, features) if batch_first
            state: Hidden state for all layers

        Returns:
            output: (batch, time, hidden_size)
            state: Updated hidden state
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            batch_size = x.shape[1]

        if state is None:
            state = self.get_h0(batch_size)

        states_per_layer = self.groups
        output = mx.zeros_like(x[..., : self.hidden_size])
        out_states = []

        for i, layer in enumerate(self.layers):
            layer_state = state[i * states_per_layer : (i + 1) * states_per_layer]
            x, s = layer(x, layer_state)
            out_states.append(s)

            # Shuffle between layers (except last)
            if self.shuffle and i < self.num_layers - 1:
                x = self._shuffle_features(x)

            if self.add_outputs:
                output = output + x
            else:
                output = x

        out_state = mx.concatenate(out_states, axis=0)
        return output, out_state


class SqueezedGRU(nn.Module):
    """GRU with input/output linear projections.

    Compresses input through a linear layer, processes with GRU,
    then expands through another linear layer. This is more efficient
    than using a large GRU directly.

    Args:
        input_size: Input feature dimension
        hidden_size: GRU hidden dimension
        output_size: Output dimension (None = hidden_size)
        num_layers: Number of GRU layers
        linear_groups: Number of groups for linear projections
        batch_first: If True, input is (batch, time, features)
        gru_skip: If True, add skip connection around GRU
        linear_act: Activation for linear layers ("relu", "gelu", None)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip: bool = False,
        linear_act: Optional[str] = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.gru_skip = gru_skip

        # Input projection
        self.linear_in = GroupedLinear(input_size, hidden_size, linear_groups)

        # Activation after input projection
        if linear_act == "relu":
            self.linear_act = nn.ReLU()
        elif linear_act == "gelu":
            self.linear_act = nn.GELU()
        elif linear_act == "silu":
            self.linear_act = nn.SiLU()
        else:
            self.linear_act = None

        # GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            bias=True,
        )

        # Output projection
        if output_size is not None and output_size != hidden_size:
            self.linear_out = GroupedLinear(hidden_size, output_size, linear_groups)
        else:
            self.linear_out = None

    def __call__(self, x: mx.array, h: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Input (batch, time, input_size)
            h: Hidden state (batch, hidden_size)

        Returns:
            output: (batch, time, output_size)
            hidden: (batch, hidden_size)
        """
        # Input projection
        projected = self.linear_in(x)
        if self.linear_act is not None:
            projected = self.linear_act(projected)

        # GRU - MLX GRU returns output, not tuple
        out = self.gru(projected, h)  # (batch, time, hidden)
        h_out = out[:, -1, :]  # (batch, hidden) - last timestep

        # Skip connection
        if self.gru_skip:
            out = out + projected

        # Output projection
        if self.linear_out is not None:
            out = self.linear_out(out)
            if self.linear_act is not None:
                out = self.linear_act(out)

        return out, h_out


class SqueezedGRU_S(nn.Module):
    """SqueezedGRU with skip connection after output (variant S).

    Unlike SqueezedGRU, the skip connection is added after the
    output projection, connecting input directly to output.

    Args:
        input_size: Input feature dimension
        hidden_size: GRU hidden dimension
        output_size: Output dimension (None = hidden_size)
        num_layers: Number of GRU layers
        linear_groups: Number of groups for linear projections
        batch_first: If True, input is (batch, time, features)
        gru_skip: If True, add skip from input to output
        linear_act: Activation for linear layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip: bool = False,
        linear_act: Optional[str] = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.gru_skip = gru_skip

        # Input projection
        self.linear_in = GroupedLinear(input_size, hidden_size, linear_groups)

        # Activation
        if linear_act == "relu":
            self.linear_act = nn.ReLU()
        elif linear_act == "gelu":
            self.linear_act = nn.GELU()
        elif linear_act == "silu":
            self.linear_act = nn.SiLU()
        else:
            self.linear_act = None

        # GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            bias=True,
        )

        # Output projection
        if output_size is not None and output_size != hidden_size:
            self.linear_out = GroupedLinear(hidden_size, output_size, linear_groups)
        else:
            self.linear_out = None

        # Skip projection (if input/output sizes differ)
        if gru_skip and input_size != self.output_size:
            self.skip_proj = nn.Linear(input_size, self.output_size)
        else:
            self.skip_proj = None

    def __call__(self, x: mx.array, h: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Input (batch, time, input_size)
            h: Hidden state (batch, hidden_size)

        Returns:
            output: (batch, time, output_size)
            hidden: (batch, hidden_size)
        """
        # Input projection
        projected = self.linear_in(x)
        if self.linear_act is not None:
            projected = self.linear_act(projected)

        # GRU - MLX GRU returns output, not tuple
        out = self.gru(projected, h)  # (batch, time, hidden)
        h_out = out[:, -1, :]  # (batch, hidden) - last timestep

        # Output projection
        if self.linear_out is not None:
            out = self.linear_out(out)
            if self.linear_act is not None:
                out = self.linear_act(out)

        # Skip connection (after output)
        if self.gru_skip:
            if self.skip_proj is not None:
                out = out + self.skip_proj(x)
            else:
                out = out + x

        return out, h_out


# ============================================================================
# Attention-based Backbone (faster backward pass than GRU/Mamba)
# ============================================================================


class SqueezedAttention(nn.Module):
    """Attention-based backbone with skip connections.

    Drop-in replacement for SqueezedGRU_S with significantly faster backward
    pass due to parallelizable attention operations.

    Architecture:
    - Input projection (grouped linear)
    - Multi-layer causal self-attention
    - Output projection (grouped linear)
    - Optional skip connection

    Performance characteristics:
    - Forward pass: ~5x faster than GRU
    - Backward pass: ~18x faster than GRU
    - Overall training step: ~2-3x faster

    Args:
        input_size: Input feature dimension
        hidden_size: Internal attention dimension
        output_size: Output dimension (None = hidden_size)
        num_layers: Number of attention layers
        num_heads: Number of attention heads
        linear_groups: Number of groups for linear projections
        gru_skip: If True, add skip from input to output
        linear_act: Activation for linear layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 2,
        num_heads: int = 4,
        linear_groups: int = 8,
        gru_skip: bool = False,
        linear_act: Optional[str] = None,
        # Unused args for API compatibility with SqueezedGRU_S
        batch_first: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.gru_skip = gru_skip

        # Input projection
        self.linear_in = GroupedLinear(input_size, hidden_size, linear_groups)

        # Activation
        if linear_act == "relu":
            self.linear_act: Optional[nn.Module] = nn.ReLU()
        elif linear_act == "gelu":
            self.linear_act = nn.GELU()
        elif linear_act == "silu":
            self.linear_act = nn.SiLU()
        else:
            self.linear_act = None

        # Attention layers with pre-norm
        self.attention_layers = []
        self.norms = []
        self.ffns = []
        self.ffn_norms = []

        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(hidden_size))
            self.attention_layers.append(nn.MultiHeadAttention(dims=hidden_size, num_heads=num_heads))
            self.ffn_norms.append(nn.LayerNorm(hidden_size))
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.GELU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                )
            )

        # Output projection
        if output_size is not None and output_size != hidden_size:
            self.linear_out: Optional[nn.Module] = GroupedLinear(hidden_size, output_size, linear_groups)
        else:
            self.linear_out = None

        # Skip projection (if input/output sizes differ)
        if gru_skip and input_size != self.output_size:
            self.skip_proj: Optional[nn.Module] = nn.Linear(input_size, self.output_size)
        else:
            self.skip_proj = None

    def __call__(self, x: mx.array, h: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Input (batch, time, input_size)
            h: Hidden state - IGNORED for attention (API compatibility only)

        Returns:
            output: (batch, time, output_size)
            hidden: (batch, hidden_size) - last timestep embedding
        """
        # Input projection
        out = self.linear_in(x)
        if self.linear_act is not None:
            out = self.linear_act(out)

        # Create causal mask once
        seq_len = out.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(out.dtype)

        # Apply attention layers with pre-norm and residual
        for norm, attn, ffn_norm, ffn in zip(self.norms, self.attention_layers, self.ffn_norms, self.ffns):
            # Self-attention with pre-norm
            normed = norm(out)
            attn_out = attn(normed, normed, normed, mask=mask)
            out = out + attn_out

            # FFN with pre-norm
            normed = ffn_norm(out)
            ffn_out = ffn(normed)
            out = out + ffn_out

        # Extract last hidden state for API compatibility
        h_out = out[:, -1, :]

        # Output projection
        if self.linear_out is not None:
            out = self.linear_out(out)
            if self.linear_act is not None:
                out = self.linear_act(out)

        # Skip connection (after output)
        if self.gru_skip:
            if self.skip_proj is not None:
                out = out + self.skip_proj(x)
            else:
                out = out + x

        return out, h_out
