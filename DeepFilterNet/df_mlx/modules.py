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
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.padding = padding
        self.stride = stride

        # Transposed convolution
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
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
