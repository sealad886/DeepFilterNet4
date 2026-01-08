"""Core operations for MLX DeepFilterNet4.

This module provides MLX implementations of core audio and signal processing
operations used in DeepFilterNet4, including:
- STFT/iSTFT with configurable parameters
- ERB filterbank generation
- Complex number operations
- Spectral operations

All operations are optimized for Apple Silicon unified memory architecture.
"""

import math
from functools import lru_cache
from typing import Optional, Tuple, Union

import mlx.core as mx
import numpy as np

# ============================================================================
# Window Functions
# ============================================================================


@lru_cache(maxsize=8)
def get_window(window_type: str, window_length: int) -> mx.array:
    """Get a window function as MLX array.

    Args:
        window_type: Type of window ("hann", "hamming", "blackman", "sqrt_hann")
        window_length: Length of the window

    Returns:
        MLX array containing the window
    """
    if window_type == "hann":
        window = np.hanning(window_length).astype(np.float32)
    elif window_type == "hamming":
        window = np.hamming(window_length).astype(np.float32)
    elif window_type == "blackman":
        window = np.blackman(window_length).astype(np.float32)
    elif window_type == "sqrt_hann":
        window = np.sqrt(np.hanning(window_length)).astype(np.float32)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    return mx.array(window)


# ============================================================================
# STFT / iSTFT
# ============================================================================


def stft(
    x: mx.array,
    n_fft: int = 960,
    hop_length: int = 480,
    win_length: Optional[int] = None,
    window: str = "sqrt_hann",
    center: bool = True,
    return_complex: bool = True,
) -> mx.array | tuple[mx.array, mx.array]:
    """Short-Time Fourier Transform.

    Args:
        x: Input audio signal of shape (batch, samples) or (samples,)
        n_fft: FFT size
        hop_length: Hop size between frames
        win_length: Window length (defaults to n_fft)
        window: Window type ("hann", "hamming", "blackman", "sqrt_hann")
        center: Whether to center-pad the signal
        return_complex: If True, return as (real, imag) tuple; if False,
                       return stacked (batch, time, freq, 2)

    Returns:
        Complex STFT output as (real, imag) tuple or stacked array
    """
    if win_length is None:
        win_length = n_fft

    # Handle input dimensions
    input_1d = x.ndim == 1
    if input_1d:
        x = mx.expand_dims(x, axis=0)

    # Center pad if requested
    if center:
        pad_amount = n_fft // 2
        # MLX doesn't support reflect padding, use constant padding instead
        # This is a simplification - for production, implement proper reflect padding
        x = mx.pad(x, [(0, 0), (pad_amount, pad_amount)])

    # Get window
    win = get_window(window, win_length)
    if win_length < n_fft:
        # Pad window to n_fft
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        win = mx.pad(win, [(pad_left, pad_right)])

    # Frame the signal
    num_samples = x.shape[1]
    num_frames = (num_samples - n_fft) // hop_length + 1

    # Create frames using striding (manually, since MLX doesn't have as_strided)
    frames = []
    for i in range(num_frames):
        start = i * hop_length
        frames.append(x[:, start : start + n_fft])
    frames = mx.stack(frames, axis=1)  # (batch, frames, n_fft)

    # Apply window
    frames = frames * win

    # Compute FFT
    # MLX FFT returns complex - we'll use rfft for real input
    fft_out = mx.fft.rfft(frames, axis=-1)

    # Extract real and imaginary parts
    real = mx.real(fft_out)
    imag = mx.imag(fft_out)

    if input_1d:
        real = mx.squeeze(real, axis=0)
        imag = mx.squeeze(imag, axis=0)

    if return_complex:
        return (real, imag)
    else:
        # Stack as (..., freq, 2)
        return mx.stack([real, imag], axis=-1)


def istft(
    spec: Union[Tuple[mx.array, mx.array], mx.array],
    n_fft: int = 960,
    hop_length: int = 480,
    win_length: Optional[int] = None,
    window: str = "sqrt_hann",
    center: bool = True,
    length: Optional[int] = None,
) -> mx.array:
    """Inverse Short-Time Fourier Transform.

    Args:
        spec: Complex spectrum as (real, imag) tuple or stacked (..., freq, 2)
        n_fft: FFT size
        hop_length: Hop size
        win_length: Window length
        window: Window type
        center: Whether signal was center-padded
        length: Desired output length

    Returns:
        Reconstructed audio signal
    """
    if win_length is None:
        win_length = n_fft

    # Parse input format
    if isinstance(spec, tuple):
        real, imag = spec
    else:
        real = spec[..., 0]
        imag = spec[..., 1]

    # Handle dimensions
    input_1d = real.ndim == 2
    if input_1d:
        real = mx.expand_dims(real, axis=0)
        imag = mx.expand_dims(imag, axis=0)

    batch_size, num_frames, num_freqs = real.shape

    # Construct complex spectrum
    complex_spec = real + 1j * imag

    # Inverse FFT
    frames = mx.fft.irfft(complex_spec, n=n_fft, axis=-1)

    # Get synthesis window
    win = get_window(window, win_length)
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        win = mx.pad(win, [(pad_left, pad_right)])

    # Apply window
    frames = frames * win

    # Overlap-add
    output_length = (num_frames - 1) * hop_length + n_fft
    output = mx.zeros((batch_size, output_length))
    window_sum = mx.zeros((output_length,))

    # Manual overlap-add (MLX doesn't have scatter_add)
    for i in range(num_frames):
        start = i * hop_length
        output = output.at[:, start : start + n_fft].add(frames[:, i, :])
        window_sum = window_sum.at[start : start + n_fft].add(win * win)

    # Normalize by window sum
    window_sum = mx.maximum(window_sum, 1e-8)
    output = output / window_sum

    # Remove center padding
    if center:
        pad_amount = n_fft // 2
        output = output[:, pad_amount:-pad_amount]

    # Trim to desired length
    if length is not None:
        output = output[:, :length]

    if input_1d:
        output = mx.squeeze(output, axis=0)

    return output


# ============================================================================
# ERB Filterbank
# ============================================================================


def erb_frequency(freq: float) -> float:
    """Convert frequency in Hz to ERB scale."""
    return 21.4 * math.log10(1 + freq / 229)


def erb_inv(erb: float) -> float:
    """Convert ERB scale to frequency in Hz."""
    return 229 * (10 ** (erb / 21.4) - 1)


@lru_cache(maxsize=4)
def erb_fb(
    sr: int,
    fft_size: int,
    nb_bands: int = 32,
    min_freq: float = 20.0,
    max_freq: Optional[float] = None,
    min_width: int = 2,
    normalized: bool = True,
    as_numpy: bool = False,
) -> Union[mx.array, np.ndarray]:
    """Generate ERB-scale filterbank matrix.

    Creates a filterbank matrix that transforms FFT bins to ERB bands.

    Args:
        sr: Sample rate in Hz
        fft_size: FFT size
        nb_bands: Number of ERB bands
        min_freq: Minimum frequency in Hz
        max_freq: Maximum frequency in Hz (defaults to sr/2)
        min_width: Minimum filter width in FFT bins
        normalized: Whether to normalize each filter to sum to 1
        as_numpy: Return as numpy array instead of MLX array

    Returns:
        Filterbank matrix of shape (n_freqs, nb_bands)
    """
    if max_freq is None:
        max_freq = sr / 2

    n_freqs = fft_size // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)

    # ERB scale boundaries
    erb_low = erb_frequency(min_freq)
    erb_high = erb_frequency(max_freq)
    erb_centers = np.linspace(erb_low, erb_high, nb_bands + 2)
    center_freqs = np.array([erb_inv(e) for e in erb_centers])

    # Build filterbank
    fb = np.zeros((n_freqs, nb_bands), dtype=np.float32)

    for i in range(nb_bands):
        f_low = center_freqs[i]
        f_center = center_freqs[i + 1]
        f_high = center_freqs[i + 2]

        # Rising slope
        rising = (freqs - f_low) / max(f_center - f_low, 1e-8)
        rising = np.clip(rising, 0, 1)

        # Falling slope
        falling = (f_high - freqs) / max(f_high - f_center, 1e-8)
        falling = np.clip(falling, 0, 1)

        # Triangular filter
        fb[:, i] = np.minimum(rising, falling)

        # Ensure minimum width
        if np.sum(fb[:, i] > 0) < min_width:
            center_idx = np.argmin(np.abs(freqs - f_center))
            start_idx = max(0, center_idx - min_width // 2)
            end_idx = min(n_freqs, start_idx + min_width)
            fb[start_idx:end_idx, i] = 1.0

        # Normalize
        if normalized and np.sum(fb[:, i]) > 0:
            fb[:, i] /= np.sum(fb[:, i])

    if as_numpy:
        return fb
    return mx.array(fb)


def erb_fb_and_inverse(
    sr: int,
    fft_size: int,
    nb_bands: int = 32,
    min_freq: float = 20.0,
    max_freq: Optional[float] = None,
    min_width: int = 2,
    normalized: bool = True,
) -> Tuple[mx.array, mx.array]:
    """Generate ERB filterbank and its inverse (transpose).

    Convenience function that returns both the forward and inverse
    ERB filterbank matrices needed by DFNetMF and other models.

    Args:
        sr: Sample rate in Hz
        fft_size: FFT size
        nb_bands: Number of ERB bands
        min_freq: Minimum frequency in Hz
        max_freq: Maximum frequency in Hz (defaults to sr/2)
        min_width: Minimum filter width in FFT bins
        normalized: Whether to normalize each filter to sum to 1

    Returns:
        Tuple of (erb_fb, erb_inv_fb):
            - erb_fb: Forward filterbank [n_freqs, nb_bands]
            - erb_inv_fb: Inverse filterbank [nb_bands, n_freqs]
    """
    fb = erb_fb(sr, fft_size, nb_bands, min_freq, max_freq, min_width, normalized, as_numpy=False)
    fb_mx = fb if isinstance(fb, mx.array) else mx.array(fb)
    return fb_mx, mx.transpose(fb_mx)


def erb_transform(spec: mx.array, fb: mx.array) -> mx.array:
    """Transform spectrogram to ERB bands.

    Args:
        spec: Magnitude spectrogram of shape (..., n_freqs)
        fb: ERB filterbank matrix of shape (n_freqs, nb_bands)

    Returns:
        ERB features of shape (..., nb_bands)
    """
    return mx.matmul(spec, fb)


def erb_inv_transform(erb_spec: mx.array, fb: mx.array) -> mx.array:
    """Inverse transform from ERB bands to spectrogram.

    Args:
        erb_spec: ERB features of shape (..., nb_bands)
        fb: ERB filterbank matrix of shape (n_freqs, nb_bands)

    Returns:
        Approximated spectrogram of shape (..., n_freqs)
    """
    # Compute pseudo-inverse of filterbank
    # fb_pinv = (fb.T @ fb)^-1 @ fb.T
    # For triangular filters, simple transpose often works well
    return mx.matmul(erb_spec, mx.transpose(fb))


# ============================================================================
# Complex Number Operations
# ============================================================================


def complex_mul(a: Tuple[mx.array, mx.array], b: Tuple[mx.array, mx.array]) -> Tuple[mx.array, mx.array]:
    """Complex multiplication of (real, imag) tuples.

    (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    """
    ar, ai = a
    br, bi = b
    return (ar * br - ai * bi, ar * bi + ai * br)


def complex_conj(x: Tuple[mx.array, mx.array]) -> Tuple[mx.array, mx.array]:
    """Complex conjugate of (real, imag) tuple."""
    return (x[0], -x[1])


def complex_abs(x: Tuple[mx.array, mx.array]) -> mx.array:
    """Complex absolute value (magnitude)."""
    return mx.sqrt(x[0] ** 2 + x[1] ** 2)


def complex_abs_squared(x: Tuple[mx.array, mx.array]) -> mx.array:
    """Complex absolute value squared (power)."""
    return x[0] ** 2 + x[1] ** 2


def complex_to_polar(x: Tuple[mx.array, mx.array]) -> Tuple[mx.array, mx.array]:
    """Convert complex (real, imag) to polar (magnitude, phase)."""
    mag = complex_abs(x)
    phase = mx.arctan2(x[1], x[0])
    return (mag, phase)


def polar_to_complex(mag: mx.array, phase: mx.array) -> Tuple[mx.array, mx.array]:
    """Convert polar (magnitude, phase) to complex (real, imag)."""
    return (mag * mx.cos(phase), mag * mx.sin(phase))


# ============================================================================
# Spectral Operations
# ============================================================================


def spec_pad(
    spec: mx.array,
    df_order: int,
    df_lookahead: int,
    dim: int = 1,
) -> mx.array:
    """Pad spectrogram for DF processing.

    Pads the time dimension to account for DF filter order and lookahead.

    Args:
        spec: Input spectrogram
        df_order: DF filter order
        df_lookahead: Number of lookahead frames
        dim: Time dimension to pad

    Returns:
        Padded spectrogram
    """
    pad_past = df_order - 1 - df_lookahead
    pad_future = df_lookahead

    # Build pad specification
    ndim = spec.ndim
    pad_spec = [(0, 0)] * ndim
    pad_spec[dim] = (pad_past, pad_future)

    return mx.pad(spec, pad_spec, mode="constant")


def as_strided_frames(
    x: mx.array,
    frame_length: int,
    hop_length: int,
    axis: int = -2,
) -> mx.array:
    """Extract overlapping frames from a tensor.

    This is a simplified version that works with MLX's limitations.

    Args:
        x: Input tensor
        frame_length: Length of each frame
        hop_length: Hop between frames
        axis: Axis along which to extract frames

    Returns:
        Tensor with an additional dimension for frame contents
    """
    # Get shape info
    shape = list(x.shape)
    axis = axis % len(shape)
    length = shape[axis]

    num_frames = (length - frame_length) // hop_length + 1

    # Extract frames
    frames = []
    for i in range(num_frames):
        start = i * hop_length
        # Build slicing
        slices = [slice(None)] * len(shape)
        slices[axis] = slice(start, start + frame_length)
        frames.append(x[tuple(slices)])

    # Stack along new axis
    result = mx.stack(frames, axis=axis)
    return result


# ============================================================================
# Normalization
# ============================================================================


def rms_normalize(
    x: mx.array,
    axis: int = -1,
    eps: float = 1e-8,
    target_rms: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """RMS normalization.

    Args:
        x: Input tensor
        axis: Axis along which to compute RMS
        eps: Small constant for numerical stability
        target_rms: Target RMS level

    Returns:
        Tuple of (normalized tensor, scaling factor)
    """
    rms = mx.sqrt(mx.mean(x**2, axis=axis, keepdims=True) + eps)
    scale = target_rms / (rms + eps)
    return x * scale, scale


def batch_rms(x: mx.array, axis: int = -1, eps: float = 1e-8) -> mx.array:
    """Compute batch-wise RMS."""
    return mx.sqrt(mx.mean(x**2, axis=axis, keepdims=True) + eps)
