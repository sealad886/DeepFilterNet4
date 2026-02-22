"""STOI (Short-Time Objective Intelligibility) metric for MLX.

Ported from df/stoi.py - provides a differentiable STOI implementation
for MLX that can be used during training validation.

Note: For final evaluation/reporting, use pystoi for reference results.
This implementation is optimized for development and fast feedback.

References:
- Original STOI: Taal et al., "A short-time objective intelligibility measure"
- pystoi: https://github.com/mpariente/pystoi
"""

from typing import List, Tuple

import mlx.core as mx
import numpy as np

EPS = float(np.finfo("float").eps)  # Cast to Python float for MLX compatibility


def thirdoct(
    fs: int,
    nfft: int,
    num_bands: int = 15,
    min_freq: float = 150.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1/3 octave band matrix and center frequencies.

    Args:
        fs: Sampling rate
        nfft: FFT size
        num_bands: Number of 1/3 octave bands
        min_freq: Center frequency of lowest band

    Returns:
        Tuple of (octave_band_matrix, center_frequencies)
    """
    f = np.linspace(0, fs, nfft + 1)
    f = f[: int(nfft / 2) + 1]

    k = np.arange(num_bands).astype(float)
    cf = np.power(2.0 ** (1.0 / 3), k) * min_freq
    freq_low = min_freq * np.power(2.0, (2 * k - 1) / 6)
    freq_high = min_freq * np.power(2.0, (2 * k + 1) / 6)

    obm = np.zeros((num_bands, len(f)))

    for i in range(len(cf)):
        # Match 1/3 octave band frequencies with FFT bins
        fl_ii = np.argmin(np.square(f - freq_low[i]))
        fh_ii = np.argmin(np.square(f - freq_high[i]))
        obm[i, fl_ii:fh_ii] = 1

    return obm, cf


def as_windowed(
    x: mx.array,
    window_length: int,
    step: int = 1,
) -> mx.array:
    """Create overlapping windows from a signal.

    Args:
        x: Input signal (time, ...) or (batch, time, ...)
        window_length: Length of each window
        step: Step between windows

    Returns:
        Windowed signal (num_windows, window_length, ...)
    """
    if x.ndim == 1:
        n_samples = x.shape[0]
        n_windows = (n_samples - window_length + step) // step
        # Use indexing to create windows
        indices = mx.arange(window_length)
        starts = mx.arange(n_windows) * step
        # Broadcast to get all window indices
        all_indices = starts[:, None] + indices[None, :]
        return x[all_indices]
    elif x.ndim == 2:
        # Batch processing
        batch, n_samples = x.shape
        n_windows = (n_samples - window_length + step) // step
        indices = mx.arange(window_length)
        starts = mx.arange(n_windows) * step
        all_indices = starts[:, None] + indices[None, :]
        # Gather for each batch item
        result = mx.stack([x[b][all_indices] for b in range(batch)], axis=0)
        return result
    else:
        raise ValueError(f"Expected 1D or 2D input, got shape {x.shape}")


def _stft_mlx(
    x: mx.array,
    win_size: int,
    fft_size: int,
    hop_size: int,
    normalized: bool = True,
) -> mx.array:
    """Compute STFT using MLX.

    Args:
        x: Input signal (samples,) or (batch, samples)
        win_size: Window size
        fft_size: FFT size
        hop_size: Hop size
        normalized: Whether to normalize by window sum

    Returns:
        Power spectrum (freq, time) or (batch, freq, time)
    """
    from .ops import stft

    # Handle dimensions
    squeeze_batch = False
    if x.ndim == 1:
        x = mx.expand_dims(x, axis=0)
        squeeze_batch = True

    # Compute STFT
    real, imag = stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_size)

    # Power spectrum
    power = real**2 + imag**2

    if squeeze_batch:
        power = mx.squeeze(power, axis=0)

    return power


def remove_silent_frames_mlx(
    x: mx.array,
    y: mx.array,
    dyn_range: float = 40.0,
    framelen: int = 256,
    hop: int = 128,
) -> Tuple[List[mx.array], List[mx.array]]:
    """Remove silent frames from paired signals.

    Identifies silent frames in the reference signal (x) and removes
    corresponding frames from both signals.

    Args:
        x: Reference signal (batch, samples)
        y: Degraded signal (batch, samples)
        dyn_range: Dynamic range threshold in dB
        framelen: Frame length in samples
        hop: Hop size in samples

    Returns:
        Tuple of (x_no_silent, y_no_silent) as lists of arrays per batch
    """
    if x.ndim == 1:
        x = mx.expand_dims(x, axis=0)
        y = mx.expand_dims(y, axis=0)

    batch = x.shape[0]

    # Pad to multiple of framelen
    samples = x.shape[1]
    pad = framelen - (samples % framelen) if samples % framelen != 0 else 0
    if pad > 0:
        x = mx.pad(x, [(0, 0), (0, pad)])
        y = mx.pad(y, [(0, 0), (0, pad)])

    # Create window
    window = mx.array(np.hanning(framelen + 2)[1:-1], dtype=x.dtype)

    x_no_sil = []
    y_no_sil = []

    for b in range(batch):
        # Create windowed frames
        x_frames = as_windowed(x[b], framelen, hop)  # (n_frames, framelen)
        y_frames = as_windowed(y[b], framelen, hop)

        # Apply window
        x_frames = x_frames * window
        y_frames = y_frames * window

        # Compute energies in dB
        frame_energy = mx.sqrt(mx.sum(x_frames**2, axis=-1)) / np.sqrt(framelen)
        energies_db = 20 * mx.log10(frame_energy + EPS)

        # Find mask for non-silent frames
        max_energy = mx.max(energies_db)
        mask = energies_db > (max_energy - dyn_range)

        # Apply mask - convert to numpy for indexing
        mask_np = np.array(mask)
        x_selected = x_frames[mask_np]
        y_selected = y_frames[mask_np]

        x_no_sil.append(x_selected)
        y_no_sil.append(y_selected)

    return x_no_sil, y_no_sil


def stoi(
    x: mx.array,
    y: mx.array,
    fs_source: int,
    extended: bool = False,
) -> mx.array:
    """Compute Short-Time Objective Intelligibility.

    This is a simplified MLX implementation for development/validation.
    For final evaluation, use pystoi.

    Args:
        x: Reference (clean) signal (batch, samples) or (samples,)
        y: Degraded signal (same shape as x)
        fs_source: Sample rate of input signals
        extended: Whether to use extended STOI (not implemented)

    Returns:
        STOI score per batch item (higher is better, range ~0-1)
    """
    if extended:
        raise NotImplementedError("Extended STOI not yet implemented for MLX")

    # Parameters matching pystoi
    fs = 10000  # Target sample rate
    dyn_range = 40
    N_frame = 256
    N_fft = 512
    N_bands = 15
    min_freq = 150
    N = 30  # Number of frames for correlation
    Beta = -15.0

    # Handle dimensions
    if x.ndim == 1:
        x = mx.expand_dims(x, axis=0)
        y = mx.expand_dims(y, axis=0)

    assert x.shape == y.shape, f"Shapes must match: {x.shape} vs {y.shape}"

    batch = x.shape[0]

    # Get octave band matrix
    obm, _ = thirdoct(fs, N_fft, N_bands, min_freq)
    obm = mx.array(obm)

    # Resample to 10kHz (simplified - use scipy for proper resampling)
    if fs_source != fs:
        # Simple decimation/interpolation
        ratio = fs_source / fs
        new_len = int(x.shape[1] / ratio)
        # Linear interpolation (simplified resampling)
        old_indices = mx.linspace(0, x.shape[1] - 1, new_len)
        old_indices_int = mx.floor(old_indices).astype(mx.int32)
        old_indices_int = mx.clip(old_indices_int, 0, x.shape[1] - 2)
        frac = old_indices - old_indices_int.astype(mx.float32)

        x_new = []
        y_new = []
        for b in range(batch):
            x_b = x[b][old_indices_int] * (1 - frac) + x[b][old_indices_int + 1] * frac
            y_b = y[b][old_indices_int] * (1 - frac) + y[b][old_indices_int + 1] * frac
            x_new.append(x_b)
            y_new.append(y_b)
        x = mx.stack(x_new)
        y = mx.stack(y_new)

    # Remove silent frames
    x_list, y_list = remove_silent_frames_mlx(x, y, dyn_range, N_frame, N_frame // 2)

    results = []

    for i in range(batch):
        x_i = x_list[i]
        y_i = y_list[i]

        if x_i.shape[0] < 2:
            # Not enough frames
            results.append(mx.array(0.0))
            continue

        # Compute power spectra from frames (already windowed)
        # For each frame, compute FFT
        n_frames = x_i.shape[0]

        # Zero-pad frames to FFT size
        if N_frame < N_fft:
            pad_amount = N_fft - N_frame
            x_i = mx.pad(x_i, [(0, 0), (0, pad_amount)])
            y_i = mx.pad(y_i, [(0, 0), (0, pad_amount)])

        # Compute FFT magnitude
        x_fft = mx.abs(mx.fft.rfft(x_i, axis=-1))
        y_fft = mx.abs(mx.fft.rfft(y_i, axis=-1))

        # Apply octave band matrix: (n_frames, n_freq) @ (n_bands, n_freq).T
        # Result: (n_frames, n_bands)
        x_bands = mx.matmul(x_fft, obm.T)
        y_bands = mx.matmul(y_fft, obm.T)

        # Segment into N-frame chunks
        if n_frames >= N:
            # Number of complete segments
            n_segments = n_frames - N + 1

            correlations = []
            for seg in range(n_segments):
                x_seg = x_bands[seg : seg + N]  # (N, n_bands)
                y_seg = y_bands[seg : seg + N]

                # Normalize degraded signal per segment
                x_norm = mx.sqrt(mx.sum(x_seg**2, axis=0, keepdims=True) + EPS)
                y_norm = mx.sqrt(mx.sum(y_seg**2, axis=0, keepdims=True) + EPS)

                # Clipping (eq. 3 in paper)
                c = 10 ** (-Beta / 20)
                y_seg_clipped = mx.minimum(y_seg, x_seg * (1 + c))

                # Normalize and clip
                y_seg_norm = y_seg_clipped * x_norm / (y_norm + EPS)

                # Subtract mean
                x_seg_zm = x_seg - mx.mean(x_seg, axis=0, keepdims=True)
                y_seg_zm = y_seg_norm - mx.mean(y_seg_norm, axis=0, keepdims=True)

                # Normalize by std
                x_seg_n = x_seg_zm / (mx.sqrt(mx.sum(x_seg_zm**2, axis=0, keepdims=True)) + EPS)
                y_seg_n = y_seg_zm / (mx.sqrt(mx.sum(y_seg_zm**2, axis=0, keepdims=True)) + EPS)

                # Correlation
                corr = mx.sum(x_seg_n * y_seg_n)
                correlations.append(corr)

            # Average correlation over all segments and bands
            avg_corr = mx.mean(mx.stack(correlations)) / N_bands
            results.append(avg_corr)
        else:
            # Too few frames - compute simple correlation
            x_zm = x_bands - mx.mean(x_bands, axis=0, keepdims=True)
            y_zm = y_bands - mx.mean(y_bands, axis=0, keepdims=True)
            x_n = x_zm / (mx.sqrt(mx.sum(x_zm**2, axis=0, keepdims=True)) + EPS)
            y_n = y_zm / (mx.sqrt(mx.sum(y_zm**2, axis=0, keepdims=True)) + EPS)
            corr = mx.mean(mx.sum(x_n * y_n, axis=0) / n_frames)
            results.append(corr)

    return mx.stack(results)


def stoi_numpy(
    x: np.ndarray,
    y: np.ndarray,
    fs: int,
    extended: bool = False,
) -> float:
    """Compute STOI using pystoi (if available) or numpy fallback.

    This is the recommended function for final evaluation.

    Args:
        x: Reference signal (samples,)
        y: Degraded signal (samples,)
        fs: Sample rate
        extended: Use extended STOI

    Returns:
        STOI score (single value)
    """
    try:
        from pystoi import stoi as pystoi_fn

        return pystoi_fn(x, y, fs, extended=extended)
    except ImportError:
        # Use our MLX implementation converted to numpy
        x_mx = mx.array(x)
        y_mx = mx.array(y)
        result = stoi(x_mx, y_mx, fs, extended=extended)
        return float(result[0])


# ============================================================================
# Batch Processing Utilities
# ============================================================================


def batch_stoi(
    references: List[np.ndarray],
    degraded: List[np.ndarray],
    fs: int,
    extended: bool = False,
) -> List[float]:
    """Compute STOI for a batch of audio pairs.

    Args:
        references: List of reference (clean) signals
        degraded: List of degraded signals
        fs: Sample rate
        extended: Use extended STOI

    Returns:
        List of STOI scores
    """
    results = []
    for ref, deg in zip(references, degraded):
        score = stoi_numpy(ref, deg, fs, extended=extended)
        results.append(score)
    return results


def stoi_loss(
    pred: mx.array,
    target: mx.array,
    fs: int,
) -> mx.array:
    """STOI-based loss for training (1 - STOI).

    Args:
        pred: Predicted signal (batch, samples)
        target: Target signal (batch, samples)
        fs: Sample rate

    Returns:
        Loss value (lower is better)
    """
    scores = stoi(target, pred, fs)
    return 1.0 - mx.mean(scores)
