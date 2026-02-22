"""Custom Metal kernels with differentiable VJP for DfOp and other hotspots.

This module provides fused GPU kernels via ``mx.fast.metal_kernel`` wrapped
with ``mx.custom_function`` so that each kernel has a proper VJP (backward
pass) and can be used inside ``nn.value_and_grad`` during training.

Forward passes use the Metal kernel for speed; backward passes use
pure-MLX ops derived from the mathematical chain rule.

A runtime availability check (``metal_kernels_available``) lets callers
fall back to pure-MLX when the API is absent.
"""

from typing import Tuple

import mlx.core as mx

_METAL_AVAILABLE: bool = hasattr(mx.fast, "metal_kernel")


def metal_kernels_available() -> bool:
    """Return True when ``mx.fast.metal_kernel`` is usable."""
    return _METAL_AVAILABLE


# ---------------------------------------------------------------------------
# DfOp: fused gather + complex MAC  (differentiable via custom_function)
# ---------------------------------------------------------------------------

_DFOP_KERNEL_SOURCE = """
    uint elem = thread_position_in_grid.x;

    // Decode flat index -> (b, t, f)
    int nb_df      = coef_real_shape[2];
    int time_steps = coef_real_shape[1];
    int df_order   = coef_real_shape[3];

    int f = elem % nb_df;
    int t = (elem / nb_df) % time_steps;
    int b = elem / (nb_df * time_steps);

    // Padded-time length lives in the spec shape array.
    int spec_padded_time = spec_real_pad_shape[1];

    // Accumulate complex MAC over taps
    T acc_real = 0;
    T acc_imag = 0;

    for (int k = 0; k < df_order; k++) {
        int spec_idx = b * spec_padded_time * nb_df
                     + (t + k) * nb_df
                     + f;
        int coef_idx = b * time_steps * nb_df * df_order
                     + t * nb_df * df_order
                     + f * df_order
                     + k;

        T sr = spec_real_pad[spec_idx];
        T si = spec_imag_pad[spec_idx];
        T cr = coef_real[coef_idx];
        T ci = coef_imag[coef_idx];

        acc_real += cr * sr - ci * si;
        acc_imag += cr * si + ci * sr;
    }

    out_real[elem] = acc_real;
    out_imag[elem] = acc_imag;
"""

if _METAL_AVAILABLE:
    _dfop_kernel = mx.fast.metal_kernel(
        name="dfop_gather_cmac",
        input_names=["spec_real_pad", "spec_imag_pad", "coef_real", "coef_imag"],
        output_names=["out_real", "out_imag"],
        source=_DFOP_KERNEL_SOURCE,
    )
else:
    _dfop_kernel = None


def _dfop_forward_metal(
    spec_real_pad: mx.array,
    spec_imag_pad: mx.array,
    coef_real: mx.array,
    coef_imag: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Raw Metal kernel dispatch for DfOp (no VJP)."""
    batch_size, _, nb_df = spec_real_pad.shape[:3]
    output_time = coef_real.shape[1]
    total_elements = batch_size * output_time * nb_df
    out_shape = (batch_size, output_time, nb_df)

    assert _dfop_kernel is not None
    outputs = _dfop_kernel(
        inputs=[spec_real_pad, spec_imag_pad, coef_real, coef_imag],
        template=[("T", coef_real.dtype)],
        grid=(total_elements, 1, 1),
        threadgroup=(min(256, total_elements), 1, 1),
        output_shapes=[out_shape, out_shape],
        output_dtypes=[coef_real.dtype, coef_real.dtype],
    )
    return outputs[0], outputs[1]


def _dfop_fallback(
    spec_real_pad: mx.array,
    spec_imag_pad: mx.array,
    coef_real: mx.array,
    coef_imag: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Pure-MLX gather + complex MAC (differentiable fallback)."""
    df_order = coef_real.shape[-1]

    frame_starts = mx.arange(coef_real.shape[1])
    offsets = mx.arange(df_order)
    indices = frame_starts[:, None] + offsets[None, :]  # (T, df_order)
    flat_idx = indices.flatten()

    in_real = mx.take(spec_real_pad, flat_idx, axis=1).reshape(
        spec_real_pad.shape[0], coef_real.shape[1], df_order, spec_real_pad.shape[2]
    )
    in_imag = mx.take(spec_imag_pad, flat_idx, axis=1).reshape(
        spec_imag_pad.shape[0], coef_real.shape[1], df_order, spec_imag_pad.shape[2]
    )
    # in_{real,imag}: (B, T, df_order, nb_df)  ->  transpose to (B, T, nb_df, df_order)
    in_real = mx.transpose(in_real, (0, 1, 3, 2))
    in_imag = mx.transpose(in_imag, (0, 1, 3, 2))

    # Complex multiplication and sum over taps:
    # (c + di)(s_r + s_i*i) = (c*s_r - d*s_i) + (c*s_i + d*s_r)*i
    df_out_real = mx.sum(coef_real * in_real - coef_imag * in_imag, axis=-1)
    df_out_imag = mx.sum(coef_real * in_imag + coef_imag * in_real, axis=-1)
    return df_out_real, df_out_imag


@mx.custom_function
def _dfop_custom(
    spec_real_pad: mx.array,
    spec_imag_pad: mx.array,
    coef_real: mx.array,
    coef_imag: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Forward: Metal kernel for DfOp gather + complex MAC."""
    return _dfop_forward_metal(spec_real_pad, spec_imag_pad, coef_real, coef_imag)


@_dfop_custom.vjp
def _dfop_vjp(primals, cotangents, _outputs):
    """Backward: pure-MLX VJP for DfOp gather + complex MAC.

    Math: out[b,t,f] = sum_k coef[b,t,f,k] * spec_pad[b,t+k,f]  (complex mul)

    Gradients (split real/imag, using conj-multiply rule):
      d_coef[b,t,f,k] = conj(spec_pad[b,t+k,f]) * d_out[b,t,f]
      d_spec_pad[b,t+k,f] += conj(coef[b,t,f,k]) * d_out[b,t,f]
    """
    spec_real_pad, spec_imag_pad, coef_real, coef_imag = primals
    d_out_real, d_out_imag = cotangents

    df_order = coef_real.shape[-1]
    output_time = coef_real.shape[1]
    batch_size = coef_real.shape[0]
    nb_df = coef_real.shape[2]

    # Gather spec frames — same indexing as the forward path
    frame_starts = mx.arange(output_time)
    offsets = mx.arange(df_order)
    indices = frame_starts[:, None] + offsets[None, :]  # (T, df_order)
    flat_idx = indices.flatten()

    in_real = mx.take(spec_real_pad, flat_idx, axis=1).reshape(batch_size, output_time, df_order, nb_df)
    in_imag = mx.take(spec_imag_pad, flat_idx, axis=1).reshape(batch_size, output_time, df_order, nb_df)
    # (B, T, df_order, nb_df) -> (B, T, nb_df, df_order)
    in_real = mx.transpose(in_real, (0, 1, 3, 2))
    in_imag = mx.transpose(in_imag, (0, 1, 3, 2))

    # Expand d_out for broadcasting over taps: (B, T, nb_df, 1)
    d_out_r = mx.expand_dims(d_out_real, axis=-1)
    d_out_i = mx.expand_dims(d_out_imag, axis=-1)

    # --- Gradient w.r.t. coef ---
    # d_coef = conj(spec) * d_out
    # conj(s)(d) = (s_r*d_r + s_i*d_i) + (s_r*d_i - s_i*d_r)*i
    d_coef_real = in_real * d_out_r + in_imag * d_out_i
    d_coef_imag = in_real * d_out_i - in_imag * d_out_r

    # --- Gradient w.r.t. spec_pad ---
    # d_spec_pad[b, t+k, f] += conj(coef[b,t,f,k]) * d_out[b,t,f]
    # conj(c)(d) = (c_r*d_r + c_i*d_i) + (c_r*d_i - c_i*d_r)*i
    # Loop over df_order taps (typically 5) — each tap shifts by 1
    d_spec_real_pad = mx.zeros_like(spec_real_pad)
    d_spec_imag_pad = mx.zeros_like(spec_imag_pad)

    for k in range(df_order):
        cr_k = coef_real[:, :, :, k]  # (B, T, nb_df)
        ci_k = coef_imag[:, :, :, k]
        grad_r = cr_k * d_out_real + ci_k * d_out_imag
        grad_i = cr_k * d_out_imag - ci_k * d_out_real
        d_spec_real_pad = d_spec_real_pad.at[:, k : k + output_time, :].add(grad_r)
        d_spec_imag_pad = d_spec_imag_pad.at[:, k : k + output_time, :].add(grad_i)

    return d_spec_real_pad, d_spec_imag_pad, d_coef_real, d_coef_imag


def df_op_kernel(
    spec_real_pad: mx.array,
    spec_imag_pad: mx.array,
    coef_real: mx.array,
    coef_imag: mx.array,
    output_time: int,
    nb_df: int,
    df_order: int,
    batch_size: int,
) -> Tuple[mx.array, mx.array]:
    """Fused Metal kernel for DfOp gather + complex MAC (differentiable).

    Uses ``mx.custom_function`` so this kernel has a proper VJP and can
    be used inside ``nn.value_and_grad`` during training.

    Args:
        spec_real_pad: Padded spectrum real part, shape ``(batch, time+pad, nb_df)``.
        spec_imag_pad: Padded spectrum imag part, shape ``(batch, time+pad, nb_df)``.
        coef_real: Filter coef real part, shape ``(batch, time, nb_df, df_order)``.
        coef_imag: Filter coef imag part, shape ``(batch, time, nb_df, df_order)``.
        output_time: Number of output time steps.
        nb_df: Number of DF frequency bins.
        df_order: Filter order (number of taps).
        batch_size: Batch dimension size.

    Returns:
        Tuple of ``(out_real, out_imag)``, each ``(batch, time, nb_df)``.

    Raises:
        RuntimeError: If ``mx.fast.metal_kernel`` is not available.
    """
    if _dfop_kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is not available")

    return _dfop_custom(spec_real_pad, spec_imag_pad, coef_real, coef_imag)


# ---------------------------------------------------------------------------
# iSTFT: fused overlap-add + window normalization  (differentiable)
# ---------------------------------------------------------------------------

_ISTFT_OLA_KERNEL_SOURCE = """
    uint elem = thread_position_in_grid.x;

    int hop_length = config_arr[0];
    int output_length = config_arr[1];
    int num_frames = frames_shape[1];
    int n_fft_k = frames_shape[2];

    int n = elem % output_length;
    int b = elem / output_length;

    // Determine which frames contribute to output position n
    int first_frame = max(0, (n - n_fft_k + hop_length) / hop_length);
    int last_frame = min(num_frames - 1, n / hop_length);

    T acc = 0;
    for (int i = first_frame; i <= last_frame; i++) {
        int offset = n - i * hop_length;
        if (offset >= 0 && offset < n_fft_k) {
            int frame_idx = b * num_frames * n_fft_k + i * n_fft_k + offset;
            acc += frames[frame_idx];
        }
    }

    // Normalize by cached window norm
    T norm = window_norm[n];
    out[elem] = (norm > T(1e-8)) ? acc / norm : acc;
"""

if _METAL_AVAILABLE:
    _istft_ola_kernel = mx.fast.metal_kernel(
        name="istft_overlap_add",
        input_names=["frames", "window_norm", "config_arr"],
        output_names=["out"],
        source=_ISTFT_OLA_KERNEL_SOURCE,
    )
else:
    _istft_ola_kernel = None


def _istft_forward_metal(
    frames: mx.array,
    window_norm: mx.array,
    hop_length_arr: mx.array,
    output_length_arr: mx.array,
) -> mx.array:
    """Raw Metal kernel dispatch for iSTFT overlap-add (no VJP)."""
    batch_size = frames.shape[0]
    hop_length = int(hop_length_arr.item())
    output_length = int(output_length_arr.item())

    config_arr = mx.array([hop_length, output_length], dtype=mx.int32)
    out_shape = (batch_size, output_length)
    total_elements = batch_size * output_length

    assert _istft_ola_kernel is not None
    outputs = _istft_ola_kernel(
        inputs=[frames, window_norm, config_arr],
        template=[("T", frames.dtype)],
        grid=(total_elements, 1, 1),
        threadgroup=(min(256, total_elements), 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[frames.dtype],
    )
    return outputs[0]


@mx.custom_function
def _istft_custom(
    frames: mx.array,
    window_norm: mx.array,
    hop_length_arr: mx.array,
    output_length_arr: mx.array,
) -> mx.array:
    """Forward: Metal kernel for iSTFT overlap-add + normalization."""
    return _istft_forward_metal(frames, window_norm, hop_length_arr, output_length_arr)


@_istft_custom.vjp
def _istft_vjp(primals, cotangent, _outputs):
    """Backward: pure-MLX VJP for iSTFT overlap-add.

    Forward: out[b, n] = (sum_f frames[b, f, n - f*hop]) / window_norm[n]
    This is a linear operation on frames, so:
      d_frames[b, f, s] = d_out[b, f*hop + s] / window_norm[f*hop + s]
    where s = n - f*hop is the offset within the frame.

    We also need d_window_norm but window_norm is a fixed precomputed
    buffer (not a learned parameter), so we return zeros for it.
    """
    frames, window_norm, hop_length_arr, output_length_arr = primals
    d_out = cotangent

    batch_size, num_frames, n_fft = frames.shape
    hop_length = int(hop_length_arr.item())
    output_length = int(output_length_arr.item())

    # Inverse of window_norm for gradient scaling
    safe_norm = mx.maximum(window_norm, 1e-8)
    inv_norm = 1.0 / safe_norm  # (output_length,)

    # Scale d_out by inv_norm: d_out_scaled[b, n] = d_out[b, n] / window_norm[n]
    d_out_scaled = d_out * inv_norm[None, :]  # (B, output_length)

    # Transpose of overlap-add: gather from d_out_scaled into frames shape
    # This is equivalent to the analysis (framing) operation
    nover = n_fft // hop_length

    d_frames = mx.zeros_like(frames)
    for g in range(nover):
        group_num_frames = frames[:, g::nover, :].shape[1]
        if group_num_frames == 0:
            continue

        start_offset = g * hop_length
        flat_len = group_num_frames * n_fft

        # Gather the corresponding output samples
        end = start_offset + flat_len
        if end <= output_length:
            flat = d_out_scaled[:, start_offset:end]
        else:
            flat = d_out_scaled[:, start_offset:output_length]
            pad_right = end - output_length
            flat = mx.pad(flat, [(0, 0), (0, pad_right)])

        # Reshape into frame structure
        group_grads = flat.reshape(batch_size, group_num_frames, n_fft)
        d_frames = d_frames.at[:, g::nover, :].add(group_grads)

    d_window_norm = mx.zeros_like(window_norm)
    d_hop = mx.zeros_like(hop_length_arr)
    d_outlen = mx.zeros_like(output_length_arr)

    return d_frames, d_window_norm, d_hop, d_outlen


def istft_overlap_add_kernel(
    frames: mx.array,
    window_norm: mx.array,
    hop_length: int,
    output_length: int,
    batch_size: int,
) -> mx.array:
    """Fused Metal kernel for iSTFT overlap-add + window normalization (differentiable).

    Uses ``mx.custom_function`` so this kernel has a proper VJP and can
    be used inside ``nn.value_and_grad`` during training.

    Args:
        frames: Windowed IRFFT output, shape ``(batch, num_frames, n_fft)``.
        window_norm: Precomputed window normalization, shape ``(output_length,)``.
        hop_length: Hop size in samples.
        output_length: Length of the output signal (before center-trim).
        batch_size: Batch dimension size.

    Returns:
        Overlap-added and normalized output, shape ``(batch, output_length)``.

    Raises:
        RuntimeError: If ``mx.fast.metal_kernel`` is not available.
    """
    if _istft_ola_kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is not available")

    # Wrap scalar config as 0-d arrays so custom_function sees them as primals
    hop_arr = mx.array(hop_length, dtype=mx.int32)
    outlen_arr = mx.array(output_length, dtype=mx.int32)
    return _istft_custom(frames, window_norm, hop_arr, outlen_arr)


# ---------------------------------------------------------------------------
# Mel spectrogram: fused power-spectrum + mel-projection + log  (differentiable)
# ---------------------------------------------------------------------------

_MEL_POWER_LOG_KERNEL_SOURCE = """
    uint elem = thread_position_in_grid.x;

    int n_mels  = mel_fb_shape[0];
    int n_freqs = mel_fb_shape[1];
    int n_frames = spec_real_shape[1];

    int m = elem % n_mels;
    int t = (elem / n_mels) % n_frames;
    int b = elem / (n_mels * n_frames);

    T acc = 0;
    for (int j = 0; j < n_freqs; j++) {
        int spec_idx = b * n_frames * n_freqs + t * n_freqs + j;
        T re = spec_real[spec_idx];
        T im = spec_imag[spec_idx];
        T power = re * re + im * im;

        int fb_idx = m * n_freqs + j;
        acc += mel_fb[fb_idx] * power;
    }

    mel_out[elem] = metal::log(metal::max(acc, T(1e-10)));
"""

if _METAL_AVAILABLE:
    _mel_power_log_kernel = mx.fast.metal_kernel(
        name="mel_power_log",
        input_names=["spec_real", "spec_imag", "mel_fb"],
        output_names=["mel_out"],
        source=_MEL_POWER_LOG_KERNEL_SOURCE,
    )
else:
    _mel_power_log_kernel = None


def _mel_forward_metal(
    spec_real: mx.array,
    spec_imag: mx.array,
    mel_fb: mx.array,
) -> mx.array:
    """Raw Metal kernel dispatch for mel power+log (no VJP)."""
    batch_size, n_frames, _ = spec_real.shape
    n_mels = mel_fb.shape[0]
    total_elements = batch_size * n_frames * n_mels
    out_shape = (batch_size, n_frames, n_mels)

    assert _mel_power_log_kernel is not None
    outputs = _mel_power_log_kernel(
        inputs=[spec_real, spec_imag, mel_fb],
        template=[("T", spec_real.dtype)],
        grid=(total_elements, 1, 1),
        threadgroup=(min(256, total_elements), 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[spec_real.dtype],
    )
    return outputs[0]


@mx.custom_function
def _mel_custom(
    spec_real: mx.array,
    spec_imag: mx.array,
    mel_fb: mx.array,
) -> mx.array:
    """Forward: Metal kernel for power-spectrum + mel-projection + log."""
    return _mel_forward_metal(spec_real, spec_imag, mel_fb)


@_mel_custom.vjp
def _mel_vjp(primals, cotangent, _outputs):
    """Backward: pure-MLX VJP for mel power + log.

    Forward: out = log(max(power @ mel_fb.T, eps))
      where power = spec_real² + spec_imag²  (element-wise)

    Chain rule (let mel_raw = power @ mel_fb.T):
      d_mel_raw = d_out / max(mel_raw, eps)              — from log
      d_power   = d_mel_raw @ mel_fb                     — from matmul
      d_spec_real = 2 * spec_real * d_power               — from x²
      d_spec_imag = 2 * spec_imag * d_power               — from x²
      d_mel_fb  = power.T @ d_mel_raw  (summed over batch+time)  — from matmul
    """
    spec_real, spec_imag, mel_fb = primals
    d_out = cotangent  # (B, n_frames, n_mels)

    # Recompute intermediates for backward
    power = spec_real * spec_real + spec_imag * spec_imag  # (B, T, n_freqs)
    mel_raw = mx.matmul(power, mx.transpose(mel_fb))  # (B, T, n_mels)
    mel_clamped = mx.maximum(mel_raw, 1e-10)

    # d_log(x) = 1/x
    d_mel_raw = d_out / mel_clamped  # (B, T, n_mels)

    # d_matmul(power, mel_fb.T) w.r.t. power = d_mel_raw @ mel_fb
    d_power = mx.matmul(d_mel_raw, mel_fb)  # (B, T, n_freqs)

    # d(x²) = 2x
    d_spec_real = 2.0 * spec_real * d_power
    d_spec_imag = 2.0 * spec_imag * d_power

    # d_mel_fb: sum over batch & time of power^T @ d_mel_raw
    # power: (B, T, n_freqs), d_mel_raw: (B, T, n_mels)
    # d_mel_fb: (n_mels, n_freqs) = sum_b sum_t d_mel_raw[b,t,:].T @ power[b,t,:]
    d_mel_fb = mx.sum(
        mx.matmul(
            mx.transpose(d_mel_raw, (0, 2, 1)),  # (B, n_mels, T)
            mx.transpose(power, (0, 1, 2)),  # (B, T, n_freqs)
        ),
        axis=0,
    )  # (n_mels, n_freqs)

    return d_spec_real, d_spec_imag, d_mel_fb


def mel_power_log_kernel(
    spec_real: mx.array,
    spec_imag: mx.array,
    mel_fb: mx.array,
    batch_size: int,
    n_frames: int,
    n_mels: int,
) -> mx.array:
    """Fused Metal kernel for power-spectrum + mel-projection + log (differentiable).

    Uses ``mx.custom_function`` so this kernel has a proper VJP and can
    be used inside ``nn.value_and_grad`` during training.

    Args:
        spec_real: Real part of FFT output, shape ``(batch, n_frames, n_freqs)``.
        spec_imag: Imaginary part of FFT output, shape ``(batch, n_frames, n_freqs)``.
        mel_fb: Mel filterbank matrix, shape ``(n_mels, n_freqs)``.
        batch_size: Batch dimension size.
        n_frames: Number of time frames.
        n_mels: Number of mel frequency bins.

    Returns:
        Log-mel spectrogram, shape ``(batch, n_frames, n_mels)``.

    Raises:
        RuntimeError: If ``mx.fast.metal_kernel`` is not available.
    """
    if _mel_power_log_kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is not available")

    return _mel_custom(spec_real, spec_imag, mel_fb)
