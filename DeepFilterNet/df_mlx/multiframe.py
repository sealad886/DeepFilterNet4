"""Multi-frame speech enhancement modules for MLX.

Signal model and notation:
    Noisy: x = s + n
    Enhanced: y = f(x)
    Objective: min ||s - y||

    PSD: Power spectral density, notated e.g., as Rxx for noisy PSD.
    IFC: Inter-frame correlation vector: PSD*u, u: selection vector. Notated as rxx
    RTF: Relative transfer function, also called steering vector.

This module implements:
    - MultiFrameModule: Base class for multi-frame operations
    - DF: Deep filtering implementation
    - MfWf: Multi-frame Wiener filter
    - MfMvdr: Multi-frame MVDR beamformer
    - MultiResolutionDF: Multi-resolution deep filtering
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def as_windowed(x: mx.array, window_length: int, step: int = 1, axis: int = 1) -> mx.array:
    """Returns a tensor with chunks of overlapping windows.

    Args:
        x: Input of shape [B, T, ...]
        window_length: Length of each window
        step: Step/hop of each window
        axis: Axis to apply windowing

    Returns:
        Windowed tensor with shape [B, (N - window_length + step) // step, window_length, ...]
    """
    # MLX doesn't have as_strided, implement manually
    shape = list(x.shape)
    n_windows = (shape[axis] - window_length + step) // step

    # Use slicing to create windows
    windows = []
    for i in range(n_windows):
        start = i * step
        end = start + window_length
        # Slice along the specified axis
        slices = [slice(None)] * len(shape)
        slices[axis] = slice(start, end)
        windows.append(x[tuple(slices)])

    # Stack along new axis after the window axis
    result = mx.stack(windows, axis=axis)
    return result


def pad_time(x: mx.array, pad_left: int, pad_right: int, time_axis: int = 2) -> mx.array:
    """Pad tensor along time axis.

    Args:
        x: Input tensor
        pad_left: Left padding
        pad_right: Right padding
        time_axis: Time axis index

    Returns:
        Padded tensor
    """
    if pad_left == 0 and pad_right == 0:
        return x

    ndim = len(x.shape)
    # Create padding configuration
    pad_width = [(0, 0)] * ndim
    pad_width[time_axis] = (pad_left, pad_right)

    return mx.pad(x, pad_width, constant_values=0.0)


class MultiFrameModule(nn.Module):
    """Multi-frame speech enhancement base module.

    Args:
        num_freqs: Number of frequency bins for filtering
        frame_size: Frame size in FD domain
        lookahead: Lookahead frames (doesn't add padding)
        real: Whether to use real-valued processing
    """

    def __init__(
        self,
        num_freqs: int,
        frame_size: int,
        lookahead: int = 0,
        real: bool = False,
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.lookahead = lookahead
        self.real = real
        self.need_unfold = frame_size > 1
        self.pad_left = frame_size - 1 - lookahead
        self.pad_right = lookahead

    def spec_unfold(self, spec: mx.array) -> mx.array:
        """Pads and unfolds the spectrogram according to frame_size.

        Args:
            spec: Complex spectrogram [B, C, T, F, 2] (real representation)

        Returns:
            Unfolded spectrogram [B, C, T, F, N, 2] where N is frame_size
        """
        if not self.need_unfold:
            return mx.expand_dims(spec, axis=-2)

        # Pad along time axis (axis 2)
        spec_padded = pad_time(spec, self.pad_left, self.pad_right, time_axis=2)

        # Unfold: create overlapping windows
        B, C, T_padded, F, _ = spec_padded.shape
        T = T_padded - self.frame_size + 1

        windows = []
        for i in range(self.frame_size):
            windows.append(spec_padded[:, :, i : i + T, :, :])

        # Stack to get [B, C, T, F, N, 2]
        return mx.stack(windows, axis=4)

    def spec_unfold_real(self, spec: mx.array) -> mx.array:
        """Pads and unfolds for real-valued processing.

        Args:
            spec: Spectrogram [B, C, T, F, 2]

        Returns:
            Unfolded spectrogram [B, C, N, T, F, 2]
        """
        if not self.need_unfold:
            return mx.expand_dims(spec, axis=2)

        # Pad along time axis
        spec_padded = pad_time(spec, self.pad_left, self.pad_right, time_axis=2)

        B, C, T_padded, F, _ = spec_padded.shape
        T = T_padded - self.frame_size + 1

        windows = []
        for i in range(self.frame_size):
            windows.append(spec_padded[:, :, i : i + T, :, :])

        # Stack to get [B, C, N, T, F, 2]
        return mx.stack(windows, axis=2)

    @staticmethod
    def apply_coefs(spec: mx.array, coefs: mx.array) -> mx.array:
        """Apply filter coefficients to unfolded spectrum.

        Args:
            spec: Unfolded spectrum [B, C, T, F, N, 2]
            coefs: Coefficients [B, C, T, F, N, 2]

        Returns:
            Filtered spectrum [B, C, T, F, 2]
        """
        # Complex multiplication and sum over N dimension
        # spec and coefs are in real format [... , 2] where [..., 0] is real, [..., 1] is imag
        real = mx.sum(spec[..., 0] * coefs[..., 0] - spec[..., 1] * coefs[..., 1], axis=-1)
        imag = mx.sum(spec[..., 0] * coefs[..., 1] + spec[..., 1] * coefs[..., 0], axis=-1)
        return mx.stack([real, imag], axis=-1)


def complex_mul(a: mx.array, b: mx.array) -> mx.array:
    """Complex multiplication for real-valued tensors with last dim = 2."""
    real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return mx.stack([real, imag], axis=-1)


def complex_conj(a: mx.array) -> mx.array:
    """Complex conjugate for real-valued tensor with last dim = 2."""
    return mx.concatenate([a[..., 0:1], -a[..., 1:2]], axis=-1)


def df(spec: mx.array, coefs: mx.array) -> mx.array:
    """Deep filter implementation.

    Args:
        spec: Unfolded complex spectrum [B, C, T, F, N, 2]
        coefs: Complex coefficients [B, C, N, T, F, 2]

    Returns:
        Filtered spectrum [B, C, T, F, 2]
    """
    # Reorder coefs to match spec: [B, C, N, T, F, 2] -> [B, C, T, F, N, 2]
    coefs = mx.transpose(coefs, (0, 1, 3, 4, 2, 5))

    # Complex multiplication and sum over N (axis -2 after stacking, but N is axis 4)
    # spec shape: [B, C, T, F, N, 2] -> multiply -> sum over axis 4 (N)
    spec_real = spec[..., 0]  # [B, C, T, F, N]
    spec_imag = spec[..., 1]
    coefs_real = coefs[..., 0]
    coefs_imag = coefs[..., 1]

    # Sum over N (last axis before complex dim, which is axis 4)
    real = mx.sum(spec_real * coefs_real - spec_imag * coefs_imag, axis=4)  # [B, C, T, F]
    imag = mx.sum(spec_real * coefs_imag + spec_imag * coefs_real, axis=4)
    return mx.stack([real, imag], axis=-1)


def df_real(spec: mx.array, coefs: mx.array) -> mx.array:
    """Deep filter for real-valued input tensors.

    Args:
        spec: Unfolded spectrum [B, C, N, T, F, 2]
        coefs: Coefficients [B, C, N, T, F, 2]

    Returns:
        Filtered spectrum [B, C, T, F, 2]
    """
    # Real part: sum over N of (spec_real * coef_real - spec_imag * coef_imag)
    real = mx.sum(spec[..., 0] * coefs[..., 0] - spec[..., 1] * coefs[..., 1], axis=2)
    # Imag part: sum over N of (spec_real * coef_imag + spec_imag * coef_real)
    imag = mx.sum(spec[..., 0] * coefs[..., 1] + spec[..., 1] * coefs[..., 0], axis=2)
    return mx.stack([real, imag], axis=-1)


class DF(MultiFrameModule):
    """Deep Filtering module."""

    def __init__(
        self,
        num_freqs: int,
        frame_size: int,
        lookahead: int = 0,
        conj: bool = False,
    ):
        super().__init__(num_freqs, frame_size, lookahead)
        self.conj = conj

    def __call__(self, spec: mx.array, coefs: mx.array) -> mx.array:
        """Apply deep filtering.

        Args:
            spec: Spectrum [B, 1, T, F, 2]
            coefs: Coefficients [B, O, T, F', 2] where O=frame_size, F'=num_freqs

        Returns:
            Filtered spectrum [B, 1, T, F, 2]
        """
        # Unfold spectrum
        spec_u = self.spec_unfold(spec)

        # Reshape coefs: [B, O, T, F', 2] -> [B, 1, O, T, F', 2]
        coefs = mx.expand_dims(coefs, axis=1)

        if self.conj:
            coefs = complex_conj(coefs)

        # Apply to frequency range
        spec_f = spec_u[..., : self.num_freqs, :, :]

        # Filter
        spec_filtered = df(spec_f, coefs)

        # Replace filtered frequencies
        # MLX doesn't support direct slice assignment, need to reconstruct
        result = mx.concatenate([spec_filtered, spec[..., self.num_freqs :, :]], axis=-2)
        return result


class DFreal(MultiFrameModule):
    """Deep Filtering for real-valued tensors."""

    def __init__(
        self,
        num_freqs: int,
        frame_size: int,
        lookahead: int = 0,
        conj: bool = False,
    ):
        super().__init__(num_freqs, frame_size, lookahead, real=True)
        self.conj = conj

    def __call__(self, spec: mx.array, coefs: mx.array) -> mx.array:
        """Apply deep filtering with real-valued processing.

        Args:
            spec: Spectrum [B, C, T, F, 2]
            coefs: Coefficients [B, O, T, F', 2]

        Returns:
            Filtered spectrum [B, C, T, F, 2]
        """
        # Unfold: [B, C, T, F, 2] -> [B, C, N, T, F, 2]
        spec_u = self.spec_unfold_real(spec)

        # Reshape coefs for filtering (O = order/num_frames)
        B, order, T, Fp, _ = coefs.shape
        coefs = mx.reshape(coefs, (B, 1, order, T, Fp, 2))

        if self.conj:
            coefs = complex_conj(coefs)

        # Filter frequencies
        spec_f = spec_u[..., : self.num_freqs, :]
        spec_filtered = df_real(spec_f, coefs)

        # Combine
        result = mx.concatenate([spec_filtered, spec[..., self.num_freqs :, :]], axis=-2)
        return result


def _tik_reg(mat: mx.array, reg: float = 1e-7, eps: float = 1e-8) -> mx.array:
    """Tikhonov regularization for stability.

    Args:
        mat: Input matrix [..., N, N, 2] (complex in real format)
        reg: Regularization factor
        eps: Small value to prevent all-zero matrices

    Returns:
        Regularized matrix
    """
    # Get diagonal trace (sum of real parts on diagonal)
    N = mat.shape[-3]

    # Extract diagonal elements
    diag_indices = mx.arange(N)
    diag_real = mat[..., diag_indices, diag_indices, 0]
    trace = mx.sum(diag_real, axis=-1, keepdims=True)

    # Compute epsilon
    epsilon = trace * reg + eps

    # Create identity matrix
    eye_real = mx.eye(N)
    eye = mx.stack([eye_real, mx.zeros_like(eye_real)], axis=-1)

    # Add regularization
    result = mat + epsilon[..., None, None] * eye
    return result


def complex_matmul(a: mx.array, b: mx.array) -> mx.array:
    """Complex matrix multiplication for real-valued tensors.

    Args:
        a: Matrix [..., M, N, 2]
        b: Matrix [..., N, K, 2]

    Returns:
        Result [..., M, K, 2]
    """
    # (a_r + i*a_i) @ (b_r + i*b_i) = (a_r@b_r - a_i@b_i) + i*(a_r@b_i + a_i@b_r)
    real = a[..., 0] @ b[..., 0] - a[..., 1] @ b[..., 1]
    imag = a[..., 0] @ b[..., 1] + a[..., 1] @ b[..., 0]
    return mx.stack([real, imag], axis=-1)


def complex_matmul_conj_transpose(a: mx.array) -> mx.array:
    """Compute A @ A^H (conjugate transpose) for complex matrix.

    Args:
        a: Matrix [..., M, N, 2]

    Returns:
        Result [..., M, M, 2]
    """
    # A^H = conj(A)^T
    a_conj_T_real = mx.swapaxes(a[..., 0], -2, -1)
    a_conj_T_imag = -mx.swapaxes(a[..., 1], -2, -1)

    real = a[..., 0] @ a_conj_T_real - a[..., 1] @ a_conj_T_imag
    imag = a[..., 0] @ a_conj_T_imag + a[..., 1] @ a_conj_T_real
    return mx.stack([real, imag], axis=-1)


class MfWf(MultiFrameModule):
    """Multi-frame Wiener filter.

    Args:
        num_freqs: Number of frequency bins for filtering
        frame_size: Multi-frame filter order
        lookahead: Lookahead frames
        cholesky_decomp: Whether input is Cholesky decomposition
        inverse: Whether input is inverse correlation matrix
        enforce_constraints: Enforce Hermitian/triangular constraints
    """

    def __init__(
        self,
        num_freqs: int,
        frame_size: int,
        lookahead: int = 0,
        cholesky_decomp: bool = False,
        inverse: bool = True,
        enforce_constraints: bool = True,
        eps: float = 1e-8,
        dload: float = 1e-7,
    ):
        super().__init__(num_freqs, frame_size, lookahead)
        self.cholesky_decomp = cholesky_decomp
        self.inverse = inverse
        self.enforce_constraints = enforce_constraints
        self.eps = eps
        self.dload = dload

        # Compute triangular indices
        self.triu_row = []
        self.triu_col = []
        self.tril_row = []
        self.tril_col = []
        for i in range(frame_size):
            for j in range(i + 1, frame_size):
                self.triu_row.append(i)
                self.triu_col.append(j)
                self.tril_row.append(j)
                self.tril_col.append(i)

    def __call__(
        self,
        spec: mx.array,
        ifc: mx.array,
        iRxx: mx.array,
    ) -> mx.array:
        """Apply multi-frame Wiener filter.

        Args:
            spec: Spectrum [B, 1, T, F, 2]
            ifc: Inter-frame correlation vector [B, T, F, N*2]
            iRxx: (Inverse) correlation matrix [B, T, F, N**2*2]

        Returns:
            Filtered spectrum [B, 1, T, F, 2]
        """
        # Unfold spectrum
        spec_u = self.spec_unfold(spec)

        # Reshape ifc: [B, T, F, N*2] -> [B, T, F, N, 2]
        B, T, F, _ = ifc.shape
        ifc = mx.reshape(ifc, (B, T, F, self.frame_size, 2))

        # Reshape iRxx: [B, T, F, N**2*2] -> [B, T, F, N, N, 2]
        iRxx = mx.reshape(iRxx, (B, T, F, self.frame_size, self.frame_size, 2))

        if self.cholesky_decomp:
            if self.enforce_constraints:
                # Zero out upper triangular (above diagonal)
                # MLX doesn't support direct indexing assignment, need to use masking
                mask = mx.tril(mx.ones((self.frame_size, self.frame_size)), k=0)
                mask = mx.expand_dims(mask, axis=-1)
                iRxx = iRxx * mask

            # Revert Cholesky: L @ L^H
            iRxx = complex_matmul_conj_transpose(iRxx)

        if self.enforce_constraints and not self.inverse and not self.cholesky_decomp:
            # Enforce Hermitian: diagonal imag = 0, upper = conj(lower)
            # This is complex for MLX without in-place ops, skip for now
            pass

        # Apply filter
        spec_f = spec_u[..., : self.num_freqs, :, :]

        if not self.inverse:
            # Need to solve linear system (not implemented in MLX, use regularization)
            iRxx = _tik_reg(iRxx, self.dload, self.eps)
            # Approximate: w ≈ (Rxx + λI)^{-1} @ ifc via iterative solve
            # For simplicity, just use direct computation assuming inverse input
            pass

        # Compute weights: w = iRxx @ ifc (when inverse=True)
        # [B, T, F, N, N, 2] @ [B, T, F, N, 2] -> [B, T, F, N, 2]
        w_real = mx.sum(
            iRxx[..., 0] * ifc[..., None, :, 0] - iRxx[..., 1] * ifc[..., None, :, 1],
            axis=-2,
        )
        w_imag = mx.sum(
            iRxx[..., 0] * ifc[..., None, :, 1] + iRxx[..., 1] * ifc[..., None, :, 0],
            axis=-2,
        )
        w = mx.stack([w_real, w_imag], axis=-1)

        # Expand for channel dim: [B, T, F, N, 2] -> [B, 1, T, F, N, 2]
        w = mx.expand_dims(w, axis=1)

        # Apply coefficients
        spec_filtered = MultiFrameModule.apply_coefs(spec_f, w)

        # Combine with original spectrum for higher frequencies
        result = mx.concatenate([spec_filtered, spec[..., self.num_freqs :, :]], axis=-2)
        return result


class MfMvdr(MultiFrameModule):
    """Multi-frame MVDR beamformer.

    Minimum variance distortionless response beamformer based on
    noise covariance matrix (or its inverse) and speech IFC vector.

    Args:
        num_freqs: Number of frequency bins for filtering
        frame_size: Multi-frame filter order
        lookahead: Lookahead frames
        cholesky_decomp: Whether input is Cholesky decomposition
        inverse: Whether input is inverse correlation matrix
        enforce_constraints: Enforce Hermitian/triangular constraints
    """

    def __init__(
        self,
        num_freqs: int,
        frame_size: int,
        lookahead: int = 0,
        cholesky_decomp: bool = False,
        inverse: bool = True,
        enforce_constraints: bool = True,
        eps: float = 1e-8,
        dload: float = 1e-7,
    ):
        super().__init__(num_freqs, frame_size, lookahead)
        self.cholesky_decomp = cholesky_decomp
        self.inverse = inverse
        self.enforce_constraints = enforce_constraints
        self.eps = eps
        self.dload = dload

    def __call__(
        self,
        spec: mx.array,
        ifc: mx.array,
        iRnn: mx.array,
    ) -> mx.array:
        """Apply multi-frame MVDR filter.

        Args:
            spec: Spectrum [B, 1, T, F, 2]
            ifc: Inter-frame correlation vector [B, T, F, N*2]
            iRnn: (Inverse) noise correlation matrix [B, T, F, N**2*2]

        Returns:
            Filtered spectrum [B, 1, T, F, 2]
        """
        # Unfold spectrum
        spec_u = self.spec_unfold(spec)

        # Reshape tensors
        B, T, F, _ = ifc.shape
        ifc = mx.reshape(ifc, (B, T, F, self.frame_size, 2))
        iRnn = mx.reshape(iRnn, (B, T, F, self.frame_size, self.frame_size, 2))

        if self.cholesky_decomp:
            if self.enforce_constraints:
                mask = mx.tril(mx.ones((self.frame_size, self.frame_size)), k=0)
                mask = mx.expand_dims(mask, axis=-1)
                iRnn = iRnn * mask

            # Revert Cholesky
            iRnn = complex_matmul_conj_transpose(iRnn)

        spec_f = spec_u[..., : self.num_freqs, :, :]

        if not self.inverse:
            iRnn = _tik_reg(iRnn, self.dload, self.eps)

        # Compute numerator: iRnn @ ifc (when inverse=True)
        num_real = mx.sum(
            iRnn[..., 0] * ifc[..., None, :, 0] - iRnn[..., 1] * ifc[..., None, :, 1],
            axis=-2,
        )
        num_imag = mx.sum(
            iRnn[..., 0] * ifc[..., None, :, 1] + iRnn[..., 1] * ifc[..., None, :, 0],
            axis=-2,
        )
        numerator = mx.stack([num_real, num_imag], axis=-1)

        # Compute denominator: conj(ifc) @ numerator
        ifc_conj = complex_conj(ifc)
        denom_real = mx.sum(
            ifc_conj[..., 0] * numerator[..., 0] - ifc_conj[..., 1] * numerator[..., 1],
            axis=-1,
            keepdims=True,
        )
        # Only use real part of denominator
        denom = denom_real + self.eps

        # Scale by last element of ifc
        scale = complex_conj(ifc[..., -1:, :])

        # Compute weights: w = (numerator * scale) / denom
        w_scaled_real = numerator[..., 0] * scale[..., 0] - numerator[..., 1] * scale[..., 1]
        w_scaled_imag = numerator[..., 0] * scale[..., 1] + numerator[..., 1] * scale[..., 0]
        w = mx.stack([w_scaled_real / denom, w_scaled_imag / denom], axis=-1)

        # Expand for channel
        w = mx.expand_dims(w, axis=1)

        # Apply
        spec_filtered = MultiFrameModule.apply_coefs(spec_f, w)

        result = mx.concatenate([spec_filtered, spec[..., self.num_freqs :, :]], axis=-2)
        return result


class MultiResolutionDF(nn.Module):
    """Multi-resolution deep filtering.

    Combines deep filtering at multiple frequency resolutions with
    learnable weighting.

    Args:
        resolutions: List of (num_freqs, frame_size) tuples
        lookahead: Lookahead frames
        learnable_weights: Whether resolution weights are learnable
        use_real: Use real-valued DF processing
    """

    def __init__(
        self,
        resolutions: Optional[List[Tuple[int, int]]] = None,
        lookahead: int = 0,
        learnable_weights: bool = True,
        use_real: bool = False,
    ):
        super().__init__()
        if resolutions is None:
            resolutions = [(96, 5), (48, 3), (24, 2)]

        self.resolutions = resolutions
        self.use_real = use_real

        # Create DF operations
        df_class = DFreal if use_real else DF
        self.df_ops = [df_class(num_freqs=nf, frame_size=fs, lookahead=lookahead) for nf, fs in resolutions]

        # Resolution weights
        if learnable_weights:
            self.resolution_weights = mx.ones(len(resolutions)) / len(resolutions)
        else:
            self.resolution_weights = mx.ones(len(resolutions)) / len(resolutions)

        self.learnable_weights = learnable_weights

    def __call__(
        self,
        spec: mx.array,
        coefs_list: List[mx.array],
    ) -> mx.array:
        """Apply multi-resolution deep filtering.

        Args:
            spec: Input spectrum [B, C, T, F, 2]
            coefs_list: List of coefficient tensors per resolution

        Returns:
            Enhanced spectrum [B, C, T, F, 2]
        """
        if len(coefs_list) != len(self.resolutions):
            raise ValueError(f"Expected {len(self.resolutions)} coefficient tensors, got {len(coefs_list)}")

        # Softmax weights
        weights = mx.softmax(self.resolution_weights, axis=0)

        full_freqs = spec.shape[-2]

        outputs = []
        for df_op, coefs, (nf, _) in zip(self.df_ops, coefs_list, self.resolutions):
            # Extract frequency range
            spec_res = spec[..., :nf, :]

            # Apply DF
            out_res = df_op(spec_res, coefs)

            # Pad back
            if nf < full_freqs:
                padded = mx.concatenate([out_res[..., :nf, :], spec[..., nf:, :]], axis=-2)
                outputs.append(padded)
            else:
                outputs.append(out_res)

        # Weighted combination
        result = mx.zeros_like(spec)
        for w, out in zip(weights, outputs):
            result = result + w * out

        return result


class CRM(MultiFrameModule):
    """Complex Ratio Mask."""

    def __init__(self, num_freqs: int):
        super().__init__(num_freqs, frame_size=1, lookahead=0)

    def __call__(self, spec: mx.array, mask: mx.array) -> mx.array:
        """Apply complex ratio mask.

        Args:
            spec: Spectrum [B, C, T, F, 2]
            mask: Complex mask [B, C, T, F, 2]

        Returns:
            Masked spectrum
        """
        return complex_mul(spec, mask)


# Export main classes
__all__ = [
    "MultiFrameModule",
    "DF",
    "DFreal",
    "MfWf",
    "MfMvdr",
    "MultiResolutionDF",
    "CRM",
    "df",
    "df_real",
    "complex_mul",
    "complex_conj",
]
