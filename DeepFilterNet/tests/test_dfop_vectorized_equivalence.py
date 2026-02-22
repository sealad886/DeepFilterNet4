import mlx.core as mx
import numpy as np

from df_mlx.modules import DfOp


def _legacy_dfop(
    spec: tuple[mx.array, mx.array],
    coef: mx.array,
    *,
    nb_df: int,
    df_order: int,
    df_lookahead: int,
) -> tuple[mx.array, mx.array]:
    """Reference implementation matching the previous stack-based DfOp path."""
    spec_real, spec_imag = spec
    _, time, n_freqs = spec_real.shape

    df_real = spec_real[:, :, :nb_df]
    df_imag = spec_imag[:, :, :nb_df]

    pad_past = df_order - 1 - df_lookahead
    pad_future = df_lookahead
    df_real_pad = mx.pad(df_real, [(0, 0), (pad_past, pad_future), (0, 0)])
    df_imag_pad = mx.pad(df_imag, [(0, 0), (pad_past, pad_future), (0, 0)])

    in_real = mx.stack([df_real_pad[:, k : k + time, :] for k in range(df_order)], axis=-1)
    in_imag = mx.stack([df_imag_pad[:, k : k + time, :] for k in range(df_order)], axis=-1)

    coef_real = coef[:, :, :, :, 0]
    coef_imag = coef[:, :, :, :, 1]

    df_out_real = mx.sum(coef_real * in_real - coef_imag * in_imag, axis=-1)
    df_out_imag = mx.sum(coef_real * in_imag + coef_imag * in_real, axis=-1)

    if n_freqs > nb_df:
        out_real = mx.concatenate([df_out_real, spec_real[:, :, nb_df:]], axis=-1)
        out_imag = mx.concatenate([df_out_imag, spec_imag[:, :, nb_df:]], axis=-1)
    else:
        out_real = df_out_real
        out_imag = df_out_imag

    return out_real, out_imag


def test_dfop_vectorized_matches_legacy_stack_path():
    np.random.seed(13)

    batch, time, n_freqs = 2, 9, 120
    nb_df, df_order, df_lookahead = 96, 5, 1

    spec_real = mx.array(np.random.randn(batch, time, n_freqs).astype(np.float32))
    spec_imag = mx.array(np.random.randn(batch, time, n_freqs).astype(np.float32))
    coef = mx.array(np.random.randn(batch, time, nb_df, df_order, 2).astype(np.float32))

    op = DfOp(nb_df=nb_df, df_order=df_order, df_lookahead=df_lookahead)

    expected_real, expected_imag = _legacy_dfop(
        (spec_real, spec_imag),
        coef,
        nb_df=nb_df,
        df_order=df_order,
        df_lookahead=df_lookahead,
    )
    actual_real, actual_imag = op((spec_real, spec_imag), coef)
    mx.eval(expected_real, expected_imag, actual_real, actual_imag)

    np.testing.assert_allclose(np.asarray(actual_real), np.asarray(expected_real), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(actual_imag), np.asarray(expected_imag), rtol=1e-5, atol=1e-5)
