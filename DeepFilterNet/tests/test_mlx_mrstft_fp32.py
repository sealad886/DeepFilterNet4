import mlx.core as mx
import numpy as np

from df_mlx.ops import istft
from df_mlx.train import MultiResolutionSTFTLoss
from df_mlx.training_waveform import compute_mrstft_loss


def test_mrstft_loss_force_fp32_is_finite():
    n_fft = 8
    hop = 4
    frames = 3
    freqs = n_fft // 2 + 1

    real = np.full((1, frames, freqs), 1000.0, dtype=np.float16)
    imag = np.zeros((1, frames, freqs), dtype=np.float16)

    out_spec = (mx.array(real), mx.array(imag))
    clean_spec = (mx.array(real), mx.array(imag))

    loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=(n_fft,),
        hop_sizes=(hop,),
        gamma=1.0,
        factor=1.0,
        f_complex=None,
    )

    target_len = (frames - 1) * hop + n_fft
    loss = compute_mrstft_loss(
        out_spec,
        clean_spec,
        istft_fn=istft,
        loss_fn=loss_fn,
        n_fft=n_fft,
        hop_length=hop,
        target_len=target_len,
        force_fp32=True,
    )

    assert loss.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(loss)))
