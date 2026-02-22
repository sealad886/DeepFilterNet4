import mlx.core as mx
import numpy as np

from df_mlx.dnsmos_proxy import MelSpectrogram


def _legacy_melspectrogram(module: MelSpectrogram, audio: mx.array) -> mx.array:
    """Reference implementation matching the previous loop-based path."""
    if audio.ndim == 3:
        audio = mx.squeeze(audio, axis=1)

    batch_size = audio.shape[0]
    n_samples = audio.shape[1]
    n_frames = (n_samples - module.n_fft) // module.hop_length + 1

    mel_specs = []
    for b in range(batch_size):
        frames = []
        for i in range(n_frames):
            start = i * module.hop_length
            frame = audio[b, start : start + module.n_fft] * module._window
            frames.append(frame)

        if not frames:
            mel_specs.append(mx.zeros((module.n_mels, 1), dtype=mx.float32))
            continue

        frames = mx.stack(frames, axis=0)
        spec_complex = mx.fft.rfft(frames)
        power = mx.abs(spec_complex) ** 2
        mel_spec = mx.matmul(module._mel_fb, mx.transpose(power))
        mel_spec = mx.log(mx.maximum(mel_spec, 1e-10))
        mel_specs.append(mel_spec)

    return mx.stack(mel_specs, axis=0)


def test_melspectrogram_matches_legacy_path():
    np.random.seed(7)
    audio = mx.array(np.random.randn(3, 4096).astype(np.float32))
    mel = MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=64)

    expected = _legacy_melspectrogram(mel, audio)
    actual = mel(audio)
    mx.eval(expected, actual)

    expected_np = np.asarray(expected)
    actual_np = np.asarray(actual)

    assert actual_np.shape == expected_np.shape
    np.testing.assert_allclose(actual_np, expected_np, rtol=1e-5, atol=1e-5)


def test_melspectrogram_short_audio_returns_single_frame():
    audio = mx.zeros((2, 128), dtype=mx.float32)
    mel = MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=32)

    out = mel(audio)
    mx.eval(out)
    out_np = np.asarray(out)

    assert out_np.shape == (2, 32, 1)
    assert np.allclose(out_np, 0.0)
