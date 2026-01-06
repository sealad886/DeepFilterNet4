import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torchaudio as ta
from loguru import logger
from numpy import ndarray
from packaging import version
from torch import Tensor

# Version detection for TorchAudio 2.9+ compatibility
TORCHAUDIO_VERSION = version.parse(ta.__version__)
USE_TORCHCODEC = TORCHAUDIO_VERSION >= version.parse("2.9.0")

# Try to import TorchCodec for TorchAudio 2.9+ metadata support
HAS_TORCHCODEC = False
if USE_TORCHCODEC:
    try:
        from torchcodec.decoders import AudioDecoder

        HAS_TORCHCODEC = True
    except ImportError:
        pass

# Try to import soundfile as fallback for audio metadata
HAS_SOUNDFILE = False
try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    pass

# Handle AudioMetaData import across TorchAudio versions
try:
    from torchaudio import AudioMetaData

    TA_RESAMPLE_SINC = "sinc_interp_hann"
    TA_RESAMPLE_KAISER = "sinc_interp_kaiser"
except ImportError:
    try:
        from torchaudio.backend.common import AudioMetaData  # type: ignore[import-unresolved]  # noqa: F401

        TA_RESAMPLE_SINC = "sinc_interpolation"
        TA_RESAMPLE_KAISER = "kaiser_window"
    except ImportError:
        # TorchAudio 2.9+: AudioMetaData may not be available, use dataclass fallback
        from dataclasses import dataclass

        @dataclass
        class AudioMetaData:
            """Fallback AudioMetaData for TorchAudio 2.9+ when using TorchCodec."""

            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int = 0
            encoding: str = ""

        TA_RESAMPLE_SINC = "sinc_interp_hann"
        TA_RESAMPLE_KAISER = "sinc_interp_kaiser"

from df.logger import warn_once  # noqa: E402
from df.utils import download_file, get_cache_dir, get_git_root  # noqa: E402


def get_audio_metadata(file: str) -> AudioMetaData:
    """Get audio metadata using TorchCodec for TorchAudio 2.9+ or torchaudio.info() for earlier versions.

    Args:
        file: Path to an audio file.

    Returns:
        AudioMetaData with sample_rate, num_frames, num_channels, etc.
    """
    if USE_TORCHCODEC and HAS_TORCHCODEC:
        decoder = AudioDecoder(file)  # type: ignore[possibly-undefined]
        metadata = decoder.metadata
        return AudioMetaData(
            sample_rate=metadata.sample_rate,
            num_frames=metadata.num_frames if hasattr(metadata, "num_frames") else 0,
            num_channels=metadata.num_channels,
            bits_per_sample=getattr(metadata, "bits_per_sample", 0),
            encoding=getattr(metadata, "codec", ""),
        )
    elif USE_TORCHCODEC and HAS_SOUNDFILE:
        # TorchAudio 2.9+ without TorchCodec: use soundfile as fallback
        info = sf.info(file)  # type: ignore[possibly-undefined]
        return AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
            bits_per_sample=0,
            encoding=info.subtype or "",
        )
    else:
        # Older TorchAudio versions with torchaudio.info()
        return ta.info(file)  # type: ignore[attr-defined]


def load_audio(file: str, sr: Optional[int] = None, verbose=True, **kwargs) -> Tuple[Tensor, AudioMetaData]:
    """Loads an audio file using torchaudio.

    Args:
        file (str): Path to an audio file.
        sr (int): Optionally resample audio to specified target sampling rate.
        **kwargs: Passed to torchaudio.load(). Depends on the backend. The resample method
            may be set via `method` which is passed to `resample()`.

    Returns:
        audio (Tensor): Audio tensor of shape [C, T], if channels_first=True (default).
        info (AudioMetaData): Meta data of the original audio file. Contains the original sr.
    """
    ikwargs = {}
    if "format" in kwargs:
        ikwargs["format"] = kwargs["format"]
    rkwargs = {}
    if "method" in kwargs:
        rkwargs["method"] = kwargs.pop("method")
    info: AudioMetaData = get_audio_metadata(file)
    if "num_frames" in kwargs and sr is not None:
        kwargs["num_frames"] *= info.sample_rate // sr
    audio, orig_sr = ta.load(file, **kwargs)
    if sr is not None and orig_sr != sr:
        if verbose:
            warn_once(f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). " "Resampling...")
        audio = resample(audio, orig_sr, sr, **rkwargs)
    return audio.contiguous(), info


def save_audio(
    file: str,
    audio: Union[Tensor, ndarray],
    sr: int,
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    log: bool = False,
    dtype=torch.int16,
):
    outpath = file
    if suffix is not None:
        file, ext = os.path.splitext(file)
        outpath = file + f"_{suffix}" + ext
    if output_dir is not None:
        outpath = os.path.join(output_dir, os.path.basename(outpath))
    if log:
        logger.info(f"Saving audio file '{outpath}'")
    audio = torch.as_tensor(audio)
    if audio.ndim == 1:
        audio.unsqueeze_(0)
    if dtype == torch.int16 and audio.dtype != torch.int16:
        audio = (audio * (1 << 15)).to(torch.int16)
    if dtype == torch.float32 and audio.dtype != torch.float32:
        audio = audio.to(torch.float32) / (1 << 15)
    ta.save(outpath, audio, sr)


try:
    from torchaudio.functional import resample as ta_resample
except ImportError:
    from torchaudio.compliance.kaldi import resample_waveform as ta_resample  # type: ignore


def get_resample_params(method: str) -> Dict[str, Any]:
    params = {
        "sinc_fast": {"resampling_method": TA_RESAMPLE_SINC, "lowpass_filter_width": 16},
        "sinc_best": {"resampling_method": TA_RESAMPLE_SINC, "lowpass_filter_width": 64},
        "kaiser_fast": {
            "resampling_method": TA_RESAMPLE_KAISER,
            "lowpass_filter_width": 16,
            "rolloff": 0.85,
            "beta": 8.555504641634386,
        },
        "kaiser_best": {
            "resampling_method": TA_RESAMPLE_KAISER,
            "lowpass_filter_width": 16,
            "rolloff": 0.9475937167399596,
            "beta": 14.769656459379492,
        },
    }
    assert method in params.keys(), f"method must be one of {list(params.keys())}"
    return params[method]


def resample(audio: Tensor, orig_sr: int, new_sr: int, method="sinc_fast"):
    params = get_resample_params(method)
    return ta_resample(audio, orig_sr, new_sr, **params)


def get_test_sample(sr: int = 48000) -> Tensor:
    dir = get_git_root()
    file_path = os.path.join("assets", "clean_freesound_33711.wav")
    if dir is None:
        url = "https://github.com/Rikorose/DeepFilterNet/raw/main/" + file_path
        save_dir = get_cache_dir()
        path = download_file(url, save_dir)
    else:
        path = os.path.join(dir, file_path)
    sample, _ = load_audio(path, sr=sr)
    return sample
