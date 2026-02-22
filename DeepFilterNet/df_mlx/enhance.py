"""Enhancement/inference module for MLX DeepFilterNet4.

This module provides:
- Single-file and batch enhancement
- Streaming real-time enhancement
- Model loading and checkpoint handling
- Audio I/O utilities

Based on df/enhance.py but adapted for MLX on Apple Silicon.
"""

import argparse
import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from loguru import logger

from .config import ModelParams4, load_config
from .model import DfNet4, StreamingDfNet4
from .vad_silero import SileroVAD, SileroVADConfig

# Default pretrained models
PRETRAINED_MODELS = ("DeepFilterNet4-MLX",)
DEFAULT_MODEL = "DeepFilterNet4-MLX"


@dataclass
class EnhanceConfig:
    """Configuration for enhancement."""

    # Model settings
    model_path: Optional[str] = None
    epoch: Union[str, int] = "best"

    # Enhancement settings
    compensate_delay: bool = True
    atten_lim_db: Optional[float] = None
    post_filter: bool = False

    # Audio settings
    target_sr: int = 48000  # DeepFilterNet4 native sample rate

    # Output settings
    output_dir: str = "."
    suffix: Optional[str] = None

    # Processing
    batch_size: int = 1
    streaming: bool = False
    chunk_size_ms: float = 100.0  # For streaming mode


@dataclass
class SpeechBoostConfig:
    """VAD-driven speech segment amplification settings."""

    gain_db: float = 0.0
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    ramp_ms: float = 8.0
    peak_limit: float = 0.99
    silero_model_path: Optional[str] = None
    silero_sample_rate: int = 16000


def _init_speech_boost_vad(speech_boost: Optional[SpeechBoostConfig]) -> Optional[SileroVAD]:
    """Initialize Silero VAD once for batch enhancement when speech boost is enabled."""
    if speech_boost is None or speech_boost.gain_db <= 0.0:
        return None

    return SileroVAD(
        SileroVADConfig(
            sample_rate=speech_boost.silero_sample_rate,
            model_path=speech_boost.silero_model_path,
            force_cpu=True,
        )
    )


def load_audio(
    path: str,
    target_sr: int = 48000,
    mono: bool = True,
) -> Tuple[mx.array, int]:
    """Load audio file and optionally resample.

    Args:
        path: Path to audio file
        target_sr: Target sample rate
        mono: Whether to convert to mono

    Returns:
        Tuple of (audio_array, original_sample_rate)
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for audio I/O: pip install soundfile")

    audio, sr = sf.read(path, dtype="float32")  # type: ignore[misc]

    # Convert to mono if needed
    if mono and audio.ndim > 1:
        audio = audio.mean(axis=-1)

    # Resample if needed
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
        orig_sr = sr
    else:
        orig_sr = sr

    return mx.array(audio), orig_sr


def save_audio(
    audio: Union[mx.array, np.ndarray],
    path: str,
    sr: int,
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """Save audio to file.

    Args:
        audio: Audio array
        path: Original file path (used for naming)
        sr: Sample rate
        output_dir: Output directory (default: same as input)
        suffix: Suffix to add to filename

    Returns:
        Path to saved file
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for audio I/O: pip install soundfile")

    if isinstance(audio, mx.array):
        audio = np.array(audio)

    # Build output path
    basename = os.path.basename(path)
    name, ext = os.path.splitext(basename)

    if suffix:
        name = f"{name}_{suffix}"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{name}{ext}")
    else:
        out_path = os.path.join(os.path.dirname(path), f"{name}{ext}")

    sf.write(out_path, audio, sr)
    return out_path


def resample(
    audio: Union[mx.array, np.ndarray],
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio using scipy.

    Args:
        audio: Input audio
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio as numpy array
    """
    if orig_sr == target_sr:
        return np.array(audio) if isinstance(audio, mx.array) else audio

    try:
        from scipy import signal
    except ImportError:
        raise ImportError("scipy is required for resampling: pip install scipy")

    if isinstance(audio, mx.array):
        audio = np.array(audio)

    # Compute resampling ratio
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)

    resampled: np.ndarray = signal.resample(audio, new_length)  # type: ignore[assignment]
    return resampled


def load_model(
    model_path: Optional[str] = None,
    epoch: Union[str, int] = "best",
    device: Optional[str] = None,  # Ignored for MLX (always uses Metal)
) -> Tuple[DfNet4, ModelParams4, str, int]:
    """Load model and configuration.

    Args:
        model_path: Path to model directory or pretrained model name
        epoch: Checkpoint epoch ('best', 'latest', or int)
        device: Compute device (ignored for MLX)

    Returns:
        Tuple of (model, params, suffix, loaded_epoch)
    """
    # Handle pretrained models
    if model_path is None:
        model_path = maybe_download_model(DEFAULT_MODEL)
    elif model_path in PRETRAINED_MODELS:
        model_path = maybe_download_model(model_path)

    epoch = normalize_epoch_spec(epoch)

    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Load config
    config_path = model_dir / "config.ini"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    params = load_config(str(config_path))

    # Initialize model
    model = DfNet4(params)

    # Load checkpoint
    checkpoint_dir = model_dir / "checkpoints"
    loaded_epoch = 0

    if checkpoint_dir.exists() and epoch != "none":
        checkpoint_path = find_checkpoint(checkpoint_dir, epoch)
        if checkpoint_path:
            weights: Dict[str, mx.array] = mx.load(str(checkpoint_path))  # type: ignore[assignment]
            model.load_weights(list(weights.items()))
            loaded_epoch = parse_epoch_from_checkpoint_name(checkpoint_path.name)
            logger.info(f"Loaded checkpoint from {checkpoint_path.name}")
        else:
            logger.warning(f"No checkpoint found in {checkpoint_dir}")

    # Model suffix for output naming
    suffix = model_dir.name

    return model, params, suffix, loaded_epoch


def find_checkpoint(
    checkpoint_dir: Path,
    epoch: Union[str, int],
) -> Optional[Path]:
    """Find checkpoint file based on epoch specification.

    Args:
        checkpoint_dir: Directory containing checkpoints
        epoch: 'best', 'latest', or integer epoch number

    Returns:
        Path to checkpoint or None
    """
    checkpoints = list(checkpoint_dir.glob("*.safetensors"))
    if not checkpoints:
        checkpoints = list(checkpoint_dir.glob("*.npz"))
    if not checkpoints:
        return None

    if isinstance(epoch, int):
        # Find specific epoch
        for cp in checkpoints:
            if f"epoch_{epoch:04d}" in cp.stem or f"epoch_{epoch}" in cp.stem:
                return cp
        return None

    elif epoch == "best":
        # Find best checkpoint
        for cp in checkpoints:
            if "best" in cp.stem:
                return cp
        # Fall back to latest
        return find_checkpoint(checkpoint_dir, "latest")

    elif epoch == "latest":
        # Find highest epoch number
        epoch_nums = []
        for cp in checkpoints:
            try:
                if "epoch_" in cp.stem:
                    num = int(cp.stem.split("epoch_")[-1].split("_")[0])
                    epoch_nums.append((num, cp))
            except ValueError:
                continue

        if epoch_nums:
            return max(epoch_nums, key=lambda x: x[0])[1]

        # Fall back to final or first available
        for cp in checkpoints:
            if "final" in cp.stem:
                return cp
        return checkpoints[0] if checkpoints else None

    return None


def normalize_epoch_spec(epoch: Union[str, int]) -> Union[str, int]:
    """Normalize epoch CLI/input value.

    Accepts:
    - int values
    - 'best', 'latest', 'none'
    - integer strings (e.g. '12')
    """
    if isinstance(epoch, int):
        return epoch
    epoch_norm = str(epoch).strip().lower()
    if epoch_norm in {"best", "latest", "none"}:
        return epoch_norm
    if epoch_norm.isdigit():
        return int(epoch_norm)
    raise ValueError(f"Invalid epoch '{epoch}'. Expected 'best', 'latest', 'none', or integer.")


def parse_epoch_from_checkpoint_name(name: str) -> int:
    """Extract epoch number from checkpoint filename if present."""
    stem = Path(name).stem
    if "epoch_" not in stem:
        return 0
    try:
        token = stem.split("epoch_")[-1].split("_")[0]
        return int(token)
    except ValueError:
        return 0


def maybe_download_model(name: str) -> str:
    """Download pretrained model if not cached.

    Args:
        name: Model name

    Returns:
        Path to model directory
    """
    cache_dir = get_cache_dir()
    model_dir = cache_dir / name

    if model_dir.exists() and (model_dir / "config.ini").exists():
        return str(model_dir)

    # Download from GitHub releases
    logger.info(f"Downloading pretrained model: {name}")

    # TODO: Implement actual download when models are released
    raise NotImplementedError(
        f"Pretrained model '{name}' not found and automatic download not yet implemented. "
        f"Please provide a local model path."
    )


def get_cache_dir() -> Path:
    """Get cache directory for models."""
    cache_dir = Path.home() / ".cache" / "deepfilternet-mlx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@mx.compile
def enhance_frame_compiled(
    spec: Tuple[mx.array, mx.array],
    feat_erb: mx.array,
    feat_spec: mx.array,
    erb_mask: mx.array,
    df_out: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Compiled enhancement kernel (stateless).

    Note: This is a stateless kernel that takes precomputed model outputs.
    For full model inference, use enhance() instead.
    """
    # This is a placeholder for future optimization
    # The actual model call happens in enhance()
    return spec


def enhance_frame(
    model: DfNet4,
    spec: Tuple[mx.array, mx.array],
    feat_erb: mx.array,
    feat_spec: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Enhanced forward pass.

    Args:
        model: DfNet4 model
        spec: Input spectrum (real, imag)
        feat_erb: ERB features
        feat_spec: DF features

    Returns:
        Enhanced spectrum (real, imag)
    """
    return model(spec, feat_erb, feat_spec, training=False)


def enhance(
    model: DfNet4,
    audio: mx.array,
    params: ModelParams4,
    compensate_delay: bool = True,
    atten_lim_db: Optional[float] = None,
) -> mx.array:
    """Enhance a single audio signal.

    Args:
        model: Loaded DfNet4 model
        audio: Audio waveform (samples,) or (batch, samples)
        params: Model parameters
        compensate_delay: Whether to pad for delay compensation
        atten_lim_db: Optional attenuation limit in dB

    Returns:
        Enhanced audio waveform
    """
    from .ops import istft, stft

    # Handle 1D input
    input_1d = audio.ndim == 1
    if input_1d:
        audio = mx.expand_dims(audio, axis=0)

    orig_len = audio.shape[-1]
    n_fft = params.fft_size
    hop = params.hop_size

    # Pad for delay compensation
    if compensate_delay:
        audio = mx.pad(audio, [(0, 0), (0, n_fft)])

    # STFT
    spec_real, spec_imag = stft(audio, n_fft=n_fft, hop_length=hop)

    # Compute features
    mag = mx.sqrt(spec_real**2 + spec_imag**2 + 1e-8)
    erb_fb = model._erb_fb
    feat_erb = mx.matmul(mag, erb_fb)

    # DF features (first nb_df bins)
    nb_df = params.nb_df
    feat_spec = mx.stack(
        [spec_real[:, :, :nb_df], spec_imag[:, :, :nb_df]],
        axis=-1,
    )

    # Enhanced forward pass
    spec_out = enhance_frame(model, (spec_real, spec_imag), feat_erb, feat_spec)
    spec_out_real, spec_out_imag = spec_out

    # Apply attenuation limit if specified
    if atten_lim_db is not None and abs(atten_lim_db) > 0:
        lim = 10 ** (-abs(atten_lim_db) / 20)
        spec_out_real = spec_real * lim + spec_out_real * (1 - lim)
        spec_out_imag = spec_imag * lim + spec_out_imag * (1 - lim)

    # iSTFT
    enhanced = istft((spec_out_real, spec_out_imag), n_fft=n_fft, hop_length=hop)

    # Compensate for delay
    if compensate_delay:
        d = n_fft - hop
        enhanced = enhanced[:, d : orig_len + d]

    # Remove batch dimension if input was 1D
    if input_1d:
        enhanced = mx.squeeze(enhanced, axis=0)

    return enhanced


def enhance_streaming(
    model: DfNet4,
    audio_iterator: Iterator[mx.array],
    params: ModelParams4,
    chunk_size_samples: int,
) -> Iterator[mx.array]:
    """Stream-based enhancement for real-time processing.

    Args:
        model: Loaded DfNet4 model
        audio_iterator: Iterator yielding audio chunks
        params: Model parameters
        chunk_size_samples: Number of samples per chunk

    Yields:
        Enhanced audio chunks
    """
    if chunk_size_samples <= 0:
        raise ValueError(f"chunk_size_samples must be > 0, got {chunk_size_samples}")

    streaming_model = StreamingDfNet4(model)
    state = streaming_model.init_state(batch_size=1)

    hop_size = params.hop_size
    buffer = mx.zeros((1, 0))

    for chunk in audio_iterator:
        # Handle 1D chunks
        if chunk.ndim == 1:
            chunk = mx.expand_dims(chunk, axis=0)

        # Append to buffer
        buffer = mx.concatenate([buffer, chunk], axis=1)

        # Process complete frames
        while buffer.shape[1] >= hop_size:
            frame = buffer[:, :hop_size]
            buffer = buffer[:, hop_size:]

            # Process frame
            enhanced_frame, state = streaming_model.process_frame(frame, state)
            yield mx.squeeze(enhanced_frame, axis=0)

    # Flush remaining samples
    if buffer.shape[1] > 0:
        # Pad to hop_size
        pad_len = hop_size - buffer.shape[1]
        if pad_len > 0:
            buffer = mx.pad(buffer, [(0, 0), (0, pad_len)])
        enhanced_frame, _ = streaming_model.process_frame(buffer, state)
        yield mx.squeeze(enhanced_frame, axis=0)


def enhance_file(
    model: DfNet4,
    params: ModelParams4,
    input_path: str,
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    compensate_delay: bool = True,
    atten_lim_db: Optional[float] = None,
    speech_boost: Optional[SpeechBoostConfig] = None,
    speech_boost_vad: Optional[SileroVAD] = None,
) -> str:
    """Enhance a single audio file.

    Args:
        model: Loaded DfNet4 model
        params: Model parameters
        input_path: Path to input audio file
        output_dir: Output directory
        suffix: Suffix for output filename
        compensate_delay: Whether to pad for delay compensation
        atten_lim_db: Optional attenuation limit in dB
        speech_boost: Optional VAD-based speech amplification settings
        speech_boost_vad: Optional pre-initialized Silero VAD instance

    Returns:
        Path to enhanced audio file
    """
    # Load audio
    audio, orig_sr = load_audio(input_path, target_sr=params.sr)

    t0 = time.time()

    # Enhance
    enhanced = enhance(
        model,
        audio,
        params,
        compensate_delay=compensate_delay,
        atten_lim_db=atten_lim_db,
    )
    mx.eval(enhanced)

    t1 = time.time()

    # Resample back to original rate if needed
    if orig_sr != params.sr:
        enhanced_np = resample(enhanced, params.sr, orig_sr)
    else:
        enhanced_np = np.array(enhanced)

    if speech_boost is not None and speech_boost.gain_db > 0.0:
        vad = speech_boost_vad or _init_speech_boost_vad(speech_boost)
        if vad is None:
            raise RuntimeError("Failed to initialize Silero VAD for speech boost")

        enhanced_np, segments = vad.apply_speech_gain(
            enhanced_np,
            sample_rate=orig_sr,
            gain_db=speech_boost.gain_db,
            threshold=speech_boost.threshold,
            min_speech_duration_ms=speech_boost.min_speech_duration_ms,
            min_silence_duration_ms=speech_boost.min_silence_duration_ms,
            speech_pad_ms=speech_boost.speech_pad_ms,
            ramp_ms=speech_boost.ramp_ms,
            peak_limit=speech_boost.peak_limit,
        )
        segment_count = len(segments[0]) if segments else 0
        logger.info(
            "Applied speech boost (+{:.1f} dB) on {} detected segment(s)".format(
                speech_boost.gain_db,
                segment_count,
            )
        )

    # Calculate RTF
    audio_duration = len(audio) / params.sr
    processing_time = t1 - t0
    rtf = processing_time / audio_duration

    # Save
    out_path = save_audio(
        enhanced_np,
        input_path,
        sr=orig_sr,
        output_dir=output_dir,
        suffix=suffix,
    )

    logger.info(f"Enhanced '{os.path.basename(input_path)}' in {processing_time:.2f}s " f"(RT factor: {rtf:.3f})")

    return out_path


def enhance_batch(
    model: DfNet4,
    params: ModelParams4,
    input_paths: List[str],
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    compensate_delay: bool = True,
    atten_lim_db: Optional[float] = None,
    chunk_size_ms: float = 100.0,
    speech_boost: Optional[SpeechBoostConfig] = None,
) -> List[str]:
    """Enhance multiple audio files.

    Args:
        model: Loaded DfNet4 model
        params: Model parameters
        input_paths: List of input file paths
        output_dir: Output directory
        suffix: Suffix for output filenames
        compensate_delay: Whether to pad for delay compensation
        atten_lim_db: Optional attenuation limit in dB

    Returns:
        List of paths to enhanced audio files
    """
    output_paths = []
    n_files = len(input_paths)

    speech_boost_vad = _init_speech_boost_vad(speech_boost)

    for i, path in enumerate(input_paths):
        progress = (i + 1) / n_files * 100
        logger.info(f"[{progress:5.1f}%] Processing: {os.path.basename(path)}")

        out_path = enhance_file(
            model,
            params,
            path,
            output_dir=output_dir,
            suffix=suffix,
            compensate_delay=compensate_delay,
            atten_lim_db=atten_lim_db,
            speech_boost=speech_boost,
            speech_boost_vad=speech_boost_vad,
        )
        output_paths.append(out_path)

    return output_paths


def enhance_file_streaming(
    model: DfNet4,
    params: ModelParams4,
    input_path: str,
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    compensate_delay: bool = True,
    atten_lim_db: Optional[float] = None,
    chunk_size_ms: float = 100.0,
    speech_boost: Optional[SpeechBoostConfig] = None,
    speech_boost_vad: Optional[SileroVAD] = None,
) -> str:
    """Enhance a single file using frame-by-frame streaming inference."""
    if atten_lim_db is not None:
        raise ValueError("--atten-lim is not supported with --streaming")

    if not compensate_delay:
        logger.warning("--no-delay-compensation is ignored in streaming mode")

    if chunk_size_ms <= 0:
        raise ValueError(f"chunk_size_ms must be > 0, got {chunk_size_ms}")

    audio, orig_sr = load_audio(input_path, target_sr=params.sr)
    chunk_size_samples = max(params.hop_size, int(params.sr * (chunk_size_ms / 1000.0)))

    def chunk_iter() -> Iterator[mx.array]:
        n_samples = int(audio.shape[-1])
        for start in range(0, n_samples, chunk_size_samples):
            yield audio[start : start + chunk_size_samples]

    t0 = time.time()
    enhanced_chunks = list(
        enhance_streaming(
            model,
            chunk_iter(),
            params,
            chunk_size_samples=chunk_size_samples,
        )
    )
    if enhanced_chunks:
        enhanced = mx.concatenate(enhanced_chunks, axis=0)
        enhanced = enhanced[: audio.shape[-1]]
    else:
        enhanced = mx.zeros_like(audio)
    mx.eval(enhanced)
    t1 = time.time()

    if orig_sr != params.sr:
        enhanced_np = resample(enhanced, params.sr, orig_sr)
    else:
        enhanced_np = np.array(enhanced)

    if speech_boost is not None and speech_boost.gain_db > 0.0:
        vad = speech_boost_vad or _init_speech_boost_vad(speech_boost)
        if vad is None:
            raise RuntimeError("Failed to initialize Silero VAD for speech boost")

        enhanced_np, segments = vad.apply_speech_gain(
            enhanced_np,
            sample_rate=orig_sr,
            gain_db=speech_boost.gain_db,
            threshold=speech_boost.threshold,
            min_speech_duration_ms=speech_boost.min_speech_duration_ms,
            min_silence_duration_ms=speech_boost.min_silence_duration_ms,
            speech_pad_ms=speech_boost.speech_pad_ms,
            ramp_ms=speech_boost.ramp_ms,
            peak_limit=speech_boost.peak_limit,
        )
        segment_count = len(segments[0]) if segments else 0
        logger.info(
            "Applied speech boost (+{:.1f} dB) on {} detected segment(s)".format(
                speech_boost.gain_db,
                segment_count,
            )
        )

    audio_duration = len(audio) / params.sr
    processing_time = t1 - t0
    rtf = processing_time / max(audio_duration, 1e-8)

    out_path = save_audio(
        enhanced_np,
        input_path,
        sr=orig_sr,
        output_dir=output_dir,
        suffix=suffix,
    )
    logger.info(
        f"Streaming-enhanced '{os.path.basename(input_path)}' in {processing_time:.2f}s " f"(RT factor: {rtf:.3f})"
    )
    return out_path


def enhance_batch_streaming(
    model: DfNet4,
    params: ModelParams4,
    input_paths: List[str],
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    compensate_delay: bool = True,
    atten_lim_db: Optional[float] = None,
    chunk_size_ms: float = 100.0,
    speech_boost: Optional[SpeechBoostConfig] = None,
) -> List[str]:
    """Enhance multiple files using streaming inference."""
    output_paths = []
    n_files = len(input_paths)
    speech_boost_vad = _init_speech_boost_vad(speech_boost)

    for i, path in enumerate(input_paths):
        progress = (i + 1) / n_files * 100
        logger.info(f"[{progress:5.1f}%] Streaming: {os.path.basename(path)}")
        out_path = enhance_file_streaming(
            model,
            params,
            path,
            output_dir=output_dir,
            suffix=suffix,
            compensate_delay=compensate_delay,
            atten_lim_db=atten_lim_db,
            chunk_size_ms=chunk_size_ms,
            speech_boost=speech_boost,
            speech_boost_vad=speech_boost_vad,
        )
        output_paths.append(out_path)
    return output_paths


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="DeepFilterNet4 MLX - Speech Enhancement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model directory path or pretrained model name",
    )
    parser.add_argument(
        "--epoch",
        "-e",
        type=str,
        default="best",
        help="Checkpoint epoch: 'best', 'latest', or integer",
    )

    # Input/output arguments
    parser.add_argument(
        "input_files",
        type=str,
        nargs="*",
        help="Input audio files",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=None,
        help="Input directory containing audio files",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".",
        help="Output directory",
    )
    parser.add_argument(
        "--suffix",
        "-s",
        type=str,
        default=None,
        help="Suffix for output filenames",
    )

    # Enhancement arguments
    parser.add_argument(
        "--no-delay-compensation",
        action="store_true",
        help="Disable delay compensation",
    )
    parser.add_argument(
        "--atten-lim",
        "-a",
        type=float,
        default=None,
        help="Attenuation limit in dB",
    )
    parser.add_argument(
        "--speech-boost-db",
        type=float,
        default=0.0,
        help="Boost dB applied only to Silero-detected speech segments (0 disables)",
    )
    parser.add_argument(
        "--speech-boost-threshold",
        type=float,
        default=0.5,
        help="Silero speech probability threshold for segment detection",
    )
    parser.add_argument(
        "--speech-boost-min-speech-ms",
        type=int,
        default=250,
        help="Minimum speech segment length in milliseconds",
    )
    parser.add_argument(
        "--speech-boost-min-silence-ms",
        type=int,
        default=100,
        help="Minimum silence length to split speech segments (milliseconds)",
    )
    parser.add_argument(
        "--speech-boost-pad-ms",
        type=int,
        default=30,
        help="Padding added around detected speech segments (milliseconds)",
    )
    parser.add_argument(
        "--speech-boost-ramp-ms",
        type=float,
        default=8.0,
        help="Fade-in/out ramp around boosted segments (milliseconds)",
    )
    parser.add_argument(
        "--speech-boost-peak-limit",
        type=float,
        default=0.99,
        help="Peak limiter after speech boost (set <=0 to disable)",
    )
    parser.add_argument(
        "--speech-boost-silero-model-path",
        type=str,
        default=None,
        help="Optional path to silero_vad.onnx for speech-segment detection",
    )
    parser.add_argument(
        "--speech-boost-silero-sample-rate",
        type=int,
        default=16000,
        help="Silero VAD sample rate used for speech-segment detection",
    )

    # Processing arguments
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for real-time processing",
    )
    parser.add_argument(
        "--streaming-chunk-ms",
        type=float,
        default=100.0,
        help="Chunk size in milliseconds for streaming mode",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    return parser


def main(args: Optional[argparse.Namespace] = None):
    """Main entry point."""
    if args is None:
        parser = setup_argument_parser()
        args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=args.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    # Get input files
    if args.input_dir:
        input_files = glob.glob(os.path.join(args.input_dir, "*"))
        # Filter to audio files
        audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".opus"}
        input_files = [f for f in input_files if os.path.splitext(f)[1].lower() in audio_exts]
    elif args.input_files:
        input_files = args.input_files
    else:
        logger.error("No input files specified")
        return 1

    if not input_files:
        logger.error("No audio files found")
        return 1

    logger.info(f"Found {len(input_files)} audio file(s)")

    # Load model
    try:
        model, params, default_suffix, epoch = load_model(
            model_path=args.model,
            epoch=args.epoch,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    suffix = args.suffix or default_suffix

    speech_boost = SpeechBoostConfig(
        gain_db=float(args.speech_boost_db),
        threshold=float(args.speech_boost_threshold),
        min_speech_duration_ms=int(args.speech_boost_min_speech_ms),
        min_silence_duration_ms=int(args.speech_boost_min_silence_ms),
        speech_pad_ms=int(args.speech_boost_pad_ms),
        ramp_ms=float(args.speech_boost_ramp_ms),
        peak_limit=float(args.speech_boost_peak_limit),
        silero_model_path=args.speech_boost_silero_model_path,
        silero_sample_rate=int(args.speech_boost_silero_sample_rate),
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Enhance files
    t_start = time.time()

    enhance_fn = enhance_batch_streaming if args.streaming else enhance_batch
    output_paths = enhance_fn(
        model,
        params,
        input_files,
        output_dir=args.output_dir,
        suffix=suffix,
        compensate_delay=not args.no_delay_compensation,
        atten_lim_db=args.atten_lim,
        chunk_size_ms=getattr(args, "streaming_chunk_ms", 100.0),
        speech_boost=speech_boost,
    )

    t_total = time.time() - t_start
    logger.info(f"Enhanced {len(output_paths)} file(s) in {t_total:.2f}s")

    return 0


def run():
    """Command-line entry point."""
    import sys

    sys.exit(main())


if __name__ == "__main__":
    run()
