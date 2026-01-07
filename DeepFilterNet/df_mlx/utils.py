"""Utility functions for MLX DeepFilterNet4.

This module provides utility functions for:
- Data loading and preprocessing
- Feature extraction
- Audio I/O
- Logging and metrics
"""

import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

import mlx.core as mx
import numpy as np

# ============================================================================
# Audio I/O
# ============================================================================


def load_audio(
    path: Union[str, Path],
    sr: int = 48000,
    mono: bool = True,
) -> Tuple[mx.array, int]:
    """Load audio file.

    Args:
        path: Path to audio file
        sr: Target sample rate (resamples if different)
        mono: Convert to mono if True

    Returns:
        Tuple of (audio array, sample rate)
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required: pip install soundfile")

    audio, file_sr = sf.read(str(path), dtype="float32")

    # Convert to mono
    if mono and audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if file_sr != sr:
        try:
            import resampy

            audio = resampy.resample(audio, file_sr, sr)
        except ImportError:
            print(f"Warning: resampy not installed, using {file_sr}Hz instead of {sr}Hz")
            sr = file_sr

    return mx.array(audio), sr


def save_audio(
    audio: mx.array,
    path: Union[str, Path],
    sr: int = 48000,
):
    """Save audio to file.

    Args:
        audio: Audio array
        path: Output path
        sr: Sample rate
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required: pip install soundfile")

    # Convert to numpy
    audio_np = np.array(audio)

    # Ensure correct shape
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()

    sf.write(str(path), audio_np, sr)


# ============================================================================
# Feature Extraction
# ============================================================================


def extract_features(
    audio: mx.array,
    fft_size: int = 960,
    hop_size: int = 480,
    nb_erb: int = 32,
    nb_df: int = 96,
    sr: int = 48000,
) -> Tuple[Tuple[mx.array, mx.array], mx.array, mx.array]:
    """Extract features for DeepFilterNet4.

    Args:
        audio: Audio waveform (batch, samples) or (samples,)
        fft_size: FFT size
        hop_size: Hop size
        nb_erb: Number of ERB bands
        nb_df: Number of DF bins
        sr: Sample rate

    Returns:
        Tuple of:
        - spec: Complex spectrum as (real, imag)
        - feat_erb: ERB features
        - feat_spec: DF features
    """
    from .ops import erb_fb, stft

    # Handle 1D input
    if audio.ndim == 1:
        audio = mx.expand_dims(audio, axis=0)

    # STFT
    spec_real, spec_imag = stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
    )

    # Magnitude for ERB features
    mag = mx.sqrt(spec_real**2 + spec_imag**2 + 1e-8)

    # ERB filterbank
    fb = erb_fb(sr=sr, fft_size=fft_size, nb_bands=nb_erb)
    feat_erb = mx.matmul(mag, fb)

    # DF features (complex spectrum of low frequencies)
    feat_spec = mx.stack(
        [
            spec_real[:, :, :nb_df],
            spec_imag[:, :, :nb_df],
        ],
        axis=-1,
    )

    return (spec_real, spec_imag), feat_erb, feat_spec


# ============================================================================
# Data Loading
# ============================================================================


class AudioDataset:
    """Simple audio dataset for training.

    Args:
        clean_dir: Directory containing clean audio files
        noisy_dir: Directory containing noisy audio files
        sr: Sample rate
        segment_length: Length of audio segments in samples
        fft_size: FFT size
        hop_size: Hop size
        nb_erb: Number of ERB bands
        nb_df: Number of DF bins
    """

    def __init__(
        self,
        clean_dir: Union[str, Path],
        noisy_dir: Union[str, Path],
        sr: int = 48000,
        segment_length: int = 48000,  # 1 second
        fft_size: int = 960,
        hop_size: int = 480,
        nb_erb: int = 32,
        nb_df: int = 96,
    ):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.sr = sr
        self.segment_length = segment_length
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.nb_erb = nb_erb
        self.nb_df = nb_df

        # Find audio files
        self.files = self._find_pairs()
        print(f"Found {len(self.files)} audio pairs")

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching clean/noisy pairs."""
        pairs = []

        extensions = [".wav", ".flac", ".mp3", ".ogg"]

        for ext in extensions:
            for clean_path in self.clean_dir.glob(f"*{ext}"):
                noisy_path = self.noisy_dir / clean_path.name
                if noisy_path.exists():
                    pairs.append((clean_path, noisy_path))

        return pairs

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        """Get a single sample."""
        clean_path, noisy_path = self.files[idx]

        # Load audio
        clean, _ = load_audio(clean_path, self.sr)
        noisy, _ = load_audio(noisy_path, self.sr)

        # Random crop to segment length
        if len(clean) > self.segment_length:
            start = np.random.randint(0, len(clean) - self.segment_length)
            clean = clean[start : start + self.segment_length]
            noisy = noisy[start : start + self.segment_length]
        elif len(clean) < self.segment_length:
            # Pad if too short
            pad_length = self.segment_length - len(clean)
            clean = mx.pad(clean, [(0, pad_length)])
            noisy = mx.pad(noisy, [(0, pad_length)])

        # Extract features
        spec, feat_erb, feat_spec = extract_features(
            noisy,
            self.fft_size,
            self.hop_size,
            self.nb_erb,
            self.nb_df,
            self.sr,
        )

        # Target features
        target_spec, _, _ = extract_features(
            clean,
            self.fft_size,
            self.hop_size,
            self.nb_erb,
            self.nb_df,
            self.sr,
        )

        return spec, feat_erb, feat_spec, target_spec


def create_dataloader(
    dataset: AudioDataset,
    batch_size: int = 8,
    shuffle: bool = True,
) -> Iterator:
    """Create a simple data loader.

    Args:
        dataset: AudioDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Yields:
        Batched data tuples
    """
    indices = list(range(len(dataset)))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]

            # Collect batch items
            batch_items = [dataset[idx] for idx in batch_indices]

            # Stack into batches
            spec_real = mx.stack([item[0][0] for item in batch_items])
            spec_imag = mx.stack([item[0][1] for item in batch_items])
            feat_erb = mx.stack([item[1] for item in batch_items])
            feat_spec = mx.stack([item[2] for item in batch_items])
            target_real = mx.stack([item[3][0] for item in batch_items])
            target_imag = mx.stack([item[3][1] for item in batch_items])

            yield (
                (spec_real, spec_imag),
                feat_erb,
                feat_spec,
                (target_real, target_imag),
            )


# ============================================================================
# Metrics
# ============================================================================


def compute_snr(
    clean: mx.array,
    enhanced: mx.array,
    eps: float = 1e-8,
) -> float:
    """Compute Signal-to-Noise Ratio.

    Args:
        clean: Clean reference signal
        enhanced: Enhanced signal
        eps: Small constant for stability

    Returns:
        SNR in dB
    """
    noise = enhanced - clean
    signal_power = float(mx.sum(clean**2))
    noise_power = float(mx.sum(noise**2))

    return 10 * np.log10((signal_power + eps) / (noise_power + eps))


def compute_pesq(
    clean: mx.array,
    enhanced: mx.array,
    sr: int = 48000,
) -> float:
    """Compute PESQ score.

    Args:
        clean: Clean reference (numpy or MLX array)
        enhanced: Enhanced signal
        sr: Sample rate

    Returns:
        PESQ score
    """
    try:
        from pesq import pesq
    except ImportError:
        print("Warning: pesq not installed, returning 0")
        return 0.0

    clean_np = np.array(clean).squeeze()
    enhanced_np = np.array(enhanced).squeeze()

    # PESQ requires 16kHz
    if sr != 16000:
        try:
            import resampy

            clean_np = resampy.resample(clean_np, sr, 16000)
            enhanced_np = resampy.resample(enhanced_np, sr, 16000)
            sr = 16000
        except ImportError:
            print("Warning: resampy not installed, cannot resample for PESQ")
            return 0.0

    return pesq(sr, clean_np, enhanced_np, "wb")


# ============================================================================
# Timing Utilities
# ============================================================================


class Timer:
    """Simple timer for profiling."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        if self.name:
            print(f"{self.name}: {self.elapsed * 1000:.2f}ms")


def benchmark_model(
    model,
    batch_size: int = 8,
    seq_length: int = 100,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict:
    """Benchmark model inference speed.

    Args:
        model: DfNet4 model
        batch_size: Batch size
        seq_length: Sequence length (time frames)
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs

    Returns:
        Dictionary with timing statistics
    """
    from .config import get_default_config

    p = get_default_config()

    # Create dummy inputs
    spec_real = mx.random.normal((batch_size, seq_length, p.n_freqs))
    spec_imag = mx.random.normal((batch_size, seq_length, p.n_freqs))
    feat_erb = mx.random.normal((batch_size, seq_length, p.nb_erb))
    feat_spec = mx.random.normal((batch_size, seq_length, p.nb_df, 2))

    # Warmup
    for _ in range(num_warmup):
        _ = model((spec_real, spec_imag), feat_erb, feat_spec)
        mx.eval(_)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        out = model((spec_real, spec_imag), feat_erb, feat_spec)
        mx.eval(out)
        times.append(time.perf_counter() - start)

    times = np.array(times)

    return {
        "mean_ms": float(times.mean() * 1000),
        "std_ms": float(times.std() * 1000),
        "min_ms": float(times.min() * 1000),
        "max_ms": float(times.max() * 1000),
        "throughput": float(batch_size / times.mean()),
    }
