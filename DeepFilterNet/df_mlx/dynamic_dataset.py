"""MLX Dynamic Dataset - Port of Rust libdfdata DataLoader.

This module provides dynamic audio mixing for training, matching the original
DeepFilterNet training pipeline:
- Dynamic speech + noise mixing at random SNR
- Random RIR convolution for reverb simulation
- Full dataset diversity (all noise/RIR files available each epoch)
- Various augmentations (clipping, bandwidth extension, etc.)

The key difference from pre-computed datastores:
- Same speech file can appear with different noise/RIR/SNR each epoch
- Full dataset diversity instead of cached subset
- Augmentations applied dynamically

Usage:
    from df_mlx.dynamic_dataset import DynamicDataset, DatasetConfig

    config = DatasetConfig(
        speech_files=speech_list,
        noise_files=noise_list,
        rir_files=rir_list,
        sample_rate=48000,
        segment_length=5.0,
    )
    dataset = DynamicDataset(config)

    # Training loop
    for epoch in range(num_epochs):
        dataset.set_epoch(epoch)  # Re-randomize combinations
        for batch in dataset.iter_batches(batch_size=8):
            # batch contains: noisy_spec, clean_spec, feat_erb, feat_spec
            ...
"""

import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from scipy import signal as scipy_signal

# Optional mlx-data import (for MLXDataStream)
try:
    import mlx.data as dx

    HAS_MLX_DATA = True
except ImportError:
    dx = None
    HAS_MLX_DATA = False

# Try to import soundfile, fall back to scipy.io.wavfile
try:
    import soundfile as sf

    def load_audio_file(path: str, sr: int) -> np.ndarray:
        """Load audio file and resample if needed."""
        audio, file_sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            num_samples = int(len(audio) * sr / file_sr)
            audio = np.asarray(scipy_signal.resample(audio, num_samples))
        return audio.astype(np.float32)

except ImportError:
    from scipy.io import wavfile

    def load_audio_file(path: str, sr: int) -> np.ndarray:
        """Load audio file and resample if needed."""
        file_sr, audio = wavfile.read(path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            num_samples = int(len(audio) * sr / file_sr)
            audio = np.asarray(scipy_signal.resample(audio, num_samples))
        return audio.astype(np.float32)


@dataclass
class DatasetConfig:
    """Configuration for dynamic dataset."""

    # Cache directory (preferred - from build_audio_cache.py)
    cache_dir: Optional[str] = None

    # File lists (used if cache_dir is None - slower, loads raw audio)
    speech_files: List[str] = field(default_factory=list)
    noise_files: List[str] = field(default_factory=list)
    rir_files: List[str] = field(default_factory=list)

    # Audio parameters
    sample_rate: int = 48000
    segment_length: float = 5.0  # seconds
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_df: int = 96

    # Mixing parameters
    snr_range: Tuple[float, float] = (-5.0, 40.0)  # dB, matching Rust [-5, 0, 5, 10, 20, 40]
    gain_range: Tuple[float, float] = (-6.0, 6.0)  # dB

    # Augmentation probabilities
    p_reverb: float = 0.5  # Probability of applying RIR
    p_clipping: float = 0.0  # Probability of clipping distortion
    p_bandwidth_ext: float = 0.0  # Probability of bandwidth extension
    p_interfer_speech: float = 0.0  # Probability of interfering speaker

    # Noise mixing
    n_noise_min: int = 2  # Minimum noises to combine
    n_noise_max: int = 5  # Maximum noises to combine
    p_random_noise: float = 0.05  # Probability of synthetic noise

    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2
    seed: int = 42

    # Splits
    train_split: float = 0.9
    valid_split: float = 0.05

    @classmethod
    def from_json(cls, path: str) -> "DatasetConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        # Handle cache_dir from build_audio_cache.py
        if "cache_dir" in data:
            data["cache_dir"] = data["cache_dir"]
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k) or k == "cache_dir"})

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


class ShardedAudioCache:
    """Load audio from pre-built sharded NPZ cache.

    This is the efficient loader that reads from the cache created by
    build_audio_cache.py. Audio is pre-processed (resampled, normalized)
    so loading is just a numpy array read.
    """

    def __init__(self, cache_dir: str, category: str):
        """Initialize cache loader.

        Args:
            cache_dir: Path to cache directory (containing index.json)
            category: 'speech', 'noise', or 'rir'
        """
        self.cache_dir = Path(cache_dir)
        self.category = category
        self.shard_dir = self.cache_dir / category

        # Load index
        index_path = self.cache_dir / "index.json"
        with open(index_path) as f:
            all_indices = json.load(f)

        self.index: Dict[str, Tuple[str, str]] = {}
        if category in all_indices:
            self.index = {k: tuple(v) for k, v in all_indices[category].items()}

        # Get list of available files
        self.files = list(self.index.keys())

        # Cache for loaded shards - keep NpzFile objects open for lazy loading
        self._shard_cache: Dict[str, Any] = {}  # NpzFile objects
        self._shard_access: List[str] = []
        self._max_shards = 20  # Keep up to 20 shards in memory (lazy, so minimal RAM)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.files)

    def _get_shard(self, shard_rel_path: str) -> Any:  # Returns NpzFile
        """Get a shard NpzFile, loading from disk if needed.

        Uses lazy loading - the NpzFile object is kept open and arrays are
        loaded on-demand when accessed, not all at once.

        Args:
            shard_rel_path: Relative path from cache_dir (e.g., "speech/shard_0000.npz")
        """
        with self._lock:
            if shard_rel_path in self._shard_cache:
                # Move to end (most recently used)
                self._shard_access.remove(shard_rel_path)
                self._shard_access.append(shard_rel_path)
                return self._shard_cache[shard_rel_path]

        # Open NpzFile lazily (arrays loaded on access, not upfront)
        shard_path = self.cache_dir / shard_rel_path
        npz_file = np.load(shard_path, mmap_mode="r")

        with self._lock:
            # Evict oldest if at capacity
            while len(self._shard_cache) >= self._max_shards:
                oldest = self._shard_access.pop(0)
                old_npz = self._shard_cache.pop(oldest)
                old_npz.close()

            self._shard_cache[shard_rel_path] = npz_file
            self._shard_access.append(shard_rel_path)

        return npz_file

    def load(self, path: str) -> np.ndarray:
        """Load audio array by original file path."""
        if path not in self.index:
            raise KeyError(f"File not in cache: {path}")

        shard_name, key = self.index[path]
        npz_file = self._get_shard(shard_name)
        # Access the specific array - this triggers lazy load of just that array
        return np.asarray(npz_file[key])

    def load_random(self) -> np.ndarray:
        """Load a random audio file from the cache."""
        path = random.choice(self.files)
        return self.load(path)

    def clear(self) -> None:
        """Clear the shard cache."""
        with self._lock:
            self._shard_cache.clear()
            self._shard_access.clear()


class AudioCache:
    """Thread-safe LRU cache for loaded audio files.

    DEPRECATED: Use ShardedAudioCache with pre-built cache instead.
    This class is kept for compatibility with raw audio file loading.
    """

    def __init__(self, max_size: int = 1000, sample_rate: int = 48000):
        self.max_size = max_size
        self.sample_rate = sample_rate
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()

    def get(self, path: str) -> Optional[np.ndarray]:
        """Get audio from cache if available."""
        with self._lock:
            if path in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(path)
                self._access_order.append(path)
                return self._cache[path]
        return None

    def put(self, path: str, audio: np.ndarray) -> None:
        """Add audio to cache, evicting oldest if necessary."""
        with self._lock:
            if path in self._cache:
                return

            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]

            self._cache[path] = audio
            self._access_order.append(path)

    def load(self, path: str) -> np.ndarray:
        """Load audio from cache or disk."""
        cached = self.get(path)
        if cached is not None:
            return cached

        audio = load_audio_file(path, self.sample_rate)
        self.put(path, audio)
        return audio

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


class NoiseGenerator:
    """Generate synthetic colored noise (white, pink, brown, etc.)."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def generate(
        self,
        f_decay: float,
        num_samples: int,
    ) -> np.ndarray:
        """Generate colored noise with given spectral decay.

        Args:
            f_decay: Decay exponent. 0=white, 1=pink, 2=brown, -1=blue
            num_samples: Number of samples to generate

        Returns:
            Generated noise signal
        """
        # Generate white noise in frequency domain
        fft_size = num_samples
        freqs = np.fft.rfftfreq(fft_size, 1.0 / self.sample_rate)

        # Avoid division by zero
        freqs[0] = 1.0

        # Create magnitude spectrum with 1/f^decay shape
        magnitudes = 1.0 / (freqs**f_decay)
        magnitudes[0] = 0  # No DC component

        # Random phases
        phases = np.random.uniform(0, 2 * np.pi, len(magnitudes))

        # Create complex spectrum
        spectrum = magnitudes * np.exp(1j * phases)

        # Convert to time domain
        noise = np.fft.irfft(spectrum, n=fft_size)

        # Normalize
        noise = noise / (np.abs(noise).max() + 1e-10)

        return noise.astype(np.float32)

    def generate_random(
        self,
        num_samples: int,
        f_decay_range: Tuple[float, float] = (-2.0, 2.0),
    ) -> np.ndarray:
        """Generate noise with random spectral characteristics."""
        f_decay = random.uniform(*f_decay_range)
        return self.generate(f_decay, num_samples)


class ReverbSimulator:
    """Apply room impulse response (RIR) convolution for reverb.

    Ports the Rust RandReverbSim functionality:
    - Trim RIR based on energy threshold
    - Optionally suppress late reflections
    - Efficient FFT-based convolution
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        p_speech: float = 0.5,
        p_noise: float = 0.3,
        rt60: float = 0.8,
    ):
        self.sample_rate = sample_rate
        self.p_speech = p_speech
        self.p_noise = p_noise
        self.rt60 = rt60

    def trim_rir(self, rir: np.ndarray, threshold_db: float = -80.0) -> np.ndarray:
        """Trim RIR based on energy threshold."""
        # Find peak
        peak_idx = np.argmax(np.abs(rir))
        peak_level = np.abs(rir[peak_idx])

        # Threshold level
        min_level = peak_level * (10 ** (threshold_db / 20))

        # Find last sample above threshold
        above_threshold = np.abs(rir) > min_level
        if not above_threshold.any():
            return rir[:1]

        last_idx = np.where(above_threshold)[0][-1]
        return rir[: last_idx + 1]

    def suppress_late(
        self,
        rir: np.ndarray,
        offset_samples: int,
        rt60: Optional[float] = None,
    ) -> np.ndarray:
        """Suppress late reflections with exponential decay."""
        if rt60 is None:
            rt60 = self.rt60

        if offset_samples >= len(rir):
            return rir

        dt = 1.0 / self.sample_rate
        rt60_level = 10 ** (-60 / 20)
        tau = -rt60 / np.log10(rt60_level)

        decay = np.ones_like(rir)
        t = np.arange(len(rir) - offset_samples) * dt
        decay[offset_samples:] = 10 ** (-t / tau)

        return rir * decay

    def convolve(
        self,
        audio: np.ndarray,
        rir: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Convolve audio with RIR using FFT."""
        # Normalize RIR
        if normalize:
            rir = rir / (np.sqrt(np.sum(rir**2)) + 1e-10)

        # Find direct path (peak) for alignment
        direct_idx = np.argmax(np.abs(rir))

        # FFT convolution
        convolved = scipy_signal.fftconvolve(audio, rir, mode="full")

        # Align to direct path
        convolved = convolved[direct_idx : direct_idx + len(audio)]

        # Ensure same length
        if len(convolved) < len(audio):
            convolved = np.pad(convolved, (0, len(audio) - len(convolved)))
        elif len(convolved) > len(audio):
            convolved = convolved[: len(audio)]

        return convolved.astype(np.float32)

    def apply(
        self,
        speech: np.ndarray,
        noise: np.ndarray,
        rir: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
        """Apply reverb to speech and/or noise based on probabilities.

        Returns:
            Tuple of (speech_out, noise_out, speech_modified, noise_modified)
        """
        # Preprocess RIR
        rir = self.trim_rir(rir)

        apply_speech = random.random() < self.p_speech
        apply_noise = random.random() < self.p_noise

        speech_out = speech
        noise_out = noise

        if apply_speech:
            speech_out = self.convolve(speech, rir)

        if apply_noise and len(rir) > 0:
            # For noise, often use different decay
            rir_noise = self.suppress_late(rir, int(0.05 * self.sample_rate))
            noise_out = self.convolve(noise, rir_noise)

        return speech_out, noise_out, apply_speech, apply_noise


class Augmentations:
    """Collection of audio augmentation transforms.

    Ports the Rust augmentations from libDF/src/augmentations.rs.
    """

    @staticmethod
    def clip(
        audio: np.ndarray,
        c: float = 0.5,
    ) -> np.ndarray:
        """Apply soft clipping to audio signal.

        Args:
            audio: Input audio
            c: Clipping threshold (0-1 range of max amplitude)

        Returns:
            Clipped audio
        """
        max_val = np.abs(audio).max()
        threshold = c * max_val
        return np.clip(audio, -threshold, threshold)

    @staticmethod
    def random_clip(
        audio: np.ndarray,
        prob: float = 0.1,
        c_range: Tuple[float, float] = (0.05, 0.5),
    ) -> np.ndarray:
        """Apply random clipping distortion.

        Args:
            audio: Input audio
            prob: Probability of applying clipping
            c_range: Range of clipping thresholds

        Returns:
            Possibly clipped audio
        """
        if random.random() > prob:
            return audio

        c = random.uniform(*c_range)
        return Augmentations.clip(audio, c)

    @staticmethod
    def biquad_filter(
        audio: np.ndarray,
        b: np.ndarray,
        a: np.ndarray,
    ) -> np.ndarray:
        """Apply biquad filter to audio."""
        result = scipy_signal.lfilter(b, a, audio)
        return np.asarray(result, dtype=np.float32)

    @staticmethod
    def high_pass(
        audio: np.ndarray,
        freq: float,
        q: float,
        sr: int,
    ) -> np.ndarray:
        """Apply high-pass filter."""
        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / (2 * q)
        cos_w0 = np.cos(w0)

        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = b0
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1 / a0, a2 / a0])
        return Augmentations.biquad_filter(audio, b, a)

    @staticmethod
    def low_pass(
        audio: np.ndarray,
        freq: float,
        q: float,
        sr: int,
    ) -> np.ndarray:
        """Apply low-pass filter."""
        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / (2 * q)
        cos_w0 = np.cos(w0)

        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = b0
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1 / a0, a2 / a0])
        return Augmentations.biquad_filter(audio, b, a)

    @staticmethod
    def bandwidth_limit(
        audio: np.ndarray,
        low_freq: float,
        high_freq: float,
        sr: int,
        q: float = 0.707,
    ) -> np.ndarray:
        """Limit bandwidth with high-pass and low-pass filters.

        Args:
            audio: Input audio
            low_freq: High-pass cutoff frequency
            high_freq: Low-pass cutoff frequency
            sr: Sample rate
            q: Q factor for filters

        Returns:
            Bandwidth-limited audio
        """
        filtered = audio
        if low_freq > 20:
            filtered = Augmentations.high_pass(filtered, low_freq, q, sr)
        if high_freq < sr / 2 - 100:
            filtered = Augmentations.low_pass(filtered, high_freq, q, sr)
        return filtered

    @staticmethod
    def random_eq(
        audio: np.ndarray,
        sr: int,
        prob: float = 0.2,
        n_bands: int = 3,
        gain_range: Tuple[float, float] = (-15.0, 15.0),
    ) -> np.ndarray:
        """Apply random EQ adjustments.

        Args:
            audio: Input audio
            sr: Sample rate
            prob: Probability of applying EQ
            n_bands: Number of EQ bands to apply
            gain_range: Range of gains in dB

        Returns:
            EQ'd audio
        """
        if random.random() > prob:
            return audio

        # Store original RMS
        rms_orig = np.sqrt(np.mean(audio**2))

        for _ in range(random.randint(1, n_bands)):
            # Random frequency (log-distributed)
            freq = np.exp(random.uniform(np.log(40), np.log(min(8000, sr / 2 - 100))))
            gain_db = random.uniform(*gain_range)
            q = random.uniform(0.5, 1.5)

            # Simple peaking EQ using scipy
            # This is a simplified version
            w0 = 2 * np.pi * freq / sr
            amp = 10 ** (gain_db / 40)
            alpha = np.sin(w0) / (2 * q)
            cos_w0 = np.cos(w0)

            b0 = 1 + alpha * amp
            b1 = -2 * cos_w0
            b2 = 1 - alpha * amp
            a0 = 1 + alpha / amp
            a1 = -2 * cos_w0
            a2 = 1 - alpha / amp

            b = np.array([b0, b1, b2]) / a0
            a = np.array([1.0, a1 / a0, a2 / a0])
            audio = Augmentations.biquad_filter(audio, b, a)

        # Restore RMS
        rms_new = np.sqrt(np.mean(audio**2))
        if rms_new > 1e-10:
            audio = audio * (rms_orig / rms_new)

        # Guard against clipping
        max_val = np.abs(audio).max()
        if max_val > 1.0 - 1e-10:
            audio = audio / (max_val + 1e-10)

        return audio

    @staticmethod
    def time_stretch(
        audio: np.ndarray,
        rate: float,
    ) -> np.ndarray:
        """Simple time stretch by resampling."""
        if abs(rate - 1.0) < 1e-6:
            return audio
        new_len = int(len(audio) / rate)
        result = scipy_signal.resample(audio, new_len)
        return np.asarray(result, dtype=np.float32)

    @staticmethod
    def random_gain(
        audio: np.ndarray,
        gain_range: Tuple[float, float] = (-6.0, 6.0),
    ) -> np.ndarray:
        """Apply random gain."""
        gain_db = random.uniform(*gain_range)
        gain = 10 ** (gain_db / 20)
        return (audio * gain).astype(np.float32)


def mix_audio(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    gain_db: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mix clean speech with noise at specified SNR.

    This matches the Rust mix_audio_signal function.

    Args:
        clean: Clean speech signal
        noise: Noise signal
        snr_db: Target SNR in dB
        gain_db: Gain to apply to speech in dB

    Returns:
        Tuple of (clean_out, noise_out, noisy_mixture)
    """
    # Apply gain to speech
    gain = 10 ** (gain_db / 20)
    clean_out = clean * gain

    # Match lengths
    if len(noise) < len(clean_out):
        repeats = int(np.ceil(len(clean_out) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[: len(clean_out)]

    # Compute mixing factor for target SNR
    clean_power = np.mean(clean_out**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    mix_factor = np.sqrt(target_noise_power / noise_power)

    noise_scaled = noise * mix_factor
    noisy = clean_out + noise_scaled

    # Guard against clipping
    max_val = max(
        np.abs(clean_out).max(),
        np.abs(noise_scaled).max(),
        np.abs(noisy).max(),
    )
    if max_val > 1.0 - 1e-10:
        scale = 1.0 / (max_val + 1e-10)
        clean_out = clean_out * scale
        noise_scaled = noise_scaled * scale
        noisy = noisy * scale

    return clean_out, noise_scaled, noisy


def combine_noises(
    noises: List[np.ndarray],
    target_len: int,
    gains_db: Optional[List[float]] = None,
) -> np.ndarray:
    """Combine multiple noise signals into one.

    Args:
        noises: List of noise signals
        target_len: Target output length
        gains_db: Optional gains for each noise

    Returns:
        Combined noise signal
    """
    if not noises:
        return np.zeros(target_len, dtype=np.float32)

    if gains_db is None:
        gains_db = [0.0] * len(noises)

    combined = np.zeros(target_len, dtype=np.float32)

    for noise, gain_db in zip(noises, gains_db):
        gain = 10 ** (gain_db / 20)

        # Random start position for this noise
        if len(noise) < target_len:
            # Repeat noise to fill
            repeats = int(np.ceil(target_len / len(noise)))
            noise = np.tile(noise, repeats)

        # Random offset
        max_offset = len(noise) - target_len
        if max_offset > 0:
            offset = random.randint(0, max_offset)
            noise = noise[offset : offset + target_len]
        else:
            noise = noise[:target_len]

        combined += noise * gain

    return combined


def compute_stft(
    audio: np.ndarray,
    fft_size: int = 960,
    hop_size: int = 480,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute STFT of audio signal.

    Args:
        audio: Input audio (samples,)
        fft_size: FFT size
        hop_size: Hop size
        window: Optional window function

    Returns:
        Complex STFT (time, freq)
    """
    if window is None:
        window = np.sqrt(np.hanning(fft_size + 1)[:-1]).astype(np.float32)

    # Pad audio
    pad_len = fft_size - hop_size
    audio_padded = np.pad(audio, (pad_len, pad_len), mode="constant")

    # Frame extraction using stride tricks
    num_frames = (len(audio_padded) - fft_size) // hop_size + 1
    shape = (num_frames, fft_size)
    strides = (audio_padded.strides[0] * hop_size, audio_padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio_padded, shape=shape, strides=strides, writeable=False)

    # Apply window and compute FFT
    windowed = frames * window
    stft = np.fft.rfft(windowed, n=fft_size, axis=-1)

    return stft


def create_erb_filterbank(
    sr: int = 48000,
    fft_size: int = 960,
    nb_erb: int = 32,
    min_freq: float = 20.0,
    max_freq: Optional[float] = None,
) -> np.ndarray:
    """Create ERB filterbank matrix."""
    if max_freq is None:
        max_freq = sr / 2

    n_freqs = fft_size // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)

    def hz_to_erb(f):
        return 9.265 * np.log(1 + f / (24.7 * 9.265))

    def erb_to_hz(erb):
        return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)

    erb_min = hz_to_erb(min_freq)
    erb_max = hz_to_erb(max_freq)
    erb_centers = np.linspace(erb_min, erb_max, nb_erb)
    center_freqs = erb_to_hz(erb_centers)

    fb = np.zeros((n_freqs, nb_erb), dtype=np.float32)

    for i in range(nb_erb):
        center = center_freqs[i]
        erb_bandwidth = 24.7 * (4.37 * center / 1000 + 1)
        low = center - erb_bandwidth / 2
        high = center + erb_bandwidth / 2

        for j, f in enumerate(freqs):
            if low <= f <= center:
                fb[j, i] = (f - low) / (center - low + 1e-10)
            elif center < f <= high:
                fb[j, i] = (high - f) / (high - center + 1e-10)

    fb = fb / (fb.sum(axis=0, keepdims=True) + 1e-10)
    return fb


def compute_erb_features(spec: np.ndarray, erb_fb: np.ndarray) -> np.ndarray:
    """Compute ERB band features from spectrum."""
    mag_sq = np.abs(spec) ** 2
    erb = np.matmul(mag_sq, erb_fb)
    erb = np.log10(np.maximum(erb, 1e-10))
    return erb.astype(np.float32)


def compute_df_features(spec: np.ndarray, nb_df: int = 96) -> np.ndarray:
    """Compute DF-band features (complex coefficients)."""
    df_spec = spec[:, :nb_df]
    df_feat = np.stack([df_spec.real, df_spec.imag], axis=-1)
    return df_feat.astype(np.float32)


@dataclass
class Sample:
    """A single training sample."""

    noisy_spec: np.ndarray  # Complex STFT of noisy mixture
    clean_spec: np.ndarray  # Complex STFT of clean speech
    feat_erb: np.ndarray  # ERB features
    feat_spec: np.ndarray  # DF-band features
    snr: float
    gain: float


class DynamicDataset:
    """Dynamic dataset with on-the-fly audio mixing.

    This is the MLX port of the Rust libdfdata DataLoader. It provides:
    - Dynamic speech + noise + RIR mixing each epoch
    - Full dataset diversity (no fixed cache)
    - Thread-safe prefetching
    - Configurable augmentations

    Supports two modes:
    1. Sharded cache (fast): Load from pre-built NPZ cache (build_audio_cache.py)
    2. Raw files (slow): Load from raw audio files on disk
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.segment_samples = int(config.segment_length * config.sample_rate)
        self.fft_size = config.fft_size
        self.hop_size = config.hop_size

        # Determine loading mode
        self._use_cache = config.cache_dir is not None

        if self._use_cache:
            # Fast path: load from sharded NPZ cache
            self.speech_cache = ShardedAudioCache(config.cache_dir, "speech")
            self.noise_cache = ShardedAudioCache(config.cache_dir, "noise")

            # RIR cache is optional
            rir_cache_dir = Path(config.cache_dir) / "rir"
            if rir_cache_dir.exists():
                self.rir_cache = ShardedAudioCache(config.cache_dir, "rir")
            else:
                self.rir_cache = None

            # Use files from cache index
            config.speech_files = self.speech_cache.files
            config.noise_files = self.noise_cache.files
            if self.rir_cache:
                config.rir_files = self.rir_cache.files
        else:
            # Slow path: load from raw audio files
            self.audio_cache = AudioCache(max_size=2000, sample_rate=config.sample_rate)

        # Split files into train/valid/test
        self._split_files()

        # Initialize components
        self.noise_generator = NoiseGenerator(sample_rate=config.sample_rate)
        self.reverb = ReverbSimulator(
            sample_rate=config.sample_rate,
            p_speech=config.p_reverb,
            p_noise=config.p_reverb * 0.5,
        )

        # Pre-compute filterbank
        self.erb_fb = create_erb_filterbank(
            sr=config.sample_rate,
            fft_size=config.fft_size,
            nb_erb=config.nb_erb,
        )

        # Pre-compute window
        self.window = np.sqrt(np.hanning(config.fft_size + 1)[:-1]).astype(np.float32)

        # Epoch and randomization
        self._epoch = 0
        self._rng = random.Random(config.seed)

        # Current split
        self._current_split = "train"
        self._indices: List[int] = []

    def _split_files(self) -> None:
        """Split speech files into train/valid/test."""
        files = self.config.speech_files.copy()
        random.Random(self.config.seed).shuffle(files)

        n = len(files)
        train_end = int(n * self.config.train_split)
        valid_end = train_end + int(n * self.config.valid_split)

        self.splits = {
            "train": files[:train_end],
            "valid": files[train_end:valid_end],
            "test": files[valid_end:],
        }

    def set_split(self, split: str) -> None:
        """Set the current split (train/valid/test)."""
        if split not in self.splits:
            raise ValueError(f"Unknown split: {split}")
        self._current_split = split
        self._regenerate_indices()

    def set_epoch(self, epoch: int) -> None:
        """Set epoch and regenerate shuffled indices."""
        self._epoch = epoch
        self._regenerate_indices()

    def _regenerate_indices(self) -> None:
        """Regenerate shuffled indices for current epoch/split."""
        n = len(self.splits[self._current_split])
        self._indices = list(range(n))
        epoch_rng = random.Random(self.config.seed + self._epoch)
        epoch_rng.shuffle(self._indices)

    def __len__(self) -> int:
        return len(self.splits[self._current_split])

    def _load_audio(self, path: str, cache_type: str = "speech") -> np.ndarray:
        """Load audio from cache or raw file.

        Args:
            path: File path (original path, used as key in cache)
            cache_type: 'speech', 'noise', or 'rir'
        """
        if self._use_cache:
            if cache_type == "speech":
                return self.speech_cache.load(path)
            elif cache_type == "noise":
                return self.noise_cache.load(path)
            elif cache_type == "rir" and self.rir_cache:
                return self.rir_cache.load(path)
        return self.audio_cache.load(path)

    def _load_speech(self, idx: int) -> Optional[np.ndarray]:
        """Load and prepare a speech sample."""
        files = self.splits[self._current_split]
        path = files[idx]

        try:
            audio = self._load_audio(path, "speech")

            # Skip if too short
            if len(audio) < self.segment_samples:
                return None

            # Extract random segment
            if len(audio) > self.segment_samples:
                start = self._rng.randint(0, len(audio) - self.segment_samples)
                audio = audio[start : start + self.segment_samples]

            return audio
        except Exception:
            return None

    def _load_noise(self) -> Tuple[np.ndarray, float]:
        """Load a random noise sample or generate synthetic noise."""
        # Occasionally generate synthetic noise
        if self._rng.random() < self.config.p_random_noise:
            noise = self.noise_generator.generate_random(self.segment_samples)
            gain = self._rng.choice([-24.0, -12.0, -6.0, 0.0])
            return noise, gain

        # Load from file
        noise_files = self.config.noise_files
        if not noise_files:
            # Fallback to white noise
            return self.noise_generator.generate(0.0, self.segment_samples), 0.0

        path = self._rng.choice(noise_files)
        try:
            noise = self._load_audio(path, "noise")
            gain = self._rng.uniform(*self.config.gain_range)
            return noise, gain
        except Exception:
            # Fallback
            return self.noise_generator.generate(0.0, self.segment_samples), 0.0

    def _load_rir(self) -> Optional[np.ndarray]:
        """Load a random RIR if available."""
        rir_files = self.config.rir_files
        if not rir_files:
            return None

        path = self._rng.choice(rir_files)
        try:
            return self._load_audio(path, "rir")
        except Exception:
            return None

    def get_sample(self, idx: int) -> Optional[Sample]:
        """Get a single processed sample.

        This implements the full mixing pipeline from the Rust DataLoader:
        1. Load speech
        2. Apply optional speech augmentations (EQ, clipping)
        3. Load and combine multiple noises
        4. Optionally apply RIR reverb
        5. Mix at random SNR/gain
        6. Compute STFT and features
        """
        # Set RNG seed for reproducibility within epoch
        sample_seed = self.config.seed + self._epoch * 1000000 + idx
        self._rng = random.Random(sample_seed)

        # Load speech
        speech = self._load_speech(self._indices[idx])
        if speech is None:
            return None

        # Sample SNR and gain
        snr = self._rng.uniform(*self.config.snr_range)
        gain = self._rng.uniform(*self.config.gain_range)

        # Load and combine multiple noises (2-5 like Rust)
        n_noises = self._rng.randint(self.config.n_noise_min, self.config.n_noise_max)
        noises = []
        noise_gains = []
        for _ in range(n_noises):
            noise, ng = self._load_noise()
            noises.append(noise)
            noise_gains.append(ng)

        combined_noise = combine_noises(noises, self.segment_samples, noise_gains)

        # Optionally apply RIR
        speech_for_mix = speech.copy()
        if self.config.rir_files and self._rng.random() < self.config.p_reverb:
            rir = self._load_rir()
            if rir is not None:
                speech_for_mix, combined_noise, _, _ = self.reverb.apply(speech, combined_noise, rir)

        # Apply augmentations to speech (training only)
        if self._current_split == "train":
            # Random clipping distortion
            if self.config.p_clipping > 0:
                speech_for_mix = Augmentations.random_clip(
                    speech_for_mix,
                    prob=self.config.p_clipping,
                    c_range=(0.1, 0.5),
                )

            # Random EQ (bandwidth extension effect)
            if self.config.p_bandwidth_ext > 0:
                speech_for_mix = Augmentations.random_eq(
                    speech_for_mix,
                    sr=self.sample_rate,
                    prob=self.config.p_bandwidth_ext,
                )

        # Mix
        clean_out, _, noisy = mix_audio(speech_for_mix, combined_noise, snr, gain)

        # Compute spectrograms
        noisy_spec = compute_stft(noisy, self.fft_size, self.hop_size, self.window)
        clean_spec = compute_stft(clean_out, self.fft_size, self.hop_size, self.window)

        # Compute features
        feat_erb = compute_erb_features(noisy_spec, self.erb_fb)
        feat_spec = compute_df_features(noisy_spec, self.config.nb_df)

        return Sample(
            noisy_spec=noisy_spec,
            clean_spec=clean_spec,
            feat_erb=feat_erb,
            feat_spec=feat_spec,
            snr=snr,
            gain=gain,
        )

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all samples in current split."""
        for idx in range(len(self)):
            sample = self.get_sample(idx)
            if sample is not None:
                yield sample

    def iter_batches(
        self,
        batch_size: int,
        drop_last: bool = True,
    ) -> Iterator[Dict[str, mx.array]]:
        """Iterate over batches with prefetching.

        Args:
            batch_size: Number of samples per batch
            drop_last: Whether to drop the last incomplete batch

        Yields:
            Dict with batched MLX arrays:
            - noisy_real: (B, T, F) noisy spectrum real part
            - noisy_imag: (B, T, F) noisy spectrum imaginary part
            - clean_real: (B, T, F) clean spectrum real part
            - clean_imag: (B, T, F) clean spectrum imaginary part
            - feat_erb: (B, T, E) ERB features
            - feat_spec: (B, T, D, 2) DF-band features
            - snr: (B,) SNR values
        """
        batch_noisy_real = []
        batch_noisy_imag = []
        batch_clean_real = []
        batch_clean_imag = []
        batch_erb = []
        batch_spec = []
        batch_snr = []

        for sample in self.iter_samples():
            batch_noisy_real.append(sample.noisy_spec.real)
            batch_noisy_imag.append(sample.noisy_spec.imag)
            batch_clean_real.append(sample.clean_spec.real)
            batch_clean_imag.append(sample.clean_spec.imag)
            batch_erb.append(sample.feat_erb)
            batch_spec.append(sample.feat_spec)
            batch_snr.append(sample.snr)

            if len(batch_noisy_real) >= batch_size:
                yield {
                    "noisy_real": mx.array(np.stack(batch_noisy_real)),
                    "noisy_imag": mx.array(np.stack(batch_noisy_imag)),
                    "clean_real": mx.array(np.stack(batch_clean_real)),
                    "clean_imag": mx.array(np.stack(batch_clean_imag)),
                    "feat_erb": mx.array(np.stack(batch_erb)),
                    "feat_spec": mx.array(np.stack(batch_spec)),
                    "snr": mx.array(np.array(batch_snr)),
                }
                batch_noisy_real = []
                batch_noisy_imag = []
                batch_clean_real = []
                batch_clean_imag = []
                batch_erb = []
                batch_spec = []
                batch_snr = []

        # Handle last batch
        if batch_noisy_real and not drop_last:
            yield {
                "noisy_real": mx.array(np.stack(batch_noisy_real)),
                "noisy_imag": mx.array(np.stack(batch_noisy_imag)),
                "clean_real": mx.array(np.stack(batch_clean_real)),
                "clean_imag": mx.array(np.stack(batch_clean_imag)),
                "feat_erb": mx.array(np.stack(batch_erb)),
                "feat_spec": mx.array(np.stack(batch_spec)),
                "snr": mx.array(np.array(batch_snr)),
            }


class PrefetchDataLoader:
    """DataLoader with background prefetching for better GPU utilization.

    Uses a thread pool to load and process samples in the background while
    the GPU is processing the current batch.
    """

    def __init__(
        self,
        dataset: DynamicDataset,
        batch_size: int,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate with background prefetching."""
        # Queue to hold prefetched batches
        prefetch_queue: Queue = Queue(maxsize=self.prefetch_factor)
        stop_event = threading.Event()

        def worker():
            """Background worker that fills the prefetch queue."""
            try:
                for batch in self.dataset.iter_batches(self.batch_size, self.drop_last):
                    if stop_event.is_set():
                        break
                    prefetch_queue.put(batch)
            finally:
                prefetch_queue.put(None)  # Signal completion

        # Start worker thread
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        try:
            while True:
                batch = prefetch_queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            stop_event.set()
            worker_thread.join(timeout=1.0)

    def __len__(self) -> int:
        """Approximate number of batches."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def read_file_list(path: str) -> List[str]:
    """Read list of audio file paths from text file."""
    files = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "\t" in line:
                    line = line.split("\t")[0]
                files.append(line)
    return files


def create_dataset_from_lists(
    speech_list: str,
    noise_list: str,
    rir_list: Optional[str] = None,
    **kwargs,
) -> DynamicDataset:
    """Convenience function to create dataset from file lists.

    Args:
        speech_list: Path to speech file list
        noise_list: Path to noise file list
        rir_list: Optional path to RIR file list
        **kwargs: Additional DatasetConfig arguments

    Returns:
        Configured DynamicDataset
    """
    speech_files = read_file_list(speech_list)
    noise_files = read_file_list(noise_list)
    rir_files = read_file_list(rir_list) if rir_list else []

    config = DatasetConfig(
        speech_files=speech_files,
        noise_files=noise_files,
        rir_files=rir_files,
        **kwargs,
    )

    return DynamicDataset(config)


# =============================================================================
# MLX-Data Integration for High-Throughput Training
# =============================================================================


@dataclass
class CheckpointState:
    """Checkpoint state for resuming interrupted training.

    This captures the minimal state needed to resume training from
    exactly where it left off, ensuring reproducibility.
    """

    epoch: int = 0
    batch_idx: int = 0
    samples_processed: int = 0
    seed: int = 42
    split: str = "train"
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
            "samples_processed": self.samples_processed,
            "seed": self.seed,
            "split": self.split,
            "timestamp": self.timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Deserialize checkpoint from dictionary."""
        return cls(
            epoch=data.get("epoch", 0),
            batch_idx=data.get("batch_idx", 0),
            samples_processed=data.get("samples_processed", 0),
            seed=data.get("seed", 42),
            split=data.get("split", "train"),
            timestamp=data.get("timestamp", ""),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save checkpoint to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CheckpointState":
        """Load checkpoint from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


class MLXDataStream:
    """High-throughput data loader using mlx-data.

    This class wraps the DynamicDataset with mlx-data's Stream API to provide:
    - Parallel sample loading via prefetch with multiple threads
    - Automatic batching with configurable batch size
    - Checkpoint/resume support for interrupted training
    - Memory-efficient streaming iteration

    The key performance improvement comes from mlx-data's prefetch mechanism
    which loads samples in parallel background threads while the GPU processes
    the current batch.

    Example:
        config = DatasetConfig(cache_dir="./audio_cache", ...)
        dataset = DynamicDataset(config)
        stream = MLXDataStream(dataset, batch_size=8, num_workers=8)

        for epoch in range(num_epochs):
            stream.set_epoch(epoch)
            for batch in stream:
                # Train on batch
                ...
            stream.save_checkpoint("checkpoint.json")

    Resume example:
        stream = MLXDataStream.from_checkpoint(dataset, "checkpoint.json")
        for batch in stream:  # Continues from where it left off
            ...
    """

    def __init__(
        self,
        dataset: DynamicDataset,
        batch_size: int = 8,
        prefetch_size: int = 8,
        num_workers: int = 8,
        drop_last: bool = True,
        checkpoint: Optional[CheckpointState] = None,
    ):
        """Initialize MLXDataStream.

        Args:
            dataset: DynamicDataset instance with audio data
            batch_size: Number of samples per batch
            prefetch_size: Number of batches to prefetch in background
            num_workers: Number of parallel worker threads for loading
            drop_last: Whether to drop the last incomplete batch
            checkpoint: Optional checkpoint state for resuming
        """
        if not HAS_MLX_DATA:
            raise ImportError("mlx-data is required for MLXDataStream. " "Install with: pip install mlx-data")

        self.dataset = dataset
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

        # Initialize checkpoint state
        self._checkpoint = checkpoint or CheckpointState(seed=dataset.config.seed)

        # Sync dataset state with checkpoint
        self.dataset.set_split(self._checkpoint.split)
        self.dataset.set_epoch(self._checkpoint.epoch)

        # Track iteration state
        self._stream: Optional[Any] = None
        self._batch_count = 0

    @classmethod
    def from_checkpoint(
        cls,
        dataset: DynamicDataset,
        checkpoint_path: Union[str, Path],
        batch_size: int = 8,
        prefetch_size: int = 8,
        num_workers: int = 8,
        drop_last: bool = True,
    ) -> "MLXDataStream":
        """Create MLXDataStream from saved checkpoint.

        Args:
            dataset: DynamicDataset instance
            checkpoint_path: Path to checkpoint JSON file
            batch_size: Number of samples per batch
            prefetch_size: Number of batches to prefetch
            num_workers: Number of parallel workers
            drop_last: Whether to drop last incomplete batch

        Returns:
            MLXDataStream configured to resume from checkpoint
        """
        checkpoint = CheckpointState.load(checkpoint_path)
        return cls(
            dataset=dataset,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            checkpoint=checkpoint,
        )

    def _sample_transform(self, sample_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Transform sample metadata to actual audio data.

        This function is called by mlx-data's prefetch mechanism in parallel
        worker threads, enabling concurrent sample loading.

        Args:
            sample_dict: Dictionary with 'idx' key for sample index

        Returns:
            Dictionary with processed audio features as numpy arrays
        """
        # Extract index from numpy array (mlx-data stores as array)
        idx_val = sample_dict["idx"]
        if isinstance(idx_val, np.ndarray):
            idx = int(idx_val.item())
        else:
            idx = int(idx_val)

        # Use retry logic for failed samples
        max_retries = 3
        for attempt in range(max_retries):
            sample = self.dataset.get_sample(idx)
            if sample is not None:
                # mlx-data requires contiguous arrays with consistent dtypes
                # numpy rfft returns complex128 -> real/imag are float64, must cast to float32
                return {
                    "noisy_real": np.ascontiguousarray(sample.noisy_spec.real, dtype=np.float32),
                    "noisy_imag": np.ascontiguousarray(sample.noisy_spec.imag, dtype=np.float32),
                    "clean_real": np.ascontiguousarray(sample.clean_spec.real, dtype=np.float32),
                    "clean_imag": np.ascontiguousarray(sample.clean_spec.imag, dtype=np.float32),
                    "feat_erb": np.ascontiguousarray(sample.feat_erb, dtype=np.float32),
                    "feat_spec": np.ascontiguousarray(sample.feat_spec, dtype=np.float32),
                    "snr": np.array([sample.snr], dtype=np.float32),
                    "gain": np.array([sample.gain], dtype=np.float32),
                    "valid": np.array([1], dtype=np.int32),
                }
            # Try a different random sample on failure
            idx = (idx + 1) % len(self.dataset)

        # All retries failed - return invalid marker
        # Create dummy data with correct shapes
        n_frames = int(self.dataset.segment_samples / self.dataset.hop_size) + 1
        n_freqs = self.dataset.fft_size // 2 + 1
        nb_erb = self.dataset.config.nb_erb
        nb_df = self.dataset.config.nb_df

        return {
            "noisy_real": np.zeros((n_frames, n_freqs), dtype=np.float32),
            "noisy_imag": np.zeros((n_frames, n_freqs), dtype=np.float32),
            "clean_real": np.zeros((n_frames, n_freqs), dtype=np.float32),
            "clean_imag": np.zeros((n_frames, n_freqs), dtype=np.float32),
            "feat_erb": np.zeros((n_frames, nb_erb), dtype=np.float32),
            "feat_spec": np.zeros((n_frames, nb_df, 2), dtype=np.float32),
            "snr": np.array([0.0], dtype=np.float32),
            "gain": np.array([0.0], dtype=np.float32),
            "valid": np.array([0], dtype=np.int32),
        }

    def _create_stream(self, skip_batches: int = 0) -> Any:
        """Create mlx-data stream for current epoch.

        Args:
            skip_batches: Number of batches to skip for resume

        Returns:
            Configured mlx-data Stream ready for iteration
        """
        # Get shuffled indices for current epoch
        n_samples = len(self.dataset)
        indices = list(range(n_samples))

        # Use deterministic shuffling based on epoch
        epoch_rng = random.Random(self._checkpoint.seed + self._checkpoint.epoch)
        epoch_rng.shuffle(indices)

        # Skip samples for resume
        skip_samples = skip_batches * self.batch_size
        if skip_samples > 0 and skip_samples < len(indices):
            indices = indices[skip_samples:]

        # Create sample metadata buffer
        samples = [{"idx": np.array([i], dtype=np.int32)} for i in indices]

        # Build mlx-data pipeline
        stream = dx.buffer_from_vector(samples)
        stream = stream.to_stream()

        # Apply our processing function (parallelized by prefetch!)
        stream = stream.sample_transform(self._sample_transform)

        # Batch samples together
        stream = stream.batch(self.batch_size)

        # Background prefetching with multiple workers
        stream = stream.prefetch(self.prefetch_size, self.num_workers)

        return stream

    def _convert_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, mx.array]:
        """Convert numpy batch to MLX arrays.

        Args:
            batch: Dictionary of numpy arrays from mlx-data

        Returns:
            Dictionary of MLX arrays ready for model
        """
        return {
            "noisy_real": mx.array(batch["noisy_real"]),
            "noisy_imag": mx.array(batch["noisy_imag"]),
            "clean_real": mx.array(batch["clean_real"]),
            "clean_imag": mx.array(batch["clean_imag"]),
            "feat_erb": mx.array(batch["feat_erb"]),
            "feat_spec": mx.array(batch["feat_spec"]),
            "snr": mx.array(batch["snr"]).squeeze(-1),
            "gain": mx.array(batch["gain"]).squeeze(-1),
        }

    def set_epoch(self, epoch: int) -> None:
        """Set epoch and reset iteration state.

        Args:
            epoch: New epoch number
        """
        self._checkpoint.epoch = epoch
        self._checkpoint.batch_idx = 0
        self._checkpoint.samples_processed = 0
        self._batch_count = 0

        # Sync dataset epoch for deterministic sample processing
        self.dataset.set_epoch(epoch)

        # Reset stream
        self._stream = None

    def set_split(self, split: str) -> None:
        """Set data split (train/valid/test).

        Args:
            split: Split name
        """
        self._checkpoint.split = split
        self.dataset.set_split(split)
        self._stream = None

    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate over batches with prefetching.

        Yields:
            Dictionary of MLX arrays for each batch
        """
        # Create stream, skipping already-processed batches on resume
        self._stream = self._create_stream(self._checkpoint.batch_idx)
        self._batch_count = self._checkpoint.batch_idx

        for batch in self._stream:
            # Check for invalid samples in batch
            valid_mask = batch.get("valid", np.ones(self.batch_size, dtype=np.int32))
            if isinstance(valid_mask, np.ndarray) and valid_mask.min() == 0:
                # Some samples invalid - skip this batch
                continue

            # Update checkpoint state
            self._batch_count += 1
            self._checkpoint.batch_idx = self._batch_count
            self._checkpoint.samples_processed += self.batch_size

            yield self._convert_batch(batch)

    def __len__(self) -> int:
        """Return approximate number of batches in current split."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    @property
    def checkpoint(self) -> CheckpointState:
        """Get current checkpoint state."""
        self._checkpoint.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return self._checkpoint

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save current checkpoint to file.

        Args:
            path: Path for checkpoint JSON file
        """
        self.checkpoint.save(path)

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information.

        Returns:
            Dictionary with progress metrics
        """
        total_batches = len(self)
        return {
            "epoch": self._checkpoint.epoch,
            "batch": self._batch_count,
            "total_batches": total_batches,
            "samples_processed": self._checkpoint.samples_processed,
            "progress_pct": 100.0 * self._batch_count / max(total_batches, 1),
        }
