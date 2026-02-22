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
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Full, Queue
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import mlx.core as mx
import numpy as np
from scipy import signal as scipy_signal

from .augment_ext import biquad_filter as _ext_biquad_filter
from .augment_ext import combine_noises as _ext_combine_noises
from .augment_ext import mix_audio as _ext_mix_audio
from .feature_ops import compute_df_features, compute_erb_features, compute_stft, create_erb_filterbank
from .file_lists import read_file_list as _read_file_list

# Optional mlx-data import (for MLXDataStream)
try:
    import mlx.data as dx

    HAS_MLX_DATA = True
    # mlx-data does not ship complete type information; cast to Any for attribute access
    dx = cast(Any, dx)
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
            audio = np.asarray(scipy_signal.resample(audio, num_samples), dtype=np.float32)
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
            audio = np.asarray(scipy_signal.resample(audio, num_samples), dtype=np.float32)
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
    snr_range_extreme: Tuple[float, float] = (-20.0, -5.0)  # dB, near-obscured speech
    snr_range_very_low: Tuple[float, float] = (-30.0, -20.0)  # dB, severely obscured speech
    p_extreme_snr: float = 0.1  # Probability of sampling from snr_range_extreme
    p_very_low_snr: float = 0.0  # Probability of sampling from snr_range_very_low
    gain_range: Tuple[float, float] = (-6.0, 6.0)  # dB (legacy; retained for compatibility)
    speech_gain_range: Tuple[float, float] = (-12.0, 12.0)  # dB, varies absolute speech loudness
    noise_gain_range: Tuple[float, float] = (-12.0, 12.0)  # dB, varies relative noise contributions

    # Augmentation probabilities
    p_reverb: float = 0.5  # Probability of applying RIR
    p_clipping: float = 0.0  # Probability of clipping distortion
    p_bandwidth_ext: float = 0.0  # Probability of bandwidth extension
    p_interfer_speech: float = 0.0  # Probability of interfering speaker
    interfer_speech_snr_range: Tuple[float, float] = (-10.0, 10.0)  # dB, SNR of interferer vs target

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
        self._shard_access: OrderedDict[str, bool] = OrderedDict()
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
                self._shard_access.move_to_end(shard_rel_path)
                return self._shard_cache[shard_rel_path]

        # Open NpzFile lazily (arrays loaded on access, not upfront)
        shard_path = self.cache_dir / shard_rel_path
        npz_file = np.load(shard_path, mmap_mode="r")

        with self._lock:
            # Evict oldest if at capacity
            while len(self._shard_cache) >= self._max_shards:
                oldest = next(iter(self._shard_access))
                del self._shard_access[oldest]
                old_npz = self._shard_cache.pop(oldest)
                old_npz.close()

            self._shard_cache[shard_rel_path] = npz_file
            self._shard_access[shard_rel_path] = True

        return npz_file

    def load(self, path: str) -> np.ndarray:
        """Load audio array by original file path."""
        if path not in self.index:
            raise KeyError(f"File not in cache: {path}")

        shard_name, key = self.index[path]
        npz_file = self._get_shard(shard_name)
        # Access the specific array - this triggers lazy load of just that array
        return np.asarray(npz_file[key], dtype=np.float32)

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
        self._access_order: OrderedDict[str, bool] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: str) -> Optional[np.ndarray]:
        """Get audio from cache if available."""
        with self._lock:
            if path in self._cache:
                # Move to end (most recently used)
                self._access_order.move_to_end(path)
                return self._cache[path]
        return None

    def put(self, path: str, audio: np.ndarray) -> None:
        """Add audio to cache, evicting oldest if necessary."""
        with self._lock:
            if path in self._cache:
                return

            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest = next(iter(self._access_order))
                del self._access_order[oldest]
                del self._cache[oldest]

            self._cache[path] = audio
            self._access_order[path] = True

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
        return _ext_biquad_filter(audio, b, a)

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

    Routes through Rust extension when available for performance.

    Args:
        clean: Clean speech signal
        noise: Noise signal
        snr_db: Target SNR in dB
        gain_db: Gain to apply to speech in dB

    Returns:
        Tuple of (clean_out, noise_out, noisy_mixture)
    """
    return _ext_mix_audio(clean, noise, snr_db, gain_db)


def combine_noises(
    noises: List[np.ndarray],
    target_len: int,
    gains_db: Optional[List[float]] = None,
) -> np.ndarray:
    """Combine multiple noise signals into one.

    Routes through Rust extension when available for performance.

    Args:
        noises: List of noise signals
        target_len: Target output length
        gains_db: Optional gains for each noise

    Returns:
        Combined noise signal
    """
    return _ext_combine_noises(noises, target_len, gains_db)


@dataclass
class Sample:
    """A single training sample."""

    noisy_spec: np.ndarray  # Complex STFT of noisy mixture
    clean_spec: np.ndarray  # Complex STFT of clean speech
    feat_erb: np.ndarray  # ERB features
    feat_spec: np.ndarray  # DF-band features
    snr: float
    gain: float


def _assemble_batch(samples: List[Sample]) -> Dict[str, mx.array]:
    """Assemble a list of Samples into a batched dict of mx.arrays.

    Pre-allocates numpy buffers based on the first sample's shape, then fills
    them via indexed assignment — avoids 7 growing Python lists + np.stack copies
    per batch.
    """
    n = len(samples)
    if n == 0:
        raise ValueError("Cannot assemble empty batch")

    s0 = samples[0]
    spec_shape = s0.noisy_spec.real.shape
    erb_shape = s0.feat_erb.shape
    spec_feat_shape = s0.feat_spec.shape

    noisy_real = np.empty((n, *spec_shape), dtype=np.float32)
    noisy_imag = np.empty((n, *spec_shape), dtype=np.float32)
    clean_real = np.empty((n, *spec_shape), dtype=np.float32)
    clean_imag = np.empty((n, *spec_shape), dtype=np.float32)
    feat_erb = np.empty((n, *erb_shape), dtype=np.float32)
    feat_spec = np.empty((n, *spec_feat_shape), dtype=np.float32)
    snr_arr = np.empty(n, dtype=np.float32)

    for i, s in enumerate(samples):
        noisy_real[i] = s.noisy_spec.real
        noisy_imag[i] = s.noisy_spec.imag
        clean_real[i] = s.clean_spec.real
        clean_imag[i] = s.clean_spec.imag
        feat_erb[i] = s.feat_erb
        feat_spec[i] = s.feat_spec
        snr_arr[i] = s.snr

    return {
        "noisy_real": mx.array(noisy_real),
        "noisy_imag": mx.array(noisy_imag),
        "clean_real": mx.array(clean_real),
        "clean_imag": mx.array(clean_imag),
        "feat_erb": mx.array(feat_erb),
        "feat_spec": mx.array(feat_spec),
        "snr": mx.array(snr_arr),
    }


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

        # Current split
        self._current_split = "train"
        self._indices: List[int] = []
        self._regenerate_indices()

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

    def _load_speech(self, idx: int, rng: random.Random) -> Optional[np.ndarray]:
        """Load and prepare a speech sample.

        Returns None if the audio is shorter than segment_samples.
        NOTE: When using a properly built cache (with --min-duration),
        all speech files should be long enough.
        """
        files = self.splits[self._current_split]
        path = files[idx]

        try:
            audio = self._load_audio(path, "speech")

            # Skip audio that's too short (shouldn't happen with properly built cache)
            if len(audio) < self.segment_samples:
                return None

            # Extract random segment
            if len(audio) > self.segment_samples:
                start = rng.randint(0, len(audio) - self.segment_samples)
                audio = audio[start : start + self.segment_samples]

            return audio
        except Exception:
            return None

    def _load_noise(self, rng: random.Random) -> Tuple[np.ndarray, float]:
        """Load a random noise sample or generate synthetic noise."""
        # Occasionally generate synthetic noise
        if rng.random() < self.config.p_random_noise:
            noise = self.noise_generator.generate_random(self.segment_samples)
            gain = rng.choice([-24.0, -12.0, -6.0, 0.0])
            return noise, gain

        # Load from file
        noise_files = self.config.noise_files
        if not noise_files:
            # Fallback to white noise
            return self.noise_generator.generate(0.0, self.segment_samples), 0.0

        path = rng.choice(noise_files)
        try:
            noise = self._load_audio(path, "noise")
            gain = rng.uniform(*self.config.noise_gain_range)
            return noise, gain
        except Exception:
            # Fallback
            return self.noise_generator.generate(0.0, self.segment_samples), 0.0

    def _load_rir(self, rng: random.Random) -> Optional[np.ndarray]:
        """Load a random RIR if available."""
        rir_files = self.config.rir_files
        if not rir_files:
            return None

        path = rng.choice(rir_files)
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
        # Use per-sample RNG so get_sample() remains thread-safe under prefetch workers.
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(
                f"Sample index {idx} out of range for split '{self._current_split}' " f"(size={len(self._indices)})."
            )

        sample_seed = self.config.seed + self._epoch * 1000000 + idx
        rng = random.Random(sample_seed)

        # Load speech
        speech = self._load_speech(self._indices[idx], rng)
        if speech is None:
            return None

        # Sample SNR and gain with 3-tier SNR distribution
        r = rng.random()
        if self.config.p_very_low_snr > 0 and r < self.config.p_very_low_snr:
            # Very low SNR: severely obscured speech (for whisper/distant mic training)
            snr = rng.uniform(*self.config.snr_range_very_low)
        elif self.config.p_extreme_snr > 0 and r < (self.config.p_very_low_snr + self.config.p_extreme_snr):
            # Extreme SNR: near-obscured speech
            snr = rng.uniform(*self.config.snr_range_extreme)
        else:
            # Base SNR: normal range
            snr = rng.uniform(*self.config.snr_range)
        gain = rng.uniform(*self.config.speech_gain_range)

        # Load and combine multiple noises (2-5 like Rust)
        n_noises = rng.randint(self.config.n_noise_min, self.config.n_noise_max)
        noises = []
        noise_gains = []
        for _ in range(n_noises):
            noise, ng = self._load_noise(rng)
            noises.append(noise)
            noise_gains.append(ng)

        combined_noise = combine_noises(noises, self.segment_samples, noise_gains)

        # Optionally apply RIR
        speech_for_mix = speech.copy()
        if self.config.rir_files and rng.random() < self.config.p_reverb:
            rir = self._load_rir(rng)
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

            # Add interfering speaker (vocal music / competing talker simulation)
            if self.config.p_interfer_speech > 0 and rng.random() < self.config.p_interfer_speech:
                # Load a different speech file as interferer
                interfer_idx = rng.randint(0, len(self._indices) - 1)
                if interfer_idx != self._indices[idx]:
                    interfer_speech = self._load_speech(interfer_idx, rng)
                    if interfer_speech is not None:
                        # Mix interferer into noise at a random SNR relative to target
                        interfer_snr = rng.uniform(*self.config.interfer_speech_snr_range)
                        # Scale interferer relative to target speech
                        target_rms = np.sqrt(np.mean(speech_for_mix**2) + 1e-8)
                        interfer_rms = np.sqrt(np.mean(interfer_speech**2) + 1e-8)
                        scale = target_rms / (interfer_rms + 1e-8) * (10 ** (-interfer_snr / 20))
                        interfer_scaled = interfer_speech * scale
                        # Add to combined noise (interferer is treated as noise, not target)
                        combined_noise = combined_noise + interfer_scaled

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
        batch_samples: List[Sample] = []

        for sample in self.iter_samples():
            batch_samples.append(sample)

            if len(batch_samples) >= batch_size:
                yield _assemble_batch(batch_samples)
                batch_samples = []

        if batch_samples and not drop_last:
            yield _assemble_batch(batch_samples)


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
        strict_failures: bool = True,
        shuffle_buffer_size: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.strict_failures = strict_failures
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate with background prefetching."""
        # Queue to hold prefetched batches
        prefetch_queue: Queue = Queue(maxsize=self.prefetch_factor)
        stop_event = threading.Event()
        worker_errors: List[BaseException] = []
        stats: Dict[str, int] = {"samples_succeeded": 0, "samples_failed": 0}

        def _to_batch(samples: List[Sample]) -> Dict[str, mx.array]:
            return _assemble_batch(samples)

        def _queue_put(item: Optional[Dict[str, mx.array]]) -> bool:
            while not stop_event.is_set():
                try:
                    prefetch_queue.put(item, timeout=0.1)
                    return True
                except Full:
                    continue
            return False

        def worker():
            """Background worker that fills the prefetch queue."""
            n_samples = len(self.dataset)
            max_workers = max(1, self.num_workers)
            max_pending = max(max_workers, max_workers * self.prefetch_factor)
            pending: Dict[int, Future[Optional[Sample]]] = {}
            next_submit = 0
            next_consume = 0
            batch_samples: List[Sample] = []

            def submit_pending(executor: ThreadPoolExecutor) -> None:
                nonlocal next_submit
                while next_submit < n_samples and len(pending) < max_pending and not stop_event.is_set():
                    pending[next_submit] = executor.submit(self.dataset.get_sample, next_submit)
                    next_submit += 1

            try:
                with ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="df-mlx-prefetch",
                ) as executor:
                    submit_pending(executor)
                    while next_consume < n_samples and not stop_event.is_set():
                        future = pending.pop(next_consume)
                        next_consume += 1
                        submit_pending(executor)

                        sample_idx = next_consume - 1
                        failed_on_exception = False
                        try:
                            sample = future.result()
                        except Exception as exc:
                            failed_on_exception = True
                            stats["samples_failed"] += 1
                            if self.strict_failures:
                                worker_errors.append(
                                    RuntimeError(
                                        f"PrefetchDataLoader failed while loading sample index " f"{sample_idx}: {exc}"
                                    )
                                )
                                return
                            sample = None
                        if sample is None:
                            if not failed_on_exception:
                                stats["samples_failed"] += 1
                            continue

                        stats["samples_succeeded"] += 1
                        batch_samples.append(sample)
                        if len(batch_samples) == self.batch_size:
                            if not _queue_put(_to_batch(batch_samples)):
                                return
                            batch_samples = []

                    if batch_samples and not self.drop_last and not stop_event.is_set():
                        _queue_put(_to_batch(batch_samples))
            finally:
                if not stop_event.is_set():
                    _queue_put(None)  # Signal completion

        # Start worker thread
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        try:
            if self.shuffle_buffer_size > 0:
                buf_rng = random.Random(getattr(self.dataset.config, "seed", 0) + getattr(self.dataset, "_epoch", 0))
                buffer: List[Dict[str, mx.array]] = []
                while True:
                    batch = prefetch_queue.get()
                    if batch is None:
                        break
                    buffer.append(batch)
                    if len(buffer) >= self.shuffle_buffer_size:
                        idx = buf_rng.randrange(len(buffer))
                        yield buffer.pop(idx)
                buf_rng.shuffle(buffer)
                yield from buffer
            else:
                while True:
                    batch = prefetch_queue.get()
                    if batch is None:
                        break
                    yield batch

            if worker_errors:
                raise worker_errors[0]

            if self.strict_failures and len(self.dataset) > 0 and stats["samples_succeeded"] == 0:
                raise RuntimeError(
                    "PrefetchDataLoader failed to load any samples from non-empty dataset. "
                    "This matches MLXDataStream failure semantics; verify input files/cache."
                )
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
    return _read_file_list(path, split_tab=True)


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

        # Track iteration state – initialise from checkpoint so that
        # get_progress() returns the correct batch count before __iter__
        # is called (e.g. during resume-position validation).
        self._stream: Optional[Any] = None
        self._batch_count = self._checkpoint.batch_idx

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
            sample_dict: Dictionary with 'idx' key for sample index and
                        'fallbacks' key for backup indices to try

        Returns:
            Dictionary with processed audio features as numpy arrays.

        Raises:
            RuntimeError: If all sample indices fail to load (no dummy data used).
        """
        # Extract primary index
        idx_val = sample_dict["idx"]
        if isinstance(idx_val, np.ndarray):
            primary_idx = int(idx_val.item())
        else:
            primary_idx = int(idx_val)

        # Extract fallback indices
        fallbacks = sample_dict.get("fallbacks", np.array([], dtype=np.int32))
        if isinstance(fallbacks, np.ndarray):
            fallback_list = fallbacks.tolist()
        else:
            fallback_list = list(fallbacks) if fallbacks is not None else []

        # Build full list of indices to try: primary first, then fallbacks
        indices_to_try = [primary_idx] + fallback_list

        # Try each index until one succeeds
        for try_idx in indices_to_try:
            sample = self.dataset.get_sample(try_idx)
            if sample is not None:
                # compute_stft guarantees complex64, so .real/.imag are float32 views.
                # np.require ensures C-contiguity without copying when already satisfied.
                noisy_r = sample.noisy_spec.real
                noisy_i = sample.noisy_spec.imag
                clean_r = sample.clean_spec.real
                clean_i = sample.clean_spec.imag
                return {
                    "noisy_real": np.require(noisy_r, dtype=np.float32, requirements="C"),
                    "noisy_imag": np.require(noisy_i, dtype=np.float32, requirements="C"),
                    "clean_real": np.require(clean_r, dtype=np.float32, requirements="C"),
                    "clean_imag": np.require(clean_i, dtype=np.float32, requirements="C"),
                    "feat_erb": np.require(sample.feat_erb, dtype=np.float32, requirements="C"),
                    "feat_spec": np.require(sample.feat_spec, dtype=np.float32, requirements="C"),
                    "snr": np.array([sample.snr], dtype=np.float32),
                    "gain": np.array([sample.gain], dtype=np.float32),
                }

        # All indices failed - raise error (no dummy data!)
        raise RuntimeError(
            f"Failed to load sample after trying {len(indices_to_try)} indices. "
            f"Primary index: {primary_idx}. This indicates a data integrity issue. "
            "Please verify your audio cache with validate_audio_cache.py."
        )

    def _create_stream(self, skip_batches: int = 0) -> Any:
        """Create mlx-data stream for current epoch.

        Args:
            skip_batches: Number of batches to skip for resume

        Returns:
            Configured mlx-data Stream ready for iteration
        """
        if not HAS_MLX_DATA or dx is None:
            raise ImportError(
                "mlx-data is required for streaming dataset iteration. Install 'mlx-data' or set DynamicDataset(..., use_mlx_data=False)."
            )
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

        # Create sample metadata generator with fallback indices for retry
        # Each sample carries a list of indices to try if primary fails
        def _sample_metadata_iter() -> Iterator[Dict[str, np.ndarray]]:
            n = len(indices)
            for i, primary_idx in enumerate(indices):
                # Create fallback indices (next 10 samples in shuffled order)
                fallbacks = [indices[(i + j) % n] for j in range(1, 11)]
                yield {
                    "idx": np.array([primary_idx], dtype=np.int32),
                    "fallbacks": np.array(fallbacks, dtype=np.int32),
                }

        # Build mlx-data pipeline lazily from a Python iterable
        stream = dx.stream_python_iterable(_sample_metadata_iter)  # type: ignore[attr-defined]

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

    def set_resume_position(self, epoch: int, batch_idx: int, *, split: str | None = None) -> None:
        """Set an explicit resume position for deterministic mid-epoch recovery.

        Args:
            epoch: Epoch index to resume within.
            batch_idx: Number of micro-batches already consumed in that epoch.
            split: Optional split override (defaults to current split).
        """
        if batch_idx < 0:
            raise ValueError(f"batch_idx must be >= 0, got {batch_idx}")

        if split is not None and split != self._checkpoint.split:
            self.set_split(split)

        self._checkpoint.epoch = epoch
        self._checkpoint.batch_idx = batch_idx
        self._checkpoint.samples_processed = batch_idx * self.batch_size
        self.dataset.set_epoch(epoch)
        self._batch_count = batch_idx
        self._stream = None

    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate over batches with prefetching.

        Yields:
            Dictionary of MLX arrays for each batch.
            All samples are real data - no dummy/fake data is ever used.
        """
        # Create stream, skipping already-processed batches on resume
        self._stream = self._create_stream(self._checkpoint.batch_idx)
        self._batch_count = self._checkpoint.batch_idx

        assert self._stream is not None

        for batch in self._stream:
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
        batch = self._checkpoint.batch_idx
        return {
            "epoch": self._checkpoint.epoch,
            "batch": batch,
            "total_batches": total_batches,
            "samples_processed": self._checkpoint.samples_processed,
            "progress_pct": 100.0 * batch / max(total_batches, 1),
        }
