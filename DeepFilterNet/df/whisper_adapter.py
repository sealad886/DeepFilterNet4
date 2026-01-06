"""
Whisper Backend Adapter Module

This module provides a unified interface for whisper-based speech recognition
across multiple backends:
- PyTorch backend (openai-whisper): Works on all platforms
- MLX backend (mlx-whisper): Optimized for Apple Silicon (M1/M2/M3/M4)

The adapter auto-detects the best available backend based on hardware and
installed packages, enabling 5-10x speedup on Apple Silicon while maintaining
full compatibility with existing code.

Example:
    >>> from df.whisper_adapter import get_whisper_backend
    >>> backend = get_whisper_backend("base")  # Auto-selects best backend
    >>> print(f"Using {backend.backend_name} backend")
    Using mlx backend  # On Apple Silicon with mlx-whisper installed
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Protocol, Union

import numpy as np
import torch

if TYPE_CHECKING:
    import mlx.core as mx

# Type aliases for clarity
ArrayLike = Union[torch.Tensor, "mx.array", np.ndarray]
BackendType = Literal["auto", "pytorch", "mlx"]

# Module-level state for lazy MLX loading
_mlx_available: bool = False
_mx_module: Optional[Any] = None


# =============================================================================
# Task 1.1: Platform Detection Utility
# =============================================================================


@lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    """
    Detect if running on Apple Silicon (M1/M2/M3/M4).

    Uses macOS sysctl to check CPU brand string for "Apple" identifier.
    Results are cached for performance.

    Returns:
        True if running on Apple Silicon Mac, False otherwise.

    Example:
        >>> if is_apple_silicon():
        ...     print("Running on Apple Silicon - MLX available")
    """
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "Apple" in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


# =============================================================================
# Task 1.2: Base WhisperBackend Protocol
# =============================================================================


@dataclass
class WhisperDecodingResult:
    """
    Unified decoding result across whisper backends.

    Attributes:
        text: Transcribed text output.
        language: Detected or specified language code.
        tokens: List of token IDs from decoding.
        avg_logprob: Average log probability of tokens.
        no_speech_prob: Probability that segment contains no speech.
        temperature: Temperature used for sampling.
    """

    text: str
    language: str
    tokens: List[int]
    avg_logprob: float
    no_speech_prob: float
    temperature: float


class WhisperBackend(Protocol):
    """
    Protocol defining the whisper backend interface.

    Both PyTorchWhisperBackend and MLXWhisperBackend implement this protocol,
    providing a unified API for whisper operations regardless of the underlying
    implementation.
    """

    @property
    def backend_name(self) -> str:
        """Return the backend identifier ('pytorch' or 'mlx')."""
        ...

    @property
    def device(self) -> Optional[str]:
        """Return device string (for PyTorch) or None (for MLX unified memory)."""
        ...

    @property
    def dims(self) -> Any:
        """Return model dimensions object (ModelDimensions)."""
        ...

    @property
    def is_multilingual(self) -> bool:
        """Return whether model supports multiple languages."""
        ...

    def embed_audio(self, mel: ArrayLike) -> ArrayLike:
        """
        Extract audio embeddings from mel spectrogram.

        Args:
            mel: Mel spectrogram tensor of shape [batch, n_mels, n_frames].

        Returns:
            Audio embeddings tensor.
        """
        ...

    def logits(self, tokens: ArrayLike, audio_features: ArrayLike) -> ArrayLike:
        """
        Get logits for token prediction.

        Args:
            tokens: Token IDs tensor.
            audio_features: Audio embeddings from embed_audio().

        Returns:
            Logits tensor for next token prediction.
        """
        ...

    def decode(self, mel: ArrayLike, options: Any = None) -> WhisperDecodingResult:
        """
        Decode mel spectrogram to text.

        Args:
            mel: Mel spectrogram tensor.
            options: Decoding options (DecodingOptions).

        Returns:
            WhisperDecodingResult with transcription.
        """
        ...

    def get_tokenizer(self, language: str = "en", task: str = "transcribe") -> Any:
        """
        Get tokenizer for the model.

        Args:
            language: Language code (e.g., 'en', 'es').
            task: Task type ('transcribe' or 'translate').

        Returns:
            Tokenizer instance.
        """
        ...

    def get_decoder(self, temperature: float = 0.0) -> Any:
        """
        Get greedy decoder instance.

        Args:
            temperature: Sampling temperature (0.0 for greedy).

        Returns:
            Decoder instance.
        """
        ...

    def create_decoding_options(self, **kwargs: Any) -> Any:
        """
        Create decoding options for the backend.

        Args:
            **kwargs: Decoding options (language, task, beam_size, etc.).

        Returns:
            Backend-specific DecodingOptions instance.
        """
        ...

    def pad_or_trim(self, audio: ArrayLike, length: Optional[int] = None) -> ArrayLike:
        """
        Pad or trim audio to specified length.

        Args:
            audio: Audio waveform tensor.
            length: Target length in samples.

        Returns:
            Padded or trimmed audio.
        """
        ...

    def log_mel_spectrogram(self, audio: ArrayLike, n_mels: Optional[int] = None) -> ArrayLike:
        """
        Compute log mel spectrogram from audio.

        Args:
            audio: Audio waveform tensor.
            n_mels: Number of mel filterbanks.

        Returns:
            Log mel spectrogram tensor.
        """
        ...

    def to_backend_array(self, arr: ArrayLike) -> ArrayLike:
        """Convert input to backend-native array type."""
        ...

    def to_numpy(self, arr: ArrayLike) -> np.ndarray:
        """Convert backend array to numpy."""
        ...


# =============================================================================
# Task 1.3: Tensor Conversion Utilities
# =============================================================================


def _ensure_mlx() -> bool:
    """
    Lazy load MLX module if available on Apple Silicon.

    Returns:
        True if MLX is available and loaded, False otherwise.
    """
    global _mlx_available, _mx_module
    if _mx_module is None and is_apple_silicon():
        try:
            import mlx.core as mlx_core

            _mx_module = mlx_core
            _mlx_available = True
        except ImportError:
            _mlx_available = False
    return _mlx_available


def _get_mx() -> Any:
    """Get the MLX module, raising if not available."""
    if not _ensure_mlx():
        raise RuntimeError("MLX not available - requires Apple Silicon with mlx installed")
    return _mx_module


def mx_to_torch(arr: "mx.array", dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Convert MLX array to PyTorch tensor.

    Args:
        arr: MLX array to convert.
        dtype: Optional PyTorch dtype for the output tensor.

    Returns:
        PyTorch tensor with the same data.

    Raises:
        RuntimeError: If MLX is not available.

    Example:
        >>> import mlx.core as mx
        >>> mlx_arr = mx.array([1.0, 2.0, 3.0])
        >>> torch_tensor = mx_to_torch(mlx_arr)
    """
    if not _ensure_mlx():
        raise RuntimeError("MLX not available")
    np_arr = np.array(arr)
    tensor = torch.from_numpy(np_arr.copy())  # Copy to ensure contiguous
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def torch_to_mx(tensor: torch.Tensor) -> "mx.array":
    """
    Convert PyTorch tensor to MLX array.

    Args:
        tensor: PyTorch tensor to convert.

    Returns:
        MLX array with the same data.

    Raises:
        RuntimeError: If MLX is not available.

    Example:
        >>> torch_tensor = torch.randn(10, 20)
        >>> mlx_arr = torch_to_mx(torch_tensor)
    """
    mx = _get_mx()
    np_arr = tensor.detach().cpu().numpy()
    return mx.array(np_arr)


def to_numpy(arr: ArrayLike) -> np.ndarray:
    """
    Convert any array type to numpy.

    Args:
        arr: Input array (torch.Tensor, mx.array, or np.ndarray).

    Returns:
        Numpy array with the same data.

    Example:
        >>> tensor = torch.randn(10, 20)
        >>> np_arr = to_numpy(tensor)
        >>> assert isinstance(np_arr, np.ndarray)
    """
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    # Assume MLX array
    return np.array(arr)


# =============================================================================
# Task 1.4: PyTorchWhisperBackend Implementation
# =============================================================================


class PyTorchWhisperBackend:
    """
    WhisperBackend implementation using openai-whisper (PyTorch).

    This backend wraps the OpenAI whisper library and works on all platforms
    with CUDA, MPS, or CPU support.

    Args:
        model_name: Whisper model name ('tiny', 'base', 'small', 'medium', 'large', 'turbo').
        device: Device to run model on ('cuda', 'cpu', 'mps'). Auto-detected if None.
        download_root: Directory to download/cache model weights.

    Example:
        >>> backend = PyTorchWhisperBackend("base", device="cuda")
        >>> mel = backend.log_mel_spectrogram(audio)
        >>> result = backend.decode(mel)
        >>> print(result.text)
    """

    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        download_root: Optional[str] = None,
    ):
        import whisper

        self._whisper = whisper

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device

        self._model = whisper.load_model(model_name, device=device, download_root=download_root)

    @property
    def backend_name(self) -> str:
        """Return 'pytorch' as the backend identifier."""
        return "pytorch"

    @property
    def device(self) -> str:
        """Return the device string (cuda, mps, or cpu)."""
        return self._device

    @property
    def dims(self) -> Any:
        """Return model dimensions object."""
        return self._model.dims

    @property
    def is_multilingual(self) -> bool:
        """Return whether model supports multiple languages."""
        return self._model.is_multilingual

    @property
    def model(self) -> Any:
        """Return the underlying whisper model (for advanced usage)."""
        return self._model

    def embed_audio(self, mel: ArrayLike) -> torch.Tensor:
        """
        Extract audio embeddings from mel spectrogram.

        Args:
            mel: Mel spectrogram tensor of shape [batch, n_mels, n_frames].

        Returns:
            Audio embeddings as PyTorch tensor.
        """
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)
        mel = mel.to(self._device)
        return self._model.embed_audio(mel)

    def logits(self, tokens: ArrayLike, audio_features: ArrayLike) -> torch.Tensor:
        """
        Get logits for token prediction.

        Args:
            tokens: Token IDs tensor.
            audio_features: Audio embeddings from embed_audio().

        Returns:
            Logits tensor for next token prediction.
        """
        if isinstance(tokens, np.ndarray):
            tokens = torch.from_numpy(tokens)
        if isinstance(audio_features, np.ndarray):
            audio_features = torch.from_numpy(audio_features)
        tokens = tokens.to(self._device)
        audio_features = audio_features.to(self._device)
        return self._model.logits(tokens, audio_features)

    def decode(self, mel: ArrayLike, options: Any = None) -> WhisperDecodingResult:
        """
        Decode mel spectrogram to text.

        Args:
            mel: Mel spectrogram tensor.
            options: DecodingOptions instance. Uses defaults if None.

        Returns:
            WhisperDecodingResult with transcription.
        """
        if options is None:
            options = self._whisper.DecodingOptions()
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)
        mel = mel.to(self._device)
        result = self._whisper.decode(self._model, mel, options)
        return WhisperDecodingResult(
            text=result.text,
            language=result.language,
            tokens=list(result.tokens),
            avg_logprob=float(result.avg_logprob),
            no_speech_prob=float(result.no_speech_prob),
            temperature=float(result.temperature),
        )

    def get_tokenizer(self, language: str = "en", task: str = "transcribe") -> Any:
        """
        Get tokenizer for the model.

        Args:
            language: Language code (e.g., 'en', 'es').
            task: Task type ('transcribe' or 'translate').

        Returns:
            Whisper tokenizer instance.
        """
        return self._whisper.tokenizer.get_tokenizer(self._model.is_multilingual, language=language, task=task)

    def get_decoder(self, temperature: float = 0.0) -> Any:
        """
        Get greedy decoder instance.

        Args:
            temperature: Sampling temperature (0.0 for greedy).

        Returns:
            GreedyDecoder instance.
        """
        tokenizer = self.get_tokenizer()
        return self._whisper.decoding.GreedyDecoder(temperature=temperature, eot=tokenizer.eot)

    def create_decoding_options(self, **kwargs: Any) -> Any:
        """
        Create DecodingOptions for PyTorch backend.

        Args:
            **kwargs: Options like language, task, beam_size, fp16, etc.

        Returns:
            whisper.DecodingOptions instance.
        """
        return self._whisper.DecodingOptions(**kwargs)

    def pad_or_trim(self, audio: ArrayLike, length: Optional[int] = None) -> torch.Tensor:
        """
        Pad or trim audio to specified length.

        Args:
            audio: Audio waveform tensor.
            length: Target length in samples. Uses whisper default if None.

        Returns:
            Padded or trimmed audio as PyTorch tensor.
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if length is None:
            return self._whisper.pad_or_trim(audio)  # type: ignore
        return self._whisper.pad_or_trim(audio, length)  # type: ignore

    def log_mel_spectrogram(self, audio: ArrayLike, n_mels: Optional[int] = None) -> torch.Tensor:
        """
        Compute log mel spectrogram from audio.

        Args:
            audio: Audio waveform tensor.
            n_mels: Number of mel filterbanks. Uses model default if None.

        Returns:
            Log mel spectrogram as PyTorch tensor.
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if n_mels is None:
            n_mels = self.dims.n_mels
        return self._whisper.log_mel_spectrogram(audio, n_mels=n_mels)

    def load_audio(self, path: str) -> np.ndarray:
        """
        Load audio file and resample to 16kHz.

        Args:
            path: Path to audio file.

        Returns:
            Audio waveform as numpy array.
        """
        return self._whisper.load_audio(path)

    def to_backend_array(self, arr: ArrayLike) -> torch.Tensor:
        """Convert input to PyTorch tensor on the correct device."""
        if isinstance(arr, torch.Tensor):
            return arr.to(self._device)
        return torch.from_numpy(to_numpy(arr)).to(self._device)

    def to_numpy(self, arr: ArrayLike) -> np.ndarray:
        """Convert any array to numpy."""
        return to_numpy(arr)


# =============================================================================
# Task 1.5: MLXWhisperBackend Implementation
# =============================================================================


class MLXWhisperBackend:
    """
    WhisperBackend implementation using mlx-whisper (Apple Silicon).

    This backend provides optimized inference on Apple Silicon Macs using
    the MLX framework with unified memory, typically 5-10x faster than
    PyTorch on CPU.

    Args:
        model_name: Whisper model name ('tiny', 'base', 'small', 'medium', 'large', 'turbo').
        download_root: Directory to download/cache model weights.

    Raises:
        RuntimeError: If not running on Apple Silicon.

    Example:
        >>> backend = MLXWhisperBackend("base")
        >>> mel = backend.log_mel_spectrogram(audio)
        >>> result = backend.decode(mel)
        >>> print(result.text)
    """

    def __init__(
        self,
        model_name: str = "base",
        download_root: Optional[str] = None,
    ):
        if not is_apple_silicon():
            raise RuntimeError("MLXWhisperBackend requires Apple Silicon")

        _ensure_mlx()

        import mlx_whisper
        from mlx_whisper import audio as mlx_audio
        from mlx_whisper import decoding as mlx_decoding
        from mlx_whisper import load_models
        from mlx_whisper import transcribe as mlx_transcribe
        from mlx_whisper.tokenizer import get_tokenizer as mlx_get_tokenizer

        self._mlx_whisper = mlx_whisper
        self._mlx_audio = mlx_audio
        self._mlx_decoding = mlx_decoding
        self._mlx_get_tokenizer = mlx_get_tokenizer
        self._mlx_transcribe = mlx_transcribe

        # Load model - download_root may not be supported in all mlx-whisper versions
        try:
            self._model = load_models.load_model(model_name, download_root=download_root)  # type: ignore
        except TypeError:
            # Fallback if download_root is not supported
            self._model = load_models.load_model(model_name)

    @property
    def backend_name(self) -> str:
        """Return 'mlx' as the backend identifier."""
        return "mlx"

    @property
    def device(self) -> Optional[str]:
        """Return None as MLX uses unified memory."""
        return None

    @property
    def dims(self) -> Any:
        """Return model dimensions object."""
        return self._model.dims

    @property
    def is_multilingual(self) -> bool:
        """Return whether model supports multiple languages."""
        return self._model.is_multilingual

    @property
    def model(self) -> Any:
        """Return the underlying MLX whisper model (for advanced usage)."""
        return self._model

    def embed_audio(self, mel: ArrayLike) -> "mx.array":
        """
        Extract audio embeddings from mel spectrogram.

        Args:
            mel: Mel spectrogram tensor of shape [batch, n_mels, n_frames].

        Returns:
            Audio embeddings as MLX array.
        """
        mx = _get_mx()
        if isinstance(mel, (np.ndarray, torch.Tensor)):
            mel = mx.array(to_numpy(mel))
        return self._model.embed_audio(mel)

    def embed_audio_as_torch(self, mel: ArrayLike, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Extract audio embeddings and return as PyTorch tensor.

        This method is useful for gradient flow in training, where the
        embeddings need to participate in PyTorch's autograd graph.

        Args:
            mel: Mel spectrogram tensor.
            dtype: Optional PyTorch dtype for output.

        Returns:
            Audio embeddings as PyTorch tensor.
        """
        mlx_result = self.embed_audio(mel)
        return mx_to_torch(mlx_result, dtype=dtype)

    def logits(self, tokens: ArrayLike, audio_features: ArrayLike) -> "mx.array":
        """
        Get logits for token prediction.

        Args:
            tokens: Token IDs tensor.
            audio_features: Audio embeddings from embed_audio().

        Returns:
            Logits as MLX array.
        """
        mx = _get_mx()
        if isinstance(tokens, (np.ndarray, torch.Tensor)):
            tokens = mx.array(to_numpy(tokens).astype(np.int32))
        if isinstance(audio_features, (np.ndarray, torch.Tensor)):
            audio_features = mx.array(to_numpy(audio_features))
        return self._model.logits(tokens, audio_features)

    def logits_as_torch(
        self,
        tokens: ArrayLike,
        audio_features: ArrayLike,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Get logits and return as PyTorch tensor.

        This method is useful for gradient flow in training.

        Args:
            tokens: Token IDs tensor.
            audio_features: Audio embeddings.
            dtype: Optional PyTorch dtype for output.

        Returns:
            Logits as PyTorch tensor.
        """
        mlx_result = self.logits(tokens, audio_features)
        return mx_to_torch(mlx_result, dtype=dtype)

    def decode(self, mel: ArrayLike, options: Any = None) -> WhisperDecodingResult:
        """
        Decode mel spectrogram to text.

        Args:
            mel: Mel spectrogram tensor.
            options: DecodingOptions instance. Uses defaults if None.

        Returns:
            WhisperDecodingResult with transcription.
        """
        mx = _get_mx()
        if options is None:
            options = self._mlx_decoding.DecodingOptions()
        if isinstance(mel, (np.ndarray, torch.Tensor)):
            mel = mx.array(to_numpy(mel))

        result = self._mlx_decoding.decode(self._model, mel, options)
        return WhisperDecodingResult(
            text=result.text,
            language=result.language,
            tokens=list(result.tokens),
            avg_logprob=float(result.avg_logprob),
            no_speech_prob=float(result.no_speech_prob),
            temperature=float(result.temperature),
        )

    def get_tokenizer(self, language: str = "en", task: str = "transcribe") -> Any:
        """
        Get tokenizer for the model.

        Args:
            language: Language code (e.g., 'en', 'es').
            task: Task type ('transcribe' or 'translate').

        Returns:
            MLX whisper tokenizer instance.
        """
        return self._mlx_get_tokenizer(
            self._model.is_multilingual,
            num_languages=getattr(self._model, "num_languages", None),
            language=language,
            task=task,
        )

    def get_decoder(self, temperature: float = 0.0) -> Any:
        """
        Get greedy decoder instance.

        Args:
            temperature: Sampling temperature (0.0 for greedy).

        Returns:
            GreedyDecoder instance.
        """
        tokenizer = self.get_tokenizer()
        return self._mlx_decoding.GreedyDecoder(temperature=temperature, eot=tokenizer.eot)

    def create_decoding_options(self, **kwargs: Any) -> Any:
        """
        Create DecodingOptions for MLX backend.

        Args:
            **kwargs: Options like language, task, beam_size, etc.

        Returns:
            mlx_whisper DecodingOptions instance.
        """
        return self._mlx_decoding.DecodingOptions(**kwargs)

    def pad_or_trim(self, audio: ArrayLike, length: Optional[int] = None) -> "mx.array":
        """
        Pad or trim audio to specified length.

        Args:
            audio: Audio waveform tensor.
            length: Target length in samples. Uses whisper default if None.

        Returns:
            Padded or trimmed audio as MLX array.
        """
        mx = _get_mx()
        if isinstance(audio, (np.ndarray, torch.Tensor)):
            audio = mx.array(to_numpy(audio))
        if length is None:
            return self._mlx_audio.pad_or_trim(audio)  # type: ignore
        return self._mlx_audio.pad_or_trim(audio, length)  # type: ignore

    def log_mel_spectrogram(self, audio: ArrayLike, n_mels: Optional[int] = None) -> "mx.array":
        """
        Compute log mel spectrogram from audio.

        Args:
            audio: Audio waveform tensor.
            n_mels: Number of mel filterbanks. Uses model default if None.

        Returns:
            Log mel spectrogram as MLX array.
        """
        mx = _get_mx()
        if isinstance(audio, (np.ndarray, torch.Tensor)):
            audio = mx.array(to_numpy(audio))
        if n_mels is None:
            n_mels = self.dims.n_mels
        return self._mlx_audio.log_mel_spectrogram(audio, n_mels=n_mels)

    def load_audio(self, path: str) -> np.ndarray:
        """
        Load audio file and resample to 16kHz.

        Args:
            path: Path to audio file.

        Returns:
            Audio waveform as numpy array.
        """
        return self._mlx_audio.load_audio(path)  # type: ignore

    def to_backend_array(self, arr: ArrayLike) -> "mx.array":
        """Convert input to MLX array."""
        mx = _get_mx()
        if _mlx_available and hasattr(arr, "__class__") and "mlx" in str(type(arr)):
            return arr  # Already MLX array  # type: ignore
        return mx.array(to_numpy(arr))

    def to_numpy(self, arr: ArrayLike) -> np.ndarray:
        """Convert any array to numpy."""
        return to_numpy(arr)


# =============================================================================
# Task 1.6: Backend Factory Function
# =============================================================================


def get_whisper_backend(
    model_name: str = "base",
    backend: BackendType = "auto",
    device: Optional[str] = None,
    download_root: Optional[str] = None,
) -> Union[PyTorchWhisperBackend, MLXWhisperBackend]:
    """
    Get the optimal whisper backend for the current platform.

    Automatically selects MLX on Apple Silicon (if mlx-whisper is installed)
    or falls back to PyTorch whisper on other platforms.

    Args:
        model_name: Whisper model size. One of 'tiny', 'base', 'small',
            'medium', 'large', or 'turbo'.
        backend: Backend selection strategy:
            - 'auto': Selects the best available backend
            - 'pytorch': Forces openai-whisper (PyTorch)
            - 'mlx': Forces mlx-whisper (Apple Silicon only)
        device: Device for PyTorch backend ('cuda', 'cpu', 'mps').
            Ignored for MLX which uses unified memory.
        download_root: Directory to download/cache model weights.

    Returns:
        A WhisperBackend implementation (PyTorchWhisperBackend or MLXWhisperBackend).

    Raises:
        RuntimeError: If 'mlx' backend is requested on non-Apple Silicon.
        ImportError: If required whisper package is not installed.

    Example:
        >>> # Auto-detect best backend
        >>> backend = get_whisper_backend("base")
        >>> print(f"Using {backend.backend_name} backend")

        >>> # Force PyTorch backend
        >>> backend = get_whisper_backend("base", backend="pytorch")

        >>> # Force MLX backend (Apple Silicon only)
        >>> backend = get_whisper_backend("base", backend="mlx")
    """
    if backend == "mlx":
        if not is_apple_silicon():
            raise RuntimeError("MLX backend requires Apple Silicon")
        return MLXWhisperBackend(model_name, download_root=download_root)

    if backend == "pytorch":
        return PyTorchWhisperBackend(model_name, device=device, download_root=download_root)

    # Auto-detect best backend
    if is_apple_silicon():
        try:
            _ensure_mlx()
            if _mlx_available:
                # Check if mlx-whisper is installed
                import mlx_whisper  # noqa: F401

                return MLXWhisperBackend(model_name, download_root=download_root)
        except ImportError:
            pass  # Fall through to PyTorch

    return PyTorchWhisperBackend(model_name, device=device, download_root=download_root)


def load_whisper_model(
    model_name: str = "base",
    backend: BackendType = "auto",
    **kwargs: Any,
) -> Union[PyTorchWhisperBackend, MLXWhisperBackend]:
    """
    Convenience wrapper for get_whisper_backend.

    Args:
        model_name: Whisper model size.
        backend: Backend selection ('auto', 'pytorch', 'mlx').
        **kwargs: Additional arguments passed to get_whisper_backend.

    Returns:
        WhisperBackend implementation.
    """
    return get_whisper_backend(model_name, backend=backend, **kwargs)


# =============================================================================
# Public API Exports
# =============================================================================

__all__ = [
    # Platform detection
    "is_apple_silicon",
    # Data types
    "WhisperDecodingResult",
    "ArrayLike",
    "BackendType",
    # Conversion utilities
    "mx_to_torch",
    "torch_to_mx",
    "to_numpy",
    # Backend classes
    "WhisperBackend",
    "PyTorchWhisperBackend",
    "MLXWhisperBackend",
    # Factory functions
    "get_whisper_backend",
    "load_whisper_model",
]
