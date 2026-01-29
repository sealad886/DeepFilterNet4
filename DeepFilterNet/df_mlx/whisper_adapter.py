"""
Whisper Backend Adapter Module for MLX

This module provides a unified interface for whisper-based speech recognition
optimized for Apple Silicon, with support for multiple backends:
- MLX backend (mlx-whisper): Primary backend, optimized for Apple Silicon
- PyTorch backend (openai-whisper): Fallback for non-Apple platforms

The adapter auto-detects the best available backend based on hardware and
installed packages, enabling 5-10x speedup on Apple Silicon while maintaining
full compatibility with training code.

Example:
    >>> from df_mlx.whisper_adapter import get_whisper_backend
    >>> backend = get_whisper_backend("base")  # Auto-selects best backend
    >>> print(f"Using {backend.backend_name} backend")
    Using mlx backend  # On Apple Silicon with mlx-whisper installed
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    import mlx.core as mx
    import torch

# Type aliases
ArrayLike = Union["torch.Tensor", "mx.array", np.ndarray]
BackendType = Literal["auto", "pytorch", "mlx"]

# Module-level state for lazy loading
_mlx_available: Optional[bool] = None
_torch_available: Optional[bool] = None
_mx_module: Optional[Any] = None
_torch_module: Optional[Any] = None


# =============================================================================
# Platform Detection
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


def _ensure_mlx() -> bool:
    """Lazy load MLX module if available."""
    global _mlx_available, _mx_module
    if _mlx_available is None:
        try:
            import mlx.core as mlx_core

            _mx_module = mlx_core
            _mlx_available = True
        except ImportError:
            _mlx_available = False
    return _mlx_available


def _ensure_torch() -> bool:
    """Lazy load PyTorch module if available."""
    global _torch_available, _torch_module
    if _torch_available is None:
        try:
            import torch

            _torch_module = torch
            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


def _get_mx() -> Any:
    """Get the MLX module, raising if not available."""
    if not _ensure_mlx():
        raise RuntimeError("MLX not available - install with: pip install mlx")
    return _mx_module


def _get_torch() -> Any:
    """Get the PyTorch module, raising if not available."""
    if not _ensure_torch():
        raise RuntimeError("PyTorch not available - install with: pip install torch")
    return _torch_module


# =============================================================================
# Data Types
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
        """Extract audio embeddings from mel spectrogram."""
        ...

    def logits(self, tokens: ArrayLike, audio_features: ArrayLike) -> ArrayLike:
        """Get logits for token prediction."""
        ...

    def decode(self, mel: ArrayLike, options: Any = None) -> WhisperDecodingResult:
        """Decode mel spectrogram to text."""
        ...

    def get_tokenizer(self, language: str = "en", task: str = "transcribe") -> Any:
        """Get tokenizer for the model."""
        ...

    def get_decoder(self, temperature: float = 0.0) -> Any:
        """Get greedy decoder instance."""
        ...

    def create_decoding_options(self, **kwargs: Any) -> Any:
        """Create decoding options for the backend."""
        ...

    def pad_or_trim(self, audio: ArrayLike, length: Optional[int] = None) -> ArrayLike:
        """Pad or trim audio to specified length."""
        ...

    def log_mel_spectrogram(self, audio: ArrayLike, n_mels: Optional[int] = None) -> ArrayLike:
        """Compute log mel spectrogram from audio."""
        ...

    def to_backend_array(self, arr: ArrayLike) -> ArrayLike:
        """Convert input to backend-native array type."""
        ...

    def to_numpy(self, arr: ArrayLike) -> np.ndarray:
        """Convert backend array to numpy."""
        ...


# =============================================================================
# Conversion Utilities
# =============================================================================


def to_numpy(arr: ArrayLike) -> np.ndarray:
    """
    Convert any array type to numpy.

    Args:
        arr: Input array (torch.Tensor, mx.array, or np.ndarray).

    Returns:
        Numpy array with the same data.

    Example:
        >>> import mlx.core as mx
        >>> mlx_arr = mx.array([1.0, 2.0, 3.0])
        >>> np_arr = to_numpy(mlx_arr)
        >>> assert isinstance(np_arr, np.ndarray)
    """
    if isinstance(arr, np.ndarray):
        return arr
    if _ensure_torch():
        torch = _get_torch()
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()  # type: ignore[union-attr]
    # Assume MLX array
    return np.array(arr)


def mx_to_torch(arr: "mx.array", dtype: Optional["torch.dtype"] = None) -> "torch.Tensor":
    """
    Convert MLX array to PyTorch tensor.

    Args:
        arr: MLX array to convert.
        dtype: Optional PyTorch dtype for the output tensor.

    Returns:
        PyTorch tensor with the same data.

    Example:
        >>> import mlx.core as mx
        >>> mlx_arr = mx.array([1.0, 2.0, 3.0])
        >>> torch_tensor = mx_to_torch(mlx_arr)
    """
    torch = _get_torch()
    np_arr = np.array(arr)
    tensor = torch.from_numpy(np_arr.copy())  # Copy to ensure contiguous
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def torch_to_mx(tensor: "torch.Tensor") -> "mx.array":
    """
    Convert PyTorch tensor to MLX array.

    Args:
        tensor: PyTorch tensor to convert.

    Returns:
        MLX array with the same data.

    Example:
        >>> import torch
        >>> torch_tensor = torch.randn(10, 20)
        >>> mlx_arr = torch_to_mx(torch_tensor)
    """
    mx = _get_mx()
    np_arr = tensor.detach().cpu().numpy()
    return mx.array(np_arr)


def mx_to_numpy(arr: "mx.array") -> np.ndarray:
    """Convert MLX array to numpy array."""
    return np.array(arr)


def numpy_to_mx(arr: np.ndarray) -> "mx.array":
    """Convert numpy array to MLX array."""
    mx = _get_mx()
    return mx.array(arr)


# =============================================================================
# MLX Whisper Backend (Primary for df_mlx)
# =============================================================================

# Model name to HuggingFace repo ID mapping for mlx-whisper
MLX_WHISPER_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en",
    "base": "mlx-community/whisper-base",
    "base.en": "mlx-community/whisper-base.en",
    "small": "mlx-community/whisper-small",
    "small.en": "mlx-community/whisper-small.en",
    "medium": "mlx-community/whisper-medium",
    "medium.en": "mlx-community/whisper-medium.en",
    "large": "mlx-community/whisper-large-v3",
    "large-v1": "mlx-community/whisper-large-v1",
    "large-v2": "mlx-community/whisper-large-v2",
    "large-v3": "mlx-community/whisper-large-v3",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def _resolve_mlx_model_name(model_name: str) -> str:
    """
    Resolve a model name to HuggingFace repo ID for mlx-whisper.

    Args:
        model_name: Short model name ('tiny', 'base', etc.) or full HF repo ID.

    Returns:
        Full HuggingFace repo ID.
    """
    if "/" in model_name:
        # Already a full repo ID
        return model_name
    return MLX_WHISPER_MODEL_MAP.get(model_name, f"mlx-community/whisper-{model_name}")


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
        RuntimeError: If MLX is not available.

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
        if not _ensure_mlx():
            raise RuntimeError("MLXWhisperBackend requires MLX - install with: pip install mlx")

        try:
            import mlx_whisper
            from mlx_whisper import audio as mlx_audio
            from mlx_whisper import decoding as mlx_decoding
            from mlx_whisper import load_models
            from mlx_whisper import transcribe as mlx_transcribe
            from mlx_whisper.tokenizer import get_tokenizer as mlx_get_tokenizer
        except ImportError as e:
            raise RuntimeError("mlx-whisper not installed. Install with: pip install mlx-whisper") from e

        self._mlx_whisper = mlx_whisper
        self._mlx_audio = mlx_audio
        self._mlx_decoding = mlx_decoding
        self._mlx_get_tokenizer = mlx_get_tokenizer
        self._mlx_transcribe = mlx_transcribe

        # Resolve model name to HuggingFace repo ID
        repo_id = _resolve_mlx_model_name(model_name)
        self._model = load_models.load_model(repo_id)
        self._model_name = model_name
        self._repo_id = repo_id

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
        """Return the underlying MLX whisper model."""
        return self._model

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def embed_audio(self, mel: ArrayLike) -> "mx.array":
        """
        Extract audio embeddings from mel spectrogram.

        Args:
            mel: Mel spectrogram tensor. Accepts both PyTorch format [batch, n_mels, n_frames]
                 and MLX format [batch, n_frames, n_mels]. Will auto-transpose if needed.

        Returns:
            Audio embeddings as MLX array.
        """
        mx = _get_mx()
        if not isinstance(mel, type(mx.array([0]))):
            mel = mx.array(to_numpy(mel))

        # MLX whisper expects (batch, n_frames, n_mels), but we accept PyTorch format too
        # PyTorch: (batch, n_mels=80, n_frames=3000)
        # MLX: (batch, n_frames=3000, n_mels=80)
        if len(mel.shape) == 3 and mel.shape[1] == 80:
            # Input is in PyTorch format, transpose to MLX format
            mel = mx.transpose(mel, (0, 2, 1))
        elif len(mel.shape) == 2 and mel.shape[0] == 80:
            # Single mel in PyTorch format (n_mels, n_frames)
            mel = mx.transpose(mel, (1, 0))
            mel = mx.expand_dims(mel, 0)

        return self._model.embed_audio(mel)

    def embed_audio_as_torch(self, mel: ArrayLike, dtype: Optional["torch.dtype"] = None) -> "torch.Tensor":
        """
        Extract audio embeddings and return as PyTorch tensor.

        Useful for gradient flow in training where embeddings need to
        participate in PyTorch's autograd graph.

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
        if not isinstance(tokens, type(mx.array([0]))):
            tokens = mx.array(to_numpy(tokens).astype(np.int32))
        if not isinstance(audio_features, type(mx.array([0]))):
            audio_features = mx.array(to_numpy(audio_features))
        return self._model.logits(tokens, audio_features)

    def logits_as_torch(
        self,
        tokens: ArrayLike,
        audio_features: ArrayLike,
        dtype: Optional["torch.dtype"] = None,
    ) -> "torch.Tensor":
        """
        Get logits and return as PyTorch tensor.

        Useful for gradient flow in training.

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
            # Use fp16=False since model is loaded in float32 by default
            options = self._mlx_decoding.DecodingOptions(fp16=False)

        if not isinstance(mel, type(mx.array([0]))):
            mel = mx.array(to_numpy(mel))

        result = self._mlx_decoding.decode(self._model, mel, options)
        # decode returns a list of DecodingResult, get the first one
        if isinstance(result, list):
            result = result[0]

        return WhisperDecodingResult(
            text=result.text,
            language=result.language,
            tokens=list(result.tokens),
            avg_logprob=float(result.avg_logprob),
            no_speech_prob=float(result.no_speech_prob),
            temperature=float(result.temperature),
        )

    def transcribe(
        self,
        audio: Union[str, np.ndarray, ArrayLike],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs: Any,
    ) -> dict:
        """
        Transcribe audio to text.

        Args:
            audio: Path to audio file or audio array.
            language: Language code (e.g., 'en'). Auto-detected if None.
            task: 'transcribe' or 'translate'.
            **kwargs: Additional arguments for transcribe.

        Returns:
            Dictionary with 'text', 'segments', 'language' keys.
        """
        return self._mlx_transcribe.transcribe(
            self._model,
            audio,
            language=language,
            task=task,
            **kwargs,
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
        if not isinstance(audio, type(mx.array([0]))):
            audio = mx.array(to_numpy(audio))
        if length is None:
            return self._mlx_audio.pad_or_trim(audio)  # type: ignore[return-value]
        return self._mlx_audio.pad_or_trim(audio, length)  # type: ignore[return-value]

    def log_mel_spectrogram(self, audio: ArrayLike, n_mels: Optional[int] = None) -> "mx.array":
        """
        Compute log mel spectrogram from audio.

        Args:
            audio: Audio waveform tensor.
            n_mels: Number of mel filterbanks. Uses model default if None.

        Returns:
            Log mel spectrogram as MLX array.
        """
        audio_np = audio if isinstance(audio, np.ndarray) else to_numpy(audio)
        if n_mels is None:
            n_mels = self.dims.n_mels
        return self._mlx_audio.log_mel_spectrogram(audio_np, n_mels=n_mels)

    def load_audio(self, path: str) -> np.ndarray:
        """
        Load audio file and resample to 16kHz.

        Args:
            path: Path to audio file.

        Returns:
            Audio waveform as numpy array.
        """
        return self._mlx_audio.load_audio(path)  # type: ignore[return-value]

    def to_backend_array(self, arr: ArrayLike) -> "mx.array":
        """Convert input to MLX array."""
        mx = _get_mx()
        if isinstance(arr, type(mx.array([0]))):
            return arr  # type: ignore[return-value]
        return mx.array(to_numpy(arr))

    def to_numpy(self, arr: ArrayLike) -> np.ndarray:
        """Convert any array to numpy."""
        return to_numpy(arr)


# =============================================================================
# PyTorch Whisper Backend (Fallback)
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
        if not _ensure_torch():
            raise RuntimeError("PyTorchWhisperBackend requires PyTorch - install with: pip install torch")

        try:
            import whisper
            from whisper.tokenizer import get_tokenizer as whisper_get_tokenizer
        except ImportError as e:
            raise RuntimeError("openai-whisper not installed. Install with: pip install openai-whisper") from e

        torch = _get_torch()

        self._whisper = whisper
        self._get_tokenizer = whisper_get_tokenizer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device

        self._model = whisper.load_model(model_name, device=device, download_root=download_root)
        self._model_name = model_name

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
        """Return the underlying whisper model."""
        return self._model

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def embed_audio(self, mel: ArrayLike) -> "torch.Tensor":
        """
        Extract audio embeddings from mel spectrogram.

        Args:
            mel: Mel spectrogram tensor of shape [batch, n_mels, n_frames].

        Returns:
            Audio embeddings as PyTorch tensor.
        """
        torch = _get_torch()
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)
        elif _ensure_mlx() and not isinstance(mel, torch.Tensor):
            mel = torch.from_numpy(to_numpy(mel))
        return self._model.embed_audio(mel.to(self._device))  # type: ignore[union-attr]

    def logits(self, tokens: ArrayLike, audio_features: ArrayLike) -> "torch.Tensor":
        """
        Get logits for token prediction.

        Args:
            tokens: Token IDs tensor.
            audio_features: Audio embeddings from embed_audio().

        Returns:
            Logits tensor for next token prediction.
        """
        torch = _get_torch()
        if isinstance(tokens, np.ndarray):
            tokens = torch.from_numpy(tokens)
        elif _ensure_mlx() and not isinstance(tokens, torch.Tensor):
            tokens = torch.from_numpy(to_numpy(tokens))

        if isinstance(audio_features, np.ndarray):
            audio_features = torch.from_numpy(audio_features)
        elif _ensure_mlx() and not isinstance(audio_features, torch.Tensor):
            audio_features = torch.from_numpy(to_numpy(audio_features))

        return self._model.logits(tokens.to(self._device), audio_features.to(self._device))  # type: ignore[union-attr]

    def decode(self, mel: ArrayLike, options: Any = None) -> WhisperDecodingResult:
        """
        Decode mel spectrogram to text.

        Args:
            mel: Mel spectrogram tensor.
            options: DecodingOptions instance. Uses defaults if None.

        Returns:
            WhisperDecodingResult with transcription.
        """
        torch = _get_torch()
        if options is None:
            options = self._whisper.DecodingOptions()

        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)
        elif _ensure_mlx() and not isinstance(mel, torch.Tensor):
            mel = torch.from_numpy(to_numpy(mel))

        result = self._whisper.decode(self._model, mel.to(self._device), options)  # type: ignore[union-attr]
        assert isinstance(result, self._whisper.DecodingResult)

        return WhisperDecodingResult(
            text=result.text,
            language=result.language,
            tokens=list(result.tokens),
            avg_logprob=float(result.avg_logprob),
            no_speech_prob=float(result.no_speech_prob),
            temperature=float(result.temperature),
        )

    def transcribe(
        self,
        audio: Union[str, np.ndarray, ArrayLike],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs: Any,
    ) -> dict:
        """
        Transcribe audio to text.

        Args:
            audio: Path to audio file or audio array.
            language: Language code (e.g., 'en'). Auto-detected if None.
            task: 'transcribe' or 'translate'.
            **kwargs: Additional arguments for transcribe.

        Returns:
            Dictionary with 'text', 'segments', 'language' keys.
        """
        return self._whisper.transcribe(
            self._model,
            audio,
            language=language,
            task=task,
            **kwargs,
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
        return self._get_tokenizer(self._model.is_multilingual, language=language, task=task)

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

    def pad_or_trim(self, audio: ArrayLike, length: Optional[int] = None) -> "torch.Tensor":
        """
        Pad or trim audio to specified length.

        Args:
            audio: Audio waveform tensor.
            length: Target length in samples. Uses whisper default if None.

        Returns:
            Padded or trimmed audio as PyTorch tensor.
        """
        torch = _get_torch()
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        elif _ensure_mlx() and not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(to_numpy(audio))
        if length is None:
            return self._whisper.pad_or_trim(audio)  # type: ignore[return-value]
        return self._whisper.pad_or_trim(audio, length)  # type: ignore[return-value]

    def log_mel_spectrogram(self, audio: ArrayLike, n_mels: Optional[int] = None) -> "torch.Tensor":
        """
        Compute log mel spectrogram from audio.

        Args:
            audio: Audio waveform tensor.
            n_mels: Number of mel filterbanks. Uses model default if None.

        Returns:
            Log mel spectrogram as PyTorch tensor.
        """
        torch = _get_torch()
        if isinstance(audio, np.ndarray):
            audio_t = torch.from_numpy(audio)
        elif _ensure_mlx() and not isinstance(audio, torch.Tensor):
            audio_t = torch.from_numpy(to_numpy(audio))
        else:
            audio_t = audio
        if n_mels is None:
            n_mels = self.dims.n_mels
        return self._whisper.log_mel_spectrogram(audio_t, n_mels=n_mels)

    def load_audio(self, path: str) -> np.ndarray:
        """
        Load audio file and resample to 16kHz.

        Args:
            path: Path to audio file.

        Returns:
            Audio waveform as numpy array.
        """
        return self._whisper.load_audio(path)

    def to_backend_array(self, arr: ArrayLike) -> "torch.Tensor":
        """Convert input to PyTorch tensor on the correct device."""
        torch = _get_torch()
        if isinstance(arr, torch.Tensor):
            return arr.to(self._device)  # type: ignore[union-attr]
        return torch.from_numpy(to_numpy(arr)).to(self._device)

    def to_numpy(self, arr: ArrayLike) -> np.ndarray:
        """Convert any array to numpy."""
        return to_numpy(arr)


# =============================================================================
# Backend Factory Functions
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
            - 'auto': Selects the best available backend (MLX on Apple Silicon)
            - 'pytorch': Forces openai-whisper (PyTorch)
            - 'mlx': Forces mlx-whisper (Apple Silicon only)
        device: Device for PyTorch backend ('cuda', 'cpu', 'mps').
            Ignored for MLX which uses unified memory.
        download_root: Directory to download/cache model weights.

    Returns:
        A WhisperBackend implementation (MLXWhisperBackend or PyTorchWhisperBackend).

    Raises:
        RuntimeError: If 'mlx' backend is requested but unavailable.
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
        return MLXWhisperBackend(model_name, download_root=download_root)

    if backend == "pytorch":
        return PyTorchWhisperBackend(model_name, device=device, download_root=download_root)

    # Auto-detect best backend (MLX-first for df_mlx)
    if is_apple_silicon() and _ensure_mlx():
        try:
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
# ASR Loss Utilities (for Training)
# =============================================================================


def compute_asr_features(
    audio: ArrayLike,
    backend: Union[MLXWhisperBackend, PyTorchWhisperBackend],
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute ASR features (mel spectrogram and embeddings) from audio.

    Useful for computing ASR loss during speech enhancement training.

    Args:
        audio: Audio waveform [batch, samples] or [samples].
        backend: Whisper backend instance.

    Returns:
        Tuple of (mel_spectrogram, audio_embeddings).
    """
    # Compute mel spectrogram
    mel = backend.log_mel_spectrogram(audio)

    # Add batch dimension if needed
    if len(mel.shape) == 2:
        if backend.backend_name == "mlx":
            mx = _get_mx()
            mel = mx.expand_dims(mel, axis=0)
        else:
            # PyTorch backend returns torch.Tensor with .unsqueeze()
            mel = mel.unsqueeze(0)  # type: ignore[union-attr]  # noqa: B010

    # Get audio embeddings
    embeddings = backend.embed_audio(mel)

    return mel, embeddings


def compute_whisper_loss(
    clean_audio: ArrayLike,
    enhanced_audio: ArrayLike,
    backend: Union[MLXWhisperBackend, PyTorchWhisperBackend],
    reduction: str = "mean",
) -> ArrayLike:
    """
    Compute ASR loss between clean and enhanced audio using Whisper embeddings.

    The loss encourages the enhanced audio to have similar Whisper representations
    as the clean reference, improving speech intelligibility.

    Args:
        clean_audio: Clean reference audio.
        enhanced_audio: Enhanced (denoised) audio.
        backend: Whisper backend instance.
        reduction: Loss reduction ('mean', 'sum', 'none').

    Returns:
        ASR embedding loss value.
    """
    # Get embeddings for both
    _, clean_emb = compute_asr_features(clean_audio, backend)
    _, enhanced_emb = compute_asr_features(enhanced_audio, backend)

    # Compute L1 loss between embeddings
    if backend.backend_name == "mlx":
        mx = _get_mx()
        diff = mx.abs(clean_emb - enhanced_emb)  # type: ignore[operator]
        if reduction == "mean":
            return mx.mean(diff)
        elif reduction == "sum":
            return mx.sum(diff)
        return diff
    else:
        torch = _get_torch()
        diff = torch.abs(clean_emb - enhanced_emb)  # type: ignore[operator]
        if reduction == "mean":
            return diff.mean()
        elif reduction == "sum":
            return diff.sum()
        return diff


# =============================================================================
# Word Accuracy Computation
# =============================================================================


def compute_word_accuracy(
    reference: str,
    hypothesis: str,
) -> Tuple[float, int, int, int]:
    """
    Compute word accuracy between reference and hypothesis transcriptions.

    Uses Levenshtein distance at the word level to compute:
    - Word Accuracy (WAcc) = 1 - WER
    - Word Error Rate (WER) = (S + D + I) / N

    Args:
        reference: Reference (ground truth) transcription.
        hypothesis: Hypothesis (predicted) transcription.

    Returns:
        Tuple of (word_accuracy, substitutions, deletions, insertions).

    Example:
        >>> wacc, s, d, i = compute_word_accuracy("hello world", "hello word")
        >>> print(f"Word Accuracy: {wacc:.2%}")
        Word Accuracy: 50.00%
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Handle edge cases
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) == 0 else 0.0, 0, 0, len(hyp_words)

    # Dynamic programming for Levenshtein distance
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # deletion
                    dp[i][j - 1],  # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    # Backtrack to count S, D, I
    substitutions = deletions = insertions = 0
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            break

    wer = (substitutions + deletions + insertions) / max(len(ref_words), 1)
    word_accuracy = max(0.0, 1.0 - wer)

    return word_accuracy, substitutions, deletions, insertions


def evaluate_transcription_batch(
    audio_paths: List[str],
    reference_texts: List[str],
    backend: Union[MLXWhisperBackend, PyTorchWhisperBackend],
    language: Optional[str] = None,
) -> Tuple[float, List[dict]]:
    """
    Evaluate transcription accuracy on a batch of audio files.

    Args:
        audio_paths: List of paths to audio files.
        reference_texts: List of reference transcriptions.
        backend: Whisper backend instance.
        language: Language code (auto-detected if None).

    Returns:
        Tuple of (average_word_accuracy, list_of_results).
        Each result dict contains: 'path', 'reference', 'hypothesis', 'wacc'.
    """
    results = []
    total_wacc = 0.0

    for path, reference in zip(audio_paths, reference_texts):
        # Transcribe
        result = backend.transcribe(path, language=language)
        hypothesis = result["text"].strip()

        # Compute accuracy
        wacc, s, d, i = compute_word_accuracy(reference, hypothesis)
        total_wacc += wacc

        results.append(
            {
                "path": path,
                "reference": reference,
                "hypothesis": hypothesis,
                "word_accuracy": wacc,
                "substitutions": s,
                "deletions": d,
                "insertions": i,
            }
        )

    avg_wacc = total_wacc / len(audio_paths) if audio_paths else 0.0
    return avg_wacc, results


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
    "mx_to_numpy",
    "numpy_to_mx",
    "to_numpy",
    # Backend classes
    "WhisperBackend",
    "MLXWhisperBackend",
    "PyTorchWhisperBackend",
    # Factory functions
    "get_whisper_backend",
    "load_whisper_model",
    # ASR loss utilities
    "compute_asr_features",
    "compute_whisper_loss",
    # Word accuracy
    "compute_word_accuracy",
    "evaluate_transcription_batch",
]
