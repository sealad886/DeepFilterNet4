from .config import config
from .enhance import enhance, init_df
from .version import version

__all__ = ["config", "version", "enhance", "init_df"]
__version__ = version

# Whisper adapter (optional, for ASR-based loss and evaluation)
# Auto-selects MLX backend on Apple Silicon for 5-10x speedup
try:
    from df.whisper_adapter import (  # noqa: F401 - exported via __all__
        PyTorchWhisperBackend,
        WhisperDecodingResult,
        get_whisper_backend,
        is_apple_silicon,
        load_whisper_model,
    )

    __all__.extend(
        [
            "get_whisper_backend",
            "load_whisper_model",
            "is_apple_silicon",
            "WhisperDecodingResult",
            "PyTorchWhisperBackend",
        ]
    )

    # Only export MLXWhisperBackend if available on Apple Silicon
    if is_apple_silicon():
        try:
            from df.whisper_adapter import MLXWhisperBackend  # noqa: F401 - exported via __all__

            __all__.append("MLXWhisperBackend")
        except ImportError:
            pass  # MLX not available on this Apple Silicon system
except ImportError:
    pass  # whisper not installed
