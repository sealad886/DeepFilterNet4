"""Tests for the whisper adapter module.

This module tests the hybrid whisper backend abstraction that supports
both PyTorch (openai-whisper) and MLX (mlx-whisper) backends.
"""

import numpy as np
import pytest
import torch

from df.whisper_adapter import WhisperDecodingResult, get_whisper_backend, is_apple_silicon, to_numpy

# =============================================================================
# Platform Detection Tests
# =============================================================================


class TestPlatformDetection:
    """Tests for is_apple_silicon() platform detection."""

    def test_is_apple_silicon_returns_bool(self):
        """Verify is_apple_silicon returns a boolean value."""
        result = is_apple_silicon()
        assert isinstance(result, bool)

    def test_is_apple_silicon_cached(self):
        """Verify is_apple_silicon returns consistent results (cached)."""
        result1 = is_apple_silicon()
        result2 = is_apple_silicon()
        assert result1 == result2

    def test_is_apple_silicon_no_exception(self):
        """Verify is_apple_silicon handles edge cases gracefully."""
        # Should never raise an exception
        try:
            _ = is_apple_silicon()
        except Exception as e:
            pytest.fail(f"is_apple_silicon() raised exception: {e}")


# =============================================================================
# Tensor Conversion Tests
# =============================================================================


class TestTensorConversion:
    """Tests for tensor conversion utilities."""

    def test_to_numpy_from_torch(self):
        """Convert PyTorch tensor to numpy array."""
        tensor = torch.randn(10, 20)
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 20)

    def test_to_numpy_from_numpy(self):
        """Convert numpy array to numpy (should return same object)."""
        arr = np.random.randn(10, 20).astype(np.float32)
        result = to_numpy(arr)
        # Should return the same object for efficiency
        assert result is arr

    def test_to_numpy_preserves_values(self):
        """Verify tensor values are preserved during conversion."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = to_numpy(tensor)
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_to_numpy_from_cuda_tensor(self):
        """Convert CUDA tensor to numpy (if CUDA available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        tensor = torch.randn(5, 5, device="cuda")
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 5)

    def test_to_numpy_from_mps_tensor(self):
        """Convert MPS tensor to numpy (if MPS available)."""
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available")
        tensor = torch.randn(5, 5, device="mps")
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 5)


@pytest.mark.skipif(not is_apple_silicon(), reason="MLX only available on Apple Silicon")
class TestMLXConversion:
    """Tests for MLX-specific tensor conversion (Apple Silicon only)."""

    def test_mx_to_torch(self):
        """Convert MLX array to PyTorch tensor."""
        pytest.importorskip("mlx")
        from df.whisper_adapter import _ensure_mlx, mx_to_torch

        _ensure_mlx()
        import mlx.core as mx

        arr = mx.random.normal((10, 20))
        result = mx_to_torch(arr)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 20)

    def test_torch_to_mx(self):
        """Convert PyTorch tensor to MLX array."""
        pytest.importorskip("mlx")
        from df.whisper_adapter import _ensure_mlx, torch_to_mx

        _ensure_mlx()
        import mlx.core as mx

        tensor = torch.randn(10, 20)
        result = torch_to_mx(tensor)
        assert isinstance(result, mx.array)
        assert result.shape == (10, 20)

    def test_roundtrip_torch_mx_torch(self):
        """Verify roundtrip conversion preserves values."""
        pytest.importorskip("mlx")
        from df.whisper_adapter import _ensure_mlx, mx_to_torch, torch_to_mx

        _ensure_mlx()

        original = torch.randn(5, 5)
        mx_arr = torch_to_mx(original)
        restored = mx_to_torch(mx_arr)

        # Check values are close (float precision may vary)
        torch.testing.assert_close(original, restored, rtol=1e-5, atol=1e-5)

    def test_mx_to_torch_with_dtype(self):
        """Convert MLX array to PyTorch tensor with specific dtype."""
        pytest.importorskip("mlx")
        from df.whisper_adapter import _ensure_mlx, mx_to_torch

        _ensure_mlx()
        import mlx.core as mx

        arr = mx.random.normal((5, 5))
        result = mx_to_torch(arr, dtype=torch.float16)
        assert result.dtype == torch.float16


# =============================================================================
# WhisperDecodingResult Tests
# =============================================================================


class TestWhisperDecodingResult:
    """Tests for the unified decoding result dataclass."""

    def test_create_decoding_result(self):
        """Create a WhisperDecodingResult instance."""
        result = WhisperDecodingResult(
            text="Hello world",
            language="en",
            tokens=[1, 2, 3, 4],
            avg_logprob=-0.5,
            no_speech_prob=0.01,
            temperature=0.0,
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.tokens == [1, 2, 3, 4]
        assert result.avg_logprob == -0.5
        assert result.no_speech_prob == 0.01
        assert result.temperature == 0.0


# =============================================================================
# Backend Factory Tests
# =============================================================================


class TestBackendFactory:
    """Tests for get_whisper_backend() factory function."""

    @pytest.mark.skipif(
        pytest.importorskip("whisper", reason="whisper required") is None,
        reason="whisper not installed",
    )
    def test_pytorch_backend_explicit(self):
        """Request PyTorch backend explicitly."""
        pytest.importorskip("whisper")
        backend = get_whisper_backend("tiny", backend="pytorch")
        assert backend.backend_name == "pytorch"

    @pytest.mark.skipif(
        pytest.importorskip("whisper", reason="whisper required") is None,
        reason="whisper not installed",
    )
    def test_pytorch_backend_properties(self):
        """Verify PyTorch backend properties."""
        pytest.importorskip("whisper")
        backend = get_whisper_backend("tiny", backend="pytorch")
        assert backend.device in ("cpu", "cuda", "mps")
        assert hasattr(backend, "dims")
        assert isinstance(backend.is_multilingual, bool)

    @pytest.mark.skipif(not is_apple_silicon(), reason="MLX only on Apple Silicon")
    def test_mlx_backend_explicit(self):
        """Request MLX backend explicitly (Apple Silicon only)."""
        pytest.importorskip("mlx_whisper")
        backend = get_whisper_backend("tiny", backend="mlx")
        assert backend.backend_name == "mlx"
        assert backend.device is None  # MLX uses unified memory

    @pytest.mark.skipif(not is_apple_silicon(), reason="MLX only on Apple Silicon")
    def test_mlx_backend_properties(self):
        """Verify MLX backend properties."""
        pytest.importorskip("mlx_whisper")
        backend = get_whisper_backend("tiny", backend="mlx")
        assert hasattr(backend, "dims")
        assert isinstance(backend.is_multilingual, bool)

    @pytest.mark.skipif(
        pytest.importorskip("whisper", reason="whisper required") is None,
        reason="whisper not installed",
    )
    def test_auto_backend_returns_valid_backend(self):
        """Auto backend should return a working backend."""
        pytest.importorskip("whisper")
        backend = get_whisper_backend("tiny", backend="auto")
        assert backend.backend_name in ("pytorch", "mlx")


# =============================================================================
# Platform Fallback Tests
# =============================================================================


class TestPlatformFallback:
    """Tests for platform fallback behavior."""

    @pytest.mark.skipif(
        pytest.importorskip("whisper", reason="whisper required") is None,
        reason="whisper not installed",
    )
    def test_auto_backend_always_works(self):
        """Auto backend should always return a working backend."""
        pytest.importorskip("whisper")
        backend = get_whisper_backend("tiny", backend="auto")
        assert backend.backend_name in ("pytorch", "mlx")

    def test_mlx_on_non_apple_raises(self):
        """Requesting MLX on non-Apple Silicon should raise RuntimeError."""
        if is_apple_silicon():
            pytest.skip("Test for non-Apple Silicon only")
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            get_whisper_backend("tiny", backend="mlx")

    @pytest.mark.skipif(
        pytest.importorskip("whisper", reason="whisper required") is None,
        reason="whisper not installed",
    )
    def test_pytorch_backend_on_any_platform(self):
        """PyTorch backend should work on any platform."""
        pytest.importorskip("whisper")
        backend = get_whisper_backend("tiny", backend="pytorch")
        assert backend.backend_name == "pytorch"


# =============================================================================
# PyTorch Backend Functionality Tests
# =============================================================================


@pytest.mark.skipif(
    pytest.importorskip("whisper", reason="whisper required") is None,
    reason="whisper not installed",
)
class TestPyTorchBackendFunctionality:
    """Tests for PyTorchWhisperBackend functionality."""

    @pytest.fixture
    def pytorch_backend(self):
        """Create a PyTorch backend for testing."""
        pytest.importorskip("whisper")
        return get_whisper_backend("tiny", backend="pytorch")

    def test_embed_audio(self, pytorch_backend):
        """Test audio embedding extraction."""
        # Create a mel spectrogram-like input (n_mels x time)
        mel = torch.randn(1, 80, 3000)
        features = pytorch_backend.embed_audio(mel)
        assert isinstance(features, torch.Tensor)
        assert len(features.shape) == 3  # (batch, time, features)

    def test_get_tokenizer(self, pytorch_backend):
        """Test tokenizer creation."""
        tokenizer = pytorch_backend.get_tokenizer(language="en", task="transcribe")
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")
        assert hasattr(tokenizer, "eot")

    def test_get_decoder(self, pytorch_backend):
        """Test decoder creation."""
        decoder = pytorch_backend.get_decoder(temperature=0.0)
        assert decoder is not None

    def test_create_decoding_options(self, pytorch_backend):
        """Test decoding options creation."""
        options = pytorch_backend.create_decoding_options(language="en", task="transcribe", beam_size=5, fp16=False)
        assert options is not None
        assert options.language == "en"
        assert options.task == "transcribe"

    def test_to_backend_array(self, pytorch_backend):
        """Test conversion to backend array type."""
        arr = np.random.randn(10, 20).astype(np.float32)
        result = pytorch_backend.to_backend_array(arr)
        assert isinstance(result, torch.Tensor)

    def test_pad_or_trim(self, pytorch_backend):
        """Test audio padding/trimming."""
        audio = torch.randn(16000)  # 1 second at 16kHz
        result = pytorch_backend.pad_or_trim(audio)
        # Default should pad to 30 seconds (480000 samples)
        assert result.shape[0] == 480000

    def test_log_mel_spectrogram(self, pytorch_backend):
        """Test mel spectrogram generation."""
        audio = torch.randn(16000)  # 1 second at 16kHz
        mel = pytorch_backend.log_mel_spectrogram(audio)
        assert isinstance(mel, torch.Tensor)
        assert mel.shape[0] == pytorch_backend.dims.n_mels


# =============================================================================
# MLX Backend Functionality Tests
# =============================================================================


@pytest.mark.skipif(not is_apple_silicon(), reason="MLX only on Apple Silicon")
class TestMLXBackendFunctionality:
    """Tests for MLXWhisperBackend functionality."""

    @pytest.fixture
    def mlx_backend(self):
        """Create an MLX backend for testing."""
        pytest.importorskip("mlx_whisper")
        return get_whisper_backend("tiny", backend="mlx")

    def test_embed_audio(self, mlx_backend):
        """Test audio embedding extraction."""
        pytest.importorskip("mlx")
        import mlx.core as mx

        # Create a mel spectrogram-like input
        mel = mx.random.normal((1, 80, 3000))
        features = mlx_backend.embed_audio(mel)
        assert isinstance(features, mx.array)
        assert len(features.shape) == 3

    def test_embed_audio_as_torch(self, mlx_backend):
        """Test audio embedding with PyTorch output."""
        pytest.importorskip("mlx")
        import mlx.core as mx

        mel = mx.random.normal((1, 80, 3000))
        features = mlx_backend.embed_audio_as_torch(mel)
        assert isinstance(features, torch.Tensor)
        assert len(features.shape) == 3

    def test_get_tokenizer(self, mlx_backend):
        """Test tokenizer creation."""
        tokenizer = mlx_backend.get_tokenizer(language="en", task="transcribe")
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")

    def test_to_backend_array(self, mlx_backend):
        """Test conversion to backend array type."""
        pytest.importorskip("mlx")
        import mlx.core as mx

        arr = np.random.randn(10, 20).astype(np.float32)
        result = mlx_backend.to_backend_array(arr)
        assert isinstance(result, mx.array)


# =============================================================================
# ASRLoss Integration Tests
# =============================================================================


class TestASRLossIntegration:
    """Integration tests for ASRLoss with whisper backends."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio tensor for testing."""
        # 1 second of random audio at 16kHz
        return torch.randn(1, 16000)

    @pytest.mark.skipif(
        pytest.importorskip("whisper", reason="whisper required") is None,
        reason="whisper not installed",
    )
    def test_asrloss_pytorch_backend(self, sample_audio):
        """Test ASRLoss with PyTorch backend."""
        pytest.importorskip("whisper")
        from df.loss import ASRLoss

        loss_fn = ASRLoss(sr=16000, model="tiny", backend="pytorch")
        loss = loss_fn(sample_audio, sample_audio)
        assert loss.item() >= 0
        # Loss should be finite and reasonable (not NaN or inf)
        assert loss.item() < 100  # Just check it's reasonable

    @pytest.mark.skipif(not is_apple_silicon(), reason="MLX only on Apple Silicon")
    def test_asrloss_mlx_backend(self, sample_audio):
        """Test ASRLoss with MLX backend."""
        pytest.importorskip("mlx_whisper")
        from df.loss import ASRLoss

        loss_fn = ASRLoss(sr=16000, model="tiny", backend="mlx")
        loss = loss_fn(sample_audio, sample_audio)
        assert loss.item() >= 0
        # Loss should be finite and reasonable (not NaN or inf)
        assert loss.item() < 100  # Just check it's reasonable

    @pytest.mark.skipif(
        pytest.importorskip("whisper", reason="whisper required") is None,
        reason="whisper not installed",
    )
    def test_asrloss_auto_backend(self, sample_audio):
        """Test ASRLoss with auto backend selection."""
        pytest.importorskip("whisper")
        from df.loss import ASRLoss

        loss_fn = ASRLoss(sr=16000, model="tiny", backend="auto")
        loss = loss_fn(sample_audio, sample_audio)
        assert loss.item() >= 0
        # Loss should be finite and reasonable (not NaN or inf)
        assert loss.item() < 100  # Just check it's reasonable


# =============================================================================
# Transcription Consistency Tests
# =============================================================================


class TestTranscriptionConsistency:
    """Tests for transcription consistency across backends."""

    @pytest.fixture
    def test_audio_path(self, tmp_path):
        """Create a simple test audio file."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        # Create 1 second of low-amplitude noise
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        path = tmp_path / "test.wav"
        sf.write(str(path), audio, 16000)
        return path

    @pytest.mark.skipif(
        pytest.importorskip("whisper", reason="whisper required") is None,
        reason="whisper not installed",
    )
    def test_transcription_runs_pytorch(self, test_audio_path):
        """Test that transcription runs without errors (PyTorch)."""
        whisper = pytest.importorskip("whisper")
        backend = get_whisper_backend("tiny", backend="pytorch")

        # Load and process audio
        audio = whisper.load_audio(str(test_audio_path))
        audio_tensor = torch.from_numpy(audio)
        audio_padded = backend.pad_or_trim(audio_tensor)
        mel = backend.log_mel_spectrogram(audio_padded)

        # Decode
        result = backend.decode(mel)
        assert isinstance(result, WhisperDecodingResult)
        assert isinstance(result.text, str)

    @pytest.mark.skipif(not is_apple_silicon(), reason="MLX only on Apple Silicon")
    def test_transcription_runs_mlx(self, test_audio_path):
        """Test that transcription runs without errors (MLX)."""
        pytest.importorskip("mlx_whisper")
        pytest.importorskip("mlx")
        import mlx.core as mx

        backend = get_whisper_backend("tiny", backend="mlx")

        # Load audio using librosa or similar
        try:
            import librosa

            audio, _ = librosa.load(str(test_audio_path), sr=16000)
        except ImportError:
            import soundfile as sf

            audio, _ = sf.read(str(test_audio_path))

        audio_mx = mx.array(audio.astype(np.float32))
        audio_padded = backend.pad_or_trim(audio_mx)
        mel = backend.log_mel_spectrogram(audio_padded)

        # Decode
        result = backend.decode(mel)
        assert isinstance(result, WhisperDecodingResult)
        assert isinstance(result.text, str)
