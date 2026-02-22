"""Tests for StreamingDfNet4 compiled inference."""

import mlx.core as mx
import pytest


class TestStreamingCompile:
    @pytest.fixture
    def model(self):
        from df_mlx.config import get_default_config
        from df_mlx.model import DfNet4

        p = get_default_config()
        model = DfNet4(p)
        mx.eval(model.parameters())
        return model

    @pytest.fixture
    def streaming(self, model):
        from df_mlx.model import StreamingDfNet4

        return StreamingDfNet4(model)

    def test_compiled_matches_eager_single_frame(self, streaming):
        """Compiled and eager paths produce identical output for one frame."""
        state_compiled = streaming.init_state(batch_size=1)
        state_eager = streaming.init_state(batch_size=1)

        audio = mx.random.normal((1, streaming.hop_length)) * 0.1
        mx.eval(audio)

        out_compiled, _ = streaming.process_frame(audio, state_compiled, compiled=True)
        out_eager, _ = streaming.process_frame(audio, state_eager, compiled=False)
        mx.eval(out_compiled, out_eager)

        assert mx.allclose(out_compiled, out_eager, atol=1e-5).item()

    def test_compiled_matches_eager_multi_frame(self, streaming):
        """Compiled and eager paths match over multiple frames."""
        state_compiled = streaming.init_state(batch_size=1)
        state_eager = streaming.init_state(batch_size=1)

        for _ in range(5):
            audio = mx.random.normal((1, streaming.hop_length)) * 0.1
            mx.eval(audio)

            out_c, state_compiled = streaming.process_frame(audio, state_compiled, compiled=True)
            out_e, state_eager = streaming.process_frame(audio, state_eager, compiled=False)
            mx.eval(out_c, out_e)

            assert mx.allclose(out_c, out_e, atol=1e-5).item()

    def test_state_updated_after_compiled_frame(self, streaming):
        """State arrays are properly updated after compiled frame processing."""
        state = streaming.init_state(batch_size=1)

        audio = mx.random.normal((1, streaming.hop_length)) * 0.1
        mx.eval(audio)

        _, state = streaming.process_frame(audio, state, compiled=True)

        assert state.frame_count == 1
        assert state.mamba_states is not None

    def test_compiled_1d_input(self, streaming):
        """1D input works with compiled path."""
        state = streaming.init_state(batch_size=1)

        audio = mx.random.normal((streaming.hop_length,)) * 0.1
        mx.eval(audio)

        out, _ = streaming.process_frame(audio, state, compiled=True)
        mx.eval(out)

        assert out.ndim == 1
        assert out.shape[0] == streaming.hop_length

    def test_ensure_mamba_initialized(self):
        """ensure_mamba_initialized creates zero state when None."""
        from df_mlx.model import StreamingState

        state = StreamingState(
            batch_size=2,
            n_fft=960,
            hop_length=480,
            d_inner=256,
            d_state=16,
            num_layers=4,
        )
        assert state.mamba_states is None
        state.ensure_mamba_initialized(256, 16, 4)
        assert state.mamba_states is not None
        assert state.mamba_states.shape == (4, 2, 256, 16)
        assert mx.all(state.mamba_states == 0).item()

    def test_ensure_mamba_initialized_noop_when_present(self):
        """ensure_mamba_initialized is a no-op when state already exists."""
        from df_mlx.model import StreamingState

        state = StreamingState(
            batch_size=1,
            n_fft=960,
            hop_length=480,
            d_inner=256,
            d_state=16,
            num_layers=4,
        )
        state.mamba_states = mx.ones((4, 1, 256, 16))
        state.ensure_mamba_initialized(256, 16, 4)
        assert mx.all(state.mamba_states == 1).item()

    def test_compiled_fn_reused(self, streaming):
        """Compiled function is created once and reused."""
        state = streaming.init_state(batch_size=1)
        audio = mx.random.normal((1, streaming.hop_length)) * 0.1
        mx.eval(audio)

        streaming.process_frame(audio, state, compiled=True)
        fn1 = streaming._compiled_fn

        streaming.process_frame(audio, state, compiled=True)
        fn2 = streaming._compiled_fn

        assert fn1 is fn2
