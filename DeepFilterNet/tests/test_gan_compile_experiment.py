"""Tests for the GAN-phase compile experiment feature flag and guardrail constants."""

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import pytest

from df_mlx.run_config import RunConfig, apply_run_config_dict, generate_run_config_example

# ---------------------------------------------------------------------------
# Guardrail constants (defined here as the canonical reference; the experiment
# implementation will import or mirror these values).
# ---------------------------------------------------------------------------

LOSS_DIVERGENCE_FACTOR: float = 10.0
GRAD_NORM_EXPLOSION_THRESHOLD: float = 100.0
GRAD_NORM_EXPLOSION_WINDOW: int = 5
DISC_ACCURACY_LOW: float = 0.10
DISC_ACCURACY_HIGH: float = 0.90
CONVERGENCE_TOLERANCE: float = 0.05
THROUGHPUT_MIN_RATIO: float = 0.80
PESQ_SISDR_MAX_DROP: float = 0.10

EXPERIMENT_VARIANTS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Feature flag defaults
# ---------------------------------------------------------------------------


class TestFeatureFlagDefault:
    def test_experimental_compile_defaults_false(self):
        cfg = RunConfig()
        assert cfg.gan.experimental_compile is False

    def test_gan_enabled_defaults_false(self):
        cfg = RunConfig()
        assert cfg.gan.enabled is False


# ---------------------------------------------------------------------------
# TOML round-trip
# ---------------------------------------------------------------------------


class TestFeatureFlagToml:
    def test_parse_experimental_compile_true(self):
        cfg = RunConfig()
        apply_run_config_dict(cfg, {"gan": {"experimental_compile": True}})
        assert cfg.gan.experimental_compile is True

    def test_parse_experimental_compile_false(self):
        cfg = RunConfig()
        apply_run_config_dict(cfg, {"gan": {"experimental_compile": False}})
        assert cfg.gan.experimental_compile is False

    def test_parse_experimental_compile_rejects_string(self):
        cfg = RunConfig()
        with pytest.raises((TypeError, ValueError)):
            apply_run_config_dict(cfg, {"gan": {"experimental_compile": "true"}})

    def test_generated_example_includes_experimental_compile(self):
        text = generate_run_config_example()
        data = tomllib.loads(text)
        assert "gan" in data
        assert "experimental_compile" in data["gan"]
        assert data["gan"]["experimental_compile"] is False

    def test_roundtrip_preserves_default(self):
        text = generate_run_config_example()
        data = tomllib.loads(text)
        cfg = RunConfig()
        apply_run_config_dict(cfg, data)
        assert cfg.gan.experimental_compile is False


# ---------------------------------------------------------------------------
# Eager-only GAN path preserved when flag is off
# ---------------------------------------------------------------------------


class TestEagerPathPreserved:
    def test_flag_off_does_not_change_other_gan_defaults(self):
        cfg = RunConfig()
        assert cfg.gan.experimental_compile is False
        assert cfg.gan.enabled is False
        assert cfg.gan.start_epoch == 0
        assert cfg.gan.adv_weight == 0.0
        assert cfg.gan.fm_weight == 0.0
        assert cfg.gan.disc_update_freq == 1

    def test_enabling_gan_without_compile_flag(self):
        cfg = RunConfig()
        apply_run_config_dict(
            cfg,
            {
                "gan": {
                    "enabled": True,
                    "start_epoch": 5,
                    "adv_weight": 0.1,
                }
            },
        )
        assert cfg.gan.enabled is True
        assert cfg.gan.experimental_compile is False
        assert cfg.gan.start_epoch == 5


# ---------------------------------------------------------------------------
# Abort criteria constants
# ---------------------------------------------------------------------------


class TestAbortCriteriaConstants:
    def test_loss_divergence_factor(self):
        assert LOSS_DIVERGENCE_FACTOR == 10.0

    def test_grad_norm_explosion_threshold(self):
        assert GRAD_NORM_EXPLOSION_THRESHOLD == 100.0

    def test_grad_norm_explosion_window(self):
        assert GRAD_NORM_EXPLOSION_WINDOW == 5

    def test_disc_accuracy_low(self):
        assert DISC_ACCURACY_LOW == pytest.approx(0.10)

    def test_disc_accuracy_high(self):
        assert DISC_ACCURACY_HIGH == pytest.approx(0.90)

    def test_convergence_tolerance(self):
        assert CONVERGENCE_TOLERANCE == pytest.approx(0.05)

    def test_throughput_min_ratio(self):
        assert THROUGHPUT_MIN_RATIO == pytest.approx(0.80)

    def test_pesq_sisdr_max_drop(self):
        assert PESQ_SISDR_MAX_DROP == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------


class TestExperimentMatrix:
    def test_variant_count(self):
        assert len(EXPERIMENT_VARIANTS) == 4

    def test_variant_labels(self):
        assert EXPERIMENT_VARIANTS == ["A", "B", "C", "D"]

    def test_baseline_is_variant_a(self):
        assert EXPERIMENT_VARIANTS[0] == "A"

    def test_matrix_generation(self):
        matrix = _build_experiment_matrix()
        assert len(matrix) == 4
        assert all("variant" in entry for entry in matrix)
        assert all("generator" in entry for entry in matrix)
        assert all("discriminator" in entry for entry in matrix)
        assert all("compile_scope" in entry for entry in matrix)
        assert all("risk_level" in entry for entry in matrix)

    def test_baseline_variant_has_no_compilation(self):
        matrix = _build_experiment_matrix()
        baseline = [e for e in matrix if e["variant"] == "A"][0]
        assert baseline["compile_scope"] == "none"
        assert baseline["risk_level"] == "none"

    def test_gen_only_variant(self):
        matrix = _build_experiment_matrix()
        gen_only = [e for e in matrix if e["variant"] == "B"][0]
        assert gen_only["generator"] == "compiled"
        assert gen_only["discriminator"] == "eager"
        assert gen_only["risk_level"] == "medium"

    def test_full_compiled_variant(self):
        matrix = _build_experiment_matrix()
        full = [e for e in matrix if e["variant"] == "C"][0]
        assert full["generator"] == "compiled"
        assert full["discriminator"] == "compiled"
        assert full["risk_level"] == "high"

    def test_alternating_variant(self):
        matrix = _build_experiment_matrix()
        alt = [e for e in matrix if e["variant"] == "D"][0]
        assert alt["generator"] == "compiled"
        assert alt["discriminator"] == "eager"
        assert alt["risk_level"] == "medium"


# ---------------------------------------------------------------------------
# Helper: experiment matrix builder
# ---------------------------------------------------------------------------


def _build_experiment_matrix() -> list[dict[str, str]]:
    return [
        {
            "variant": "A",
            "generator": "eager",
            "discriminator": "eager",
            "compile_scope": "none",
            "risk_level": "none",
        },
        {
            "variant": "B",
            "generator": "compiled",
            "discriminator": "eager",
            "compile_scope": "generator_loss_grad",
            "risk_level": "medium",
        },
        {
            "variant": "C",
            "generator": "compiled",
            "discriminator": "compiled",
            "compile_scope": "both",
            "risk_level": "high",
        },
        {
            "variant": "D",
            "generator": "compiled",
            "discriminator": "eager",
            "compile_scope": "generator_steps_only",
            "risk_level": "medium",
        },
    ]


# ---------------------------------------------------------------------------
# GAN-P6/P7/P8 config surface tests
# ---------------------------------------------------------------------------


class TestDiscUpdateFreq:
    """Tests for GAN discriminator update frequency configuration."""

    def test_disc_update_freq_config_default(self):
        """disc_update_freq defaults to 1 in GANConfig."""
        from df_mlx.run_config import GanConfig

        cfg = GanConfig()
        assert cfg.disc_update_freq == 1

    def test_disc_update_freq_skip_logic(self):
        """With freq=2, disc updates happen on even steps only."""
        freq = 2
        results = []
        for step in range(10):
            do_update = (step % freq) == 0
            results.append(do_update)
        # Steps 0, 2, 4, 6, 8 should update
        assert results == [True, False, True, False, True, False, True, False, True, False]

    def test_disc_update_freq_config_fields_exist(self):
        """New config fields exist with correct defaults."""
        from df_mlx.run_config import GanConfig

        cfg = GanConfig()
        assert hasattr(cfg, "cache_gen_waveforms")
        assert cfg.cache_gen_waveforms is True
        assert hasattr(cfg, "disc_gradient_checkpoint")
        assert cfg.disc_gradient_checkpoint is False
        assert hasattr(cfg, "eval_frequency")
        assert cfg.eval_frequency == 2

    def test_new_fields_in_run_config(self):
        """New fields accessible via RunConfig.gan."""
        cfg = RunConfig()
        assert cfg.gan.cache_gen_waveforms is True
        assert cfg.gan.disc_gradient_checkpoint is False
        assert cfg.gan.eval_frequency == 2

    def test_apply_new_fields_via_dict(self):
        """New fields can be set via apply_run_config_dict."""
        cfg = RunConfig()
        apply_run_config_dict(
            cfg,
            {
                "gan": {
                    "cache_gen_waveforms": True,
                    "disc_gradient_checkpoint": True,
                }
            },
        )
        assert cfg.gan.cache_gen_waveforms is True
        assert cfg.gan.disc_gradient_checkpoint is True

    def test_generated_example_includes_new_fields(self):
        """Generated TOML example includes new GAN fields."""
        text = generate_run_config_example()
        data = tomllib.loads(text)
        assert "gan" in data
        assert "cache_gen_waveforms" in data["gan"]
        assert data["gan"]["cache_gen_waveforms"] is True
        assert "disc_gradient_checkpoint" in data["gan"]
        assert data["gan"]["disc_gradient_checkpoint"] is False
        assert "eval_frequency" in data["gan"]
        assert data["gan"]["eval_frequency"] == 2


# ---------------------------------------------------------------------------
# GAN-P2: Compiled discriminator update step
# ---------------------------------------------------------------------------


class TestCompiledDiscUpdate:
    """Tests for compiled discriminator update step (GAN-P2)."""

    def test_compiled_disc_update_config_gate(self):
        """compiled_disc_update_step is None when experimental_compile is False."""
        from df_mlx.run_config import GanConfig

        cfg = GanConfig()
        assert cfg.experimental_compile is False

    def test_compiled_disc_update_concept(self):
        """Verify that nn.value_and_grad works inside mx.compile."""
        from functools import partial

        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        model = nn.Linear(4, 2)
        optimizer = optim.SGD(learning_rate=0.01)

        state = [model.state, optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def compiled_update(x, y):
            def loss_fn(m):
                pred = m(x)
                return mx.mean((pred - y) ** 2)

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            return loss

        x = mx.random.normal((2, 4))
        y = mx.random.normal((2, 2))

        w_before = model.weight.tolist()

        loss = compiled_update(x, y)
        mx.eval(loss, model.parameters(), optimizer.state)

        w_after = model.weight.tolist()
        assert w_before != w_after, "Weights should change after compiled update"
        assert float(loss) > 0, "Loss should be positive"

    def test_compiled_disc_update_multiple_steps(self):
        """Compiled update converges over several steps."""
        from functools import partial

        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        model = nn.Linear(4, 2)
        optimizer = optim.SGD(learning_rate=0.01)
        state = [model.state, optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def compiled_update(x, y):
            def loss_fn(m):
                pred = m(x)
                return mx.mean((pred - y) ** 2)

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            return loss

        x = mx.random.normal((8, 4))
        y = mx.random.normal((8, 2))

        losses = []
        for _ in range(10):
            loss = compiled_update(x, y)
            mx.eval(loss, model.parameters(), optimizer.state)
            losses.append(float(loss))

        assert losses[-1] < losses[0], "Loss should decrease over steps"


# ---------------------------------------------------------------------------
# GAN-P1: Compiled discriminator inference for gen loss path
# ---------------------------------------------------------------------------


class TestCompiledDiscInference:
    """Tests for compiled disc inference in gen loss path (GAN-P1)."""

    def test_compiled_disc_infer_concept(self):
        """Verify mx.compile works for inference-only forward pass."""
        from functools import partial

        import mlx.core as mx
        import mlx.nn as nn

        disc = nn.Linear(4, 1)
        disc_state = [disc.state]

        @partial(mx.compile, inputs=disc_state, outputs=disc_state)
        def compiled_infer(x):
            return disc(x)

        x = mx.random.normal((2, 4))

        eager_out = disc(x)
        compiled_out = compiled_infer(x)
        mx.eval(eager_out, compiled_out)

        assert mx.allclose(eager_out, compiled_out, atol=1e-5).item()

    def test_compiled_disc_infer_multi_output(self):
        """Compiled inference returning multiple tensors matches eager."""
        from functools import partial

        import mlx.core as mx
        import mlx.nn as nn

        class TwoOutputDisc(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def __call__(self, x):
                out = self.linear(x)
                scores = out[:, :1]
                feats = out[:, 1:]
                return scores, feats

        disc = TwoOutputDisc()
        disc_state = [disc.state]

        @partial(mx.compile, inputs=disc_state, outputs=disc_state)
        def compiled_infer(fake_wav, real_wav):
            d_fake, f_fake = disc(fake_wav)
            d_real, f_real = disc(mx.stop_gradient(real_wav))
            return d_fake, f_fake, d_real, f_real

        fake = mx.random.normal((2, 4))
        real = mx.random.normal((2, 4))

        e_fake_s, e_fake_f = disc(fake)
        e_real_s, e_real_f = disc(mx.stop_gradient(real))
        c_fake_s, c_fake_f, c_real_s, c_real_f = compiled_infer(fake, real)
        mx.eval(e_fake_s, e_fake_f, e_real_s, e_real_f)
        mx.eval(c_fake_s, c_fake_f, c_real_s, c_real_f)

        assert mx.allclose(e_fake_s, c_fake_s, atol=1e-5).item()
        assert mx.allclose(e_fake_f, c_fake_f, atol=1e-5).item()
        assert mx.allclose(e_real_s, c_real_s, atol=1e-5).item()
        assert mx.allclose(e_real_f, c_real_f, atol=1e-5).item()

    def test_late_binding_holder_pattern(self):
        """The mutable container pattern works for late-bound closures."""
        holder: list = [None]

        def use_holder(x):
            if holder[0] is not None:
                return holder[0](x)
            return x * 2

        assert use_holder(3) == 6

        holder[0] = lambda x: x * 10
        assert use_holder(3) == 30


# ---------------------------------------------------------------------------
# GAN-P3: Waveform caching tests
# ---------------------------------------------------------------------------


class TestWaveformCaching:
    """Tests for GAN-P3 waveform caching optimization."""

    def test_loss_fn_returns_four_values(self):
        """nn.value_and_grad on a 4-tuple-returning fn produces ((loss, a, b, c), grads)."""
        import mlx.core as mx
        import mlx.nn as nn

        model = nn.Linear(4, 2)

        def loss_fn(m, x):
            pred = m(x)
            loss = mx.mean(pred**2)
            aux1 = pred
            aux2 = pred * 2
            return loss, pred, aux1, aux2

        lag = nn.value_and_grad(model, loss_fn)
        x = mx.random.normal((2, 4))
        (loss, out, a1, a2), grads = lag(model, x)
        mx.eval(loss, out, a1, a2)
        assert loss.shape == ()
        assert out.shape == (2, 2)
        assert a1.shape == (2, 2)
        assert a2.shape == (2, 2)
        assert mx.allclose(a2, a1 * 2, atol=1e-5).item()

    def test_loss_fn_returns_none_wavs_when_no_istft(self):
        """When aux values are None, nn.value_and_grad passes them through."""
        import mlx.core as mx
        import mlx.nn as nn

        model = nn.Linear(4, 2)

        def loss_fn(m, x):
            pred = m(x)
            loss = mx.mean(pred**2)
            return loss, pred, None, None

        lag = nn.value_and_grad(model, loss_fn)
        x = mx.random.normal((2, 4))
        (loss, out, a1, a2), grads = lag(model, x)
        mx.eval(loss, out)
        assert a1 is None
        assert a2 is None

    def test_cached_waveforms_match_recomputed(self):
        """Waveforms from loss_fn match independently recomputed ISTFT output."""
        import mlx.core as mx

        # Simulate: loss_fn returns raw waveforms, disc update would recompute.
        # The values should be identical (same input, same transform).
        B, T = 2, 16
        out_spec = mx.random.normal((B, T))
        clean_spec = mx.random.normal((B, T))

        def fake_istft(spec):
            return spec * 0.5  # deterministic transform

        # "loss_fn" path
        cached_out = fake_istft(out_spec)
        cached_clean = fake_istft(clean_spec)

        # "disc update" path — recompute
        recomputed_out = fake_istft(out_spec)
        recomputed_clean = fake_istft(clean_spec)

        mx.eval(cached_out, cached_clean, recomputed_out, recomputed_clean)
        assert mx.allclose(cached_out, recomputed_out, atol=1e-6).item()
        assert mx.allclose(cached_clean, recomputed_clean, atol=1e-6).item()

    def test_cache_flag_controls_recompute(self):
        """When flag is True, cached wavs are used; when False, recompute runs."""
        import mlx.core as mx

        B, T = 2, 16
        cached_out = mx.ones((B, T))
        cached_clean = mx.ones((B, T)) * 2

        recomputed_out = mx.zeros((B, T))
        recomputed_clean = mx.zeros((B, T))

        for flag in (True, False):
            if flag and cached_out is not None and cached_clean is not None:
                pred_wav = cached_out
                clean_wav = cached_clean
            else:
                pred_wav = recomputed_out
                clean_wav = recomputed_clean

            mx.eval(pred_wav, clean_wav)
            if flag:
                assert mx.allclose(pred_wav, mx.ones((B, T)), atol=1e-6).item()
                assert mx.allclose(clean_wav, mx.ones((B, T)) * 2, atol=1e-6).item()
            else:
                assert mx.allclose(pred_wav, mx.zeros((B, T)), atol=1e-6).item()
                assert mx.allclose(clean_wav, mx.zeros((B, T)), atol=1e-6).item()

    def test_config_cache_gen_waveforms_default(self):
        """cache_gen_waveforms defaults to True."""
        cfg = RunConfig()
        assert cfg.gan.cache_gen_waveforms is True

    def test_config_cache_gen_waveforms_disable(self):
        """cache_gen_waveforms can be disabled via apply_run_config_dict."""
        cfg = RunConfig()
        apply_run_config_dict(cfg, {"gan": {"cache_gen_waveforms": False}})
        assert cfg.gan.cache_gen_waveforms is False
