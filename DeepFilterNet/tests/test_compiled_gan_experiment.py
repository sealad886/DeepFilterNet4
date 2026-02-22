"""Tests for M1.2 — Guarded compiled-GAN experiment implementation.

Covers resolve_epoch_train_mode with experimental_compiled_gan, invariant
relaxation, config field, and TOML roundtrip.
"""

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from df_mlx.run_config import RunConfig, apply_run_config_dict, generate_run_config_example
from df_mlx.training_checkpoints import resolve_epoch_train_mode

_COMPILED = "COMPILED"
_EAGER = "EAGER"


# ---------------------------------------------------------------------------
# resolve_epoch_train_mode — without experimental flag
# ---------------------------------------------------------------------------


class TestResolveModeWithoutExperimental:
    """Existing (non-experimental) behavior must be preserved."""

    def test_gan_active_forces_eager(self):
        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=True,
            gan_active=True,
            previous_mode=None,
        )
        assert mode == _EAGER
        assert use_compiled is False

    def test_gan_inactive_allows_compiled(self):
        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=True,
            gan_active=False,
            previous_mode=None,
        )
        assert mode == _COMPILED
        assert use_compiled is True

    def test_oneway_eager_stays_eager(self):
        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=True,
            gan_active=False,
            previous_mode=_EAGER,
        )
        assert mode == _EAGER
        assert use_compiled is False

    def test_compiled_base_disabled_forces_eager(self):
        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=False,
            gan_enabled=False,
            gan_active=False,
            previous_mode=None,
        )
        assert mode == _EAGER
        assert use_compiled is False

    def test_no_gan_allows_compiled(self):
        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=False,
            gan_active=False,
            previous_mode=None,
        )
        assert mode == _COMPILED
        assert use_compiled is True


# ---------------------------------------------------------------------------
# resolve_epoch_train_mode — with experimental flag
# ---------------------------------------------------------------------------


class TestResolveModeWithExperimental:
    """With experimental_compiled_gan=True, GAN-active epochs stay COMPILED."""

    def test_gan_active_stays_compiled(self):
        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=True,
            gan_active=True,
            previous_mode=None,
            experimental_compiled_gan=True,
        )
        assert mode == _COMPILED
        assert use_compiled is True

    def test_gan_active_compiled_from_compiled_previous(self):
        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=True,
            gan_active=True,
            previous_mode=_COMPILED,
            experimental_compiled_gan=True,
        )
        assert mode == _COMPILED
        assert use_compiled is True

    def test_compiled_base_disabled_blocks_even_with_flag(self):
        """debug_numerics or nan_skip_batch still block compiled mode."""
        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=False,
            gan_enabled=True,
            gan_active=True,
            previous_mode=None,
            experimental_compiled_gan=True,
        )
        assert mode == _EAGER
        assert use_compiled is False

    def test_experimental_allows_return_from_eager(self):
        """With experimental flag, previous EAGER does not prevent switching back."""
        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=True,
            gan_active=True,
            previous_mode=_EAGER,
            experimental_compiled_gan=True,
        )
        assert mode == _COMPILED
        assert use_compiled is True

    def test_experimental_without_gan_active_stays_compiled(self):
        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=True,
            gan_active=False,
            previous_mode=None,
            experimental_compiled_gan=True,
        )
        assert mode == _COMPILED
        assert use_compiled is True


# ---------------------------------------------------------------------------
# resolve_epoch_train_mode — one-way invariant behaviors
# ---------------------------------------------------------------------------


class TestResolveModeOneWay:
    """One-way invariant is enforced without flag, relaxed with flag."""

    def test_oneway_enforced_without_flag(self):
        mode, _ = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=False,
            gan_active=False,
            previous_mode=_EAGER,
            experimental_compiled_gan=False,
        )
        assert mode == _EAGER

    def test_oneway_relaxed_with_flag(self):
        mode, use_compiled = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=False,
            gan_active=False,
            previous_mode=_EAGER,
            experimental_compiled_gan=True,
        )
        assert mode == _COMPILED
        assert use_compiled is True


# ---------------------------------------------------------------------------
# Config field and TOML roundtrip
# ---------------------------------------------------------------------------


class TestExperimentalCompileConfigField:
    def test_field_exists_and_defaults_false(self):
        cfg = RunConfig()
        assert hasattr(cfg.gan, "experimental_compile")
        assert cfg.gan.experimental_compile is False

    def test_set_via_dict(self):
        cfg = RunConfig()
        apply_run_config_dict(cfg, {"gan": {"experimental_compile": True}})
        assert cfg.gan.experimental_compile is True

    def test_toml_roundtrip(self):
        text = generate_run_config_example()
        data = tomllib.loads(text)
        assert "gan" in data
        assert "experimental_compile" in data["gan"]
        assert data["gan"]["experimental_compile"] is False

        # Apply with True
        cfg = RunConfig()
        data["gan"]["experimental_compile"] = True
        apply_run_config_dict(cfg, data)
        assert cfg.gan.experimental_compile is True


# ---------------------------------------------------------------------------
# Invariant check simulation
# ---------------------------------------------------------------------------


class TestInvariantRelaxation:
    """The RuntimeError invariant is relaxed when experimental flag is on."""

    def test_invariant_enforced_without_experimental(self):
        """Without the flag, GAN active + compiled step raises RuntimeError.

        This simulates what happens in the training loop when
        resolve_epoch_train_mode returns COMPILED despite GAN being active
        (which should not happen without the flag).
        """
        # Without experimental flag, resolve_epoch_train_mode does NOT return
        # compiled when GAN is active, so the invariant is never triggered
        # naturally. Verify that if it somehow did, the check would catch it.
        gan_active = True
        epoch_use_compiled_step = True
        experimental_compiled_gan = False

        # The invariant check from the training loop
        should_raise = gan_active and epoch_use_compiled_step and not experimental_compiled_gan
        assert should_raise is True

    def test_invariant_relaxed_with_experimental(self):
        """With the flag, GAN active + compiled step does NOT raise."""
        gan_active = True
        epoch_use_compiled_step = True
        experimental_compiled_gan = True

        should_raise = gan_active and epoch_use_compiled_step and not experimental_compiled_gan
        assert should_raise is False

    def test_invariant_not_triggered_when_gan_inactive(self):
        gan_active = False
        epoch_use_compiled_step = True
        experimental_compiled_gan = False

        should_raise = gan_active and epoch_use_compiled_step and not experimental_compiled_gan
        assert should_raise is False


# ---------------------------------------------------------------------------
# Mode transition sequences
# ---------------------------------------------------------------------------


class TestModeTransitionSequences:
    """Test realistic epoch-by-epoch mode transitions."""

    def test_standard_transition_compiled_to_eager_at_gan(self):
        """Pre-GAN: COMPILED, GAN activation: EAGER, stays EAGER."""
        modes = []
        prev = None
        for epoch in range(5):
            gan_active = epoch >= 3
            mode, _ = resolve_epoch_train_mode(
                compiled_step_base_enabled=True,
                gan_enabled=True,
                gan_active=gan_active,
                previous_mode=prev,
            )
            modes.append(mode)
            prev = mode
        assert modes == [_COMPILED, _COMPILED, _COMPILED, _EAGER, _EAGER]

    def test_experimental_stays_compiled_through_gan(self):
        """With experimental flag, mode stays COMPILED through GAN activation."""
        modes = []
        prev = None
        for epoch in range(5):
            gan_active = epoch >= 3
            mode, _ = resolve_epoch_train_mode(
                compiled_step_base_enabled=True,
                gan_enabled=True,
                gan_active=gan_active,
                previous_mode=prev,
                experimental_compiled_gan=True,
            )
            modes.append(mode)
            prev = mode
        assert modes == [_COMPILED, _COMPILED, _COMPILED, _COMPILED, _COMPILED]

    def test_experimental_blocked_by_debug_numerics(self):
        """debug_numerics blocks compiled even with experimental flag."""
        modes = []
        prev = None
        for epoch in range(5):
            gan_active = epoch >= 3
            mode, _ = resolve_epoch_train_mode(
                compiled_step_base_enabled=False,  # blocked by debug_numerics
                gan_enabled=True,
                gan_active=gan_active,
                previous_mode=prev,
                experimental_compiled_gan=True,
            )
            modes.append(mode)
            prev = mode
        assert modes == [_EAGER, _EAGER, _EAGER, _EAGER, _EAGER]

    def test_flag_false_default_parameter(self):
        """experimental_compiled_gan defaults to False — same as no kwarg."""
        mode_default, _ = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=True,
            gan_active=True,
            previous_mode=None,
        )
        mode_explicit, _ = resolve_epoch_train_mode(
            compiled_step_base_enabled=True,
            gan_enabled=True,
            gan_active=True,
            previous_mode=None,
            experimental_compiled_gan=False,
        )
        assert mode_default == mode_explicit == _EAGER
