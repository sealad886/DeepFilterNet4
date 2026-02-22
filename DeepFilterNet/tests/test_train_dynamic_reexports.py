"""Verify all training_* module public symbols are re-exported from train_dynamic.

This test prevents drift between the canonical training_* modules and the
backward-compatibility re-export shim in train_dynamic.py.
"""

import importlib
import inspect

_TRAINING_MODULES = [
    "df_mlx.training_losses",
    "df_mlx.training_checkpoints",
    "df_mlx.training_cli",
    "df_mlx.training_cli_main",
    "df_mlx.training_ops",
    "df_mlx.training_session",
    "df_mlx.training_signals",
    "df_mlx.training_waveform",
]

_SKIP_NAMES = frozenset(
    {
        "__annotations__",
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
        "__path__",
        "__all__",
    }
)


def _public_symbols(mod):
    """Return symbols defined in a module (not imported from other df_mlx or third-party modules).

    Skips symbols starting with underscore unless they are listed in __all__.
    """
    mod_name = mod.__name__
    mod_all = set(getattr(mod, "__all__", []))
    symbols = set()
    for name in dir(mod):
        if name in _SKIP_NAMES:
            continue
        # Skip underscore-prefixed names unless they're in __all__
        if name.startswith("_") and name not in mod_all:
            continue
        obj = getattr(mod, name)
        if inspect.ismodule(obj):
            continue
        # Only include symbols whose defining module IS this module
        defining_mod = getattr(obj, "__module__", None)
        if defining_mod and defining_mod != mod_name:
            continue
        symbols.add(name)
    return symbols


def test_reexport_completeness():
    """All training_* public symbols must be re-exported from train_dynamic."""
    import df_mlx.train_dynamic as td

    missing = []
    for mod_name in _TRAINING_MODULES:
        mod = importlib.import_module(mod_name)
        for name in _public_symbols(mod):
            if not hasattr(td, name):
                missing.append(f"{mod_name}.{name}")

    assert missing == [], "Symbols not re-exported from train_dynamic:\n" + "\n".join(
        f"  - {m}" for m in sorted(missing)
    )


def test_reexported_symbols_are_same_objects():
    """Re-exported symbols must be the exact same objects (not copies)."""
    import df_mlx.train_dynamic as td

    mismatches = []
    for mod_name in _TRAINING_MODULES:
        mod = importlib.import_module(mod_name)
        for name in _public_symbols(mod):
            if not hasattr(td, name):
                continue
            orig = getattr(mod, name)
            reex = getattr(td, name)
            if orig is not reex:
                mismatches.append(f"{mod_name}.{name}")

    assert mismatches == [], "Re-exported symbols differ from originals:\n" + "\n".join(
        f"  - {m}" for m in sorted(mismatches)
    )


def test_hardware_diagnostics_reexported():
    """print_hardware_diagnostics moved to hardware.py must be re-exported."""
    from df_mlx.hardware import print_hardware_diagnostics as orig
    from df_mlx.train_dynamic import print_hardware_diagnostics

    assert print_hardware_diagnostics is orig
