from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

from df_mlx.run_config import RunConfig


def _collect_parser_flags(module: ast.Module) -> set[str]:
    flags: set[str] = set()
    for node in ast.walk(module):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        option_strings = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-")
        ]
        if not option_strings:
            continue
        long_flags = [flag for flag in option_strings if flag.startswith("--")]
        flags.add(long_flags[0] if long_flags else option_strings[0])
    return flags


def _help_block(help_text: str, flag: str) -> str:
    options_idx = help_text.find("\noptions:\n")
    search_start = options_idx if options_idx >= 0 else 0

    idx = help_text.find(flag, search_start)
    if idx == -1:
        return ""

    line_start = help_text.rfind("\n", search_start, idx)
    start = 0 if line_start == -1 else line_start + 1

    next_option_idx = help_text.find("\n  -", idx + len(flag))
    end = next_option_idx if next_option_idx != -1 else len(help_text)
    return help_text[start:end]


def _get_help_text() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "df_mlx.train_dynamic", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_all_runtime_config_flags_show_default_in_help() -> None:
    train_dynamic = Path(__file__).resolve().parents[1] / "df_mlx" / "train_dynamic.py"
    module = ast.parse(train_dynamic.read_text(encoding="utf-8"))
    parser_flags = _collect_parser_flags(module)

    meta_only_flags = {"--run-config", "--print-run-config", "--preset"}
    runtime_flags = sorted(parser_flags - meta_only_flags)

    help_text = _get_help_text()
    missing: list[str] = []
    for flag in runtime_flags:
        block = _help_block(help_text, flag)
        if not block:
            missing.append(f"{flag} (missing from help)")
            continue
        if "(default:" not in block:
            missing.append(f"{flag} (missing default)")

    assert missing == [], "Help output missing defaults for runtime config flags:\n" + "\n".join(missing)


def test_help_defaults_match_run_config_for_key_overrides() -> None:
    cfg = RunConfig()
    help_text = _get_help_text()

    expected_by_flag = {
        "--epochs": cfg.training.epochs,
        "--mrstft-factor": cfg.loss.mrstft.factor,
        "--mrstft-gamma": cfg.loss.mrstft.gamma,
        "--mrstft-fft-sizes": list(cfg.loss.mrstft.fft_sizes),
        "--gan-mpd-periods": list(cfg.gan.mpd_periods),
        "--dynamic-loss": cfg.loss.dynamic_loss,
    }

    for flag, expected_default in expected_by_flag.items():
        block = _help_block(help_text, flag)
        assert block, f"Could not find {flag} in help output"
        pattern = rf"default:\s*{re.escape(str(expected_default))}"
        assert re.search(pattern, block), f"{flag} help should show default {expected_default!r}; block was: {block!r}"
