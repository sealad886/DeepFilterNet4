---
name: deepfilternet-python-mlx
description: Python and MLX implementation overlay for DeepFilterNet.
applyTo: "DeepFilterNet/**/*.py"
---

# Python and MLX Overlay

This repository's most active path is `DeepFilterNet/df_mlx/`.

## Before editing

1. Read the relevant `df_mlx` module plus adjacent tests.
2. Read any matching docs in `docs/` when the change affects losses, presets,
   run config, checkpoints, or VAD behavior.
3. Confirm whether the task is MLX-native, PyTorch-path, or shared utility work.

## Current training entrypoints

- Supported CLI module: `python -m df_mlx.train_dynamic`
- Run config precedence:
  `defaults < preset < legacy train-config INI compatibility < run-config TOML < CLI flags`

## Validation anchors

- Environment: `python3 -m pip install -e ./DeepFilterNet[train,eval]` plus `python3 -m pip install -r DeepFilterNet/requirements_mlx.txt`
- Tests: `python3 -m pytest` from `DeepFilterNet/`
- Targeted MLX path checks should stay close to changed modules and docs.

## Preferred references

- `README.md`
- `DeepFilterNet/df_mlx/README.md`
- `docs/RUN_CONFIG_PRESETS.md`
- `docs/LOSSES.md`
