---
name: DeepFilterNet4 Repository Instructions
description: Custom GitHub Copilot instructions for the DeepFilterNet4 repository.
applyTo: "**"
---
# GitHub Copilot Workspace Instructions

## Repository Identity

- **This is sealad886/DeepFilterNet** — a standalone fork
- **There is NO upstream repository relationship**
- **NEVER create PRs to or reference Rikorose/DeepFilterNet**
- All work stays within this repository only

## Issue Tracking

If `.beads/` exists in the project root, use the `bd` CLI for issue tracking. The beads skill (`.github/skills/beads/`) provides
detailed guidance in a structured progressive disclosure format.

Quick commands: `bd prime` (get context), `bd ready` (find work), `bd sync` (save to git)

## Project Structure

- `DeepFilterNet/` - Main Python package (training, inference, configs)
- `DeepFilterNet/df/` - Core Python code
- `DeepFilterNet/tests/` - Python tests (pytest)
- `libDF/`, `ladspa/` - Rust crates for DSP/runtime and LADSPA plugin
- `models/` - Pretrained model archives
- `pyDF/`, `pyDF-data/` - Python bindings and data utilities

## Build & Test

- Python: `poetry -C DeepFilterNet install`, `pytest` (from `DeepFilterNet/`)
- Rust: `cargo build`, `cargo test`

## Coding Style

- Python: Black (`line-length = 100`), isort, Pyright for type checking
- Rust: `cargo fmt` (follows `rustfmt.toml`)
- Tests: pytest, name test files `test_*.py`

## Commit Guidelines

- Follow Conventional Commits style (e.g., `feat(whisper): ...`, `chore(lint): ...`)
- Always push changes before ending a session
