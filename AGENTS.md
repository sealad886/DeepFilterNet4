# Repository Guidelines

## ⚠️ CRITICAL: Repository Identity
- **This is sealad886/DeepFilterNet4** — a standalone fork
- **There is NO upstream repository relationship**
- **NEVER create PRs to or reference Rikorose/DeepFilterNet**
- All work stays within this repository only

## Issue Tracking & AI Integration

This project uses **bd (beads)** for issue tracking with full AI integration.

### For AI Agents
- **Skill location:** `.claude/skills/beads/` — comprehensive bd integration patterns
- **Run `bd prime` at session start** to inject workflow context
- **Git hooks auto-inject** context on commits if installed
- Consult `SKILL.md` for decision trees, `resources/` for specific patterns

### Quick Reference

| Command | Purpose |
|---------|---------|
| `bd ready` | Find unblocked work to claim |
| `bd create "Title" --type task --priority 2` | Create new issue |
| `bd update <id> --status in-progress` | Claim work |
| `bd close <id>` | Complete work |
| `bd sync` | Sync with git (run at session end) |
| `bd prime` | Get full workflow context |

### Session Workflow
1. **Start:** `bd prime` → `bd ready` → claim work
2. **During:** Reference issues in commits, update status as you go
3. **End:** `bd sync` → `git push` (work isn't complete until pushed!)

## Project Structure & Module Organization
- `DeepFilterNet/` is the main Python package (training, inference, configs, scripts). Core code lives in `DeepFilterNet/df/`.
- `DeepFilterNet/tests/` contains Python tests (pytest).
- `libDF/` and `ladspa/` host Rust crates for DSP/runtime and the LADSPA plugin.
- `models/` stores packaged pretrained model archives.
- `docs/`, `assets/`, and `demo/` contain documentation, media, and the demo app.
- `pyDF/` and `pyDF-data/` provide Python bindings and data loading utilities.

## Build, Test, and Development Commands
- `poetry -C DeepFilterNet install` — install Python deps for the DeepFilterNet package.
- `poetry -C DeepFilterNet lock --regenerate` — refresh `DeepFilterNet/poetry.lock`.
- `python -m pytest` (run inside `DeepFilterNet/`) — execute Python tests in `DeepFilterNet/tests/`.
- `python df/train.py --model-type dfnet4 ...` (run inside `DeepFilterNet/`) — train DFNet4 models.
- `cargo build` / `cargo test` — build and test Rust crates from repo root.
- `cargo +nightly run -p df-demo --features ui --bin df-demo --release` — run the UI demo (Linux).

## Coding Style & Naming Conventions
- Python is formatted with Black (`line-length = 100`) and imports organized with isort (see `pyproject.toml`).
- Use `snake_case` for functions/variables, `PascalCase` for classes, and keep module names lowercase.
- Rust formatting follows `rustfmt.toml`; prefer `cargo fmt` before commits.
- Type checking uses Pyright (`pyrightconfig.json`).

## Testing Guidelines
- Primary framework: pytest (`DeepFilterNet/tests/`).
- Name new tests `test_*.py` and place in the closest relevant module folder.
- Use the `mps` marker for Apple Silicon–specific tests (see `DeepFilterNet/pyproject.toml`).

## Commit & Pull Request Guidelines
- Commit messages follow a Conventional Commits style (e.g., `feat(whisper): ...`, `chore(lint): ...`).
- PRs should include a concise summary, tests run (or why not), and any model/data changes.
- If a change affects checkpoints or configs, mention the expected model directory layout (`config.ini` + `checkpoints/`).
