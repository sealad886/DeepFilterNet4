# Repository Guidelines

## ⚠️ CRITICAL: Repository Identity
- **This is sealad886/DeepFilterNet4** — a standalone fork
- **There is NO upstream repository relationship**
- **NEVER create PRs to or reference Rikorose/DeepFilterNet**
- All work stays within this repository only

## Issue Tracking

This project uses **bd (beads)** for issue tracking.
Run `bd prime` for workflow context. This repo keeps the Beads hook shims in
`.githooks/`, so enable them with `./setup.sh` or `scripts/install-hooks.sh`.
Use `bd hooks install` only if you intentionally want upstream-managed hooks
instead of the repo-managed shims.

**Quick reference:**
- `bd ready` - Find unblocked work
- `bd create "Title" --type task --priority 2` - Create issue
- `bd close <id>` - Complete work
- `bd dolt remote list` - Check whether a Dolt remote is configured
- `bd dolt push` - Push beads to remote **when a Dolt remote is configured**

For full workflow details: `bd prime`

## Project Structure & Module Organization
- `DeepFilterNet/` is the main Python package (training, inference, configs, scripts). Core code lives in `DeepFilterNet/df/`.
- `DeepFilterNet/tests/` contains Python tests (pytest).
- `libDF/` and `ladspa/` host Rust crates for DSP/runtime and the LADSPA plugin.
- `models/` stores packaged pretrained model archives.
- `docs/`, `assets/`, and `demo/` contain documentation, media, and the demo app.
- `pyDF/` and `pyDF-data/` provide Python bindings and data loading utilities.

## Build, Test, and Development Commands
- `python3 -m pip install -e ./DeepFilterNet[train,eval]` — install the DeepFilterNet package plus training/eval deps in the active environment.
- `python3 -m pip install -r DeepFilterNet/requirements_mlx.txt` — install MLX-specific extras for `df_mlx` work on Apple Silicon.
- `python3 -m pytest` (run inside `DeepFilterNet/`) — execute Python tests in `DeepFilterNet/tests/`.
- `python3 df/train.py --model-type dfnet4 ...` (run inside `DeepFilterNet/`) — train DFNet4 models.
- `cargo build` / `cargo test` — build and test Rust crates from repo root.
- `cargo +nightly run -p df-demo --features ui --bin df-demo --release` — run the UI demo (Linux).

## Coding Style & Naming Conventions
- Python is formatted with Black (`line-length = 120`) and imports organized with isort (see `pyproject.toml`).
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

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt status
   bd dolt remote list
   # If a Dolt remote is configured, then:
   bd dolt pull
   bd dolt commit -m "Sync beads state"   # when there are pending Dolt changes
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
- Do **not** run `bd sync`; the current local CLI uses `bd dolt ...` commands instead
