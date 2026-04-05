# Repository Guidelines

**Generated:** 2026-04-05
**Commit:** 387b666
**Branch:** feat/background-music-dataset

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

**Workspace structure:** Root-level monorepo with 6 Rust crates + 1 Python package
- `DeepFilterNet/` is the main Python package (training, inference, configs, scripts). Core code lives in `DeepFilterNet/df/`.
- `DeepFilterNet/tests/` contains Python tests (pytest).
- `libDF/` and `ladspa/` host Rust crates for DSP/runtime and the LADSPA plugin.
- `models/` stores packaged pretrained model archives.
- `docs/`, `assets/`, and `demo/` contain documentation, media, and the demo app.
- `pyDF/` and `pyDF-data/` provide Python bindings and data loading utilities.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| MLX training/inference | `DeepFilterNet/df_mlx/` | **Active dev path** — native Apple Silicon |
| Core Python models | `DeepFilterNet/df/` | Legacy PyTorch implementation |
| DSP/Rust runtime | `libDF/src/` | STFT, ISTFT, data loading |
| LADSPA plugin | `ladspa/src/` | Real-time noise suppression |
| Python bindings | `pyDF/` | `maturin develop` for rebuild |
| Data loader bindings | `pyDF-data/` | Requires HDF5 headers |
| Training scripts | `DeepFilterNet/df/scripts/` | Legacy train.py, prepare_data.py |
| Test data handling | `scripts/datasets/` | Dataset creation utilities |

## CONVENTIONS
- Python: Black (`line-length = 120`), isort, Pyright type checking
- Rust: `cargo fmt` (follows `rustfmt.toml`)
- Commit messages: Conventional Commits (`feat(whisper): ...`)
- Tests: pytest, `test_*.py` files, `mps` marker for Apple Silicon
- **MUST use Beads for non-trivial work** (`bd prime`, issue tracking)
- **MUST push before handoff** (mandatory closeout)

## ANTI-PATTERNS (THIS PROJECT)
- **NEVER** create PRs to `Rikorose/DeepFilterNet` — this is a standalone fork
- **NEVER** use `bd sync` — use `bd dolt ...` commands instead
- **NEVER** stop before pushing — work is NOT complete until `git push` succeeds
- **NEVER** assume PyTorch backend for training — MLX is primary for Apple Silicon
- **NEVER** skip quality gates (tests, linters) before committing

## UNIQUE STYLES
- Dual config system: legacy `.ini` + new TOML `run_config.toml` for df_mlx
- Hardware presets for MLX: `entry`, `pro`, `max`, `ultra`, `debug`
- VAD head with soft gating in df_mlx for speech-aware enhancement
- Beads hook shims in `.githooks/` (not upstream-managed)

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
