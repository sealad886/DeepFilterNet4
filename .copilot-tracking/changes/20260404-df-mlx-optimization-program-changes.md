<!-- markdownlint-disable-file -->
# df_mlx optimization program changes

## Scope

Implement the approved, measurement-first `df_mlx` optimization program inside MLX/Metal-related paths only.

## Milestone log

### 2026-04-04

- Created the isolated execution worktree at `.worktrees/feat-df-mlx-optimization-program`.
- Added a replacement task implementation scaffold at `/Users/andrew/.github/prompts/task-implementation.instructions.md` because the prompt-required file was missing.
- Created this tracking file for continuous implementation notes.
- Switched baseline setup to the repo-standard project-root `.venv` instead of a Poetry-managed environment.
- Installed the baseline with the repo-native CI-style flow: local `.venv`, `maturin develop -m pyDF/Cargo.toml`, editable `DeepFilterNet`, and MLX requirements.
- Verified the repaired baseline with:
  - `python -m pytest --noconftest DeepFilterNet/tests/test_validation_dataset_isolation.py -q`
  - `cargo test -p deep_filter -q`
- Completed Task 1.1 doc reconciliation for the active `df_mlx` optimization program.
- Corrected the backlog and performance audit so they now reflect the current benchmark surfaces (`benchmark_train_step.py`, `benchmark_pipeline.py`, `benchmark_hotspots.py`) and the approved execution sequence.
- Removed stale assumptions that `benchmark_hotspots.py` still needed to be added, that `StreamingDfNet4.process_frame()` was uncompiled, or that `PrefetchDataLoader` was in scope as an optimization target.
