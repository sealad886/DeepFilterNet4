# DeepFilterNet Codex Instructions

This file is the repo-local Codex entrypoint. Read it together with:

- `AGENTS.md` at the repository root for the authoritative repo contract
- `DeepFilterNet/AGENTS.md` when working inside the Python package subtree
- `docs/CONVENTIONS.md` for contributor-facing conventions that agents should
  preserve

## Repository identity

- This repository is `sealad886/DeepFilterNet4`.
- There is no upstream PR flow to `Rikorose/DeepFilterNet`.
- The most actively developed code path is `DeepFilterNet/df_mlx/`.

## Required operating model

1. For work beyond a small edit, define done criteria, keep a compact plan, and
   use multi-agent delegation intentionally.
2. Use Beads for tracking: run `bd prime`, claim or create the issue, keep it
   current, and close it before handoff.
3. Investigate code with Codanna first. Use text search only for exact-literal
   checks, cleanup validation, or when Codanna cannot answer the question.
4. Reuse existing abstractions before adding new modules, helpers, or config
   paths.
5. Treat tests, logs, and command output as evidence. No evidence means the
   work is not done.

## Local control plane

- Core operating contract: `#file:./instructions/codex-instructions.instructions.md`
- Search methodology: `#file:./instructions/codebase-search-methodology.instructions.md`
- Beads workflow: `#file:./instructions/beads.instructions.md`
- Python and MLX overlay: `#file:./instructions/python-mlx.instructions.md`
- Rust DSP overlay: `#file:./instructions/rust-dsp.instructions.md`
- Release closeout overlay: `#file:./instructions/release.instructions.md`
- Tool map: `#file:./TOOLSET.md`

## Default specialist map

- Orchestration: `#file:./agents/dfn-shadow-orchestrator.agent.md`
- Architecture and canonicalization: `#file:./agents/dfn-architect-strategist.agent.md`
- Python and MLX implementation: `#file:./agents/dfn-python-mlx-engineer.agent.md`
- Rust and DSP implementation: `#file:./agents/dfn-rust-dsp-engineer.agent.md`
- Root cause debugging: `#file:./agents/dfn-debugger-problem-solver.agent.md`
- Tests and regressions: `#file:./agents/dfn-test-engineer.agent.md`
- Documentation alignment: `#file:./agents/dfn-documentation-writer.agent.md`
- Release and closeout: `#file:./agents/dfn-release-engineer.agent.md`

## Preferred skills

- `#skill:deepfilternet-multi-agent-orchestration`
- `#skill:deepfilternet-repo-research`
- `#skill:deepfilternet-training-workflows`
- `#skill:deepfilternet-rust-dsp`
- `#skill:deepfilternet-release-readiness`
- `#skill:deepfilternet-verification`

## Quick command anchors

- Python environment: `python3 -m pip install -e ./DeepFilterNet[train,eval]`
- MLX extras: `python3 -m pip install -r DeepFilterNet/requirements_mlx.txt`
- Python tests: `python3 -m pytest` from `DeepFilterNet/`
- Rust build/test: `cargo build`, `cargo test`
- Active MLX training entrypoint: `python3 -m df_mlx.train_dynamic`
- Agentic setup validation: `python3 .codex/scripts/validate_agentic_setup.py`

## Hard stop rule

Do not hand off work until:

- requested scope is complete,
- relevant checks were run or the blocker is explicit,
- Beads state is updated,
- git commit and push completed,
- remaining risks are bounded and evidenced.
