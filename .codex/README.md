# DeepFilterNet Codex Setup

This directory contains the repo-local Codex control plane for DeepFilterNet.
It is intentionally narrower than the shared `/Users/andrew/.github` corpus:
the files here are tuned to this repository's Python, MLX, Rust DSP, Beads,
and release workflows.

## What lives here

- `config.toml`: repo-scoped Codex settings that use supported relative-path
  configuration only. The `model_instructions_file` path is relative to
  `.codex/config.toml`, so the sibling entrypoint is `AGENTS.md`.
- `AGENTS.md`: the Codex entrypoint loaded via `model_instructions_file`.
- `TOOLSET.md`: recommended tool groupings and search order for this repo.
- `agents/`: thin role definitions for orchestration and specialist work.
- `instructions/`: reusable operating contracts and language/domain overlays.
- `skills/`: deeper workflows for research, orchestration, training, release,
  and verification.
- `prompts/`: high-value prompt playbooks for audits and regression sweeps.
- `scripts/validate_agentic_setup.py`: consistency checker for internal
  references and config wiring.

## Design rules

1. Root `AGENTS.md` remains authoritative for repo identity and closeout.
2. `.codex/AGENTS.md` supplements the root file and points Codex at the local
   agentic assets.
3. Agents stay thin; skills carry the heavier workflow detail.
4. Codanna-first repository investigation is the default.
5. Beads is the task system for non-trivial work.

## Validation

Run the local validator after editing this setup:

```bash
python3 .codex/scripts/validate_agentic_setup.py
```
