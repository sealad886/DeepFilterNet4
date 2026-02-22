---
name: beads
description: Git-backed issue tracking with bd CLI. Use when you see .beads/ in a repository, or when tasks involve multi-session work, dependency management, or project planning that must survive context loss.
---

# Beads Issue Tracking

If `.beads/` exists → use `bd` for task tracking.

## Quick Start

```bash
bd prime              # Get AI-optimized context (run first!)
bd ready --json       # Find unblocked work
bd update <id> --status in_progress --json  # Claim work
bd close <id> --reason "..." --json         # Complete work
bd sync               # Persist to git (ALWAYS run at session end)
```

## Session Protocol

**Start**: `bd prime` → `bd ready` → claim work with `bd update`

**During**: Create discoveries with `--deps discovered-from:<parent-id>`, update notes

**End** (MANDATORY):
```bash
bd sync && git push   # Work is NOT complete until pushed
```

## bd vs TodoWrite

| bd | TodoWrite |
|----|-----------|
| Multi-session, survives compaction | Single-session only |
| Complex dependencies | Linear checklist |
| Git-backed persistence | Conversation-scoped |

**Test**: "Will I need this context in 2 weeks?" → YES = use bd

## Core Commands

| Command | Purpose |
|---------|---------|
| `bd create "Title" -t task -p 2 --json` | Create issue |
| `bd show <id> --json` | View issue details |
| `bd update <id> --status in_progress --json` | Claim/update |
| `bd close <id> --reason "..." --json` | Complete |
| `bd dep add <blocker> <blocked>` | Add dependency |
| `bd list --status open --json` | Search issues |

## Dependency Direction (Critical)

`bd dep add A B` means **A blocks B** (A must complete before B can start).

```bash
# "Setup blocks Implementation" (setup first)
bd dep add setup-id impl-id    # ✓ CORRECT
```

## Discovery Pattern

When you find related work during a task:

```bash
bd create "Found: auth bug" -t bug -p 1 --deps discovered-from:bd-42 --json
```

## Priorities

| P | Name | Use |
|---|------|-----|
| 0 | Critical | Production down, security |
| 1 | High | Major features, significant bugs |
| 2 | Medium | Standard work (default) |
| 3 | Low | Polish, nice-to-have |
| 4 | Backlog | Future ideas |

## Issue Types

`bug`, `feature`, `task`, `epic`, `chore`

## References

For detailed documentation, see:

- [REFERENCE.md](references/REFERENCE.md) - CLI syntax, workflows, patterns, dependencies
- [ADVANCED.md](references/ADVANCED.md) - Molecules, chemistry, multi-agent coordination, worktrees
- [TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) - Error handling and recovery
