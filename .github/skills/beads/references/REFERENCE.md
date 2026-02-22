# Beads Reference Guide

Comprehensive reference for bd CLI, workflows, patterns, and dependencies.

---

## CLI Command Reference

### Issue Management

```bash
# Create issues
bd create "Title" -t bug|feature|task|epic|chore -p 0-4 -d "Description" --json
bd create "Title" --id custom-id -p 1 --json           # Explicit ID
bd create "Title" -t bug -p 1 -l bug,critical --json   # With labels
bd create "Title" --deps discovered-from:bd-42 --json  # Link discovery

# View issues
bd show <id> --json                    # Single issue details
bd show <id> <id> <id> --json          # Multiple issues
bd dep tree <id>                       # Show dependency tree

# Update issues
bd update <id> --status in_progress --json
bd update <id> --priority 1 --json
bd update <id> --notes "Progress update" --json
bd update <id> --design "Architecture decisions" --json

# Close/reopen
bd close <id> --reason "Completed" --json
bd close <id> <id> <id> --reason "Batch complete" --json
bd reopen <id> --reason "Need more work" --json
```

### Finding Work

```bash
# Ready work (no blockers)
bd ready --json

# Stale issues
bd stale --days 30 --json
bd stale --days 90 --status in_progress --json

# List with filters
bd list --status open --json
bd list --status open --priority 1 --json
bd list --type bug --json
bd list --assignee alice --json
bd list --label bug,critical --json        # AND: has ALL labels
bd list --label-any frontend,backend --json # OR: has ANY label

# Text search
bd list --title "auth" --json
bd list --title-contains "auth" --json
bd list --desc-contains "implement" --json
bd list --notes-contains "TODO" --json

# Date filters
bd list --created-after 2024-01-01 --json
bd list --updated-after 2024-06-01 --json

# Special filters
bd list --empty-description --json
bd list --no-assignee --json
bd list --no-labels --json
bd list --priority-min 0 --priority-max 1 --json
```

### Dependencies & Labels

```bash
# Dependencies
bd dep add <from-id> <to-id>                    # Default: blocks
bd dep add <from-id> <to-id> --type blocks      # Explicit blocks
bd dep add <from-id> <to-id> --type related     # Soft link
bd dep add <from-id> <to-id> --type parent-child
bd dep add <from-id> <to-id> --type discovered-from
bd dep remove <from-id> <to-id>
bd dep tree <id>

# Labels
bd label add <id> <label> --json
bd label remove <id> <label> --json
bd label list <id> --json
bd label list-all --json
```

### Database & Sync

```bash
# Sync to git
bd sync

# Import/export
bd import -i .beads/issues.jsonl
bd import -i .beads/issues.jsonl --force       # Force metadata update
bd import -i .beads/issues.jsonl --dry-run     # Preview

# Database info
bd info --json

# Daemon management
bd daemons list --json
bd daemons health --json
bd daemons restart /path/to/workspace --json
bd daemons killall --json
```

### Global Flags

```bash
bd --json <command>           # JSON output
bd --sandbox <command>        # Sandbox mode (no daemon)
bd --allow-stale <command>    # Skip staleness check
bd --no-daemon <command>      # Direct SQLite mode
bd --db /path/to/db <command> # Custom database
bd --actor alice <command>    # Custom audit actor
```

---

## Dependency Types

### blocks (Hard Blocker)

**Semantics**: A blocks B means B cannot start until A completes.

**Affects `bd ready`**: YES - blocked issues are excluded.

```bash
bd dep add prerequisite-id blocked-id
# prerequisite-id must close before blocked-id appears in bd ready
```

**Use for**: Prerequisites, sequential steps, build order.

### related (Soft Link)

**Semantics**: Issues are related but don't block each other.

**Affects `bd ready`**: NO - informational only.

```bash
bd dep add issue-1 issue-2 --type related
```

**Use for**: Similar work, shared context, alternative approaches.

### parent-child (Hierarchy)

**Semantics**: A is parent of B (epic contains subtask).

**Affects `bd ready`**: NO - structural only.

```bash
bd dep add child-id parent-epic-id --type parent-child
```

**Use for**: Epics, work breakdown, hierarchical organization.

### discovered-from (Provenance)

**Semantics**: B was discovered while working on A.

**Affects `bd ready`**: NO - tracks origin.

```bash
bd dep add original-work-id discovered-id --type discovered-from
# Or at creation time:
bd create "Found bug" --deps discovered-from:original-id --json
```

**Use for**: Side quests, research findings, bug discoveries.

---

## Workflow Patterns

### Daily Workflow

```bash
# Morning
bd ready --json                          # What's available?
bd list --status in_progress --json      # What did I start?
bd update bd-42 --status in_progress     # Claim work

# During work
bd create "Found: bug" -t bug -p 1 --deps discovered-from:bd-42 --json
bd update bd-42 --notes "Progress: completed step 1" --json

# End of day
bd update bd-42 --notes "EOD: steps 1-3 done, tomorrow: step 4" --json
bd sync
```

### Epic Decomposition

```bash
# Create epic
bd create "Epic: User auth" -t epic -p 1 --json  # bd-100

# Create subtasks
bd create "DB schema" -t task -p 1 --json        # bd-101
bd create "Endpoints" -t task -p 1 --json        # bd-102
bd create "Tests" -t task -p 2 --json            # bd-103

# Set hierarchy
bd dep add bd-100 bd-101 --type parent-child
bd dep add bd-100 bd-102 --type parent-child
bd dep add bd-100 bd-103 --type parent-child

# Set sequence
bd dep add bd-101 bd-102   # Schema before endpoints
bd dep add bd-102 bd-103   # Endpoints before tests
```

### Bug Investigation

```bash
# Create bug
bd create "Bug: login fails" -t bug -p 1 \
  -d "Steps: 1. Go to login 2. Enter creds 3. Fails" --json

# Update with findings
bd update bd-bug --design "Root cause: race condition in session" --json

# Create fix tasks if needed
bd create "Fix race condition" -t task -p 1 --json
bd dep add bd-bug bd-fix --type parent-child
```

### Side Quest Handling

```bash
# Working on bd-42, discover related issue
bd create "Bug: auth missing validation" -t bug -p 1 \
  --deps discovered-from:bd-42 --json

# If it blocks current work:
bd dep add bd-new bd-42   # New issue blocks current
# bd-42 leaves ready queue until bd-new closes

# If independent:
# Continue on bd-42, bd-new queued for later
```

### Research Pattern

```bash
# Create research task
bd create "Research: caching options" -t task -p 2 \
  -d "Question: Redis vs Memcached
      Time box: 4 hours
      Deliverable: Recommendation" --json

# Capture findings
bd create "Finding: Redis supports persistence" \
  --deps discovered-from:bd-research --json
bd create "Finding: Memcached simpler" \
  --deps discovered-from:bd-research --json

# Create decision
bd create "Decision: Use Redis" \
  -d "Based on findings, selecting Redis for persistence" \
  --deps discovered-from:bd-research --json
```

---

## Session Protocols

### Session Start

```bash
bd prime                                    # Get AI context (if available)
bd ready --json                             # Find unblocked work
bd list --status in_progress --json         # Check ongoing work
bd update bd-42 --status in_progress --json # Claim issue
bd show bd-42 --json                        # Read full context
```

### Session End (MANDATORY)

```bash
# 1. Update current issue
bd update bd-42 --notes "Session end: [progress summary]" --json

# 2. Create discovered issues
bd create "TODO: Found during work" --deps discovered-from:bd-42 --json

# 3. Close if complete
bd close bd-42 --reason "Completed and tested" --json

# 4. ALWAYS sync and push
bd sync
git push

# 5. Verify
git status  # Must show "up to date with origin"
```

### Resumability

When returning after time away:

```bash
# Recover context
bd ready --json                              # What's available
bd list --status in_progress --json          # What was I doing?
bd show bd-42 --json                         # Full context of issue
bd list --updated-after $(date -v-7d +%Y-%m-%d) --json  # Recent activity
```

Write notes as if explaining to a future session with zero context:

```bash
bd update bd-42 --notes "
COMPLETED: JWT auth with RS256
KEY DECISION: RS256 over HS256 for key rotation
IN PROGRESS: Password reset flow (step 2 of 4)
NEXT: Implement rate limiting
BLOCKED BY: None
" --json
```

---

## Static Reference

### Issue Statuses

| Status | In `bd ready`? | Meaning |
|--------|----------------|---------|
| `open` | Yes (if unblocked) | Not started |
| `in_progress` | No | Being worked on |
| `closed` | No | Completed |

### Priority Levels

| P | Name | Response | Examples |
|---|------|----------|----------|
| 0 | Critical | Immediate | Security, production down |
| 1 | High | Same day | Major features, significant bugs |
| 2 | Medium | This week | Standard work |
| 3 | Low | When possible | Polish, optimization |
| 4 | Backlog | Future | Ideas, speculative |

### Issue Types

| Type | Use |
|------|-----|
| `bug` | Defects, errors |
| `feature` | New functionality |
| `task` | Tests, docs, research |
| `epic` | Container for related issues |
| `chore` | Maintenance, cleanup |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `BD_DATABASE` | Database path (default: `.beads/beads.db`) |
| `BD_PREFIX` | Issue ID prefix (default: `bd-`) |
| `BD_NO_DAEMON` | Disable daemon |
| `BD_NO_AUTO_FLUSH` | Disable auto-export |
| `BD_NO_AUTO_IMPORT` | Disable auto-import |

### File Locations

| File | Purpose |
|------|---------|
| `.beads/beads.db` | SQLite database |
| `.beads/issues.jsonl` | JSONL export (git-tracked) |
| `.beads/config.toml` | Local configuration |

---

## bd vs TodoWrite Decision Matrix

### Use bd when:

- **Multi-session work**: Spans multiple days or compaction cycles
- **Complex dependencies**: Blockers, prerequisites, hierarchy
- **Knowledge work**: Fuzzy boundaries, exploration, decisions
- **Side quests**: Work that might pause main task
- **Project memory**: Need to resume after weeks away

### Use TodoWrite when:

- **Single-session tasks**: Completes within conversation
- **Linear execution**: Step-by-step, no branching
- **Immediate context**: All info in current conversation
- **Simple tracking**: Just need visible checklist

### Decision Test

**"Will I need this context in 2 weeks?"**

- YES → use bd
- NO → TodoWrite is fine

### Integration Pattern

Use both strategically:

- **bd**: High-level issues and dependencies (strategic)
- **TodoWrite**: Current session's execution steps (tactical)

```
bd issue: "Implement authentication" (epic)
  └─ Currently working on: "Create login endpoint"

TodoWrite (for login endpoint):
- [ ] Install JWT library
- [ ] Create token middleware
- [ ] Add tests
```

---

## Best Practices

### Issue Creation

1. **Atomic issues**: One coherent piece of work per issue
2. **Actionable titles**: "Fix null pointer in UserService.getProfile()"
3. **Clear scope**: Define what's in and out of scope
4. **Testable acceptance criteria**: Specific, verifiable conditions

### Dependency Management

1. **Only use `blocks` for true blockers**: Not preferences
2. **Keep graphs shallow**: Avoid deep dependency chains
3. **Check before adding**: `bd dep tree <id>` to understand existing structure

### Context Preservation

1. **Capture decisions**: Use `--design` for architectural choices
2. **Track progress**: Update notes regularly
3. **Link everything**: Use `discovered-from` for side quests

### Session Discipline

1. **Always sync at session end**: `bd sync && git push`
2. **Write for future you**: Notes should be self-contained
3. **Close what's done**: Don't leave completed work open
