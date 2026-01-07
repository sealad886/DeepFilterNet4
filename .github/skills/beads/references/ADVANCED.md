# Beads Advanced Features

Power-user features: molecules, chemistry patterns, async gates, agents, and worktrees.

---

## Molecules & Chemistry Patterns

### Molecule Concept

A **molecule** is a self-contained unit combining multiple issues with defined boundaries. Think of it as a mini-project with clear scope and dependencies.

### Standard Molecule Structure

```
Epic: Feature X (container)
├── Research: Explore options (discover)
├── Design: Architecture decision (decide)
├── Implement: Core functionality (build)
├── Test: Verification (verify)
└── Docs: Documentation (document)
```

### Creating a Molecule

```bash
# Create container epic
bd create "Feature: User Authentication" -t epic -p 1 --json  # bd-100

# Create phase issues
bd create "Research auth patterns" -t task -p 1 --json         # bd-101
bd create "Design: auth architecture" -t task -p 1 --json      # bd-102
bd create "Implement auth flow" -t task -p 1 --json            # bd-103
bd create "Test: auth coverage" -t task -p 2 --json            # bd-104
bd create "Docs: auth guide" -t task -p 2 --json               # bd-105

# Establish hierarchy
bd dep add bd-100 bd-101 --type parent-child
bd dep add bd-100 bd-102 --type parent-child
bd dep add bd-100 bd-103 --type parent-child
bd dep add bd-100 bd-104 --type parent-child
bd dep add bd-100 bd-105 --type parent-child

# Set sequence (using blocks)
bd dep add bd-101 bd-102   # Research before design
bd dep add bd-102 bd-103   # Design before implement
bd dep add bd-103 bd-104   # Implement before test
bd dep add bd-104 bd-105   # Test before docs
```

### Chemistry Patterns

**Pattern: Sequential Reaction**

Each step enables the next:

```
A ──blocks──▶ B ──blocks──▶ C ──blocks──▶ D
```

Use for: Waterfall phases, build pipelines, staged releases.

**Pattern: Parallel Synthesis**

Independent work streams:

```
       ┌──▶ B1 ──┐
A ────┼──▶ B2 ──┼────▶ C
       └──▶ B3 ──┘
```

Use for: Parallel development, concurrent tasks, resource optimization.

**Pattern: Catalyst**

One issue enables multiple:

```
       ┌──▶ B
A ────┼──▶ C
       └──▶ D
```

Use for: Foundation work, shared infrastructure, enabling tasks.

**Pattern: Convergence**

Multiple streams merge:

```
A1 ──┐
A2 ──┼────▶ B
A3 ──┘
```

Use for: Integration points, coordination, final assembly.

### Boundary Rules

**Molecule boundaries** define scope:

1. **Entry point**: Clear starting condition (issue becomes ready)
2. **Exit criteria**: Definition of done for the molecule
3. **Side effects**: What this molecule produces for others

When work spans boundaries:

```bash
# Document the boundary crossing
bd update bd-inside --notes "HANDOFF: Results passed to bd-outside" --json
bd dep add bd-inside bd-outside --type related
```

---

## Async Gates

### Gate Concept

An **async gate** is a checkpoint that automatically triggers when conditions are met. It's a pattern for managing dependencies across async work streams.

### Manual Gate Pattern

Since bd doesn't have automatic gates, implement them manually:

```bash
# Create gate issue
bd create "GATE: All components ready for integration" -t task -p 1 \
  -d "Close when: bd-101, bd-102, bd-103 all closed" --json

# Make gate block downstream work
bd dep add bd-gate bd-integration

# When checking gate status
bd show bd-101 bd-102 bd-103 --json | jq '.[] | .status'

# When all prerequisites done, close gate
bd close bd-gate --reason "All prerequisites complete" --json
```

### Gate Patterns

**All-of Gate**: Requires ALL prerequisites

```
bd-101 ──┐
bd-102 ──┼──▶ GATE ──▶ downstream
bd-103 ──┘
```

**Any-of Gate**: Requires ANY prerequisite (conceptual)

```bash
# Document as: close gate when first option succeeds
bd create "GATE: First approach succeeds" -t task \
  -d "Close when: bd-option-1 OR bd-option-2 succeeds" --json
```

**Timed Gate**: Deadline-based

```bash
bd create "GATE: Review deadline 2024-03-15" -t task \
  -d "Auto-close on deadline regardless of blockers" --json
```

### Gate Best Practices

1. **Document gate conditions explicitly** in description
2. **Keep gates simple**: Avoid complex logic
3. **Check gates regularly**: Manual review of status
4. **Clean up obsolete gates**: Close or remove when no longer needed

---

## Agent Patterns

### AI Agent Integration

When AI agents use beads:

**Prime at session start**:
```bash
bd prime  # Inject workflow context
```

**Session workflow**:
```bash
# 1. Find work
bd ready --json | jq '.[0]'  # Get first available

# 2. Claim work
bd update bd-42 --status in_progress --json

# 3. Read context
bd show bd-42 --json
bd dep tree bd-42

# 4. Work...
# 5. Update progress
bd update bd-42 --notes "Agent session: completed X, Y pending" --json

# 6. ALWAYS end with
bd sync && git push
```

### Agent-Specific Patterns

**Handoff Pattern**:

```bash
# Agent creates handoff issue for human review
bd create "REVIEW: Agent implementation needs human review" -t task -p 1 \
  -d "Agent completed: [work]
      Confidence: [high/medium/low]
      Review needed: [specific aspects]" \
  --deps discovered-from:bd-original --json
```

**Uncertainty Escalation**:

```bash
# Agent uncertain, creates decision request
bd create "DECISION: Which approach for caching?" -t task -p 1 \
  -d "Options identified:
      1. Redis - pros/cons
      2. Memcached - pros/cons
      Human input needed for final decision" --json
```

**Research Capture**:

```bash
# Agent captures findings during research
bd create "FINDING: Library X deprecated" \
  --deps discovered-from:bd-research \
  -d "Found during research: X is deprecated
      Alternative: Y
      Migration guide: [link]" --json
```

### Multi-Agent Coordination

When multiple agents work in same repo:

1. **Use `--actor` flag** for audit trail:
   ```bash
   bd --actor agent-1 update bd-42 --status in_progress --json
   bd --actor agent-2 create "Task" --json
   ```

2. **Coordinate via issues**: Each agent claims specific issues
3. **Avoid conflicts**: Only one agent per issue at a time
4. **Sync frequently**: `bd sync` after each significant change

---

## Worktree Patterns

### Git Worktree Integration

When using git worktrees for parallel work:

**Setup worktree**:
```bash
git worktree add ../feature-branch feature-branch
```

**Beads in worktrees**:

Each worktree shares the same `.beads/` database (if in main worktree) OR has its own copy. Choose strategy based on needs:

**Shared Database** (default):
```bash
# .beads/ stays in main worktree
# All worktrees see same issues
cd ../feature-branch
bd --db ../main/.beads/beads.db list --json
```

**Isolated Database** (for experiments):
```bash
# Copy database to worktree
cp ../main/.beads/beads.db .beads/
# Work in isolation
bd list --json
```

### Worktree Workflow

```bash
# Main worktree: feature planning
bd create "Feature X" -t epic --json

# Feature worktree: implementation
git worktree add ../feature-x feature-x
cd ../feature-x
bd update bd-feature-x --status in_progress --json
# ... work ...
bd update bd-feature-x --notes "Implementation complete" --json

# Back to main: integration
cd ../main
bd close bd-feature-x --reason "Merged" --json
git worktree remove ../feature-x
```

### Branch-Issue Association

Pattern for linking branches to issues:

```bash
# Create branch with issue reference
git checkout -b bd-42/short-description

# Reference in commits
git commit -m "feat: implement auth (bd-42)"

# Update issue with branch
bd update bd-42 --notes "Branch: bd-42/short-description" --json
```

---

## Integration Patterns

### CI/CD Integration

**Build triggers based on issue status**:

```yaml
# .github/workflows/build.yml
on:
  push:
    paths:
      - '.beads/issues.jsonl'

jobs:
  check-ready:
    steps:
      - run: |
          bd ready --json > ready.json
          # Use ready issues for deployment decisions
```

**Status updates from CI**:

```bash
# In CI script
bd update bd-42 --notes "CI: Build ${{ github.run_id }} passed" --json
bd sync
```

### External Tool Integration

**Export for external tools**:

```bash
# Export to JSONL
bd list --json | jq -c '.[]' > export.jsonl

# Import from external source
# Convert to JSONL format first, then:
bd import -i external.jsonl
```

**Webhook-style patterns** (manual implementation):

```bash
# On issue close, trigger external action
bd close bd-42 --json
# Custom script checks .beads/issues.jsonl changes
# Triggers external webhook/action
```

### IDE Integration

**VS Code tasks.json example**:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "bd: ready",
      "type": "shell",
      "command": "bd ready --json | jq",
      "problemMatcher": []
    },
    {
      "label": "bd: sync",
      "type": "shell",
      "command": "bd sync && git push",
      "problemMatcher": []
    }
  ]
}
```

---

## Advanced CLI Usage

### JSON Processing

```bash
# Extract specific fields
bd list --json | jq '.[] | {id, title, status}'

# Filter in jq
bd list --json | jq '[.[] | select(.priority <= 1)]'

# Count by status
bd list --json | jq 'group_by(.status) | map({status: .[0].status, count: length})'

# Get blocked issues
bd list --json | jq '[.[] | select(.blocked_by | length > 0)]'
```

### Bulk Operations

```bash
# Close multiple issues
bd close bd-1 bd-2 bd-3 --reason "Sprint complete" --json

# Update multiple (loop required)
for id in bd-1 bd-2 bd-3; do
  bd update $id --priority 1 --json
done

# Add label to multiple
for id in $(bd list --json | jq -r '.[].id'); do
  bd label add $id needs-review --json
done
```

### Scripting Patterns

```bash
#!/bin/bash
# Daily standup script

echo "=== Ready Work ==="
bd ready --json | jq -r '.[] | "[\(.priority)] \(.id): \(.title)"'

echo ""
echo "=== In Progress ==="
bd list --status in_progress --json | jq -r '.[] | "\(.id): \(.title)"'

echo ""
echo "=== Stale (>7 days) ==="
bd stale --days 7 --json | jq -r '.[] | "\(.id): \(.title)"'
```

### Debug Mode

```bash
# Verbose output for debugging
BD_LOG_LEVEL=debug bd list --json

# Direct SQLite access (emergency only)
sqlite3 .beads/beads.db "SELECT * FROM issues WHERE status='open'"

# Check daemon health
bd daemons health --json
```

---

## Architecture Decisions

### Why JSONL for Git

- **Line-based diffs**: Each issue is one line, clean diffs
- **Merge friendly**: Conflicts are per-issue, not whole file
- **Incremental sync**: Append-friendly format
- **Human readable**: Can inspect with standard tools

### Why SQLite + Daemon

- **Fast queries**: SQLite indexes for complex queries
- **Concurrent access**: Daemon serializes writes
- **Atomic operations**: Transaction safety
- **Rich queries**: Full SQL when needed via direct access

### Data Flow

```
CLI commands
    ↓
Daemon (serializes writes)
    ↓
SQLite database (.beads/beads.db)
    ↓
JSONL export (.beads/issues.jsonl)
    ↓
Git (track changes)
```

### bd prime

The `bd prime` command is the authoritative source for workflow context. It outputs comprehensive context that includes:

- Current issue state
- Ready work queue
- In-progress items
- Recent history
- Workflow guidance

AI agents should run `bd prime` at session start rather than manually reconstructing context from multiple commands.
