# Async Gates

Async gates are synchronization points for parallel agent work where multiple agents must complete their tasks before dependent work can proceed.

## What Are Async Gates?

An **async gate** is a coordination pattern where:

1. Multiple parallel tasks (inputs) must complete
2. Before a dependent task (output) can start
3. bd's dependency system manages this automatically

Think of it like an AND gate in electronics: all inputs must be "high" (complete) before the output activates.

## Basic Pattern

```
Task A ──┐
Task B ──┼──▶ Task D (blocked until A, B, C all complete)
Task C ──┘
```

### Creating an Async Gate

```bash
# Create parallel tasks
bd create "Research option A" -t task -p 2 --json  # bd-1
bd create "Research option B" -t task -p 2 --json  # bd-2
bd create "Research option C" -t task -p 2 --json  # bd-3

# Create dependent task
bd create "Decision: Choose option" -t task -p 1 --json  # bd-4

# Set up gate: all research blocks decision
bd dep add bd-1 bd-4  # A blocks decision
bd dep add bd-2 bd-4  # B blocks decision
bd dep add bd-3 bd-4  # C blocks decision
```

**Result**: `bd-4` only appears in `bd ready` when `bd-1`, `bd-2`, AND `bd-3` are all closed.

## Use Cases

### Multi-Source Research

Wait for multiple research tracks to complete before making a decision.

```bash
# Research tracks
bd create "Research: Database options" -t task -p 2 --json      # bd-r1
bd create "Research: Caching strategies" -t task -p 2 --json    # bd-r2
bd create "Research: API frameworks" -t task -p 2 --json        # bd-r3

# Architecture decision (gated by all research)
bd create "Architecture decision document" -t task -p 1 --json  # bd-d1

# Gate setup
bd dep add bd-r1 bd-d1
bd dep add bd-r2 bd-d1
bd dep add bd-r3 bd-d1
```

### Parallel Implementation

Wait for parallel feature work before integration.

```bash
# Feature tracks
bd create "Implement auth service" -t feature -p 1 --json    # bd-f1
bd create "Implement user service" -t feature -p 1 --json    # bd-f2
bd create "Implement order service" -t feature -p 1 --json   # bd-f3

# Integration task
bd create "Integration tests for all services" -t task -p 1 --json  # bd-int

# Gate setup
bd dep add bd-f1 bd-int
bd dep add bd-f2 bd-int
bd dep add bd-f3 bd-int
```

### Multi-Stakeholder Review

Wait for multiple reviewers before proceeding.

```bash
# Review tasks
bd create "Security review" -t task -p 1 --json       # bd-rev1
bd create "Performance review" -t task -p 1 --json    # bd-rev2
bd create "UX review" -t task -p 1 --json             # bd-rev3

# Proceed to launch
bd create "Launch preparation" -t task -p 0 --json    # bd-launch

# Gate setup
bd dep add bd-rev1 bd-launch
bd dep add bd-rev2 bd-launch
bd dep add bd-rev3 bd-launch
```

## Chained Gates

Gates can be chained for complex workflows.

```
Research A ──┐
Research B ──┼──▶ Analysis ──┐
Research C ──┘                │
                              ├──▶ Final Decision
Review X ──┐                  │
Review Y ──┼──▶ Approval ────┘
Review Z ──┘
```

```bash
# Research gate
bd create "Research A" -t task -p 2 --json  # bd-ra
bd create "Research B" -t task -p 2 --json  # bd-rb
bd create "Research C" -t task -p 2 --json  # bd-rc
bd create "Analysis" -t task -p 1 --json    # bd-analysis
bd dep add bd-ra bd-analysis
bd dep add bd-rb bd-analysis
bd dep add bd-rc bd-analysis

# Review gate
bd create "Review X" -t task -p 1 --json    # bd-rx
bd create "Review Y" -t task -p 1 --json    # bd-ry
bd create "Review Z" -t task -p 1 --json    # bd-rz
bd create "Approval" -t task -p 1 --json    # bd-approval
bd dep add bd-rx bd-approval
bd dep add bd-ry bd-approval
bd dep add bd-rz bd-approval

# Final gate
bd create "Final Decision" -t task -p 0 --json  # bd-final
bd dep add bd-analysis bd-final
bd dep add bd-approval bd-final
```

## Partial Gates

Sometimes you want to proceed when some (not all) inputs are complete.

**bd doesn't have built-in partial gate support**, but you can work around this:

### Option 1: Timeout with Manual Override

```bash
# Create time-boxed research
bd create "Research A (time-boxed 2hr)" -t task -p 2 --json
bd create "Research B (time-boxed 2hr)" -t task -p 2 --json

# If one times out, close with status
bd close bd-ra --reason "Time-boxed: partial results" --json

# Decision can now proceed
```

### Option 2: Majority Rule

Create a "quorum reached" intermediate task:

```bash
# Reviewers
bd create "Review by Alice" -t task -p 1 --json  # bd-r1
bd create "Review by Bob" -t task -p 1 --json    # bd-r2
bd create "Review by Carol" -t task -p 1 --json  # bd-r3

# Quorum task (human/agent marks when 2 of 3 done)
bd create "Quorum reached (2 of 3 reviews)" -t task -p 1 --json  # bd-quorum

# Final task blocked by quorum, not individual reviews
bd create "Proceed with launch" -t task -p 0 --json
bd dep add bd-quorum bd-proceed

# When 2 reviews done, manually close quorum
bd close bd-quorum --reason "2 of 3 reviews complete" --json
```

## Agent Coordination with Gates

### Orchestrator Pattern

One agent manages gate progression:

```bash
# Orchestrator creates structure
bd create "Research A" -t task -p 2 --json
bd create "Research B" -t task -p 2 --json
bd create "Decision (gated)" -t task -p 1 --json
bd dep add bd-ra bd-decision
bd dep add bd-rb bd-decision
bd sync

# Worker agents claim parallel tasks
# Agent 1: bd update bd-ra --status in_progress --json
# Agent 2: bd update bd-rb --status in_progress --json

# Orchestrator monitors progress
bd list --status in_progress --json

# When gate opens, orchestrator or worker claims gated task
bd ready --json  # bd-decision now appears
bd update bd-decision --status in_progress --json
```

### Self-Organizing Pattern

Agents autonomously claim gated work:

```bash
# Any agent checking bd ready will see gated task once unblocked
bd ready --json

# First agent to claim it wins
bd update bd-decision --status in_progress --json
```

## Visualizing Gates

Use `bd dep tree` to visualize gate structure:

```bash
bd dep tree bd-final

# Output:
# bd-final: Final Decision [P0, open, blocked]
# ├── bd-analysis: Analysis [P1, open, blocked]
# │   ├── bd-ra: Research A [P2, in_progress]
# │   ├── bd-rb: Research B [P2, closed]
# │   └── bd-rc: Research C [P2, open]
# └── bd-approval: Approval [P1, open, blocked]
#     ├── bd-rx: Review X [P1, in_progress]
#     ├── bd-ry: Review Y [P1, closed]
#     └── bd-rz: Review Z [P1, closed]
```

## Best Practices

### 1. Clear Gating Criteria

Document what the gate is waiting for:

```bash
bd create "Decision: DB choice (gated by research)" -t task -p 1 \
  -d "Blocked until:
      - bd-ra: PostgreSQL research complete
      - bd-rb: MySQL research complete
      - bd-rc: MongoDB research complete

      Will synthesize findings and make recommendation." \
  --json
```

### 2. Reasonable Parallelism

Don't create gates with too many inputs:
- 2-5 inputs: Good
- 6-10 inputs: Consider grouping
- 10+ inputs: Break into phases

### 3. Explicit Timeout Plans

For time-sensitive gates:

```bash
bd create "Research (2hr time-box)" -t task -p 2 \
  -d "Time-boxed to 2 hours.
      If incomplete, close with partial findings.
      Decision can proceed with available information." \
  --json
```

### 4. Status Visibility

Regularly check gate status:

```bash
# See all blocked issues
bd list --status open --json | jq '[.[] | select(.blocked == true)]'

# Check specific gate
bd dep tree bd-gated-task
```

## See Also

- [MOLECULES.md](MOLECULES.md) - Parallel agent coordination
- [DEPENDENCIES.md](DEPENDENCIES.md) - Dependency types
- [AGENTS.md](AGENTS.md) - Agent workflows
- [WORKFLOWS.md](WORKFLOWS.md) - Common patterns
