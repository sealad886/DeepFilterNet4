# Resumability: Surviving Context Loss

How bd enables work resumption after conversation compaction, session breaks, and context resets.

## The Problem bd Solves

AI agents face a fundamental challenge: **context loss**.

### When Context Is Lost

1. **Conversation compaction**: Long conversations get summarized, losing detail
2. **Session breaks**: New conversation starts fresh
3. **Token limits**: Context window fills, old content dropped
4. **Human breaks**: Return after days/weeks away
5. **Agent handoffs**: Different agent continues work

### What Gets Lost

Without bd:
- Current task progress
- Design decisions made
- Blockers discovered
- Side quests found
- Dependencies identified
- Acceptance criteria refined

### How bd Preserves Context

bd creates **persistent, structured context** that survives all these scenarios:

```
Git-backed database
  ↓
Issue with full context
  ↓
Survives compaction
  ↓
Resume with bd ready
```

## Designing for Resumability

### Principle 1: Capture Context Early

Don't wait until you're blocked to create an issue.

**Before starting significant work**:
```bash
bd create "Implement user authentication" -t feature -p 1 \
  -d "Goal: Allow users to log in with email/password
      Approach: Use bcrypt for hashing, JWT for sessions
      Acceptance:
      - [ ] Login endpoint works
      - [ ] Sessions persist across requests
      - [ ] Tests cover happy path and errors" \
  --json
```

**Why**: If compaction happens mid-work, you have the full context captured.

### Principle 2: Update as You Learn

Design notes capture evolving understanding.

**When you discover something**:
```bash
bd update bd-42 --design "
Found that bcrypt has configurable cost factor.
Using cost=12 for balance of security and performance.
See: https://example.com/bcrypt-tuning
" --json
```

**Why**: Future sessions see your thinking, not just the outcome.

### Principle 3: Link Discoveries

Use `discovered-from` to preserve provenance.

**When you find related work**:
```bash
bd create "Bug: Password hashing doesn't use constant-time compare" \
  -t bug -p 1 \
  --deps discovered-from:bd-42 \
  --json
```

**Why**: Returns show what led to what. Side quests don't get lost.

### Principle 4: Mark Blockers Explicitly

Dependencies create automatic resumability.

**When blocked**:
```bash
# Create the blocker
bd create "Need to upgrade auth library for security patch" \
  -t task -p 0 --json
# Returns: bd-43

# Link as blocker
bd dep add bd-43 bd-42

# bd-42 disappears from ready queue
# When bd-43 closes, bd-42 automatically returns
```

**Why**: You can't accidentally resume blocked work. Unblocking is automatic.

## Session Start Protocol

Every session should start with context recovery.

### Step 1: Check Ready Work

```bash
bd ready --json
```

This shows all unblocked issues. Pick up where you left off.

### Step 2: Review In-Progress

```bash
bd list --status in_progress --json
```

These are issues someone (possibly past you) started but didn't finish.

### Step 3: Check Recent Activity

```bash
bd list --updated-after $(date -v-7d +%Y-%m-%d) --json
```

See what changed in the last week for context.

### Step 4: Claim Your Work

```bash
bd update bd-42 --status in_progress --json
```

Mark what you're working on so future sessions know.

## Session End Protocol

Every session should capture state for future resumption.

### Step 1: Update Progress

```bash
# Add notes about current state
bd update bd-42 --notes "
Session end: Completed login endpoint.
Next: Add logout and test coverage.
Blocked: None currently.
" --json
```

### Step 2: Create Any Discovered Issues

```bash
# Capture anything found but not addressed
bd create "Noticed: Error messages leak internal paths" \
  -t bug -p 2 \
  --deps discovered-from:bd-42 \
  --json
```

### Step 3: Sync to Git

```bash
bd sync
```

This ensures all changes are committed and pushed.

### Step 4: Set Status

```bash
# If pausing work
bd update bd-42 --status open --json  # Return to queue

# If done for now but in progress
# Keep as in_progress, notes explain state
```

## Resumability Patterns

### Pattern 1: Breadcrumb Trail

Leave explicit markers for future sessions.

```bash
bd update bd-42 --notes "
=== Session 2024-01-15 ===
Completed: Database schema, password hashing
In progress: Login endpoint (50% done)
Next: Finish login, add tests
Discovered: bd-50 (error handling), bd-51 (rate limiting)
Blockers: None

=== Session 2024-01-14 ===
Started work. Set up project structure.
" --json
```

### Pattern 2: Checkpoint Issues

Create explicit checkpoint markers for long work.

```bash
bd create "CHECKPOINT: Auth phase 1 complete" \
  -t task -p 4 \
  -d "Marking completion of:
      - Database schema
      - Password hashing
      - Basic login endpoint

      Remaining for phase 2:
      - Session management
      - Logout
      - Tests" \
  --json
```

### Pattern 3: Resumption Tags

Use labels to mark resumption points.

```bash
bd label add bd-42 resume-point --json
bd label add bd-42 needs-review --json

# Later, find resumption points
bd list --label resume-point --json
```

### Pattern 4: Dependency Chain

Create explicit dependency chains that guide work order.

```bash
# Phase 1
bd create "Phase 1: Core auth" --json  # bd-100
bd create "Schema" --json              # bd-101
bd create "Hashing" --json             # bd-102
bd dep add bd-101 bd-102               # Schema before hashing

# Phase 2 blocked by phase 1
bd create "Phase 2: Sessions" --json   # bd-200
bd dep add bd-102 bd-200               # Hashing before sessions
```

**Result**: `bd ready` naturally guides you through the correct order.

## Recovery Scenarios

### Scenario 1: Return After Vacation

```bash
# What's ready to work on?
bd ready --json

# What was I doing?
bd list --assignee me --status in_progress --json

# What changed while I was gone?
bd list --updated-after 2024-01-01 --json

# Any new blockers?
bd list --label blocked --json
```

### Scenario 2: Conversation Compaction

Mid-conversation, context gets compacted. You lose the detailed history.

```bash
# Still know what to work on
bd ready --json

# Get full context for current issue
bd show bd-42 --json
# Returns: title, description, design notes, acceptance criteria, links

# See what's blocking/blocked
bd dep tree bd-42
```

### Scenario 3: Agent Handoff

A different agent needs to continue your work.

```bash
# New agent starts fresh
bd ready --json

# Picks up in-progress work
bd show bd-42 --json

# Sees full history:
# - Original description
# - Design notes from previous agent
# - Discovered issues linked
# - Acceptance criteria
# - Current status and notes
```

### Scenario 4: Context Window Full

You've been working for hours and context is maxed.

```bash
# Don't rely on conversation history
# All context is in bd

bd show bd-42 --json  # Full issue context
bd dep tree bd-42     # Dependency context
bd list --label current-sprint --json  # Broader context
```

## Best Practices

### Do: Capture Decisions

```bash
bd update bd-42 --design "
Decided to use JWT instead of sessions because:
1. Stateless - easier to scale
2. Works with mobile clients
3. Team has JWT experience
" --json
```

### Do: Link Everything

```bash
# Every discovery gets linked
bd create "Found: ..." --deps discovered-from:bd-42 --json

# Every blocker gets linked
bd dep add blocker-id bd-42
```

### Do: Update Frequently

```bash
# After each significant step
bd update bd-42 --notes "Completed step X, starting Y" --json
```

### Don't: Rely on Conversation

Don't assume you'll remember or can scroll back. Capture in bd.

### Don't: Leave Issues Stale

If work pauses, update the issue to reflect current state.

### Don't: Skip Session Protocols

Start and end every session with the proper protocols.

## Summary

**Resumability requires**:
1. **Early capture**: Create issues before starting work
2. **Continuous updates**: Capture decisions and progress
3. **Explicit links**: Connect related and discovered work
4. **Session protocols**: Consistent start and end procedures
5. **Structured context**: Use bd fields (design, notes, acceptance)

**The payoff**:
- Return after any gap with full context
- Survive compaction without losing work
- Enable agent handoffs seamlessly
- Track long-horizon projects effectively

bd is your persistent memory. Use it well.

## See Also

- [WORKFLOWS.md](WORKFLOWS.md) - Session workflows
- [AGENTS.md](AGENTS.md) - Agent protocols
- [PATTERNS.md](PATTERNS.md) - Common patterns
- [BOUNDARIES.md](BOUNDARIES.md) - When bd helps most
