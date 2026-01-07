# Agent Guide

Comprehensive guide for AI agents using bd for task tracking.

## Quick Start for Agents

### Session Start

```bash
# 1. Check ready work
bd ready --json

# 2. Pick and claim an issue
bd update bd-42 --status in_progress --json

# 3. Read full context
bd show bd-42 --json
```

### During Work

```bash
# Capture discoveries
bd create "Found bug in auth" -t bug -p 1 --deps discovered-from:bd-42 --json

# Update progress
bd update bd-42 --notes "Completed: steps 1-3. Next: step 4" --json
```

### Session End

```bash
# Update final state
bd update bd-42 --notes "Session end: [status]" --json

# Close if complete
bd close bd-42 --reason "Implemented and tested" --json

# Always sync
bd sync
```

## Core Agent Workflows

### Workflow 1: Execute Ready Work

The standard flow for completing available tasks.

```bash
# Find work
bd ready --json

# Claim issue (prevents duplicate work)
bd update bd-42 --status in_progress --json

# Get full context
bd show bd-42 --json

# Work on implementation...

# When done
bd close bd-42 --reason "Completed successfully" --json
bd sync
```

### Workflow 2: Manage Blockers

Handle issues that can't proceed.

```bash
# Hit a blocker during work
bd create "Need database credentials" -t task -p 0 --json
# Returns: bd-43

# Mark original as blocked
bd dep add bd-43 bd-42  # New issue blocks current

# bd-42 disappears from ready queue
# When bd-43 is closed, bd-42 returns
```

### Workflow 3: Side Quest Discovery

Capture related work without losing context.

```bash
# Working on bd-42, find a bug
bd create "Bug: validation missing" -t bug -p 1 \
  --deps discovered-from:bd-42 --json
# Returns: bd-44

# Decide: switch or continue?
# Option A: Continue on bd-42, bd-44 queued for later
# Option B: Pause bd-42, work on bd-44

# If switching:
bd update bd-42 --status open --json  # Return to queue
bd update bd-44 --status in_progress --json
```

### Workflow 4: Epic Execution

Work through structured multi-part tasks.

```bash
# Check epic structure
bd dep tree bd-epic-1

# bd ready shows only unblocked children
bd ready --json

# Work through sequentially
# As each task completes, next becomes ready
bd close bd-task-1 --reason "Done" --json
bd ready --json  # Now shows bd-task-2
```

## Agent-Specific Considerations

### JSON Output

Always use `--json` for programmatic parsing:

```bash
bd ready --json
bd show bd-42 --json
bd create "Issue" -p 1 --json
```

### Sandbox Mode

In restricted environments (no daemon control):

```bash
bd --sandbox ready --json
bd --sandbox create "Issue" -p 1 --json
```

Or set environment:
```bash
export BD_NO_DAEMON=true
export BD_NO_AUTO_FLUSH=true
```

### Explicit IDs

For parallel agent work, use explicit IDs:

```bash
bd create "Issue" --id agent1-001 -p 1 --json
```

### Sync Timing

**Critical**: Always sync at session end:

```bash
bd sync
```

This ensures changes are committed to git immediately.

## Context Preservation

### Capture Decisions

```bash
bd update bd-42 --design "
Using approach X because:
1. Better performance
2. Team familiarity
3. Simpler to test
" --json
```

### Track Progress

```bash
bd update bd-42 --notes "
=== Progress ===
[x] Step 1: Database schema
[x] Step 2: API endpoint
[ ] Step 3: Frontend integration
[ ] Step 4: Tests
" --json
```

### Link Everything

```bash
# Discoveries
bd create "..." --deps discovered-from:bd-42 --json

# Blockers
bd dep add blocker-id bd-42

# Related work
bd dep add related-id bd-42 --type related
```

## Common Agent Tasks

### Find High Priority Work

```bash
bd ready --json | jq '[.[] | select(.priority <= 1)]'
```

### Find Work by Type

```bash
bd list --type bug --status open --json
```

### Check Dependencies

```bash
bd dep tree bd-42
```

### Batch Close

```bash
bd close bd-41 bd-42 bd-43 --reason "Batch complete" --json
```

### Check Stale Issues

```bash
bd stale --days 30 --status in_progress --json
```

## Error Handling

### Issue Not Found

```bash
# Verify ID exists
bd list --json | jq '.[].id'
```

### Database Stale

```bash
# Force import
bd import --force -i .beads/issues.jsonl

# Or skip check
bd --allow-stale ready --json
```

### Sync Fails

```bash
# Check git status
git status

# Resolve conflicts manually
git add .beads/issues.jsonl
git commit -m "Resolve bd conflict"
bd sync
```

## Best Practices for Agents

### Do

- Always use `--json` flag
- Sync at session end
- Capture discoveries immediately
- Update notes as you work
- Use meaningful close reasons

### Don't

- Leave issues in_progress across sessions without notes
- Create duplicate issues (search first)
- Skip sync at session end
- Use blocking dependencies for preferences

### Decision Making

**When to create issue**:
- Multi-session work
- Has dependencies
- Others need visibility
- Context preservation needed

**When to use TodoWrite instead**:
- Single-session task
- Linear execution
- No dependencies
- All context in conversation

## See Also

- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Full command reference
- [WORKFLOWS.md](WORKFLOWS.md) - Detailed workflow patterns
- [BOUNDARIES.md](BOUNDARIES.md) - bd vs TodoWrite decision guide
- [RESUMABILITY.md](RESUMABILITY.md) - Context preservation
