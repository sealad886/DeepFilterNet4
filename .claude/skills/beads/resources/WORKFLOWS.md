# Workflow Patterns

Step-by-step guides for common bd workflows.

## Daily Workflow

### Morning Start

```bash
# 1. Check what's ready to work on
bd ready --json

# 2. Review any in-progress work
bd list --status in_progress --json

# 3. Pick an issue and claim it
bd update bd-42 --status in_progress --json

# 4. Work on the issue...
```

### During Work

```bash
# When you discover something
bd create "Found: validation bug in form" -t bug -p 1 \
  --deps discovered-from:bd-42 --json

# When you hit a blocker
bd create "Need API key for external service" -t task -p 0 --json
bd dep add bd-new bd-42  # New issue blocks current work

# When you make progress
bd update bd-42 --notes "Completed step 1, starting step 2" --json
```

### End of Day

```bash
# 1. Update current issue with status
bd update bd-42 --notes "
End of day status:
- Completed: Steps 1-3
- In progress: Step 4
- Tomorrow: Finish step 4, write tests
" --json

# 2. Create any pending issues
bd create "TODO: Add error handling" -t task -p 2 \
  --deps discovered-from:bd-42 --json

# 3. Sync everything
bd sync
```

## Starting a New Feature

### Step 1: Create Epic

```bash
bd create "Epic: User notifications system" -t epic -p 1 \
  -d "Allow users to receive and manage notifications.

      Goals:
      - Real-time notifications
      - Email digest option
      - Notification preferences

      Out of scope:
      - Push notifications (phase 2)
      - SMS notifications" \
  --json
# Returns: bd-100
```

### Step 2: Break Down into Tasks

```bash
# Database layer
bd create "Design notification schema" -t task -p 1 --json  # bd-101
bd create "Create notification model" -t task -p 1 --json   # bd-102

# Backend
bd create "Add notification endpoints" -t task -p 1 --json  # bd-103
bd create "Implement email digest job" -t task -p 2 --json  # bd-104

# Frontend
bd create "Add notification UI" -t task -p 1 --json         # bd-105
bd create "Add preferences page" -t task -p 2 --json        # bd-106

# Testing
bd create "Write notification tests" -t task -p 2 --json    # bd-107
```

### Step 3: Set Up Hierarchy

```bash
# All tasks under epic
bd dep add bd-100 bd-101 --type parent-child
bd dep add bd-100 bd-102 --type parent-child
bd dep add bd-100 bd-103 --type parent-child
bd dep add bd-100 bd-104 --type parent-child
bd dep add bd-100 bd-105 --type parent-child
bd dep add bd-100 bd-106 --type parent-child
bd dep add bd-100 bd-107 --type parent-child
```

### Step 4: Set Up Dependencies

```bash
# Database first
bd dep add bd-101 bd-102  # Schema before model

# Model before endpoints
bd dep add bd-102 bd-103
bd dep add bd-102 bd-104

# Endpoints before UI
bd dep add bd-103 bd-105
bd dep add bd-103 bd-106

# Everything before tests
bd dep add bd-105 bd-107
bd dep add bd-106 bd-107
```

### Step 5: Start Work

```bash
# Check what's ready
bd ready --json
# Shows: bd-101 (Design notification schema)

# Claim and start
bd update bd-101 --status in_progress --json
```

## Bug Investigation

### Step 1: Create Bug Issue

```bash
bd create "Bug: Users can't save profile changes" -t bug -p 1 \
  -d "**Reported**: Customer support ticket #1234
      **Frequency**: 100% reproducible
      **Steps**:
      1. Go to profile settings
      2. Change any field
      3. Click Save
      4. Changes don't persist

      **Expected**: Changes saved
      **Actual**: Spinner forever, no error" \
  --json
# Returns: bd-bug-1
```

### Step 2: Investigation

```bash
# Update as you learn
bd update bd-bug-1 --design "
Investigation notes:
- Frontend logs show API call succeeding
- Backend logs show 500 error in ProfileService
- Database query times out

Root cause: N+1 query in profile update
" --json
```

### Step 3: Create Fix Tasks

```bash
# If fix is multi-step
bd create "Fix N+1 query in ProfileService" -t task -p 1 --json  # bd-fix-1
bd create "Add database indexes for profile" -t task -p 1 --json # bd-fix-2
bd create "Add regression test" -t task -p 2 --json              # bd-fix-3

# Link to bug
bd dep add bd-bug-1 bd-fix-1 --type parent-child
bd dep add bd-bug-1 bd-fix-2 --type parent-child
bd dep add bd-bug-1 bd-fix-3 --type parent-child

# Set dependencies
bd dep add bd-fix-1 bd-fix-3  # Fix before test
bd dep add bd-fix-2 bd-fix-3  # Index before test
```

### Step 4: Execute Fix

```bash
# Work through ready queue
bd ready --json

# Complete each task
bd update bd-fix-1 --status in_progress --json
# ... fix query ...
bd close bd-fix-1 --reason "Fixed N+1 query" --json

bd update bd-fix-2 --status in_progress --json
# ... add indexes ...
bd close bd-fix-2 --reason "Added indexes" --json

bd update bd-fix-3 --status in_progress --json
# ... write test ...
bd close bd-fix-3 --reason "Added regression test" --json
```

### Step 5: Close Bug

```bash
bd close bd-bug-1 --reason "Fixed: N+1 query + indexes. Test added." --json
bd sync
```

## Code Review Workflow

### Reviewer Perspective

```bash
# Create review issue
bd create "Review: PR #123 - Add notifications" -t task -p 1 \
  -d "Review PR from @developer
      Files: src/notifications/*
      Related: bd-100 (notifications epic)" \
  --json
# Returns: bd-review-1

# During review, capture feedback
bd create "Review finding: Missing input validation" -t bug -p 2 \
  --deps discovered-from:bd-review-1 --json

bd create "Review finding: Consider caching here" -t task -p 3 \
  --deps discovered-from:bd-review-1 --json

# Complete review
bd close bd-review-1 --reason "Review complete. 2 issues created." --json
```

### Author Perspective

```bash
# Check for review feedback
bd list --title-contains "Review finding" --status open --json

# Address each finding
bd update bd-finding-1 --status in_progress --json
# ... fix issue ...
bd close bd-finding-1 --reason "Fixed in commit abc123" --json
```

## Research Workflow

### Step 1: Define Question

```bash
bd create "Research: Best caching strategy for our API" -t task -p 2 \
  -d "**Question**: What caching strategy should we use?

      **Context**: API response times are slow (>500ms)

      **Options to evaluate**:
      - In-memory (Redis)
      - CDN (Cloudflare)
      - Application-level caching

      **Deliverable**: Recommendation document
      **Time box**: 4 hours" \
  --json
# Returns: bd-research-1
```

### Step 2: Capture Findings

```bash
# As you research, create finding issues
bd create "Finding: Redis supports our data patterns well" \
  -t task -p 3 \
  -d "Redis JSONGET supports nested queries we need.
      Latency: <1ms for our data sizes.
      Cost: ~$50/month for our volume." \
  --deps discovered-from:bd-research-1 \
  --json

bd create "Finding: CDN won't work with our auth model" \
  -t task -p 3 \
  -d "CDN caching requires public endpoints.
      Our API is fully authenticated.
      Would need significant rearchitecture." \
  --deps discovered-from:bd-research-1 \
  --json

bd create "Finding: App caching adds complexity" \
  -t task -p 3 \
  -d "In-process caching requires cache invalidation.
      Multi-instance deployment complicates this.
      Not recommended without Redis." \
  --deps discovered-from:bd-research-1 \
  --json
```

### Step 3: Make Decision

```bash
bd create "Decision: Use Redis for API caching" \
  -t task -p 1 \
  -d "Based on research (bd-research-1):

      **Decision**: Implement Redis caching

      **Rationale**:
      - Supports our data patterns (Finding 1)
      - Works with authenticated endpoints (Finding 2)
      - Simpler than app caching (Finding 3)

      **Next steps**:
      - Set up Redis instance
      - Add caching layer to API
      - Update deployment config" \
  --deps discovered-from:bd-research-1 \
  --json
```

### Step 4: Close Research

```bash
bd close bd-research-1 \
  --reason "Complete. Decision: Use Redis. See bd-decision-1." \
  --json
```

## Sprint Planning

### Step 1: Review Backlog

```bash
# See all open issues by priority
bd list --status open --json | jq 'sort_by(.priority)'

# See stale issues
bd stale --days 30 --json
```

### Step 2: Label for Sprint

```bash
# Add sprint label to selected issues
bd label add bd-42 sprint-23 --json
bd label add bd-43 sprint-23 --json
bd label add bd-44 sprint-23 --json
```

### Step 3: Verify Dependencies

```bash
# Check that sprint items aren't blocked by non-sprint items
bd list --label sprint-23 --json | jq '.[].id' | while read id; do
  bd dep tree "$id"
done
```

### Step 4: During Sprint

```bash
# See sprint progress
bd list --label sprint-23 --json | jq 'group_by(.status) | map({status: .[0].status, count: length})'

# Find blocked items
bd list --label sprint-23,blocked --json
```

### Step 5: Sprint End

```bash
# Review incomplete items
bd list --label sprint-23 --status open --json

# Move to next sprint or backlog
bd label remove bd-42 sprint-23 --json
bd label add bd-42 sprint-24 --json
```

## Handoff Workflow

When handing work to another person or agent.

### Outgoing Handoff

```bash
# 1. Update all in-progress issues
bd list --status in_progress --assignee me --json | jq '.[].id' | while read id; do
  bd update "$id" --notes "HANDOFF: [Current state and next steps]" --json
done

# 2. Create handoff summary issue
bd create "Handoff: Work from @me to @newperson" -t task -p 1 \
  -d "**Handing off**:
      - bd-42: In progress, step 3 of 5
      - bd-43: Blocked on external API key
      - bd-44: Ready to start

      **Key context**:
      - Design decisions in bd-42 design notes
      - API documentation in /docs/api.md

      **Questions**: Reach me at @me" \
  --json

# 3. Sync everything
bd sync
```

### Incoming Handoff

```bash
# 1. Find handoff summary
bd list --title-contains "Handoff" --status open --json

# 2. Review handed-off issues
bd show bd-42 --json  # Read notes, design, etc.
bd show bd-43 --json
bd show bd-44 --json

# 3. Claim issues
bd update bd-42 --assignee newperson --json
bd update bd-43 --assignee newperson --json
bd update bd-44 --assignee newperson --json

# 4. Close handoff issue
bd close handoff-1 --reason "Handoff received and understood" --json
```

## See Also

- [PATTERNS.md](PATTERNS.md) - Common issue patterns
- [BOUNDARIES.md](BOUNDARIES.md) - When to use bd
- [RESUMABILITY.md](RESUMABILITY.md) - Session protocols
- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Command details
