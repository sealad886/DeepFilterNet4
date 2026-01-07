# Common Patterns

Reusable patterns for organizing work with bd.

## Epic Decomposition

### Pattern: Breaking Down Large Features

```
1. Create epic for high-level goal
2. Create child tasks for implementable units
3. Add blocking dependencies for sequence
4. Work through bd ready queue
```

### Example: Authentication System

```bash
# Create epic
bd create "Implement user authentication" -t epic -p 1 --json
# Returns: bd-auth-1

# Create subtasks
bd create "Set up auth database schema" -t task -p 1 --json     # bd-auth-2
bd create "Implement password hashing" -t task -p 1 --json       # bd-auth-3
bd create "Create login endpoint" -t task -p 1 --json            # bd-auth-4
bd create "Create logout endpoint" -t task -p 1 --json           # bd-auth-5
bd create "Add session management" -t task -p 1 --json           # bd-auth-6
bd create "Write authentication tests" -t task -p 2 --json       # bd-auth-7

# Set up hierarchy
bd dep add bd-auth-1 bd-auth-2 --type parent-child
bd dep add bd-auth-1 bd-auth-3 --type parent-child
bd dep add bd-auth-1 bd-auth-4 --type parent-child
bd dep add bd-auth-1 bd-auth-5 --type parent-child
bd dep add bd-auth-1 bd-auth-6 --type parent-child
bd dep add bd-auth-1 bd-auth-7 --type parent-child

# Set up blocking dependencies (sequence)
bd dep add bd-auth-2 bd-auth-3  # Schema before hashing
bd dep add bd-auth-3 bd-auth-4  # Hashing before login
bd dep add bd-auth-4 bd-auth-5  # Login before logout
bd dep add bd-auth-4 bd-auth-6  # Login before sessions
bd dep add bd-auth-6 bd-auth-7  # Sessions before tests
```

**Result**: `bd ready` shows only bd-auth-2 initially. As you complete each task, the next becomes ready.

## Research and Decision

### Pattern: Structured Investigation

```
1. Create research issue with clear question
2. Create finding issues as you discover information
3. Link findings with discovered-from
4. Create decision issue summarizing conclusions
```

### Example: Database Selection

```bash
# Research question
bd create "Research: Choose database for user data" -t task -p 2 \
  -d "Evaluate PostgreSQL, MySQL, MongoDB for our use case.
      Consider: performance, scalability, team expertise, cost." \
  --json
# Returns: bd-db-1

# Findings (created during research)
bd create "Finding: PostgreSQL JSONB supports our schema flexibility needs" \
  -t task -p 3 --deps discovered-from:bd-db-1 --json

bd create "Finding: Team has most experience with PostgreSQL" \
  -t task -p 3 --deps discovered-from:bd-db-1 --json

bd create "Finding: MongoDB requires separate deployment expertise" \
  -t task -p 3 --deps discovered-from:bd-db-1 --json

# Decision
bd create "Decision: Use PostgreSQL for user data" \
  -t task -p 1 \
  -d "Based on research (bd-db-1), selecting PostgreSQL.
      Rationale: JSONB flexibility, team expertise, simpler ops." \
  --deps discovered-from:bd-db-1 --json
```

## Bug Triage

### Pattern: Systematic Bug Handling

```
1. Create bug issue with reproduction steps
2. Investigate and document root cause
3. Create fix tasks if complex
4. Link related bugs with related dependency
```

### Example: Login Bug

```bash
# Initial bug report
bd create "Bug: Login fails intermittently" -t bug -p 1 \
  -d "**Reported by**: Customer support
      **Frequency**: ~5% of login attempts
      **Steps**:
      1. Go to login page
      2. Enter valid credentials
      3. Click submit
      4. Sometimes shows 'Server error'" \
  --json
# Returns: bd-bug-1

# After investigation, update with findings
bd update bd-bug-1 \
  -d "**Root cause**: Race condition in session creation when concurrent requests.
      **Fix approach**: Add mutex lock in SessionManager.create()" \
  --json

# If fix is complex, create subtasks
bd create "Add SessionManager mutex" -t task -p 1 --json
bd create "Add regression test for concurrent logins" -t task -p 2 --json

# Link back to bug
bd dep add bd-bug-1 bd-fix-1 --type parent-child
bd dep add bd-bug-1 bd-fix-2 --type parent-child
```

## Side Quest Handling

### Pattern: Capturing Discoveries Without Losing Context

```
1. Working on main issue
2. Discover related problem
3. Create new issue with discovered-from
4. Decide: continue or switch
5. Resume main work with full context
```

### Example: Found Bug During Feature

```bash
# Working on bd-100: "Add user profile page"
# Discover auth doesn't handle profile permissions

# Create discovery issue
bd create "Bug: Auth doesn't check profile permissions" \
  -t bug -p 1 \
  --deps discovered-from:bd-100 \
  --json
# Returns: bd-101

# Assess: Does this block the feature?
# Yes - profile page needs proper permissions

# Mark feature as blocked
bd dep add bd-101 bd-100  # Bug blocks feature

# Work on bug first
bd update bd-101 --status in_progress --json
# ... fix bug ...
bd close bd-101 --reason "Added profile permission check" --json

# Resume feature (now unblocked)
bd update bd-100 --status in_progress --json
```

## Parallel Workstreams

### Pattern: Managing Independent Work Tracks

```
1. Create separate epics for each workstream
2. No blocking dependencies between streams
3. Use labels to categorize
4. Related links for context
```

### Example: Frontend and Backend Work

```bash
# Frontend epic
bd create "Epic: New dashboard UI" -t epic -p 1 -l frontend --json
# Returns: bd-fe-1

# Backend epic
bd create "Epic: Dashboard API" -t epic -p 1 -l backend --json
# Returns: bd-be-1

# Link as related (not blocking)
bd dep add bd-fe-1 bd-be-1 --type related

# Create frontend tasks
bd create "Design dashboard layout" -t task -p 2 -l frontend --json
bd create "Implement dashboard components" -t task -p 2 -l frontend --json
bd dep add bd-fe-1 bd-fe-2 --type parent-child
bd dep add bd-fe-1 bd-fe-3 --type parent-child
bd dep add bd-fe-2 bd-fe-3  # Design before implement

# Create backend tasks
bd create "Design API schema" -t task -p 2 -l backend --json
bd create "Implement API endpoints" -t task -p 2 -l backend --json
bd dep add bd-be-1 bd-be-2 --type parent-child
bd dep add bd-be-1 bd-be-3 --type parent-child
bd dep add bd-be-2 bd-be-3  # Schema before implement

# Integration task (blocked by both)
bd create "Integrate dashboard with API" -t task -p 1 --json
bd dep add bd-fe-3 bd-int-1  # Frontend blocks integration
bd dep add bd-be-3 bd-int-1  # Backend blocks integration
```

**Result**: Frontend and backend can proceed in parallel. Integration only becomes ready when both are done.

## Migration Pattern

### Pattern: Large-Scale Code Migration

```
1. Create migration epic
2. Create prep task (not blocking)
3. Create migration tasks with sequence
4. Create verification task
5. Create cleanup task
```

### Example: Database Migration

```bash
# Epic
bd create "Epic: Migrate to PostgreSQL" -t epic -p 0 --json
# Returns: bd-mig-1

# Preparation (can start immediately)
bd create "Set up PostgreSQL staging environment" -t task -p 1 --json
bd create "Write data migration scripts" -t task -p 1 --json
bd create "Document rollback procedure" -t task -p 2 --json

# Migration sequence
bd create "Backup production MySQL" -t task -p 0 --json           # bd-mig-5
bd create "Run migration scripts on staging" -t task -p 0 --json  # bd-mig-6
bd create "Verify staging data integrity" -t task -p 0 --json     # bd-mig-7
bd create "Switch production to PostgreSQL" -t task -p 0 --json   # bd-mig-8
bd create "Verify production data integrity" -t task -p 0 --json  # bd-mig-9

# Set up sequence
bd dep add bd-mig-5 bd-mig-6  # Backup before migrate
bd dep add bd-mig-6 bd-mig-7  # Migrate before verify
bd dep add bd-mig-7 bd-mig-8  # Verify staging before prod switch
bd dep add bd-mig-8 bd-mig-9  # Switch before verify prod

# Prep tasks block migration start
bd dep add bd-mig-2 bd-mig-5  # Staging env before backup
bd dep add bd-mig-3 bd-mig-6  # Scripts before migration
bd dep add bd-mig-4 bd-mig-5  # Rollback doc before any migration

# All under epic
bd dep add bd-mig-1 bd-mig-2 --type parent-child
bd dep add bd-mig-1 bd-mig-3 --type parent-child
bd dep add bd-mig-1 bd-mig-4 --type parent-child
bd dep add bd-mig-1 bd-mig-5 --type parent-child
bd dep add bd-mig-1 bd-mig-6 --type parent-child
bd dep add bd-mig-1 bd-mig-7 --type parent-child
bd dep add bd-mig-1 bd-mig-8 --type parent-child
bd dep add bd-mig-1 bd-mig-9 --type parent-child
```

## Recurring Work

### Pattern: Template for Repeated Tasks

```
1. Create template issue (don't close)
2. Clone for each occurrence
3. Close clones when done
4. Keep template for future
```

### Example: Weekly Review

```bash
# Template
bd create "TEMPLATE: Weekly code review" -t task -p 3 \
  -d "Weekly code review checklist:
      - [ ] Review open PRs
      - [ ] Check test coverage
      - [ ] Update documentation
      - [ ] Clear stale branches" \
  -l template \
  --json
# Returns: bd-tmpl-1

# Each week, create instance
bd create "Week 23 code review" -t task -p 2 \
  -d "Copy from bd-tmpl-1" \
  --json

# Work through and close
bd update bd-w23-1 --status in_progress --json
# ... do review ...
bd close bd-w23-1 --reason "Week 23 review complete" --json
```

## Anti-Patterns to Avoid

### Anti-Pattern: Everything Blocks Everything

**Problem**: Creating excessive blocking dependencies

```bash
# Bad: Linear chain of unrelated work
bd dep add task-1 task-2  # Not actually dependent
bd dep add task-2 task-3  # Not actually dependent
```

**Solution**: Only use `blocks` for true technical dependencies. Use `related` for connections.

### Anti-Pattern: Giant Issues

**Problem**: Issues too large to track progress

```bash
# Bad
bd create "Build the entire authentication system" -t epic -p 1 --json
```

**Solution**: Break down into implementable tasks. Epic should be container, not work item.

### Anti-Pattern: No Dependencies

**Problem**: Flat list with no structure

```bash
# Bad: 50 unrelated tasks
bd create "Task 1" --json
bd create "Task 2" --json
# ... no relationships ...
```

**Solution**: Group with parent-child, sequence with blocks, connect with related.

### Anti-Pattern: Duplicate Issues

**Problem**: Same work tracked multiple times

**Solution**: Search before creating: `bd list --title-contains "keyword" --json`

## Summary

**Key patterns**:
1. **Epic Decomposition**: Break large work into sequenced tasks
2. **Research and Decision**: Structure investigation with findings
3. **Bug Triage**: Systematic handling with root cause
4. **Side Quest**: Capture discoveries without losing context
5. **Parallel Workstreams**: Manage independent tracks
6. **Migration**: Sequenced large-scale changes
7. **Recurring Work**: Templates for repeated tasks

**Golden rules**:
- Use `blocks` only for true dependencies
- Break work into atomic, completable units
- Link related work for context
- Track discoveries with `discovered-from`

## See Also

- [DEPENDENCIES.md](DEPENDENCIES.md) - Dependency types in detail
- [BOUNDARIES.md](BOUNDARIES.md) - When to use bd
- [WORKFLOWS.md](WORKFLOWS.md) - Step-by-step guides
- [ISSUE_CREATION.md](ISSUE_CREATION.md) - Writing good issues
