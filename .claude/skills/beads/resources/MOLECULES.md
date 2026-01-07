# Molecules: Parallel Agent Coordination

Multi-agent work patterns using bd's molecule system.

## What Are Molecules?

A **molecule** is a coordinated unit of parallel agent work:

- Multiple agents working simultaneously
- Shared issue database for coordination
- Dependencies prevent conflicts
- Merge strategy for combining results

Molecules enable faster completion of large tasks by distributing work across agents while maintaining coherence.

## Core Concepts

### Bonding Agents

Agents "bond" to form a molecule when they:
1. Share the same bd database
2. Work on related issues with dependencies
3. Follow the same merge strategy

### Atomic Operations

Each agent's work is atomic:
- Complete their assigned issue(s)
- Don't interfere with other agents' work
- Results can be cleanly merged

### Valence (Parallelism Capacity)

How many agents can work in parallel depends on:
- Independence of issues (no blocking dependencies)
- File-level conflicts (agents modifying same files)
- Resource constraints (API limits, compute)

## Creating a Molecule

### Step 1: Identify Parallelizable Work

Work is parallelizable when:
- Issues don't block each other
- Issues modify different files
- Issues don't share mutable state

**Example**: API endpoints for different resources
```
bd-1: "Add /users endpoint"     # modifies users.py
bd-2: "Add /products endpoint"  # modifies products.py
bd-3: "Add /orders endpoint"    # modifies orders.py
```

All three can proceed in parallel.

### Step 2: Create Issues with Dependencies

```bash
# Create parallel issues (no blocking dependencies between them)
bd create "Add /users endpoint" -t task -p 2 --json
bd create "Add /products endpoint" -t task -p 2 --json
bd create "Add /orders endpoint" -t task -p 2 --json

# Add related links (not blocking)
bd dep add bd-1 bd-2 --type related
bd dep add bd-2 bd-3 --type related
bd dep add bd-1 bd-3 --type related
```

### Step 3: Assign to Agents

Each agent claims their issue:

```bash
# Agent 1
bd update bd-1 --status in_progress --assignee agent-1 --json

# Agent 2
bd update bd-2 --status in_progress --assignee agent-2 --json

# Agent 3
bd update bd-3 --status in_progress --assignee agent-3 --json
```

### Step 4: Work in Parallel

Agents work independently, periodically syncing:

```bash
# Each agent syncs periodically
bd sync

# Check for conflicts or updates
bd ready --json
bd list --status in_progress --json
```

### Step 5: Merge Results

When agents complete:

```bash
# Each agent closes their issue
bd close bd-1 --reason "Implemented" --json

# Sync to share results
bd sync

# Main agent verifies all complete
bd list --status open --json  # Should show remaining work
```

## Molecule Patterns

### Pattern 1: File-Parallel

Each agent works on different files.

**Use case**: Adding features to different modules

```
Agent 1: auth/login.py, auth/logout.py
Agent 2: users/profile.py, users/settings.py
Agent 3: products/catalog.py, products/cart.py
```

**Merge strategy**: Automatic (no conflicts)

### Pattern 2: Test-Parallel

Each agent writes tests for different components.

**Use case**: Expanding test coverage

```
Agent 1: test_auth.py (testing auth module)
Agent 2: test_users.py (testing users module)
Agent 3: test_products.py (testing products module)
```

**Merge strategy**: Automatic (independent test files)

### Pattern 3: Research-Parallel

Each agent researches different aspects.

**Use case**: Technology evaluation

```
Agent 1: Research Redis caching options
Agent 2: Research CDN providers
Agent 3: Research database optimization
```

**Merge strategy**: Manual synthesis (combine findings)

### Pattern 4: Pipeline-Parallel

Agents work on different pipeline stages.

**Use case**: Data processing pipeline

```
Agent 1: Data ingestion (stage 1)
Agent 2: Data transformation (stage 2, blocked by stage 1)
Agent 3: Data output (stage 3, blocked by stage 2)
```

**Merge strategy**: Sequential (pass outputs between stages)

## Conflict Resolution

### File Conflicts

When agents modify the same file:

1. **Detect early**: Check issue descriptions for file lists
2. **Partition work**: Assign different file sections
3. **Resolve on merge**: Use git merge strategies

### Logic Conflicts

When agents make incompatible changes:

1. **Document assumptions**: Each agent notes their approach
2. **Review before merge**: Compare approaches
3. **Select winner**: Choose best implementation

### State Conflicts

When agents depend on shared state:

1. **Lock resources**: Use bd labels to mark in-use resources
2. **Queue access**: Use blocking dependencies
3. **Atomic updates**: Complete all state changes together

## Best Practices

### 1. Clear Boundaries

Define exactly what each agent owns:

```bash
bd create "Add user API" -t task -p 2 \
  -d "Files: src/api/users.py, tests/test_users.py
      NOT: src/api/auth.py (owned by bd-10)" \
  --json
```

### 2. Regular Syncs

Sync frequently to catch conflicts early:

```bash
# Every agent should sync periodically
bd sync  # Push/pull changes

# Check for new work or changes
bd ready --json
bd list --updated-after $(date -v-1H +%Y-%m-%dT%H:%M:%S) --json
```

### 3. Progress Visibility

Update status so other agents know:

```bash
# Starting work
bd update bd-42 --status in_progress --json

# Hit a blocker
bd label add bd-42 blocked --json

# Completed
bd close bd-42 --reason "Done" --json
bd sync  # Important: sync immediately
```

### 4. Explicit Handoffs

When work passes between agents:

```bash
# Agent 1 completes prerequisite
bd close bd-1 --reason "API ready for integration" --json
bd sync

# Agent 2 picks up blocked work
bd ready --json  # bd-2 now appears
bd update bd-2 --status in_progress --json
```

## Molecule Lifecycle

### Formation

```
1. Identify parallelizable work
2. Create issues with appropriate dependencies
3. Assign agents to non-blocking issues
4. Agents begin work
```

### Active Phase

```
1. Agents work independently
2. Regular syncs share progress
3. Completed work unblocks dependent issues
4. Agents claim newly-unblocked work
```

### Dissolution

```
1. All issues closed
2. Final sync ensures all changes merged
3. Results verified
4. Molecule complete
```

## Example: Full Molecule Workflow

### Setup (Orchestrator Agent)

```bash
# Create epic
bd create "API Expansion" -t epic -p 1 --json
# Returns: bd-100

# Create parallel tasks
bd create "Users API" -p 2 --json       # bd-101
bd create "Products API" -p 2 --json    # bd-102
bd create "Orders API" -p 2 --json      # bd-103

# Create dependent integration task
bd create "API Integration Tests" -p 2 --json  # bd-104

# Set dependencies
bd dep add bd-100 bd-101 --type parent-child
bd dep add bd-100 bd-102 --type parent-child
bd dep add bd-100 bd-103 --type parent-child
bd dep add bd-100 bd-104 --type parent-child

# Integration tests blocked by all three APIs
bd dep add bd-101 bd-104
bd dep add bd-102 bd-104
bd dep add bd-103 bd-104

bd sync
```

### Execution (Parallel Agents)

```bash
# Agent 1
bd update bd-101 --status in_progress --json
# ... implement users API ...
bd close bd-101 --reason "Users API complete" --json
bd sync

# Agent 2 (simultaneously)
bd update bd-102 --status in_progress --json
# ... implement products API ...
bd close bd-102 --reason "Products API complete" --json
bd sync

# Agent 3 (simultaneously)
bd update bd-103 --status in_progress --json
# ... implement orders API ...
bd close bd-103 --reason "Orders API complete" --json
bd sync
```

### Integration (After All Complete)

```bash
# Orchestrator checks status
bd ready --json
# bd-104 now ready (all blockers closed)

# Agent 4 (or orchestrator)
bd update bd-104 --status in_progress --json
# ... write integration tests ...
bd close bd-104 --reason "Integration tests pass" --json

# Close epic
bd close bd-100 --reason "API expansion complete" --json
bd sync
```

## See Also

- [AGENTS.md](AGENTS.md) - Agent workflow basics
- [DEPENDENCIES.md](DEPENDENCIES.md) - Dependency types
- [WORKFLOWS.md](WORKFLOWS.md) - Common workflow patterns
- [WORKTREES.md](WORKTREES.md) - Git worktree integration
