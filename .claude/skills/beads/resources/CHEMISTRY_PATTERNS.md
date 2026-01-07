# Chemistry Patterns

Advanced patterns using bd's molecule and bonding metaphors for complex agent coordination.

## The Chemistry Model

bd uses chemistry metaphors for multi-agent coordination:

| Chemistry Term | bd Concept |
|----------------|------------|
| **Atom** | Individual issue |
| **Molecule** | Coordinated unit of parallel work |
| **Bond** | Dependency between issues |
| **Valence** | Parallelism capacity |
| **Reaction** | Work transformation |
| **Catalyst** | Agent that enables work |

## Bonding Patterns

### Ionic Bonds (Strong Dependencies)

Use for strict sequential requirements.

```bash
# Issue A must complete before B can start
bd dep add bd-a bd-b  # blocks dependency
```

**Characteristics**:
- One-way dependency
- Strict ordering
- Automatic unblocking

### Covalent Bonds (Shared Work)

Use for issues that share context but can proceed independently.

```bash
# Issues share context but don't block
bd dep add bd-a bd-b --type related
```

**Characteristics**:
- Bidirectional awareness
- No blocking
- Shared context

### Metallic Bonds (Flexible Sharing)

Use for work pools where any agent can contribute.

```bash
# Create pool of similar tasks
bd create "Process file batch 1" -t task -p 2 -l processing-pool --json
bd create "Process file batch 2" -t task -p 2 -l processing-pool --json
bd create "Process file batch 3" -t task -p 2 -l processing-pool --json

# Link as related
bd dep add bd-b1 bd-b2 --type related
bd dep add bd-b2 bd-b3 --type related
bd dep add bd-b1 bd-b3 --type related

# Any agent can claim from pool
bd list --label processing-pool --status open --json
```

**Characteristics**:
- Interchangeable agents
- Flexible assignment
- Pool-based work

## Reaction Patterns

### Synthesis (Combining Work)

Multiple inputs combine into one output.

```bash
# Inputs
bd create "Research component A" -t task -p 2 --json  # bd-a
bd create "Research component B" -t task -p 2 --json  # bd-b
bd create "Research component C" -t task -p 2 --json  # bd-c

# Synthesis output
bd create "Combined analysis" -t task -p 1 --json     # bd-out

# Bonds
bd dep add bd-a bd-out
bd dep add bd-b bd-out
bd dep add bd-c bd-out

# When A, B, C complete → out becomes ready
```

### Decomposition (Breaking Down Work)

One input breaks into multiple outputs.

```bash
# Input
bd create "Epic: User management" -t epic -p 1 --json  # bd-epic

# Decomposition
bd create "User registration" -t task -p 1 --json      # bd-1
bd create "User authentication" -t task -p 1 --json    # bd-2
bd create "User profile" -t task -p 1 --json           # bd-3

# Parent-child bonds
bd dep add bd-epic bd-1 --type parent-child
bd dep add bd-epic bd-2 --type parent-child
bd dep add bd-epic bd-3 --type parent-child
```

### Exchange (Trading Information)

Two agents exchange outputs.

```bash
# Agent 1's work
bd create "Design API schema" -t task -p 1 --json      # bd-api

# Agent 2's work
bd create "Design database schema" -t task -p 1 --json # bd-db

# Integration point
bd create "Integrate API with DB" -t task -p 1 --json  # bd-int

# Both feed into integration
bd dep add bd-api bd-int
bd dep add bd-db bd-int
```

### Catalysis (Enabling Work)

A catalyst issue enables other work without being consumed.

```bash
# Catalyst: Setup task that enables multiple workstreams
bd create "Set up development environment" -t task -p 0 --json  # bd-setup

# Multiple workstreams enabled
bd create "Develop feature A" -t feature -p 1 --json   # bd-a
bd create "Develop feature B" -t feature -p 1 --json   # bd-b
bd create "Develop feature C" -t feature -p 1 --json   # bd-c

# Setup enables all (but isn't "consumed")
bd dep add bd-setup bd-a
bd dep add bd-setup bd-b
bd dep add bd-setup bd-c

# Catalyst stays open as reference, or close it:
bd close bd-setup --reason "Environment ready for all features" --json
```

## Molecular Structures

### Linear Molecule (Pipeline)

Sequential processing through stages.

```
A → B → C → D
```

```bash
bd create "Ingest data" -t task -p 1 --json      # bd-a
bd create "Transform data" -t task -p 1 --json   # bd-b
bd create "Validate data" -t task -p 1 --json    # bd-c
bd create "Store data" -t task -p 1 --json       # bd-d

bd dep add bd-a bd-b
bd dep add bd-b bd-c
bd dep add bd-c bd-d
```

### Branched Molecule (Fork/Join)

Parallel paths that converge.

```
    ┌─ B ─┐
A ──┼─ C ─┼── E
    └─ D ─┘
```

```bash
bd create "Initial analysis" -t task -p 1 --json       # bd-a
bd create "Path B analysis" -t task -p 2 --json        # bd-b
bd create "Path C analysis" -t task -p 2 --json        # bd-c
bd create "Path D analysis" -t task -p 2 --json        # bd-d
bd create "Synthesize findings" -t task -p 1 --json    # bd-e

# Fork
bd dep add bd-a bd-b
bd dep add bd-a bd-c
bd dep add bd-a bd-d

# Join
bd dep add bd-b bd-e
bd dep add bd-c bd-e
bd dep add bd-d bd-e
```

### Ring Molecule (Iterative)

Work that cycles through phases.

```
A → B → C → D
↑           ↓
└───────────┘
```

**Note**: bd doesn't support circular dependencies. Model iterations as:

```bash
# Iteration 1
bd create "Iteration 1: Design" -t task -p 1 --json
bd create "Iteration 1: Implement" -t task -p 1 --json
bd create "Iteration 1: Review" -t task -p 1 --json
bd dep add bd-i1d bd-i1i
bd dep add bd-i1i bd-i1r

# Iteration 2 (blocked by iteration 1 review)
bd create "Iteration 2: Design" -t task -p 1 --json
bd dep add bd-i1r bd-i2d
# ... continue pattern
```

### Complex Molecule (Network)

Interconnected work with multiple dependencies.

```bash
# Create issues
bd create "Core library" -t task -p 0 --json          # bd-core
bd create "Auth module" -t task -p 1 --json           # bd-auth
bd create "API module" -t task -p 1 --json            # bd-api
bd create "UI components" -t task -p 1 --json         # bd-ui
bd create "Integration layer" -t task -p 1 --json     # bd-int
bd create "End-to-end tests" -t task -p 2 --json      # bd-e2e

# Complex dependencies
bd dep add bd-core bd-auth
bd dep add bd-core bd-api
bd dep add bd-core bd-ui
bd dep add bd-auth bd-int
bd dep add bd-api bd-int
bd dep add bd-ui bd-int
bd dep add bd-int bd-e2e
```

## Equilibrium States

### Stable State

All work is either:
- Closed (complete)
- Open with no blockers (ready)
- In progress (being worked)

```bash
# Check stability
bd ready --json        # Ready work
bd list --status in_progress --json  # Active work
# If both empty and no closed today, system is stable
```

### Excited State

Work is blocked waiting for external input.

```bash
# Identify excited (blocked) state
bd list --label blocked --json
```

### Metastable State

Work can proceed but is paused for strategic reasons.

```bash
# Mark as on-hold
bd label add bd-42 on-hold --json
bd update bd-42 --notes "Paused: waiting for stakeholder decision" --json
```

## Catalyst Patterns

### Shared Resource Catalyst

A resource that enables multiple independent work streams.

```bash
# Catalyst: API credentials
bd create "Obtain API credentials" -t task -p 0 --json  # bd-creds

# Dependent features
bd create "Feature: OAuth login" -t feature -p 1 --json
bd create "Feature: Data sync" -t feature -p 1 --json
bd create "Feature: Analytics" -t feature -p 2 --json

# All blocked by credentials
bd dep add bd-creds bd-oauth
bd dep add bd-creds bd-sync
bd dep add bd-creds bd-analytics
```

### Knowledge Catalyst

Research that enables implementation decisions.

```bash
# Catalyst: Research
bd create "Research: Performance requirements" -t task -p 1 --json

# Enabled work
bd create "Design caching strategy" -t task -p 1 --json
bd create "Choose database" -t task -p 1 --json
bd create "Plan scaling approach" -t task -p 1 --json

# Research enables all design decisions
bd dep add bd-research bd-cache
bd dep add bd-research bd-db
bd dep add bd-research bd-scale
```

## Best Practices

### 1. Match Pattern to Problem

- Linear: Sequential processing
- Branched: Parallel exploration
- Synthesis: Combining findings
- Catalyst: Shared enablers

### 2. Minimize Bond Count

Keep dependencies simple:
- Direct dependencies only
- Avoid transitive chains where possible
- Group related work under epics

### 3. Document Molecular Structure

Add structure notes to epic issues:

```bash
bd update bd-epic --design "
Molecular structure:
- Core (bd-core) → enables all modules
- Modules (auth, api, ui) → parallel development
- Integration (bd-int) → synthesis point
- Tests (bd-e2e) → final validation
" --json
```

### 4. Monitor Reaction Progress

```bash
# Overall structure
bd dep tree bd-epic

# Status by phase
bd list --label phase-1 --json
bd list --label phase-2 --json
```

## See Also

- [MOLECULES.md](MOLECULES.md) - Basic molecule patterns
- [ASYNC_GATES.md](ASYNC_GATES.md) - Synchronization patterns
- [DEPENDENCIES.md](DEPENDENCIES.md) - Dependency types
- [PATTERNS.md](PATTERNS.md) - Common patterns
