# Git Worktrees Integration

Using bd with git worktrees for parallel development environments.

## What Are Git Worktrees?

Git worktrees allow multiple working directories from the same repository:

```bash
# Main checkout
/project           # main branch

# Additional worktrees
/project-feature   # feature branch
/project-bugfix    # bugfix branch
```

Each worktree has its own working directory but shares the git database.

## bd and Worktrees

### Shared vs Separate Databases

**Option 1: Shared Database (Recommended)**

All worktrees share one bd database:

```
/project/.beads/beads.db      # Main database
/project-feature/.beads/      # Symlink to main
/project-bugfix/.beads/       # Symlink to main
```

```bash
# In feature worktree
ln -s ../project/.beads .beads

# All worktrees see same issues
bd ready --json  # Same everywhere
```

**Option 2: Separate Databases**

Each worktree has independent tracking:

```
/project/.beads/beads.db           # Main issues
/project-feature/.beads/beads.db   # Feature issues
/project-bugfix/.beads/beads.db    # Bugfix issues
```

Use when:
- Different teams on different worktrees
- Experiments that shouldn't affect main tracking
- Long-running branches with distinct scope

### Setting Up Shared Database

```bash
# Main repository has .beads/
cd /project
ls .beads/beads.db  # Exists

# Create worktree
git worktree add ../project-feature feature-branch

# Link to main database
cd ../project-feature
ln -s ../project/.beads .beads

# Verify
bd info --json  # Shows main database path
```

## Workflow Patterns

### Pattern 1: Branch-Per-Issue

Each issue gets its own worktree and branch.

```bash
# Main worktree: view and manage issues
cd /project
bd ready --json  # See bd-42 is ready

# Create worktree for issue
git worktree add ../project-bd-42 -b bd-42-fix-auth

# Link database
cd ../project-bd-42
ln -s ../project/.beads .beads

# Claim issue
bd update bd-42 --status in_progress --json

# Work in isolated environment
# ... make changes ...

# Complete
bd close bd-42 --reason "Fixed in branch bd-42-fix-auth" --json
cd /project
git merge bd-42-fix-auth
git worktree remove ../project-bd-42
```

### Pattern 2: Parallel Development

Multiple developers/agents work simultaneously.

```bash
# Developer 1: Main worktree
cd /project
bd update bd-42 --status in_progress --json
# Works on bd-42

# Developer 2: Feature worktree
cd /project-feature
bd update bd-43 --status in_progress --json
# Works on bd-43 in parallel

# Both see each other's progress via shared database
bd list --status in_progress --json
```

### Pattern 3: Experiment Isolation

Experimental work in isolated worktree.

```bash
# Create experiment worktree with separate database
git worktree add ../project-experiment -b experiment
cd ../project-experiment
mkdir .beads  # Fresh database, not linked

# Create experiment-specific issues
bd create "Experiment: Try new architecture" -t task -p 2 --json

# If experiment succeeds, migrate issues to main
bd export -o experiment-issues.jsonl
cd /project
bd import -i ../project-experiment/experiment-issues.jsonl
```

## Daemon Considerations

### Multiple Daemons

Each worktree path gets its own daemon if using daemon mode:

```bash
# Check daemons
bd daemons list --json

# Output might show:
# /project: daemon running (PID 1234)
# /project-feature: daemon running (PID 5678)
```

### Shared Database with Multiple Daemons

When worktrees share a database, coordinate carefully:

```bash
# Option 1: One daemon, multiple access points
cd /project
bd daemons start .  # Main daemon

cd /project-feature
bd --db /project/.beads/beads.db ready --json  # Use main db

# Option 2: Sandbox mode in worktrees
cd /project-feature
bd --sandbox ready --json  # Direct access, no daemon
```

## Sync Considerations

### Git Sync Across Worktrees

bd syncs to git, so worktrees stay coordinated:

```bash
# Main: make changes
cd /project
bd create "New issue" -p 1 --json
bd sync  # Commits to .beads/issues.jsonl

# Feature: get changes
cd /project-feature
git pull  # Gets .beads/issues.jsonl
bd import -i .beads/issues.jsonl  # Refresh database
```

### Conflict Resolution

If worktrees diverge:

```bash
# Feature worktree: local changes
bd create "Feature issue" -p 1 --json

# Main worktree: also has changes
bd create "Main issue" -p 1 --json

# Merge feature into main
cd /project
git merge feature-branch  # May conflict in .beads/issues.jsonl

# Resolve: JSONL is line-based, usually both can be kept
# Manual merge: take all unique lines
bd import -i .beads/issues.jsonl  # Re-import merged file
```

## Best Practices

### 1. Consistent Database Strategy

Choose shared or separate and stick with it:

```bash
# Document in README
# All worktrees should link to main .beads/
# or
# Each worktree maintains independent .beads/
```

### 2. Clear Ownership

Label issues by worktree/branch:

```bash
bd create "Feature: new login" -t feature -p 1 -l feature-branch --json
bd create "Bugfix: auth error" -t bug -p 0 -l hotfix-branch --json
```

### 3. Regular Sync

Sync frequently when sharing database:

```bash
# Before starting work
git pull
bd import -i .beads/issues.jsonl

# After completing work
bd sync
```

### 4. Clean Up Worktrees

Remove worktrees when done:

```bash
# After merging feature
cd /project
git worktree remove ../project-feature
git branch -d feature-branch

# Close related issues
bd close bd-42 --reason "Merged, worktree removed" --json
```

## Troubleshooting

### Database Not Found

```bash
# Check symlink
ls -la .beads

# Fix broken symlink
rm .beads
ln -s ../project/.beads .beads
```

### Stale Data

```bash
# Force refresh from JSONL
bd import --force -i .beads/issues.jsonl
```

### Conflicting Daemons

```bash
# Kill all daemons
bd daemons killall --json

# Use sandbox mode
bd --sandbox ready --json
```

### Diverged JSONL

```bash
# Export both versions
cd /project
bd export -o main-issues.jsonl

cd /project-feature
bd export -o feature-issues.jsonl

# Combine (JSONL is append-friendly)
cat main-issues.jsonl feature-issues.jsonl | sort -u > combined.jsonl

# Import combined
bd import -i combined.jsonl
```

## Example Setup Script

```bash
#!/bin/bash
# setup-worktree.sh

MAIN_DIR="/project"
BRANCH_NAME="$1"
WORKTREE_DIR="/project-$BRANCH_NAME"

# Create worktree
git -C "$MAIN_DIR" worktree add "$WORKTREE_DIR" -b "$BRANCH_NAME"

# Link to main database
ln -s "$MAIN_DIR/.beads" "$WORKTREE_DIR/.beads"

echo "Worktree created at $WORKTREE_DIR"
echo "Linked to main bd database"

# Find related issues
echo "Related issues:"
bd list --title-contains "$BRANCH_NAME" --json
```

## See Also

- [MOLECULES.md](MOLECULES.md) - Parallel work patterns
- [INTEGRATION_PATTERNS.md](INTEGRATION_PATTERNS.md) - Git integration
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
