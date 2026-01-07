# Troubleshooting Guide

Common issues and their solutions when using bd.

## Installation Issues

### bd: command not found

**Problem**: bd is not in your PATH.

**Solution**:
```bash
# Check if installed
which bd

# If not found, install
cargo install beads

# Or add cargo bin to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

### Permission denied

**Problem**: Can't write to .beads directory.

**Solution**:
```bash
# Check permissions
ls -la .beads/

# Fix permissions
chmod 755 .beads/
chmod 644 .beads/beads.db
```

## Database Issues

### No database found

**Problem**: bd can't find the database.

**Solution**:
```bash
# Check if .beads exists
ls -la .beads/

# Initialize if needed
mkdir -p .beads
bd create "Initial issue" -p 3 --json
```

### Database locked

**Problem**: Another process has the database locked.

**Solution**:
```bash
# Find locking process
lsof .beads/beads.db

# If daemon, restart it
bd daemons restart /path/to/workspace --json

# Or kill and retry
bd daemons killall --json
```

### Staleness Error

**Problem**: Database out of sync with JSONL.

**Solution**:
```bash
# Force import from JSONL
bd import -i .beads/issues.jsonl

# If import shows "0 created, 0 updated" but staleness persists
bd import --force -i .beads/issues.jsonl

# Emergency: skip staleness check
bd --allow-stale ready --json
```

### Database Corruption

**Problem**: SQLite database is corrupted.

**Solution**:
```bash
# Rebuild from JSONL (source of truth)
rm .beads/beads.db
bd import -i .beads/issues.jsonl

# If JSONL is also bad, recover from git
git checkout HEAD -- .beads/issues.jsonl
bd import -i .beads/issues.jsonl
```

## Sync Issues

### Sync fails with git error

**Problem**: bd sync can't push/pull.

**Solution**:
```bash
# Check git status
git status

# Resolve any conflicts
git add .beads/issues.jsonl
git commit -m "Resolve bd sync conflict"

# Retry sync
bd sync
```

### JSONL conflicts

**Problem**: Git merge conflict in issues.jsonl.

**Solution**:
```bash
# Accept both versions (JSONL is append-only friendly)
git checkout --ours .beads/issues.jsonl
git checkout --theirs .beads/issues.jsonl

# Or manually merge
# Each line is independent, take all unique lines

# Then import
bd import -i .beads/issues.jsonl
```

### Changes not syncing

**Problem**: Changes made but not appearing in git.

**Solution**:
```bash
# Force immediate sync
bd sync

# Check daemon status
bd info --json

# If daemon not running, changes may be batched
# Wait for auto-sync or manually sync
```

## Daemon Issues

### Daemon not starting

**Problem**: bd daemon fails to start.

**Solution**:
```bash
# Check logs
bd daemons logs /path/to/workspace

# Check for port conflicts
lsof -i :7890  # Default daemon port

# Start manually
bd daemons restart /path/to/workspace --json
```

### Daemon version mismatch

**Problem**: Daemon running old version.

**Solution**:
```bash
# Check health
bd daemons health --json

# Restart with new version
bd daemons killall --json
# Next command will start new daemon
```

### Sandbox mode not detected

**Problem**: bd not auto-detecting sandboxed environment.

**Solution**:
```bash
# Explicitly enable sandbox mode
bd --sandbox ready --json

# Or set environment
export BD_NO_DAEMON=true
export BD_NO_AUTO_FLUSH=true
export BD_NO_AUTO_IMPORT=true
```

## Command Issues

### Invalid issue ID

**Problem**: Issue ID not recognized.

**Solution**:
```bash
# Check exact ID
bd list --json | jq '.[].id'

# IDs are case-sensitive
bd show bd-ABC123 --json  # Not bd-abc123
```

### Missing required fields

**Problem**: Create fails with missing field error.

**Solution**:
```bash
# Always quote titles
bd create "My issue title" -p 1 --json

# Include required flags
bd create "Title" -t bug -p 1 --json
```

### Dependency cycle

**Problem**: Can't add dependency due to cycle.

**Solution**:
```bash
# Check existing dependencies
bd dep tree <id>

# Remove conflicting dependency first
bd dep remove <from> <to>

# Then add new one
bd dep add <from> <to>
```

## Performance Issues

### Slow queries

**Problem**: bd commands taking too long.

**Solution**:
```bash
# Check database size
ls -lh .beads/beads.db

# Clean up closed issues
bd admin cleanup --older-than 90 --force --json

# Compact old issues
bd admin compact --auto --all --tier 1
```

### Memory issues

**Problem**: bd using too much memory.

**Solution**:
```bash
# Use pagination for large lists
bd list --limit 50 --json

# Process in batches
bd list --status open --json | head -100
```

## Migration Issues

### Old database format

**Problem**: Database from older bd version.

**Solution**:
```bash
# Run migration
bd migrate --dry-run  # Preview
bd migrate            # Execute

# If issues, check migration plan
bd migrate --inspect --json
```

### Missing fields after upgrade

**Problem**: New fields not present in old issues.

**Solution**:
```bash
# Export and reimport
bd export -o backup.jsonl
bd import -i backup.jsonl

# Or update individual issues
bd update <id> --design "" --json  # Initialize field
```

## Common Error Messages

### "Issue not found"

Issue ID doesn't exist in database.

```bash
# List all issues to find correct ID
bd list --json
```

### "Blocked by open issues"

Can't close issue that blocks open issues.

```bash
# Check what it blocks
bd dep tree <id>

# Either close blockers first or remove dependency
bd dep remove <id> <blocked-id>
```

### "Stale database"

Database differs from JSONL source of truth.

```bash
# Import to refresh
bd import -i .beads/issues.jsonl

# Or with force
bd import --force -i .beads/issues.jsonl
```

### "Daemon connection refused"

Can't connect to bd daemon.

```bash
# Check if running
bd info --json

# Restart if needed
bd daemons restart /path/to/workspace --json

# Or run without daemon
bd --no-daemon ready --json
```

### "Permission denied" on sync

Git push/pull failing.

```bash
# Check git remote
git remote -v

# Check credentials
git fetch origin

# Fix authentication
ssh -T git@github.com  # For SSH
git credential fill    # For HTTPS
```

## Debug Mode

For detailed debugging:

```bash
# Enable verbose logging
RUST_LOG=debug bd ready --json

# Check daemon logs
bd daemons logs /path/to/workspace -f

# Get full system info
bd info --schema --json
```

## Getting Help

If issues persist:

1. Check bd version: `bd --version`
2. Check database info: `bd info --json`
3. Check daemon health: `bd daemons health --json`
4. Review daemon logs: `bd daemons logs . -n 100`
5. Try sandbox mode: `bd --sandbox ready --json`

## See Also

- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Full command reference
- [STATIC_DATA.md](STATIC_DATA.md) - Error codes and exit values
- [WORKFLOWS.md](WORKFLOWS.md) - Proper workflows to avoid issues
