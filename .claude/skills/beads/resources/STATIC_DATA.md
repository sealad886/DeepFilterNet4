# Static Data Reference

Fixed reference data for bd usage.

## Issue Types

| Type | Description | Use When |
|------|-------------|----------|
| `bug` | Something broken | Defect, error, unexpected behavior |
| `feature` | New functionality | Adding capabilities |
| `task` | Work item | Tests, docs, refactoring, research |
| `epic` | Large feature | Container for related issues |
| `chore` | Maintenance | Dependencies, tooling, cleanup |

## Priority Levels

| Priority | Name | Response Time | Examples |
|----------|------|---------------|----------|
| 0 | Critical | Immediate | Security vuln, production down, data loss |
| 1 | High | Same day | Major features, significant bugs |
| 2 | Medium | This week | Standard work, minor bugs |
| 3 | Low | When possible | Polish, optimization |
| 4 | Backlog | Future | Ideas, speculative |

## Issue Statuses

| Status | Meaning | Appears in `bd ready`? |
|--------|---------|------------------------|
| `open` | Not started | Yes (unless blocked) |
| `in_progress` | Being worked on | No |
| `closed` | Completed | No |

## Dependency Types

| Type | Direction | Affects Ready? | Use For |
|------|-----------|----------------|---------|
| `blocks` | A blocks B | Yes | Prerequisites, sequence |
| `related` | Bidirectional | No | Context, similar work |
| `parent-child` | A contains B | No | Epics, hierarchy |
| `discovered-from` | Found B in A | No | Side quests, findings |

## CLI Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Database error |
| 4 | Network error (sync) |
| 5 | Git error |

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `BD_DATABASE` | Database path | `.beads/beads.db` |
| `BD_PREFIX` | Issue ID prefix | `bd-` |
| `BD_ACTOR` | Audit trail actor | Current user |
| `BD_NO_DAEMON` | Disable daemon | false |
| `BD_NO_AUTO_FLUSH` | Disable auto-export | false |
| `BD_NO_AUTO_IMPORT` | Disable auto-import | false |

## File Locations

| File | Purpose |
|------|---------|
| `.beads/beads.db` | SQLite database |
| `.beads/issues.jsonl` | JSONL export (git-tracked) |
| `.beads/config.toml` | Local configuration |
| `~/.config/bd/config.toml` | Global configuration |

## JSON Output Fields

### Issue Object

```json
{
  "id": "bd-abc123",
  "title": "Issue title",
  "description": "Full description",
  "type": "bug|feature|task|epic|chore",
  "status": "open|in_progress|closed",
  "priority": 0-4,
  "labels": ["label1", "label2"],
  "assignee": "username",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "closed_at": "2024-01-01T00:00:00Z",
  "design": "Design notes",
  "notes": "Additional notes",
  "acceptance_criteria": "Criteria text",
  "source_repo": "repo-name"
}
```

### Dependency Object

```json
{
  "from_id": "bd-abc123",
  "to_id": "bd-def456",
  "type": "blocks|related|parent-child|discovered-from"
}
```

## Common Label Conventions

### Status Labels
- `blocked` - Waiting on external dependency
- `needs-review` - Ready for code review
- `needs-design` - Needs design input
- `wip` - Work in progress

### Category Labels
- `frontend` - UI/client work
- `backend` - Server/API work
- `infrastructure` - DevOps/deployment
- `documentation` - Docs work
- `testing` - Test-related

### Special Labels
- `good-first-issue` - Good for new contributors
- `tech-debt` - Technical debt reduction
- `security` - Security-related
- `performance` - Performance optimization

## Version Compatibility

| bd Version | Key Features | Breaking Changes |
|------------|--------------|------------------|
| 0.43.0+ | Current stable | None |
| 0.40.0+ | MCP server | None |
| 0.35.0+ | Molecules | None |
| 0.30.0+ | Daemon mode | Config format |
| 0.21.0+ | Sandbox auto-detect | None |
| 0.20.0+ | JSONL sync | Database format |

## Quick Reference Card

### Most Used Commands

```bash
bd ready --json          # Find work
bd create "..." --json   # Create issue
bd update <id> --status in_progress --json  # Claim work
bd close <id> --json     # Complete work
bd sync                  # Force sync to git
```

### Filters

```bash
--status open|in_progress|closed
--priority 0-4
--type bug|feature|task|epic|chore
--label label1,label2
--assignee username
```

### Dependencies

```bash
bd dep add <from> <to>              # from blocks to
bd dep add <from> <to> --type related
bd dep tree <id>                    # Show tree
```

## See Also

- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Full command reference
- [DEPENDENCIES.md](DEPENDENCIES.md) - Dependency details
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Error handling
