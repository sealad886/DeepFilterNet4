# Integration Patterns

How bd integrates with other tools, systems, and workflows.

## Version Control Integration

### Git Hooks

bd automatically syncs with git through optional hooks.

**Post-commit hook**: Updates issue status based on commit messages

```bash
# Enable git integration
bd config set git.auto_sync true
bd config set git.commit_prefix "bd-"
```

**Commit message patterns recognized**:
- `bd-42: Fix authentication bug` - Links commit to issue
- `Fixes bd-42` - Closes issue on commit
- `Closes bd-42` - Closes issue on commit
- `Refs bd-42` - References without closing

### Branch Naming

bd can track which issues are being worked on in which branches:

```bash
# Branch naming convention
git checkout -b bd-42-fix-auth

# bd can find issues associated with current branch
bd list --branch HEAD --json
```

### Multi-repo Workflows

When working across multiple repositories:

```bash
# Set source_repo for issues
bd create "Fix shared library bug" -t bug -p 1 --source-repo shared-lib --json

# Filter by repository
bd list --source-repo shared-lib --json
```

## IDE Integration

### VS Code

bd works in VS Code through the terminal or via Claude extensions.

**Workflow**:
1. Open terminal in VS Code
2. Use bd commands normally
3. Issues tracked in `.beads/` directory

### JetBrains IDEs

Same terminal-based workflow applies to JetBrains products.

**Tip**: Add a terminal profile for bd-focused work:
```bash
alias bdr="bd ready --json"
alias bdc="bd create"
alias bds="bd sync"
```

## CI/CD Integration

### Issue Validation in CI

Ensure issues are properly tracked before merge:

```yaml
# .github/workflows/issue-check.yml
name: Issue Check
on: [pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check for issue reference
        run: |
          # Verify PR references a bd issue
          PR_BODY="${{ github.event.pull_request.body }}"
          if ! echo "$PR_BODY" | grep -q "bd-[a-zA-Z0-9]"; then
            echo "Warning: No bd issue referenced in PR"
          fi
```

### Auto-close on Merge

```yaml
# .github/workflows/auto-close.yml
name: Auto Close Issues
on:
  pull_request:
    types: [closed]
jobs:
  close:
    if: github.event.pull_request.merged == true
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install bd
        run: cargo install beads
      - name: Close referenced issues
        run: |
          # Extract issue IDs from PR and close them
          ISSUES=$(echo "${{ github.event.pull_request.body }}" | grep -oE "bd-[a-zA-Z0-9]+")
          for issue in $ISSUES; do
            bd close $issue --reason "Closed via PR merge" --json
          done
```

## Claude Integration

### Claude Code (Cline)

bd is designed to work with Claude Code agents:

```bash
# Agent reads current work
bd ready --json

# Agent claims work
bd update bd-42 --status in_progress --json

# Agent completes work
bd close bd-42 --reason "Implemented feature" --json
```

### Claude API

When building custom Claude integrations:

```python
# Example: Claude function calling with bd
tools = [
    {
        "name": "bd_ready",
        "description": "Get list of ready issues",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "bd_create",
        "description": "Create a new issue",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "type": {"type": "string"},
                "priority": {"type": "integer"}
            },
            "required": ["title"]
        }
    }
]

# Process Claude's tool use
if tool_name == "bd_ready":
    result = subprocess.run(["bd", "ready", "--json"], capture_output=True)
    return json.loads(result.stdout)
```

## MCP Server Integration

bd provides an MCP (Model Context Protocol) server for AI assistants:

```bash
# Start MCP server
bd mcp

# Server provides tools:
# - bd_ready: Get ready issues
# - bd_create: Create new issue
# - bd_update: Update issue status
# - bd_close: Close issue
# - bd_list: Search issues
# - bd_show: Get issue details
```

### MCP Configuration

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "bd": {
      "command": "bd",
      "args": ["mcp"],
      "env": {
        "BD_DATABASE": "/path/to/.beads/beads.db"
      }
    }
  }
}
```

## Project Management Tool Integration

### Linear Integration

Export bd issues to Linear:

```bash
# Export issues as Linear-compatible CSV
bd list --status open --format csv > issues.csv

# Import into Linear via API or UI
```

### Jira Integration

Sync with Jira (requires custom script):

```bash
# Example: Create Jira tickets from bd issues
bd list --status open --json | jq -r '.[] | "\(.id)\t\(.title)\t\(.priority)"' | while read id title priority; do
  # Create corresponding Jira ticket
  curl -X POST "$JIRA_URL/rest/api/2/issue" \
    -H "Authorization: Bearer $JIRA_TOKEN" \
    -d "{
      \"fields\": {
        \"summary\": \"[$id] $title\",
        \"priority\": {\"name\": \"P$priority\"}
      }
    }"
done
```

## Notification Integration

### Slack Notifications

```bash
# Notify on issue creation
bd create "Urgent bug fix needed" -t bug -p 0 --json | \
  jq -r '"New P0 bug: \(.title) (\(.id))"' | \
  curl -X POST -d "{\"text\": \"$(cat -)\"}" "$SLACK_WEBHOOK"
```

### Email Notifications

```bash
# Daily digest of ready work
bd ready --json | \
  jq -r '.[] | "- [\(.id)] \(.title) (P\(.priority))"' | \
  mail -s "bd Ready Issues" team@example.com
```

## Database Integration

### SQLite Access

bd uses SQLite internally. You can query directly for advanced use cases:

```bash
# Direct SQLite query (read-only recommended)
sqlite3 .beads/beads.db "SELECT id, title, status FROM issues WHERE status = 'open'"
```

**Warning**: Direct writes may cause sync issues. Use bd commands for modifications.

### Backup Integration

```bash
# Backup bd database
cp .beads/beads.db backups/beads-$(date +%Y%m%d).db

# Or use git (already integrated)
git add .beads/issues.jsonl
git commit -m "Backup bd issues"
```

## Custom Tool Integration

### Building Custom Tools

bd's JSON output makes integration straightforward:

```python
import subprocess
import json

def get_ready_issues():
    result = subprocess.run(
        ["bd", "ready", "--json"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

def create_issue(title, issue_type="task", priority=2):
    result = subprocess.run(
        ["bd", "create", title, "-t", issue_type, "-p", str(priority), "--json"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

def close_issue(issue_id, reason="Completed"):
    result = subprocess.run(
        ["bd", "close", issue_id, "--reason", reason, "--json"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)
```

### Webhook Integration

For event-driven workflows:

```bash
# Watch for changes and trigger webhooks
bd config set hooks.on_create "curl -X POST $WEBHOOK_URL -d @-"
bd config set hooks.on_close "curl -X POST $WEBHOOK_URL -d @-"
```

## Best Practices

### Keep It Simple

- Use native bd commands when possible
- Avoid complex integrations unless necessary
- JSON output covers most use cases

### Maintain Single Source of Truth

- bd database is the source of truth
- External systems should sync from bd, not to bd
- Use bd sync for git integration

### Handle Errors Gracefully

```bash
# Always check exit codes
if bd create "Issue" -p 1 --json; then
    echo "Created successfully"
else
    echo "Failed to create issue"
fi
```

### Test Integrations

- Use `--dry-run` flags when available
- Test in isolated environments first
- Monitor sync status with `bd info --json`

## See Also

- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Full command reference
- [WORKFLOWS.md](WORKFLOWS.md) - Common workflow patterns
- [AGENTS.md](AGENTS.md) - Agent-specific integration
