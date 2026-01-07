# Issue Creation Best Practices

Guidelines for creating well-structured bd issues that maximize productivity.

## Core Principles

### 1. Atomic Issues

Each issue should represent a single, coherent piece of work.

**Good**:
```
Title: Add email validation to registration form
Type: task
Description: Implement client-side and server-side email format validation
```

**Bad**:
```
Title: Fix registration and also update the dashboard
Description: Multiple unrelated things
```

### 2. Actionable Titles

Titles should clearly indicate what needs to be done.

**Good**:
- "Fix null pointer exception in UserService.getProfile()"
- "Add pagination to /api/users endpoint"
- "Implement OAuth2 authorization code flow"

**Bad**:
- "Bug"
- "Auth stuff"
- "Work on the thing"

### 3. Clear Scope

Define what's in and out of scope explicitly.

**Good**:
```
Title: Add user avatar upload
Description: Allow users to upload profile avatars
Acceptance Criteria:
- [ ] Accept PNG/JPG files up to 2MB
- [ ] Resize to 200x200 on server
- [ ] Display in profile header
Out of scope: Avatar editing, GIF support
```

## Issue Templates

### Bug Report

```bash
bd create "Fix: [brief description of bug]" \
  -t bug \
  -p 1 \
  -d "**Problem**: What's broken
**Steps to reproduce**:
1. First step
2. Second step
3. Bug appears

**Expected**: What should happen
**Actual**: What actually happens
**Environment**: OS, browser, version" \
  --json
```

### Feature Request

```bash
bd create "Add [feature name]" \
  -t feature \
  -p 2 \
  -d "**Goal**: What problem does this solve
**User story**: As a [user], I want [capability] so that [benefit]
**Acceptance criteria**:
- [ ] Criterion 1
- [ ] Criterion 2

**Design notes**: Initial thoughts on implementation" \
  --json
```

### Research Task

```bash
bd create "Research: [topic]" \
  -t task \
  -p 2 \
  -d "**Question**: What we need to learn
**Why**: Why this matters
**Deliverable**: Expected output (comparison doc, recommendation, etc.)
**Time box**: Maximum time to spend" \
  --json
```

### Epic

```bash
bd create "Epic: [high-level goal]" \
  -t epic \
  -p 1 \
  -d "**Vision**: What we're building
**Success criteria**: How we know it's done
**Dependencies**: What must exist first
**Risks**: What could go wrong" \
  --json
```

## Priority Guidelines

### Priority 0 - Critical

Use for issues that block all other work or affect production.

- Security vulnerabilities
- Production outages
- Data corruption risks
- Build/deploy failures

### Priority 1 - High

Important work that should be addressed soon.

- Major features on roadmap
- Significant bugs affecting users
- Blockers for other high-priority work
- Performance issues affecting UX

### Priority 2 - Medium (Default)

Standard work that should be done in normal course.

- Regular features and improvements
- Minor bugs
- Code cleanup and refactoring
- Documentation updates

### Priority 3 - Low

Nice-to-have work that can wait.

- Polish and optimization
- Minor UX improvements
- Technical debt that isn't urgent
- Experimental features

### Priority 4 - Backlog

Ideas and future possibilities.

- Speculative features
- Long-term improvements
- Research topics
- "Someday/maybe" items

## Writing Good Descriptions

### Structure

```
**Context**: Why does this issue exist?

**Problem/Goal**: What needs to change?

**Approach** (optional): Initial ideas on how to solve

**Acceptance Criteria**:
- [ ] Testable criterion 1
- [ ] Testable criterion 2

**Out of Scope** (optional): What this issue does NOT cover
```

### Be Specific

**Good**:
```
Add rate limiting to the /api/login endpoint.
Limit to 5 attempts per minute per IP address.
Return 429 status code when exceeded.
```

**Bad**:
```
Make login more secure.
```

### Include Context

**Good**:
```
Users are reporting slow dashboard loads. Initial investigation shows
the /api/metrics endpoint returning 50MB of data. This issue addresses
adding pagination to reduce payload size.

See related issue bd-42 for backend optimization.
```

**Bad**:
```
Dashboard is slow, fix it.
```

## Using Labels Effectively

### Common Label Patterns

**Status labels**:
- `blocked` - Waiting on external dependency
- `needs-review` - Ready for code review
- `needs-design` - Needs design input

**Category labels**:
- `frontend`
- `backend`
- `infrastructure`
- `documentation`

**Special labels**:
- `good-first-issue` - Good for new contributors
- `tech-debt` - Technical debt reduction
- `security` - Security-related

### Creating with Labels

```bash
bd create "Fix XSS vulnerability in comments" \
  -t bug \
  -p 0 \
  -l security,frontend \
  --json
```

## Linking Related Work

### When to Create Dependencies

**Use `blocks`**:
- When one issue literally cannot start until another is done
- Technical prerequisites
- Sequential workflow steps

**Use `related`**:
- Related but independent issues
- Alternative approaches
- Similar work in different areas

**Use `discovered-from`**:
- Issues found while working on something else
- Side quests and tangents
- Follow-up work

### Creating with Dependencies

```bash
# Create blocked issue
bd create "Add user preferences API" -p 2 --json
# Returns: bd-100

bd create "Add preferences UI" -p 2 --json
# Returns: bd-101

bd dep add bd-100 bd-101  # API blocks UI

# Create discovered issue in one command
bd create "Found: validation missing" -t bug -p 1 \
  --deps discovered-from:bd-100 --json
```

## Common Mistakes to Avoid

### 1. Too Vague

**Problem**: Issue lacks enough detail to act on.

**Solution**: Include specific acceptance criteria, reproduction steps, or technical details.

### 2. Too Broad

**Problem**: Issue scope is too large to complete in reasonable time.

**Solution**: Break down into smaller, atomic issues. Use epics for tracking.

### 3. Missing Type

**Problem**: Using default type when bug/feature/task distinction matters.

**Solution**: Always set appropriate type: bug, feature, task, epic, chore.

### 4. Wrong Priority

**Problem**: Everything is P0 or nothing is prioritized.

**Solution**: Use full range. Reserve P0 for true emergencies.

### 5. No Acceptance Criteria

**Problem**: No way to know when issue is done.

**Solution**: Include testable criteria that define completion.

### 6. Duplicate Issues

**Problem**: Creating issues that already exist.

**Solution**: Search before creating: `bd list --title-contains "keyword" --json`

## Batch Creation

### From Markdown File

Create multiple related issues from a markdown file:

```markdown
# feature-auth.md

## Login Flow
- Type: feature
- Priority: 1
- Description: Implement OAuth login flow

## Session Management
- Type: task
- Priority: 2
- Description: Add session timeout handling
- Depends on: Login Flow
```

```bash
bd create -f feature-auth.md --json
```

### Programmatic Creation

For bulk imports or automation:

```bash
# Create multiple issues from JSON
cat issues.json | jq -c '.[]' | while read issue; do
  title=$(echo "$issue" | jq -r '.title')
  type=$(echo "$issue" | jq -r '.type // "task"')
  priority=$(echo "$issue" | jq -r '.priority // 2')
  bd create "$title" -t "$type" -p "$priority" --json
done
```

## Summary

**Good issues have**:
- Clear, actionable titles
- Specific descriptions with context
- Appropriate type and priority
- Testable acceptance criteria
- Relevant labels and dependencies

**Remember**:
- Atomic scope (one thing per issue)
- Enough detail to act on
- Clear definition of done
- Links to related work

This structure enables effective AI agent work, human collaboration, and long-term project tracking.

## See Also

- [BOUNDARIES.md](BOUNDARIES.md) - When to use bd vs TodoWrite
- [DEPENDENCIES.md](DEPENDENCIES.md) - Dependency types guide
- [PATTERNS.md](PATTERNS.md) - Common issue patterns
