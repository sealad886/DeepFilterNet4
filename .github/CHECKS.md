# Repository Checks Summary

This document provides an overview of all the checks and quality controls that have been added to the DeepFilterNet repository.

## Overview

The repository now includes comprehensive automated checks to ensure code quality, security, and maintainability. These checks run automatically on pull requests and pushes to the main branch.

## Automated Workflows

### 1. Python Linting (`python_lint.yml`)
**Triggers:** Push to main, Pull requests to main, Daily at 18:00 UTC

**Checks:**
- `black` - Code formatting (line length: 100)
- `isort` - Import sorting
- `flake8` - Code linting and style checking

**Purpose:** Ensures Python code follows consistent formatting and style guidelines.

### 2. Rust Linting (`rust_lint.yml`)
**Triggers:** Push to any branch, Daily at 18:00 UTC

**Checks:**
- `rustfmt` - Code formatting
- `clippy` - Linting and best practices
- Build tests for various packages
- Unit tests with coverage

**Purpose:** Ensures Rust code follows standard formatting and best practices.

### 3. CodeQL Security Scan (`codeql.yml`)
**Triggers:** Push to main, Pull requests to main, Weekly on Mondays at 02:30 UTC

**Checks:**
- Python security vulnerability scanning
- Rust security vulnerability scanning (via C++ analysis)
- Security and quality queries

**Purpose:** Automatically detects security vulnerabilities in the codebase.

### 4. Dependency Review (`dependency-review.yml`)
**Triggers:** Pull requests to main

**Checks:**
- Scans for vulnerable dependencies
- Checks for license compatibility
- Fails on moderate or higher severity vulnerabilities
- Comments summary on PR when issues are found

**Purpose:** Prevents introduction of vulnerable or incompatible dependencies.

### 5. PR Checks (`pr-checks.yml`)
**Triggers:** Pull requests to main

**Checks:**
- Detects modified lockfiles (Cargo.lock, poetry.lock)
- Validates commit message format (informational)
- Provides summary of PR status

**Purpose:** Provides additional context and validation for pull requests.

### 6. Integration Tests (`test_df.yml`)
**Triggers:** Workflow dispatch, Push to main, Pull requests to main, Weekly on Sundays

**Checks:**
- Builds and tests DeepFilterNet on multiple platforms (Ubuntu, Windows)
- Tests Python package functionality
- Tests Rust binary functionality
- Validates audio output quality with DNSMOS

**Purpose:** Ensures the software works correctly across different platforms.

### 7. Existing Workflows
The repository also maintains these existing workflows:
- `build_demo.yml` - Builds the demo application
- `build_wasm.yml` - Builds WebAssembly version
- `build_capi.yml` - Builds C API (weekly)
- `publish.yml` - Publishes releases
- `test_pypi_release.yml` - Tests PyPI releases
- `stale.yml` - Manages stale issues/PRs
- `combine-prs.yml` - Combines dependabot PRs

## Issue and PR Templates

### Pull Request Template
**Location:** `.github/pull_request_template.md`

**Features:**
- Type of change checklist
- Code quality checklist
- Testing requirements
- Documentation requirements

**Purpose:** Ensures contributors provide necessary information and follow best practices.

### Issue Templates

#### Bug Report (`bug_report.yml`)
**Features:**
- Structured bug report form
- Required fields: description, component, OS, version, reproduction steps
- Optional: logs and additional context

#### Feature Request (`feature_request.yml`)
**Features:**
- Structured feature request form
- Required fields: solution description, affected component
- Optional: problem statement, alternatives, additional context

#### Configuration (`config.yml`)
**Features:**
- Links to GitHub Discussions for questions
- Links to Security Advisories for vulnerabilities
- Allows blank issues

**Purpose:** Guides users to report issues in a structured way that helps maintainers respond effectively.

## Documentation

### 1. CONTRIBUTING.md
**Contents:**
- How to report bugs and suggest features
- Development setup instructions
- Coding standards for Python and Rust
- Testing guidelines
- Commit message conventions
- Overview of automated checks

**Purpose:** Provides comprehensive guide for contributors.

### 2. SECURITY.md
**Contents:**
- Supported versions
- How to report vulnerabilities
- Response timeline
- Security best practices
- Known security considerations

**Purpose:** Establishes clear security policy and reporting process.

### 3. CODEOWNERS
**Location:** `.github/CODEOWNERS`

**Features:**
- Defines code ownership for different parts of the repository
- Automatically requests reviews from appropriate owners
- Covers Rust components, Python components, GitHub workflows, and documentation

**Purpose:** Ensures the right people review changes to critical parts of the codebase.

### 4. README.md Updates
**Added Section:**
- Contributing guidelines reference
- Overview of code quality and security checks
- Instructions for running checks locally
- Links to issue templates
- Security policy reference

**Purpose:** Makes checks visible to contributors and users.

## How to Use These Checks

### For Contributors

1. **Before Creating a PR:**
   ```bash
   # Python formatting
   black .
   isort .
   flake8
   
   # Rust formatting
   cargo fmt
   cargo clippy --all-features -- -D warnings
   cargo test --all-features
   ```

2. **When Creating a PR:**
   - Fill out the PR template completely
   - Link related issues
   - Ensure all automated checks pass

3. **If Checks Fail:**
   - Review the error messages in the GitHub Actions tab
   - Fix issues locally and push updates
   - Ask for help in PR comments if needed

### For Maintainers

1. **Review Automated Check Results:**
   - All PRs must pass linting checks
   - CodeQL findings should be reviewed
   - Dependency review warnings should be investigated

2. **Use Issue Templates:**
   - Guide users to use appropriate templates
   - Structured information helps with triage

3. **Monitor Security:**
   - Review CodeQL alerts regularly
   - Respond to dependency vulnerability notifications
   - Keep SECURITY.md updated

## Check Status Requirements

### Required for Merge
- Python linting must pass
- Rust linting must pass
- CodeQL analysis must complete (warnings reviewed)
- Dependency review must pass (no high/critical vulnerabilities)

### Informational
- PR checks (lockfile changes, commit format)
- Integration tests (provide confidence but may be run post-merge)

## Future Enhancements

Potential additions to consider:

1. **Branch Protection Rules** - Enforce required checks via GitHub settings
2. **Code Coverage** - Track test coverage metrics
3. **Performance Benchmarks** - Detect performance regressions
4. **Documentation Generation** - Auto-generate API documentation
5. **Release Automation** - Automate changelog generation

## Troubleshooting

### Common Issues

**"Black/isort formatting failed"**
- Run `black .` and `isort .` locally and commit changes

**"Clippy warnings"**
- Run `cargo clippy --all-features -- -D warnings` locally
- Fix all warnings before pushing

**"Dependency review failed"**
- Check which dependency has the vulnerability
- Update to a patched version or find an alternative
- Document any exceptions with maintainers

**"CodeQL found issues"**
- Review the security advisory details
- Fix the vulnerability or mark as false positive with explanation
- Re-run CodeQL after fixes

## Conclusion

These checks help maintain high code quality and security standards while making it easier for contributors to understand expectations and requirements. All checks are automated and run on GitHub Actions, providing fast feedback to contributors.

For questions about these checks, see [CONTRIBUTING.md](CONTRIBUTING.md) or open a discussion.
