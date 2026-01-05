# Contributing to DeepFilterNet

Thank you for your interest in contributing to DeepFilterNet! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Version information (`deepFilter --version` or `deep-filter --version`)
- Operating system and environment details
- Relevant log output

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.yml) when creating bug reports.

### Suggesting Features

Feature requests are welcome! Please use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.yml) and provide:

- Clear description of the feature
- Use cases and benefits
- Possible implementation approach (if applicable)

### Pull Requests

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** following our coding standards
3. **Test your changes** thoroughly
4. **Commit your changes** with clear, descriptive commit messages
5. **Push to your fork** and submit a pull request

#### Pull Request Guidelines

- Fill out the pull request template completely
- Link related issues using keywords (e.g., "Fixes #123")
- Keep changes focused and atomic
- Ensure all tests pass
- Update documentation as needed

## Development Setup

### Prerequisites

- Rust (via [rustup](https://rustup.rs/))
- Python 3.8+ 
- Poetry for Python dependency management
- Maturin for building Python wheels

### Setup Instructions

```bash
cd path/to/DeepFilterNet/

# Install Python dependencies
pip install maturin poetry

# For Python development
poetry -C DeepFilterNet install -E train -E eval --no-root
export PYTHONPATH=$PWD/DeepFilterNet

# Build libDF Python package
maturin develop --release -m pyDF/Cargo.toml

# Optional: Build libdfdata for dataset functionality
maturin develop --release -m pyDF-data/Cargo.toml
```

## Coding Standards

### Python Code

We use the following tools for Python code quality:

- **black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **flake8**: Linting

Run formatters and linters:

```bash
# Format code
black .
isort .

# Check linting
flake8
```

Configuration files:
- `.flake8` - Flake8 configuration
- `pyproject.toml` - Black and isort configuration

### Rust Code

We use standard Rust tooling:

- **rustfmt**: Code formatting
- **clippy**: Linting

Run formatters and linters:

```bash
# Format code
cargo fmt

# Check linting
cargo clippy --all-features -- -D warnings
```

Configuration files:
- `rustfmt.toml` - Rustfmt configuration
- `clippy.toml` - Clippy configuration

## Testing

### Python Tests

```bash
cd DeepFilterNet
poetry run python df/scripts/test_df.py
```

### Rust Tests

```bash
cargo test --all-features
```

## Building

### Rust Binary (deep-filter)

```bash
cargo build --release -p deep_filter
```

### Python Wheels

```bash
maturin build --release -m pyDF/Cargo.toml
```

## Commit Message Guidelines

While not strictly enforced, we encourage following the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(libDF): add new STFT windowing option
fix(python): correct delay compensation calculation
docs: update installation instructions for Windows
```

## Automated Checks

All pull requests run through automated checks:

### Required Checks
- **Python Lint**: black, isort, flake8 must pass
- **Rust Lint**: rustfmt and clippy must pass
- **CodeQL**: Security analysis for vulnerabilities
- **Dependency Review**: Checks for vulnerable or incompatible dependencies

### Informational Checks
- **PR Checks**: Validates lockfile changes and commit format
- **Tests**: Integration tests (run on main branch and PRs)

You can run these checks locally before submitting your PR to catch issues early.

## Release Process

Releases are managed by maintainers. Version numbers follow [Semantic Versioning](https://semver.org/).

## Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/Rikorose/DeepFilterNet/discussions)
- **Bugs**: File an issue using the bug report template
- **Features**: File an issue using the feature request template

## License

By contributing, you agree that your contributions will be dual-licensed under MIT and Apache-2.0 licenses, matching the project's existing license terms.

## Recognition

Contributors will be acknowledged in release notes and the project documentation.

Thank you for contributing to DeepFilterNet! 🎉
