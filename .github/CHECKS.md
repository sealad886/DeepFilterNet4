# Repository Checks Summary

This document describes the GitHub Actions and related support files that currently exist in `sealad886/DeepFilterNet4`.
It intentionally reflects the repository as it is today rather than an earlier setup rollout.

## Current workflow inventory

Only workflows present in `.github/workflows/` are listed here.

### Python CI (`python_lint.yml`)

- **Triggers:** push to `main`, pull requests to `main`, daily at 18:00 UTC
- **Runner/tooling:** Ubuntu + Python 3.10 + `pre-commit`
- **Checks:** `flake8`, `black`, and `isort` via the repo's pre-commit hooks
- **Purpose:** fast Python formatting and lint validation on the main integration path

### Rust CI (`rust_lint.yml`)

- **Triggers:** push to any branch, daily at 18:00 UTC
- **Checks:** `cargo fmt --check`, `cargo clippy`, targeted `cargo build`, and `cargo test`
- **Packages covered:** `df-demo`, `deep_filter`, `DeepFilterLib`, `DeepFilterDataLoader`
- **Notes:** this workflow currently does **not** run on `pull_request`

### CodeQL Security Scan (`codeql.yml`)

- **Triggers:** push to `main`, pull requests to `main`, weekly on Mondays at 02:30 UTC
- **Jobs:** `Analyze Python Code`, `Analyze Rust Code`
- **Purpose:** GitHub code scanning for security and quality issues
- **Implementation detail:** the Rust job initializes CodeQL with `cpp` and builds the workspace before analysis

### Dependency Review (`dependency-review.yml`)

- **Triggers:** pull requests to `main`
- **Checks:** dependency vulnerability review and license-policy enforcement
- **Policy:** fails on `moderate` or higher severity; denies `GPL-3.0` and `AGPL-3.0`

### PR Checks (`pr-checks.yml`)

- **Triggers:** pull requests to `main`
- **Checks:**
  - warns when `Cargo.lock` or `poetry.lock` changed
  - emits commit-message guidance for conventional commits
  - writes a PR summary
- **Purpose:** lightweight PR hygiene checks without duplicating full CI

### Test DF (`test_df.yml`)

- **Triggers:** manual dispatch, push to `main`, pull requests to `main`, weekly on Sundays, and `workflow_run` for `publish-pypi-wheels`
- **Platform matrix:** Ubuntu on pull requests; Ubuntu + Windows on push/schedule/manual runs
- **Checks:**
  - builds `pyDF` with `maturin`
  - installs DeepFilterNet runtime dependencies
  - runs `python -m df.scripts.test_df`
  - exercises the Python and Rust CLIs on sample audio
  - validates expected DNSMOS outputs
- **Purpose:** end-to-end smoke coverage for the packaged enhancement paths

### CI Status (`ci-status.yml`)

- **Triggers:** push to `main`, pull requests to `main`
- **Behavior:** writes a human-readable summary to the Actions job output
- **Important:** this is a summary job only; it does not currently aggregate upstream workflow results via `needs`

## Supporting build and maintenance workflows

These workflows are part of the repository automation, but they are not the primary PR gatekeepers.

- **`build_demo.yml`** — builds the `df-demo` UI binary on Ubuntu, macOS, and Windows
- **`build_wasm.yml`** — builds the `libDF` WebAssembly package
- **`build_capi.yml`** — produces scheduled/manual C API artifacts across Linux, macOS, and Windows targets
- **`test_pypi_release.yml`** — validates the published PyPI package on Ubuntu, macOS, and Windows; also listens for `publish-pypi-wheels`
- **`stale.yml`** — marks and closes stale issues
- **`combine-prs.yml`** — manual helper for combining compatible PRs into a single branch

## What is not part of the current setup

- There is no repository-managed Copilot setup workflow anymore.
- The former `.github/workflows/copilot-setup-steps.yml` file was template residue and did not match this repository's Python/Rust toolchain.
- This document does **not** list a `publish.yml` workflow because no such file exists in `.github/workflows/`.

## Related support files

- `.github/pull_request_template.md` — PR checklist and submission guidance
- `.github/ISSUE_TEMPLATE/bug_report.yml` — structured bug reports
- `.github/ISSUE_TEMPLATE/feature_request.yml` — structured feature requests
- `.github/ISSUE_TEMPLATE/config.yml` — issue creation links and defaults
- `.github/CODEOWNERS` — code-owner routing for reviews
- `CONTRIBUTING.md` — contributor workflow and local development guidance
- `SECURITY.md` — vulnerability reporting and security policy

## Practical local equivalents

GitHub Actions should mirror checks contributors can run locally.

### Python-side

- `pre-commit run flake8 --all-files`
- `pre-commit run black --all-files`
- `pre-commit run isort --all-files`
- `poetry -C DeepFilterNet install`
- `python -m pytest` from `DeepFilterNet/` when touching Python functionality

### Rust-side

- `cargo fmt --all -- --check`
- `cargo clippy -p df-demo --tests --all-features -- -D warnings`
- `cargo clippy -p deep_filter --tests --all-features -- -D warnings`
- `cargo test --all-features -p deep_filter`

## Suggested branch-protection checks

If branch protection is enabled for `main`, prefer the concrete job names that actually run on pull requests today:

- `lint`
- `Analyze Python Code`
- `Analyze Rust Code`
- `dependency-review`
- `PR Check Status`
- optionally `test-df-output` if you want end-to-end smoke coverage required before merge

`Rust CI` can still remain a push/scheduled safeguard, but its `test` job should not be marked as a required PR check unless that workflow is later extended to `pull_request`.

`All Checks Status` can stay informational, but it should not be treated as a substitute for the underlying workflow jobs.
