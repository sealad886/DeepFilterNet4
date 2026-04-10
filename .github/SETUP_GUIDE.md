# GitHub Checks Setup Guide

This guide tracks the current GitHub-side setup for `sealad886/DeepFilterNet4`.
It is a maintenance reference, not a changelog for a one-time rollout.

## Current active workflow set

The repository currently ships these workflow files under `.github/workflows/`:

- `python_lint.yml`
- `rust_lint.yml`
- `codeql.yml`
- `dependency-review.yml`
- `pr-checks.yml`
- `ci-status.yml`
- `test_df.yml`
- `build_demo.yml`
- `build_wasm.yml`
- `build_capi.yml`
- `test_pypi_release.yml`
- `stale.yml`
- `combine-prs.yml`

There is no separate Copilot setup workflow in the current repository layout. The old `copilot-setup-steps.yml` file was template-only residue and is no longer part of the active setup.

## What should be enabled in GitHub

### Branch protection for `main`

Recommended settings:

1. Require pull requests before merging
2. Require status checks to pass before merging
3. Require branches to be up to date before merging
4. Optionally require linear history and include administrators

When selecting required status checks, use the job names that actually run on pull requests today:

- `lint`
- `Analyze Python Code`
- `Analyze Rust Code`
- `dependency-review`
- `PR Check Status`

Optional but useful:

- `test-df-output` for end-to-end smoke coverage

`Rust CI` currently runs on pushes and its daily schedule, not on pull requests, so its `test` job should stay out of the required PR-check list unless the workflow trigger changes.

`All Checks Status` from `ci-status.yml` is summary-only and should stay informational unless that workflow is later turned into a real aggregator.

### Security and dependency features

Under **Settings → Security & analysis**, enable:

- Dependency graph
- Dependabot alerts
- Dependabot security updates
- Code scanning
- Secret scanning
- Push protection

`dependabot.yml` is already present in `.github/`.

## Support files tied to the workflow setup

- `.github/CHECKS.md` — current workflow/tooling reference
- `.github/BADGES.md` — optional README badge snippets
- `.github/CODEOWNERS` — review routing
- `.github/pull_request_template.md` — PR checklist
- `.github/ISSUE_TEMPLATE/bug_report.yml` — structured bug reports
- `.github/ISSUE_TEMPLATE/feature_request.yml` — structured feature requests
- `.github/ISSUE_TEMPLATE/config.yml` — issue creation defaults and links
- `CONTRIBUTING.md` — contributor workflow
- `SECURITY.md` — vulnerability reporting policy

## Verification checklist

Use this list after changing GitHub workflows or repo support files:

- [ ] The Actions tab shows the expected workflow list above
- [ ] The Actions tab does not show a `Copilot Setup Steps` workflow
- [ ] A PR to `main` triggers Python CI, CodeQL, Dependency Review, PR Checks, CI Status, and Test DF
- [ ] Pushes still trigger Rust CI
- [ ] Issue templates appear in the new-issue flow
- [ ] The PR template appears when opening a pull request
- [ ] CODEOWNERS review requests still route as expected
- [ ] Any README badges referenced from `.github/BADGES.md` still point at real workflows

## Maintaining these docs

When workflow files change:

1. Update `.github/CHECKS.md` to match the actual workflow inventory and behavior
2. Update this guide if the recommended GitHub settings or verification steps change
3. Avoid documenting placeholder or template workflows as if they are part of the real toolchain

If a workflow is referenced by name from another workflow using `workflow_run`, document that relationship carefully and only if the referenced workflow actually exists in the repository.

## Where to look next

- For exact workflow behavior, start with `.github/CHECKS.md`
- For contributor-facing expectations, use `CONTRIBUTING.md`
- For security reporting, use `SECURITY.md`
