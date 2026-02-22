#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v git >/dev/null 2>&1; then
  echo "git not found; skipping git hook installation." >&2
  exit 0
fi

if ! git -C "$ROOT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  echo "Not a git repository: $ROOT_DIR" >&2
  exit 0
fi

git -C "$ROOT_DIR" config core.hooksPath "$ROOT_DIR/.githooks"

chmod +x "$ROOT_DIR/scripts/guard-upstream-remote.sh"
chmod +x "$ROOT_DIR/.githooks/pre-commit"
# chmod +x "$ROOT_DIR/.githooks/pre-push"
chmod +x "$ROOT_DIR/.githooks/post-checkout"
chmod +x "$ROOT_DIR/.githooks/post-merge"
chmod +x "$ROOT_DIR/.githooks/post-rewrite"

echo "Git hooks installed at $ROOT_DIR/.githooks"
