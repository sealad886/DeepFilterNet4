#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-enforce}"

if ! command -v git >/dev/null 2>&1; then
  exit 0
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  exit 0
fi

if git remote get-url upstream >/dev/null 2>&1; then
  case "$MODE" in
    cleanup)
      git remote remove upstream
      echo "Removed forbidden 'upstream' remote."
      exit 0
      ;;
    enforce)
      echo "ERROR: 'upstream' remote is forbidden in this repository." >&2
      echo "Removing it now to prevent accidental use." >&2
      git remote remove upstream || true
      exit 1
      ;;
    *)
      echo "ERROR: Unknown mode '$MODE' (expected 'cleanup' or 'enforce')." >&2
      exit 2
      ;;
  esac
fi

exit 0
