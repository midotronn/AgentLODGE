#!/usr/bin/env bash
# Create the AgentLODGE repository on GitHub and push the current branch.
set -euo pipefail

REPO_NAME="${1:-AgentLODGE}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GH_BIN="${GH_BIN:-gh}"

if ! command -v "$GH_BIN" >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required. Install from https://cli.github.com/ or set GH_BIN."
  exit 1
fi

if ! "$GH_BIN" auth status >/dev/null 2>&1; then
  echo "Run 'gh auth login' first, then re-run this script."
  exit 1
fi

cd "$ROOT"
"$GH_BIN" repo create "$REPO_NAME" --public --source=. --remote=origin --push
echo "Repository created and pushed: $("$GH_BIN" repo view --json url -q .url)"
