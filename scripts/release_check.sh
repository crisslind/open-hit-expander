#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo '=== Git state ==='
git status --short
git remote -v
echo '=== Diff summary against current HEAD ==='
git diff --stat
git diff --check
git diff --cached --stat
echo '=== Python checks ==='
python -m compileall -q src
python -m pytest -q tests
echo '=== Saved October 13 run (no inference) ==='
python scripts/release_smoke.py --timestamp 20261013_111348_145796
echo 'All automated release checks passed. Review git diff manually before tagging.'
