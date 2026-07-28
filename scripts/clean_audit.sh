#!/usr/bin/env bash
set -euo pipefail

uv sync --frozen --extra dev --extra deep-learning
uv run --frozen ruff check .
uv run --frozen pytest
audit_output="outputs/clean-audit"
uv run --frozen edgefault-summarize \
  --results-dir results/v1 \
  --output "${audit_output}/SUMMARY.md" \
  --robustness-output "${audit_output}/ROBUSTNESS.md" \
  --hardware-output "${audit_output}/HARDWARE.md"
cmp -s results/v1/SUMMARY.md "${audit_output}/SUMMARY.md"
cmp -s results/v1/ROBUSTNESS.md "${audit_output}/ROBUSTNESS.md"
cmp -s results/v1/HARDWARE.md "${audit_output}/HARDWARE.md"
uv run --frozen edgefault-verify-demo
uv run --frozen edgefault-infer --warmup 2 --repeats 5
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status --short --branch
fi
