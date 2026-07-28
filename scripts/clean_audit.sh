#!/usr/bin/env bash
set -euo pipefail

uv sync --frozen --extra dev --extra deep-learning
uv run --frozen ruff check .
uv run --frozen pytest
uv run --frozen edgefault-summarize
git diff --exit-code -- \
  results/v1/SUMMARY.md \
  results/v1/ROBUSTNESS.md \
  results/v1/HARDWARE.md
uv run --frozen edgefault-verify-demo
uv run --frozen edgefault-infer --warmup 2 --repeats 5
git status --short --branch
