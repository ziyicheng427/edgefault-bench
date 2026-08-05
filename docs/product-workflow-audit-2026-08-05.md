# Product Workflow Clean-Checkout Audit — 2026-08-05

## Scope

This audit verifies the first Decision 0005 product slice from a new checkout. It covers package
installation, the unified CLI, the raw-data-free quickstart, regression safety, and the existing
demonstration asset. It does not claim PyPI availability, external adoption, or JOSS readiness.

## Procedure

Commit `4751103` was cloned with `git clone --no-local` into a new temporary directory. The
existing `scripts/clean_audit.sh` created a new Python 3.10.13 environment and installed the
project plus locked development and deep-learning dependencies.

The audit then ran the installed console entry point rather than importing source files from
the maintainer workspace:

```bash
edgefault --version
edgefault dataset inspect --manifest registry/mehran_v2.json
edgefault task audit \
  --task registry/tasks/mehran_load_100_to_300_v1.json \
  --dataset-manifest registry/mehran_v2.json
edgefault results validate \
  results/v1.1/mehran/mehran-load-100-to-300-v1__signal_features_logreg.json
```

## Results

- Wheel build and installation: passed.
- Ruff: passed.
- Pytest: 77 passed.
- Existing core, robustness, and hardware table regeneration: passed.
- v1.0 demonstration asset SHA-256 and synthetic inference: passed.
- Installed `edgefault` entry point: present; reports the latest released package version
  `1.0.0` while v1.2 remains under development.
- Dataset inspection: 38 registered files, 36 selected recordings, two exclusions, accepted
  protocol, DOI and CC-BY-4.0 metadata present.
- Task audit: seven checks passed; 12 train, 12 validation, and 12 test recordings.
- Result validation: passed for all registered seeds and source commit provenance.
- Final cloned worktree: clean and synchronized with its origin.

## Compatibility evidence

The unified CLI delegates acquisition and auditing to the same tested implementation used by
the existing dataset-specific commands. No legacy entry point was removed. The HUST v1 result
tables and inference asset remained reproducible in the clean audit.

## Remaining product gaps

- Benchmark execution and report generation are not yet exposed through the unified CLI.
- Public dataset and model plugin discovery is not implemented.
- The package is not yet published to PyPI.
- API documentation and a hosted documentation site are not yet available.
- The workflow has no external-user validation.

These are roadmap items, not completed claims.
