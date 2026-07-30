# Mehran v2 Provisional Integration Audit — 2026-07-30

## Scope

This audit covers the provisional implementation proposed by Decision 0004. It validates the
source registry, adapter boundary, checksum-enforced download path, two task assignments, and
regression safety. It does not ratify the protocol or report model performance.

## Traceable stages

| Commit | Stage |
|---|---|
| `a0758bb` | Published candidate/source/license review and proposed Decision 0004 |
| `a98a154` | Pinned the complete official 38-file v2 registry |
| `b476a3d` | Added the 36-recording adapter and two provisional tasks |
| `1783af6` | Added shared checksum-enforced downloads and validation tests |
| `04bd7a3` | Published the provisional data card and user-facing instructions |

Every stage was pushed independently to the public `main` branch. No commit dates were altered
or backdated.

## Source validation

- Official dataset/version: `10.17632/fm6xzxnf36.2`.
- Official repository license metadata: CC BY 4.0.
- Registry: 38 unique files and 159,324,234 total bytes.
- Proposed selection: 36 files; two healthy files retain explicit exclusion reasons.
- Factorial selection: 2 fault locations × 6 defect sizes × 3 loads.
- Inner sample: exact byte-size and SHA-256 match; `(143348, 3)` finite `float32` signal.
- Outer sample: exact byte-size and SHA-256 match; `(130115, 3)` finite `float32` signal.
- Raw sample files were stored in `/private/tmp` and were not committed.

## Protocol and leakage validation

Both provisional manifests pass the generic recording audit:

| Task | Train | Validation | Test | Shared/unassigned |
|---|---:|---:|---:|---:|
| 100 → 300 W | 12 | 12 | 12 | 0 |
| 300 → 100 W | 12 | 12 | 12 | 0 |

Canonical recording identity is the upstream CSV file. No window extraction or model training
is required to establish these assignment counts.

## Clean-checkout result

The repository was cloned with `--no-local` into a new temporary directory at commit
`04bd7a3`. `./scripts/clean_audit.sh` then completed successfully under Python 3.10.13:

- Ruff: passed;
- pytest: 56 passed;
- generated summary, robustness, and hardware tables: passed;
- v1.0 demonstration asset SHA-256: passed;
- synthetic CPU inference check: passed;
- final worktree status: clean and synchronized with `origin/main`.

GitHub Actions runs `30553282338`, `30553707527`, and `30553740449` also completed
successfully for the adapter, download, and data-card stages respectively.

## Acceptance boundary

All mechanical validation items in proposed Decision 0004 are now evidenced. The decision
remains **Proposed** because the maintainer has not yet ratified the dataset, healthy-file
exclusion, two labels, or task directions. Benchmark training and frozen-result claims must
wait for that explicit decision.
