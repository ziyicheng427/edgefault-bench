# Integration-Effort Audit (RQ2) — 2026-11-06

## Claim under test

Adding a heterogeneous public dataset requires only adapter-side artefacts; the
benchmark kernel gains no dataset-specific logic. This audit quantifies the two
most recent integrations (Decision 0007 gearbox, Decision 0008 KAIST acoustic)
and lists every kernel file they touched, with the reason.

## Measured integration effort

| Integration | Adapter code (LOC) | Tests (LOC) | Total incl. registry/task/card/decision | Files added |
|---|---:|---:|---:|---:|
| `gearbox-two-stage-v1` | 239 | 175 | 812 | 6 |
| `kaist-rotating-acoustic-v6` | 261 | 158 | 595 | 5 |

Both integrations were verified against the real upstream archives before any
result was produced, and both run end-to-end through the unified
`benchmark run` command (gearbox including a frozen cross-load task and group
statistics; KAIST acoustic at adapter/provenance level, no task — see its data
card).

## Kernel files touched, and why

| Kernel file | Change | Dataset-specific? |
|---|---|---|
| `contracts.py` | `sample_rate_hz` became optional where a source may not state one; consumers fail closed | No — generic honest-metadata capability |
| `windows.py` | stopped asserting a rate | No — consequence of the same generic change |
| `plugins.py` | one registration-table line per adapter | Declaration, not logic (same role as an entry point) |
| `datasets/__init__.py` | re-exports | No |
| `cli.py` | one branch per downloader in the fetch dispatcher | **Yes — registered debt** until downloaders become plugins |

No kernel file contains dataset identifiers outside the registration table and
the fetch dispatcher; `tests/test_gearbox_adapter.py::test_kernel_contains_no_dataset_specific_branches`
enforces this for all lifecycle modules on every run, with the known debt
listed explicitly in the test.

## Traditional-path comparison (RQ2 narrative)

| Operation | Typical paper-repository approach | EdgeFault-Bench |
|---|---|---|
| Add a dataset | new parsing + split + training scripts per dataset | one adapter package/file + registry + task manifest |
| Assert source integrity | ad hoc or absent | registry SHA-256, verified at download and at every `open` |
| Prevent recording leakage | manual review | fail-closed audit + window inheritance + cross-audit |
| Change the held-out conditions | edit experiment code | edit a versioned task manifest |
| Reproduce a result | re-read the paper, rebuild the environment | result bundle + `results validate --check-hashes` |

The claimed "zero kernel edits" holds for all lifecycle logic; the two honest
exceptions above are one generic contract relaxation and the fetch-dispatcher
debt.

## Boundary

The two audited integrations were performed by the project author; the
stronger RQ2 evidence — an integration authored by a non-author — remains open
(T17).
