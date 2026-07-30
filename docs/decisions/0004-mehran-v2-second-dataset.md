# Decision 0004: Mehran v2 as the Second Dataset

- Status: Proposed
- Proposed: 2026-07-30
- Scope: EdgeFault-Bench v1.1 second-dataset integration

## Proposed decision

Integrate version 2 of the *Triaxial Bearing Vibration Dataset of Induction Motor under Varying
Load Conditions*, DOI `10.17632/fm6xzxnf36.2`, as the second independent EdgeFault-Bench data
source.

Pin the 36 inner- and outer-race recordings that form a complete six-severity by three-load by
two-label grid. Exclude the two healthy recordings from the initial cross-load tasks because
their filenames and source description do not assign them to 100, 200, or 300 W.

Create forward and reverse load-held-out tasks using 200 W only for validation. Use defect size
as the within-domain evaluation group. Do not compare absolute performance directly with HUST
as though the sensors, labels, sampling process, and fault construction were identical.

## Rationale

- The official Mendeley record assigns CC BY 4.0 and an immutable versioned DOI.
- The public API supplies exact file IDs, byte sizes, SHA-256 values, and download URLs.
- The complete 36-file grid supports leakage-resistant load holdout without inferred labels.
- CSV and triaxial input exercise materially different adapter paths from the HUST MAT files.
- The total source size is practical for independent download and audit.

## Alternatives considered

- Ottawa UODS-VAFDC has strong device and multimodal potential but primarily uses one load; it
  remains a candidate after the generic multichannel interface matures.
- Paderborn provides rich operating conditions but its CC BY-NC license limits broad reuse.
- Ferrara is CC BY 4.0 but has only outer-race faults and bundles signals in a RAR archive.

## Required validation before acceptance

- Snapshot and validate all 38 v2 API file entries, retaining the reason for selecting 36.
- Download at least one file from every fault location and verify size and SHA-256.
- Validate CSV columns, finite samples, channel count, and minimum usable recording length.
- Demonstrate that every selected recording maps exactly once to both proposed tasks.
- Confirm that the HUST v1 tasks, tests, and result tables remain unchanged.

## Ratification

This decision remains proposed until the maintainer explicitly accepts or modifies the dataset,
exclusion, labels, and task directions. Implementation work before ratification is provisional
and must not be represented as a frozen benchmark protocol.
