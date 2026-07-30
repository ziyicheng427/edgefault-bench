# Mehran Triaxial Bearing v2 Data Card

## Status

Decision 0004 was accepted by the maintainer on 2026-07-30. The selected recordings, labels,
partitions, window settings, and seeds in the two v1 task manifests are frozen. Acceptance of
the protocol does not validate a future model score or scientific claim.

## Source and rights

- Title: *Triaxial Bearing Vibration Dataset of Induction Motor under Varying Load Conditions*
- Provider: Mehran University of Engineering and Technology / NCRA
- Repository: Mendeley Data
- Version: 2
- DOI: [`10.17632/fm6xzxnf36.2`](https://doi.org/10.17632/fm6xzxnf36.2)
- License reported by the repository: CC BY 4.0
- Official record: <https://data.mendeley.com/datasets/fm6xzxnf36/2>

The repository contains metadata and download tooling only. Source recordings remain under
their original license and are not redistributed with EdgeFault-Bench.

## Registry and selection

`registry/mehran_v2.json` snapshots all 38 upstream v2 file entries, including their immutable
file IDs, byte sizes, SHA-256 digests, and stable download URLs. The snapshot totals
159,324,234 bytes. It can be independently regenerated from the official public API with
`scripts/snapshot_mehran_v2.py`.

The frozen benchmark selection contains 36 recordings in a complete factorial grid:

- fault location: inner race or outer race;
- defect size: 0.7, 0.9, 1.1, 1.3, 1.5, or 1.7 mm;
- motor load: 100, 200, or 300 W;
- channels: X, Y, and Z acceleration at the source-reported 10 kHz rate.

The two healthy recordings are retained in the registry but excluded from the frozen tasks.
Their names distinguish pulley state, not a 100/200/300 W load, so assigning them to a load
would create unsupported metadata. This initial task therefore diagnoses fault location among
faulted recordings; it is not a healthy-versus-fault detector.

## Frozen tasks

| Manifest | Train | Validation | Test | Labels |
|---|---:|---:|---:|---|
| `mehran_load_100_to_300_v1.json` | 100 W | 200 W | 300 W | inner/outer race |
| `mehran_load_300_to_100_v1.json` | 300 W | 200 W | 100 W | inner/outer race |

Each source CSV is one canonical recording. The generic leakage audit assigns all 36 selected
recordings exactly once in each task: 12 train, 12 validation, and 12 test. Defect size is an
evaluation grouping variable, not a class label.

## Download and verification

Download one registered recording or the entire selected set:

```bash
uv run edgefault-download-mehran --files 0.7outer-100watt-lB5LIS.csv
uv run edgefault-download-mehran
uv run edgefault-download-mehran --verify-only
```

Downloads are written atomically and accepted only when both byte size and SHA-256 match the
committed registry. Invalid existing files fail closed unless the user explicitly requests
`--repair`.

On 2026-07-30, two official files were independently downloaded and loaded through the adapter:

| Fault | File | Bytes | Rows | Observed SHA-256 |
|---|---|---:|---:|---|
| inner race | `0.7inner-100watt-67V2Iv.csv` | 4,520,076 | 143,348 | `3b1f1ebd2499cd75a4443455afcc4101bd64136b02ea7c1ba3caed3c86780724` |
| outer race | `0.7outer-100watt-lB5LIS.csv` | 4,125,202 | 130,115 | `782dcac141a504b10c39e2fe4a236f92930d8a8a99264f344a541cd4cd7f2487` |

Both observed digests and byte sizes matched the official API metadata. Both files had the
expected timestamp plus X/Y/Z header, finite numeric samples, and at least one 4,096-sample
window. Raw validation files were kept outside the repository.

## Known limitations

- Laboratory seeded faults do not establish performance on naturally degraded field assets.
- There is one physical CSV per location/size/load cell; windows from a recording are not
  independent physical assets.
- The timestamp text is ignored because acceleration columns provide the analysis signal.
- Absolute scores should not be compared with HUST as if acquisition, labels, and fault
  construction were equivalent.
- Acceptance of the source does not imply acceptance of a model or a scientific claim.

Use should cite the upstream dataset and comply with CC BY 4.0 attribution requirements.
