# Dataset Adapter Guide

Dataset adapters translate source-specific files and metadata into the stable EdgeFault-Bench
contracts. They do not redefine benchmark metrics, silently relabel source data, or redistribute
files without permission.

## Required contract

An adapter implements `edgefault_bench.contracts.DatasetAdapter` and exposes:

- one immutable `DatasetMetadata` value with source, version, license, and domain vocabulary;
- a sequence of immutable `Recording` values, one per independent acquisition;
- stable recording identifiers, labels, domain values, sample counts, and source-file names;
- source SHA-256 values when the upstream repository publishes immutable files.

The minimal shape is:

```python
from edgefault_bench.contracts import DatasetMetadata, Recording


class ExampleAdapter:
    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            dataset_id="example-v1",
            version="1",
            title="Example industrial sensor dataset",
            license_spdx="CC-BY-4.0",
            source_url="https://example.org/dataset",
            domain_fields=("device", "load"),
        )

    def recordings(self) -> tuple[Recording, ...]:
        return (
            Recording(
                dataset_id="example-v1",
                recording_id="device-a-load-0",
                source_file="device-a-load-0.csv",
                label="healthy",
                domains={"device": "a", "load": 0},
                sample_rate_hz=12_800,
                sample_count=128_000,
            ),
        )
```

## Source and license review

Before implementation, open a public issue that records:

1. the official landing page, version, DOI or stable identifier;
2. access and redistribution terms from the source itself;
3. available devices, operating conditions, labels, and missing combinations;
4. which files will be referenced and whether their byte size and hash can be pinned;
5. known experimental limitations and any excluded labels.

Do not copy raw recordings into the repository unless the source license explicitly permits
redistribution and doing so is necessary. Prefer a downloader that verifies the official files.

## Task and leakage requirements

Every dataset contribution includes at least one versioned `TaskSpec`. Train, validation, and
test domain values must be explicit and disjoint. Run the metadata audit before extracting
windows:

```bash
edgefault-audit-task \
  --task registry/tasks/hust_load_0_to_400_v1.json \
  --dataset-manifest registry/hust_v3.json \
  --output outputs/audits/hust-load-0-to-400-v1.json
```

The report checks dataset identity, required domain fields, label support, complete domain
assignment, non-empty partitions, unique recording identity, and recording exclusivity. A
failed report must block training. Signal-window overlap checks remain the responsibility of
the window builder and its tests.

## Contribution checklist

- Add adapter unit tests using small synthetic fixtures rather than committing source data.
- Test corrupt files, missing metadata, unsupported labels, and unassigned domains.
- Demonstrate that every recording is assigned exactly once by the proposed task.
- Add a data card with source rights and limitations.
- Document preprocessing without using test-domain population statistics.
- Preserve raw metadata needed to reproduce labels and domain assignments.
- Run the full test suite and clean audit.
- Disclose material AI assistance and human verification in the pull request.

The first implementation is `HustV3Adapter`. Future adapters should reuse the contracts and
audit rather than copying HUST-specific parsing rules.
