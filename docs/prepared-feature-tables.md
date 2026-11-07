# Prepared Feature Tables

Prepared feature tables are the portable boundary between dataset-specific preprocessing and
backend-specific model execution. They keep every row tied to an independent recording, label,
and operating-condition vocabulary so a versioned task can assign partitions before fitting.

## v1 JSON contract (legacy)

Schema v1 is deliberately readable and suited to small examples. Rows carry a single
`recording_id` and no window position, so window-level leakage audits cannot see where a
row came from:

```json
{
  "schema_version": 1,
  "dataset_id": "example-v1",
  "feature_names": ["rms", "kurtosis"],
  "preparation": {
    "method": "documented-feature-extractor-v1"
  },
  "samples": [
    {
      "recording_id": "device-a-load-0-window-0",
      "label": "healthy",
      "domains": {"device": "a", "load": 0},
      "features": [0.12, 2.95]
    }
  ]
}
```

The earlier convention of embedding a window index into `recording_id` (as above) defeats
recording-level grouping and is no longer accepted practice; new tables must use v2.

## v2 contract with window identity

Schema v2 requires window identity on **every** row, which makes the
partition-inheritance invariant checkable: all windows of one recording must sit in the
same partition, and relabelling a test recording into the training side is rejected when
canonical dataset metadata is available:

```json
{
  "schema_version": 2,
  "dataset_id": "example-v1",
  "feature_names": ["acc_x_mean", "acc_x_kurtosis"],
  "preparation": {
    "preprocessor": {
      "plugin_id": "signal_features",
      "version": "1",
      "state_digest": "…64 hex chars…",
      "fit_scope": ["train", "validation"],
      "fitted_row_count": 128
    },
    "channel_layout": ["acc_x", "acc_y", "acc_z"],
    "sample_rate_hz": 12800.0
  },
  "samples": [
    {
      "window_id": "0.7inner-100watt.csv#s000000000",
      "recording_id": "0.7inner-100watt.csv",
      "start_sample": 0,
      "end_sample": 4096,
      "label": "inner_race",
      "domains": {"load_w": 100, "defect_size_mm": 0.7},
      "features": [0.12, 2.95]
    }
  ]
}
```

The loader rejects unsupported schema versions, empty or duplicate feature names, non-finite
values, inconsistent row widths, partial window identity, duplicate `window_id` values, and
inconsistent metadata for rows sharing a recording identifier. Task preparation additionally
rejects dataset mismatch, unknown labels, missing domain fields, unassigned conditions, empty
partitions, and — when adapter recordings are supplied — rows whose induced partition
contradicts their recording's canonical partition.

Validate the portable structure before binding the table to a task (the CLI dispatches on
`schema_version` automatically):

```bash
edgefault schema validate --kind prepared-feature-table prepared/example-v1.json
```

## Storage layering

JSON remains the auditable interchange format for metadata and small tables. For larger
matrices, v2 tables can be written and read as Parquet through the optional `columnar`
extra (`pip install 'edgefault-bench[columnar]'`); the header, feature names, and
preparation provenance travel in the Parquet schema metadata and are validated on load.
Raw signals are never stored in tables — they are streamed from checksum-verified source
files through the dataset signal interface.

## Building tables from a registered dataset

The kernel can build a v2 table end-to-end from a dataset manifest, streaming windows
from verified recordings, fitting the preprocessing plugin on the training side only,
and recording the fitted-state digest:

```bash
edgefault benchmark run \
  --task registry/tasks/example-v1.json \
  --dataset-manifest registry/example_v1.json \
  --preprocessor signal_features \
  --model third_party_model \
  --output outputs/example-v1__third_party_model.json
```

## Execution

An installed model plugin — scikit-learn or PyTorch — consumes a validated table through
the unified CLI:

```bash
edgefault benchmark run \
  --task registry/tasks/example-v1.json \
  --features prepared/example-v1.json \
  --model third_party_model \
  --output outputs/example-v1__third_party_model.json
edgefault results validate --check-hashes outputs/example-v1__third_party_model.json
```

The runner constructs a fresh estimator for every task seed, fits only on the training
view under a versioned training policy, evaluates validation and test indices separately,
and writes a schema-v2 result bundle with content-hashed input artifacts, per-seed
`ok`/`failed` status, and framework overhead timings. Failed seeds are retained as
placeholders; they never disappear from the seed sequence.

## Evidence boundary

A prepared table is derived data. Its creator remains responsible for source rights, checksum
verification, preprocessing disclosure, recording-level split integrity, and avoiding statistics
computed from held-out domains. A successful schema or runner validation establishes structural
reproducibility, not scientific validity or deployment safety.
