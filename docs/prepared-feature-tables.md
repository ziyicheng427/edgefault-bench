# Prepared Feature Tables

Prepared feature tables are the portable boundary between dataset-specific preprocessing and
backend-specific model execution. They keep every row tied to an independent recording, label,
and operating-condition vocabulary so a versioned task can assign partitions before fitting.

## JSON contract

The initial schema is deliberately readable and suited to small examples:

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

The loader rejects unsupported schema versions, empty or duplicate feature names, non-finite
values, and inconsistent row widths. Task preparation additionally rejects dataset mismatch,
unknown labels, missing domain fields, unassigned conditions, empty partitions, and inconsistent
metadata for rows sharing a recording identifier.

JSON is not intended as the final high-volume storage backend. A future binary or columnar format
must preserve these semantics and publish hashes rather than silently weakening provenance.

## Execution

An installed scikit-learn model plugin can consume a validated table through the unified CLI:

```bash
edgefault benchmark run \
  --task registry/tasks/example-v1.json \
  --features prepared/example-v1.json \
  --model third_party_model \
  --output outputs/example-v1__third_party_model.json
edgefault results validate outputs/example-v1__third_party_model.json
```

The runner constructs a fresh model for every task seed, fits only the training indices, evaluates
validation and test indices separately, and records the code commit, plugin distribution/version,
preparation metadata, environment, partition sizes, metrics, serialized size, learned numeric
value count, and batch-one latency protocol.

The current runner accepts only the explicit `sklearn` backend. PyTorch execution remains blocked
until a versioned policy defines optimization, checkpoint selection, early stopping, and optional
domain-generalization objectives. This prevents incomparable training procedures from appearing
under one generic command.

## Evidence boundary

A prepared table is derived data. Its creator remains responsible for source rights, checksum
verification, preprocessing disclosure, recording-level split integrity, and avoiding statistics
computed from held-out domains. A successful schema or runner validation establishes structural
reproducibility, not scientific validity or deployment safety.
