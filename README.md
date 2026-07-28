# EdgeFault-Bench

**A reproducible benchmark for cross-condition and resource-efficient fault diagnosis.**

EdgeFault-Bench evaluates whether fault-diagnosis models remain reliable when operating
conditions change and compute resources are limited. The project focuses on transparent
experimental protocols, reproducible baselines, and deployment-oriented measurements
rather than accuracy from random within-condition splits.

## Research questions

1. How much performance is lost when a model is evaluated on operating conditions that
   were not observed during training?
2. Which compact architectures offer the best trade-off among worst-condition Macro-F1,
   parameter count, memory footprint, and CPU latency?
3. How robust are these models to limited labels, class imbalance, and measurement noise?

## Implemented benchmark scope

- A checksum-pinned public rotating-machinery dataset, downloaded by scripts rather than
  redistributed.
- Three explicit load- or device-held-out tasks with recording-level leakage controls.
- Signal-feature, standard CNN, compact depthwise CNN, and compact CORAL baselines.
- Three-seed aggregate, worst-condition, label-scarcity, class-imbalance, and noise results.
- Parameter count, serialized size, scoped MACs, repeated Apple M1 latency, and isolated
  process RSS measurements.
- Machine-readable configurations, raw result JSON, generated tables, tests, a model card,
  and a technical report.

The measured pre-release results are public, but v1.0 is not released yet. A tagged release
will be made only after the edge-inference demonstration and clean-environment reproduction
audit are complete.

## Status

The core 3-task by 4-model matrix and all three robustness tracks are complete. The strongest
negative result is severe device-holdout failure despite good load-holdout scores. The compact
model reduces parameters and MACs substantially but is not faster than the standard CNN on the
measured Apple M1 CPU. See
[`docs/project-charter.md`](docs/project-charter.md),
[`docs/evaluation-protocol.md`](docs/evaluation-protocol.md),
[`docs/technical-report.md`](docs/technical-report.md),
[`docs/model-card.md`](docs/model-card.md), and [`ROADMAP.md`](ROADMAP.md).

The primary v1 data source is the CC BY 4.0 HUST bearing dataset. EdgeFault-Bench pins 60
source files and three condition-held-out tasks without redistributing the recordings. See
the [`HUST v3 data card`](docs/datasets/hust-bearing-v3.md).

Development is AI-assisted and governed by explicit provenance and research-integrity rules.
See [`docs/research-integrity.md`](docs/research-integrity.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Development smoke check

The smoke check uses generated periodic signals only. It validates the split, feature,
model, and metric interfaces; its output is not a machinery benchmark result.

```bash
uv sync --extra dev
uv run edgefault-smoke
uv run pytest
```

Download and verify the pinned HUST subset, or start with one recording:

```bash
uv run edgefault-download-hust --files N400.mat
uv run edgefault-download-hust --verify-only --files N400.mat
```

The full selected subset is approximately 414 MB. Every source file is accepted only after
its byte size and SHA-256 match the committed registry. Downloads use four workers by
default; `--workers 1` provides a serial fallback.

Run a frozen neural baseline (three seeds, validation-only early stopping):

```bash
uv sync --extra dev --extra deep-learning
uv run edgefault-run-neural \
  --model compact_depthwise_cnn_1d \
  --task registry/tasks/hust_load_0_to_400_v1.json
```

Rebuild the committed core, robustness, and hardware Markdown tables from raw JSON:

```bash
uv run edgefault-summarize
```

The source tables are in [`results/v1`](results/v1). Checkpoints remain local pre-release
artifacts and are not yet advertised as downloadable models.

## License

The code is released under the Apache License 2.0. Dataset files remain under
their original owners' terms and will not be committed to this repository.
