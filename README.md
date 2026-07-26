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

## Planned scope

- Public rotating-machinery datasets, downloaded by scripts rather than redistributed.
- Explicit train/validation/test separation by operating condition.
- Classical, convolutional, and lightweight neural baselines.
- Three-seed reporting with aggregate and worst-condition metrics.
- Parameter count, serialized model size, and CPU latency measurements.
- Reproducible configurations, tests, result tables, and a concise technical report.

No benchmark results are claimed yet. The first public release will be made only after the
data protocol and baselines are independently reproducible from a clean environment.

## Status

The protocol is frozen and the first public-data registry and task manifests are available.
See
[`docs/project-charter.md`](docs/project-charter.md),
[`docs/evaluation-protocol.md`](docs/evaluation-protocol.md), and [`ROADMAP.md`](ROADMAP.md).

The primary v1 data source is the CC BY 4.0 HUST bearing dataset. EdgeFault-Bench pins 60
source files and three condition-held-out tasks without redistributing the recordings. See
the [`HUST v3 data card`](docs/datasets/hust-bearing-v3.md).

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

## License

The code is released under the Apache License 2.0. Dataset files remain under
their original owners' terms and will not be committed to this repository.
