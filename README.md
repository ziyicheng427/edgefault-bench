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

The protocol is frozen and the dataset-independent core is under active development. See
[`docs/project-charter.md`](docs/project-charter.md),
[`docs/evaluation-protocol.md`](docs/evaluation-protocol.md), and [`ROADMAP.md`](ROADMAP.md).

## Development smoke check

The smoke check uses generated periodic signals only. It validates the split, feature,
model, and metric interfaces; its output is not a machinery benchmark result.

```bash
uv sync --extra dev
uv run edgefault-smoke
uv run pytest
```

## License

The code is released under the Apache License 2.0. Dataset files remain under
their original owners' terms and will not be committed to this repository.
