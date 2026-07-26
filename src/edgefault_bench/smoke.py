"""End-to-end smoke check on explicitly synthetic data."""

from __future__ import annotations

import json

from edgefault_bench.baseline import make_feature_baseline
from edgefault_bench.data import make_condition_split
from edgefault_bench.evaluation import condition_metrics
from edgefault_bench.features import extract_features
from edgefault_bench.synthetic import make_synthetic_dataset


def main() -> None:
    dataset = make_synthetic_dataset()
    split = make_condition_split(
        dataset.index,
        train_conditions={"speed-low"},
        validation_conditions={"speed-mid"},
        test_conditions={"speed-high"},
    )
    features = extract_features(dataset.windows, sampling_rate=dataset.sampling_rate)
    model = make_feature_baseline()
    model.fit(features[split.train], dataset.labels[split.train])
    predictions = model.predict(features[split.test])
    metrics = condition_metrics(
        dataset.labels[split.test], predictions, dataset.conditions[split.test]
    )
    result = {
        "warning": "Synthetic smoke-test output is not a benchmark result.",
        "samples": len(dataset.labels),
        "features": features.shape[1],
        "metrics": metrics.to_dict(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

