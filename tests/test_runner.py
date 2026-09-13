import json
from pathlib import Path

import pytest

from edgefault_bench.plugins import ModelPlugin
from edgefault_bench.reporting import load_result
from edgefault_bench.runner import run_prepared_benchmark


class ThresholdEstimator:
    def __init__(self, seed: int):
        self.seed = seed
        self.threshold_ = 0.0

    def fit(self, features, labels):
        self.threshold_ = float(features[:, 0].mean())
        return self

    def predict(self, features):
        return ["fault" if row[0] > self.threshold_ else "healthy" for row in features]

    def get_params(self):
        return {"seed": self.seed}


def _write_inputs(root: Path) -> tuple[Path, Path]:
    task = root / "task.json"
    task.write_text(
        json.dumps(
            {
                "task_id": "example-load-v1",
                "dataset_id": "example-v1",
                "domain_field": "load",
                "evaluation_group_field": "device",
                "partitions": {"train": [0], "validation": [100], "test": [200]},
                "labels": ["healthy", "fault"],
                "window": {"length": 4, "stride": 4, "normalization": "none"},
                "seeds": [17, 29, 43],
                "description": "Synthetic runner test.",
            }
        ),
        encoding="utf-8",
    )
    features = root / "features.json"
    samples = []
    for load in (0, 100, 200):
        samples.extend(
            [
                {
                    "recording_id": f"a-{load}-healthy",
                    "label": "healthy",
                    "domains": {"load": load, "device": "a"},
                    "features": [0.0, 0.1],
                },
                {
                    "recording_id": f"b-{load}-fault",
                    "label": "fault",
                    "domains": {"load": load, "device": "b"},
                    "features": [10.0, 9.9],
                },
            ]
        )
    features.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "example-v1",
                "feature_names": ["rms", "kurtosis"],
                "preparation": {"kind": "synthetic-test"},
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return task, features


def test_runs_plugin_for_frozen_seeds_and_writes_valid_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, features = _write_inputs(tmp_path)
    output = tmp_path / "result.json"
    plugin = ModelPlugin(
        "threshold",
        "sklearn",
        "example-distribution",
        lambda context: ThresholdEstimator(context.seed),
    )
    monkeypatch.setattr("edgefault_bench.runner._git_commit", lambda: "abcdef1234567890")

    result_path = run_prepared_benchmark(
        task_path=task,
        prepared_path=features,
        model_id="threshold",
        output_path=output,
        latency_warmup=0,
        latency_repeats=2,
        plugins={"threshold": plugin},
    )
    payload = load_result(result_path)

    assert [run["seed"] for run in payload["runs"]] == [17, 29, 43]
    assert all(run["test"]["macro_f1"] == 1.0 for run in payload["runs"])
    assert payload["model_plugin"]["source_distribution"] == "example-distribution"
    assert payload["protocol"]["test_used_for_selection"] is False


def test_rejects_backend_without_implemented_execution_policy(tmp_path: Path) -> None:
    task, features = _write_inputs(tmp_path)
    plugin = ModelPlugin("neural", "pytorch", "example", lambda context: object())

    with pytest.raises(ValueError, match="requires backend 'sklearn'"):
        run_prepared_benchmark(
            task_path=task,
            prepared_path=features,
            model_id="neural",
            output_path=tmp_path / "result.json",
            plugins={"neural": plugin},
        )
