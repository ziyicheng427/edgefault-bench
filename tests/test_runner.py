import json
from pathlib import Path

import pytest

from edgefault_bench.plugins import ModelPlugin
from edgefault_bench.reporting import load_result, verify_input_artifact_hashes
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
    monkeypatch.setattr("edgefault_bench.execution._git_commit", lambda: "abcdef1234567890")

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

    assert payload["schema_version"] == 2
    assert [run["seed"] for run in payload["runs"]] == [17, 29, 43]
    assert all(run["status"] == "ok" for run in payload["runs"])
    assert all(run["test"]["macro_f1"] == 1.0 for run in payload["runs"])
    assert payload["model_plugin"]["source_distribution"] == "example-distribution"
    assert payload["protocol"]["test_used_for_selection"] is False
    assert payload["input_artifacts"]["task_manifest"]["sha256"] != ""
    assert payload["framework_overhead"]["load_seconds"] >= 0.0

    hashes = verify_input_artifact_hashes(result_path)
    assert hashes["passed"] is True


def test_input_artifact_hash_check_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, features = _write_inputs(tmp_path)
    plugin = ModelPlugin(
        "threshold", "sklearn", "example-distribution",
        lambda context: ThresholdEstimator(context.seed),
    )
    monkeypatch.setattr("edgefault_bench.execution._git_commit", lambda: "abcdef1234567890")
    output = tmp_path / "result.json"
    run_prepared_benchmark(
        task_path=task,
        prepared_path=features,
        model_id="threshold",
        output_path=output,
        latency_warmup=0,
        latency_repeats=1,
        plugins={"threshold": plugin},
    )
    features.write_text(features.read_text().replace("0.1", "0.2"), encoding="utf-8")

    report = verify_input_artifact_hashes(output)
    assert report["passed"] is False
    assert report["artifacts"]["prepared_feature_table"]["status"] == "mismatch"


class UnstableEstimator:
    """Module-level so the kernel's pickle-based size probe can serialize it."""

    def __init__(self, seed: int):
        if seed == 29:
            raise RuntimeError("synthetic training failure")

    def fit(self, features, labels):
        return self

    def predict(self, features):
        return ["healthy"] * len(features)

    def get_params(self):
        return {}


def test_failed_seed_is_retained_as_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, features = _write_inputs(tmp_path)
    plugin = ModelPlugin(
        "unstable", "sklearn", "example-distribution",
        lambda context: UnstableEstimator(context.seed),
    )
    monkeypatch.setattr("edgefault_bench.execution._git_commit", lambda: "abcdef1234567890")
    output = run_prepared_benchmark(
        task_path=task,
        prepared_path=features,
        model_id="unstable",
        output_path=tmp_path / "result.json",
        latency_warmup=0,
        latency_repeats=1,
        plugins={"unstable": plugin},
    )
    payload = load_result(output)
    statuses = {run["seed"]: run["status"] for run in payload["runs"]}
    assert statuses == {17: "ok", 29: "failed", 43: "ok"}
    failed = next(run for run in payload["runs"] if run["seed"] == 29)
    assert "synthetic training failure" in failed["error"]


def test_pytorch_plugin_runs_end_to_end_on_tabular_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")

    class TinyTorchNet(torch.nn.Module):
        def __init__(self, num_classes: int, in_features: int):
            super().__init__()
            self.flatten = torch.nn.Flatten()
            self.head = torch.nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.head(self.flatten(x))

    def factory(context):
        return TinyTorchNet(context.num_classes, context.input_channels)

    plugin = ModelPlugin("tiny_net", "pytorch", "example-distribution", factory)
    task, features = _write_inputs(tmp_path)
    monkeypatch.setattr("edgefault_bench.execution._git_commit", lambda: "abcdef1234567890")

    output = run_prepared_benchmark(
        task_path=task,
        prepared_path=features,
        model_id="tiny_net",
        output_path=tmp_path / "result.json",
        latency_warmup=0,
        latency_repeats=1,
        plugins={"tiny_net": plugin},
    )
    payload = load_result(output)
    assert payload["schema_version"] == 2
    assert payload["model_plugin"]["backend"] == "pytorch"
    assert all(run["status"] == "ok" for run in payload["runs"])
    assert all(
        run["complexity"]["trainable_parameters"] > 0 for run in payload["runs"]
    )
