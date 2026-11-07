"""Group-aware statistics tests (T15): intervals, pairing, bundle wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import edgefault_bench.cli as cli
from edgefault_bench.execution import run_benchmark
from edgefault_bench.plugins import ModelPlugin
from edgefault_bench.reporting import load_result
from edgefault_bench.statistics import (
    compare_bundles,
    group_bootstrap,
    macro_f1,
    paired_group_bootstrap,
    result_bundle_statistics,
)


class PerfectEstimator:
    def __init__(self, seed: int):
        self.seed = seed

    def fit(self, features, labels):
        return self

    def predict(self, features):
        # Feature 0 encodes the label: 0 -> healthy, 1 -> fault.
        return ["healthy" if int(row[0]) == 0 else "fault" for row in np.asarray(features)]

    def get_params(self):
        return {"seed": self.seed}


class NoisyEstimator(PerfectEstimator):
    def predict(self, features):
        # Feature 2 carries the row index; every fifth fault is degraded.
        perfect = super().predict(features)
        return [
            "healthy" if int(row[0]) == 1 and int(row[2]) % 5 == 0 else value
            for row, value in zip(np.asarray(features), perfect, strict=True)
        ]


def _write_task(root: Path) -> Path:
    task = root / "task.json"
    task.write_text(
        json.dumps(
            {
                "task_id": "stats-fixture-v1",
                "dataset_id": "stats-v1",
                "domain_field": "load",
                "evaluation_group_field": "device",
                "partitions": {"train": [0], "validation": [100], "test": [200]},
                "labels": ["healthy", "fault"],
                "window": {"length": 4, "stride": 4, "normalization": "none"},
                "seeds": [17, 29],
                "description": "statistics fixture",
            }
        ),
        encoding="utf-8",
    )
    return task


def _write_features(root: Path) -> Path:
    samples = []
    for load in (0, 100, 200):
        for recording in range(4):  # four recordings per partition
            for window in range(3):  # three windows per recording
                index = len(samples)
                label = "healthy" if (recording + load // 100) % 2 == 0 else "fault"
                marker = 0 if label == "healthy" else 1
                samples.append(
                    {
                        "window_id": f"rec-{load}-{recording}#w{window}",
                        "recording_id": f"rec-{load}-{recording}",
                        "start_sample": window * 4,
                        "end_sample": (window + 1) * 4,
                        "label": label,
                        "domains": {"load": load, "device": f"dev-{recording % 2}"},
                        "features": [float(marker), 0.5, float(index)],
                    }
                )
    features = root / "features.json"
    features.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_id": "stats-v1",
                "feature_names": ["marker", "const", "offset"],
                "preparation": {"kind": "stats-fixture"},
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return features


def _run(root: Path, model_id: str, factory) -> Path:
    import edgefault_bench.execution as execution

    root.mkdir(parents=True, exist_ok=True)
    original = execution._git_commit
    execution._git_commit = lambda: "abcdef1234567890"
    try:
        return run_benchmark(
            task_path=_write_task(root),
            model_id=model_id,
            output_path=root / f"{model_id}.json",
            features_path=_write_features(root),
            model_plugins={
                model_id: ModelPlugin(model_id, "sklearn", "stats-fixture", factory)
            },
            latency_warmup=0,
            latency_repeats=1,
        )
    finally:
        execution._git_commit = original


def test_group_bootstrap_reports_interval_and_determinism() -> None:
    rng = np.random.default_rng(3)
    units = np.asarray([f"r{index}" for index in range(10) for _ in range(5)])
    truth = np.asarray(["a", "b"] * 25)
    predictions = truth.copy()
    predictions[rng.choice(50, size=8, replace=False)] = "a"

    first = group_bootstrap(truth, predictions, units, n_resamples=500, seed=17)
    second = group_bootstrap(truth, predictions, units, n_resamples=500, seed=17)
    third = group_bootstrap(truth, predictions, units, n_resamples=500, seed=29)

    assert first.ci_low <= first.point <= first.ci_high
    assert first.unit_count == 10
    assert first.to_dict()["n_resamples"] == 500
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)
    assert (first.ci_low, first.ci_high) != (third.ci_low, third.ci_high)


def test_group_bootstrap_requires_two_groups_and_alignment() -> None:
    with pytest.raises(ValueError, match="two groups"):
        group_bootstrap(["a"], ["a"], ["only"])
    with pytest.raises(ValueError, match="equal lengths"):
        group_bootstrap(["a", "b"], ["a"], ["r1", "r1"])


def test_paired_group_bootstrap_direction() -> None:
    units = np.asarray([f"r{index}" for index in range(6) for _ in range(4)])
    truth = np.asarray(["a", "b"] * 12)
    perfect = truth.copy()
    degraded = truth.copy()
    degraded[::4] = "b"  # six 'a' samples are misclassified

    paired = paired_group_bootstrap(truth, perfect, degraded, units, n_resamples=400)
    assert paired.delta_point == pytest.approx(macro_f1(truth, perfect) - macro_f1(truth, degraded))
    assert paired.delta_point > 0.0
    assert paired.fraction_favouring_a == pytest.approx(1.0)


def test_bundle_statistics_and_prediction_retention(tmp_path: Path) -> None:
    output = _run(tmp_path, "perfect", lambda context: PerfectEstimator(context.seed))
    payload = load_result(output)

    assert payload["protocol"]["retain_predictions"] is True
    for run in payload["runs"]:
        retained = run["test_predictions"]
        assert len(retained["truth"]) == 12  # 4 recordings x 3 windows
        assert len(set(retained["recording_ids"])) == 4

    report = result_bundle_statistics(payload, unit="recording", n_resamples=200)
    assert report["unit"] == "recording"
    assert len(report["per_seed"]) == 2
    for entry in report["per_seed"]:
        interval = entry["group_bootstrap"]
        assert interval["unit_count"] == 4
        assert interval["ci_low"] <= entry["window_level_macro_f1"] <= interval["ci_high"] + 1e-9


def test_compare_bundles_pairs_shared_test_sets(tmp_path: Path) -> None:
    perfect = load_result(
        _run(tmp_path / "a", "perfect", lambda context: PerfectEstimator(context.seed))
    )
    noisy = load_result(
        _run(tmp_path / "b", "noisy", lambda context: NoisyEstimator(context.seed))
    )
    comparison = compare_bundles(perfect, noisy, unit="recording", n_resamples=200)
    assert comparison["model_a"] == "perfect"
    assert comparison["model_b"] == "noisy"
    for entry in comparison["per_seed"]:
        assert entry["comparison"]["delta_point"] >= 0.0


def test_statistics_cli_reports_intervals(tmp_path: Path, capsys) -> None:
    output = _run(tmp_path, "perfect", lambda context: PerfectEstimator(context.seed))
    exit_code = cli.main(
        [
            "statistics",
            "bootstrap",
            "--result",
            str(output),
            "--unit",
            "recording",
            "--resamples",
            "150",
        ]
    )
    rendered = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert rendered["per_seed"][0]["group_bootstrap"]["unit_count"] == 4
