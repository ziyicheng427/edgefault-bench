#!/usr/bin/env bash
# Framework-overhead benchmark (T16, RQ4): kernel load/audit/validate costs at
# three table scales, compared against bare json parsing of the same files.
# Writes results/verification/framework-overhead-<date>.json.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run python - <<'EOF'
import json
import statistics
import tempfile
import time
from pathlib import Path

from edgefault_bench.prepared import (
    PreparedFeatureTable,
    PreparedSample,
    load_prepared_feature_table,
    write_prepared_feature_table,
)
from edgefault_bench.reporting import load_result
from edgefault_bench.schema import validate_artifact_schema

rng_rows = __import__("numpy").random.default_rng(11)


def build_table(rows: int) -> PreparedFeatureTable:
    import numpy as np

    rng = np.random.default_rng(rows)
    samples = tuple(
        PreparedSample(
            recording_id=f"rec-{row % 30}",
            label="healthy" if row % 2 else "fault",
            domains={"load": row % 3, "device": f"dev-{row % 5}"},
            features=tuple(float(v) for v in rng.normal(size=10)),
            window_id=f"rec-{row % 30}#w{row}",
            start_sample=row * 4,
            end_sample=row * 4 + 4,
        )
        for row in range(rows)
    )
    return PreparedFeatureTable(
        dataset_id="overhead-v1",
        feature_names=tuple(f"f{i}" for i in range(10)),
        samples=samples,
        preparation={"kind": "overhead-benchmark"},
    )


def timed(callback, repeats: int = 5) -> float:
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        callback()
        durations.append(time.perf_counter() - started)
    return round(statistics.median(durations), 6)


report = {"schema_version": 1, "scales": [], "note": (
    "Kernel cost = load_prepared_feature_table + schema validation; bare cost = "
    "json.loads on the same file. Median of 5 repeats; synthetic tables."
)}
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    for rows in (1_000, 10_000, 100_000):
        table = build_table(rows)
        path = write_prepared_feature_table(table, tmp_path / f"table-{rows}.json")
        kernel = timed(
            lambda: validate_artifact_schema(  # noqa: B023
                load_prepared_feature_table(path) and path, "prepared-feature-table"
            )
        )
        bare = timed(lambda: json.loads(path.read_text(encoding="utf-8")))  # noqa: B023
        report["scales"].append(
            {
                "rows": rows,
                "kernel_load_and_validate_seconds": kernel,
                "bare_json_load_seconds": bare,
                "overhead_ratio": round(kernel / bare, 2) if bare > 0 else None,
            }
        )

# Result-bundle validation cost on a realistic bundle.
import edgefault_bench.execution as execution
from edgefault_bench.plugins import ModelPlugin

class Trivial:
    def __init__(self, seed):
        self.seed = seed
    def fit(self, features, labels):
        return self
    def predict(self, features):
        return ["healthy"] * len(features)
    def get_params(self):
        return {"seed": self.seed}

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    task = tmp_path / "task.json"
    task.write_text(json.dumps({
        "task_id": "overhead-v1", "dataset_id": "overhead-v1",
        "domain_field": "load", "evaluation_group_field": "device",
        "partitions": {"train": [0], "validation": [1], "test": [2]},
        "labels": ["healthy", "fault"],
        "window": {"length": 4, "stride": 4, "normalization": "none"},
        "seeds": [17, 29, 43], "description": "overhead",
    }), encoding="utf-8")
    features = write_prepared_feature_table(build_table(3_000), tmp_path / "features.json")
    original = execution._git_commit
    execution._git_commit = lambda: "abcdef1234567890"
    try:
        bundle = execution.run_benchmark(
            task_path=task, model_id="trivial",
            output_path=tmp_path / "result.json", features_path=features,
            model_plugins={"trivial": ModelPlugin("trivial", "sklearn", "fixture",
                                                  lambda c: Trivial(c.seed))},
            latency_warmup=0, latency_repeats=1,
        )
    finally:
        execution._git_commit = original
    report["result_validation_seconds"] = timed(lambda: load_result(bundle))

out = Path(f"results/verification/framework-overhead-{__import__('datetime').date.today().isoformat()}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
EOF
