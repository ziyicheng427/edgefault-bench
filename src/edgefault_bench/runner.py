"""Backend-aware execution of installed model plugins on prepared feature tables."""

from __future__ import annotations

import json
import os
import pickle
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import sklearn

from edgefault_bench.evaluation import condition_metrics
from edgefault_bench.plugins import ModelBuildContext, ModelPlugin, discover_model_plugins
from edgefault_bench.prepared import load_prepared_feature_table, prepare_task_data
from edgefault_bench.tasks import load_task_spec


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _distribution_version(source: str) -> str:
    try:
        return metadata.version(source)
    except metadata.PackageNotFoundError:
        return "unknown"


def _latency_ms(model: object, sample: np.ndarray, *, warmup: int, repeats: int) -> dict:
    if warmup < 0 or repeats <= 0:
        raise ValueError("latency warmup must be non-negative and repeats must be positive")
    for _ in range(warmup):
        model.predict(sample)
    durations = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        started = time.perf_counter_ns()
        model.predict(sample)
        durations[index] = (time.perf_counter_ns() - started) / 1_000_000.0
    return {
        "batch_size": 1,
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
    }


def _stored_numeric_values(model: object) -> int:
    """Count learned numeric array values without claiming they are trainable weights."""

    total = 0
    for name, value in vars(model).items():
        if not name.endswith("_"):
            continue
        if isinstance(value, dict):
            total += sum(np.asarray(item).size for item in value.values())
        elif isinstance(value, (np.ndarray, list, tuple)):
            total += np.asarray(value).size
    return int(total)


def run_prepared_benchmark(
    *,
    task_path: Path,
    prepared_path: Path,
    model_id: str,
    output_path: Path,
    latency_warmup: int = 20,
    latency_repeats: int = 100,
    plugins: dict[str, ModelPlugin] | None = None,
) -> Path:
    """Fit one installed sklearn plugin per frozen seed and write a validated result shape."""

    task = load_task_spec(task_path)
    table = load_prepared_feature_table(prepared_path)
    data = prepare_task_data(table, task)
    registry = discover_model_plugins() if plugins is None else plugins
    try:
        plugin = registry[model_id]
    except KeyError as error:
        raise ValueError(f"no model plugin is registered for {model_id!r}") from error
    if plugin.backend != "sklearn":
        raise ValueError(
            f"prepared feature runner requires backend 'sklearn', found {plugin.backend!r}"
        )

    runs: list[dict[str, object]] = []
    for seed in task.seeds:
        model = plugin.create(ModelBuildContext(len(task.labels), len(table.feature_names), seed))
        started = time.perf_counter()
        model.fit(data.features[data.train], data.labels[data.train])
        fit_seconds = time.perf_counter() - started
        validation_predictions = model.predict(data.features[data.validation])
        test_predictions = model.predict(data.features[data.test])
        validation = condition_metrics(
            data.labels[data.validation],
            validation_predictions,
            data.evaluation_groups[data.validation],
        )
        test = condition_metrics(
            data.labels[data.test], test_predictions, data.evaluation_groups[data.test]
        )
        runs.append(
            {
                "seed": seed,
                "fit_seconds": fit_seconds,
                "validation": validation.to_dict(),
                "test": test.to_dict(),
                "complexity": {
                    "trainable_parameters": _stored_numeric_values(model),
                    "parameter_count_scope": "learned numeric values stored in fitted attributes",
                    "serialized_size_bytes": len(pickle.dumps(model)),
                },
                "latency": _latency_ms(
                    model,
                    data.features[data.test[:1]],
                    warmup=latency_warmup,
                    repeats=latency_repeats,
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "benchmark_id": "edgefault-bench-plugin-v1",
        "model_id": model_id,
        "task_id": task.task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "protocol": {
            "task_manifest": str(task_path),
            "prepared_feature_table": str(prepared_path),
            "feature_names": list(table.feature_names),
            "preparation": dict(table.preparation),
            "test_used_for_selection": False,
        },
        "model_plugin": {
            "backend": plugin.backend,
            "source_distribution": plugin.source,
            "source_version": _distribution_version(plugin.source),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "partition_sizes": {
            "train": len(data.train),
            "validation": len(data.validation),
            "test": len(data.test),
        },
        "runs": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
