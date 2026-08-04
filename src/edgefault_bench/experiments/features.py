"""Three-seed signal-feature baseline for registered frozen tasks."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn

from edgefault_bench.baseline import make_feature_baseline
from edgefault_bench.datasets.hust import (
    build_window_records,
    load_hust_manifest,
    load_hust_signal,
    verify_hust_file,
)
from edgefault_bench.datasets.mehran import (
    build_mehran_window_records,
    load_mehran_manifest,
    load_mehran_signal,
    preprocess_mehran_windows,
    verify_mehran_file,
)
from edgefault_bench.evaluation import condition_metrics
from edgefault_bench.features import (
    FEATURE_NAMES,
    extract_features,
    extract_multichannel_features,
    multichannel_feature_names,
)
from edgefault_bench.tasks import load_task_manifest, split_window_records


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _load_hust_feature_table(
    manifest_path: Path,
    raw_dir: Path,
    *,
    window_length: int,
    stride: int,
):
    _, files = load_hust_manifest(manifest_path)
    all_features: list[np.ndarray] = []
    for specification in sorted(files, key=lambda item: item.filename):
        path = raw_dir / specification.filename
        verify_hust_file(path, specification)
        signal, _ = load_hust_signal(path)
        starts = range(0, len(signal) - window_length + 1, stride)
        windows = np.stack([signal[start : start + window_length] for start in starts])
        all_features.append(extract_features(windows, sampling_rate=51_200.0))
    features = np.vstack(all_features)
    records = build_window_records(files, window_length=window_length, stride=stride)
    if len(features) != len(records):
        raise RuntimeError("feature rows and deterministic window records are misaligned")
    return features, records, FEATURE_NAMES


def _load_mehran_feature_table(
    manifest_path: Path,
    raw_dir: Path,
    *,
    window_length: int,
    stride: int,
):
    payload, files = load_mehran_manifest(manifest_path)
    all_features: list[np.ndarray] = []
    sample_counts: dict[str, int] = {}
    for specification in sorted(files, key=lambda item: item.filename):
        path = raw_dir / specification.filename
        verify_mehran_file(path, specification)
        signal = load_mehran_signal(path, minimum_samples=window_length)
        sample_counts[specification.filename] = len(signal)
        windows = preprocess_mehran_windows(
            signal, window_length=window_length, stride=stride
        )
        all_features.append(
            extract_multichannel_features(
                windows, sampling_rate=float(payload["sampling_rate_hz"])
            )
        )
    features = np.vstack(all_features)
    records = build_mehran_window_records(
        files,
        sample_counts,
        window_length=window_length,
        stride=stride,
    )
    if len(features) != len(records):
        raise RuntimeError("feature rows and Mehran window records are misaligned")
    return features, records, multichannel_feature_names(("x", "y", "z"))


def load_feature_table(
    manifest_path: Path,
    raw_dir: Path,
    *,
    window_length: int = 4096,
    stride: int = 4096,
):
    """Load verified files one at a time through the registered adapter."""

    dataset_id = json.loads(manifest_path.read_text(encoding="utf-8")).get("dataset_id")
    if dataset_id == "hust-bearing-v3":
        return _load_hust_feature_table(
            manifest_path,
            raw_dir,
            window_length=window_length,
            stride=stride,
        )
    if dataset_id == "mehran-triaxial-bearing-v2":
        return _load_mehran_feature_table(
            manifest_path,
            raw_dir,
            window_length=window_length,
            stride=stride,
        )
    raise ValueError(f"unsupported feature dataset: {dataset_id!r}")


def _latency_ms(model, sample: np.ndarray, *, warmup: int = 100, repeats: int = 1000):
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


def run_task(
    *, task_path: Path, manifest_path: Path, raw_dir: Path, output_dir: Path
) -> Path:
    task = load_task_manifest(task_path)
    features, records, feature_names = load_feature_table(
        manifest_path,
        raw_dir,
        window_length=task.window_length,
        stride=task.stride,
    )
    selected, split = split_window_records(records, task)
    labels = np.asarray([record.label for record in selected])
    evaluation_groups = np.asarray(
        [str(record.domain_value(task.evaluation_group_field)) for record in selected]
    )
    if len(selected) != len(features):
        raise RuntimeError("task selection changed the frozen feature row set")

    runs: list[dict[str, object]] = []
    for seed in task.seeds:
        model = make_feature_baseline(seed=seed)
        started = time.perf_counter()
        model.fit(features[split.train], labels[split.train])
        fit_seconds = time.perf_counter() - started
        validation_predictions = model.predict(features[split.validation])
        test_predictions = model.predict(features[split.test])
        validation = condition_metrics(
            labels[split.validation],
            validation_predictions,
            evaluation_groups[split.validation],
        )
        test = condition_metrics(
            labels[split.test], test_predictions, evaluation_groups[split.test]
        )
        classifier = model.named_steps["classifier"]
        parameters = int(classifier.coef_.size + classifier.intercept_.size)
        runs.append(
            {
                "seed": seed,
                "fit_seconds": fit_seconds,
                "validation": validation.to_dict(),
                "test": test.to_dict(),
                "complexity": {
                    "trainable_parameters": parameters,
                    "serialized_size_bytes": len(pickle.dumps(model)),
                    "process_peak_rss_bytes": _peak_rss_bytes(),
                },
                "latency": _latency_ms(model, features[split.test[:1]]),
            }
        )

    payload = {
        "schema_version": 1,
        "benchmark_id": (
            "edgefault-bench-v1"
            if task.dataset_id == "hust-bearing-v3"
            else "edgefault-bench-v1.1"
        ),
        "model_id": "signal_features_logreg",
        "task_id": task.task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "protocol": {
            "task_manifest": str(task_path),
            "dataset_manifest": str(manifest_path),
            "feature_names": list(feature_names),
            "input_channels": len(feature_names) // len(FEATURE_NAMES),
            "test_used_for_selection": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "partition_sizes": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "runs": runs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task.task_id}__signal_features_logreg.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("registry/hust_v3.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hust_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v1"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = run_task(
        task_path=args.task,
        manifest_path=args.manifest,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )
    print(output)


if __name__ == "__main__":
    main()
