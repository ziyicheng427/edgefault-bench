"""Motivated-example experiment: does recording-level leakage inflate scores?

The paper's Figure 1 contrasts a conventional window-level random split
(windows of one recording straddle train/test) with the grouping the kernel
enforces. To keep the figure free of invented numbers, this script measures
both splits on the real Mehran v2 archive with the same model, policy, and
seeds, and archives the result as a verification artefact.

Only the kernel's public API is used; the leaky split deliberately bypasses
the kernel's partition assignment because that is exactly what a conventional
script pipeline would do (the kernel itself rejects such splits, M-LEAK-01).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from edgefault_bench.execution import DataView  # noqa: E402
from edgefault_bench.datasets import MehranV2Adapter  # noqa: E402
from edgefault_bench.evaluation import condition_metrics  # noqa: E402
from edgefault_bench.execution import (  # noqa: E402
    TorchEstimator,
    TrainingPolicy,
    seed_everything,
)
from edgefault_bench.models import build_model  # noqa: E402
from edgefault_bench.preprocessing import PerWindowZScore  # noqa: E402
from edgefault_bench.tasks import load_task_spec  # noqa: E402
from edgefault_bench.windows import extract_windows  # noqa: E402

TASK = ROOT / "registry/tasks/mehran_load_100_to_300_v1.json"
MANIFEST = ROOT / "registry/mehran_v2.json"
SEEDS = (17, 29, 43)
POLICY = TrainingPolicy(
    policy_version="audit-1",
    optimizer="adam",
    learning_rate=1e-3,
    max_epochs=2,
    batch_size=64,
    patience=1,
)


def split_indices(
    recordings: np.ndarray, *, by_recording: bool, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window-level random split, or recording-grouped random split."""

    total = len(recordings)
    if not by_recording:
        order = np.random.default_rng(seed).permutation(total)
        n_train, n_val = int(0.6 * total), int(0.2 * total)
        return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]

    unique = np.unique(recordings)
    order = np.random.default_rng(seed).permutation(len(unique))
    n_train, n_val = int(0.6 * len(unique)), max(1, int(0.2 * len(unique)))
    groups = {
        "train": set(unique[order[:n_train]]),
        "validation": set(unique[order[n_train : n_train + n_val]]),
        "test": set(unique[order[n_train + n_val :]]),
    }
    buckets: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for index, recording in enumerate(recordings):
        for name, members in groups.items():
            if recording in members:
                buckets[name].append(index)
                break
    return (
        np.asarray(buckets["train"]),
        np.asarray(buckets["validation"]),
        np.asarray(buckets["test"]),
    )


def run_split(
    x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    recordings: np.ndarray,
    task,
    *,
    by_recording: bool,
) -> list[dict]:
    runs = []
    for seed in SEEDS:
        seed_everything(seed)
        train_idx, val_idx, test_idx = split_indices(
            recordings, by_recording=by_recording, seed=seed
        )
        model = build_model(
            "compact_depthwise_cnn_1d",
            num_classes=len(task.labels),
            in_channels=x.shape[1],
        )
        estimator = TorchEstimator(model, labels=task.labels, seed=seed)
        estimator.fit(
            DataView(x[train_idx], labels[train_idx], groups[train_idx]),
            DataView(x[val_idx], labels[val_idx], groups[val_idx]),
            POLICY,
        )
        predictions = estimator.predict(x[test_idx])
        metrics = condition_metrics(labels[test_idx], predictions, groups[test_idx])
        overlap = len(set(recordings[train_idx]) & set(recordings[test_idx]))
        runs.append(
            {
                "seed": seed,
                "test_macro_f1": metrics.macro_f1,
                "recordings_shared_between_train_and_test": overlap,
                "train_windows": int(len(train_idx)),
                "test_windows": int(len(test_idx)),
            }
        )
    return runs


def main() -> None:
    task = load_task_spec(TASK)
    adapter = MehranV2Adapter(MANIFEST)
    batch = extract_windows(adapter, task)
    x = PerWindowZScore().transform(batch)
    labels = np.asarray([sample.label for sample in batch.samples], dtype=str)
    groups = np.asarray(
        [str(sample.domains[task.evaluation_group_field]) for sample in batch.samples],
        dtype=str,
    )
    recordings = np.asarray(
        [sample.recording_id for sample in batch.samples], dtype=str
    )

    leaky = run_split(x, labels, groups, recordings, task, by_recording=False)
    honest = run_split(x, labels, groups, recordings, task, by_recording=True)
    leaky_mean = float(np.mean([r["test_macro_f1"] for r in leaky]))
    honest_mean = float(np.mean([r["test_macro_f1"] for r in honest]))

    report = {
        "schema_version": 1,
        "claim": (
            "Window-level random splitting lets windows of one recording "
            "straddle the train/test boundary; recording-grouped splitting "
            "with the same model, policy, and seeds does not."
        ),
        "task_id": task.task_id,
        "dataset": "mehran-triaxial-bearing-v2",
        "model": "compact_depthwise_cnn_1d",
        "policy": POLICY.to_dict(),
        "window_level_random_split": leaky,
        "recording_grouped_split": honest,
        "mean_macro_f1_leaky": leaky_mean,
        "mean_macro_f1_grouped": honest_mean,
        "inflation": leaky_mean - honest_mean,
        "scope": (
            "Audit-policy demonstration for the motivated example; both splits "
            "draw train and test from the same operating conditions, so the "
            "score gap isolates the leakage effect. Not a benchmark result."
        ),
        "generated": date.today().isoformat(),
    }
    out = ROOT / "results/verification/leakage-motivated-example-2026-11-07.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
