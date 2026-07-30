"""Versioned task specifications and deterministic partition assignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edgefault_bench.contracts import TaskSpec, WindowSpec
from edgefault_bench.data import SampleIndex, make_condition_split
from edgefault_bench.datasets.hust import WindowRecord

TaskManifest = TaskSpec

_FROZEN_HUST_V1_TASKS = {
    "hust-load-0-to-400-v1",
    "hust-load-400-to-0-v1",
    "hust-device-6204-6206-to-6208-v1",
}


def load_task_spec(path: str | Path) -> TaskSpec:
    """Load a dataset-independent task while enforcing known frozen profiles."""

    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = TaskSpec(
        task_id=payload["task_id"],
        dataset_id=payload["dataset_id"],
        domain_field=payload["domain_field"],
        evaluation_group_field=payload["evaluation_group_field"],
        partitions={
            name: tuple(payload["partitions"][name])
            for name in ("train", "validation", "test")
        },
        labels=tuple(payload["labels"]),
        window=WindowSpec(
            length=int(payload["window"]["length"]),
            stride=int(payload["window"]["stride"]),
            normalization=payload["window"]["normalization"],
        ),
        seeds=tuple(int(seed) for seed in payload["seeds"]),
        description=payload["description"],
        metadata={
            key: value
            for key, value in payload.items()
            if key
            not in {
                "task_id",
                "dataset_id",
                "domain_field",
                "evaluation_group_field",
                "partitions",
                "labels",
                "window",
                "seeds",
                "description",
            }
        },
    )
    _validate_frozen_profile(manifest)
    return manifest


def load_task_manifest(path: str | Path) -> TaskManifest:
    """Backward-compatible name for :func:`load_task_spec`."""

    return load_task_spec(path)


def _validate_frozen_profile(manifest: TaskSpec) -> None:
    if manifest.task_id not in _FROZEN_HUST_V1_TASKS:
        return
    if manifest.dataset_id != "hust-bearing-v3":
        raise ValueError("frozen HUST v1 tasks must reference the pinned HUST v3 dataset")
    if manifest.domain_field not in {"load_w", "bearing_type"}:
        raise ValueError(f"unsupported HUST v1 domain field: {manifest.domain_field!r}")
    if manifest.evaluation_group_field not in {"load_w", "bearing_type"}:
        raise ValueError(
            f"unsupported HUST v1 evaluation group: {manifest.evaluation_group_field!r}"
        )
    if manifest.seeds != (17, 29, 43):
        raise ValueError("v1 benchmark seeds are frozen as 17, 29, and 43")
    if manifest.normalization != "per_window_zscore":
        raise ValueError("v1 HUST tasks require per-window z-score normalization")


def split_window_records(records: tuple[WindowRecord, ...], manifest: TaskSpec):
    """Return a validated split whose recording identifiers cannot cross partitions."""

    allowed_labels = set(manifest.labels)
    selected = tuple(record for record in records if record.label in allowed_labels)
    samples = tuple(
        SampleIndex(
            condition=str(record.domain_value(manifest.domain_field)),
            recording=record.recording,
            label=record.label,
        )
        for record in selected
    )
    partition_values = {
        name: {str(value) for value in values} for name, values in manifest.partitions.items()
    }
    return selected, make_condition_split(
        samples,
        train_conditions=partition_values["train"],
        validation_conditions=partition_values["validation"],
        test_conditions=partition_values["test"],
    )
