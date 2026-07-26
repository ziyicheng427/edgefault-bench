"""Versioned task manifests and deterministic partition assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edgefault_bench.data import SampleIndex, make_condition_split
from edgefault_bench.datasets.hust import WindowRecord


@dataclass(frozen=True)
class TaskManifest:
    task_id: str
    dataset_id: str
    domain_field: str
    evaluation_group_field: str
    partitions: dict[str, tuple[str | int, ...]]
    labels: tuple[str, ...]
    window_length: int
    stride: int
    normalization: str
    seeds: tuple[int, ...]
    description: str


def load_task_manifest(path: str | Path) -> TaskManifest:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    partitions = {
        name: tuple(payload["partitions"][name]) for name in ("train", "validation", "test")
    }
    manifest = TaskManifest(
        task_id=payload["task_id"],
        dataset_id=payload["dataset_id"],
        domain_field=payload["domain_field"],
        evaluation_group_field=payload["evaluation_group_field"],
        partitions=partitions,
        labels=tuple(payload["labels"]),
        window_length=int(payload["window"]["length"]),
        stride=int(payload["window"]["stride"]),
        normalization=payload["window"]["normalization"],
        seeds=tuple(int(seed) for seed in payload["seeds"]),
        description=payload["description"],
    )
    _validate_task_manifest(manifest)
    return manifest


def _validate_task_manifest(manifest: TaskManifest) -> None:
    if manifest.dataset_id != "hust-bearing-v3":
        raise ValueError("v1 task manifests must reference the pinned HUST v3 dataset")
    if manifest.domain_field not in {"load_w", "bearing_type"}:
        raise ValueError(f"unsupported domain field: {manifest.domain_field!r}")
    if manifest.evaluation_group_field not in {"load_w", "bearing_type"}:
        raise ValueError(f"unsupported evaluation group: {manifest.evaluation_group_field!r}")
    if manifest.evaluation_group_field == manifest.domain_field:
        raise ValueError("evaluation group must expose variation inside the held-out domain")
    if manifest.seeds != (17, 29, 43):
        raise ValueError("v1 benchmark seeds are frozen as 17, 29, and 43")
    if manifest.normalization != "per_window_zscore":
        raise ValueError("v1 HUST tasks require per-window z-score normalization")
    domains = {name: set(values) for name, values in manifest.partitions.items()}
    if any(not values for values in domains.values()):
        raise ValueError("every task partition must contain at least one domain")
    if domains["train"] & domains["validation"]:
        raise ValueError("train and validation domains overlap")
    if domains["train"] & domains["test"]:
        raise ValueError("train and test domains overlap")
    if domains["validation"] & domains["test"]:
        raise ValueError("validation and test domains overlap")


def split_window_records(records: tuple[WindowRecord, ...], manifest: TaskManifest):
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
