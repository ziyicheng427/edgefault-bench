"""Portable feature-table contract between dataset and model plugins."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from edgefault_bench.contracts import DomainValue, TaskSpec


@dataclass(frozen=True)
class PreparedSample:
    """One model-ready row retaining recording and operating-condition identity."""

    recording_id: str
    label: str
    domains: Mapping[str, DomainValue]
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.recording_id.strip():
            raise ValueError("recording_id must be non-empty")
        if not self.label.strip():
            raise ValueError("label must be non-empty")
        if not self.domains:
            raise ValueError("domains must not be empty")
        object.__setattr__(self, "domains", MappingProxyType(dict(self.domains)))
        if not self.features:
            raise ValueError("features must not be empty")
        if not all(math.isfinite(value) for value in self.features):
            raise ValueError("features must be finite")


@dataclass(frozen=True)
class PreparedFeatureTable:
    """Versioned model input with provenance retained at sample level."""

    dataset_id: str
    feature_names: tuple[str, ...]
    samples: tuple[PreparedSample, ...]
    preparation: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        if not self.samples:
            raise ValueError("samples must not be empty")
        width = len(self.feature_names)
        if any(len(sample.features) != width for sample in self.samples):
            raise ValueError("every sample must match the declared feature width")
        object.__setattr__(self, "preparation", MappingProxyType(dict(self.preparation)))


@dataclass(frozen=True)
class PreparedTaskData:
    """Arrays and indices validated against one frozen task."""

    features: np.ndarray
    labels: np.ndarray
    evaluation_groups: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def load_prepared_feature_table(path: str | Path) -> PreparedFeatureTable:
    """Load a readable JSON feature table and reject malformed rows."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("prepared feature table requires schema_version 1")
    samples = tuple(
        PreparedSample(
            recording_id=item["recording_id"],
            label=item["label"],
            domains=item["domains"],
            features=tuple(float(value) for value in item["features"]),
        )
        for item in payload["samples"]
    )
    return PreparedFeatureTable(
        dataset_id=payload["dataset_id"],
        feature_names=tuple(payload["feature_names"]),
        samples=samples,
        preparation=payload.get("preparation", {}),
    )


def prepare_task_data(table: PreparedFeatureTable, task: TaskSpec) -> PreparedTaskData:
    """Validate and partition prepared rows without exposing test data to fitting."""

    if table.dataset_id != task.dataset_id:
        raise ValueError(
            f"prepared dataset {table.dataset_id!r} does not match task {task.dataset_id!r}"
        )
    allowed_labels = set(task.labels)
    partitions: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    recording_metadata: dict[str, tuple[str, DomainValue, DomainValue]] = {}
    evaluation_groups: list[str] = []
    for index, sample in enumerate(table.samples):
        if sample.label not in allowed_labels:
            raise ValueError(f"sample label {sample.label!r} is not declared by the task")
        try:
            held_out_value = sample.domains[task.domain_field]
            evaluation_value = sample.domains[task.evaluation_group_field]
        except KeyError as error:
            raise ValueError(
                f"sample {sample.recording_id!r} lacks required domain {error.args[0]!r}"
            ) from error
        partition = task.partition_for(held_out_value)
        if partition is None:
            raise ValueError(
                f"sample {sample.recording_id!r} is unassigned for {task.domain_field!r}"
            )
        identity = (sample.label, held_out_value, evaluation_value)
        prior = recording_metadata.setdefault(sample.recording_id, identity)
        if prior != identity:
            raise ValueError(
                f"recording {sample.recording_id!r} has inconsistent label or domain metadata"
            )
        partitions[partition].append(index)
        evaluation_groups.append(str(evaluation_value))

    empty = [name for name, indices in partitions.items() if not indices]
    if empty:
        raise ValueError(f"prepared task partitions must be non-empty: {empty}")
    features = np.asarray([sample.features for sample in table.samples], dtype=np.float64)
    labels = np.asarray([sample.label for sample in table.samples])
    return PreparedTaskData(
        features=features,
        labels=labels,
        evaluation_groups=np.asarray(evaluation_groups),
        train=np.asarray(partitions["train"], dtype=np.int64),
        validation=np.asarray(partitions["validation"], dtype=np.int64),
        test=np.asarray(partitions["test"], dtype=np.int64),
    )
