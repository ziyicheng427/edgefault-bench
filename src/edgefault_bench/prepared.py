"""Portable feature-table contract between dataset and model plugins."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from edgefault_bench.contracts import DomainValue, TaskSpec

SUPPORTED_SCHEMA_VERSIONS = (1, 2)


@dataclass(frozen=True)
class PreparedSample:
    """One model-ready row retaining recording and window identity."""

    recording_id: str
    label: str
    domains: Mapping[str, DomainValue]
    features: tuple[float, ...]
    window_id: str | None = None
    start_sample: int | None = None
    end_sample: int | None = None

    def __post_init__(self) -> None:
        if not self.recording_id.strip():
            raise ValueError("recording_id must be non-empty")
        if not self.label.strip():
            raise ValueError("label must be non-empty")
        if not self.domains:
            raise ValueError("domains must not be empty")
        object.__setattr__(self, "domains", MappingProxyType(dict(self.domains)))
        if not self.features:
            raise ValueError("features must be non-empty")
        if not all(math.isfinite(value) for value in self.features):
            raise ValueError("features must be finite")
        window_fields = (self.window_id, self.start_sample, self.end_sample)
        if any(field is None for field in window_fields):
            if any(field is not None for field in window_fields):
                raise ValueError(
                    "window identity fields must be present together or absent together"
                )
        else:
            if not self.window_id or not self.window_id.strip():  # type: ignore[union-attr]
                raise ValueError("window_id must be non-empty")
            if self.start_sample < 0 or self.end_sample <= self.start_sample:  # type: ignore[operator]
                raise ValueError("end_sample must exceed non-negative start_sample")

    @property
    def has_window_identity(self) -> bool:
        return self.window_id is not None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "recording_id": self.recording_id,
            "label": self.label,
            "domains": dict(self.domains),
            "features": list(self.features),
        }
        if self.has_window_identity:
            payload.update(
                {
                    "window_id": self.window_id,
                    "start_sample": self.start_sample,
                    "end_sample": self.end_sample,
                }
            )
        return payload


@dataclass(frozen=True)
class PreparedFeatureTable:
    """Versioned model input with provenance retained at sample level."""

    dataset_id: str
    feature_names: tuple[str, ...]
    samples: tuple[PreparedSample, ...]
    preparation: Mapping[str, object]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"schema_version must be one of {SUPPORTED_SCHEMA_VERSIONS}")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        if not self.samples:
            raise ValueError("samples must not be empty")
        width = len(self.feature_names)
        if any(len(sample.features) != width for sample in self.samples):
            raise ValueError("every sample must match the declared feature width")
        window_ids = [sample.window_id for sample in self.samples if sample.has_window_identity]
        if window_ids:
            if len(window_ids) != len(self.samples):
                raise ValueError(
                    "window identity must cover every sample; partial coverage defeats "
                    "window-level leakage audits"
                )
            if len(set(window_ids)) != len(window_ids):
                raise ValueError("window_id values must be unique across the table")
        object.__setattr__(self, "preparation", MappingProxyType(dict(self.preparation)))

    @property
    def has_window_identity(self) -> bool:
        return bool(self.samples) and self.samples[0].has_window_identity

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "feature_names": list(self.feature_names),
            "preparation": dict(self.preparation),
            "samples": [sample.to_dict() for sample in self.samples],
        }


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
    """Load a JSON or Parquet feature table and reject malformed rows."""

    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        return _load_parquet_table(table_path)
    payload = json.loads(table_path.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"prepared feature table requires schema_version in {SUPPORTED_SCHEMA_VERSIONS}"
        )
    samples = tuple(
        PreparedSample(
            recording_id=item["recording_id"],
            label=item["label"],
            domains=item["domains"],
            features=tuple(float(value) for value in item["features"]),
            window_id=item.get("window_id"),
            start_sample=item.get("start_sample"),
            end_sample=item.get("end_sample"),
        )
        for item in payload["samples"]
    )
    return PreparedFeatureTable(
        dataset_id=payload["dataset_id"],
        feature_names=tuple(payload["feature_names"]),
        samples=samples,
        preparation=payload.get("preparation", {}),
        schema_version=int(version),
    )


def write_prepared_feature_table(
    table: PreparedFeatureTable, path: str | Path
) -> Path:
    """Persist a table as pretty JSON or Parquet, selected by file suffix."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        _write_parquet_table(table, output)
    else:
        output.write_text(
            json.dumps(table.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return output


def _table_metadata_bytes(table: PreparedFeatureTable) -> dict[bytes, bytes]:
    header = {
        "schema_version": table.schema_version,
        "dataset_id": table.dataset_id,
        "feature_names": list(table.feature_names),
        "preparation": dict(table.preparation),
    }
    return {b"edgefault_prepared_table": json.dumps(header).encode("utf-8")}


def _write_parquet_table(table: PreparedFeatureTable, path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:  # pragma: no cover - exercised via docs
        raise ValueError(
            "Parquet output requires the optional 'columnar' extra "
            "(pip install 'edgefault-bench[columnar]')"
        ) from error

    columns: dict[str, pa.Array] = {}
    if table.has_window_identity:
        columns["window_id"] = pa.array([sample.window_id for sample in table.samples])
        columns["start_sample"] = pa.array(
            [sample.start_sample for sample in table.samples], type=pa.int64()
        )
        columns["end_sample"] = pa.array(
            [sample.end_sample for sample in table.samples], type=pa.int64()
        )
    columns["recording_id"] = pa.array([sample.recording_id for sample in table.samples])
    columns["label"] = pa.array([sample.label for sample in table.samples])
    columns["domains"] = pa.array(
        [json.dumps(dict(sample.domains), sort_keys=True) for sample in table.samples]
    )
    features = np.asarray([sample.features for sample in table.samples], dtype=np.float64)
    for index, name in enumerate(table.feature_names):
        columns[name] = pa.array(features[:, index])
    arrow_table = pa.table(columns).replace_schema_metadata(_table_metadata_bytes(table))
    pq.write_table(arrow_table, path)


def _load_parquet_table(path: Path) -> PreparedFeatureTable:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:  # pragma: no cover - exercised via docs
        raise ValueError(
            "Parquet input requires the optional 'columnar' extra "
            "(pip install 'edgefault-bench[columnar]')"
        ) from error

    arrow_table = pq.read_table(path)
    raw_metadata = arrow_table.schema.metadata or {}
    header_payload = raw_metadata.get(b"edgefault_prepared_table")
    if header_payload is None:
        raise ValueError(f"Parquet table is missing EdgeFault-Bench metadata: {path}")
    header = json.loads(header_payload.decode("utf-8"))
    version = int(header["schema_version"])
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"prepared feature table requires schema_version in {SUPPORTED_SCHEMA_VERSIONS}"
        )
    column_names = arrow_table.column_names
    has_windows = "window_id" in column_names
    samples = []
    count = arrow_table.num_rows
    recordings = arrow_table.column("recording_id").to_pylist()
    labels = arrow_table.column("label").to_pylist()
    domains = [json.loads(value) for value in arrow_table.column("domains").to_pylist()]
    feature_names = list(header["feature_names"])
    features = np.column_stack(
        [
            np.asarray(arrow_table.column(name).to_pylist(), dtype=np.float64)
            for name in feature_names
        ]
    )
    for index in range(count):
        samples.append(
            PreparedSample(
                recording_id=recordings[index],
                label=labels[index],
                domains=domains[index],
                features=tuple(float(value) for value in features[index]),
                window_id=arrow_table.column("window_id")[index].as_py()
                if has_windows
                else None,
                start_sample=arrow_table.column("start_sample")[index].as_py()
                if has_windows
                else None,
                end_sample=arrow_table.column("end_sample")[index].as_py()
                if has_windows
                else None,
            )
        )
    return PreparedFeatureTable(
        dataset_id=str(header["dataset_id"]),
        feature_names=tuple(feature_names),
        samples=tuple(samples),
        preparation=header.get("preparation", {}),
        schema_version=version,
    )


def prepare_task_data(
    table: PreparedFeatureTable, task: TaskSpec, *, recordings: Sequence = ()
) -> PreparedTaskData:
    """Validate and partition prepared rows without exposing test data to fitting.

    When ``recordings`` (canonical adapter metadata) is provided, every row's
    induced partition is cross-checked against the partition of its recording,
    which detects a test recording whose rows were relabelled into the training
    side even when the table itself looks internally consistent.
    """

    if table.dataset_id != task.dataset_id:
        raise ValueError(
            f"prepared dataset {table.dataset_id!r} does not match task {task.dataset_id!r}"
        )
    allowed_labels = set(task.labels)
    partitions: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    recording_metadata: dict[str, tuple[object, ...]] = {}
    evaluation_groups: list[str] = []
    for index, sample in enumerate(table.samples):
        if sample.label not in allowed_labels:
            raise ValueError(f"sample label {sample.label!r} is not declared by the task")
        missing = [
            field
            for field in (*task.held_out_fields, task.evaluation_group_field)
            if field not in sample.domains
        ]
        if missing:
            raise ValueError(
                f"sample {sample.recording_id!r} lacks required domains {missing}"
            )
        partition = task.partition_for_sample(sample.domains)
        if partition is None:
            raise ValueError(
                f"sample {sample.recording_id!r} has an unassigned held-out condition"
            )
        evaluation_value = sample.domains[task.evaluation_group_field]
        identity = (
            sample.label,
            *(sample.domains[field] for field in task.held_out_fields),
            evaluation_value,
        )
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
    if recordings:
        _audit_rows_against_recordings(table, task, recordings, partitions)
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


def _audit_rows_against_recordings(
    table: PreparedFeatureTable,
    task: TaskSpec,
    recordings: Sequence,
    partitions: Mapping[str, Sequence[int]],
) -> None:
    """Fail closed when table rows contradict canonical recording partitions."""

    from edgefault_bench.contracts import Recording

    canonical: dict[str, Recording] = {}
    for record in recordings:
        if not isinstance(record, Recording):
            raise TypeError("recordings must contain Recording values")
        canonical[record.recording_id] = record
    index_partitions = {
        index: name for name, indices in partitions.items() for index in indices
    }
    for index, sample in enumerate(table.samples):
        record = canonical.get(sample.recording_id)
        if record is None:
            raise ValueError(
                f"table references unknown recording {sample.recording_id!r}; "
                "it is not part of the canonical dataset metadata"
            )
        if sample.label != record.label:
            raise ValueError(
                f"recording {sample.recording_id!r} is labelled {sample.label!r} in the "
                f"table but {record.label!r} in the dataset metadata"
            )
        row_partition = index_partitions[index]
        canonical_partition = task.partition_for_sample(record.domains)
        if canonical_partition != row_partition:
            raise ValueError(
                f"recording {sample.recording_id!r} belongs to partition "
                f"{canonical_partition!r} but its table rows were assigned "
                f"{row_partition!r}: relabelled partitions are rejected"
            )
