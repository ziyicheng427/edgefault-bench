"""Dataset-agnostic contracts for extensible condition-shift benchmarks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

DomainValue: TypeAlias = str | int | float | bool
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PARTITION_NAMES = ("train", "validation", "test")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _freeze_domains(values: Mapping[str, DomainValue]) -> Mapping[str, DomainValue]:
    frozen: dict[str, DomainValue] = {}
    for name, value in values.items():
        _require_text(name, "domain name")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"domain {name!r} has unsupported value: {value!r}")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"domain {name!r} has an empty value")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"domain {name!r} must be finite")
        frozen[name] = value
    if not frozen:
        raise ValueError("domains must contain at least one field")
    return MappingProxyType(frozen)


def _freeze_partitions(
    values: Mapping[str, Sequence[DomainValue]],
) -> Mapping[str, tuple[DomainValue, ...]]:
    if set(values) != set(_PARTITION_NAMES):
        raise ValueError("partitions must contain exactly train, validation, and test")
    frozen: dict[str, tuple[DomainValue, ...]] = {}
    for name in _PARTITION_NAMES:
        partition = tuple(values[name])
        if not partition:
            raise ValueError(f"partition {name!r} must not be empty")
        if len(set(partition)) != len(partition):
            raise ValueError(f"partition {name!r} contains duplicate domain values")
        frozen[name] = partition
    for position, left_name in enumerate(_PARTITION_NAMES):
        for right_name in _PARTITION_NAMES[position + 1 :]:
            overlap = set(frozen[left_name]) & set(frozen[right_name])
            if overlap:
                raise ValueError(
                    f"partitions {left_name!r} and {right_name!r} overlap: "
                    f"{sorted(overlap, key=str)}"
                )
    return MappingProxyType(frozen)


def _freeze_condition_partitions(
    values: Mapping[str, Sequence[Sequence[DomainValue]]],
) -> Mapping[str, tuple[tuple[DomainValue, ...], ...]]:
    """Freeze partitions over tuples of combined condition values."""

    if set(values) != set(_PARTITION_NAMES):
        raise ValueError(
            "condition_partitions must contain exactly train, validation, and test"
        )
    frozen: dict[str, tuple[tuple[DomainValue, ...], ...]] = {}
    for name in _PARTITION_NAMES:
        partition = tuple(tuple(combination) for combination in values[name])
        if not partition:
            raise ValueError(f"condition partition {name!r} must not be empty")
        if len(set(partition)) != len(partition):
            raise ValueError(
                f"condition partition {name!r} contains duplicate combinations"
            )
        frozen[name] = partition
    for position, left_name in enumerate(_PARTITION_NAMES):
        for right_name in _PARTITION_NAMES[position + 1 :]:
            overlap = set(frozen[left_name]) & set(frozen[right_name])
            if overlap:
                raise ValueError(
                    f"condition partitions {left_name!r} and {right_name!r} overlap"
                )
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class DatasetMetadata:
    """Stable identity and domain vocabulary exposed by a dataset adapter."""

    dataset_id: str
    version: str
    title: str
    license_spdx: str
    source_url: str
    domain_fields: tuple[str, ...]
    doi: str | None = None

    def __post_init__(self) -> None:
        for name in ("dataset_id", "version", "title", "license_spdx", "source_url"):
            _require_text(getattr(self, name), name)
        if not self.domain_fields or len(set(self.domain_fields)) != len(self.domain_fields):
            raise ValueError("domain_fields must contain unique field names")
        for field_name in self.domain_fields:
            _require_text(field_name, "domain field")
        if self.doi is not None:
            _require_text(self.doi, "doi")

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "title": self.title,
            "license_spdx": self.license_spdx,
            "source_url": self.source_url,
            "domain_fields": list(self.domain_fields),
            "doi": self.doi,
        }


@dataclass(frozen=True)
class Recording:
    """Canonical metadata for one independently acquired sensor recording."""

    dataset_id: str
    recording_id: str
    source_file: str
    label: str
    domains: Mapping[str, DomainValue]
    sample_rate_hz: float | None = None
    sample_count: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("dataset_id", "recording_id", "source_file", "label"):
            _require_text(getattr(self, name), name)
        object.__setattr__(self, "domains", _freeze_domains(self.domains))
        if self.sample_rate_hz is not None and (
            not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0
        ):
            raise ValueError("sample_rate_hz must be finite and positive")
        if self.sample_count is not None and self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.sha256 is not None and _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")

    def domain_value(self, field_name: str) -> DomainValue:
        try:
            return self.domains[field_name]
        except KeyError as error:
            raise ValueError(
                f"recording {self.recording_id!r} has no domain field {field_name!r}"
            ) from error

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "recording_id": self.recording_id,
            "source_file": self.source_file,
            "label": self.label,
            "domains": dict(self.domains),
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": self.sample_count,
            "sha256": self.sha256,
        }


@runtime_checkable
class DatasetAdapter(Protocol):
    """Minimal extension point implemented by a public dataset integration."""

    @property
    def metadata(self) -> DatasetMetadata: ...

    def recordings(self) -> Sequence[Recording]: ...


@dataclass(frozen=True)
class SignalInfo:
    """Channel layout and extent of one openable recording.

    ``sample_rate_hz`` may be ``None`` when the upstream source does not state
    a rate; consumers that need physical frequencies (spectral features) must
    fail closed on ``None`` instead of assuming a value.
    """

    channels: tuple[str, ...]
    total_samples: int
    sample_rate_hz: float | None = None
    units: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must contain unique non-empty names")
        for channel in self.channels:
            _require_text(channel, "channel name")
        if self.sample_rate_hz is not None and (
            not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0
        ):
            raise ValueError("sample_rate_hz must be finite and positive when stated")
        if self.total_samples <= 0:
            raise ValueError("total_samples must be positive")
        if self.units is not None:
            if len(self.units) != len(self.channels):
                raise ValueError("units must align with channels when provided")
            for unit in self.units:
                _require_text(unit, "unit")


@runtime_checkable
class SignalStream(Protocol):
    """Chunked read access to one recording without requiring a full-file load."""

    @property
    def info(self) -> SignalInfo: ...

    def read(self, start: int, count: int) -> NDArray[np.float64]: ...


@runtime_checkable
class SignalDatasetAdapter(DatasetAdapter, Protocol):
    """Dataset adapter that can also serve raw signal payloads.

    Metadata-only adapters remain valid for audits and inspection. Kernel stages
    that need signal values (window extraction) require this capability and fail
    closed with a "capability missing" error instead of silently degrading.
    """

    def open(self, recording_id: str) -> SignalStream: ...


@dataclass(frozen=True)
class WindowSpec:
    """Window extraction and sample-local preprocessing contract."""

    length: int
    stride: int
    normalization: str

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("window length must be positive")
        if self.stride < self.length:
            raise ValueError("window stride must prevent overlap")
        _require_text(self.normalization, "normalization")

    def to_dict(self) -> dict[str, object]:
        return {
            "length": self.length,
            "stride": self.stride,
            "normalization": self.normalization,
        }


@dataclass(frozen=True)
class TaskSpec:
    """Dataset-independent definition of a condition-held-out classification task.

    Two partition modes exist. Single-field tasks (the v1 form) hold out values
    of ``domain_field`` via ``partitions``. Combined-condition tasks set
    ``condition_fields`` (two or more domain fields) and assign partitions over
    tuples of their values via ``condition_partitions``; in that mode
    ``domain_field`` must equal ``condition_fields[0]`` (the primary shift
    axis) and ``partitions`` must stay empty so no silent precedence exists.
    """

    task_id: str
    dataset_id: str
    domain_field: str
    evaluation_group_field: str
    partitions: Mapping[str, Sequence[DomainValue]]
    labels: tuple[str, ...]
    window: WindowSpec
    seeds: tuple[int, ...]
    description: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    condition_fields: tuple[str, ...] = ()
    condition_partitions: Mapping[str, Sequence[Sequence[DomainValue]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "dataset_id",
            "domain_field",
            "evaluation_group_field",
            "description",
        ):
            _require_text(getattr(self, name), name)
        if self.domain_field == self.evaluation_group_field:
            raise ValueError("evaluation group must differ from the held-out domain field")
        if self.condition_fields:
            if self.partitions:
                raise ValueError(
                    "combined-condition tasks must leave single-field partitions empty"
                )
            if not all(isinstance(name, str) and name.strip() for name in self.condition_fields):
                raise ValueError("condition_fields must contain non-empty names")
            if len(set(self.condition_fields)) != len(self.condition_fields):
                raise ValueError("condition_fields must be unique")
            if self.domain_field != self.condition_fields[0]:
                raise ValueError(
                    "domain_field must equal condition_fields[0] in combined-condition tasks"
                )
            object.__setattr__(self, "partitions", MappingProxyType({}))
            object.__setattr__(
                self,
                "condition_partitions",
                _freeze_condition_partitions(self.condition_partitions),
            )
        else:
            object.__setattr__(self, "partitions", _freeze_partitions(self.partitions))
            object.__setattr__(self, "condition_partitions", MappingProxyType({}))
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must contain unique class names")
        for label in self.labels:
            _require_text(label, "label")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must contain unique integer values")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise TypeError("seeds must contain integers")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def held_out_fields(self) -> tuple[str, ...]:
        """Fields whose combined values define the held-out conditions."""

        return self.condition_fields or (self.domain_field,)

    def partition_for(self, value: DomainValue) -> str | None:
        """Single-field partition lookup (v1 tasks)."""

        if self.condition_fields:
            raise ValueError(
                "combined-condition tasks assign partitions over tuples; use "
                "partition_for_sample"
            )
        for name in _PARTITION_NAMES:
            if value in self.partitions[name]:
                return name
        return None

    def partition_for_sample(
        self, domains: Mapping[str, DomainValue]
    ) -> str | None:
        """Mode-agnostic partition lookup for one recording or window."""

        if self.condition_fields:
            try:
                key = tuple(domains[field] for field in self.condition_fields)
            except KeyError as error:
                raise ValueError(
                    f"sample lacks a held-out condition field {error.args[0]!r}"
                ) from error
            for name in _PARTITION_NAMES:
                if key in self.condition_partitions[name]:
                    return name
            return None
        try:
            value = domains[self.domain_field]
        except KeyError as error:
            raise ValueError(
                f"sample lacks the held-out domain field {self.domain_field!r}"
            ) from error
        for name in _PARTITION_NAMES:
            if value in self.partitions[name]:
                return name
        return None

    @property
    def window_length(self) -> int:
        """Compatibility accessor for v1 task consumers."""

        return self.window.length

    @property
    def stride(self) -> int:
        """Compatibility accessor for v1 task consumers."""

        return self.window.stride

    @property
    def normalization(self) -> str:
        """Compatibility accessor for v1 task consumers."""

        return self.window.normalization

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
            "domain_field": self.domain_field,
            "evaluation_group_field": self.evaluation_group_field,
            "partitions": {
                name: [list(combination) for combination in values]
                if self.condition_fields
                else list(values)
                for name, values in (
                    self.condition_partitions.items()
                    if self.condition_fields
                    else self.partitions.items()
                )
            },
            "labels": list(self.labels),
            "window": self.window.to_dict(),
            "seeds": list(self.seeds),
            "description": self.description,
            "metadata": dict(self.metadata),
        }
        if self.condition_fields:
            payload["condition_fields"] = list(self.condition_fields)
        return payload
