"""Preprocessing contracts with enforced fit-scope separation (T05, INV-3).

A preprocessor learns its fitted state exclusively from training-side windows
(``train`` and ``validation`` partitions) and then transforms any partition
deterministically. The kernel helper :func:`fit_preprocessor` is the only
sanctioned way to reach ``fit``: it selects the training view itself, rejects
recordings that span partitions, and records the fitted-state digest for the
result bundle.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from edgefault_bench.contracts import DomainValue, TaskSpec
from edgefault_bench.features import extract_multichannel_features, multichannel_feature_names

PREPROCESSOR_ENTRY_POINT_GROUP = "edgefault_bench.preprocessors"
TRAINING_PARTITIONS: frozenset[str] = frozenset({"train", "validation"})


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class WindowSample:
    """Identity of one extracted window inside its source recording."""

    window_id: str
    recording_id: str
    start_sample: int
    end_sample: int
    label: str
    domains: Mapping[str, DomainValue]

    def __post_init__(self) -> None:
        for name in ("window_id", "recording_id", "label"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("end_sample must exceed non-negative start_sample")
        if not self.domains:
            raise ValueError("domains must not be empty")
        object.__setattr__(self, "domains", MappingProxyType(dict(self.domains)))

    def to_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "recording_id": self.recording_id,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "label": self.label,
            "domains": dict(self.domains),
        }


@dataclass(frozen=True)
class WindowBatch:
    """Windows with identity metadata and their ``(n, channels, time)`` signals.

    ``sample_rate_hz`` is ``None`` when the dataset source does not state one.
    """

    samples: tuple[WindowSample, ...]
    signals: NDArray[np.float64]
    channels: tuple[str, ...]
    sample_rate_hz: float | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.signals, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("signals must have shape (windows, channels, time)")
        if len(self.samples) != values.shape[0]:
            raise ValueError("samples and signals must describe the same windows")
        if values.shape[1] != len(self.channels) or not self.channels:
            raise ValueError("channels must align with the signal channel axis")
        if not np.isfinite(values).all():
            raise ValueError("signals contain non-finite values")
        if self.sample_rate_hz is not None and (
            not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0
        ):
            raise ValueError("sample_rate_hz must be finite and positive when stated")
        object.__setattr__(self, "signals", values)

    def __len__(self) -> int:
        return len(self.samples)

    def take(self, indices: Sequence[int]) -> WindowBatch:
        picked = np.asarray(indices, dtype=np.int64)
        if picked.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        return WindowBatch(
            samples=tuple(self.samples[index] for index in picked),
            signals=self.signals[picked],
            channels=self.channels,
            sample_rate_hz=self.sample_rate_hz,
        )


@runtime_checkable
class Preprocessor(Protocol):
    """Fit-on-training-side, transform-anywhere preprocessing contract."""

    @property
    def plugin_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def output(self) -> str:
        """``"windows"`` keeps the (n, channels, time) shape; ``"features"`` yields vectors."""

    def fit(self, batch: WindowBatch) -> None: ...

    def transform(self, batch: WindowBatch) -> NDArray[np.float64]: ...

    def fitted_state_digest(self) -> str: ...


@dataclass(frozen=True)
class PreprocessorFitRecord:
    """Provenance of one fitted preprocessing state (INV-3 evidence)."""

    plugin_id: str
    version: str
    state_digest: str
    fit_scope: tuple[str, ...]
    fitted_row_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "plugin_id": self.plugin_id,
            "version": self.version,
            "state_digest": self.state_digest,
            "fit_scope": list(self.fit_scope),
            "fitted_row_count": self.fitted_row_count,
        }


def partition_window_indices(
    batch: WindowBatch, task: TaskSpec
) -> dict[str, tuple[int, ...]]:
    """Assign every window to its task partition, failing closed on violations.

    Enforces the window-level partition invariants: every window's held-out
    domain value must map to exactly one partition, and all windows of one
    recording must inherit that recording's partition (INV-2). Unassigned or
    cross-partition recordings raise before any preprocessing can run.
    """

    indices: dict[str, list[int]] = {name: [] for name in ("train", "validation", "test")}
    recording_partitions: dict[str, str] = {}
    for index, sample in enumerate(batch.samples):
        try:
            partition = task.partition_for_sample(sample.domains)
        except ValueError as error:
            raise ValueError(f"window {sample.window_id!r}: {error}") from error
        if partition is None:
            raise ValueError(
                f"window {sample.window_id!r} has an unassigned held-out condition"
            )
        previous = recording_partitions.setdefault(sample.recording_id, partition)
        if previous != partition:
            raise ValueError(
                f"recording {sample.recording_id!r} spans partitions "
                f"{previous!r} and {partition!r}"
            )
        indices[partition].append(index)
    for name, values in indices.items():
        if not values:
            raise ValueError(f"partition {name!r} contains no windows")
    return {name: tuple(values) for name, values in indices.items()}


def training_view(batch: WindowBatch, task: TaskSpec) -> WindowBatch:
    """Return the train+validation sub-batch that alone may reach ``fit``."""

    indices = partition_window_indices(batch, task)
    selected = list(indices["train"]) + list(indices["validation"])
    return batch.take(selected)


def fit_preprocessor(
    preprocessor: Preprocessor, batch: WindowBatch, task: TaskSpec
) -> PreprocessorFitRecord:
    """Fit a preprocessor on the training side only and record its state digest."""

    if not isinstance(preprocessor, Preprocessor):
        raise TypeError("preprocessor must satisfy the Preprocessor contract")
    view = training_view(batch, task)
    preprocessor.fit(view)
    digest = preprocessor.fitted_state_digest()
    if len(digest) != 64 or not all(char in "0123456789abcdef" for char in digest):
        raise ValueError("fitted_state_digest must be a 64-character hex string")
    return PreprocessorFitRecord(
        plugin_id=preprocessor.plugin_id,
        version=preprocessor.version,
        state_digest=digest,
        fit_scope=("train", "validation"),
        fitted_row_count=len(view),
    )


class PerWindowZScore:
    """Stateless per-window, per-channel z-score (``per_window_zscore``)."""

    plugin_id = "per_window_zscore"
    version = "1"
    output = "windows"

    def __init__(self, *, epsilon: float = 1e-8) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._epsilon = epsilon

    def fit(self, batch: WindowBatch) -> None:
        """Intentionally stateless; accepting the call keeps the contract uniform."""

    def transform(self, batch: WindowBatch) -> NDArray[np.float64]:
        windows = batch.signals
        centered = windows - windows.mean(axis=2, keepdims=True)
        scale = np.maximum(centered.std(axis=2, keepdims=True), self._epsilon)
        return centered / scale

    def fitted_state_digest(self) -> str:
        return _canonical_digest({"plugin_id": self.plugin_id, "version": self.version})


class ChannelStandardization:
    """Per-channel standardization whose statistics come from ``fit`` alone."""

    plugin_id = "channel_standardization"
    version = "1"
    output = "windows"

    def __init__(self) -> None:
        self._mean: NDArray[np.float64] | None = None
        self._scale: NDArray[np.float64] | None = None

    def fit(self, batch: WindowBatch) -> None:
        values = batch.signals.reshape(batch.signals.shape[0], batch.signals.shape[1], -1)
        self._mean = values.mean(axis=(0, 2))
        self._scale = np.maximum(values.std(axis=(0, 2)), 1e-8)

    def transform(self, batch: WindowBatch) -> NDArray[np.float64]:
        if self._mean is None or self._scale is None:
            raise ValueError("transform requires a fitted state; call fit first")
        return (batch.signals - self._mean[None, :, None]) / self._scale[None, :, None]

    def fitted_state_digest(self) -> str:
        if self._mean is None or self._scale is None:
            raise ValueError("no fitted state to digest")
        return _canonical_digest(
            {
                "plugin_id": self.plugin_id,
                "version": self.version,
                "mean": [float(value) for value in self._mean],
                "scale": [float(value) for value in self._scale],
            }
        )


class SignalFeatureExtractor:
    """Deterministic 10-feature-per-channel extractor (``signal_features``)."""

    plugin_id = "signal_features"
    version = "1"
    output = "features"

    def fit(self, batch: WindowBatch) -> None:
        """Stateless; feature definitions are frozen, not learned."""

    def transform(self, batch: WindowBatch) -> NDArray[np.float64]:
        if batch.sample_rate_hz is None:
            raise ValueError(
                "signal_features requires a declared sample rate; this dataset's "
                "source does not state one, so spectral features are unavailable"
            )
        return extract_multichannel_features(
            batch.signals, sampling_rate=batch.sample_rate_hz
        )

    def feature_names(self, channels: Sequence[str]) -> tuple[str, ...]:
        return multichannel_feature_names(tuple(channels))

    def fitted_state_digest(self) -> str:
        return _canonical_digest({"plugin_id": self.plugin_id, "version": self.version})


PreprocessorFactory = Callable[[], Preprocessor]

_BUILTIN_FACTORIES: dict[str, PreprocessorFactory] = {
    PerWindowZScore.plugin_id: PerWindowZScore,
    ChannelStandardization.plugin_id: ChannelStandardization,
    SignalFeatureExtractor.plugin_id: SignalFeatureExtractor,
}


def builtin_preprocessors() -> dict[str, PreprocessorFactory]:
    return dict(_BUILTIN_FACTORIES)


def discover_preprocessor_plugins(
    entry_points: Iterable[Any] | None = None,
) -> dict[str, PreprocessorFactory]:
    """Return built-in and installed preprocessor factories, rejecting collisions."""

    plugins = dict(_BUILTIN_FACTORIES)
    candidates = (
        tuple(metadata.entry_points(group=PREPROCESSOR_ENTRY_POINT_GROUP))
        if entry_points is None
        else tuple(entry_points)
    )
    for entry_point in candidates:
        plugin_id = entry_point.name
        if plugin_id in plugins:
            raise ValueError(f"duplicate preprocessor plugin for {plugin_id!r}")
        factory = entry_point.load()
        if not callable(factory):
            raise TypeError(f"preprocessor plugin {plugin_id!r} must expose a callable factory")
        plugins[plugin_id] = factory
    return plugins


def create_preprocessor(
    plugin_id: str,
    *,
    entry_points: Iterable[Any] | None = None,
) -> Preprocessor:
    """Instantiate one preprocessor plugin and validate its contract."""

    plugins = discover_preprocessor_plugins(entry_points)
    try:
        factory = plugins[plugin_id]
    except KeyError as error:
        raise ValueError(f"no preprocessor plugin is registered for {plugin_id!r}") from error
    preprocessor = factory()
    if not isinstance(preprocessor, Preprocessor):
        raise TypeError(f"preprocessor plugin {plugin_id!r} did not return a Preprocessor")
    return preprocessor
