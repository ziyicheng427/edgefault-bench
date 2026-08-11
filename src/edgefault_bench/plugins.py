"""Discovery and validation for dataset plugins.

Third-party packages register a callable under the ``edgefault_bench.datasets``
entry-point group. The entry-point name is the dataset identifier and the
callable accepts a manifest path and returns a :class:`DatasetAdapter`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any

from edgefault_bench.contracts import DatasetAdapter, Recording

DATASET_ENTRY_POINT_GROUP = "edgefault_bench.datasets"
MODEL_ENTRY_POINT_GROUP = "edgefault_bench.models"
DatasetAdapterFactory = Callable[[Path], DatasetAdapter]
ModelFactory = Callable[["ModelBuildContext"], object]


@dataclass(frozen=True)
class DatasetPlugin:
    """A discoverable dataset adapter factory and its provenance."""

    dataset_id: str
    source: str
    factory: DatasetAdapterFactory

    def create(self, manifest: Path) -> DatasetAdapter:
        adapter = self.factory(manifest)
        validate_dataset_adapter(adapter, expected_dataset_id=self.dataset_id)
        return adapter


@dataclass(frozen=True)
class ModelBuildContext:
    """Runtime-independent dimensions and seed supplied to a model factory."""

    num_classes: int
    input_channels: int
    seed: int

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


@dataclass(frozen=True)
class ModelPlugin:
    """A model factory with an explicit execution backend and provenance."""

    model_id: str
    backend: str
    source: str
    factory: ModelFactory

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if self.backend not in {"pytorch", "sklearn"}:
            raise ValueError("model plugin backend must be 'pytorch' or 'sklearn'")
        if not self.source.strip():
            raise ValueError("model plugin source must be non-empty")
        if not callable(self.factory):
            raise TypeError("model plugin factory must be callable")

    def create(self, context: ModelBuildContext) -> object:
        model = self.factory(context)
        validate_model_implementation(model, backend=self.backend)
        return model


def _builtin_plugins() -> tuple[DatasetPlugin, ...]:
    # Imports stay local so importing the plugin registry remains lightweight.
    from edgefault_bench.datasets import HustV3Adapter, MehranV2Adapter

    return (
        DatasetPlugin("hust-bearing-v3", "edgefault-bench", HustV3Adapter),
        DatasetPlugin("mehran-triaxial-bearing-v2", "edgefault-bench", MehranV2Adapter),
    )


def _installed_entry_points() -> Sequence[Any]:
    return tuple(metadata.entry_points(group=DATASET_ENTRY_POINT_GROUP))


def _build_feature_baseline(context: ModelBuildContext) -> object:
    from edgefault_bench.baseline import make_feature_baseline

    return make_feature_baseline(seed=context.seed)


def _neural_factory(model_id: str) -> ModelFactory:
    def build(context: ModelBuildContext) -> object:
        from edgefault_bench.models import build_model

        return build_model(
            model_id,
            num_classes=context.num_classes,
            in_channels=context.input_channels,
        )

    return build


def _builtin_model_plugins() -> tuple[ModelPlugin, ...]:
    return (
        ModelPlugin(
            "signal_features_logreg", "sklearn", "edgefault-bench", _build_feature_baseline
        ),
        ModelPlugin(
            "standard_cnn_1d", "pytorch", "edgefault-bench", _neural_factory("standard_cnn_1d")
        ),
        ModelPlugin(
            "compact_depthwise_cnn_1d",
            "pytorch",
            "edgefault-bench",
            _neural_factory("compact_depthwise_cnn_1d"),
        ),
        ModelPlugin(
            "compact_coral_cnn_1d",
            "pytorch",
            "edgefault-bench",
            _neural_factory("compact_coral_cnn_1d"),
        ),
    )


def discover_dataset_plugins(
    entry_points: Iterable[Any] | None = None,
) -> dict[str, DatasetPlugin]:
    """Return built-in and installed plugins, rejecting identifier collisions."""

    plugins = {plugin.dataset_id: plugin for plugin in _builtin_plugins()}
    candidates = _installed_entry_points() if entry_points is None else tuple(entry_points)
    for entry_point in candidates:
        dataset_id = entry_point.name
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("dataset plugin entry-point names must be non-empty")
        if dataset_id in plugins:
            raise ValueError(f"duplicate dataset plugin for {dataset_id!r}")
        distribution = getattr(entry_point, "dist", None)
        source = getattr(distribution, "name", None) or getattr(entry_point, "module", "unknown")
        factory = entry_point.load()
        if not callable(factory):
            raise TypeError(f"dataset plugin {dataset_id!r} must expose a callable factory")
        plugins[dataset_id] = DatasetPlugin(dataset_id, source, factory)
    return plugins


def discover_model_plugins(
    entry_points: Iterable[Any] | None = None,
) -> dict[str, ModelPlugin]:
    """Return built-in and installed model plugins, rejecting collisions."""

    plugins = {plugin.model_id: plugin for plugin in _builtin_model_plugins()}
    candidates = (
        tuple(metadata.entry_points(group=MODEL_ENTRY_POINT_GROUP))
        if entry_points is None
        else tuple(entry_points)
    )
    for entry_point in candidates:
        model_id = entry_point.name
        if model_id in plugins:
            raise ValueError(f"duplicate model plugin for {model_id!r}")
        plugin = entry_point.load()
        if not isinstance(plugin, ModelPlugin):
            raise TypeError(f"model plugin {model_id!r} must expose a ModelPlugin value")
        if plugin.model_id != model_id:
            raise ValueError(
                f"model plugin identity mismatch: registered as {model_id!r}, "
                f"declared {plugin.model_id!r}"
            )
        distribution = getattr(entry_point, "dist", None)
        source = getattr(distribution, "name", None) or getattr(entry_point, "module", "unknown")
        plugins[model_id] = replace(plugin, source=source)
    return plugins


def dataset_id_from_manifest(manifest: Path) -> str:
    """Read the plugin routing key from a dataset manifest."""

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("dataset_id"), str):
        raise ValueError(f"dataset manifest has no dataset_id: {manifest}")
    dataset_id = payload["dataset_id"].strip()
    if not dataset_id:
        raise ValueError(f"dataset manifest has no dataset_id: {manifest}")
    return dataset_id


def load_dataset_adapter(
    manifest: Path,
    *,
    entry_points: Iterable[Any] | None = None,
) -> DatasetAdapter:
    """Create the adapter selected by a manifest, or fail closed if unavailable."""

    dataset_id = dataset_id_from_manifest(manifest)
    plugins = discover_dataset_plugins(entry_points)
    try:
        plugin = plugins[dataset_id]
    except KeyError as error:
        raise ValueError(f"no dataset plugin is registered for {dataset_id!r}") from error
    return plugin.create(manifest)


def validate_dataset_adapter(
    adapter: object,
    *,
    expected_dataset_id: str | None = None,
) -> None:
    """Validate the public adapter boundary without reading signal payloads."""

    if not isinstance(adapter, DatasetAdapter):
        raise TypeError("dataset plugin did not return a DatasetAdapter")
    metadata_value = adapter.metadata
    if expected_dataset_id is not None and metadata_value.dataset_id != expected_dataset_id:
        raise ValueError(
            "dataset plugin identity mismatch: "
            f"registered as {expected_dataset_id!r}, returned {metadata_value.dataset_id!r}"
        )
    recordings = adapter.recordings()
    if not isinstance(recordings, Sequence):
        raise TypeError("DatasetAdapter.recordings() must return a sequence")
    recording_ids: set[str] = set()
    for recording in recordings:
        if not isinstance(recording, Recording):
            raise TypeError("DatasetAdapter.recordings() must contain Recording values")
        if recording.dataset_id != metadata_value.dataset_id:
            raise ValueError(
                f"recording {recording.recording_id!r} belongs to {recording.dataset_id!r}, "
                f"not {metadata_value.dataset_id!r}"
            )
        missing_domains = set(metadata_value.domain_fields) - set(recording.domains)
        if missing_domains:
            raise ValueError(
                f"recording {recording.recording_id!r} is missing domains: "
                f"{sorted(missing_domains)}"
            )
        if recording.recording_id in recording_ids:
            raise ValueError(f"duplicate recording_id {recording.recording_id!r}")
        recording_ids.add(recording.recording_id)


def validate_model_implementation(model: object, *, backend: str) -> None:
    """Check the minimum executable boundary without fitting or training a model."""

    methods = {
        "sklearn": ("fit", "predict", "get_params"),
        "pytorch": ("train", "eval", "state_dict", "parameters"),
    }
    try:
        required = methods[backend]
    except KeyError as error:
        raise ValueError(f"unsupported model backend {backend!r}") from error
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(f"{backend} model is missing required methods: {missing}")
