"""Discovery and validation for dataset plugins.

Third-party packages register a callable under the ``edgefault_bench.datasets``
entry-point group. The entry-point name is the dataset identifier and the
callable accepts a manifest path and returns a :class:`DatasetAdapter`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from edgefault_bench.contracts import DatasetAdapter, Recording

DATASET_ENTRY_POINT_GROUP = "edgefault_bench.datasets"
DatasetAdapterFactory = Callable[[Path], DatasetAdapter]


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


def _builtin_plugins() -> tuple[DatasetPlugin, ...]:
    # Imports stay local so importing the plugin registry remains lightweight.
    from edgefault_bench.datasets import HustV3Adapter, MehranV2Adapter

    return (
        DatasetPlugin("hust-bearing-v3", "edgefault-bench", HustV3Adapter),
        DatasetPlugin("mehran-triaxial-bearing-v2", "edgefault-bench", MehranV2Adapter),
    )


def _installed_entry_points() -> Sequence[Any]:
    return tuple(metadata.entry_points(group=DATASET_ENTRY_POINT_GROUP))


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
