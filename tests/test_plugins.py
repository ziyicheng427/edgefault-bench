import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from edgefault_bench.contracts import DatasetMetadata, Recording
from edgefault_bench.plugins import (
    ModelBuildContext,
    ModelPlugin,
    discover_dataset_plugins,
    discover_model_plugins,
    load_dataset_adapter,
    validate_dataset_adapter,
)


class ExampleAdapter:
    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            dataset_id="example-v1",
            version="1",
            title="Example",
            license_spdx="CC-BY-4.0",
            source_url="https://example.org",
            domain_fields=("load",),
        )

    def recordings(self) -> tuple[Recording, ...]:
        return (
            Recording(
                dataset_id="example-v1",
                recording_id="r1",
                source_file="r1.csv",
                label="healthy",
                domains={"load": 0},
            ),
        )


class FakeEntryPoint:
    name = "example-v1"
    module = "example_plugin"
    dist = SimpleNamespace(name="edgefault-example")

    def load(self):
        return lambda manifest: ExampleAdapter()


class ExampleEstimator:
    def fit(self, features, labels):
        return self

    def predict(self, features):
        return []

    def get_params(self):
        return {}


class FakeModelEntryPoint:
    name = "example_estimator"
    module = "example_models"
    dist = SimpleNamespace(name="edgefault-example-models")

    def load(self):
        return ModelPlugin(
            model_id="example_estimator",
            backend="sklearn",
            source="replaced-on-discovery",
            factory=lambda context: ExampleEstimator(),
        )


def test_discovers_third_party_entry_point() -> None:
    plugins = discover_dataset_plugins([FakeEntryPoint()])

    assert plugins["example-v1"].source == "edgefault-example"
    assert plugins["example-v1"].create(Path("unused.json")).metadata.title == "Example"


def test_loads_plugin_selected_by_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "example.json"
    manifest.write_text(json.dumps({"dataset_id": "example-v1"}), encoding="utf-8")

    adapter = load_dataset_adapter(manifest, entry_points=[FakeEntryPoint()])

    assert adapter.recordings()[0].recording_id == "r1"


def test_rejects_entry_point_collision_with_builtin() -> None:
    entry_point = FakeEntryPoint()
    entry_point.name = "hust-bearing-v3"

    with pytest.raises(ValueError, match="duplicate dataset plugin"):
        discover_dataset_plugins([entry_point])


def test_rejects_plugin_identity_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "wrong.json"
    manifest.write_text(json.dumps({"dataset_id": "different-v1"}), encoding="utf-8")
    entry_point = FakeEntryPoint()
    entry_point.name = "different-v1"

    with pytest.raises(ValueError, match="identity mismatch"):
        load_dataset_adapter(manifest, entry_points=[entry_point])


def test_rejects_recording_with_missing_domain() -> None:
    adapter = ExampleAdapter()
    adapter.recordings = lambda: (  # type: ignore[method-assign]
        Recording(
            dataset_id="example-v1",
            recording_id="r1",
            source_file="r1.csv",
            label="healthy",
            domains={"speed": 1000},
        ),
    )

    with pytest.raises(ValueError, match="missing domains"):
        validate_dataset_adapter(adapter)


def test_discovers_and_builds_third_party_model_plugin() -> None:
    plugins = discover_model_plugins([FakeModelEntryPoint()])
    plugin = plugins["example_estimator"]

    model = plugin.create(ModelBuildContext(num_classes=2, input_channels=3, seed=29))

    assert isinstance(model, ExampleEstimator)
    assert plugin.source == "edgefault-example-models"


def test_rejects_invalid_model_context_and_implementation() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ModelBuildContext(num_classes=1, input_channels=1, seed=17)
    plugin = ModelPlugin("broken", "sklearn", "example", lambda context: object())
    with pytest.raises(TypeError, match="missing required methods"):
        plugin.create(ModelBuildContext(num_classes=2, input_channels=1, seed=17))


def test_rejects_model_plugin_identity_mismatch() -> None:
    entry_point = FakeModelEntryPoint()
    entry_point.name = "different-name"

    with pytest.raises(ValueError, match="identity mismatch"):
        discover_model_plugins([entry_point])


def test_rejects_model_plugin_collision_with_builtin() -> None:
    entry_point = FakeModelEntryPoint()
    entry_point.name = "standard_cnn_1d"

    with pytest.raises(ValueError, match="duplicate model plugin"):
        discover_model_plugins([entry_point])
