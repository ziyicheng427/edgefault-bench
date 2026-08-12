import importlib.util
from pathlib import Path

import numpy as np

from edgefault_bench.plugins import ModelBuildContext, validate_dataset_adapter

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/edgefault-example-plugin"


def _load_example_module():
    module_path = EXAMPLE / "src/edgefault_example_plugin/__init__.py"
    spec = importlib.util.spec_from_file_location("edgefault_example_plugin_source", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_dataset_uses_public_adapter_boundary() -> None:
    module = _load_example_module()
    adapter = module.ExampleDatasetAdapter(EXAMPLE / "example_manifest.json")

    validate_dataset_adapter(adapter, expected_dataset_id="edgefault-example-v1")
    assert len(adapter.recordings()) == 2


def test_example_model_factory_can_fit_and_predict() -> None:
    module = _load_example_module()
    model = module.model_plugin.create(ModelBuildContext(2, 1, 29))
    features = np.asarray([[0.0], [0.2], [9.8], [10.0]])
    labels = np.asarray(["healthy", "healthy", "fault", "fault"])

    predictions = model.fit(features, labels).predict([[0.1], [9.9]])

    assert predictions.tolist() == ["healthy", "fault"]
    assert model.get_params() == {"seed": 29}
