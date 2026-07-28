import json
from pathlib import Path

import pytest
import torch

from edgefault_bench.demo import (
    INPUT_LENGTH,
    _state_to_json,
    infer,
    normalize_window,
)
from edgefault_bench.models import build_model
from edgefault_bench.torch_data import LABELS


def test_normalize_window_enforces_contract():
    normalized = normalize_window(torch.arange(INPUT_LENGTH, dtype=torch.float32))

    assert normalized.shape == (1, 1, INPUT_LENGTH)
    assert float(normalized.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(normalized.std(correction=0)) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="4096"):
        normalize_window(torch.zeros(12))
    invalid = torch.zeros(INPUT_LENGTH)
    invalid[0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        normalize_window(invalid)


def test_synthetic_inference_returns_probabilities(tmp_path: Path):
    model = build_model("compact_depthwise_cnn_1d", num_classes=4)
    asset = {
        "asset_id": "test-asset",
        "model_id": "compact_depthwise_cnn_1d",
        "input_length": INPUT_LENGTH,
        "labels": list(LABELS),
        "state_dict": _state_to_json(model.state_dict()),
    }
    path = tmp_path / "asset.json"
    path.write_text(json.dumps(asset), encoding="utf-8")

    result = infer(asset_path=path, input_path=None, warmup=1, repeats=2)

    assert result["input_source"] == "synthetic_demo"
    assert result["input_is_benchmark_evidence"] is False
    assert set(result["probabilities"]) == set(LABELS)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
