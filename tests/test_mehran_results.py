import json
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/v1.1/mehran"
TASK_IDS = {
    "mehran-load-100-to-300-v1",
    "mehran-load-300-to-100-v1",
}
MODEL_IDS = {
    "signal_features_logreg",
    "standard_cnn_1d",
    "compact_depthwise_cnn_1d",
    "compact_coral_cnn_1d",
}
DEFECT_SIZES = {"0.7", "0.9", "1.1", "1.3", "1.5", "1.7"}


def _payloads() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RESULT_DIR.glob("*.json"))
    ]


def test_mehran_result_matrix_is_complete_and_unique() -> None:
    payloads = _payloads()
    combinations = {(item["task_id"], item["model_id"]) for item in payloads}

    assert len(payloads) == 8
    assert combinations == {
        (task_id, model_id) for task_id in TASK_IDS for model_id in MODEL_IDS
    }


@pytest.mark.parametrize("payload", _payloads(), ids=lambda item: item["model_id"])
def test_mehran_result_retains_frozen_protocol_and_all_seeds(payload: dict) -> None:
    assert payload["benchmark_id"] == "edgefault-bench-v1.1"
    assert re.fullmatch(r"[0-9a-f]{40}", payload["git_commit"])
    assert payload["protocol"]["test_used_for_selection"] is False
    assert payload["protocol"]["input_channels"] == 3
    assert payload["partition_sizes"].keys() == {"train", "validation", "test"}
    assert all(count > 0 for count in payload["partition_sizes"].values())
    assert [run["seed"] for run in payload["runs"]] == [17, 29, 43]

    for run in payload["runs"]:
        for partition in ("validation", "test"):
            metrics = run[partition]
            assert math.isfinite(metrics["macro_f1"])
            assert math.isfinite(metrics["balanced_accuracy"])
            assert set(metrics["per_condition_macro_f1"]) == DEFECT_SIZES
            assert metrics["worst_condition_macro_f1"] == min(
                metrics["per_condition_macro_f1"].values()
            )

    if payload["model_id"] == "signal_features_logreg":
        assert len(payload["protocol"]["feature_names"]) == 30
    else:
        assert payload["protocol"]["labels"] == ["inner_race", "outer_race"]
        assert all(run["history"] for run in payload["runs"])
