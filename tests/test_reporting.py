import json
from pathlib import Path

import pytest

from edgefault_bench.reporting import load_result, summarize


def _payload(seeds=(17, 29, 43)):
    return {
        "task_id": "task-v1",
        "model_id": "model-v1",
        "git_commit": "abcdef1234567890",
        "environment": {"platform": "test-platform", "python": "3.12.0"},
        "runs": [
            {
                "seed": seed,
                "validation": {"macro_f1": 0.7},
                "test": {"macro_f1": 0.8, "worst_condition_macro_f1": 0.6},
                "complexity": {"trainable_parameters": 12, "serialized_size_bytes": 34},
                "latency": {"median_ms": 0.5},
            }
            for seed in seeds
        ],
    }


def test_summary_contains_provenance_and_metrics(tmp_path: Path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    text = summarize([path])

    assert "task-v1 | model-v1" in text
    assert "0.8000 ± 0.0000" in text
    assert "abcdef1" in text
    assert "test-platform / Python 3.12.0" in text


def test_load_result_rejects_incomplete_seed_set(tmp_path: Path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_payload(seeds=(17, 29))), encoding="utf-8")

    with pytest.raises(ValueError, match="expected seeds"):
        load_result(path)
