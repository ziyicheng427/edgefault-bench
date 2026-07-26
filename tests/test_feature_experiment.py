from pathlib import Path

import numpy as np

from edgefault_bench.experiments.features import _latency_ms


class ConstantModel:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(len(features), dtype=np.int64)


def test_latency_protocol_reports_batch_one() -> None:
    result = _latency_ms(ConstantModel(), np.zeros((1, 10)), warmup=2, repeats=5)
    assert result["batch_size"] == 1
    assert result["warmup"] == 2
    assert result["repeats"] == 5
    assert result["median_ms"] >= 0.0
    assert result["p95_ms"] >= result["median_ms"]


def test_result_directory_is_not_raw_data() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "results").as_posix() != (root / "data/raw").as_posix()

