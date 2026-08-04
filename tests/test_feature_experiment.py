import json
from pathlib import Path

import numpy as np
import pytest

import edgefault_bench.experiments.features as feature_experiment
from edgefault_bench.datasets.mehran import MehranFile
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


def test_mehran_feature_loader_returns_30_features_and_aligned_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"dataset_id": "mehran-triaxial-bearing-v2"}), encoding="utf-8"
    )
    specification = MehranFile(
        filename="sample.csv",
        file_id="sample",
        size_bytes=1,
        sha256="0" * 64,
        download_url="https://example.invalid/file",
        label="inner_race",
        defect_size_mm=0.7,
        load_w=100,
    )
    monkeypatch.setattr(
        feature_experiment,
        "load_mehran_manifest",
        lambda _: ({"sampling_rate_hz": 10_000}, (specification,)),
    )
    monkeypatch.setattr(feature_experiment, "verify_mehran_file", lambda *_: None)
    monkeypatch.setattr(
        feature_experiment,
        "load_mehran_signal",
        lambda *_args, **_kwargs: np.column_stack(
            (
                np.arange(8, dtype=np.float32),
                np.arange(8, dtype=np.float32) * 2,
                np.arange(8, dtype=np.float32) * -3,
            )
        ),
    )

    features, records, names = feature_experiment.load_feature_table(
        manifest, tmp_path, window_length=4, stride=4
    )

    assert features.shape == (2, 30)
    assert len(records) == 2
    assert names[0] == "x_mean"
    assert names[-1] == "z_spectral_entropy"
