import json
from pathlib import Path

import pytest

from edgefault_bench.contracts import TaskSpec, WindowSpec
from edgefault_bench.prepared import load_prepared_feature_table, prepare_task_data


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="example-load-v1",
        dataset_id="example-v1",
        domain_field="load",
        evaluation_group_field="device",
        partitions={"train": [0], "validation": [100], "test": [200]},
        labels=("healthy", "fault"),
        window=WindowSpec(length=4, stride=4, normalization="none"),
        seeds=(17, 29, 43),
        description="Synthetic contract test.",
    )


def _write_table(path: Path, samples: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "example-v1",
                "feature_names": ["rms", "kurtosis"],
                "preparation": {"kind": "synthetic-test"},
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )


def _sample(recording_id: str, label: str, load: int, device: str, values=(1.0, 2.0)):
    return {
        "recording_id": recording_id,
        "label": label,
        "domains": {"load": load, "device": device},
        "features": list(values),
    }


def test_loads_and_partitions_prepared_table_by_frozen_task(tmp_path: Path) -> None:
    path = tmp_path / "features.json"
    _write_table(
        path,
        [
            _sample("train-h", "healthy", 0, "a"),
            _sample("train-f", "fault", 0, "b"),
            _sample("validation-h", "healthy", 100, "a"),
            _sample("validation-f", "fault", 100, "b"),
            _sample("test-h", "healthy", 200, "a"),
            _sample("test-f", "fault", 200, "b"),
        ],
    )

    table = load_prepared_feature_table(path)
    prepared = prepare_task_data(table, _task())

    assert prepared.features.shape == (6, 2)
    assert prepared.train.tolist() == [0, 1]
    assert prepared.validation.tolist() == [2, 3]
    assert prepared.test.tolist() == [4, 5]
    assert prepared.evaluation_groups.tolist() == ["a", "b", "a", "b", "a", "b"]


def test_rejects_nonfinite_or_wrong_width_features(tmp_path: Path) -> None:
    path = tmp_path / "features.json"
    _write_table(path, [_sample("r1", "healthy", 0, "a", values=(float("nan"), 2.0))])
    with pytest.raises(ValueError, match="finite"):
        load_prepared_feature_table(path)

    _write_table(path, [_sample("r1", "healthy", 0, "a", values=(1.0,))])
    with pytest.raises(ValueError, match="feature width"):
        load_prepared_feature_table(path)


def test_rejects_unassigned_domain_and_inconsistent_recording(tmp_path: Path) -> None:
    path = tmp_path / "features.json"
    _write_table(path, [_sample("r1", "healthy", 999, "a")])
    with pytest.raises(ValueError, match="unassigned"):
        prepare_task_data(load_prepared_feature_table(path), _task())

    _write_table(
        path,
        [
            _sample("same-recording", "healthy", 0, "a"),
            _sample("same-recording", "fault", 0, "a"),
            _sample("validation", "healthy", 100, "a"),
            _sample("test", "healthy", 200, "a"),
        ],
    )
    with pytest.raises(ValueError, match="inconsistent"):
        prepare_task_data(load_prepared_feature_table(path), _task())


def test_rejects_empty_partition(tmp_path: Path) -> None:
    path = tmp_path / "features.json"
    _write_table(
        path,
        [
            _sample("train", "healthy", 0, "a"),
            _sample("validation", "healthy", 100, "a"),
        ],
    )

    with pytest.raises(ValueError, match="partitions must be non-empty"):
        prepare_task_data(load_prepared_feature_table(path), _task())
