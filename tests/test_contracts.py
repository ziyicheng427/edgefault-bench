from dataclasses import FrozenInstanceError

import pytest

from edgefault_bench.contracts import (
    DatasetAdapter,
    DatasetMetadata,
    Recording,
    TaskSpec,
    WindowSpec,
)


class ExampleAdapter:
    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            dataset_id="example-v1",
            version="1",
            title="Example sensor dataset",
            license_spdx="CC-BY-4.0",
            source_url="https://example.org/dataset",
            domain_fields=("device", "load"),
        )

    def recordings(self) -> tuple[Recording, ...]:
        return (
            Recording(
                dataset_id="example-v1",
                recording_id="device-a-load-0",
                source_file="a.csv",
                label="healthy",
                domains={"device": "a", "load": 0},
                sample_rate_hz=1000.0,
                sample_count=4096,
            ),
        )


def test_dataset_adapter_contract_is_runtime_checkable() -> None:
    adapter = ExampleAdapter()
    assert isinstance(adapter, DatasetAdapter)
    assert adapter.metadata.dataset_id == "example-v1"
    assert adapter.recordings()[0].domain_value("load") == 0


def test_recording_metadata_is_immutable_and_serializable() -> None:
    record = ExampleAdapter().recordings()[0]
    with pytest.raises(TypeError):
        record.domains["load"] = 200  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        record.label = "fault"  # type: ignore[misc]
    assert record.to_dict()["domains"] == {"device": "a", "load": 0}
    with pytest.raises(ValueError, match="no domain field"):
        record.domain_value("speed")


def test_recording_rejects_invalid_provenance() -> None:
    with pytest.raises(ValueError, match="sha256"):
        Recording(
            dataset_id="example-v1",
            recording_id="r1",
            source_file="r1.csv",
            label="healthy",
            domains={"device": "a"},
            sha256="not-a-hash",
        )
    with pytest.raises(ValueError, match="finite"):
        Recording(
            dataset_id="example-v1",
            recording_id="r1",
            source_file="r1.csv",
            label="healthy",
            domains={"load": float("nan")},
        )


def test_task_spec_validates_and_resolves_partitions() -> None:
    task = TaskSpec(
        task_id="example-load-holdout-v1",
        dataset_id="example-v1",
        domain_field="load",
        evaluation_group_field="device",
        partitions={"train": [0], "validation": [100], "test": [200]},
        labels=("healthy", "fault"),
        window=WindowSpec(length=1024, stride=1024, normalization="per_window_zscore"),
        seeds=(17, 29, 43),
        description="Example condition-held-out task.",
    )
    assert task.partition_for(0) == "train"
    assert task.partition_for(300) is None
    assert task.to_dict()["window"] == {
        "length": 1024,
        "stride": 1024,
        "normalization": "per_window_zscore",
    }


def test_task_spec_rejects_overlapping_domains() -> None:
    with pytest.raises(ValueError, match="overlap"):
        TaskSpec(
            task_id="invalid",
            dataset_id="example-v1",
            domain_field="load",
            evaluation_group_field="device",
            partitions={"train": [0], "validation": [0], "test": [200]},
            labels=("healthy", "fault"),
            window=WindowSpec(length=1024, stride=1024, normalization="none"),
            seeds=(17,),
            description="Invalid overlap.",
        )


def test_window_spec_rejects_overlapping_windows() -> None:
    with pytest.raises(ValueError, match="prevent overlap"):
        WindowSpec(length=1024, stride=512, normalization="none")


def test_combined_condition_task_spec_modes() -> None:
    import pytest

    from edgefault_bench.contracts import TaskSpec, WindowSpec

    def combined(partitions, *, fields=("load_w", "defect_size_mm")):
        return TaskSpec(
            task_id="combined-v1",
            dataset_id="example-v1",
            domain_field=fields[0],
            evaluation_group_field="device",
            partitions={},
            labels=("a", "b"),
            window=WindowSpec(length=4, stride=4, normalization="none"),
            seeds=(17,),
            description="combined fixture",
            condition_fields=fields,
            condition_partitions=partitions,
        )

    spec = combined(
        {
            "train": [(100, 0.7), (200, 0.7)],
            "validation": [(100, 1.1)],
            "test": [(300, 1.7)],
        }
    )
    assert spec.held_out_fields == ("load_w", "defect_size_mm")
    assert spec.partition_for_sample({"load_w": 200, "defect_size_mm": 0.7}) == "train"
    assert spec.partition_for_sample({"load_w": 300, "defect_size_mm": 1.7}) == "test"
    assert spec.partition_for_sample({"load_w": 999, "defect_size_mm": 1.7}) is None
    with pytest.raises(ValueError, match="condition field"):
        spec.partition_for_sample({"load_w": 100})
    with pytest.raises(ValueError, match="tuples"):
        spec.partition_for(100)

    with pytest.raises(ValueError, match="overlap"):
        combined(
            {
                "train": [(100, 0.7)],
                "validation": [(100, 0.7)],
                "test": [(300, 1.7)],
            }
        )
    with pytest.raises(ValueError, match="single-field partitions empty"):
        combined(
            {
                "train": [(100, 0.7), (200, 0.7)],
                "validation": [(100, 1.1)],
                "test": [(300, 1.7)],
            },
        ).__class__(
            task_id="x", dataset_id="d", domain_field="load_w",
            evaluation_group_field="device",
            partitions={"train": [1], "validation": [2], "test": [3]},
            labels=("a",), window=WindowSpec(4, 4, "none"), seeds=(1,),
            description="d", condition_fields=("load_w", "size"),
            condition_partitions={
                "train": [(1, 1)], "validation": [(2, 2)], "test": [(3, 3)]
            },
        )
    with pytest.raises(ValueError, match="condition_fields\\[0\\]"):
        TaskSpec(
            task_id="mismatched",
            dataset_id="example-v1",
            domain_field="other_field",
            evaluation_group_field="device",
            partitions={},
            labels=("a",),
            window=WindowSpec(4, 4, "none"),
            seeds=(1,),
            description="d",
            condition_fields=("load_w", "size"),
            condition_partitions={
                "train": [(1, 1)], "validation": [(2, 2)], "test": [(3, 3)]
            },
        )
