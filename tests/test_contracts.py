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
