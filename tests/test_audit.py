import json
from pathlib import Path

import pytest

from edgefault_bench.audit import audit_recordings, main
from edgefault_bench.contracts import DatasetAdapter, Recording, TaskSpec
from edgefault_bench.datasets import HustV3Adapter
from edgefault_bench.tasks import load_task_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "task_name,expected_counts",
    [
        ("hust_load_0_to_400_v1.json", {"train": 20, "validation": 20, "test": 20}),
        (
            "hust_device_6204_6206_to_6208_v1.json",
            {"train": 36, "validation": 12, "test": 12},
        ),
    ],
)
def test_hust_adapter_passes_generic_recording_audit(
    task_name: str, expected_counts: dict[str, int]
) -> None:
    adapter = HustV3Adapter(ROOT / "registry/hust_v3.json")
    task = load_task_spec(ROOT / "registry/tasks" / task_name)
    report = audit_recordings(adapter.recordings(), task)

    assert isinstance(adapter, DatasetAdapter)
    assert adapter.metadata.domain_fields == ("bearing_type", "load_w")
    assert report.passed
    assert report.partition_counts == expected_counts
    assert json.loads(json.dumps(report.to_dict()))["passed"] is True


def test_audit_returns_actionable_failures_without_discarding_report() -> None:
    task = load_task_spec(ROOT / "registry/tasks/hust_load_0_to_400_v1.json")
    shared = {
        "dataset_id": task.dataset_id,
        "recording_id": "duplicated-recording",
        "source_file": "recording.mat",
        "label": "N",
        "sample_rate_hz": 51_200.0,
        "sample_count": 4096,
    }
    records = (
        Recording(**shared, domains={"load_w": 0, "bearing_type": "6204"}),
        Recording(**shared, domains={"load_w": 400, "bearing_type": "6204"}),
        Recording(
            dataset_id=task.dataset_id,
            recording_id="validation-recording",
            source_file="validation.mat",
            label="N",
            domains={"load_w": 200, "bearing_type": "6205"},
        ),
    )
    report = audit_recordings(records, task)
    failures = {check.check_id for check in report.checks if not check.passed}

    assert not report.passed
    assert {"recording_identity", "recording_exclusivity"} <= failures
    with pytest.raises(ValueError, match="leakage audit failed"):
        report.require_pass()


def test_generic_task_loader_does_not_apply_hust_seed_freeze(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "task_id": "example-device-holdout-v1",
        "dataset_id": "example-v1",
        "domain_field": "device",
        "evaluation_group_field": "load",
        "partitions": {"train": ["a"], "validation": ["b"], "test": ["c"]},
        "labels": ["healthy", "fault"],
        "window": {"length": 1024, "stride": 1024, "normalization": "none"},
        "seeds": [1, 2, 3, 4, 5],
        "description": "A dataset-independent task.",
    }
    path = tmp_path / "task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    task = load_task_spec(path)

    assert isinstance(task, TaskSpec)
    assert task.seeds == (1, 2, 3, 4, 5)


def test_audit_cli_writes_machine_readable_report(tmp_path: Path, capsys) -> None:
    output = tmp_path / "audit.json"
    exit_code = main(
        [
            "--task",
            str(ROOT / "registry/tasks/hust_load_0_to_400_v1.json"),
            "--dataset-manifest",
            str(ROOT / "registry/hust_v3.json"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    assert json.loads(capsys.readouterr().out)["partition_counts"] == {
        "test": 20,
        "train": 20,
        "validation": 20,
    }
