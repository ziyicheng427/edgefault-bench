from pathlib import Path

import numpy as np
import pytest

from edgefault_bench.audit import audit_recordings, main
from edgefault_bench.contracts import DatasetAdapter
from edgefault_bench.datasets import (
    MehranV2Adapter,
    build_mehran_window_records,
    load_mehran_manifest,
    load_mehran_signal,
    parse_mehran_filename,
    preprocess_mehran_windows,
    verify_mehran_file,
)
from edgefault_bench.tasks import load_task_spec

ROOT = Path(__file__).resolve().parents[1]


def test_mehran_registry_retains_complete_selected_grid() -> None:
    payload, files = load_mehran_manifest(ROOT / "registry/mehran_v2.json")

    assert payload["license"]["spdx"] == "CC-BY-4.0"
    assert payload["protocol_status"] == "accepted"
    assert payload["selection"] == {
        "selected_count": 36,
        "excluded_count": 2,
        "reason": (
            "Select the complete inner/outer, six-defect-size, three-load grid; exclude "
            "healthy files whose source metadata do not assign a load."
        ),
    }
    assert len(files) == 36
    assert {item.label for item in files} == {"inner_race", "outer_race"}
    assert {item.load_w for item in files} == {100, 200, 300}
    assert {item.defect_size_mm for item in files} == {0.7, 0.9, 1.1, 1.3, 1.5, 1.7}


def test_mehran_filename_parser_handles_source_suffixes() -> None:
    assert parse_mehran_filename("0.7inner-100watt-67V2Iv.csv") == (
        "inner_race",
        0.7,
        100,
    )
    assert parse_mehran_filename("1.7outer-300watt.csv") == ("outer_race", 1.7, 300)
    with pytest.raises(ValueError, match="invalid selected"):
        parse_mehran_filename("Healthy with pulley.csv")


@pytest.mark.parametrize(
    "task_name",
    ["mehran_load_100_to_300_v1.json", "mehran_load_300_to_100_v1.json"],
)
def test_frozen_tasks_pass_generic_recording_audit(task_name: str) -> None:
    adapter = MehranV2Adapter(ROOT / "registry/mehran_v2.json")
    task = load_task_spec(ROOT / "registry/tasks" / task_name)
    report = audit_recordings(adapter.recordings(), task)

    assert isinstance(adapter, DatasetAdapter)
    assert task.metadata["protocol_status"] == "frozen"
    assert report.passed
    assert report.partition_counts == {"train": 12, "validation": 12, "test": 12}


def test_mehran_csv_loader_reads_triaxial_values(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text(
        "Time Stamp, X-axis, Y-axis, Z-axis\n"
        "13:34:14,0.1,0.2,0.3\n"
        "13:34:14,0.4,0.5,0.6\n",
        encoding="utf-8",
    )

    signal = load_mehran_signal(path, minimum_samples=2)

    assert signal.shape == (2, 3)
    assert signal.dtype == np.float32
    assert signal[1].tolist() == pytest.approx([0.4, 0.5, 0.6])


def test_mehran_windows_are_non_overlapping_and_channel_normalized() -> None:
    _, files = load_mehran_manifest(ROOT / "registry/mehran_v2.json")
    selected = (files[0],)
    sample_counts = {files[0].filename: 10}
    records = build_mehran_window_records(
        selected, sample_counts, window_length=4, stride=4
    )
    signal = np.column_stack(
        (
            np.arange(10, dtype=np.float32),
            np.arange(10, dtype=np.float32) * 2,
            np.arange(10, dtype=np.float32) * -3,
        )
    )
    windows = preprocess_mehran_windows(signal, window_length=4, stride=4)

    assert len(records) == 2
    assert records[0].stop <= records[1].start
    assert windows.shape == (2, 3, 4)
    assert windows.dtype == np.float32
    assert windows.mean(axis=2) == pytest.approx(np.zeros((2, 3)), abs=1e-6)
    assert windows.std(axis=2) == pytest.approx(np.ones((2, 3)), abs=1e-6)


def test_mehran_window_records_require_exact_sample_count_keys() -> None:
    _, files = load_mehran_manifest(ROOT / "registry/mehran_v2.json")
    with pytest.raises(ValueError, match="sample-count keys"):
        build_mehran_window_records((files[0],), {})


def test_mehran_file_verification_fails_closed(tmp_path: Path) -> None:
    _, files = load_mehran_manifest(ROOT / "registry/mehran_v2.json")
    (tmp_path / files[0].filename).write_bytes(b"not the registered source file")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_mehran_file(tmp_path / files[0].filename, files[0])


def test_audit_cli_dispatches_mehran_adapter(capsys) -> None:
    exit_code = main(
        [
            "--task",
            str(ROOT / "registry/tasks/mehran_load_100_to_300_v1.json"),
            "--dataset-manifest",
            str(ROOT / "registry/mehran_v2.json"),
        ]
    )

    assert exit_code == 0
    assert '"record_count": 36' in capsys.readouterr().out
