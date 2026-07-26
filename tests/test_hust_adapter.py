import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from edgefault_bench.datasets.hust import (
    build_window_records,
    load_hust_manifest,
    load_hust_signal,
    parse_hust_filename,
    preprocess_window,
)
from edgefault_bench.tasks import load_task_manifest, split_window_records

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_hust_registry_is_complete() -> None:
    payload, files = load_hust_manifest(ROOT / "registry/hust_v3.json")
    assert payload["license"]["spdx"] == "CC-BY-4.0"
    assert len(files) == 60
    assert {item.label for item in files} == {"N", "I", "O", "IO"}
    assert {item.bearing_type for item in files} == {"6204", "6205", "6206", "6207", "6208"}
    assert {item.load_w for item in files} == {0, 200, 400}


def test_filename_parser_uses_paper_convention() -> None:
    assert parse_hust_filename("I402.mat") == ("I", "6204", 200)
    assert parse_hust_filename("IO804.mat") == ("IO", "6208", 400)
    with pytest.raises(ValueError, match="invalid HUST filename"):
        parse_hust_filename("I420.mat")


def test_window_records_are_non_overlapping_and_complete() -> None:
    _, files = load_hust_manifest(ROOT / "registry/hust_v3.json")
    records = build_window_records(files)
    assert len(records) == 7_500
    first_recording = [record for record in records if record.filename == records[0].filename]
    assert len(first_recording) == 125
    neighbours = zip(first_recording, first_recording[1:], strict=False)
    assert all(left.stop <= right.start for left, right in neighbours)


@pytest.mark.parametrize(
    "task_name,expected_sizes",
    [
        ("hust_load_0_to_400_v1.json", (2_500, 2_500, 2_500)),
        ("hust_load_400_to_0_v1.json", (2_500, 2_500, 2_500)),
        ("hust_device_6204_6206_to_6208_v1.json", (4_500, 1_500, 1_500)),
    ],
)
def test_frozen_tasks_have_expected_partition_sizes(
    task_name: str, expected_sizes: tuple[int, int, int]
) -> None:
    _, files = load_hust_manifest(ROOT / "registry/hust_v3.json")
    records = build_window_records(files)
    task = load_task_manifest(ROOT / "registry/tasks" / task_name)
    selected, split = split_window_records(records, task)
    assert len(selected) == 7_500
    assert (len(split.train), len(split.validation), len(split.test)) == expected_sizes


def test_mat_loader_and_per_window_preprocessing(tmp_path: Path) -> None:
    path = tmp_path / "N400.mat"
    raw = np.linspace(-2.0, 3.0, 512_000, dtype=np.float64)
    savemat(path, {"data": raw[:, None], "fs": np.array([[24.93]])})
    signal, shaft_frequency = load_hust_signal(path)
    processed = preprocess_window(signal[:4096])
    assert shaft_frequency == pytest.approx(24.93)
    assert processed.dtype == np.float32
    assert float(processed.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(processed.std()) == pytest.approx(1.0, abs=1e-6)


def test_mat_loader_drops_documented_io_extra_sample(tmp_path: Path) -> None:
    path = tmp_path / "IO400.mat"
    raw = np.arange(512_001, dtype=np.float64)
    savemat(path, {"data": raw[:, None], "fs": np.array([[24.0]])})
    signal, _ = load_hust_signal(path)
    assert signal.shape == (512_000,)
    assert signal[-1] == 511_999


def test_task_loader_rejects_seed_changes(tmp_path: Path) -> None:
    source = ROOT / "registry/tasks/hust_load_0_to_400_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["seeds"] = [1, 2, 3]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        load_task_manifest(changed)
