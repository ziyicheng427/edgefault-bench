import json
from pathlib import Path

import numpy as np
import pytest

import edgefault_bench.torch_data as torch_data
from edgefault_bench.datasets.mehran import MehranFile


def _mehran_file(filename: str, label: str, load_w: int) -> MehranFile:
    return MehranFile(
        filename=filename,
        file_id=filename,
        size_bytes=1,
        sha256="0" * 64,
        download_url="https://example.invalid/file",
        label=label,
        defect_size_mm=0.7,
        load_w=load_w,
    )


def test_mehran_tensor_dispatch_preserves_channels_labels_and_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"dataset_id": "mehran-triaxial-bearing-v2"}), encoding="utf-8"
    )
    files = (
        _mehran_file("inner.csv", "inner_race", 100),
        _mehran_file("outer.csv", "outer_race", 300),
    )
    monkeypatch.setattr(
        torch_data, "load_mehran_manifest", lambda _: ({"dataset_id": "test"}, files)
    )
    monkeypatch.setattr(torch_data, "verify_mehran_file", lambda *_: None)

    def fake_signal(path: Path, *, minimum_samples: int) -> np.ndarray:
        offset = 0.0 if path.name == "inner.csv" else 10.0
        base = np.arange(8, dtype=np.float32) + offset
        assert minimum_samples == 4
        return np.column_stack((base, base * 2, base * -3))

    monkeypatch.setattr(torch_data, "load_mehran_signal", fake_signal)
    table = torch_data.load_tensor_table(
        manifest,
        tmp_path,
        window_length=4,
        stride=4,
        normalization="per_window_channel_zscore",
    )

    assert table.signals.shape == (4, 3, 4)
    assert table.labels == ("inner_race", "outer_race")
    assert table.targets.tolist() == [0, 0, 1, 1]
    assert [record.recording for record in table.records] == [
        "inner.csv",
        "inner.csv",
        "outer.csv",
        "outer.csv",
    ]
    assert table.signals.mean(dim=2).numpy() == pytest.approx(np.zeros((4, 3)), abs=1e-6)


def test_tensor_dispatch_rejects_unknown_dataset(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset_id": "unknown"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported tensor dataset"):
        torch_data.load_tensor_table(manifest, tmp_path)
