"""Signal capability tests for adapters and the plugin boundary (T04)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from edgefault_bench.contracts import (
    DatasetMetadata,
    Recording,
    SignalInfo,
)
from edgefault_bench.datasets import resolve_raw_dir
from edgefault_bench.datasets.hust import HustV3Adapter
from edgefault_bench.datasets.mehran import MehranV2Adapter
from edgefault_bench.plugins import open_signal, validate_signal_adapter


class MetadataOnlyAdapter:
    """v1-style adapter without the signal capability."""

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            dataset_id="example-v1",
            version="1",
            title="Example",
            license_spdx="CC-BY-4.0",
            source_url="https://example.org",
            domain_fields=("load",),
        )

    def recordings(self) -> tuple[Recording, ...]:
        return (
            Recording(
                dataset_id="example-v1",
                recording_id="a-0",
                source_file="a-0.npy",
                label="healthy",
                domains={"load": 0},
                sample_rate_hz=1000.0,
                sample_count=64,
            ),
        )


class _SyntheticStream:
    def __init__(self, signal: np.ndarray, rate: float) -> None:
        self._signal = signal
        self._info = SignalInfo(
            channels=("ch0", "ch1"),
            sample_rate_hz=rate,
            total_samples=int(signal.shape[1]),
        )

    @property
    def info(self) -> SignalInfo:
        return self._info

    def read(self, start: int, count: int) -> np.ndarray:
        if count <= 0 or start < 0 or start + count > self._info.total_samples:
            raise ValueError("read out of bounds")
        return self._signal[:, start : start + count].copy()


class SyntheticSignalAdapter(MetadataOnlyAdapter):
    """Signal-capable adapter serving one deterministic in-memory recording."""

    def __init__(self, *, rate: float = 1000.0, samples: int = 64) -> None:
        self._rate = rate
        self._samples = samples
        rng = np.random.default_rng(17)
        self._signal = rng.normal(size=(2, samples))

    def recordings(self) -> tuple[Recording, ...]:
        return (
            Recording(
                dataset_id="example-v1",
                recording_id="a-0",
                source_file="a-0.npy",
                label="healthy",
                domains={"load": 0},
                sample_rate_hz=self._rate,
                sample_count=self._samples,
            ),
        )

    def open(self, recording_id: str) -> _SyntheticStream:
        if recording_id != "a-0":
            raise ValueError(f"unknown recording {recording_id!r}")
        return _SyntheticStream(self._signal, self._rate)


def test_open_signal_rejects_metadata_only_adapter() -> None:
    with pytest.raises(TypeError, match="signal capability"):
        open_signal(MetadataOnlyAdapter(), "a-0")


def test_open_signal_returns_metadata_consistent_stream() -> None:
    record, stream = open_signal(SyntheticSignalAdapter(), "a-0")
    assert record.recording_id == "a-0"
    assert stream.info.total_samples == 64
    assert stream.read(8, 16).shape == (2, 16)


def test_open_signal_rejects_unknown_recording() -> None:
    with pytest.raises(ValueError, match="unknown recording"):
        open_signal(SyntheticSignalAdapter(), "missing")


def test_validate_signal_adapter_probes_bounds_and_shape() -> None:
    validate_signal_adapter(SyntheticSignalAdapter(), probe_samples=8)

    class _BadShapeStream:
        @property
        def info(self) -> SignalInfo:
            return SignalInfo(
                channels=("ch0", "ch1"), sample_rate_hz=1000.0, total_samples=64
            )

        def read(self, start: int, count: int) -> np.ndarray:
            return np.zeros((3, count))

    class BadShapeAdapter(SyntheticSignalAdapter):
        def open(self, recording_id: str) -> _BadShapeStream:
            return _BadShapeStream()

    with pytest.raises(ValueError, match="probe"):
        validate_signal_adapter(BadShapeAdapter(), probe_samples=8)


def test_open_signal_rejects_metadata_contradicting_stream() -> None:
    class RateMismatchAdapter(SyntheticSignalAdapter):
        def open(self, recording_id: str) -> _SyntheticStream:
            stream = super().open(recording_id)
            object.__setattr__(  # type: ignore[attr-defined]
                stream._info, "sample_rate_hz", 999.0
            )
            return stream

    with pytest.raises(ValueError, match="contradicts"):
        open_signal(RateMismatchAdapter(), "a-0")


def test_signal_info_rejects_inconsistent_channel_metadata() -> None:
    with pytest.raises(ValueError, match="channels"):
        SignalInfo(channels=(), sample_rate_hz=1.0, total_samples=10)
    with pytest.raises(ValueError, match="units"):
        SignalInfo(channels=("a",), sample_rate_hz=1.0, total_samples=10, units=("m", "s"))


def test_resolve_raw_dir_honours_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_raw_dir("hust_v3") == Path("data/raw/hust_v3")
    monkeypatch.setenv("EDGEFAULT_RAW_DIR", "/tmp/elsewhere")
    assert resolve_raw_dir("hust_v3") == Path("/tmp/elsewhere/hust_v3")


HUST_LABELS = ("N", "I", "O", "B", "IO", "IB", "OB")
HUST_LOADS = ("00", "02", "04")


def _hust_filenames() -> list[str]:
    """The first 60 unique names of the full label x bearing x load grid."""

    grid = [
        f"{label}{bearing}{load}.mat"
        for label in HUST_LABELS
        for bearing in range(4, 9)
        for load in HUST_LOADS
    ]
    unique = list(dict.fromkeys(grid))
    assert len(unique) == 105
    return unique[:60]


def _write_hust_registry(root: Path, real_file: Path) -> Path:
    """A 60-entry registry whose only materialized recording is ``real_file``."""

    names = _hust_filenames()
    entries = []
    for name in names:
        if name == real_file.name:
            size, digest = real_file.stat().st_size, hashlib.sha256(
                real_file.read_bytes()
            ).hexdigest()
        else:
            size, digest = 1, "0" * 64
        entries.append(
            {
                "id": name,
                "filename": name,
                "size_bytes": size,
                "sha256": digest,
                "download_url": f"https://example.org/{name}",
            }
        )
    manifest = root / "registry" / "hust_v3.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "hust-bearing-v3",
                "version": 3,
                "title": "HUST fixture registry",
                "doi": "10.17632/cbv7jyx4p9.3",
                "landing_page": "https://data.mendeley.com/datasets/cbv7jyx4p9/3",
                "license": {"spdx": "CC-BY-4.0"},
                "sampling_rate_hz": 64_000,
                "files": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_hust_adapter_opens_registered_file_and_streams_windows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    signal = np.linspace(-1.0, 1.0, 512_000)
    mat_path = raw_dir / "N400.mat"
    savemat(mat_path, {"data": signal, "fs": np.array([12_000.0])})

    manifest = _write_hust_registry(tmp_path, mat_path)
    adapter = HustV3Adapter(manifest, raw_dir=raw_dir)
    assert len(adapter.recordings()) == 60

    stream = adapter.open("N400.mat")
    assert stream.info.total_samples == 512_000
    assert stream.info.sample_rate_hz == 64_000.0
    block = stream.read(1_000, 32)
    assert block.shape == (1, 32)
    np.testing.assert_array_equal(block.ravel(), signal[1_000:1_032])
    with pytest.raises(ValueError, match="exceeds"):
        stream.read(511_999, 2)


def test_hust_adapter_open_enforces_checksum(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    savemat(raw_dir / "N400.mat", {"data": np.zeros(512_000), "fs": np.array([12_000.0])})
    manifest = _write_hust_registry(tmp_path, raw_dir / "N400.mat")
    adapter = HustV3Adapter(manifest, raw_dir=raw_dir)

    # Tamper after registration: the byte content no longer matches the hash.
    savemat(raw_dir / "N400.mat", {"data": np.ones(512_000), "fs": np.array([12_000.0])})
    with pytest.raises(ValueError, match="mismatch"):
        adapter.open("N400.mat")


def test_hust_adapter_open_rejects_missing_raw_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    savemat(raw_dir / "N400.mat", {"data": np.zeros(512_000), "fs": np.array([12_000.0])})
    manifest = _write_hust_registry(tmp_path, raw_dir / "N400.mat")
    adapter = HustV3Adapter(manifest, raw_dir=raw_dir)
    (raw_dir / "N400.mat").unlink()
    with pytest.raises(FileNotFoundError, match="dataset fetch"):
        adapter.open("N400.mat")


MEHRAN_TOTAL_BYTES = 159_324_234


def _write_mehran_csv(path: Path, rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, 3))
    lines = ["Time Stamp,X-axis,Y-axis,Z-axis"]
    for index, row in enumerate(values):
        lines.append(
            f"{index / 12_800:.6f},{row[0]:.8f},{row[1]:.8f},{row[2]:.8f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return values


def _write_mehran_registry(root: Path, real_file: Path) -> Path:
    """A 38-entry registry matching the frozen 36-file grid and byte total."""

    real_digest = hashlib.sha256(real_file.read_bytes()).hexdigest()
    real_size = real_file.stat().st_size
    entries = []
    for location in ("inner", "outer"):
        for size in (0.7, 0.9, 1.1, 1.3, 1.5, 1.7):
            for load in (100, 200, 300):
                filename = f"{size}{location}-{load}watt.csv"
                is_real = filename == real_file.name
                entries.append(
                    {
                        "id": filename,
                        "filename": filename,
                        "size_bytes": real_size if is_real else 4_000_000,
                        "sha256": real_digest if is_real else "0" * 64,
                        "download_url": "https://example.org",
                        "selected": True,
                        "exclusion_reason": None,
                    }
                )
    # Two excluded entries push the file count from 36 to the frozen 38.
    for suffix in ("extra-a", "extra-b"):
        entries.append(
            {
                "id": suffix,
                "filename": f"0.7inner-100watt-{suffix}.csv",
                "size_bytes": 4_000_000,
                "sha256": "0" * 64,
                "download_url": "https://example.org",
                "selected": False,
                "exclusion_reason": "excluded by the frozen fixture selection",
            }
        )
    total = sum(item["size_bytes"] for item in entries)
    remainder = MEHRAN_TOTAL_BYTES - total
    entries[-1]["size_bytes"] += remainder
    assert sum(item["size_bytes"] for item in entries) == MEHRAN_TOTAL_BYTES
    assert len(entries) == 38

    manifest = root / "registry" / "mehran_v2.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "mehran-triaxial-bearing-v2",
                "version": 2,
                "title": "Mehran fixture registry",
                "doi": "10.17632/fm6xzxnf36.2",
                "landing_page": "https://example.org",
                "license": {"spdx": "CC-BY-4.0"},
                "sampling_rate_hz": 12_800,
                "protocol_status": "accepted",
                "files": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_mehran_adapter_opens_selected_file_and_streams(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_path = raw_dir / "0.7inner-100watt.csv"
    values = _write_mehran_csv(csv_path, rows=4_096, seed=29)

    manifest = _write_mehran_registry(tmp_path, csv_path)
    adapter = MehranV2Adapter(manifest, raw_dir=raw_dir)
    assert len(adapter.recordings()) == 36

    stream = adapter.open("0.7inner-100watt.csv")
    assert stream.info.channels == ("acc_x", "acc_y", "acc_z")
    assert stream.info.total_samples == 4_096
    block = stream.read(64, 128)
    assert block.shape == (3, 128)
    np.testing.assert_allclose(block, values[64:192].T, atol=1e-6)


def test_mehran_adapter_open_rejects_unknown_recording(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_path = raw_dir / "0.7inner-100watt.csv"
    _write_mehran_csv(csv_path, rows=4_096, seed=7)
    manifest = _write_mehran_registry(tmp_path, csv_path)
    adapter = MehranV2Adapter(manifest, raw_dir=raw_dir)
    with pytest.raises(ValueError, match="unknown Mehran recording"):
        adapter.open("1.9inner-100watt.csv")
