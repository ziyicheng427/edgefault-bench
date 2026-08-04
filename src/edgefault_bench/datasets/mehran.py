"""Adapter for the accepted CC BY 4.0 Mehran triaxial bearing dataset, version 2."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from edgefault_bench.contracts import DatasetMetadata, Recording
from edgefault_bench.datasets.checksums import verify_registered_file

_FILENAME_PATTERN = re.compile(
    r"^(?P<size>0\.7|0\.9|1\.1|1\.3|1\.5|1\.7)"
    r"(?P<location>inner|outer)-(?P<load>100|200|300)watt"
    r"(?:-[A-Za-z0-9]+)?\.csv$"
)
_EXPECTED_SIZES = (0.7, 0.9, 1.1, 1.3, 1.5, 1.7)
_EXPECTED_LOADS = (100, 200, 300)
_EXPECTED_HEADER = ("Time Stamp", "X-axis", "Y-axis", "Z-axis")


@dataclass(frozen=True)
class MehranFile:
    filename: str
    file_id: str
    size_bytes: int
    sha256: str
    download_url: str
    label: str
    defect_size_mm: float
    load_w: int


@dataclass(frozen=True)
class MehranWindowRecord:
    """Metadata for one non-overlapping window from an upstream CSV recording."""

    filename: str
    recording: str
    label: str
    defect_size_mm: float
    load_w: int
    start: int
    stop: int

    def domain_value(self, field: str) -> float | int:
        if field == "load_w":
            return self.load_w
        if field == "defect_size_mm":
            return self.defect_size_mm
        raise ValueError(f"unsupported Mehran domain field: {field!r}")


def parse_mehran_filename(filename: str) -> tuple[str, float, int]:
    """Return normalized fault location, defect size in millimetres, and load."""

    match = _FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"invalid selected Mehran filename: {filename!r}")
    return (
        f"{match.group('location')}_race",
        float(match.group("size")),
        int(match.group("load")),
    )


def load_mehran_manifest(path: str | Path) -> tuple[dict[str, Any], tuple[MehranFile, ...]]:
    """Load and validate the accepted immutable Mehran v2 registry snapshot."""

    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("dataset_id") != "mehran-triaxial-bearing-v2" or payload.get("version") != 2:
        raise ValueError("expected the pinned Mehran triaxial bearing v2 registry")
    if payload.get("doi") != "10.17632/fm6xzxnf36.2":
        raise ValueError("unexpected Mehran dataset DOI")
    if payload.get("protocol_status") != "accepted":
        raise ValueError("Mehran registry must retain the Decision 0004 acceptance status")
    if payload.get("license", {}).get("spdx") != "CC-BY-4.0":
        raise ValueError("Mehran registry must retain the official CC-BY-4.0 license")

    raw_files = payload.get("files", [])
    if len(raw_files) != 38 or len({item["filename"] for item in raw_files}) != 38:
        raise ValueError("the Mehran v2 registry must contain exactly 38 unique files")
    if sum(int(item["size_bytes"]) for item in raw_files) != 159_324_234:
        raise ValueError("unexpected total byte size for the Mehran v2 registry")

    files: list[MehranFile] = []
    excluded = 0
    for item in raw_files:
        sha256 = item["sha256"].lower()
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError(f"invalid SHA-256 for {item['filename']}")
        if not item["selected"]:
            excluded += 1
            if not item.get("exclusion_reason"):
                raise ValueError(f"excluded file has no reason: {item['filename']}")
            continue
        if item.get("exclusion_reason") is not None:
            raise ValueError(f"selected file has an exclusion reason: {item['filename']}")
        label, defect_size_mm, load_w = parse_mehran_filename(item["filename"])
        files.append(
            MehranFile(
                filename=item["filename"],
                file_id=item["id"],
                size_bytes=int(item["size_bytes"]),
                sha256=sha256,
                download_url=item["download_url"],
                label=label,
                defect_size_mm=defect_size_mm,
                load_w=load_w,
            )
        )
    expected_grid = {
        (label, size, load)
        for label in ("inner_race", "outer_race")
        for size in _EXPECTED_SIZES
        for load in _EXPECTED_LOADS
    }
    actual_grid = {(item.label, item.defect_size_mm, item.load_w) for item in files}
    if len(files) != 36 or excluded != 2 or actual_grid != expected_grid:
        raise ValueError("selected Mehran files do not form the expected 2x6x3 grid")
    return payload, tuple(files)


def verify_mehran_file(path: str | Path, specification: MehranFile) -> None:
    verify_registered_file(path, specification)


def load_mehran_signal(
    path: str | Path, *, minimum_samples: int = 4096
) -> NDArray[np.float32]:
    """Load X/Y/Z acceleration columns while ignoring the display timestamp."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = tuple(value.strip() for value in next(csv.reader(stream)))
    if header != _EXPECTED_HEADER:
        raise ValueError(f"unexpected Mehran CSV header in {file_path}: {header}")
    values = np.loadtxt(
        file_path,
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3),
        dtype=np.float64,
        ndmin=2,
    )
    if values.shape[1] != 3 or values.shape[0] < minimum_samples:
        raise ValueError(
            f"expected at least {minimum_samples} triaxial samples, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Mehran CSV contains non-finite acceleration values: {file_path}")
    return values.astype(np.float32)


def build_mehran_window_records(
    files: tuple[MehranFile, ...],
    sample_counts: Mapping[str, int],
    *,
    window_length: int = 4096,
    stride: int = 4096,
) -> tuple[MehranWindowRecord, ...]:
    """Build deterministic metadata from observed CSV lengths without padding."""

    if window_length <= 0 or stride < window_length:
        raise ValueError("window_length must be positive and stride must prevent overlap")
    expected_names = {item.filename for item in files}
    if set(sample_counts) != expected_names:
        missing = sorted(expected_names - set(sample_counts))
        extra = sorted(set(sample_counts) - expected_names)
        raise ValueError(
            f"sample-count keys do not match registry; missing={missing}, extra={extra}"
        )
    records: list[MehranWindowRecord] = []
    for item in sorted(files, key=lambda value: value.filename):
        sample_count = sample_counts[item.filename]
        if sample_count < window_length:
            raise ValueError(f"recording is shorter than one window: {item.filename}")
        for start in range(0, sample_count - window_length + 1, stride):
            records.append(
                MehranWindowRecord(
                    filename=item.filename,
                    recording=item.filename,
                    label=item.label,
                    defect_size_mm=item.defect_size_mm,
                    load_w=item.load_w,
                    start=start,
                    stop=start + window_length,
                )
            )
    return tuple(records)


def preprocess_mehran_windows(
    signal: NDArray[np.float32],
    *,
    window_length: int = 4096,
    stride: int = 4096,
    epsilon: float = 1e-8,
) -> NDArray[np.float32]:
    """Return ``(windows, channels, time)`` with independent channel z-scores."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Mehran signal must have shape (time, 3)")
    if window_length <= 0 or stride < window_length:
        raise ValueError("window_length must be positive and stride must prevent overlap")
    if values.shape[0] < window_length:
        raise ValueError("Mehran signal is shorter than one window")
    starts = range(0, values.shape[0] - window_length + 1, stride)
    windows = np.stack([values[start : start + window_length].T for start in starts])
    centered = windows - windows.mean(axis=2, keepdims=True)
    scale = np.maximum(centered.std(axis=2, keepdims=True), epsilon)
    return (centered / scale).astype(np.float32)


class MehranV2Adapter:
    """Expose the selected 36-file factorial grid through the generic contract."""

    def __init__(self, manifest_path: str | Path) -> None:
        payload, files = load_mehran_manifest(manifest_path)
        self._files = files
        self._metadata = DatasetMetadata(
            dataset_id=payload["dataset_id"],
            version=str(payload["version"]),
            title=payload["title"],
            license_spdx=payload["license"]["spdx"],
            source_url=payload["landing_page"],
            domain_fields=("load_w", "defect_size_mm"),
            doi=payload["doi"],
        )
        self._sample_rate_hz = float(payload["sampling_rate_hz"])

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    @property
    def files(self) -> tuple[MehranFile, ...]:
        return self._files

    def recordings(self) -> tuple[Recording, ...]:
        return tuple(
            Recording(
                dataset_id=self.metadata.dataset_id,
                recording_id=item.filename,
                source_file=item.filename,
                label=item.label,
                domains={"load_w": item.load_w, "defect_size_mm": item.defect_size_mm},
                sample_rate_hz=self._sample_rate_hz,
                sha256=item.sha256,
            )
            for item in self.files
        )
