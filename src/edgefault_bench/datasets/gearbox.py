"""Adapter for the CC BY 4.0 two-stage gearbox fault-diagnosis dataset, v1.

Source: DOE / Data.gov catalogue entry "Gearbox Fault Diagnosis Data"
(https://catalog.data.gov/dataset/gearbox-fault-diagnosis-data), distributed as
one ZIP of tab-separated four-accelerometer text files. Twenty recordings cover
a healthy gear and a broken-tooth gear at ten load levels (0–90 %) with a 30 Hz
shaft speed. The source states no sampling rate; the adapter records ``None``
and spectral preprocessing fails closed instead of assuming a value.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from edgefault_bench.contracts import DatasetMetadata, Recording, SignalInfo
from edgefault_bench.datasets.checksums import sha256_file, verify_registered_file
from edgefault_bench.datasets.raw import resolve_raw_dir

_FILENAME_PATTERN = re.compile(r"^(?P<condition>h|b)30hz(?P<load>[0-9]{1,2})\.txt$")
_LOADS = tuple(range(0, 100, 10))
_CHANNELS = ("accelerometer_1", "accelerometer_2", "accelerometer_3", "accelerometer_4")


@dataclass(frozen=True)
class GearboxFile:
    recording_id: str
    filename: str
    zip_path: str
    size_bytes: int
    sha256: str
    row_count: int
    label: str
    load_percent: int


def load_gearbox_manifest(path: str | Path) -> tuple[dict[str, Any], tuple[GearboxFile, ...]]:
    """Load and validate the immutable gearbox v1 registry snapshot."""

    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("dataset_id") != "gearbox-two-stage-v1" or payload.get("version") != 1:
        raise ValueError("expected the pinned gearbox two-stage v1 registry")
    if payload.get("license", {}).get("spdx") != "CC-BY-4.0":
        raise ValueError("gearbox registry must retain the verified CC-BY-4.0 license")
    if payload.get("sampling_rate_hz") is not None:
        raise ValueError("gearbox registry must not assert an unstated sampling rate")

    raw_files = payload.get("files", [])
    if len(raw_files) != 20 or len({item["filename"] for item in raw_files}) != 20:
        raise ValueError("the gearbox v1 registry must contain exactly 20 unique files")
    files: list[GearboxFile] = []
    for item in raw_files:
        match = _FILENAME_PATTERN.fullmatch(item["filename"])
        if match is None:
            raise ValueError(f"invalid gearbox filename: {item['filename']!r}")
        condition, load = match.group("condition"), int(match.group("load"))
        expected_label = "healthy" if condition == "h" else "broken_tooth"
        if item["label"] != expected_label or item["load_percent"] != load:
            raise ValueError(f"gearbox metadata contradicts filename: {item['filename']!r}")
        sha256 = item["sha256"].lower()
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError(f"invalid SHA-256 for {item['filename']}")
        files.append(
            GearboxFile(
                recording_id=item["recording_id"],
                filename=item["filename"],
                zip_path=item["zip_path"],
                size_bytes=int(item["size_bytes"]),
                sha256=sha256,
                row_count=int(item["row_count"]),
                label=expected_label,
                load_percent=load,
            )
        )
    expected = {
        (label, load)
        for label in ("healthy", "broken_tooth")
        for load in _LOADS
    }
    actual = {(item.label, item.load_percent) for item in files}
    if actual != expected:
        raise ValueError("gearbox files do not form the expected 2x10 load grid")
    archive = payload.get("archive", {})
    if not archive.get("sha256") or not archive.get("url"):
        raise ValueError("gearbox registry must pin the source archive hash and URL")
    return payload, tuple(files)


def _load_gearbox_matrix(path: str | Path) -> NDArray[np.float64]:
    # Source rows end with a trailing tab; usecols ignores the empty last field.
    values = np.loadtxt(path, delimiter="\t", usecols=(0, 1, 2, 3), dtype=np.float64, ndmin=2)
    if values.ndim != 2 or values.shape[1] != len(_CHANNELS):
        raise ValueError(f"gearbox file must have four sensor columns: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"gearbox file contains non-finite values: {path}")
    return values


class GearboxSignalStream:
    """Four-channel accelerometer stream over one tab-separated recording."""

    def __init__(self, path: str | Path) -> None:
        self._values = _load_gearbox_matrix(path).T
        self._info = SignalInfo(
            channels=_CHANNELS,
            total_samples=int(self._values.shape[1]),
            sample_rate_hz=None,
        )

    @property
    def info(self) -> SignalInfo:
        return self._info

    def read(self, start: int, count: int) -> NDArray[np.float64]:
        if count <= 0 or start < 0 or start + count > self._info.total_samples:
            raise ValueError(
                f"read [{start}, {start + count}) exceeds recording of "
                f"{self._info.total_samples} samples"
            )
        return self._values[:, start : start + count].copy()


class GearboxV1Adapter:
    """Expose the pinned 20-file gearbox grid through the generic contracts."""

    def __init__(self, manifest_path: str | Path, *, raw_dir: str | Path | None = None) -> None:
        payload, files = load_gearbox_manifest(manifest_path)
        self._files = files
        self._metadata = DatasetMetadata(
            dataset_id=payload["dataset_id"],
            version=str(payload["version"]),
            title=payload["title"],
            license_spdx=payload["license"]["spdx"],
            source_url=payload["landing_page"],
            domain_fields=("load_percent", "gear_condition"),
        )
        self._raw_dir = Path(raw_dir) if raw_dir is not None else resolve_raw_dir("gearbox_v1")

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def recordings(self) -> tuple[Recording, ...]:
        return tuple(
            Recording(
                dataset_id=self.metadata.dataset_id,
                recording_id=item.recording_id,
                source_file=item.filename,
                label=item.label,
                domains={"load_percent": item.load_percent, "gear_condition": item.label},
                sample_rate_hz=None,
                sample_count=item.row_count,
                sha256=item.sha256,
            )
            for item in self._files
        )

    def open(self, recording_id: str) -> GearboxSignalStream:
        known = {item.recording_id for item in self._files}
        if recording_id not in known:
            raise ValueError(
                f"unknown gearbox recording {recording_id!r}; expected one of 20 files"
            )
        specification = next(
            item for item in self._files if item.recording_id == recording_id
        )
        path = self._raw_dir / specification.filename
        if not path.is_file():
            raise FileNotFoundError(
                f"raw gearbox file is missing: {path}; run 'edgefault dataset fetch' first"
            )
        verify_registered_file(path, specification)
        return GearboxSignalStream(path)


def _extract_archive(archive_path: Path, raw_dir: Path, files: tuple[GearboxFile, ...]) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for item in files:
            if item.zip_path not in names:
                raise ValueError(f"archive does not contain {item.zip_path!r}")
            destination = raw_dir / item.filename
            if destination.exists():
                continue
            destination.write_bytes(archive.read(item.zip_path))
        for item in files:
            verify_registered_file(raw_dir / item.filename, item)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the pinned gearbox archive, verify its SHA-256, extract the "
            "twenty recordings, and verify every file against the registry."
        )
    )
    parser.add_argument("--manifest", default="registry/gearbox_v1.json")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    return parser


def main(argv: list[str] | None = None) -> None:
    from edgefault_bench.download import download_file

    args = build_parser().parse_args(argv)
    payload, files = load_gearbox_manifest(args.manifest)
    archive = payload["archive"]
    raw_dir = Path(args.raw_dir) if args.raw_dir else resolve_raw_dir("gearbox_v1")
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "gearboxdata.zip"

    if not archive_path.exists():
        class _Specification:
            filename = "gearboxdata.zip"
            size_bytes = int(archive["size_bytes"])
            sha256 = archive["sha256"]
            download_url = archive["url"]

        download_file(_Specification(), raw_dir)  # type: ignore[arg-type]
    else:
        if sha256_file(archive_path) != archive["sha256"]:
            raise ValueError("existing gearbox archive failed SHA-256 verification")

    print(f"gearbox v{payload['version']} | CC-BY-4.0 | files={len(files)}")
    _extract_archive(archive_path, raw_dir, files)
    print(f"verified and extracted {len(files)} recordings into {raw_dir}")


if __name__ == "__main__":
    main()
