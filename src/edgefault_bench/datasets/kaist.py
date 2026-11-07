"""Adapter for the CC BY 4.0 KAIST rotating-machine acoustic subset, v6.

Source: "Vibration, Acoustic, Temperature, and Motor Current Dataset of
Rotating Machine Under Varying Load Conditions for Fault Diagnosis"
(https://data.mendeley.com/datasets/ztmf3m7h5x/6, DOI 10.17632/ztmf3m7h5x.6).
This first integration pins the acoustic archive: five Siemens Test.Lab MAT
recordings (60 s, 51.2 kHz, Pascal) covering Normal, BPFI, and BPFO conditions
at the single 0 N·m load available in that archive. Vibration and
current/temperature archives remain future integrations, so no cross-load task
is frozen yet; see the data card for the boundary.
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
from scipy.io import loadmat

from edgefault_bench.contracts import DatasetMetadata, Recording, SignalInfo
from edgefault_bench.datasets.checksums import sha256_file, verify_registered_file
from edgefault_bench.datasets.raw import resolve_raw_dir

_FILENAME_PATTERN = re.compile(
    r"^(?P<load>[0-9]+)Nm_(?P<fault>Normal|BPFI|BPFO)(?:_(?P<severity>[0-9]{2}))?\.mat$"
)
_SAMPLE_RATE_HZ = 51_200.0
_TOTAL_SAMPLES = 3_072_000


@dataclass(frozen=True)
class KaistAcousticFile:
    recording_id: str
    filename: str
    size_bytes: int
    sha256: str
    label: str
    fault_family: str
    severity_mm: int | None
    load_nm: int


def load_kaist_manifest(path: str | Path) -> tuple[dict[str, Any], tuple[KaistAcousticFile, ...]]:
    """Load and validate the pinned KAIST acoustic v6 registry snapshot."""

    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("dataset_id") != "kaist-rotating-acoustic-v6" or payload.get("version") != 6:
        raise ValueError("expected the pinned KAIST acoustic v6 registry")
    if payload.get("doi") != "10.17632/ztmf3m7h5x.6":
        raise ValueError("unexpected KAIST dataset DOI")
    if payload.get("license", {}).get("spdx") != "CC-BY-4.0":
        raise ValueError("KAIST registry must retain the official CC-BY-4.0 license")
    if payload.get("sampling_rate_hz") != _SAMPLE_RATE_HZ:
        raise ValueError("KAIST acoustic registry must declare the 51.2 kHz rate")

    raw_files = payload.get("files", [])
    if len(raw_files) != 5 or len({item["filename"] for item in raw_files}) != 5:
        raise ValueError("the KAIST acoustic registry must contain exactly 5 unique files")
    files: list[KaistAcousticFile] = []
    for item in raw_files:
        match = _FILENAME_PATTERN.fullmatch(item["filename"])
        if match is None:
            raise ValueError(f"invalid KAIST filename: {item['filename']!r}")
        sha256 = item["sha256"].lower()
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError(f"invalid SHA-256 for {item['filename']}")
        severity = None if match.group("severity") is None else int(match.group("severity"))
        files.append(
            KaistAcousticFile(
                recording_id=item["recording_id"],
                filename=item["filename"],
                size_bytes=int(item["size_bytes"]),
                sha256=sha256,
                label=item["label"],
                fault_family=item["fault_family"],
                severity_mm=severity,
                load_nm=int(item["load_nm"]),
            )
        )
    expected = {
        ("normal", "Normal", None),
        ("bpfi_03", "Inner", 3),
        ("bpfi_10", "Inner", 10),
        ("bpfo_03", "Outer", 3),
        ("bpfo_10", "Outer", 10),
    }
    actual = {(item.label, item.fault_family, item.severity_mm) for item in files}
    if actual != expected or {item.load_nm for item in files} != {0}:
        raise ValueError("KAIST acoustic files do not form the expected five-condition grid")
    return payload, tuple(files)


def _load_testlab(
    path: str | Path, *, expected_rate_hz: float
) -> tuple[NDArray[np.float64], float]:
    """Read a KAIST Test.Lab MAT file as ``(channels, samples)`` plus its rate."""

    payload = loadmat(
        Path(path), struct_as_record=False, squeeze_me=True, variable_names=["Signal"]
    )
    if "Signal" not in payload:
        raise ValueError(f"KAIST MAT file has no 'Signal' struct: {path}")
    signal = payload["Signal"]
    try:
        y_values = signal.y_values
        x_values = signal.x_values
        values = np.asarray(y_values.values, dtype=np.float64)
        increment = float(np.asarray(x_values.increment).reshape(-1)[0])
        count = int(np.asarray(x_values.number_of_values).reshape(-1)[0])
    except AttributeError as error:
        raise ValueError(f"unexpected Test.Lab structure in {path}") from error
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] != count:
        raise ValueError(
            f"KAIST signal length {values.shape[0]} contradicts x_values count "
            f"{count}: {path}"
        )
    rate = 1.0 / increment
    if abs(rate - expected_rate_hz) > 1.0:
        raise ValueError(
            f"KAIST signal rate {rate:.2f} Hz contradicts {expected_rate_hz:.2f} Hz: {path}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"KAIST signal contains non-finite values: {path}")
    return values.T.copy(), rate


def load_kaist_acoustic_signal(path: str | Path) -> NDArray[np.float64]:
    """Read the Test.Lab acoustic channel as ``(1, samples)`` float64."""

    values, _ = _load_testlab(path, expected_rate_hz=_SAMPLE_RATE_HZ)
    return values


class KaistAcousticSignalStream:
    """Single-channel acoustic pressure stream (Pascal) from one MAT file."""

    def __init__(self, path: str | Path) -> None:
        self._values = load_kaist_acoustic_signal(path)
        self._info = SignalInfo(
            channels=("acoustic_pa",),
            total_samples=int(self._values.shape[1]),
            sample_rate_hz=_SAMPLE_RATE_HZ,
            units=("Pa",),
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


class KaistAcousticV6Adapter:
    """Expose the five acoustic recordings through the generic contracts."""

    def __init__(self, manifest_path: str | Path, *, raw_dir: str | Path | None = None) -> None:
        payload, files = load_kaist_manifest(manifest_path)
        self._files = files
        self._metadata = DatasetMetadata(
            dataset_id=payload["dataset_id"],
            version=str(payload["version"]),
            title=payload["title"],
            license_spdx=payload["license"]["spdx"],
            source_url=payload["landing_page"],
            domain_fields=("load_nm", "fault_family"),
        )
        self._raw_dir = (
            Path(raw_dir) if raw_dir is not None else resolve_raw_dir("kaist_acoustic_v6")
        )

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
                domains={
                    "load_nm": item.load_nm,
                    "fault_family": item.fault_family,
                    "severity_mm": item.severity_mm if item.severity_mm is not None else 0,
                },
                sample_rate_hz=_SAMPLE_RATE_HZ,
                sample_count=_TOTAL_SAMPLES,
                sha256=item.sha256,
            )
            for item in self._files
        )

    def open(self, recording_id: str) -> KaistAcousticSignalStream:
        known = {item.recording_id for item in self._files}
        if recording_id not in known:
            raise ValueError(
                f"unknown KAIST recording {recording_id!r}; expected one of 5 files"
            )
        specification = next(
            item for item in self._files if item.recording_id == recording_id
        )
        path = self._raw_dir / specification.filename
        if not path.is_file():
            raise FileNotFoundError(
                f"raw KAIST file is missing: {path}; run 'edgefault dataset fetch' first"
            )
        verify_registered_file(path, specification)
        return KaistAcousticSignalStream(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the pinned KAIST acoustic archive, verify its SHA-256, "
            "extract the five MAT recordings, and verify each against the registry."
        )
    )
    parser.add_argument("--manifest", default="registry/kaist_acoustic_v6.json")
    parser.add_argument("--raw-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    from edgefault_bench.download import download_file

    args = build_parser().parse_args(argv)
    payload, files = load_kaist_manifest(args.manifest)
    archive = payload["archive"]
    raw_dir = Path(args.raw_dir) if args.raw_dir else resolve_raw_dir("kaist_acoustic_v6")
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / archive["filename"]

    if not archive_path.exists():
        class _Specification:
            filename = archive["filename"]
            size_bytes = int(archive["size_bytes"])
            sha256 = archive["sha256"]
            download_url = archive["url"]

        download_file(_Specification(), raw_dir)  # type: ignore[arg-type]
    elif sha256_file(archive_path) != archive["sha256"]:
        raise ValueError("existing KAIST archive failed SHA-256 verification")

    print(f"kaist acoustic v{payload['version']} | CC-BY-4.0 | files={len(files)}")
    with zipfile.ZipFile(archive_path) as bundle:
        names = set(bundle.namelist())
        for item in files:
            if item.filename not in names:
                raise ValueError(f"archive does not contain {item.filename!r}")
            destination = raw_dir / item.filename
            if not destination.exists():
                destination.write_bytes(bundle.read(item.filename))
    for item in files:
        verify_registered_file(raw_dir / item.filename, item)
    print(f"verified and extracted {len(files)} recordings into {raw_dir}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Vibration subset (four-channel acceleration, three loads)
# ---------------------------------------------------------------------------

_VIBRATION_RATE_HZ = 25_600.0
_VIBRATION_CHANNELS = ("vib_ax_g", "vib_ay_g", "vib_bx_g", "vib_by_g")
_VIBRATION_PATTERN = re.compile(
    r"^(?P<load>[0-9]+)Nm_(?P<fault>Normal|BPFI|BPFO|Misalign|Unbalalnce|Unbalance)"
    r"(?:_(?P<sev>[0-9a-z]+))?\.mat$"
)
_VIBRATION_FAMILY = {
    "Normal": "normal",
    "BPFI": "bpfi",
    "BPFO": "bpfo",
    "Misalign": "misalign",
    "Unbalalnce": "unbalance",  # upstream misspelling in the 2 N·m files
    "Unbalance": "unbalance",
}


@dataclass(frozen=True)
class KaistVibrationFile:
    recording_id: str
    filename: str
    size_bytes: int
    sha256: str
    sample_count: int
    label: str
    load_nm: int
    severity: str


def load_kaist_vibration_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], tuple[KaistVibrationFile, ...]]:
    """Load and validate the pinned KAIST vibration v6 registry snapshot."""

    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("dataset_id") != "kaist-rotating-vibration-v6" or payload.get("version") != 6:
        raise ValueError("expected the pinned KAIST vibration v6 registry")
    if payload.get("doi") != "10.17632/ztmf3m7h5x.6":
        raise ValueError("unexpected KAIST dataset DOI")
    if payload.get("license", {}).get("spdx") != "CC-BY-4.0":
        raise ValueError("KAIST registry must retain the official CC-BY-4.0 license")
    if payload.get("sampling_rate_hz") != _VIBRATION_RATE_HZ:
        raise ValueError("KAIST vibration registry must declare the 25.6 kHz rate")

    raw_files = payload.get("files", [])
    if len(raw_files) != 45 or len({item["filename"] for item in raw_files}) != 45:
        raise ValueError("the KAIST vibration registry must contain exactly 45 unique files")
    files: list[KaistVibrationFile] = []
    for item in raw_files:
        match = _VIBRATION_PATTERN.fullmatch(item["filename"])
        if match is None:
            raise ValueError(f"invalid KAIST vibration filename: {item['filename']!r}")
        sha256 = item["sha256"].lower()
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError(f"invalid SHA-256 for {item['filename']}")
        label = _VIBRATION_FAMILY[match.group("fault")]
        if item["label"] != label:
            raise ValueError(f"registry label contradicts filename: {item['filename']!r}")
        files.append(
            KaistVibrationFile(
                recording_id=item["recording_id"],
                filename=item["filename"],
                size_bytes=int(item["size_bytes"]),
                sha256=sha256,
                sample_count=int(item["sample_count"]),
                label=label,
                load_nm=int(match.group("load")),
                severity=match.group("sev") or "none",
            )
        )
    grid = {(item.load_nm, item.label) for item in files}
    expected = {
        (load, family)
        for load in (0, 2, 4)
        for family in ("normal", "bpfi", "bpfo", "misalign", "unbalance")
    }
    if grid != expected:
        raise ValueError("vibration files do not form the expected 3x5 condition grid")
    return payload, tuple(files)


class KaistVibrationSignalStream:
    """Four-channel acceleration stream (g) from one Test.Lab MAT file."""

    def __init__(self, path: str | Path) -> None:
        values, rate = _load_testlab(path, expected_rate_hz=_VIBRATION_RATE_HZ)
        if values.shape[0] != len(_VIBRATION_CHANNELS):
            raise ValueError(
                f"vibration file must expose four channels, got {values.shape[0]}: {path}"
            )
        self._values = values
        self._info = SignalInfo(
            channels=_VIBRATION_CHANNELS,
            total_samples=int(values.shape[1]),
            sample_rate_hz=rate,
            units=("g",) * len(_VIBRATION_CHANNELS),
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


class KaistVibrationV6Adapter:
    """Expose the 45 vibration recordings through the generic contracts."""

    def __init__(self, manifest_path: str | Path, *, raw_dir: str | Path | None = None) -> None:
        payload, files = load_kaist_vibration_manifest(manifest_path)
        self._files = files
        self._metadata = DatasetMetadata(
            dataset_id=payload["dataset_id"],
            version=str(payload["version"]),
            title=payload["title"],
            license_spdx=payload["license"]["spdx"],
            source_url=payload["landing_page"],
            domain_fields=("load_nm", "fault_family"),
        )
        self._raw_dir = (
            Path(raw_dir) if raw_dir is not None else resolve_raw_dir("kaist_vibration_v6")
        )

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
                domains={
                    "load_nm": item.load_nm,
                    "fault_family": item.label,
                    "severity": item.severity,
                },
                sample_rate_hz=_VIBRATION_RATE_HZ,
                sample_count=item.sample_count,
                sha256=item.sha256,
            )
            for item in self._files
        )

    def open(self, recording_id: str) -> KaistVibrationSignalStream:
        known = {item.recording_id for item in self._files}
        if recording_id not in known:
            raise ValueError(
                f"unknown KAIST vibration recording {recording_id!r}; expected one of 45 files"
            )
        specification = next(
            item for item in self._files if item.recording_id == recording_id
        )
        path = self._raw_dir / specification.filename
        if not path.is_file():
            raise FileNotFoundError(
                f"raw KAIST vibration file is missing: {path}; run 'edgefault dataset fetch' first"
            )
        verify_registered_file(path, specification)
        return KaistVibrationSignalStream(path)
