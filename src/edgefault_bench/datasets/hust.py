"""Adapter for the CC BY 4.0 HUST bearing dataset, version 3."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

_FILENAME_PATTERN = re.compile(
    r"^(?P<label>IO|IB|OB|N|I|O|B)(?P<bearing>[4-8])(?P<load>00|02|04)\.mat$"
)
_LOAD_CODES = {"00": 0, "02": 200, "04": 400}


@dataclass(frozen=True)
class HustFile:
    filename: str
    file_id: str
    size_bytes: int
    sha256: str
    download_url: str
    label: str
    bearing_type: str
    load_w: int


@dataclass(frozen=True)
class WindowRecord:
    filename: str
    recording: str
    label: str
    bearing_type: str
    load_w: int
    start: int
    stop: int

    def domain_value(self, field: str) -> str | int:
        if field == "load_w":
            return self.load_w
        if field == "bearing_type":
            return self.bearing_type
        raise ValueError(f"unsupported HUST domain field: {field!r}")


def parse_hust_filename(filename: str) -> tuple[str, str, int]:
    """Return fault label, ISO-style bearing number, and load in watts."""

    match = _FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"invalid HUST filename: {filename!r}")
    return (
        match.group("label"),
        f"620{match.group('bearing')}",
        _LOAD_CODES[match.group("load")],
    )


def load_hust_manifest(path: str | Path) -> tuple[dict[str, Any], tuple[HustFile, ...]]:
    """Load and validate the immutable v3 registry snapshot."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != "hust-bearing-v3" or payload.get("version") != 3:
        raise ValueError("expected the pinned HUST bearing v3 manifest")
    if payload.get("doi") != "10.17632/cbv7jyx4p9.3":
        raise ValueError("unexpected HUST dataset DOI")
    if payload.get("license", {}).get("spdx") != "CC-BY-4.0":
        raise ValueError("HUST manifest must retain the verified CC-BY-4.0 license")

    files: list[HustFile] = []
    for item in payload.get("files", []):
        label, bearing_type, load_w = parse_hust_filename(item["filename"])
        sha256 = item["sha256"].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"invalid SHA-256 for {item['filename']}")
        files.append(
            HustFile(
                filename=item["filename"],
                file_id=item["id"],
                size_bytes=int(item["size_bytes"]),
                sha256=sha256,
                download_url=item["download_url"],
                label=label,
                bearing_type=bearing_type,
                load_w=load_w,
            )
        )
    if len(files) != 60 or len({item.filename for item in files}) != len(files):
        raise ValueError("the v1 HUST registry must contain exactly 60 unique MAT files")
    return payload, tuple(files)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hust_file(path: str | Path, specification: HustFile) -> None:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    actual_size = file_path.stat().st_size
    if actual_size != specification.size_bytes:
        raise ValueError(
            f"size mismatch for {specification.filename}: "
            f"expected {specification.size_bytes}, got {actual_size}"
        )
    actual_sha256 = sha256_file(file_path)
    if actual_sha256 != specification.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {specification.filename}: "
            f"expected {specification.sha256}, got {actual_sha256}"
        )


def load_hust_signal(path: str | Path) -> tuple[NDArray[np.float64], float]:
    """Load the 10-second steady-state vibration and measured shaft frequency."""

    payload = loadmat(Path(path), variable_names=["data", "fs"])
    if "data" not in payload or "fs" not in payload:
        raise ValueError(f"HUST MAT file is missing 'data' or 'fs': {path}")
    signal = np.asarray(payload["data"], dtype=np.float64).reshape(-1)
    shaft_frequency = float(np.asarray(payload["fs"]).reshape(-1)[0])
    if signal.size not in {512_000, 512_001}:
        raise ValueError(
            f"expected 512,000 or 512,001 steady-state samples, got {signal.size}: {path}"
        )
    if not np.isfinite(signal).all() or not np.isfinite(shaft_frequency):
        raise ValueError(f"HUST MAT file contains non-finite values: {path}")
    return signal[:512_000], shaft_frequency


def build_window_records(
    files: tuple[HustFile, ...], *, window_length: int = 4096, stride: int = 4096
) -> tuple[WindowRecord, ...]:
    """Build deterministic, non-overlapping window metadata without reading signals."""

    if window_length <= 0 or stride < window_length:
        raise ValueError("window_length must be positive and stride must prevent overlap")
    records: list[WindowRecord] = []
    for item in sorted(files, key=lambda value: value.filename):
        for start in range(0, 512_000 - window_length + 1, stride):
            records.append(
                WindowRecord(
                    filename=item.filename,
                    recording=item.filename,
                    label=item.label,
                    bearing_type=item.bearing_type,
                    load_w=item.load_w,
                    start=start,
                    stop=start + window_length,
                )
            )
    return tuple(records)


def preprocess_window(window: NDArray[np.float64], *, epsilon: float = 1e-8) -> NDArray[np.float32]:
    """Apply per-window centering and scaling without cross-sample test statistics."""

    values = np.asarray(window, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("window must be a non-empty one-dimensional array")
    centered = values - values.mean()
    scale = centered.std()
    return (centered / max(float(scale), epsilon)).astype(np.float32)
