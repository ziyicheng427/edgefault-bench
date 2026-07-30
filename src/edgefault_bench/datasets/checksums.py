"""Shared checksum helpers for public dataset adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class RegisteredFile(Protocol):
    """Minimum immutable registry fields required for download verification."""

    filename: str
    size_bytes: int
    sha256: str
    download_url: str


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_registered_file(path: str | Path, specification: RegisteredFile) -> None:
    """Fail unless a local file matches its registry byte size and SHA-256."""

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
