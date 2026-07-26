from pathlib import Path

import pytest

from edgefault_bench.datasets.hust import HustFile
from edgefault_bench.download import download_file, select_files


def make_specification(payload: bytes, filename: str = "N400.mat") -> HustFile:
    import hashlib

    return HustFile(
        filename=filename,
        file_id="test-id",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        download_url="https://example.invalid/file",
        label="N",
        bearing_type="6204",
        load_w=0,
    )


def test_existing_valid_file_is_not_downloaded(tmp_path: Path) -> None:
    payload = b"verified test fixture"
    specification = make_specification(payload)
    (tmp_path / specification.filename).write_bytes(payload)
    assert download_file(specification, tmp_path) == "verified"


def test_existing_invalid_file_fails_closed(tmp_path: Path) -> None:
    specification = make_specification(b"expected")
    (tmp_path / specification.filename).write_bytes(b"different")
    with pytest.raises(ValueError, match="--repair"):
        download_file(specification, tmp_path)


def test_file_selection_rejects_unregistered_names() -> None:
    specification = make_specification(b"content")
    with pytest.raises(ValueError, match="not in the pinned registry"):
        select_files((specification,), ["unknown.mat"])
