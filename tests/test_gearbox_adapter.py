"""Gearbox adapter tests (T13): registry, streaming, and fail-closed checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from edgefault_bench.datasets.gearbox import (
    GearboxSignalStream,
    GearboxV1Adapter,
    load_gearbox_manifest,
)

LOADS = list(range(0, 100, 10))


def _write_recording(path: Path, *, rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    block = rng.normal(size=(rows, 4))
    # Match the source format: tab separated with a trailing tab on each row.
    lines = ["\t".join(f"{value:.5e}" for value in row) + "\t" for row in block]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_registry(root: Path) -> Path:
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    entries = []
    for condition, label in (("h", "healthy"), ("b", "broken_tooth")):
        for index, load in enumerate(LOADS):
            name = f"{condition}30hz{load}.txt"
            path = raw_dir / name
            rows = 64 + index  # distinct lengths exercise per-file sample counts
            _write_recording(path, rows=rows, seed=index)
            entries.append(
                {
                    "id": name.removesuffix(".txt"),
                    "recording_id": name.removesuffix(".txt"),
                    "filename": name,
                    "zip_path": f"{'Healthy' if condition == 'h' else 'BrokenTooth'} Data/{name}",
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "row_count": rows,
                    "label": label,
                    "load_percent": load,
                    "selected": True,
                    "exclusion_reason": None,
                }
            )
    manifest = root / "gearbox.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "gearbox-two-stage-v1",
                "title": "fixture",
                "landing_page": "https://example.org",
                "license": {"spdx": "CC-BY-4.0"},
                "version": 1,
                "sampling_rate_hz": None,
                "archive": {"url": "https://example.org/gearboxdata.zip", "sha256": "0" * 64},
                "files": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_registry_requires_the_full_two_by_ten_grid(tmp_path: Path) -> None:
    manifest = _write_registry(tmp_path)
    payload, files = load_gearbox_manifest(manifest)
    assert len(files) == 20
    assert payload["sampling_rate_hz"] is None

    payload["files"] = payload["files"][:-1]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="20 unique files"):
        load_gearbox_manifest(broken)


def test_adapter_streams_four_channels_without_a_sample_rate(tmp_path: Path) -> None:
    manifest = _write_registry(tmp_path)
    adapter = GearboxV1Adapter(manifest, raw_dir=tmp_path / "raw")
    recordings = adapter.recordings()
    assert len(recordings) == 20
    assert recordings[0].sample_rate_hz is None
    assert recordings[0].sample_count == 64

    stream = adapter.open("h30hz0")
    assert stream.info.channels == (
        "accelerometer_1",
        "accelerometer_2",
        "accelerometer_3",
        "accelerometer_4",
    )
    assert stream.info.sample_rate_hz is None
    block = stream.read(8, 16)
    assert block.shape == (4, 16)
    assert np.isfinite(block).all()


def test_adapter_open_enforces_checksum_and_length(tmp_path: Path) -> None:
    manifest = _write_registry(tmp_path)
    adapter = GearboxV1Adapter(manifest, raw_dir=tmp_path / "raw")

    tampered = tmp_path / "raw" / "h30hz0.txt"
    tampered.write_text(tampered.read_text() + "1.0\t2.0\t3.0\t4.0\t\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        adapter.open("h30hz0")


def test_registry_rejects_asserted_sampling_rate(tmp_path: Path) -> None:
    manifest = _write_registry(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sampling_rate_hz"] = 10_000
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unstated sampling rate"):
        load_gearbox_manifest(manifest)


def test_signal_stream_rejects_out_of_range_reads(tmp_path: Path) -> None:
    path = tmp_path / "small.txt"
    _write_recording(path, rows=32, seed=1)
    stream = GearboxSignalStream(path)
    assert stream.info.total_samples == 32
    with pytest.raises(ValueError, match="exceeds"):
        stream.read(30, 4)


_DATASET_FREE_KERNEL_MODULES = (
    "contracts.py",
    "preprocessing.py",
    "prepared.py",
    "windows.py",
    "execution.py",
    "statistics.py",
    "audit.py",
    "reporting.py",
    "schema.py",
    "runner.py",
)

# Intentionally outside the assertion:
# - cli.py: the fetch dispatcher is known debt until downloaders are plugins;
# - demo.py: the frozen v1.0 HUST inference asset;
# - tasks.py: the legacy WindowRecord import;
# - plugins.py: _builtin_plugins is the built-in registration table (a
#   declaration, not a dispatch branch);
# - mutation.py: fault-injection scenarios build fixtures for concrete datasets
#   by design.
_KNOWN_DATASET_DEBT = ("cli.py", "demo.py", "tasks.py", "plugins.py", "mutation.py")


def test_kernel_contains_no_dataset_specific_branches() -> None:
    """RQ2: dataset integration must not leak identifiers into kernel code."""

    root = Path(__file__).resolve().parents[1] / "src" / "edgefault_bench"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        relative = str(path.relative_to(root))
        if path.parent != root or relative in _KNOWN_DATASET_DEBT:
            continue  # adapters, experiments, and documented debt are excluded
        if relative not in _DATASET_FREE_KERNEL_MODULES:
            continue
        text = path.read_text(encoding="utf-8")
        for identifier in ("gearbox", "kaist", "hust", "mehran"):
            if identifier in text.lower():
                offenders.append(f"{relative} mentions {identifier!r}")
    assert not offenders, f"kernel files carry dataset identifiers: {offenders}"
