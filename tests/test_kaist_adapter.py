"""KAIST acoustic adapter tests (T14): Test.Lab parsing and registry checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from edgefault_bench.datasets.kaist import (
    KaistAcousticV6Adapter,
    load_kaist_acoustic_signal,
    load_kaist_manifest,
)

SAMPLES = 4_096
RATE = 51_200.0
CONDITIONS = (
    ("0Nm_Normal.mat", "normal", "Normal", None),
    ("0Nm_BPFI_03.mat", "bpfi_03", "Inner", 3),
    ("0Nm_BPFI_10.mat", "bpfi_10", "Inner", 10),
    ("0Nm_BPFO_03.mat", "bpfo_03", "Outer", 3),
    ("0Nm_BPFO_10.mat", "bpfo_10", "Outer", 10),
)


def _write_testlab_mat(path: Path, *, samples: int = SAMPLES, seed: int = 5) -> None:
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=0.05, size=samples)
    savemat(
        path,
        {
            "Signal": {
                "y_values": {"values": values},
                "x_values": {
                    "increment": 1.0 / RATE,
                    "number_of_values": samples,
                },
            }
        },
    )


def _write_registry(root: Path) -> tuple[Path, Path]:
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    entries = []
    for filename, label, family, severity in CONDITIONS:
        path = raw_dir / filename
        _write_testlab_mat(path, seed=len(label))
        entries.append(
            {
                "id": filename.removesuffix(".mat"),
                "recording_id": filename.removesuffix(".mat"),
                "filename": filename,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sample_count": SAMPLES,
                "label": label,
                "fault_family": family,
                "severity_mm": severity,
                "load_nm": 0,
                "selected": True,
                "exclusion_reason": None,
            }
        )
    manifest = root / "kaist.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "kaist-rotating-acoustic-v6",
                "title": "fixture",
                "landing_page": "https://example.org",
                "doi": "10.17632/ztmf3m7h5x.6",
                "version": 6,
                "protocol_status": "accepted",
                "license": {"spdx": "CC-BY-4.0"},
                "sampling_rate_hz": RATE,
                "channels": ["acoustic_pa"],
                "archive": {
                    "filename": "acoustic.zip",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                    "url": "https://example.org/acoustic.zip",
                },
                "files": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest, raw_dir


def test_loads_testlab_signal_with_declared_rate(tmp_path: Path) -> None:
    path = tmp_path / "sample.mat"
    _write_testlab_mat(path, seed=17)
    signal = load_kaist_acoustic_signal(path)
    assert signal.shape == (1, SAMPLES)
    assert np.isfinite(signal).all()

    wrong = tmp_path / "wrong.mat"
    savemat(
        wrong,
        {
            "Signal": {
                "y_values": {"values": np.zeros(SAMPLES)},
                "x_values": {"increment": 1.0 / 1_000.0, "number_of_values": SAMPLES},
            }
        },
    )
    with pytest.raises(ValueError, match="contradicts"):
        load_kaist_acoustic_signal(wrong)


def test_adapter_exposes_five_conditions_and_streams(tmp_path: Path) -> None:
    manifest, raw_dir = _write_registry(tmp_path)
    adapter = KaistAcousticV6Adapter(manifest, raw_dir=raw_dir)
    recordings = adapter.recordings()
    assert len(recordings) == 5
    by_id = {record.recording_id: record for record in recordings}
    assert by_id["0Nm_Normal"].label == "normal"
    assert by_id["0Nm_BPFI_10"].domains == {
        "load_nm": 0,
        "fault_family": "Inner",
        "severity_mm": 10,
    }
    assert by_id["0Nm_Normal"].sample_rate_hz == RATE

    stream = adapter.open("0Nm_BPFO_03")
    assert stream.info.channels == ("acoustic_pa",)
    assert stream.info.units == ("Pa",)
    block = stream.read(128, 256)
    assert block.shape == (1, 256)
    with pytest.raises(ValueError, match="exceeds"):
        stream.read(SAMPLES - 1, 2)


def test_registry_rejects_incomplete_condition_grid(tmp_path: Path) -> None:
    manifest, _ = _write_registry(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"] = payload["files"][1:] + [payload["files"][0]]
    payload["files"][0] = dict(payload["files"][0], label="unexpected")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="five-condition grid"):
        load_kaist_manifest(manifest)


def test_adapter_open_enforces_checksum(tmp_path: Path) -> None:
    manifest, raw_dir = _write_registry(tmp_path)
    adapter = KaistAcousticV6Adapter(manifest, raw_dir=raw_dir)
    target = raw_dir / "0Nm_Normal.mat"
    _write_testlab_mat(target, samples=SAMPLES, seed=99)  # different bytes
    with pytest.raises(ValueError, match="mismatch"):
        adapter.open("0Nm_Normal")


def test_vibration_registry_normalizes_misspelled_unbalance(tmp_path: Path) -> None:
    """T14 vibration subset: 3x5 grid, label normalization, four channels."""

    import hashlib

    from scipy.io import savemat

    from edgefault_bench.datasets.kaist import (
        KaistVibrationV6Adapter,
        load_kaist_vibration_manifest,
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    entries = []
    families = {
        "Normal": ("normal", None),
        "BPFI_03": ("bpfi", "03"),
        "BPFI_10": ("bpfi", "10"),
        "BPFI_30": ("bpfi", "30"),
        "BPFO_03": ("bpfo", "03"),
        "BPFO_10": ("bpfo", "10"),
        "BPFO_30": ("bpfo", "30"),
        "Misalign_01": ("misalign", "01"),
        "Misalign_03": ("misalign", "03"),
        "Misalign_05": ("misalign", "05"),
        "Unbalance_0583mg": ("unbalance", "0583mg"),
        "Unbalance_1169mg": ("unbalance", "1169mg"),
        "Unbalance_1751mg": ("unbalance", "1751mg"),
        "Unbalance_2239mg": ("unbalance", "2239mg"),
        "Unbalance_3318mg": ("unbalance", "3318mg"),
    }
    for load in (0, 2, 4):
        for suffix, (label, severity) in families.items():
            # Alternate the upstream misspelling on the 2 N·m unbalance files.
            stem = f"{load}Nm_{suffix}"
            if suffix == "Normal":
                stem = f"{load}Nm_Normal"
            elif label == "unbalance" and load == 2:
                stem = f"{load}Nm_Unbalalnce_{severity}"
            filename = f"{stem}.mat"
            path = raw_dir / filename
            samples = 256
            savemat(
                path,
                {
                    "Signal": {
                        "y_values": {"values": np.zeros((samples, 4))},
                        "x_values": {
                            "increment": 1.0 / 25_600.0,
                            "number_of_values": samples,
                        },
                    }
                },
            )
            entries.append(
                {
                    "id": stem,
                    "recording_id": stem,
                    "filename": filename,
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "sample_count": samples,
                    "label": label,
                    "load_nm": load,
                    "severity": severity or "none",
                    "selected": True,
                    "exclusion_reason": None,
                }
            )
    manifest = tmp_path / "vibration.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "kaist-rotating-vibration-v6",
                "title": "fixture",
                "landing_page": "https://example.org",
                "doi": "10.17632/ztmf3m7h5x.6",
                "version": 6,
                "license": {"spdx": "CC-BY-4.0"},
                "sampling_rate_hz": 25600,
                "archive": {
                    "filename": "vibration.zip",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                    "url": "https://example.org",
                },
                "files": entries,
            }
        ),
        encoding="utf-8",
    )

    payload, files = load_kaist_vibration_manifest(manifest)
    assert len(files) == 45  # full 3 loads x 15 conditions, tiny signals
    adapter = KaistVibrationV6Adapter(manifest, raw_dir=raw_dir)
    recordings = adapter.recordings()
    misspelled = [r for r in recordings if "Unbalalnce" in r.recording_id]
    assert misspelled and all(r.label == "unbalance" for r in misspelled)

    stream = adapter.open("2Nm_Unbalalnce_0583mg")
    assert stream.info.channels == ("vib_ax_g", "vib_ay_g", "vib_bx_g", "vib_by_g")
    assert stream.info.sample_rate_hz == 25600.0
    assert stream.read(8, 16).shape == (4, 16)
