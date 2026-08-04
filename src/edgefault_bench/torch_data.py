"""In-memory tensor preparation for registered time-series datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from edgefault_bench.datasets.hust import (
    WindowRecord,
    build_window_records,
    load_hust_manifest,
    load_hust_signal,
    verify_hust_file,
)
from edgefault_bench.datasets.mehran import (
    MehranWindowRecord,
    build_mehran_window_records,
    load_mehran_manifest,
    load_mehran_signal,
    preprocess_mehran_windows,
    verify_mehran_file,
)

LABELS = ("N", "I", "O", "IO")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
MEHRAN_LABELS = ("inner_race", "outer_race")
WindowRecordType = WindowRecord | MehranWindowRecord


@dataclass(frozen=True)
class TensorTable:
    signals: Tensor
    targets: Tensor
    records: tuple[WindowRecordType, ...]
    labels: tuple[str, ...]


def _load_hust_tensor_table(
    manifest_path: str | Path,
    raw_dir: str | Path,
    *,
    window_length: int,
    stride: int,
    normalization: str,
) -> TensorTable:
    if normalization != "per_window_zscore":
        raise ValueError("HUST tensor loading requires per_window_zscore")

    _, files = load_hust_manifest(manifest_path)
    raw_path = Path(raw_dir)
    tensors: list[np.ndarray] = []
    for specification in sorted(files, key=lambda item: item.filename):
        path = raw_path / specification.filename
        verify_hust_file(path, specification)
        signal, _ = load_hust_signal(path)
        starts = range(0, len(signal) - window_length + 1, stride)
        windows = np.stack([signal[start : start + window_length] for start in starts])
        centered = windows - windows.mean(axis=1, keepdims=True)
        scale = np.maximum(centered.std(axis=1, keepdims=True), 1e-8)
        tensors.append((centered / scale).astype(np.float32))
    signals = torch.from_numpy(np.vstack(tensors)[:, None, :])
    records = build_window_records(files, window_length=window_length, stride=stride)
    targets = torch.tensor([LABEL_TO_INDEX[record.label] for record in records], dtype=torch.long)
    if len(signals) != len(records):
        raise RuntimeError("tensor rows and deterministic window records are misaligned")
    return TensorTable(signals=signals, targets=targets, records=records, labels=LABELS)


def _load_mehran_tensor_table(
    manifest_path: str | Path,
    raw_dir: str | Path,
    *,
    window_length: int,
    stride: int,
    normalization: str,
) -> TensorTable:
    if normalization != "per_window_channel_zscore":
        raise ValueError("Mehran tensor loading requires per_window_channel_zscore")
    _, files = load_mehran_manifest(manifest_path)
    raw_path = Path(raw_dir)
    tensors: list[np.ndarray] = []
    sample_counts: dict[str, int] = {}
    for specification in sorted(files, key=lambda item: item.filename):
        path = raw_path / specification.filename
        verify_mehran_file(path, specification)
        signal = load_mehran_signal(path, minimum_samples=window_length)
        sample_counts[specification.filename] = len(signal)
        tensors.append(
            preprocess_mehran_windows(
                signal, window_length=window_length, stride=stride
            )
        )
    signals = torch.from_numpy(np.vstack(tensors))
    records = build_mehran_window_records(
        files,
        sample_counts,
        window_length=window_length,
        stride=stride,
    )
    label_to_index = {label: index for index, label in enumerate(MEHRAN_LABELS)}
    targets = torch.tensor(
        [label_to_index[record.label] for record in records], dtype=torch.long
    )
    if len(signals) != len(records):
        raise RuntimeError("tensor rows and Mehran window records are misaligned")
    return TensorTable(
        signals=signals,
        targets=targets,
        records=records,
        labels=MEHRAN_LABELS,
    )


def load_tensor_table(
    manifest_path: str | Path,
    raw_dir: str | Path,
    *,
    window_length: int = 4096,
    stride: int = 4096,
    normalization: str | None = None,
) -> TensorTable:
    """Dispatch verified recordings to a dataset-specific tensor adapter."""

    manifest = Path(manifest_path)
    dataset_id = json.loads(manifest.read_text(encoding="utf-8")).get("dataset_id")
    if dataset_id == "hust-bearing-v3":
        return _load_hust_tensor_table(
            manifest,
            raw_dir,
            window_length=window_length,
            stride=stride,
            normalization=normalization or "per_window_zscore",
        )
    if dataset_id == "mehran-triaxial-bearing-v2":
        return _load_mehran_tensor_table(
            manifest,
            raw_dir,
            window_length=window_length,
            stride=stride,
            normalization=normalization or "per_window_channel_zscore",
        )
    raise ValueError(f"unsupported tensor dataset: {dataset_id!r}")


def encode_domains(records: tuple[WindowRecordType, ...], field: str) -> Tensor:
    values = [str(record.domain_value(field)) for record in records]
    mapping = {value: index for index, value in enumerate(sorted(set(values)))}
    return torch.tensor([mapping[value] for value in values], dtype=torch.long)
