"""In-memory tensor preparation for the pinned HUST v1 subset."""

from __future__ import annotations

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

LABELS = ("N", "I", "O", "IO")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


@dataclass(frozen=True)
class TensorTable:
    signals: Tensor
    targets: Tensor
    records: tuple[WindowRecord, ...]


def load_tensor_table(manifest_path: str | Path, raw_dir: str | Path) -> TensorTable:
    """Load verified recordings and apply independent per-window z-scoring."""

    _, files = load_hust_manifest(manifest_path)
    raw_path = Path(raw_dir)
    tensors: list[np.ndarray] = []
    for specification in sorted(files, key=lambda item: item.filename):
        path = raw_path / specification.filename
        verify_hust_file(path, specification)
        signal, _ = load_hust_signal(path)
        windows = signal.reshape(125, 4096)
        centered = windows - windows.mean(axis=1, keepdims=True)
        scale = np.maximum(centered.std(axis=1, keepdims=True), 1e-8)
        tensors.append((centered / scale).astype(np.float32))
    signals = torch.from_numpy(np.vstack(tensors)[:, None, :])
    records = build_window_records(files)
    targets = torch.tensor([LABEL_TO_INDEX[record.label] for record in records], dtype=torch.long)
    if len(signals) != len(records):
        raise RuntimeError("tensor rows and deterministic window records are misaligned")
    return TensorTable(signals=signals, targets=targets, records=records)


def encode_domains(records: tuple[WindowRecord, ...], field: str) -> Tensor:
    values = [str(record.domain_value(field)) for record in records]
    mapping = {value: index for index, value in enumerate(sorted(set(values)))}
    return torch.tensor([mapping[value] for value in values], dtype=torch.long)

