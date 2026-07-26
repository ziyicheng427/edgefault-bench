"""Synthetic signals for tests and smoke checks; not a benchmark dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from edgefault_bench.data import SampleIndex


@dataclass(frozen=True)
class SyntheticDataset:
    windows: NDArray[np.float64]
    labels: NDArray[np.str_]
    conditions: NDArray[np.str_]
    index: tuple[SampleIndex, ...]
    sampling_rate: float


def make_synthetic_dataset(
    *, seed: int = 17, windows_per_class: int = 24, window_length: int = 512
) -> SyntheticDataset:
    """Create condition-shifted periodic signals solely for fast pipeline validation."""

    if windows_per_class < 2 or window_length < 32:
        raise ValueError("windows_per_class must be >= 2 and window_length must be >= 32")
    generator = np.random.default_rng(seed)
    sampling_rate = 2_048.0
    time = np.arange(window_length) / sampling_rate
    condition_speeds = {"speed-low": 0.90, "speed-mid": 1.00, "speed-high": 1.12}
    class_frequencies = {"healthy": 45.0, "inner-race": 115.0, "outer-race": 175.0}

    windows: list[NDArray[np.float64]] = []
    labels: list[str] = []
    conditions: list[str] = []
    index: list[SampleIndex] = []
    for condition, speed_factor in condition_speeds.items():
        for label, base_frequency in class_frequencies.items():
            for sample_number in range(windows_per_class):
                phase = generator.uniform(0.0, 2.0 * np.pi)
                carrier = np.sin(2.0 * np.pi * base_frequency * speed_factor * time + phase)
                harmonic = 0.35 * np.sin(
                    2.0 * np.pi * 2.0 * base_frequency * speed_factor * time + phase / 2.0
                )
                noise = generator.normal(
                    0.0, 0.18 + 0.03 * (sample_number % 3), window_length
                )
                windows.append(carrier + harmonic + noise)
                labels.append(label)
                conditions.append(condition)
                index.append(
                    SampleIndex(
                        condition=condition,
                        recording=f"{condition}-{label}-{sample_number:03d}",
                        label=label,
                    )
                )
    return SyntheticDataset(
        windows=np.asarray(windows),
        labels=np.asarray(labels),
        conditions=np.asarray(conditions),
        index=tuple(index),
        sampling_rate=sampling_rate,
    )
