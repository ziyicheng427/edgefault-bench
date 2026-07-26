"""Leakage-resistant condition-held-out split construction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SampleIndex:
    """Metadata needed to assign windows without inspecting their signal values."""

    condition: str
    recording: str
    label: str


@dataclass(frozen=True)
class ConditionSplit:
    """Indices for mutually exclusive train, validation, and test partitions."""

    train: NDArray[np.int64]
    validation: NDArray[np.int64]
    test: NDArray[np.int64]

    def as_dict(self) -> dict[str, NDArray[np.int64]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def _normalise_conditions(values: Iterable[str], name: str) -> frozenset[str]:
    result = frozenset(values)
    if not result:
        raise ValueError(f"{name} must contain at least one operating condition")
    if any(not value.strip() for value in result):
        raise ValueError(f"{name} contains an empty operating condition")
    return result


def make_condition_split(
    samples: Sequence[SampleIndex],
    *,
    train_conditions: Iterable[str],
    validation_conditions: Iterable[str],
    test_conditions: Iterable[str],
) -> ConditionSplit:
    """Assign samples using explicit and disjoint operating-condition sets.

    Every sample must be assigned exactly once. A recording identifier may occur in only
    one partition, which prevents overlapping windows from the same recording from leaking
    across the experimental boundary.
    """

    if not samples:
        raise ValueError("samples must not be empty")

    condition_sets = {
        "train": _normalise_conditions(train_conditions, "train_conditions"),
        "validation": _normalise_conditions(validation_conditions, "validation_conditions"),
        "test": _normalise_conditions(test_conditions, "test_conditions"),
    }
    names = tuple(condition_sets)
    for position, left_name in enumerate(names):
        for right_name in names[position + 1 :]:
            overlap = condition_sets[left_name] & condition_sets[right_name]
            if overlap:
                raise ValueError(
                    f"condition sets {left_name!r} and {right_name!r} overlap: {sorted(overlap)}"
                )

    all_allowed = frozenset().union(*condition_sets.values())
    unknown = sorted({sample.condition for sample in samples} - all_allowed)
    if unknown:
        raise ValueError(f"samples contain unassigned conditions: {unknown}")

    partitions: dict[str, list[int]] = {name: [] for name in names}
    recording_partition: dict[str, str] = {}
    for index, sample in enumerate(samples):
        partition = next(
            name for name, conditions in condition_sets.items() if sample.condition in conditions
        )
        previous = recording_partition.setdefault(sample.recording, partition)
        if previous != partition:
            raise ValueError(
                f"recording {sample.recording!r} spans {previous!r} and {partition!r} partitions"
            )
        partitions[partition].append(index)

    empty = [name for name, indices in partitions.items() if not indices]
    if empty:
        raise ValueError(f"split produced empty partitions: {empty}")

    return ConditionSplit(
        train=np.asarray(partitions["train"], dtype=np.int64),
        validation=np.asarray(partitions["validation"], dtype=np.int64),
        test=np.asarray(partitions["test"], dtype=np.int64),
    )
