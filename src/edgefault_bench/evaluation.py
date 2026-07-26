"""Condition-aware classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


@dataclass(frozen=True)
class ConditionMetrics:
    macro_f1: float
    balanced_accuracy: float
    worst_condition_macro_f1: float
    per_condition_macro_f1: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def condition_metrics(
    truth: Sequence[str], predictions: Sequence[str], conditions: Sequence[str]
) -> ConditionMetrics:
    """Compute aggregate and worst-domain scores without hiding weak conditions."""

    truth_array = np.asarray(truth)
    prediction_array = np.asarray(predictions)
    condition_array = np.asarray(conditions)
    if truth_array.ndim != 1:
        raise ValueError("truth, predictions, and conditions must be one-dimensional")
    if not (len(truth_array) == len(prediction_array) == len(condition_array)):
        raise ValueError("truth, predictions, and conditions must have equal lengths")
    if len(truth_array) == 0:
        raise ValueError("metrics require at least one sample")

    per_condition = {
        str(condition): float(
            f1_score(
                truth_array[condition_array == condition],
                prediction_array[condition_array == condition],
                average="macro",
                zero_division=0,
            )
        )
        for condition in sorted(np.unique(condition_array))
    }
    return ConditionMetrics(
        macro_f1=float(f1_score(truth_array, prediction_array, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(truth_array, prediction_array)),
        worst_condition_macro_f1=min(per_condition.values()),
        per_condition_macro_f1=per_condition,
    )
