"""Group-aware statistics for condition-shift evaluation (T15).

Windows cut from one recording are not independent samples, so a plain
window-level mean overstates the effective sample size. This module resamples
whole groups (recordings or evaluation groups) with replacement, reports
percentile confidence intervals, and compares two models with a paired
group-level bootstrap on the shared test samples.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score

Metric = Callable[[np.ndarray, np.ndarray], float]

DEFAULT_RESAMPLES = 2_000
DEFAULT_CONFIDENCE = 0.95


def macro_f1(truth: Sequence[str], predictions: Sequence[str]) -> float:
    """The primary metric used throughout the benchmark protocols."""

    return float(
        f1_score(
            np.asarray(truth, dtype=str),
            np.asarray(predictions, dtype=str),
            average="macro",
            zero_division=0,
        )
    )


def _unit_indices(units: Sequence[str]) -> dict[str, np.ndarray]:
    unit_array = np.asarray(units, dtype=str)
    return {
        str(unit): np.flatnonzero(unit_array == unit)
        for unit in np.unique(unit_array)
    }


@dataclass(frozen=True)
class BootstrapInterval:
    """Point estimate with a percentile interval over resampled groups."""

    metric: str
    unit: str
    point: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    unit_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "n_resamples": self.n_resamples,
            "unit_count": self.unit_count,
        }


def group_bootstrap(
    truth: Sequence[str],
    predictions: Sequence[str],
    units: Sequence[str],
    *,
    metric: Metric = macro_f1,
    metric_name: str = "macro_f1",
    unit: str = "recording",
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 17,
) -> BootstrapInterval:
    """Bootstrap a metric by resampling whole groups with replacement.

    ``units`` assigns every sample to a resampling group (for example its
    recording identifier). Each resample draws ``len(groups)`` groups with
    replacement, pools their samples, and recomputes the metric.
    """

    truth_array = np.asarray(truth, dtype=str)
    prediction_array = np.asarray(predictions, dtype=str)
    if not (len(truth_array) == len(prediction_array) == len(units)):
        raise ValueError("truth, predictions, and units must have equal lengths")
    if len(truth_array) == 0:
        raise ValueError("statistics require at least one sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    groups = _unit_indices(units)
    if len(groups) < 2:
        raise ValueError("group bootstrap requires at least two groups")

    point = metric(truth_array, prediction_array)
    rng = np.random.default_rng(seed)
    unit_names = sorted(groups)
    statistics = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        picked = rng.choice(unit_names, size=len(unit_names), replace=True)
        pooled = np.concatenate([groups[name] for name in picked])
        statistics[index] = metric(truth_array[pooled], prediction_array[pooled])

    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        metric=metric_name,
        unit=unit,
        point=point,
        ci_low=float(np.quantile(statistics, alpha)),
        ci_high=float(np.quantile(statistics, 1.0 - alpha)),
        confidence=confidence,
        n_resamples=n_resamples,
        unit_count=len(groups),
    )


@dataclass(frozen=True)
class PairedComparison:
    """Paired group-level comparison of two models on the shared test set."""

    delta_point: float
    delta_ci_low: float
    delta_ci_high: float
    confidence: float
    n_resamples: int
    unit_count: int
    fraction_favouring_a: float

    def to_dict(self) -> dict[str, object]:
        return {
            "delta_point": self.delta_point,
            "delta_ci_low": self.delta_ci_low,
            "delta_ci_high": self.delta_ci_high,
            "confidence": self.confidence,
            "n_resamples": self.n_resamples,
            "unit_count": self.unit_count,
            "fraction_favouring_a": self.fraction_favouring_a,
        }


def paired_group_bootstrap(
    truth: Sequence[str],
    predictions_a: Sequence[str],
    predictions_b: Sequence[str],
    units: Sequence[str],
    *,
    metric: Metric = macro_f1,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 17,
) -> PairedComparison:
    """Bootstrap the metric difference of two models over shared groups.

    ``fraction_favouring_a`` is the fraction of resamples whose difference
    favours model A; it summarizes the resampling distribution rather than a
    parametric p-value.
    """

    truth_array = np.asarray(truth, dtype=str)
    a_array = np.asarray(predictions_a, dtype=str)
    b_array = np.asarray(predictions_b, dtype=str)
    if not (len(truth_array) == len(a_array) == len(b_array) == len(units)):
        raise ValueError("truth, both prediction sets, and units must align")
    groups = _unit_indices(units)
    if len(groups) < 2:
        raise ValueError("paired bootstrap requires at least two groups")

    delta_point = metric(truth_array, a_array) - metric(truth_array, b_array)
    rng = np.random.default_rng(seed)
    unit_names = sorted(groups)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        picked = rng.choice(unit_names, size=len(unit_names), replace=True)
        pooled = np.concatenate([groups[name] for name in picked])
        deltas[index] = metric(truth_array[pooled], a_array[pooled]) - metric(
            truth_array[pooled], b_array[pooled]
        )

    alpha = (1.0 - confidence) / 2.0
    return PairedComparison(
        delta_point=float(delta_point),
        delta_ci_low=float(np.quantile(deltas, alpha)),
        delta_ci_high=float(np.quantile(deltas, 1.0 - alpha)),
        confidence=confidence,
        n_resamples=n_resamples,
        unit_count=len(groups),
        fraction_favouring_a=float(np.mean(deltas > 0.0)),
    )


def _run_predictions(
    run: Mapping[str, object],
) -> tuple[list[str], list[str], list[str], list[str]]:
    retained = run.get("test_predictions")
    if not isinstance(retained, Mapping):
        raise ValueError(
            "run does not retain test_predictions; rerun with a 1.2 runner to "
            "enable group statistics"
        )
    return (
        [str(value) for value in retained["truth"]],
        [str(value) for value in retained["predictions"]],
        [str(value) for value in retained["groups"]],
        [str(value) for value in retained["recording_ids"]],
    )


def result_bundle_statistics(
    payload: Mapping[str, object],
    *,
    unit: str = "recording",
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 17,
) -> dict[str, object]:
    """Window-level and group-bootstrap statistics for every seed of a bundle."""

    per_seed = []
    for run in payload["runs"]:
        if run.get("status") != "ok":
            continue
        truth, predictions, groups, recordings = _run_predictions(run)
        resampling_units = recordings if unit == "recording" else groups
        interval = group_bootstrap(
            truth,
            predictions,
            resampling_units,
            unit=unit,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed + int(run["seed"]),
        )
        per_seed.append(
            {
                "seed": run["seed"],
                "window_level_macro_f1": macro_f1(truth, predictions),
                "group_bootstrap": interval.to_dict(),
            }
        )
    if not per_seed:
        raise ValueError("bundle contains no successful runs with retained predictions")
    return {
        "task_id": payload.get("task_id"),
        "model_id": payload.get("model_id"),
        "unit": unit,
        "per_seed": per_seed,
        "mean_group_bootstrap_point": float(
            np.mean([entry["group_bootstrap"]["point"] for entry in per_seed])
        ),
    }


def compare_bundles(
    payload_a: Mapping[str, object],
    payload_b: Mapping[str, object],
    *,
    unit: str = "recording",
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 17,
) -> dict[str, object]:
    """Paired per-seed comparison of two result bundles on the same task."""

    if payload_a.get("task_id") != payload_b.get("task_id"):
        raise ValueError("paired comparison requires the same task on both sides")
    runs_b = {run["seed"]: run for run in payload_b["runs"]}
    comparisons = []
    for run in payload_a["runs"]:
        if run.get("status") != "ok":
            continue
        counterpart = runs_b.get(run["seed"])
        if counterpart is None or counterpart.get("status") != "ok":
            continue
        truth, predictions_a, _, recordings_a = _run_predictions(run)
        truth_b, predictions_b, _, recordings_b = _run_predictions(counterpart)
        if truth != truth_b or recordings_a != recordings_b:
            raise ValueError(
                f"seed {run['seed']} test samples differ between the bundles; "
                "paired comparison requires identical test sets"
            )
        resampling_units = recordings_a if unit == "recording" else _run_predictions(run)[2]
        paired = paired_group_bootstrap(
            truth,
            predictions_a,
            predictions_b,
            resampling_units,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed + int(run["seed"]),
        )
        comparisons.append({"seed": run["seed"], "comparison": paired.to_dict()})
    if not comparisons:
        raise ValueError("no seed with successful runs on both sides")
    return {
        "task_id": payload_a.get("task_id"),
        "model_a": payload_a.get("model_id"),
        "model_b": payload_b.get("model_id"),
        "unit": unit,
        "per_seed": comparisons,
    }
