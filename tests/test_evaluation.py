import pytest

from edgefault_bench.evaluation import condition_metrics


def test_worst_condition_is_reported() -> None:
    metrics = condition_metrics(
        truth=["a", "b", "a", "b"],
        predictions=["a", "b", "a", "a"],
        conditions=["easy", "easy", "hard", "hard"],
    )
    assert metrics.per_condition_macro_f1["easy"] == pytest.approx(1.0)
    assert metrics.worst_condition_macro_f1 == metrics.per_condition_macro_f1["hard"]


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        condition_metrics(["a"], ["a", "b"], ["condition"])

