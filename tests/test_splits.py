import pytest

from edgefault_bench.data import SampleIndex, make_condition_split


def test_condition_split_assigns_every_sample_once() -> None:
    samples = tuple(
        SampleIndex(condition=condition, recording=f"recording-{condition}", label="healthy")
        for condition in ("load-1", "load-2", "load-3")
    )
    split = make_condition_split(
        samples,
        train_conditions={"load-1"},
        validation_conditions={"load-2"},
        test_conditions={"load-3"},
    )
    assigned = set(split.train) | set(split.validation) | set(split.test)
    assert assigned == set(range(len(samples)))


def test_condition_split_rejects_overlapping_conditions() -> None:
    samples = (SampleIndex(condition="load-1", recording="r1", label="healthy"),)
    with pytest.raises(ValueError, match="overlap"):
        make_condition_split(
            samples,
            train_conditions={"load-1"},
            validation_conditions={"load-1"},
            test_conditions={"load-2"},
        )


def test_condition_split_rejects_recording_leakage() -> None:
    samples = (
        SampleIndex(condition="load-1", recording="shared", label="healthy"),
        SampleIndex(condition="load-2", recording="shared", label="healthy"),
        SampleIndex(condition="load-3", recording="other", label="healthy"),
    )
    with pytest.raises(ValueError, match="spans"):
        make_condition_split(
            samples,
            train_conditions={"load-1"},
            validation_conditions={"load-2"},
            test_conditions={"load-3"},
        )

