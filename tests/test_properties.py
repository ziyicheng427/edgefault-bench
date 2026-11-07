"""Property-based tests (T10): generated valid/invalid artefacts vs. oracle outcome.

Each property pairs a Hypothesis-generated input with an independent oracle that
recomputes the expected accept/reject decision from the invariant definitions in
``docs/formal-spec-draft.md``. The system under test must agree with the oracle
on every generated case.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edgefault_bench.audit import audit_recordings
from edgefault_bench.contracts import (
    DatasetMetadata,
    Recording,
    TaskSpec,
    WindowSpec,
)
from edgefault_bench.execution import TrainingPolicy
from edgefault_bench.plugins import ModelPlugin
from edgefault_bench.prepared import (
    PreparedFeatureTable,
    PreparedSample,
    load_prepared_feature_table,
    write_prepared_feature_table,
)
from edgefault_bench.preprocessing import WindowBatch, WindowSample, partition_window_indices
from edgefault_bench.reporting import load_result

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

LABELS = ("healthy", "fault")
PARTITION_VALUES = (0, 100, 200)

_METADATA = DatasetMetadata(
    dataset_id="prop-v1",
    version="1",
    title="property fixture",
    license_spdx="CC0-1.0",
    source_url="https://example.org",
    domain_fields=("load", "device"),
)


def _task(partitions: dict[str, list[int]], *, labels: tuple[str, ...] = LABELS) -> TaskSpec:
    return TaskSpec(
        task_id="prop-task-v1",
        dataset_id="prop-v1",
        domain_field="load",
        evaluation_group_field="device",
        partitions=partitions,
        labels=labels,
        window=WindowSpec(length=4, stride=4, normalization="none"),
        seeds=(17, 29),
        description="property test",
    )


@st.composite
def _partition_lists(draw: st.DrawFn) -> tuple[list[int], list[int], list[int]]:
    pool = [0, 1, 2, 3, 4, 5, 6, 7]
    train = draw(st.lists(st.sampled_from(pool), min_size=1, max_size=3, unique=True))
    remaining = [value for value in pool if value not in train]
    validation = draw(st.lists(st.sampled_from(remaining), min_size=1, max_size=3, unique=True))
    remaining = [value for value in remaining if value not in validation]
    test = draw(st.lists(st.sampled_from(remaining), min_size=1, max_size=3, unique=True))
    return train, validation, test


def _partition_lists_overlap(
    train: list[int], validation: list[int], test: list[int]
) -> bool:
    return bool(
        set(train) & set(validation)
        or set(train) & set(test)
        or set(validation) & set(test)
    )


@SETTINGS
@given(partitions=_partition_lists())
def test_task_spec_rejects_partitions_iff_overlapping(partitions) -> None:
    train, validation, test = partitions
    overlap = _partition_lists_overlap(train, validation, test)
    try:
        spec = _task({"train": train, "validation": validation, "test": test})
    except ValueError:
        assert overlap, "TaskSpec rejected disjoint partitions"
        return
    assert not overlap, "TaskSpec accepted overlapping partitions"
    assert spec.partition_for(train[0]) == "train"


@st.composite
def _fault_kinds(draw: st.DrawFn) -> tuple[str, bool]:
    """Draw a fault injection flag: none, duplicate id, bad label, unassigned load."""

    return draw(
        st.sampled_from(
            [
                ("none", False),
                ("duplicate-id", True),
                ("bad-label", True),
                ("unassigned-load", True),
            ]
        )
    )


def _recordings_with_fault(fault: str) -> list[Recording]:
    records = [
        Recording(
            dataset_id="prop-v1",
            recording_id=f"rec-{value}",
            source_file=f"rec-{value}.csv",
            label=LABELS[index % 2],
            domains={"load": value, "device": f"dev-{value}"},
        )
        for index, value in enumerate(PARTITION_VALUES)
    ]
    if fault == "duplicate-id":
        records.append(replace(records[0]))
    elif fault == "bad-label":
        records.append(
            Recording(
                dataset_id="prop-v1",
                recording_id="rec-extra",
                source_file="rec-extra.csv",
                label="unknown-label",
                domains={"load": 0, "device": "dev-0"},
            )
        )
    elif fault == "unassigned-load":
        records.append(
            Recording(
                dataset_id="prop-v1",
                recording_id="rec-extra",
                source_file="rec-extra.csv",
                label=LABELS[0],
                domains={"load": 999, "device": "dev-999"},
            )
        )
    return records


@SETTINGS
@given(fault=_fault_kinds())
def test_recording_audit_matches_independent_oracle(fault) -> None:
    fault_kind, _ = fault
    task = _task(
        {"train": [0], "validation": [100], "test": [200]},
    )
    records = _recordings_with_fault(fault_kind)
    report = audit_recordings(records, task)

    # Oracle derived directly from INV-2 and its audit preconditions.
    ids = [record.recording_id for record in records]
    labels_ok = all(record.label in task.labels for record in records)
    loads_assigned = all(
        task.partition_for(record.domains["load"]) is not None for record in records
    )
    expected = len(set(ids)) == len(ids) and labels_ok and loads_assigned
    assert report.passed is expected, (
        f"audit passed={report.passed} but oracle says {expected} for fault={fault_kind}"
    )
    if not expected:
        try:
            report.require_pass()
        except ValueError as error:
            assert "leakage audit failed" in str(error)
        else:  # pragma: no cover - unreachable on failure
            raise AssertionError("require_pass did not fail on a violated invariant")


@st.composite
def _window_batches(draw: st.DrawFn) -> tuple[WindowBatch, TaskSpec, int]:
    # At least two windows per recording: relabelling one window must then make
    # its recording span partitions. Single-window recordings are invisible to
    # the window-level check alone and rely on the recording-level cross audit.
    windows_per_recording = draw(st.integers(min_value=2, max_value=3))
    samples: list[WindowSample] = []
    signals = []
    for position, value in enumerate(PARTITION_VALUES):
        label = LABELS[position % 2]
        for window_index in range(windows_per_recording):
            samples.append(
                WindowSample(
                    window_id=f"rec-{value}#s{window_index:03d}",
                    recording_id=f"rec-{value}",
                    start_sample=window_index * 4,
                    end_sample=(window_index + 1) * 4,
                    label=label,
                    domains={"load": value, "device": f"dev-{value}"},
                )
            )
            signals.append(np.zeros((2, 4)))
    batch = WindowBatch(
        samples=tuple(samples),
        signals=np.stack(signals),
        channels=("ch0", "ch1"),
        sample_rate_hz=64.0,
    )
    task = _task({"train": [0], "validation": [100], "test": [200]})
    return batch, task, windows_per_recording


@SETTINGS
@given(case=_window_batches())
def test_window_partitioning_is_total_and_recording_exclusive(case) -> None:
    batch, task, _ = case
    indices = partition_window_indices(batch, task)

    assigned = [index for values in indices.values() for index in values]
    assert sorted(assigned) == list(range(len(batch)))
    recording_partition = {}
    for name, values in indices.items():
        for index in values:
            recording = batch.samples[index].recording_id
            recording_partition.setdefault(recording, name)
            assert recording_partition[recording] == name

    # Any single window moved to another partition's value makes its recording
    # span partitions and must be rejected (INV-2 at window level).
    victim = 0 if len(indices["train"]) > 1 else len(indices["train"])
    moved = replace(
        batch.samples[victim],
        domains={"load": 200, "device": batch.samples[victim].domains["device"]},
    )
    tampered_samples = list(batch.samples)
    tampered_samples[victim] = moved
    tampered = WindowBatch(
        samples=tuple(tampered_samples),
        signals=batch.signals,
        channels=batch.channels,
        sample_rate_hz=batch.sample_rate_hz,
    )
    try:
        partition_window_indices(tampered, task)
    except ValueError as error:
        assert "spans partitions" in str(error) or "unassigned" in str(error)
    else:
        raise AssertionError("window relabelling was not rejected")


@st.composite
def _tables(draw: st.DrawFn) -> tuple[PreparedFeatureTable, TaskSpec]:
    rows = draw(st.integers(min_value=3, max_value=8))
    width = draw(st.integers(min_value=1, max_value=4))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=99)))
    samples = tuple(
        PreparedSample(
            recording_id=f"rec-{PARTITION_VALUES[index % 3]}",
            label=LABELS[index % 2],
            domains={"load": PARTITION_VALUES[index % 3], "device": "dev"},
            features=tuple(float(value) for value in rng.normal(size=width)),
            window_id=f"rec-{PARTITION_VALUES[index % 3]}#s{index:03d}",
            start_sample=index * 4,
            end_sample=(index + 1) * 4,
        )
        for index in range(rows)
    )
    table = PreparedFeatureTable(
        dataset_id="prop-v1",
        feature_names=tuple(f"f{index}" for index in range(width)),
        samples=samples,
        preparation={"kind": "property"},
    )
    return table, _task({"train": [0], "validation": [100], "test": [200]})


@SETTINGS
@given(case=_tables())
def test_prepared_table_round_trip_preserves_identity(case) -> None:
    import tempfile

    table, _task_spec = case
    with tempfile.TemporaryDirectory() as directory:
        path = write_prepared_feature_table(table, Path(directory) / "table.json")
        reloaded = load_prepared_feature_table(path)

    assert reloaded.schema_version == 2
    assert reloaded.dataset_id == table.dataset_id
    assert reloaded.feature_names == table.feature_names
    assert [sample.window_id for sample in reloaded.samples] == [
        sample.window_id for sample in table.samples
    ]
    assert all(sample.has_window_identity for sample in reloaded.samples)


@SETTINGS
@given(case=_tables())
def test_partial_or_duplicate_window_identity_is_rejected(case) -> None:
    table, _ = case
    # Duplicate window identity.
    duplicated = list(table.samples)
    duplicated[1] = replace(duplicated[1], window_id=duplicated[0].window_id)
    try:
        PreparedFeatureTable(
            dataset_id=table.dataset_id,
            feature_names=table.feature_names,
            samples=tuple(duplicated),
            preparation=table.preparation,
        )
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate window_id accepted")

    # Partial identity on one row defeats the audit and must be rejected.
    partial = list(table.samples)
    try:
        partial[0] = replace(partial[0], window_id=None)  # type: ignore[arg-type]
        PreparedFeatureTable(
            dataset_id=table.dataset_id,
            feature_names=table.feature_names,
            samples=tuple(partial),
            preparation=table.preparation,
        )
    except ValueError as error:
        assert "together" in str(error)
    else:
        raise AssertionError("partial window identity accepted")


@SETTINGS
@given(
    seeds=st.lists(st.sampled_from([17, 29, 43]), min_size=0, max_size=5),
)
def test_result_seed_bijection_matches_oracle(seeds: list[int]) -> None:
    registered = [17, 29]
    payload = {
        "schema_version": 2,
        "task_id": "prop-task-v1",
        "model_id": "prop-model",
        "git_commit": "abcdef1234567890",
        "environment": {"python": "3.11"},
        "protocol": {"task_seeds": registered},
        "model_plugin": {"backend": "sklearn"},
        "input_artifacts": {},
        "runs": [
            {"seed": seed, "status": "failed", "error": "synthetic"} for seed in seeds
        ],
    }
    path = Path("/tmp/prop-result.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    bijection = seeds == registered
    try:
        load_result(path)
    except ValueError:
        assert not bijection, "valid bijection rejected"
    else:
        assert bijection, "broken bijection accepted"


@SETTINGS
@given(
    optimizer=st.sampled_from(["none", "adam", "sgd", "adagrad", "lbfgs"]),
    lr=st.floats(min_value=0.0, max_value=1.0),
    epochs=st.integers(min_value=-2, max_value=30),
    patience=st.integers(min_value=-1, max_value=9),
)
def test_training_policy_validation_matches_oracle(
    optimizer: str, lr: float, epochs: int, patience: int
) -> None:
    valid = (
        optimizer in {"none", "adam", "sgd"}
        and lr > 0
        and epochs >= 0
        and patience > 0
    )
    try:
        policy = TrainingPolicy(
            optimizer=optimizer, learning_rate=lr, max_epochs=epochs, patience=patience
        )
    except ValueError:
        assert not valid
        return
    assert valid
    assert TrainingPolicy.from_dict(policy.to_dict()) == policy


@SETTINGS
@given(
    model_id=st.sampled_from(["", "  ", "good-model"]),
    backend=st.sampled_from(["sklearn", "pytorch", "jax", ""]),
)
def test_model_plugin_contract_rejects_invalid_declarations(
    model_id: str, backend: str
) -> None:
    valid = bool(model_id.strip()) and backend in {"sklearn", "pytorch"}
    try:
        ModelPlugin(model_id=model_id, backend=backend, source="s", factory=dict)
    except (ValueError, TypeError):
        assert not valid
    else:
        assert valid
