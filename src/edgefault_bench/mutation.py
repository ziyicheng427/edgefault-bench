"""Fault-injection catalogue for integrity verification (T11, RQ1 evidence).

Each mutation plants one defect from ``docs/formal-spec-draft.md`` §6 into a
real fixture and runs the genuine kernel mechanism against it. A mutation is
*detected* only when the system rejects the artefact fail-closed (or the hash
check reports a mismatch); silent acceptance marks the catalogue entry — and
therefore the design — as failing. ``edgefault mutation audit`` renders the
detection matrix used as RQ1 evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from edgefault_bench.audit import audit_recordings
from edgefault_bench.contracts import (
    DatasetMetadata,
    Recording,
    TaskSpec,
    WindowSpec,
)
from edgefault_bench.execution import run_benchmark
from edgefault_bench.plugins import (
    ModelPlugin,
    discover_model_plugins,
    validate_model_implementation,
)
from edgefault_bench.prepared import (
    PreparedFeatureTable,
    PreparedSample,
    prepare_task_data,
    write_prepared_feature_table,
)
from edgefault_bench.preprocessing import (
    ChannelStandardization,
    WindowBatch,
    WindowSample,
    fit_preprocessor,
    training_view,
)
from edgefault_bench.reporting import load_result, verify_input_artifact_hashes

LABELS = ("healthy", "fault")


@dataclass(frozen=True)
class MutationOutcome:
    """One planted defect and whether the kernel caught it."""

    mutation_id: str
    target_invariant: str
    description: str
    detected: bool
    mechanism: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mutation_id": self.mutation_id,
            "target_invariant": self.target_invariant,
            "description": self.description,
            "detected": self.detected,
            "mechanism": self.mechanism,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _metadata() -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id="mutation-v1",
        version="1",
        title="mutation fixture",
        license_spdx="CC0-1.0",
        source_url="https://example.org",
        domain_fields=("load", "device"),
    )


def _task(seeds: tuple[int, ...] = (17, 29)) -> TaskSpec:
    return TaskSpec(
        task_id="mutation-load-v1",
        dataset_id="mutation-v1",
        domain_field="load",
        evaluation_group_field="device",
        partitions={"train": [0], "validation": [100], "test": [200]},
        labels=LABELS,
        window=WindowSpec(length=8, stride=8, normalization="none"),
        seeds=seeds,
        description="mutation fixture",
    )


def _recordings(*, extra_test: bool = False) -> list[Recording]:
    loads = [(0, "rec-0"), (100, "rec-100"), (200, "rec-200a")]
    if extra_test:
        loads.append((200, "rec-200b"))
    return [
        Recording(
            dataset_id="mutation-v1",
            recording_id=name,
            source_file=f"{name}.csv",
            label=LABELS[0] if load == 0 else LABELS[1],
            domains={"load": load, "device": name},
        )
        for load, name in loads
    ]


def _batch(recordings: list[Recording], *, windows: int = 2) -> WindowBatch:
    rng = np.random.default_rng(5)
    samples: list[WindowSample] = []
    signals = []
    for record in recordings:
        for index in range(windows):
            samples.append(
                WindowSample(
                    window_id=f"{record.recording_id}#s{index:03d}",
                    recording_id=record.recording_id,
                    start_sample=index * 8,
                    end_sample=(index + 1) * 8,
                    label=record.label,
                    domains=dict(record.domains),
                )
            )
            signals.append(rng.normal(size=(2, 8)))
    return WindowBatch(
        samples=tuple(samples),
        signals=np.stack(signals),
        channels=("ch0", "ch1"),
        sample_rate_hz=64.0,
    )


def _table_from_batch(batch: WindowBatch) -> PreparedFeatureTable:
    rows = [
        PreparedSample(
            recording_id=sample.recording_id,
            label=sample.label,
            domains=dict(sample.domains),
            features=tuple(float(value) for value in batch.signals[index, 0]),
            window_id=sample.window_id,
            start_sample=sample.start_sample,
            end_sample=sample.end_sample,
        )
        for index, sample in enumerate(batch.samples)
    ]
    return PreparedFeatureTable(
        dataset_id="mutation-v1",
        feature_names=tuple(f"ch0_s{index}" for index in range(8)),
        samples=tuple(rows),
        preparation={"kind": "mutation-fixture"},
    )


class _ThresholdEstimator:
    def __init__(self, seed: int):
        self.seed = seed
        self.threshold_ = 0.0

    def fit(self, features, labels):
        self.threshold_ = float(np.asarray(features)[:, 0].mean())
        return self

    def predict(self, features):
        return [
            "fault" if row[0] > self.threshold_ else "healthy"
            for row in np.asarray(features)
        ]

    def get_params(self):
        return {"seed": self.seed}


class _RowSpyEstimator:
    """Records the leading feature of every row handed to fit."""

    seen_first_features: list[float] = []

    def __init__(self, seed: int):
        self.seed = seed

    def fit(self, features, labels):
        _RowSpyEstimator.seen_first_features.extend(
            float(row) for row in np.asarray(features)[:, 0]
        )

    def predict(self, features):
        return ["healthy"] * len(features)

    def get_params(self):
        return {"seed": self.seed}


class _FakeModelEntryPoint:
    def __init__(self, name: str, exposed_id: str):
        self.name = name
        self._exposed_id = exposed_id

    def load(self):
        return ModelPlugin(
            model_id=self._exposed_id,
            backend="sklearn",
            source="mutation-fixture",
            factory=lambda context: _ThresholdEstimator(context.seed),
        )


def _minimal_bundle(seeds: list[int], *, runs: list[dict] | None = None) -> dict:
    if runs is None:
        runs = [
            {"seed": seed, "status": "failed", "error": "synthetic"} for seed in seeds
        ]
    return {
        "schema_version": 2,
        "task_id": "mutation-load-v1",
        "model_id": "mutation-model",
        "git_commit": "abcdef1234567890",
        "environment": {"python": "3.11"},
        "protocol": {"task_seeds": seeds},
        "model_plugin": {"backend": "sklearn"},
        "input_artifacts": {},
        "runs": runs,
    }


def _write_bundle(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Mutation scenarios
# ---------------------------------------------------------------------------


def _m_leak_01(workdir: Path) -> MutationOutcome:
    """A recording duplicated across train and test partitions (INV-2)."""

    records = _recordings()
    task = _task()
    duplicated = records + [
        Recording(
            dataset_id="mutation-v1",
            recording_id="rec-0",
            source_file="rec-0.csv",
            label="healthy",
            domains={"load": 200, "device": "rec-0"},  # same id, test partition
        )
    ]
    report = audit_recordings(duplicated, task)
    failed_checks = [check.check_id for check in report.checks if not check.passed]
    try:
        report.require_pass()
    except ValueError as error:
        return MutationOutcome(
            "M-LEAK-01", "INV-2", "recording duplicated across partitions",
            True, "pre-training audit", f"failed checks {failed_checks}: {error}",
        )
    return MutationOutcome(
        "M-LEAK-01", "INV-2", "recording duplicated across partitions",
        False, "none", "audit passed despite cross-partition duplication",
    )


def _m_leak_02(workdir: Path) -> MutationOutcome:
    """Test-recording rows relabelled into the training partition (INV-2)."""

    records = _recordings(extra_test=True)
    task = _task()
    honest = _table_from_batch(_batch(records))
    rows = []
    for sample in honest.samples:
        domains = dict(sample.domains)
        if sample.recording_id == "rec-200a":
            domains["load"] = 0
        rows.append(
            PreparedSample(
                recording_id=sample.recording_id,
                label=sample.label,
                domains=domains,
                features=sample.features,
                window_id=sample.window_id,
                start_sample=sample.start_sample,
                end_sample=sample.end_sample,
            )
        )
    tampered = PreparedFeatureTable(
        dataset_id=honest.dataset_id,
        feature_names=honest.feature_names,
        samples=tuple(rows),
        preparation=honest.preparation,
    )
    try:
        prepare_task_data(tampered, task, recordings=records)
    except ValueError as error:
        return MutationOutcome(
            "M-LEAK-02", "INV-2", "test recording relabelled as train rows",
            True, "recording cross-audit", str(error)[:200],
        )
    return MutationOutcome(
        "M-LEAK-02", "INV-2", "test recording relabelled as train rows",
        False, "none", "relabelled rows accepted by task preparation",
    )


def _m_leak_03(workdir: Path) -> MutationOutcome:
    """Preprocessor fitted on rows that include the test partition (INV-3)."""

    records = _recordings()
    task = _task()
    batch = _batch(records)
    kernel_fit = ChannelStandardization()
    record = fit_preprocessor(kernel_fit, batch, task)

    train_only = ChannelStandardization()
    train_only.fit(training_view(batch, task))
    all_rows = ChannelStandardization()
    all_rows.fit(batch)

    isolated = (
        record.state_digest == train_only.fitted_state_digest()
        and record.state_digest != all_rows.fitted_state_digest()
    )
    return MutationOutcome(
        "M-LEAK-03", "INV-3", "preprocessing fitted with test statistics",
        isolated,
        "kernel fit-scope guard",
        (
            f"kernel digest matches train-only fit and differs from all-rows fit; "
            f"fit_scope={record.fit_scope}"
            if isolated
            else "fitted state did not differ from an all-rows fit"
        ),
    )


def _m_leak_04(workdir: Path) -> MutationOutcome:
    """Training attempt that tries to see test rows (INV-3, architectural)."""

    records = _recordings()
    task = _task()
    batch = _batch(records)
    table = _table_from_batch(batch)
    features = workdir / "m-leak-04-features.json"
    write_prepared_feature_table(table, features)
    # Encode the row index into the first feature so seen rows are identifiable.
    payload = table.to_dict()
    for index, sample in enumerate(payload["samples"]):
        values = list(sample["features"])
        values[0] = float(index)
        sample["features"] = values
    features.write_text(json.dumps(payload), encoding="utf-8")

    test_rows = {
        index for index, sample in enumerate(table.samples)
        if task.partition_for_sample(sample.domains) == "test"
    }
    allowed = set(range(len(table.samples))) - test_rows

    _RowSpyEstimator.seen_first_features = []
    import edgefault_bench.execution as execution

    original = execution._git_commit
    execution._git_commit = lambda: "abcdef1234567890"
    try:
        run_benchmark(
            task_path=_write_task_file(workdir),
            model_id="row_spy",
            output_path=workdir / "m-leak-04-result.json",
            features_path=features,
            model_plugins={
                "row_spy": ModelPlugin(
                    "row_spy", "sklearn", "mutation-fixture",
                    lambda context: _RowSpyEstimator(context.seed),
                )
            },
            latency_warmup=0,
            latency_repeats=1,
        )
    finally:
        execution._git_commit = original

    seen = {int(value) for value in _RowSpyEstimator.seen_first_features}
    leaked = seen - allowed
    return MutationOutcome(
        "M-LEAK-04", "INV-3", "fit call receives test rows",
        not leaked,
        "partition-restricted views",
        (
            f"fit saw rows {sorted(seen)}; test rows {sorted(test_rows)} absent"
            if not leaked
            else f"fit unexpectedly saw test rows {sorted(leaked)}"
        ),
    )


def _m_hash_01(workdir: Path) -> MutationOutcome:
    """A registered source file whose bytes no longer match its SHA-256 (INV-1)."""

    from scipy.io import savemat

    from edgefault_bench.datasets.hust import HustV3Adapter

    raw_dir = workdir / "hust-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    signal = np.zeros(512_000)
    mat_path = raw_dir / "N400.mat"
    savemat(mat_path, {"data": signal, "fs": np.array([12_000.0])})

    grid = [
        f"{label}{bearing}{load}.mat"
        for label in ("N", "I", "O", "B", "IO", "IB", "OB")
        for bearing in range(4, 9)
        for load in ("00", "02", "04")
    ][:60]
    entries = []
    for name in grid:
        if name == mat_path.name:
            size, digest = mat_path.stat().st_size, hashlib.sha256(
                mat_path.read_bytes()
            ).hexdigest()
        else:
            size, digest = 1, "0" * 64
        entries.append(
            {
                "id": name,
                "filename": name,
                "size_bytes": size,
                "sha256": digest,
                "download_url": f"https://example.org/{name}",
            }
        )
    manifest = workdir / "hust-mutation.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "hust-bearing-v3",
                "version": 3,
                "title": "mutation registry",
                "doi": "10.17632/cbv7jyx4p9.3",
                "landing_page": "https://example.org",
                "license": {"spdx": "CC-BY-4.0"},
                "sampling_rate_hz": 64_000,
                "files": entries,
            }
        ),
        encoding="utf-8",
    )
    adapter = HustV3Adapter(manifest, raw_dir=raw_dir)
    savemat(mat_path, {"data": np.ones(512_000), "fs": np.array([12_000.0])})  # tamper
    try:
        adapter.open("N400.mat")
    except ValueError as error:
        return MutationOutcome(
            "M-HASH-01", "INV-1", "source bytes differ from registered SHA-256",
            True, "adapter checksum verification", str(error)[:200],
        )
    return MutationOutcome(
        "M-HASH-01", "INV-1", "source bytes differ from registered SHA-256",
        False, "none", "tampered source file was opened without verification",
    )


def _write_task_file(workdir: Path) -> Path:
    path = workdir / "mutation-task.json"
    path.write_text(
        json.dumps(
            {
                "task_id": "mutation-load-v1",
                "dataset_id": "mutation-v1",
                "domain_field": "load",
                "evaluation_group_field": "device",
                "partitions": {"train": [0], "validation": [100], "test": [200]},
                "labels": list(LABELS),
                "window": {"length": 8, "stride": 8, "normalization": "none"},
                "seeds": [17, 29],
                "description": "mutation fixture",
            }
        ),
        encoding="utf-8",
    )
    return path


def _m_seed_01(workdir: Path) -> MutationOutcome:
    """A result bundle with one registered seed dropped (INV-4)."""

    path = _write_bundle(
        workdir / "m-seed-01.json", _minimal_bundle([17, 29], runs=[
            {"seed": 17, "status": "failed", "error": "synthetic"},
        ])
    )
    try:
        load_result(path)
    except ValueError as error:
        return MutationOutcome(
            "M-SEED-01", "INV-4", "seed run dropped from bundle",
            True, "result validator", str(error)[:200],
        )
    return MutationOutcome(
        "M-SEED-01", "INV-4", "seed run dropped from bundle",
        False, "none", "incomplete seed sequence accepted",
    )


def _m_seed_02(workdir: Path) -> MutationOutcome:
    """One seed's run replaced by a copy of another (INV-4)."""

    path = _write_bundle(
        workdir / "m-seed-02.json", _minimal_bundle([17, 29], runs=[
            {"seed": 17, "status": "failed", "error": "synthetic"},
            {"seed": 17, "status": "failed", "error": "synthetic"},
        ])
    )
    try:
        load_result(path)
    except ValueError as error:
        return MutationOutcome(
            "M-SEED-02", "INV-4", "seed duplicated to impersonate another",
            True, "result validator", str(error)[:200],
        )
    return MutationOutcome(
        "M-SEED-02", "INV-4", "seed duplicated to impersonate another",
        False, "none", "duplicated seed accepted",
    )


def _m_prov_01(workdir: Path) -> MutationOutcome:
    """Environment provenance stripped from a bundle (INV-5)."""

    payload = _minimal_bundle([17, 29])
    payload.pop("environment")
    path = _write_bundle(workdir / "m-prov-01.json", payload)
    try:
        load_result(path)
    except ValueError as error:
        return MutationOutcome(
            "M-PROV-01", "INV-5", "environment provenance removed",
            True, "result validator", str(error)[:200],
        )
    return MutationOutcome(
        "M-PROV-01", "INV-5", "environment provenance removed",
        False, "none", "bundle without environment accepted",
    )


def _m_prov_02(workdir: Path) -> MutationOutcome:
    """Input artefact tampered after the bundle was written (INV-5, M-PROV-02)."""

    records = _recordings()
    table = _table_from_batch(_batch(records))
    features = workdir / "m-prov-02-features.json"
    write_prepared_feature_table(table, features)
    import edgefault_bench.execution as execution

    original = execution._git_commit
    execution._git_commit = lambda: "abcdef1234567890"
    try:
        output = run_benchmark(
            task_path=_write_task_file(workdir),
            model_id="threshold",
            output_path=workdir / "m-prov-02-result.json",
            features_path=features,
            model_plugins={
                "threshold": ModelPlugin(
                    "threshold", "sklearn", "mutation-fixture",
                    lambda context: _ThresholdEstimator(context.seed),
                )
            },
            latency_warmup=0,
            latency_repeats=1,
        )
    finally:
        execution._git_commit = original

    payload = json.loads(features.read_text(encoding="utf-8"))
    payload["samples"][0]["features"][0] = 999.0
    features.write_text(json.dumps(payload), encoding="utf-8")

    report = verify_input_artifact_hashes(output)
    detected = not report["passed"]
    return MutationOutcome(
        "M-PROV-02", "INV-5", "input artefact modified after result creation",
        detected,
        "input-artifact hash check",
        json.dumps(report["artifacts"]["prepared_feature_table"])[:200],
    )


def _m_plugin_01(workdir: Path) -> MutationOutcome:
    """Plugin entry-point name differs from its declared identity (INV-6)."""

    entry = _FakeModelEntryPoint("registered-name", "declared-other-id")
    try:
        discover_model_plugins([entry])
    except (ValueError, TypeError) as error:
        return MutationOutcome(
            "M-PLUGIN-01", "INV-6", "entry-point identity mismatch",
            True, "plugin registry", str(error)[:200],
        )
    return MutationOutcome(
        "M-PLUGIN-01", "INV-6", "entry-point identity mismatch",
        False, "none", "mismatched identity accepted",
    )


def _m_plugin_02(workdir: Path) -> MutationOutcome:
    """Model object missing its backend contract methods (INV-6)."""

    try:
        validate_model_implementation(object(), backend="pytorch")
    except (ValueError, TypeError) as error:
        return MutationOutcome(
            "M-PLUGIN-02", "INV-6", "model missing contract methods",
            True, "plugin contract validation", str(error)[:200],
        )
    return MutationOutcome(
        "M-PLUGIN-02", "INV-6", "model missing contract methods",
        False, "none", "contract-less model accepted",
    )


def _m_plugin_03(workdir: Path) -> MutationOutcome:
    """External plugin colliding with a built-in identifier (INV-6)."""

    entry = _FakeModelEntryPoint("standard_cnn_1d", "standard_cnn_1d")
    try:
        discover_model_plugins([entry])
    except (ValueError, TypeError) as error:
        return MutationOutcome(
            "M-PLUGIN-03", "INV-6", "built-in identifier collision",
            True, "plugin registry", str(error)[:200],
        )
    return MutationOutcome(
        "M-PLUGIN-03", "INV-6", "built-in identifier collision",
        False, "none", "collision accepted",
    )


MUTATIONS: dict[str, Callable[[Path], MutationOutcome]] = {
    "M-LEAK-01": _m_leak_01,
    "M-LEAK-02": _m_leak_02,
    "M-LEAK-03": _m_leak_03,
    "M-LEAK-04": _m_leak_04,
    "M-HASH-01": _m_hash_01,
    "M-SEED-01": _m_seed_01,
    "M-SEED-02": _m_seed_02,
    "M-PROV-01": _m_prov_01,
    "M-PROV-02": _m_prov_02,
    "M-PLUGIN-01": _m_plugin_01,
    "M-PLUGIN-02": _m_plugin_02,
    "M-PLUGIN-03": _m_plugin_03,
}


def run_mutation_catalogue(workdir: Path) -> list[MutationOutcome]:
    """Execute every catalogue entry against the live kernel mechanisms."""

    workdir.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for mutation_id, scenario in sorted(MUTATIONS.items()):
        scenario_dir = workdir / mutation_id.lower()
        scenario_dir.mkdir(parents=True, exist_ok=True)
        outcomes.append(scenario(scenario_dir))
    return outcomes


def mutation_audit(workdir: Path) -> dict[str, Any]:
    """Render the RQ1 detection matrix."""

    outcomes = run_mutation_catalogue(workdir)
    undetected = [
        outcome.mutation_id for outcome in outcomes if not outcome.detected
    ]
    return {
        "schema_version": 1,
        "catalogue": "formal-spec-draft.md#6",
        "total": len(outcomes),
        "detected": len(outcomes) - len(undetected),
        "passed": not undetected,
        "undetected": undetected,
        "mutations": [outcome.to_dict() for outcome in outcomes],
    }
