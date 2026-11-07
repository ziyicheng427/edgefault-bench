"""Unified execution tests (T06/T07/T08): window path, isolation, bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from edgefault_bench.contracts import (
    DatasetMetadata,
    Recording,
    SignalInfo,
)
from edgefault_bench.execution import (
    TrainingPolicy,
    load_training_policy,
    run_benchmark,
    write_training_policy,
)
from edgefault_bench.plugins import ModelPlugin
from edgefault_bench.prepared import (
    PreparedFeatureTable,
    PreparedSample,
    load_prepared_feature_table,
    prepare_task_data,
    write_prepared_feature_table,
)
from edgefault_bench.preprocessing import SignalFeatureExtractor
from edgefault_bench.reporting import load_result, verify_input_artifact_hashes
from edgefault_bench.schema import validate_artifact_schema
from edgefault_bench.windows import build_prepared_table, extract_windows

WINDOW = 32


class SpyEstimator:
    """Records the number of rows reaching fit; module-level for pickling."""

    seen_row_counts: list[int] = []

    def __init__(self, seed: int):
        self.seed = seed

    def fit(self, features, labels):
        SpyEstimator.seen_row_counts.append(len(features))
        return self

    def predict(self, features):
        return ["fault"] * len(features)

    def get_params(self):
        return {"seed": self.seed}


class SignalDatasetAdapter:
    """Two-channel synthetic dataset with load-separated recordings."""

    def __init__(
        self, *, window: int = WINDOW, samples: int = 128, second_test: bool = False
    ) -> None:
        rng = np.random.default_rng(11)
        self._recordings = []
        self._signals = {}
        rows = [(0, 0.0, "rec-0"), (100, 3.0, "rec-100"), (200, 6.0, "rec-200a")]
        if second_test:
            rows.append((200, 6.0, "rec-200b"))
        for load, base, name in rows:
            signal = rng.normal(loc=base, scale=0.5, size=(2, samples))
            self._signals[name] = signal
            self._recordings.append(
                Recording(
                    dataset_id="synthetic-v1",
                    recording_id=name,
                    source_file=f"{name}.npy",
                    label="healthy" if load != 200 else "fault",
                    domains={"load": load, "device": name},
                    sample_rate_hz=512.0,
                    sample_count=samples,
                )
            )

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            dataset_id="synthetic-v1",
            version="1",
            title="Synthetic execution fixture",
            license_spdx="CC0-1.0",
            source_url="https://example.org",
            domain_fields=("load", "device"),
        )

    def recordings(self) -> tuple[Recording, ...]:
        return tuple(self._recordings)

    def open(self, recording_id: str) -> _FixtureStream:
        if recording_id not in self._signals:
            raise ValueError(f"unknown recording {recording_id!r}")
        return _FixtureStream(self._signals[recording_id])


class _FixtureStream:
    def __init__(self, signal: np.ndarray) -> None:
        self._signal = signal
        self._info = SignalInfo(
            channels=("ch0", "ch1"),
            sample_rate_hz=512.0,
            total_samples=int(signal.shape[1]),
        )

    @property
    def info(self) -> SignalInfo:
        return self._info

    def read(self, start: int, count: int) -> np.ndarray:
        if count <= 0 or start < 0 or start + count > self._info.total_samples:
            raise ValueError("read out of bounds")
        return self._signal[:, start : start + count].copy()


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "task_id": "synthetic-load-v1",
                "dataset_id": "synthetic-v1",
                "domain_field": "load",
                "evaluation_group_field": "device",
                "partitions": {"train": [0], "validation": [100], "test": [200]},
                "labels": ["healthy", "fault"],
                "window": {"length": WINDOW, "stride": WINDOW, "normalization": "none"},
                "seeds": [17, 29],
                "description": "synthetic execution fixture",
            }
        ),
        encoding="utf-8",
    )
    return task


def _manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset_id": "synthetic-v1"}), encoding="utf-8")
    return manifest


def test_extract_windows_produces_identity_and_partitions(tmp_path: Path) -> None:
    adapter = SignalDatasetAdapter()
    task = _task(tmp_path)
    from edgefault_bench.tasks import load_task_spec

    batch = extract_windows(adapter, load_task_spec(task))
    assert len(batch) == 3 * (128 // WINDOW)
    first = batch.samples[0]
    assert first.window_id == "rec-0#s000000000"
    assert (first.start_sample, first.end_sample) == (0, WINDOW)
    assert {sample.recording_id for sample in batch.samples} == {
        "rec-0",
        "rec-100",
        "rec-200a",
    }
    assert batch.channels == ("ch0", "ch1")


def test_build_prepared_table_carries_window_identity_and_fit_scope(
    tmp_path: Path,
) -> None:
    from edgefault_bench.tasks import load_task_spec

    adapter = SignalDatasetAdapter()
    task = load_task_spec(_task(tmp_path))
    batch = extract_windows(adapter, task)
    result = build_prepared_table(
        batch, SignalFeatureExtractor(), task, dataset_id="synthetic-v1"
    )
    table = result.table
    assert table.schema_version == 2
    assert table.has_window_identity
    assert table.feature_names[0] == "ch0_mean"
    assert table.preparation["preprocessor"]["fit_scope"] == ["train", "validation"]
    assert table.preparation["preprocessor"]["fitted_row_count"] == 2 * (
        128 // WINDOW
    )

    data = prepare_task_data(table, task, recordings=adapter.recordings())
    assert len(data.train) + len(data.validation) == 2 * (128 // WINDOW)
    assert len(data.test) == 128 // WINDOW


def test_relabeled_test_recording_is_rejected_by_cross_audit(tmp_path: Path) -> None:
    """M-LEAK-02: a test recording whose rows claim the training partition fails."""

    from edgefault_bench.tasks import load_task_spec

    adapter = SignalDatasetAdapter(second_test=True)
    task = load_task_spec(_task(tmp_path))
    batch = extract_windows(adapter, task)
    result = build_prepared_table(
        batch, SignalFeatureExtractor(), task, dataset_id="synthetic-v1"
    )
    honest = result.table
    assert len({s.recording_id for s in honest.samples if s.recording_id == "rec-200b"}) > 0

    rows = []
    for sample in honest.samples:
        domains = dict(sample.domains)
        if sample.recording_id == "rec-200a":
            domains["load"] = 0  # relabel the test recording into the train side
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
        dataset_id="synthetic-v1",
        feature_names=honest.feature_names,
        samples=tuple(rows),
        preparation=honest.preparation,
    )
    with pytest.raises(ValueError, match="relabelled partitions"):
        prepare_task_data(tampered, task, recordings=adapter.recordings())


def test_training_plugins_never_receive_test_rows(tmp_path: Path) -> None:
    """INV-3, architectural: fit sees train+validation features only."""

    SpyEstimator.seen_row_counts = []
    plugin = ModelPlugin(
        "spy", "sklearn", "example-distribution",
        lambda context: SpyEstimator(context.seed),
    )
    task = _task(tmp_path)
    adapter = SignalDatasetAdapter()
    from edgefault_bench.tasks import load_task_spec

    batch = extract_windows(adapter, load_task_spec(task))
    result = build_prepared_table(
        batch, SignalFeatureExtractor(), load_task_spec(task), dataset_id="synthetic-v1"
    )
    features = tmp_path / "features.json"
    write_prepared_feature_table(result.table, features)

    import edgefault_bench.execution as execution

    original = execution._git_commit
    execution._git_commit = lambda: "abcdef1234567890"
    try:
        run_benchmark(
            task_path=task,
            model_id="spy",
            output_path=tmp_path / "out.json",
            features_path=features,
            model_plugins={"spy": plugin},
            latency_warmup=0,
            latency_repeats=1,
        )
    finally:
        execution._git_commit = original

    # 128 samples per recording = 4 windows each. The sklearn-style fit receives
    # only the train rows (4); validation is a separate view and test rows (4)
    # never reach the plugin.
    assert SpyEstimator.seen_row_counts == [4, 4]


def _tiny_torch_plugin() -> ModelPlugin:
    torch = pytest.importorskip("torch")

    class TinyConvNet(torch.nn.Module):
        def __init__(self, num_classes: int, in_channels: int):
            super().__init__()
            self.conv = torch.nn.Conv1d(in_channels, 4, kernel_size=3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool1d(1)
            self.head = torch.nn.Linear(4, num_classes)

        def forward(self, x):
            pooled = self.pool(torch.relu(self.conv(x))).squeeze(-1)
            return self.head(pooled)

    return ModelPlugin(
        "tiny_conv",
        "pytorch",
        "example-distribution",
        lambda context: TinyConvNet(context.num_classes, context.input_channels),
    )


def test_pytorch_plugin_executes_from_dataset_manifest_window_path(
    tmp_path: Path,
) -> None:
    plugin = _tiny_torch_plugin()
    task = _task(tmp_path)
    manifest = _manifest(tmp_path)

    # The manifest must route through plugin discovery: register the fixture.
    import edgefault_bench.execution as execution
    import edgefault_bench.plugins as plugins_module

    original_loader = plugins_module.load_dataset_adapter
    plugins_module.load_dataset_adapter = lambda path: SignalDatasetAdapter()
    original_commit = execution._git_commit
    execution._git_commit = lambda: "abcdef1234567890"
    try:
        output = run_benchmark(
            task_path=task,
            model_id="tiny_conv",
            output_path=tmp_path / "torch-result.json",
            dataset_manifest=manifest,
            policy_path=None,
            model_plugins={"tiny_conv": plugin},
            latency_warmup=0,
            latency_repeats=1,
        )
    finally:
        plugins_module.load_dataset_adapter = original_loader
        execution._git_commit = original_commit

    payload = load_result(output)
    assert payload["schema_version"] == 2
    assert payload["model_plugin"]["backend"] == "pytorch"
    assert [run["status"] for run in payload["runs"]] == ["ok", "ok"]
    assert payload["input_artifacts"]["prepared_feature_table"] is None
    assert payload["input_artifacts"]["dataset_manifest"]["sha256"]
    assert payload["protocol"]["preparation"]["preprocessor"]["plugin_id"] == (
        "per_window_zscore"
    )
    validate_artifact_schema(output, "result")
    assert verify_input_artifact_hashes(output)["passed"] is True


def test_sklearn_plugin_on_window_path_requires_feature_output(tmp_path: Path) -> None:
    class TrivialEstimator:
        def fit(self, features, labels):
            return self

        def predict(self, features):
            return ["healthy"] * len(features)

        def get_params(self):
            return {}

    plugin = ModelPlugin(
        "trivial", "sklearn", "example-distribution",
        lambda context: TrivialEstimator(),
    )
    import edgefault_bench.plugins as plugins_module

    original = plugins_module.load_dataset_adapter
    plugins_module.load_dataset_adapter = lambda path: SignalDatasetAdapter()
    try:
        with pytest.raises(ValueError, match="output='features'"):
            run_benchmark(
                task_path=_task(tmp_path),
                model_id="trivial",
                output_path=tmp_path / "out.json",
                dataset_manifest=_manifest(tmp_path),
                preprocessor_id="per_window_zscore",
                model_plugins={"trivial": plugin},
            )
    finally:
        plugins_module.load_dataset_adapter = original


def test_training_policy_round_trip_and_validation(tmp_path: Path) -> None:
    policy = TrainingPolicy(optimizer="sgd", learning_rate=0.05, max_epochs=3)
    path = write_training_policy(policy, tmp_path / "policy.json")
    loaded = load_training_policy(path)
    assert loaded == policy
    assert loaded.content_digest() == policy.content_digest()

    with pytest.raises(ValueError, match="optimizer"):
        TrainingPolicy(optimizer="adagrad")
    with pytest.raises(ValueError, match="patience"):
        TrainingPolicy(patience=0)


def test_prepared_table_json_and_parquet_round_trip(tmp_path: Path) -> None:
    from edgefault_bench.tasks import load_task_spec

    adapter = SignalDatasetAdapter()
    task = load_task_spec(_task(tmp_path))
    batch = extract_windows(adapter, task)
    table = build_prepared_table(
        batch, SignalFeatureExtractor(), task, dataset_id="synthetic-v1"
    ).table

    json_path = write_prepared_feature_table(table, tmp_path / "table.json")
    validate_artifact_schema(json_path, "prepared-feature-table")
    reloaded = load_prepared_feature_table(json_path)
    assert reloaded.has_window_identity
    assert reloaded.feature_names == table.feature_names
    assert reloaded.samples[0].window_id == table.samples[0].window_id

    pyarrow = pytest.importorskip("pyarrow")
    parquet_path = write_prepared_feature_table(table, tmp_path / "table.parquet")
    assert pyarrow is not None
    from_parquet = load_prepared_feature_table(parquet_path)
    assert from_parquet.dataset_id == table.dataset_id
    assert from_parquet.feature_names == table.feature_names
    assert from_parquet.samples[-1].window_id == table.samples[-1].window_id
    np.testing.assert_allclose(
        np.asarray([sample.features for sample in from_parquet.samples]),
        np.asarray([sample.features for sample in table.samples]),
        rtol=1e-12,
    )
