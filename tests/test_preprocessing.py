"""Preprocessing contract tests (T05): fit scope, digests, invariants."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from edgefault_bench.contracts import TaskSpec, WindowSpec
from edgefault_bench.preprocessing import (
    ChannelStandardization,
    PerWindowZScore,
    SignalFeatureExtractor,
    WindowBatch,
    WindowSample,
    create_preprocessor,
    discover_preprocessor_plugins,
    fit_preprocessor,
    partition_window_indices,
    training_view,
)


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="example-load-v1",
        dataset_id="example-v1",
        domain_field="load",
        evaluation_group_field="device",
        partitions={"train": [0], "validation": [100], "test": [200]},
        labels=("healthy", "fault"),
        window=WindowSpec(length=8, stride=8, normalization="none"),
        seeds=(17,),
        description="synthetic preprocessing test",
    )


def _batch(seed: int = 3, *, windows_per_partition: int = 3) -> WindowBatch:
    """Deterministic two-channel windows spread across loads 0/100/200."""

    rng = np.random.default_rng(seed)
    samples: list[WindowSample] = []
    signals: list[np.ndarray] = []
    offset = 0.0
    for load, windows in (
        (0, windows_per_partition),
        (100, windows_per_partition),
        (200, windows_per_partition),
    ):
        for index in range(windows):
            recording = f"rec-{load}"
            samples.append(
                WindowSample(
                    window_id=f"{recording}#w{index}",
                    recording_id=recording,
                    start_sample=index * 8,
                    end_sample=(index + 1) * 8,
                    label="healthy" if load == 0 else "fault",
                    domains={"load": load, "device": f"dev-{load}"},
                )
            )
            signals.append(rng.normal(loc=offset, scale=1.0 + load / 100, size=(2, 8)))
        offset += 5.0
    return WindowBatch(
        samples=tuple(samples),
        signals=np.stack(signals),
        channels=("ch0", "ch1"),
        sample_rate_hz=1_000.0,
    )


def test_partition_window_indices_assigns_and_rejects_spanning() -> None:
    batch = _batch()
    task = _task()
    indices = partition_window_indices(batch, task)
    assert set(indices) == {"train", "validation", "test"}
    assert all(len(values) == 3 for values in indices.values())

    spanning = WindowSample(
        window_id="evil#w0",
        recording_id="rec-0",
        start_sample=0,
        end_sample=8,
        label="fault",
        domains={"load": 200, "device": "dev-0"},
    )
    tampered = WindowBatch(
        samples=batch.samples + (spanning,),
        signals=np.concatenate([batch.signals, np.zeros((1, 2, 8))]),
        channels=batch.channels,
        sample_rate_hz=batch.sample_rate_hz,
    )
    with pytest.raises(ValueError, match="spans partitions"):
        partition_window_indices(tampered, task)


def test_training_view_excludes_test_rows() -> None:
    batch = _batch()
    task = _task()
    view = training_view(batch, task)
    loads = {sample.domains["load"] for sample in view.samples}
    assert loads == {0, 100}
    assert all(sample.recording_id != "rec-200" for sample in view.samples)


def test_channel_standardization_fits_on_training_side_only() -> None:
    batch = _batch()
    task = _task()
    preprocessor = ChannelStandardization()
    record = fit_preprocessor(preprocessor, batch, task)

    assert record.fit_scope == ("train", "validation")
    assert record.fitted_row_count == 6
    # The fitted mean must reflect train+validation rows only: the test rows in
    # this fixture are shifted by +10.0, so an all-rows fit would differ.
    shifted = WindowBatch(
        samples=batch.samples,
        signals=batch.signals + 10.0,
        channels=batch.channels,
        sample_rate_hz=batch.sample_rate_hz,
    )
    all_rows = ChannelStandardization()
    all_rows.fit(shifted)
    assert preprocessor.fitted_state_digest() != all_rows.fitted_state_digest()

    transformed = preprocessor.transform(batch)
    assert transformed.shape == batch.signals.shape
    train_view = training_view(batch, task)
    np.testing.assert_allclose(
        preprocessor.transform(train_view).mean(axis=(0, 2)), 0.0, atol=1e-9
    )


def test_state_digest_is_stable_and_sensitive() -> None:
    task = _task()
    first = fit_preprocessor(ChannelStandardization(), _batch(seed=1), task)
    second = fit_preprocessor(ChannelStandardization(), _batch(seed=1), task)
    third = fit_preprocessor(ChannelStandardization(), _batch(seed=2), task)
    assert first.state_digest == second.state_digest
    assert first.state_digest != third.state_digest
    assert len(first.state_digest) == 64


def test_transform_without_fit_fails_closed() -> None:
    with pytest.raises(ValueError, match="fitted"):
        ChannelStandardization().transform(_batch())


def test_per_window_zscore_matches_reference_semantics() -> None:
    batch = _batch()
    from edgefault_bench.datasets.hust import preprocess_window

    preprocessor = PerWindowZScore()
    fit_preprocessor(preprocessor, batch, _task())
    transformed = preprocessor.transform(batch)
    expected = preprocess_window(batch.signals[0, 0])
    np.testing.assert_allclose(transformed[0, 0], expected, atol=1e-6)
    np.testing.assert_allclose(
        transformed.std(axis=2), 1.0, atol=1e-5
    )


def test_signal_feature_extractor_returns_vector_features() -> None:
    batch = _batch()
    preprocessor = SignalFeatureExtractor()
    record = fit_preprocessor(preprocessor, batch, _task())
    features = preprocessor.transform(batch)

    assert preprocessor.output == "features"
    assert features.shape == (len(batch), 2 * 10)
    assert len(preprocessor.feature_names(batch.channels)) == 20
    assert record.state_digest == preprocessor.fitted_state_digest()
    assert np.isfinite(features).all()


def test_window_sample_and_batch_validation() -> None:
    with pytest.raises(ValueError, match="end_sample"):
        WindowSample(
            window_id="w",
            recording_id="r",
            start_sample=8,
            end_sample=8,
            label="healthy",
            domains={"load": 0},
        )
    batch = _batch()
    with pytest.raises(ValueError, match="same windows"):
        WindowBatch(
            samples=batch.samples[:-1],
            signals=batch.signals,
            channels=batch.channels,
            sample_rate_hz=batch.sample_rate_hz,
        )
    with pytest.raises(ValueError, match="non-finite"):
        WindowBatch(
            samples=batch.samples,
            signals=batch.signals * np.nan,
            channels=batch.channels,
            sample_rate_hz=batch.sample_rate_hz,
        )


def test_discover_preprocessors_rejects_collisions() -> None:
    class FakePreprocessorEntryPoint:
        name = "per_window_zscore"
        module = "evil"
        dist = SimpleNamespace(name="evil-package")

        def load(self):
            return PerWindowZScore

    with pytest.raises(ValueError, match="duplicate preprocessor plugin"):
        discover_preprocessor_plugins([FakePreprocessorEntryPoint()])


def test_create_preprocessor_validates_contract() -> None:
    class NotAPreprocessor:
        pass

    class BadFactoryEntryPoint:
        name = "bad_preprocessor"
        module = "bad"
        dist = SimpleNamespace(name="bad-package")

        def load(self):
            return lambda: NotAPreprocessor()

    with pytest.raises(TypeError, match="Preprocessor"):
        create_preprocessor("bad_preprocessor", entry_points=[BadFactoryEntryPoint()])
    with pytest.raises(ValueError, match="no preprocessor plugin"):
        create_preprocessor("missing", entry_points=[])

    created = create_preprocessor("per_window_zscore", entry_points=[])
    assert isinstance(created, PerWindowZScore)
    assert created.output == "windows"
