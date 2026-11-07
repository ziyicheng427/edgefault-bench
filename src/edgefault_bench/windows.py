"""Kernel-side window extraction from signal-capable dataset adapters (T07).

This module owns the ``DATA_PREPARED`` lifecycle transition: it audits the
recording metadata, streams each registered recording through the adapter,
cuts deterministic non-overlapping windows, and emits batches whose window
identity (``window_id``, ``recording_id``, ``start_sample``, ``end_sample``)
makes the partition-inheritance invariant checkable at any later stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from edgefault_bench.audit import audit_recordings
from edgefault_bench.contracts import Recording, TaskSpec
from edgefault_bench.plugins import open_signal
from edgefault_bench.prepared import PreparedFeatureTable, PreparedSample
from edgefault_bench.preprocessing import (
    Preprocessor,
    PreprocessorFitRecord,
    WindowBatch,
    WindowSample,
    fit_preprocessor,
)


def window_id(recording_id: str, start_sample: int) -> str:
    """Deterministic, collision-free window identifier."""

    return f"{recording_id}#s{start_sample:09d}"


def extract_windows(
    adapter: object,
    task: TaskSpec,
    *,
    recordings: Sequence[Recording] | None = None,
) -> WindowBatch:
    """Audit, stream, and window every recording assigned by the task.

    Fails closed when the pre-training audit fails, the adapter lacks the
    signal capability, a stream contradicts its registered metadata, or the
    channel layout differs across recordings.
    """

    records = tuple(adapter.recordings()) if recordings is None else tuple(recordings)  # type: ignore[attr-defined]
    audit_recordings(records, task).require_pass()

    samples: list[WindowSample] = []
    signals: list[NDArray[np.float64]] = []
    channels: tuple[str, ...] | None = None
    sample_rate_hz: float | None = None
    length = task.window.length
    stride = task.window.stride

    for record in sorted(records, key=lambda item: item.recording_id):
        _, stream = open_signal(adapter, record.recording_id)
        info = stream.info
        if channels is None:
            channels = info.channels
            sample_rate_hz = info.sample_rate_hz
        elif info.channels != channels or info.sample_rate_hz != sample_rate_hz:
            raise ValueError(
                f"recording {record.recording_id!r} exposes channels {info.channels} at "
                f"{info.sample_rate_hz} Hz; expected {channels} at {sample_rate_hz}"
            )
        total = info.total_samples
        if total < length:
            raise ValueError(
                f"recording {record.recording_id!r} is shorter than one window"
            )
        for start in range(0, total - length + 1, stride):
            block = stream.read(start, length)
            samples.append(
                WindowSample(
                    window_id=window_id(record.recording_id, start),
                    recording_id=record.recording_id,
                    start_sample=start,
                    end_sample=start + length,
                    label=record.label,
                    domains=dict(record.domains),
                )
            )
            signals.append(block)

    assert channels is not None  # the audited recording set is never empty
    return WindowBatch(
        samples=tuple(samples),
        signals=np.stack(signals),
        channels=channels,
        sample_rate_hz=sample_rate_hz,
    )


@dataclass(frozen=True)
class PreparedPipelineResult:
    """A feature table plus the fit provenance that produced it."""

    table: PreparedFeatureTable
    fit_record: PreprocessorFitRecord
    channel_layout: tuple[str, ...]
    sample_rate_hz: float


def build_prepared_table(
    batch: WindowBatch,
    preprocessor: Preprocessor,
    task: TaskSpec,
    *,
    dataset_id: str,
) -> PreparedPipelineResult:
    """Fit a feature preprocessor on the training side and transform all windows.

    ``fit`` receives only the train+validation view; ``transform`` then maps
    every window deterministically, so held-out rows never influence the
    fitted state (INV-3).
    """

    if preprocessor.output != "features":
        raise ValueError(
            f"preprocessor {preprocessor.plugin_id!r} outputs {preprocessor.output!r}; "
            "prepared tables require an output='features' preprocessor"
        )
    fit_record = fit_preprocessor(preprocessor, batch, task)
    features = preprocessor.transform(batch)
    if features.ndim != 2 or features.shape[0] != len(batch):
        raise ValueError(
            f"preprocessor {preprocessor.plugin_id!r} must return one vector per window"
        )
    if not np.isfinite(features).all():
        raise ValueError("preprocessor produced non-finite features")
    names = getattr(preprocessor, "feature_names", None)
    feature_names: tuple[str, ...]
    if callable(names):
        feature_names = tuple(names(batch.channels))
    else:
        feature_names = tuple(f"f{index}" for index in range(features.shape[1]))

    rows = []
    for index, sample in enumerate(batch.samples):
        rows.append(
            PreparedSample(
                recording_id=sample.recording_id,
                label=sample.label,
                domains=dict(sample.domains),
                features=tuple(float(value) for value in features[index]),
                window_id=sample.window_id,
                start_sample=sample.start_sample,
                end_sample=sample.end_sample,
            )
        )
    table = PreparedFeatureTable(
        dataset_id=dataset_id,
        feature_names=feature_names,
        samples=tuple(rows),
        preparation={
            "preprocessor": fit_record.to_dict(),
            "channel_layout": list(batch.channels),
            "sample_rate_hz": batch.sample_rate_hz,
            "window": task.window.to_dict(),
        },
        schema_version=2,
    )
    return PreparedPipelineResult(
        table=table,
        fit_record=fit_record,
        channel_layout=batch.channels,
        sample_rate_hz=batch.sample_rate_hz,
    )
