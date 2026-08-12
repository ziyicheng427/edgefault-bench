"""Public-interface-only example plugins for EdgeFault-Bench."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from edgefault_bench.contracts import DatasetMetadata, Recording
from edgefault_bench.plugins import ModelBuildContext, ModelPlugin


class ExampleDatasetAdapter:
    """Load a tiny metadata-only dataset manifest without kernel imports."""

    def __init__(self, manifest: Path):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("dataset_id") != "edgefault-example-v1":
            raise ValueError("example manifest has an unexpected dataset_id")
        self._metadata = DatasetMetadata(
            dataset_id=payload["dataset_id"],
            version=str(payload["version"]),
            title=payload["title"],
            license_spdx=payload["license_spdx"],
            source_url=payload["source_url"],
            domain_fields=tuple(payload["domain_fields"]),
        )
        self._recordings = tuple(
            Recording(
                dataset_id=payload["dataset_id"],
                recording_id=item["recording_id"],
                source_file=item["source_file"],
                label=item["label"],
                domains=item["domains"],
                sample_rate_hz=item["sample_rate_hz"],
                sample_count=item["sample_count"],
            )
            for item in payload["recordings"]
        )

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def recordings(self) -> tuple[Recording, ...]:
        return self._recordings


class NearestCentroidClassifier:
    """Small dependency-free estimator used only to exercise the plugin boundary."""

    def __init__(self, *, seed: int):
        self.seed = seed
        self.centroids_: dict[object, np.ndarray] = {}

    def fit(self, features, labels):
        features_array = np.asarray(features, dtype=float)
        labels_array = np.asarray(labels)
        self.centroids_ = {
            label: features_array[labels_array == label].mean(axis=0)
            for label in np.unique(labels_array)
        }
        return self

    def predict(self, features):
        if not self.centroids_:
            raise RuntimeError("estimator must be fitted before prediction")
        features_array = np.asarray(features, dtype=float)
        labels = tuple(self.centroids_)
        distances = np.stack(
            [np.linalg.norm(features_array - self.centroids_[label], axis=1) for label in labels],
            axis=1,
        )
        return np.asarray([labels[index] for index in distances.argmin(axis=1)])

    def get_params(self, deep: bool = True) -> dict[str, int]:
        return {"seed": self.seed}


def _build_model(context: ModelBuildContext) -> NearestCentroidClassifier:
    return NearestCentroidClassifier(seed=context.seed)


model_plugin = ModelPlugin(
    model_id="example_centroid_classifier",
    backend="sklearn",
    source="resolved-from-installed-distribution",
    factory=_build_model,
)

__all__ = ["ExampleDatasetAdapter", "NearestCentroidClassifier", "model_plugin"]
