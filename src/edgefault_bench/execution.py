"""Unified backend execution under versioned training policies (T06/T08).

The kernel builds partition-restricted data views, resolves a versioned
:class:`TrainingPolicy`, and drives one :class:`Estimator` per frozen seed.
Training receives only the train and validation views; the test view is
materialized exclusively inside the evaluation step (INV-3). Every run —
including failures — is retained in a schema-v2 result bundle whose input
artifacts are content-hashed (INV-5) and whose seed sequence must biject with
the task's registered seeds (INV-4).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Protocol

import numpy as np
import sklearn

from edgefault_bench.contracts import Recording, TaskSpec
from edgefault_bench.evaluation import condition_metrics
from edgefault_bench.plugins import ModelBuildContext, ModelPlugin, discover_model_plugins
from edgefault_bench.prepared import (
    PreparedFeatureTable,
    load_prepared_feature_table,
    prepare_task_data,
)
from edgefault_bench.preprocessing import (
    Preprocessor,
    WindowBatch,
    create_preprocessor,
    fit_preprocessor,
    partition_window_indices,
)
from edgefault_bench.tasks import load_task_spec
from edgefault_bench.windows import extract_windows

BENCHMARK_ID = "edgefault-bench-plugin-v2"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _distribution_version(source: str) -> str:
    try:
        return importlib_metadata.version(source)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: object) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and (when installed) PyTorch RNGs deterministically."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
    except ModuleNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Versioned training policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingPolicy:
    """Kernel-owned, versioned training hyperparameters (plugins cannot redefine)."""

    policy_version: str = "1"
    optimizer: str = "none"
    learning_rate: float = 1e-3
    max_epochs: int = 20
    batch_size: int = 32
    early_stopping_metric: str = "validation_macro_f1"
    patience: int = 5
    deterministic: bool = True
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.policy_version.strip():
            raise ValueError("policy_version must be non-empty")
        if self.optimizer not in {"none", "adam", "sgd"}:
            raise ValueError("optimizer must be 'none', 'adam', or 'sgd'")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.max_epochs < 0:
            raise ValueError("max_epochs must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.early_stopping_metric != "validation_macro_f1":
            raise ValueError("early stopping must use 'validation_macro_f1'")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        object.__setattr__(self, "extra", dict(self.extra))

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "early_stopping_metric": self.early_stopping_metric,
            "patience": self.patience,
            "deterministic": self.deterministic,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrainingPolicy:
        return cls(
            policy_version=str(payload.get("policy_version", "1")),
            optimizer=str(payload.get("optimizer", "none")),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            max_epochs=int(payload.get("max_epochs", 20)),
            batch_size=int(payload.get("batch_size", 32)),
            early_stopping_metric=str(
                payload.get("early_stopping_metric", "validation_macro_f1")
            ),
            patience=int(payload.get("patience", 5)),
            deterministic=bool(payload.get("deterministic", True)),
            extra=dict(payload.get("extra", {})),  # type: ignore[arg-type]
        )

    def content_digest(self) -> str:
        return _canonical_digest(self.to_dict())


DEFAULT_SKLEARN_POLICY = TrainingPolicy(
    policy_version="1", optimizer="none", max_epochs=0, batch_size=1
)
DEFAULT_TORCH_POLICY = TrainingPolicy(
    policy_version="1", optimizer="adam", learning_rate=1e-3,
    max_epochs=20, batch_size=32, patience=5,
)


def load_training_policy(path: str | Path) -> TrainingPolicy:
    """Load and validate a versioned policy manifest from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"training policy must be a JSON object: {path}")
    return TrainingPolicy.from_dict(payload)


def write_training_policy(policy: TrainingPolicy, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


# ---------------------------------------------------------------------------
# Partition-restricted data views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataView:
    """One partition's model inputs; the only shape plugins ever see."""

    x: np.ndarray
    labels: np.ndarray
    evaluation_groups: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x)
        if x.ndim == 0 or x.shape[0] == 0:
            raise ValueError("DataView.x must be a non-empty array")
        if not (len(self.labels) == len(self.evaluation_groups) == x.shape[0]):
            raise ValueError("DataView arrays must agree on the sample axis")
        object.__setattr__(self, "x", x)

    def __len__(self) -> int:
        return int(self.x.shape[0])


class Estimator(Protocol):
    """Backend-neutral fit/predict lifecycle driven by a TrainingPolicy."""

    def fit(self, train: DataView, validation: DataView, policy: TrainingPolicy) -> None: ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------


def _stored_numeric_values(model: object) -> int:
    """Count learned numeric array values without claiming they are weights."""

    total = 0
    for name, value in vars(model).items():
        if not name.endswith("_"):
            continue
        if isinstance(value, dict):
            total += sum(np.asarray(item).size for item in value.values())
        elif isinstance(value, (np.ndarray, list, tuple)):
            total += np.asarray(value).size
    return int(total)


class SklearnEstimator:
    """Adapt an sklearn-style plugin (``fit``/``predict``/``get_params``)."""

    def __init__(self, model: object) -> None:
        for name in ("fit", "predict", "get_params"):
            if not callable(getattr(model, name, None)):
                raise TypeError(f"sklearn model is missing required method {name!r}")
        self._model = model

    def fit(self, train: DataView, validation: DataView, policy: TrainingPolicy) -> None:
        if policy.optimizer != "none":
            raise ValueError("sklearn execution requires optimizer='none' policies")
        self._model.fit(train.x, train.labels)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(x), dtype=str)

    def complexity(self) -> dict[str, object]:
        return {
            "trainable_parameters": _stored_numeric_values(self._model),
            "parameter_count_scope": "learned numeric values stored in fitted attributes",
            "serialized_size_bytes": len(pickle.dumps(self._model)),
        }


class TorchEstimator:
    """Train a PyTorch plugin module under the kernel's versioned policy."""

    def __init__(self, model: object, *, labels: tuple[str, ...], seed: int) -> None:
        try:
            import torch
        except ModuleNotFoundError as error:  # pragma: no cover - optional extra
            raise ValueError(
                "PyTorch execution requires the 'deep-learning' extra"
            ) from error
        for name in ("train", "eval", "state_dict", "parameters"):
            if not callable(getattr(model, name, None)):
                raise TypeError(f"pytorch model is missing required method {name!r}")
        self._torch = torch
        self.model = model
        self._labels = tuple(labels)
        self._label_index = {name: index for index, name in enumerate(self._labels)}
        self._seed = int(seed)
        self._batch_size = 64

    def fit(self, train: DataView, validation: DataView, policy: TrainingPolicy) -> None:
        torch = self._torch
        if policy.optimizer not in {"adam", "sgd"}:
            raise ValueError(
                "pytorch execution requires optimizer='adam' or 'sgd' policies"
            )
        if policy.deterministic:
            torch.use_deterministic_algorithms(True)
        self._batch_size = policy.batch_size
        x_train = torch.as_tensor(np.asarray(train.x, dtype=np.float32))
        y_train = torch.as_tensor(
            np.asarray([self._label_index[label] for label in train.labels], dtype=np.int64)
        )
        x_val = torch.as_tensor(np.asarray(validation.x, dtype=np.float32))
        y_val = np.asarray(
            [self._label_index[label] for label in validation.labels], dtype=np.int64
        )

        if policy.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=policy.learning_rate
            )
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=policy.learning_rate, momentum=0.9
            )
        loss_fn = torch.nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(self._seed * 1_003 + 1)

        best_f1 = -1.0
        best_state = copy.deepcopy(self.model.state_dict())
        stale = 0
        sample_count = int(x_train.shape[0])
        self.model.train()
        for _epoch in range(policy.max_epochs):
            permutation = torch.randperm(sample_count, generator=generator)
            for start in range(0, sample_count, policy.batch_size):
                picked = permutation[start : start + policy.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.model(x_train[picked]), y_train[picked])
                loss.backward()
                optimizer.step()
            score = float(self._score(x_val, y_val))
            if score > best_f1 + 1e-12:
                best_f1 = score
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= policy.patience:
                    break
        self.model.load_state_dict(best_state)
        self.model.eval()

    def _score(self, x_val, y_true: np.ndarray) -> float:
        predictions = self._predict_index(x_val)
        truth = [self._labels[index] for index in y_true]
        predicted = [self._labels[index] for index in predictions]
        return condition_metrics(
            truth, predicted, ["validation"] * len(truth)
        ).macro_f1

    def _predict_index(self, x) -> np.ndarray:
        torch = self._torch
        self.model.eval()
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, int(x.shape[0]), self._batch_size):
                logits = self.model(x[start : start + self._batch_size])
                outputs.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.asarray([], dtype=np.int64)

    def predict(self, x: np.ndarray) -> np.ndarray:
        tensor = self._torch.as_tensor(np.asarray(x, dtype=np.float32))
        indices = self._predict_index(tensor)
        return np.asarray([self._labels[index] for index in indices], dtype=str)

    def complexity(self) -> dict[str, object]:
        import io

        torch = self._torch
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        parameters = sum(
            value.numel() for value in self.model.parameters() if value.requires_grad
        )
        return {
            "trainable_parameters": int(parameters),
            "parameter_count_scope": "trainable torch parameters",
            "serialized_size_bytes": buffer.getbuffer().nbytes,
        }


def _latency_ms(
    estimator: Estimator, sample: np.ndarray, *, warmup: int, repeats: int
) -> dict[str, object]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("latency warmup must be non-negative and repeats positive")
    for _ in range(warmup):
        estimator.predict(sample)
    durations = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        started = time.perf_counter_ns()
        estimator.predict(sample)
        durations[index] = (time.perf_counter_ns() - started) / 1_000_000.0
    return {
        "batch_size": 1,
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
    }


# ---------------------------------------------------------------------------
# Unified benchmark execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PreparedInputs:
    x: np.ndarray
    labels: np.ndarray
    evaluation_groups: np.ndarray
    recording_ids: np.ndarray
    train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray
    input_artifacts: dict[str, dict[str, str] | None]
    preparation: dict[str, object]
    input_dim: int


def _resolve_inputs(
    *,
    task: TaskSpec,
    task_path: Path,
    features_path: Path | None,
    dataset_manifest: Path | None,
    preprocessor_id: str | None,
    recordings: Sequence[Recording] | None,
    model_plugins: Mapping[str, ModelPlugin],
    model_id: str,
) -> _PreparedInputs:
    plugin = model_plugins[model_id]

    def artifact(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": sha256_file(path)}

    if features_path is not None:
        table: PreparedFeatureTable = load_prepared_feature_table(features_path)
        canonical: Sequence[Recording] | None = None
        if dataset_manifest is not None:
            from edgefault_bench.plugins import load_dataset_adapter

            adapter = load_dataset_adapter(dataset_manifest)
            canonical = tuple(adapter.recordings())
        data = prepare_task_data(table, task, recordings=canonical or ())
        return _PreparedInputs(
            x=data.features,
            labels=data.labels,
            evaluation_groups=data.evaluation_groups,
            recording_ids=np.asarray(
                [sample.recording_id for sample in table.samples], dtype=str
            ),
            train_idx=data.train,
            validation_idx=data.validation,
            test_idx=data.test,
            input_artifacts={
                "task_manifest": artifact(task_path),
                "prepared_feature_table": artifact(features_path),
                "dataset_manifest": artifact(dataset_manifest)
                if dataset_manifest is not None
                else None,
            },
            preparation=dict(table.preparation),
            input_dim=len(table.feature_names),
        )

    if dataset_manifest is None:
        raise ValueError(
            "benchmark execution needs either --features or --dataset-manifest"
        )
    from edgefault_bench.plugins import load_dataset_adapter

    adapter = load_dataset_adapter(dataset_manifest)
    batch: WindowBatch = extract_windows(
        adapter, task, recordings=None if recordings is None else tuple(recordings)
    )
    resolved_id = preprocessor_id
    if resolved_id is None:
        resolved_id = (
            "per_window_zscore" if plugin.backend == "pytorch" else "signal_features"
        )
    preprocessor: Preprocessor = create_preprocessor(resolved_id)
    if plugin.backend == "sklearn" and preprocessor.output != "features":
        raise ValueError(
            "sklearn plugins require an output='features' preprocessor; "
            f"got {resolved_id!r}"
        )
    fit_record = fit_preprocessor(preprocessor, batch, task)
    transformed = preprocessor.transform(batch)
    partitions = partition_window_indices(batch, task)
    labels = np.asarray([sample.label for sample in batch.samples], dtype=str)
    evaluation_groups = np.asarray(
        [
            str(sample.domains[task.evaluation_group_field])
            for sample in batch.samples
        ],
        dtype=str,
    )
    input_dim = (
        int(transformed.shape[1]) if transformed.ndim >= 2 else 1
    )
    return _PreparedInputs(
        x=transformed,
        labels=labels,
        evaluation_groups=evaluation_groups,
        recording_ids=np.asarray(
            [sample.recording_id for sample in batch.samples], dtype=str
        ),
        train_idx=np.asarray(partitions["train"], dtype=np.int64),
        validation_idx=np.asarray(partitions["validation"], dtype=np.int64),
        test_idx=np.asarray(partitions["test"], dtype=np.int64),
        input_artifacts={
            "task_manifest": artifact(task_path),
            "prepared_feature_table": None,
            "dataset_manifest": artifact(dataset_manifest),
        },
        preparation={
            "preprocessor": fit_record.to_dict(),
            "channel_layout": list(batch.channels),
            "sample_rate_hz": batch.sample_rate_hz,
            "window": task.window.to_dict(),
        },
        input_dim=input_dim,
    )


def run_benchmark(
    *,
    task_path: Path,
    model_id: str,
    output_path: Path,
    features_path: Path | None = None,
    dataset_manifest: Path | None = None,
    preprocessor_id: str | None = None,
    policy_path: Path | None = None,
    recordings: Sequence[Recording] | None = None,
    model_plugins: Mapping[str, ModelPlugin] | None = None,
    latency_warmup: int = 20,
    latency_repeats: int = 100,
) -> Path:
    """Execute one model plugin across every frozen seed and write a v2 bundle."""

    started = time.perf_counter()
    task = load_task_spec(task_path)
    registry = dict(model_plugins) if model_plugins is not None else discover_model_plugins()
    if model_id not in registry:
        raise ValueError(f"no model plugin is registered for {model_id!r}")
    plugin = registry[model_id]

    inputs = _resolve_inputs(
        task=task,
        task_path=Path(task_path),
        features_path=Path(features_path) if features_path is not None else None,
        dataset_manifest=Path(dataset_manifest) if dataset_manifest is not None else None,
        preprocessor_id=preprocessor_id,
        recordings=recordings,
        model_plugins=registry,
        model_id=model_id,
    )
    load_seconds = time.perf_counter() - started

    policy: TrainingPolicy
    policy_artifact: dict[str, str] | None = None
    if policy_path is not None:
        policy = load_training_policy(policy_path)
        policy_artifact = {
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
        }
    else:
        policy = (
            DEFAULT_SKLEARN_POLICY if plugin.backend == "sklearn" else DEFAULT_TORCH_POLICY
        )

    train_view = DataView(
        x=inputs.x[inputs.train_idx],
        labels=inputs.labels[inputs.train_idx],
        evaluation_groups=inputs.evaluation_groups[inputs.train_idx],
    )
    validation_view = DataView(
        x=inputs.x[inputs.validation_idx],
        labels=inputs.labels[inputs.validation_idx],
        evaluation_groups=inputs.evaluation_groups[inputs.validation_idx],
    )
    test_x = inputs.x[inputs.test_idx]
    test_labels = inputs.labels[inputs.test_idx]
    test_groups = inputs.evaluation_groups[inputs.test_idx]

    runs: list[dict[str, object]] = []
    for seed in task.seeds:
        try:
            seed_everything(seed, deterministic=policy.deterministic)
            context = ModelBuildContext(
                num_classes=len(task.labels),
                input_channels=inputs.input_dim,
                seed=seed,
            )
            raw_model = plugin.create(context)
            estimator: Estimator
            if plugin.backend == "sklearn":
                estimator = SklearnEstimator(raw_model)
            elif plugin.backend == "pytorch":
                estimator = TorchEstimator(raw_model, labels=task.labels, seed=seed)
            else:  # pragma: no cover - guarded by ModelPlugin validation
                raise ValueError(f"unsupported backend {plugin.backend!r}")

            fit_started = time.perf_counter()
            estimator.fit(train_view, validation_view, policy)
            fit_seconds = time.perf_counter() - fit_started

            validation_predictions = estimator.predict(validation_view.x)
            validation = condition_metrics(
                validation_view.labels,
                validation_predictions,
                validation_view.evaluation_groups,
            )
            test_predictions = estimator.predict(test_x)
            test = condition_metrics(test_labels, test_predictions, test_groups)
            complexity = getattr(estimator, "complexity", lambda: {
                "trainable_parameters": 0,
                "parameter_count_scope": "unavailable for this estimator",
                "serialized_size_bytes": 1,
            })()
            latency = _latency_ms(
                estimator,
                test_x[:1],
                warmup=latency_warmup,
                repeats=latency_repeats,
            )
            runs.append(
                {
                    "seed": seed,
                    "status": "ok",
                    "fit_seconds": fit_seconds,
                    "validation": validation.to_dict(),
                    "test": test.to_dict(),
                    "complexity": complexity,
                    "latency": latency,
                    "test_predictions": {
                        "truth": [str(value) for value in test_labels],
                        "predictions": [str(value) for value in test_predictions],
                        "groups": [str(value) for value in test_groups],
                        "recording_ids": [
                            str(value) for value in inputs.recording_ids[inputs.test_idx]
                        ],
                    },
                }
            )
        except Exception as error:  # noqa: BLE001 - failures are retained per seed
            runs.append(
                {
                    "seed": seed,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}"[:500],
                }
            )

    serialize_started = time.perf_counter()
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if plugin.backend == "pytorch":
        try:
            import torch

            environment["torch"] = torch.__version__
        except ModuleNotFoundError:
            environment["torch"] = "not-installed"

    payload = {
        "schema_version": 2,
        "benchmark_id": BENCHMARK_ID,
        "model_id": model_id,
        "task_id": task.task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "protocol": {
            "task_manifest": str(task_path),
            "feature_source": (
                str(features_path) if features_path is not None else "dataset-manifest"
            ),
            "task_seeds": list(task.seeds),
            "retain_predictions": True,
            "preparation": inputs.preparation,
            "training_policy": policy.to_dict(),
            "training_policy_digest": policy.content_digest(),
            "test_used_for_selection": False,
        },
        "model_plugin": {
            "backend": plugin.backend,
            "source_distribution": plugin.source,
            "source_version": _distribution_version(plugin.source),
        },
        "environment": environment,
        "partition_sizes": {
            "train": int(len(inputs.train_idx)),
            "validation": int(len(inputs.validation_idx)),
            "test": int(len(inputs.test_idx)),
        },
        "input_artifacts": {
            "task_manifest": inputs.input_artifacts["task_manifest"],
            "prepared_feature_table": inputs.input_artifacts["prepared_feature_table"],
            "dataset_manifest": inputs.input_artifacts["dataset_manifest"],
            "training_policy": policy_artifact,
        },
        "framework_overhead": {
            "load_seconds": load_seconds,
            "scope": "kernel load/audit/serialization time; per-seed fit is per run",
        },
        "runs": runs,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = json.dumps(payload, indent=2, sort_keys=True)
    payload["framework_overhead"]["serialize_seconds"] = (
        time.perf_counter() - serialize_started
    )
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
