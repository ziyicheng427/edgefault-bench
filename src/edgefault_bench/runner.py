"""Backward-compatible entry module for prepared-table benchmark runs.

The v1 single-purpose runner has been absorbed into the unified backend
execution in :mod:`edgefault_bench.execution`. This module keeps the historic
``run_prepared_benchmark`` signature working while emitting the versioned
schema-v2 result bundle.
"""

from __future__ import annotations

from pathlib import Path

from edgefault_bench.execution import (  # noqa: F401 - re-exported for compatibility
    DEFAULT_SKLEARN_POLICY,
    DEFAULT_TORCH_POLICY,
    TrainingPolicy,
    _git_commit,
    load_training_policy,
    run_benchmark,
    seed_everything,
    sha256_bytes,
    sha256_file,
    write_training_policy,
)


def run_prepared_benchmark(
    *,
    task_path: Path,
    prepared_path: Path,
    model_id: str,
    output_path: Path,
    latency_warmup: int = 20,
    latency_repeats: int = 100,
    plugins: dict | None = None,
) -> Path:
    """Fit one installed plugin per frozen seed on a prepared feature table."""

    return run_benchmark(
        task_path=Path(task_path),
        model_id=model_id,
        output_path=Path(output_path),
        features_path=Path(prepared_path),
        model_plugins=plugins,
        latency_warmup=latency_warmup,
        latency_repeats=latency_repeats,
    )
