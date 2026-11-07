"""Raw-data directory resolution shared by dataset adapters and downloaders."""

from __future__ import annotations

import os
from pathlib import Path

RAW_ROOT_ENV = "EDGEFAULT_RAW_DIR"
DEFAULT_RAW_ROOT = "data/raw"


def resolve_raw_dir(dataset_dirname: str) -> Path:
    """Return the raw directory for one dataset, honouring the environment override.

    The ``EDGEFAULT_RAW_DIR`` variable replaces the ``data/raw`` root while the
    dataset-specific directory name (for example ``hust_v3``) stays fixed, so
    tests and reproducers can relocate downloads without editing manifests.
    """

    if not dataset_dirname.strip():
        raise ValueError("dataset_dirname must be non-empty")
    root = os.environ.get(RAW_ROOT_ENV, DEFAULT_RAW_ROOT)
    return Path(root) / dataset_dirname
