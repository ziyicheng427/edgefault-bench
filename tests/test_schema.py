import json
from pathlib import Path

import pytest

from edgefault_bench.schema import (
    ArtifactSchemaError,
    load_schema,
    schema_names,
    validate_artifact_schema,
)

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_schemas_are_valid_draft_2020_12() -> None:
    assert schema_names() == ("prepared-feature-table", "result", "task")
    for name in schema_names():
        assert load_schema(name)["$schema"] == "https://json-schema.org/draft/2020-12/schema"


@pytest.mark.parametrize("path", sorted((ROOT / "registry/tasks").glob("*.json")))
def test_all_registered_tasks_match_schema(path: Path) -> None:
    validate_artifact_schema(path, "task")


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "results/v1").glob("*.json"))
    + sorted((ROOT / "results/v1.1/mehran").glob("*.json")),
)
def test_all_core_results_match_schema(path: Path) -> None:
    validate_artifact_schema(path, "result")


def test_example_prepared_table_matches_schema() -> None:
    validate_artifact_schema(
        ROOT / "examples/edgefault-example-plugin/prepared_features.json",
        "prepared-feature-table",
    )


def test_error_reports_stable_json_path(tmp_path: Path) -> None:
    path = tmp_path / "invalid-task.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "invalid",
                "dataset_id": "example",
                "domain_field": "load",
                "evaluation_group_field": "device",
                "partitions": {"train": [], "validation": [1], "test": [2]},
                "labels": ["healthy"],
                "window": {"length": 4, "stride": 4, "normalization": "none"},
                "seeds": [17],
                "description": "Invalid empty train partition.",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactSchemaError, match=r"\$\.partitions\.train"):
        validate_artifact_schema(path, "task")
