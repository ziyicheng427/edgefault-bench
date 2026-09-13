"""Access and validation for packaged EdgeFault-Bench JSON Schemas."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_FILES = {
    "prepared-feature-table": "prepared-feature-table-v1.schema.json",
    "result": "result-v1.schema.json",
    "task": "task-v1.schema.json",
}


class ArtifactSchemaError(ValueError):
    """Raised with stable JSON paths when an artifact violates its schema."""


def schema_names() -> tuple[str, ...]:
    return tuple(sorted(SCHEMA_FILES))


def load_schema(name: str) -> dict[str, object]:
    """Load one schema from package resources and verify its own structure."""

    try:
        filename = SCHEMA_FILES[name]
    except KeyError as error:
        raise ValueError(f"unknown schema {name!r}; choose from {schema_names()}") from error
    resource = files("edgefault_bench.schema_specs").joinpath(filename)
    schema = json.loads(resource.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


def validate_artifact_schema(path: str | Path, schema_name: str) -> dict[str, object]:
    """Validate a JSON document and return its decoded object on success."""

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    schema = load_schema(schema_name)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.absolute_path))
    if errors:
        rendered = []
        for error in errors:
            location = "$"
            for component in error.absolute_path:
                location += f"[{component}]" if isinstance(component, int) else f".{component}"
            rendered.append(f"{location}: {error.message}")
        raise ArtifactSchemaError(f"{artifact_path}: " + "; ".join(rendered))
    if not isinstance(payload, dict):
        raise ArtifactSchemaError(f"{artifact_path}: root must be an object")
    return payload
