"""Access and validation for packaged EdgeFault-Bench JSON Schemas."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_FILES = {
    "prepared-feature-table": "prepared-feature-table-v1.schema.json",
    "prepared-feature-table-v2": "prepared-feature-table-v2.schema.json",
    "result": "result-v1.schema.json",
    "result-v2": "result-v2.schema.json",
    "task": "task-v1.schema.json",
    "task-v2": "task-v2.schema.json",
}

_VERSIONED_KINDS = {
    "prepared-feature-table": {1: "prepared-feature-table", 2: "prepared-feature-table-v2"},
    "result": {1: "result", 2: "result-v2"},
    "task": {1: "task", 2: "task-v2"},
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


def resolve_schema_name(kind: str, payload: dict[str, object]) -> str:
    """Map a versioned artifact kind plus its payload to a concrete schema."""

    versions = _VERSIONED_KINDS.get(kind)
    if versions is None:
        return kind
    declared = payload.get("schema_version")
    if not isinstance(declared, int) or declared not in versions:
        raise ArtifactSchemaError(
            f"schema_version {declared!r} is not one of {sorted(versions)} for kind {kind!r}"
        )
    return versions[declared]


def validate_artifact_schema(path: str | Path, schema_name: str) -> dict[str, object]:
    """Validate a JSON document and return its decoded object on success.

    For the versioned kinds (``task`` stays fixed), the payload's
    ``schema_version`` selects the concrete schema, so a v2 bundle is never
    forced through its v1 predecessor.
    """

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactSchemaError(f"{artifact_path}: root must be an object")
    concrete = resolve_schema_name(schema_name, payload)
    schema = load_schema(concrete)
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
    return payload
