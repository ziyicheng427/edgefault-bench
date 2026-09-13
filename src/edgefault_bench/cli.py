"""Unified command-line interface for the EdgeFault-Bench research workflow."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from edgefault_bench.audit import main as audit_main
from edgefault_bench.download import main as download_hust_main
from edgefault_bench.download_mehran import main as download_mehran_main
from edgefault_bench.plugins import (
    ModelBuildContext,
    discover_dataset_plugins,
    discover_model_plugins,
    load_dataset_adapter,
)
from edgefault_bench.reporting import load_result
from edgefault_bench.runner import run_prepared_benchmark
from edgefault_bench.schema import load_schema, schema_names, validate_artifact_schema


def _version() -> str:
    try:
        return version("edgefault-bench")
    except PackageNotFoundError:
        return "unknown"


def _manifest_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("dataset_id"):
        raise ValueError(f"dataset manifest has no dataset_id: {path}")
    return payload


def _adapter(manifest: Path):
    return load_dataset_adapter(manifest)


def _plugin_list(args: argparse.Namespace) -> int:
    if args.kind == "dataset":
        plugins = discover_dataset_plugins()
        items = [
            {"dataset_id": plugin.dataset_id, "source": plugin.source}
            for plugin in sorted(plugins.values(), key=lambda item: item.dataset_id)
        ]
    else:
        model_plugins = discover_model_plugins()
        items = [
            {
                "model_id": plugin.model_id,
                "backend": plugin.backend,
                "source": plugin.source,
            }
            for plugin in sorted(model_plugins.values(), key=lambda item: item.model_id)
        ]
    print(
        json.dumps(
            {
                "schema_version": 1,
                "plugin_type": args.kind,
                "plugins": items,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _plugin_validate(args: argparse.Namespace) -> int:
    adapter = load_dataset_adapter(args.manifest)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "passed": True,
                "dataset_id": adapter.metadata.dataset_id,
                "recording_count": len(adapter.recordings()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _plugin_validate_model(args: argparse.Namespace) -> int:
    plugins = discover_model_plugins()
    try:
        plugin = plugins[args.model]
    except KeyError as error:
        raise ValueError(f"no model plugin is registered for {args.model!r}") from error
    context = ModelBuildContext(args.num_classes, args.input_channels, args.seed)
    plugin.create(context)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "passed": True,
                "model_id": plugin.model_id,
                "backend": plugin.backend,
                "source": plugin.source,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _dataset_inspect(args: argparse.Namespace) -> int:
    payload = _manifest_payload(args.manifest)
    adapter = _adapter(args.manifest)
    registry_files = payload.get("files", [])
    selection = payload.get("selection", {})
    rendered = {
        "schema_version": 1,
        "metadata": adapter.metadata.to_dict(),
        "registry_file_count": len(registry_files),
        "selected_recording_count": len(adapter.recordings()),
        "excluded_file_count": int(selection.get("excluded_count", 0)),
        "protocol_status": payload.get("protocol_status", "frozen-v1"),
    }
    print(json.dumps(rendered, indent=2, sort_keys=True))
    return 0


def _dataset_fetch(args: argparse.Namespace) -> int:
    payload = _manifest_payload(args.manifest)
    forwarded = ["--manifest", str(args.manifest), "--workers", str(args.workers)]
    if args.raw_dir is not None:
        forwarded.extend(("--raw-dir", str(args.raw_dir)))
    if args.files is not None:
        forwarded.append("--files")
        forwarded.extend(args.files)
    if args.verify_only:
        forwarded.append("--verify-only")
    if args.repair:
        forwarded.append("--repair")
    if payload["dataset_id"] == "hust-bearing-v3":
        download_hust_main(forwarded)
    elif payload["dataset_id"] == "mehran-triaxial-bearing-v2":
        download_mehran_main(forwarded)
    else:
        raise ValueError(f"no dataset downloader is registered for {payload['dataset_id']!r}")
    return 0


def _task_audit(args: argparse.Namespace) -> int:
    forwarded = [
        "--task",
        str(args.task),
        "--dataset-manifest",
        str(args.dataset_manifest),
    ]
    if args.output is not None:
        forwarded.extend(("--output", str(args.output)))
    return audit_main(forwarded)


def _results_validate(args: argparse.Namespace) -> int:
    results = []
    for path in args.paths:
        payload = load_result(path)
        results.append(
            {
                "path": str(path),
                "task_id": payload["task_id"],
                "model_id": payload["model_id"],
                "git_commit": payload["git_commit"],
                "seeds": [run["seed"] for run in payload["runs"]],
            }
        )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "passed": True,
                "result_count": len(results),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _benchmark_run(args: argparse.Namespace) -> int:
    output = run_prepared_benchmark(
        task_path=args.task,
        prepared_path=args.features,
        model_id=args.model,
        output_path=args.output,
        latency_warmup=args.latency_warmup,
        latency_repeats=args.latency_repeats,
    )
    print(output)
    return 0


def _schema_list(args: argparse.Namespace) -> int:
    schemas = []
    for name in schema_names():
        specification = load_schema(name)
        schemas.append(
            {
                "name": name,
                "schema_id": specification["$id"],
                "title": specification["title"],
            }
        )
    print(json.dumps({"schema_version": 1, "schemas": schemas}, indent=2, sort_keys=True))
    return 0


def _schema_validate(args: argparse.Namespace) -> int:
    validated = []
    for path in args.paths:
        payload = validate_artifact_schema(path, args.kind)
        identity = (
            payload.get("task_id")
            or payload.get("dataset_id")
            or payload.get("benchmark_id")
            or path.name
        )
        validated.append({"path": str(path), "identity": identity})
    print(
        json.dumps(
            {
                "schema_version": 1,
                "passed": True,
                "kind": args.kind,
                "artifact_count": len(validated),
                "artifacts": validated,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgefault",
        description="Audit and run reproducible condition-shift fault-diagnosis benchmarks.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    schema = commands.add_parser("schema", help="Inspect or apply packaged JSON Schemas")
    schema_commands = schema.add_subparsers(dest="schema_command", required=True)
    list_schemas = schema_commands.add_parser("list", help="List packaged schema identifiers")
    list_schemas.set_defaults(handler=_schema_list)
    validate_schema = schema_commands.add_parser(
        "validate", help="Validate one or more JSON artifacts against a packaged schema"
    )
    validate_schema.add_argument("--kind", choices=schema_names(), required=True)
    validate_schema.add_argument("paths", nargs="+", type=Path)
    validate_schema.set_defaults(handler=_schema_validate)

    benchmark = commands.add_parser("benchmark", help="Run a versioned benchmark workflow")
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    run = benchmark_commands.add_parser(
        "run", help="Run an installed sklearn plugin on a prepared feature table"
    )
    run.add_argument("--task", required=True, type=Path)
    run.add_argument("--features", required=True, type=Path)
    run.add_argument("--model", required=True)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--latency-warmup", type=int, default=20)
    run.add_argument("--latency-repeats", type=int, default=100)
    run.set_defaults(handler=_benchmark_run)

    plugin = commands.add_parser("plugin", help="Discover or validate dataset plugins")
    plugin_commands = plugin.add_subparsers(dest="plugin_command", required=True)
    list_plugins = plugin_commands.add_parser(
        "list", help="List built-in and installed dataset or model plugins"
    )
    list_plugins.add_argument("--kind", choices=("dataset", "model"), default="dataset")
    list_plugins.set_defaults(handler=_plugin_list)
    validate_plugin = plugin_commands.add_parser(
        "validate", help="Instantiate and validate the plugin selected by a manifest"
    )
    validate_plugin.add_argument("--manifest", required=True, type=Path)
    validate_plugin.set_defaults(handler=_plugin_validate)
    validate_model_plugin = plugin_commands.add_parser(
        "validate-model", help="Instantiate and validate a registered model plugin"
    )
    validate_model_plugin.add_argument("--model", required=True)
    validate_model_plugin.add_argument("--num-classes", type=int, default=4)
    validate_model_plugin.add_argument("--input-channels", type=int, default=1)
    validate_model_plugin.add_argument("--seed", type=int, default=17)
    validate_model_plugin.set_defaults(handler=_plugin_validate_model)

    dataset = commands.add_parser("dataset", help="Inspect or acquire a registered dataset")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    inspect = dataset_commands.add_parser(
        "inspect", help="Validate a manifest and print canonical metadata"
    )
    inspect.add_argument("--manifest", required=True, type=Path)
    inspect.set_defaults(handler=_dataset_inspect)

    fetch = dataset_commands.add_parser(
        "fetch", help="Download or verify registered files with checksums"
    )
    fetch.add_argument("--manifest", required=True, type=Path)
    fetch.add_argument("--raw-dir", type=Path)
    fetch.add_argument("--files", nargs="*")
    fetch.add_argument("--verify-only", action="store_true")
    fetch.add_argument("--repair", action="store_true")
    fetch.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    fetch.set_defaults(handler=_dataset_fetch)

    task = commands.add_parser("task", help="Inspect or audit a versioned task")
    task_commands = task.add_subparsers(dest="task_command", required=True)
    audit = task_commands.add_parser(
        "audit", help="Fail closed on invalid recording partitions"
    )
    audit.add_argument("--task", required=True, type=Path)
    audit.add_argument("--dataset-manifest", required=True, type=Path)
    audit.add_argument("--output", type=Path)
    audit.set_defaults(handler=_task_audit)

    results = commands.add_parser("results", help="Validate result provenance")
    result_commands = results.add_subparsers(dest="results_command", required=True)
    validate = result_commands.add_parser(
        "validate", help="Validate registered seeds and required result sections"
    )
    validate.add_argument("paths", nargs="+", type=Path)
    validate.set_defaults(handler=_results_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
