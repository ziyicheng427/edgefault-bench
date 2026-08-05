"""Unified command-line interface for the EdgeFault-Bench research workflow."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from edgefault_bench.audit import main as audit_main
from edgefault_bench.datasets import HustV3Adapter, MehranV2Adapter
from edgefault_bench.download import main as download_hust_main
from edgefault_bench.download_mehran import main as download_mehran_main
from edgefault_bench.reporting import load_result


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
    dataset_id = _manifest_payload(manifest)["dataset_id"]
    if dataset_id == "hust-bearing-v3":
        return HustV3Adapter(manifest)
    if dataset_id == "mehran-triaxial-bearing-v2":
        return MehranV2Adapter(manifest)
    raise ValueError(f"no dataset adapter is registered for {dataset_id!r}")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgefault",
        description="Audit and run reproducible condition-shift fault-diagnosis benchmarks.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

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
