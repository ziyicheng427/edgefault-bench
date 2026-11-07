"""Validate result provenance and render a deterministic Markdown summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

EXPECTED_SEEDS = (17, 29, 43)

_V1_REQUIRED = {"task_id", "model_id", "git_commit", "runs", "environment"}
_V2_REQUIRED = _V1_REQUIRED | {"protocol", "model_plugin", "input_artifacts"}


def _mean_sd(values: list[float]) -> str:
    return f"{statistics.mean(values):.4f} ± {statistics.pstdev(values):.4f}"


def _require_fields(payload: dict, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")


def _check_commit(payload: dict, path: Path) -> None:
    if payload["git_commit"] == "unknown" or len(payload["git_commit"]) < 7:
        raise ValueError(f"{path}: result is not bound to a valid Git commit")


def load_result(path: Path, *, expected_seeds: tuple[int, ...] = EXPECTED_SEEDS) -> dict:
    """Load a v1 or v2 result and reject incomplete seed provenance.

    v1 bundles keep the frozen 17/29/43 seed order. v2 bundles carry their own
    registered seed list in ``protocol.task_seeds`` and may retain failed runs
    as explicit placeholders instead of dropping seeds (INV-4).
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("schema_version", 1)
    if version == 1:
        _require_fields(payload, _V1_REQUIRED, path)
        seeds = tuple(run.get("seed") for run in payload["runs"])
        if seeds != expected_seeds:
            raise ValueError(f"{path}: expected seeds {expected_seeds}, found {seeds}")
        _check_commit(payload, path)
        for run in payload["runs"]:
            for section in ("validation", "test", "complexity", "latency"):
                if section not in run:
                    raise ValueError(f"{path}: seed {run['seed']} is missing {section}")
        return payload
    if version == 2:
        _require_fields(payload, _V2_REQUIRED, path)
        _check_commit(payload, path)
        registered = tuple(payload.get("protocol", {}).get("task_seeds", ()))
        if not registered:
            raise ValueError(f"{path}: v2 result does not declare protocol.task_seeds")
        seeds = tuple(run.get("seed") for run in payload["runs"])
        if seeds != registered:
            raise ValueError(
                f"{path}: runs {list(seeds)} do not biject with registered seeds "
                f"{list(registered)}"
            )
        for run in payload["runs"]:
            status = run.get("status")
            if status == "ok":
                for section in ("validation", "test", "complexity", "latency"):
                    if section not in run:
                        raise ValueError(
                            f"{path}: ok run for seed {run['seed']} is missing {section}"
                        )
            elif status == "failed":
                if not run.get("error"):
                    raise ValueError(
                        f"{path}: failed run for seed {run['seed']} has no error record"
                    )
            else:
                raise ValueError(
                    f"{path}: run for seed {run['seed']} has invalid status {status!r}"
                )
        return payload
    raise ValueError(f"{path}: unsupported schema_version {version!r}")


def verify_input_artifact_hashes(path: Path) -> dict:
    """Recompute the SHA-256 of every referenced input artifact (INV-5, M-PROV-02)."""

    payload = load_result(path)
    artifacts = payload.get("input_artifacts", {})
    if not artifacts:
        raise ValueError(f"{path}: result predates content-hashed input artifacts")
    outcomes: dict[str, dict[str, object]] = {}
    passed = True
    for name in sorted(artifacts):
        reference = artifacts[name]
        if reference is None:
            outcomes[name] = {"status": "not-applicable"}
            continue
        artifact_path = Path(reference["path"])
        if not artifact_path.is_file():
            outcomes[name] = {
                "status": "missing",
                "path": str(artifact_path),
                "expected_sha256": reference["sha256"],
            }
            passed = False
            continue
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if digest != reference["sha256"]:
            outcomes[name] = {
                "status": "mismatch",
                "path": str(artifact_path),
                "expected_sha256": reference["sha256"],
                "actual_sha256": digest,
            }
            passed = False
        else:
            outcomes[name] = {"status": "verified", "sha256": digest}
    return {"passed": passed, "artifacts": outcomes}


def summarize(paths: list[Path]) -> str:
    """Create a stable table from raw result JSON files."""

    rows: list[tuple[str, ...]] = []
    environments: set[str] = set()
    for path in sorted(paths):
        payload = load_result(path)
        runs = payload["runs"]
        macro = [float(run["test"]["macro_f1"]) for run in runs]
        worst = [float(run["test"]["worst_condition_macro_f1"]) for run in runs]
        parameters = [int(run["complexity"]["trainable_parameters"]) for run in runs]
        sizes = [int(run["complexity"]["serialized_size_bytes"]) for run in runs]
        latency = [float(run["latency"]["median_ms"]) for run in runs]
        environment = payload["environment"]
        platform_name = environment.get("platform", "unknown")
        python_version = environment.get("python", "unknown")
        environments.add(f"{platform_name} / Python {python_version}")
        rows.append(
            (
                payload["task_id"],
                payload["model_id"],
                _mean_sd(macro),
                _mean_sd(worst),
                str(round(statistics.mean(parameters))),
                str(round(statistics.mean(sizes))),
                _mean_sd(latency),
                payload["git_commit"][:7],
            )
        )

    lines = [
        "# EdgeFault-Bench result summary",
        "",
        "This file is generated from the committed raw JSON files. Scores and latency are",
        "population mean ± standard deviation over the fixed seeds 17, 29, and 43.",
        "Latency is batch-one median CPU latency per run; it is not an edge-device claim.",
        "",
        "| Task | Model | Test Macro-F1 | Worst-condition Macro-F1 | Parameters | "
        "Serialized bytes | Median latency (ms) | Code commit |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(["", "## Recorded execution environments", ""])
    lines.extend(f"- {item}" for item in sorted(environments))
    lines.append("")
    return "\n".join(lines)


def summarize_robustness(paths: list[Path]) -> str:
    """Validate completed robustness matrices and render their degradation table."""

    rows: list[tuple[str, ...]] = []
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("complete"):
            raise ValueError(f"{path}: robustness result is incomplete")
        if payload.get("git_commit") in (None, "unknown"):
            raise ValueError(f"{path}: robustness result has no Git provenance")
        experiments = payload.get("experiments", [])
        seeds = tuple(payload.get("seeds", ()))
        if seeds != EXPECTED_SEEDS:
            raise ValueError(f"{path}: unexpected robustness seeds {seeds}")
        if payload["track"] == "measurement_noise":
            if tuple(item["seed"] for item in experiments) != EXPECTED_SEEDS:
                raise ValueError(f"{path}: incomplete measurement-noise seed matrix")
            for level in ("clean", "20_db", "10_db", "0_db"):
                metrics = [item["test_by_snr"][level] for item in experiments]
                rows.append(
                    (
                        payload["track"],
                        level,
                        "full",
                        _mean_sd([item["macro_f1"] for item in metrics]),
                        _mean_sd([item["worst_condition_macro_f1"] for item in metrics]),
                        payload["git_commit"][:7],
                    )
                )
        else:
            for level in payload["levels"]:
                selected = [item for item in experiments if item["level"] == level]
                if tuple(item["seed"] for item in selected) != EXPECTED_SEEDS:
                    raise ValueError(f"{path}: incomplete {payload['track']} level {level}")
                mean_train_samples = round(
                    statistics.mean(item["train_sample_count"] for item in selected)
                )
                rows.append(
                    (
                        payload["track"],
                        f"{float(level):g}",
                        str(mean_train_samples),
                        _mean_sd([item["run"]["test"]["macro_f1"] for item in selected]),
                        _mean_sd(
                            [
                                item["run"]["test"]["worst_condition_macro_f1"]
                                for item in selected
                            ]
                        ),
                        payload["git_commit"][:7],
                    )
                )
    lines = [
        "# EdgeFault-Bench robustness summary",
        "",
        "Generated from completed three-seed robustness JSON files. Values are population",
        "mean ± standard deviation over seeds 17, 29, and 43.",
        "",
        "| Track | Level | Training samples | Test Macro-F1 | "
        "Worst-condition Macro-F1 | Code commit |",
        "|---|---:|---:|---:|---:|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    return "\n".join(lines)


def summarize_hardware(path: Path) -> str:
    """Render repeated isolated-process measurements without hiding repetitions."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    repeats = int(payload["protocol"]["process_repeats_per_model"])
    rows = []
    for model_id in sorted({item["model_id"] for item in payload["measurements"]}):
        selected = [item for item in payload["measurements"] if item["model_id"] == model_id]
        if len(selected) != repeats:
            raise ValueError(f"{path}: expected {repeats} measurements for {model_id}")
        first = selected[0]
        median_rss_mib = statistics.median(
            item["isolated_process_peak_rss_bytes"] for item in selected
        ) / 2**20
        rows.append(
            (
                model_id,
                str(first["trainable_parameters"]),
                str(first["serialized_state_bytes"]),
                str(first["multiply_accumulates"]),
                f"{statistics.median(item['latency']['median_ms'] for item in selected):.4f}",
                f"{statistics.median(item['latency']['p95_ms'] for item in selected):.4f}",
                f"{median_rss_mib:.1f}",
            )
        )
    environment = payload["environment"]
    lines = [
        "# EdgeFault-Bench hardware summary",
        "",
        f"Measured on {environment['cpu']} ({environment['machine']}) with "
        f"PyTorch {environment['torch']}. Values are medians across {repeats} independent",
        "processes; each process contains 1,000 timed batch-one calls after warm-up.",
        "",
        "| Model | Parameters | Serialized bytes | MACs | Median latency (ms) | "
        "p95 latency (ms) | Peak process RSS (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(
        [
            "",
            "RSS covers the entire isolated Python/PyTorch process. MACs cover Conv1d and",
            "Linear operations only. See `docs/hardware-benchmark.md` for boundaries.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/v1"))
    parser.add_argument("--output", type=Path, default=Path("results/v1/SUMMARY.md"))
    parser.add_argument(
        "--robustness-output", type=Path, default=Path("results/v1/ROBUSTNESS.md")
    )
    parser.add_argument("--hardware-output", type=Path, default=Path("results/v1/HARDWARE.md"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = sorted(args.results_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no result JSON files found in {args.results_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summarize(paths), encoding="utf-8")
    print(args.output)
    robustness_paths = sorted((args.results_dir / "robustness").glob("*.json"))
    if robustness_paths:
        args.robustness_output.write_text(
            summarize_robustness(robustness_paths), encoding="utf-8"
        )
        print(args.robustness_output)
    hardware_paths = sorted((args.results_dir / "hardware").glob("*.json"))
    if len(hardware_paths) == 1:
        args.hardware_output.write_text(
            summarize_hardware(hardware_paths[0]), encoding="utf-8"
        )
        print(args.hardware_output)
    elif len(hardware_paths) > 1:
        raise SystemExit("expected at most one v1 hardware JSON file")


if __name__ == "__main__":
    main()
