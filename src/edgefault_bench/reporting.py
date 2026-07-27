"""Validate result provenance and render a deterministic Markdown summary."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

EXPECTED_SEEDS = (17, 29, 43)


def _mean_sd(values: list[float]) -> str:
    return f"{statistics.mean(values):.4f} ± {statistics.pstdev(values):.4f}"


def load_result(path: Path, *, expected_seeds: tuple[int, ...] = EXPECTED_SEEDS) -> dict:
    """Load a result and reject missing, duplicated, or reordered predefined seeds."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"task_id", "model_id", "git_commit", "runs", "environment"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")
    seeds = tuple(run.get("seed") for run in payload["runs"])
    if seeds != expected_seeds:
        raise ValueError(f"{path}: expected seeds {expected_seeds}, found {seeds}")
    if payload["git_commit"] == "unknown" or len(payload["git_commit"]) < 7:
        raise ValueError(f"{path}: result is not bound to a valid Git commit")
    for run in payload["runs"]:
        for section in ("validation", "test", "complexity", "latency"):
            if section not in run:
                raise ValueError(f"{path}: seed {run['seed']} is missing {section}")
    return payload


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/v1"))
    parser.add_argument("--output", type=Path, default=Path("results/v1/SUMMARY.md"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = sorted(args.results_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no result JSON files found in {args.results_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summarize(paths), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
