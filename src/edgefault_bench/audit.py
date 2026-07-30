"""Structured pre-training audits for condition-held-out task metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from edgefault_bench.contracts import Recording, TaskSpec


@dataclass(frozen=True)
class AuditCheck:
    """One machine-readable audit assertion."""

    check_id: str
    passed: bool
    message: str
    examples: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "message": self.message,
            "examples": list(self.examples),
        }


@dataclass(frozen=True)
class LeakageReport:
    """Serializable evidence that recording metadata respects a task boundary."""

    task_id: str
    dataset_id: str
    record_count: int
    partition_counts: Mapping[str, int]
    checks: tuple[AuditCheck, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "partition_counts", MappingProxyType(dict(self.partition_counts)))

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    def require_pass(self) -> None:
        failures = [check for check in self.checks if not check.passed]
        if failures:
            summary = "; ".join(f"{check.check_id}: {check.message}" for check in failures)
            raise ValueError(f"leakage audit failed: {summary}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
            "passed": self.passed,
            "record_count": self.record_count,
            "partition_counts": dict(self.partition_counts),
            "checks": [check.to_dict() for check in self.checks],
        }


def _examples(values: Sequence[str], *, limit: int = 5) -> tuple[str, ...]:
    return tuple(sorted(set(values))[:limit])


def audit_recordings(records: Sequence[Recording], task: TaskSpec) -> LeakageReport:
    """Audit canonical recordings before window extraction or model training.

    The report is returned even when checks fail so callers can persist diagnostic evidence.
    Use :meth:`LeakageReport.require_pass` at execution boundaries that must fail closed.
    """

    dataset_mismatches = [
        record.recording_id for record in records if record.dataset_id != task.dataset_id
    ]
    missing_domain: list[str] = []
    missing_group: list[str] = []
    unsupported_labels: list[str] = []
    unassigned: list[str] = []
    partition_counts: Counter[str] = Counter()
    recording_partitions: dict[str, set[str]] = {}
    recording_occurrences: Counter[str] = Counter()

    for record in records:
        recording_occurrences[record.recording_id] += 1
        if task.domain_field not in record.domains:
            missing_domain.append(record.recording_id)
            continue
        if task.evaluation_group_field not in record.domains:
            missing_group.append(record.recording_id)
        if record.label not in task.labels:
            unsupported_labels.append(record.recording_id)
        partition = task.partition_for(record.domains[task.domain_field])
        if partition is None:
            unassigned.append(record.recording_id)
            continue
        partition_counts[partition] += 1
        recording_partitions.setdefault(record.recording_id, set()).add(partition)

    repeated = [name for name, count in recording_occurrences.items() if count > 1]
    cross_partition = [
        name for name, partitions in recording_partitions.items() if len(partitions) > 1
    ]
    empty_partitions = [
        name for name in ("train", "validation", "test") if partition_counts[name] == 0
    ]

    checks = (
        AuditCheck(
            "dataset_identity",
            not dataset_mismatches,
            "all recordings reference the task dataset"
            if not dataset_mismatches
            else f"{len(dataset_mismatches)} recordings reference another dataset",
            _examples(dataset_mismatches),
        ),
        AuditCheck(
            "domain_metadata",
            not missing_domain and not missing_group,
            "all required domain fields are present"
            if not missing_domain and not missing_group
            else (
                f"missing held-out domain on {len(missing_domain)} recordings and "
                f"evaluation group on {len(missing_group)} recordings"
            ),
            _examples(missing_domain + missing_group),
        ),
        AuditCheck(
            "label_support",
            not unsupported_labels,
            "all recording labels belong to the task"
            if not unsupported_labels
            else f"{len(unsupported_labels)} recordings use unsupported labels",
            _examples(unsupported_labels),
        ),
        AuditCheck(
            "domain_assignment",
            not unassigned,
            "all recordings map to an explicit partition"
            if not unassigned
            else f"{len(unassigned)} recordings have unassigned domain values",
            _examples(unassigned),
        ),
        AuditCheck(
            "partition_nonempty",
            not empty_partitions,
            "train, validation, and test contain recordings"
            if not empty_partitions
            else f"empty partitions: {', '.join(empty_partitions)}",
            tuple(empty_partitions),
        ),
        AuditCheck(
            "recording_identity",
            not repeated,
            "recording identifiers are unique"
            if not repeated
            else f"{len(repeated)} recording identifiers are duplicated",
            _examples(repeated),
        ),
        AuditCheck(
            "recording_exclusivity",
            not cross_partition,
            "no recording identifier crosses partitions"
            if not cross_partition
            else f"{len(cross_partition)} recording identifiers cross partitions",
            _examples(cross_partition),
        ),
    )
    return LeakageReport(
        task_id=task.task_id,
        dataset_id=task.dataset_id,
        record_count=len(records),
        partition_counts={name: partition_counts[name] for name in ("train", "validation", "test")},
        checks=checks,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit canonical recording metadata before window extraction or training."
    )
    parser.add_argument("--task", required=True, help="Versioned task JSON")
    parser.add_argument("--dataset-manifest", required=True, help="Pinned dataset registry JSON")
    parser.add_argument("--output", help="Optional path for the JSON audit report")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the registered adapter for a task and emit its structured audit report."""

    from edgefault_bench.datasets import HustV3Adapter, MehranV2Adapter
    from edgefault_bench.tasks import load_task_spec

    args = build_parser().parse_args(argv)
    task = load_task_spec(args.task)
    if task.dataset_id == "hust-bearing-v3":
        adapter = HustV3Adapter(args.dataset_manifest)
    elif task.dataset_id == "mehran-triaxial-bearing-v2":
        adapter = MehranV2Adapter(args.dataset_manifest)
    else:
        raise ValueError(f"no dataset adapter is registered for {task.dataset_id!r}")
    report = audit_recordings(adapter.recordings(), task)
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
