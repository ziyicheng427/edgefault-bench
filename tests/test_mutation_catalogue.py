"""Mutation catalogue tests (T11): every planted fault must be caught."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import edgefault_bench.cli as cli
from edgefault_bench import mutation
from edgefault_bench.mutation import MutationOutcome, mutation_audit, run_mutation_catalogue

CATALOGUE_IDS = {
    "M-LEAK-01", "M-LEAK-02", "M-LEAK-03", "M-LEAK-04",
    "M-HASH-01",
    "M-SEED-01", "M-SEED-02",
    "M-PROV-01", "M-PROV-02",
    "M-PLUGIN-01", "M-PLUGIN-02", "M-PLUGIN-03",
}


def test_catalogue_matches_formal_spec_fault_list() -> None:
    assert set(mutation.MUTATIONS) == CATALOGUE_IDS


def test_every_mutation_is_detected_fail_closed(tmp_path: Path) -> None:
    outcomes = run_mutation_catalogue(tmp_path)

    assert {outcome.mutation_id for outcome in outcomes} == CATALOGUE_IDS
    undetected = [o.mutation_id for o in outcomes if not o.detected]
    assert not undetected, f"mutations escaped detection: {undetected}"
    for outcome in outcomes:
        assert outcome.mechanism != "none"
        assert outcome.target_invariant.startswith("INV-")


def test_detection_matrix_reports_undetected_faults_honestly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reporter must be able to say 'undetected' when a guard is broken."""

    def sabotaged(_workdir: Path) -> MutationOutcome:
        return MutationOutcome(
            "M-PLUGIN-02", "INV-6", "sabotaged scenario",
            False, "none", "canary: mechanism disabled",
        )

    monkeypatch.setitem(mutation.MUTATIONS, "M-PLUGIN-02", sabotaged)
    report = mutation_audit(tmp_path)
    assert report["passed"] is False
    assert report["detected"] == len(CATALOGUE_IDS) - 1
    assert report["undetected"] == ["M-PLUGIN-02"]


def test_cli_writes_detection_matrix(tmp_path: Path, capsys) -> None:
    output = tmp_path / "matrix.json"
    exit_code = cli.main(["mutation", "audit", "--output", str(output)])
    rendered = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert rendered["passed"] is True
    assert rendered["total"] == len(CATALOGUE_IDS)
    assert rendered["detected"] == len(CATALOGUE_IDS)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["passed"] is True
    assert {entry["mutation_id"] for entry in persisted["mutations"]} == CATALOGUE_IDS
