import json
from pathlib import Path

import edgefault_bench.cli as cli

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_inspect_exposes_canonical_mehran_metadata(capsys) -> None:
    exit_code = cli.main(
        [
            "dataset",
            "inspect",
            "--manifest",
            str(ROOT / "registry/mehran_v2.json"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["metadata"]["dataset_id"] == "mehran-triaxial-bearing-v2"
    assert payload["registry_file_count"] == 38
    assert payload["selected_recording_count"] == 36
    assert payload["excluded_file_count"] == 2
    assert payload["protocol_status"] == "accepted"


def test_dataset_fetch_dispatches_without_removing_legacy_commands(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    monkeypatch.setattr(cli, "download_mehran_main", lambda argv: calls.append(argv))

    exit_code = cli.main(
        [
            "dataset",
            "fetch",
            "--manifest",
            str(ROOT / "registry/mehran_v2.json"),
            "--raw-dir",
            str(tmp_path),
            "--files",
            "0.7inner-100watt-67V2Iv.csv",
            "--verify-only",
            "--workers",
            "1",
        ]
    )

    assert exit_code == 0
    assert calls == [
        [
            "--manifest",
            str(ROOT / "registry/mehran_v2.json"),
            "--workers",
            "1",
            "--raw-dir",
            str(tmp_path),
            "--files",
            "0.7inner-100watt-67V2Iv.csv",
            "--verify-only",
        ]
    ]


def test_task_audit_is_available_through_unified_cli(capsys) -> None:
    exit_code = cli.main(
        [
            "task",
            "audit",
            "--task",
            str(ROOT / "registry/tasks/mehran_load_100_to_300_v1.json"),
            "--dataset-manifest",
            str(ROOT / "registry/mehran_v2.json"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["passed"] is True
    assert payload["partition_counts"] == {"train": 12, "validation": 12, "test": 12}


def test_results_validate_reports_provenance(capsys) -> None:
    result = (
        ROOT
        / "results/v1.1/mehran/mehran-load-100-to-300-v1__signal_features_logreg.json"
    )
    exit_code = cli.main(["results", "validate", str(result)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["passed"] is True
    assert payload["result_count"] == 1
    assert payload["results"][0]["seeds"] == [17, 29, 43]
