"""Frozen robustness tracks for the primary HUST v1 task."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from edgefault_bench.experiments.neural import (
    _git_commit,
    _score,
    train_one_seed,
)
from edgefault_bench.models import build_model
from edgefault_bench.robustness import add_awgn, stratified_fraction_indices
from edgefault_bench.tasks import load_task_manifest, split_window_records
from edgefault_bench.torch_data import LABEL_TO_INDEX, encode_domains, load_tensor_table

TRACK_LEVELS = {
    "label_scarcity": (0.25, 0.1),
    "class_imbalance": (0.5, 0.25),
    "measurement_noise": (None, 20.0, 10.0, 0.0),
}
ROBUSTNESS_MODELS = ("compact_depthwise_cnn_1d",)


def _write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_track(
    *,
    track: str,
    model_id: str,
    task_path: Path,
    manifest_path: Path,
    raw_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
) -> Path:
    if track not in TRACK_LEVELS:
        raise ValueError(f"unsupported track: {track}")
    if model_id not in ROBUSTNESS_MODELS:
        raise ValueError(f"unsupported robustness model: {model_id}")
    task = load_task_manifest(task_path)
    table = load_tensor_table(manifest_path, raw_dir)
    selected, split = split_window_records(table.records, task)
    if len(selected) != len(table.records):
        raise RuntimeError("task selection changed the frozen tensor row set")
    source_domains = encode_domains(table.records, task.evaluation_group_field)
    evaluation_groups = np.asarray(
        [str(record.domain_value(task.evaluation_group_field)) for record in table.records]
    )
    train_indices = torch.from_numpy(split.train)
    validation_indices = torch.from_numpy(split.validation)
    test_indices = torch.from_numpy(split.test)
    output_path = output_dir / f"{task.task_id}__{model_id}__{track}.json"
    payload = {
        "schema_version": 1,
        "benchmark_id": "edgefault-bench-v1",
        "task_id": task.task_id,
        "model_id": model_id,
        "track": track,
        "levels": TRACK_LEVELS[track],
        "seeds": task.seeds,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "complete": False,
        "protocol": {
            "task_manifest": str(task_path),
            "dataset_manifest": str(manifest_path),
            "sampling_after_split": True,
            "minority_label": "IO" if track == "class_imbalance" else None,
            "noise_stage": "post-window-zscore then re-zscore"
            if track == "measurement_noise"
            else None,
            "test_used_for_selection": False,
        },
        "training": {
            "device": "cpu",
            "epochs_max": 15,
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 4,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
        },
        "experiments": [],
    }
    _write(payload, output_path)

    if track == "measurement_noise":
        for seed in task.seeds:
            checkpoint = checkpoint_dir / f"robustness__{task.task_id}__{model_id}__seed{seed}.pt"
            run = train_one_seed(
                model_id=model_id,
                seed=seed,
                signals=table.signals,
                targets=table.targets,
                source_domains=source_domains,
                evaluation_groups=evaluation_groups,
                train_indices=train_indices,
                validation_indices=validation_indices,
                test_indices=test_indices,
                device=torch.device("cpu"),
                epochs=15,
                batch_size=64,
                learning_rate=0.001,
                weight_decay=0.0001,
                patience=4,
                coral_weight=0.0,
                checkpoint_path=checkpoint,
            )
            model = build_model(model_id, num_classes=4)
            checkpoint_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint_payload["state_dict"])
            test_groups = evaluation_groups[test_indices.numpy()]
            noise_results: dict[str, dict] = {"clean": run["test"]}
            clean_test = table.signals[test_indices]
            for snr_db in (20.0, 10.0, 0.0):
                noisy_test = add_awgn(clean_test, snr_db=snr_db, seed=seed + int(snr_db * 10))
                metrics = _score(
                    model,
                    noisy_test,
                    table.targets[test_indices],
                    test_groups,
                    device=torch.device("cpu"),
                    batch_size=64,
                )
                noise_results[f"{snr_db:g}_db"] = metrics.to_dict()
            payload["experiments"].append(
                {"seed": seed, "training_run": run, "test_by_snr": noise_results}
            )
            _write(payload, output_path)
    else:
        for level in TRACK_LEVELS[track]:
            for seed in task.seeds:
                selected_train = stratified_fraction_indices(
                    train_indices,
                    table.targets,
                    source_domains,
                    fraction=level,
                    seed=seed,
                    target_class=LABEL_TO_INDEX["IO"] if track == "class_imbalance" else None,
                )
                checkpoint = checkpoint_dir / (
                    f"robustness__{task.task_id}__{model_id}__{track}-{level:g}__seed{seed}.pt"
                )
                run = train_one_seed(
                    model_id=model_id,
                    seed=seed,
                    signals=table.signals,
                    targets=table.targets,
                    source_domains=source_domains,
                    evaluation_groups=evaluation_groups,
                    train_indices=selected_train,
                    validation_indices=validation_indices,
                    test_indices=test_indices,
                    device=torch.device("cpu"),
                    epochs=15,
                    batch_size=64,
                    learning_rate=0.001,
                    weight_decay=0.0001,
                    patience=4,
                    coral_weight=0.0,
                    checkpoint_path=checkpoint,
                )
                payload["experiments"].append(
                    {
                        "level": level,
                        "seed": seed,
                        "train_sample_count": len(selected_train),
                        "run": run,
                    }
                )
                _write(payload, output_path)
    payload["complete"] = True
    _write(payload, output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, choices=tuple(TRACK_LEVELS))
    parser.add_argument("--model", default=ROBUSTNESS_MODELS[0], choices=ROBUSTNESS_MODELS)
    parser.add_argument(
        "--task", type=Path, default=Path("registry/tasks/hust_load_0_to_400_v1.json")
    )
    parser.add_argument("--manifest", type=Path, default=Path("registry/hust_v3.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hust_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v1/robustness"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("artifacts/robustness-checkpoints")
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = run_track(
        track=args.track,
        model_id=args.model,
        task_path=args.task,
        manifest_path=args.manifest,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(output)


if __name__ == "__main__":
    main()
