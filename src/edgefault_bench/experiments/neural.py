"""Neural baselines and lightweight CORAL domain generalization for HUST v1 tasks."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import platform
import random
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from edgefault_bench.evaluation import condition_metrics
from edgefault_bench.models import build_model, coral_loss, trainable_parameter_count
from edgefault_bench.tasks import load_task_manifest, split_window_records
from edgefault_bench.torch_data import LABELS, encode_domains, load_tensor_table

MODEL_IDS = ("standard_cnn_1d", "compact_depthwise_cnn_1d", "compact_coral_cnn_1d")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict(model: nn.Module, signals: Tensor, *, device: torch.device, batch_size: int) -> Tensor:
    model.eval()
    predictions: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(signals), batch_size):
            logits = model(signals[start : start + batch_size].to(device))
            predictions.append(logits.argmax(dim=1).cpu())
    return torch.cat(predictions)


def _score(
    model: nn.Module,
    signals: Tensor,
    targets: Tensor,
    groups: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
):
    predictions = _predict(model, signals, device=device, batch_size=batch_size).numpy()
    truth_names = np.asarray([LABELS[index] for index in targets.numpy()])
    prediction_names = np.asarray([LABELS[index] for index in predictions])
    return condition_metrics(truth_names, prediction_names, groups)


def _cpu_latency(model: nn.Module, sample: Tensor, *, warmup: int = 25, repeats: int = 200):
    model = copy.deepcopy(model).cpu().eval()
    torch.set_num_threads(1)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        durations = np.empty(repeats, dtype=np.float64)
        for index in range(repeats):
            started = time.perf_counter_ns()
            model(sample)
            durations[index] = (time.perf_counter_ns() - started) / 1_000_000.0
    return {
        "device": "cpu",
        "threads": 1,
        "batch_size": 1,
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
    }


def _serialized_state_size(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def train_one_seed(
    *,
    model_id: str,
    seed: int,
    signals: Tensor,
    targets: Tensor,
    source_domains: Tensor,
    evaluation_groups: np.ndarray,
    train_indices: Tensor,
    validation_indices: Tensor,
    test_indices: Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    coral_weight: float,
    checkpoint_path: Path,
) -> dict[str, object]:
    _set_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = build_model(model_id, num_classes=len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    train_dataset = TensorDataset(
        signals[train_indices], targets[train_indices], source_domains[train_indices]
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )

    best_validation = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    use_coral = model_id == "compact_coral_cnn_1d"
    validation_groups = evaluation_groups[validation_indices.numpy()]

    for epoch in range(1, epochs + 1):
        model.train()
        classification_total = 0.0
        coral_total = 0.0
        sample_count = 0
        for batch_signals, batch_targets, batch_domains in loader:
            batch_signals = batch_signals.to(device)
            batch_targets = batch_targets.to(device)
            batch_domains = batch_domains.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings = model(batch_signals, return_embedding=True)
            classification = criterion(logits, batch_targets)
            domain_penalty = (
                coral_loss(embeddings, batch_domains) if use_coral else embeddings.sum() * 0.0
            )
            loss = classification + coral_weight * domain_penalty
            loss.backward()
            optimizer.step()
            batch_count = len(batch_targets)
            classification_total += float(classification.detach()) * batch_count
            coral_total += float(domain_penalty.detach()) * batch_count
            sample_count += batch_count

        validation = _score(
            model,
            signals[validation_indices],
            targets[validation_indices],
            validation_groups,
            device=device,
            batch_size=batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "classification_loss": classification_total / sample_count,
                "coral_loss": coral_total / sample_count,
                "validation_macro_f1": validation.macro_f1,
            }
        )
        if validation.macro_f1 > best_validation + 1e-8:
            best_validation = validation.macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    fit_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("training did not produce a selectable model state")
    model.load_state_dict(best_state)
    validation = _score(
        model,
        signals[validation_indices],
        targets[validation_indices],
        validation_groups,
        device=device,
        batch_size=batch_size,
    )
    test_groups = evaluation_groups[test_indices.numpy()]
    test = _score(
        model,
        signals[test_indices],
        targets[test_indices],
        test_groups,
        device=device,
        batch_size=batch_size,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": model_id,
            "seed": seed,
            "labels": LABELS,
            "state_dict": best_state,
        },
        checkpoint_path,
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "fit_seconds": fit_seconds,
        "history": history,
        "validation": validation.to_dict(),
        "test": test.to_dict(),
        "complexity": {
            "trainable_parameters": trainable_parameter_count(model),
            "serialized_size_bytes": _serialized_state_size(model),
            "process_peak_rss_bytes": _peak_rss_bytes(),
        },
        "latency": _cpu_latency(model, signals[test_indices[:1]]),
        "checkpoint": str(checkpoint_path),
    }


def run_task(
    *,
    model_id: str,
    task_path: Path,
    manifest_path: Path,
    raw_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    device_name: str = "cpu",
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 4,
    coral_weight: float = 0.1,
) -> Path:
    if model_id not in MODEL_IDS:
        raise ValueError(f"unsupported model: {model_id}")
    task = load_task_manifest(task_path)
    table = load_tensor_table(manifest_path, raw_dir)
    selected, split = split_window_records(table.records, task)
    if len(selected) != len(table.records):
        raise RuntimeError("task selection changed the frozen tensor row set")
    source_domains = encode_domains(table.records, task.evaluation_group_field)
    evaluation_groups = np.asarray(
        [str(record.domain_value(task.evaluation_group_field)) for record in table.records]
    )
    indices = {
        "train": torch.from_numpy(split.train),
        "validation": torch.from_numpy(split.validation),
        "test": torch.from_numpy(split.test),
    }
    device = torch.device(device_name)
    runs = [
        train_one_seed(
            model_id=model_id,
            seed=seed,
            signals=table.signals,
            targets=table.targets,
            source_domains=source_domains,
            evaluation_groups=evaluation_groups,
            train_indices=indices["train"],
            validation_indices=indices["validation"],
            test_indices=indices["test"],
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            coral_weight=coral_weight,
            checkpoint_path=checkpoint_dir / f"{task.task_id}__{model_id}__seed{seed}.pt",
        )
        for seed in task.seeds
    ]
    payload = {
        "schema_version": 1,
        "benchmark_id": "edgefault-bench-v1",
        "model_id": model_id,
        "task_id": task.task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "protocol": {
            "task_manifest": str(task_path),
            "dataset_manifest": str(manifest_path),
            "test_used_for_selection": False,
            "source_alignment_field": task.evaluation_group_field if "coral" in model_id else None,
        },
        "training": {
            "device": device_name,
            "epochs_max": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "coral_weight": coral_weight if "coral" in model_id else 0.0,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "torch": torch.__version__,
        },
        "partition_sizes": {name: len(value) for name, value in indices.items()},
        "runs": runs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task.task_id}__{model_id}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_IDS)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("registry/hust_v3.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hust_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v1"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = run_task(
        model_id=args.model,
        task_path=args.task,
        manifest_path=args.manifest,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(output)


if __name__ == "__main__":
    main()
