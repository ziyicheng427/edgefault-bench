"""Portable JSON model export and batch-one edge inference demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from edgefault_bench.models import build_model
from edgefault_bench.torch_data import LABELS

INPUT_LENGTH = 4096
DEFAULT_ASSET = Path("demo/assets/compact_depthwise_seed17.json")


def normalize_window(window: Tensor) -> Tensor:
    """Validate and independently z-score one vibration window."""

    flattened = window.detach().cpu().to(torch.float32).flatten()
    if len(flattened) != INPUT_LENGTH:
        raise ValueError(f"expected {INPUT_LENGTH} samples, found {len(flattened)}")
    if not bool(torch.isfinite(flattened).all()):
        raise ValueError("input contains non-finite samples")
    centered = flattened - flattened.mean()
    scale = centered.std(correction=0).clamp_min(1e-8)
    return (centered / scale).reshape(1, 1, INPUT_LENGTH)


def synthetic_window() -> Tensor:
    """Create a deterministic non-dataset input for interface demonstration only."""

    time_axis = torch.arange(INPUT_LENGTH, dtype=torch.float32) / 51_200.0
    carrier = torch.sin(2.0 * torch.pi * 1200.0 * time_axis)
    modulation = 0.35 * torch.sin(2.0 * torch.pi * 32.0 * time_axis)
    impulses = (torch.arange(INPUT_LENGTH) % 320 == 0).to(torch.float32)
    return carrier * (1.0 + modulation) + impulses


def load_window(path: Path | None) -> tuple[Tensor, str]:
    if path is None:
        return synthetic_window(), "synthetic_demo"
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".csv":
        values = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError("input must be a .npy or .csv file")
    return torch.as_tensor(values, dtype=torch.float32), str(path)


def _state_to_json(state_dict: dict[str, Tensor]) -> dict[str, dict[str, object]]:
    payload = {}
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu()
        if tensor.dtype not in (torch.float32, torch.int64):
            raise ValueError(f"unsupported state tensor dtype for {name}: {tensor.dtype}")
        payload[name] = {
            "dtype": "float32" if tensor.dtype == torch.float32 else "int64",
            "shape": list(tensor.shape),
            "values": tensor.flatten().tolist(),
        }
    return payload


def _state_from_json(payload: dict[str, dict[str, object]]) -> dict[str, Tensor]:
    state = {}
    for name, specification in payload.items():
        dtypes = {"float32": torch.float32, "int64": torch.int64}
        if specification["dtype"] not in dtypes:
            raise ValueError(f"unsupported tensor dtype for {name}")
        shape = tuple(int(value) for value in specification["shape"])
        tensor = torch.tensor(specification["values"], dtype=dtypes[specification["dtype"]])
        state[name] = tensor.reshape(shape)
    return state


def export_asset(
    *, source_checkpoint: Path, source_result: Path, output_path: Path, seed: int = 17
) -> Path:
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    result = json.loads(source_result.read_text(encoding="utf-8"))
    model_id = checkpoint["model_id"]
    if model_id != "compact_depthwise_cnn_1d" or checkpoint["seed"] != seed:
        raise ValueError("source checkpoint does not match the frozen compact seed-17 demo")
    matching_runs = [run for run in result["runs"] if run["seed"] == seed]
    if result["model_id"] != model_id or len(matching_runs) != 1:
        raise ValueError("source result does not contain the matching model and seed")
    payload = {
        "schema_version": 1,
        "asset_id": "edgefault-compact-depthwise-hust-load-0-to-400-seed17",
        "model_id": model_id,
        "task_id": result["task_id"],
        "seed": seed,
        "labels": list(checkpoint["labels"]),
        "input_length": INPUT_LENGTH,
        "normalization": "per_window_zscore",
        "training_git_commit": result["git_commit"],
        "selection_rule": "first pre-registered seed; not selected by test performance",
        "source_dataset": "HUST Bearing v3, DOI 10.17632/cbv7jyx4p9.3, CC BY 4.0",
        "state_dict": _state_to_json(checkpoint["state_dict"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def load_asset(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["input_length"] != INPUT_LENGTH or tuple(payload["labels"]) != LABELS:
        raise ValueError("model asset does not match the v1 inference contract")
    model = build_model(payload["model_id"], num_classes=len(LABELS)).cpu().eval()
    model.load_state_dict(_state_from_json(payload["state_dict"]), strict=True)
    return model, payload


def infer(
    *, asset_path: Path, input_path: Path | None, warmup: int = 20, repeats: int = 200
) -> dict[str, object]:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    torch.set_num_threads(1)
    model, asset = load_asset(asset_path)
    raw_window, input_source = load_window(input_path)
    sample = normalize_window(raw_window)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        durations = []
        logits = model(sample)
        for _ in range(repeats):
            started = time.perf_counter_ns()
            logits = model(sample)
            durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    prediction = int(probabilities.argmax())
    return {
        "asset_id": asset["asset_id"],
        "asset_sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
        "input_source": input_source,
        "input_is_benchmark_evidence": False,
        "predicted_label": LABELS[prediction],
        "confidence": float(probabilities[prediction]),
        "probabilities": {
            label: float(probabilities[index]) for index, label in enumerate(LABELS)
        },
        "latency": {
            "device": "cpu",
            "threads": 1,
            "batch_size": 1,
            "warmup": warmup,
            "repeats": repeats,
            "median_ms": statistics.median(durations),
            "p95_ms": sorted(durations)[int(0.95 * (repeats - 1))],
        },
    }


def export_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the fixed seed-17 demo asset")
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_ASSET)
    args = parser.parse_args(argv)
    print(
        export_asset(
            source_checkpoint=args.source_checkpoint,
            source_result=args.source_result,
            output_path=args.output,
        )
    )


def infer_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            infer(
                asset_path=args.asset,
                input_path=args.input,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            indent=2,
            sort_keys=True,
        )
    )
