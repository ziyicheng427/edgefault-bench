"""Isolated batch-one CPU resource benchmark for EdgeFault-Bench models."""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor, nn

from edgefault_bench.experiments.neural import _git_commit
from edgefault_bench.models import build_model, trainable_parameter_count

MODEL_IDS = ("standard_cnn_1d", "compact_depthwise_cnn_1d")


def serialized_state_size(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def multiply_accumulate_count(model: nn.Module, sample: Tensor) -> int:
    """Count Conv1d and Linear multiply-accumulates for one forward pass."""

    total = 0

    def conv_hook(module: nn.Conv1d, _inputs, output: Tensor) -> None:
        nonlocal total
        output_elements = output.numel()
        kernel_ops = module.kernel_size[0] * module.in_channels // module.groups
        total += output_elements * kernel_ops

    def linear_hook(module: nn.Linear, _inputs, output: Tensor) -> None:
        nonlocal total
        total += output.numel() * module.in_features

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        with torch.inference_mode():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
    return total


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _cpu_name() -> str:
    name = platform.processor()
    if name:
        return name
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return "unknown"


def benchmark_model(
    model_id: str, *, warmup: int = 50, repeats: int = 1000
) -> dict[str, object]:
    if model_id not in MODEL_IDS:
        raise ValueError(f"unsupported hardware model: {model_id}")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = build_model(model_id, num_classes=4).cpu().eval()
    sample = torch.linspace(-1.0, 1.0, 4096, dtype=torch.float32).reshape(1, 1, -1)
    macs = multiply_accumulate_count(model, sample)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        durations_ns = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            model(sample)
            durations_ns.append(time.perf_counter_ns() - started)
    durations_ms = [value / 1_000_000.0 for value in durations_ns]
    return {
        "model_id": model_id,
        "input_shape": list(sample.shape),
        "trainable_parameters": trainable_parameter_count(model),
        "serialized_state_bytes": serialized_state_size(model),
        "multiply_accumulates": macs,
        "latency": {
            "device": "cpu",
            "threads": 1,
            "batch_size": 1,
            "warmup": warmup,
            "repeats": repeats,
            "median_ms": statistics.median(durations_ms),
            "p95_ms": sorted(durations_ms)[int(0.95 * (repeats - 1))],
        },
        "isolated_process_peak_rss_bytes": _peak_rss_bytes(),
        "memory_scope": "entire isolated Python/PyTorch inference process",
    }


def run_isolated(*, output_path: Path, warmup: int, repeats: int) -> Path:
    measurements = []
    for model_id in MODEL_IDS:
        command = [
            sys.executable,
            "-m",
            "edgefault_bench.hardware",
            "--worker",
            "--model",
            model_id,
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        measurements.append(json.loads(completed.stdout))
    payload = {
        "schema_version": 1,
        "benchmark_id": "edgefault-bench-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "protocol": {
            "independent_process_per_model": True,
            "synthetic_input": "deterministic linear ramp",
            "trained_weights_required": False,
            "operator_scope": (
                "Conv1d and Linear MACs; pooling, normalization, activations excluded"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": _cpu_name(),
            "logical_cpu_count": os.cpu_count(),
            "torch": torch.__version__,
        },
        "measurements": measurements,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/v1/hardware/cpu.json"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=MODEL_IDS, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.worker:
        if args.model is None:
            raise SystemExit("--worker requires --model")
        print(json.dumps(benchmark_model(args.model, warmup=args.warmup, repeats=args.repeats)))
        return
    print(run_isolated(output_path=args.output, warmup=args.warmup, repeats=args.repeats))


if __name__ == "__main__":
    main()
