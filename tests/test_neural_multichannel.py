import json
from pathlib import Path

import torch

import edgefault_bench.experiments.neural as neural
from edgefault_bench.datasets.mehran import MehranWindowRecord
from edgefault_bench.torch_data import TensorTable

ROOT = Path(__file__).resolve().parents[1]


def test_neural_runner_uses_task_labels_and_three_input_channels(
    tmp_path: Path, monkeypatch
) -> None:
    records = []
    targets = []
    for load_w in (100, 200, 300):
        for label_index, label in enumerate(("inner_race", "outer_race")):
            for size in (0.7, 0.9):
                filename = f"{load_w}-{label}-{size}.csv"
                records.append(
                    MehranWindowRecord(
                        filename=filename,
                        recording=filename,
                        label=label,
                        defect_size_mm=size,
                        load_w=load_w,
                        start=0,
                        stop=128,
                    )
                )
                targets.append(label_index)
    generator = torch.Generator().manual_seed(7)
    table = TensorTable(
        signals=torch.randn(12, 3, 128, generator=generator),
        targets=torch.tensor(targets, dtype=torch.long),
        records=tuple(records),
        labels=("inner_race", "outer_race"),
    )
    monkeypatch.setattr(neural, "load_tensor_table", lambda *_, **__: table)
    monkeypatch.setattr(
        neural,
        "_cpu_latency",
        lambda *_args, **_kwargs: {
            "device": "cpu",
            "threads": 1,
            "batch_size": 1,
            "warmup": 0,
            "repeats": 1,
            "median_ms": 0.0,
            "p95_ms": 0.0,
        },
    )

    output = neural.run_task(
        model_id="compact_depthwise_cnn_1d",
        task_path=ROOT / "registry/tasks/mehran_load_100_to_300_v1.json",
        manifest_path=ROOT / "registry/mehran_v2.json",
        raw_dir=tmp_path / "raw",
        output_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
        epochs=1,
        batch_size=4,
        patience=1,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["benchmark_id"] == "edgefault-bench-v1.1"
    assert payload["protocol"]["labels"] == ["inner_race", "outer_race"]
    assert payload["protocol"]["input_channels"] == 3
    assert payload["protocol"]["window_normalization"] == "per_window_channel_zscore"
    assert payload["partition_sizes"] == {"train": 4, "validation": 4, "test": 4}
    assert [run["seed"] for run in payload["runs"]] == [17, 29, 43]
    checkpoint = torch.load(
        payload["runs"][0]["checkpoint"], map_location="cpu", weights_only=True
    )
    assert checkpoint["labels"] == ("inner_race", "outer_race")
    assert checkpoint["input_channels"] == 3
