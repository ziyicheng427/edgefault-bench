# Edge Inference Demonstration

This demonstration runs one 4,096-sample vibration window through the released compact model
on a CPU with batch size one. It is an inference-interface demonstration, not a robot, field
deployment, or new benchmark result.

After installing the deep-learning dependency, run the deterministic synthetic demo:

```bash
uv sync --extra deep-learning
uv run edgefault-infer
```

Use an external window stored as one-dimensional NumPy or comma-separated data:

```bash
uv run edgefault-infer --input path/to/window.npy
uv run edgefault-infer --input path/to/window.csv
```

The input must contain exactly 4,096 finite numeric samples. The CLI applies the frozen
per-window z-score transformation and returns all four probabilities, the selected label,
the model-asset SHA-256, and warmed batch-one CPU latency.

The built-in synthetic waveform never came from HUST and its predicted class is not scientific
evidence. It exists only so a clean checkout can exercise the complete loading, normalization,
forward-pass, and output interface without downloading third-party data.

## Model choice and provenance

The asset uses seed 17 because it is the first pre-registered seed, not because it achieved the
best test score. Its metadata binds the training task and Git commit. Aggregate claims remain
based on all three seeds in `results/v1`, not on this demonstration checkpoint.

The trained asset derives from HUST Bearing v3 (DOI `10.17632/cbv7jyx4p9.3`, CC BY 4.0).
Dataset attribution and limitations in `docs/datasets/hust-bearing-v3.md` continue to apply.
The demo asset is distributed under CC BY 4.0; project source code remains Apache-2.0.
