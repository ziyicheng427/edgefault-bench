# EdgeFault example plugin

This directory is an independently packaged integration fixture. It registers one metadata-only
dataset adapter and one scikit-learn-compatible model through Python entry points while importing
only documented EdgeFault-Bench contracts.

It is intentionally small and synthetic. It proves packaging, discovery, provenance, and boundary
validation; it is not a machinery dataset, competitive baseline, benchmark result, or separately
maintained community project.

From the repository root, install it into an environment containing EdgeFault-Bench:

```bash
uv pip install --python .venv/bin/python --no-deps ./examples/edgefault-example-plugin
.venv/bin/edgefault plugin list
.venv/bin/edgefault plugin list --kind model
.venv/bin/edgefault plugin validate \
  --manifest examples/edgefault-example-plugin/example_manifest.json
.venv/bin/edgefault plugin validate-model --model example_centroid_classifier
```

The main CI workflow performs this installation after the core test suite on every supported
Python version.

Run the complete synthetic execution path and validate its result bundle:

```bash
.venv/bin/python -m edgefault_bench.cli benchmark run \
  --task examples/edgefault-example-plugin/example_task.json \
  --features examples/edgefault-example-plugin/prepared_features.json \
  --model example_centroid_classifier \
  --output outputs/example-plugin-result.json
.venv/bin/python -m edgefault_bench.cli results validate \
  outputs/example-plugin-result.json
```

The inputs and scores are synthetic interface fixtures, not machinery research results.
