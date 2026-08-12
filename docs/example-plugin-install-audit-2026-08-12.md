# Independent Example Plugin Installation Audit — 2026-08-12

## Purpose

This audit tests the falsifiable product claim that an independently packaged dataset adapter
and model implementation can be discovered through public EdgeFault-Bench interfaces without
editing the benchmark kernel.

The example is maintained in this repository and therefore is not evidence of external adoption,
an independent contributor, or a separate community project. It is packaging and extensibility
evidence only.

## Audited package

- Distribution: `edgefault-example-plugin==0.1.0`
- Dataset entry point: `edgefault-example-v1`
- Model entry point: `example_centroid_classifier`
- Dataset payload: metadata-only synthetic manifest with two recordings
- Model payload: small scikit-learn-compatible nearest-centroid estimator
- Core-source modifications required by plugin: none

The package imports only documented values from `edgefault_bench.contracts` and
`edgefault_bench.plugins`. It contains its own `pyproject.toml`, source tree, and entry-point
declarations.

## Procedure and observed evidence

The local Python 3.10 audit performed these steps from the repository root:

```bash
uv pip install --python .venv/bin/python --no-deps ./examples/edgefault-example-plugin
.venv/bin/python -m edgefault_bench.cli plugin list
.venv/bin/python -m edgefault_bench.cli plugin list --kind model
.venv/bin/python -m edgefault_bench.cli plugin validate \
  --manifest examples/edgefault-example-plugin/example_manifest.json
.venv/bin/python -m edgefault_bench.cli plugin validate-model \
  --model example_centroid_classifier
```

Observed results:

- the wheel built and installed as `edgefault-example-plugin==0.1.0`;
- `importlib.metadata` exposed both registered entry points;
- dataset discovery attributed the plugin to the installed distribution;
- dataset validation passed with two canonical recordings;
- model discovery attributed the estimator to the installed distribution;
- model construction and backend validation passed;
- the full suite passed with 92 tests after plugin installation;
- Ruff passed for core, tests, and example-plugin source.

The installation initially revealed that two CLI tests assumed no external plugins could be
installed. Those tests were corrected to assert the built-in subset while permitting legitimate
third-party entries. This was an ecosystem-compatibility defect in the test boundary, not a
failure of plugin discovery.

## Continuous audit

The main GitHub Actions workflow now installs the example distribution after the core test suite
on Python 3.10 and 3.12, then runs all four discovery and validation commands. This prevents the
documented entry-point contract from drifting away from the installable implementation.

## Remaining gap

This audit does not yet run the example model through a versioned benchmark task or emit a result
bundle. The next product layer is a backend-aware execution contract that connects installed model
plugins to frozen tasks, seeds, metrics, provenance, and result validation. Genuine external use
must still be obtained separately before JOSS submission.
