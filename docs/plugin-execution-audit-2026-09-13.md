# Installed Plugin Execution Audit — 2026-09-13

## Claim under test

An independently packaged scikit-learn model plugin can be discovered, instantiated once per
frozen seed, evaluated against a versioned condition-held-out task, and recorded in a result bundle
accepted by the existing provenance validator without modifying the benchmark kernel.

## Procedure

The maintained `edgefault-example-plugin==0.1.0` distribution was installed in the audit
environment. Commit `0a601fedc21078ce6a8c74231bde8d8583924f80` then executed:

```bash
python -m edgefault_bench.cli benchmark run \
  --task examples/edgefault-example-plugin/example_task.json \
  --features examples/edgefault-example-plugin/prepared_features.json \
  --model example_centroid_classifier \
  --output /tmp/example-plugin-result.json \
  --latency-warmup 1 \
  --latency-repeats 3
python -m edgefault_bench.cli results validate /tmp/example-plugin-result.json
```

## Observed result

- plugin provenance: `edgefault-example-plugin==0.1.0`;
- task: `edgefault-example-load-v1`;
- seeds: 17, 29, and 43 in the frozen order;
- partition sizes: four train, four validation, and four test rows;
- result provenance: full commit `0a601fedc21078ce6a8c74231bde8d8583924f80`;
- result validation: passed;
- complete local suite: 98 tests passed;
- source lint: passed.

All three synthetic test scores equal 1.0 because the fixture intentionally provides two clearly
separable numerical clusters. The values test wiring only. They are not machinery-diagnosis
performance, evidence of generalization, or a result suitable for scientific comparison.

GitHub Actions repeats installation, execution, and result validation on Python 3.10 and 3.12.

## Remaining limitations

- the example plugin is maintained by this project and is not evidence of external adoption;
- the JSON input is appropriate for small audits, not large production feature matrices;
- only the scikit-learn execution policy is implemented;
- the generic learned-value count is disclosed separately from trainable neural parameters;
- genuine third-party use and feedback remain required before JOSS submission.
