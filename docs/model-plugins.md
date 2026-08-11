# Model Plugin Guide

Model plugins let an independent Python distribution expose a model factory without editing
EdgeFault-Bench's built-in model table. The current interface freezes construction, identity,
backend, and provenance. A later execution-runner milestone will connect these factories to the
complete versioned benchmark workflow; discovery alone does not claim that an external model has
produced a valid benchmark result.

## Required object

An external entry point exposes one `edgefault_bench.plugins.ModelPlugin` value. Its factory
receives a `ModelBuildContext` containing the task's class count, input-channel count, and seed:

```python
from edgefault_bench.plugins import ModelBuildContext, ModelPlugin
from sklearn.linear_model import LogisticRegression


def build_model(context: ModelBuildContext) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=context.seed,
    )


plugin = ModelPlugin(
    model_id="example_logreg",
    backend="sklearn",
    source="replaced-from-installed-distribution",
    factory=build_model,
)
```

Register the object under an entry-point name identical to `model_id`:

```toml
[project.entry-points."edgefault_bench.models"]
example_logreg = "edgefault_example_models:plugin"
```

The plugin registry records the installed distribution as `source`; it does not trust a package
to self-report its provenance. Built-in model identifiers cannot be overridden.

## Backend boundary

The first interface supports two explicit backends:

- `sklearn`: the returned object provides callable `fit`, `predict`, and `get_params` methods;
- `pytorch`: the returned object provides callable `train`, `eval`, `state_dict`, and
  `parameters` methods.

These checks establish executable compatibility, not scientific validity. A publishable result
must still record its frozen task, preprocessing, seeds, training policy, dependency versions,
hardware, metrics, and git commit. Backend-specific objectives such as CORAL also require an
execution policy; a model factory by itself does not silently introduce a new training method.

## Discovery and validation

After installing the external distribution, run:

```bash
edgefault plugin list --kind model
edgefault plugin validate-model \
  --model example_logreg \
  --num-classes 4 \
  --input-channels 1 \
  --seed 17
```

Validation constructs the model but does not train it or endorse its results. Installation and
discovery execute third-party Python code, so users must review and trust external packages.

## Contribution checklist

- Use a stable, descriptive model identifier and document the backend.
- Derive stochastic initialization from the provided seed.
- Test at least two class counts and, for raw-signal models, multiple input-channel counts.
- Keep training and test-domain selection outside the factory.
- Include a small synthetic fixture; do not require proprietary data for interface tests.
- Disclose optional dependencies and material AI assistance.
- Do not describe discovery or successful construction as benchmark performance.
