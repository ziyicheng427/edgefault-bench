# Evaluation Protocol

This document freezes the principles of the benchmark before model results are available.
Dataset-specific task manifests will be versioned alongside the code.

## Unit of separation

The principal test is condition-held-out evaluation. Samples derived from one continuous
recording must not be divided across train and test sets. When machine or bearing identifiers
are available, grouped separation will also prevent identity leakage.

## Roles of the splits

- **Training:** model fitting and training-time augmentation.
- **Validation:** architecture and hyperparameter selection from allowed source conditions.
- **Test:** one or more unseen operating conditions, used only for final evaluation.

No normalization statistic may be estimated from test samples unless a separately labelled
test-time adaptation experiment explicitly permits it. Such experiments will never be mixed
with the primary domain-generalization results.

## Primary metrics

- Macro-F1 across all test samples.
- Worst-condition Macro-F1.
- Balanced accuracy.
- Trainable parameter count.
- Serialized model size.
- Batch-one CPU latency distribution after warm-up.

Accuracy, per-class recall, confusion matrices, calibration, and memory measurements are
secondary metrics where supported.

## Repetition and reporting

Neural experiments will use at least three predetermined random seeds. Reports will include
mean and standard deviation, every individual run, and the configuration and commit that
produced it. Classical deterministic baselines will document any remaining randomness.

## Robustness tracks

- Additive noise at predefined signal-to-noise ratios.
- Reduced labelled training fractions.
- Controlled class imbalance.

Robustness transformations are applied after split assignment and will not create duplicate
signals across splits.

## Efficiency benchmarking

Latency measurements use batch size one, a fixed input length, warm-up iterations, repeated
timed iterations, and a documented CPU/runtime configuration. Cross-machine latency values
are not compared as though they were measured on identical hardware.

## Change control

Any material change to the primary split or metric after results are observed requires a
new protocol version and an explanation in the changelog. Prior results remain available.

