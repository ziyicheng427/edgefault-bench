# Decision 0002: V1 Robustness Tracks

- Status: Proposed for maintainer ratification
- Proposed: 2026-07-28
- Scope: EdgeFault-Bench v1.0 robustness evaluation

## Decision

Use `hust-load-0-to-400-v1` as the primary robustness task and
`compact_depthwise_cnn_1d` as the resource-efficient reference model. Preserve the frozen
seeds 17, 29, and 43 and the original validation/test partitions.

Evaluate three tracks:

1. label scarcity at 25% and 10% of training windows, sampled within every class and source
   bearing stratum;
2. class imbalance by retaining 50% and 25% of the `IO` training windows within every source
   bearing, while retaining all other classes;
3. deterministic additive Gaussian test noise at 20, 10, and 0 dB, plus the clean reference.

Sampling occurs only after the frozen domain split. Noise is added to independently
standardized test windows and each noisy window is standardized again. Validation data remain
clean, and no robustness transformation is used to choose a test-specific checkpoint.

## Rationale

- The primary load task showed meaningful performance separation without the near-total
  device-holdout failure, making degradation interpretable.
- The compact model directly represents the release's accuracy-size-latency trade-off.
- Class-domain stratification prevents a reduced-label sample from accidentally deleting an
  entire label or source device.
- Per-window seeded noise makes every perturbation exactly reproducible.

## Limitations

- Window-level label scarcity is an annotation-budget proxy, not a claim about collecting
  fewer independent machines.
- Gaussian noise does not represent every sensor, mounting, or environmental disturbance.
- Robustness results for one reference model do not establish universal model rankings.
- Any additional level or model must be versioned rather than silently added to the v1 table.

## Ratification record

The maintainer should mark this decision accepted only after personally reviewing the task,
levels, and stated limitations. Automated tooling must not record acceptance on the
maintainer's behalf.
