# Mehran v2 Compact CNN Baseline — 2026-08-04

## Scope

This report adds the first neural result under accepted Decision 0004. The compact
depthwise-separable 1D CNN consumes all three independently normalized axes. Validation
Macro-F1 selects the epoch; test data are evaluated only after selection. All registered
seeds, including weak runs, are retained.

## Three-seed results

| Train → test load | Validation Macro-F1 mean | Test Macro-F1 mean ± sample SD | Test range | Worst-size Macro-F1 mean |
|---|---:|---:|---:|---:|
| 100 → 300 W | 0.761 | 0.548 ± 0.040 | 0.504–0.583 | 0.271 |
| 300 → 100 W | 0.604 | 0.468 ± 0.148 | 0.329–0.624 | 0.331 |

The reverse direction is highly seed-sensitive: seed 17 reached 0.624 test Macro-F1, while
seed 43 reached only 0.329. Selecting or highlighting seed 17 alone would materially
overstate reliability. The forward task also shows a substantial validation-to-test gap.

Compared with the linear feature baseline, the compact CNN improves the mean worst-defect-size
score but does not improve mean aggregate Macro-F1 in either direction. This is a mixed and
unstable result, not evidence of model superiority.

## Resource observations

- Trainable parameters: 5,634.
- Serialized state: 37,791 bytes per run.
- Measured batch-one median CPU latency: approximately 0.73–0.83 ms.
- Input: three channels × 4,096 samples.

Latency is an in-process measurement on the current machine and is not yet an isolated
multi-hardware result. Process RSS includes the Python runtime.

## Evidence boundary

Raw JSON files preserve training history, selected epoch, validation/test condition metrics,
resource fields, environment metadata, and source commit:

- `results/v1.1/mehran/mehran-load-100-to-300-v1__compact_depthwise_cnn_1d.json`;
- `results/v1.1/mehran/mehran-load-300-to-100-v1__compact_depthwise_cnn_1d.json`.

Standard CNN and CORAL comparisons remain incomplete. No deployment, field-validity, or
healthy-versus-fault claim follows from this experiment.
