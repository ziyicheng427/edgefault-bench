# Mehran v2 Core Model Comparison — 2026-08-04

## Completed matrix

The accepted Decision 0004 matrix now contains two frozen cross-load tasks and four baselines.
Every neural entry retains seeds 17, 29, and 43, uses validation-only early stopping, and
evaluates the test partition after model selection. Values below are three-seed means; SD is
the sample standard deviation of test Macro-F1.

| Train → test | Model | Val. Macro-F1 | Test Macro-F1 | SD | Worst-size F1 | Parameters | Median latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| 100 → 300 W | Signal features | 0.551 | 0.552 | 0.000 | 0.031 | 31 | 0.077 |
| 100 → 300 W | Compact CNN | 0.761 | 0.548 | 0.040 | 0.271 | 5,634 | 0.773 |
| 100 → 300 W | Compact CORAL | 0.763 | 0.551 | 0.041 | 0.271 | 5,634 | 0.872 |
| 100 → 300 W | Standard CNN | 0.791 | 0.580 | 0.026 | 0.290 | 77,922 | 0.842 |
| 300 → 100 W | Signal features | 0.716 | 0.550 | 0.000 | 0.120 | 31 | 0.078 |
| 300 → 100 W | Compact CNN | 0.604 | 0.468 | 0.148 | 0.331 | 5,634 | 0.810 |
| 300 → 100 W | Compact CORAL | 0.607 | 0.465 | 0.145 | 0.331 | 5,634 | 0.920 |
| 300 → 100 W | Standard CNN | 0.799 | 0.632 | 0.117 | 0.323 | 77,922 | 0.880 |

Feature-baseline SD is zero because the convex solver produced identical predictions for the
three registered seeds. It should not be interpreted as independent experimental replication.

## Findings

1. The standard CNN has the best mean aggregate Macro-F1 in both directions, but uses about
   13.8 times as many parameters as the compact networks.
2. The standard model does not solve worst-defect-size reliability. Its reverse-task mean
   worst-size score (0.323) is slightly below the compact model (0.331).
3. CORAL alignment over source defect-size groups produces no material benefit over the same
   compact architecture without CORAL. The differences are within approximately 0.003
   Macro-F1, and mean worst-size scores are unchanged at the reported precision.
4. The compact networks are not faster than the standard CNN in these in-process Apple M1
   measurements. Lower parameter count does not guarantee lower latency for depthwise kernels.
5. Reverse-direction neural results are seed-sensitive. Any single best-seed claim would be
   misleading.

## Interpretation boundary

These results support a software demonstration and a narrowly defined laboratory benchmark.
They do not establish healthy-versus-fault detection, generalization to natural damage,
multi-asset field reliability, or safety for autonomous inspection. Window counts are not
counts of independent bearings. HUST and Mehran scores must not be compared as if their labels,
sensors, and fault construction were identical.

The raw JSON files in `results/v1.1/mehran` are the authoritative evidence. They retain each
seed, epoch history, validation and test metrics, per-defect-size scores, environment, source
commit, parameter count, state size, process RSS, and latency protocol.
