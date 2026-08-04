# Mehran v2 Signal-Feature Baseline — 2026-08-04

## Scope

This is the first measured result under accepted Decision 0004. It evaluates a balanced
logistic-regression reference over 30 documented features: the existing ten time/frequency
features applied independently to X, Y, and Z. It is a baseline result, not a claim of field
readiness or superiority.

All 36 selected source files passed byte-size and SHA-256 verification before extraction.
Windows are 4,096 samples, non-overlapping, contained within one upstream recording, and
z-scored independently per channel. Test data were not used for model selection.

## Results

| Train → test load | Windows (train/validation/test) | Validation Macro-F1 | Test Macro-F1 | Worst-size Macro-F1 |
|---|---:|---:|---:|---:|
| 100 → 300 W | 398 / 393 / 363 | 0.551 | 0.552 | 0.031 |
| 300 → 100 W | 363 / 393 / 398 | 0.716 | 0.550 | 0.120 |

The three registered seeds produced identical predictions for this convex linear baseline.
Seed-specific rows are retained in the raw JSON for protocol consistency; they are not three
independent acquisitions.

The aggregate test Macro-F1 of approximately 0.55 hides severe defect-size instability. For
100 → 300 W, per-size test Macro-F1 ranged from 0.031 at 1.3 mm to 0.951 at 0.9 mm. For
300 → 100 W, it ranged from 0.120 at 1.3 mm to 0.921 at 1.5 mm. The worst-group metric is
therefore more informative than aggregate accuracy for this result.

## Resource measurements

The fitted two-class linear classifier has 31 learned coefficients/intercepts and a serialized
pipeline size of 2,154 bytes. Batch-one prediction median was approximately 0.078 ms over 1,000
repeats on the current machine. Process peak RSS is recorded in raw JSON but includes the whole
Python process and should not be interpreted as incremental model memory.

## Evidence and limitations

- Raw results:
  `results/v1.1/mehran/mehran-load-100-to-300-v1__signal_features_logreg.json` and
  `results/v1.1/mehran/mehran-load-300-to-100-v1__signal_features_logreg.json`.
- Source commit embedded in both files: `e6572571adabdf75e80783bbeb6974cfb14e2abe`.
- A CSV is treated as one recording, but windows from a CSV are correlated observations from
  one laboratory run rather than independent physical assets.
- These tasks distinguish inner- from outer-race faults among faulted recordings; they do not
  detect healthy operation.
- The dataset contains seeded laboratory defects. No conclusion about natural degradation or
  robotic inspection deployment is supported by this result alone.
- Neural baselines and isolated hardware measurements remain to be completed before a v1.1
  comparative claim is prepared.
