# Mehran v2 Core Results Audit — 2026-08-04

The v1.1 Mehran result directory contains exactly eight unique task/model combinations:
two frozen load directions crossed with signal features, standard CNN, compact CNN, and compact
CORAL. An automated test now fails unless this matrix remains complete and unique.

For every result file, the audit requires:

- benchmark identity `edgefault-bench-v1.1` and a 40-character source commit;
- three non-empty train/validation/test partitions;
- explicit `test_used_for_selection: false`;
- three-channel input and seeds 17, 29, and 43 in order;
- finite validation and test Macro-F1 and balanced accuracy;
- all six defect-size group scores and a worst-group value equal to their minimum;
- 30 named features for the linear baseline, or frozen two-label metadata plus non-empty
  training history for neural models.

On 2026-08-04, Ruff and all 73 tests passed locally. GitHub Actions run `30898944453` passed
the pre-audit 64-test code and result matrix on Python 3.10 and 3.12. The subsequent audit
commit is separately validated by CI.

This is a structural and provenance audit. It confirms that reported files are internally
complete; it does not make the dataset field-representative or the measured models reliable
for operational decisions.
