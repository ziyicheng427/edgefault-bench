# Changelog

All notable changes and public milestones are recorded here. The project follows semantic
versioning after the first tagged release.

## Unreleased

### Added

- Accepted Decision 0003 and opened the v1.1 extensible-kernel milestone while preserving the
  frozen v1.0 evidence chain.
- Added dataset-independent contracts, a HUST v3 adapter, structured recording-leakage audits,
  an audit CLI, and contributor documentation for future dataset integrations.
- Published a source, license, domain, and auditability review proposing the CC BY 4.0 Mehran
  triaxial bearing v2 dataset as the second independent integration.
- Added a complete 38-file Mehran v2 registry snapshot, provisional 36-recording adapter and
  cross-load tasks, shared checksum-enforced downloads, and inner/outer source validation.
- Passed a clean-checkout 56-test integration audit while retaining Decision 0004 as proposed.
- Accepted Decision 0004 and froze the Mehran v2 selection, labels, cross-load partitions,
  window settings, and seeds before benchmark training.
- Added deterministic triaxial windowing, dataset-dispatched tensor loading, task-specific
  model channels/labels, and a 30-feature multichannel reference baseline.
- Completed the two frozen Mehran signal-feature tasks, retaining poor worst-defect-size
  performance as a documented negative result rather than selecting on test outcomes.
- Completed both three-seed compact-CNN tasks and documented high reverse-direction seed
  sensitivity and the absence of an aggregate advantage over the linear feature baseline.
- Completed both three-seed standard-CNN controls: aggregate Macro-F1 improved over the
  compact model, while worst-defect-size reliability remained weak at roughly 13.8× the
  trainable parameter count.

## 1.0.0 - 2026-07-28

### Added

- Pinned HUST bearing v3 registry with 60 SHA-256-verified CC BY 4.0 source files.
- Three frozen load- and device-held-out tasks with seeds 17, 29, and 43.
- Leakage-resistant recording groups and non-overlapping window protocol.
- Signal-feature baseline with condition-aware reliability and CPU measurements.
- Standard 1D CNN, compact depthwise 1D CNN, and compact CORAL model definitions.
- Automated tests on Python 3.10 and 3.12.

### Measured

- Signal-feature baseline completed on all three frozen HUST tasks.
- Standard 1D CNN, compact depthwise 1D CNN, and compact CORAL completed on both frozen
  load-held-out tasks.
- Standard 1D CNN, compact depthwise 1D CNN, and compact CORAL completed on the frozen
  device-held-out task, completing the 3-task by 4-model core matrix.
- Compact-model label-scarcity evaluation completed at 25% and 10% training-label fractions.
- Compact-model class-imbalance evaluation completed at 50% and 25% `IO` retention.
- Compact-model measurement-noise evaluation completed at clean, 20, 10, and 0 dB.
- Three-process Apple M1 CPU measurements completed for the standard and compact CNNs,
  including latency, isolated-process RSS, serialized size, and scoped MAC counts.
- Generated robustness and hardware tables, a technical report, and a compact-model
  card with negative results and deployment limitations.
- Portable seed-17 JSON model asset, SHA-256 verification, and batch-one CPU inference demo.
- One-command clean audit, full-data reproduction runner, and documented new-directory audit.
- v1.0.0 release notes and an explicit evidence/mechanical release checklist.
- Maintainer ratification of the frozen core and robustness protocols and release claims.
