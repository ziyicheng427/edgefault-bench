# Changelog

All notable changes and public milestones are recorded here. The project follows semantic
versioning after the first tagged release.

## Unreleased

### Added

- Pinned HUST bearing v3 registry with 60 SHA-256-verified CC BY 4.0 source files.
- Three frozen load- and device-held-out tasks with seeds 17, 29, and 43.
- Leakage-resistant recording groups and non-overlapping window protocol.
- Signal-feature baseline with condition-aware reliability and CPU measurements.
- Standard 1D CNN, compact depthwise 1D CNN, and compact CORAL model definitions.
- Automated tests on Python 3.10 and 3.12.

### Measured, not yet released

- Signal-feature baseline completed on all three frozen HUST tasks.
- Standard 1D CNN, compact depthwise 1D CNN, and compact CORAL completed on both frozen
  load-held-out tasks.
- Standard 1D CNN, compact depthwise 1D CNN, and compact CORAL completed on the frozen
  device-held-out task, completing the 3-task by 4-model core matrix.
- Compact-model label-scarcity evaluation completed at 25% and 10% training-label fractions.
- Compact-model class-imbalance evaluation completed at 50% and 25% `IO` retention.
- Compact-model measurement-noise evaluation completed at clean, 20, 10, and 0 dB.

No v1.0 performance claim is final until the complete experiment matrix and release audit
are published.
