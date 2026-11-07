# Changelog

All notable changes and public milestones are recorded here. The project follows semantic
versioning after the first tagged release.

## 1.2.0 - 2026-11-07

### Added

- Drafted the SEN manuscript (`paper/`, elsarticle, 21 compiled pages):
  frozen eight-section structure with the four RQs, all numbers traced to
  committed artefacts via `paper/README.md`, and an independently verified
  bibliography (one year corrected; three citation-level placeholders
  flagged for the T18 full pass).
- Integrated the KAIST vibration archive (Decision 0008 extension): 45
  four-channel recordings at 25.6 kHz across three loads × fifteen conditions,
  the upstream misspelling normalized with original filenames still pinned, and
  the first frozen non-bearing cross-load task
  `kaist-vibration-load-0-2-to-4-v1` (five machine-condition classes,
  audit-passed). Registry, adapter, and real-archive streaming verified; full
  training deferred to a ≥16 GB machine per the data card.
- Added combined-condition and leave-one-device-out task support (T15): task
  schema v2 introduces `condition_fields` with tuple-valued
  `condition_partitions`; `TaskSpec.partition_for_sample` is the single
  mode-agnostic partition entry point used by the audit, prepared-table, and
  window kernels. Frozen `mehran-load-size-combined-v1` (train loads 100/200 W
  across all defect sizes, select at 300 W small defects, test at 300 W large
  defects — verified end-to-end on real data with group statistics) and five
  HUST LODO folds `hust-lodo-{6204..6208}-v1`, every fold audit-passed.
- Added a cross-OS reproduction matrix to CI (`reproduce` on ubuntu/macos +
  `reproduce-compare`) that fails unless the canonical per-seed metrics are
  identical across operating systems.
- Executed the full leave-one-device-out outer validation on real HUST data:
  all five `hust-lodo-{device}-v1` folds completed end-to-end under an audit
  policy; fold means range 0.27–0.60 with an outer mean of 0.456, archived with
  scope boundaries in `results/verification/hust-lodo-outer-2026-11-07.json`.
- Added the third-party integration issue template with an acceptance
  checklist (`.github/ISSUE_TEMPLATE/third-party-integration.yml`) to solicit
  T17 evidence from non-authors.
- Published the RQ2 integration-effort audit
  (`docs/integration-effort-2026-11-06.md`): adapter-side LOC and file
  inventory per dataset integration, with every kernel touch classified as
  generic capability, registration declaration, or registered debt.

- Formalized and model-checked the experiment lifecycle (T09):
  `formal/ExperimentLifecycle.tla` encodes the state machine and guards of
  `docs/formal-spec-draft.md`; TLC exhaustively verified INV-1…INV-6, the
  fail-closed discipline, and deadlock freedom over 46,686 distinct states
  with no errors. `scripts/run_tlc.sh` reproduces the check and a CI `formal`
  job runs it on every push; the archived log is
  `formal/tlc-report-2026-11-06.txt`.
- Added property-based verification (T10): `tests/test_properties.py` pairs
  Hypothesis-generated valid/invalid artefacts with independently recomputed
  oracles for task partitions, recording audits, window partitioning,
  prepared-table identity, seed bijection, training policies, and plugin
  contracts.
- Added the fault-injection catalogue (T11): `edgefault mutation audit`
  executes the twelve mutations of `docs/formal-spec-draft.md` §6 against the
  live kernel and renders the RQ1 detection matrix; all twelve are rejected
  fail-closed and the reporter is tested to surface undetected faults honestly.
- Integrated the CC BY 4.0 data.gov gearbox two-stage dataset as
  `gearbox-two-stage-v1` (Decision 0007): 20 checksum-pinned recordings across
  healthy/broken-tooth × ten loads, a frozen cross-load task, honest handling of
  the unstated sampling rate (`sample_rate_hz=None` contract support with
  fail-closed spectral preprocessing), and a verified real-data end-to-end run.
- Integrated the KAIST rotating-machine acoustic subset as
  `kaist-rotating-acoustic-v6` (Decision 0008): five Test.Lab MAT recordings at
  51.2 kHz with unit-aware channels, archive and per-file hashes verified against
  the Mendeley-published values. No task is frozen because this archive covers a
  single load; the boundary is documented on the data card.
- Added group-aware statistics (T15): result bundles now retain per-sample test
  predictions (truth, predictions, evaluation groups, recording ids), and
  `edgefault statistics bootstrap|compare` reports recording-level bootstrap
  95% CIs beside window-level Macro-F1 plus paired per-seed model comparisons.
- Added first RQ3/RQ4 evidence points: the framework-overhead benchmark
  (`scripts/framework_overhead.sh`, archived JSON) and a Python 3.11 vs 3.12
  reproduction check with identical metrics across all seeds.
- Executed the real-data window path on the verified 36-file Mehran v2
  registry (T12 mechanism layer): streamed windows with full identity,
  train-side-only preprocessing fit, PyTorch plugin executed under a versioned
  policy across all frozen seeds, schema-v2 bundle with verified input hashes;
  plus an independent Python 3.12 environment install/execution audit. Evidence
  and boundaries: `docs/window-execution-audit-2026-11-06.md`.

- Accepted the SEN research repositioning (Decision 0006): the paper target moves from
  JOSS to Software Engineering Nexus around four research questions on integrity,
  extensibility, reproducibility, and scale; the design is frozen in
  `docs/research-design.md`, `docs/formal-spec-draft.md` (lifecycle state machine with
  invariants INV-1 through INV-6), and `docs/architecture-gap-analysis.md`.
- Added the dataset signal capability: `SignalInfo`, the `SignalStream` protocol, and the
  `SignalDatasetAdapter` contract with `open(recording_id)`. The HUST v3 and Mehran v2
  adapters stream their registered recordings through chunked reads and re-verify byte
  size and SHA-256 at open time (INV-1 at the signal boundary). Metadata-only adapters
  keep working and fail closed with a capability error where signals are required.
- Added the preprocessing contract (`edgefault_bench.preprocessing`): window identity
  (`window_id`, `recording_id`, `start_sample`, `end_sample`), window batches, a
  `Preprocessor` protocol with fit/transform separation, training-side-only fitting
  enforced by the kernel, fitted-state digests for provenance, and three built-in
  preprocessors (`per_window_zscore`, `channel_standardization`, `signal_features`)
  discoverable through the `edgefault_bench.preprocessors` entry-point group.
- Added kernel window extraction (`edgefault_bench.windows`): streaming, audited,
  deterministic windowing straight from a registered dataset manifest, with
  partition-inheritance checks and prepared-table v2 builders that carry window identity
  on every row.
- Added versioned training policies and unified backend execution
  (`edgefault_bench.execution`): `TrainingPolicy` manifests (optimizer, early stopping on
  validation Macro-F1, patience, determinism), `DataView` partition isolation,
  `SklearnEstimator` and `TorchEstimator` adapters, and a single `benchmark run` command
  that trains and evaluates installed sklearn and PyTorch plugins under the same
  protocol. Training plugins receive only train/validation views; test rows are
  materialized exclusively at evaluation (INV-3 as an architectural property).
- Added result-bundle schema v2: content-hashed input artifacts (task manifest, prepared
  table, dataset manifest, training policy), per-seed `ok`/`failed` status with error
  placeholders instead of dropped seeds (INV-4), framework overhead timings, and
  `results validate --check-hashes` recomputation (M-PROV-02 detection). Result and
  prepared-table JSON Schemas now dispatch on `schema_version`; v1 artifacts remain
  valid and untouched.
- Added optional Parquet storage for prepared tables through the `columnar` extra.
- Removed the dataset-id dispatch from the legacy task-audit entry point; it now routes
  through the plugin registry like the unified CLI.
- Raised the package version to 1.2.0 to match the documented command surface.

## Unreleased (pre-1.2.0 development)

### Added

- Accepted Decision 0005 and repositioned EdgeFault-Bench as an installable, extensible,
  audit-first research software framework rather than a collection of benchmark results.
- Published an initial state-of-the-field comparison and falsifiable build-vs-contribute
  rationale covering domain-specific benchmarks and complementary research software.
- Added the unified `edgefault` CLI with dataset inspection/acquisition, task auditing, and
  result validation while retaining existing commands as compatibility aliases.
- Added a raw-data-free external-user quickstart for the first unified workflow slice.
- Passed a new-checkout installation audit covering 77 tests and the installed unified CLI.
- Added fail-closed dataset plugin discovery through standard Python entry points, public
  plugin listing and validation commands, and adapter-boundary tests.
- Added backend-aware model plugin discovery and construction contexts for independently
  packaged scikit-learn and PyTorch implementations, with fail-closed identity checks.
- Added an independently packaged synthetic dataset/model plugin and install-level CI audit on
  Python 3.10 and 3.12, without representing the maintained fixture as external adoption.
- Added a portable prepared-feature-table contract and a fail-closed scikit-learn plugin runner
  that emits seed-complete, provenance-bound result bundles from versioned tasks.
- Shipped Draft 2020-12 JSON Schemas for tasks, prepared feature tables, and core results, with
  multi-file CLI validation and repository-wide CI enforcement.
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
- Completed both compact-CORAL tasks and the full Mehran 2-task × 4-model matrix; CORAL did not
  materially improve aggregate or worst-defect-size results over the compact architecture.
- Added an automated completeness and provenance audit for all eight Mehran core result files.

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
