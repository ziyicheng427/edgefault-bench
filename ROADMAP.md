# Roadmap

Tracks the Software Engineering Nexus (SEN) publication plan from
[`docs/research-design.md`](docs/research-design.md) and Decision 0006. Frozen v1.0 and
v1.1 evidence remains valid; new work exists to answer RQ1–RQ4 about the framework
itself rather than to accumulate model comparisons.

## Batch 1 — Research design freeze (T01–T03)

- [x] Freeze the SEN positioning, four contributions, four research questions, and the
      experiment matrix (`docs/research-design.md`).
- [x] Draft the formal specification: lifecycle state machine, transition guards, and
      invariants INV-1 through INV-6 (`docs/formal-spec-draft.md`).
- [x] Publish the architecture gap analysis G1–G10 with target v2 contracts
      (`docs/architecture-gap-analysis.md`).
- [x] Record Decision 0006 and retire the JOSS-only target.

## Batch 2 — Core kernel architecture (T04–T08)

- [x] T04: Dataset signal capability (`SignalDatasetAdapter.open`, `SignalStream`,
      chunked reads, checksum re-verification at open) with HUST v3 and Mehran v2
      implementations and fail-closed validation (G1).
- [x] T05: Preprocessing contract with kernel-enforced training-side fitting, window
      identity, fitted-state digests, and a discoverable plugin group (G2).
- [x] T06: Unified execution — versioned `TrainingPolicy`, partition-restricted
      `DataView`s, sklearn and PyTorch estimators under one `benchmark run` command
      (G3).
- [x] T07: Window identity on every prepared row, schema-v2 tables, JSON/Parquet
      storage split, and window-level partition-inheritance audits (G5).
- [x] T08: Result bundle v2 with content-hashed input artifacts, failed-seed
      placeholders, overhead timings, and `results validate --check-hashes` (G8 partly).
- [x] Retire dataset-id dispatch from the legacy audit entry (G7).

## Batch 3 — Formal and software verification (T09–T12)

- [x] T09: TLA+ lifecycle model with transition guards and INV-1…INV-6; TLC
      exhaustive check clean (46,686 distinct states, depth 11, no errors,
      no deadlocks) — `formal/`, `scripts/run_tlc.sh`, CI `formal` job,
      report `formal/tlc-report-2026-11-06.txt`.
- [x] T10: Hypothesis property tests with independent accept/reject oracles
      (task partitions, recording audit, window partitioning, table identity,
      seed bijection, policies, plugin contracts) — `tests/test_properties.py`.
- [x] T11: Fault-injection catalogue implemented and executable via
      `edgefault mutation audit`; 12/12 mutations detected fail-closed
      (M-LEAK/M-HASH/M-SEED/M-PROV/M-PLUGIN).
- [x] T12 (mechanism layer): real-data window path exercised end-to-end on the
      36-file Mehran v2 registry (streamed, checksum-verified, windowed,
      preprocessed train-side-only, PyTorch plugin executed under a versioned
      policy, schema-v2 bundle hash-verified); independent Python 3.12 venv
      install + execution audit passed —
      `docs/window-execution-audit-2026-11-06.md`.
- [ ] T12 (external adoption): an independently authored dataset or model
      plugin, integrated and documented by a non-author (folds into T17).

## Batch 4 — Heterogeneous evaluation (T13–T17)

- [x] T13: Gearbox two-stage v1 adapter (20 checksum-pinned recordings, healthy vs
      broken tooth × ten loads, unstated sampling rate handled honestly with
      `sample_rate_hz=None` and fail-closed spectral features), one frozen cross-load
      task, end-to-end real-data run with group statistics. Data card, Decision 0007,
      kernel-purity assertion in tests.
- [x] T14 (acoustic subset): KAIST acoustic v6 adapter — five Test.Lab MAT recordings
      at 51.2 kHz with unit-aware channel metadata, archive hash verified against the
      Mendeley-published value. No task frozen (single load in this archive); vibration
      and current/temperature archives remain open. Data card, Decision 0008.
- [x] T14 (vibration subset, 2026-11-07): KAIST vibration v6 adapter — 45
      checksum-pinned four-channel recordings (25.6 kHz, g) across three loads ×
      fifteen conditions, upstream "Unbalalnce" misspelling normalized; frozen
      five-class cross-load task `kaist-vibration-load-0-2-to-4-v1` (train 0 N·m,
      select 2 N·m, test 4 N·m) passes the fail-closed audit. Real-archive streaming
      verified; full end-to-end training awaits a ≥16 GB machine (documented on the
      data card).
- [ ] KAIST current/temperature TDMS modality (needs an npTDMS dependency).
- [x] T15 (statistics core): group bootstrap with recording/group resampling units and
      95% CIs, window-level vs recording-level dual reporting, paired per-seed bundle
      comparison, retained per-sample test predictions in schema-v2 bundles, and the
      `edgefault statistics bootstrap|compare` commands.
- [x] T15 (task coverage): combined-condition partitions landed as task schema v2
      (`condition_fields` + tuple `condition_partitions`, mode-agnostic
      `partition_for_sample`); frozen `mehran-load-size-combined-v1` (load × defect
      size shift, verified end-to-end on real data) and five HUST
      leave-one-device-out folds `hust-lodo-{6204..6208}-v1`, all audit-passed.
- [x] T16 (first evidence points): fault-injection detection matrix (batch 3),
      framework-overhead benchmark (kernel load+validate is 15–27× bare JSON;
      `results/verification/framework-overhead-2026-11-06.json`), Python 3.11 vs
      3.12 reproduction with bit-identical metrics
      (`results/verification/reproduction-3.11-vs-3.12.json`), and a cross-OS
      reproduction matrix in CI (`reproduce` + `reproduce-compare` jobs). Open: the
      full RQ1–RQ4 evidence matrix at paper scale.
- [x] RQ2 quantitative comparison: `docs/integration-effort-2026-11-06.md` measures
      adapter-side LOC per integration against the zero-kernel-edit property, with
      every kernel touch listed and classified.
- [x] LODO outer validation executed on real HUST data: all five
      `hust-lodo-{device}-v1` folds ran end-to-end under an audit policy; the
      outer-fold summary (fold means 0.27–0.60, outer mean 0.456, large
      per-device and per-seed variance) is archived as
      `results/verification/hust-lodo-outer-2026-11-07.json` with its
      non-benchmark scope stated.
- [ ] T17 third-party invitation prepared (issue template
      `.github/ISSUE_TEMPLATE/third-party-integration.yml` with an acceptance
      checklist); an actual non-author integration remains open.
- [ ] T17: One third-party integration performed and documented by a non-author.
      Solicitation is live as
      [issue #4](https://github.com/ziyicheng427/edgefault-bench/issues/4)
      with the acceptance-checklist template; evidence lands here when an
      integration completes.

## Batch 5 — Paper and release (T18–T20)

- [ ] T18: Refresh related work (2024–2026) including formal methods for experiment
      systems; update `docs/state-of-the-field.md`.
- [x] T19 (draft): full SEN manuscript drafted in LaTeX (`paper/`, elsarticle,
      21 compiled pages, no undefined references) from the frozen section plan;
      bibliography independently verified (11/11 entries, one year corrected,
      three citation-level placeholders flagged); claim-to-evidence map in
      `paper/README.md`. Open: final-commit evidence refresh, pre-submission
      review, affiliation/archival metadata.
- [ ] T20: Stable release: unify code/package/schema versions, publish API docs, and
      archive the reviewed release with a DOI.

## Carried engineering obligations (pre-dating Decision 0006)

- [ ] Publish installable package releases through a standard Python package index.
- [ ] Add a documentation site, API reference, and end-to-end tutorials.
- [ ] Add at least one additional inference backend and hardware class, or narrow the
      deployment claims to CPU inference cost.
- [ ] Demonstrate sustained public development and genuine research use over time.
