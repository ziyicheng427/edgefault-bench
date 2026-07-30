# Roadmap

## Milestone 0 — Protocol freeze

- [x] Define mission, evidence boundaries, and success levels.
- [x] Define leakage controls and primary metrics.
- [x] Select the first dataset after verifying source terms and metadata.
- [x] Publish versioned task manifests and a data card.

## Milestone 1 — Auditable baseline

- [x] Implement package structure, configuration validation, and tests.
- [x] Implement download/preparation pipeline without committing raw data.
- [x] Add signal-feature, standard 1D-CNN, and compact depthwise CNN baselines.
- [x] Run condition-held-out experiments for three seeds.
- [x] Generate machine-readable results and the first technical report.

## Milestone 2 — Resource-efficient generalization

- [x] Add a lightweight condition-generalization method.
- [x] Benchmark parameters, model size, CPU latency, scoped MACs, and isolated-process RSS.
- [x] Add noise, label-scarcity, and class-imbalance tracks.
- [x] Validate across five bearing-device domains in addition to load domains.

## Milestone 3 — Public release

- [x] Add continuous integration, model card, contribution guide, and citation metadata.
- [x] Publish a tagged release with reproducibility instructions.
- [ ] Archive an eligible release with a persistent identifier.
- [x] Pass a new-directory, locked-environment maintainer reproduction audit.
- [ ] Invite technically relevant external reproduction and feedback without representing
      unanswered outreach as endorsement or collaboration.

## Milestone 4 — Inspection integration

- [x] Define a stable inference interface for time-series sensor windows.
- [x] Build a clearly labelled CPU edge-inference demonstration with a trained model asset.
- [x] Report demonstration constraints separately from benchmark conclusions.

## Milestone 5 — Extensible benchmark kernel

- [x] Ratify the dataset, task, leakage-audit, and artifact-boundary architecture.
- [x] Add dataset-agnostic recording and adapter interfaces.
- [x] Add a serializable leakage audit without changing frozen v1 results.
- [x] Adapt HUST metadata and tasks through the generic interfaces.
- [x] Publish contributor documentation for adding a dataset adapter.
- [x] Select a second public dataset only after license and source verification.

## Milestone 6 — Research-software maturity

- [ ] Publish installable package releases through a standard Python package index.
- [ ] Add a documentation site, API reference, and end-to-end tutorials.
- [ ] Add at least one additional inference backend and hardware class.
- [ ] Demonstrate sustained public development and genuine research use over time.
- [ ] Prepare the JOSS Markdown paper and archive the reviewed release with a DOI.
