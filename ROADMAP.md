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
- [ ] Publish a tagged release with reproducibility instructions.
- [ ] Archive an eligible release with a persistent identifier.
- [ ] Invite technically relevant external reproduction and feedback without representing
      unanswered outreach as endorsement or collaboration.

## Milestone 4 — Inspection integration

- [ ] Define a stable inference interface for time-series sensor windows.
- [ ] Build a clearly labelled robot/simulator or edge-device demonstration.
- [ ] Report demonstration constraints separately from benchmark conclusions.
