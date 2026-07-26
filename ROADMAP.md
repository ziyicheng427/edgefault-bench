# Roadmap

## Milestone 0 — Protocol freeze

- [x] Define mission, evidence boundaries, and success levels.
- [x] Define leakage controls and primary metrics.
- [ ] Select the first dataset after verifying source terms and metadata.
- [ ] Publish versioned task manifests and a data card.

## Milestone 1 — Auditable baseline

- [ ] Implement package structure, configuration validation, and tests.
- [ ] Implement download/preparation pipeline without committing raw data.
- [ ] Add signal-feature, standard 1D-CNN, and compact depthwise CNN baselines.
- [ ] Run condition-held-out experiments for three seeds.
- [ ] Generate machine-readable results and the first technical report.

## Milestone 2 — Resource-efficient generalization

- [ ] Add a lightweight condition-generalization method.
- [ ] Benchmark parameters, model size, CPU latency, and memory where supported.
- [ ] Add noise, label-scarcity, and class-imbalance tracks.
- [ ] Validate on a second public dataset or device domain.

## Milestone 3 — Public release

- [ ] Add continuous integration, model card, contribution guide, and citation metadata.
- [ ] Publish a tagged release with reproducibility instructions.
- [ ] Archive an eligible release with a persistent identifier.
- [ ] Invite technically relevant external reproduction and feedback without representing
      unanswered outreach as endorsement or collaboration.

## Milestone 4 — Inspection integration

- [ ] Define a stable inference interface for time-series sensor windows.
- [ ] Build a clearly labelled robot/simulator or edge-device demonstration.
- [ ] Report demonstration constraints separately from benchmark conclusions.

