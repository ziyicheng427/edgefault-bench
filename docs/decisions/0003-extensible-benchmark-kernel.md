# Decision 0003: Extensible Benchmark Kernel

- Status: Accepted
- Proposed: 2026-07-30
- Accepted: 2026-07-30
- Scope: EdgeFault-Bench v1.1 and later

## Decision

Evolve EdgeFault-Bench from a HUST-specific experiment package into an extensible,
hardware-aware evaluation toolkit for condition-shifted industrial sensor diagnosis.

The reusable kernel will separate four concepts:

1. dataset adapters expose immutable dataset metadata and canonical recording metadata;
2. task specifications define label support, domain-held-out partitions, windowing, and seeds;
3. leakage audits validate partition disjointness and recording/window boundaries before model
   training;
4. artifact bundles retain task, dataset, environment, prediction, metric, and hardware
   provenance.

The existing HUST v1 manifests, result files, command-line interfaces, and release artifacts
remain valid. New interfaces will be introduced through backward-compatible adapters before
any legacy implementation is deprecated.

## Initial acceptance criteria

- Define dataset-agnostic `Recording`, `DatasetMetadata`, and `DatasetAdapter` interfaces.
- Define a general task specification without weakening the frozen HUST v1 checks.
- Produce a structured leakage report that can be serialized and inspected before training.
- Adapt HUST metadata to the generic interfaces with no changes to frozen v1 partition sizes.
- Test both successful audits and explicit leakage failures.
- Document the extension path for a second independently licensed public dataset.

## Rationale

- A research software package must support more than the analysis that originally created it.
- Explicit interfaces let researchers add datasets and models without copying benchmark logic.
- A first-class leakage report turns an existing protocol rule into reusable scientific tooling.
- Backward compatibility preserves the public v1 evidence chain and prevents silent protocol
  drift.

## Boundaries

The initial extensible kernel remains focused on labelled industrial sensor classification
under explicit domain shifts. Remaining-useful-life prediction, generic AutoML, robot control,
and safety-critical operational decisions are outside this phase.

New datasets will be accepted only after source identity, access terms, and redistribution
rights are documented. Development history must reflect genuine work and review; commits,
issues, releases, or external engagement will not be manufactured for publication optics.

## Ratification record

Ziyi Cheng explicitly agreed on 2026-07-30 to begin the next project phase based on the
proposed extensible architecture and maturity roadmap. This record captures that stated
maintainer decision; it does not represent automated approval.
