# Decision 0005: Research-Software Product Positioning

- Status: Accepted
- Proposed: 2026-08-04
- Accepted: 2026-08-05
- Scope: EdgeFault-Bench v1.2 and JOSS-readiness work

## Decision

Position EdgeFault-Bench as an installable, auditable evaluation framework for
condition-shifted industrial time-series diagnosis:

> EdgeFault-Bench turns heterogeneous machinery datasets and model implementations into
> versioned, leakage-audited, reproducible benchmarks that report both reliability under
> operating-condition shift and edge-inference cost.

The primary product is the reusable evaluation workflow, not the bundled HUST or Mehran
results and not a new fault-classification model.

## Primary users and jobs

1. **Fault-diagnosis and domain-generalization researchers** add or compare models without
   rebuilding dataset parsing, condition splits, metrics, and provenance capture.
2. **Dataset maintainers and laboratory researchers** publish adapters and frozen tasks with
   source identity, license, checksums, recording boundaries, and explicit exclusions.
3. **Reviewers and reproducibility researchers** verify that recordings do not cross
   partitions, test domains do not select models, all registered seeds are retained, and
   results map to source code and environment metadata.

Edge-deployment researchers and educators are secondary users. Maintenance technicians,
robot operators, and safety-critical production systems are not current target users.

## Core user workflow

An external user should be able to:

1. install the package with a standard Python package manager;
2. validate a registered dataset or add an adapter;
3. define or select a versioned condition-held-out task;
4. run a fail-closed recording and partition audit;
5. evaluate a bundled or third-party model across all task seeds;
6. validate and publish a machine-readable result bundle and comparison report.

No step should require copying a dataset-specific training script or editing the benchmark
kernel.

## Distinguishing contribution

EdgeFault-Bench combines capabilities that are often separated in paper-specific code:

- source/version/license identity and checksum-enforced public-data acquisition;
- canonical recording metadata and pre-training leakage audits;
- task manifests that make operating-condition partitions reviewable and immutable;
- dataset- and model-extension contracts instead of copied experiment scripts;
- condition-aware reliability, seed retention, negative-result reporting, and resource cost;
- provenance-rich artifacts whose completeness can be tested in continuous integration.

The contribution is an audit trail and extension architecture for experiments, not a claim to
be the first domain-generalization benchmark or to contain the best diagnostic algorithm.

## Non-goals

- Live condition monitoring, streaming ingestion, alarms, maintenance scheduling, or robot
  control.
- Certification or safety claims for operational machinery.
- A universal ontology that makes absolute scores directly comparable across incompatible
  datasets.
- AutoML, leaderboard optimization, or test-domain hyperparameter selection.
- Accumulating datasets or models solely to increase repository size.

## Product success measures

The project is successful as research software when all of the following are true:

- a new user can install, run a small end-to-end example, and interpret its audit and report;
- a third-party dataset or model can be added without changing the benchmark kernel;
- the public API, CLI, schemas, and compatibility policy are documented and versioned;
- package releases, tests, CI, tutorials, and archival metadata are maintained over time;
- genuine external use produces feedback, reproduction, integration, or contribution evidence;
- the JOSS paper can focus on software need, design, state of the field, and research impact
  rather than new model scores.

## Consequences

Near-term work prioritizes a unified CLI/API, plugin boundaries, schema validation, packaging,
documentation, and external usability. Additional benchmark experiments are lower priority
unless they exercise a missing software capability or respond to real user feedback.

The existing HUST v1.0 and Mehran v1.1 evidence remains valid, but is treated as validation of
the software rather than the product definition itself.

## Ratification record

During the 2026-08-04 stage review, Ziyi Cheng required an explicit objective, audience,
differentiation, research gap, and user value aligned with eventual JOSS publication. On
2026-08-05, the maintainer instructed the project to continue on the resulting productization
route. This record captures that direction; it does not claim JOSS eligibility or external
adoption.
