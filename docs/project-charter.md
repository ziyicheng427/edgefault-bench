# Project Charter

## Mission

Build an installable, auditable evaluation framework that turns heterogeneous machinery
datasets and models into versioned, leakage-resistant, reproducible cross-condition
benchmarks with reliability and edge-inference cost reported together.

## Problem statement

Many fault-diagnosis experiments randomly divide windows drawn from the same machines and
operating conditions. Such splits can overstate performance when the intended deployment
contains unseen loads, speeds, devices, or noise levels. At the same time, reporting only
classification accuracy omits constraints that matter on industrial edge devices.

EdgeFault-Bench therefore treats the operating condition as a first-class experimental
domain and evaluates reliability and efficiency together.

## Primary users

- Researchers comparing fault-diagnosis and domain-generalization methods.
- Dataset maintainers publishing source-audited condition-shift tasks.
- Reviewers and reproducibility researchers auditing experimental claims.

Edge-deployment researchers and educators are secondary users. Maintenance technicians,
robot operators, and safety-critical production systems are outside the present product scope.

## Product promise

An external user should be able to install the package, validate or add a dataset adapter,
select a frozen task, audit recording partitions, evaluate a model across registered seeds,
and publish a validated result bundle without copying dataset-specific training code.

The reusable workflow is the product. Bundled datasets, models, and measured results are
reference implementations and verification evidence.

## Core deliverables

1. Dataset adapters with checksum/provenance metadata and leakage-resistant segmentation.
2. Versioned cross-condition protocols in which test conditions are not used for training.
3. At least three baseline families:
   - signal features plus a conventional classifier;
   - a standard one-dimensional convolutional network;
   - a depthwise-separable lightweight network.
4. Evaluation across at least three random seeds, including Macro-F1, balanced accuracy,
   worst-condition Macro-F1, and confusion matrices.
5. Efficiency evaluation including trainable parameters, model size, peak inference memory
   where measurable, and batch-one CPU latency with warm-up and repeated runs.
6. Robustness evaluation under controlled noise and reduced-label settings.
7. Automated tests, pinned environments, machine-readable result files, and a technical
   report that includes limitations and negative results.

## Non-goals for the first release

- Claiming universal generalization across all machines or sensor types.
- Redistributing third-party datasets without explicit permission.
- Using private employer data, code, or operational details.
- Presenting simulation or benchmark performance as field deployment.
- Tuning hyperparameters on the held-out test condition.
- Providing live monitoring, alarms, maintenance scheduling, or robot control.
- Adding datasets or models solely to increase the apparent project scope.

## Research integrity rules

- Predefine data splits and primary metrics before comparing final models.
- Keep validation and held-out operating conditions distinct.
- Report all planned seeds and failed runs with explanations.
- Separate measured results from hypotheses and engineering targets.
- Record dependency versions, hardware class, configuration, and commit identifier.
- Do not backdate work or describe planned integrations as completed.

## Definition of success

The project succeeds when an external user can install and extend the documented benchmark,
reproduce its reports, and understand the audit trail and limitations. A new model
outperforming every baseline is neither required nor the primary software objective.

### Level 1 — Auditable baseline

- One public dataset and at least three condition-held-out tasks.
- Three baseline families and three-seed results.
- Automated unit/smoke tests and a reproducibility command.
- A technical report with the exact protocol and limitations.

### Level 2 — Recommended public release

- A second dataset or device domain.
- Lightweight model at no more than 25% of the standard CNN's parameter count, with its
  accuracy and latency trade-off reported without selective omission.
- Noise, label-scarcity, and class-imbalance experiments.
- Release artifacts, model card, citation metadata, and continuous integration.

### Level 3 — Independent-use evidence

- External reproduction, issue, pull request, or documented evaluation by a non-collaborator.
- Archival release with a persistent identifier when eligible.
- A robot, simulator, or edge-device demonstration that consumes the benchmarked inference
  interface without changing the reported benchmark protocol.

### Level 4 — JOSS-ready research software

- Standard package-index distribution and a stable, documented public API/CLI.
- End-to-end tutorials for users, dataset contributors, and model contributors.
- A state-of-the-field comparison and explicit build-vs-contribute justification.
- Sustained public development and genuine external use or community feedback.
- A JOSS-format paper with software-design, research-impact, and AI-usage sections.

## Privacy and publication boundary

This repository contains only original project work and appropriately referenced public
inputs. Immigration records, personal identifiers, recommendation correspondence,
employer-confidential information, and private strategy documents are explicitly outside
the repository scope.
