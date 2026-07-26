# Project Charter

## Mission

Build an auditable, reusable benchmark that lowers the effort required to evaluate
cross-condition generalization and edge-deployment efficiency in industrial fault
diagnosis.

## Problem statement

Many fault-diagnosis experiments randomly divide windows drawn from the same machines and
operating conditions. Such splits can overstate performance when the intended deployment
contains unseen loads, speeds, devices, or noise levels. At the same time, reporting only
classification accuracy omits constraints that matter on industrial edge devices.

EdgeFault-Bench therefore treats the operating condition as a first-class experimental
domain and evaluates reliability and efficiency together.

## Intended users

- Researchers comparing domain-generalization methods for machinery health monitoring.
- Engineers selecting compact diagnostic models for edge or robotic inspection systems.
- Students who need a small, documented, reproducible reference implementation.

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

## Research integrity rules

- Predefine data splits and primary metrics before comparing final models.
- Keep validation and held-out operating conditions distinct.
- Report all planned seeds and failed runs with explanations.
- Separate measured results from hypotheses and engineering targets.
- Record dependency versions, hardware class, configuration, and commit identifier.
- Do not backdate work or describe planned integrations as completed.

## Definition of success

The project succeeds when an external user can reproduce the documented benchmark from a
clean checkout and understand the trade-offs and limitations. A new model outperforming
every baseline is desirable but is not required for an honest and useful benchmark.

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

## Privacy and publication boundary

This repository contains only original project work and appropriately referenced public
inputs. Immigration records, personal identifiers, recommendation correspondence,
employer-confidential information, and private strategy documents are explicitly outside
the repository scope.

