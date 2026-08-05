# State of the Field and Build-vs-Contribute Rationale

Reviewed: 2026-08-05

## Research need

Condition shift is a recognized evaluation problem in intelligent machinery diagnosis. Recent
benchmarking work identifies ad hoc dataset partitions, labels, and evaluation criteria as a
barrier to comparable domain-adaptation results. EdgeFault-Bench addresses the narrower
software problem of making cross-condition experiments source-audited, leakage-resistant,
extensible, and reproducible.

It does not claim to introduce domain generalization, time-series classification, experiment
tracking, or fault-diagnosis benchmarking.

## Related software and methods

| Project | Primary scope | Capabilities relevant here | Boundary relative to EdgeFault-Bench |
|---|---|---|---|
| [Domain-Generalization Fault-Diagnosis Benchmark](https://github.com/CHAOZHAO-1/Domain-generalization-fault-diagnosis-benchmark) | Comparing machinery domain-generalization algorithms | Cross-working-condition and cross-machine settings; eight method implementations | Strong algorithm benchmark and closest domain-specific work. Its public repository is organized around model/data scripts and prepared datasets; EdgeFault-Bench focuses on installable extension contracts, upstream file identity, recording audits, result schemas, and resource evidence. |
| [DomainBed](https://github.com/facebookresearch/DomainBed) | General domain-generalization algorithm comparison | Standard algorithms, datasets, test environments, sweeps, and explicit model-selection methods | Establishes important DG evaluation practice but is primarily built around its own benchmark datasets and is now archived. It does not model machinery recording identity, source-file rights/checksums, or edge diagnostic resource reports. |
| [aeon](https://www.aeon-toolkit.org/) | General scikit-learn-compatible time-series machine learning | Classification, regression, clustering, anomaly detection, segmentation, similarity search, and benchmark tooling | A mature algorithm ecosystem that EdgeFault-Bench should interoperate with rather than reproduce. It does not prescribe condition-held-out machinery tasks or upstream recording/provenance audits. |
| [MLflow Tracking](https://mlflow.org/docs/latest/tracking/) | General experiment and model lifecycle tracking | Logs parameters, code versions, metrics, datasets, checkpoints, and artifacts | Complementary storage and visualization infrastructure. It records what a user logs but does not define or enforce machinery task partitions, recording exclusivity, registered seeds, or worst-condition metrics. |
| [Standard DA benchmarking framework for intelligent fault diagnosis](https://doi.org/10.1109/ACCESS.2025.3537817) | Methodology for controlled fault-level and operating-condition evaluation | Formalizes domain-shift factors, consistent partitions, and comparative evaluation on CWRU and Paderborn | Supports the need for standard protocols. EdgeFault-Bench contributes reusable software contracts, validation, packaging, provenance, and community extension paths rather than another fixed study alone. |

This comparison is representative, not an assertion that no other related package exists. It
must be revisited before a JOSS submission and updated when reviewers or users identify closer
alternatives.

## What EdgeFault-Bench uniquely combines

The contribution is the integration of six concerns into one domain-specific workflow:

1. **Source audit:** versioned source identity, license, stable file IDs, expected byte sizes,
   SHA-256 values, and explicit exclusions.
2. **Recording semantics:** one canonical identity per independent upstream acquisition before
   windows are created.
3. **Reviewable tasks:** immutable domain partitions, labels, window rules, validation policy,
   seeds, and evaluation groups stored as data rather than hidden in scripts.
4. **Fail-closed validation:** training is blocked when recordings are missing, duplicated,
   unassigned, or shared across partitions.
5. **Reliability and cost together:** aggregate and worst-condition metrics are retained with
   seed variability, parameter/state size, latency, and documented measurement boundaries.
6. **Auditable artifacts:** result bundles retain task, dataset, code, environment, training
   history, and completeness checks suitable for continuous integration.

Any one of these exists elsewhere. The research-software gap is a reusable machinery-diagnosis
framework that enforces all of them consistently across independently sourced datasets and
third-party models.

## Why a standalone package is justified

Contributing a classifier to aeon or another general time-series library would not express the
domain-specific source, recording, condition, and artifact invariants above. Adding more model
scripts to an existing fault-diagnosis benchmark would also leave package installation,
adapter contracts, result validation, and hardware evidence unresolved.

A standalone orchestration and validation layer is therefore justified, with deliberate reuse
rather than replacement of established ecosystems:

- accept scikit-learn-compatible estimators where practical;
- keep PyTorch optional for neural baselines;
- expose portable JSON artifacts instead of inventing a tracking server;
- consider optional MLflow export rather than implementing experiment storage;
- evaluate an aeon estimator bridge before JOSS submission;
- cite and compare against domain-specific benchmark protocols.

## Falsifiable product test

The build justification fails if an external researcher must copy a bundled training script to
add a dataset or model. Before JOSS submission, at least one third-party-style adapter and one
third-party-style model integration must exercise only documented public interfaces. Ideally,
one of these is implemented or reviewed by a non-author.

The project should be reconsidered as a contribution to an existing ecosystem if its generic
interfaces cannot deliver that independence.

## JOSS relevance

JOSS currently requires a statement of need, comparison with commonly used packages,
build-vs-contribute justification, meaningful software-design decisions, and concrete research
impact or credible near-term significance. This document supplies research for those eventual
paper sections; it is not itself a claim that the project is ready for submission.
