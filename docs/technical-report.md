# EdgeFault-Bench v1.0 Pre-release Technical Report

## Abstract

EdgeFault-Bench evaluates rotating-machinery fault classifiers under unseen loads and devices
while reporting worst-condition reliability and deployment cost. The pre-release study pins
60 CC BY 4.0 HUST Bearing v3 recordings, three leakage-resistant tasks, four baseline model
families, and seeds 17, 29, and 43. It additionally evaluates label scarcity, controlled class
imbalance, and additive measurement noise. Results show high load-held-out performance can
coexist with severe device-held-out failure. The compact network reduces parameter and scoped
operation counts by about 93% but is slower than the standard CNN on the measured Apple M1
runtime. These negative results are part of the benchmark contribution.

## Research questions

1. How reliable are common baselines on an unseen load or bearing device?
2. Does a depthwise-separable network preserve accuracy while reducing deployment cost?
3. Does lightweight CORAL alignment improve worst-condition generalization?
4. How does the compact baseline degrade with fewer labels, minority-class reduction, and
   additive noise?

## Dataset and protocol

The source is HUST Bearing v3 (dataset DOI `10.17632/cbv7jyx4p9.3`; paper DOI
`10.1186/s13104-023-06400-4`). EdgeFault-Bench does not redistribute recordings. Its registry
pins the official identifiers, sizes, URLs, and SHA-256 hashes.

Four labels shared by five bearing devices and three loads yield 60 recordings and 7,500
non-overlapping 4,096-sample windows. The three tasks are 0 W→400 W, 400 W→0 W, and devices
6204–6206→6208, with 200 W or device 6207 used only for validation as applicable. The five
bearing identifiers provide multiple device domains, satisfying the v1 requirement to test
the protocol across at least two public datasets or device domains without overstating this
as a second dataset.

Splits are assigned before robustness transformations. No recording crosses partitions.
Each window is normalized independently, so no test-set population statistic is learned.
Early stopping uses validation Macro-F1; the frozen test domain is scored only after the best
validation state is restored.

## Baselines

- Twelve signal features with standardized logistic regression (44 fitted classifier
  parameters).
- A standard three-stage 1D CNN (77,220 parameters).
- A compact depthwise-separable 1D CNN (5,476 parameters).
- The compact network with a training-only CORAL covariance penalty across source subdomains.

Training uses AdamW for at most 15 epochs with three predetermined seeds. Machine-readable
histories retain every validation score used by early stopping.

## Core findings

The authoritative generated table is `results/v1/SUMMARY.md`.

- On 0 W→400 W, the standard CNN achieves 0.9715±0.0092 Macro-F1 and 0.9167±0.0349
  worst-bearing Macro-F1. The compact model reaches 0.8988±0.0184 and 0.6768±0.0080.
- On 400 W→0 W, the compact model reaches 0.9434±0.0210 Macro-F1, slightly above the standard
  model's 0.9221±0.0034, although worst-bearing behavior remains variable.
- On the unseen 6208 device, every model is weak. The feature baseline is highest at 0.3460
  Macro-F1; compact and CORAL models are about 0.306, and the standard CNN is about 0.204.
- CORAL provides no consistent advantage. It is marginally higher than compact CNN on one load
  direction, lower on the other, and essentially identical on device holdout.

These findings contradict any broad claim that success on an unseen load establishes robust
transfer to a new device.

## Robustness findings

The authoritative generated table is `results/v1/ROBUSTNESS.md`.

- At 25% labelled training windows, the compact model retains 0.8466±0.0229 Macro-F1.
- At 10%, the mean falls to 0.4737 with a 0.2643 standard deviation; one seed collapses to
  0.10 Macro-F1.
- Retaining 50% or 25% of the `IO` class preserves aggregate Macro-F1 near 0.93, but
  worst-bearing Macro-F1 remains around 0.67–0.69. Aggregate performance therefore does not
  eliminate reliability concerns in the weakest device group.
- Noise degradation is progressive: 0.8758±0.0201 at 20 dB, 0.6474±0.0092 at 10 dB, and
  0.4629±0.0738 at 0 dB. Worst-bearing Macro-F1 at 0 dB is 0.3140±0.1263.

Label scarcity is a window-annotation proxy, not an experiment with fewer independent
machines. Gaussian noise is deterministic and auditable but does not reproduce all mounting,
sensor, or environmental disturbances.

## Hardware findings

The authoritative generated table is `results/v1/HARDWARE.md`; measurement boundaries are in
`docs/hardware-benchmark.md`.

On an Apple M1 CPU, the compact network reduces parameters from 77,220 to 5,476, serialized
state from 317,351 to 37,151 bytes, and scoped MACs from 17.76 million to 1.24 million. Despite
this, its cross-process median batch-one latency is 0.606 ms versus 0.513 ms for the standard
CNN. Peak isolated-process RSS is similar because the Python/PyTorch runtime dominates. A
smaller architecture therefore cannot be described as faster without hardware measurement.

## Reproducibility and provenance

Every source file is checksum verified. Task manifests, model settings, and robustness levels
are committed before their corresponding result JSON. Result files retain their generating
Git commit, environment, seed, per-epoch history, condition metrics, and resource values.
Generated Markdown tables reject missing seed matrices and incomplete robustness files.

Development assistance and maintainer responsibilities are disclosed in
`docs/research-integrity.md`. Decision records remain pending until the maintainer personally
ratifies them. A clean-clone audit, packaged edge demo, tagged release, and archival identifier
remain release gates and are not claimed complete in this report.

## Limitations

- The source contains artificial faults in a laboratory setup and only one sensor dataset.
- Device holdout mixes bearing geometry and crack-severity changes.
- Five device domains satisfy the stated domain-count requirement but do not substitute for
  future validation on a genuinely independent dataset.
- Only the compact model is evaluated in the robustness tracks.
- MACs exclude pooling, normalization, and activation operations; RSS includes runtime memory.
- CPU latency on Apple M1 does not establish latency, energy, or real-time behavior on other
  edge devices.
- No external reproduction, adoption, field deployment, or safety validation is claimed.
