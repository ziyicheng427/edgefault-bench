# Compact Depthwise 1D CNN Model Card

## Status

This is a pre-release model-family card for `compact_depthwise_cnn_1d`. The repository
publishes architecture code, training protocol, raw results, and a deterministic JSON demo
asset for seed 17. The asset is an inference demonstration, not the basis for aggregate claims.

## Model details

- Input: one independently z-scored vibration window with shape `[1, 4096]`.
- Output labels: healthy (`N`), inner-race (`I`), outer-race (`O`), and combined (`IO`).
- Architecture: one conventional stem followed by three depthwise-separable 1D blocks,
  global average pooling, a 64-dimensional embedding, and a four-class linear head.
- Trainable parameters: 5,476.
- Serialized state: 37,151 bytes under the recorded PyTorch environment.
- Scoped computational cost: 1,240,320 Conv1d/Linear multiply-accumulates.
- Training: AdamW, learning rate 0.001, weight decay 0.0001, batch size 64, maximum 15
  epochs, validation Macro-F1 early stopping with patience 4.

The `compact_coral_cnn_1d` variant uses the same inference architecture and adds a training-only
covariance-alignment loss across source subdomains with weight 0.1.

## Data and evaluation

The model is evaluated on the HUST Bearing v3 subset pinned in `registry/hust_v3.json` under
the source dataset's CC BY 4.0 license. The three tasks hold out either a load or a bearing
device. Windows from a recording never cross task partitions. Seeds 17, 29, and 43 are
reported individually and in aggregate.

Authoritative values are generated from raw JSON into:

- `results/v1/SUMMARY.md` for the core matrix;
- `results/v1/ROBUSTNESS.md` for label scarcity, class imbalance, and noise;
- `results/v1/HARDWARE.md` for repeated Apple M1 measurements.

## Intended use

- Reproducible research on cross-condition machinery fault diagnosis.
- A compact baseline for studying accuracy-size-latency trade-offs.
- Educational or engineering evaluation on the pinned tasks.
- CPU inference demonstrations that consume the stable 4,096-sample interface.

## Out-of-scope use

- Safety-critical maintenance decisions without independent validation and human review.
- Claims about natural degradation, remaining useful life, fault severity, or unseen sensor
  types.
- Treating benchmark confidence as calibrated probability or operational risk.
- Inferring performance on a new machine from the load-held-out results alone.

## Limitations and failure modes

- Device holdout is poor: the compact model averages about 0.306 Macro-F1 on bearing 6208,
  showing that load transfer does not establish device transfer.
- The CORAL variant does not consistently improve the compact baseline and is nearly identical
  on the device-held-out task.
- At 10% labelled training windows, one seed collapses to 0.10 Macro-F1, producing high
  three-seed variance.
- At 0 dB additive Gaussian noise, aggregate and worst-condition performance degrade sharply.
- Artificial cracks, differing crack depths, and laboratory acquisition limit field validity.
- The compact model has about 7% of the standard model's parameters and scoped MACs but is
  slower on the measured Apple M1 CPU, likely reflecting kernel/runtime efficiency.
- Gaussian noise and window-level label scarcity are controlled proxies, not complete field
  simulations.

## Hardware interpretation

Latency uses a real Apple M1 CPU, one PyTorch thread, batch size one, three independent
processes, and 1,000 timed calls per process after warm-up. RSS includes the complete isolated
Python/PyTorch process, not model tensors alone. Results must not be compared to another
machine as if hardware and runtime were identical.

## Development provenance

Development is AI-assisted as disclosed in `docs/research-integrity.md`. The repository
maintainer is responsible for protocol review, scientific claims, release approval, and
corrections. Git ownership is not represented as proof of unaided manual authorship.
