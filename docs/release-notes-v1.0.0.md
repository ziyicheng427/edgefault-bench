# EdgeFault-Bench v1.0.0 Release Notes

Release date: 2026-07-28.

## What v1.0 provides

- A CC BY 4.0 HUST Bearing v3 registry with 60 official files, byte sizes, URLs, and SHA-256
  checksums; raw recordings are not redistributed.
- Three frozen recording-safe tasks covering two unseen-load directions and one unseen-device
  direction across five bearing-device domains.
- Signal-feature logistic regression, standard 1D CNN, compact depthwise 1D CNN, and compact
  CORAL baselines, each evaluated with seeds 17, 29, and 43.
- Macro-F1, balanced accuracy, per-condition and worst-condition Macro-F1, parameters,
  serialized size, CPU latency, memory, and scoped MACs.
- Label-scarcity, `IO` class-imbalance, and clean/20/10/0 dB noise tracks.
- Raw JSON with Git and environment provenance plus deterministic Markdown table generation.
- A model card, technical report, research-integrity disclosure, decision records, and dataset
  card.
- A deterministic seed-17 JSON model asset and stable 4,096-sample CPU inference interface.
- One-command clean audit and a long-form script for downloading data and rerunning the entire
  experimental matrix.

## Principal findings

- Load-held-out performance is much stronger than device-held-out performance. The best
  device-held-out aggregate result is the feature baseline at about 0.346 Macro-F1.
- Lightweight CORAL does not consistently improve the compact baseline.
- Ten-percent labelled training produces high seed variance and one 0.10 Macro-F1 collapse.
- Noise causes substantial monotonic aggregate degradation down to about 0.463 Macro-F1 at
  0 dB.
- The compact CNN removes roughly 93% of parameters and scoped MACs but is slower than the
  standard CNN on the measured Apple M1 PyTorch runtime.

These are intentionally retained negative findings, not release defects or selectively omitted
runs.

## Reproduction evidence

- All local tests pass.
- GitHub Actions passes on Python 3.10 and 3.12.
- A new-directory locked-environment audit passes without raw data or local checkpoints.
- A source snapshot obtained from GitHub's official archive API passes 37 tests, byte-for-byte
  table regeneration, asset verification, and installed-CLI inference.

See `docs/reproduction-audit-2026-07-28.md` for scope and commands.

## Important limitations

- One laboratory dataset with artificial faults is used; five devices satisfy the domain
  validation target but are not equivalent to a second independent dataset.
- Device shift mixes geometry and crack-severity differences.
- Robustness tracks cover only the compact model.
- Gaussian noise and window-level label scarcity are controlled proxies.
- Apple M1 latency and RSS do not establish performance on other CPUs, embedded boards, energy
  budgets, or field systems.
- The inference demo is not a robot deployment, operational diagnosis, external adoption, or
  safety validation.

## Upgrade and compatibility

This is the first stable protocol release. Future material changes to task partitions, labels,
metrics, or robustness levels require versioned manifests and must retain the v1 results.
