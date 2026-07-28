# EdgeFault-Bench hardware summary

Measured on Apple M1 (arm64) with PyTorch 2.13.0. Values are medians across 3 independent
processes; each process contains 1,000 timed batch-one calls after warm-up.

| Model | Parameters | Serialized bytes | MACs | Median latency (ms) | p95 latency (ms) | Peak process RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| compact_depthwise_cnn_1d | 5476 | 37151 | 1240320 | 0.6064 | 0.8073 | 240.6 |
| standard_cnn_1d | 77220 | 317351 | 17760768 | 0.5130 | 0.6668 | 245.5 |

RSS covers the entire isolated Python/PyTorch process. MACs cover Conv1d and
Linear operations only. See `docs/hardware-benchmark.md` for boundaries.
