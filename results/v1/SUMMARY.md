# EdgeFault-Bench result summary

This file is generated from the committed raw JSON files. Scores and latency are
population mean ± standard deviation over the fixed seeds 17, 29, and 43.
Latency is batch-one median CPU latency per run; it is not an edge-device claim.

| Task | Model | Test Macro-F1 | Worst-condition Macro-F1 | Parameters | Serialized bytes | Median latency (ms) | Code commit |
|---|---|---:|---:|---:|---:|---:|---|
| hust-device-6204-6206-to-6208-v1 | signal_features_logreg | 0.3460 ± 0.0000 | 0.3328 ± 0.0000 | 44 | 1732 | 0.0817 ± 0.0033 | 10f8b6b |
| hust-load-0-to-400-v1 | compact_coral_cnn_1d | 0.9044 ± 0.0105 | 0.6739 ± 0.0073 | 5476 | 37151 | 0.8656 ± 0.0092 | 00549e6 |
| hust-load-0-to-400-v1 | compact_depthwise_cnn_1d | 0.8988 ± 0.0184 | 0.6768 ± 0.0080 | 5476 | 37151 | 0.8705 ± 0.0918 | 00549e6 |
| hust-load-0-to-400-v1 | signal_features_logreg | 0.7730 ± 0.0000 | 0.5428 ± 0.0000 | 44 | 1732 | 0.0819 ± 0.0023 | 10f8b6b |
| hust-load-0-to-400-v1 | standard_cnn_1d | 0.9715 ± 0.0092 | 0.9167 ± 0.0349 | 77220 | 317351 | 0.7429 ± 0.0579 | 00549e6 |
| hust-load-400-to-0-v1 | signal_features_logreg | 0.8827 ± 0.0000 | 0.7018 ± 0.0000 | 44 | 1732 | 0.0817 ± 0.0027 | 10f8b6b |

## Recorded execution environments

- macOS-26.3.1-arm64-arm-64bit / Python 3.10.13
