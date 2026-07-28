# EdgeFault-Bench robustness summary

Generated from completed three-seed robustness JSON files. Values are population
mean ± standard deviation over seeds 17, 29, and 43.

| Track | Level | Training samples | Test Macro-F1 | Worst-condition Macro-F1 | Code commit |
|---|---:|---:|---:|---:|---|
| class_imbalance | 0.5 | 2185 | 0.9316 ± 0.0027 | 0.6893 ± 0.0319 | 98706b9 |
| class_imbalance | 0.25 | 2030 | 0.9294 ± 0.0093 | 0.6696 ± 0.0041 | 98706b9 |
| label_scarcity | 0.25 | 620 | 0.8466 ± 0.0229 | 0.6507 ± 0.0113 | 611f921 |
| label_scarcity | 0.1 | 240 | 0.4737 ± 0.2643 | 0.3636 ± 0.1864 | 611f921 |
| measurement_noise | clean | full | 0.8988 ± 0.0184 | 0.6768 ± 0.0080 | 76a87e9 |
| measurement_noise | 20_db | full | 0.8758 ± 0.0201 | 0.6710 ± 0.0062 | 76a87e9 |
| measurement_noise | 10_db | full | 0.6474 ± 0.0092 | 0.6101 ± 0.0302 | 76a87e9 |
| measurement_noise | 0_db | full | 0.4629 ± 0.0738 | 0.3140 ± 0.1263 | 76a87e9 |
