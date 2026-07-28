#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-outputs/reproduction-v1}"
core_dir="${output_root}"
robustness_dir="${output_root}/robustness"
hardware_dir="${output_root}/hardware"

uv sync --frozen --extra dev --extra deep-learning
uv run --frozen edgefault-download-hust
uv run --frozen edgefault-download-hust --verify-only

tasks=(
  registry/tasks/hust_load_0_to_400_v1.json
  registry/tasks/hust_load_400_to_0_v1.json
  registry/tasks/hust_device_6204_6206_to_6208_v1.json
)
models=(
  standard_cnn_1d
  compact_depthwise_cnn_1d
  compact_coral_cnn_1d
)

for task in "${tasks[@]}"; do
  uv run --frozen edgefault-run-features --task "${task}" --output-dir "${core_dir}"
  for model in "${models[@]}"; do
    uv run --frozen edgefault-run-neural \
      --task "${task}" \
      --model "${model}" \
      --output-dir "${core_dir}" \
      --checkpoint-dir "${output_root}/checkpoints"
  done
done

for track in label_scarcity class_imbalance measurement_noise; do
  uv run --frozen edgefault-run-robustness \
    --track "${track}" \
    --output-dir "${robustness_dir}" \
    --checkpoint-dir "${output_root}/robustness-checkpoints"
done

uv run --frozen edgefault-benchmark-hardware \
  --output "${hardware_dir}/cpu.json"
uv run --frozen edgefault-summarize \
  --results-dir "${core_dir}" \
  --output "${core_dir}/SUMMARY.md" \
  --robustness-output "${core_dir}/ROBUSTNESS.md" \
  --hardware-output "${core_dir}/HARDWARE.md"

echo "Reproduction artifacts: ${output_root}"
