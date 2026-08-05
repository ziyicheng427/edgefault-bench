# Quickstart: Audit a Benchmark Without Downloading Raw Data

This quickstart exercises the research-software workflow from a clean checkout. It validates
registered source metadata, audits a frozen task, and checks result provenance without
downloading third-party recordings or training a model.

## Install the development environment

```bash
git clone https://github.com/ziyicheng427/edgefault-bench.git
cd edgefault-bench
uv sync --extra dev
```

The package targets Python 3.10 or newer. PyTorch remains optional unless a neural experiment
is requested.

## 1. Inspect a registered dataset

```bash
uv run edgefault dataset inspect --manifest registry/mehran_v2.json
```

The JSON output identifies the upstream dataset, version, DOI, license, domain fields, complete
registry size, selected recording count, exclusions, and protocol status. Inspection validates
the registered adapter but does not assert that raw files are present.

## 2. Audit a frozen task

```bash
uv run edgefault task audit \
  --task registry/tasks/mehran_load_100_to_300_v1.json \
  --dataset-manifest registry/mehran_v2.json
```

A passing report confirms dataset identity, required domains, label support, explicit domain
assignment, non-empty partitions, unique recording IDs, and recording exclusivity. A failed
audit returns a non-zero exit status and must block training.

## 3. Validate a committed result

```bash
uv run edgefault results validate \
  results/v1.1/mehran/mehran-load-100-to-300-v1__signal_features_logreg.json
```

The validator rejects missing required sections, unknown code provenance, or seeds that are
missing, duplicated, or reordered. Its output is suitable for scripts and CI.

## Optional: acquire or verify source recordings

Start with one file:

```bash
uv run edgefault dataset fetch \
  --manifest registry/mehran_v2.json \
  --raw-dir data/raw/mehran_v2 \
  --files 0.7inner-100watt-67V2Iv.csv
```

Verify it later without downloading:

```bash
uv run edgefault dataset fetch \
  --manifest registry/mehran_v2.json \
  --raw-dir data/raw/mehran_v2 \
  --files 0.7inner-100watt-67V2Iv.csv \
  --verify-only
```

Raw data directories are ignored by Git. Files are accepted only after their byte size and
SHA-256 match the committed registry.

## Compatibility policy

The unified `edgefault` command is the preferred public entry point from v1.2 onward. Existing
commands such as `edgefault-download-hust`, `edgefault-download-mehran`, and
`edgefault-audit-task` remain supported aliases during the compatibility period. Their removal
would require a documented deprecation release rather than an unannounced change.

The next CLI slices will add benchmark execution and report generation after their common
configuration and extension contracts are stable.
