# Clean-environment Reproduction Audit — 2026-07-28

## Scope

This maintainer audit tests whether committed code, generated tables, and the inference demo
work in a new checkout without the development virtual environment, raw HUST files, local
training checkpoints, or ignored artifacts. It does not claim external independent adoption.

## Checkout provenance

- Audited commit: `78c17b61e9e34f2c6243cd4ee8a503e1f3c48d1b`.
- Clean directory: a newly created temporary directory outside the project workspace.
- Transfer source: `git clone --no-local` from the clean, fully pushed local Git history.
- Confirmed absent before installation: `.venv`, `data/raw`, and `artifacts`.
- Public HTTPS clone attempts: two attempts failed before checkout because GitHub returned an
  empty response or connection timeout. A public-URL clone remains a release audit item.

## Environment construction

`uv 0.11.14` created a new Python 3.10.13 virtual environment from the committed `uv.lock`.
It installed 26 locked packages, including PyTorch 2.13.0, NumPy 2.2.6, SciPy 1.15.3,
scikit-learn 1.7.2, pytest 8.4.2, and ruff 0.16.0.

## Commands and outcomes

```bash
uv sync --frozen --extra dev --extra deep-learning
uv run --frozen ruff check .
uv run --frozen pytest
uv run --frozen edgefault-summarize
git diff --exit-code -- results/v1/SUMMARY.md results/v1/ROBUSTNESS.md results/v1/HARDWARE.md
uv run --frozen edgefault-verify-demo
uv run --frozen edgefault-infer --warmup 2 --repeats 5
```

Observed outcomes:

- lint: passed;
- tests: 36 passed;
- generated core, robustness, and hardware tables: byte-for-byte unchanged in Git;
- demo asset SHA-256: passed;
- installed inference CLI: returned four probabilities, asset hash, label, and CPU latency;
- final clean-checkout Git status: clean.

The synthetic demo predicted `N` with high confidence. That prediction is explicitly not
benchmark evidence and is recorded only to establish execution of the public interface.

## Full experimental reproduction

`scripts/reproduce_v1.sh` provides the long-form path that downloads and verifies all 60 HUST
recordings, reruns the complete core and robustness matrices, repeats hardware measurements,
and generates tables under an ignored output directory. The present clean audit does not rerun
the approximately 414 MB download and all training jobs; the original formal jobs and their
machine-readable histories are committed separately.

## Remaining release checks

- Repeat this audit from the public HTTPS clone when GitHub connectivity permits.
- Confirm the latest GitHub Actions matrix is green for the release candidate.
- Record maintainer ratification of Decisions 0001 and 0002.
- Create the signed or annotated v1.0 tag and GitHub release only after those checks.
