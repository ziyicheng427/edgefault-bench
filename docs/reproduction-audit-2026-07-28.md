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
- Public HTTPS clone attempts: three attempts failed before checkout because GitHub returned an
  empty response, connection timeout, or incomplete packfile.
- Public archive source: GitHub's official archive API successfully downloaded commit
  `fd86e4c8a8515bf1c260cbe81086495abf40f2b1` into a separate new directory. The full audit
  below also passed in that public-source snapshot.

## Environment construction

`uv 0.11.14` created a new Python 3.10.13 virtual environment from the committed `uv.lock`.
It installed 26 locked packages, including PyTorch 2.13.0, NumPy 2.2.6, SciPy 1.15.3,
scikit-learn 1.7.2, pytest 8.4.2, and ruff 0.16.0.

## Commands and outcomes

```bash
uv sync --frozen --extra dev --extra deep-learning
uv run --frozen ruff check .
uv run --frozen pytest
uv run --frozen edgefault-summarize --results-dir results/v1 --output outputs/clean-audit/SUMMARY.md --robustness-output outputs/clean-audit/ROBUSTNESS.md --hardware-output outputs/clean-audit/HARDWARE.md
cmp -s results/v1/SUMMARY.md outputs/clean-audit/SUMMARY.md
cmp -s results/v1/ROBUSTNESS.md outputs/clean-audit/ROBUSTNESS.md
cmp -s results/v1/HARDWARE.md outputs/clean-audit/HARDWARE.md
uv run --frozen edgefault-verify-demo
uv run --frozen edgefault-infer --warmup 2 --repeats 5
```

Observed outcomes:

- lint: passed;
- tests: 36 passed;
- generated core, robustness, and hardware tables: byte-for-byte equal to the committed files;
- demo asset SHA-256: passed;
- installed inference CLI: returned four probabilities, asset hash, label, and CPU latency;
- final clean-checkout Git status: clean.

The audit script also supports a GitHub source archive, where `.git` metadata is absent; Git
status is reported only when available.

## Public CI evidence

GitHub Actions run
[`30338096350`](https://github.com/ziyicheng427/edgefault-bench/actions/runs/30338096350)
checked out commit `fd86e4c8a8515bf1c260cbe81086495abf40f2b1`, installed dependencies, ran
ruff, and passed the complete tests on Python 3.10 and 3.12. Both matrix jobs completed
successfully on 2026-07-28.

The synthetic demo predicted `N` with high confidence. That prediction is explicitly not
benchmark evidence and is recorded only to establish execution of the public interface.

## Final v1.0.0 candidate audit

After maintainer ratification and the version-metadata update, the same locked audit script
was rerun on commit `1c3feea31052a49a51b5350d97009b3b0e1510f4`. The environment installed
`edgefault-bench==1.0.0`; ruff passed; all 37 tests passed; all three generated tables remained
byte-for-byte equal; and asset verification plus installed-CLI inference passed. This is the
final local release-candidate audit. The final GitHub Actions run and distribution steps are
tracked separately because they require the pushed commit.

## Full experimental reproduction

`scripts/reproduce_v1.sh` provides the long-form path that downloads and verifies all 60 HUST
recordings, reruns the complete core and robustness matrices, repeats hardware measurements,
and generates tables under an ignored output directory. The present clean audit does not rerun
the approximately 414 MB download and all training jobs; the original formal jobs and their
machine-readable histories are committed separately.

## Remaining release checks

- Repeat the ordinary public HTTPS Git clone when connectivity permits; public archive and
  Actions checkout evidence are already complete.
- Connect an eligible maintainer-controlled archival service before claiming a persistent
  identifier; no such verified account or repository integration was available in this
  release workflow.

## Release verification

GitHub Actions run
[`30340220112`](https://github.com/ziyicheng427/edgefault-bench/actions/runs/30340220112)
passed on Python 3.10 and 3.12 for commit
`34a1155c5caf3859ff9d0eb0c661206944e11e5c`. Annotated tag `v1.0.0` resolves to that commit.
The public GitHub release was verified as neither a draft nor a prerelease:
[`EdgeFault-Bench v1.0.0`](https://github.com/ziyicheng427/edgefault-bench/releases/tag/v1.0.0).
