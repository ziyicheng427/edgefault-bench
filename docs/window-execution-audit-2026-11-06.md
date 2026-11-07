# Window-Path Execution and Verification Audit — 2026-11-06

## Claims under test

1. **T12 (mechanism layer):** a registered dataset manifest alone — no prepared
   feature table, no dataset-specific kernel code — is sufficient to stream,
   checksum-verify, window, preprocess, train, and evaluate a PyTorch model
   plugin through the unified `benchmark run` command, producing a schema-v2
   result bundle whose input artifacts pass hash re-verification.
2. **T09:** the abstract experiment lifecycle satisfies invariants
   INV-1…INV-6 and the fail-closed discipline under exhaustive model checking.
3. **T10:** kernel accept/reject behaviour agrees with independent oracles on
   Hypothesis-generated valid and invalid artefacts.
4. **T11:** every mutation in the fault catalogue of
   `docs/formal-spec-draft.md` §6 is rejected fail-closed by at least one
   mechanism.
5. **Independent environment:** the package and an external plugin install and
   execute in a Python 3.12 virtual environment separate from the development
   environment.

## Procedure and observed results

### Real-data window path (Mehran v2, CC BY 4.0)

All 36 registered CSV recordings were downloaded with mandatory byte-size and
SHA-256 verification (`edgefault-download-mehran`, 145 MB), then:

```bash
edgefault benchmark run \
  --task registry/tasks/mehran_load_100_to_300_v1.json \
  --dataset-manifest registry/mehran_v2.json \
  --model compact_depthwise_cnn_1d \
  --policy /tmp/mehran-window-policy.json \
  --output /tmp/mehran-window-result.json
edgefault results validate --check-hashes /tmp/mehran-window-result.json
```

- windows streamed from 36 recordings: 398 train / 393 validation / 363 test,
  every window carrying `(window_id, recording_id, start_sample, end_sample)`;
- preprocessing `per_window_zscore` fitted on the training side only
  (`fitted_row_count=791` = train + validation; state digest recorded);
- `compact_depthwise_cnn_1d` (5,634 parameters) executed under the versioned
  policy `audit-1` (Adam, 1e-3, 2 epochs, batch 64, deterministic): all three
  frozen seeds completed with status `ok`;
- test Macro-F1 0.341 / 0.325 / 0.325 — a **2-epoch audit run**: these values
  demonstrate the mechanism and are NOT comparable to the frozen v1.1 protocol
  results (which used the full training budget) and are not a benchmark claim;
- `results validate --check-hashes`: task manifest, dataset manifest, and
  policy SHA-256 all `verified`; framework load/audit overhead 1.4 s for the
  36-recording extraction; environment records torch 2.13.0.

### Model checking (T09)

`formal/ExperimentLifecycle.tla` checked with TLC2 2026.11.06 (Java 17 class
compatibility; reproduced via `scripts/run_tlc.sh`):

> Model checking completed. No error has been found.
> 89,476 states generated, 46,686 distinct states, search depth 11.

All nine properties hold: TypeOK, Inv1SourceIntegrity,
Inv2PartitionExclusivity, Inv2WindowInheritance, Inv3TestIsolation,
Inv4SeedCompleteness, Inv5Provenance, Inv6PluginContract, FailClosed; no
deadlocks. Full log: `formal/tlc-report-2026-11-06.txt`. CI re-runs this on
every push.

### Property-based tests (T10)

`tests/test_properties.py` (Hypothesis 6.170): 8 properties pairing generated
artefacts with independently recomputed oracles — task-partition validation,
recording-audit agreement, window-partition totality/exclusivity, prepared-table
round trips, window-identity uniqueness, seed-bijection validation, training-
policy validation, and plugin-contract rejection. All pass.

### Fault-injection catalogue (T11)

`edgefault mutation audit` executes all 12 mutations of
`docs/formal-spec-draft.md` §6 against the live kernel:

> total 12, detected 12, passed true

Detection mechanisms: pre-training audit (M-LEAK-01), recording cross-audit
(M-LEAK-02), kernel fit-scope guard (M-LEAK-03), partition-restricted views
(M-LEAK-04), adapter checksum verification (M-HASH-01), result validator
(M-SEED-01/02, M-PROV-01), input-artifact hash check (M-PROV-02), plugin
registry and contract validation (M-PLUGIN-01/02/03). The matrix reporter
itself is tested to report `undetected` honestly when a guard is sabotaged.

### Independent environment

A fresh `uv` virtual environment on Python 3.12.0 (development uses 3.11)
installed `edgefault-bench==1.2.0` and the separately packaged
`edgefault-example-plugin` from source. `plugin list --kind model` discovered
5 model plugins (4 built-in + 1 external); `benchmark run` on the example
table produced a schema-v2 bundle that passed `results validate
--check-hashes`.

## Evidence boundaries

- The example plugin is maintained by this project; it is mechanism evidence,
  not external adoption. A non-author third-party integration (T17) remains
  open.
- The Mehran run used a reduced 2-epoch policy purely to exercise the
  pipeline; it is not a research result and must not be cited as one.
- The TLA+ model verifies the abstract guard logic; it is not a proof of the
  Python implementation (see `formal/README.md`, claim discipline).
- The kernel has not yet been exercised on real data for the HUST adapter's
  window path or on any dataset beyond the two bearings sets (T13/T14 open).
