# Formal Model — Experiment Lifecycle

This directory holds the TLA+ model of the EdgeFault-Bench experiment lifecycle
(T09 of the SEN research plan, `docs/research-design.md`).

## What is modelled

`ExperimentLifecycle.tla` abstracts one `(task, model-plugin)` experiment from
data acquisition to publication, exactly along the transitions and guards of
`docs/formal-spec-draft.md` §4:

| Transition | Guard | Runtime mechanism |
|---|---|---|
| `Acquire` (t1) | source integrity (INV-1) | checksum-verified download + `adapter.open` re-verification |
| `Audit` (t2) | partition validity (INV-2, recording level) | `edgefault_bench.audit.audit_recordings` |
| `Prepare` (t3) | window inheritance (INV-2) + fit scope (INV-3) | `partition_window_indices`, `fit_preprocessor` |
| `Construct` (t4) | plugin contract (INV-6) | `validate_dataset_adapter` / `validate_model_implementation` |
| `TrainStep` (t5) | test isolation (INV-3); a test-scope attempt exits via `REJECTED` | partition-restricted `DataView`s |
| `EvaluateStep` (t6) | test read exactly once per trained seed | evaluation step in `execution.run_benchmark` |
| `Validate` (t7) | seed bijection (INV-4) + provenance (INV-5) | `reporting.load_result`, `verify_input_artifact_hashes` |
| `Publish` (t8) | immutable release | release checklist |

Bad candidates (tampered partitions, test-claiming windows, test-scope fits,
contract-less plugins, incomplete bundles) are explored non-deterministically;
every guard violation must land in `REJECTED` with a reason (`FailClosed`).

## Checked properties

TLC exhaustively checks, over all reachable states of the configured model:

- `TypeOK`, `Inv1SourceIntegrity`, `Inv2PartitionExclusivity`,
  `Inv2WindowInheritance`, `Inv3TestIsolation`, `Inv4SeedCompleteness`,
  `Inv5Provenance`, `Inv6PluginContract`, `FailClosed`;
- absence of deadlocks (committed phases can always continue or reject).

## Result

`tlc-report-2026-11-06.txt` records the run on TLC2 2026.11.06:

> Model checking completed. No error has been found.
> 89,476 states generated, 46,686 distinct states, search depth 11.

## Reproducing

```bash
# Java 11+ and the TLA+ tools jar are required.
scripts/run_tlc.sh          # downloads tla2tools.jar if absent, then runs TLC
```

CI repeats the check on every push (see the `formal` job in
`.github/workflows/ci.yml`).

## Claim discipline

This is bounded model checking of an **abstract** lifecycle model. It verifies
the guard logic of the design, not the Python implementation; implementation
coverage comes from the runtime contract checks, the Hypothesis property tests
(`tests/test_properties.py`), and the fault-injection catalogue
(`edgefault mutation audit`).
