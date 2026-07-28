# Hardware Benchmark Protocol

The release hardware table is measured on a real CPU with batch size one and a 4,096-sample
input. Each model runs in three independent processes with one PyTorch thread, 50 warm-up
calls, and 1,000 timed calls per process. The raw JSON records every process repetition plus
the operating system, architecture, CPU identifier, Python version, PyTorch version, Git
commit, median latency, and p95 latency.

## Memory boundary

`isolated_process_peak_rss_bytes` is the operating system's peak resident-set measurement for
the entire isolated Python/PyTorch inference process. It includes the interpreter and runtime;
it is not presented as model-only tensor memory. Running every model in a fresh process avoids
contamination from a previously trained or loaded model.

## Computational-cost boundary

The reported multiply-accumulate count covers `Conv1d` and `Linear` operations observed in a
real forward pass. It excludes pooling, normalization, activation, and indexing operations.
It is therefore an auditable operator-scope estimate, not a claim of exact CPU instructions or
energy consumption.

## Input and weights

The resource benchmark uses a deterministic synthetic input and initialized weights because
tensor shapes, operation counts, serialized state size, and dense-kernel timing do not depend
on trained parameter values. Diagnostic performance remains sourced only from the separately
published trained-model result files.
