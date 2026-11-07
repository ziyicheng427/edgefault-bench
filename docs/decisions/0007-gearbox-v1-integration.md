# Decision 0007: Gearbox Two-Stage v1 Integration

- Status: Accepted
- Proposed: 2026-11-06
- Accepted: 2026-11-06
- Scope: third dataset integration (different machinery type); no kernel changes

## Decision

Integrate the CC BY 4.0 data.gov "Gearbox Fault Diagnosis Data" as
`gearbox-two-stage-v1`: 20 checksum-pinned recordings (healthy vs broken tooth ×
ten loads), streamed through the generic signal contracts, with one frozen
cross-load task `gearbox-load-low-to-high-v1`.

## Rationale

1. RQ2/RQ4 require heterogeneous machinery beyond bearings; a gearbox with an
   unstated sampling rate also exercises the honest-metadata path
   (`sample_rate_hz=None`, spectral preprocessing fails closed).
2. Licence and source verified on the catalogue page; the archive URL is stable
   and hash-pinnable; all twenty files selected, nothing excluded.
3. The integration added only adapter-side code (`datasets/gearbox.py`,
   registry, task, data card, tests). Two kernel edits were made as **generic**
   contract changes, not dataset branches: `sample_rate_hz` became optional
   wherever a source may legitimately not state one, and the window extractor
   stopped asserting a rate. A test asserts that no dataset identifier appears
   anywhere in the kernel outside `datasets/` and the fetch dispatcher.

## Consequences

- Contract change: `SignalInfo.sample_rate_hz` and `WindowBatch.sample_rate_hz`
  became `float | None` with fail-closed consumers; existing datasets are
  unaffected (they state rates).
- Registry row counts follow the loader's parse; the invariant check caught a
  two-row discrepancy during the first real run and the registry was corrected
  before any result was produced.
- The `edgefault dataset fetch` dispatcher gained a gearbox branch; the branch
  table remains the only dataset-specific dispatch outside adapters (known
  limitation until downloaders become plugins).
