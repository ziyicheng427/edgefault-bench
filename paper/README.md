# Paper Draft — EdgeFault-Bench (SEN submission)

LaTeX sources for the manuscript. Build with `elsarticle`
(`latexmk -pdf main.tex`).

## Claim → evidence map

Every number in the manuscript traces to a repository artefact:

| Manuscript claim | Artefact |
|---|---|
| TLC: 89,476 states / 46,686 distinct / depth 11 / no errors | `formal/tlc-report-2026-11-06.txt` (reproduce: `scripts/run_tlc.sh`) |
| Six invariants, eight states, transition guards | `docs/formal-spec-draft.md` §3–4, `formal/ExperimentLifecycle.tla` |
| Mutation matrix 12/12 detected, 9 mechanisms, honest canary | `results/verification/mutation-matrix-2026-11-06.json`, `tests/test_mutation_catalogue.py` |
| Oracle property tests (8) | `tests/test_properties.py` |
| Registry row-count catch during gearbox integration | `docs/decisions/0007-gearbox-v1-integration.md` |
| Integration effort (239/175/6 and 261/158/5 LOC) | `docs/integration-effort-2026-11-06.md` |
| Cross-Python identical metrics | `results/verification/reproduction-3.11-vs-3.12.json` |
| Cross-OS reproduction gate | `.github/workflows/ci.yml` (`reproduce`, `reproduce-compare`) |
| Overhead table (27.3/26.2/15.2×, 8.37 s, 0.5 ms) | `results/verification/framework-overhead-2026-11-06.json` (`scripts/framework_overhead.sh`) |
| Gearbox / combined-condition audit-policy scores + CIs | committed run bundles (2026-11-06/07), policy `audit-1` |
| LODO fold means 0.565/0.377/0.597/0.472/0.268, outer 0.456 | `results/verification/hust-lodo-outer-2026-11-07.json` |
| v1.0/v1.1 protocol matrices and negative results | `results/v1`, `results/v1.1`, `docs/technical-report.md` |
| 198 tests, six dataset integrations, 13 frozen tasks | `pytest`, `registry/` |
| Honest-metadata rules (unstated rate, misspelling) | `docs/datasets/gearbox-two-stage-v1.md`, `docs/datasets/kaist-rotating-vibration-v6.md` |

## Verified citations

Bibliography entries verified against retrieval on 2026-11-07:
DomainBed (ICLR 2021, arXiv:2007.01434); Zhao/Zio/Shen RESS 2024 (Vol. 245);
aeon (JMLR 25(289), 2025); IEEE Access 2025 standardization
(doi:10.1109/ACCESS.2025.3537817). Dataset DOIs verified against their
registries. Three 2025–2026 items are marked citation-level in
`references.bib` and need full bibliographic details before submission.

## Pre-submission checklist (open)

- [ ] Complete the three citation-level bibliography entries (T18 full pass).
- [ ] Re-run TLC / mutation audit / overhead script at the final commit and
      refresh the numbers above.
- [ ] Execute the KAIST vibration task on a ≥16 GB machine; add to RQ4 or keep
      the boundary statement.
- [ ] Author affiliation + ORCID; repository archival DOI (Zenodo) once the
      reviewed release exists.
- [ ] Run `pre-submission-reviewer` on the compiled PDF.
