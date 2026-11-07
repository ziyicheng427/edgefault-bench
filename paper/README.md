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
| Overhead table (30.1/27.1/15.6×, 8.81 s, 0.66 ms) | `results/verification/framework-overhead-2026-11-07.json` (`scripts/framework_overhead.sh`; the 2026-11-06 run is also archived) |
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
- [ ] Re-check the DG benchmark repository URL in `references.bib`
      (`CHAOZHAO-1/Domain-generalization-fault-diagnosis-benchmark` vs the
      possibly renamed `CHAOZHAO-1/DG-PHM`) against the paper's data
      availability statement.
- [x] Evidence re-run at the final pre-tag commit (2026-11-07): TLC identical
      (89,476/46,686, no errors), mutation 12/12, overhead re-archived; paper
      table and abstract updated to the fresh numbers.
- [ ] Execute the KAIST vibration task on a ≥16 GB machine; add to RQ4 or keep
      the boundary statement.
- [x] v1.2.0 frozen and pushed: tag
      [v1.2.0](https://github.com/ziyicheng427/edgefault-bench/releases/tag/v1.2.0)
      with the evidence re-run at the final commit.
- [x] Third-party integration solicited:
      [issue #4](https://github.com/ziyicheng427/edgefault-bench/issues/4).
- [ ] Zenodo DOI: maintainer steps in `docs/zenodo-archival-steps.md`
      (account action required), then add it here, to `CITATION.cff`, and to
      the availability statement.
- [x] Guide for Authors verified (2026-11-07, full text): double-anonymized
      review (manuscript anonymized, `titlepage.tex` carries author details);
      LaTeX `.tex` accepted as source, two-column allowed; length limit 20
      pages (two-column build is ~7); abstract limit 250 words (ours: 238);
      1-7 keywords (ours: 7); official Word/LaTeX templates required;
      research-data policy is Option A (deposit + cite encouraged, not
      mandatory); generative-AI use MUST be declared with the standard
      statement (added to main.tex); declarations (competing interests,
      funding) submitted via the declarations tool as .doc/.docx.
- [ ] Download the journal's official LaTeX template and diff against our
      elsarticle setup (ours compiles; confirm class/options match).
- [ ] Add 1-3 figures (lifecycle state machine, architecture overview, RQ
      results) — the manuscript currently has tables only; Elsevier artwork
      rules: vector PDF/EPS, 300/1000 dpi rasters.
- [ ] Prepare the declarations .docx, ORCID, APC decision (USD 1,600; check
      waiver eligibility), and the anonymized-repository link strategy for
      reviewers (e.g., an anonymous 4open.science mirror until acceptance).
- [ ] Author affiliation + ORCID.
- [ ] Run `pre-submission-reviewer` on the compiled PDF.
