# v1.0.0 Release Checklist

## Completed evidence gates

- [x] Dataset source, license, size, URL, and SHA-256 registry.
- [x] Three frozen tasks with recording-level leakage controls.
- [x] Four model families and seeds 17, 29, and 43.
- [x] Core 3-task by 4-model result matrix.
- [x] Label-scarcity, class-imbalance, and noise tracks.
- [x] Performance, worst-condition, parameter, size, scoped MAC, repeated CPU latency, and
  isolated-process RSS reporting.
- [x] Five bearing-device domains plus load domains.
- [x] Generated result tables, model card, technical report, negative findings, and limitations.
- [x] Portable trained model asset and CPU inference demonstration.
- [x] Long-form reproduction runner and clean audit.
- [x] GitHub archive-source audit and Python 3.10/3.12 CI evidence.
- [x] AI-assisted development and maintainer responsibility disclosure.

## Maintainer-only gates

- [ ] Personally review and accept or modify Decision 0001.
- [ ] Personally review and accept or modify Decision 0002.
- [ ] Confirm the technical report accurately states the intended project claims.
- [ ] Confirm release notes do not overstate external use, field validation, or authorship.

## Final mechanical gates

- [ ] Set package and citation metadata to `1.0.0` with the actual release date.
- [ ] Convert the changelog's Unreleased section to `1.0.0` without deleting prior history.
- [ ] Run `scripts/clean_audit.sh` on the final release-candidate commit.
- [ ] Confirm the final GitHub Actions Python 3.10/3.12 matrix is green.
- [ ] Create an annotated `v1.0.0` tag.
- [ ] Push the tag and create a non-draft GitHub release using the reviewed notes.
- [ ] Archive the release with a persistent identifier when the configured repository/account
  is eligible; otherwise record the exact missing external prerequisite without inventing one.

## Post-release, not v1.0 completion claims

- [ ] Invite relevant external reproduction and feedback.
- [ ] Record genuine external issues, reproductions, pull requests, or evaluations if they occur.
- [ ] Validate on an independent second dataset or embedded board in a later version.

Unanswered outreach, repository views, stars, and the maintainer's own audit must not be
described as external adoption or independent endorsement.
