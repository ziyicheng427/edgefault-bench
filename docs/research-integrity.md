# Research Integrity and Development Provenance

## Purpose

EdgeFault-Bench is designed to be auditable by an independent user. Public claims must be
traceable to a task manifest, source-data checksum, code commit, environment record, and raw
machine-readable result.

## Maintainer responsibility

Ziyi Cheng is the repository owner and project maintainer. The maintainer is responsible for:

- approving the research question, benchmark scope, and release criteria;
- reviewing data and licensing decisions;
- reviewing and accepting code and documentation changes;
- running or supervising reproducibility checks;
- approving public releases and ensuring claims match measured evidence;
- responding to external issues and corrections.

These responsibilities are substantive project contributions. They should be evidenced by
actual review, decisions, signed/tagged releases, technical writing, and maintenance rather
than inferred solely from repository ownership.

## AI-assisted development disclosure

Codex has been used as an AI-assisted development tool for repository scaffolding, code and
test implementation, documentation drafting, command execution, and experiment orchestration.
The maintainer remains responsible for technical review and for every accepted scientific
claim. Commit authorship must not be described as proof that every line was written manually
without assistance.

Future external contributions must disclose material AI assistance under the same standard.

## Evidence chain

| Evidence | Public location | Meaning |
|---|---|---|
| Source identity and rights | `registry/hust_v3.json`, data card | Pins DOI, version, license, files, sizes, and hashes |
| Experimental intent | Task manifests and benchmark config | Freezes domains, seeds, models, tracks, and metrics |
| Implementation | Git commits and CI | Shows chronological code changes and automated checks |
| Execution | Result JSON and checkpoints | Binds measurements to seeds, commit, environment, and hardware |
| Interpretation | Technical report and model card | States results, failures, limitations, and appropriate use |
| Independent use | External issue, reproduction, or pull request | Demonstrates use beyond the maintainer when it genuinely occurs |

## Prohibited practices

- Backdating commits, releases, experiments, or outreach.
- Representing AI-generated work as unaided human implementation.
- Treating unanswered outreach as endorsement or collaboration.
- Omitting failed predefined seeds or robustness tracks.
- Changing a frozen test protocol after seeing results without versioning the change.
- Uploading confidential employer code, data, or personal immigration records.

