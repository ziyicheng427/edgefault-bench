# Contributing

EdgeFault-Bench welcomes reproducibility reports, dataset-adapter corrections, benchmark
extensions, and hardware measurements.

## Research-integrity requirements

- Do not tune a model on a frozen test domain.
- Do not replace or remove an unfavorable completed run without documenting why.
- Include the exact task manifest, seed, commit, dependency versions, and hardware class.
- Keep raw third-party data outside Git; add only source metadata, checksums, and scripts.
- State whether a result is measured, simulated, reproduced, or planned.
- Disclose material use of code-generation or AI assistance in the pull request.

## Development checks

```bash
uv sync --extra dev --extra deep-learning
uv run ruff check .
uv run pytest
```

## Authorship and review

Git commit metadata identifies the account that accepted and recorded a change; it does not
by itself prove unaided manual authorship. Contributors should describe their conceptual,
implementation, validation, data-curation, and writing roles accurately. The maintainer is
responsible for reviewing generated changes and for approving releases and scientific
claims.

