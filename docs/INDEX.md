# Documentation Index

Use the smallest document that answers the current question.

| Need | Read |
|---|---|
| Project goal, fixed scope, current stage | `PROJECT_CONTEXT.md` |
| Next implementation tasks | `TODO.md` |
| Why a methodology choice was accepted | `DECISIONS.md` |
| Exact parameters, versions, hashes, and code mapping | `METHODOLOGY_TRACEABILITY.md` |
| Chronological run/pilot evidence | `EXPERIMENT_LOG.md` |
| Verified detector metrics | `RESULTS_SUMMARY.md` |
| Manuscript structure and planned figures | `PAPER_OUTLINE.md` |
| Complete paper-writing source | `PAPER_TECHNICAL_RECORD_PRIVATE.md` (local, Git-ignored) |

## Maintenance rules

- Keep `PROJECT_CONTEXT.md` stable and concise; do not copy run history into it.
- Keep only unfinished work and milestone summaries in `TODO.md`.
- Append verified executions to `EXPERIMENT_LOG.md`; never rewrite history to
  make a failed pilot appear successful.
- Put exact numeric methodology values in `METHODOLOGY_TRACEABILITY.md` once.
- Put detector metrics in `RESULTS_SUMMARY.md`; link rather than duplicate them.
- Preserve paper-relevant environment, provenance, artifact, and visual-example
  details in the private technical record.
- Mark planned, implemented, executed, accepted, and rejected states explicitly.
