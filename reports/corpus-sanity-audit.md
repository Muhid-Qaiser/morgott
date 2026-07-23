# Corpus sanity audit

This audit covers the canonical source shards, routing views, quarantine outputs, labels, grouping, leakage controls, and current model evidence.
Exact current counts, hashes, and split distributions remain in `data/manifest.json` as the sole machine source of truth.

## Checks that passed

- Every manifest-tracked file was hash-verified and its recorded row count was recomputed.
- Canonical schemas, source revisions, downloaded-file digests, and required source contracts fail closed.
- Exact normalized text does not cross train, validation, and dev-test.
- Split groups do not cross train, validation, and dev-test.
- Exact label conflicts, strict near-overlaps, and source-level sensitive-text matches remain visible in quarantine instead of silently entering fitting.
- Official evaluation lineages remain in dev-test, and exact duplicates touching those lineages are held out with them.

## Critical red flags

- Source identity predicts most of the training label.
  Most direct-user sources contain only benign rows or only review-required rows, so a model can score well by recognizing dataset style instead of intent.
- The labels do not represent production truth or production prevalence.
  They combine task construction, challenge participation, human safety annotation, model labels, and generated weak supervision.
- `review_required` deliberately mixes instruction subversion, harmful requests, toxicity, and unresolved cases.
  It is a conservative routing target, not a claim that every positive prompt is malicious or an injection.
- Text alone cannot determine authority or provenance.
  The same short instruction can be legitimate from an authorized user and hostile when embedded in retrieved content, tool output, email, or memory.
- The direct-user neural recipe has no long benign denominator above its context limit.
  Increasing the token limit on the current selected rows would expose more positive text without measuring the corresponding false-positive cost.
- Exact deduplication removes repeated text, but many source families still contain templated, paraphrased, adaptive, or derived examples.
  SimHash quarantine is conservative and cannot prove semantic independence.
- Development evaluation has already influenced source and model decisions.
  The current dev-test is useful for repeated diagnosis but is not a prospective final test.
- Multilingual, multi-turn, memory, tool-output, clarification-state, and action-level coverage remains too thin for broad product claims.
- Regex-based privacy screening is only a quarantine aid.
  Passing it does not prove that public issue or agent-task text contains no personal or sensitive information.
- Corrected source-heldout experiments did not find an architecture fix for the data problem.
  Full tuning overfit source artifacts, the frozen multipool head retained double-digit macro-source false-positive rates, OR-Bench reduced recall, and paired splices did not improve ranking materially.

## Consequence

No current classifier should block users, approve transactions, or grant tool authority.
The viable near-term target is a shadow-only `no_security_signal` versus `review_required` sensor, evaluated by source-heldout and explicitly qualified paired diagnostics, behind a deterministic reference monitor.
