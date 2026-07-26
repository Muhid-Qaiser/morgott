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
- Boundary Pairs preserve official splits, pair IDs, scenario IDs, and source context.
  Every pair and scenario stays within one official role, and authorization-only families remain auxiliary diagnostics.

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
- The masked multitask frozen encoder also failed the quick promotion gates.
  Its winning auxiliary-BCE recipe escalated 59.80% of legitimate held-out finance rows and every benign held-out BrowseSafe document at validation-selected operating points.
  Pair ranking improved matched Boundary ordering without repairing either false-positive failure.
- Global label-support counts overstate evaluability when a held-out source lacks explicit matching-axis negatives.
  Masked per-head metrics can be undefined even while the derived route imposes a severe benign review load.
- The Rogue Security benchmark is materially contaminated by current public-source families.
  An exact normalized audit matched 54.28% of its rows to the canonical corpus, so it cannot serve as an independent headline evaluation.

## Consequence

No current classifier should block users, approve transactions, or grant tool authority.
The next work is matched finance and long-document data collection, not additional encoder tuning.
A future candidate remains a shadow-only advisory sensor behind the deterministic reference monitor.
