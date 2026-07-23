# Source-heldout false-positive audit

Generated 2026-07-23.

## What was reviewed

This audit joins the retained ModernBERT validation scores back to the canonical source-heldout validation rows by normalized-text hash.
It reviews the two-epoch ModernBERT observation and the recomputed word 1-2 gram control.
No dev-test source was read for this analysis.

The qualitative review inspected the highest-scoring distinct groups per source, not just the highest-scoring rows.
Automatic keyword tags below are overlapping weak analysis metadata.
They do not rewrite source labels.

## ModernBERT false positives by source

| Source | Benign denominator | FPR at 0.50 | FPR at 0.90 | FPR at 0.95 |
|---|---:|---:|---:|---:|
| WildJailbreak | 5,359 | 22.50% | 3.38% | 1.59% |
| Schema-Guided Dialogue | 13,277 | 4.30% | 0.38% | 0.11% |
| Taskmaster | 33,385 | 9.72% | 1.51% | 0.67% |
| BANKING77 | 1,237 | 9.70% | 1.54% | 0.73% |
| HarperValleyBank | 382 | 4.45% | 0.00% | 0.00% |
| TAT-QA questions | 1,837 | 0.76% | 0.05% | 0.05% |
| LMSYS Arena | 4 | 25.00% | 25.00% | 0.00% |
| MASSIVE en-US | 1,404 | 17.59% | 5.06% | 2.85% |
| Mind2Web | 103 | 19.42% | 4.85% | 1.94% |

The four-row LMSYS denominator is too small for a stable rate.
The largest actionable high-cutoff failures are MASSIVE, Mind2Web, Taskmaster, BANKING77, and nominally benign WildJailbreak.

## Error composition

| Model and cutoff | False positives | 32 characters or fewer | Dialogue acknowledgements | Action or side-effect terms | Security or harm terms |
|---|---:|---:|---:|---:|---:|
| ModernBERT 0.50 | 5,441 | 33.47% | 34.07% | 7.88% | 2.48% |
| ModernBERT 0.90 | 831 | 41.99% | 31.53% | 9.03% | 4.93% |
| ModernBERT 0.95 | 374 | 45.99% | 32.09% | 10.43% | 6.68% |
| Word 0.50 | 5,008 | 41.71% | 13.92% | 3.83% | 0.78% |
| Word 0.90 | 248 | 47.98% | 14.11% | 5.65% | 2.02% |
| Word 0.95 | 81 | 66.67% | 11.11% | 7.41% | 2.47% |

The dominant high-score error is short, context-dependent language.
Raising the cutoff concentrates rather than removes this failure.

## Critical findings

- Source-supported benign is not always benign under the router ontology.
  WildJailbreak nominal negatives include requests about creating computer viruses, obtaining credentials, and violent game actions.
  The broad router explicitly routes harmful non-injection content to review, so these are label-policy conflicts rather than clean false positives.

- Both models learn source and dialogue-template shortcuts.
  ModernBERT assigns near-one scores to ordinary Taskmaster and Schema-Guided Dialogue fragments such as thanks, a location name, a time, and a single proper noun.
  The word model assigns near-one scores to conversational phrases about being correct or asking for more.
  This is direct evidence against interpreting either model as robust intent understanding.

- Flattened turns lose the context needed to classify them.
  A time, name, acknowledgement, or terse follow-up is not independently classifiable as malicious or benign.
  Training and evaluating single turns forces the detector to guess from source style.

- Legitimate side effects are being confused with attacks.
  High-scoring BANKING77, MASSIVE, and Mind2Web negatives include account deletion, passcode help, email sending, job applications, social sharing, cart operations, and other ordinary agent tasks.
  Prompt text alone cannot establish whether those actions are authorized.
  Authorization must remain a trusted runtime and reference-monitor decision.

- Financial question wording is a smaller but real shortcut.
  TAT-QA errors often contain evidence constraints such as reported results, tables, balances, and financial statements.
  These resemble instruction constraints but are ordinary finance QA.

- The highest aggregate precision is an artifact of validation prevalence.
  More than half of the validation rows are positive.
  The high-score false positives that remain would dominate alerts at a realistic low attack prevalence.

- A higher cutoff changes the operating point but does not repair the representation.
  At ModernBERT 0.95, almost half of remaining false positives are very short utterances.
  The tail-format PR-AUC remains worse than the word model even when aggregate recall improves.

## Data decisions

- Do not add flagged MASSIVE, Mind2Web, Taskmaster, or Schema-Guided Dialogue validation rows to their held-out fold training data.
  Those sources already exist in the canonical training corpus.
  Adding the failed examples would destroy the unseen-source diagnostic and reward memorization.

- Keep JBB, NotInject, FinanceBench, and the human FalseReject test outside training.
  They are already viewed development diagnostics.
  Training on their errors would retire them as evaluation, and FalseReject also contains prompts that conflict with the broad `review_required` ontology.

- Do not add more OR-Bench to the default recipe.
  Its controlled ablation reduced recall from 70.78% to 65.67% for only a 0.32-point FPR change and did not fix ranking.

- The useful missing data is matched within-style support for both labels.
  Priority examples are very short dialogue fragments with surrounding context, matched legitimate action versus instruction-subversion pairs, and finance or security wording with the same surface form on both sides.
  Conversation, scenario, source, and pair lineage must stay grouped.

- Agentic Prompt Injection Boundary Pairs is a reasonable auxiliary pair-balanced ablation, not default supervision.
  Pin revision `a5682e7573e1c7bc4b12e64d49c0dcd90ca776cf`.
  The release has 1,200 synthetic rows in 600 pairs with official 840, 120, and 240 row splits.
  Keep each pair and scenario together, train only on official train, preserve source channel and risk metadata, and screen exact and strict near overlap.
  Do not add it to default fitting yet because it is templated weak supervision, is much longer than the dominant short-turn errors, and needs a predeclared pair-ranking and both-correct ablation.

## Model decisions

- Keep the word model as the cheap baseline and possible first-stage score.

- Do not interpret the frozen ModernBERT head as a dominant model.
  It improves recall at the cost of more false positives and does not improve fold-macro PR-AUC.

- Do not resume full fine-tuning or add larger heads until the label-source dependency is repaired.
  Earlier full tuning drove training loss near zero while held-out FPR exceeded 50%.

- Split future learned targets into independent masked heads for injection, harmfulness, and toxicity.
  Keep authorization and tool capability checks deterministic rather than learning them from prompt wording.

- Preserve full conversations as lineage and add a separate context-aware experiment.
  A single-turn model should be allowed to return uncertain for fragments that are not independently classifiable.
