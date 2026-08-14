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

## Cross-domain false positives, registered mmBERT shadow and public guards (2026-08-07)

Measured while scoring the pinned guard baselines. Every figure below comes from
`artifacts/comparisons/<slug>/{evaluation.json,scores.npz}` on canonical dev-test
(161,910 negatives for the current corpus, 161,482 for the older Prompt Guard run).

### Aggregate benign flag rate

Each system read twice: at its own native cutoff, and at the shared calibration
protocol threshold. The two columns differ enormously for the saturating models.

| System | Benign rows | At native 0.5 | At calibrated threshold |
|---|---:|---:|---:|
| mmbert-lora-full-s42 (registered at the time) | 161,910 | 4.62% | 1.50% |
| Llama Prompt Guard 2 86M | 161,482 | 1.81% | 1.38% |
| Qwen3Guard-Stream-4B, query risk head | 161,910 | 10.89% | 0.21% |
| Qwen3Guard-Stream-4B, jailbreak head | 161,910 | 0.01% | 0.01% |
| ModernGuard-1 | 161,910 | 13.59% | 0.00% |

ModernGuard and the Qwen3Guard query head saturate at exactly 1.0, so the shared
protocol selects a threshold at or above 1.0 and they flag nothing. Their 0.00%
and 0.21% columns are abstention, not precision, and must never be quoted as
selectivity. At their own default cutoffs both are two to three times less
selective than the then-registered shadow, which matches their behaviour on the
red-team reserve, where they flag 81.3% and 95.7% of bare-harmful rows.

### Per source, then-registered shadow at its calibrated threshold

| Source | Benign denominator | FPR | Prompt Guard 2 |
|---|---:|---:|---:|
| Finance (banking77, tatqa, harper_valley_bank, financebench) | 7,043 | 0.00% | 0.01% |
| Taskmaster | 62,733 | 0.03% | 0.01% |
| Schema-Guided Dialogue | 51,731 | 0.00% | 0.00% |
| XSTest | 450 | 0.00% | 0.00% |
| FalseReject | 1,187 | 0.00% | 1.18% |
| CoCoNot | 379 | 0.00% | 0.00% |
| NotInject | 339 | 0.88% | 2.36% |
| BrowseSafe | 1,856 | 0.32% | 0.16% |
| do_not_answer | 6,312 | 0.78% | 0.30% |
| SWE-bench verified | 492 | 5.89% | not in that corpus |
| HarmBench (harmful, no subversion) | 400 | 13.50% | 5.00% |
| **multi_turn generated controls** | **2,400** | **86.75%** | **75.58%** |
| **llmail challenge FP controls** | **160** | **100.00%** | not in that corpus |

HarmBench rows are `harmful_non_injection` by contract, so flagging them is
off-target rather than a benign error.

`llmail` was missing from the first version of this table. Its 160 negatives are
all `challenge_false_positive_control` on the `untrusted_content` channel — the
competition's own purpose-built false-positive controls — and every one of them
is flagged. It is the second-largest contributor after `multi_turn` and the only
other saturated benign source.

### Finding: the multi_turn controls dominate any aggregate benign number

The `multi_turn` negatives are 2,400 generated control conversations shipped as
the benign half of an obfuscated multi-turn jailbreak set. Scoring the two view
files separately through the registered bundle at the calibrated threshold:

| View | Rows | label_basis | Flagged | Median score |
|---|---:|---|---:|---:|
| `multi_turn_benign` | 1,200 | `generated_benign_control` | 82.58% | 1.0000 |
| `multi_turn_semi_benign` | 1,200 | `generated_semantically_benign_control_with_harmful_terms` | 90.92% | 1.0000 |

The `multi_turn_benign` figure was first recorded as 82.50%. A re-score through
the registered bundle in canonical dev-test row order gives 82.58%, one row
apart. Batch composition perturbs padded scores, as
`experiments/guard_baselines/README.md` records for the shared scorer; treat the
last digit of any single-row-margin figure on this source as batch-dependent.

The failure is not confined to the deliberately borderline half, and the median
score is maximal, so the model is maximally confident on rows the corpus labels
benign. Prompt Guard 2 fails the same source at 75.58%, which points at the
scaffolding shared by that generation process rather than at either model's
weights. Both variants carry `data_role: dev_test` and
`injection_subtype_training_eligible: true`.

These 2,400 rows are 1.48% of the benign denominator and contribute 85.86% of all
canonical false positives at the calibrated threshold — the first version of this
section said "roughly two thirds", which understates it. The explicit current
evaluation records 2,082 of the 2,425 false flags on this source.
Any corpus-wide benign FPR for this model is therefore dominated by one synthetic
source and should not be presented as representative. Report per source.

Adding `llmail`, the other purpose-built control source, accounts for almost
everything left:

| Population | Benign rows | False positives | FPR |
|---|---:|---:|---:|
| Canonical dev-test as reported | 161,910 | 2,425 | 1.4977% |
| `multi_turn` + `llmail` controls | 2,560 | 2,242 | 87.58% |
| Everything else | 159,350 | 183 | **0.1148%** |

The two adversarial-control sources are 1.58% of the benign denominator and
92.45% of all false positives. On the remaining ordinary benign text the model
flags at 0.1148%, thirteen times below the figure the corpus-wide number
reports.

### Actions

- Do not quote an aggregate canonical benign FPR without naming `multi_turn`.
- Audit whether `injection_label: 0` is correct for those rows under
  `docs/data-contract.md`, and whether siblings by `split_group_id` reach any
  training partition despite the dev-test `data_role`.
- SWE-bench software-issue text at 5.89% is the second-worst domain and wants
  the same review.
- When comparing against saturating guard models, report their native cutoff.
  A calibrated-threshold comparison silently rewards abstention.

## multi_turn control audit: cause, label verdict, and remedy (2026-08-07)

Answers the second action above. Reproduced through
`experiments/guard_baselines/adapters.py build_baseline("mmbert-lora-full-s42-rescore")`,
which loads the registered bundle, over the canonical dev-test rows assembled by
`canonical_rows`, at the shared threshold `0.9997965730219449`.
No raw prompt text is quoted here; every characterisation below is a structural
measurement over the projected detector text.

### What the generation process produced

The upstream source is `tom-gibbs/multi-turn_jailbreak_attack_datasets` at
revision `e3b30257c4d6be5438ea19f0989ac82c24234fe4`, loaded by `_load_multi_turn`
in `src/morgott/data.py`. It ships three parallel CSVs — harmful, semi-benign,
and completely-benign — and the adapter projects the same `Prompt` column from
each, taking `injection_label` from the file the row came from and `category`
from that row's `Input-cipher`/`Output-cipher` pair.

Those three families are the same artifact with a different goal inside:

- All 6,536 rows, attacks and controls alike, share a byte-identical 133-
  character prefix and 198-character suffix. That is roughly a third of a median
  1,000-character row, and it is identical across every one of the eight
  techniques.
- The eight `category` values are the same in all three families: `word_mapping_random`
  and `word_mapping_perp_filter`, each alone and each crossed with Base64,
  Caesar, and Leetspeak. Each control family holds exactly 150 rows per
  technique, the attack family 517.
- 64 word types appear in at least 95% of attack rows. Mean coverage of that
  vocabulary inside the benign controls is 100.0%.
- Structural probes agree: 100% of rows in every family carry cipher and
  word-mapping language, 50% carry a respond-in-cipher instruction. No family
  carries override, refusal-suppression, roleplay, or system-impersonation
  phrasing.

So the controls are not ordinary benign text that happens to score badly. They
are the attack carrier with a safe goal substituted in, which is what
`reports/dataset-selection.md` already calls them: "goal-grouped obfuscated
attacks and matched contrast families."

### The model scores the carrier, not the goal

Flag rate at the shared threshold, by technique:

| Technique | benign | semi-benign | harmful |
|---|---:|---:|---:|
| `word_mapping_perp_filter` | 11.33% | 40.00% | 59.77% |
| `word_mapping_random` | 51.33% | 87.33% | 93.81% |
| `word_mapping_perp_filter/Base64` | 100.00% | 100.00% | 100.00% |
| `word_mapping_perp_filter/Caesar` | 100.00% | 100.00% | 100.00% |
| `word_mapping_perp_filter/Leetspeak` | 100.00% | 100.00% | 100.00% |
| `word_mapping_random/Base64` | 99.33% | 100.00% | 100.00% |
| `word_mapping_random/Caesar` | 99.33% | 100.00% | 100.00% |
| `word_mapping_random/Leetspeak` | 99.33% | 100.00% | 100.00% |

Six of the eight techniques — every one that applies an output cipher — flag at
99.33% to 100% regardless of label. Only the two techniques with no output
cipher separate the classes at all. Deleting the shared 331-character
scaffolding and re-scoring moves the flag rate by only 1.83 points on the benign
controls, 4.50 on semi-benign, and 1.67 on the attacks, so the driver is the
inline cipher body rather than the fixed framing.

Across the whole source the attacks flag at 94.20% and the benign controls at
82.58%: an 11.62-point separation between rows that differ only in the goal
carried inside an identical wrapper. Prompt Guard 2 86M fails the same source at
75.58%. This is a property of the row construction, not of these weights.

### Is `injection_label: 0` correct? No, and neither is 1

`docs/data-contract.md` reserves `routing_label=0` for "source-supported benign
content" and assigns `routing_label=1` to injection, jailbreak, or "an
unresolved row that must not silently pass as benign". `injection_label` may be
null, and "null is unknown, never negative".

The source supports a benign **goal**. The projected detector text is the
**attack-transformed prompt**. Inheriting the goal's label and attaching it to
the carrier asserts something the source never claimed, so `injection_label: 0`
and `routing_label: 0` are not supportable for these 2,400 rows.

Relabelling them to 1 would be equally wrong. It asserts instruction-subversion
intent that the source explicitly denies, contradicts the row's own
`goal_policy_status: safe`, and would hand the model 2,400 easy positives that
inflate recall. The contract's own instruction not to collapse these concepts
applies in both directions.

The rows are genuinely unresolved: the subversion mechanism is present, the
harmful intent is absent. That is what the uncertain path is for.

### These rows are not in any training partition

Checked directly rather than inferred:

- No rows with `source: multi_turn`, `group_id` matches,
  `split_group_id` matches, and zero `normalized_text_sha256` matches in
  `data/views/routing/train.jsonl`, `validation.jsonl`, or `uncertain.jsonl`;
  current partition totals live only in `data/manifest.json`.
- All routing-eligible `multi_turn` rows sit in `dev_test.jsonl`.
  `complete_harmful` rows are `source_role: auxiliary` and enter no routing view.
- The three families share no `group_id`, `split_group_id`, `source_goal_sha256`,
  or text hash with each other, so no benign control is a lineage sibling of an
  attack.

`injection_subtype_training_eligible` is derived in `src/morgott/data.py` as
`label is not None`. It records that the source supplied an injection label; it
is not a partition flag and cannot place a row in training. Seeing it true
alongside `data_role: dev_test` is the designed behaviour, not a leak. There is
no leakage finding here.

### Remedy

The defect is a label defect with a reporting consequence. It is not a training
defect and not a reason to drop the source.

- **Correct the label, do not delete the rows.** For the 2,400 `benign` and
  `semi_benign` rows set `injection_label` to null, `security_label` to
  `uncertain`, and `routing_label` to 1, on the grounds that the source supports
  the goal and not the prompt. `_model_eligible` then drops them from the model
  population, so they leave the benign denominator without leaving the corpus,
  and the attack family is untouched. This is a canonical-corpus change: it needs
  a full `uv run morgott data` rebuild, manifest verification,
  and an Azure push.
- **Keep them as a named specificity diagnostic.** The matched
  harmful/semi-benign/benign triple over one identical carrier is the only clean
  technique-versus-intent contrast in the corpus. Report the carrier specificity
  gap — attack flag rate minus benign-control flag rate, 11.62 points today — in
  place of an FPR. A model that closes that gap by learning to read the goal is
  improving; one that closes it by flagging less obfuscation is not.
- **Do not exclude the source.** Excluding it would hide the finding. The 86.75%
  is the most informative number the panel produces about this model: it says the
  detector keys on obfuscation form and cannot separate carrier from intent.
- **Do not train on these rows.** They are dev-test with no leakage, and fitting
  on them would teach the detector to discount cipher scaffolding — the opposite
  of what an obfuscated-jailbreak detector should learn — while retiring the only
  contrast set available to check that. This is the same reasoning already
  applied above to JBB, NotInject, FinanceBench, and the human FalseReject test.
- **Fix the reported denominator regardless of the label decision.** Every
  aggregate benign FPR in the repository currently includes these rows:
  `reports/model-experiments.md` (1.48%), `experiments/guard_baselines/README.md`
  (1.4825%), the table earlier in this file (1.50%), and the stored
  `canonical_dev_test` figures for every sibling model artifact. Quote the
  ordinary-benign rate and the control rate separately, never pooled.

### Still open

- `llmail`'s 160 `challenge_false_positive_control` rows fail at 100.00% and are
  the same class of defect — a purpose-built adversarial control counted as
  ordinary benign text — but they were not audited row by row here.
- SWE-bench verified at 5.89% is a different failure: ordinary long software-issue
  text with no attack carrier. It remains the first genuine benign domain failure
  on the list and is still unreviewed.
