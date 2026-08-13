# First-party red-team campaign corpus

Decision record for the campaign data at `data-archive/redteam/`, added 2026-08-06.
The data card and reading rules are in `data-archive/README.md`; this file records what
the corpus is, why it is not a canonical source, and what it measured.

## Decision

Retain as a frozen evaluation reserve. It does not enter `data/sources/`, and it does not
enter routing train, validation, or dev-test.

## What it is

96 automated red-team campaigns run 2026-07-20 to 2026-07-28. Attacker models generated
attack prompts against target `z-ai/glm-5.2`; `openai/gpt-5.4-mini` and `openai/gpt-4o`
judged the outcomes. 5,133 rows in, **5,112 retained, 21 quarantined** as unusable detector
input. 26 of the retained rows carry `text_source: "generator_envelope"` — see the recovery
note under Known defects.

| `source_record_type` | rows | target response | judge verdict |
| --- | ---: | ---: | ---: |
| `classic_memory` | 3,277 | 0 | 0 |
| `concise_attempt` | 1,605 | 1,587 | 1,605 |
| `classic_attempt` | 226 | 221 | 226 |
| `agentic_episode_turn` | 21 | 19 | 6 |
| `classic_trace` | 4 | 0 | 0 |

Roughly two thirds of the file is unlabelled attacker-candidate text that was never sent to
a target. 1,837 rows carry a verdict (162 Breached, 68 Partial, 1,607 Failed); 3,296 do not.

## Why it is not a canonical source

1. **Scale makes it inert as training data.** 5,133 rows against 1,477,802 routing-train
   rows is 0.35% of train and 0.24% of the supervised corpus.
2. **It would deepen the known shortcut.** It is 100% positive, effectively one input
   channel, one target model. `reports/model-experiments.md` records that thirteen of
   fourteen direct-user training sources are already single-class, that raw source identity
   explains 87.75% of direct-user label entropy, and that a source-majority lookup reaches
   95.94% validation accuracy without reading any text. This would be the fourteenth
   single-class direct-user source.
3. **It is none of the four prerequisites.** `docs/data-contract.md:160` gates the next
   encoder run on matched transaction tasks, paired multilingual transformations,
   known-span long-document attacks, and stronger same-source controls. Short direct-user
   attack text is already covered by hackaprompt, tensor_trust_raw, and wildjailbreak.
4. **This source is reserved for evaluation.** Its rare fresh positives are
   more valuable as untouched evidence than as another single-class fit source.
5. **It is uncontaminated, and that is scarce.** 5,888 distinct normalized texts across all
   three archived files were checked against all 2,177,717 routing-view rows — train,
   validation, dev_test, and uncertain — under the corpus's own `normalized_text_sha256`.
   **Zero exact overlap in every view.** `docs/roadmap.md:241` requires a genuinely
   untouched final test, which does not yet exist. Spending this set on 0.35% of train
   would destroy the only fresh held-out positives available.

## Measured: the matched subversion control

The single most useful thing in this corpus is a control the canonical data lacks.

864 rows use one fixed jailbreak wrapper and end in a literal `Request: <inner>` tail. All
864 parse, and all 864 inner requests are distinct and appear nowhere else in the file as
standalone text. Splitting the wrapper off constructs 864 pairs whose halves carry
identical harmful content and differ only by the presence of instruction subversion.
(A 865th wrapper row exists in the source but is degenerate — a 52,341-character run of
`█` — and is quarantined rather than paired.)

82 of the 864 pairs are excluded before scoring: their wrapped half exceeds the scorer's
512-token limit, and `batch_logits` truncates from the end without windowing
(`core.py:129-135`). Because the request sits at the *end* of the wrapper, those wrapped
halves would be scored having seen less of the harmful request than their bare
counterparts — no longer a matched comparison. **782 pairs are scored**, through the
registered advisory shadow `mmbert-lora-full-s42` at the direct-user pass floor of `0.2`:

| half | flag rate | median score |
| --- | ---: | ---: |
| `wrapped` (subversion present) | **100.00%** (782/782) | 0.99980 |
| `inner` (identical request, no wrapper) | **18.03%** (141/782) | 0.00232 |

Delta 81.97 points. 641 pairs flag only when wrapped; **0** pairs flag only when bare; **0**
pairs flag neither. Flag rate on `inner` by category: cyber_misuse 123/402 (30.6%),
weapons_and_explosives 12/219 (5.5%), physical_harm 6/161 (3.7%).

**Reading.** The detector keys on instruction subversion, not on harm topic. Holding
content fixed and adding the wrapper moves the median score from 0.003 to 0.9997.

This settles how to read the positive-only recall figures in
`experiments/cascade_showcase/`. That panel reports 39.44% attempt recall for the cascade
over 5,132 rows, which has been read as a 61% miss rate. It is not. A marker scan puts
about 67% of the corpus in the bare-harmful-request class, and `docs/data-contract.md:46-56`
makes `harmful_non_injection` a different label from injection — the same treatment
HarmBench already gets in this corpus (`injection_label: 0`,
`label_basis: task_definition:harmful_goal_not_injection`). The already-published slices
agree: `concise_attack_prompt` scores 68.2% and `attack_prompt` 26.3%. The aggregate is a
weighted average across two label classes and should never be quoted as attack recall.

Artifacts: `artifacts/redteam_subversion_control/{summary.json,scored_pairs.jsonl.gz}`,
regenerated by `experiments/redteam_subversion_control/run.py`. Advisory only; `decision`
remains `allow`.

**Scores are batch-composition dependent.** The panel scores the two halves as two separate
`score_file` calls. Measured both ways on an earlier 864-pair run: at 864 — exactly 108 ×
the scorer's batch size of 8 — merging the halves into one file reproduced all 1,728 scores
**bit for bit**, because every batch stayed length-homogeneous. At 865 the boundary batch
mixed a long wrapped text with short bare ones and **146 scores shifted, by up to 0.0646**
(none crossed the 0.2 floor). So padded batching does perturb scores, and it only bites
when a batch mixes very different lengths. The two-call structure is kept for that reason:
it guarantees homogeneous batching at any pair count, not just multiples of 8. This is a
property of the shared scorer, not of this panel, and it bounds how exactly any per-row
comparison against these artifacts can be expected to match.

## Known defects

- `record_id` is not unique (4,986 distinct across 5,133 rows), so the projection derives a
  composite id from `run_id`, `record_id`, and the normalized text hash.
- 865 source rows (16.9%) share one wrapper prefix with only two rule-block variants; 864
  survive quarantine.
- TAP and PAIR campaigns are attack trees; children are mutations of parents. The leakage
  atom is `split_group_id`, not the row.
- `category` is confounded with `attack_mode`: all 464 `illicit_drugs`, all 410
  `financial_fraud`, and all 184 `chemical_biological_harm` rows ran under `tap`.
- **Generator envelopes, recovered.** 33 source rows carry
  `{"strategy": ..., "prompt": "<attack>"}` instead of a bare prompt. The JSON does not
  parse — the prompt text holds unescaped double quotes, raw newlines, and bad escapes, so
  `json.loads` fails on all 33 and `strict=False` rescues only 6. Anchoring on the key
  boundaries instead of the grammar recovers **26**; those rows are retained with
  `text_source: "generator_envelope"`, marking `text` as not byte-equal to the source
  `prompt`. The 7 that do not recover have no `prompt` key at all — they are AgentDojo
  tool-return reprs (`{'message': 'Transaction ... sent.'}`), correctly not attack prompts.
- **Degenerate generations, now caught.** Three rows are model degeneration: runs of
  126,720 `A` and two of 45,000+ `█`. One was previously retained as a valid prompt and one
  was a wrapper row that would otherwise have become a matched pair. A 100-identical-
  character guard removes all three; the count is unchanged at thresholds of 200 and 500,
  so it is not tuned to this data.
- 21 rows remain unusable: 8 non-prompt fragments, 5 empty or trivial, 4 attacker-model
  refusals recorded as prompts, 3 degenerate repetitions, 1 special-token spam.
- One retained row still embeds a JSON envelope inside a markdown transcript (it starts
  with `"""` and a code fence, so the envelope rule does not fire). Left as-is: widening
  the trigger for a single row would risk false positives on prompts that discuss JSON.
- The raw parquet's path columns embed the generating machine's user directory. The
  projection drops them.
- Goal seeds partly derive from public benchmarks already locked to dev-test: 2 of the 26
  distinct `goal` strings exact-match corpus text.

## Open gap before any final-test use

Exact overlap is measured and zero. **Near-duplicate overlap is not measured.** Run the
repo's own audit-strict plus SimHash path (`morgott.overlap`) over the projection against
`data/views/routing/dev_test.jsonl` and record the result before this set is treated as a
prospective final test. `docs/roadmap.md` disqualifies the Rogue Security benchmark on
exactly this ground.

## What would justify promotion

All three, not any one: a benign denominator produced by the same generator, at least a
second target model so the outcome metadata is not glm-specific, and an authorized encoder
run under a prospectively frozen, same-row comparison protocol.

## AgentDojo

`raw/agentdojo_workspace.parquet` is archived for provenance and cross-check only.

Its 6,330 rows contain 693 distinct payloads and 63 distinct user prompts; each payload
recurs 1–29 times (median 4). The payload is a function of attack template, injection task,
and vector multiplicity, and is independent of the user prompt — the most-replicated
payload appears 29 times against 29 different prompts. Each of the 11 attack templates
yields exactly 63 payloads over 14 injection tasks, and the six `important_instructions*`
templates are 0.907–0.978 similar to one another. Scoring all 6,267 attack rows would
weight each payload by how often the harness replayed it.

The carrier document is absent: 0 of the 693 payloads contain an email or document wrapper.
What the benchmark actually sent was a full tool return with the payload substituted into a
vector placeholder, and that text was not materialized into the parquet. A pair-level
detector evaluation therefore cannot be built from this file.

A workspace and travel detector panel remains possible but is a separate study, not an
extension of `experiments/agentdojo_detector_eval`: that runner hardcodes
`SUITE_NAME = "banking"`, raises unless a case exposes exactly one injection vector and
exactly one message changed — workspace has 16 vectors and travel 13, reached several at a
time — and self-hashes its own source, so it cannot be edited in place. It is also
network-mandatory. AgentDojo text stays out of the training corpus either way, per
`reports/agentdojo-integration-research.md:7`.
