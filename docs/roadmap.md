# Roadmap

morgott is still at the data-foundation stage. The broad corpus is not a trained
or promoted model, and the old POC results do not choose the final architecture.

## P0 - finish and freeze the corpus

- Build every manifest-declared source through one command and publish one manifest.
- Verify source/output digests, canonical schema, row counts, label mappings,
  exact uniqueness, group separation, and quarantine reasons.
- Preserve every valid detector-text projection and required lineage; keep
  sampling and source weighting out of corpus construction.
- Treat `train`, `validation`, and `dev_test` as development roles. Record that a
  genuinely untouched final test does not yet exist.
- Keep auxiliary, uncertain, and quarantine rows visibly separate from ordinary
  supervision.
- Routing publication now applies the audit-strict text hash across development roles before the existing SimHash check, preserving dev-test over train and train over validation.
  A full independent readback found zero normalized-text, audit-strict-text, or lineage-group crossings after publication; exact totals remain only in `data/manifest.json`.

Exit condition: a clean rebuild produces the versioned manifest, all maintained
tests and manifest invariants pass, and no second manifest or legacy data root
exists.

## P1 - first proper routing model

Start with the smallest fair comparison:

1. A character/word linear baseline on the canonical routing train split.
2. One frozen encoder comparison on identical grouped rows before any limited top-layer unfreezing.
3. Independent masked heads for direct instruction subversion, indirect instruction subversion, jailbreak, and harmful intent.
4. Derived advisory routing for benign, uncertain, and review-required outcomes.
5. Toxicity only after two independent positive source families and matched negatives exist.
6. Explicit source-balanced weighting or sampling ablations so HackAPrompt,
   LLMail, Tensor Trust, or another large family cannot dominate by volume.
7. Threshold and recipe selection on validation only.

The matched 2,000-per-source-label cap ablation was rejected because it reduced aggregate recall and increased aggregate FPR on both validation and dev-test.
Its small macro-source improvement does not satisfy item 6; another source-balance attempt needs a targeted weighting hypothesis rather than a lower global cap.
A subsequent capped inverse-square-root source-label weighting hypothesis kept every selected row and improved macro-source recall and FPR, but it missed its predeclared gate after losing 0.41 validation and 0.30 dev-test aggregate recall points and slightly reducing ranking quality.
Keep the unweighted control and do not sweep weighting constants on the consumed development roles.
The active word control now counts exact-merged rows once in aggregate metrics and in every applicable origin source slice, reports TP/FP/TN/FN, fixed-prevalence precision substitutions, and normalized-character length slices, and produces byte-reproducible versioned reports.
The maintained checks bind that report to the current canonical manifest so a corpus rebuild cannot silently leave stale baseline evidence.
Its selected fit has no benign row above 1,024 normalized characters, and its longer dev-test benign slices have 47% to 71% FPR on small denominators.
Treat that as confirmation of the genuine long-benign data prerequisite, not permission to tune a threshold or synthesize negative labels from the opened slices.

Required reporting:

- TP, FP, TN, and FN at stated operating points;
- per-source and leave-one-source-out recall;
- precision at realistic attack-prevalence scenarios;
- language, topic, text length, channel, and source-family slices;
- document-level false positives by number of windows;
- latency, memory, model revision, data-manifest hash, and random seeds.

Do not add ensembles, LoRA, custom losses, remote reviewers, or long-context
machinery until a measured error analysis predicts what each addition should
fix. Do not train every window of a positive document as positive without a
known attack span.

The repository owner authorized one bounded exception on 2026-07-28 after the external tail failure was measured and later authorized one full-mixture rank-8 LoRA seed.
That scope contains the full-data frozen mmBERT head, one update-matched reduced-mixture LoRA engineering gate, and one full-mixture LoRA seed.
On 2026-08-05 the owner separately authorized one LP-FT comparison using the rebuilt canonical mixture plus repository-grouped SWE-rebench V2 matched pairs.
LP-FT reduced held-out same-source long-task clean flags from 7.31% to 0.17%, but reduced attack recall from 98.02% to 85.20%, collapsed PromptShield ranking, and nearly eliminated BrowseSafe and BIPIA recall.
The LP-FT candidate was rejected, its weights are not registered for inference, and the full-mixture LoRA and maintained cascade remain unchanged.
On 2026-08-12 a later bounded exception completed the matched no-harm
512/1,024-token campaign. At its frozen comparison update, 1,024-token training
improved canonical and indirect-document low-FPR recall and long-code clean
specificity, but regressed PromptShield transfer and matched-pair ordering. A
later packaged-selector checkpoint exposed a different trade rather than a
clear winner.
At the campaign's close, the advisory study did not promote either checkpoint, and its retained weights, scores, archived training source, and result records were comparison and reproducibility material only; the resumable checkpoint was deleted after its digest was recorded.
On 2026-08-14, the exact update-17,000 weights were later registered for the advisory Azure preview after artifact, licensing, privacy, and reproducibility review; the other campaign material remains provenance only.
These materials do not by themselves justify more seeds, another LoRA or LP-FT
run, production calibration, promotion, or blocking.

Exit condition: one reproducible shadow candidate improves on the cheap control
across source-held-out evidence without unacceptable benign review load. This is
still not permission to block users.

### July 2026 experiments and stop decision

The frozen four-head quick comparison failed its finance and BrowseSafe false-positive gates.
Longer 2,048-token training made BrowseSafe ranking worse, while document bags and pair-balanced BIPIA data produced only a narrow indirect-route improvement.
A later direct repair added group-held-out multi-turn rows and clean non-adversarial WildGuardMix counterexamples.
The clean repair reduced the English ModernBERT direct-suite FPR to 0.221% at 75.68% recall and the mmBERT-base FPR to 0.106% at 74.73% recall under the nominal 0.1% per-head profile.
A validation-calibrated mean of the two backbones' direct-route probabilities reached 0.101% FPR and 77.16% recall on the open direct suite.
These are single-seed development shadows, not a promoted global model.
No model is approved for global routing or blocking.

### July 2026 external validation - the shadow does not transfer

The above numbers were subsequently checked outside the corpus and against a
retrying adversary. Both checks failed. Full detail and reproduction commands
are in `reports/model-experiments.md`; the summary that changes how the
numbers above should be read:

- **PromptShield public test split (23,516 rows, 1.374% benign-only overlap
  with morgott training views): ROC AUC 0.624 and 0.00% TPR at 1% FPR.** Last
  place against every published baseline on that split, below ProtectAI v2.
  Every head, both members and all fusions cap at 0.52% TPR. The cause is not
  truncation (chunked scoring lifts AUC to 0.661 and leaves TPR at 0.00%) and
  not head selection. Benign application prompts that carry instructional
  language outrank every attack.
- **Attempt scaling: ASR@1 is 49.20% and ASR@32 is 98.30%** using free
  surface mutations, so the 77.16% recall figure corresponds to roughly
  **1.38% effective recall** against a 32-attempt adversary.
- **The 0.101% FPR describes short text.** 98.5% of the negatives it averages
  over are 64 tokens or fewer; above that the measured FPR is 1.1%-6.7%.
- **Multi-turn recall at the same operating point is 1.79%.** Multi-turn rows
  were filtered out of the suite the headline was computed on.
- **The leakage check misses obfuscated duplicates.** Twelve groups span more
  than one split under a normaliser that handles zero-width and homoglyph
  evasion; one payload appears in train, validation and dev-test at once.

Treat the 0.101%/77.16% pair as an in-corpus artifact, not a detector result.

### July 2026 bounded generic first-pass artifacts

The full-data frozen mmBERT and full-mixture LoRA recipes fit leakage-filtered canonical rows, PromptShield train, and the retained generated pairs with balanced BCE plus pair ranking.
Three frozen-head seeds are retained to show dispersion.
The reduced-mixture rank-8 LoRA is retained only as an update-matched engineering gate against a frozen control; it omits most canonical rows and all generated pairs.
The full-mixture rank-8 LoRA is one seed selected at epoch 2 after 25,071 updates and is retained as historical advisory evidence without promotion.
PromptShield validation selects checkpoints, while canonical calibration alone selects reported applied thresholds.
PromptShield test is source-disjoint within its own release and remains already-consumed benchmark development data.
Known cross-corpus overlap prevents a source-OOD claim relative to the complete Morgott plus PromptShield fit.
SEP remains already-consumed transfer development data.
All retained downstream checkpoints are advisory and must be compared independently rather than combined into an unevaluated ensemble.
The historical 512-token scores remain comparison evidence and are not registered for maintained inference.
The sole registered Azure preview uses the update-17,000 candidate with 1,024-token windows, while complete long-document aggregation remains an explicit evidence gap.

### Downstream cascade candidate

The July no-manual-review development candidate used the partial-data `mmbert-lora-s42` artifact followed by DeepSeek V4 Flash only for mmBERT scores from `0.2` through values below `0.999`.
Scores below `0.2` pass the advisory sensor, scores at or above `0.999` restrict, and middle-zone rows restrict when the normalized DeepSeek two-token probability is at least `0.9`.
An exhausted DeepSeek failure also restricts.
The `0.9` threshold was selected on the fixed 6,000-row calibration split and applied once to the separate 14,000-row evaluation split.
It is an operating threshold, not a calibrated production probability.
Exact evidence, provider settings, metrics, and limitations are in [the OpenRouter downstream evaluation](../reports/openrouter-downstream-evaluation.md).

The completed full-data LoRA cannot inherit the partial-LoRA score gates directly.
The copied policy reaches 78.40% recall at 3.62% FPR, while a post-hoc `0.99999` high-gate extension reaches 66.79% recall at 1.81% FPR with 22.17% DeepSeek calls.
That July extension was not in the predeclared grid.

The August channel-aware follow-up supersedes the July maintained route.
It uses a `0.2` direct-user floor, a `0.1` untrusted-content floor, a shared `0.99999` high gate, trusted `input_channel` metadata, and a hybrid outer-intent prompt.
The channel-specific floor keeps calibration FPR at 1.9808% while adding 10 true positives and 49 provider calls over 6,000 rows.
The selected reviewer is now DeepSeek V4 Flash 0731 on Cloudflare with its separately calibrated `0.6224593312018547` threshold.
On the frozen 14,000-row evaluation role, the then-registered 512-token FP32 route reached 71.235% / 1.773% / 96.779% recall / FPR / precision at a 22.914% provider call rate.
This adds 58 true positives with no additional false positives over the April route, although PromptShield recall falls by 2.176 points and untrusted-content FPR rises by 0.375 points.
Its OpenVINO BF16 runtime reached 71.386% / 1.798% / 96.742%, differed from FP32 on 27 of 20,000 final routes, and passed every serving-equivalence gate.
A frozen-ledger-only call-minimization check rejected further scalar gate tightening: the exact-decision rule saved seven evaluation calls and lost one true positive, while the best simple noninferior calibration grid saved 87 calls but lost nine true positives on evaluation.
Keep the existing gates; any future low-call policy must be a materially different prospectively frozen architecture rather than another retune on this consumed panel.
The selection follows the owner's aggregate-quality criterion and remains already-open advisory development evidence rather than a production promotion.
See [the 0731 research and decision report](../reports/deepseek-v4-flash-0731-research.md).
Two separately frozen Ionstream canaries then failed the exact classifier reliability gate at both four and one concurrent requests because 8 of 20 and 2 of 20 rows respectively exhausted HTTP 429 retries.
Ionstream calibration and evaluation were not opened, provider fallback remains disabled, and the measured Cloudflare route remains selected.

A separate matched-boundary sequence tested script-aware normalization, the broader April-reviewer prompt, and the existing word n-gram control without changing the maintained route.
The final frozen 1,008-row multilingual panel used all 15 remaining scenario-disjoint source scenarios and passed every predeclared gate at 97.02% / 1.79% English recall / FPR, 98.21% / 0.60% Russian, and 95.83% / 1.19% Ukrainian.
The linear gate added 13 low-tail reviews, all attacks, while total candidate review load remained 88.29%.
This is successful source-specific synthetic development evidence, but all source scenarios are now consumed and the remote-call rate is too high to supersede the maintained cascade.
Do not integrate or retune it until an independently sourced matched panel and representative benign denominator reproduce the gain.

A frozen provider-free transfer check then scored 571 privacy-filtered, zero-fit-overlap benign prompts from the public Operant AI false-positive dataset.
The linear gate admitted only 1.40% overall but admitted 14.29% of the 49 English rows, above its frozen 10% per-language cap, while the current mmBERT high gate admitted two source-benign rows.
The linear candidate was rejected and its planned OpenRouter reviewer phase was canceled.

The next provider-free diagnostic froze 280 unseen payload families from the ACL Inj-SQuAD and Inj-TriviaQA release into 1,657 complete clean-plus-three-position groups.
The registered local gate admitted only 65.10% of attacked documents and 61.80% of exact payload spans, with SQuAD head admission at 32.62% and clean TriviaQA review at 52.14%.
A post-hoc threshold grid found no scalar repair, and a 128-token development candidate increased benign load and compute too sharply.
The remote phase was canceled because low-zone attacks cannot reach the reviewer.
This narrows the next architecture target to request-conditioned task-versus-content assessment and deterministic authorization rather than text-only window retuning.

A subsequent task-conditioned canary paired the trusted source question with one privacy-safe document per source and each of the 20 already-consumed payload families.
The reviewer produced zero clean flags and 100% attack-over-clean ordering, but its fixed `0.85` threshold reached only 75% SQuAD-head recall, 45% TriviaQA-head recall, and 25% TriviaQA-midpoint recall.
The canary therefore failed its frozen operating-point gates and the remaining 280 payload families were not remotely reviewed.
A post-hoc lower threshold is diagnosis, not promotion evidence.
Any follow-up must use independent request-conditioned calibration and evaluation data and solve the all-document call-load problem before integration is considered.

The independent StruQ follow-up froze both splits before calls after privacy screening and full-fit overlap filtering.
Calibration retained `0.85`, and the untouched 157-pair evaluation flagged all 471 attacks with 100% paired ordering while flagging 1 clean input, or 0.637%.
This supports the task-conditioned architecture hypothesis only for three explicit fixed-output attack constructions.
It does not solve the 100% remote-review rate or establish transfer to natural documents, varied attacker goals, tool outputs, retrieval content, or adaptive attacks, so no maintained interface or threshold changed.

The next fixed transfer used all 1,054 InjecAgent user-and-attacker pairs across 62 varied tool-output goals.
The unchanged response-only cascade reached 70.40% base recall and 100% enhanced recall with 0 of 17 clean templates restricted.
Task-conditioned DeepSeek V4 Flash 0731 at the transferred `0.85` threshold raised base recall to 78.27%, retained zero clean flags, and ranked every attack above its paired clean response, but failed aggregate and worst-slice gates.
A post-hoc `0.3` grid point reaches 96.96% recall with zero clean flags on this consumed source, so it diagnoses score-scale mismatch but cannot select a maintained threshold.
The next threshold gate needs a larger independent clean tool-output denominator, separately sourced varied attacks, and a frozen invocation strategy.

The subsequent API-Bank clean study first exposed a projection defect in the benchmark adapter.
Sending the complete serialized API call and result as untrusted content produced a 20.31% exact-unique flag rate at `0.3`, while sending only the returned value reduced it to 3.68%.
The corrected projection still failed the frozen clean gates at `0.3`.
A sealed post-hoc grid found `0.5` to be the first fixed `0.1` point that passed all retained API-Bank gates, with 0.70% exact-unique flags and 94.88% InjecAgent recall.

That `0.5` candidate was frozen before AgentDyn reviewer outcomes were opened.
After retrying every and only 55 rate-limited calls at concurrency `8`, it flagged all 560 task-and-goal attacks across all suites, goals, and tasks.
This validates transfer for one explicit fixed instruction template, but not integration: the threshold used consumed clean evidence, API-Bank excluded fit-overlapping outputs, and AgentDyn had no clean, adaptive, or tool-execution arm.

The prospectively frozen AgentPIMA gate then supplied 672 progressive attacks and their matched clean artifacts across 112 trusted tasks.
Threshold `0.5` retained about 95% attack recall but flagged 38.52% and 27.63% of the two exact-unique clean variants and at least one clean input for 91.07% of tasks.
An exhaustive post-hoc diagnostic found that satisfying every clean gate reduced worst-variant attack recall to 50.45%, so the candidate is rejected rather than retuned.
The next architecture target is deterministic authorization plus a frozen low-call invocation policy, not another scalar threshold over this all-row reviewer.

The independent SafeClawBench follow-up then tested whether direct-user prompt specialization could repair the context-free boundary detector.
The maintained cascade detected only 9 of 89 DPI prompts because 80 positives passed below the local reviewer floor, while all-row 0731 with the current prompt reached 48 positives at 11 of 94 ADI false positives.
The specialized prompt reduced all-row performance to 3 positives and 2 false positives, and no threshold on the consumed panel met its fixed quality targets.
This rejects prompt and scalar-gate repair and narrows the next architecture target to trusted task, policy, and capability context plus deterministic action outcomes on an independent source.

The maintained code verifies and serves the registered full-mixture LoRA through ordered windows, strictly parses DeepSeek decision-token log probabilities, and fails conservatively after bounded retries.
For remote-enabled multi-window untrusted content without a local high, it now reviews the complete normalized artifact once before the existing middle-window branch.
A full-context flag restricts early, while a clear result falls back to middle-window reviews in batches of at most 4.
Direct-user and single-window behavior remain unchanged.
A prospectively frozen one-review replacement rejected itself after losing one PIArena completion catch, while the retained full-first union replays at 725 of 732 multi-window attack restrictions versus 697 for the old branch with the same 2 of 183 matched-clean restrictions.
See [the long-context reviewer report](../reports/long-context-reviewer-research.md).
Middle-window batches stop before later batches after a definitive flag or exhausted failure, so a decisive fallback batch can avoid up to 124 of the 128 permitted window calls.
Production reviewer initialization also suppresses LiteLLM's unsolicited error banners so handled retries cannot corrupt the CLI's machine-readable JSON output.
All-clear multi-window artifacts now incur the full review and every eligible fallback window, so this remains uncalibrated for representative benign traffic.
The prospectively frozen LongBench Pro control stopped before OpenRouter after 7 of 471 natural documents hit the local-high branch and the remaining all-clear workload projected 17,877 fallback reviews against an 8,192-window ceiling.
A post-hoc PIArena replay found that the eight highest-scoring fallback windows preserve the consumed panel's exhaustive-union decisions and would fit the LongBench workload budget, but this rule is not integrated or promoted.
The first such panel used 59 fit-disjoint LongBench v2 pairs and stopped before OpenRouter because none of its clean artifacts reached the local high zone, despite nine high-zone attacks.
That panel is consumed and did not authorize a cascade change.
The next panel supplied 100 fit-disjoint pairs balanced across all five official Chinese LongBench tasks and contained 3 local-high clean artifacts plus 44 local-high attacks.
Its exact full-context plus top-eight candidate caught 99 attacks and lost no local-high attack catch, but restricted 11 clean controls against the frozen maximum of 2 and reached only 88 fully correct pairs against the minimum of 90.
The candidate is rejected and the maintained exhaustive fallback remains unchanged.
Do not spend another consumed benchmark on scalar threshold or window-count tuning.
The next architecture target is trusted task or policy context plus deterministic authorization outcomes, evaluated prospectively on realistic stateful tasks.
The serving verifier now binds typed reviewer evidence to the current prompt, request, provider, panel identity, and trusted channel, while harmful non-injection classification stays outside the subversion route.
The following integration work remains:

1. Map `restrict` to privilege reduction or deterministic reference-monitor policy rather than treating a learned score as authorization.
2. Shadow the complete pipeline on representative traffic and recalibrate before deployment.

The disposable OpenRouter experiment runner is reproducibility code, not the production adapter.
Do not import it into the maintained package.

Before another encoder-tuning experiment, collect realistic matched transaction attacks and benign tasks plus prospective traffic-like negatives.
The completed PIArena static panel now supplies same-row clean and attacked retrieval, question-answering, summarization, and long-context development evidence, but it is evaluation-only, publicly predates the remote model, and cannot authorize fitting.
The repository-grouped SWE-bench Verified problem-statement slice now supplies a dev-test-only long-benign FPR denominator, not fit-data evidence or the missing same-format attack arm.
Paired multilingual transformations and an independently sourced known-span long-document diagnostic now exist, but the latter rejects text-only window and threshold repairs rather than supplying fit-data evidence.
The future fit-leakage audit now removes U+034F and supplementary variation selectors U+E0100 through U+E01EF and filters one same-label intra-training duplicate.
The registered model-input normalizer still preserves those code points; correct that behavior only as a prospectively evaluated model-input contract, not as an inference-only patch to retained weights.
Do not repeat context-length, document-bag, or BIPIA augmentation ablations on the same labels.
Do not sweep reinforcement-learning, focal-loss, or source-weighting objectives over the current open dev roles as a substitute for new evidence.
The rejected LP-FT comparison confirms that matched long-task data can repair same-source clean workload while damaging external transfer; any future run requires identical-row incumbent scoring and frozen cross-source and indirect-document gates.
Do not use the Rogue Security benchmark as independent evidence because more than half of its rows exactly overlap current canonical public sources.

## P2 - prospective evaluation

- Freeze a genuinely untouched final test before using its outcomes for model, threshold, or architecture selection.
- Score every future learned candidate and the incumbent on identical rows, including source-heldout detection, PromptShield, indirect documents, the retained mutation curve, and long legitimate workloads.
- Keep SWE-bench Verified, SWE-rebench V1, SWE-chat, PIArena, Adaptive Adversaries, FORCE-Bench, BFCL, AgentAbstain, and every other opened panel out of fitting and threshold selection.
- Add realistic application traffic only when independently labelled; report source, time, language, length, channel, participant, and provider-failure slices.
- Treat workload restriction and review incidence as workload measurements rather than production FPR when benignity was not adjudicated.
- Do not spend another consumed benchmark on scalar threshold, window-count, or prompt tuning.
- Require a materially different architecture and a prospectively frozen evaluation before revisiting task-conditioned review or long-document routing.

Completed outcomes and exact limitations are in [the model ledger](../reports/model-experiments.md), [the DeepSeek report](../reports/deepseek-v4-flash-0731-research.md), [the long-context report](../reports/long-context-reviewer-research.md), and [the task-conditioned report](../reports/task-conditioned-reviewer-evaluation.md).

## P3 - stateful agent containment

- Keep learned scores advisory and enforce every proposed side effect through typed, task-scoped capabilities.
- Bind variable arguments to trusted stable source identities and propagate provenance and sensitivity monotonically through transformations and egress.
- Require every containment experiment to establish a vulnerable control, load-bearing source observation, exact legitimate state, exact unauthorized state, and a monitored denial before claiming causal containment.
- Replace oracle capability construction with a trusted application adapter and capability broker before making deployment claims.
- Treat clarification replies as typed values bound to the original missing slots; require a separate trusted authorization event for scope expansion.
- Broker short-lived credentials outside the planner, bind human approval to exact irreversible actions, and prevent untrusted content from creating capabilities or durable memory.
- Measure task success, attack success, denied calls, unauthorized state, secret egress, and provider failures rather than classifier accuracy alone.

The bounded AgentDojo and load-bearing stateful findings are in [the AgentDojo report](../reports/agentdojo-integration-research.md) and [the stateful benchmark report](../reports/agent-security-benchmark-options.md).

The core security thesis remains: prediction reduces exposure; deterministic
authorization bounds impact when prediction fails.
