# Roadmap

The corpus foundation and first maintained advisory model milestone are complete.
The active work is prospective traffic evaluation, long-input error reduction, and stateful containment.
No model is approved for blocking.

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

## P1 - first proper routing model (completed advisory milestone)

The phase began with the smallest fair comparison:

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

This was the gate before the later bounded LoRA, reviewer, and long-context exceptions.
Do not train every window of a positive document as positive without a known attack span.

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
The sole registered Azure preview used the update-17,000 candidate with 1,024-token windows; the complete all-window cascade replay was added later in the 2026-08-16 pipeline benchmark.

### Downstream cascade candidate

Each consumed cascade experiment below is recorded as its decision plus any
operative constraint; complete narratives, metrics, and provider settings stay
in the linked reports and [the model ledger](../reports/model-experiments.md).

- July no-manual-review candidate (partial-data `mmbert-lora-s42` plus DeepSeek
  middle zone at `0.9`): consumed development evidence with an operating
  threshold, not a calibrated production probability. See
  [the OpenRouter downstream evaluation](../reports/openrouter-downstream-evaluation.md).
  The completed full-data LoRA cannot inherit the partial-LoRA score gates
  directly, and the July `0.99999` high-gate extension was not in the
  predeclared grid.
- August channel-aware route (floors `0.2`/`0.1`, shared `0.99999` high gate,
  DeepSeek V4 Flash 0731 on Cloudflare): the maintained incumbent through
  2026-08-16; its OpenVINO BF16 runtime passed every serving-equivalence gate.
  A frozen-ledger-only call-minimization check rejected further scalar gate
  tightening.
  Keep the existing gates; any future low-call policy must be a materially different prospectively frozen architecture rather than another retune on this consumed panel.
  See [the 0731 research and decision report](../reports/deepseek-v4-flash-0731-research.md).

On 2026-08-17, the owner promoted the benchmark-selected exact balanced profile as the maintained advisory default.
It uses direct and untrusted floors `0.2` and `0.025`, local high `0.9999`, reviewer threshold `0.5`, the unchanged Cloudflare strict-logprob request, and the existing 1,024/128 full-context-first window flow.
On 12,352 provider-safe consumed evaluation artifacts, CUDA BF16 reached 72.00% recall and 1.37% FPR; transport of the frozen profile to the full OpenVINO CPU ledger reached 72.10% recall and 1.36% FPR.
The source-complete provider audit selected no strict no-logprob winner; Decart remains diagnostic because its HackAPrompt recall loss exceeded the declared 2 percentage-point slice gate.
Cloudflare remains the selected strict-logprob route.
The registry binds the exact profile and evidence record, but the promotion changes advisory defaults only and preserves `decision: allow`.
The long-document warning, absence of representative adjudicated traffic, and consumed development roles still prohibit blocking, SLA, or production-quality claims.
See [the complete pipeline benchmark](../reports/pipeline-benchmark-20260816.md).
Two separately frozen Ionstream canaries failed the exact classifier reliability gate; Ionstream calibration and evaluation were not opened, provider fallback remains disabled, and the measured Cloudflare route remains selected.

On 2026-08-19 the owner promoted the registry-bound `balanced-retrieval-20260819` profile as the maintained advisory default (recorded here 2026-08-20).
The review route now runs the registered source-lineage HNSW plus BM25/RRF example retriever before the fixed DeepSeek reviewer, with direct and untrusted floors `0.2` and `0.025` and local high `0.9999`.
The consolidated evidence and the selected integration recipe are in [the retrieval-assisted reviewer findings](../reports/retrieval-assisted-reviewer-findings-20260819.md).
The promotion changes advisory defaults only: every assessment still returns `decision: allow`, and the blocking, SLA, and production-quality prohibitions above are unchanged.

The subsequent consumed reviewer and boundary diagnostics, as decisions plus
operative constraints (full narratives in the ledger and topic reports):

- Matched-boundary multilingual sequence (script-aware normalization, April
  prompt, word n-gram gate): passed every predeclared gate on the final frozen
  panel, but all source scenarios are consumed and the remote-call rate is too
  high to supersede the maintained cascade.
  Do not integrate or retune it until an independently sourced matched panel and representative benign denominator reproduce the gain.
- Operant AI provider-free transfer check: the linear gate broke its frozen 10%
  per-language cap on English rows; the candidate was rejected and its planned
  OpenRouter reviewer phase was canceled.
- Inj-SQuAD/Inj-TriviaQA known-span diagnostic: the registered local gate
  admits too few attacked documents and payload spans, no scalar repair exists,
  and the remote phase was canceled because low-zone attacks cannot reach the
  reviewer.
  This narrows the next architecture target to request-conditioned task-versus-content assessment and deterministic authorization rather than text-only window retuning.
- Task-conditioned canary: zero clean flags with perfect ordering but failed
  its frozen operating-point gates; a post-hoc lower threshold is diagnosis,
  not promotion evidence.
  Any follow-up must use independent request-conditioned calibration and evaluation data and solve the all-document call-load problem before integration is considered.
- StruQ follow-up: supports the task-conditioned hypothesis only for three
  explicit fixed-output attack constructions; no maintained interface or
  threshold changed.
- InjecAgent transfer: failed aggregate and worst-slice gates at the
  transferred `0.85`; the post-hoc `0.3` point diagnoses score-scale mismatch
  but cannot select a maintained threshold.
  The next threshold gate needs a larger independent clean tool-output denominator, separately sourced varied attacks, and a frozen invocation strategy.
- API-Bank clean study plus AgentDyn: the output-only projection is the correct
  runtime boundary and a sealed grid found `0.5`, which then flagged all
  AgentDyn attacks; this validates one explicit fixed instruction template, not
  integration, because the threshold used consumed clean evidence and AgentDyn
  had no clean, adaptive, or tool-execution arm.
- AgentPIMA gate: threshold `0.5` was rejected rather than retuned because
  satisfying every clean gate halves worst-variant attack recall.
  The next architecture target is deterministic authorization plus a frozen low-call invocation policy, not another scalar threshold over this all-row reviewer.
- SafeClawBench follow-up: rejects prompt and scalar-gate repair and narrows
  the next architecture target to trusted task, policy, and capability context
  plus deterministic action outcomes on an independent source.

The maintained code verifies and serves the registered full-mixture LoRA through ordered windows, strictly parses DeepSeek decision-token log probabilities, and fails conservatively after bounded retries.
For remote-enabled multi-window untrusted content without a local high, it now reviews the complete normalized artifact once before the existing middle-window branch.
A full-context flag restricts early, while a clear result falls back to middle-window reviews in batches of at most 4.
Direct-user and single-window behavior remain unchanged.
A prospectively frozen one-review replacement rejected itself after losing one PIArena completion catch, while the retained full-first union replays at 725 of 732 multi-window attack restrictions versus 697 for the old branch with the same 2 of 183 matched-clean restrictions.
See [the long-context reviewer report](../reports/long-context-reviewer-research.md).
Middle-window batches stop before later batches after a definitive flag or exhausted failure, so a decisive fallback batch can avoid up to 124 of the 128 permitted window calls.
Production reviewer initialization also suppresses LiteLLM's unsolicited error banners so handled retries cannot corrupt the CLI's machine-readable JSON output.
All-clear multi-window artifacts now incur the full review and every eligible fallback window, so this remains uncalibrated for representative benign traffic.
The three consumed long-document panels each ended in a decision without a cascade change: the LongBench Pro control stopped before OpenRouter because the all-clear workload projected far past the window ceiling; the LongBench v2 panel stopped before OpenRouter and did not authorize a change; and the Chinese LongBench panel rejected the full-context-plus-top-eight candidate on its clean gates, so the maintained exhaustive fallback remains unchanged.
The post-hoc top-eight fallback replay is diagnosis only and is not integrated or promoted.
Do not spend another consumed benchmark on scalar threshold or window-count tuning.
The next architecture target is trusted task or policy context plus deterministic authorization outcomes, evaluated prospectively on realistic stateful tasks.
The serving verifier now binds typed reviewer evidence to the current prompt, request, provider, panel identity, and trusted channel, while harmful non-injection classification stays outside the subversion route.
The following integration work remains:

1. Map `restrict` to privilege reduction or deterministic reference-monitor policy rather than treating a learned score as authorization.
2. Shadow the complete pipeline on representative traffic before any blocking or production-quality claim.

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
