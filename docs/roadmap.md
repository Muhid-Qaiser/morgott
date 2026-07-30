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
The completed scope contains the full-data frozen mmBERT head, one update-matched reduced-mixture LoRA engineering gate, and one full-mixture LoRA seed.
It does not authorize more seeds, a LoRA sweep, production calibration, promotion, or blocking.
The maintained trainer remains reproducibility code; code availability is not evidence or approval for another run.

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
The full-mixture rank-8 LoRA is one seed selected at epoch 2 after 25,071 updates and is registered as an advisory research result without promotion.
PromptShield validation selects checkpoints, while canonical calibration alone selects reported applied thresholds.
PromptShield test is source-disjoint within its own release and remains already-consumed benchmark development data.
Known cross-corpus overlap prevents a source-OOD claim relative to the complete Morgott plus PromptShield fit.
SEP remains already-consumed transfer development data.
All retained downstream checkpoints are advisory and must be compared independently rather than combined into an unevaluated ensemble.
Every score is truncated to the first 512 normalized tokens, so long web, retrieval, and tool content remains an explicit evidence gap.

### July 2026 downstream cascade candidate

The selected no-manual-review development candidate uses the partial-data `mmbert-lora-s42` artifact followed by DeepSeek V4 Flash only for mmBERT scores from `0.2` through values below `0.999`.
Scores below `0.2` pass the advisory sensor, scores at or above `0.999` restrict, and middle-zone rows restrict when the normalized DeepSeek two-token probability is at least `0.9`.
An exhausted DeepSeek failure also restricts.
The `0.9` threshold was selected on the fixed 6,000-row calibration split and applied once to the separate 14,000-row evaluation split.
It is an operating threshold, not a calibrated production probability.
Exact evidence, provider settings, metrics, and limitations are in [the OpenRouter downstream evaluation](../reports/openrouter-downstream-evaluation.md).

The completed full-data LoRA cannot inherit the partial-LoRA score gates directly.
The copied policy reaches 78.40% recall at 3.62% FPR, while a post-hoc `0.99999` high-gate extension reaches 66.79% recall at 1.81% FPR with 22.17% DeepSeek calls.
The extension was not in the predeclared grid and does not replace the retained route without prospective recalibration.

The maintained code currently provides mmBERT artifact loading and scoring plus a pure advisory routing function.
The following integration work remains:

1. Add a maintained orchestrator that sends text through `mmbert-lora-s42`, invokes the advisory route, and calls DeepSeek only for the middle zone.
2. Add an OpenRouter adapter with the frozen subversion-only prompt, CoreWeave fp8 routing, reasoning disabled, strict integer JSON, and both decision-token log probabilities.
3. Compute and retain raw decision-token log odds, convert them with the stable sigmoid helper, and pass `p_subversion` to the advisory route.
4. Add bounded concurrency, timeouts, retry handling, and operational metrics keyed by versioned model, prompt, provider, and threshold hashes; exhausted failures must restrict.
5. Map `restrict` to privilege reduction or deterministic reference-monitor policy rather than treating a learned score as authorization.
6. Keep harmful non-injection classification separate from the subversion prompt and route.
7. Shadow the complete pipeline on representative traffic and recalibrate before deployment.

The disposable OpenRouter experiment runner is reproducibility code, not the production adapter.
Do not import it into the maintained package.

Before another encoder-tuning experiment, collect realistic matched transaction attacks and benign tasks, paired multilingual transformations on both labels, independently sourced known-span long-document attacks, and prospective traffic-like negatives.
Do not repeat context-length, document-bag, or BIPIA augmentation ablations on the same labels.
Do not sweep reinforcement-learning, focal-loss, or source-weighting objectives over the current open dev roles as a substitute for new evidence.
Do not use the Rogue Security benchmark as independent evidence because more than half of its rows exactly overlap current canonical public sources.

## P2 - prospective evaluation

- Freeze a new final test before using its results for model or threshold choice.
- Add known-span long-document attacks with matched clean controls.
- Add realistic application traffic only when independently labelled; model
  agreement alone is not ground truth.
- Evaluate distribution shift by source, time, language, attack family, and
  provenance channel.

Exit condition: evidence supports a narrowly stated shadow-deployment claim.
Blocking requires a separate product and risk review.

## P3 - stateful agent containment

- Integrate the reference monitor with a stateful environment such as AgentDojo.
- Propagate provenance/taint through retrieval, summaries, tool outputs, memory,
  and egress sinks.
- Broker short-lived credentials outside the planner.
- Bind human approval to the exact irreversible action and arguments.
- Quarantine durable memory writes and prevent untrusted content from creating
  capabilities.
- Measure task success, attack success, unauthorized side effects, and secret
  exfiltration, not only text-classifier accuracy.

The core security thesis remains: prediction reduces exposure; deterministic
authorization bounds impact when prediction fails.
