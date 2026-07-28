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

The repository owner authorized one bounded exception on 2026-07-28 after the external tail failure was measured.
It covers the completed full-data frozen mmBERT head and one update-matched rank-8 mmBERT LoRA engineering gate.
It does not authorize a LoRA sweep, a full-mixture LoRA claim, production calibration, or blocking.
The maintained trainer can execute a full-mixture LoRA run if one is later authorized, but code availability is not evidence or approval.

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

The full-data frozen mmBERT recipe fits leakage-filtered canonical rows, PromptShield train, and the retained generated pairs with balanced BCE plus pair ranking.
Three frozen-head seeds are retained to show dispersion.
One rank-8 LoRA seed is retained only as an update-matched engineering gate against a frozen control; it omits most canonical rows and all generated pairs.
PromptShield validation selects checkpoints, while canonical calibration alone selects reported applied thresholds.
PromptShield test is source-disjoint within its own release and remains already-consumed benchmark development data.
Known cross-corpus overlap prevents a source-OOD claim relative to the complete Morgott plus PromptShield fit.
SEP remains already-consumed transfer development data.
All four downstream checkpoints are advisory and must be compared independently rather than combined into an unevaluated ensemble.
Every score is truncated to the first 512 normalized tokens, so long web, retrieval, and tool content remains an explicit evidence gap.

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
