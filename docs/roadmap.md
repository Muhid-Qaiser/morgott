# Jailbreak-defense roadmap

The product thesis is defense in depth: a detector can lower the probability
that an attack reaches the planner, but only a trusted reference monitor can
bound the impact after a detector miss. False positives are also a security and
product failure, so no learned score blocks users in the initial system.

## P0 — reproducible baseline (complete)

- Twelve pinned, ungated public sources; exact deduplication and source/group-
  aware splits where metadata permits it.
- Separate direct-user and untrusted-content character sensors. Untrusted input
  runs both its provenance-specific sensor and the direct-override fallback.
- Direct-user thresholds selected only on grouped development data at 80%, 85%,
  90%, and 95% minimum precision. The 85% floor is the current high-precision
  shadow-review knee; 0.1%, 0.5%, 1%, 2%, and 5% FPR budgets remain diagnostics.
  External hard-negative, obfuscation, indirect-injection, human-attack, and
  source-shift evaluations remain frozen.
- A fail-closed reference monitor mediates every simulated side effect with
  exact schemas, a static caller-supplied capability allowlist, constrained
  arguments, and sensitive-data checks. Task/user binding, expiry, credential
  issuance, and provenance-aware authorization are not implemented yet.
- Classifier decisions are shadow-only. The reference monitor commits 0/8
  unauthorized actions and 2/2 benign actions; an input-filter-only ablation
  commits all eight unauthorized actions.

The baseline is intentionally cheap. Frozen E5, frozen ModernBERT (mean and
CLS), PIGuard and ProtectAI DeBERTa checkpoints, two current ModernBERT/mmBERT
injection checkpoints, and two one-shot remote reviewers were measured at the
stringent 0.1% diagnostic and did not earn promotion there. Plausible future
candidates must be compared over the full declared precision/FPR grid; the
old result does not establish dominance at every review tradeoff.

## P1 — production-shaped evidence

1. Define four independent annotation axes: harmful intent, direct jailbreak,
   direct prompt injection, and indirect prompt injection. Add provenance and
   attack-span fields rather than deriving them from text prefixes.
2. Because no human labelers are available, do not make human adjudication an
   operational dependency. Keep public source labels and model-produced labels
   separate, preserve provenance, and leave the detector in shadow mode.
3. Run a deterministic 5,000-sampled-row WildChat weak-label pilot stratified by
   language, length, toxicity, topic, and security-trigger terms. Two
   independent model families must agree with high confidence before a row can
   become a weak benign training example. Discard disagreement and uncertainty;
   send every detector-hard candidate plus a deterministic 10% audit sample to
   a third model family. Keep toxicity as an independent field. First compare
   zero weak rows with all accepted negatives from the 5k-row sample. Only if a
   predeclared development gate improves may the sampled pool grow until 5k,
   20k, and 50k accepted-negative ablations are possible. Stop when recall does
   not improve or any normal-chat subgroup worsens.
4. Add positive long-document tests with known attack spans at 5%, 50%, and 95%
   positions. Keep source, generator family, goal, document, and mutation seed
   grouped across splits.
5. Report leave-one-source-out results and expected precision at realistic
   attack prevalences, not just balanced accuracy or AUROC.

The WildChat labels are weak training labels only. Judge agreement and a third
model audit measure consistency, not correctness; neither may be reported as a
production FPR. Public suites used to select whether an ablation scales are
frozen repeated development comparisons, not untouched final tests. Without
independently labelled product traffic, the exit gate for blocking is
intentionally unavailable. The useful P1 outcome is a more robust,
reproducible shadow sensor with no regression on those public controls.

The pilot is now complete. It accepted 2,430 weak negatives, but the weighted
ablation decreased both direct and indirect macro recall at the 85% precision
profile, including a drop from 908/4,136 to 291/4,136 on multi-turn attacks.
The same gate failed at all four declared precision profiles, so collection
stops at 5,000 sampled rows and the weak rows are not added to the baseline.
See [`reports/wildchat-ablation.md`](../reports/wildchat-ablation.md).

## P2 — evidence-gated encoder model (initial screen complete)

The next trainable candidate is a trust-conditioned segment classifier:

```text
trusted runtime provenance
        ├── direct user ────────┐
        └── untrusted content ──┼─> shared encoder -> channel head -> shadow score
                               └─> 512-token windows / max document score
```

The first two runs were completed as one atomic, one-seed/one-epoch screen:

1. ModernBERT-base fit in 2,955 MiB but underfit: 6/66 TP and 1/7,120 FP at the
   85% profile, with no multi-turn detection.
2. DeBERTa-v3-base used 3,547 MiB and reached 36/66 TP with 6/7,120 FP, but added
   four hard-negative alerts and caught only 5/4,136 multi-turn attacks. Its 95%
   profile is the only continuation candidate; neither model is promoted.
3. Rerun DeBERTa with three seeds and a fuller predeclared schedule before any
   architectural embellishment. Check score calibration/saturation and preserve
   the full precision/FPR grid.
4. Add known-span positive long documents plus matched clean documents before
   comparing 512-token windows with whole-text 2K inference. Train positive
   windows from known payload spans—never label every window positive.
5. Add two tiny channel-specific heads and independently calibrated thresholds.
   Provenance routing must come from immutable runtime metadata.
6. Try OOD-energy/tokenizer perturbation, scalar mixing, or LoRA only if the
   fuller control and error analysis predict a specific fix.

Promotion requires a better precision-first tradeoff than the character model
across the same 80%, 85%, 90%, and 95% validation-precision profiles and the
diagnostic FPR grid, no regression on harmful-but-non-injection or normal-chat
strata, paired group-bootstrap confidence intervals, batch-1/p95 latency, and
document-level FPR stratified by window count.

Do not spend on ModernBERT Decoder, Longformer, a learned attention head over
frozen embeddings, a model ensemble, or an LLM/ReAct loop without new evidence.
ModernBERT Decoder is causal-generation machinery, Longformer is superseded for
this test by native long-context encoders, and the measured remote reviewer adds
seconds of latency plus provider/unavailability risk.

## P3 — impact containment

- Integrate the reference monitor with AgentDojo or an equivalent stateful agent
  environment and real tool schemas.
- Propagate provenance/taint through summaries, tool outputs, memory, and RAG;
  enforce egress policy at every sink.
- Broker short-lived credentials outside the planner. Add exact-action human
  approval for irreversible or high-impact operations.
- Quarantine durable memory writes and prevent tool output or stored content
  from granting new authority.
- Evaluate adaptive attacks against the full agent, not only text classifiers:
  task success, attack success, utility, unauthorized side effects, and secret
  exfiltration are the release metrics.
- After the static end-to-end harness is reproducible, add PIArena-style
  defense-adaptive attacks and cross-benchmark evaluation. Track browser/image
  prompt injection separately; a text-only win does not establish multimodal
  web-agent safety.

The highest-value architectural novelty is this split between provenance-aware
prediction and deterministic authority—not a larger binary text classifier.
