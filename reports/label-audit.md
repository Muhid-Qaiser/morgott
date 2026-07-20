# Initial label audit

Method: a Codex agent qualitatively inspected the highest-confidence false
positives and false negatives from each evaluation family at the then-retained
0.1% direct diagnostic and zero-observed-FP indirect point. This was not
independent human annotation or adjudication, and no visible test label was
changed after inspection. The later precision-first shadow-review recommendation
changes counts but does not turn this inspection into ground truth.

## Findings

- At the 0.1% diagnostic, the direct model alerts on 18/4,630 ToxicChat
  source-labeled negatives. The
  highest-scoring examples are long role assignments, instruction-heavy
  adventure prompts, forced response prefixes, and requests for hidden
  instructions. Several are plausible source-label errors or genuinely
  ambiguous, not ordinary small talk.
- At that diagnostic, the direct model misses 46/60 deepset positives. The
  highest missed scores include explicit overrides just below the
  threshold; lower-scoring multilingual and context-dependent fragments expose
  both calibration and label-context problems.
- At that diagnostic, OASST1 produces 0/1,582 held-out alerts. Its highest scores
  are role-play,
  virtual-assistant command formats, and conversational corrections. Treating
  every accepted OASST1 prompt as a true negative is still a weak-label
  assumption.
- The indirect model's two clean-context false positives are Stack Overflow
  answers containing imperative code and URL instructions. This is a realistic
  ambiguity for a text-only external-content detector.
- The indirect model misses 41/125 standalone BIPIA payloads and 123/375
  poisoned contexts. Misses span output-format changes, unrelated questions,
  and harmful code insertion; provenance helps choose the sensor but does not
  make intent identifiable from text alone.
- At the stringent 0.1% diagnostic, the cipher aggregate hides a severe subgroup
  gap: the two plain word-mapping variants reach 0.0% and 0.2% recall, while
  encoded variants range from 52.8% to 69.3%. At the recommended 85%-precision
  profile, both plain variants fall to 0% and encoded variants reach only
  19.3%–39.8%. The high-precision preference exposes this recall failure rather
  than fixing it.
- At the stringent 0.1% diagnostic, the direct model alerts on 91/18,195
  JailbreaksOverTime source-labeled negatives. All 91 come from the WildChat
  subset. The highest-scoring audit
  rows contain DAN personas, explicit jailbreak framing, instruction overrides,
  or requests to evade safeguards, so the source annotation is visibly noisy
  for this task. Keep the source label in metrics, but do not interpret its
  0.50% source-label FPR as the false-positive rate on ordinary production chat.
- JailbreaksOverTime mixes source, collection period, attack family, and label
  process. Its 84.88% source-labeled attack recall at the 0.1% diagnostic is
  useful distribution-shift evidence, not a causal claim about robustness
  improving or degrading over time.
- Tensor Trust contributes 908 unique attack-only texts and 1,346 attack-plus-
  defense contexts. At the 0.1% diagnostic the direct baseline detects 33.48%
  attack-only and the two shadow sensors together detect 67.09% of contexts; at
  the precision-first review point those figures are 28.85% and 62.48%. The higher context
  score is not automatically better intent recognition: benchmark defenses
  themselves contain phrases about secrets, instructions, and attacks, and no
  matched defense-only controls are published in these two suites. Treat
  attack-only as the cleaner transfer check and the context view as a provenance
  stress test.

## Decision

Keep source labels for reproducibility, expose weak-label assumptions and exact
denominators, and never tune on inspected test errors. Keep a label schema that
separates harmful intent, direct jailbreak, direct prompt injection, indirect
injection, uncertainty, and toxicity even though no human labelers are
available. Cross-family LLM agreement may supply conservative weak training
labels, but disagreements must be discarded and model-labelled rows must stay
out of threshold calibration, locked tests, and production-FPR claims. Without
independent product labels, the learned detector remains shadow-only.
