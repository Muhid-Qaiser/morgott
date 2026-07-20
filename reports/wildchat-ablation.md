# WildChat weak-negative pilot

## Decision

**Stop at the 5,000-row sampled pilot. Do not scale to 20,000 or 50,000
accepted negatives.** The accepted weak negatives reduced some alerts, but they
did not improve the predeclared precision-first attack tradeoff. This is a
useful negative result: broad model-agreed chat negatives are not a free fix for
over-defense when the positive training set is small and narrow.

The decision is deterministic development evidence, not a statistical claim or
a production release gate. Full machine-readable results are in
[`wildchat-ablation-results.json`](wildchat-ablation-results.json).

## Data and weak-label run

- Sampled 5,000 user turns from three exact-revision WildChat-1M Parquet shards,
  after local redaction, exact/near deduplication, and overlap removal against
  all fit and evaluation rows.
- Covered 11 language buckets, four length buckets, eight coarse topics, and
  security-trigger/no-trigger strata. The pinned public sample did not expose a
  useful source-toxicity field, so toxicity was not fabricated.
- Made 10,266 one-shot, ZDR-required OpenRouter calls: 5,000 each to Mistral
  Small 2603 and Qwen 3.5 27B, plus 266 blind audits by Claude Sonnet 4.6.
  Calls used strict JSON, temperature zero, no tools, no fallback, no retries,
  no ReAct, and denied provider data collection.
- Accepted 2,430 rows as weak benign negatives: 2,173 on two-family unanimous
  high-confidence agreement and 257 after third-family high-confidence
  agreement. Nine third-audited rows were rejected. Agreement is not accuracy.
- Provider-reported aggregate cost was 2.595 credits; p50/p95 latency was
  971/2,893 ms. Unavailable or invalid primary outcomes were retained only as
  aggregate diagnostics and never retried into agreement.
- Raw text and normalized judgment journals remain ignored local artifacts.
  Versioned reports contain hashes, provenance, counts, and aggregates only.

## Controlled ablation

Both candidates use the same base fit rows, character 3--5 gram vocabulary,
grouped validation, public development suites, and threshold-selection rules.
Weak rows enter classifier fitting only; they never enter vocabulary fitting,
threshold calibration, or evaluation. Their total weight is capped at 10% of
the base negative mass.

At the selected minimum-85%-validation-precision profile:

| Set | No weak rows | +2,430 weak rows | Change |
|---|---:|---:|---:|
| Grouped validation | 34 TP / 4 FP | 34 TP / 4 FP | unchanged |
| ToxicChat | 44/73 TP; 18/4,630 FP | 44/73 TP; 17/4,630 FP | -1 FP |
| deepset injection positives | 12/60 | 14/60 | +2 TP |
| Multi-turn positives | 908/4,136 | 291/4,136 | **-617 TP** |
| JailbreaksOverTime positives/negatives | 3,203/3,901; 81/18,195 | 3,203/3,901; 41/18,195 | same TP; -40 FP |
| Tensor Trust isolated attacks | 262/908 | 282/908 | +20 TP |
| BIPIA payload/context | 84/125; 252/375 | 84/125; 252/375 | unchanged |
| Tensor Trust embedded attacks | 841/1,346 | 825/1,346 | -16 TP |
| External hard negatives | 0/4,208 alerts | 0/4,208 alerts | unchanged |
| NotInject | 0/339 alerts | 0/339 alerts | unchanged |
| Position-stress normal chat | 0/500 alerts | 0/500 alerts | unchanged |

Across the direct attack suites, macro recall decreased by 1.88 percentage
points; across the indirect attack suites it decreased by 0.40 points. The
candidate also failed the same improvement rule at the 80%, 90%, and 95%
precision profiles. The non-accepted WildChat remainder is not benign ground
truth, so its 20/2,570 candidate alerts are reported only as an alert rate, not
an FPR.

## Interpretation

The weak negatives made the classifier more conservative around language that
also appears in obfuscated and contextual attacks. The large multi-turn recall
loss dominates the modest false-alert reductions. More rows under the same
labeling and weighting recipe would amplify an unhelpful direction rather than
answer the actual representation and positive-coverage gaps.

Keep the 2,430 rows and provenance as an archived experimental asset, but do not
include them in the baseline artifact. Future broad-chat work should wait for a
stronger encoder and better grouped positive coverage, then be reevaluated as a
fresh experiment. These model-only labels must never be used to claim a
production false-positive rate.
