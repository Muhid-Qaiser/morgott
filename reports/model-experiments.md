# Historical model experiment ledger

These results used the earlier twelve-source injection corpus. They are retained
only to avoid repeating failed ideas. No model below was trained on the new
canonical routing corpus, no model is promoted, and none is an authorization
boundary.

The old direct validation partition contained 66 positives and roughly 7,120
negatives. That number was a grouped validation denominator—not a corpus limit,
training cap, or context limit.

## What was tried

| Candidate | Useful observation | Decision |
|---|---|---|
| Character 3–5 gram TF-IDF + logistic regression | Cheap and surprisingly strong on obfuscated text, but incomplete and bypassable | Keep only as an optional smoke/control baseline |
| Frozen multilingual E5 + linear head | Lower transfer and near-zero multi-turn recall | Do not use frozen embeddings as the next primary path |
| Frozen ModernBERT mean/CLS probes | Both missed all multi-turn rows at the precision-first point | Frozen features rejected; this did not test full fine-tuning |
| PIGuard and ProtectAI checkpoints | Public training-family overlap, high hard-negative cost, or threshold saturation | Not independent promotion candidates |
| SiberianCat and Wolf checkpoints | Card thresholds over-defended; locally tightened thresholds collapsed recall | Not promoted |
| One-epoch ModernBERT-base fine-tune | Underfit: 6/66 validation positives at the old 85% precision profile | No conclusion about a proper data-rich run |
| One-epoch DeBERTa-v3-base fine-tune | 36/66 validation positives with 6/7,120 false alerts, but only 5/4,136 multi-turn attacks and 4/4,208 hard-negative alerts | Interesting control, not a finalized architecture |
| One-shot OpenRouter reviewers | Seconds of latency, unavailable outcomes, provider/privacy dependencies, no security boundary | Removed from active code; do not put in the request path |
| WildChat weak benign negatives | 2,430 accepted rows reduced multi-turn recall from 908/4,136 to 291/4,136 in the declared ablation | Stop; rows remain outside the corpus |

## Transfer warnings

- At the old precision-first threshold, the character control detected 0/676
  exact-unique Nemotron agentic injections. The source is positive-only, so this
  was a transfer-recall failure rather than an FPR measurement.
- On PromptShield, the same control detected only 194/6,486 source positives and
  alerted on 240/17,030 source negatives at the old locked profile. PromptShield
  lacks row-level family/group lineage and overlaps active public families, so
  it remains unsuitable for training.
- Tensor Trust attack-in-defense contexts scored far higher than standalone
  attacks for several models. Because those contexts contain defensive
  instruction/security language and lack matched clean controls, the increase
  may be a shortcut rather than improved attack understanding.
- JailbreaksOverTime source-negative rows contain obvious jailbreak language.
  Its source-label false-positive rate is not ordinary-chat product friction.

## Why the old results do not choose the new model

The routing corpus changes both scale and target. It adds large human and
synthetic attack families, harmful non-injection rows, explicit uncertainty,
long HTML, and independent tags. Old aggregate results are therefore not a fair
ranking for the new task. Off-the-shelf checkpoints also have unknown or known
training overlap with public evaluations.

## Next controlled comparison

Build a fresh, minimal trainer rather than reviving the deleted runners:

1. Fit a character/word linear routing baseline.
2. Fine-tune one encoder on the identical grouped rows.
3. Use explicit source-balanced weights or sampling as an ablation, not a corpus
   cap.
4. Train a binary routing head and mask subtype losses when labels are unknown.
5. Select thresholds on validation and report per-source plus
   leave-one-source-out results on dev-test.
6. Freeze a prospective final test before making any generalization or blocking
   claim.

Do not add an ensemble, remote judge, custom loss, LoRA, long-context mechanism,
or extra head until measured errors predict what it should fix. The reference
monitor remains the enforcement boundary regardless of classifier choice.
