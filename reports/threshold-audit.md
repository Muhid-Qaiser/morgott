# Source-heldout threshold and AUC audit

Generated 2026-07-23.

## Scope and integrity

This audit uses 131,695 source-heldout validation rows from four independently trained folds.
It excludes 316 exact-merged rows whose origins cross folds.
No dev-test rows, labels, or scores were read.

The word model is the group-uniform, unweighted word 1-2 gram linear control trained for three epochs.
The neural model is the frozen ModernBERT-base CLS + mean + max multipool candidate trained on the same baseline recipe for one three-epoch schedule.
The neural score files retain four checkpoints at 0.5, 1, 2, and 3 epochs.

All four compressed ModernBERT score files were verified against their report SHA-256 values and expected row counts.
All four reports bind the same manifest, fold assignment, evaluator, and trainer source hashes.
The final dialogue fold was rerun in an isolated copy of that exact source set after an unrelated new package file correctly tripped the broad source-change guard.
Every checkpoint exactly reproduced the rejected run.

The word scores were recomputed from manifest SHA-256 `fbb4f5e90a731d0131e0ccd1f34e4bb33e783d82b3d3892057207e8c8912d2ca`.
The recomputation exactly matched the prior fixed-cutoff counts.

## ModernBERT checkpoint observations

All confusion counts below pool only fixed numeric cutoff decisions.
PR-AUC and ROC-AUC are computed inside each independently trained fold and then macro-averaged.
AUC is not computed by pooling scores from different fold models.

| Progress | Recall | FPR | Precision | Fold-macro PR-AUC | Fold-macro ROC-AUC | Mean macro-source FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 epoch | 88.56% | 14.07% | 89.19% | 0.8838 | 0.9139 | 20.50% |
| 1.0 epoch | 84.53% | 11.55% | 90.56% | 0.8864 | 0.9195 | 10.73% |
| 2.0 epochs | 86.08% | 9.55% | 92.20% | 0.9013 | 0.9190 | 13.79% |
| 3.0 epochs | 85.05% | 11.36% | 90.75% | 0.8994 | 0.9206 | 17.32% |

The two-epoch row is the strongest pooled validation observation, not a selected production checkpoint.
Its intermediate state shares the three-epoch learning-rate schedule and is not an independent two-epoch recipe.

## Ranking metrics by fold

The ModernBERT columns use the two-epoch observation.

| Held-out fold | Rows | Word PR-AUC | Word ROC-AUC | ModernBERT PR-AUC | ModernBERT ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Mixed WildJailbreak | 19,793 | 0.9043 | 0.7802 | 0.9406 | 0.8350 |
| Task dialogue vs competition | 80,787 | 0.9634 | 0.9679 | 0.9765 | 0.9806 |
| Short finance agent | 29,446 | 0.9924 | 0.9467 | 0.9940 | 0.9554 |
| Tail formats | 1,669 | 0.7491 | 0.9208 | 0.6942 | 0.9049 |
| Fold macro | n/a | 0.9023 | 0.9039 | 0.9013 | 0.9190 |
| Row-size-weighted fold mean | 131,695 | 0.9583 | 0.9343 | 0.9714 | 0.9521 |

ModernBERT does not beat the word control on fold-macro PR-AUC.
It improves ROC-AUC and recall at comparable fixed cutoffs, but performs worse on the small tail-format fold.
The row-size-weighted means are dominated by the 80,787-row task-dialogue fold and are not promotion metrics.

## Fixed threshold sweep

These thresholds were not selected against a target.
They show the observed validation tradeoff only.
The score is not calibrated confidence.

### Word 1-2 gram control

| Score cutoff | Recall | FPR | Precision | False positives | False signals per 10k benign |
|---:|---:|---:|---:|---:|---:|
| 0.500 | 70.78% | 8.7878% | 91.35% | 5,008 | 878.8 |
| 0.700 | 58.05% | 2.9708% | 96.24% | 1,693 | 297.1 |
| 0.800 | 50.58% | 1.4494% | 97.86% | 826 | 144.9 |
| 0.900 | 40.67% | 0.4352% | 99.19% | 248 | 43.5 |
| 0.950 | 32.77% | 0.1421% | 99.67% | 81 | 14.2 |
| 0.975 | 26.16% | 0.0456% | 99.87% | 26 | 4.6 |
| 0.990 | 18.77% | 0.0140% | 99.94% | 8 | 1.4 |

### ModernBERT two-epoch observation

| Score cutoff | Recall | FPR | Precision | False positives | False signals per 10k benign |
|---:|---:|---:|---:|---:|---:|
| 0.500 | 86.08% | 9.5476% | 92.20% | 5,441 | 954.8 |
| 0.700 | 80.69% | 5.0590% | 95.44% | 2,883 | 505.9 |
| 0.800 | 76.84% | 3.2182% | 96.90% | 1,834 | 321.8 |
| 0.900 | 70.38% | 1.4582% | 98.44% | 831 | 145.8 |
| 0.950 | 63.93% | 0.6563% | 99.22% | 374 | 65.6 |
| 0.975 | 57.56% | 0.2825% | 99.63% | 161 | 28.3 |
| 0.990 | 48.83% | 0.0912% | 99.86% | 52 | 9.1 |

Raising the cutoff sharply reduces false positives for both models.
The cost is substantial recall loss.
At every shown cutoff, ModernBERT retains more recall and produces more false positives than the word control.

## Why the reported precision is misleading

The validation pool is 56.73% positive.
Production attack prevalence will probably be far lower, so validation precision cannot be transferred to traffic.

For illustration only, if the measured recall and FPR transferred unchanged:

| Candidate | Observed validation precision | Expected precision at 0.1% prevalence | At 1% | At 5% |
|---|---:|---:|---:|---:|
| Word at 0.90 | 99.19% | 8.56% | 48.56% | 83.11% |
| Word at 0.95 | 99.67% | 18.75% | 69.96% | 92.39% |
| ModernBERT at 0.90 | 98.44% | 4.61% | 32.77% | 71.75% |
| ModernBERT at 0.95 | 99.22% | 8.88% | 49.60% | 83.68% |
| ModernBERT at 0.99 | 99.86% | 34.88% | 84.39% | 96.57% |

These are prevalence substitutions, not production estimates.
The source-heldout rates themselves may not transfer.

## Fixed sample cascade

The previously proposed fixed cascade was evaluated without choosing cutoffs to optimize this result:

- Word score below 0.50: return `no_security_signal` and skip the neural model.
- Word score at least 0.50 but below 0.90: run ModernBERT and recommend review when its two-epoch score is at least 0.90.
- Word score at least 0.90: immediately recommend review.
- Any known-malicious exact or Bloom-filter hit may only add review.
- A failed or unavailable middle-zone check fails closed to review.

The primary word zones contained 73,811 low rows, 27,250 middle rows, and 30,634 high rows.
The fixed cascade observed 60.38% recall, 0.8072% FPR, and 460 false positives.
It improved recall over word at 0.90 by 19.70 percentage points, but increased FPR by 0.3720 points.
Its FPR still reached 3.99% on MASSIVE, 0.97% on Mind2Web, and 0.96% on Taskmaster.
This is a validation tradeoff, not a promoted pipeline.

No runnable cascade is retained because no model is promoted.
A representative future integration is:

```python
score = word_score(text)
review_required = (
    malicious_bloom.maybe_contains(text_hash(text))
    or score >= 0.90
    or (score >= 0.50 and secondary_check(text))
)
```

A Bloom filter should contain known-malicious exact signatures, not benign allow entries.
It has no false negatives only for keys that were actually inserted.
Paraphrases and new attacks are unseen keys, and positive matches can still be false positives.
Therefore a Bloom positive may recommend review, but a Bloom negative may never grant authority or declare text benign.

Every returned detector decision remains `allow`.
Every side effect still passes through the deterministic reference monitor.

## Decision

Promote no model and select no threshold.
Keep the word model as the cheap control.
Keep the frozen multipool ModernBERT only as a research candidate.
Do not make a deployment or production-FPR claim until a prospective traffic-like evaluation is frozen and finalists are repeated across seeds.
