# Historical model experiment ledger

No model is promoted or approved for blocking.

## Completed current-corpus experiments

Character 3-5 gram and word 1-2 gram linear classifiers were trained on the grouped routing views at the untouched 0.5 cutoff.
No threshold was selected against an FPR, precision, or recall target.
The scores are not calibrated to application traffic.

| Task | Model | Dev recall | Dev FPR | Dev PR-AUC | Natural unseen-source recall | Natural unseen-source FPR |
|---|---|---:|---:|---:|---:|---:|
| Direct source-supported | Character balanced | 88.81% | 11.00% | 96.15% | 70.86% | 37.78% |
| Direct source-supported | Character unweighted | 91.26% | 2.43% | 98.79% | 70.41% | 24.31% |
| Direct source-supported | Word balanced | 86.06% | 9.12% | 96.29% | 63.54% | 27.08% |
| Direct source-supported | Word unweighted | 92.35% | 2.94% | 98.60% | 77.33% | 18.39% |
| Direct with generated weak-benign labels | Character balanced | 89.40% | 7.14% | 97.47% | 69.58% | 22.52% |
| Direct with generated weak-benign labels | Character unweighted | 85.72% | 1.36% | 98.43% | 56.16% | 15.19% |
| Direct with generated weak-benign labels | Word balanced | 90.04% | 6.66% | 97.37% | 73.35% | 21.39% |
| Direct with generated weak-benign labels | Word unweighted | 86.27% | 1.80% | 98.06% | 58.91% | 17.01% |
| Untrusted-content injection | Character balanced | 98.57% | 37.57% | 99.26% | 94.90% | 88.61% |
| Untrusted-content injection | Character unweighted | 98.91% | 37.50% | 99.43% | 96.31% | 89.29% |
| Untrusted-content injection | Word balanced | 96.86% | 28.15% | 99.14% | 82.53% | 63.90% |
| Untrusted-content injection | Word unweighted | 95.89% | 19.32% | 99.54% | 73.45% | 40.91% |
| Aegis harmfulness | Character balanced | 80.67% | 23.45% | 88.64% | n/a | n/a |
| Aegis harmfulness | Word balanced | 80.77% | 24.25% | 88.76% | n/a | n/a |

The high aggregate PR-AUC values do not establish robust separation.
Many sources are nearly single-class, so source style is an easy label shortcut.
The natural unseen-source slice is useful evidence but is not a proper leave-one-source-out retraining experiment.

Critical failures at the same default cutoff:

- Direct-model false-positive rates were 74% to 82% on JBB benign, 48% to 72% on NotInject, and 13% to 49% on FalseReject.
- The direct models recalled only 0% to 8% of XSTest positives.
- FinanceBench direct-question false-positive rates were 12% to 42%, despite very low rates on the much larger TAT-QA question set.
- Generated weak-benign data reduced aggregate false positives but also reduced recall in the matched unweighted comparison.
- On held-out BrowseSafe documents, untrusted-model PR-AUC remained about 0.52.
- On BIPIA, untrusted-model FPR remained 30% to 34% while recall ranged from 72% to 82%.
- Source balancing was a failed default for the direct source-supported task at cutoff 0.5.

Decision: promote no linear model.

## Fixed ModernBERT direct-user comparison

One `answerdotai/ModernBERT-base` candidate was fully fine-tuned for one epoch on the same capped examples as the unweighted direct linear controls.
The recipe used FP32 master weights, BF16 autocast, a 256-token limit, gradient checkpointing, and the untouched 0.5 cutoff.
Dev-test remained unread until the model artifact was frozen, then one sealed evaluation was performed.

| Split | Recall | FPR | PR-AUC | Group-weighted recall | Group-weighted FPR |
|---|---:|---:|---:|---:|---:|
| Validation | 98.03% | 0.42% | 99.96% | 97.58% | 0.79% |
| Dev-test | 96.55% | 1.52% | 99.61% | 92.03% | 2.90% |
| Naturally unseen dev sources | 83.47% | 29.43% | 97.46% | n/a | n/a |

Critical dev-test failures at the same default cutoff were 81.0% FPR on JBB benign, 69.6% on NotInject, 40.8% on FalseReject, 20.0% on FinanceBench, and only 4.0% recall on XSTest positives.
The large gap between row-weighted and unseen-source results is direct evidence of source and task shortcuts.
The apparent aggregate PR-AUC is not representative of production traffic.

Training took about 56 minutes and peaked at about 2.9 GiB allocated VRAM on the local RTX 4050 Laptop GPU.
The 256-token input cap truncated 5.42% of selected training rows, 1.76% of validation, and 3.65% of dev-test.
Only one seed was run, and CUDA is not guaranteed to be bitwise deterministic.

A pinned Hugging Face FlashAttention-2 v3 kernel passed padded forward and backward parity.
It improved the synthetic median training step by only 1.2% at length 256, by 6.9% at length 512, and by 15.8% at length 1024 with checkpointing.
The data, not the kernel, blocked a valid longer-context comparison.

Decision: preserve the model and results as research artifacts, but do not use them for blocking, production-FPR claims, or authority decisions.

## Shortcut and length audit

Thirteen of fourteen selected direct-user training sources contained only one routing class.
Raw source identity explained 87.75% of direct-user label entropy and 91.34% of full routing-train label entropy.
After the row-uniform per-source-label cap, source identity still explained 79.94% of direct-user label entropy.

A source-majority lookup reached 95.94% validation accuracy with 100% recall and 9.40% FPR without reading prompt text.
A character n-gram source classifier reached 87.78% top-1 accuracy and 77.81% macro source recall.
Mapping its predicted source to the source-majority class yielded 92.77% recall and 5.25% FPR.

All 8,355 selected training rows above 256 tokens and all 2,323 validation rows above 256 tokens were non-benign.
The frozen ModernBERT also alerted on 99.88% of WildJailbreak adversarial-benign rows and 43.94% of accepted English OASST1 direct-user rows at cutoff 0.5.
Those auxiliary labels are not formal broad-benign ground truth, but they rule out interpreting the aggregate result as robust prompt understanding.

## Source-heldout capacity and data ablations

The corrected quick comparison assigned a validation row to a source fold only when every retained origin belonged to that fold.
It excluded 316 exact-merged rows whose origins crossed folds, leaving 131,695 source-heldout validation rows.
Training and validation origins were asserted disjoint.

The group-uniform word n-gram control reached 70.78% pooled recall, 8.79% pooled FPR, 91.35% precision, and 0.9023 fold-macro PR-AUC.
Adding 1,157 leakage-screened OR-Bench hard-benign rows reduced recall to 65.67% for a small FPR change to 8.47%.
Matched-context splices raised word-model recall to 72.64% with 8.91% pooled FPR, but macro-source FPR increased from 6.71% to 10.45%.

The following three-epoch results held out WildJailbreak as a mixed-label source:

| ModernBERT mode | Trainable parameters | Recall | FPR | PR-AUC | Decision |
|---|---:|---:|---:|---:|---|
| True linear probe | 1,538 | 75.07% | 30.55% | 0.9291 | Signal exists, but default-threshold FPR is unacceptable |
| Pretrained nonlinear head probe | 592,130 | 76.20% | 31.33% | 0.9313 | No material gain over the linear probe |
| CLS + mean + max multipool candidate | 890,498 | 76.44% | 24.59% | 0.9429 | Best frozen candidate on this fold, but still poor |
| Top-four-layer partial tune | 20,653,058 | 76.56% | 35.14% | 0.9238 | Worse transfer than frozen candidates |
| Ordinary full tune at 2e-5 | 149,606,402 | 84.36% | 51.22% | 0.9296 | Severe shortcut overfit |
| Full tune with discriminative rates | 149,606,402 | 78.37% | 40.77% | 0.9232 | Lower rates did not fix overfit |

The ordinary full-tune training loss fell to 0.0036 while held-out FPR exceeded 50%.

The multipool candidate was then run across all four source folds:

| Progress | Pooled recall | Pooled FPR | Precision | Fold-macro PR-AUC | Mean macro-source FPR |
|---:|---:|---:|---:|---:|---:|
| 0.5 epoch | 88.56% | 14.07% | 89.19% | 0.8838 | 20.50% |
| 1.0 epoch | 84.53% | 11.55% | 90.56% | 0.8864 | 10.73% |
| 2.0 epochs | 86.08% | 9.55% | 92.20% | 0.9013 | 13.79% |
| 3.0 epochs | 85.05% | 11.36% | 90.75% | 0.8994 | 17.32% |

At the observed two-epoch point, fold-macro PR-AUC and ROC-AUC were 0.9013 and 0.9190, versus 0.9023 and 0.9039 for the word control.
At score cutoff 0.90, ModernBERT observed 70.38% recall and 1.46% FPR, while the word control observed 40.67% recall and 0.44% FPR.
These are uncalibrated source-heldout validation observations.

At three epochs, multipool FPR was 24.59% on WildJailbreak, 24.75% on the tail-format fold, and 10.11% on the task-dialogue fold.
Within those folds it alerted on 24.50% of MASSIVE, 27.18% of Mind2Web, 5.44% of Schema-Guided Dialogue, and 11.96% of Taskmaster.

Replacing 1,000 positive and 1,000 benign parent rows with paired splices changed the three-epoch WildJailbreak result from 76.44% recall and 24.59% FPR to 74.11% recall and 19.24% FPR.
PR-AUC changed only from 0.9429 to 0.9449, and the held-out paired diagnostic contained only five pairs.

Decision: promote no neural model or synthetic-data recipe.
Keep only the minimal word n-gram implementation as an active cheap control.
Detailed threshold, prevalence, cascade, and error analysis remains in `reports/threshold-audit.md` and `reports/false-positive-audit.md`.
The completed runners are intentionally not part of the active package.
Their existing raw generated reports remain local and ignored, while this versioned ledger preserves their metrics and decisions.

## Earlier corpus experiments

The results below used the earlier twelve-source injection corpus.
They are retained only to avoid repeating failed ideas.

| Candidate | Useful observation | Decision |
|---|---|---|
| Character 3-5 gram TF-IDF and logistic regression | Cheap and strong on some obfuscated text, but incomplete and bypassable | Keep only as a smoke control |
| Frozen multilingual E5 and linear head | Lower transfer and near-zero multi-turn recall | Rejected |
| Frozen ModernBERT mean and CLS probes | Both missed all multi-turn rows at the precision-first point | Rejected |
| PIGuard and ProtectAI checkpoints | Public-family overlap, hard-negative cost, or threshold saturation | Not independent promotion candidates |
| SiberianCat and Wolf checkpoints | Published thresholds over-defended; tighter thresholds collapsed recall | Not promoted |
| One-epoch ModernBERT-base fine-tune | Underfit on the old validation split | Inconclusive |
| One-epoch DeBERTa-v3-base fine-tune | Better aggregate recall but almost no multi-turn transfer | Not promoted |
| One-shot OpenRouter reviewers | Slow, unreliable, provider-dependent, and not a security boundary | Removed |
| WildChat weak benign negatives | Reduced multi-turn recall substantially | Stopped |

## Durable warnings

- Aggregate results are dominated by source and task style.
- Public evaluation families may overlap checkpoint training data.
- Positive-only sources measure transfer recall, not false-positive cost.
- A detector score never grants authority.
- A prospective traffic-like final test is required before any production claim.
