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

## Masked multitask ModernBERT and Boundary Pairs

A July 2026 quick experiment used the pinned `answerdotai/ModernBERT-base` encoder with frozen weights and four masked output logits for direct instruction subversion, indirect instruction subversion, jailbreak, and harmful intent.
Toxicity was deferred because the corpus lacks two independent positive source families with matched negatives.
The head pooled masked CLS, mean, and maximum representations into a 384-unit GELU projection.
Direct inputs used 256 tokens, while untrusted content used 512-token windows with 128-token overlap.
Unknown and conflicting axis labels were loss-masked.
Every inference result remained advisory with `decision: allow`.

The quick selector used seed 42, a group-uniform cap per source and security-label stratum, validation-selected checkpoints through six epochs, and the WildJailbreak, short finance-agent, and BrowseSafe-heldout folds.
The exact run used data manifest SHA-256 `27bdd9c244fbf479d699cb7c8d826385c0bd0f2f39e5154051db10e927c58f81` and runner SHA-256 `a84cdfdb1e68017dfcb3b4cebd8241720cfc42e32406495bc1b5ab99219d0a44`.
Raw text-free results and checkpoints remain ignored under `artifacts/routing_encoder_runs/`, and the compact selector report remains ignored at `reports/modernbert-multitask-quick.json`.

| Recipe | Mean source-macro PR-AUC | Boundary both-correct | Decision |
|---|---:|---:|---|
| Core masked BCE | 0.7846 | 9.17% | Rejected |
| Boundary pairs plus masked BCE | 0.8245 | 68.33% | Best PR-AUC, but rejected |
| Boundary pairs plus BCE and ranking | 0.8192 | 90.00% | Pair gate passed, but rejected |
| Word 1-2 gram control for winning rows | 0.5683 | n/a | Retained control only |
| Source-only control for winning rows | 0.3880 | n/a | Shortcut diagnostic |
| Length-only control for winning rows | 0.5790 | n/a | Shortcut diagnostic |

Pair ranking passed its predeclared aggregate rule, but Boundary BCE retained the higher source-macro PR-AUC.
On WildJailbreak, Boundary BCE selected epoch 5 and reached 0.9683 direct-subversion and 0.9802 jailbreak source-macro PR-AUC.
Its validation-thresholded combined-route FPR was still 3.95%, compared with 2.35% for core BCE.

The short finance-agent fold was the decisive failure.
Boundary BCE escalated 59.80% of legitimate held-out finance rows at validation-selected 1% per-head profiles, with 100% FPR on Harper Valley Bank, 65.60% on TAT-QA, and 38.00% on BANKING77.
The word control observed 7.93% overall finance FPR and 11.20% worst-source FPR on the same selected rows.
Pair ranking improved Boundary pair separation but still produced 58.77% finance FPR and 100% worst-source FPR.
Masked per-head PR-AUC was undefined for this fold because the legitimate held-out rows did not carry explicit matching-axis negatives, which is why the broad combined-route denominator is required.

BrowseSafe transfer was approximately random.
Core BCE reached 0.5296 indirect PR-AUC and 0.5500 ROC-AUC, while Boundary BCE reached 0.5251 and 0.5470.
The validation-thresholded combined route escalated every one of the 682 benign held-out BrowseSafe documents for all three neural recipes.
Therefore every nonempty benign document-window-count slice also had 100% FPR.
The runner excluded 339 long positive training documents without known payload spans instead of assigning their clean windows positive labels.

The saved historical binary ModernBERT had a different manifest hash.
On Boundary validation it achieved 0% both-correct and 7.5% ordering, so it remains a historical control and was not modified.

The pinned FA2 binary passed the 512-token parity gate.
At 256 tokens, finite outputs, predictions, and the `0.125` BF16 logit tolerance passed, but classifier-head gradient cosine was `0.9248`, below the predeclared `0.995` threshold.
The quick suite used FA2 only under the user's explicit override and records `runtime_gate_passed=false` plus `runtime_gate_override=true`.
This is not represented as a parity pass.

Decision: stop encoder work before full folds, top-layer unfreezing, dev-test selection, or a consolidated research demo.
Return to matched finance attack and benign collection, known-span long-document data, and broader same-source contrasts.
No model is promoted.

## Frozen public guard comparison

Four pinned public checkpoints were scored locally on the exact ModernBERT quick-suite validation roles under the same 256-token direct and 512-token overlapping document policy.
Wolf was also rerun on the identical rows using its published 2,048-token training context for direct inputs and overlapping documents.
The comparison did not open dev-test.
The versioned summary below preserves the decision evidence.
Detailed text-free results and the completed evaluator remain in the ignored local research archive.

| Checkpoint | WildJailbreak direct PR-AUC | Finance direct PR-AUC | BrowseSafe indirect PR-AUC | Boundary both-correct | Decision |
|---|---:|---:|---:|---:|---|
| Llama Prompt Guard 2 86M | 0.8422 | 0.8754 | 0.7838 | 35.00% | Keep as the general precision-first shadow |
| Wolf Defender, common windows | 0.6265 | 0.8442 | 0.6943 | 50.00% | Context policy superseded |
| Wolf Defender, native 2K | 0.6270 | 0.8453 | **0.7963** | 50.00% | Keep only as a transaction-specific shadow candidate |
| ProtectAI DeBERTa v2 | 0.9545 | 0.8836 | 0.6924 | 25.00% | Rejected |
| Sentinel v2 | 0.5173 | 0.9131 | 0.5175 | 12.50% | Rejected |

Prompt Guard materially improved long-document ranking and avoided the worst finance false positives under a conservative global threshold, but its finance recall fell to 18.4% and its boundary rate remained below both boundary-trained ModernBERT recipes.
Native 2K raised Wolf's BrowseSafe PR-AUC from 0.6943 to 0.7963 and halved default BrowseSafe FPR from 67.16% to 34.16%.
It did not materially change finance or WildJailbreak ranking, and its all-source default FPR remained 18.06% despite known training overlap with four Morgott source families.
The five static transaction-demo presets separated cleanly at 0.5, so Wolf remains useful only for a prospective transaction-specific shadow trial.
The three selected ModernBERT direct-instruction heads also separated all five demo prompts.
On 10,818 direct held-out rows, the ModernBERT direct-plus-jailbreak route reached 61.78% recall at 0.92% FPR under its 0.1% validation profile and 81.35% recall at 4.34% FPR under its 0.5% profile.
Wolf thresholds of 0.8 and 0.9 remained near 24% and 23% FPR, and simple ModernBERT-Wolf OR or AND rules did not improve the measured frontier.
ProtectAI produced 99.12% BrowseSafe FPR after global validation calibration.
Sentinel was approximately random on BrowseSafe, weak on pairs, and the slowest checkpoint.
None replaces Morgott's explicit masked axes or deterministic reference monitor, and no checkpoint is promoted for blocking.

## Consolidated ModernBERT repair

After the public-checkpoint comparison, one bounded experiment consolidated the supported training and validation roles into a single frozen `answerdotai/ModernBERT-base` checkpoint.
The shared head still uses masked CLS, mean, and maximum pooling followed by a 384-unit GELU projection and four sigmoid outputs.
The loss mask is independent per output, but the normalization and projection are shared, so training data for one axis can shift the other output scores.
Direct instruction subversion, indirect instruction subversion, and jailbreak select the transaction-security route.
Harmful intent remains a trained and thresholded content-safety metadata head, but it cannot select the transaction route or the training epoch.
Every result remains advisory and returns `decision: allow`.

The repair copied techniques rather than Wolf weights or its binary ontology.
It added 420 training-only matched clean and attacked long-context pairs using 140 known indirect payloads, three distinct clean BrowseSafe documents per payload, and prefix, middle, and suffix placement.
Tokenizer offsets label only windows that overlap a known payload span.
Long positive documents without a known span remain excluded from fitting rather than assigning every clean window a positive label.
Exact normalized-text collisions with training or locked evaluation are rejected.

The consolidated selector used the full capped development train and validation roles, seed 42, six frozen-head epochs, a maximum of 2,000 rows per source and security-label stratum, and no pair-ranking loss.
It selected epoch 6 without using harmful intent in checkpoint selection.
The exact run used the same data-manifest hash as the public-checkpoint comparison and did not open dev-test.
The raw-text-free result, score cache, configuration, and head are under `artifacts/routing_encoder_runs/consolidated_known_span/seed_42/`.

The default route uses separately calibrated development thresholds.
Direct and short untrusted inputs use the nominal 0.1% profile.
Untrusted inputs longer than 512 tokens use the indirect head's nominal 1% long-document profile.

| Validation scope | FPR | Recall | PR-AUC |
|---|---:|---:|---:|
| All transaction-security rows | 0.245% | 82.25% | 0.9551 |
| Direct-user rows | 0.203% | 85.68% | 0.9905 |
| Finance rows | 0.138% | 64.61% | 0.9823 |
| Untrusted-content rows | 0.534% | 74.58% | 0.9764 |
| BrowseSafe documents | 1.026% | 5.04% | 0.6287 |
| All untrusted rows over 512 tokens at the 1% profile | 1.070% | 26.45% | 0.7996 |
| BrowseSafe rows over 512 tokens at the 1% profile | 0.919% | 3.79% | 0.6125 |

The repair raised BrowseSafe PR-AUC from 0.5711 to 0.6287 and default contextual recall from 0.59% to 5.04%.
It retained 87.5% Boundary both-correct and 100% Boundary ordering.
Those improvements are real but insufficient.
The predeclared shadow gate failed because BrowseSafe PR-AUC remained below 0.65 and BrowseSafe recall remained below 25%.
The apparently acceptable aggregate long-document recall is dominated by easier LLMail rows and must not hide the BrowseSafe failure.

The calibrated 0.1% thresholds are `0.9942697287` for direct instruction subversion, `0.9086179733` for short indirect instruction subversion, `0.2538610101` for jailbreak, and `0.9830850959` for harmful-intent metadata.
The separately calibrated 1% threshold for indirect inputs over 512 tokens is `0.7461389899`.
These are development-selected values, not portable production-FPR guarantees.

The exact checkpoint was also reevaluated on the same validation rows using 2,048-token untrusted windows with 512-token overlap.
This was an inference-context audit, not a new 2,048-token training run.
For untrusted rows over 512 tokens, PR-AUC fell from 0.7996 to 0.7774 while recall remained approximately flat at 26.10%.
For long BrowseSafe rows, PR-AUC fell from 0.6125 to 0.5521 and recall at the separately calibrated 1% profile fell from 3.79% to 1.42%.
Increasing context therefore does not repair this checkpoint.

A separate training run then used 2,048-token untrusted windows with 512-token overlap from the beginning.
It changed no model, data cap, seed, epoch selector, or validation rule.
The larger context retained 184 real BrowseSafe positive training documents instead of 87, but still excluded 1,816 whose full documents required multiple windows and lacked payload spans.
It selected epoch 6 and failed the BrowseSafe ranking, recall, and worst-source FPR gates.

| Controlled recipe | BrowseSafe PR-AUC | BrowseSafe FPR / recall | Long BrowseSafe PR-AUC | Long BrowseSafe FPR / recall | Direct FPR / recall | Finance FPR / recall | Boundary both-correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512/128 known-span repair | 0.6287 | 1.026% / 5.04% | 0.6125 | 0.919% / 3.79% | 0.203% / 85.68% | 0.138% / 64.61% | 87.5% |
| 2,048/512 trained context | 0.5636 | 0.880% / 1.04% | 0.5459 | 0.766% / 0.95% | 0.890% / 90.25% | 0.414% / 75.78% | 90.0% |
| 512/128 plus BrowseSafe bags | **0.7674** | **0.880% / 18.07%** | **0.7727** | 0.919% / **18.77%** | **0.174% / 84.85%** | **0.110% / 62.13%** | **95.0%** |

The fully trained 2,048-token model therefore confirmed the inference-only diagnostic instead of overturning it.
Its best epoch-level BrowseSafe PR-AUC was only 0.5710, so checkpoint selection did not hide a good long-context model.
At a raw score cutoff of 0.5 it reached 7.11% BrowseSafe recall at 3.67% FPR.
Longer context alone is rejected.

A second one-variable ablation returned to 512/128 windows and used the remaining 1,913 BrowseSafe positive training documents as multiple-instance bags.
Every bag contributed one positive indirect-subversion loss over the maximum window logit.
The individual windows were loss-masked because the source does not disclose the payload span, so ordinary HTML windows were never labelled positive.
The run trained on all 2,000 BrowseSafe positives and 2,000 clean BrowseSafe documents, selected epoch 6, and left validation and dev-test untouched.
Its artifacts are under `artifacts/routing_encoder_runs/consolidated_known_span_browsesafe_bags/seed_42/`.

Document-bag supervision repaired ranking rather than merely moving a threshold.
BrowseSafe PR-AUC rose by 0.1387 and long BrowseSafe PR-AUC rose by 0.1602 over the known-span repair.
The 9-plus-window BrowseSafe slice rose from 0.6181 to 0.7690 PR-AUC, although it still supplied six false positives and only 112 of 595 detected attacks at the contextual default route.
The predeclared gate passed every check except BrowseSafe recall of at least 25%.
That gate set did not constrain positive-only long-document source recall.
At the same 1% long-document profile, LLMail recall fell from 89.64% to 75.23%, and the six-positive BIPIA slice fell from five detections to zero.
Those slices are too narrow for portable threshold claims, but the regression is evidence of source specialization.

The bag objective substantially changed score scale.
The raw 0.5 cutoff produced 86.07% BrowseSafe FPR and 97.78% recall, so it is unusable.
The calibrated long threshold `0.9732407331` produced 0.92% FPR and 18.77% recall on BrowseSafe documents over 512 tokens.
The complete contextual route produced 0.88% FPR and 18.07% recall across all BrowseSafe documents.
A BrowseSafe-specific threshold of `0.9591543078` was required to cross the 25% recall gate, reaching 25.63% recall at 3.08% FPR.
A lower threshold of `0.9518632293` reached 28.0% recall at 4.84% FPR.
Both thresholds were selected on the same development source and conflict with the precision-first goal.
Neither threshold is adopted.

The new ModernBERT scores were aligned row-for-row with cached Prompt Guard scores on all 1,357 BrowseSafe validation documents.
At the existing transferred strict thresholds, ModernBERT flagged 122 attacks and six benign documents, while Prompt Guard flagged 248 attacks and five benign documents.
Their OR flagged 330 attacks and 11 benign documents, giving 48.89% recall at 1.61% FPR.
Their AND flagged 40 attacks and no benign documents in this development set, giving only 5.93% recall.
Across all 1,516 long untrusted rows, ModernBERT reached 33.18% recall at 0.92% FPR and Prompt Guard reached 34.22% recall at 0.76% FPR.
Their OR reached 59.16% recall at 1.68% FPR, while their AND reached only 8.24% recall with no observed false positives.
At 1% attack prevalence, the measured OR rates imply a 2.09% review rate and 23.44% precision among reviewed documents.
At 0.1% attack prevalence, they imply a 1.66% review rate and only 2.94% precision.
These are development projections, not production guarantees.

Interim decision before the later BIPIA and dev-test comparison: retain the BrowseSafe-bag checkpoint as the strongest internal validation shadow, using calibrated thresholds and never the raw 0.5 cutoff.
Use Prompt Guard alone as the precision-first default for ordinary long untrusted content because its transferred 0.73% FPR and 36.74% recall remain materially better.
For high-impact agent contexts, run both models and send their OR to deep review if the measured review budget is acceptable.
Treat agreement as a high-confidence escalation signal, not as an allow or deny decision.
Keep native-2K Wolf as a transaction-specific disagreement shadow only.
Do not use ModernBERT, Prompt Guard, or Wolf as an authorization boundary or blocking approval.

## BIPIA matched-context ablation

One controlled data-repair experiment reused only BIPIA payloads and clean contexts whose lineage was eligible for training.
It selected 109 payloads and 164 distinct code, email, and table contexts, then inserted every payload into three different contexts at prefix, middle, and suffix positions.
Tokenizer offsets retained the exact payload span, validation and dev-test context lineages were excluded, and exact collisions with locked evaluation hashes failed the recipe.
The canonical corpus and `data/manifest.json` did not change.

The first pilot retained all 327 attacked variants but deduplicated clean contexts against existing training rows.
That accidentally added 327 positive rows and only 86 new clean rows.
Because all four output logits share the same normalization and 384-unit projection, the positive-heavy indirect objective also shifted direct and finance scores.
The repaired recipe instead contributed one clean training copy for every attacked row, for 327 clean and 327 attacked copies grouped as matched pairs.
This is explicit model-recipe weighting rather than duplication in the canonical corpus.

Both treatments used the same BrowseSafe bag control, seed 42, frozen encoder, six-epoch selector, validation rows, context policy, and per-head masking.
The contextual route used each checkpoint's validation-selected nominal 0.1% standard profile and nominal 1% long-untrusted profile.

| Validation scope | BrowseSafe-bag control FPR / recall / PR-AUC | Positive-heavy pilot FPR / recall / PR-AUC | Pair-balanced repair FPR / recall / PR-AUC |
|---|---:|---:|---:|
| All transaction-security rows | 0.203% / 81.60% / 0.9858 | 0.769% / 84.07% / 0.9783 | 0.228% / 83.35% / 0.9873 |
| Direct-user rows | 0.174% / 84.85% / 0.9917 | 0.793% / 87.86% / 0.9834 | 0.203% / 86.69% / 0.9919 |
| Untrusted-content rows | 0.400% / 74.32% / 0.9815 | 0.600% / 75.57% / 0.9812 | 0.400% / 75.86% / 0.9817 |
| BIPIA | 0.000% / 28.57% / 0.9915 | 0.000% / 47.62% / 0.9936 | 0.000% / 41.27% / 0.9972 |
| BrowseSafe | 0.880% / 18.07% / 0.7674 | 1.320% / 19.41% / 0.7563 | 0.880% / 19.70% / 0.7652 |
| Finance | 0.110% / 62.13% / 0.9871 | 0.525% / 70.08% / 0.9739 | 0.083% / 66.59% / 0.9882 |

Pair balancing removed the large direct and finance regression while retaining most of the BIPIA recall gain.
It selected epoch 6 and improved BIPIA ranking, not merely its score cutoff.
Its standard thresholds were `0.9893474579` for direct instruction subversion, `0.9793540239` for short indirect instruction subversion, and `0.3259495199` for jailbreak.
Its separately calibrated long-indirect threshold was `0.9674102664`.
Harmful intent retained threshold `0.9752018452` as metadata and did not select the transaction route.
The pair-balanced model still failed the predeclared shadow gate because BrowseSafe recall remained below 25%.

The validation result justified one frozen three-way dev-test comparison.
It encoded the same 43,137 selected rows once and applied each checkpoint's validation-selected thresholds without retuning.
This opens that dev-test role permanently for repeated development; it is not a pristine final test.
The unified comparison is under `artifacts/bipia_matched_ablation/dev_test_comparison_all.json`.

| Dev-test scope | BrowseSafe-bag control FPR / recall / PR-AUC | Pair-balanced repair FPR / recall / PR-AUC |
|---|---:|---:|
| All transaction-security rows | 10.993% / 82.33% / 0.8606 | 11.023% / 83.42% / 0.8601 |
| Direct-user rows | 12.868% / 92.87% / 0.8732 | 12.971% / 93.61% / 0.8693 |
| Untrusted-content rows | 1.467% / 63.61% / 0.9606 | **1.130% / 65.31% / 0.9628** |
| BIPIA | 1.681% / 29.46% / 0.9641 | **0.840% / 41.28% / 0.9786** |
| BrowseSafe | 0.916% / 17.82% / 0.7718 | 1.078% / 19.63% / 0.7720 |
| LLMail | 18.125% / 90.35% / 0.9949 | **10.000% / 91.30% / 0.9967** |
| Finance | 0.327% / 78.25% / 0.9875 | 0.348% / 80.31% / 0.9877 |

The pair-balanced repair therefore generalizes as a narrow indirect-route improvement.
It reduces untrusted-content false positives while increasing recall, and its BIPIA gain survives on 499 positive and 119 negative dev-test rows.
The single observed pair-balanced BIPIA false positive is not evidence of a portable 0.84% rate because that negative denominator is small.
BrowseSafe ranking is essentially unchanged, so BIPIA augmentation does not repair the long-HTML weakness.
Boundary both-correct changed from 87.5% to 86.25%, while ordering remained 100%.

The full-route aggregate remains unacceptable.
Both models flag every one of the 2,000 multi-turn benign rows, and their multi-turn PR-AUC remains below random.
The pair-balanced model also flags 28.61% of NotInject and 3.64% of FinanceBench.
After removing the multi-turn source only as a diagnostic, control FPR is still 1.045% and pair-balanced FPR is 1.079%.
This is distribution-shift evidence, not a threshold inconvenience.

Decision: reject the positive-heavy pilot.
Retain the pair-balanced checkpoint only as the better indirect-content research shadow, not as a global replacement or blocking model.
For a later architecture run, give the indirect axis its own projection branch or run a separate tiny indirect head over the same frozen encoder features so BIPIA updates cannot perturb direct outputs.
Use trusted runtime provenance to choose the direct or indirect branch.
Prompt Guard remains the complementary precision-first sensor for long untrusted content, while the pair-balanced ModernBERT head supplies BIPIA-like signal that the strict public checkpoints missed.

The combined injection-route evaluator previously omitted `harmful_non_injection` rows even though those rows are negatives for instruction subversion.
The evaluator now includes both `benign` and `harmful_non_injection` as route negatives while continuing to exclude unknown labels.
Recomputing the retained pair-balanced validation scores raises its contextual FPR from the previously reported 0.228% over 11,834 benign rows to 0.252% over 13,866 route-negative rows.
It flags 8 of 2,032 harmful-but-non-injection validation rows.
This correction does not explain the much larger out-of-source Do-Not-Answer and HarmBench failures.

A one-variable BrowseSafe experiment then applied document-level maximum-window BCE to both attacked and clean long documents so training aggregation exactly matched inference aggregation.
It used the same pair-balanced BIPIA data, frozen English ModernBERT backbone, seed, window policy, validation rows, and six-epoch selector.
The symmetric recipe formed 3,857 BrowseSafe bags and reduced optimizer updates from 118,991 to 48,649, but it selected epoch 5 and damaged the useful ranking.

| Validation scope | Positive-only bag control | Symmetric positive and negative bags |
|---|---:|---:|
| BrowseSafe PR-AUC | 0.7652 | 0.7023 |
| BrowseSafe FPR / recall | 0.880% / 19.70% | 1.026% / 3.11% |
| Long BrowseSafe PR-AUC | 0.7704 | 0.7195 |
| Long BrowseSafe FPR / recall | 0.919% / 20.19% | 0.919% / 3.15% |
| Finance FPR / recall | 0.083% / 66.59% | 0.110% / 58.04% |
| BIPIA FPR / recall | 0.000% / 41.27% | 0.000% / 50.79% |
| Boundary both-correct / ordering | 87.5% / 100% | 62.5% / 95.0% |

The symmetric maximum loss is rejected.
For a clean document, only its current highest-scoring window receives gradient, which overcorrected shared HTML features and suppressed true attacks without reducing the calibrated long-document FPR.
Focal loss, class weighting, or another threshold cannot repair that ranking loss.
The artifact is retained only under `artifacts/failure_repair_ablation/routing_encoder_runs/consolidated_known_span_browsesafe_bags_symmetric_bipia_pair_balanced/seed_42/`.

Licensing was not used as an inclusion filter.
BIPIA WebQA and Summarization were not mixed into this ablation because they require regenerating contexts from NewsQA and XSum and would change the source-task mix at the same time as the pairing recipe.
The [official BIPIA acquisition instructions](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/benchmark/README.md) make that a separate corpus-expansion experiment.
That expansion is lower priority than repairing the demonstrated multi-turn and NotInject failures.

## Direct-user failure verification and repair

The first diagnostic corrected the combined-route evaluator to include `harmful_non_injection` as a negative for instruction subversion.
The retained pair-balanced checkpoint then produced 6.081% FPR and 92.24% recall on the direct dev suite after excluding the multi-turn source.
On the group-held-out multi-turn diagnostic it produced 100% FPR, 100% recall, 0.6379 PR-AUC, and 0.4491 ROC-AUC.
The poor ranking proves that this was not a threshold-only failure.
Every multi-turn input was already below the 256-token direct limit, so neither 2,048-token training nor a rolling window could repair it.
The benchmark's Base64, Caesar, and Leetspeak labels describe requested output transformations while the harmful goal remains readable in the input, so a decoder is not the missing component.

The multi-turn source was split by complete goal hash into 572 training goals, 77 validation goals, and 168 held-out goals.
This produced 3,760 training rows, 616 validation rows, and 1,344 held-out rows.
The templates and techniques still repeat across roles, so this is an already-open research diagnostic rather than an independent benchmark.

| Frozen-head treatment | Direct suite FPR / recall | Multi-turn FPR / recall | Multi-turn PR-AUC / ROC-AUC | Boundary both-correct |
|---|---:|---:|---:|---:|
| Shared projection, no multi-turn repair | 5.920% / 90.88% | 100.000% / 100.00% | 0.6421 / 0.4556 | 70.0% |
| Shared projection, multi-turn repair | **4.895% / 88.67%** | 0.446% / **29.69%** | **0.9321 / 0.8752** | 65.0% |
| Independent projection per head, no multi-turn repair | 6.081% / 90.70% | 100.000% / 100.00% | 0.6470 / 0.4666 | 85.0% |
| Independent projection per head, multi-turn repair | 5.106% / 89.22% | **0.223% / 27.23%** | 0.9288 / 0.8701 | 65.0% |

The style-matched data changed multi-turn ranking from below random to useful.
Four fully independent projection towers did not improve the result enough to justify approximately four times the head parameters.
This rejects architecture expansion as the first repair and supports separate direct and indirect branches only where their data and input provenance actually differ.

A single route score can still improve the operating tradeoff.
For the shared repaired head, averaging the direct and jailbreak logits reduced the non-multi-turn suite to 2.267% FPR at 80.19% recall and produced 1.339% FPR at 37.95% recall on multi-turn.
That remained worse than the precision target, so route fusion alone was not accepted as the repair.

The clean repair then added non-adversarial WildGuardMix train prompts as direct-subversion and jailbreak negatives.
The recipe used 2,000 source-benign and 2,000 source-harmful prompts for training and 500 of each for validation.
It excluded 7,682 exact hashes from locked evaluation and 969 exact hashes already present in the base training role.
The source-model harmfulness label is weak supervision, but both selected classes provide valid counterexamples to the false rule that harmful subject matter implies instruction subversion.
No Do-Not-Answer, HarmBench, or NotInject row entered this clean-transfer fit.

A separate upper-bound diagnostic split those three already-open failure sources by lineage into 4,857 train, 677 validation, and 1,474 held-out rows.
Forty-one exact hashes crossed roles, so 82 complete lineage groups were removed before fitting.
The upper bound reached zero observed FPR on its held-out failure rows while retaining 0.117% FPR and 73.01% recall on unaffected English sources.
This establishes model capacity but is not a promotable result because it fits the benchmark families under diagnosis.

Three pinned frozen backbones were compared using the identical clean WildGuard transfer selection and seed 42.
All rows used the same 256-token direct-input policy and validation-selected nominal 0.1% per-head thresholds.

| Frozen backbone | Full direct FPR / recall | Precision | PR-AUC / ROC-AUC | Failure holdout FPR | Multi-turn FPR / recall | Boundary both-correct |
|---|---:|---:|---:|---:|---:|---:|
| English ModernBERT-base | 0.221% / **75.68%** | 99.38% | 0.9867 / 0.9932 | 0.475% | **0.000% / 23.10%** | 65.0% |
| mmBERT-small | 0.281% / 70.98% | 99.16% | 0.9859 / 0.9927 | 0.339% | 0.000% / 0.67% | 45.0% |
| mmBERT-base | **0.106% / 74.73%** | **99.70%** | **0.9916 / 0.9948** | **0.136%** | 0.893% / 18.86% | **75.0%** |

mmBERT-small is rejected because it loses recall, boundary separation, and multi-turn transfer without reducing aggregate FPR.
mmBERT-base is the strongest single precision-first direct backbone in this experiment.
English ModernBERT-base remains complementary because it retains more multi-turn, Jailbreaks Over Time, and WildJailbreak recall.

| Source at the strict per-head route | English ModernBERT-base | mmBERT-base |
|---|---:|---:|
| Do-Not-Answer FPR | 0.30% | **0.05%** |
| HarmBench FPR | 3.75% | **0.50%** |
| NotInject FPR | 1.77% | **1.18%** |
| FinanceBench FPR | 0.00% | 0.00% |
| WildGuardMix FPR | 0.00% | 0.00% |
| HackAPrompt recall | 80.90% | **86.20%** |
| Jailbreaks Over Time recall | **70.15%** | 54.55% |
| WildJailbreak recall | **95.00%** | 91.45% |

The failure repair is primarily a data result, not a context-length, loss-weighting, or architecture result.
It cuts the old direct-suite FPR by more than an order of magnitude while preserving useful attack ranking.
The remaining weaknesses are low strict recall on multi-turn attacks, elevated NotInject and ToxicChat false positives, and positive-source transfer that varies materially by attack family.

## Cross-backbone direct-route fusion

The English ModernBERT-base and mmBERT-base clean-transfer heads were scored row-for-row on one shared 20,114-row grouped validation set and five already-open evaluation suites.
Each candidate used one route score and one threshold for all direct inputs.
Member route probability is the maximum of that member's direct-subversion and jailbreak probabilities.
The selected ensemble score is the arithmetic mean of the two member route probabilities.

| Validation profile and rule | Full direct FPR / recall | Precision | Multi-turn FPR / recall | Boundary FPR / recall |
|---|---:|---:|---:|---:|
| 0.1%, English member | 0.146% / 66.86% | 99.54% | 0.000% / 0.45% | 0.0% / 27.5% |
| 0.1%, mmBERT-base member | **0.075% / 66.01%** | **99.76%** | 0.000% / 2.46% | 0.0% / 17.5% |
| 0.1%, mean route ensemble | 0.101% / **77.16%** | 99.72% | 0.000% / 4.13% | 0.0% / **30.0%** |
| 0.5%, English member | 1.246% / 91.41% | 97.16% | **0.670% / 35.60%** | 2.5% / 65.0% |
| 0.5%, mmBERT-base member | 0.804% / 92.02% | 98.16% | 3.125% / 42.41% | 2.5% / 57.5% |
| 0.5%, mean route ensemble | **0.794% / 92.92%** | **98.20%** | 4.911% / **52.01%** | **0.0% / 62.5%** |

The strict ensemble gains 11.16 recall points over the stronger individual route at only 0.025 percentage points more held-out FPR than mmBERT-base.
The 0.5% ensemble slightly improves both full-suite FPR and recall over mmBERT-base and raises multi-turn recall by 9.60 points, but its multi-turn benign FPR also rises by 1.79 points.
The gain is not supplied by one oversized attack source.
At the strict profile the ensemble recalls 88.10% of HackAPrompt, 67.45% of Jailbreaks Over Time, 85.79% of Tensor Trust, and 88.85% of WildJailbreak.

The strict ensemble's remaining source failures include 1.48% NotInject FPR, 2.66% ToxicChat-negative FPR, 0.25% HarmBench FPR, 33.90% deepset prompt-injections recall, and 4.13% multi-turn recall.
Finance sources and WildJailbreak benign rows produced zero observed false positives at this operating point, but zero observations are not guarantees.

The exact strict ensemble threshold is `0.9699193633`.
The exact 0.5% ensemble threshold is `0.7654141188`.
These values apply only to the arithmetic mean of the two recorded route probabilities and the exact member revisions and head digests in `artifacts/direct_failure_repair_ensemble/ensemble-audit.json`.
They are development-selected thresholds and must be recalibrated against representative traffic before any deployment claim.

Decision: retain the strict mean-route ensemble as the strongest current direct-user research shadow for the user's precision-first goal.
Use the 0.5% point only for high-impact actions whose review budget tolerates the measured increase in false positives.
Keep the pair-balanced English ModernBERT checkpoint as the separate indirect-content branch because the mmBERT direct experiment did not evaluate BrowseSafe or BIPIA as an indirect replacement.
Trusted runtime provenance chooses the direct or indirect branch, so an attacker-controlled prompt never selects its own detector or threshold.

## Optimization strategy decision

Reinforcement learning is not the next training method.
This detector has static supervised labels and no sequential environment, delayed reward, or trustworthy preference signal.
An RL objective would optimize a proxy over the same biased development data and add variance without supplying the matched counterexamples that repaired the observed failures.

Focal loss and broad class weighting are also rejected as first repairs.
The failures were source, topic, language, and template shortcuts rather than a shortage of aggregate positive weight.
The symmetric BrowseSafe loss already demonstrated that pushing harder on the wrong objective can reduce both ranking and recall.

PCGrad is worth testing only if one shared trainable encoder is required and measured direct-versus-indirect gradients conflict.
The fully independent frozen-head towers did not beat the shared projection, while separate provenance-selected branches remove the interference without a new optimizer.

LP-FT is the only encoder-tuning strategy worth a later bounded test.
The current frozen head supplies the linear-probe initialization, after which only the top layers should be unfrozen with a much smaller learning rate, strong early stopping, and source-heldout selection.
Earlier Morgott top-layer and full fine-tunes overfit source style and produced worse held-out FPR, so LP-FT must wait for stronger matched data and a prospective evaluation role.

Group DRO is also deferred.
It can optimize worst-group loss only after the groups represent real deployment shifts and it requires strong regularization or early stopping.
Using current source IDs as groups would risk optimizing benchmark identity rather than robust prompt semantics.

Longer 2,048-token training and rolling windows remain appropriate only for genuinely long untrusted documents.
The controlled 2,048-token run was worse on BrowseSafe, and the direct failures were already shorter than 256 tokens.
For indirect content, known payload spans, matched clean documents, symmetric position variation, and document-level evaluation remain more important than raw context length.

The next evidence-producing work is not another loss-function sweep on the same open dev set.
It is a prospective source-heldout suite with realistic transaction attacks and matched benign tasks, paired multilingual transformations on both labels, known-span long-document attacks with clean controls, and multiple training seeds.
Further recipe selection on the current dev sources would increase selection multiplicity without establishing production FPR.

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

## External validation of the retained ensemble (2026-07-27)

The retained English ModernBERT plus mmBERT-base direct-route ensemble was
scored on the public PromptShield test split (`hendzh/PromptShield`, 23,516
rows), the first benign distribution this project did not choose. Contamination
measured at 323 rows (1.374%), all label 0.

- Ensemble ROC AUC 0.624 and **0.00% TPR at 1% FPR**; last against every
  baseline in that paper's Table 4, below ProtectAI v2.
- Not truncation: chunked untrusted-channel scoring reaches AUC 0.661 and
  leaves TPR at 0.00%. Not head, member or fusion selection: every head and both
  members cap at 0.52%.
- Mechanism: the highest-scoring benign rows outrank every attack. Maximum
  attack score 0.9942 against a benign 99th percentile of 0.9975. The rows
  responsible are ordinary application prompts carrying instructional language.
- The retained 0.101% FPR is a statement about short text: 98.5% of its
  negatives are 64 tokens or fewer, and above that the measured FPR is
  1.1%-6.7% over 303 rows.
- Multi-turn recall at the same operating point is 1.79%; multi-turn rows were
  filtered out of the suite the headline was computed on.

## Attempt scaling (2026-07-27)

Seven gradient-free surface mutation families, threshold calibrated at 0.1% FPR
on the suite's own negatives, 300 caught attacks, 10,500 mutations.

- ASR@1 49.20%, ASR@4 84.48%, ASR@32 98.30%.
- Clean recall 81.14% corresponds to **1.38% effective recall** against a
  32-attempt adversary.
- Per-family evasion: homoglyph 72.6%, zero-width 65.9%, case 61.6%, encoding
  wrap 60.2%, padding 53.0%, whitespace 24.7%, duplication 6.9%.

Any future recipe should report ASR@k beside clean recall; the two move in
opposite directions.

## Strict normalisation and generated matched pairs (2026-07-27)

Run through the archived `run_direct_failure_repair` with only preprocessing and
training data varied, so the baseline reproduces by construction (10,896
optimizer updates, matching the original exactly). Three seeds per condition
where stated; `wildguard_weak_transfer` recipe only.

| condition | n | dev recall | PromptShield AUC | TPR@1% | SEP AUC |
|---|---|---|---|---|---|
| raw | 3 | 58.33% | 0.6226 | 0.00% | 0.7275 |
| strict | 3 | 65.48% | 0.6391 | 0.01% | 0.7134 |
| strict+pairs 25% | 2 | 68.53% | 0.7008 | 0.03% | 0.7324 |
| strict+pairs 50% | 2 | 54.94% | 0.7056 | 0.03% | 0.7244 |
| strict+pairs 100% | 3 | 33.33% | 0.7386 | 0.01% | 0.7336 |
| strict+pairs+ranking 0.25 | 2 | 47.72% | 0.7584 | 0.00% | 0.7506 |
| strict+mmBERT | 2 | 78.14% | 0.6957 | 0.00% | 0.7563 |
| strict+pairs+mmBERT | 2 | 61.60% | **0.7814** | 0.02% | 0.7846 |

- Strict normalisation (NFKC plus invisible stripping, homoglyph folding,
  combining-mark removal, repeat capping) adds 7.15 points of dev recall with
  near-disjoint seed ranges, and drives case, whitespace, homoglyph and
  zero-width evasion to **exactly 0.00%** because those variants normalise to
  the same string. `encoding_wrap` worsens to 80.93% and `padding` stays at
  58.07%; both add real text and no normaliser reaches them.
- The mixing ratio is monotone in external AUC (0.639, 0.701, 0.706, 0.739),
  which distinguishes shortcut removal from label noise in favour of the
  former. At 25% mixing dev recall also improves.
- `aligned_pair_ranking_loss` at 0.25, given genuine matched pairs for the first
  time, raises external AUC to 0.7584 while recovering dev recall from 33.3% to
  47.7%.
- **TPR at 1% FPR remains at or below 0.03% in every condition.** Ranking
  improved by 0.16 AUC; the operating point did not move. This is the open
  problem, and it implicates the objective: at 0.1% FPR roughly twenty negatives
  set the threshold while the loss optimises mean likelihood over twenty
  thousand.
- Several conditions are n=2 and the 50% dev-recall range is [42-68]; 25% and
  50% should not be ranked against each other on this evidence.

Reproduction and caveats: `experiments/README.md`. Raw data:
`artifacts/phase3_archived/phase3_summary.json`.

Reporting rule: any PromptShield number must be labelled Track A (trained on the morgott corpus only; the split measures transfer) or Track B (trained on morgott plus PromptShield train; PromptShield-internal source-disjoint development rather than IID capability).
Known LMSYS-family overlap and missing row-level PromptShield provenance mean Track B is not source-OOD relative to the complete Morgott plus PromptShield fit.
Everything above is Track A. SEP (`artifacts/external_eval_data/sep/`, 18,320
rows, contamination 0.126% with zero on positives) is the replacement transfer
set; its positives are benign-intent imperatives, so it measures
instruction-in-data separation rather than harmfulness.

### Rejected tail and OOD-score diagnostics

A seed-42 global top-1% negative penalty with weight 0.1 after one warmup epoch was tested against its same-seed control.
PromptShield direct ROC AUC fell from 0.8004 to 0.7051, PR-AUC fell from 0.4724 to 0.3546, and descriptive TPR at approximately 1% row FPR fell from 2 of 6,486 positives (0.0308%) to zero.
SEP descriptive TPR at approximately 1% row FPR was effectively flat at 3.3079% versus 3.3188%.
Its higher transported SEP TPR came only with row FPR worsening from 3.30% to 6.07%.
Decision: reject the hard-negative penalty and do not retain its implementation.

Post-hoc energy computed from one binary logit is a monotonic transformation of the existing score and cannot change fixed-FPR ranking.
Revisit OOD scoring only with a separately trained OOD head or genuine outlier exposure.

## Full-data frozen mmBERT first-line shadow (2026-07-28)

This experiment trains a generic instruction-subversion head over the pinned `jhu-clsp/mmBERT-base` representation with strict normalization.
The fitted mixture contains 1,069,607 leakage-filtered canonical training rows, 18,197 leakage-filtered PromptShield training rows, and 11,041 retained generated pairs containing 22,082 rows.
PromptShield validation selects checkpoints but never fits parameters or selects operating thresholds.
A disjoint 116,488-row canonical calibration role selects thresholds.
PromptShield test and SEP remain already-open development evaluations.

The retained objective gives equal domain loss mass to canonical rows, PromptShield rows, and generated pairs, then adds aligned pair-ranking loss at weight 0.25.
The generic binary target is known for every selected row, so its main classification loss is ordinary BCE.
Masked losses remain appropriate only for future subtype heads whose labels can be unknown.
The frozen encoder and identical three-epoch schedules isolate the loss change, although validation-selected checkpoints can come from different epochs.
The scorer strictly normalizes and truncates every row to its first 512 tokens.
It does not chunk long documents or localize payload spans, so long web, retrieval, and tool content remains outside the supported evidence.
The three retained run directories are under `artifacts/combined_generic/full_runs/` and end in `_pair-rank-0p25_s42`, `_pair-rank-0p25_s43`, and `_pair-rank-0p25_s44`.
Their `result.json` SHA-256 digests are `401b8b20c6620fa1c34fa3c82eb59c6f387474f49ea3f7cd97877092d155d0ae`, `2fe0233828456dbb2ca4ee7d91e1e1e7e5d393fe1814926364975859acfb1f27`, and `ed3ca99f94adb1ee0c0af6bcb4aaa70b28a0d89f6b2fb03add43be503ff4bc6a`.
Their current-source `evaluation.json` SHA-256 digests are `737e8aab8c984841d0404bcfaef4b06442b75995f59738f870fbc9e81e146077`, `4c45cae5be97205f25a71045ffe722461ff9c95b4cc51988c621b938bd805360`, and `4ae05dbf3df0497948a44b655313c3165c3976ac1ff593bc1bf2f96eb2d9f2c2`.
The corresponding no-ranking control evaluation digests are `51c1e653057e2db3fff242d9c05e8b547f9cd7ecafeb9e1e43a2084a505fa5c9`, `5fb8296b23119b3ba39555c78ff5e91d6a5f7fa2a76c6f78fb4f1889fdc8054a`, and `25a2e10fb21e41de75800e5fe56434f0442453bf3b8d5a29f10c73c6fa5f0e71`.
All 54 recorded evaluation arrays were independently rehashed and checked for shape and dtype consistency.
All six v3 metric sections and score arrays are exactly equal to their v2 counterparts.
The v3 files pin the committed evaluator SHA-256 `5c536552a0007a4e89e2ed829f97e89183a0bf0c16f81505b1e061990c64e83c`, reverify every scored input and head before atomic publication, and treat the large training feature cache as recorded training provenance rather than an evaluation dependency.

| Seed | Validation macro BCE, BCE control to pair ranking | SEP descriptive TPR at empirical approximately 1% row FPR | SEP canonical-threshold TPR / row FPR, control to pair ranking |
|---|---:|---:|---:|
| 42 | 0.094521 to 0.093509 | 0.20% to 1.47% | 0.0109% / 0.1747% to 0.0109% / 0.0546% |
| 43 | 0.105266 to 0.088313 | 0.80% to 4.32% | 0.1092% / 0.3821% to 0.3712% / 0.3493% |
| 44 | 0.108474 to 0.090422 | 0.47% to 2.98% | 0.1528% / 0.6004% to 0.4148% / 0.3930% |

Pair ranking improved the descriptive SEP tail in all three seeds.
Its mean SEP TPR at the empirical approximately 1% row-FPR point rose from 0.4876% to 2.9258%.
The mean improvement was 2.4381 percentage points, with sample standard deviation 1.1262 points and a seed range from 1.2773 to 3.5262 points.
No confidence interval is claimed: three seeds estimate only limited run dispersion, and the flattened external releases do not expose enough lineage for a defensible independent bootstrap.
SEP ROC AUC and PR-AUC did not improve in every seed, so this is specifically a tail result rather than a uniform ranking gain.

At the canonical-calibrated threshold, mean SEP TPR rose from 0.0910% to 0.2656% while mean SEP row FPR fell from 0.3857% to 0.2656%.
Canonical dev-test mean TPR rose from 50.19% to 54.48% while mean row FPR fell from 0.3641% to 0.2931%.
PromptShield-internal source-disjoint mean TPR rose from 1.81% to 6.27% while mean row FPR fell from 0.689% to 0.644%.
PromptShield seed 43 is a material exception because both TPR and row FPR increased.
Finance false positives fell from a mean of 6.67 to 4.33 among 7,054 canonical finance negatives, with no finance source worsening.

The 1% calibration target is the primary operating diagnostic.
The 0.1% target remains unavailable because the untrusted-content calibration channel contains only 979 negative components.
Even zero observed false positives there has a 97.5% one-sided upper bound of 0.3761%, so a 0.1% claim would be unsupported.

Decision: retain frozen mmBERT with full-balanced BCE and pair ranking at weight 0.25 as the first-pass advisory research shadow.
It moved the SEP tail in all three seeds without an aggregate finance false-positive regression.
Absolute transfer remains weak because mean canonical-threshold SEP TPR is only 0.27%.
The model is not wired into the CLI and is not approved for blocking, authorization, transaction approval, or privilege grants.
The standalone `experiments/score_shadow_model.py` path verifies the registered artifacts and emits raw scores plus provenance without an authorization decision.
Trusted provenance must route its score, and every financial or Web3 side effect must still pass the deterministic reference monitor.

The old 81.14% number is not a competing result on this task.
It was the clean recall of a two-model ModernBERT plus mmBERT ensemble after generous same-suite recalibration near 0.1% row FPR.
That setup scored 0.00% TPR at 1% FPR on PromptShield and fell to 1.38% effective recall under 32 mutation attempts.
The new generic shadow has a stronger transfer-tail protocol and a broader training mixture, but its current absolute recall does not exceed 81.14% on a comparable test because no comparable test exists.

### LoRA engineering gate

The matched frozen seed-42 control fits 18,197 Morgott rows plus 18,197 PromptShield rows for 429 updates and selects epoch 3 at macro validation BCE 0.136361.
The LoRA gate uses the same rows, seed, update count, pinned encoder revision, and validation roles.
Only the base attention `Wqkv` and `Wo` projections receive rank-8 adapters with alpha 16 and dropout 0.05; the base encoder weights remain frozen.
Generated pairs are deliberately excluded from this gate so encoder adaptation is the only material training change.
The LoRA scorer uses the same strict normalization and first-512-token truncation as the frozen head.
One seed is sufficient for a preliminary engineering comparison, but it is insufficient for a robust method claim.
The LoRA run selected epoch 3 at macro validation BCE 0.051875, a 61.96% reduction from the matched frozen control.
Its `result.json` SHA-256 is `570db76e153f6d08bceb9a5688203e619e297d01f092effa25335743215ef90b`.
The matched frozen and LoRA current-source `evaluation.json` SHA-256 digests are `306637a7d10e5d2d9bc743db30c0fca75b7b78a93e53b4832364bd0b1370ccaa` and `c76956729eba694e378d5e8246f475dd6c4443a4c109b0550e49b4c13ade354e`.
Their 22 score arrays and all seven metric sections are exactly equal to their v2 counterparts and pin the same committed evaluator as the full-data runs.

| Metric | Matched frozen head | Rank-8 LoRA |
|---|---:|---:|
| SEP ROC AUC / PR AUC | 0.8321 / 0.7902 | 0.8758 / 0.8857 |
| SEP canonical-threshold TPR / row FPR | 2.03% / 0.71% | 31.71% / 0.92% |
| SEP descriptive TPR at approximately 1% row FPR | 3.89% | 32.48% |
| PromptShield-internal source-disjoint ROC AUC / PR AUC | 0.7452 / 0.4899 | 0.8410 / 0.7429 |
| PromptShield descriptive TPR at approximately 1% row FPR | 2.41% | 26.10% |
| PromptShield canonical-threshold TPR / row FPR | 43.54% / 12.31% | 64.29% / 10.28% |
| Canonical dev-test descriptive TPR at approximately 1% row FPR | 63.75% | 50.23% |
| Canonical dev-test canonical-threshold TPR / row FPR | 73.38% / 1.69% | 87.93% / 2.03% |
| Finance false positives among 7,054 negatives | 9 | 4 |

The ordinary checkpoint-selection and repeated canonical development metrics are:

| Candidate | n | Morgott checkpoint validation ROC AUC / PR-AUC | PromptShield checkpoint validation ROC AUC / PR-AUC | Canonical dev-test applied precision / recall / F1 |
|---|---:|---:|---:|---:|
| Full-data frozen plus pair ranking | 3 | 0.9956 / 0.9952 | 0.9961 / 0.9958 | 99.39% / 54.48% / 69.74% |
| Update-matched frozen | 1 | 0.9945 / 0.9943 | 0.9892 / 0.9889 | 97.44% / 73.38% / 83.71% |
| Update-matched rank-8 LoRA | 1 | 0.9969 / 0.9968 | 0.9999 / 0.9999 | 97.44% / 87.93% / 92.44% |

The full-data row is the arithmetic mean across seeds.
Canonical dev-test precision uses the canonical-calibrated 1% target and a development set with 46.77% positives, so it is not an estimate of review precision on deployment traffic.

Decision: LoRA passes this one-seed preliminary engineering gate because it materially improves SEP transfer, PromptShield ranking, and finance false positives.
It does not establish that LoRA is generally superior, and it does not yet replace the retained full-data frozen shadow.
It is not a strict Pareto win because SEP row FPR rises by 0.21 percentage points and canonical dev-test row FPR rises by 0.34 points at their transported thresholds.
FinanceBench also moves from zero to one false positive among 150 rows even though aggregate finance false positives fall from nine to four.
PromptShield row FPR remains 10.28% when the canonical threshold is transported, canonical same-test tail recall falls by 13.53 percentage points, and this gate excludes the generated pairs and most canonical training rows.
The next model experiment, when modelling resumes, is one full-mixture LoRA run with the retained pair-ranking objective and the same external evaluator.

### Deferred improvement ledger

1. Train one full-mixture LoRA run with the retained pair-ranking objective and repeat the same external evaluation.
2. Run additional LoRA seeds only if that full-mixture externally applied result remains materially better.
3. Build a prospective, lineage-grouped finance and Web3 suite with matched benign transaction tasks, realistic direct and indirect attacks, multilingual transformations, and a long-benign denominator.
4. Report mutation ASR at multiple attempts beside clean recall before any deployment claim.
5. Evaluate the deterministic policy monitor in AgentDojo Banking so the advisory sensor is measured as one layer rather than mistaken for the security boundary.
