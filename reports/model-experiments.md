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

The completed runner source is preserved by Git commit `91e8c829c8b39c8ff37a6ca2479c8fc057168d39`.
Raw local data was written to `artifacts/phase3_archived/phase3_summary.json`.

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
The three evaluated run directories were under `artifacts/combined_generic/full_runs/` and ended in `_pair-rank-0p25_s42`, `_pair-rank-0p25_s43`, and `_pair-rank-0p25_s44`.
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
The model is not wired into `morgott scan` and is not approved for blocking, authorization, transaction approval, or privilege grants.
At the time, `morgott shadow-score` verified the registered 512-token artifacts and emitted raw scores plus provenance without an authorization decision; that command was retired when maintained inference became 1,024-only.
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
This gate motivated the subsequently authorized full-mixture LoRA run below.

### Full-mixture rank-8 LoRA completion

The authorized full-mixture seed-42 run completed all 25,071 updates across three epochs and selected epoch 2 by the equal-domain mean of Morgott and PromptShield validation BCE.
It fitted 1,069,607 leakage-filtered canonical rows, cycled all 18,197 PromptShield rows class-balanced, cycled all 11,041 matched pairs, and applied pair-ranking weight `0.25`.
Those counts describe the immutable completed run.
The 2026-08-02 audit preflight used an audit-only fingerprint that additionally removes U+034F and supplementary variation selectors, exposing one same-label intra-training HackAPrompt duplicate and yielding 1,069,606 prospective canonical fit rows under that recorded preparation revision.
This does not alter the retained weights or their registered runtime normalizer.
The run kept the pinned rank-8 adapter contract, BF16 mixed precision, microbatch 8, and disabled gradient checkpointing after the measured memory preflight.
The machine-readable preflight is [retained with the experiment artifacts](../artifacts/mmbert/full-lora-runtime-benchmark.json) at SHA-256 `5158a674c884aca8643690f3f919bde2acedb42c1c91cf6b5b00c163f69b9a6a`.
The selected epoch reached Morgott validation BCE 0.04548, PromptShield validation BCE 0.05638, and macro BCE 0.05093.
The final result records 82,700 seconds of successful trainer runtime on the RTX 4050; interrupted lid-triggered and PTY work made campaign wall time longer.

The primary comparison is the complete-mixture LoRA against `mmbert-frozen-s42`.
The reduced-mixture LoRA remains historical context because its rows, update count, and missing pair loss differ materially.

| Already-open development metric | Full-data frozen | Full-data rank-8 LoRA | Reduced-mixture rank-8 LoRA |
|---|---:|---:|---:|
| Canonical AUROC / PR-AUC | 0.9877 / 0.9839 | **0.9896** / 0.9838 | 0.9870 / 0.9815 |
| Canonical descriptive TPR at approximately 1% row FPR | **73.84%** | 54.48% | 50.23% |
| Canonical transported TPR / row FPR | 39.59% / **0.22%** | 85.95% / 1.48% | **87.93%** / 2.03% |
| PromptShield AUROC / PR-AUC | 0.7634 / 0.5259 | **0.9015 / 0.8473** | 0.8410 / 0.7429 |
| PromptShield descriptive TPR at approximately 1% row FPR | 3.22% | **47.36%** | 26.10% |
| PromptShield transported TPR / row FPR | 0.35% / **0.16%** | **68.50%** / 4.96% | 64.29% / 10.28% |
| SEP AUROC / PR-AUC | 0.8368 / 0.7838 | 0.8144 / 0.8627 | **0.8758 / 0.8857** |
| SEP descriptive TPR at approximately 1% row FPR | 1.47% | **38.88%** | 32.48% |
| SEP transported TPR / row FPR | 0.01% / **0.05%** | 30.71% / 0.48% | **31.71%** / 0.92% |
| Finance false positives among 7,054 negatives | 5 | **0** | 4 |

The full LoRA materially improves PromptShield ranking and tail recall over the frozen head, reaches zero false positives on the retained finance slice, and orders the attack above its clean counterpart in 83.35% of SEP pairs.
It is not a strict win.
Its canonical descriptive 1% tail recall is 19.36 points below frozen mmBERT, its canonical transported FPR rises from 0.22% to 1.48%, its PromptShield transported FPR rises from 0.16% to 4.96%, and its SEP AUROC is lower.
The complete-mixture run also does not dominate the reduced-mixture LoRA on SEP ranking.

The fixed downstream DeepSeek policy cannot simply inherit the partial-LoRA score gates because the encoder score scales differ.
Copying `0.2 / 0.999 / 0.9` raises the full-LoRA evaluation result to 78.40% recall but also 3.62% FPR, including 8.40% PromptShield FPR.
A post-hoc `0.99999` high-gate extension reaches 66.79% recall at 1.81% FPR with 22.17% DeepSeek calls, but it was not in the predeclared grid and remains already-open development evidence.
The exact cascade comparison is in [the OpenRouter downstream evaluation](openrouter-downstream-evaluation.md).

The pinned local Llama Prompt Guard 2 86M comparison used revision `a8ded8e697ce7c355e395a0df51f94adb4a2fd27`, FP16 inference, native tokenization, and the first 512 model tokens over the same 461,700 row identities.
At its native 0.5 cutoff it reached 56.09% canonical recall at 1.81% FPR, 27.83% PromptShield recall at 2.76% FPR, and 0.02% SEP recall at zero FPR.
Under the shared canonical component protocol at threshold `0.9740303159`, those figures fell to 50.49% / 1.38%, 22.46% / 2.33%, and 0% / 0%.
Prompt Guard orders the SEP attack above its clean counterpart in 94.16% of pairs, but its score scale collapses near zero there and neither tested threshold converts that ranking into useful recall.
It produced one false positive among the 7,054 retained finance negatives under both thresholds.
Its local scorer processed 461,700 rows in 3,404 seconds with 2.18 GiB peak reserved memory; the complete inhibited service took 73.9 wall-clock minutes including preparation and publication.
The verified evaluation JSON and score-array SHA-256 digests are `e3b82fba87d2ca8494254b10353cf63d8a4627943a842713d30df84d7587f842` and `a95f3132c94aa170fa37f46d64ccc3343999fd2d49ac7b21997d389dba43124a`.

### Full-LoRA serving precision study

The retained post-hoc cascade was replayed over the frozen 20,000-row panel to select a CPU serving runtime.
This is already-open shadow engineering evidence, not a new model evaluation or a production claim.
The owner explicitly accepted a small paired numerical difference instead of requiring exact route parity.

| Runtime | Calibration FPR | Evaluation recall / FPR / precision | DeepSeek call rate | 512-token CPU p95 / QPS |
|---|---:|---:|---:|---:|
| Retained FP32 reference | 1.98% | 66.79% / 1.81% / 96.50% | 22.17% | ONNX Runtime: 400 ms / 2.59 |
| OpenVINO CPU BF16 | 2.01% | **67.06%** / 1.84% / 96.47% | **22.16%** | **155 ms / 6.66** |
| ONNX Runtime dynamic INT8 | 2.04% | 60.95% / **1.72%** / 96.36% | 32.71% | 224 ms / 4.74 |

OpenVINO BF16 changed 40 of 20,000 final routes, added one calibration false positive, improved evaluation recall by 0.27 points, increased evaluation FPR by 0.025 points, and left the provider call rate effectively unchanged.
It consumes the single registered FP32 ONNX graph and lowers eligible operations to BF16 at startup, so no second precision-specific model artifact is stored.
Dynamic INT8 reduced the model from 1.232 GB to 309.6 MB, but changed 846 final routes, lost 5.84 evaluation recall points, exceeded 2% calibration FPR, and raised provider calls by 10.54 points.
OpenVINO BF16 is therefore the selected shadow serving runtime, while INT8 is rejected and the temporary INT8 model is not retained.
A provider-free dynamic-batch probe over sixteen full 512-token windows preserved every BF16 probability exactly, but batch size two improved throughput by only 1.27% from 6.661 to 6.745 windows per second and batch sizes four and eight were slower.
Dynamic batching is rejected as immaterial for the registered latency-oriented CPU runtime, so the simpler one-window inference loop remains unchanged.

A full-panel NOOA PredictStrategy comparison was also rejected in favor of the maintained `CompletionClient` path.
Its copied `p=0.9` threshold improved evaluation recall from 66.79% to 71.22% but exceeded the calibration cap at 2.21% FPR, while its feasible `p=0.9706877673` threshold fell to 61.89% evaluation recall.
PredictStrategy also reduced valid-output coverage from 99.955% to 97.12% and increased prompt tokens, latency, and cost; the detailed results are in [the OpenRouter downstream evaluation](openrouter-downstream-evaluation.md#nooa-predictstrategy-comparison-2026-07-30).

The maintained word n-gram routing baseline was rerun separately against the strict-cross-role-clean canonical manifest recorded in the retained JSON snapshot.
At its untouched 0.5 cutoff it reached 92.46% dev-test recall, 3.18% row FPR, 96.49% precision, and 0.9827 AUROC over its broad `routing_label` target.
At that cutoff the validation TP/FP/TN/FN counts are 69,576/974/55,758/5,099, and the dev-test counts are 119,699/4,352/132,428/9,765.
Its origin-membership macro source recall and FPR were 80.02% and 15.24%, which exposes strong source variation hidden by the aggregate.
Exact-merged rows count once in aggregate metrics and in every applicable origin source slice, rather than only under the arbitrary representative source.
Substituting its measured dev-test recall and FPR gives only 2.83%, 22.69%, and 60.46% expected precision at 0.1%, 1%, and 5% attack prevalence respectively; these are arithmetic scenarios, not production estimates.
Length slicing exposes a separate denominator failure: dev-test benign FPR is 2.96% at 256 normalized characters or fewer, then 47.00%, 70.89%, and 62.96% across the longer bands, whose benign denominators are only 283, 237, and 27.
The selected fit contains only 64 benign rows from 257 through 1,024 characters and none above 1,024, while the corresponding positive counts are 17,196, 8,927, and 41.
The retained [JSON](routing-baseline.json) and [Markdown](routing-baseline.md) snapshots have SHA-256 digests `9366b13422099510f987e438d2140c71c18ee1dcce9f5c695b8e30b96fceb9d7` and `2adc2a6465eb7846d0ba99147a94f42ba07e7f9148413feb2a62e1c9c70a91d4`.
Both versioned snapshots reproduced byte for byte on an immediate second run after removing the non-evidentiary wall-clock field.
Aggregate differences from the earlier historical snapshot come from the data-manifest change, while the source-macro change on this manifest is only the corrected membership accounting; neither is evidence that the same model improved or regressed.
This control is not directly comparable to the narrower instruction-subversion models because broad routing labels also include harmful intent, toxicity, and unresolved rows.

#### Source-cap ablation

A matched source-balance ablation on manifest SHA-256 `0c5e22107975588dfa2628bf4ac290ec9dc8f344c9c9a3da69c72aa208ca5edc` reduced the per-source-label training cap from 20,000 to 2,000 while keeping the seed, three epochs, unweighted loss, features, and untouched 0.5 cutoff fixed.
The cap reduced selected training rows by 85.80%, from 154,049 to 21,870.

| Per-source-label cap | Validation recall / FPR / macro recall / macro FPR | Dev-test recall / FPR / macro recall / macro FPR |
|---:|---:|---:|
| 20,000 control | **93.19%** / **1.69%** / 85.79% / 2.83% | **92.42%** / **3.17%** / 79.94% / 15.13% |
| 2,000 candidate | 91.32% / 2.04% / **88.22%** / **2.70%** | 91.20% / 3.43% / **80.25%** / **13.36%** |

The smaller cap improved unweighted macro-source recall and FPR, but it degraded aggregate recall, FPR, precision, PR-AUC, ROC-AUC, and Brier score on both splits.
On dev-test it lost 1.22 recall points and added 0.27 FPR points while gaining only 0.32 macro-recall points.
The 2,000 cap is rejected as the maintained default because the compute reduction does not improve routing quality.
The unchanged 20,000-cap control remains the cheapest retained recipe, and a future source-balance study needs a better weighting hypothesis rather than a more aggressive global cap.

#### Capped inverse-square-root source-weight ablation

A second matched source-balance ablation on that same canonical manifest retained all 154,049 selected training rows and changed only each row's training weight.
For source-label count `n`, the raw weight was `min(4, sqrt(20000 / n))`, followed by normalization to mean one; resulting weights ranged from `0.870315` to `3.481261`.
The seed, three epochs, feature map, optimizer, 20,000 selection cap, and untouched `0.5` cutoff stayed fixed.
The predeclared gate required at least one percentage point of macro-source recall improvement on both roles, no macro-source FPR increase, no aggregate FPR increase, at most 0.25 percentage points of aggregate recall loss, and no PR-AUC or ROC-AUC loss.

| Recipe | Validation recall / FPR / macro recall / macro FPR | Dev-test recall / FPR / macro recall / macro FPR |
|---|---|---|
| Unweighted control | 93.1717% / 1.7168% / 86.3617% / 4.2122% | 92.4574% / 3.1818% / 80.0217% / 15.2425% |
| Capped inverse-square-root weights | 92.7633% / 1.6499% / 89.7729% / 2.5522% | 92.1577% / 3.1160% / 80.5222% / 14.7384% |

The weighted candidate reduced aggregate and macro-source FPR on both roles, but validation aggregate recall fell by 0.4084 percentage points and dev-test recall fell by 0.2997 points.
Validation PR-AUC and ROC-AUC fell from `0.993599` and `0.990317` to `0.993411` and `0.990072`, while dev-test PR-AUC fell from `0.985367` to `0.985256`.
Its dev-test macro-source recall gain was only 0.5006 percentage points, below the one-point gate, and Brier loss worsened on both roles.
An exact deterministic repeat reproduced the candidate metrics.

Decision: reject the weighting candidate and keep the unweighted control unchanged.
Do not sweep weighting exponents or caps on these consumed development roles; revisit source balance only with new source-held-out evidence or a qualitatively different, predeclared hypothesis.

Decision: register `mmbert-lora-full-s42` as a one-seed advisory research artifact without promoting it over the retained frozen model or the existing partial-LoRA downstream route.
No learned model is approved for blocking, authorization, or privilege grants, and `morgott scan` remains advisory with `decision: allow`.

### Auxiliary matched boundary diagnostic

The complete official test split of the already-open auxiliary `agentic_boundary_pairs` source was scored without fitting or threshold selection on that source.
A total of 120 matched pairs were scored: 80 instruction-subversion pairs for detector metrics and 40 authorization-boundary pairs reported separately.
A strict-normalized exact audit found no overlap with any canonical routing role, PromptShield split, SEP, or the retained generated-pair archive.
This is a small synthetic development diagnostic from an auxiliary source already present in the canonical data manifest, not prospective or independent external evidence.
Only the registered seed-42 frozen head is available for this replay, so the frozen result does not measure seed dispersion.

| Registered artifact | Instruction-subversion AUROC | Attack ranked above paired clean | Recall / row FPR at unchanged canonical 1% component-target threshold | Authorization-only attack-side / clean-side flag rate |
|---|---:|---:|---:|---:|
| Frozen full-data seed 42 | 0.7852 | 88.75% | 6.25% / 0.00% | 0.00% / 0.00% |
| Reduced-mixture LoRA seed 42 | 0.7291 | 77.50% | 50.00% / 18.75% | 20.00% / 10.00% |
| Full-mixture LoRA seed 42 | **0.8370** | **91.25%** | 48.75% / 16.25% | 10.00% / 2.50% |

The full LoRA improves pair ordering over both registered alternatives, but its transported operating point does not preserve the calibration false-positive rate on these matched controls.
It flagged every clean obfuscation control, which accounts for most of its instruction-side false positives and strengthens the existing requirement for symmetric benign and attack transformations.
The authorization-only columns are diagnostics rather than detector recall or FPR because those pairs deliberately exercise approval, authority, egress, and tool-use boundaries outside the instruction-subversion target.
They reinforce that learned routing cannot replace the deterministic reference monitor.

### Channel-aware downstream repair

The maintained DeepSeek reviewer previously discarded trusted `input_channel` metadata and over-relied on the presence of an attack payload.
A first outer-intent prompt was rejected because it lowered mixed-domain calibration AUROC and nearly eliminated SEP recall.
The selected hybrid keeps the indirect-injection rule for task-like instructions inside untrusted content while distinguishing analysis of a payload from requests to execute it.

On the frozen 6,000-row calibration role, its standalone AUROC / average precision improved from 0.9343 / 0.9114 to 0.9460 / 0.9202.
With full-LoRA local gates fixed at `0.2 / 0.99999`, threshold `0.7310585786300049` raised calibration recall from 66.58% to 69.11% at the identical 68 false positives and 1.9808% FPR.
On the untouched 14,000-row evaluation role, recall rose from 66.79% to 69.85%, FPR fell from 1.810% to 1.760%, and precision rose from 96.50% to 96.74% at the unchanged 22.17% provider call rate.
SEP recall rose by 9.43 points and canonical recall by 1.40 points, while PromptShield recall fell by 2.90 points.

Reviewer-only scoring of the complete already-open instruction Boundary Pairs test improved from 98.75% recall / 15.00% FPR under the old remote contract to 100% / 5.00% under the selected contract and threshold.
All clean obfuscation controls passed when every row reached the reviewer; three memory-context controls and one multi-agent trust control remained false positives.
The actual three-zone cascade reached 90.00% recall / 12.50% FPR and 77.50% fully correct pairs because eight attacks fell below the local gate and eight clean rows crossed the local high gate.
A calibration-only gate search found no material generalizing improvement under the 2% FPR and 35% call-rate caps, so the `0.2 / 0.99999` local gates remain unchanged.
OpenVINO BF16 serving equivalence passed with 70.02% evaluation recall, 1.785% FPR, 96.70% precision, and one extra calibration false positive relative to the new FP32 reference.

A later AgentDojo transfer diagnostic exposed a narrower untested option: change only the low floor selected by trusted `input_channel`.
The fixed follow-up retained the direct-user floor `0.2`, high gate `0.99999`, and reviewer threshold, then compared untrusted-content floors `0.2`, `0.1`, and `0.05` on the same 6,000-row calibration role.
The `0.1` floor is the highest-recall feasible point under the unchanged 2% FPR and 35% call-rate caps.
It adds 10 calibration true positives and 49 provider calls with no false positives, keeping calibration FPR at 1.9808%.
On the already-open 14,000-row evaluation role it adds 25 true positives, one false positive, and 104 calls, reaching 70.27% recall, 1.773% FPR, 96.74% precision, and a 22.91% provider call rate.
The gain is concentrated in untrusted content: SEP recall rises from 48.06% to 49.37% at unchanged 0.114% FPR, while direct-only PromptShield is unchanged.
This is a post-hoc development improvement, not prospective validation.
Fresh OpenVINO BF16 verification under the channel-specific floor passes every serving-equivalence check.
It reaches 70.42% evaluation recall, 1.798% FPR, 96.70% precision, and a 22.89% provider call rate, compared with FP32 at 70.27% / 1.773% / 96.74% / 22.91%.
The runtimes differ on 89 local zones and 24 final routes over 20,000 rows, and BF16 adds one calibration false positive.

Decision: use this contract only in the advisory cascade and retain the full-LoRA artifact's one-seed research status.
The detailed prompt-selection protocol, per-dataset results, immutable evidence hashes, and limitations are in [the OpenRouter downstream evaluation](openrouter-downstream-evaluation.md#channel-aware-outer-intent-repair-2026-08-02).

### August 2026 cascade, workload, and containment ledger

Detailed protocols, complete slices, and hashes live in the dedicated reports linked below.
This ledger retains only decisions that affect maintained behavior or the next experiment.

#### Retained cascade decision through 2026-08-16

- DeepSeek V4 Flash 0731 replaced the April reviewer after adding 58 true positives with the same 142 false positives on the frozen evaluation.
  FP32 recall, FPR, precision, and call rate are 71.235%, 1.773%, 96.779%, and 22.914%; OpenVINO BF16 reaches 71.386%, 1.798%, 96.742%, and 22.893%.
  PromptShield recall fell 2.176 points and untrusted-content FPR rose 0.375 points, so this is an aggregate advisory improvement rather than uniform transfer.
  The incumbent local gates were direct-user 0.2, untrusted-content 0.1, and high 0.99999; the reviewer threshold was 0.6224593312018547 and concurrency was capped at four.
  See [the replacement report](deepseek-v4-flash-0731-research.md).
- The selected cascade remains advisory, is not approved for blocking, and never grants authority.
  Full-context-first review applies only to remote-enabled multi-window untrusted content; scalar call-minimization and Ionstream fallback candidates were rejected.

#### Balanced advisory promotion on 2026-08-17

- The owner promoted the exact benchmark-selected balanced profile as the maintained advisory default.
  Its direct and untrusted floors are 0.2 and 0.025, its local-high gate is 0.9999, and its reviewer threshold is 0.5.
  The DeepSeek V4 Flash 0731 prompt, Cloudflare strict-logprob request, reasoning-disabled contract, concurrency four, 1,024-token windows, 128-token overlap, full-context-first review, ordered fallback batches, failure behavior, and `decision: allow` authority are unchanged.
- Selection used only the frozen 1,024-row calibration provider panel and its required windows.
  On 12,352 provider-safe consumed evaluation artifacts, CUDA BF16 reached 72.00% recall, 1.37% FPR, 96.68% precision, and a 13.96% artifact review rate.
  The unchanged thresholds transported to the full OpenVINO CPU ledger reached 72.10% recall, 1.36% FPR, 96.72% precision, and a 13.94% artifact review rate.
- A post-review source-slice audit disqualified every strict no-logprob provider under the declared 2 percentage-point slice-loss rule.
  Decart's exact result remains a diagnostic, but its HackAPrompt source recall was 3.10 points below the best provider, so the benchmark has no strict hard-verdict winner.
  This correction does not change the selected Cloudflare strict-logprob cascade.
- The sealed LogInject remote replay supports the selected profile on synthetic out-of-distribution logs at 96.57% attack recall and 0.54% paired clean false restrictions.
  The revised reviewer prompt and channel split are rejected because the broader screen lost 29 of 142 attacks for one fewer false restriction.
  High reasoning did not dominate reasoning-disabled review and is not integrated.
- This is a maintained advisory promotion, not production calibration or blocking authorization.
  The evaluation roles are consumed development evidence, representative traffic is absent, the complete long-character diagnostic remains a false-positive warning, mutation evidence is local-only, and the Azure load run predates the promoted profile.
  The next decision-changing evidence is a prospective task-bearing long benign and matched-attack shadow panel.
- The registry-bound promotion record is `artifacts/models/mmbert-lora-full-ctx1024-u17000-s42/serving/promotion.json`.
  Azure revision `morgott-api--0000016` serves it at 100% traffic with matching policy, threshold, model, and ONNX identities; its retained 30-request check is smoke evidence rather than a promoted-profile load benchmark.
  The complete benchmark and provider/runtime limitations are in [the pipeline benchmark](pipeline-benchmark-20260816.md).

#### Learned-detector and workload findings

- The Financial AI CTF diagnostic reaches 83.33% row recall but only 63.26% participant-macro recall, and most protected-field leaks fall outside instruction subversion.
  FORCE-Bench records 0 restrictions across 251 legitimate finance tasks.
  See [the finance audit](finance-web3-benchmark-audit.md).
- AgentAbstain shows at least 3.45% restrictions overall and 14% on high-stakes tasks.
  SafeClawBench reaches only 10.11% recall at 4.26% FPR.
  BFCL v4 records 0.878% restrictions and 14.34% review incidence across 2,050 live controls.
- Long-document and task-conditioned repairs did not transfer cleanly.
  Known-span documents retained weak payload recall and high clean review, while LongBench, StruQ, InjecAgent, API-Bank, AgentDyn, and AgentPIMA showed that scalar thresholds or trusted-task prompting do not solve the source shift.
  See [the long-context report](long-context-reviewer-research.md), [the InjecAgent report](injecagent-detector-evaluation.md), and [the task-conditioned report](task-conditioned-reviewer-evaluation.md).
- SWE-bench Verified records 1.42% hard restrictions and 69.11% review on 492 long legitimate tasks.
  SWE-chat records 1.133% hard restrictions and 52.255% review on 5,121 earliest-session prompts, with 93.471% review at 256 or more words.
  SWE-rebench records 0.954% hard restrictions and 52.061% review across 20,762 tasks, with roughly 96% to 97% review on 2,048-character or longer tasks.
  These are workload measurements rather than production FPR estimates.
  See [the SWE-chat report](swe-chat-traffic-research.md).
- PIArena reaches 95.96% attack recall, 0.77% clean restrictions, and 83.85% fully correct matched tasks after a bounded transport completion.
  The original run still records 37 provider failures.
- Adaptive Adversaries retained 93.18% attack recall but failed its clean-pair gate because flattening mixed provenance was the wrong projection.
  See [the adaptive report](adaptive-adversaries-evaluation.md).
- ASPI shows 31.41% of retained clarification attacks below the local review gate, including 71.03% of Travel attacks.
  Clarifications must bind typed missing slots and require separate authorization for scope expansion.
  See [the stateful benchmark report](agent-security-benchmark-options.md#aspi-clarification-state-audit-exposed-a-routing-blind-spot).
- The tau2 retail construction did not create an independently supported matched training source and did not authorize fitting.

#### Stateful containment findings

- The provider-free AgentDojo Banking replay preserves all 144 paired legitimate outcomes while reducing upstream attacker-goal completion from 144 of 144 without monitoring to 0 of 144 with monitoring.
- The fixed-agent Banking comparison improves unattacked utility from 6 of 16 to 8 of 16 and attacked utility from 56 of 144 to 71 of 144 while retaining zero unauthorized mutations.
  Warning-only and Slack policy variants were rejected or left incomplete on utility.
  See [the AgentDojo report](agentdojo-integration-research.md).
- One load-bearing Nemotron case and one coupled Agent-Diff Linear incident demonstrate vulnerable controls, exact legitimate state, deterministic denial, and zero unauthorized mutations.
  Other Agent-Diff, STATE-Bench, AgentLure, and workspace transfers were inconclusive because the attack was not proposed, the required source was not observed, or transport failed.
  See [the stateful benchmark report](agent-security-benchmark-options.md).
- The maintained security boundary remains typed capabilities, stable source identities, monotone provenance and sensitivity propagation, synchronous pre-effect enforcement, and deterministic egress policy.

#### Robustness and next-model evidence

- Multi-attempt mutation conditional evasion at 1, 2, 4, 8, 16, and 25 attempts is 0.87%, 1.30%, 1.71%, 2.14%, 2.57%, and 2.79%.
  Effective end-to-end recall falls from 73.93% to 72.50%; no mutation crosses below the local pass gate.
  The manifest, base result, mutation result, and summary digests are `5b4138ef08cd84456749fc9ad8100024bc07d65417a6dbe4f39295e1c1b8cdb8`, `183c1ff8680e7b097b3d93fa0bd4e61f6fd20c81b038a2b76b12661eaeabdbfb`, `25aba3e75ba563f4eeaf659d54a331f79a27d3ac59cff2b50ad057b11e6709f1`, and `57ba976f7736a0744ea7b954ba1c56433b1506af22209eb4dcf71071a6aa769b`.
- Another learned-model run still requires explicit authorization, same-row incumbent scoring, source-heldout and indirect-document gates, a long legitimate workload, and the retained mutation curve.

#### Frozen mmBERT LP-FT comparison (2026-08-05)

The owner-authorized LP-FT candidate trained encoder layers 20 and 21, the final normalization, and the retained frozen head on the rebuilt canonical mixture plus 22,716 repository-grouped SWE-rebench V2 matched pairs, with seed 42, three epochs, 25,083 updates, and the unchanged objective and checkpoint rule.
Training completed in 10.11 hours on the local RTX 4050 at 1.04 GiB peak reserved VRAM, and the trainer now writes an atomic progress checkpoint every 500 updates that restores the same update sequence; validation and best-epoch selection remain epoch-boundary only.
The resume path does not enable deterministic CUDA algorithms and does not claim bit-identical floating-point results across process or hardware changes.
Epoch validation macro BCE was 0.08856, 0.09102, and 0.08478, so epoch 3 was selected after its PromptShield validation component recovered from the epoch-2 regression.

Evaluation first failed closed because the dev-test real-finance negative pin still recorded the July count.
The corpus rebuild removed 11 harper_valley_bank rows from the scored slice: the manifest records one additional routing-quarantine row and a three-row dev-test reduction for that source, while the remaining eight rows stayed in dev-test but left the scored direct-user channel through whole-conversation lineage regrouping.
The pin therefore moved from 7,054 to 7,043 with no other evaluator change.
Population guards should run before scoring in a future maintenance pass because the stale pin cost one complete scoring run.

Against the incumbent full-mixture LoRA, threshold-free ranking ties on canonical dev-test (ROC AUC 0.98967 versus 0.98962), improves SEP (0.87567 versus 0.81440), and collapses on PromptShield test (0.75142 versus 0.90149), with recall 14.82% versus 68.50% at each run's own calibration-selected component threshold.
The candidate removes the incumbent's worst benign slice (multi_turn FPR 86.75% leaves the worst list) but loses indirect-document recall (BrowseSafe 34.05% to 0.16%, BIPIA 41.08% to 8.82%) and flags 9 of 7,043 real-finance negatives where the incumbent flags 0.
Evaluation populations differ slightly because the corpus was rebuilt, so aggregate rows are not identical between the two evaluations.

On the repository-held-out SWE-rebench V2 slices scored with window-max over ordered 512-token windows, the descriptive validation-calibrated 1% coordinate gives the candidate 0.17% dev-test clean flags, 0.03% clean high-gate restrictions, and 85.20% attack recall against the incumbent's 7.31% clean flags, 1.50% clean high-gate restrictions, and 98.02% recall; threshold-free pair ordering is 99.86% versus 99.42%.
On clean documents of at least 8,192 normalized characters the candidate flags 0.00% while the incumbent flags 44.44%; that slice holds only 18 pairs (0 versus 8 flags), so it is directional evidence rather than a precise rate.
Repository grouping holds out task contexts, not attack templates: 2,267 of 2,590 unique dev-test attack spans also occur in training.
Attack recall, AUROC, and pair ordering are therefore in-family template evidence rather than attack-side generalization, and the candidate's clean-side advantage partly reflects same-source familiarity that the incumbent lacks.

Decision: the candidate is not retained because the PromptShield and indirect-document regressions fail the frozen Pareto rule; the maintained artifacts, gates, and cascade are unchanged.
The handoff stopped after the decisive core result, so the sealed LogInject panel remained unopened and the mutation ASR and SWE-rebench workload reruns were not started.
This is an explicit protocol deviation and those panels provide no evidence for the candidate.
The durable positive result is that the matched long-task pairs work: they remove most long-clean flagging on exactly the workload that rejected the current cascade.
The retained result, evaluation, new-data manifest, and held-out comparison SHA-256 digests are `abf09b1a50eb767ce250ed9e29f90bcfbf3e29cd356d1a0786317e39ae5219b4`, `8890a302d4808f414ef22fdb6dec8615fc556c50af74d28dca6710bbbf8c05f7`, `1f3a7dfdf9a35c4910f2762cc8b971dc54cdf1af1036b754a2753b4d2f0fb45b`, and `63b5147631ab073cefc66e5e27309d5a87b5ae81d6ad9ad4d2bf22c8271fe6b0`.
The rejected encoder, head, and scores are retained through Git LFS for later comparison but remain unregistered and unavailable to maintained inference.
Their SHA-256 digests are `271df253cd4fc807c6060059e9bb62dc85e0c317aed15a892be5c7186cf3d515`, `2b5dbd647484e8753441118bf45bccd7c3836982474978f653c2de606bef98b5`, and `1cf04f41f76c2048d0170880a4794dae1d97da1efb38d1523a8cfd4c6c415aa5` respectively.
The owner approved deleting the resumable progress checkpoint after its SHA-256 `0e46cad31c0a92284ee85a6011c7ba723b78e4fcaac76f8669a4a69929585c2e` was recorded; the retained weights, scores, and evaluation records remain available for later comparison.
The natural disentangling candidate is rank-8 LoRA retrained on the identical
new mixture at matched updates under its own frozen protocol.

### Deferred improvement ledger

1. Replace the rejected Adaptive Adversaries composite-text projection with an end-to-end adaptive suite that preserves field-level provenance and has deterministic task, action, and clean-utility oracles.
2. Treat the rejected SWE-chat traffic proxy as consumed development evidence, preserve the frozen PromptShield regression diagnosis without retuning either source, and keep the AgentPIMA-rejected task-conditioned candidate out of integration unless a materially different architecture with a frozen low-call invocation policy succeeds on new independent evidence.
3. Use the completed multi-attempt mutation ASR curve beside clean recall in every candidate comparison.
4. Keep the rejected advisory-warning result as fixed development evidence, and test any revision only on a prospective task or attack family with a warning-aware adaptive attack.
5. Replace the inconclusive coupled transfer sequence with an independently versioned task whose clean completion demonstrably consumes the untrusted record and whose vulnerable control passes before the same proposal is forked, while retaining stable identity, oracle authority, one-shot capabilities, and complete-state gates.
6. Treat clarification replies as typed data bound to the original task scope, and require a separate authorization event before a reply can add actions or widen capabilities; evaluate any task-conditioned reviewer only as an advisory layer on a new prospective source with legitimate scope-expansion controls.

## Capacity ladder arms and pinned guard baselines (2026-08-07 to 2026-08-12)

All numbers from `artifacts/mmbert/runs*/`, `artifacts/comparisons/<slug>/`, and
the frozen Aug 6 red-team reserve. Advisory only; no promotion is implied and
no new ladder arm in this section is registered.
The checksum-bound context-study index is
[`mmbert-context-comparison.json`](mmbert-context-comparison.json); it, rather
than the Trackio UI, is the compact machine-readable source for the comparison.

### Batch-ordering ablation strongly implicates length grouping

The Aug 7 full-mixture LoRA (`mmbert-lora-full-s42-mb24`) improved canonical and
SEP but lost public transfer. The
`runs-ablation/mmbert-lora-full-s42-mb24` `--no-length-grouped` arm is a
batch-ordering-focused ablation. Preserved provenance records intervening
training/data-source changes, so this is not a single-variable experiment and
does not prove grouping is the sole cause. It nevertheless strongly implicates
grouping in the observed trade. Detection at a fixed descriptive 1% FPR:

| Panel | Jul 28 | Aug 7, length-grouped | Aug 7, grouping off |
|---|---:|---:|---:|
| Canonical dev-test | 54.5% | 70.2% | 56.3% |
| SEP | 38.9% | 59.5% | 49.8% |
| PromptShield test | 47.4% | 40.2% | 54.8% |

In this ablation, turning grouping off recovers PromptShield above both prior
models and gives back most of the in-distribution and indirect-injection gain.
Because provenance does not support a single-variable claim, treat length
grouping as a leading mechanism rather than a proven sole cause. Choose the
operating point per deployment target and state which arm produced a quoted
number.

### Arm 6 and its no-harm control completed

The rank-8, no-length-grouping multitask arm
`mmbert-lora-full-s42-mb24-nolengthgroup-harmful-balanced` completed all 25,083
updates and three epochs in 14,436 seconds of packaged runtime on the L40S. It
used compiled SDPA at 512 tokens, the complete 1.07-million-row canonical
primary objective, PromptShield fitting rows, and the retained matched-pair
archive. The run result is
`artifacts/mmbert/runs/mmbert-lora-full-s42-mb24-nolengthgroup-harmful-balanced/result.json`.

The pre-registered update 17,000 and the later update 23,000 diagnostic were
both scored on the complete current panels. The TPR columns are threshold-free
same-panel coordinates at descriptive 1% FPR; the finance column uses each
checkpoint's calibration-selected 1% component threshold.

| Checkpoint | Canonical AUROC / TPR@1% | PromptShield AUROC / TPR@1% | SEP AUROC / TPR@1% / pair order | Real-finance flags |
|---|---:|---:|---:|---:|
| Arm 6 at 17,000, pre-registered primary | 0.98784 / 45.88% | 0.93224 / 54.18% | 0.78937 / **47.31%** / 79.47% | **0 / 7,043** |
| Arm 6 at 23,000, post-run diagnostic | 0.99000 / **61.35%** | 0.92668 / 43.11% | **0.86299** / **56.92%** / **87.58%** | 13 / 7,043 |
| No-harm control at 17,000, explicit 512/512 | **0.99013** / 60.15% | **0.93299 / 57.00%** | 0.81941 / 44.81% / 82.53% | **0 / 7,043** |

At their transported thresholds, update 17,000 flags 2,304 of 2,400
multi-turn controls, 159 of 160 llmail controls, and one of 492 SWE-bench
Verified tasks; update 23,000 flags 2,233, 160, and zero respectively. These
adversarial-control slices are diagnostics, not ordinary-benign FPR estimates.
The full records are the two `evaluation-update-*/evaluation.json` files under
the Arm 6 run directory.

The explicit 512/512 no-harm evaluation flags 2,152 of 2,400 multi-turn
controls, 160 of 160 llmail controls, and one of 492 SWE-bench Verified tasks.
Its calibration-selected threshold is 0.9902915235. Training packaged in
14,854.14 seconds; the later resumable full-panel evaluation took 6,124.06
seconds. The package records update 17,000 as its exact weights provenance, so
a duplicate "final package" evaluation would score the same weights and is not
required.

The frozen 5,112-row red-team reserve does not provide a benign denominator.
"Bare harmful" below is an off-target flag rate on harmful requests without
source-attested instruction subversion, not FPR.

| Checkpoint | Source-attested subversion | Bare-harmful off-target | Pooled flag rate | Auxiliary mean score |
|---|---:|---:|---:|---:|
| Arm 6 at 17,000 | 981 / 1,122 = 87.43% | 582 / 3,990 = 14.59% | 1,563 / 5,112 = 30.58% | 0.5044 |
| Arm 6 at 23,000 | **986 / 1,122 = 87.88%** | 626 / 3,990 = 15.69% | 1,612 / 5,112 = 31.53% | 0.0973 |
| No-harm control at 17,000, explicit 512/512 | 955 / 1,122 = 85.12% | **413 / 3,990 = 10.35%** | **1,368 / 5,112 = 26.76%** | n/a |

The reserve gain at update 23,000 is only 0.45 percentage points while its
off-target rate is 1.10 points worse. More importantly, no frozen selection
rule authorized choosing 23,000 after looking at PromptShield, SEP, finance, or
the reserve. Update 17,000 therefore remains the pre-registered primary result;
update 23,000 is a useful trade-off diagnostic, not a replacement or promotion.
The aggregate-only records are the two
`redteam-reserve-evaluation-update-*/evaluation.json` files under the run.

The matched comparison is therefore mixed but useful. Against the explicit
512/512 control, adding the harmful head lowers canonical TPR@1% by 14.27
points and PromptShield TPR@1% by 2.82 points, lowers canonical and PromptShield
AUROC, and raises bare-harmful off-target flags by 4.24 points. It improves SEP
TPR@1% by 2.50 points and reserve source-attested recall by 2.32 points,
although the no-harm control has better SEP AUROC and pair ordering. The head
hurts most primary panels and specificity for a narrow gain on two recall
coordinates; that is not a case for keeping the current auxiliary recipe.

The no-harm artifacts are under
`artifacts/mmbert/runs/mmbert-lora-full-s42-mb24-nolengthgroup-noharm-current-control/`.
The no-harm recipe became the 512-token base for the completed matched
1,024-token study below. Its explicit-cap records supersede the earlier
implicit-cap full and reserve JSONs; the old files remain indexed for
provenance. Compact curves may be viewed in the curated Trackio project
`morgott`, but UI run IDs are not evidence. This remains advisory development
evidence and does not promote or register either run.

The live Trackio store was non-destructively curated on Aug 12 after a verified
SQLite backup. All 19 historical run identities and every metric-row record
were retained; redundant JSON metric keys and historical per-core telemetry
were projected out of the live viewer, while the complete pre-curation history
remains at `/workspace/hf_cache/trackio-archive/20260812T105500Z/` with its own
checksum manifest. Two compact `decision-summaries` runs add only eight
headline metrics: the fixed-update native 512/1,024 comparison and the native
1,024 update-17,000/update-18,500 diagnostic. The live project now has 21 runs
and 68 user-metric keys. All eight historical positive-row
`val_bce_missed_attacks/*` curves are retained, but these legacy BCEs are not
miss counts or recall; future runs use label-aware validation metrics. The
checksum-bound JSON records remain authoritative.

### The auxiliary harmful head is source-confounded

This limitation concerns only the auxiliary harmful-intent head. It does not
mean the instruction-subversion objective was trained on a small corpus: that
primary objective used more than one million canonical rows plus PromptShield
and 33,757 long-task and other matched pairs.

Only 49.60% of Arm 6 canonical training rows have a known harmfulness label.
WildJailbreak supplies 99.7149% of all harmful-positive supervision and
ToxicChat supplies the remainder. The harmfulness 2x2 contains no
`injection=1, harmful=0` examples, so the auxiliary head never sees an explicit
counterexample to "subversion cues imply harmfulness." Checkpoint selection is
more confounded still: 5,614 of its 5,622 harmful positives are WildJailbreak,
whose checkpoint slice contains only one harmful-negative row.

At update 17,000, pooled canonical-dev harmful AUROC / AP is 0.9268 / 0.8273,
but the unweighted macro over the four sources containing both classes is only
0.7647 / 0.7862. Calibration BCE / AUROC / AP is 0.2394 / 0.9871 / 0.9659.
At update 23,000, pooled dev ranking rises to 0.9418 / 0.8606 while calibration
BCE worsens to 0.4642 and the reserve's unlabeled auxiliary mean falls from
0.5044 to 0.0973. The correct conclusion is narrow WildJailbreak/style learning
with unstable score scale, not broad harmfulness generalization. There is no
supported harmful-score operating threshold and no evidence for extending this
recipe beyond three epochs.

The completed no-harm control isolates the current auxiliary loss as a net
negative for the primary objective under the pre-registered comparison. Any
future harmful branch needs the data, sampling, and validation repairs above;
longer training on the existing branch is not the next experiment.

### The 1,024-token context study completed

The original 512-versus-1,024 canary showed that 1,024 tokens fit on the L40S
at 13.90 GiB peak but ran about 1.557 times slower. Its first numerical
comparison mixed efficient-only compiled SDPA with an eager path that could also
choose Flash, so that failure did not isolate compilation.

The replacement same-backend diagnostic forced efficient SDPA for all paths
and ran three fresh replicas each of uncompiled, Dynamo-eager, AOT-eager, and
Inductor. All 12 probes and every cross-mode and within-mode gate passed the
unchanged loss-relative, gradient-cosine, and gradient-relative-L2 thresholds
of 0.05, 0.99, and 0.20. The result is
`artifacts/mmbert/arm6-context-parity-cap1024-efficient-s42-r3-mb24.json`,
SHA-256 `ab2ca66759bd31064f85ef3b6c931310dd4784247ad7060fe86519c2a626df36`.
This cleared the numerical correctness prerequisite.

The launch-binding V2 tail audit then counted every fitting, selection,
calibration, and already-open evaluation row instance using the exact prepared
population and tokenizer. It found 13,397 clean and 71,210 attack instances
whose tokenized input exceeds the 512-token baseline, against a predeclared
minimum of 300 per label. The gate passes. The aggregate-only artifact is
`artifacts/mmbert/context-tail-audit-cap512-vs1024-s42-v2.json`, SHA-256
`dd6aad99efba698f9cfb32da3aa7e1a2f1e51a759f3e8ccd7baf3142f84e7b4a`;
it records 1,647,409 audited row instances and 717.05 seconds runtime. V2 binds
the patched training source used for launch and supersedes the pre-patch V1
audit even though the affected counts are unchanged.

The first 1,024-token launch reached update 500 and failed before validation
completed. The cap-specific validation path relied on `train_encoder=False`,
whose no-gradient context covered the encoder but not the trainable head;
converting the head output to NumPy therefore raised on a tensor requiring
gradients. No score or model claim was recovered from that attempt. Its Trackio
run remains preserved under
`mmbert-lora-full-s42-mb24-nolengthgroup-noharm-ctx1024-failed-u500`.

The narrow correction wrapped the complete cap-specific validation loop in
`torch.no_grad()`. Its regression test explicitly observed disabled gradient
mode in the 1,024 path. The 13-test context-length module, focused Ruff checks,
formatting, the then-current full repository check of 398 tests with 10 skipped,
and a finite CUDA smoke all passed before relaunch. No smoke input text was
retained.

The corrected run
`mmbert-lora-full-s42-mb24-nolengthgroup-noharm-ctx1024` completed all 25,083
updates and three epochs. Its cumulative packaged runtime was 51,070.33 seconds
across the interrupted and resumed campaign. The final resumed process reports
an RTX 4090 and 14.90 GB peak reserved VRAM, while earlier work ran on an L40S.
The 512 control also ran on an L40S, so these runtimes are operational
observations rather than a hardware speed comparison. Both runs use the same
seed, data/input hashes, base model,
rank-8 LoRA objective, optimizer, effective batch, no-length-group ordering,
and fixed update 17,000 comparison point. Their trainer hashes differ because
the 1,024 launch required explicit cap, validation, and provenance support; the
machine index preserves both.

Update 17,000 is the pre-registered primary context comparison. Every full row
below was rescored with an explicit training/evaluation context. TPR values are
descriptive same-panel 1% FPR coordinates; finance flags use each evaluation's
own calibration-selected component threshold. Long-code flags are clean
SWE-rebench V2 dev-test tasks out of 2,926; every row had 100% attack recall and
100% pair ordering.

| Train / eval tokens | Canonical AUROC / TPR@1% | PromptShield AUROC / TPR@1% | SEP AUROC / TPR@1% / pair order | Finance flags | Long-code clean flags |
|---|---:|---:|---:|---:|---:|
| 512 / 512 | 0.99013 / 60.15% | 0.93299 / 57.00% | 0.81941 / 44.81% / **82.53%** | **0 / 7,043** | 15 / 2,926 |
| 512 / 1,024 | 0.99002 / 59.44% | **0.94486 / 57.11%** | 0.81941 / 44.81% / **82.53%** | 2 / 7,043 | 6 / 2,926 |
| 1,024 / 512 | 0.99154 / 71.83% | 0.93062 / 51.76% | 0.77051 / **49.83%** / 76.78% | 1 / 7,043 | 3 / 2,926 |
| 1,024 / 1,024 | **0.99155 / 71.86%** | 0.94419 / 52.85% | 0.77051 / **49.83%** / 76.78% | 2 / 7,043 | **2 / 2,926** |

Training at 1,024 is the material lever. At native context, it gains 11.71
points canonical TPR@1%, 5.01 points SEP TPR@1%, and reduces long-code clean
flags from 15 to 2. It loses 4.15 points PromptShield TPR@1% and 5.75 points SEP
pair ordering, while adding two finance flags. Changing inference context alone
is modest and mixed: 512-trained PromptShield changes by +0.11 points and
canonical by -0.71 points at 1,024 evaluation; for the 1,024-trained checkpoint,
PromptShield changes by +1.09 points and canonical by +0.03 points.

The native frozen reserve also favors neither model uniformly. The 512 model
recalls 955 of 1,122 source-attested rows (85.12%), flags 413 of 3,990
bare-harmful controls (10.35%), and truncates 543 rows. The 1,024 model recalls
966 (86.10%), flags 414 (10.38%), and truncates only 65. The extra context
therefore raises attested recall by 0.98 points with essentially unchanged
off-target rate and materially less truncation.

The historical packager selected update 18,500 for the 1,024 run because its
recorded equal-domain selection loss, 0.0588949, narrowly beat update 17,000 at
0.0593469. That 0.0004519 improvement hides opposed components: PromptShield
BCE improves by 0.0040309, while Morgott source-macro BCE worsens by 0.0031271,
row-micro BCE worsens from 0.08668 to 0.16742, and worst-source BCE worsens from
0.47408 to 0.75890.

| Native 1,024 checkpoint | Canonical AUROC / TPR@1% | PromptShield AUROC / TPR@1% | SEP AUROC / TPR@1% / pair order | Finance flags | Reserve attested / bare-harmful | Long-code clean flags |
|---|---:|---:|---:|---:|---:|---:|
| Update 17,000, pre-registered | **0.99155 / 71.86%** | 0.94419 / 52.85% | **0.77051** / 49.83% / **76.78%** | 2 / 7,043 | 86.10% / **10.38%** | **2 / 2,926** |
| Update 18,500, packaged selector | 0.99010 / 66.91% | **0.96146 / 58.53%** | 0.76173 / **49.93%** / 75.38% | 2 / 7,043 | **87.52%** / 13.21% | 3 / 2,926 |

Update 18,500 gains 5.67 points PromptShield TPR@1%, 0.11 points SEP TPR@1%,
and 1.43 points reserve attested recall. It loses 4.94 points canonical TPR@1%,
1.40 points pair ordering, adds 2.83 points bare-harmful off-target flags, and
adds one long-code clean flag. Its 512-token rescore is also complete: 66.98%
canonical, 56.98% PromptShield, and 49.93% SEP TPR@1%, two finance flags, and
five long-code clean flags. Evaluation context does not reverse the checkpoint
trade-off.

Decision: retain update 17,000 as the primary fixed-update 1,024 context result
and update 18,500 as the packaged-selector diagnostic. The study supports
continuing the 1,024-token research direction, while retaining 512 as the
PromptShield and SEP-pair counterpoint. It does not establish that update
18,500 is best overall, promote either checkpoint, or authorize blocking. A
2,048-token training campaign requires a new tail and numerical gate plus a
frozen checkpoint-selection protocol before launch.

Exact paths, hashes, identities, metrics, deltas, superseded implicit-cap
records, and the 512/1,024 matrix are in
[`mmbert-context-comparison.json`](mmbert-context-comparison.json). The exact
campaign source and launchers are preserved in
`reports/provenance/mmbert-context-campaign-source-20260812.tar.gz`, SHA-256
`7326148fd92f2486afb908ae73f90c2ecb212d0c6bd68f8ef06fd0d6494dca11`;
every embedded checksum was reverified before consolidation.
Because the RunPod volume is disposable, Git also retains byte-exact,
checksum-verified copies of all 21 aggregate result JSONs under
`reports/provenance/mmbert-context-results-20260812/`, the three evaluated
snapshots through Git LFS under
`reports/provenance/mmbert-context-checkpoints-20260812/`, and both curated and
pre-curation Trackio databases through Git LFS under
`reports/provenance/trackio-20260812/`.
At consolidation time the snapshots remained advisory and unregistered; the
update-17,000 registration decision below occurred later.
Per-row score journals are reproducible caches rather than the only copy of any
finding.

### Update 17,000 registered for the advisory preview (2026-08-14)

`mmbert-lora-full-ctx1024-u17000-s42` is the sole model registered for
maintained inference.
The registration packages the checksum-verified update-17,000 snapshot into
safe head and PEFT adapter files, exports one FP32 ONNX graph, and binds the
native 1,024-token evaluation plus the ONNX Runtime/OpenVINO serving-equivalence
evidence recorded in `model-artifacts.json`.

Licensing review permits private internal use only: the base encoder is MIT,
while mixed-corpus redistribution remains unresolved, so neither the derived
weights nor the image may be pushed to a public or external model registry.
The privacy review found no explicit corpus rows, prompts, provider responses,
or credentials in the packaged head, adapter, tokenizer, ONNX, or evidence
files.
The reproducibility review binds the archived snapshot digest, pinned base-model
revision, tokenizer, source result and evaluation, materialized safe artifacts,
and serving evidence before load.
The model remains advisory, every assessment returns `decision: allow`, and
registration does not promote it to a blocking control.

### LP-FT arm rejected again

`mmbert-base-full-lpft-s42-top22-mb24` (top-22 layers unfrozen from the frozen
head) is best-in-class in distribution and collapses on transfer: 75.3% canonical
at 1% FPR against 3.0% PromptShield and 3.3% SEP; at its own threshold, canonical
recall 66.85% at 0.60% FPR but PromptShield 6.60% and SEP 0.73%. Consistent with
the 2026-08-05 LP-FT rejection. Stays outside `model-artifacts.json`.

### Pinned guard baselines on identical rows

Open-panel columns use the threshold-free descriptive 1% FPR coordinate.
Reserve columns use each baseline's recorded native operating point or, where
there is no native cutoff, its shared calibration threshold. Reserve slices
follow `by_subversion_basis`, never the pooled aggregate, so those columns are
not on one common threshold scale.

| System | Canonical | PromptShield | SEP | Reserve attested | Reserve bare-harmful |
|---|---:|---:|---:|---:|---:|
| mmbert-lora-full-s42 (registered at the time) | 55.2% | 48.0% | 38.8% | 43.0% | 9.3% |
| Full pipeline (encoder + selective review) | n/a | n/a | n/a | 91.7% | 22.9% |
| Llama Prompt Guard 2 86M | 43.1% | 15.7% | 3.3% | 80.0% | 2.6% |
| ModernGuard-1 | 0.0% | 0.1% | 2.4% | 99.2% | 81.3% |
| Qwen3Guard-Stream-4B, query head | 1.7% | 5.9% | 0.5% | 98.8% | 95.7% |
| Qwen3Guard-Stream-4B, jailbreak head | 37.7% | 9.2% | 8.0% | 0.0% | 0.1% |
| Kanana Safeguard Prompt 2.1B | 37.7% | 1.7% | 3.0% | 84.0% | 10.8% |

Reading rules. ModernGuard and the Qwen3Guard query head score near-perfectly on
attested reserve attacks while flagging 81.3% and 95.7% of bare-harmful rows and
10.9% to 13.6% of canonical benign rows at native cutoff: they approximate
flag-everything, and their low-FPR coordinates are near the floor because their
scores saturate at 1.0. The jailbreak head is the opposite degenerate case. Only
the then-registered shadow and Prompt Guard 2 separate the two reserve classes, and
Prompt Guard 2 is the more selective of the two on the off-target class.

The reserve also reproduces the open-panel gap: the then-registered shadow recalls
43.05% of attested reserve rows against 86.57% on canonical. The harness records
this as descriptive, since the populations differ in composition as well as in
publication status.

The then-registered row above previously carried the Aug 7 length-grouped arm's
70.2% / 40.2% / 59.5% values under the registered model's name. The corrected
55.2% / 48.0% / 38.8% values come from the then-current registered rescore at
`artifacts/comparisons/mmbert-lora-full-s42-rescore/evaluation.json`.

Prompt Guard's current-panel artifact is
`artifacts/comparisons/prompt-guard-2-86m-current-panel/evaluation.json`, pinned
at revision `a8ded8e697ce7c355e395a0df51f94adb4a2fd27`. It scores 303,645 canonical
rows at 43.116% TPR@1%, rather than silently reusing the historical
`prompt-guard-2-86m-full-mixture` artifact's 303,376-row 42.527% result. The
PromptShield coordinate is unchanged at 15.742%; SEP moves from 3.352% in the
historical artifact to 3.308% on the current panel. The 80.0% / 2.6% reserve
figures are from this same current-panel run at its native 0.5 cutoff.

Both Qwen3Guard Stream 4B readouts are complete on all current panels and the
reserve at exact revision `27a8f4e52e66dc01a03d20f41e362bb9c9bda7bf`:
the generic query risk score and the separate Jailbreak category score. They
are different projections of one checkpoint, not two independently trained
models, and neither is a promotion candidate.

Kanana Safeguard Prompt 2.1B is complete at revision
`167d74d4706b236580b0e48318337c7ac6ba7848`. Its pooled unsafe mass uses the
shared calibration threshold 0.9997723699 because the documented three-way
first-token argmax has no equivalent scalar cutoff. Canonical AUROC / TPR@1%
is 0.91054 / 37.73%; PromptShield is 0.73882 / 1.70%; SEP is 0.66168 / 3.02%,
with 69.63% pair ordering. It flags zero of 7,043 real-finance controls. On the
positive-only reserve it recalls 942 of 1,122 source-attested rows (83.96%),
flags 432 of 3,990 bare-harmful rows (10.83%), and flags 1,374 of 5,112 rows
pooled (26.88%).

The finalized evidence source is
`artifacts/comparisons/kanana-safeguard-prompt-2.1b/evaluation.json`. The
current invocation scored 414,507 rows in 3,719.51 seconds after resuming
52,224 completed calibration rows whose earlier shard timing is not retained;
the reserve portion took 54.68 seconds. The compact `morgott`
Trackio project is visualization only. Kanana is not promoted: despite clean
finance behavior, it is weaker than the compact no-harm candidate on every
open-panel 1% recall coordinate, does not improve the reserve target/off-target
trade, and costs a 2.1-billion-parameter guard to obtain that result.

### LLM reviewers on the reserve, bounded owner-approved run

Shared binary prompt contract, valid-output basis. DeepSeek V4 Flash 0731 scored
27.2% pooled; on attested rows the every-input configuration reached 88.8%.
Alibaba Qwen 3.7 Flash reached 93.0% attested and 23.0% bare-harmful but only on
the 93.3% of rows its provider answered after a full retry pass, with failures
correlated to content, so its rows are not identical to the other systems'.
GPT-OSS Safeguard 20B reached 22.9% attested at 97.1% coverage.

### Guard-baseline status and deferred work

Granite Guardian 4.1 8B is terminally recorded as `extraction_unavailable`, not
as a zero score. At pinned revision
`69820a3f3c8f265e2fe61b5a8fcea2146c2fcb16`, the documented `risk_name` setting
does not change the rendered prompt, the 4.1 template carries no guardian
logic, and the card's regex-parsed generated answer does not expose the
continuous first-position probability required by this ROC harness. The
fail-closed record is
`artifacts/comparisons/granite-guardian-4.1-8b/evaluation.json`; rerunning the
same revision cannot create a comparable scalar.

A Granite Guardian 3.2 3B A800M canary completed 4,096 length-stratified rows in
16 minutes 28 seconds of queue wall time, but the harness's stdout suppression
also swallowed the final JSON. The 90-minute projection predicate therefore
failed closed and no full panel ran. This is a throughput-harness failure, not a
quality result, so Granite 3.2 has no comparison row.

A separate hash-ranked 4,096-row canary scored the current ProtectAI DeBERTa-v3
prompt-injection v2 checkpoint at pinned revision
`90c9989b1a342275dd0d1a95aad283c04e075671`. BF16 512-token scoring at batch 32
took 52.53 seconds after 53.26 seconds of loading, or 77.98 rows per second.
Its conservative full-panel projection was 116.86 minutes, so it failed the
60-minute gate and no full evaluation ran. On the canary only, native-cutoff
canonical / PromptShield / SEP AUROC was 0.96910 / 0.67992 / 0.69185, while the
same-sample descriptive TPR@1% was 65.34% / 2.05% / 3.39%. These are sampled
polarity diagnostics, not shared-threshold comparison evidence. The record is
`artifacts/comparisons/protectai-deberta-v3-prompt-injection-v2-canary-4096/evaluation.json`,
with an exact tracked copy at
`reports/provenance/protectai-v2-canary-20260812.json`, SHA-256
`e43bae32ad230804477e3881214d19aebb2f28973301089cd44d41344cd3a585`.
The companion
`reports/provenance/protectai-v2-canary-20260812.md` discloses that the exact
producing canary/adapter source bytes were not retained; the JSON binds their
hashes, while the current reusable harness is corrected but not byte-identical.

Full-panel Granite 3.2 and ProtectAI v2 scoring is explicitly deferred future
baseline work under the current runtime gates. Granite must first be rerun
through the artifact-writing canary and establish a valid runtime projection;
ProtectAI requires a deliberately longer gate or
a faster evaluation path. Their absent full-panel rows are deliberate, do not
imply zero scores, and do not block the completed context decision.

AprielGuard remains unscored because access was unavailable and no speculative
result is useful. It is not a required blocker for this ladder; no result should
be inferred from the absence of an artifact.

## Retrieval-assisted reviewer selection, 2026-08-19

Retrieval-assisted DeepSeek review improved the advisory cascade enough to justify a maintained candidate, but it did not authorize blocking.
The consolidated evidence and exact qualifications are in `reports/retrieval-assisted-reviewer-findings-20260819.md`.

PPLX Embed V1 4B at 256 dimensions remained the best tested embedding configuration.
The prospective WMT comparison favored the source-lineage bank over the all-row bank by 4.198 recall points, with a paired 95% interval of `[1.358, 7.037]` points and no FPR difference.
The lineage bank is therefore the implementation candidate, while the completed all-row HNSW work remains scale and mechanics evidence.

The fixed full-row HNSW `efSearch=1024`, top-160, exact-rescore configuration reached 99.409% mean set Recall@20 and matched exact NumPy downstream recall and FPR.
Its persistent provider-free runtime measured 55.476 ms p95 at four workers and about 1.17 GiB process RSS, without co-resident mmBERT, live query embedding, or DeepSeek.

The full-row HNSW plus partitioned Unicode BM25 and 2:1 RRF arm moved recall from 93.182% to 93.636% and FPR from 0.249% to 0.124%.
The paired intervals included no change, so this is favorable exploratory evidence rather than a statistically established hybrid gain.
The owner selected the hybrid as an advisory defense-in-depth candidate, with dense-only behavior on sparse failure and the existing no-example reviewer on dense failure.

The maintained source-lineage bank now uses an exact provider-egress license allowlist and excludes ambiguous mixed-license rows.
Every retained document vector was reused by verified identity, so the migration required no document-embedding calls.
Its fresh HNSW run reached 99.818% mean Recall@20, reproduced all exact-dense selected packets, and reduced four-worker dense search p95 from 37.791 ms to 11.481 ms.
Fresh baseline, dense, and hybrid reviewer evidence moved recall from 71.818% without retrieval to 94.091% for both retrieval arms, while FPR moved from 0.373% to 0.249%.
Dense and hybrid tied on aggregate quality, so the owner-selected hybrid remains a defense-in-depth choice rather than a demonstrated incremental gain.
The rebuilt source-lineage HNSW bundle, sparse sidecar, exact fallback tests, and reviewed hybrid packet parity are complete and registry-bound.
The retained provider-free resource canary is bound to an earlier manifest revision, so it is not treated as registered-bundle evidence.
The rebuilt bundle passed exact zero-traffic validation on 2026-08-20 and was then owner-promoted to 100% preview traffic for POC use, with the previous revision retained at 0% for rollback.
That rollout is not a latency-gate pass and does not replace genuinely fresh source-and-time-heldout quality evidence.
ColBERT, learned sparse retrieval, raw-attention token selection, Qdrant, GraphRAG, and output-verification machinery are deferred.

#### Azure zero-traffic deployment canary on 2026-08-19

The candidate remained at 0% traffic and was not promoted.
The first single-probe AB/BA canary passed the latency gate, while a later run of the same protocol failed and rolled back before traffic moved.
The contradictory results make latency inconclusive and keep Azure promotion blocked pending a larger predeclared multi-probe paired protocol.
The complete measurements and qualifications are in [retrieval-assisted-reviewer-findings-20260819.md](retrieval-assisted-reviewer-findings-20260819.md).
Machine-readable evidence is in [azure-preview-retrieval-canary-20260819T174113Z.json](azure-preview-retrieval-canary-20260819T174113Z.json).
