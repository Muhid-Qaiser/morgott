# OpenRouter downstream evaluation

Date: 2026-07-29; updated 2026-08-02

Status update, 2026-08-19: this report preserves the earlier cascade selection as historical evidence.
The maintained advisory default is now the registry-bound `balanced-retrieval-20260819` profile documented in [the retrieval findings](retrieval-assisted-reviewer-findings-20260819.md).

## Decision

The evaluated LLMs should not be used as unconditional vetoes after a high-recall mmBERT first stage.

At an exploratory first-stage target near 99% recall, a pure two-zone cascade falls to about 70% final recall with DeepSeek V4 Flash, 51% with Qwen 3.7 Flash, and 14% with GPT-OSS Safeguard.
The LLM clears thousands of true attacks that mmBERT correctly escalated.
This directly conflicts with the recall-first goal.

DeepSeek V4 Flash with reasoning disabled is the best LLM tested for recall, speed, and cost.
Qwen 3.7 Flash with default reasoning is the higher-precision option, but its recall is much lower and its reasoning tokens make it slower and more expensive.
GPT-OSS Safeguard 20B is not viable for this corpus because its valid-output instruction-subversion recall is 12.75% overall, 6.63% on PromptShield, and 0% on SEP.
This conclusion is specific to the shared binary prompt contract and is not a claim about every possible Safeguard policy format.

DeepSeek V4 Pro is not a better recall-first replacement.
Strict structured output fixes its high-reasoning serialization failures, but Pro still trails Flash on recall and costs much more.
Changing from fp4 to fp8 does not materially change Flash or Pro high-reasoning classification on the paired sample.

The maintained shadow recommendation after the channel-aware follow-up is:

1. Use the full-mixture rank-8 LoRA only as the local advisory research shadow for this selected development route.
2. Pass direct-user scores below `0.2` and untrusted-content scores below `0.1`, restrict scores at or above `0.99999`, and send only the middle zone to DeepSeek V4 Flash.
3. Pass the trusted runtime input channel to DeepSeek and restrict at decision-token probability `p >= 0.7310585786`; otherwise pass.
4. Retry transient or invalid provider results under the bounded maintained contract, and restrict after retries are exhausted.
5. Keep every learned output advisory and privilege-reducing; it never grants authority or changes `morgott scan`.
6. Treat this as already-open development evidence until a prospective traffic-like evaluation confirms the gain.

## Evaluation contract

The fixed panel has 20,000 rows:

| Dataset | Rows | Negative | Positive | Input channels |
|---|---:|---:|---:|---|
| Canonical grouped dev-test sample | 10,000 | 5,322 | 4,678 | 8,834 direct user, 1,166 untrusted content |
| PromptShield test | 5,000 | 3,621 | 1,379 | direct user |
| SEP paired sample | 5,000 | 2,500 | 2,500 | untrusted content |

The canonical 10,000 are a deterministic source, channel, and label-stratified sample of the established 303,376-row narrow evaluator.
The SEP sample contains 2,500 complete clean and instructed pairs.
The panel SHA-256 is `45e798955418e8990d5f56b5d1ae542ae3499f8d585a0ea2c4fe6712b4d06e14`.

Every primary LLM evaluated every panel row, regardless of either mmBERT score.
Each LLM received exactly the raw row text and the same system contract.
No source, channel, dataset identity, label, or mmBERT score was shown to the LLM.
No row was redacted, rewritten, dropped, or filtered before transmission.

The primary response contract was:

```json
{"subversion": true, "harmful_request": false}
```

Both values were binary JSON booleans.
Malformed output, refusal, timeout, HTTP failure, and provider failure were recorded separately.
Operational failures route to review in fail-safe and cascade metrics.
Raw provider responses and panel text were not persisted in experiment outputs.

The selected primary configurations were:

| Configuration | OpenRouter model | Provider | Reasoning | Max completion |
|---|---|---|---|---:|
| DeepSeek Flash off | `deepseek/deepseek-v4-flash-20260423` | DeepInfra fp4 | disabled | 64 |
| Qwen default | `qwen/qwen3.7-flash-20260727` | Alibaba | provider default, reasoning excluded from response | 1,024 |
| Safeguard default | `openai/gpt-oss-safeguard-20b` | Groq | provider default, reasoning excluded from response | 1,024 |

All calls used temperature 0.
The seed was 42 where supported.
Provider fallback was disabled.

## Full 20,000-row LLM results

The following classifier metrics use valid outputs only.
The failure-safe metrics below them treat every operational failure as review-positive.

| Model | Valid outputs | Recall | FPR | Precision | Accuracy | Mean latency | P95 latency | Observed cost per 1,000 inputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek V4 Flash, reasoning off | 99.84% | 70.11% | 4.44% | 92.18% | 84.68% | 1.69 s | 3.75 s | $0.0328 |
| Qwen 3.7 Flash, default reasoning | 99.86% | 51.08% | 1.31% | 96.68% | 78.32% | 5.23 s | 11.02 s | $0.1034 |
| GPT-OSS Safeguard 20B, default reasoning | 99.37% | 12.75% | 0.06% | 99.36% | 62.74% | 1.64 s | 10.58 s | $0.0687 |

Failure-safe routing changes DeepSeek to 70.18% recall and 4.54% FPR.
It changes Qwen to 51.15% recall and 1.45% FPR.
It changes Safeguard to 13.54% recall and 0.49% FPR.

DeepSeek produced 19,968 valid outputs, 28 malformed verdicts, and four non-stop completions.
Qwen produced 19,972 valid outputs, 19 malformed verdicts, eight provider API errors, and one timeout.
Safeguard produced 19,873 valid outputs, 89 provider API errors, 37 exhausted HTTP failures, and one non-stop completion.

### Dataset transfer

These are valid-output recall and FPR:

| Model | Canonical | PromptShield | SEP |
|---|---:|---:|---:|
| DeepSeek V4 Flash off | 76.47% / 2.82% | 79.62% / 9.70% | 53.00% / 0.28% |
| Qwen default | 59.05% / 1.28% | 45.39% / 2.27% | 39.34% / 0.00% |
| Safeguard default | 21.47% / 0.04% | 6.63% / 0.14% | 0.00% / 0.00% |

DeepSeek has the strongest transfer recall, but its PromptShield FPR is high.
Qwen is more conservative and misses more attacks.
Safeguard nearly always returns negative outside the easiest canonical positives.

## Existing mmBERT models

Both existing models were evaluated on exactly the same 20,000 rows using their already-aligned historical score arrays.
The unfinished full-data LoRA was excluded.

| Encoder | AUROC | Average precision | Recall at 0.5 | FPR at 0.5 | Precision at 0.5 |
|---|---:|---:|---:|---:|---:|
| Frozen full-data mmBERT s42 | 0.9067 | 0.8805 | 74.34% | 13.87% | 80.03% |
| Partial-data LoRA mmBERT s42 | 0.9194 | 0.9049 | 77.80% | 10.28% | 84.99% |

The partial-data LoRA is better on aggregate ranking and at the untouched 0.5 cutoff.
It also transfers better at 0.5 to PromptShield and SEP.

The extreme high-recall tail gives a different result.
The maximum threshold reaching at least 99% recall on this same open panel is `0.0014103585` for frozen full-data mmBERT and `0.0003920418` for partial-data LoRA.
Those thresholds are exploratory and are not independent calibration evidence.

| Encoder | Actual first-stage recall | First-stage row FPR | Escalation rate |
|---|---:|---:|---:|
| Frozen full-data mmBERT | 99.04% | 53.43% | 72.95% |
| Partial-data LoRA mmBERT | 99.08% | 73.57% | 84.48% |

Frozen full-data mmBERT is more efficient as an extreme-recall first-stage router on this panel.
Partial-data LoRA is stronger at ordinary operating points and supplies a much larger retained high-confidence zone.
At that stage, the encoder choice remained unresolved until the full-data LoRA could be completed and evaluated under the same contract.

## Pure two-zone cascade

The simulated two-zone policy is:

```text
mmBERT score below low threshold -> pass
mmBERT score at or above low threshold -> use the LLM binary verdict
LLM operational failure -> review
```

The low boundary below is the exploratory panel threshold that reaches about 99% first-stage recall.

| Encoder and LLM | LLM call rate | Final recall | Final FPR | Final precision | Attacks below mmBERT boundary | Escalated attacks cleared by LLM |
|---|---:|---:|---:|---:|---:|---:|
| Frozen + DeepSeek | 72.95% | 70.15% | 4.46% | 92.17% | 82 | 2,472 |
| Frozen + Qwen | 72.95% | 51.12% | 1.41% | 96.45% | 82 | 4,101 |
| Frozen + Safeguard | 72.95% | 13.54% | 0.34% | 96.74% | 82 | 7,316 |
| Partial LoRA + DeepSeek | 84.48% | 70.16% | 4.49% | 92.11% | 79 | 2,474 |
| Partial LoRA + Qwen | 84.48% | 51.14% | 1.42% | 96.43% | 79 | 4,102 |
| Partial LoRA + Safeguard | 84.48% | 13.53% | 0.43% | 95.94% | 79 | 7,320 |

This architecture does not preserve the first stage's recall.
At a sufficiently low mmBERT threshold, final recall converges to the LLM's standalone recall because the LLM can clear every escalation.
No threshold adjustment can repair that structural property.

## Three-zone diagnostic

The non-tuned three-zone diagnostic is:

```text
score below exploratory low boundary -> pass
score between low boundary and existing canonical-validation boundary -> LLM
score at or above canonical-validation boundary -> retain review regardless of LLM
```

The high boundary is `0.9999771720` for frozen full-data mmBERT and `0.9219218349` for partial-data LoRA.
It was selected previously on canonical validation for a 1% component-level FPR target.
Its row-level behavior on this mixed panel is not a production guarantee.

| Encoder and LLM | Pass / LLM / retained-review zones | Final recall | Final FPR | Final precision |
|---|---:|---:|---:|---:|
| Frozen + DeepSeek | 27.05% / 63.98% / 8.97% | 74.69% | 4.55% | 92.46% |
| Frozen + Qwen | 27.05% / 63.98% / 8.97% | 60.29% | 1.50% | 96.77% |
| Frozen + Safeguard | 27.05% / 63.98% / 8.97% | 28.49% | 0.45% | 97.91% |
| Partial LoRA + DeepSeek | 15.52% / 52.99% / 31.50% | 82.33% | 6.83% | 90.02% |
| Partial LoRA + Qwen | 15.52% / 52.99% / 31.50% | 77.31% | 4.81% | 92.32% |
| Partial LoRA + Safeguard | 15.52% / 52.99% / 31.50% | 68.68% | 4.89% | 91.31% |

The retained high zone helps, especially with partial-data LoRA, but the result still falls far short of the 99% first-stage recall target.
This is a useful diagnostic, not a promoted policy.

## Reasoning ablation

The reasoning gate used the same fixed 200 rows for every configuration.
The sample contains 82 positives and 118 negatives.

| Model configuration | Valid | Recall | FPR | Accuracy | Mean latency | Cost for 200 |
|---|---:|---:|---:|---:|---:|---:|
| Safeguard default | 195 | 18.75% | 0.00% | 66.67% | 1.21 s | $0.0135 |
| Safeguard low | 190 | 12.99% | 0.00% | 64.74% | 1.05 s | $0.0128 |
| Qwen default | 200 | 50.00% | 0.85% | 79.00% | 6.34 s | $0.0195 |
| Qwen reasoning off | 200 | 40.24% | 0.85% | 75.00% | 2.48 s | $0.0027 |
| DeepSeek Flash reasoning off | 199 | 69.51% | 4.27% | 84.92% | 2.08 s | $0.0055 |
| DeepSeek Flash high reasoning | 171 | 60.00% | 3.77% | 82.46% | 6.19 s | $0.0099 |

All eight Qwen prediction changes from reasoning-off to default corrected false negatives without adding a false positive.
DeepSeek reasoning-off uniquely corrected ten paired disagreements while high reasoning uniquely corrected two.
DeepSeek high reasoning also produced 29 malformed verdicts.
Safeguard default uniquely corrected four paired disagreements while low reasoning uniquely corrected one.

The selected reasoning settings are therefore Qwen default, DeepSeek off, and Safeguard default.

## DeepSeek Pro and fp8 samples

The Pro and quantization comparisons used the same 200 audit rows.
These samples are too small to support claims about small differences.

| Configuration | Valid | Recall | FPR | Accuracy | Mean latency | Cost for 200 |
|---|---:|---:|---:|---:|---:|---:|
| Flash fp4 off, DeepInfra | 199 | 69.51% | 4.27% | 84.92% | 2.08 s | $0.0055 |
| Flash fp8 off, Alibaba | 200 | 68.29% | 4.24% | 84.50% | 1.26 s | $0.0096 |
| Pro fp4 off, DeepInfra | 198 | 46.91% | 1.71% | 77.27% | 3.55 s | $0.0668 |
| Pro fp8 off, Alibaba | 200 | 51.22% | 0.85% | 79.50% | 1.68 s | $0.0923 |
| Pro fp4 high strict, DeepInfra | 198 | 62.50% | 1.69% | 83.84% | 6.22 s | $0.1784 |
| Pro fp8 high strict, Alibaba | 200 | 62.20% | 1.69% | 83.50% | 4.59 s | $0.2184 |

Flash fp4 and fp8 differ on only four paired valid rows, split two unique wins each.
Pro high strict fp4 and fp8 differ on six paired valid rows, split three unique wins each.
The apparent latency differences are confounded by provider choice.

Pro reasoning-off improves somewhat at fp8, but it still has much lower recall than Flash.
Pro high strict has lower FPR than Flash, but also lower recall, slightly lower accuracy, much higher cost, and much higher latency.
For this recall-first layer, Pro is not promoted.

Ordinary JSON mode made only 132 of 200 Pro high-reasoning outputs valid.
Strict structured output repaired this to 198 of 200 on DeepInfra and 200 of 200 on Alibaba.
The remaining failures were one HTTP failure and one length-limited completion on DeepInfra.

## Harmful-request axis

PromptShield and SEP have no matching harmful-request labels, so harmful metrics are masked to 1,052 canonical `harmful_intent` positives and 4,742 canonical `benign` negatives.

| Model | Harmful recall | Harmful FPR | Harmful precision |
|---|---:|---:|---:|
| DeepSeek Flash off | 55.95% | 1.14% | 91.46% |
| Qwen default | 53.71% | 0.76% | 94.01% |
| Safeguard default | 48.91% | 0.47% | 95.74% |

The 580 source-supported harmful-non-injection rows remain a separate diagnostic:

| Model | Valid rows | Incorrectly called subversion | Called harmful request |
|---|---:|---:|---:|
| DeepSeek Flash off | 580 | 3.45% | 48.79% |
| Qwen default | 580 | 0.69% | 45.69% |
| Safeguard default | 567 | 0.18% | 41.27% |

The two axes are not collapsed.
However, fewer than half of the harmful-non-injection rows are recognized as harmful by any model.
This output is not a general moderation system.

The prompt-interference audit compares the two-field prompt with an otherwise identical subversion-only prompt:

| Model | Paired valid rows | Subversion disagreements | Two-field prompt uniquely correct | Subversion-only prompt uniquely correct |
|---|---:|---:|---:|---:|
| DeepSeek Flash off | 199 | 10 | 5 | 5 |
| Qwen default | 197 | 8 | 2 | 6 |
| Safeguard default | 198 | 2 | 0 | 2 |

The harmful field does measurably alter subversion decisions.
The effect is neutral for DeepSeek on this sample and slightly harmful for Qwen and Safeguard.
The next routing experiment should therefore use a subversion-only primary contract and evaluate harmfulness in a separate call or separate downstream stage.

## Repeatability

The first primary call was reused as repetition one.
Two additional calls used the same prompt, text, seed where supported, temperature, model, and provider.

| Model | Rows with three valid outputs | Subversion unanimous | Harmful unanimous |
|---|---:|---:|---:|
| DeepSeek Flash off | 199 | 96.98% | 100.00% |
| Qwen default | 197 | 96.95% | 95.43% |
| Safeguard default | 198 | 98.48% | 99.49% |

Temperature 0 does not make the result deterministic.
Binary labels are reasonably stable on this sample, but the observed 1.5% to 3.0% subversion flip rate is material near any automatic policy boundary.

## Cost and execution accounting

The clean in-scope study contains 64,200 calls:
These are logical model-row jobs, and transient HTTP retries inside a job are recorded separately in its `attempts` field.

| Phase | Calls | Observed cost |
|---|---:|---:|
| Three selected models on all 20,000 rows | 60,000 | $4.1000 |
| Repeatability and prompt-interference audits | 1,800 | $0.1050 |
| Reasoning, Pro, and fp8 paired samples | 2,400 | $0.8019 |
| Total | 64,200 | $5.0069 |

The earlier 330-call concurrency canary cost $0.0100 and is archived separately.
It is excluded from all model metrics.

There were 405 non-billable setup attempts excluded from model metrics.
Four hundred were 404 responses from an OpenRouter Pro alias route, two were 404 responses from its dated native route, and three were exhausted 429 responses from Baidu fp8 canaries.
All are retained in separate operational ledgers.

## Interpretation and next experiment

The binary LLM prompt is precision-biased.
That is why it reduces false positives when allowed to veto mmBERT, but also destroys recall.
The LLM is not merely refining mistakes.
It is reclassifying the escalated set with a stricter boundary.

The next useful experiment is not another threshold sweep over these same binary outputs.
It should test one of these narrower hypotheses:

1. A subversion-only, explicitly recall-biased LLM contract can preserve substantially more escalated attacks.
2. A three-zone design can reserve high-confidence mmBERT rows for review and use the LLM only in a genuinely ambiguous middle zone.
3. A stronger model can beat DeepSeek Flash's recall at an acceptable operational cost.
4. A calibrated fusion model over mmBERT score and one or more LLM features can improve the frontier without allowing the LLM to veto every escalation.

The first follow-up should use the subversion-only prompt because the current audit already shows harmful-field interference.
It should keep the same 20,000-row panel for comparability, but model or prompt selection must not turn this open panel into a final test.
Any production threshold still requires a separate calibration set and representative traffic.

## Log-probability three-zone follow-up (2026-07-29)

This follow-up executes the first two hypotheses above with the same fixed 20,000-row panel.
A deterministic group-aware split assigned 6,000 rows to calibration and 14,000 rows to evaluation, with no group overlap and complete SEP pairs.
The split contains 10,000 canonical rows, 5,000 PromptShield rows, and 5,000 SEP rows.
Every model received the same stored text unchanged and received neither input-channel metadata, source identity, labels, mmBERT scores, nor task context.
The contract is subversion-only and recall-biased, with harmful non-injection explicitly negative.

DeepSeek V4 Flash used CoreWeave fp8, reasoning disabled, strict integer JSON, `max_tokens=16`, `temperature=0`, and top-20 token log probabilities.
Qwen 3.7 Flash used Alibaba, default reasoning, JSON-object mode, `max_tokens=1024`, `temperature=0`, and the endpoint maximum of five token alternatives.
The evaluator stores both raw `logprob(1) - logprob(0)` and its numerically stable sigmoid.
The selected runtime interface uses the sigmoid value as `p_subversion`, while raw log odds remain available for diagnostics.
The two values have identical rankings, so this representation change does not alter any selected row or reported metric.
Missing log probabilities, malformed output, and exhausted transport errors route to review.

### Standalone LLM results

| Configuration and split | Valid score coverage | ROC AUC / PR AUC on valid rows | Recall / FPR at zero log odds with failures routed to review | Mean / p95 latency | Observed cost |
|---|---:|---:|---:|---:|---:|
| DeepSeek CoreWeave fp8, calibration | 99.97% | 0.9343 / 0.9114 | 68.10% / 3.58% | 3.00 s / 8.09 s | $0.3176 |
| DeepSeek CoreWeave fp8, evaluation | 99.95% | 0.9393 / 0.9244 | 71.15% / 3.32% | 2.97 s / 7.96 s | $0.7523 |
| Qwen default reasoning, calibration | 98.95% | 0.7729 / 0.7848 | 56.45% / 2.97% | 4.15 s / 10.55 s | $0.3301 |
| DeepSeek StreamLake fp8, calibration diagnostic | 68.22% | 0.9233 / 0.9037 | 77.44% / 39.62% | 2.12 s / 6.10 s | $0.2037 |

CoreWeave returned 19,991 valid scores and nine exhausted HTTP 429 failures over 20,000 rows.
Qwen returned 5,937 valid scores, 36 invalid verdicts, 26 exhausted HTTP 429 failures, and one provider API error over 6,000 calibration rows.
Qwen cleared 1.86 percentage points fewer benign rows with the frozen encoder and 2.83 points fewer with the partial LoRA than CoreWeave at the primary calibration constraint.
It therefore failed the predeclared requirement to clear at least two percentage points more benign rows, and no 14,000-row Qwen evaluation run was performed.

The StreamLake diagnostic was not a small-canary accident.
It returned only 4,093 valid scores, 1,807 rows where both binary alternatives were unavailable, 98 exhausted HTTP 429 failures, and two provider API errors over the same 6,000 calibration rows.
Among rows valid on both DeepSeek routes, binary decisions agreed 97.75% of the time, but StreamLake's valid-row ranking was also slightly worse.
Its lower price does not compensate for forcing approximately one-third of inputs to review.

### Calibration-selected three-zone result

The primary exploratory constraint is 99.5% review-tier recall with a 1% high-confidence flag FPR cap on calibration.
The selection rule first maximizes high-confidence flag true positives and then benign auto-passes.
The following numbers apply the selected thresholds once to the 14,000-row evaluation split.

| First-stage encoder | Review-tier recall | Overall auto-pass / benign auto-pass | High-confidence flag recall / FPR / precision | LLM middle zone | Manual-review zone | Projected LLM cost per 1,000 inputs |
|---|---:|---:|---:|---:|---:|---:|
| Full-data frozen mmBERT | **99.82%** | **20.84% / 36.29%** | 44.94% / 0.85% / 97.54% | 69.46% | 59.44% | $0.0439 |
| Partial-data rank-8 LoRA | 99.40% | 12.26% / 20.99% | **47.75%** / 0.81% / **97.78%** | 80.82% | 66.84% | $0.0470 |

The frozen full-data encoder is the better primary cascade at this constraint.
It transfers above the 99.5% recall anchor and clears substantially more benign rows.
The partial LoRA has slightly better high-confidence flagging, but its evaluation review recall falls to 99.40% and its review workload is higher.
The 99.5% value is an exploratory calibration anchor, not an execution gate or a production guarantee.

The frozen primary calibration thresholds are `mmbert_low=0.0002611903`, `mmbert_high=0.9998766054`, LLM clear log odds `-9.2500`, and LLM flag log odds `4.3750`.
The corresponding LLM sigmoid values are approximately `0.0000961` and `0.987568`.
These values are development candidates selected on an already-open panel and should not be copied into production.

### Rounded-threshold sensitivity

Decimal simplicity is not robustness when scores are uncalibrated and highly polarized.
A frozen-mmBERT low threshold of `0.1` retains only 87.18% of evaluation attacks before the LLM, while `0.2` retains 83.29%.
Allowing the LLM to clear middle-zone rows reduces final review recall to 81.84% and 79.20% with an LLM `0.1` clear and `0.9` flag band.
Using the LLM's direct binary decision reduces it further to 79.47% and 77.03%.
The mmBERT high threshold `0.9` also has 6.94% evaluation FPR for the frozen encoder and 4.96% for the partial LoRA before downstream combination.
Those tenths-based thresholds are therefore incompatible with the recall-first and low-FPR goals.

A small predeclared logarithmic grid is a better anti-overfitting simplification.
The following frozen and partial-LoRA candidates use `mmbert_low=0.0001` and `llm_clear_probability=0.0001`.
They were selected as simple sensitivity points rather than fitted order statistics.

| Encoder and candidate | mmBERT high | LLM flag probability | Evaluation review recall | Overall / benign auto-pass | Flag recall / FPR / precision | Manual-review zone |
|---|---:|---:|---:|---:|---:|---:|
| Full-data frozen, conservative | 0.9999 | 0.99 | 99.88% | 17.45% / 30.41% | 42.25% / 0.74% / 97.72% | 64.05% |
| Partial-data LoRA, conservative | 0.9999 | 0.99 | 99.88% | 6.23% / 10.80% | 34.07% / 0.55% / 97.89% | 78.88% |
| Full-data frozen, higher flag recall | 0.999 | 0.95 | 99.88% | 17.45% / 30.41% | 61.27% / 1.75% / 96.33% | 55.34% |
| Partial-data LoRA, higher flag recall | 0.999 | 0.95 | 99.88% | 6.23% / 10.80% | 61.27% / 1.54% / 96.76% | 66.68% |

The conservative rounded candidate gives up 3.88 percentage points of benign auto-pass and 2.69 points of flag recall relative to the calibration-optimized primary point, but it transfers with higher review recall and has easier operational semantics.
The higher-flag candidate reaches the requested 60% to 70% flag-recall range without using fitted decimal cutpoints.
The partial LoRA slightly reduces flag FPR under the higher-flag policy, but it sends 11.34 percentage points more traffic to manual review and clears 19.61 points fewer benign rows.
The rounded comparison therefore does not change the recommendation to use the full-data frozen encoder.
The full-data LoRA was still training when this rounded comparison was selected and was not represented by these partial-data LoRA results.

Do not collapse the LLM stage to its selected `0` or `1` token for automatic clearance.
On the rounded frozen setup, a zero-log-odds binary rule yields only 75% to 77% review recall and approximately 3.4% to 3.7% FPR because almost every valid middle-zone row is either passed or flagged.
Use two LLM thresholds instead: clear only below the low confidence bound, flag only above the high confidence bound, route the interval and every operational failure to review.

### No-manual-review diagnostic

Without a review queue, the cascade must emit only pass or flag.
This diagnostic sends low mmBERT scores directly to pass, high scores directly to flag, applies one DeepSeek threshold in the middle, and sends operational failures to flag.
It selects only from the coarse threshold grid on calibration, maximizes recall under each calibration FPR cap, and applies the selected point once to evaluation.
The selected `0.9` DeepSeek threshold was chosen on the 6,000-row calibration split and then evaluated on the separate 14,000 rows.
It is an empirically selected operating threshold over the normalized two-token signal, not a claim that DeepSeek outputs calibrated real-world probabilities.

| Encoder | Calibration FPR cap | mmBERT low / high | LLM flag probability | Evaluation recall / FPR / precision | LLM call rate |
|---|---:|---:|---:|---:|---:|
| Full-data frozen | 0.5% | 0.001 / 0.9999 | 0.999 | 28.95% / 0.27% / 98.75% | 63.95% |
| Partial-data LoRA | 0.5% | 0.001 / 0.9999 | 0.99 | 34.12% / 0.57% / 97.80% | 69.96% |
| Full-data frozen | 1% | 0.001 / 0.9999 | 0.99 | 42.29% / 0.76% / 97.65% | 63.95% |
| Partial-data LoRA | 1% | 0.001 / 0.9999 | 0.99 | 34.12% / 0.57% / 97.80% | 69.96% |
| Full-data frozen | 2% | 0.001 / 0.9999 | 0.95 | 57.41% / 1.29% / 97.09% | 63.95% |
| Partial-data LoRA | 2% | 0.2 / 0.999 | 0.9 | 61.82% / 1.70% / 96.46% | 26.79% |
| Full-data frozen | 5% | 0.001 / 0.999 | 0.5 | 77.03% / 3.68% / 93.99% | 59.46% |
| Partial-data LoRA | 5% | 0.0001 / 0.99 | 0.5 | 79.10% / 3.92% / 93.78% | 71.87% |

The 1% region favors frozen mmBERT because it finds materially more attacks at a similar FPR.
The approximately 2% region favors the partial LoRA when higher recall and lower LLM traffic matter more than its 0.41-point FPR increase.
At the 5% calibration cap, neither encoder reaches 80% evaluation recall.
Reaching 99.88% recall by automatically flagging every former review row produces 69.59% FPR with frozen mmBERT and 89.20% FPR with the partial LoRA.
There is therefore no credible high-recall, low-FPR, no-review operating point in this experiment.

Precision in this table reflects the artificial 42.8% positive panel prevalence.
For example, the frozen 1% setting has 97.65% panel precision, but its recall and FPR imply only approximately 35.9% precision if attacks are 1% of live traffic.
Deployment precision must be recomputed using representative prevalence.

### Completed-encoder refresh (2026-07-30)

The completed full-data rank-8 mmBERT LoRA and bounded full-data LFM2.5 frozen-head scores were joined to the exact frozen panel by retained source index and checked against all 20,000 row identities and labels.
This refresh reuses the already-stored DeepSeek CoreWeave outputs and sends no additional corpus text to a provider.
The mmBERT and LFM frozen controls trained only their classification heads, so neither is an all-weights fine-tune.
The hash-linked [machine summary](../artifacts/comparisons/deepseek-v4-flash-encoder-refresh/summary.json) has SHA-256 `a806088e0a6046fcc285c6c04961e9e925885302f8657e1f10ad3598eb99808f`.

The first table literally copies the selected partial-LoRA policy to every encoder: pass below `0.2`, flag at or above `0.999`, call DeepSeek in the middle, and flag at DeepSeek `p >= 0.9` or after an exhausted operational failure.

| First-stage encoder | Calibration FPR | Evaluation recall / FPR / panel precision | DeepSeek call rate | Projected DeepSeek cost per 1,000 inputs |
|---|---:|---:|---:|---:|
| Full-data rank-8 mmBERT LoRA | 3.87% | **78.40%** / 3.62% / 94.18% | **11.69%** | **$0.0129** |
| Partial-data rank-8 mmBERT LoRA | **1.98%** | 61.82% / **1.70%** / **96.46%** | 26.79% | $0.0256 |
| Full-data frozen mmBERT head | 2.62% | 62.39% / 2.11% / 95.67% | 31.65% | $0.0293 |
| Full-data frozen LFM2.5 head | 3.70% | 64.41% / 3.63% / 92.99% | 29.14% | $0.0265 |

The fixed full-LoRA policy is not a valid replacement for the partial-LoRA policy because its calibration FPR exceeds the 2% selection cap.
Its aggregate recall gain also hides an 8.40% PromptShield FPR.
The exact evaluation recall and FPR slices are:

| First-stage encoder | Canonical | PromptShield | SEP |
|---|---:|---:|---:|
| Full-data rank-8 mmBERT LoRA | 90.63% / 1.69% | 81.35% / 8.40% | 53.89% / 0.80% |
| Partial-data rank-8 mmBERT LoRA | 76.49% / 1.61% | 69.02% / 2.96% | 30.40% / 0.06% |
| Full-data frozen mmBERT head | 83.39% / 1.85% | 55.85% / 3.71% | 26.69% / 0.34% |
| Full-data frozen LFM2.5 head | 79.63% / 2.52% | 72.44% / 7.57% | 31.49% / 0.29% |

Raw encoder probabilities are not calibrated across adaptation recipes.
The original coarse selector was therefore rerun on the same 6,000 calibration rows and evaluated once on the retained 14,000 evaluation rows.
It maximizes calibration recall subject to the aggregate 2% calibration FPR cap over low gates `{0.0001, 0.001, 0.01, 0.1, 0.2}`, high gates `{0.99, 0.999, 0.9999}`, and DeepSeek gates `{0.5, 0.9, 0.95, 0.99, 0.999}`.

| First-stage encoder and grid | Low / high / DeepSeek threshold | Calibration FPR | Evaluation recall / FPR / panel precision | DeepSeek call rate |
|---|---:|---:|---:|---:|
| Full-data frozen mmBERT, original grid | `0.001 / 0.9999 / 0.95` | 1.69% | 57.41% / 1.29% / 97.09% | 63.95% |
| Partial-data mmBERT LoRA, original grid | `0.2 / 0.999 / 0.9` | 1.98% | 61.82% / 1.70% / 96.46% | 26.79% |
| Full-data mmBERT LoRA, original grid | no feasible point; nearest `0.0001 / 0.9999 / 0.999` | minimum 2.27% | nearest point: 64.61% / 1.92% / 96.17% | 60.14% |
| Full-data frozen LFM2.5, original grid | `0.001 / 0.9999 / 0.95` | 1.69% | 56.84% / 1.51% / 96.57% | 60.71% |
| Full-data mmBERT LoRA, post-hoc high-gate extension | `0.2 / 0.99999 / 0.9` | 1.98% | **66.79%** / 1.81% / **96.50%** | **22.17%** |

The post-hoc full-LoRA extension adds only the `0.99999` high gate, but it was not in the predeclared grid and remains already-open development evidence.
Against the retained partial-LoRA policy, it gains 4.97 percentage points of aggregate recall and reduces DeepSeek calls by 4.61 points while increasing FPR by 0.11 points.
Its canonical, PromptShield, and SEP recall and FPR pairs are 78.69% / 1.61%, 77.51% / 3.27%, and 38.63% / 0.11%, respectively.
The full-LoRA result is promising enough for prospective recalibration, but it does not justify silently replacing the registered partial-LoRA route.

### Finance-negative cascade check

The fixed full-LoRA `0.2 / 0.99999 / 0.9` cascade was subsequently applied without threshold changes to all 7,054 retained direct-user finance negatives.
The local encoder passed 6,926 rows below `0.2`, sent 128 rows to DeepSeek, and placed no row in the local high-flag zone.
All 128 CoreWeave fp8 calls completed successfully, all were passed by DeepSeek, and the maximum observed `p_subversion` was `0.4687906266`.
The complete cascade therefore produced zero false positives and 0.00% empirical FPR, with a two-sided 95% Clopper-Pearson upper bound of 0.0523%.
It called DeepSeek for 1.81% of finance inputs, cost `$0.00354592` in total, added 38.8 milliseconds of mean provider latency per input, and retained the full LoRA's improvement over the frozen and partial-LoRA controls, which produced five and four false positives respectively.
This negative-only slice measures false positives and cannot establish finance-attack recall.
The hash-linked [finance comparison artifact](../artifacts/comparisons/deepseek-v4-flash-finance-full-lora/summary.json) records the complete result without raw corpus text or raw provider response content.

Only the full-data frozen-head LFM2.5 artifact exists.
No LFM2.5 partial-data LoRA, full-data LoRA, or all-weights fine-tune was trained, and the retained mmBERT LoRA adapter contract does not match LFM2.5 attention modules.
Producing those missing numbers requires newly authorized training runs and a separate pinned LFM adapter implementation; they cannot be inferred from the frozen-head result.

### NOOA PredictStrategy comparison (2026-07-30)

The disposable NOOA `PredictStrategy` experiment replayed the frozen 6,000-row calibration and 14,000-row evaluation panel through 32 stateless agents.
It reused the maintained prompt, CoreWeave fp8 route, strict log-probability parser, and three-call retry ceiling.
The ledger contains 20,000 unique panel IDs, stores no corpus text or raw provider response, and cost `$1.84803752` for the full-panel experiment.

| Downstream path | Valid output coverage | Calibration FPR | Evaluation recall / FPR / panel precision | Projected call rate |
|---|---:|---:|---:|---:|
| Maintained `CompletionClient`, `p >= 0.9` | **99.955%** | **1.981%** | 66.79% / 1.81% / 96.50% | 22.17% |
| `PredictStrategy`, copied `p >= 0.9` | 97.12% | 2.214% | **71.22%** / 1.91% / 96.54% | 22.17% |
| `PredictStrategy`, calibration-selected `p >= 0.9706877673` | 97.12% | 1.952% | 61.89% / **1.67%** / 96.51% | 22.17% |

The copied Predict threshold gains 4.42 evaluation recall points over the maintained route, but it violates the predeclared 1.98% calibration FPR cap.
Selecting the highest-recall feasible Predict threshold loses 4.91 recall points relative to `CompletionClient`.
Its canonical, PromptShield, and SEP evaluation recall / FPR pairs are 77.34% / 1.45%, 65.70% / 3.04%, and 30.86% / 0.17%.

| Full-panel downstream measurement | Maintained `CompletionClient` | NOOA `PredictStrategy` |
|---|---:|---:|
| Successful outputs | 19,991 / 20,000 | 19,424 / 20,000 |
| Mean prompt tokens on successful outputs | **366.69** | 663.59 |
| Mean / p95 client latency | **2.98 s / 7.99 s** | 8.19 s / 13.44 s |
| Actual cost for 20,000 calls | **$1.06983170** | $1.84803752 |
| Projected cascade cost per 1,000 inputs | **$0.0192** | $0.0280 |

Across 19,415 jointly valid rows, verdict agreement was 91.17%, probability mean absolute error was `0.10287`, and probability correlation was `0.89965`.
CompletionClient versus PredictStrategy AUROC was `0.93791` versus `0.93614`, and average precision was `0.92073` versus `0.91204`, on those common rows.
PredictStrategy's concrete advantage is its typed Pydantic return and its higher copied-threshold recall.
Its typed result does not expose the decision-token alternatives needed by this cascade, so the experiment still required a captured raw completion for strict probability parsing.
That leaves more integration code while measured coverage, ranking, latency, prompt tokens, and cost are worse.

Decision: do not promote `PredictStrategy`; retain NOOA `CompletionClient` in the maintained cascade.
The PredictStrategy machine-summary SHA-256 is `48494bd95b540b8f374561af7d061196617935b40f46a8f3937632b4c6a82629`.
Its parsed append-only ledger has SHA-256 `8254cad13f48f3ce4a4b2dc52f5931b26abccb7bb9b0560bb8758aa3445bdb73`, configuration SHA-256 `c1addf24929cbd23a6918e609d7c60dac54834c0202df9a93e6172438864c264`, and no raw content.

### Discarded fusion diagnostics

Two-feature logistic fusion raised the partial-LoRA aggregate result from 61.82% recall at 1.70% FPR to 67.45% recall at 1.84% FPR with the same 26.79% LLM call rate.
It also raised PromptShield FPR from 2.96% to 3.63%, so the aggregate gain was not uniform.
A grouped monotonic-boosting follow-up reduced aggregate FPR to 0.32% but collapsed recall to 30.45%, including 3.31% on SEP, at 35.21% LLM calls.
Both fusion implementations were rejected in favor of the simpler rounded cascade.

At the more permissive 2% calibration flag-FPR cap, the frozen cascade reaches 59.88% evaluation flag recall at 1.46% FPR and 96.84% precision while retaining 99.82% review recall.
The partial LoRA reaches 65.69% flag recall at 1.79% FPR and 96.49% precision while retaining 99.40% review recall.
This is the first tested operating region that reaches the requested approximately 60% to 70% high-confidence flag recall at a relatively low FPR.
At a 5% calibration cap, flag recall rises to 78.71% for frozen and 80.40% for partial LoRA, but evaluation FPR also rises to 4.31% and 4.46%, which is too costly for a default automatic-flag policy.

Decision at the end of the July study: retain DeepSeek V4 Flash on CoreWeave fp8 with reasoning disabled as the downstream research route.
The partial-LoRA `0.2 / 0.999 / 0.9` recommendation below was superseded by the channel-aware follow-up in the next section.
The review-capable 99.5% recall family remains a separate high-recall experiment rather than the maintained no-manual-review route.
No result establishes a source-robust production guarantee.

## Channel-aware outer-intent repair (2026-08-02)

The retained prompt classified payload presence too readily and did not receive `input_channel`, even though the cascade already required that value from trusted runtime metadata.
In reviewer-only scoring of the complete 160-row instruction side of the scenario-held-out Boundary Pairs test, it reached 98.75% recall and 15.00% FPR at `p >= 0.9`.
It flagged every clean obfuscation control that asked for analysis of an encoded attack while explicitly forbidding execution.

The first repair made outer intent explicit but treated ordinary requests as benign regardless of provenance.
That candidate was rejected after full calibration because standalone AUROC fell from `0.93429` to `0.92764` and SEP recall at `p >= 0.9` collapsed from 33.07% to 0.13%.
The selected hybrid instead preserves a high-recall indirect-injection rule for untrusted content, distinguishes analysis from execution, and receives trusted `direct_user` or `untrusted_content` metadata in the system message.

The original 6,000-row calibration and 14,000-row evaluation split, full-LoRA local gates, provider, quantization, strict response schema, and retry behavior stayed fixed.
Only the prompt template and the DeepSeek threshold changed.
The candidate produced 20,000 of 20,000 valid typed outputs with no retries in the final run.
The old retained records had 19,991 valid outputs.
The candidate used a mean 524.00 input tokens per call versus 366.53 prompt tokens for the old contract, a material approximately 43% token increase.
Provider timing was measured in a different run window and is not used as a selection claim.

Standalone decision-token ranking improved on both roles:

| Prompt contract | Calibration AUROC / AP | Evaluation AUROC / AP |
|---|---:|---:|
| Old text-only prompt | 0.93429 / 0.91145 | 0.93931 / 0.92440 |
| Channel-aware hybrid | **0.94604 / 0.92025** | **0.95361 / 0.93816** |

With full-LoRA gates fixed at `0.2 / 0.99999`, calibration selected `p >= 0.7310585786300049` by maximizing recall subject to the same aggregate 2% FPR cap.
The old `p >= 0.9` route had 68 calibration false positives, 1.9808% FPR, and 66.58% recall.
The hybrid has the same 68 false positives and 1.9808% FPR with 69.11% recall.

| Frozen evaluation result | Recall | FPR | Precision | DeepSeek call rate |
|---|---:|---:|---:|---:|
| Old prompt, FP32 full LoRA | 66.79% | 1.810% | 96.50% | 22.17% |
| Channel-aware hybrid, FP32 full LoRA | **69.85%** | **1.760%** | **96.74%** | 22.17% |
| Channel-aware hybrid, OpenVINO BF16 | **70.02%** | 1.785% | 96.70% | **22.16%** |

The FP32 comparison changes 357 of 14,000 final routes.
It adds 252 true positives, loses 69 true positives, adds 16 false positives, and clears 20 false positives.
This is a strict aggregate improvement on the already-open evaluation role, not a uniform slice win.

| Evaluation slice | Old recall / FPR | Hybrid recall / FPR | Change |
|---|---:|---:|---:|
| Canonical | 78.69% / 1.611% | 80.09% / 1.933% | Recall +1.40 points; FPR +0.32 |
| PromptShield | 77.51% / 3.274% | 74.61% / 2.643% | Recall -2.90 points; FPR -0.63 |
| SEP | 38.63% / 0.114% | 48.06% / 0.114% | Recall +9.43 points; FPR unchanged |

At the selected `0.7310585786` cutoff, reviewer-only scoring of the complete instruction-side Boundary Pairs test reaches 100% recall, 5.00% FPR, and 95.00% fully correct matched pairs.
All ten clean obfuscation controls are correct when every row reaches the reviewer.
The four remaining reviewer false positives are three memory-context controls and one multi-agent trust control.

The retained end-to-end three-zone cascade reaches only 90.00% recall, 12.50% FPR, and 77.50% fully correct matched pairs on the same 80 instruction pairs.
Of the 80 attack rows, 8 fall below the local low gate, 50 reach DeepSeek, and 22 cross the local high gate.
Of the 80 clean rows, 61 fall below the local low gate, 11 reach DeepSeek, and 8 cross the local high gate; those eight local false positives are six obfuscation controls and two RAG controls.
The separate authorization diagnostic is likewise 97.5% attack-side / 0% clean-side under reviewer-only scoring but 45% / 0% end to end.
Neither pair is detector recall or FPR because approval, tool authority, and data egress belong to the deterministic reference monitor.

A calibration-only gate search then varied the existing low gate, high gate, and reviewer cutoff under the same 2% FPR and 35% call-rate caps.
Its row-recall winner lowered the local gate from `0.2` to `0.1`, adding 4 calibration true positives at the same 68 false positives but requiring 114 more reviewer calls.
On the already-open evaluation role it added only 2 true positives while adding 3 false positives and 236 reviewer calls, so the change was rejected.
Separating direct-user and untrusted-content high gates also produced no feasible calibration improvement, so that stage retained the scalar `0.2 / 0.99999` local gates.

### Channel-specific low-floor follow-up

The later AgentDojo transfer panel exposed five low-cue untrusted-content misses below the scalar `0.2` floor.
AgentDojo did not select a threshold, but it motivated testing the one trusted-metadata degree of freedom omitted from the earlier gate search: separate low floors by input channel.

The follow-up fixes the direct-user floor at `0.2`, the high gate at `0.99999`, and the reviewer threshold at `0.7310585786300049`.
It compares only untrusted-content floors `0.2`, `0.1`, and `0.05` on the retained 6,000-row calibration role, maximizing recall under the same 2% aggregate FPR and 35% call-rate caps.
The `0.05` point is infeasible at 2.010% calibration FPR.
The selected `0.1` point adds 10 true positives and 49 calls at the same 68 false positives as the scalar-floor route.

| FP32 full-LoRA policy | Calibration recall / FPR / calls | Evaluation recall / FPR / precision / calls |
|---|---:|---:|
| Scalar low floor `0.2` | 69.11% / 1.9808% / 22.02% | 69.85% / 1.760% / 96.74% / 22.17% |
| Direct `0.2`, untrusted `0.1` | **69.50%** / **1.9808%** / 22.83% | **70.27%** / 1.773% / **96.74%** / 22.91% |

The evaluation delta is 25 additional true positives, one additional false positive, and 104 additional provider calls over 14,000 rows.
SEP recall rises from 48.06% to 49.37% at unchanged 0.114% FPR.
Canonical recall rises from 80.09% to 80.15% while FPR rises from 1.933% to 1.960%, and direct-only PromptShield is unchanged.

The write-once [machine ablation](../artifacts/comparisons/channel-low-gate/summary.json) has SHA-256 `1d4eb89230700de7a7ce47c314e8c521ad2f66010c5bbbdd34ac0f78356ce6bc`.
Its runner has SHA-256 `220552c86a831b845ef21c20f03d0427cc6f9019296a1f15f56d2c965af13b71`.
The selected threshold-contract SHA-256 is `0d7edddf5fa86d791d4457a7e59cf229b03b093e826dd19ffc8556919f70b1ae`.
The ablation reuses the retained 20,000 reviewer records and sends no additional corpus text to a provider.

Decision: use the channel-specific low floor inside the advisory cascade.
The evaluation role is already open and the gain is small, so this remains a post-hoc engineering improvement pending representative prospective calibration.

The earlier scalar-floor OpenVINO BF16 verification used the same channel-aware ledger for the registered FP32 reference and BF16 candidate.
It found 97 local-zone differences and 32 final-route differences over 20,000 rows.
BF16 added one calibration false positive, raised evaluation recall by 0.17 points, raised FPR by 0.025 points, and passed every existing serving-equivalence check.

The channel-specific verification finds 89 local-zone differences and 24 final-route differences.
FP32 reaches 70.27% evaluation recall, 1.773% FPR, 96.74% precision, and a 22.91% call rate.
OpenVINO BF16 reaches 70.42% recall, 1.798% FPR, 96.70% precision, and a 22.89% call rate.
BF16 adds one calibration false positive and passes every quality gate.

The parsed text-free evidence is `artifacts/openrouter_downstream_eval/outer_intent_hybrid_results.jsonl.gz`.
Its compressed SHA-256 is `ac030484c5472b4500b2ad7f9bad1fb7b5f818276a9a630fa156f4707a064a5e`, and its decompressed ledger SHA-256 is `d4e930c004a52790a5ed2775b9ebb6f5abf2e0b3e14b7d953ee38aa1d899e096`.
The prompt, provider request, and version-2 scalar-floor threshold SHA-256 values are `6793cd3df00ea49c6da801692ef94b8200b212056fba27d298830186843b99a1`, `91cebeab5f248fe21677bf3b30afd6fe4df8bee61d2271852ac6a67c1c664b3e`, and `d75b25e472bb7219ce7a4948ee9abff77b04c2100dee1aa4875a0e02ceac8fb7`.
The superseded version-2 BF16 verification digest is `fdcf8621deaff51e44774783d057ee21ab239eae31207534509543cac30e720e`.
The selected channel-specific threshold SHA-256 is `0d7edddf5fa86d791d4457a7e59cf229b03b093e826dd19ffc8556919f70b1ae`.
The superseded version-3 BF16 verification digest is `3a267e7d930d89d7b1c379c360ed9419d4642aa7e59cb0a04b4e81a8a91c86f6`.

Decision: promote the channel-aware hybrid prompt and calibrated threshold inside the advisory cascade.
Do not interpret the gain as production validation.
The prompt was selected with already-open synthetic boundary and mixed-domain development data, PromptShield recall regressed, the provider is external and mutable, and no prospective prevalence-matched final test exists.
Learned output still cannot block, approve, grant authority, or bypass the reference monitor.

The later DeepSeek V4 Flash 0731 comparison superseded only the reviewer identity, provider, and threshold in this historical contract.
Under the owner's aggregate-quality criterion, the route used 0731 through Cloudflare at `0.6224593312018547` through 2026-08-16; the historical April metrics and artifacts above remain unchanged provenance.
The replacement evidence and its PromptShield limitation are recorded in [the 0731 research report](deepseek-v4-flash-0731-research.md).

## Artifacts and limitations

The machine summary is `artifacts/openrouter_downstream_eval/summary.json`.
Its SHA-256 is `728521a998124066924765e2601a0c35b29bdf3dc5f25b778ae1c7bb9f823e54`.
The panel, primary ledger, audit ledger, ablation ledger, setup-failure ledgers, and exact input hashes are recorded in that summary.
Versioned row-level ledgers use deterministic `.jsonl.gz` copies in the same directory, and the analyzer reads either the working `.jsonl` files or those compressed copies.

The completed runner and analyzer were disposable research code and are not retained in the maintained tree.
They do not modify `morgott scan`, shadow scoring, policy enforcement, authorization, or the maintained mmBERT trainer.

The follow-up machine summary is `artifacts/openrouter_downstream_eval/followup_summary.json`.
Its SHA-256 is `e1821ccb1cb8e4f2c6a573398d96f2cdce6c805abc98887f217f7c47dc856eee`.
It pins follow-up manifest SHA-256 `fe20a42462d1c4929b1f3927bdd34692a44175d3ef9853daee81090f562ac64a`, main parsed-ledger SHA-256 `002bf9286ef0021a427611c688f4b4e881851d6f7fad1dab57da9531d33b563f`, and StreamLake diagnostic-ledger SHA-256 `1a094006d85f556db5cbc99ceacea2594b18cc74f36312d144ed74dae7e035c8`.
The follow-up ledger stores parsed decisions, log probabilities, status, usage, cost, and latency, but no corpus text, system prompt, or raw provider response.

The fusion machine summary is `artifacts/openrouter_downstream_eval/fusion_summary.json`.
Its SHA-256 is `ac40eec09967c3bc9c29eb491ba1cc5c3c41c59fe14a07453ca41418ee0d60c4`.

The source-robust machine summary is `artifacts/openrouter_downstream_eval/robust_fusion_summary.json`.
Its SHA-256 is `148dc7e5cabc81f19b535cca87ceecf0a9a21278ccdf70231616b8ad145241a0`.
The rejected disposable fusion analyzers were removed after their stop decisions.

The selected pure advisory route is `src/morgott/models/downstream.py`.
It does not load a model, call a provider, authorize an action, or change `morgott scan`.
A future provider adapter should retry HTTP 408, 429, 5xx, timeout, and connection failures for at most three total attempts while honoring `Retry-After` and using jittered exponential backoff.
It should retry malformed output or missing log probabilities once and should not retry configuration or authentication failures such as HTTP 400, 401, or 403.
Only then should it pass `llm_failed=True`.

This is already-open development evidence.
The panel has 42.8% positive prevalence and is not representative production traffic.
Precision, accuracy, cost per input, and review rate will change with production prevalence and prompt length.
The recall-target thresholds were selected and evaluated on the same panel.
Latency was observed under concurrent research traffic and is not a service-level objective.
Provider and quantization comparisons can be confounded when the provider differs.
The 200-row ablations are directional evidence, not precise estimates of small differences.
No result approves blocking or grants authority.
