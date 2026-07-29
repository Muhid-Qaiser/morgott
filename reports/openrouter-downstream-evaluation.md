# OpenRouter downstream evaluation

Date: 2026-07-29

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

The current practical recommendation is:

1. Keep the mmBERT output advisory and use it to route, never to grant authority.
2. Do not let a binary LLM `false` automatically clear every mmBERT escalation.
3. If an LLM is needed now, use DeepSeek V4 Flash with reasoning off for review prioritization, subtyping, or evidence collection, not final automatic clearance.
4. Run instruction-subversion and harmful-request classification as separate prompts in the next experiment.
5. Revisit encoder selection after the full-data LoRA finishes.
6. Calibrate any production boundary on representative traffic that is separate from this open development panel.

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
The encoder choice therefore remains unresolved until the full-data LoRA is complete and evaluated under the same contract.

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

## Artifacts and limitations

The machine summary is `artifacts/openrouter_downstream_eval/summary.json`.
Its SHA-256 is `728521a998124066924765e2601a0c35b29bdf3dc5f25b778ae1c7bb9f823e54`.
The panel, primary ledger, audit ledger, ablation ledger, setup-failure ledgers, and exact input hashes are recorded in that summary.
Versioned row-level ledgers use deterministic `.jsonl.gz` copies in the same directory, and the analyzer reads either the working `.jsonl` files or those compressed copies.

The runner and analyzer are disposable research code under `experiments/openrouter_downstream_eval/`.
They do not modify `morgott scan`, shadow scoring, policy enforcement, authorization, or the maintained mmBERT trainer.

This is already-open development evidence.
The panel has 42.8% positive prevalence and is not representative production traffic.
Precision, accuracy, cost per input, and review rate will change with production prevalence and prompt length.
The recall-target thresholds were selected and evaluated on the same panel.
Latency was observed under concurrent research traffic and is not a service-level objective.
Provider and quantization comparisons can be confounded when the provider differs.
The 200-row ablations are directional evidence, not precise estimates of small differences.
No result approves blocking or grants authority.
