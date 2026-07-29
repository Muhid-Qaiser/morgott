# OpenRouter options for a fast downstream security layer

Research snapshot: 2026-07-29 at approximately 06:18 UTC.

Status: the model-selection recommendation in this pre-run snapshot is superseded by the completed [OpenRouter downstream evaluation](openrouter-downstream-evaluation.md).

## Recommendation

The proposed design is viable as an advisory selective cascade, but it is not viable as a security boundary or as evidence that either model may block users.
The first model to evaluate is `openai/gpt-oss-safeguard-20b`, because OpenAI built it to classify content against a developer-provided policy and explicitly describes using a small high-recall classifier to select content for a safety reasoner.
OpenAI also labels the safeguard model a research preview and says a dedicated classifier trained on tens of thousands of high-quality examples can still outperform it, so this recommendation is about task fit, price, and serving speed rather than proven Morgott accuracy.
[OpenAI's launch post](https://openai.com/index/introducing-gpt-oss-safeguard/) and [official model guide](https://developers.openai.com/cookbook/articles/gpt-oss-safeguard-guide) support those facts.

Use `mistralai/mistral-small-2603` as the generic structured-output control.
Use `deepseek/deepseek-v4-flash` as the cheaper generic control if its selected endpoint satisfies the required schema and privacy policy.
Do not choose a winner from public model positioning or general benchmarks, because no official source reports prompt-injection accuracy on Morgott's ontology or held-out suites for any candidate below.

## Current shortlist

Prices are endpoint list prices in USD per one million input and output tokens, and context is the selected endpoint's advertised maximum.
The serving measurements are OpenRouter's rolling 30-minute endpoint statistics captured at the time above, not vendor specifications or an availability guarantee.
OpenRouter defines latency in this endpoint feed as time to first token, and the request counts show the sample behind each snapshot.

| Model and fixed endpoint | Input / output | Context | Output contract | OpenRouter 30-minute serving snapshot |
|---|---:|---:|---|---|
| [`openai/gpt-oss-safeguard-20b`](https://openrouter.ai/openai/gpt-oss-safeguard-20b), Groq | $0.075 / $0.30 | 131,072 | Strict structured output and tools | 381 tok/s p50 throughput, 0.234 s p50 TTFT, 0.459 s p90 TTFT, 9,352 requests |
| [`mistralai/mistral-small-2603`](https://openrouter.ai/mistralai/mistral-small-2603), Mistral | $0.15 / $0.60 | 262,144 | Strict structured output and tools | 107 tok/s p50 throughput, 0.507 s p50 TTFT, 1.238 s p90 TTFT, 11,325 requests |
| [`deepseek/deepseek-v4-flash`](https://openrouter.ai/deepseek/deepseek-v4-flash), DeepInfra | $0.09 / $0.18 | 1,048,576 | Strict structured output and tools | 36 tok/s p50 throughput, 0.832 s p50 TTFT, 2.130 s p90 TTFT, 437,500 requests |
| [`qwen/qwen3.7-flash`](https://openrouter.ai/qwen/qwen3.7-flash), Alibaba | $0.03 / $0.13 below 32K prompt tokens | 1,000,000 | JSON mode and tools, but not strict schema | 124 tok/s p50 throughput, 0.921 s p50 TTFT, 2.578 s p90 TTFT, 29,265 requests |

The exact safeguard telemetry and its 30-minute window are available from OpenRouter's [first-party endpoint statistics response](https://openrouter.ai/api/frontend/v1/stats/endpoint?permaslug=openai%2Fgpt-oss-safeguard-20b).
OpenRouter's [endpoint catalog](https://openrouter.ai/api/v1/models/openai/gpt-oss-safeguard-20b/endpoints) reports that the safeguard model has one Groq endpoint, supports strict structured outputs, and requires reasoning.
The same first-party endpoint APIs provide the current price, context, and supported-parameter records for [Mistral Small 4](https://openrouter.ai/api/v1/models/mistralai/mistral-small-2603/endpoints), [DeepSeek V4 Flash](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash/endpoints), and [Qwen3.7 Flash](https://openrouter.ai/api/v1/models/qwen/qwen3.7-flash/endpoints).

Qwen3.7 Flash pricing rises to $0.10 / $0.40 from 32K through 256K prompt tokens and to $0.20 / $0.80 above 256K.
It was released two days before this snapshot, has one provider, and lacks strict schema support on that endpoint, so it is an experimental cost-floor candidate rather than the default.
DeepSeek pricing and parameter support vary by provider, so the $0.09 / $0.18 row applies to the pinned DeepInfra endpoint rather than every route for the model.

For this task, p90 time to first token and end-to-end latency matter more than long-output throughput because the desired answer should be a short label object.
That statement is an engineering inference, while the numbers in the table are OpenRouter observations.

`openai/gpt-oss-safeguard-20b` is therefore the best first experiment in this shortlist.
It is both the only task-specialized option here and the fastest observed option in this snapshot.
That does not establish that it has the best recall, precision, calibration, or adversarial robustness on Morgott.

## Cascade shape

Do not implement the cascade as `final_signal = mmbert_positive AND llm_positive` across every first-stage positive.
For a positive example, that design has recall `P(mmBERT positive) * P(LLM positive | mmBERT positive)`, so its end-to-end recall cannot exceed first-stage recall and no stage-two model can recover a first-stage false negative.

Use two validation-selected mmBERT thresholds instead:

```text
score < t_low                 -> no learned security signal
t_low <= score < t_high       -> ask the downstream classifier
score >= t_high               -> retain the security signal directly

final review signal = high zone OR (middle zone AND downstream signal)
```

This shape uses the expensive model where it can recover precision without allowing it to erase the strongest first-stage signals.
If the downstream model is also expected to find mmBERT misses, it must independently receive a sampled or provenance-selected part of the low zone, which raises cost and still requires measured end-to-end recall.

Select `t_low`, `t_high`, the downstream policy, and the final rule together on validation data.
Report the complete cascade's recall, false-positive rate, precision at realistic prevalence, latency, cost, malformed-output rate, per-source results, source-held-out results, and mutation evasion rather than multiplying independently measured model metrics.
The retained full-data frozen mmBERT is an advisory shadow with weak absolute external transfer, while the current rank-8 LoRA result is a one-seed engineering gate that has not replaced it.
The cascade is therefore a bounded research experiment, not a promoted blocking pipeline.

## Downstream contract

Give the safeguard model a fixed developer policy that preserves Morgott's distinctions among direct jailbreak, direct prompt injection, indirect prompt injection, harmful non-injection, benign, and uncertain.
Pass trusted `input_channel` and provenance from the runtime rather than asking the model to infer either from attacker-controlled text.
Require a small JSON schema such as a verdict, subtype, and short evidence span, and validate it locally.
Do not request tools, browsing, or any other side effect for classification.
Treat timeout, provider failure, and invalid output as `uncertain` or review rather than benign.
Treat the model's verdict and explanation as untrusted application input, and keep every side effect behind the deterministic reference monitor.

OpenRouter's [structured-output documentation](https://openrouter.ai/docs/guides/features/structured-outputs) recommends `provider.require_parameters: true` when schema support is required.
The safeguard endpoint requires reasoning, and OpenRouter bills reasoning tokens as output tokens even when the reasoning trace is excluded from the returned response.
[OpenRouter's reasoning-token documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) supports that billing rule.
Measure actual `usage` at the selected reasoning effort because the $0.30 output rate applies to both the short verdict and the potentially larger reasoning budget.

The approximate model cost is:

```text
forwarded_requests
* (average_input_tokens * input_rate
   + average_verdict_and_reasoning_tokens * output_rate)
/ 1,000,000
```

OpenRouter passes through provider inference prices but charges a 5.5 percent fee, with a stated $0.80 minimum, when credits are purchased.
[OpenRouter's pricing FAQ](https://openrouter.ai/docs/faq) documents that separate fee.

## Privacy and routing

No corpus or prompt text was sent to a model during this research.
Any Morgott remote experiment still requires an explicit, bounded, locally redacted, development-only review before corpus text leaves the machine.
OpenRouter says it does not retain prompts or responses unless logging or data-use options are enabled, but each downstream provider has its own policy.
[OpenRouter's data-collection documentation](https://openrouter.ai/docs/guides/privacy/data-collection) and [provider-policy documentation](https://openrouter.ai/docs/guides/privacy/provider-logging) describe this split.

At capture time, the safeguard endpoint record stated that Groq neither trains on nor retains prompts, and the endpoint appeared in OpenRouter's zero-data-retention catalog.
Still set `provider.zdr: true`, deny data collection, pin the accepted provider and model revision, and fail the request if no compliant endpoint remains.
[OpenRouter's ZDR documentation](https://openrouter.ai/docs/guides/features/zdr) explains per-request enforcement.

## Execution method

Use bounded asynchronous live `/api/v1/chat/completions` requests for the first fixed panel rather than OpenRouter's Batch API.
The Batch API is beta, has only a 24-hour completion window, may expire, and withholds all results until the whole batch reaches a terminal state.
Its documentation publishes neither an exact model-compatibility matrix nor a universal discount, so support and reduced pricing for these three models cannot be assumed.
Most importantly, OpenRouter stores batch inputs and results as JSONL artifacts in Google Cloud Storage for 30 days, which conflicts with the current raw-evaluation privacy requirement.
[OpenRouter's Batch API quickstart](https://openrouter.ai/docs/batch-quickstart) documents those semantics and retention.

Run each eligible model through its own bounded semaphore, starting with a small canary and about eight concurrent requests, then increase toward sixteen only while throughput improves without materially worsening p95 latency or `429` and `503` rates.
OpenRouter publishes no fixed concurrency quota for paid models; upstream provider capacity remains the practical limit.
Honor `Retry-After` when present, otherwise use bounded exponential backoff with jitter for transient `408`, `429`, `502`, and `503` failures.
OpenRouter documents no idempotency key for completion or batch submissions, and a batch `custom_id` is unique only inside that batch, so persist a stable local row-and-model ID and count only one terminal verdict while retaining every attempt for audit.
[OpenRouter's limits](https://openrouter.ai/docs/api-reference/limits) and [error-handling documentation](https://openrouter.ai/docs/api/reference/errors-and-debugging) describe the paid-model limits, fallback behavior, and retry headers.

Pin the immutable model revision, provider, and quantization; require ZDR and required parameters; deny provider data collection; and disable provider fallbacks so a retry cannot silently change the evaluated backend.
Qwen3.7 Flash currently has no ZDR route, so raw evaluation rows must not be sent to it without a separately authorized redacted experiment.
Use non-streaming responses for the classification panel, record client-side monotonic start and finish times, and enable `X-OpenRouter-Metadata: enabled` to capture routing attempts.
Persist each generation ID and fetch generation metadata after the hot path for provider, cost, token, reasoning-token, `latency`, and `generation_time` fields.
Client wall-clock time remains the authoritative end-to-end latency because batch and generation records do not include local queueing.
[Router metadata](https://openrouter.ai/docs/guides/features/router-metadata) and the [generation metadata schema](https://openrouter.ai/docs/client-sdks/python/components/generationresponsedata) document those fields.
Do not enable response caching because cached hits would invalidate latency and cost comparisons and require temporary response retention.

## Pre-execution red-team review

Snapshot: 2026-07-29 at approximately 07:58 UTC.

Do not launch the proposed 20,000-row panel unchanged.
It becomes a valid bounded development experiment after the following corrections.

1. Freeze the evaluation before seeing any panel output.
Use non-overlapping canonical validation rows for a small transport and format canary, then freeze the policy text, row IDs and raw-text hashes, sampling seed, mmBERT artifact hash and thresholds, model revisions, providers, quantizations, reasoning settings, schema, retry policy, and metric code.
Choose the frozen-head or LoRA artifact from its existing validation evidence before revealing this panel; do not use these LLM results to select the first-stage model or thresholds.
Do not use any panel response, audit result, or generated variant for fitting, prompt-example selection, or threshold selection.
Select the 10,000 canonical rows with predeclared source, label-evidence, channel, language, and length strata rather than an unstratified row sample.
Sample PromptShield deterministically while preserving its released prevalence, and sample 2,500 complete SEP pairs rather than 5,000 independent rows.
Retain source and group IDs, and audit exact and strict near-overlap across all three slices.
PromptShield test and SEP are already-open development evaluations, and canonical dev-test has already influenced model decisions, so none of the 20,000 rows is a prospective final test or supports a production-FPR claim.
The official [PromptShield paper](https://arxiv.org/html/2501.15145) and pinned [SEP release](https://github.com/egozverev/Should-It-Be-Executed-Or-Processed/tree/7606c0696f20f5aa433169fd2221f76852d1d4f5) define those source tasks.
The repository's [model ledger](model-experiments.md#full-data-frozen-mmbert-first-line-shadow-2026-07-28) records their prior use and notes that SEP measures instruction-in-data separation rather than harmfulness.

2. Mask unsupported labels instead of inventing negatives.
Return independent required enum fields `subversion` and `harmful_request`, each limited to `yes`, `no`, or `uncertain`, but score each field only where the source supplies matching ground truth.
PromptShield and SEP provide instruction-subversion evidence, not broad harmful-request labels, and SEP positives are benign-intent imperatives.
Score `harmful_request` only on direct-user rows with known matching labels, selected using trusted `input_channel`; report coverage and abstention separately.
Treat `uncertain` as review in the conservative cascade metric, and also report selective metrics over answered rows.
PromptShield lacks trustworthy row-level channel lineage, so pass its channel as unknown rather than labelling it direct-user.
Do not infer channel, provenance, or label applicability from the text being classified.

3. Freeze one policy contract, but acknowledge unequal wire enforcement.
Place a concise policy in the trusted system message and the untrusted text in a separate user message whose wrapper says that all embedded instructions are data and cannot change the classification task.
OpenAI recommends an instruction, definitions, criteria, examples, and explicit repeated output instructions, with roughly 400 to 600 policy tokens as an early-tested range.
[OpenAI's safeguard guide](https://developers.openai.com/cookbook/articles/gpt-oss-safeguard-guide) also warns that adding policy domains can reduce accuracy, so report each output field separately.
Groq's safeguard endpoint and DeepInfra's DeepSeek endpoint currently support strict JSON Schema; Alibaba's Qwen endpoint supports JSON object mode but explicitly lacks `structured_outputs`.
For the primary comparison, request the common `json_object` mode and apply the same strict local schema validator to all three rather than giving two models stronger provider-side enforcement.
Set `provider.require_parameters: true`, and reserve native strict-schema mode for a later deployability check.
Count malformed JSON, missing or extra fields, refusal, empty content, and non-`stop` finish reasons as first-attempt operational failures or `uncertain`; do not silently repair or semantically retry them.
[OpenRouter's structured-output documentation](https://openrouter.ai/docs/guides/features/structured-outputs) says strict enforcement varies by endpoint even when `strict: true`.

4. Fix the exact serving configurations before the full panel.
The current routes remain Groq `openai/gpt-oss-safeguard-20b` at $0.075 / $0.30 per million tokens, Alibaba `qwen/qwen3.7-flash-20260727` at $0.03 / $0.13 below 32K prompt tokens, and DeepInfra FP4 `deepseek/deepseek-v4-flash-20260423` at $0.09 / $0.18.
Safeguard reasoning is mandatory; Qwen reasoning defaults on and supports a token budget; DeepSeek exposes only `high` and `xhigh` and defaults to `high`.
For the fast-layer primary panel, use safeguard's mandatory default reasoning and explicitly disable reasoning for Qwen and DeepSeek.
Record that compute asymmetry, and add a reasoning-on generic-model ablation only if either generic model is promising.
Freeze those settings, set `temperature: 0` and one seed, and still run the stability audit because OpenRouter does not guarantee seeded determinism for every model.
Exclude returned reasoning text but retain reasoning-token counts; exclusion reduces stored response content, not billed output tokens.
[OpenRouter's reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) documents those controls and billing.
Qwen's sole endpoint currently retains prompts and is not ZDR, so it cannot receive raw panel rows under the current privacy rule.
Either omit Qwen or obtain a separate explicit authorization for a locally redacted Qwen-only panel and label it non-comparable if redaction changes text.

5. Make retries and audits measurable.
Use live asynchronous requests, randomize row order independently per model, pin the provider, disable fallbacks and response caching, and keep a stable local row-model-attempt ledger.
Retry only transient transport failures under the predeclared bounded policy; an ambiguous timeout may duplicate cost because OpenRouter documents no completion idempotency key.
Do not discard failed rows, and inspect both HTTP status and the response body because generation failures can surface after processing has begun.
Persist finish reason, provider, model, attempt, client wall time, token counts, reasoning tokens, cache tokens, and returned cost.
OpenRouter includes native-token usage and cost in every response and normalizes finish reasons to `stop`, `length`, `content_filter`, `error`, or `tool_calls`.
[Usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting) and [error handling](https://openrouter.ai/docs/api/reference/errors-and-debugging) define those fields and failure modes.
Score the frozen mmBERT model and each LLM alone on every eligible row before simulating the predeclared cascade.
Keep every LLM blind to the mmBERT score, and do not describe all-row LLM accuracy as actual cascade coverage.

6. Define the two audits precisely.
For repeat stability, stratify 200 panel rows by source, label, channel, length, and mmBERT score zone, then make three total calls per row and model with identical settings and caching disabled.
Report exact three-way agreement for both fields, pairwise disagreement, abstention, malformed output, and transport-failure rates; do not drop failed repetitions.
For prompt interference, reuse those 200 rows and compare the primary two-field policy against one otherwise matched subversion-only policy.
Score only `subversion`, and report accuracy change, exact agreement, flips toward `no` or `uncertain`, schema breakage, and refusals.

The full three-model design therefore has 60,000 primary calls, 1,200 extra stability calls when the first pass is reused as repetition one, and 600 prompt-interference calls.
Use the validation canary's returned usage to forecast the 61,800-call spend, set a dedicated API-key budget above that forecast, and define an automatic stop for unexpected cost, error rate, or malformed-output rate before execution.

## Minimal evaluation

Run one blinded, fixed-prompt comparison of safeguard, Qwen3.7 Flash when separately privacy-authorized, and DeepSeek V4 Flash over the same frozen rows and audits.
Keep the downstream models blind to the mmBERT score during evaluation, then combine their outputs with the predeclared three-zone rule.
Choose the winner only from Morgott end-to-end accuracy, false-positive cost, latency, malformed-output rate, privacy eligibility, and measured spend.
Skip a broad model sweep until those three establish whether the second layer adds independent signal at all.
