# OpenRouter log-probability support for the downstream experiment

Checked on 2026-07-29.

## Decision

OpenRouter Chat Completions supports `logprobs` and `top_logprobs` in its common API, but the exact endpoint used by the initial panel experiment did not advertise either parameter.

The initial pinned route was `deepseek/deepseek-v4-flash-20260423` through `deepinfra/fp4`, with provider fallbacks disabled and `require_parameters=true`.
That configuration is recorded in the [runner](../experiments/openrouter_downstream_eval/run.py) and [experiment manifest](../artifacts/openrouter_downstream_eval/manifest.json).
The later bounded follow-up moved to CoreWeave fp8 and retained decision-token log probabilities; its final measured route and limitations are recorded in the [downstream evaluation](openrouter-downstream-evaluation.md).

The current OpenRouter endpoint catalog lists `reasoning`, `response_format`, and `structured_outputs` for the DeepInfra fp4 endpoint, but omits `logprobs` and `top_logprobs`.
Endpoint capability is the relevant level because OpenRouter documents that support can differ between providers serving the same model and can change over time.
See the official [DeepSeek V4 Flash provider catalog](https://openrouter.ai/deepseek/deepseek-v4-flash-20260423/providers), [endpoint metadata](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-20260423/endpoints), and [endpoint-level support guidance](https://openrouter.ai/docs/guides/features/structured-outputs#model-support).

Therefore, usable token log probabilities could not be collected while preserving the exact initial DeepInfra fp4 route.
With `require_parameters=true`, adding these parameters makes that endpoint ineligible.
Without it, OpenRouter warns that providers may ignore unsupported parameters, which would not produce a reliable threshold signal.
See [provider parameter routing](https://openrouter.ai/docs/guides/routing/provider-selection#requiring-providers-to-support-all-parameters).

## General request and response shape

The Chat Completions request schema accepts:

```json
{
  "logprobs": true,
  "top_logprobs": 5
}
```

OpenRouter documents `logprobs` as a boolean and `top_logprobs` as an integer from 0 through 20.
See the [Chat Completions API reference](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request) and [official Python SDK reference](https://openrouter.ai/docs/client-sdks/python/api-reference/chat).

For a supporting endpoint, the non-streaming response places the data at `choices[i].logprobs.content`.
Each emitted content token has `token`, `bytes`, `logprob`, and a `top_logprobs` array containing the same fields for alternatives.
The field can be absent or null, so the evaluator must fail closed when it is missing.
See the official SDK definitions for [the choice](https://github.com/OpenRouterTeam/typescript-sdk/blob/main/src/models/chatchoice.ts), [the log-probability container](https://github.com/OpenRouterTeam/typescript-sdk/blob/main/src/models/chattokenlogprobs.ts), and [each token](https://github.com/OpenRouterTeam/typescript-sdk/blob/main/src/models/chattokenlogprob.ts).

## JSON output, reasoning, and interpretation

If an endpoint supports both log probabilities and `response_format`, the JSON returned in `message.content` is still generated content.
By the response schema, its JSON punctuation, field name, and boolean value are output tokens and receive token-level entries.
This is a schema-based inference, not a promise that the resulting values are calibrated probabilities of the semantic class.

Structured decoding may constrain the available tokens, and tokenization may attach whitespace or split a value across tokens.
Do not assume a fixed array index or that `true` and `false` are each one token.
Locate the boolean value from the reconstructed token bytes, and treat missing or ambiguous alternatives as unscorable.

The raw chosen-token value `exp(logprob)` is the model probability of that token under the provider's decoding path.
It is not automatically a normalized binary class probability.
Only when both verdict alternatives are present at the same decision position is a local binary score defensible:

```text
p(true | true-or-false) = exp(lp_true) / (exp(lp_true) + exp(lp_false))
```

If one alternative is absent from `top_logprobs`, do not assign it zero probability.
A multi-token alternative also needs sequence probability, which a single sampled path does not provide directly.
Use the derived value only as an empirically calibrated ranking signal.

Reasoning and log probabilities are separate parameters.
OpenRouter documents `reasoning.effort="none"` as disabling reasoning, while `reasoning.exclude=true` merely hides reasoning that the model may still perform.
The current experiment used `enabled=false` together with `exclude=true`.
Log probabilities, when supported, describe completion content tokens rather than hidden reasoning tokens.
See [OpenRouter reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#reasoning-effort-level).

## Best route alternatives

These current DeepSeek V4 Flash endpoints advertise `reasoning`, `response_format`, `logprobs`, and `top_logprobs` together:
The endpoint names, capabilities, prices, quantizations, and data-policy fields below come from the official [provider catalog and endpoint metadata](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-20260423/endpoints).

| Choice | Provider endpoint | Quantization | Listed input/output price per 1M tokens | Why choose it |
|---|---|---:|---:|---|
| Lowest listed price | `streamlake/fp8` | fp8 | $0.091 / $0.182 | Cheapest supporting route in the current catalog |
| Low price without listed prompt retention | `akashml/fp8` | fp8 | $0.098 / $0.196 | Small price increase and no listed prompt retention |
| Closest comparison to the existing run | `ionstream/fp4` | fp4 | $0.14 / $0.28 | Preserves fp4 while adding both log-probability parameters |

The catalog also currently lists DigitalOcean, Morph, Parasail fp8, Fireworks, Ambient fp4, CoreWeave fp8, DeepSeek, and Mancer fp4 as supporting the required parameter combination.
Provider behavior, latency, and tokenization can differ, so switching routes is a new provider experiment rather than a continuation of the exact DeepInfra comparison.

For the fastest useful turnaround, run a small schema-and-tokenization canary on `streamlake/fp8`, then use that route for the bounded panel only if every response contains parseable boolean-token log probabilities.
Use `provider.order=["streamlake/fp8"]`, `allow_fallbacks=false`, and `require_parameters=true` so unsupported fallbacks cannot silently enter the results.
