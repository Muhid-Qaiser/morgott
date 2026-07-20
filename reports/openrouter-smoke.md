# OpenRouter one-shot reviewer smoke test

Generated: 2026-07-17

This exploratory test asks whether a remote LLM reviewer adds enough signal to
justify a multi-stage or ReAct-style hot path. It is not part of the guard and
did not select any locked detector threshold.

## Protocol

The frozen sample has 100 rows: 50 attacks and 50 benign controls, with ten
rows from each of ToxicChat attacks, deepset prompt injections, obfuscated
multi-turn attacks, standalone BIPIA payloads, poisoned BIPIA contexts, clean
BIPIA contexts, XSTest, accepted OASST chat, harmful-but-non-injection prompts,
and NotInject trigger-word controls.

Each item received one structured-output request. There were no tools, plugins,
retries, fallback providers, response repair, or agent loop. Routing requested
zero data retention and denied provider data collection. Raw prompts and
responses were not persisted; the ignored result files retain hashes, enum
verdicts, aggregate usage, latency, and cost only.

## Results

| Requested model | Availability | Attack recall among available | FPR among available | Accuracy among available | p50 / p95 latency | Reported cost |
|---|---:|---:|---:|---:|---:|---:|
| `google/gemini-2.5-flash-lite` | 100/100 | 35/50 (70.0%) | 1/50 (2.0%) | 84.0% | 1,417 / 2,629 ms | $0.0033793 |
| `openai/gpt-5.4-nano` | 84/100 | 24/36 (66.7%) | 0/48 (0%) | 85.7% | 1,844 / 3,074 ms | $0.0069230 |

Gemini detected 8/10 ToxicChat, 5/10 deepset, 10/10 obfuscated,
8/10 standalone BIPIA, and 4/10 poisoned-context attacks. Its one false
positive was a harmful-but-non-injection request.

GPT-5.4 nano returned 16 unusable responses, concentrated on the inputs most
important to defend: 7/10 ToxicChat attacks, 6/10 obfuscated attacks, 1/10
BIPIA payloads, and 2/10 harmful controls were unavailable. The zero observed
FPR therefore applies only to the 84 available results and must not be read as
a 0% end-to-end false-positive or failure rate. One preflight response was
confirmed as provider content filtering; the harness conservatively records
all non-`stop` or invalid responses as unavailable.

Across the two full runs, reported successful-call cost was $0.0103023. Cost is
not the limiting factor here; availability, privacy exposure, and roughly
1.4–1.8 second median latency are.

## Decision

Do not add a remote reviewer, ReAct loop, or LLM OR-gate to the request path.
Neither model beats the cheap local baseline on the evidence needed for this
project, and an agent loop would add more opportunities for prompt-controlled
state, latency, provider failure, and data exposure. Keep the harness as an
offline adjudication/red-team experiment only. If revisited, evaluate a single
bounded call on a pre-registered disagreement slice and count unavailable
responses explicitly rather than silently retrying them.

The reproducible harness and frozen sampling specification are in
[`experiments/openrouter_review/`](../experiments/openrouter_review/README.md).
