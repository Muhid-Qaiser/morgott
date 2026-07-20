# OpenRouter one-shot reviewer experiment

This is an optional, advisory experiment—not a production gate. It evaluates one
strictly structured LLM classification call per item on a locked, 100-row sample:
50 attacks and 50 benign controls spanning direct injection, obfuscation, BIPIA
indirect injection/clean context, XSTest, ordinary chat, harmful non-injection
requests, and NotInject over-defense controls.

The default command is offline. It validates the pinned sample and makes no call:

```bash
python experiments/openrouter_review/review.py --model vendor/model
```

Run the tests without network access:

```bash
python -m unittest discover -s experiments/openrouter_review
```

Only an explicit `--execute` makes the 100 calls. Supply the key through the
environment, or install `python-dotenv` to load the repository `.env` silently:

```bash
python experiments/openrouter_review/review.py \
  --model vendor/model \
  --execute \
  --output experiments/openrouter_review/results/vendor-model.json
```

The request uses OpenRouter's strict JSON Schema mode, disables provider fallback,
sorts providers by latency, disables reasoning and response caching, requires
parameter support, requests zero-data-retention routing, denies provider data
collection, and enables no tools/plugins. There is no retry, ReAct loop, or
response-healing call. Non-`stop` completions, timeouts, HTTP/network errors, and
validation failures are recorded as `unavailable`.

Results contain only sample/request/response hashes, labels, enum verdicts,
redaction/truncation counts, latency, token usage, reported cost, model, provider,
router-attempt count, and aggregates. Selected-provider metadata is requested but
only those two fields are retained. Raw prompts, responses, routing payloads,
exception messages, and the API key are never persisted or printed.

The two completed 100-row smoke runs are summarized in the versioned
[`reports/openrouter-smoke.md`](../../reports/openrouter-smoke.md). Neither model
was promoted: the fastest median was still 1.42 seconds, Gemini produced a 2%
FPR on the small control slice, and GPT-5.4 nano returned no usable verdict for
16% of rows. The ignored per-call result files can be regenerated locally.

OpenRouter references: [structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs),
[usage accounting](https://openrouter.ai/docs/api/reference/overview), and
[provider privacy/routing](https://openrouter.ai/docs/guides/routing/provider-selection),
and [router metadata](https://openrouter.ai/docs/guides/features/router-metadata).
