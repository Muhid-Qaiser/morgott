# Voyage retrieval models for Morgott

Research snapshot: 2026-07-29.

## Bottom line

`voyageai/voyage-4` is an embedding model that converts each text into a reusable semantic vector for search, clustering, and nearest-neighbor retrieval.
`voyageai/rerank-2.5` is a cross-encoder that receives one query plus an already-selected list of documents and returns that list ordered by relevance.
Neither model is a small LLM safety judge, and neither output is a calibrated probability that text is a prompt injection.
They can support retrieval of policy text or labelled exemplars, but they should not replace mmBERT, the downstream safety reasoner, trusted provenance, or the deterministic reference monitor.
The simplest first experiment is the existing mmBERT uncertainty band followed directly by the safety reasoner, with retrieval added only if a controlled ablation shows that retrieved evidence improves end-to-end errors.

## Exact difference

| Property | `voyageai/voyage-4` | `voyageai/rerank-2.5` |
|---|---|---|
| Job | It independently encodes text for approximate semantic retrieval. | It jointly compares one query with each supplied candidate for more precise relevance ordering. |
| Input | It accepts one text or a batch of texts. | It accepts one query and a list of candidate documents. |
| Output | It returns one numeric vector per input. | It returns document indices and relevance scores sorted by relevance. |
| Typical placement | It creates the initial searchable index and retrieves a broad top-k set. | It optionally refines that top-k set before evidence is sent to an LLM. |
| Reuse | Document vectors can be stored and reused across queries. | Every new query must be evaluated with its candidate documents. |
| Security meaning | Vector proximity means semantic similarity. | The score means relevance to the supplied query. |

OpenRouter describes embeddings as semantic vectors used for retrieval, search, clustering, classification, duplicate detection, and anomaly detection, while its RAG guide describes reranking as a more precise but additional cross-encoder pass over retrieved candidates.
The official [OpenRouter embeddings guide](https://openrouter.ai/docs/api_reference/embeddings) and [OpenRouter RAG guide](https://openrouter.ai/docs/cookbook/evaluate-and-optimize/rag) support that distinction.
Voyage gives the same distinction in its [reranker documentation](https://docs.voyageai.com/docs/reranker).

## `voyageai/voyage-4`

OpenRouter lists `voyage-4` as a text-to-embeddings model with a 32,000-token context and one Voyage AI by MongoDB endpoint at $0.06 per million input tokens.
The current facts are available on the [OpenRouter model page](https://openrouter.ai/voyageai/voyage-4) and its [endpoint catalog response](https://openrouter.ai/api/v1/models/voyageai/voyage-4/endpoints).
Voyage documents a default dimension of 1,024 and supported Matryoshka dimensions of 256, 512, 1,024, and 2,048.
Voyage also supports float, int8, uint8, binary, and unsigned-binary outputs for this model.
The direct Voyage API accepts at most 1,000 texts and 320,000 total tokens in one `voyage-4` request, while each text has the 32,000-token model limit.
Voyage's direct API truncates over-length input by default, so silent loss is possible unless truncation is disabled and errors are handled.
These model capabilities and direct API limits are documented in the [Voyage embedding reference](https://docs.voyageai.com/reference/embeddings-api) and [Voyage embedding guide](https://docs.voyageai.com/docs/embeddings).

For retrieval, Voyage recommends marking inputs as queries or documents because it applies different retrieval prompts.
Voyage's direct API uses `input_type="query"` and `input_type="document"`, while OpenRouter's generic embedding schema documents `input_type` with examples such as `search_query` and `search_document`.
OpenRouter's generic API also accepts `dimensions`, but the exact `voyage-4` endpoint currently advertises an empty model-specific supported-parameter list.
An integration should therefore contract-test the requested input type and dimension against OpenRouter before building an index, and it should reject a response with an unexpected vector length.
The relevant schemas are the [OpenRouter embedding request reference](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request) and the [Voyage embedding reference](https://docs.voyageai.com/reference/embeddings-api).

Voyage's direct-account pricing page promises the first 200 million `voyage-4` tokens free, but OpenRouter's listing does not promise those direct-account credits.
OpenRouter traffic should therefore be budgeted at its listed $0.06 per million input tokens.
The direct pricing terms are in [Voyage pricing](https://docs.voyageai.com/docs/pricing).

## `voyageai/rerank-2.5`

OpenRouter exposes the model through `POST /api/v1/rerank`, where `model`, `query`, and `documents` are required and `top_n` is optional.
The response contains sorted results with an input index and relevance score, and may include the corresponding document.
The exact request and response contract is in the [OpenRouter rerank reference](https://openrouter.ai/docs/api/api-reference/rerank/create-rerank).

Voyage describes `rerank-2.5` as a quality-focused, multilingual, instruction-following generalist reranker with a 32,000-token query-document context.
Its query may contain at most 8,000 tokens, each query-document pair may contain at most 32,000 tokens, a request may contain at most 1,000 documents, and total processed input may not exceed 600,000 tokens.
Voyage defines total processed tokens as the query token count multiplied by the number of documents, plus the sum of all document token counts.
Voyage's direct API truncates by default and can instead reject over-limit inputs when truncation is disabled.
These limits and semantics are documented in the [Voyage reranker reference](https://docs.voyageai.com/reference/reranker-api).

The defensible current price is $0.05 per million processed rerank input tokens.
Voyage publishes that exact price and formula, while OpenRouter's model-page payload contains the rerank-specific `rerank:input-tokens` price of $0.00000005 per token.
OpenRouter's same page is internally inconsistent because its generated metadata and FAQ call the model free and its generic endpoint JSON reports zero prompt and completion price.
The generic zero fields should not be treated as a free-service promise because the page also marks the endpoint as non-free and carries the separate rerank input-token SKU.
Budget at $0.05 per million processed tokens and verify the first billed generation before any material run.
The conflicting OpenRouter presentation is visible on the [exact model page](https://openrouter.ai/voyageai/rerank-2.5), while Voyage's unambiguous price is in [Voyage pricing](https://docs.voyageai.com/docs/pricing).

## Endpoint and privacy facts

Each exact OpenRouter slug currently has only one endpoint, operated by Voyage AI by MongoDB, so there is no independent provider fallback for the same model.
The endpoint catalogs currently expose no 30-minute latency or throughput observations, so there is no first-party basis for calling either exact OpenRouter route fast.
The [voyage-4 endpoint response](https://openrouter.ai/api/v1/models/voyageai/voyage-4/endpoints) and [rerank-2.5 endpoint response](https://openrouter.ai/api/v1/models/voyageai/rerank-2.5/endpoints) support those facts.

OpenRouter's current model-page data marks both Voyage endpoints as not training on inputs but retaining prompts.
Neither endpoint appears in OpenRouter's current [ZDR endpoint catalog](https://openrouter.ai/api/v1/endpoints/zdr).
OpenRouter says its own proxy does not retain prompt bodies unless input-output logging is enabled, but the request is still sent to the upstream provider and provider retention is a separate policy.
OpenRouter defines `provider.zdr=true` as restricting a request to ZDR endpoints, so the current single retained-prompt endpoints should not be assumed usable under that restriction.
The relevant controls and distinctions are documented in OpenRouter's [ZDR guide](https://openrouter.ai/docs/guides/features/zdr), [provider-routing guide](https://openrouter.ai/docs/guides/routing/provider-selection), and [logging FAQ](https://openrouter.ai/docs/faq).

Voyage's direct terms grant model-improvement rights by default and say that opting out applies only to later content, after which later content is deleted after processing.
That direct-account rule should not be conflated with OpenRouter's endpoint-specific no-training representation, but it remains relevant if the integration later bypasses OpenRouter.
The direct rule is in the [Voyage terms of service](https://www.voyageai.com/tos).

Morgott should not send corpus rows, sensitive prompts, or provider-review samples to either endpoint merely because an API key exists.
A bounded remote evaluation needs an explicit privacy decision, local redaction, a fixed sample manifest, and confirmation of the applicable endpoint policy at execution time.

## Fit in the Morgott pipeline

An embedding or reranking score is not interchangeable with the mmBERT score.
Semantic similarity can be high for a benign article explaining prompt injection and low for a novel or obfuscated attack, so a nearest-attack threshold has no intrinsic recall or false-positive interpretation.
An instruction-following reranker answers the relevance question it is given, and attacker-controlled query text can influence that question.
Neither score should clear a high mmBERT signal, grant authority, or become a blocking rule without its own prospectively evaluated and calibrated classifier.

A defensible optional flow is:

1. Trusted runtime metadata supplies the input channel and provenance, and the deterministic reference monitor remains authoritative for every action.
2. mmBERT produces its advisory first-pass score, and only a validation-selected uncertain band proceeds to expensive remote work.
3. An embedding index retrieves a small set of versioned policy passages or labelled exemplars, preferably from a locally stored index whose document vectors are not recomputed per request.
4. A reranker is added only if the embedding top-k set is large or noisy enough that measured reranking improves evidence selection.
5. The downstream safety reasoner receives the original text, trusted channel, and retrieved evidence, returns strict structured output, and maps timeout or malformed output to `uncertain`.

This design uses retrieval as evidence selection rather than pretending retrieval is detection.
It also keeps retrieved text and every model output untrusted, consistent with Morgott's [threat model](../docs/threat-model.md).

Using one 32,000-token embedding for an entire long document is not a fix for mmBERT's first-512-token truncation because a small injected span can be diluted in a document-level semantic vector.
If long-document localization is the goal, chunk the document with trusted lineage, preserve all chunks for policy, and evaluate whether retrieval finds the injected span before adding the result to the cascade.
The simpler control is direct chunk-wise advisory scoring, which should be compared against retrieval rather than assumed inferior.

## Current alternatives worth testing

The following is a short OpenRouter snapshot rather than a quality ranking.
OpenRouter prices are list prices at the snapshot, and per-search Cohere prices are not directly comparable with per-token prices until candidate count and text length are fixed.
Provider counts come from the linked endpoint pages, while retention and ZDR status come from each model page and OpenRouter's current [ZDR endpoint catalog](https://openrouter.ai/api/v1/endpoints/zdr).

### Embedding candidates

| Candidate | Current OpenRouter price | Context, modality, and dimensions | Providers and endpoint privacy | Defensible reason to test |
|---|---:|---|---|---|
| [`voyageai/voyage-4-lite`](https://openrouter.ai/voyageai/voyage-4-lite) | $0.02 per million input tokens | 32,000 tokens, text, and 256, 512, 1,024 default, or 2,048 dimensions | One provider, no training, prompt retention, and no ZDR | It is the cheapest compatible Voyage 4-series control. |
| [`perplexity/pplx-embed-v1-0.6b`](https://openrouter.ai/perplexity/pplx-embed-v1-0.6b) | $0.004 per million input tokens | 32,000 tokens, text, and 1,024 dimensions | One provider, no training and no retention, but absent from the current ZDR catalog | It is the cheapest current paid 32K text candidate and has local MIT-licensed weights. |
| [`qwen/qwen3-embedding-8b`](https://openrouter.ai/qwen/qwen3-embedding-8b) | $0.01 per million on Nebius and DeepInfra, or $0.04 per million on SiliconFlow | 32K tokens, text, and up to 4,096 Matryoshka dimensions | Three providers, all with no training, no retention, and ZDR | It adds provider redundancy, instruction-aware multilingual retrieval, and Apache-2.0 local weights. |
| [`baai/bge-m3`](https://openrouter.ai/baai/bge-m3) | $0.01 per million input tokens | 8,192 tokens on the smaller endpoint, text, and a 1,024-dimension dense output | Two providers, both with no training, no retention, and ZDR | It is an open multilingual cloud and local privacy control. |
| [`nvidia/nemotron-3-embed-1b:free`](https://openrouter.ai/nvidia/nemotron-3-embed-1b:free) | Free | OpenRouter advertises 32,768 tokens, while NVIDIA validates 4,096 tokens; text and a native 2,048-dimension output | One provider, training enabled, prompt retention, and no ZDR | Its roughly 1.14B local weights are a useful self-hosted control, but the free remote route and conflicting limits are unsuitable for sensitive production traffic. |

Perplexity's official [0.6B model card](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) documents the 32K context, 1,024 dimensions, MIT license, local SentenceTransformers and ONNX paths, and the requirement to compare its unnormalized int8 vectors with cosine similarity.
Qwen reports that Qwen3-Embedding-8B scored 70.58 on the MTEB multilingual leaderboard as of 2025-06-05, supports more than 100 languages, and has a 32K context with a 4,096-dimension Matryoshka output in its [official release](https://qwenlm.github.io/blog/qwen3-embedding/).
BGE-M3's [official model card](https://huggingface.co/BAAI/bge-m3) documents its multilingual 8,192-token model and its local dense, sparse, and ColBERT-style outputs, but OpenRouter's unified embedding API should only be assumed to expose the advertised dense vector.
NVIDIA reports average NDCG@10 values of 72.38 on RTEB and 71.04 on MMTEB Retrieval for its BF16 Nemotron 3 Embed 1B family member at a 4,096-token evaluation length in the [official model card](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16).
NVIDIA's current [NIM support matrix](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html) validates that model at 4,096 tokens and 2,048 dimensions without reduced dimensions, so OpenRouter's 32,768-token listing requires a contract test.
Those are vendor-run general retrieval results, not evidence of prompt-injection retrieval, calibration, or downstream Morgott improvement.
OpenRouter's generic `dimensions` parameter still needs a contract test for every exact endpoint before an index is built.

### Reranking candidates

| Candidate | Current OpenRouter price | Context and modality | Providers and endpoint privacy | Defensible reason to test |
|---|---:|---|---|---|
| [`voyageai/rerank-2.5-lite`](https://openrouter.ai/voyageai/rerank-2.5-lite) | $0.02 per million processed input tokens | 32,000 tokens and text | One provider, no training, prompt retention, and no ZDR | It is the direct lower-cost and lower-latency Voyage control. |
| [`cohere/rerank-4-fast`](https://openrouter.ai/cohere/rerank-4-fast) | $0.002 per search unit | 32,768 tokens and text | One provider, no training, 30-day retention, and no ZDR | It is positioned for low latency, high throughput, multilingual text, and structured data. |
| [`nvidia/llama-nemotron-rerank-vl-1b-v2:free`](https://openrouter.ai/nvidia/llama-nemotron-rerank-vl-1b-v2:free) | Free | 10,240 tokens and text plus image | One provider, training enabled, prompt retention, and no ZDR | It is a useful local or future multimodal control, but its remote privacy policy is unacceptable for sensitive text. |
| [`Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | Local infrastructure cost only | 32K tokens and text | No current OpenRouter endpoint, with retention controlled by the local deployment | It is a small Apache-2.0 text reranker with an official reproducible model card. |

Cohere documents a 32,768-token context, support for more than 100 languages, and the intended Fast versus Pro latency-quality tradeoff in its [Rerank 4 documentation](https://docs.cohere.com/changelog/rerank-v4.0) and [best-practices guide](https://docs.cohere.com/docs/reranking-best-practices).
Cohere's public documents provide positioning rather than a numerical comparison that establishes Rerank 4 Fast as better for Morgott.
NVIDIA reports that its multimodal reranker raised average Recall@5 from 71.04% to 76.12% for text input and from 73.24% to 77.64% for image-plus-text input across five visual-document retrieval datasets in its [official model card](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2).
Qwen reports general retrieval benchmark results for its local rerankers in the [Qwen3 Embedding release](https://qwenlm.github.io/blog/qwen3-embedding/), but those scores use Qwen-retrieved top-100 candidates and do not measure Morgott's security task.

### Recommendation

Start with an offline shadow ablation that compares mmBERT alone, mmBERT plus the safety reasoner, and the same cascade with retrieved exemplars.
Use local `pplx-embed-v1-0.6b` as the simplest privacy-preserving embedding control.
Use an explicitly ZDR-routed `qwen3-embedding-8b` or `bge-m3` request for a serving-parity check only after the remote-evaluation gate is approved.
Use `qwen3-embedding-8b` only as a stronger and heavier multilingual retrieval control, not as a presumed winner.
Skip reranking until errors show that the correct evidence is already inside the broad retrieved set but ordered too poorly for the reasoner.
If that condition is met, compare local Qwen3-Reranker-0.6B with `rerank-2.5-lite`, and reserve NVIDIA's multimodal pair for a future image or document-page threat model.
Do not send sensitive text through the free NVIDIA or Cohere routes under their current OpenRouter retention policies.
Measure end-to-end recall, false-positive rate, conditional recall inside the uncertainty band, source-held-out behavior, long-benign behavior, mutation evasion, p50 and p95 latency, remote failure rate, and cost.
Do not promote a similarity threshold using the same dev-test suite that selected the exemplars or query wording.
The downstream safety reasoner remains the component that applies the explicit security policy, and the deterministic reference monitor remains the authority boundary.
