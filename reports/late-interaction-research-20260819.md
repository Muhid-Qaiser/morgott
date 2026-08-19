# Late-interaction retrieval research (2026-08-19)

Current implementation decisions are superseded by [retrieval-assisted-reviewer-findings-20260819.md](retrieval-assisted-reviewer-findings-20260819.md).
Late interaction is deferred to a later iteration.

## Decision

Morgott should integrate late interaction as an optional fail-soft experimental reranker of the already-locked PPLX 4B 256-dimensional HNSW plus partitioned-BM25 candidate union after its target-shaped operational canary passes.

Morgott should not make it the default, replace BM25, deploy a second vector service, or build a all-row ColBERT search index before prospective quality evidence supports those larger decisions.

The first model should be the Apache-2.0 `lightonai/mLateOn` revision `edd378f99593c0ac8a15518b97ad89786b02685e` run locally, because its 8,192-token query contract avoids the 32-token query truncation used by several ColBERT-family checkpoints and its 307M-parameter size is materially smaller than the incumbent 4B embedder.

This recommendation is an experiment choice rather than a production endorsement, because all public quality and speed results are generic or vendor-reported rather than Morgott cascade evidence.

The smallest experiment and optional integration need no provider calls, no full late-interaction search index, no Qdrant or Vespa deployment, and no change to maintained inference defaults.

If exact reranking cannot improve the frozen candidate union, the experimental path may remain available for more representative research, but a full multi-vector index cannot repair that model-level mismatch cheaply enough to justify default promotion.

If exact reranking wins but the frozen union lacks useful examples, then and only then should Morgott test full-corpus PLAID retrieval as a candidate generator.

## What ColBERT changes

Dense retrieval compresses every query and example into one vector before comparison, while ColBERT keeps a contextual vector for each retained token and delays query-document interaction until scoring.

ColBERT's MaxSim score takes the best document-token match for each query token and sums those maxima, as defined by the [original ColBERT paper](https://arxiv.org/abs/2004.12832) and [official implementation](https://github.com/stanford-futuredata/ColBERT).

This token-level matching can preserve exact phrases, dispersed concepts, and local instruction patterns that one pooled vector may blur.

It is still a learned semantic scorer, so it does not guarantee BM25's exact lexical coverage and should not be treated as a sparse-retrieval replacement without slice evidence.

It also cannot create a useful example that neither HNSW nor BM25 placed in its reranking pool.

ColBERT is a principled version of the user's token-level hypothesis, but it does not select tokens by raw transformer-attention outliers.

Raw attention scores vary by layer, head, length, and input, and they are not trained to preserve retrieval rankings, so mean-and-standard-deviation token pruning would create an unstable unvalidated index contract.

Model-supported projection, learned token filtering, pooling, residual quantization, or document-side binarization are safer compression hypotheses because each can be measured against uncompressed MaxSim.

## Current Morgott baseline

The current full-row growth candidate uses PPLX Embed 4B at 256 dimensions, four Faiss HNSW indexes with `M=32`, `efConstruction=200`, `efSearch=1024`, top-160 over-retrieval, exact float32 rescoring, and a final top 20 per label.

Its persisted HNSW indexes and row maps occupy 939.3 MiB, its separate bank occupies about 721 MiB, and measured runtime RSS is about 1.15 to 1.17 GiB.

At four workers it measured 55.476 ms p95 and 167.2 queries per second for dense retrieval.

The locked hybrid adds partitioned Unicode BM25, 2:1 dense-to-sparse RRF, deterministic balanced selection, and an exact dense fail-soft fallback.

On the completed 110-query comparison, HNSW alone reached 93.182% recall and 0.249% FPR, while hybrid reached 93.636% recall and 0.124% FPR.

Those favorable hybrid point estimates did not pass the predeclared paired material-gain gate because their confidence intervals included zero.

The hybrid nevertheless changed 59 packets, selected 47 slots absent from the saved dense top 20, and rescued three of six dense packet failures, which makes ordering the candidate union a real and bounded question.

BM25 plus its 1,000 ms execution budget measured 510.972 ms p95, 663.568 ms p99, 21.220 queries per second at four workers, and zero timeouts across 330 confirmation searches.

RRF plus selection measured only 0.802 ms p95, so replacing RRF can improve ranking quality but cannot remove the SQLite candidate-generation latency.

These figures and their qualifications come from the [current retrieval evidence ledger](retrieval-model-selection-research-20260817.md).

## Core ColBERT engines and scale evidence

[ColBERTv2](https://arxiv.org/abs/2112.01488) introduced denoised supervision and residual compression that reduced its reported late-interaction index footprint by 6 to 10 times.

Its encoding stores a four-byte centroid identifier plus either 16 residual bytes at one bit per 128 dimensions or 32 residual bytes at two bits per 128 dimensions, for approximately 20 or 36 bytes per stored token before surrounding structures.

The paper reports an approximately 154 GiB uncompressed MS MARCO passage index, a 16 GiB one-bit index, and a 25 GiB two-bit index for roughly nine million passages.

[PLAID](https://arxiv.org/abs/2205.09707) prunes documents by centroid interaction before decompressing and applying exact MaxSim, and it reported speedups up to 45 times on CPU and seven times on GPU over vanilla ColBERTv2.

PLAID reported 101.3 ms at top 1,000 on eight CPU cores and 38.4 ms on a Titan V for 8.8 million passages, excluding query encoding.

PLAID reported 181.9 ms at top 100 and 251.3 ms at top 1,000 on eight CPU cores for 138.4 million passages, while its GPU ran out of memory at top 1,000.

Those numbers used dual Xeon Gold 6132 servers and are evidence of scalability rather than predictions for Morgott's 2-vCPU target.

A later [PLAID reproducibility study](https://arxiv.org/abs/2404.14989) found its latency-quality frontier sensitive to the three pruning parameters and found ColBERTv2 reranking over lexical candidates competitive at low-latency operating points.

[WARP](https://arxiv.org/abs/2501.17788) and its [official repository](https://github.com/jlscheerer/xtr-warp) optimize the XTR retrieval objective for CPU search and recommend a GPU for index construction.

WARP is not a drop-in speed switch for arbitrary ColBERT checkpoints because its candidate generation and scoring are designed around XTR-trained representations.

The Apache-2.0 [Google XTR checkpoint](https://huggingface.co/google/xtr-base-en) is English-only and useful as an engine research control, but a 2026 [independent replication](https://arxiv.org/abs/2605.00646) did not reproduce its claimed overall effectiveness advantage.

## Practical model shortlist

| Model | Contract | License and serving | Morgott judgment |
| --- | --- | --- | --- |
| [`lightonai/mLateOn`](https://huggingface.co/lightonai/mLateOn) | 307M parameters; 128 dimensions per token; query and document length up to 8,192; MaxSim; multilingual and code training | Apache-2.0; local PyLate, Sentence Transformers, ONNX, FastPlaid | Primary exact-rerank quality anchor because it preserves long live queries and has the strongest current first-party open-model evidence among the reviewed practical checkpoints |
| [`perplexity-ai/pplx-embed-v1-late-0.6b`](https://huggingface.co/perplexity-ai/pplx-embed-v1-late-0.6b) | 596M parameters; 128 dimensions per token; 32 returned query vectors; approximately 512 document-token contract; MaxSim; multilingual benchmark evidence | MIT; local Sentence Transformers or PyLate; custom code must be pinned and reviewed | Relevant family control, but its fixed 32-token query representation risks discarding late prompt-injection evidence and its size weakens the CPU case |
| [`lightonai/LateOn`](https://huggingface.co/lightonai/LateOn) | 149M parameters; 128 dimensions per token; English; ModernBERT; MaxSim | Apache-2.0; local PyLate, ONNX, FastPlaid | Strong English efficiency candidate after mLateOn, but not the first test because Morgott needs multilingual and long-query robustness |
| [`answerdotai/answerai-colbert-small-v1`](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | 33M parameters; 96 dimensions per token; query length 32; recommended document length 512; English | Apache-2.0; local Sentence Transformers, RAGatouille, Stanford ColBERT, or Qdrant FastEmbed | Useful CPU lower-bound and serving canary, but its own card calls it a proof of concept and a negative result would not reject late interaction generally |
| [`LiquidAI/LFM2.5-ColBERT-350M`](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | 353M parameters; 128 dimensions per token; query length 32; document length 512; 11 languages | LFM Open License v1.0; local PyLate and GGUF variants | Practical edge candidate, but the short-query contract and nonstandard license make it secondary for Morgott |
| [`jinaai/jina-colbert-v2`](https://huggingface.co/jinaai/jina-colbert-v2) | 559M parameters; 64, 96, or 128 dimensions per token; query length 32; document length 8,192; 89 languages | CC-BY-NC-4.0 weights; local or Jina API; commercial terms require review | Best verified hosted ColBERT-specific fallback, but licensing and query truncation make it a bounded research arm rather than the local default |
| [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | 568M parameters; dense, learned sparse, and multi-vector outputs; 8,192-token multilingual context; 1,024-dimensional token vectors | MIT; local FlagEmbedding | Broad unified-model control, but its token vectors are eight times wider than 128-dimensional ColBERT vectors and are not the efficient first experiment |

The 4B multimodal [Jina Embeddings v4](https://huggingface.co/jinaai/jina-embeddings-v4) can emit late-interaction vectors, but its Qwen Research License, model size, and image capability are unnecessary burdens for Morgott's text-only study.

Generic BEIR, MIRACL, or MLDR scores nominate these models but cannot decide Morgott's binary-reviewer cascade, adversarial slices, or latency envelope.

## Token counts and storage for this corpus

I performed a read-only tokenizer pass over every eligible row in the all-row bank without sending text to a provider or writing an artifact.

With the pinned PPLX late tokenizer and document prefix, rows average 35.019 raw document tokens, with p50 17, p90 96, p95 149, p99 237, and maximum 969.

Only 332 rows, or 0.0438%, exceed a 511-token PPLX late document payload after reserving its prefix token.

With the pinned mLateOn tokenizer, rows average 37.561 raw tokens, with p50 19, p90 100, p95 152, p99 245, and maximum 991.

No row exceeds mLateOn's 8,192-token contract.

The following calculations use PPLX late's conservative 35.019 tokens per row so that the formats can be compared on one token count.

They include token payloads only and exclude centroid tables, IVF postings, document maps, metadata, allocator slack, duplicated mutable shards, model weights, query tensors, and page cache.

| Representation | the all-row bank | 1,000,000 rows at the same token mean |
| --- | ---: | ---: |
| 128d float32, 512 bytes per token | 12.650 GiB | 16.698 GiB |
| 128d float16, 256 bytes per token | 6.325 GiB | 8.349 GiB |
| ColBERTv2 one-bit residual, about 20 bytes per token | 0.494 GiB | 0.652 GiB |
| ColBERTv2 two-bit residual, about 36 bytes per token | 0.889 GiB | 1.174 GiB |
| Vespa sign-binary document vector, 16 bytes per token | 0.395 GiB | 0.522 GiB |

Using mLateOn's measured 37.561-token mean raises the corresponding all-row payloads to 13.568 GiB float32, 0.530 GiB ColBERTv2 one-bit residual, and 0.424 GiB Vespa sign-binary.

The raw float formats plainly do not fit a 4 GiB co-resident service beside the existing bank, dense HNSW indexes, reviewer, model weights, and runtime overhead.

The compressed payload estimates may fit on disk, but no production claim is valid until an actual index reports total bytes, peak build RSS, warm serving RSS, load time, and page-cache behavior.

The original ColBERTv2 paper's MS MARCO indexes show that non-payload structures can be several additional GiB at larger scale, so payload arithmetic is a lower bound rather than a capacity plan.

## Serving choices

[Sentence Transformers 6 MultiVectorEncoder](https://www.sbert.net/docs/package_reference/sentence_transformer/models.html) is the least invasive library for exact candidate reranking because it loads the checkpoint directly without installing a retrieval service or PyLate's FastPlaid dependency.

[FastPlaid](https://github.com/lightonai/fast-plaid) provides CPU and GPU search, four-bit product quantization by default, centroid probes, full-score candidate controls, and a `freeze` operation that removes duplicate mutable shards after indexing.

[NextPlaid](https://github.com/lightonai/next-plaid) is the strongest reviewed local serving candidate if the experiment advances because it adds memory-mapped indexes, two-bit or four-bit quantization, incremental updates, metadata prefiltering, ONNX encoding, CPU and CUDA modes, and built-in FTS5 hybrid search.

NextPlaid is still a new independent service and should not be introduced merely to run an exact reranking experiment that MultiVectorEncoder can answer in process.

[Qdrant multivectors](https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/) natively support variable-height token matrices and MaxSim.

Qdrant's own [dense plus BM25 plus ColBERT tutorial](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/) disables HNSW for the multi-vector field with `m=0`, retrieves candidates through dense and sparse branches, and applies ColBERT only to rerank them.

That official design directly supports the proposed Morgott role and warns that HNSW over every token vector adds high RAM use and slow insertion without benefiting a rerank-only field.

Qdrant becomes reasonable if Morgott later needs one online service for updates, filtering, dense, sparse, and multivector fields, but it adds no scientific value to the first model test.

[Vespa's native ColBERT embedder](https://blog.vespa.ai/announcing-colbert-embedder-in-vespa/) supports phased MaxSim reranking, variable token tensors, document-side sign binarization from 128 floats to 16 bytes per token, and paged attributes for disk-backed storage.

Vespa is a credible eventual search-platform choice, especially if compressed phased ranking and disk paging are required, but it is too large an architectural decision for a single retrieval hypothesis.

An HNSW nearest-neighbor search over individual token vectors is not equivalent to the sum-of-MaxSim ColBERT score across every query token.

Morgott should therefore keep HNSW on the single-vector PPLX candidate generator and avoid building a token-level HNSW unless a separately defined approximation proves candidate and cascade parity.

## Remote APIs, privacy, and failure behavior

Jina exposes `jina-colbert-v2` through a [multi-vector embedding endpoint and a rerank endpoint](https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/).

The embedding endpoint returns 128- or 64-dimensional token vectors, truncates documents at 8,192 tokens, and always returns 32 query-token vectors including padding.

The rerank endpoint accepts a query plus candidate documents and is the cleanest hosted late-interaction comparison if a provider arm is later authorized.

LightOn's managed [search API](https://lighton.ai/api) combines dense, sparse, and late-interaction retrieval and currently advertises EUR 0.006 per query and under 200 ms p50.

That managed search product would confound encoder, candidate generation, fusion, and reranking, so it is not an interpretable first comparison against Morgott's frozen pool.

The current [OpenRouter embeddings API](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings) returns one flat `embedding` array per input rather than a ragged token matrix, and the reviewed catalog exposed no ColBERT multi-vector model.

The existing OpenRouter key therefore does not make a ColBERT experiment available through OpenRouter's present embedding schema.

Morgott's repository policy still requires every remote study to be explicit, bounded, provider-safe or privacy-filtered, frozen before calls, and recorded with a budget even though zero data retention is not a user requirement.

No remote API should receive the corpus during the first experiment because local weights are available and provider routing is not part of the hypothesis.

Every embedding cache must be keyed by model repository, immutable revision, tokenizer and preprocessing contract, input-text hash, input role, maximum length, output dimension, and numeric dtype.

A remote timeout, malformed shape, non-finite vector, revision mismatch, dimension mismatch, or score failure must return the exact previously-selected RRF hybrid packet without retrying through a different model or vector space.

The reviewer must remain fail-soft and advisory, and late-interaction output must never grant authority or become a blocking decision.

## Smallest bounded experiment

### Hypothesis

The experiment asks whether exact mLateOn MaxSim orders useful examples inside the frozen HNSW plus BM25 candidate union better than 2:1 RRF does.

It does not ask whether full-corpus ColBERT retrieval beats HNSW, whether Qdrant beats Faiss, or whether a hosted search product beats Morgott.

### Frozen inputs

Use the fresh source-and-time-held-out panel already required for the locked hybrid, then preserve one untouched confirmation block.

Freeze the current all-row bank hash, HNSW `efSearch=1024/top160` artifacts, exact-rescored dense top 20 per label, partitioned Unicode BM25 implementation, 1,000 ms sparse budget, candidate limits, lineage deduplication, 2:1 RRF control, balanced four-example selector, reviewer, prompt, threshold, concurrency, and seed.

Persist the union of dense and sparse candidates before RRF, together with branch ranks and provenance, and do not let ColBERT retrieve outside that union.

For each label in the request's fixed channel, the union contains the exact-rescored dense top 20 plus the first 50 unique-source-lineage BM25 results, so the branch-output ceiling before cross-branch ID deduplication is 70 candidates per label and 140 candidates per request.

Blindly adjudicate which union examples are actually useful for the query and which four-example balanced packets are viable before inspecting cascade outcomes.

### Arms

Arm A is the current HNSW-only control with its exact fallback behavior.

Arm B is the locked HNSW plus BM25 plus 2:1 RRF challenger.

Arm C encodes the exact same Arm B candidate union with pinned local mLateOn, computes exact uncompressed float32 MaxSim, ranks within each label, and passes the result through the unchanged balanced selector.

Arm C uses mLateOn's unmodified query and document encoders, retains every model-emitted non-padding token, applies no attention filtering or token pooling, truncates only at the model's 8,192-token contract, and compares candidates only within their fixed label partition.

Do not add AnswerAI, PPLX late, compression, pooling, PLAID approximation, weighted score fusion, or a provider to this first causal comparison.

If Arm C wins quality but misses target latency, one subsequent engineering screen may compare LateOn 149M and AnswerAI 33M against the same exact mLateOn rankings and cascade outputs.

### Candidate gates

The frozen union must contain at least one adjudicated useful balanced four-example packet for at least 95% of routed queries.

RRF must omit at least one adjudicated useful example from its selected packet on at least 10% of routed queries, because otherwise there is too little ordering error for any reranker to repair.

Arm C must retain at least 99% of adjudicated-useful candidates found by the union in its top 20 per label and at least 95% in every adequately sized critical slice.

Report candidate recall, nDCG, packet availability, selected-packet agreement, source diversity, lineage diversity, and results by label, channel, source, domain, language, length, mutation family, and obfuscation family.

### Cascade gates

Advance Arm C only if its recall improvement over Arm B is at least 1.0 percentage point and the paired 95% confidence interval lower bound is above zero while the FPR delta upper bound is at most plus 0.10 percentage points.

An alternative pass is a statistically supported FPR reduction with no recall loss, using the same paired interval and no-regression requirements.

No adequately sized critical slice may lose more than 3.0 recall percentage points, and neither terminal failures nor invalid packets may increase.

The untouched confirmation block must reproduce the direction of both recall and FPR effects before any serving work begins.

### Latency and reliability gates

Measure query encoding, candidate document encoding or cache lookup, MaxSim, selection, and total retrieval separately at one and four workers on the intended 2-vCPU and 4-GiB target shape.

The added late-interaction stage must stay at or below 100 ms p95 at four workers and keep end-to-end retrieval at or below the existing 1,000 ms feature budget.

The stage must have at most 1% timeouts or invalid results, deterministic packet hashes across three warm repeats, and byte-identical fallback packets for every injected failure.

The test must include cold start, p50, p95, p99, throughput, CPU saturation, peak RSS, and model-load time rather than quoting GPU or high-core-count vendor numbers.

### Memory and cost gates

The experiment may cache only the distinct candidate-document tensors required by the frozen panel and must report their actual bytes.

Peak co-resident RSS must remain below 3.5 GiB so that the nominal 4 GiB deployment retains at least 512 MiB headroom.

If model weights plus candidate tensors violate that bound, the result may continue as an offline quality study but cannot be called co-resident production-feasible.

The primary local experiment has a provider-spend ceiling of zero and should record wall time, CPU-hours, GPU-hours if any, energy-visible utilization, and downloaded model bytes.

Any later hosted Jina arm requires a separately frozen text count, request count, quoted price, absolute spend ceiling, one-retry maximum, 80% budget stop, and projected cost per routed request.

### Optional fail-soft integration contract

An inconclusive small-panel quality result does not prohibit an off-by-default experimental integration if the architecture, licensing, determinism, latency, memory, and fallback gates pass.

The preferred model remains `lightonai/mLateOn` at immutable revision `edd378f99593c0ac8a15518b97ad89786b02685e`, with every model, tokenizer, configuration, and preprocessing file hashed into the experiment manifest.

The executed loader is Sentence Transformers 6.0.0 `MultiVectorEncoder`, which officially converts the checkpoint's legacy PyLate modules without importing PyLate.

A normal PyLate 1.5.0 installation was rejected because its mandatory FastPlaid dependency pins Torch 2.9.0 and would replace the repository's locked Torch 2.13.0 runtime.

Production-shaped loading should resolve and hash the snapshot once, then pass its local directory with `local_files_only=True` and `trust_remote_code=False` so runtime startup cannot fetch mutable code or weights.

The model repository contains no custom Python module or `auto_map`, so `trust_remote_code=True` is neither required nor permitted for this contract.

The executed dependency cell is Sentence Transformers 6.0.0, Transformers 5.14.1, Torch 2.13.0 with CUDA 13.0, and NumPy 2.4.6.

The [pinned mLateOn configuration](https://huggingface.co/lightonai/mLateOn/blob/edd378f99593c0ac8a15518b97ad89786b02685e/config_sentence_transformers.json) declares query prefix `[Q] `, document prefix `[D] `, query length 8,192, document length 8,192, 128 output dimensions, MaxSim, no query expansion, no expansion-token attention, and an empty document skip list.

The frozen encoding call preserves the checkpoint's native token vectors with `normalize_embeddings=False`, matching the current first-party Sentence Transformers example and published reference scores.

The integration uses the exact Sentence Transformers preprocessing path with truncation disabled and rejects any role-prefixed sequence above the loaded 8,192-token limit before encoding.

The integration asserts those loaded values because generic tokenizer metadata is not the authoritative role-specific length contract.

For a single query, `encode_query` returns a tensor shaped `(Lq, 128)`, and `encode_document` returns a ragged list whose item `i` is shaped `(Ld_i, 128)`.

With query expansion disabled, `Lq` is the retained attended query length rather than a fixed padded length, while each `Ld_i` retains all attended document tokens because the checkpoint's skip list is empty.

The following is the executed Stage 0 local load and encode API.

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder(
    "/absolute/path/to/hash-verified-mLateOn-snapshot",
    device="cuda",
    local_files_only=True,
    trust_remote_code=False,
)

query_vectors = model.encode_query(
    query_text,
    convert_to_numpy=False,
    normalize_embeddings=False,
)
document_vectors = model.encode_document(
    candidate_texts,
    batch_size=32,
    convert_to_numpy=False,
    normalize_embeddings=False,
)
scores = model.similarity(query_vectors, document_vectors)
```

The executed implementation separately recomputes exact uncompressed float32 NumPy MaxSim over each ragged document and requires it to match `model.similarity` within `1e-4`.

Do not use pooling, embedding quantization, `rank.rerank`, FastPlaid, or another scorer in Stage 0 until each is separately shown to preserve these exact scores and stable-ID rankings.

Reject and fall back before encoding when a live query or candidate exceeds the role-specific 8,192-token total rather than silently right-truncating security-relevant suffixes.

Sort equal MaxSim scores by the immutable bank-row identifier so repeated runs produce the same per-label ranking and packet hash.

The integration must first materialize the normal hybrid candidate union of at most 70 candidates per label and 140 per request, while retaining the exact RRF packet before any late-interaction work begins.

It must then load locally cached mLateOn document tensors, encode the query once, compute unpooled MaxSim in float32, rerank independently inside each label, and invoke the unchanged balanced selector.

The initial cache should contain only documents reached by the frozen evaluation and target-shaped canary panels, stored as a hash-bound ragged float32 tensor file plus a uint64 offset table and row-to-content-hash map outside Git.

For live experimental use, a missing cache entry must return the saved RRF packet immediately and may enqueue that document for local offline encoding, so a cache miss never adds encoder latency to the request or changes its result.

A complete float32 mLateOn document store is optional rather than a launch requirement and would require approximately 13.568 GiB of token payload for the current bank before offsets and metadata.

The research replay applies a post-execution 1,000 ms fail-soft threshold, while production advancement still requires four-worker p95 at or below 100 ms when every candidate is warm, total retrieval within 1,000 ms, and peak co-resident RSS below 3.5 GiB.

On a timeout, cache miss, model-load failure, malformed tensor, non-finite score, label mismatch, revision mismatch, or selector error, the result must preserve the RRF status, failure, selected IDs, and candidate IDs exactly.

The mode should be exposed only through an explicit research flag or shadow cohort, should log both RRF and ColBERT packet hashes, and should remain advisory and non-blocking.

Default promotion still requires the prospective cascade and untouched-confirmation gates because optional integration measures operational behavior but does not convert an underpowered or reused benchmark into quality proof.

### Stop rules

Stop before reviewer calls if useful balanced-packet coverage is below 95%, because candidate generation rather than ordering is the bottleneck.

Stop before default deployment work if exact MaxSim fails the cascade gate, because compression or a vector database cannot rescue an ineffective scorer.

Do not enable even the optional experimental path if the target-shaped reranker cannot meet its latency, reliability, memory, and exact-fallback gates.

## Only if exact reranking wins

Build an exact float32 MaxSim reference over the deterministic curated-screen bank, then compare NextPlaid or FastPlaid at four bits, two bits, and at most one documented token-pooling setting.

Require at least 99% top-20 recall against exact MaxSim, at least 99% selected-packet parity, no cascade regression, and the same critical-slice and fail-soft gates.

Report actual disk bytes, peak build RSS, build time, warm RSS, load time, cold start, p50, p95, p99, throughput, and results at one and four workers.

Advance to the source-lineage bank before the duplicate-heavy all-row bank, because Morgott already found the lineage bank more accurate, faster, and more reliable for the incumbent retriever.

Build the complete all-row multi-vector index only if the lineage bank leaves measured useful-candidate coverage on the table or the expected production corpus genuinely requires all rows.

If a production service is then warranted, benchmark NextPlaid first, Qdrant only when unified online vector operations are valuable, and Vespa only when broader search-platform consolidation and paged compressed tensors justify its operational surface.

## Executed result on 2026-08-19

The repository now contains an off-by-default local mLateOn reranker over the frozen ef1024 HNSW plus first-eight BM25 candidate union.

It is an experiment CLI and does not alter maintained inference, send corpus text to a provider, build a full-corpus token index, or call the reviewer.

Stage 0 verified all 13 required files from revision `edd378f99593c0ac8a15518b97ad89786b02685e`, whose aggregate manifest hash is `38b00c76c345661e34883d31c0b634cabb03a63c385c02c058b5766fb07e8b07`.

Exact NumPy MaxSim matched Sentence Transformers within `9.54e-7` and the model card's four published scores within `4.20e-5`.

The model loaded in 6.60 seconds, the repeated warm four-document canary took 23.52 ms, peak process RSS was 2,604,988 KiB, and peak CUDA reserved memory was 1,298,137,088 bytes.

The 110-unit consumed-panel replay rebuilt first-eight BM25 with the separate 1,000 ms SQLite budget and observed zero sparse timeouts, nine sparse fallbacks, and the same one pre-existing terminal RRF packet failure.

The in-memory candidate cache encoded 8,204 unique documents and 465,870 token vectors into 238,525,440 bytes in 64.01 seconds, with zero failed candidates.

ColBERT changed 109 of 110 selected packets and preserved the one failed RRF decision through its fail-soft path.

Across the 436 selected examples from successful packets, 214 were present only in the saved dense branch, 163 only in the sparse branch, and 59 in both branches.

The new packet retained a mean of 1.248 of the four RRF-selected examples, so the reranker is not a cosmetic tie breaker and requires prospective downstream validation before default use.

Sequential local added latency was 12.36 ms p50, 235.21 ms p95, 423.43 ms p99, and 995.54 ms maximum.

The longer untrusted-content queries dominated the tail: query length was 18 tokens at p50 and about 1,025 tokens at p95, with a 0.733 Pearson correlation between token count and reranker latency.

A row-matched local component estimate using the saved 9.74 ms HNSW p95, measured BM25 time, and ColBERT time was 378.89 ms p95, but one request reached 1,005.28 ms and this was not an end-to-end concurrent service test.

Replay peak process RSS was 2,627,984 KiB, peak CUDA allocation was 2,162,881,024 bytes, and peak CUDA reservation was 2,814,377,984 bytes.

The implementation therefore passes checkpoint integrity, reference-score parity, deterministic scoring, cache coverage, and fail-soft correctness, but it fails the predeclared 100 ms added-stage latency target and has not been tested on the 2-vCPU, 4-GiB deployment shape.

The correct status is integrated as an optional research challenger, not selected for maintained or default inference.

## Final answer

ColBERT is worthwhile for Morgott as an optional fail-soft reranker of the promising dense-plus-BM25 union rather than as a replacement for BM25 or a second million-document first-stage index.

The local mLateOn integration now exists and passes its correctness canary, while default promotion remains blocked by the measured latency tail and the lack of prospective downstream evidence.
