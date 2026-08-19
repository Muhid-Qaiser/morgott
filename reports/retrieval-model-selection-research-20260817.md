# Retrieval-assisted DeepSeek review research

Research snapshot: 2026-08-17.
Execution updates: 2026-08-18 and 2026-08-19.

Current implementation decisions are superseded by [retrieval-assisted-reviewer-findings-20260819.md](retrieval-assisted-reviewer-findings-20260819.md).
This file remains the chronological experiment record.

## Decision

Morgott should study retrieval-assisted in-context examples, but it should not assume that RAG, hybrid fusion, a reranker, HNSW, a vector database, GraphRAG, or any named model improves the classifier.

The completed study confirmed a full-cascade quality gain for dense PPLX examples on the frozen dev-test, but it did not establish stable production latency and makes no maintained inference change.

The first shippable candidate is intentionally small:

1. Build one immutable, leakage-audited, train-only example bank.
2. Run local SQLite FTS5 sparse retrieval and one remote dense embedding request in parallel.
3. Search the dense vectors locally.
4. Compare sparse-only, dense-only, and RRF hybrid retrieval.
5. Deterministically select two positive and two negative, source-diverse examples from the same input channel.
6. Give those examples to the already-selected DeepSeek reviewer while keeping its model, binary output contract, and threshold fixed.
7. Fall back to the current no-example reviewer on any retrieval failure.

The study should begin with exact dense search.
HNSW should be added only if a full-bank concurrency benchmark shows that exact search cannot meet the feature's total p95 latency budget.
Qdrant should not be the initial deployment because the bank is immutable and the maintained service does not currently need independent scaling, online writes, or complex payload filters.

## What this is and is not

This is retrieval-assisted in-context classification, sometimes called case-based prompting.
It is not conventional factual RAG because the retrieved rows are labeled analogies, not evidence that logically entails the correct label.

The embedding model supplies semantic candidate retrieval.
BM25 supplies lexical candidate retrieval.
RRF combines rankings that have incompatible score scales.
A reranker can reorder a small candidate set.
Exact search, HNSW, IVF, and vector databases are alternative ways to search stored dense vectors.
They do not make the vectors more semantic.

The only result that matters for promotion is the full Morgott cascade result.
Generic embedding leaderboards, nearest-neighbor label agreement, and reviewer-only scores are screening diagnostics.

## Locked study decisions

The decisions already established with the owner are:

- Improve DeepSeek's binary instruction-subversion classification result.
- Use the full maintained cascade as the primary evaluation unit.
- Keep the selected DeepSeek reviewer fixed.
- Require at least a 1.0 percentage-point recall gain.
- Allow at most a 0.25 absolute percentage-point FPR increase.
- Allow no critical-slice recall loss greater than 3 percentage points.
- Keep the whole feature's added p95 latency below 1 second on requests that already reach DeepSeek.
- Count retrieval, optional reranking, and longer DeepSeek prefill in that latency delta.
- Compare a curated bank with the full prompt-eligible training bank.
- Use validation for selection and freeze 12,000 previously unconsumed dev-test rows for one final paired confirmation.
- Keep the DeepSeek threshold fixed rather than recalibrating it around the new prompt.
- Fall back to the current no-example reviewer on timeout, malformed output, invalid index state, or provider failure.
- Keep remote research spend at or below $50.
- Allow remote embeddings, including live remote query embeddings.
- Do not require zero data retention.
- Shadow a winner before making it the maintained default.

The unresolved provider-policy decision is whether providers may train on corpus rows or live inputs.
The recommended default is to allow documented logging and retention while still prohibiting training use.

## Data boundary

Validation and dev-test rows must never become retrieval memory.
Putting them in the index would leak answers into the prompt and invalidate the evaluation.

The full bank means the full prompt-eligible train bank, not every row that happens to exist in `data/`.
Eligibility should reuse the maintained model-data rules and add prompt-specific constraints:

- `data_role` is `train`.
- The instruction-subversion label is a high-confidence integer 0 or 1.
- The row is not weak, unverified, uncertain, or label-conflicted.
- The input channel is `direct_user` or `untrusted_content`.
- Positive rows contain a maintained instruction-subversion security tag.
- Exact, strict-normalized, near-duplicate, group, and lineage leakage checks pass.
- The example fits the frozen prompt budget without silent truncation.
- The row's license and provenance remain attached to its identifier.

Long positive documents should not be split into many positive examples unless a trusted annotation identifies the positive span.
Artifact-level labels do not imply that every chunk contains an attack.
The safer first bank excludes rows that do not fit the example limit instead of manufacturing noisy chunk labels.

The curated bank should be a deterministic source, label, channel, and subtype-balanced cap over the same eligible population.
It is a quality and latency candidate, not a hand-picked showcase.

## Prompt design

The current DeepSeek baseline must remain an exact arm in every comparison.

The smallest useful prompt tournament is:

1. Current reviewer with no added context.
2. Current reviewer plus trusted task metadata, where that metadata genuinely exists.
3. Current reviewer plus four fixed, balanced examples.
4. Current reviewer plus four sparse-retrieved examples.
5. Current reviewer plus four dense-retrieved examples.
6. Current reviewer plus four RRF-hybrid examples.
7. The best retrieval arm plus a reranker, only as an ablation.

Two positive and two negative examples prevent a nearest-neighbor class majority from silently steering every answer.
Examples should match the trusted input channel and should be diverse by source and lineage.
The study should include an example-order reversal diagnostic because in-context classification is sensitive to demonstration order and label recency.

Retrieved attack examples remain untrusted text.
They must never be inserted as system-level instructions.
The prompt must explicitly identify them as inert labeled data, preserve the original text-to-classify as the final item, and retain strict JSON output validation.

Research on retrieval-selected in-context examples reports gains over random examples, but it also reports label, domain, and order biases.
Those findings justify the experiment and the controls rather than proving that it will help Morgott.
See [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804), [Learning To Retrieve Prompts for In-Context Learning](https://arxiv.org/abs/2112.08633), and [Mitigating Label Biases for In-context Learning](https://arxiv.org/abs/2305.19148).

## Trusted task metadata

Trusted task context may be more valuable than retrieval for quoted or analyzed attacks, so it must be measured separately.

It cannot be a free-form claim that an unauthenticated API caller can set.
It is trusted only when an authenticated runtime supplies it outside the attacker-controlled text boundary.
If it is absent or untrusted, the reviewer must behave exactly as it does today.

This follows Morgott's existing rule that provenance and authority labels come from trusted runtime metadata, not from text being classified.

## Sparse retrieval

The first sparse implementation should use Python's built-in SQLite interface and SQLite FTS5.
FTS5 already provides BM25 and both Unicode word and trigram tokenizers.
The official [SQLite FTS5 documentation](https://www.sqlite.org/fts5.html) documents its BM25 function and trigram tokenizer.

Compare two sparse indexes on validation:

- Unicode word BM25 for ordinary lexical and phrase overlap.
- Trigram BM25 for obfuscations, fragments, and unusual token boundaries.

Pick one sparse branch for production unless combining them yields a measured cascade gain.
Do not start with SPLADE or a custom learned-sparse index.
Those add another model and index format before Morgott has shown that ordinary BM25 misses useful candidates.

Raw attacker text must never be interpolated directly into FTS query syntax.
The query builder must parameterize SQL, escape FTS terms, cap the number of terms, and handle a no-term query as a valid empty sparse result.

## Dense model shortlist

No vendor benchmark is Morgott evidence.
The shortlist is designed to cover the deployment tradeoffs with few candidates.

### Primary remote family: Voyage 4

Use `voyage-4-large` as the quality-oriented document-index candidate and test remote `voyage-4-large`, `voyage-4`, and `voyage-4-lite` query encoders against that index.
Also resource-test local `voyage-4-nano` as a possible query encoder.

Voyage documents that the four models share an embedding space, so this comparison does not require rebuilding the document index.
The models support 32K input and 256, 512, 1,024, and 2,048 dimensions.
The local Nano model is Apache-2.0 and has roughly 180M non-embedding plus 160M embedding parameters.
See the official [Voyage 4 Nano model card](https://huggingface.co/voyageai/voyage-4-nano), [Voyage pricing](https://docs.voyageai.com/docs/pricing), and [Voyage FAQ](https://docs.voyageai.com/docs/faq).

The first OpenRouter call must contract-test query/document input types, requested dimensions, encoding format, normalization, finite values, and exact output length.
OpenRouter currently lists the models but does not advertise those provider-specific parameters in the model's `supported_parameters` metadata.
If OpenRouter does not preserve the required contract, use the direct Voyage endpoint rather than building an ambiguous index.

### Independent remote challenger: Perplexity Embed V1

Test `pplx-embed-v1-4b` at 256 dimensions as the independent quality challenger.
Test the 0.6B variant only as a latency or local-inference control, not as an assumed compatible query encoder for a 4B-built index.

Perplexity documents a 32K context, Matryoshka dimensions down to 128, native INT8 or binary output, and prices of $0.03 and $0.004 per million tokens for the 4B and 0.6B models.
See the official [standard embeddings documentation](https://docs.perplexity.ai/docs/embeddings/standard-embeddings) and [technical report](https://arxiv.org/abs/2602.11151).

### Tie-breaker only: Gemini Embedding 2

Gemini Embedding 2 is a reasonable third-vendor tie-breaker if Voyage and Perplexity are inconclusive.
It supports 8,192 text tokens and dimensions from 128 to 3,072.
Its paid text price is $0.20 per million tokens.
See the official [Gemini embedding guide](https://ai.google.dev/gemini-api/docs/embeddings) and [pricing](https://ai.google.dev/gemini-api/docs/pricing).

Multimodality is irrelevant to the current text classifier, so it is not a reason to prefer Gemini.

### Local fallbacks

The first local candidates are Voyage 4 Nano and Snowflake Arctic Embed M v2.0 at 256 dimensions.

Arctic is an Apache-2.0, encoder-style model with 305M total parameters, 113M non-embedding parameters, 8,192-token context, and Matryoshka compression to 256 dimensions.
See the official [Snowflake model card](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0).

Local inference is not automatically faster in this deployment.
It competes with mmBERT on four CPUs, consumes memory, enlarges the image, and currently requires runtime support not installed by Morgott.
Run a resource canary before embedding a material bank.

### Models not in the first tournament

BGE-M3, Qwen3-Embedding-0.6B, GTE multilingual base, Nomic Embed v2, and larger local rerankers are legitimate research controls but redundant in the first tournament.
They should enter only if the shortlisted models expose a specific failure, such as learned-sparse candidate recall, language coverage, or local latency.

## Dense search: exact first, HNSW if measured

The full prompt-eligible bank is roughly at the one-million-vector scale.
At 256 dimensions, a flat float32 matrix is roughly 1.1 GiB and a float16 matrix is roughly 0.55 GiB before metadata.
An exact dot-product scan is the correctness baseline and may fit the one-second feature budget on optimized CPU code, but that must be tested under the service's current concurrency of four.

The exact baseline is also required to measure any approximate index's recall.

If exact full-bank search fails the latency or CPU-headroom gate, test an in-process Faiss HNSW index before introducing a vector service.
Faiss documents `IndexHNSWFlat`, scalar-quantized HNSW, and the `M`, `efConstruction`, and `efSearch` quality-memory-speed controls.
Its current guidance describes Flat as the exact baseline and HNSW as a fast, accurate choice when RAM permits.
See [Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) and [Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

For each dense model, the HNSW gate should require:

- Recall@20 of at least 0.98 against exact top-20 results on the fixed validation queries.
- No material loss in the full-cascade quality metrics.
- The total feature p95 remains below the one-second budget at concurrency four.
- Resident memory leaves safe headroom for mmBERT, Python, the sparse index, and request concurrency.
- The index is immutable, version-pinned, and hash-verified at load time.

The initial HNSW screen can sweep a small fixed set such as `M=32` with `efSearch` 64, 128, and 256.
Parameters should be selected by the stated gates rather than copied from an unrelated benchmark.

## Why not Qdrant initially

Qdrant is a capable vector database that uses filterable HNSW, payload indexes, quantization, and optional on-disk storage.
Those features become useful when vectors change online, filters are numerous or dynamic, several applications share an index, replicas must scale independently, or the index no longer fits a process comfortably.
See Qdrant's official [indexing documentation](https://qdrant.tech/documentation/manage-data/indexing/) and [overview](https://qdrant.tech/documentation/overview/).

Morgott's first bank is immutable and can be split into four fixed partitions by channel and label.
It therefore does not need online vector writes or complex filter-aware graph traversal.

Running Qdrant would add another process or managed service, network and authentication policy, backup and upgrade work, and a second failure boundary.
Qdrant's own installation guidance distinguishes local development from production deployments such as Qdrant Cloud or Kubernetes.
See the official [installation guide](https://qdrant.tech/documentation/installation/).

Qdrant should be reconsidered only when an in-process exact or HNSW index measurably fails an operational requirement that Qdrant solves.

Azure AI Search is another later option because it natively combines BM25, HNSW, and RRF.
It is also a separate managed service and is therefore outside the first one-service design.

## Fusion

Always measure sparse-only, dense-only, and hybrid retrieval.
Do not keep two retrieval branches merely because hybrid search is fashionable.

RRF with the conventional constant near 60 is the untuned baseline because it combines ranks without pretending BM25 and cosine scores are calibrated to each other.
The original [RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) and current [Azure RRF documentation](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) describe the method.

RRF is not universally optimal.
If the union top-20 contains useful examples but RRF orders them poorly, compare one validation-tuned weighted fusion before paying for a reranker.
Do not tune a large fusion search space.

## Do we need a reranker?

Probably not, unless the experiment demonstrates an ordering problem.

A reranker is sequential after candidate retrieval, so its entire latency is on the critical path.
It also repeats the query across candidate documents and creates another remote failure mode.

Include a no-rerank versus rerank arm in the study, but ship the reranker only if it improves the full cascade rather than just NDCG or a vendor relevance score.

The first quality arm is Voyage `rerank-2.5` over the same fused top-20 candidates.
If it helps but misses latency, test `rerank-2.5-lite` and then Cohere Rerank 4 Fast as lower-latency remote alternatives.
Voyage documents a 32K pair context, an 8K query limit, and $0.05 per million processed tokens for `rerank-2.5`.
See [Voyage reranker documentation](https://docs.voyageai.com/docs/reranker) and [pricing](https://docs.voyageai.com/docs/pricing).
Cohere documents its Fast and Pro variants in the official [Rerank documentation](https://docs.cohere.com/v2/docs/rerank).

A local reranker should be tested only if a remote reranker first proves that reordering has enough quality value to justify its CPU and memory cost.

The reranker earns production inclusion only if:

- The paired full-cascade validation result improves materially over the same hybrid candidates without reranking.
- The improvement survives label-order and source-held-out diagnostics.
- The entire retrieval feature still meets the p95 budget.
- Failure cleanly returns to the current no-example reviewer.

## Long documents

Do not silently truncate a long artifact to an embedding model's context window.

For a DeepSeek review of one mmBERT window, retrieve using that same normalized window.
For the current untrusted multi-window full-context review, retrieve using the already-computed highest-scoring mmBERT window while still sending the full normalized artifact to DeepSeek.

This keeps one dense query on the critical path and avoids missing an injection merely because it occurs beyond an embedding cutoff.
Compare top-one with top-two mmBERT-window rank fusion only if top-one shows measured localization failures.

## Retrieval and vector validation

Every index must be bound to:

- The data manifest digest.
- The exact eligibility and normalization contract.
- The embedding provider, model, and immutable revision.
- Query and document input types.
- Vector dimension, encoding, normalization, and similarity metric.
- The ordered row identifiers and index-byte digest.

Every remote response must have the expected number of finite vectors and the exact expected dimension.
Unexpected dimensions, NaNs, missing rows, truncation, alias drift, or digest mismatch invalidate retrieval for that request or index.

No prompt bodies, raw provider responses, or embedding vectors should enter normal application logs.
Shadow telemetry needs only hashes, row identifiers, ranks, latency, model identity, branch status, and fallback reason.

## Evaluation sequence

### Phase 0: contract and resource canary

Use a small, frozen set spanning channels, labels, languages, lengths, quoted attacks, obfuscations, and long documents.

Reject a candidate before material indexing if it cannot satisfy its dimension and normalization contract, local RSS headroom, query p95, or provider reliability requirements.

### Phase 1: cheap retrieval screen

Build the deterministic source-balanced screening bank for every surviving embedding candidate.

Use validation-only queries to measure:

- Exact and mutation-stable top-k behavior.
- Same-channel coverage.
- Source and lineage diversity.
- Weighted k-nearest-neighbor classification as a diagnostic.
- Source-masked and source-held-out behavior.
- Query latency, index size, build cost, and failure rate.

Advance at most two dense candidates.

### Phase 2: prompt and reranking ablation

On one frozen validation panel, run the current reviewer, trusted-task-only, fixed-example, sparse-only, dense-only, hybrid, and hybrid-plus-rerank arms.

Use identical rows and paired group-aware bootstrap intervals.
Record reviewer-only outcomes as diagnostics and reconstruct the full cascade for the primary result.

### Phase 3: curated versus full bank

Build the curated and full prompt-eligible train indexes only for the winning retrieval configuration.

Benchmark exact search first.
Add HNSW only if the full-bank exact benchmark fails.

### Phase 4: one frozen confirmation

Run the chosen configuration once on the frozen 12,000-row dev-test confirmation panel.
Do not adjust the prompt, bank, fusion, threshold, model, or reranker after seeing that result.

### Phase 5: shadow rollout

Shadow the winner with retrieval unable to alter authoritative runtime decisions.
Confirm real p50 and p95 latency, provider and index failure rate, cost, example stability, memory, and fallback behavior.

Promotion still leaves all learned output advisory and every side effect behind the deterministic reference monitor.

## Promotion gates

A winner must satisfy every locked quality and latency gate:

- Recall improves by at least 1.0 percentage point on the full cascade.
- Absolute FPR increases by no more than 0.25 percentage point.
- No critical-slice recall falls by more than 3 percentage points.
- The whole feature adds less than 1 second p95 on requests that reach DeepSeek.
- Research remote spend stays at or below $50.
- Retrieval failures return the exact current no-example reviewer behavior.

The report should also show paired uncertainty intervals, direct and untrusted channels, quoted-analysis controls, long clean tasks, security and finance discussion, obfuscations, indirect documents, source macro results, source-held-out results, and retrieval failure simulations.

If fixed examples or trusted task metadata pass while dynamic retrieval does not, ship the simpler winner.
If no arm passes, do not ship retrieval.

## GraphRAG

Do not use GraphRAG for this study.

Microsoft GraphRAG extracts entities, relationships, claims, communities, and LLM-generated community reports for relational and global questions over document collections.
Morgott needs nearest labeled examples for a binary classifier, not multi-hop entity reasoning.
Its source and lineage relations are deterministic metadata for leakage control, not a knowledge graph to infer.

GraphRAG would add expensive LLM indexing, generated summaries, and new failure modes without solving the present selection problem.
Microsoft's own [GraphRAG overview](https://github.com/microsoft/graphrag) warns that indexing can be expensive, and its [indexing dataflow](https://github.com/microsoft/graphrag/blob/main/docs/index/default_dataflow.md) shows the entity, relationship, and community pipeline.

Reconsider a graph only if the product later asks multi-hop questions whose answer depends on explicit relationships across documents.

## zkg and output verification

Do not put `leochlon/zkg` on this one-bit classification path.

The current project judges each cited factual claim twice, with and without cited evidence, and proves that its release bit matches committed verifier confidences.
It does not prove that an instruction-subversion label is correct.
Morgott's retrieved examples are analogies, not citations that entail the label.
The project also requires two verifier calls per claim, directly conflicting with the latency target.
See the current [zkg README](https://github.com/leochlon/zkg).

The right verification for this classifier is strict schema validation, logprob validation, fixed failure behavior, paired labeled evaluation, shadow monitoring, and the deterministic reference monitor.
If Morgott later produces cited factual claims, zkg may deserve a separate offline or asynchronous study for evidence dependence.

## Observed benchmark evidence

The disposable runner is implemented under `experiments/retrieval_assisted_reviewer/` and does not change maintained inference or advisory authority.
It freezes a train-only retrieval bank, a 1,024-row validation panel, and a sealed 12,000-row dev-test panel.
It keeps the DeepSeek reviewer, prompt contract, threshold, and cascade fixed, and sends four balanced same-channel examples only as inert user-level JSON data.
Remote ledgers exclude raw prompts and responses, and one durable ledger enforces the $50 cap with a $2 reserve.

### Bank construction

The all-row bank contains every eligible training row after the license, sensitive-text, prompt-size, label-confidence, and panel-overlap gates.
The curated-screen bank is deterministically balanced across the available source, label, channel, and subtype strata.
The lineage bank keeps deterministic representatives and collapses correlated variants.
The all-row bank retains every eligible row for the requested scale comparison.

The all-row bank includes repeated source-lineage groups and one especially concentrated lineage.
That concentration is why the all-row arm is a comparison rather than an assumed improvement.

### Curated screening bank

Perplexity `pplx-embed-v1-4b` at 256 dimensions won the dense model screen.
Its exact-search p95 was 9.1 ms, and dense retrieval including the remote query embedding had 102 successes and 8 fallbacks across 110 review units.
Dense PPLX improved full-cascade recall from 71.36% to 89.55%, a paired gain of 18.18 percentage points with a 95% interval from 13.18 to 23.18 points.
FPR moved from 0.249% to 0.124%, with a paired 95% delta interval from -0.373 to 0 percentage points.
The example-order reversal diagnostic reached 90.45% recall with the same FPR, so the gain was not dependent on the selected label order.

Dense-only PPLX beat its trigram-RRF hybrid on recall, 89.55% versus 87.27%.
The hybrid also failed the latency gate in that run.
This supplied no evidence that a second retrieval branch or reranker would improve the cascade.

### Source-lineage comparison

The lineage bank removed every dense retrieval fallback on the same 110 review units.
Exact-search p95 was 19.1 ms, and dense retrieval including the remote query embedding had a 756.0 ms p95.
At four simultaneous exact-search workers, p95 was 43.6 ms, throughput was about 95 queries per second, and peak process RSS was 578 MiB.

Dense PPLX improved full-cascade recall from 69.55% to 93.64%, a paired gain of 24.09 percentage points with a 95% interval from 18.64 to 30.00 points.
FPR remained 0.249%, and no critical recall slice regressed.
The arm did not pass the locked promotion gate because its measured added reviewer p95 was 1.149 seconds, above the 1-second limit.
This miss cannot be repaired by HNSW because the local exact component contributes only tens of milliseconds.

### All-row comparison

The all-row bank had 104 successful dense retrievals and 6 fallbacks across the 110 validation review units.
Exact-search p95 was 65.1 ms, and dense retrieval including the remote query embedding had a 723.7 ms p95.
At four simultaneous exact-search workers, p95 was 170.3 ms, throughput was 23.6 queries per second, and peak process RSS was 2.0 GiB.
This real full-scale benchmark does not justify HNSW or Qdrant.

Dense PPLX improved validation recall from 71.82% to 92.73%, a paired gain of 20.91 percentage points with a 95% interval from 15.90 to 26.36 points.
FPR remained 0.249%, no critical recall slice regressed, and all reviewer retries were exhausted before the selection was sealed.
The marginal p95 difference passed at 0.490 seconds, although the paired latency-delta p95 was 7.17 seconds and demonstrates substantial provider noise.

The all-row bank did not beat the lineage bank on validation recall, retrieval reliability, local latency, or memory.
Its value is the requested full-data comparison, not evidence that retaining every variant is the best maintained design.

The full-scale sparse branches were also weak operationally.
Trigram BM25 had a 1.36-second retrieval p95 and 20 failures, while Unicode BM25 had a 0.92-second p95 and 22 failures.
When combined with dense PPLX, the trigram and Unicode RRF records had 22 and 24 failures respectively.
Those full-scale hybrid candidates did not receive post-selection DeepSeek quality calls, so the evidence does not claim that hybrid can never help.

### Frozen 12,000-row confirmation

The all-row dense winner was frozen before the dev-test was opened.
The unchanged local cascade routed 1,333 of the 12,000 artifacts to DeepSeek.
Dense retrieval succeeded for 1,321 units and used exact-baseline fallback for 12 units.
Its retrieval p95 was 621.0 ms, of which exact search contributed 57.3 ms.

Full-cascade recall improved from 89.07% to 94.81%, a paired gain of 5.73 percentage points with a 95% interval from 4.84 to 6.63 points.
FPR increased from 1.168% to 1.274%, a paired increase of 0.106 percentage points whose 95% upper bound was 0.223 points.
That remains inside the locked 0.25-point limit, and no critical recall slice regressed.

Direct-user recall improved from 88.36% to 94.99%, while direct-user FPR increased from 0.851% to 0.969%.
On the smaller untrusted-content slice, recall improved from 92.69% to 93.87% and FPR decreased from 23.13% to 22.39%.
The untrusted-content result has only 134 negatives and should not be treated as a precise production FPR estimate.

One dense reviewer job remained terminal after both allowed attempts, and 12 retrieval queries used the exact baseline prompt.
The marginal p95 comparison reported only 1.2 ms added latency, but the paired latency-delta p95 was 2.93 seconds.
This contradiction means the experiment confirms quality, not production latency.

The final ledger total was $18.71 including conservative embedding ceilings and settled reviewer cost, below the $50 cap with the $2 reserve intact.
No result changes maintained inference, blocking authority, the model registry, or the deterministic reference monitor.

## Completed follow-up experiments (2026-08-18)

The follow-ups below are post-selection diagnostics on already-open development rows.
They do not reopen the frozen dev-test decision or upgrade the evidence to prospective robustness.

### PPLX dimension ablation on the curated screening bank

The dimension study rebuilt a separate deterministic curated-screen bank.
The bank was byte-identical to the first curated bank, and the decompressed frozen panel rows were identical.
The already-paid 256-dimensional document vectors were reused only after the bank and vector hashes matched.
The 512-dimensional build required one resume after a transport timeout.
The 1,024-dimensional build required two resumes, one after HTTP 429 and one after a transport timeout.

All dimensions used fresh query embeddings and fresh paired DeepSeek calls in one isolated reviewer ledger.
Each dense arm succeeded on 102 of the 110 review units and used exact-baseline fallback on 8.

| Arm | Full-cascade recall | FPR | Exact-search p95 | Raw screen-bank vector memory | Isolated process peak RSS |
|---|---:|---:|---:|---:|---:|
| Baseline reviewer | 71.36% | 0.373% | n/a | n/a | n/a |
| PPLX 256d | **90.45%** | **0.124%** | **10.9 ms** | **48.8 MiB** | **347.6 MiB** |
| PPLX 512d | 89.55% | **0.124%** | 12.7 ms | 97.7 MiB | 444.7 MiB |
| PPLX 1,024d | 88.64% | 0.249% | 16.1 ms | 195.3 MiB | 640.7 MiB |

Compared directly with 256d, 512d changed recall by -0.91 percentage point with a paired 95% interval from -2.27 to 0 points and did not change FPR.
Compared directly with 256d, 1,024d changed recall by -1.82 points with a paired 95% interval from -4.09 to 0 points and increased FPR by 0.124 point with an interval from 0 to 0.373 points.
Remote query timing moved in the opposite direction from local search timing, which is further evidence that provider latency from these short runs is noisy.

The 256-dimensional arm is the winner.
Neither larger dimension improved full-cascade quality, both consumed more memory, and the 1,024-dimensional document build was less reliable.
No all-row 512d or 1,024d index was built.

### BM25 candidate contribution and locked hybrid follow-up

The candidate gate used the existing full-bank top-20 records before making another reviewer call.
Because the corpus has no passage-level relevance judgments, novelty alone was not called useful.
The diagnostic instead measured dense-missing candidates, source and lineage novelty, subtype agreement on the query's gold-label branch, and sparse coverage of dense failures.

Across the 88 validation units where full-bank dense and trigram BM25 both succeeded, trigram returned 2,911 candidate occurrences absent from dense top-20.
Of those occurrences, 2,344 came from source-lineage pairs absent from the corresponding dense list and 380 came from sources absent from that dense list.
On the query's gold-label branch, 966 of 1,368 dense-missing trigram candidates matched the query subtype proxy.
Trigram also succeeded for 2 units where the recorded dense query failed.
These are candidate-level proxies, not proof that the examples improve the reviewer, but they were sufficient to run the one predeclared fusion follow-up.

The only new setting over-retrieved 50 candidates per label, removed repeated source-lineage groups within each branch, and used RRF with a locked 2:1 dense-to-trigram weight and `k=60`.
No outcome-driven weight or candidate-count sweep was performed.
The new retriever succeeded on 108 of 110 units, changed 64 selected example packets relative to the recorded dense arm, and placed 13 BM25-only examples into selected prompt slots.
Its retrieval p95 was 1.518 seconds, including a 1.382-second sparse p95 and a 62.3 ms exact-dense p95, so it failed the feature latency budget before reviewer timing was considered.

Fresh baseline, dense, equal-weight trigram RRF, and weighted lineage-RRF reviewer calls were then run together on the same frozen validation rows.

| Arm | Recall | FPR | Retrieval fallbacks | Added end-to-end p95 | Gate result |
|---|---:|---:|---:|---:|---|
| Baseline reviewer | 72.73% | 0.249% | 0 | 0 | Control |
| Dense PPLX 256d | **94.09%** | 0.249% | 6 | 0.431 s | Pass |
| Equal-weight trigram RRF | 88.18% | 0.249% | 22 | 1.239 s | Fail |
| Weighted lineage-RRF | 93.64% | 0.249% | 2 | 6.588 s | Fail |

Compared directly with dense, equal-weight RRF lost 5.91 recall points with a paired 95% interval from -9.55 to -2.73 points.
The weighted lineage arm changed recall by -0.45 point with an interval from -2.73 to +1.82 points and did not change FPR.
Its reviewer-inclusive p95 was dominated by provider noise, but its retrieval-only p95 independently exceeded the locked budget.

BM25 therefore demonstrated candidate novelty but no full-cascade advantage.
Dense-only remains the winner, and there is no ordering gain that justifies paying for a reranker.

### Exact Faiss FlatIP benchmark

Faiss CPU 1.15.0 was installed ephemerally for this benchmark and was not added to the project dependencies.
The full all-row 256d bank was searched through four fixed channel-label partitions with `IndexFlatIP`.
The same 110 normalized query vectors were used for NumPy and Faiss.
The four-worker timing run repeated the panel three times for 330 request-level searches and attempted to limit each implementation to one native thread per request.
That limit is not established for the Faiss worker threads because `faiss.omp_set_num_threads(1)` was called in the parent thread, while Faiss documents that the OpenMP setting is thread-local.

| Exact implementation | Four-worker p50 | Four-worker p95 | Throughput |
|---|---:|---:|---:|
| NumPy ground truth | **155.8 ms** | **176.3 ms** | **31.4 QPS** |
| Faiss `IndexFlatIP` | 164.8 ms | 225.4 ms | 27.8 QPS |

Faiss matched 216 of 220 ordered top-20 label rankings and 219 of 220 top-20 sets, for mean Recall@20 of 99.977% against NumPy.
The four discrepancies had maximum float32 score differences no larger than 5.96e-7.
The only set difference exchanged two candidates whose NumPy scores were exactly tied at the rank-20 boundary.
FlatIP therefore failed the predeclared identical-order gate, but it did not demonstrate a meaningful retrieval-recall failure.
A future exact-backend contract should accept score-tolerant tied order and require candidate-set, selected-packet, and full-cascade parity.

The raw vector matrix was 739.8 MiB.
The measured process held 1.97 GiB after the current NumPy loader and 2.26 GiB with both NumPy and Faiss resident; the net allocator change is not a standalone Faiss memory estimate.
FlatIP built in 0.206 seconds and was slower than NumPy in this measured configuration.
That result does not establish that Faiss is universally slower because the benchmark used singleton queries, always timed NumPy first, retained both indexes in one process, omitted the CPU, BLAS, Faiss build, and SIMD metadata from its artifact, and did not verify per-worker OpenMP settings.
Faiss documents that singleton queries are a weak operating point compared with batched search and that OpenMP settings are thread-local in its [FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ#why-are-searches-with-a-single-query-vector-slow) and [threading guide](https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls).

`IndexHNSWFlat` was not tested.
The exact NumPy implementation remained below the predeclared local-search trigger on the benchmark machine, and FlatIP supplied no performance reason to replace it.
The test did not run on the intended two-vCPU, four-GiB service shape, so it cannot establish production headroom there.
HNSW is now justified as a growth benchmark, but not yet as the selected deployment.
The same evidence does not justify Qdrant or another vector service for this immutable bank.

## Prospective source-heldout confirmation (2026-08-18)

The pinned Apache-2.0 [WMT 2024 English-to-German prompt-injection suite](https://github.com/Avmb/adversarial_MT_prompt_injection/tree/0d2107adc2515193a39919b672979223b67dbc7c) had not been used for fitting, retrieval selection, prompt selection, threshold selection, or any earlier Morgott result.
This run reserves it as evaluation-only evidence, so it must not enter later fitting or selection.

The protocol assigned one of the source's five attack forms to every TruthfulQA question by a deterministic hash-balanced rule before any model outcome was read.
Every retained lineage contributed one clean source sentence and its assigned attacked counterpart under `untrusted_content`.
The preparation removed three complete pairs for normalized, strict, or near fit overlap, removed no pair for the sensitive-text screen, and dropped four additional pairs only to preserve exact attack-type balance.
The sealed panel therefore contains 810 matched pairs and 1,620 artifacts, with 162 pairs per attack type.
Raw source text is reloaded from content-addressed public files and is not retained in the evidence artifacts.

The local cascade sent 577 artifacts to review.
Each review query was embedded once, then searched against the already-frozen source-lineage index and all-row index.
The reviewer model, provider, prompt, threshold, failure behavior, and example count remained fixed.
The first DeepSeek pass had 18 transient HTTP failures across 1,731 calls; the one allowed retry recovered all 18, leaving zero terminal reviewer failures.

| Arm | Full-cascade recall | FPR | Retrieval fallbacks | Exact-search p95 | Added end-to-end p95 |
|---|---:|---:|---:|---:|---:|
| No-example baseline | 17.28% | 0.123% | 0 | n/a | 0 |
| PPLX 256d, all-row bank | 39.14% | 0% | 3 | 18.5 ms | 0.712 s |
| PPLX 256d, lineage bank | **43.33%** | **0%** | **0** | **11.4 ms** | 0.821 s |

Against the no-example baseline, the all-row bank gained 21.85 recall points with a paired 95% interval from 18.52 to 25.06 points.
The lineage bank gained 26.05 points with an interval from 22.59 to 29.63 points.
Both dense arms removed the baseline's one false restriction among 810 clean rows.

The predeclared matched-group comparison is stronger than noninferiority.
Lineage minus all-row recall was +4.20 percentage points with a 95% interval from +1.36 to +7.04 points, while FPR differed by 0 points with a 0-to-0 interval.
The lineage bank did not regress any attack subtype, had no retrieval failure, and reduced exact-search p95 by 38%.
Every locked bank-comparison gate passed, so the lineage bank is confirmed as the bank for any future shadow.

The source also exposes a hard architectural limit.
Both dense banks had 0% recall on the one-shot task-switch slice because all 162 attacks passed below the local reviewer floor and therefore never reached retrieval or DeepSeek.
On the zero-shot task-switch slice, 137 of 162 attacks also passed locally; final recall was 10.49% for lineage and 9.26% for all-row.
This is not a retrieval-ranking failure and cannot be repaired by a reranker, HNSW, Qdrant, or more examples after the reviewer gate.
Do not tune the local threshold or prompt on this consumed panel; any repair needs a materially different invocation architecture and new independent evidence.

The marginal added p95 values happened to pass the one-second feature gate, but paired reviewer-latency deltas remained noisy at 4.72 seconds for lineage and 5.14 seconds for all-row.
The result confirms quality and relative exact-search behavior, not production latency.
Representative shadow traffic is still required.
The public 2024 benchmark may also have been present in later model pretraining, so this is source-heldout from Morgott fitting rather than proof of foundation-training independence.

### Final decision

The completed evidence retains PPLX 256d, dense-only retrieval, the source-lineage bank, and exact NumPy search as the research winner.
It does not select a larger full-bank dimension, either tested trigram fusion strategy, a reranker, the measured Faiss FlatIP configuration, HNSW, Qdrant, or GraphRAG for the current bank.
HNSW and Qdrant were not benchmarked and are gated growth options rather than technologies rejected in general.
The global ledger now reserves or settles $21.20 under the $50 cap, leaving $26.80 available after preserving the $2 reserve.

Nothing is promoted into maintained inference.
The prospective WMT panel confirms the lineage bank over the all-row bank on one independent source-heldout synthetic transfer suite, but its 43.33% recall is not broad robustness and its translation workload is not representative traffic.
The current canonical validation and dev-test, PromptShield, SEP, and WMT roles are now consumed evidence.
No representative independently adjudicated live shadow stream exists in this repository.

The next action is to collect prospective task-bearing long benign traffic and matched attacks with source, time, language, length, channel, participant, and provider-failure metadata.
Only the lineage bank should enter that advisory-only shadow.
The shadow must measure p50 and p95 latency, index and provider failure rates, exact fallback behavior, example stability, memory, CPU, call rate, and provider-native cost without changing authoritative decisions.
If representative traffic cannot be supplied, adding a dormant runtime integration or simulating traffic would not satisfy the remaining gate.

## Dense index and scale re-evaluation (2026-08-19)

### What was and was not measured

Only Faiss `IndexFlatIP` was measured slower than the current NumPy implementation.
HNSW, IVF, DiskANN or Vamana, and Qdrant were not benchmarked.
Calling those alternatives slower was incorrect.

`IndexFlatIP` and NumPy both perform an exhaustive `O(Nd)` inner-product search.
Morgott's NumPy path is a contiguous OpenBLAS matrix-vector multiply followed by `argpartition`, while Faiss adds its own singleton-query and result-selection path.
Faiss explicitly documents that it is optimized for query batches and that one-query searches can be slower.
The recorded benchmark also has order, resident-memory, build-metadata, and thread-setting confounders, so it establishes only that one measured configuration rather than a general library ranking.

The strict FlatIP parity gate was also too strong.
All score differences were at most `5.96e-7`, and the only candidate-set difference was an exact score tie at the rank-20 boundary.
Future exact comparisons must use a numeric score tolerance, tie-aware candidate recall, selected four-example packet parity, and full-cascade parity rather than arbitrary tied-ID order.

### Current scaling evidence

| Bank | Four-worker exact p95 | Throughput | Process RSS |
|---|---:|---:|---:|
| Lineage representatives | 43.6 ms | 95 QPS | 578 MiB |
| All eligible rows | 176.3 ms | 31.4 QPS | 1.97 GiB |

The all-row process used 1.97 GiB for a raw float32 vector matrix of 739.8 MiB.
The first memory-scaling issue is therefore the current SQLite-BLOB-to-Python loader and duplicated resident representations, not exact inner-product search itself.
That process measurement also leaves little confidence that the all-row design fits beside the maintained model on the intended two-vCPU, four-GiB service shape.
The benchmark machine had materially more CPU and memory, so target-hardware measurement is mandatory.

Scaling decisions must use the number of distinct eligible lineage representatives rather than raw corpus rows.
The lineage bank collapsed correlated variants and then beat the all-row bank on prospective WMT recall, failures, local latency, and memory.
Adding data should mean adding new, provenance-controlled semantic coverage, not indexing every mutation of the same source example.

Approximate raw index storage at 256 float32 dimensions is:

| Distinct vectors | Flat | HNSW32 lower estimate | IVFFlat |
|---:|---:|---:|---:|
| 1 million | 0.954 GiB | 1.192 GiB | 0.961 GiB |
| 10 million | 9.537 GiB | 11.921 GiB | 9.611 GiB |

These estimates exclude higher HNSW layers, allocators, process overhead, payload metadata, and duplicate rollout copies.
Faiss documents Flat storage as `4*d` bytes per vector, HNSW storage as the vector plus graph links, and IVFFlat storage as `4*d+8` bytes per vector in its [index summary](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).
HNSW trades `M`, `efConstruction`, and `efSearch` against memory, build time, latency, and recall, while IVF adds representative k-means training and an `nprobe` recall-latency control.
Faiss's [index-selection guidance](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) treats Flat as the exact baseline and HNSW as a strong in-memory option when RAM permits.

### Smallest useful growth benchmark

The next local-index experiment should be bounded to the existing full all-row vectors and should make no maintained runtime change.

1. Re-run NumPy and FlatIP in isolated processes, alternate timing order, record CPU, NumPy, BLAS, Faiss, compiler, and SIMD metadata, and set native thread environment variables before process start.
2. Build one Faiss `IndexHNSWFlat` curve with `M=32`, `efConstruction=200`, and `efSearch` values 64, 128, and 256.
3. Retrieve more than the final top 20, then exactly rescore the returned candidates with the original float32 vectors before deterministic example selection.
4. Measure singleton requests at concurrency four because that is the reviewer workload, and separately record any batch result only as an offline-build diagnostic.
5. Report p50, p95, p99, QPS, CPU, RSS, index bytes, build time, load time, cold start, top-20 recall, four-example packet stability, and full-cascade deltas by channel, label, source, language, and domain.
6. Run the same benchmark on the intended deployment shape before selecting a backend.

Exact NumPy remains the incumbent while four-worker local-search p95 is at most 200 ms, sustained throughput is at least twice expected peak reviewer traffic, steady RSS is at most 60% of the process limit, and an immutable replacement snapshot can be loaded within rollout memory.
HNSW should be tested when any gate fails or when the actual lineage bank reaches one million distinct vectors.
It should replace exact search only if top-20 recall is at least 0.98 overall and 0.95 in every adequately sized critical slice, the selected packet and cascade pass the existing quality gates, and local p95 improves by at least twofold.

At ten million distinct vectors, add one IVFFlat curve if HNSW's graph memory or rebuild window is unacceptable.
Test DiskANN only when a justified in-memory configuration no longer fits the deployment RAM budget, because DiskANN's purpose is SSD-backed, memory-constrained approximate search rather than improving semantic quality.
The original [DiskANN paper](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html) demonstrates that design at much larger scales but does not predict Morgott's latency or recall.

Qdrant is not a better embedding or ranking algorithm by itself.
It supplies a service around dense HNSW, sparse indexes, payload filtering, replication, online writes, and storage tiers.
It should enter when Morgott needs independent scaling, high availability, shared consumers, dynamic updates, filter-aware search, or an index larger than the application should own in process.
Raw row count alone is not that trigger.
See Qdrant's official [indexing](https://qdrant.tech/documentation/manage-data/indexing/), [storage-tier](https://qdrant.tech/documentation/ops-configuration/memory-tiers/), and [distributed-deployment](https://qdrant.tech/documentation/scaling/distributed_deployment/) documentation.
If that operational trigger appears after sparse complementarity is proven, benchmark Qdrant's native [full-text BM25](https://qdrant.tech/documentation/search/text-search/full-text-search/) and [hybrid fusion](https://qdrant.tech/documentation/search/hybrid-queries/) as one unified service against the same frozen exact, HNSW, and cascade gates.

Future corpus releases should build a versioned lineage snapshot offline, bind it to the data and embedding manifests, validate it against exact search, and atomically replace the prior immutable snapshot.
Frequent online inserts are a Qdrant trigger, while periodic corpus releases are not.

## Sparse and hybrid robustness re-evaluation (2026-08-19)

### What actually caused the latency and failures

RRF was not the latency bottleneck.
In the weighted full-row arm, fusion and final selection had a 4.64 ms p95, while sparse search had a 1.382-second p95.
The performance result therefore applies to the current SQLite sparse implementation rather than to rank fusion as a class.

The current query performs two sequential sparse searches, constructs a broad OR over up to the first 32 normalized terms, searches a global FTS table, joins channel and label metadata afterward, and sorts by `bm25(...)`.
SQLite documents that `ORDER BY rank` can be faster than calling the ranking function in the ordering expression when a query has a `LIMIT` in its [FTS5 ranking documentation](https://www.sqlite.org/fts5.html#sorting_by_auxiliary_function_results).
The current global-table design also makes common-term posting lists and duplicate-heavy lineages expensive before the fixed channel-label filters can help.

Bank composition materially changed sparse latency.
Successful Unicode and trigram p95 values on the lineage bank were about 218 ms and 382 ms, compared with about 916 ms and 1.344 seconds on the all-row bank.
The all-row bank contains one highly concentrated lineage, which inflates postings, changes document frequency, and can fill a top-k list with effectively the same example.

Most recorded selection failures were also specific to the current design.
Nineteen of the 20 all-row trigram failures came from HackAPrompt.
Nine failing queries returned no candidates, and several positive top-20 lists contained only one source-lineage despite returning 20 rows.
Over-retrieving 50 per label and deduplicating by lineage reduced those 20 failures to two.
A fail-soft replay of the saved lineage candidates produced selectable packets for all 110 queries, but selected no example unique to BM25, so it repaired reliability without establishing sparse complementarity.

The current FTS5 trigram arm is substring search rather than fuzzy bag-of-character-grams retrieval.
The query builder quotes whole terms, and SQLite documents that its trigram tokenizer matches substrings and cannot match a token shorter than three Unicode characters in its [tokenizer documentation](https://www.sqlite.org/fts5.html#the_trigram_tokenizer).
It should not be described as a complete obfuscation retriever.

### Domain slices do not support a universal hybrid default

| Arm | Overall recall | HackAPrompt recall | WildJailbreak recall |
|---|---:|---:|---:|
| Dense PPLX | 94.09% | 89.38% | 98.96% |
| Equal-weight RRF | 88.18% | 78.76% | 97.92% |
| Weighted top-50 RRF | 93.64% | 91.15% | 95.83% |

The weighted hybrid improved HackAPrompt by 1.77 points relative to dense but reduced WildJailbreak by 3.13 points and lost 0.45 point overall.
This is evidence that branch utility varies by domain, not evidence that an aggregate benchmark should decide every production slice.
It also does not justify enabling the branch for HackAPrompt alone because that slice decision was observed after the run and lacks fresh confirmation.

The original [RRF study](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) combined TREC document-retrieval runs rather than examples for in-context binary classification.
RRF discards score magnitude, and with `k=60` it gives rank 1 weight `1/61` and rank 20 weight `1/80`, so a weak branch retains substantial influence across its top 20.
A later [fusion analysis](https://arxiv.org/abs/2210.11934) found RRF parameter-sensitive and found normalized convex score fusion stronger in its tested in-domain and out-of-domain settings.
Neither paper proves which fusion rule improves Morgott's fixed DeepSeek reviewer.

### One redesigned sparse experiment

BM25 deserves one fair, bounded re-test because lexical matches can supply exact-token and domain-shift coverage that dense embeddings miss.
The [BEIR study](https://arxiv.org/abs/2104.08663) found BM25 to be a strong zero-shot baseline across heterogeneous document-retrieval datasets, which is enough to justify the diagnostic but not enough to predict Morgott's reviewer outcome.
The experiment should still stop if that candidate coverage does not improve the downstream cascade.

1. Use the source-lineage bank rather than the duplicate-heavy all-row bank.
2. Split SQLite FTS5 into the four fixed channel-label partitions, use Unicode BM25 as the primary lexical branch, change bounded top-k sorting to `ORDER BY rank`, and retain parameterized, escaped query construction.
3. Retrieve 50 raw candidates per label, remove repeated lineage and source concentration before selection, and allow any valid sparse candidates to join the pool even when sparse alone cannot produce a balanced four-example packet.
4. Run dense and sparse retrieval concurrently, and return dense-only results on sparse timeout, empty output, invalid output, or insufficient diversity.
5. Compare dense-only, sparse-only, equal RRF as the untuned control, and one normalized convex score fusion whose single weight is selected through leave-one-source-out development folds.
6. Adjudicate the blinded union top 50 for actual example usefulness, because candidate novelty and same-subtype agreement are not relevance labels.
7. Freeze one candidate before a new source-and-time-heldout confirmation, and require a full-cascade gain within the existing latency, FPR, critical-slice, failure, and cost gates.

If the partitioned lineage SQLite arm contains useful dense-missing examples but misses the latency target, benchmark one optimized local inverted-index implementation such as Tantivy or BM25S.
Tantivy documents memory-mapped indexes and Block-WAND top-k pruning in its [architecture](https://github.com/quickwit-oss/tantivy/blob/main/ARCHITECTURE.md), while the [BM25S paper](https://arxiv.org/abs/2407.03618) describes an eager sparse-matrix implementation for static corpora.
Do not add either engine merely to reduce a number if the sparse candidates still have no downstream utility.

SPLADE or BGE-M3 should enter only if ordinary BM25 proves useful lexical complementarity but still suffers vocabulary mismatch.
[SPLADE](https://arxiv.org/abs/2107.05720) adds a transformer query encoder and learned impact index, while [BGE-M3](https://huggingface.co/BAAI/bge-m3) emits dense, learned-sparse, and multi-vector representations from one 1,024-dimensional multilingual model.
That is a replacement retrieval stack with its own memory and latency costs, not a free robustness layer on PPLX.

### Production robustness protocol

No consumed validation or public transfer benchmark can establish production robustness.
The next decision needs a newly collected, independently adjudicated, advisory-only shadow block followed by a later untouched confirmation block.

The protocol must predeclare trusted-metadata slices for ordinary conversation, coding, security, finance, long task-bearing requests, quoted analysis, web and email content, retrieved documents, tool outputs, direct and indirect attacks, attack subtype, obfuscation, language, length, channel, source, and time.
It must report micro-average, macro-domain, adequately sized worst-domain, paired clean-attack results, source-heldout and time-heldout results, mutation stability, top-four packet stability, duplicate and source concentration, provider and index failures, p50 and p95 latency, CPU, RSS, call rate, and provider-native cost.
The arms are the no-example reviewer, dense PPLX incumbent, redesigned sparse-only retrieval, and the single frozen hybrid candidate.
Hybrid advances only if it contributes adjudicated useful examples absent from dense and improves the full cascade without hiding a critical domain loss behind the aggregate.
The shadow cannot change Morgott's authoritative `decision: allow` behavior.

## Embedding model re-evaluation and challenger tournament (2026-08-19)

### Decision

PPLX Embed V1 4B at 256 dimensions remains the evidence-backed Morgott incumbent, but the evidence does not establish that it is the best available embedding model.
It is the only candidate that has passed Morgott's dimension ablation, full-cascade comparison, lineage-bank scale run, and prospective WMT confirmation.
That is stronger evidence than a generic embedding leaderboard, but it is evidence for one tested configuration rather than proof of global model superiority.

The curated screening bank did not show a decisive quality gap within the tested leaders.
The original PPLX arm reached 89.55% recall, the reversed-order PPLX diagnostic reached 90.45%, and the mixed Voyage 4 large-document plus Voyage 4 lite-query arm reached 90.00%, with the same 0.124% FPR for all three.
PPLX won the locked study because its selected arm passed every declared gate, while the Voyage arms failed the noisy added-latency gate and never received the later full-lineage and prospective confirmation work.
The correct conclusion is therefore "retain PPLX while testing credible challengers," not "PPLX is universally best."

The highest-priority new challenger is Qwen3 Embedding 8B at 256 dimensions through a pinned OpenRouter provider.
OpenRouter listed it at $0.01 per million input tokens with three providers on 2026-08-19, while Qwen3 Embedding 4B was $0.02 per million with one provider, so the 8B route currently has better price and failover options despite the larger model.
Those are mutable service facts, not durable model properties, and they must be snapshotted again when the experiment is frozen.

The most informative local control is Qwen3 Embedding 0.6B at 256 dimensions.
It shares Qwen3's 32K context, Matryoshka dimension support, multilingual coverage, Apache-2.0 license, and query-instruction contract while removing the remote embedding dependency.
It should be treated as an efficiency control rather than presumed inferior from parameter count.

Nemotron 3 Embed 1B is a valid secondary challenger at 512 dimensions, not an immediate 256-dimensional drop-in.
NVIDIA documents a 2,048-dimensional native vector and explicitly demonstrates 1,024- and 512-dimensional prefix slicing with L2 renormalization, but its model card does not establish Morgott-quality retention at 256 dimensions.
Nemotron 3 Embed 8B is a local or NVIDIA-serving quality anchor only if the 1B family result is promising, because no current OpenRouter 8B embedding route was found and NVIDIA's documented examples start from a 4,096-dimensional vector with much larger prefixes.

No reranker should be added now.
The current evidence shows sparse candidate novelty without an ordering or full-cascade gain, and a reranker cannot recover candidates absent from its input or queries that never pass the local reviewer gate.

### Model and serving contracts

All service prices and provider counts in this table are point-in-time observations from 2026-08-19 and must not be copied into a permanent cost assumption.

| Candidate | Required query and document contract | Dimensions and context | Language claim | Delivery, license, and current price | Morgott role |
|---|---|---|---|---|---|
| PPLX Embed V1 4B | Embed raw queries and documents without an instruction prefix, then compare L2-normalized vectors with cosine similarity. | 2,560 native, Matryoshka prefixes from 128 through 2,560 through the API, 32K context. | Multilingual. | OpenRouter listed one Perplexity provider at $0.03/M input tokens; open MIT weights support SentenceTransformers, ONNX, and TEI. | Retained 256d incumbent and local-versus-remote transport parity test. |
| Voyage 4 large documents plus Voyage 4 lite queries | Set `input_type=document` for the bank and `input_type=query` for live queries, which prepends Voyage's published document and query prompts. | 256, 512, 1,024, or 2,048 dimensions, 32K context, and a shared Voyage 4 embedding space. | Multilingual. | OpenRouter listed one Voyage provider at $0.12/M for large and $0.02/M for lite; proprietary large and lite services, with Apache-2.0 Voyage 4 Nano available locally. | Already-tested close comparator and a useful shared-space deployment design. |
| Qwen3 Embedding 4B | Prepend `Instruct: {task_description}\nQuery:{query}` to queries and leave documents unprefixed. | User-selected 32 through 2,560 dimensions, 32K context, and Matryoshka support. | 100+ languages. | Apache-2.0 local weights; OpenRouter listed one DeepInfra provider at $0.02/M. | Reserve challenger if 8B fails latency, contract, or provider-parity gates. |
| Qwen3 Embedding 8B | Use the same explicit textual query instruction as the official model card and leave documents unprefixed. | User-selected 32 through 4,096 dimensions, 32K context, and Matryoshka support. | 100+ languages. | Apache-2.0 local weights; OpenRouter listed three providers at $0.01/M. | Primary remote challenger at 256d. |
| Nemotron 3 Embed 1B | Prefix raw queries with `query: ` and bank documents with `passage: `, or use NVIDIA's `/v2/embed` endpoint with the matching input types. | 2,048 native, documented 1,024 and 512 prefix slices, 32K model context. | Evaluated by NVIDIA in 34 languages. | OpenMDW-1.1 local weights and NVIDIA serving; OpenRouter listed one free provider but exposed inconsistent 16K and 33K context descriptions during this research. | Secondary challenger at 512d after a contract canary. |
| Nemotron 3 Embed 8B | Use the same `query: ` and `passage: ` role prefixes. | 4,096 native with documented large prefix slices, 32K model context. | Evaluated by NVIDIA in 34 languages. | OpenMDW-1.1 local weights and NVIDIA serving, with no verified OpenRouter embedding route found. | Conditional family quality anchor, not a first-round index build. |
| Qwen3 Embedding 0.6B | Use the same explicit Qwen task instruction on queries and no document instruction. | User-selected 32 through 1,024 dimensions, 32K context, and Matryoshka support. | 100+ languages. | Apache-2.0 local weights with SentenceTransformers, Transformers, vLLM, and TEI examples. | Primary local 256d control. |
| BGE-M3 | No retrieval instruction is required for the documented dense path. | Fixed 1,024-dimensional dense vectors and 8,192-token context. | Multilingual. | MIT local weights; one model can emit dense, learned sparse, and ColBERT-style representations. | Optional learned-sparse diagnostic, not a default dense challenger. |

The point-in-time OpenRouter service facts came from the official [PPLX](https://openrouter.ai/perplexity/pplx-embed-v1-4b), [Voyage 4 large](https://openrouter.ai/voyageai/voyage-4-large), [Voyage 4 lite](https://openrouter.ai/voyageai/voyage-4-lite), [Qwen3 4B](https://openrouter.ai/qwen/qwen3-embedding-4b), [Qwen3 8B](https://openrouter.ai/qwen/qwen3-embedding-8b), and [Nemotron 3 1B](https://openrouter.ai/nvidia/nemotron-3-embed-1b) model pages.

Sources for PPLX's independent-text contract, dimensions, context, quantization, local runtimes, and MIT license are the [Perplexity standard embedding documentation](https://docs.perplexity.ai/docs/embeddings/standard-embeddings) and [official PPLX model card](https://huggingface.co/perplexity-ai/pplx-embed-v1-4b).
The standard PPLX model is the correct family member because Morgott stores independently meaningful examples rather than chunks whose meaning depends on neighboring document context.
The contextual PPLX variant would add a document-structure assumption that this bank does not have.

Sources for Voyage's role prompts, dimensions, quantization, context, and shared embedding space are the [Voyage embedding documentation](https://docs.voyageai.com/docs/embeddings), [flexible-dimension guide](https://docs.voyageai.com/docs/flexible-dimensions-and-quantization), and [Voyage 4 Nano model card](https://huggingface.co/voyageai/voyage-4-nano).
The shared Voyage 4 space is operationally attractive because a high-quality document encoder can coexist with a smaller query encoder without a re-index, but that feature only matters if the paired configuration wins Morgott's downstream evaluation.

Sources for Qwen's role-specific instruction format, dimensions, languages, local runtimes, and Apache-2.0 license are the official [0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B), and [8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) model cards.
Qwen reports that task instructions usually improve its downstream retrieval results by 1% to 5%, so sending raw queries or assuming that a generic API `input_type` reproduces the Qwen instruction would be a protocol error.
The frozen Morgott instruction should be exactly `Given text from an LLM application, retrieve labeled examples with similar instruction-subversion behavior.` and must not be tuned after outcomes are read.

Sources for Nemotron's role prefixes, vector slicing, context, languages, serving routes, and OpenMDW-1.1 terms are NVIDIA's official [1B](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) and [8B](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16) model cards.
OpenMDW-1.1 is not interchangeable with MIT or Apache-2.0, so a repository licensing review remains required before either Nemotron artifact could enter maintained inference.
The OpenRouter Nemotron listing's context inconsistency is a reason for a fail-closed contract canary, not a reason to infer that either advertised limit is correct.

The [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) documents fixed 1,024-dimensional dense vectors, 8,192-token inputs, and dense, learned-sparse, and multi-vector modes.
BGE-M3 would consume four times the raw float32 vector memory of a 256-dimensional bank, and its generic recommendation to combine retrieval modes is not evidence that such fusion helps Morgott.
It should enter only if slice diagnostics show useful lexical candidates that the stronger dense models consistently miss.

OpenRouter's [embedding API](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request) accepts dimensions and provider preferences, while its [provider-routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection) says the default behavior load-balances among providers and permits fallbacks.
Selection runs must pin one provider, require support for every requested parameter, disable provider fallback, and record the returned model and provider identifiers.
After a winner is selected, same-model failover may be enabled only if a cross-provider canary preserves dimension, preprocessing, vector norms, repeated-input stability, and top-20 neighbor overlap.
Routing to a different embedding model is never a safe fallback for an existing index because the vector spaces are not interchangeable.

PPLX is not API-only.
Its open weights make local query embedding a transport variant of the incumbent, and a target-hardware canary should compare local and OpenRouter vectors before any assumption that they are index-compatible.
Local and remote implementations can differ in precision, quantization, pooling code, or revision, so model-name equality alone is not a parity result.

### Why MTEB and RTEB only nominate candidates

The original [MTEB paper](https://arxiv.org/abs/2210.07316) spans heterogeneous embedding tasks and reports that no single method dominates every task.
The later [MMTEB paper](https://arxiv.org/abs/2502.13595) broadens the benchmark to more than 500 tasks and more than 250 languages, but breadth does not reproduce Morgott's reviewer pipeline.
Generic retrieval leaderboards primarily measure whether known relevant passages appear near the top, commonly with nDCG@10, while Morgott retrieves labeled analogies, enforces label and channel balance, removes source-lineage duplication, inserts four examples into a fixed reviewer prompt, and finally measures binary recall, FPR, call rate, fallbacks, and latency.
This mismatch means leaderboard results are useful screening evidence for Qwen and Nemotron, but they cannot select a production model or justify an 8B model over a 0.6B model.
Vendor-authored benchmark claims receive the same limited role even when their reported scores are strong.

### Frozen candidate tournament

The consumed canonical validation, dev-test, PromptShield, SEP, and WMT panels must not be used to select a new embedding model.
The tournament needs a newly frozen, independently adjudicated selection panel stratified by label, channel, source, language, length, subtype, mutation family, and task-bearing benign content, followed by a separate prospective confirmation source.

Stage 0 is a no-reviewer contract and resource canary on public or approved non-sensitive text.
It must freeze model ID and revision, provider, requested dimension, exact query and document transformation, over-length handling, output encoding, local normalization, retry policy, and timeout behavior.
Every route must reject over-length inputs locally when the provider API cannot explicitly disable truncation.
Each arm must return exactly the requested finite nonzero dimension, remain stable within a declared numeric tolerance on repeated inputs, and preserve its own query-document similarity contract across batch sizes.
The canary must measure p50, p95, errors, retries, tokens, cost, and target-hardware memory before any document-bank build.

Stage 1 uses the byte-identical curated-screen sample from the confirmed lineage bank and exact NumPy search for every arm.
The first-round arms are PPLX 4B 256d, Voyage 4 large-document plus Voyage 4 lite-query 256d, Qwen3 8B 256d, Nemotron 3 1B 512d, and local Qwen3 0.6B 256d.
PPLX local query embedding is a separate transport-parity arm against the existing PPLX document index rather than a new model arm.
Qwen3 4B advances only if 8B fails its contract or operational gate, and Nemotron 3 8B advances only if the 1B result establishes a family-quality reason to pay its serving cost.
BGE-M3 advances only after a dense-miss audit finds recurring domain or language slices with useful lexical complementarity.

Candidate quality cannot be inferred from subtype agreement alone because the corpus has no passage-level relevance labels.
A blinded union of top-50 results from every first-round arm should therefore be adjudicated for whether each example is genuinely useful to distinguish instruction subversion from a legitimate same-domain request, while retaining source-lineage identity for leakage checks.
Stage 1 reports candidate Recall@50, nDCG@10, complete balanced-packet availability, source and lineage diversity, mutation stability, slice results, exact-search latency, embedding latency, errors, and raw vector memory.
At most two challengers advance, and no dimension or query-instruction sweep is allowed after the frozen settings are scored.

Stage 2 runs the advancing challengers and the PPLX incumbent through the unchanged DeepSeek reviewer, prompt, threshold, local gate, four-example selector, and fallback behavior on the same frozen selection rows.
The primary endpoints remain full-cascade recall and FPR with paired intervals, followed by local call rate, retrieval fallbacks, per-slice deltas, prompt-token growth, p50 and p95 latency, and provider-native cost.
A challenger may replace PPLX for prospective confirmation only if the lower paired recall difference is at least -1.0 percentage point, the upper paired FPR difference is at most +0.10 point, no critical predeclared slice loses more than 3 points, and the arm passes the existing one-second feature-latency gate.
Replacement additionally requires either a recall interval wholly above zero at noninferior FPR or noninferior quality plus a material operational win such as at least 20% lower representative p95 or at least half as many terminal embedding failures.
Price is a tiebreaker after quality, robustness, and reliability rather than a substitute for them.

Stage 3 rebuilds only the winning challenger's source-lineage index and confirms it once on an untouched prospective source.
The all-row bank should not be rebuilt for each candidate because the completed paired WMT result already found the smaller lineage bank more accurate, faster, and more reliable.
Future corpus growth should enter through the same provenance, split, deduplication, and lineage controls, then trigger a versioned re-index with the model revision and preprocessing contract bound into its manifest.
More rows are not automatically more robust when they add correlated variants or evaluation-adjacent leakage.

### Conditional reranker gate

The reranker trigger is a measured separation between candidate coverage and selected-example order.
It fires only if the adjudicated top-50 pool contains a complete useful four-example packet for at least 95% of routed queries, the current top-four selection misses at least one such useful example for at least 10% of queries, and an offline oracle packet improves recall without increasing FPR or lowers FPR without reducing recall.
If top-50 coverage is poor, the remedy is a better embedder, a justified sparse candidate branch, or better bank construction rather than a reranker.

If and only if that trigger fires, the first quality experiment should rerank the frozen top-50 candidates with Voyage `rerank-2.5`, a trusted task instruction, and truncation disabled.
Voyage documents a 32K query-document limit, multilingual instruction following, and a current price of $0.05 per million processed tokens in its [reranker guide](https://docs.voyageai.com/docs/reranker) and [pricing page](https://docs.voyageai.com/docs/pricing).
The reranker must beat the dense ordering in the full cascade, not merely improve nDCG, and its separate remote-call p95 and failure rate count against the existing feature budget.
Only after `rerank-2.5` shows a material downstream gain should `rerank-2.5-lite` be tested as the $0.02/M latency alternative.
The Apache-2.0 [Qwen3 Reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) is the local fallback if remote reliability remains the only failing gate, but running it in parallel before the quality trigger would add an unmotivated arm.

### Resulting recommendation

Keep PPLX Embed V1 4B at 256d, dense-only, on the lineage bank for the next advisory shadow because it remains the only end-to-end confirmed choice.
Do not describe it as the final best model.
Run the staged tournament with Qwen3 8B as the primary remote challenger, Qwen3 0.6B as the primary local control, the already-close Voyage mixed encoder as the replication arm, and Nemotron 3 1B at 512d as the conditional NVIDIA challenger.
Do not add a reranker, BGE-M3 sparse fusion, Nemotron 8B, Qwen 4B, or a full-bank rebuild unless the preceding diagnostic gate gives that extra component a specific job.
This design treats embedding quality, index scale, transport reliability, and reranking as separate hypotheses, which is the shortest path to a result that is both scalable and defensible.

## Executed next steps: dense indexes and tournament stop (2026-08-19)

### Isolated full-row index result

The planned NumPy, Faiss Flat, and Faiss HNSW comparison is complete on all all stored 256-dimensional vectors and the same 110 frozen validation review units.
Each backend ran in a fresh child process, native search threads were fixed at one, singleton queries were measured with one and four Python workers, and every timing cell contains three repeats or 330 searches.
The complete machine-readable result is `artifacts/retrieval_assisted_reviewer_full_rows/validation-dense-indexes-pplx-4b.json`.

| Backend | Build | Four-worker p50 | Four-worker p95 | Four-worker p99 | Throughput | Mean set Recall@20 | Exact selected-packet parity |
|---|---:|---:|---:|---:|---:|---:|---:|
| NumPy exact | `<0.001` s | 137.048 ms | 174.707 ms | 179.513 ms | 34.107 QPS | ground truth | ground truth |
| Faiss FlatIP | 0.407 s | 161.764 ms | 180.002 ms | 205.209 ms | 31.888 QPS | 99.977% | 104/110, or 94.545% |
| HNSW32, `efSearch=64` | 344.990 s shared build | 2.000 ms | 2.784 ms | 3.522 ms | 1,965.487 QPS | 89.023% | 57/110, or 51.818% |
| HNSW32, `efSearch=128` | same index | 2.027 ms | 2.869 ms | 3.331 ms | 1,940.108 QPS | 93.023% | 70/110, or 63.636% |
| HNSW32, `efSearch=256` | same index | 2.848 ms | 3.815 ms | 4.115 ms | 1,417.458 QPS | 96.250% | 84/110, or 76.364% |

FlatIP was faster than NumPy with one worker, at 46.343 ms versus 59.646 ms p95, but it was slightly slower and lower-throughput at the required four-worker workload.
Its mean set recall difference came from one rank-20 boundary choice: all 220 rankings had 100% tie-aware score recall and maximum score regret was `2.68e-8`.
FlatIP is therefore semantically exact within the frozen tolerance, but it supplies no measured operational reason to replace NumPy.

HNSW supplied a large local latency and throughput gain, but every tested setting failed the frozen quality gate.
The best setting, `efSearch=256` with `M=32`, `efConstruction=200`, over-retrieval of 80, and exact rescoring of 20 candidates, reached only 96.250% mean set Recall@20 against the required 98%, fell to 20% on its worst ranking, and reproduced only 84 of 110 selected packets.
Increasing `efSearch` improved recall but did not approach packet parity, so no HNSW cascade run or backend promotion is justified from this curve.

These are local-host diagnostics, not deployment measurements.
The host was a non-exclusive AMD Ryzen 7 7445HS machine with 6 physical cores, 12 logical CPUs, and about 30 GiB RAM, while the target Azure preview has 2 vCPUs and 4 GiB RAM.
The source file cache was not cleared between worker-count cells, the raw vector matrix occupied 739.802 MiB, and peak process RSS was 1.997 GiB for NumPy and about 2.245 GiB for each Faiss backend.
The artifact therefore explicitly prohibits a deployment conclusion, and NumPy remains the incumbent until the same isolated test passes on the target service shape or a later HNSW curve meets the quality gate.

### Model-tournament execution boundary

Stage 1 retrieval quality and Stage 2 DeepSeek cascade comparison are a scientific no-go now, not an engineering no-go.
The canonical validation, dev-test, PromptShield, SEP, and WMT evidence has already influenced decisions, and the repository contains no fresh independently adjudicated selection panel with usefulness labels for the blinded top-50 candidate union.
Uncalled rows from the same consumed roles can support an engineering smoke test, but treating them as fresh selection evidence would optimize another model against the same source distributions and would not answer the production-robustness question.

The only valid model-tournament execution now is Stage 0 Qwen contract and resource canaries on approved public non-sensitive text.
That means a pinned-provider Qwen3 Embedding 8B 256-dimensional remote canary and a pinned-revision local Qwen3 Embedding 0.6B 256-dimensional resource canary, with the exact frozen query instruction, dimension and finite-vector checks, repeated-input and batch-shape stability, explicit over-length behavior, latency, failures, tokens, cost, and memory recorded before any bank build.
Do not build challenger banks, adjudicate retrieved examples, or call the fixed DeepSeek reviewer until a new source-and-time-bounded selection block is frozen and independently adjudicated, with a separate untouched confirmation block reserved.

## Executed next step: partitioned sparse re-test (2026-08-19)

The redesigned Unicode sparse diagnostic is complete on the source-lineage bank.
It uses a bank-hash-bound 21 MiB contentless FTS5 sidecar with one table per channel and label, `ORDER BY rank`, the first eight normalized terms, top-50 retrieval per label, lineage deduplication, and the existing deterministic four-example selector.
The hybrid applies a frozen 2:1 dense-to-sparse RRF and returns the exact saved dense packet when the sparse branch is empty, invalid, or fails during fusion.
The machine-readable evidence is under `artifacts/retrieval_assisted_reviewer_full_sparse_v2/` and remains local and gitignored.

| Retrieval arm | Successful packets | p50 | p95 | Maximum | Dense fallback |
|---|---:|---:|---:|---:|---:|
| Partitioned Unicode sparse | 91/110 | 6.124 ms | 87.905 ms | 214.170 ms | not applicable |
| Dense plus partitioned sparse replay | 110/110 | 379.532 ms | 755.973 ms | 1,379.891 ms | 10/110 |

The hybrid latency is a concurrent replay estimate formed from the saved historical dense latency and the newly measured sparse latency, not a new live end-to-end concurrency measurement.
Sparse search itself had an 87.905 ms p95 and fusion had a 1.192 ms p95.
Sparse alone failed to form a balanced packet for 19 units, consisting of 10 empty results and 9 insufficient-diversity results, while partial sparse candidates still allowed hybrid fusion to produce all 110 packets.

The sparse branch supplied 7,624 of 8,339 candidate slots that were absent from the saved dense top 20 and changed 78 of 110 selected hybrid packets.
That established lexical novelty but not relevance.
A bounded post-hoc DeepSeek development diagnostic therefore compared the two new arms with the saved baseline and dense evidence at an actual provider cost of `$0.129700224`.

| Reviewer arm | Recall | FPR | HackAPrompt recall | WildJailbreak recall |
|---|---:|---:|---:|---:|
| No-example baseline | 69.545% | 0.249% | 49.558% | 89.583% |
| Dense PPLX 4B 256d | 93.636% | 0.249% | 89.381% | 97.917% |
| Partitioned Unicode sparse | 86.364% | 0.124% | 74.336% | 98.958% |
| Dense plus partitioned sparse RRF | 93.636% | 0.249% | 89.381% | 97.917% |

The hybrid and dense arms had identical aggregate and reported source-slice quality despite six of 110 unit-level verdicts changing in opposite directions.
The analyzer nominally chose the hybrid because its separately sampled remote reviewer latency was lower, but that provider-latency fluctuation is not an operational win and cannot establish that the extra branch is better.
The substantive finding is that the redesigned sparse implementation fixed its earlier latency and reliability mechanics but did not improve the dense reviewer's quality on this consumed panel.
Dense PPLX therefore remains the default, and BM25 remains a research branch pending fresh usefulness adjudication and source-and-time-heldout confirmation.
The unchanged aggregate quality also does not trigger a reranker experiment.

## Executed next step: Qwen3 8B remote contract canaries (2026-08-19)

Pinned OpenRouter Stage 0 canaries are complete for the Nebius and DeepInfra Qwen3 Embedding 8B routes at 256 dimensions.
Each route received the same two hardcoded public-safe samples as unprefixed documents and with the frozen Qwen query instruction, with fallback disabled, required parameter support enabled, and no ZDR request.
Each canary made eight calls covering repeated batches and batch-versus-single execution, stored only hashes and aggregate telemetry, and used no corpus text or reviewer call.

| Pinned route | Status at `1e-5` max-absolute tolerance | Worst repeat drift | Worst batch-single drift | Minimum cosine | Call p50 | Call p95 | Recorded cost |
|---|---|---:|---:|---:|---:|---:|---:|
| Nebius | failed | 0.003870 | 0.003461 | 0.999900 | 1,460 ms | 3,121 ms | $0.00000276 |
| DeepInfra | failed | 0.002403 | 0.002269 | 0.999933 | 1,851 ms | 4,524 ms | $0.00000276 |

Both routes returned finite, nonzero, normalized 256-dimensional vectors and the canonical `Qwen/Qwen3-Embedding-8B` model identity.
The request routing and returned diagnostic provider fields agreed, although the OpenRouter response cannot independently prove provider identity.
Both routes failed the frozen componentwise stability threshold for documents and queries, so neither route may proceed to a bank build under the current contract.

The cosine agreement remained high, which means the failed `1e-5` componentwise threshold may be stricter than the ranking behavior requires.
It would be outcome tuning to loosen that threshold after observing these calls merely to obtain a passing status.
The next valid remote diagnostic is therefore an independently specified neighbor-overlap and packet-stability gate on approved fresh evidence, not a retroactive tolerance change.
The multi-second per-call latency also needs a representative batched-query measurement before either route can satisfy the one-second feature budget.
The sanitized failure artifacts are `artifacts/retrieval_assisted_reviewer_qwen_stage0/qwen3-embedding-8b-256-stage0-nebius.json` and `artifacts/retrieval_assisted_reviewer_qwen_stage0/qwen3-embedding-8b-256-stage0-deepinfra.json`.

## Executed next step: Qwen3 0.6B local resource canaries (2026-08-19)

The local Qwen control is complete at the pinned public revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` using the official Transformers last-token pooling recipe.
The model was loaded from a verified local Hugging Face cache with network fallback and remote code disabled, queries used the same frozen instruction, documents remained unprefixed, the first 256 Matryoshka dimensions were selected before L2 normalization, and tokenizer truncation was disabled.
No dependency was added to the repository and the model remains unavailable to `embed-bank`.

| Local route | Status at `1e-5` max-absolute tolerance | Worst repeat drift | Worst batch-single drift | Minimum cosine | Call p50 | Call p95 | Load | Resource peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 4050 CUDA BF16 | failed | 0 | 0.006406 | 0.999715 | 19.060 ms | 465.211 ms | 3.562 s | 1.227 GB CUDA reserved, 2.276 GB process RSS |
| Two-core-constrained CPU BF16 | failed | 0 | 0.003885 | 0.999807 | 174.802 ms | 1,583.103 ms | 2.413 s | 1.908 GB process RSS |

Both local runs were exactly repeatable for an unchanged batch and failed only when the same texts moved between batch and singleton execution.
The CUDA route demonstrates a material steady-call latency advantage over both remote 8B routes on this development machine, and its model allocation fits comfortably in the available 6 GiB GPU.
The CPU run suggests that the 0.6B model can fit by itself inside a 4 GiB process limit, but co-residency with the maintained detector and the difference between this host and the target two-vCPU Azure shape still prohibit a deployment conclusion.
The eight-call p95 values also include first-inference warmup and are resource-canary evidence rather than production load-test estimates.

The uniform batch-shape effect across remote 8B, local CUDA 0.6B, and local CPU 0.6B shows that componentwise equality is not the right final retrieval-stability criterion by itself.
That observation does not convert these failed artifacts into passes.
Before any challenger bank build, a fresh protocol must predeclare cosine, exact top-k neighbor-overlap, and four-example packet-stability gates and then rerun the canary without tuning those gates to the observed outcomes.
Until that happens and a fresh independently adjudicated selection panel exists, PPLX 4B 256d remains the dense incumbent and no Qwen route advances to reviewer comparison.
The sanitized local artifacts are `artifacts/retrieval_assisted_reviewer_qwen_local_stage0/qwen3-embedding-0.6b-256-local-stage0-cuda.json` and `artifacts/retrieval_assisted_reviewer_qwen_local_stage0/qwen3-embedding-0.6b-256-local-stage0-cpu.json`.

## Final executed result: full-row HNSW extension and cascade (2026-08-19)

### Full-row retrieval benchmark

The extension benchmark used all all stored 256-dimensional float32 vectors and 110 frozen validation review units, which produced 220 per-label rankings.
A single fresh PPLX query matrix was reused byte-for-byte by isolated NumPy and Faiss HNSW child processes.
The four HNSW arms shared `M=32` and `efConstruction=200`, varied `efSearch` between 512 and 1,024 and over-retrieval between 160 and 320, and applied exact float32 rescoring to the final top 20.
Native search threads were fixed at one, singleton queries ran with one and four Python workers, and every timing cell contained three repeats or 330 searches.
The machine-readable result is `artifacts/retrieval_assisted_reviewer_full_rows/validation-hnsw-extension-pplx-4b.json`.

| HNSW arm | Mean set Recall@20 | Tie-aware score Recall@20 | Worst ranking set Recall@20 | Worst adequately sized slice | Exact selected-packet parity | Four-worker p95 | Speedup versus NumPy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ef512/top160` | 98.318% | 90.409% | 60.000% | 97.541% | 97/110, or 88.182% | 5.667 ms | 27.776x |
| `ef512/top320` | 98.318% | 90.409% | 60.000% | 97.541% | 97/110, or 88.182% | 7.733 ms | 20.353x |
| `ef1024/top160` | 99.409% | 95.477% | 85.000% | 99.057% | 101/110, or 91.818% | 9.740 ms | 16.160x |
| `ef1024/top320` | 99.409% | 95.477% | 85.000% | 99.057% | 101/110, or 91.818% | 10.778 ms | 14.603x |

The worst adequately sized slice was `source:hackaprompt` with 122 rankings for every arm.
All four arms passed the predeclared retrieval advancement gates of at least 98% mean set Recall@20, at least 95% recall in every adequately sized slice, and at least 2x local p95 speedup.
For each `efSearch`, top-320 produced byte-identical exact-rescored rankings and selected packets to top-160 while increasing latency, so both top-320 arms were dominated and excluded from the cascade.
The same-matrix NumPy ground truth had a four-worker p95 of 157.396 ms.

The raw vector matrix was 739.802 MiB, and the shared HNSW build took 367.768 seconds.
NumPy held 1.954 GiB RSS with its backend loaded and peaked at 1.997 GiB, while HNSW held 1.450 GiB after releasing the source matrix and peaked at 2.165 GiB during construction.
These measurements came from a non-exclusive 6-core local development host rather than the target 2-vCPU, 4-GiB Azure service shape.

The one fresh remote PPLX embedding call covered all 110 queries and 21,881 tokens, took 2.255 seconds, and cost `$0.00065643`.
That batched remote call is not a per-request latency measurement, but it is much larger than the measured local-search p95 values and leaves remote query embedding as the dominant unresolved end-to-end latency component.

### Gated reviewer cascade

The cascade compared the imported no-example baseline, fresh same-matrix NumPy packets, and both non-dominated top-160 HNSW arms with the fixed DeepSeek reviewer.
Exact request-identity reuse reduced the 330 new retrieval-arm records to 119 provider calls and copied 211 byte-identical responses, while 110 successful baseline records were imported at zero new cost.
The machine-readable analysis is `artifacts/retrieval_assisted_reviewer_hnsw_cascade/validation-hnsw-cascade-analysis.json`.

| Reviewer arm | Recall | FPR | Recall delta versus baseline, paired 95% CI | Recall delta versus fresh NumPy, paired 95% CI | Retrieval fallbacks | Terminal review failures |
|---|---:|---:|---:|---:|---:|---:|
| Imported no-example baseline | 71.818% | 0.249% | reference | not applicable | 0 | 0 |
| Fresh NumPy | 93.182% | 0.249% | +21.364 pp `[+15.909, +26.818]` | reference | 6 | 0 |
| HNSW `ef512/top160` | 92.727% | 0.249% | +20.909 pp `[+15.455, +26.364]` | -0.455 pp `[-1.364, 0.000]` | 6 | 0 |
| HNSW `ef1024/top160` | 93.182% | 0.249% | +21.364 pp `[+15.909, +26.818]` | 0.000 pp `[0.000, 0.000]` | 6 | 0 |

Every retrieval arm had the same aggregate FPR, whose Wilson 95% interval was 0.068% to 0.902%, and every paired FPR delta and paired 95% interval was exactly 0.000 percentage points.
Relative to fresh NumPy, the worst critical-slice recall delta was -0.885 percentage points for `ef512/top160` and 0.000 percentage points for `ef1024/top160`.
Both HNSW arms passed the direct quality, FPR, critical-slice, fallback, and terminal-failure gates and independently retained the baseline quality gain.

The 119 new reviewer calls cost `$0.037944432` against a reserved ceiling of `$1.27975188`.
Including the fresh query-embedding call, measured new provider spend was `$0.038600862`.
Per-arm reviewer spend after cross-arm response reuse is execution-order dependent, so the total is the defensible cost comparison.

`ef1024/top160` is the robustness-first growth candidate because it gave up some local speed to improve retrieval recall and exactly matched fresh NumPy on downstream recall, FPR, critical slices, fallbacks, and failures.
There is still no production selection because the local benchmark excludes remote embedding, prompt assembly, reviewer latency, service concurrency, index startup, and co-resident resource pressure.
Promotion requires a target-shaped total-latency and resource test on the intended deployment before any maintained inference change.

The 110-unit validation panel is consumed, and this cascade is a post-hoc comparison on evidence already used to develop the retrieval study.
It is not evidence of real-world robustness and does not replace a prospective source-and-time-heldout evaluation.

## Executed result: full-row ef1024 plus partitioned BM25 and RRF (2026-08-19)

### Frozen retrieval design

The combined diagnostic used the same all-row bank, bank SHA, PPLX 4B 256-dimensional query matrix, and saved `efSearch=1024/top160` HNSW evidence as the preceding cascade.
The HNSW artifact retained only the exact-rescored top 20 candidates per label, so those top 20 entered fusion and the unretained raw top 160 did not.
A new 58.5 MiB contentless SQLite FTS5 sidecar was built over the same full-row bank in 9.918 seconds.
It used four fixed channel-label partitions, Unicode tokenization, the first eight normalized terms, `ORDER BY rank`, 320 raw hits per label, and retention of the first 50 unique source-lineages.
The hybrid applied RRF with `k=60`, dense weight 2, sparse weight 1, and the existing deterministic balanced four-example selector.
The sparse FTS query had an approximate 250 ms SQLite execution budget and returned the byte-identical HNSW packet on interruption, empty output, invalid output, or fusion failure.
SQLite checked that budget every 1,000 virtual-machine operations, while row metadata lookup occurred outside it, so it was not a hard 250 ms request deadline.
The machine-readable retrieval and review evidence is under `artifacts/retrieval_assisted_reviewer_full_rows_hnsw_sparse/`.

| Retrieval measurement | Result |
|---|---:|
| HNSW saved four-worker p95 | 9.740 ms |
| BM25 one-worker p50 / p95 | 126.435 / 251.017 ms |
| BM25 four-worker p50 / p95 | 180.085 / 259.463 ms |
| RRF plus selection p95 | 0.802 ms |
| Conservative serial component p95 | 270.004 ms |
| Four-worker BM25 deadline hits | 123/330 |
| Hybrid packets changed from HNSW | 59/110 |
| Selected prompt slots absent from saved dense top 20 | 47 |
| Dense packet failures rescued | 3/6 |
| Full replay sparse fallbacks | 28/110, including 19 timeouts |

The retrieval gate passed because the conservative local component sum remained below one second, the hybrid reduced packet failures from six to three, and every sparse branch failure reproduced the exact dense packet.
The four-worker deadline rate remains a serious operational weakness: 123 of 330 repeated searches timed out, compared with 53 of 330 at one worker.
Fail-soft behavior limits FTS work and preserves dense quality, but progress-handler granularity and metadata work can exceed the nominal budget and make sparse contribution load-dependent.
All 19 interruptions in the sequential materialization were direct-user attack units: 14 of 21 WildJailbreak review units and 5 of 61 HackAPrompt units, while none of the 20 untrusted-content units timed out.
All 28 materialization fallbacks were also direct-user positive units.
The 0.802 ms fusion p95 confirms that RRF itself was not the latency source.
SQLite BM25 over the duplicate-heavy full-row postings was the slow component.

### Post-hoc reviewer result

Exact request matching imported all 110 completed HNSW responses and reused 51 of them for byte-identical hybrid packets.
Only the 59 changed packets reached the fixed DeepSeek reviewer, at an actual new cost of `$0.025870504` under a `$0.69990888` ceiling.
All 220 final arm records completed without a reviewer failure.

| Reviewer arm | Recall | FPR | Retrieval fallbacks | Terminal failures |
|---|---:|---:|---:|---:|
| HNSW `ef1024/top160` | 93.182% | 0.249% | 6 | 0 |
| HNSW plus partitioned BM25 RRF | 93.636% | 0.124% | 3 | 0 |

Hybrid minus dense recall was +0.455 percentage point with a paired 95% interval from -0.909 to +2.273 points.
Hybrid minus dense FPR was -0.124 point with a paired 95% interval from -0.373 to 0 points.
The hybrid converted one additional positive and avoided one false restriction while leaving the total restriction rate unchanged.
HackAPrompt recall moved from 88.496% to 89.381%, WildJailbreak recall remained 97.917%, and no adequately sized critical slice lost recall.
This quality result came from the one sequential packet materialization with 19 sparse interruptions.
The separate four-worker load replay had 123 interruptions across 330 searches and did not retain per-unit packets, so the reviewer result must not be presented as the quality of a stable four-worker serving policy.

A later provider-free fixed-ranking replay isolated the timeout setting from the retrieval recipe.
With one pass over the same 110 queries at four workers, timeout counts were 37 at a 250 ms budget, 7 at 500 ms, and zero at 1,000 ms.
The 1,000 ms cell completed with 521.601 ms p95 and 598.822 ms p99, while every query successful under all three budgets produced the same candidate hash.
A three-repeat 1,000 ms confirmation again had zero timeouts across 330 searches, with 510.972 ms p95, 663.568 ms p99, and 21.220 QPS at four workers.
The corresponding one-worker confirmation had zero timeouts, 322.460 ms p95, 373.602 ms p99, and 9.118 QPS.
These warm-cache local results show that the historical 250 ms cutoff was too aggressive and that a separately versioned 1,000 ms fail-soft budget is the appropriate frozen setting for the next prospective hybrid arm.
They do not change the saved 250 ms artifact, establish target end-to-end latency, or upgrade the consumed-panel reviewer result.

A second provider-free diagnostic tested a cheap lexical-salience version of the user's token-selection hypothesis.
It considered the first 32 normalized unique query terms, discarded terms absent from both label partitions in the request channel, and selected at most eight terms with the lowest combined document frequency.
The method selected 3.673 terms per query on average and matched the original first eight terms for only 11 of 110 queries.
At a 1,000 ms execution budget it had zero timeouts, 367.529 ms p95, 507.173 ms p99, and 33.091 QPS with four workers over one warm-cache pass.
Its corpus-frequency lookup cost was 3.979 ms p95 and 11.446 ms p99.
At the historical 250 ms budget it still timed out on 19 of 110 four-worker searches, compared with 37 of 110 for first-eight selection in the matched one-pass diagnostic.

The resulting HNSW plus IDF-selected BM25 RRF packets changed 70 of 110 dense packets, rescued five of the six dense packet failures, and left one hybrid packet failure.
They supplied 73 selected slots absent from the saved dense top 20 and differed from the completed first-eight hybrid on 49 packets.
The first-eight hybrid had changed 59 dense packets, rescued three dense failures, and supplied 47 dense-missing selected slots.
No reviewer call was made because additional novelty on consumed validation evidence is not proof that the examples are useful.

A matched retrieval-only control then ran the original first-eight selector with the same 1,000 ms execution budget.
It had zero timeouts, changed 77 of 110 dense packets, rescued five of six dense packet failures, supplied 70 selected slots absent from dense top 20, returned nine sparse fallbacks, and left one hybrid packet failure.
The IDF selector changed 70 dense packets, rescued the same five failures, supplied 73 dense-missing slots, returned 14 sparse fallbacks, and left the same one hybrid failure.
IDF reduced four-worker p95 from 510.972 ms to 367.529 ms, but it increased unavailable sparse packets without rescuing another dense failure.
It therefore failed the predeclared no-worse-packet-availability gate and does not replace first-eight BM25 for the fresh hybrid experiment.
This matched result also shows why the historical 250 ms artifact could not be used as the IDF control: raising the first-eight budget alone changed another 18 packets and improved dense-failure rescue from three to five.
A dedicated first-eight determinism run at four workers and a 1,000 ms budget then completed all 330 searches with identical candidate hashes across three repeats for all 110 units.
That run measured 571.593 ms p95, 737.099 ms p99, and 20.614 QPS, which is slower than the prior 510.972 ms replay but still below the frozen execution budget.
The latency range is retained as local run variance rather than selecting the more favorable measurement.

This is directionally promising because every observed point estimate moved in the desired direction, but it is not a demonstrated quality gain.
Both paired intervals include no change, the recall point gain is below the locked one-point complexity threshold, and the validation panel was already consumed during retrieval development.
The combined arm therefore failed its material-gain gate and cannot replace dense ef1024, enter an advisory shadow, or justify a reranker.
It remains the locked hybrid challenger for a fresh prospective selection panel because its observed recall and FPR both moved in the desired direction.
The 47 selected sparse slots were absent from the saved HNSW dense top 20, but they cannot be called absent from the unretained raw HNSW top 160.
Its useful conclusion is narrower: partitioned BM25 can complement ef1024 on some full-row queries, but the present SQLite branch is too timeout-prone and the observed cascade gain is too uncertain.
A production decision still requires a fresh independently adjudicated source-and-time-heldout selection block, an untouched confirmation block, and target-shaped end-to-end latency and memory evidence.

## Executed result: persistent full-row ef1024 runtime canary (2026-08-19)

### Serialized bundle

The fixed `efSearch=1024/top160` candidate was rebuilt once from the hash-bound all-row PPLX 4B 256-dimensional store and serialized as four Faiss HNSW indexes.
The indexes used `M=32`, `efConstruction=200`, one native build thread, deterministic partition order, exact float32 rescoring from 160 candidates to 20, and compact uint32 bank-row maps.
The four indexes and row maps totaled about 939.3 MiB, while the separate immutable metadata bank was about 721 MiB.
The dense source matrix was not copied into the runtime bundle.
The offline build took 334.857 seconds with Faiss 1.15.0.

Eight deterministic persisted document vectors, two from each channel-label partition, were retained as integrity canaries.
Every candidate ranking, score, balanced four-example selection, and packet hash matched exactly before and after serialization.
These document-vector canaries establish persistence and runtime integrity only.
They do not replay the expired PPLX validation query matrix, estimate retrieval quality, or represent live traffic.
The write-once manifest is `artifacts/retrieval_assisted_reviewer_hnsw_persistent/persistent-hnsw-manifest.json`.

### Quota-constrained local runtime

The saved bundle was loaded in a fresh transient service configured with a 200% CPU quota, 4 GiB memory maximum, and zero swap.
The service property requested CPUs 0 and 1, but the process reported affinity to all 12 logical host CPUs, so this is a two-CPU-quota proxy rather than a strict two-core pin or an Azure hardware result.
Native BLAS and Faiss search threads were fixed at one.
Bundle hash verification took 0.895 seconds and serialized-index loading took 0.474 seconds with the operating-system page cache left intact.
Process RSS was about 1.15 GiB after index load and 1.17 GiB after the warm benchmark.
The machine-readable result is `artifacts/retrieval_assisted_reviewer_hnsw_persistent/persistent-hnsw-local-resource.json`.

| Runtime cell | Total p50 | Total p95 | Total p99 | Search plus rescore p95 | Bank lookup plus packet p95 | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| Concurrency 1 | 4.291 ms | 8.304 ms | 9.046 ms | 6.200 ms | 2.117 ms | 189.2 QPS |
| Concurrency 4 | 16.018 ms | 55.476 ms | 59.556 ms | 39.948 ms | 50.386 ms | 167.2 QPS |

Each cell repeated the same eight integrity canaries ten times, so the timings are a runtime mechanics canary rather than a representative production load distribution.
All eight post-load ranking and packet checks remained exact and the process made no provider call.

### Decision

Persistent Faiss clears the local serialization, integrity, memory, and search-only mechanics gate for this static all-row index.
This result means row count alone does not justify Qdrant or another network vector service.
Qdrant should enter only if an operational requirement such as online writes, independent availability, shared consumers, or filter-aware scaling appears, or if Faiss later fails the actual target-shape gate.

There is still no production selection.
This run did not load the maintained mmBERT process, measure a cold artifact download, clear the page cache, call PPLX for singleton live-query embeddings, measure longer DeepSeek prompt prefill, or execute on Azure hardware.
The minimum remaining deployment check is a target-shaped co-resident load and end-to-end safe-query latency run with the already-fixed index.
The minimum remaining hybrid quality check is the locked ef1024 plus partitioned-BM25 RRF arm on a fresh source-and-time-heldout adjudicated panel followed by an untouched confirmation block.
For the quality objective, that hybrid comparison is the next selection experiment, while dense ef1024 remains its paired control and fail-soft fallback rather than a new dense-only challenger.

## Executed result: exact mLateOn reranking of the locked hybrid union (2026-08-19)

The off-by-default local experiment now reranks the ef1024 HNSW plus first-eight BM25 candidate union with pinned mLateOn and exact float32 MaxSim.
It uses Sentence Transformers 6.0.0, verifies every required checkpoint file, rejects overlength inputs before encoding, and preserves the RRF decision fields on every failure.
The complete implementation contract, model comparison, and executed evidence are in `reports/late-interaction-research-20260819.md`.

Stage 0 passed against the model card's published reference scores, with NumPy MaxSim matching the library within `9.54e-7`.
The 110-unit replay had zero sparse timeouts, zero document-cache failures, and the same one pre-existing RRF packet failure.
ColBERT changed 109 of 110 selected packets, showing that it is a material ordering stage rather than a tie breaker.
Its 8,204-document in-memory cache occupied 238,525,440 bytes and took 64.01 seconds to build.

Sequential local added latency was 12.36 ms p50, 235.21 ms p95, 423.43 ms p99, and 995.54 ms maximum.
The row-matched HNSW, BM25, and ColBERT component estimate was 378.89 ms p95, but it was not a concurrent target-service test and one request exceeded one second.
The result therefore passes correctness and fail-soft mechanics but fails the predeclared 100 ms added-stage target.

ColBERT remains integrated as a research challenger and is not wired into maintained inference.
The next quality experiment remains a fresh source-and-time-heldout hybrid panel with dense ef1024 as its paired fallback, followed by optional ColBERT reranking over the frozen candidate union.
Raw-attention outlier selection is not a competing next step; the evidence and alternatives are in `reports/retrieval-token-selection-research-20260819.md`.
The first learned-sparse challenger is OpenSearch `doc-v3-gte`, with the full shortlist and gates in `reports/learned-sparse-model-research-20260819.md`.

## Deferred next iteration: late-interaction checkpoint comparison

This iteration stops after the executed mLateOn correctness and candidate-reranking study.
It will not add another ColBERT-family checkpoint, build a full-corpus multi-vector index, or use the consumed 110-unit panel to select among late-interaction models.

The next iteration should first freeze a new independently adjudicated source-and-time-heldout selection panel and a separate untouched confirmation block.
On that fresh evidence, rerank the exact same frozen HNSW plus BM25 candidate union with mLateOn as the quality anchor, LateOn 149M as the English efficiency control, PPLX Embed Late 0.6B as the incumbent-provider late-interaction control, and original ColBERTv2 only as a historical baseline if the first three leave an unresolved architectural question.
Keep candidate generation, label partitions, balanced selection, reviewer, prompt, threshold, and failure behavior identical across checkpoints.

Compare adjudicated candidate recall and nDCG, selected-packet availability and diversity, full-cascade recall and FPR, critical slices, p50, p95, p99, throughput, peak memory, cache size, and deterministic fail-soft behavior.
Do not build PLAID, FastPlaid, Qdrant multivectors, or a full all-row ColBERT index unless a late-interaction checkpoint first demonstrates a prospective downstream gain over the locked RRF hybrid.
