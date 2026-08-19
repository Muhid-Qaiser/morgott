# Learned sparse retrieval model research

Date: 2026-08-19.

Current implementation decisions are superseded by [retrieval-assisted-reviewer-findings-20260819.md](retrieval-assisted-reviewer-findings-20260819.md).
Learned sparse retrieval is deferred to a later iteration.

## Decision

The best first production-eligible learned-sparse challenger for Morgott is `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte`, run locally with its frozen tokenizer and IDF lookup for queries and its pinned ONNX or PyTorch document encoder for the offline bank.

It is the strongest practical first challenger because query-time encoding is only tokenization plus a weight lookup, its 133M-parameter checkpoint is Apache-2.0, it expands documents into a standard 30,522-dimensional weighted vocabulary, and it can be searched by any exact sparse dot-product engine rather than requiring OpenSearch itself.

For inputs in its published 15-language set and within 512 tokens, `opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1` is the lower-risk multilingual challenger because it also removes neural query inference.

BGE-M3 sparse should be the conditional broad-multilingual and long-context challenger, not a free addition to the current PPLX dense path.

BGE-M3 needs the approximately 2.27 GB XLM-R-based model for both document and live-query inference, and OpenRouter exposes only its 1,024-dimensional dense output rather than its sparse token IDs and weights.

`naver/splade-v3` remains a useful English research ceiling, but its CC-BY-NC-SA-4.0 license and gated download make it ineligible as Morgott's maintained production default without a separate legal decision.

MILCO is the most material newer multilingual learned-sparse option found, but its fresh custom-code checkpoint, neural query path, sparse vocabulary of up to 280,524 pivot-plus-source features, and declared dependence on a SPLADE-v3 checkpoint require a reproducibility and license-provenance canary before any Morgott bank build.

DeepImpact, uniCOIL, miniCOIL, BM42, ELSER, and Pinecone's hosted sparse model are useful historical or operational controls, but none displaces the OpenSearch document-only family or conditional BGE-M3 arm for the first local experiment.

The first experiment should test candidate usefulness with an exact sparse scorer, not deploy Qdrant, OpenSearch, Elasticsearch, a reranker, or sparse ANN before encoder complementarity is established.

Only the winning encoder should receive a all-row scale build and exact-engine benchmark.

These recommendations are hypotheses for Morgott, not conclusions from BEIR, MIRACL, MTEB, MS MARCO, or vendor benchmarks.

## Morgott objective and constraints

The retrieval unit is an example for a fixed DeepSeek binary reviewer rather than a web document that is itself the final answer.

The relevant endpoint is therefore full-cascade recall and false-positive rate after deterministic four-example selection, not standalone nDCG.

The incumbent is dense PPLX Embed 4B at 256 dimensions over the lineage bank, with the current partitioned Unicode BM25 arm retained as a lexical control.

The current repository gate requires at least a 1.0 percentage-point cascade-recall improvement, no more than a 0.25-point absolute FPR increase, no critical-slice recall loss above 3 points, and less than one second of added p95 latency on requests that reach DeepSeek.

Learned sparse must also contribute adjudicated useful examples absent from dense retrieval, because novel token IDs and changed packets are not evidence of useful context.

The bank is versioned and rebuilt offline, so query latency, memory, immutable loading, and deterministic fallback matter more than online insertion speed.

The corpus includes direct and indirect prompt injection, ordinary conversation, quoted analysis, finance, security, obfuscation, multiple sources, and multiple languages, so a single aggregate English retrieval score cannot establish robustness.

## Representation families

BM25 uses lexical terms, collection statistics, term frequency, and document-length normalization without a neural encoder.

Learned lexical weighting methods such as DeepImpact, uniCOIL, miniCOIL, and BGE-M3 sparse usually assign contextual weights to terms that occur in the input, although some pipelines add separate document expansion.

Expansion models such as SPLADE, the OpenSearch v3 document model, ELSER, and MILCO can activate vocabulary terms absent from the original text.

Every candidate in this report ultimately produces sparse `(feature_id, weight)` pairs scored by a dot product, but the feature vocabulary, query contract, expansion behavior, and collection-statistics contract are not interchangeable.

Dense HNSW is unrelated to sparse indexing and cannot search these weighted postings.

RRF can fuse dense and sparse rankings without calibrating their incompatible raw score scales, while normalized score fusion can retain magnitude information if it is fitted only on a development partition.

## Frozen model shortlist

| Role | Frozen model | Query contract | Document contract | Language and length | License and footprint | Disposition |
|---|---|---|---|---|---|---|
| Primary deployable challenger | `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte` | Tokenize with the pinned BERT vocabulary and multiply present token IDs by the checkpoint's frozen IDF table | Run the 133M document-expansion model and emit positive weights in a 30,522-feature vocabulary | English, with an 8,192-position model configuration | Apache-2.0, about 550 MB FP32 weights or a 648 MB Qdrant ONNX export | Advance directly to the exact candidate screen |
| Conditional static-query multilingual challenger | `opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1` | Tokenize with its pinned multilingual BERT vocabulary and apply the supplied IDF lookup | Run the 160M document-expansion model and freeze maximum-ratio pruning at 0.1 | 15 published MIRACL languages and 512 positions | Apache-2.0, about 670 MB FP32 weights, 105,879 features, and 75 reported average nonzeros after pruning | Include when the production language set fits its published coverage |
| Conditional multilingual challenger | `BAAI/bge-m3` sparse head | Run BGE-M3 with no query instruction and request `lexical_weights` | Run the same model and request `lexical_weights` | More than 100 languages and up to 8,192 tokens | MIT, approximately 2.27 GB FP32 weights and a 250,002-token vocabulary | Include only if multilingual or long-input coverage is a predeclared production requirement |
| Research-only English ceiling | `naver/splade-v3` | Neural SPLADE query encoding | Neural SPLADE document encoding with vocabulary expansion | English and 512 tokens, with 256 tokens used for the published reproduction | CC-BY-NC-SA-4.0, gated, BERT-sized, and 30,522 features | Run only on the bounded screen if noncommercial research use is approved |
| Reproducibility canary | `omai-research/milco-650m` | Neural multilingual encoding into an English pivot vocabulary, optionally with the LexEcho source view | Same encoder, with mass-based pruning frozen before evaluation | Model card tags 58 languages and the underlying multilingual configuration supports 8,192 positions | Apache-2.0 tag, about 2.37 GB FP32 weights, custom code, and SPLADE-v3 named as an upstream checkpoint | Do not build a bank until code, revision, output, benchmark, and license provenance pass review |

The shortlist intentionally does not include a reranker because a reranker cannot create a useful example that both retrieval branches missed.

The shortlist also does not replace the PPLX dense incumbent, because the question is whether a learned sparse branch adds complementary candidates.

## OpenSearch v3 document-only model

The official [v3-gte model card](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte) reports a 133M-parameter, inference-free-at-retrieval model with a mean nDCG@10 of 0.546 and mean FLOPS proxy of 1.7 on its selected 13-dataset BEIR subset.

That result is stronger than the card's v3-distill, v2, and v1 document-only variants, but it is vendor-reported first-stage retrieval evidence rather than Morgott evidence.

The card's query example constructs a bag of tokenizer IDs weighted by a frozen `idf.json`, while the document path runs a masked-language-model head, max-pools token logits, applies the v3 activation, removes special tokens, and returns weighted expansion terms.

This asymmetric contract is valuable for Morgott because the expensive neural computation occurs only during the offline bank build.

The model's [configuration](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte/blob/main/config.json) specifies 12 layers, hidden size 768, 30,522 vocabulary entries, and 8,192 maximum positions.

The reference PyTorch path creates token-by-vocabulary logits before max pooling, so the 8,192-position maximum can require substantial transient memory and is a capability ceiling rather than a safe default batch length.

The experiment must freeze the shortest maximum length that preserves Morgott's existing reviewer text boundary and report truncation and peak memory rather than silently selecting the advertised maximum.

The official model card requires custom remote code for the PyTorch path and demonstrates pinning `code_revision=40ced75c3017eb27626c9d4ea981bde21a2662f4`, so a Morgott build must pin both model and code revisions rather than trusting a moving branch.

Qdrant publishes an Apache-2.0 [FastEmbed ONNX conversion](https://huggingface.co/Qdrant/opensearch-neural-sparse-encoding-doc-v3-gte) with distinct `query_embed()` and `embed()` methods, which avoids executing Hugging Face remote code at runtime.

The ONNX model is not currently served by a Hugging Face inference provider, so the supported low-risk path is local inference.

The original 550 MB FP32 checkpoint is practical on a GPU for bulk encoding and plausible on CPU for an offline build, but its actual documents-per-second rate on Morgott text must be measured.

The model expands documents, so its average nonzero count and index size cannot be inferred from its 30,522-dimensional vocabulary or from the dense bank size.

The card reports no Morgott-like adversarial, obfuscated, or example-selection evaluation, so no deployment claim follows from its BEIR average.

## OpenSearch multilingual document-only model

The Apache-2.0 [OpenSearch multilingual v1 model card](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1) describes a 160M document encoder with static IDF-weighted query tokenization, a 105,879-feature vocabulary, and 15 evaluated MIRACL languages.

Those languages are Bengali, Telugu, Spanish, French, Indonesian, Hindi, Russian, Arabic, Chinese, Persian, Japanese, Finnish, Swahili, Korean, and English.

The card excludes Thai because its uncased backbone cannot encode it, which is a concrete reminder that the word multilingual does not mean language-universal.

The model configuration is a 12-layer multilingual BERT with 512 maximum positions, so it cannot replace BGE-M3's 8,192-token coverage for long inputs.

The card reports mean MIRACL nDCG@10 of 0.629 with 138 average nonzeros and 0.626 with 75 average nonzeros after maximum-ratio pruning at 0.1.

That small reported benchmark change and nearly halved representation size make the published 0.1 pruning setting the correct frozen starting contract for this model, but Morgott still must measure candidate loss by language and obfuscation.

The model card advertises a Hugging Face feature-extraction provider, but that label does not document a stable remote sparse `(indices, values)` response contract, so the study should use local model output unless a separate public-safe canary proves the provider schema.

This model should precede BGE-M3 when all required non-English languages and lengths fit its declared envelope because it retains a cheap non-neural query path.

## SPLADE family

The [SPLADE-v3 checkpoint](https://huggingface.co/naver/splade-v3) maps queries and documents into 30,522-dimensional sparse vectors with max pooling, dot-product scoring, and a maximum sequence length of 512 tokens.

Its model card reports MS MARCO MRR@10 of 40.2 and mean nDCG@10 of 51.7 on BEIR-13.

The [SPLADE-v3 paper](https://arxiv.org/abs/2403.06789) reports statistically significant gains over BM25 and SPLADE++ across a meta-analysis of more than 40 query sets, but it also reports significant losses on Touché-2020 and two TREC-MQ sets and notes that long documents may need passage decomposition.

The paper's smaller `splade-v3-distilbert` variant reports 38.7 MS MARCO MRR@10 and 50.0 BEIR-13 nDCG@10.

The `splade-v3-lexical` variant removes query expansion and reports 40.0 and 49.1 on those measures, while `splade-v3-doc` removes query computation and reports 37.8 and 47.0.

The full SPLADE-v3 checkpoint is the relevant quality ceiling because the efficient variants trade away the out-of-domain strength that would justify testing learned sparse in Morgott.

Both query and document inference are neural for full SPLADE-v3, which adds live-query CPU or GPU latency that the OpenSearch document-only model avoids.

The model card is gated and declares CC-BY-NC-SA-4.0, so it is not a clean commercial or maintained-runtime dependency.

## BGE-M3 sparse lexical weights

The official [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3/raw/main/README.md) describes one model that emits dense, lexical-weight, and ColBERT-style representations for more than 100 languages and inputs up to 8,192 tokens under an MIT license.

The [configuration](https://huggingface.co/BAAI/bge-m3/blob/main/config.json) is a 24-layer XLM-R architecture with hidden size 1,024 and a 250,002-token vocabulary.

The FP32 model file is approximately 2.27 GB, which makes local GPU batching preferable for the corpus build and makes a CPU query canary mandatory before accepting it in a 2-vCPU service.

The current contract requires no instruction prefix for queries.

`BGEM3FlagModel.encode(..., return_sparse=True)` returns a `lexical_weights` mapping for both queries and documents, and similarity is the dot product over matching token IDs.

The model card's examples show weights only for tokens present in the input, so this sparse head is contextual lexical weighting rather than SPLADE-style vocabulary expansion.

Its long-context and multilingual coverage are real architectural advantages, but long documents can also create many unique postings and make both encoding and search slower.

The model card says sparse weights are available at no additional model-forward cost when BGE-M3 dense vectors are generated in the same pass.

That saving does not apply to Morgott while PPLX remains the dense encoder, because BGE-M3 would be an additional model forward and an additional bank.

OpenRouter lists [BAAI BGE-M3](https://openrouter.ai/baai/bge-m3/providers) as an 8K model that returns a 1,024-dimensional dense vector.

OpenRouter's [embedding response schema](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request) contains one dense `embedding` float array and has no sparse indices or values field.

An OpenRouter key therefore does not provide access to BGE-M3's learned-sparse head as of the report date.

A local FlagEmbedding service, a pinned custom inference endpoint that exposes `lexical_weights`, or a supported vector database integration is required for sparse BGE-M3.

## MILCO as the newest material open option

The official [MILCO-650m model card](https://huggingface.co/omai-research/milco-650m) describes an ICLR 2026 multilingual learned-sparse model that projects text into an English lexical space and optionally adds a source-language LexEcho view for rare entities and code-switched terms.

The card tags 58 languages, reports evaluation across more than 39 languages, and declares Apache-2.0.

The model returns sparse COO tensors or token-weight dictionaries, and the source-view representation spans the 30,522 English vocabulary plus the 250,002 multilingual vocabulary.

The authors report that mass-pruned documents average 30 active dimensions and that their 560M configuration beats a similarly sized Qwen3 dense encoder while using a smaller index and lower retrieval latency on their multilingual benchmarks.

Those claims are promising but do not test Morgott, and the public checkpoint is new, uses `trust_remote_code`, has no hosted inference provider, and has an approximately 2.37 GB FP32 file.

The card's configuration names `naver/splade-v3` and `BAAI/bge-m3-unsupervised` as upstream checkpoints even though its repository declares Apache-2.0.

That is not proof of a license conflict, but it is sufficient reason to require provenance review before treating the Apache tag as production clearance.

MILCO also requires neural inference for live queries, so it must beat the simpler OpenSearch static-query model by a material multilingual or cross-lingual margin to justify its cost.

The frozen pre-bank canary should verify revision pinning, offline loading, query-document determinism, finite nonnegative weights, vocabulary stability, pruning determinism, maximum-length behavior, CPU and GPU latency, RSS, and exact reproduction of at least one published example.

## DeepImpact and uniCOIL

The [DeepImpact paper](https://arxiv.org/abs/2104.12016) uses DocT5Query to expand documents and a contextual model to assign one impact value to each resulting term for inverted-index retrieval.

Its query path is cheap and lexical, but its separate generation pipeline, English MS MARCO training, and 2021 checkpoint make it a historical architecture reference rather than the best first challenger.

The [uniCOIL paper](https://arxiv.org/abs/2106.14807) places one contextual scalar on each query and document term and commonly combines that representation with DocT5Query expansion.

The official [Pyserini reproduction](https://github.com/castorini/pyserini/blob/master/docs/experiments-unicoil.md) uses a 3.4 GB preprocessed corpus for 8,841,823 MS MARCO passages, an impact index, and a BERT query encoder.

That guide reports about 30 minutes to encode and search 6,980 queries on CPU with live inference, under 10 minutes with pre-encoded queries, and MRR@10 near 0.351.

The public uniCOIL checkpoint does not expose clear model-license metadata in its card, which is another reason not to prefer it over the Apache-licensed OpenSearch v3 model.

DeepImpact and uniCOIL remain useful conceptual baselines for contextual term importance, but they should not consume a Morgott full-bank build.

## miniCOIL and the attention hypothesis

The Apache-2.0 [miniCOIL model card](https://huggingface.co/Qdrant/minicoil-v1) creates a four-dimensional contextual vector per English word stem, weights it with BM25 logic, and falls back to one-dimensional BM25 behavior for out-of-vocabulary terms.

It runs through local ONNX FastEmbed with an approximately 130 MB FP32 file or 33 MB int8 file, but both queries and documents require model inference and the representation must use Qdrant's IDF modifier or an exactly equivalent external calculation.

miniCOIL does not expand into semantically related vocabulary terms, and its official card does not provide broad retrieval evidence strong enough to make it a core challenger.

The user's attention-outlier idea is closest to Qdrant's BM42, which averages the final-layer `[CLS]` attention over heads and combines token importance with collection IDF.

Qdrant's corrected [BM42 report](https://qdrant.tech/articles/bm42/) states that BM42 did not outperform a correct Tantivy BM25 implementation and explicitly labels the method experimental rather than production-ready.

Selecting tokens outside a mean-plus-standard-deviation band would add an uncalibrated threshold to a signal whose scale varies by layer, head, sequence, and model.

That hypothesis is worth revisiting only if a frozen attention extractor first beats BM25 on dense-missing useful-candidate coverage, which the closest published implementation has not shown.

BM42 and a new attention-outlier variant should therefore stay outside the frozen challenger set.

## Hosted and platform-bound alternatives

Elastic's [ELSER v2 documentation](https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-elser) describes an English query-and-document expansion model stored in Elasticsearch sparse-vector fields.

ELSER requires an eligible Elastic subscription or trial, encodes only the first 512 tokens of a field unless text is chunked, and requires at least a 4 GB dedicated ML node in the cited Elastic Cloud configuration.

Elastic reports better relevance and indexing efficiency than ELSER v1 and separate optimized x86-64 and cross-platform builds, but adopting it would couple Morgott to Elasticsearch and a platform-specific model.

ELSER can be self-hosted to keep text local or used through Elastic Inference Service, where corpus and live-query text leave Morgott's boundary.

Pinecone's hosted [sparse English v0 model](https://docs.pinecone.io/models/pinecone-sparse-english-v0) is a DeepImpact-derived English encoder with distinct `query` and `passage` input types, a 512-token default, an optional 2,048-token limit, and sparse token output.

Pinecone reports average gains over BM25 on TREC DL and BEIR, while its current documentation recommends ordinary full-text BM25 for general-purpose retrieval and positions the sparse model for upstream sparse or single-index hybrid workflows.

It is a credible managed-service control, but it would send the entire bank and live queries to a new provider and introduce a proprietary model and index dependency before local complementarity is proven.

No paid or corpus-bearing provider call was made for this research.

## Index-engine comparison

An encoder and an index engine solve different problems, so encoder quality should be frozen before engine selection.

An exact compressed-sparse-row matrix multiplication on the bounded bank is the ground-truth scorer because it introduces no ANN recall loss and makes every engine compare against the same sparse vectors.

Qdrant's [sparse index documentation](https://qdrant.tech/documentation/manage-data/indexing/#sparse-vector-index) states that its sparse index is exact, uses an inverted-index-like structure, immediately indexes mutable data, later builds a compact immutable index, supports pinned, cached, and cold memory tiers, and scores only by dot product.

Qdrant can store named dense and sparse vectors together, but its dense branch uses HNSW while its sparse branch does not.

Qdrant's optional IDF modifier computes collection statistics per shard, so it is appropriate for miniCOIL or BM25 but must not be silently applied to model outputs whose query contract already includes frozen IDF.

OpenSearch neural sparse search uses Lucene inverted indexes and can run the v3-gte model and its compatible query analyzer natively.

OpenSearch reports that expansion models can produce indexes four to seven times larger than BM25 and that weight pruning can cut index size by up to 60 percent with about a 1 percent relevance effect in its benchmarks, which must not be assumed for Morgott [without measurement](https://opensearch.org/blog/opensearch-project-update-performance-progress-in-opensearch-3-0/#pruning-for-neural-sparse-search).

OpenSearch 3.3 also offers SEISMIC sparse ANN, but its [billion-scale study](https://opensearch.org/blog/scaling-neural-sparse-search-to-billions-of-vectors-with-approximate-search/) recommends segments of 5 million to 10 million documents and measured a large 15-data-node cluster.

That study reports 90.209 percent Recall@10, 11.77 ms single-thread mean latency, 27 ms p99, and about 1 GB of memory per million documents for SEISMIC at 1.29 billion documents, but those numbers do not predict a 2-vCPU, 4-GiB Morgott service.

At 758,000 to slightly above 1 million rows, exact sparse search should be tested before sparse ANN because the collection is below the engine's recommended SEISMIC segment size and because approximate recall would add another variable to the encoder comparison.

Elasticsearch supports sparse-vector storage, token pruning, ELSER inference, and hybrid retrieval, but it is not justified as a second service merely to evaluate an encoder that can be scored exactly elsewhere.

Qdrant should be the first production-shaped exact sparse engine benchmark only after an encoder passes the usefulness gate, because it can accept generic sparse vectors and later place dense HNSW and sparse postings behind one API.

OpenSearch should be the second engine only if Qdrant misses the target-hardware p95 or memory gate, or if native v3 query analysis materially simplifies a maintained deployment.

Switching the existing dense index into Qdrant is a separate decision that must retain the dense HNSW exact-recall and selected-packet gates.

## Scale, postings, and latency

Sparse storage scales with total nonzero postings, `sum(document_nnz)`, rather than with declared vocabulary size alone.

The OpenSearch multilingual model card is one of the few official sources that publishes average representation sizes: 138 nonzeros without pruning and 75 with a maximum-ratio threshold of 0.1 [on its MIRACL setup](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1).

At 758,000 documents, 75 nonzeros would mean about 56.9 million postings, whose uncompressed four-byte ID plus four-byte float payload is already about 455 MB before offsets, compression, metadata, and engine overhead.

At 1 million documents, the same illustrative density would mean 75 million postings and a 600 MB raw pair payload.

Those calculations are not estimates for v3-gte, BGE-M3, SPLADE, or MILCO, whose nonzero distributions must be measured on the actual bank.

BGE-M3 can emit at most one retained weight per unique input token ID, so its posting count grows with token diversity and length but it does not activate absent vocabulary terms.

SPLADE, OpenSearch v3-gte, ELSER, and MILCO expansion can create postings for absent terms, so pruning has a larger effect and a larger risk of removing useful obfuscation or domain-transfer signals.

The scale report must record mean, median, p95, and maximum document nonzeros, postings by partition and source, index bytes, peak build RSS, build throughput, load time, cold and warm RSS, and cold and warm query p50, p95, and p99.

It must also report the same metrics for the source-lineage bank and the all-row stress bank because duplicate-heavy lineages can change document frequency, postings, diversity, and latency without adding independent evidence.

The lineage bank remains the quality bank unless a fresh comparison proves that the all-row bank improves the cascade rather than amplifying duplicates.

An actual future bank of more than 1 million distinct examples should rerun the same resource gates, while duplicated vectors may be used only as a labeled throughput stress test and never as robustness evidence.

The local sparse branch must meet a four-concurrent-query p95 of at most 200 ms on the intended 2-vCPU, 4-GiB shape and keep steady total process RSS at or below 60 percent of the process limit.

The full feature must still meet the inherited less-than-one-second added p95 gate, and the sparse branch must run in parallel with dense retrieval so its elapsed time is not added serially.

No vendor latency number can waive those target-hardware measurements.

## Fusion with PPLX dense HNSW

The learned sparse branch should retrieve 50 candidates per channel-label partition before lineage and source-concentration controls, matching the redesigned BM25 diagnostic.

Dense PPLX HNSW and learned sparse search should execute concurrently from the identical reviewer query text and identical bank snapshot.

If sparse times out, returns nonfinite or malformed weights, violates its bank hash, or lacks sufficient diversity, the hybrid must return the exact saved dense result rather than an empty or partially corrupted packet.

If all retrieval fails, the reviewer must receive the exact current no-example behavior.

Equal-weight RRF is the untuned control, and the current locked `k=60` must be specified explicitly because Qdrant's [RRF implementation](https://qdrant.tech/documentation/search/hybrid-queries/#reciprocal-rank-fusion-rrf) defaults to `k=2` and would otherwise define a materially different ranker.

The current 2:1 dense-to-sparse RRF with `k=60` is the continuity control.

One normalized convex score fusion may be tuned through leave-one-source-out development folds and then frozen, because raw PPLX cosine scores and unbounded learned-sparse dot products are not on a common scale.

The candidate union, fusion rule, lineage deduplication, label balance, source cap, tie breaking, and four-example selector must all be deterministic and hash-bound.

A reranker should remain absent until the union contains adjudicated useful candidates that the frozen fusion repeatedly orders below unhelpful candidates and the cascade misses its gain gate for that reason.

## Minimal experiment

### Stage 0: freeze evidence and contracts

Create a new source-and-time-bounded, independently adjudicated selection block and reserve a later untouched confirmation block before any reviewer calls.

Power the paired panel for the smallest deployment-worthy recall gain and FPR noninferiority margin rather than selecting a convenient row count.

Freeze the PPLX dense incumbent, DeepSeek reviewer, local gate, prompt, threshold, four-example selector, bank membership, normalization, query text, and fallback behavior.

Freeze model repository revisions, tokenizer revisions, custom-code revisions, query and document methods, maximum lengths, output pruning, numeric dtype, and sparse feature maps.

Run only public non-sensitive contract samples for the MILCO canary and do not include MILCO in the bank build unless its reproducibility and provenance checks pass.

### Stage 1: exact candidate screen

Encode the source-lineage bank locally with unpruned OpenSearch v3-gte and, only when multilingual text is predeclared, the pruned OpenSearch multilingual model or BGE-M3 sparse according to the frozen language and length contract.

Run SPLADE-v3 only as a bounded research ceiling if its noncommercial license is accepted for the study.

Use one exact CSR dot-product implementation as ground truth and query the fixed channel-label partitions for top 50 results before deduplication.

Run the existing Unicode BM25 and dense PPLX top 50 over the identical rows as controls.

Blindly adjudicate the union for whether each example is actually useful to the fixed reviewer, with adjudicators unable to see the producing model or score.

Report useful-candidate coverage at 20 and 50, useful candidates absent from dense top 50, useful candidates absent from BM25 top 50, source and lineage concentration, label and channel balance, language and domain slices, obfuscation stability, and packet stability.

A learned-sparse encoder advances only if the union of dense plus that encoder increases the fraction of queries with at least one adjudicated useful candidate by at least 5 absolute percentage points over dense alone and the paired 95 percent interval excludes zero.

The dense-plus-learned-sparse union must also avoid a greater than 3-point candidate-coverage loss relative to the dense-plus-BM25 union in every adequately sized predeclared critical language, domain, channel, and attack slice.

Its exact scorer must have zero malformed results, at most a 0.5 percent terminal query-failure rate on representative non-injected requests, exact dense fallback for every injected sparse failure, four-worker p95 at or below 200 ms, and memory within the 60 percent service limit.

If no encoder passes candidate usefulness, stop without fusion, reviewer calls, engine deployment, pruning sweeps, reranking, or a full-row build.

### Stage 2: frozen fusion and full cascade

For each advancing encoder, compare dense only, sparse only, equal RRF, the current 2:1 RRF continuity control, and one leave-one-source-out tuned normalized convex fusion.

Use the same saved candidate sets for all fusion arms so the comparison spends no additional provider budget on retrieval noise.

Select exactly one hybrid on the selection block and freeze it before the untouched confirmation block.

The hybrid advances only if its full-cascade recall point estimate improves by at least 1.0 percentage point over dense and the paired 95 percent recall-difference interval is above zero.

The upper paired 95 percent bound for absolute FPR increase must be at most 0.25 point, and no adequately sized critical slice may lose more than 3 recall points.

The whole feature must add less than one second p95, retrieval failure must preserve the exact fallback contract, and provider plus index failures, tokens, cost, prompt growth, CPU, and RSS must be reported.

The confirmation report must include micro recall and FPR, macro-domain results, worst adequately sized domain, direct and indirect channels, source-heldout and time-heldout results, ordinary security and finance discussion, quoted analysis, long clean tasks, mutation stability, language, source concentration, and selected-packet stability.

If a hybrid matches dense on aggregate but wins only in a post-hoc domain, it remains a challenger for a newly powered domain-specific confirmation rather than becoming the default.

### Stage 3: scale and engine only for the winner

Build the winning sparse representation over both the lineage bank and the all-row stress bank with immutable manifests and exact vector hashes.

Benchmark exact CSR as ranking ground truth and Qdrant's exact sparse index first on the intended deployment shape.

Require exact top-50 set parity and deterministic four-example packet parity before attributing any quality change to the encoder rather than the engine.

Measure four-worker and expected-peak concurrency, build and load behavior, index bytes, postings, RSS, CPU, p50, p95, p99, QPS, timeout rate, and dense-fallback rate.

Test one frozen pruning threshold only if the unpruned winner misses a declared memory or p95 gate, and require at least 0.99 top-50 recall overall, at least 0.97 in every critical slice, unchanged useful-candidate coverage, and full-cascade gate parity.

Do not test SEISMIC or another sparse ANN method until exact sparse search misses the target at several million distinct rows or the exact index cannot fit within the deployment envelope.

Do not migrate dense PPLX HNSW into Qdrant merely because the sparse benchmark uses Qdrant.

## Privacy and provider boundary

Local OpenSearch v3-gte, BGE-M3, SPLADE-v3, MILCO, miniCOIL, Qdrant, and self-hosted OpenSearch can keep corpus and live-query text inside Morgott's environment.

OpenRouter BGE-M3 is not a sparse endpoint, so sending text there cannot satisfy this experiment's output contract.

Pinecone integrated inference, Elastic Inference Service, or a custom hosted sparse endpoint would receive corpus text during bank creation and live text during query encoding.

Any remote arm therefore requires an explicit bounded study, provider and retention review, privacy filtering or provider-safe text, a frozen request contract, a recorded budget, and a fail-soft local fallback.

Key availability does not authorize corpus upload, and this research neither inspected credentials nor called a paid provider.

## Final recommendation

Run one local, exact, candidate-first comparison of OpenSearch v3-gte against current BM25 and dense PPLX on the lineage bank.

Add the OpenSearch multilingual model when its 15-language and 512-token contract fits a predeclared requirement, and use BGE-M3 sparse only for broader language or long-context coverage that the static-query models cannot provide.

Use SPLADE-v3 only as a noncommercial bounded ceiling and use MILCO only after its canary and provenance review.

Do not spend engineering effort on DeepImpact, uniCOIL, miniCOIL, BM42, ELSER, Pinecone, sparse ANN, a reranker, or a vector-service migration until the primary challenger supplies adjudicated dense-missing useful examples.

If it clears that gate, freeze one fusion, prove the fixed DeepSeek cascade on new source-and-time-heldout evidence, and only then benchmark the full all-row index and its growth path beyond one million distinct examples.
