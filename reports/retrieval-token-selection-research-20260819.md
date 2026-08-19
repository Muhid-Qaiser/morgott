# Retrieval token-selection research

Date: 2026-08-19

Current implementation decisions are superseded by [retrieval-assisted-reviewer-findings-20260819.md](retrieval-assisted-reviewer-findings-20260819.md).
This file remains the token-selection experiment record.

## Decision

The proposed attention-outlier heuristic is scientifically testable, but it should not be a Morgott experiment arm yet.

The current PPLX embedding API does not expose token embeddings or attention tensors, raw self-attention does not define one stable token-importance score, and a mean-or-standard-deviation cutoff has no retrieval-specific training objective.

The smallest justified next experiment is a provider-free comparison of the existing first-eight-term BM25 query with a corpus-aware top-eight IDF query over the existing SQLite FTS5 index.

If corpus-aware term selection cannot make BM25 fast and reliable enough, the next model-backed sparse candidate should be BGE-M3 lexical retrieval on the curated-screen bank, not an invented raw-attention score.

SPLADE and ColBERT remain later escalations with materially larger indexing, serving, and licensing consequences.

## Current Morgott baseline

The dense incumbent is the all-row PPLX 4B 256-dimensional index searched with Faiss HNSW at `efSearch=1024`, over-retrieval depth 160, and exact float32 rescoring to 20 candidates.

The sparse branch is a four-partition contentless SQLite FTS5 index using Unicode tokenization, the first eight normalized query terms joined by `OR`, 320 raw results per label, and retention of the first 50 unique source-lineages.

The frozen fusion uses RRF with `k=60`, dense weight 2, sparse weight 1, and the existing balanced four-example selector.

The 250 ms value that the sparse branch exceeded was an approximate SQLite virtual-machine execution cutoff, not a financial budget and not a hard end-to-end request deadline.

SQLite checked elapsed time only every 1,000 virtual-machine operations, and later metadata work occurred outside the cutoff.

At four workers, that cutoff interrupted 123 of 330 searches in the original load run.

A later fixed-ranking replay had 37 timeouts at 250 ms, 7 at 500 ms, and none at 1,000 ms over the same 110 queries.

The three-repeat 1,000 ms confirmation had no timeouts across 330 searches, but its four-worker BM25 p95 was 510.972 ms and p99 was 663.568 ms.

The fusion itself had a p95 of only 0.802 ms, so term processing and postings traversal in SQLite BM25, rather than RRF, caused the latency.

On the consumed 110-unit quality panel, hybrid minus dense moved recall from 93.182% to 93.636% and FPR from 0.249% to 0.124%.

Those changes represented one additional true positive and one fewer false positive, and both paired confidence intervals included no change.

That makes the hybrid the correct prospective challenger, while dense remains its paired control and fail-soft fallback.

It does not make the consumed panel valid for selecting additional learned heuristics.

## What the attention hypothesis could mean

There are three materially different implementations hiding inside the rough hypothesis.

1. A query-only sparse selector would use another model's attention to choose words sent to BM25.
2. An attention-pooled dense retriever would use token states and attention to create a new query vector.
3. A token-pruned encoder would remove low-scoring tokens during model inference to reduce compute.

Only the first interpretation can reuse the current FTS index without rebuilding the corpus.

The second interpretation requires corpus and query embeddings produced by the same model and pooling contract, so it cannot be mixed with the existing PPLX document vectors.

The third interpretation is an encoder-efficiency method rather than a retrieval-ranking method.

These interpretations should not be combined in one experiment because a result would not identify whether token selection, embedding geometry, or encoder pruning caused the change.

## Why raw attention outliers are not the next arm

### The current API cannot supply the signal

The official PPLX endpoint accepts input, model, Matryoshka dimension, and output encoding, and returns one embedding per input.

It exposes no token-level representation or attention output in its documented contract.

See the [Perplexity standard embeddings documentation](https://docs.perplexity.ai/docs/embeddings/standard-embeddings).

Obtaining PPLX attention would therefore require an undocumented provider behavior or local access to matching model weights and internals, neither of which belongs in a reproducible production design.

Using attention from a different local model to edit a PPLX query would introduce a second model whose salience objective is not aligned with the PPLX embedding space.

### Attention is a matrix family, not a token scalar

Transformer tooling exposes one attention tensor per layer with shape `(batch, heads, sequence, sequence)` after softmax.

See the [Hugging Face model-output contract](https://huggingface.co/docs/transformers/main_classes/output).

A token score therefore requires decisions about layer, head, direction, aggregation across source positions, handling of special tokens, and mapping subword pieces back to FTS terms.

Changing any of those decisions changes the ranking signal.

Materializing all returned attention tensors also has storage proportional to layers times heads times squared sequence length, even when the final selector keeps only a few tokens.

The mean and standard deviation of a bounded, context-dependent softmax distribution do not supply a model-independent threshold.

The resulting number of retained terms would vary with sequence length and head behavior, making both recall and sparse latency less predictable.

### High attention is not established retrieval importance

The original attention-faithfulness study found that learned attention weights were often uncorrelated with gradient feature importance and that very different attention distributions could yield equivalent predictions.

See [Attention is not Explanation](https://aclanthology.org/N19-1357/).

This finding does not prove attention is useless as a feature, but it rules out treating a large raw weight as self-validating evidence of token relevance.

An analysis of BERT found heads that attended to delimiters, fixed positional offsets, broad sentence regions, syntax, and coreference.

See [What Does BERT Look at?](https://aclanthology.org/W19-4828/).

That diversity is useful inside a trained model, but it makes a universal outlier rule especially under-specified.

For Morgott, special-token, punctuation, positional, or attacker-induced attention spikes could become sparse query terms without being useful examples for instruction-subversion classification.

That last risk is an architectural inference and has not been measured on Morgott.

### Published attention methods do more than threshold outliers

Ditto is the closest direct precedent for attention-weighted sentence embeddings.

It uses the diagonal of one selected attention head to weight hidden states, chooses the head on a development set, and evaluated semantic textual similarity rather than information retrieval.

Its authors explicitly left information-retrieval evaluation as future work.

See [Ditto: A Simple and Efficient Approach to Improve Sentence Embeddings](https://aclanthology.org/2023.emnlp-main.359/).

PoWER-BERT uses attention-derived word-vector significance for progressive token elimination, but it learns a layer-by-layer retention configuration with the model and task loss.

It is evidence for trained attention-based pruning, not for a query-time mean-or-deviation rule.

See [PoWER-BERT](https://arxiv.org/abs/2001.08950).

Neither method validates taking arbitrary attention outliers from an embedding model and treating them as BM25 terms.

## Better-supported alternatives

### 1. Corpus-IDF query-term selection

BM25 already gives rarer corpus terms greater weight through inverse document frequency.

The standard formulation and its document-frequency term are described in the [Stanford Introduction to Information Retrieval BM25 chapter](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html).

The current first-eight rule ignores that corpus signal before BM25 sees the query.

Selecting up to eight terms with the greatest positive partition-specific IDF should traverse shorter postings lists more often and is therefore a direct hypothesis about the measured bottleneck.

This remains a hypothesis because rare identifiers, typos, or obfuscations can be unhelpful despite high IDF.

The source-lineage deduplication and prospective cascade evaluation must remain in place.

SQLite documents both FTS5 query processing and its built-in BM25 rank function in the [official FTS5 reference](https://www.sqlite.org/fts5.html).

This option requires no new model, no remote call, no corpus text export, no dense rebuild, and only document-frequency metadata derived from the existing local sidecar.

### 2. BGE-M3 learned lexical weights

BGE-M3 is explicitly trained to support dense, sparse lexical, and multi-vector retrieval from one encoder, across more than 100 languages and inputs up to 8,192 tokens.

See the [BGE-M3 paper](https://arxiv.org/abs/2402.03216).

The official implementation exposes `dense_vecs`, `lexical_weights`, and `colbert_vecs` separately and computes lexical matching from learned token weights.

See the [FlagEmbedding BGE-M3 implementation](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/embedder/encoder_only/m3.py).

This is a retrieval-supervised, contextual token-weight signal and is better grounded than raw attention outliers for a multilingual Unicode corpus.

The model and official toolkit are MIT-licensed according to the [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) and [FlagEmbedding repository](https://github.com/FlagOpen/FlagEmbedding).

The cost is a complete local corpus encoding pass, a new weighted inverted index, local model memory, and a model inference on every live query.

Its index size and target CPU latency must be measured on Morgott because benchmark results do not determine the sparsity of this corpus.

### 3. SPLADE-family learned sparse retrieval

SPLADE learns sparse query and document expansion through a BERT masked-language-model head with explicit sparsity regularization.

See the [SPLADE v2 paper](https://arxiv.org/abs/2109.10086) and [official repository](https://github.com/naver/splade).

This can recover related terms that exact BM25 cannot match, which is a stronger capability than merely choosing among observed query terms.

It also requires a full corpus encoding and inverted-index build, and expansion can create substantially more postings than plain BM25 depending on the chosen model and regularization.

The official code and listed weights are licensed CC BY-NC-SA 4.0, so production use would require a licensing decision even if a research pilot passed.

The published training recipe is centered on MS MARCO, while Morgott needs multilingual, Unicode, obfuscation, and source-held-out evidence.

BGE-M3 is therefore the cleaner first learned-sparse pilot for this repository.

### 4. ColBERT-style late interaction

ColBERT represents each query and document with multiple contextual token vectors and scores documents by late token-level interaction rather than compressing every text into one vector.

See the [ColBERT paper](https://arxiv.org/abs/2004.12832) and [official implementation](https://github.com/stanford-futuredata/ColBERT).

This is a principled way to retain token-level evidence without deciding that a few attention outliers are the only important tokens.

ColBERTv2 reports a 6 to 10 times reduction from the original late-interaction footprint through residual compression, but the paper also notes that uncompressed late interaction has an order-of-magnitude larger footprint than single-vector retrieval.

See [ColBERTv2](https://arxiv.org/abs/2112.01488).

PLAID improves late-interaction search and reports tens of milliseconds on GPU and tens to a few hundreds of milliseconds on CPU at its evaluated scales.

See the [PLAID paper](https://arxiv.org/abs/2205.09707).

Those figures are not evidence that a ColBERT index and encoder fit Morgott's two-CPU, 4 GiB co-resident target.

At the all-row bank, token-vector count, compressed bytes per token, document lengths, model memory, and target latency must be measured before any full build.

Late interaction belongs after a candidate-recall audit shows that single-vector dense plus sparse retrieval still misses useful examples, or as a bounded reranker over an already-good candidate union.

## Smallest bounded next experiment

### Question

Can corpus-aware term selection reduce the current BM25 latency and load sensitivity without discarding the sparse candidates that make the hybrid promising?

### Frozen arms

Run exactly two sparse-query selectors against the existing full-row FTS5 sidecar.

The control is the current first eight unique normalized terms.

The challenger is the eight in-vocabulary unique normalized terms with highest positive Robertson-Sparck Jones IDF in each channel-label partition, with original query order as the deterministic tie-breaker.

Use the existing Unicode tokenizer's terms so no second tokenizer or subword-to-word mapper is introduced.

Discard terms absent from a partition because they cannot retrieve a row there.

Do not add a learned stoplist, document-frequency floor, phrase operator, query expansion, attention score, new sparse engine, fusion change, or reranker in this experiment.

Keep raw top 320, lineage-deduplicated top 50, HNSW `efSearch=1024/top160`, exact rescore to 20, RRF `k=60`, dense-to-sparse weight 2:1, and balanced example selection unchanged.

Use the separately versioned approximate 1,000 ms fail-soft SQLite execution budget for both arms.

Any empty, interrupted, invalid, or nondeterministic sparse result must reproduce the exact dense packet.

### Development evidence

Use the already-consumed 110 queries only for provider-free latency, determinism, packet-difference, and failure diagnostics.

Do not call the reviewer, retune `k=8`, retune RRF, or claim a cascade quality win from those rows.

Measure three warm repeats at concurrency one and four, plus a fresh-process load run under the two-CPU, 4 GiB resource envelope.

Record p50, p95, p99, throughput, timeout count, empty and invalid results, post-dedup candidate count, selected packet hash, dense fallback hash, per-source and per-language diagnostics, and peak RSS.

### Operational advancement gates

The IDF selector advances only if all of these conditions hold.

- It has zero sparse timeouts, invalid results, and nondeterministic rankings across the three concurrency-four repeats under the 1,000 ms execution budget.
- Its concurrency-four BM25 p95 is at most 400 ms, which is more than a 20% reduction from the observed 510.972 ms control p95.
- Its concurrency-four BM25 p99 stays below the 1,000 ms fail-soft execution budget.
- It does not increase empty or incomplete balanced packets relative to the first-eight control.
- It preserves all three previously observed dense-failure rescues as a conservative diagnostic, without treating those three cases as proof of general quality.
- Sparse failure reproduces the byte-identical dense packet.
- The sidecar plus document-frequency metadata and runtime process still fit the target resource envelope with the maintained model loaded in the later co-resident check.

If the IDF selector misses any gate, retain first-eight terms with the 1,000 ms fail-soft budget for the prospective hybrid comparison.

Do not tune another deterministic selector on the consumed rows.

### Prospective quality gate

Freeze the one operationally eligible hybrid before inspecting a new independently adjudicated source-and-time-held-out selection panel.

Compare it only with dense HNSW `efSearch=1024/top160`, which remains both the paired control and runtime fallback.

Pre-register panel size and strata for label, channel, source, time, language, length, subtype, mutation family, and task-bearing benign content.

The hybrid advances only if full-cascade recall improves by at least 1.0 percentage point with the paired lower 95% bound above zero, absolute FPR increases by no more than 0.25 percentage point with its paired upper bound within that limit, no critical slice loses more than 3 points, and the full feature adds less than one second p95 on requests reaching DeepSeek.

It must then pass one untouched confirmation block without changing term selection, timeout, RRF, prompt, reviewer, threshold, or fallback behavior.

Research remote spend remains capped at the repository's existing $50 ceiling.

## Escalation order after the IDF test

1. If IDF preserves sparse usefulness but SQLite remains too slow, benchmark a purpose-built inverted-index engine with the identical frozen postings and rankings before changing the retrieval model.
2. If sparse candidate quality rather than engine latency is the limit, run one BGE-M3 lexical pilot on the curated-screen bank with exact retrieval and a blinded usefulness adjudication.
3. Build the all-row BGE-M3 sparse index only if the pilot materially improves candidate utility and clears extrapolated storage and target latency gates.
4. Consider SPLADE only if BGE-M3 fails and its noncommercial license and English-centered transfer risk are acceptable for the intended use.
5. Consider ColBERT or PLAID only if a candidate-recall audit shows that dense plus learned sparse still misses useful examples, or if a bounded late-interaction reranker can improve the union without violating latency and memory limits.
6. Revisit attention only with a published, fully specified pooling or pruning method, a matching open-weight encoder, and a retrieval-specific held-out test.

## Cost, storage, latency, and privacy summary

| Approach | Corpus rebuild | Added online work | Index or storage effect | Current decision |
|---|---|---|---|---|
| Partition-IDF top eight | No | Local dictionary lookup plus existing FTS5 | Tiny document-frequency metadata | Run next |
| Raw attention outliers for BM25 | No | A second local model plus attention extraction and token mapping | Attention tensors are transient but quadratic in sequence length | Do not run now |
| Attention-pooled dense embeddings | Yes, both documents and queries | New encoder inference | New dense index and model | Do not mix with PPLX |
| BGE-M3 lexical | Yes | Local learned-sparse query encoding | New weighted postings plus model | First learned-sparse escalation |
| SPLADE | Yes | Local learned-sparse query encoding | Expanded weighted postings plus model | Later, license-gated |
| ColBERT or PLAID | Yes | Local query encoding and token-level interaction | Compressed multi-vector token index plus model | Later, resource-gated |

The IDF experiment has zero provider cost and keeps corpus text local.

A local BGE-M3, SPLADE, or ColBERT pilot also avoids sending corpus or live text to a provider, but requires frozen model revisions, local artifact hashes, and measured GPU build cost if a GPU is used.

A remote learned-sparse or attention service would transmit live queries and potentially corpus rows, so it requires an explicit provider data-use decision even though zero data retention is not a Morgott requirement.

No remote service should be selected merely because an OpenRouter key is available.

## Bottom line

The intuition behind the hypothesis is useful: not every token deserves equal retrieval influence.

Raw attention outliers are the wrong first implementation because the signal is unavailable from PPLX, under-specified across heads and layers, and not trained for Morgott retrieval.

Corpus-IDF top-eight selection tests the same core idea with the existing index, no new model, no privacy expansion, and a direct path to reducing the measured BM25 postings cost.

If that simple test fails, BGE-M3 lexical weights are the most defensible learned token-weighting experiment before SPLADE or ColBERT.

## Executed local follow-up

The provider-free budget sweep confirmed that 250 ms was an aggressive cutoff rather than a monetary or end-to-end budget.
First-eight BM25 had 37 of 110 four-worker timeouts at 250 ms, 7 at 500 ms, and zero at 1,000 ms.
A three-repeat 1,000 ms determinism run completed all 330 four-worker searches and returned identical candidate hashes for all 110 units, with 571.593 ms p95 and 737.099 ms p99.

The executed corpus-frequency challenger considered the first 32 normalized unique terms and selected at most eight with the lowest combined document frequency across both labels in the request channel.
This differs from the partition-specific Robertson-Sparck Jones formulation proposed above and is recorded as a cheaper preliminary control rather than a substitute for it.
At four workers and a 1,000 ms budget, combined-channel IDF had zero timeouts and 367.529 ms p95.
At the matched 1,000 ms budget, first-eight BM25 returned nine sparse fallbacks, changed 77 dense packets, rescued five of six dense failures, supplied 70 dense-missing selected slots, and left one hybrid packet failure.
Combined-channel IDF returned 14 sparse fallbacks, changed 70 dense packets, rescued the same five failures, supplied 73 dense-missing slots, and left the same one hybrid packet failure.
The IDF control therefore improved latency but failed the no-worse-packet-availability gate.
The frozen sparse branch for the fresh hybrid quality experiment remains first-eight BM25 with a separately versioned 1,000 ms fail-soft execution budget.
