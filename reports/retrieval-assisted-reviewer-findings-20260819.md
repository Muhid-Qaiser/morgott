# Retrieval-assisted reviewer consolidated findings

Date: 2026-08-19.
Updated: 2026-08-20.

This report consolidates the executed retrieval-assisted DeepSeek studies and records the current implementation decision.
It does not replace the machine-readable artifacts or claim that a consumed development panel represents production traffic.

## Current decision

The owner-selected advisory production candidate is:

1. Keep the maintained mmBERT routing thresholds and the fixed DeepSeek reviewer unchanged.
2. Retrieve only for requests that already reach DeepSeek.
3. Use the source-lineage bank rather than the duplicate-heavy all-row bank.
4. Embed the review query with PPLX Embed V1 4B at 256 dimensions.
5. Search four fixed channel-label partitions with a hash-bound Faiss HNSW index using `M=32`, `efConstruction=200`, `efSearch=1024`, top-160 over-retrieval, and exact float32 rescoring to top 20.
6. In parallel, search four contentless SQLite FTS5 partitions with Unicode tokenization, the first eight normalized unique terms, and a separately versioned approximate 250 ms fail-soft execution budget.
7. Fuse dense and sparse rankings with RRF `k=60` and dense-to-sparse weight 2:1.
8. Deterministically select two positive and two negative same-channel examples with source-lineage diversity.
9. Treat every retrieved example as inert user-level data and preserve the fixed binary reviewer schema.
10. Return the byte-identical dense packet when sparse retrieval is empty, interrupted, malformed, or cannot form a valid fusion result.
11. Use the current no-example DeepSeek request when the query embedding, dense bundle, or balanced example packet is invalid.
12. Preserve Morgott's existing `decision: allow` behavior because all learned results remain advisory.

The hybrid branch is an explicit owner-selected defense-in-depth choice.
On the superseding strict-provider-safe development rerun, dense and hybrid retrieval tied on aggregate recall and FPR, while the paired recall interval allowed either loss or gain.
This implementation decision must not be restated as a statistically demonstrated hybrid advantage.

The provider-egress bank now uses an exact public-license allowlist and excludes ambiguous mixed-license rows.
Every retained document vector was reused after an identity join, so the migration made no document-embedding calls.
Fresh HNSW retrieval reproduced every exact-dense selected packet, and a provider-free replay reproduced every strict-provider-safe HNSW-plus-BM25 packet.
The completed all-row HNSW work remains scale and algorithm evidence rather than the production bank selection.
The rebuilt strict-provider-safe bundle passed its exact zero-traffic runtime smoke on 2026-08-20.
The owner then authorized a POC traffic rollout without treating the earlier contradictory latency results as a gate pass.

ColBERT, a learned-sparse model, raw-attention token selection, Qdrant, GraphRAG, and output-verification machinery are not part of this candidate.
They add no required capability to the smallest tested flow.

## Evidence contract

The study kept the full cascade as the evaluation unit and froze these principal gates before the later diagnostics:

- Full-cascade recall gain of at least 1.0 percentage point.
- Absolute FPR increase no greater than 0.25 percentage point.
- No adequately sized critical-slice recall loss greater than 3 percentage points.
- Less than 1 second of added p95 latency on requests already routed to DeepSeek.
- Deterministic fail-soft behavior and no additional terminal review failures.
- A fixed reviewer model, prompt contract, threshold, and four-example balance.

The machine-readable source of truth is the artifact set linked beside each result.
The longer experimental narrative is [retrieval-model-selection-research-20260817.md](retrieval-model-selection-research-20260817.md).

## Embedding selection

### Initial curated screening bank

The first screen compared the same fixed cascade with balanced examples from each candidate.
The PPLX and Voyage arms were close enough that the result supports an incumbent, not a universal model ranking.

| Arm | Recall | FPR | Retrieval fallbacks | Finding |
|---|---:|---:|---:|---|
| No-example baseline | 71.364% | 0.249% | 0 | Control |
| PPLX 4B 256d | 89.545% | 0.124% | 8 | Selected arm |
| PPLX 4B 256d, reversed example order | 90.455% | 0.124% | 8 | Order diagnostic remained favorable |
| Voyage 4 query against Voyage 4 large documents | 89.091% | 0.249% | 8 | Close quality, noisy latency gate failure |
| Voyage 4 lite-query in the shared Voyage space | 90.000% | 0.124% | 6 | Close quality, noisy latency gate failure |
| PPLX plus equal-weight trigram RRF | 87.273% | 0.124% | 24 | Worse than PPLX dense-only |

PPLX dense improved recall over the paired baseline by 18.182 points with a paired 95% interval of `[13.182, 23.182]` points.
Its FPR delta was -0.124 point with a paired interval of `[-0.373, 0.000]` points.
The exact local search p95 was 9.1 ms, while the remote query path produced eight fallbacks across 110 routed review units.
Machine-readable evidence: `artifacts/retrieval_assisted_reviewer/validation-analysis.json` and `artifacts/retrieval_assisted_reviewer/selection.json`.

Perplexity's current first-party contract describes the 4B model as a 2,560-dimensional, 32K-context Matryoshka model and allows API dimensions from 128 through 2,560.
It emits unnormalized embeddings intended for cosine comparison, so Morgott normalizes vectors before inner-product search.
See the [Perplexity embedding documentation](https://docs.perplexity.ai/docs/embeddings/standard-embeddings) and [official model card](https://huggingface.co/perplexity-ai/pplx-embed-v1-4b).

### PPLX dimension ablation

The separate dimension study reused the byte-identical curated-screen bank and gave each arm fresh query embeddings and paired reviewer calls.

| Dimension | Recall | FPR | Exact-search p95 | Raw vector memory | Isolated peak RSS |
|---:|---:|---:|---:|---:|---:|
| 256 | **90.455%** | **0.124%** | **10.9 ms** | **48.8 MiB** | **347.6 MiB** |
| 512 | 89.545% | **0.124%** | 12.7 ms | 97.7 MiB | 444.7 MiB |
| 1,024 | 88.636% | 0.249% | 16.1 ms | 195.3 MiB | 640.7 MiB |

Relative to 256d, 512d changed recall by -0.909 point with a paired 95% interval of `[-2.273, 0.000]` points and did not change FPR.
Relative to 256d, 1,024d changed recall by -1.818 points with a paired interval of `[-4.091, 0.000]` and raised FPR by 0.124 point with an interval of `[0.000, 0.373]`.
No larger-dimension full bank was built because neither candidate improved downstream quality and both increased memory.
Machine-readable evidence: `artifacts/retrieval_assisted_reviewer_dimensions/validation-analysis.json`.

### Qwen contract and resource canaries

The Qwen canaries used public-safe fixed text and made no corpus-bank build or reviewer call.

| Route | Status | p50 | p95 | Worst batch-single drift | Minimum cosine | Resource note |
|---|---|---:|---:|---:|---:|---|
| Qwen3 Embedding 8B 256d via Nebius | Failed frozen `1e-5` component gate | 1,460 ms | 3,121 ms | 0.003461 | 0.999900 | Eight remote calls |
| Qwen3 Embedding 8B 256d via DeepInfra | Failed frozen `1e-5` component gate | 1,851 ms | 4,524 ms | 0.002269 | 0.999933 | Eight remote calls |
| Qwen3 Embedding 0.6B 256d local CUDA BF16 | Failed frozen `1e-5` component gate | 19.060 ms | 465.211 ms | 0.006406 | 0.999715 | 1.227 GB CUDA reserved, 2.276 GB peak RSS |
| Qwen3 Embedding 0.6B 256d local two-core CPU BF16 | Failed frozen `1e-5` component gate | 174.802 ms | 1,583.103 ms | 0.003885 | 0.999807 | 1.908 GB peak RSS |

Every route returned finite normalized 256-dimensional vectors and showed high cosine agreement.
The uniform batch-shape effect means componentwise equality alone is too strict for a future retrieval gate, but changing the threshold after seeing the result would be outcome tuning.
A new canary must predeclare cosine, exact neighbor-overlap, and selected-packet stability before any Qwen bank build.
Machine-readable evidence: `artifacts/retrieval_assisted_reviewer_qwen_stage0/qwen3-embedding-8b-256-stage0-nebius.json`, `artifacts/retrieval_assisted_reviewer_qwen_stage0/qwen3-embedding-8b-256-stage0-deepinfra.json`, `artifacts/retrieval_assisted_reviewer_qwen_local_stage0/qwen3-embedding-0.6b-256-local-stage0-cuda.json`, and `artifacts/retrieval_assisted_reviewer_qwen_local_stage0/qwen3-embedding-0.6b-256-local-stage0-cpu.json`.

## Bank size and transfer

The eligible train population yielded three main bank designs: a curated screening bank, source-lineage representatives, and all eligible rows.
The lineage bank collapses correlated variants rather than treating mutations of the same source example as independent coverage.

### Consumed validation comparisons

| Bank | Paired no-example recall | Dense recall | Dense FPR | Retrieval fallbacks | Exact-search p95 | Dense retrieval p95 including embedding |
|---|---:|---:|---:|---:|---:|---:|
| Source-lineage representatives | 69.545% | **93.636%** | 0.249% | 0 | 19.1 ms | 756.0 ms |
| All eligible rows | 71.818% | 92.727% | 0.249% | 6 | 65.1 ms | 723.7 ms |

The lineage gain over its paired baseline was 24.091 points with a paired 95% interval of `[18.636, 30.000]` points.
The all-row gain over its paired baseline was 20.909 points with a paired 95% interval of `[15.898, 26.364]` points.
The validation calls were separate executions with provider noise, so their absolute reviewer latency and slightly different baselines are not a clean direct bank comparison.
Machine-readable evidence: `artifacts/retrieval_assisted_reviewer_full/validation-analysis.json` and `artifacts/retrieval_assisted_reviewer_full_rows/validation-analysis.json`.

### Frozen 12,000-row confirmation

The all-row dense arm was frozen before this dev-test was opened.
The local cascade routed 1,333 of 12,000 artifacts to DeepSeek.

| Arm | Recall | FPR | Retrieval fallbacks | Terminal review failures |
|---|---:|---:|---:|---:|
| No-example reviewer | 89.074% | 1.168% | 0 | 0 |
| All-row PPLX 256d | **94.808%** | 1.274% | 12 | 1 |

Recall improved by 5.734 points with a paired 95% interval of `[4.843, 6.625]` points.
FPR increased by 0.106 point, and the paired 95% upper bound was 0.223 point, inside the frozen 0.25-point limit.
The marginal p95 changed by only 1.2 ms, while the paired latency-delta p95 was 2.928 seconds, so this run confirms quality rather than production latency.
Machine-readable evidence: `artifacts/retrieval_assisted_reviewer_full_rows/dev_test-analysis.json`.

### Prospective WMT source-heldout comparison

The WMT panel was independent of Morgott fitting and retrieval selection when opened.
It contained 810 clean-and-attacked matched pairs across five attack forms.

| Arm | Recall | FPR | Retrieval fallbacks | Exact-search p95 | Added marginal p95 |
|---|---:|---:|---:|---:|---:|
| No-example reviewer | 17.284% | 0.123% | 0 | n/a | 0 |
| All-row PPLX 256d | 39.136% | 0% | 3 | 18.5 ms | 0.712 s |
| Lineage PPLX 256d | **43.333%** | **0%** | **0** | **11.4 ms** | 0.821 s |

Lineage minus all-row recall was +4.198 points with a paired matched-group 95% interval of `[+1.358, +7.037]` points.
FPR differed by exactly zero, no attack subtype regressed, and lineage exact search was 38% faster.
The paired reviewer-latency p95 values remained noisy at 4.716 seconds for lineage and 5.135 seconds for all rows.
Both banks had 0% recall on the one-shot task-switch slice because all 162 attacks passed below the local route-to-review floor, which retrieval after routing cannot repair.
The WMT suite establishes source-heldout transfer for this synthetic translation task, not broad production robustness or foundation-training independence.
Machine-readable evidence: `artifacts/retrieval_assisted_reviewer_wmt_external/external-analysis.json`.

This is the strongest direct bank comparison and is the reason the production candidate uses lineage representatives.

## Dense search and HNSW

Faiss documents `IndexFlatIP` as exhaustive inner-product search and `IndexHNSWFlat` as approximate graph search with a memory and recall-latency tradeoff controlled by `M` and `efSearch`.
Cosine retrieval requires normalized vectors before inner-product search.
See the official [Faiss index summary](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), [index guidance](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index), and [metric documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances).

### Full-row backend benchmark

| Backend | Four-worker p95 | Throughput | Mean set Recall@20 | Exact packet matches | Downstream recall |
|---|---:|---:|---:|---:|---:|
| NumPy exact, HNSW-extension run | 157.396 ms | 38.819 QPS | Ground truth | Ground truth | 93.182% |
| Faiss FlatIP, isolated rerun | 180.002 ms | 31.888 QPS | 99.977% | 104/110 | Not separately called |
| HNSW `ef256/top80` | 3.815 ms | 1,417.458 QPS | 96.250% | 84/110 | Not called because retrieval gate failed |
| HNSW `ef512/top160` | 5.667 ms | 898.497 QPS | 98.318% | 97/110 | 92.727% |
| HNSW `ef1024/top160` | **9.740 ms** | **546.684 QPS** | **99.409%** | **101/110** | **93.182%** |
| HNSW `ef1024/top320` | 10.778 ms | 473.445 QPS | 99.409% | 101/110 | Dominated by top160 |

FlatIP was score-tolerant exact, including 100% tie-aware score recall, but supplied no four-worker speed or throughput win over NumPy.
The first HNSW sweep was fast but its best `efSearch=256` arm missed the frozen 98% mean Recall@20 gate.
Extending the curve to `efSearch=1024` and exact-rescoring 160 candidates raised mean Recall@20 to 99.409%, kept the worst adequately sized slice at 99.057%, and produced a 16.16-fold p95 speedup over its same-matrix NumPy control.

The fixed reviewer result for `ef1024/top160` exactly matched fresh NumPy at 93.182% recall, 0.249% FPR, six retrieval fallbacks, and zero terminal review failures.
The recall delta and paired 95% interval versus fresh NumPy were both exactly zero.
Machine-readable evidence: `artifacts/retrieval_assisted_reviewer_full_rows/validation-dense-indexes-pplx-4b.json`, `artifacts/retrieval_assisted_reviewer_full_rows/validation-hnsw-extension-pplx-4b.json`, and `artifacts/retrieval_assisted_reviewer_hnsw_cascade/validation-hnsw-cascade-analysis.json`.

### Persistent full-row runtime

The serialized four-index bundle and row maps occupied about 939.3 MiB, while its separate metadata bank occupied about 721 MiB.
The one-thread build took 334.857 seconds with Faiss 1.15.0.
Eight deterministic integrity queries matched ranking, scores, selected packet, and packet hash exactly across serialization.

| Runtime cell | Total p50 | Total p95 | Total p99 | Throughput |
|---|---:|---:|---:|---:|
| Concurrency 1 | 4.291 ms | 8.304 ms | 9.046 ms | 189.2 QPS |
| Concurrency 4 | 16.018 ms | 55.476 ms | 59.556 ms | 167.2 QPS |

The fresh quota-constrained process used about 1.17 GiB RSS after the warm run.
It had a two-CPU quota and 4 GiB memory maximum, but CPU affinity still covered the host's 12 logical CPUs and the page cache was warm.
It did not load mmBERT, embed a live query, assemble the longer prompt, or call DeepSeek.
Machine-readable evidence: `artifacts/retrieval_assisted_reviewer_hnsw_persistent/persistent-hnsw-manifest.json` and `artifacts/retrieval_assisted_reviewer_hnsw_persistent/persistent-hnsw-local-resource.json`.

The conclusion is narrow: in-process HNSW is proven viable for the immutable all-row scale and removes row count alone as a reason to deploy Qdrant.

### Maintained source-lineage HNSW bundle

The rebuilt strict-provider-safe source-lineage HNSW run reached 99.818% mean Recall@20 and 99.713% on the worst adequately sized slice.
It reproduced all selected packets from the same-query exact NumPy control.
Replaying the fixed Unicode BM25 and 2:1 RRF branch reproduced every freshly reviewed hybrid packet.

| Runtime evidence | Exact NumPy | HNSW candidate | Finding |
|---|---:|---:|---|
| Four-worker search p95 | 37.791 ms | 11.481 ms | About 3.3 times faster |
| Selected-packet parity | Reference | 100% | No reviewer-input change |

The strict-provider-safe reviewer rerun moved recall from 71.818% without retrieval to 94.091% for both dense and hybrid retrieval.
FPR moved from 0.373% to 0.249% for both retrieval arms.
The recall gain versus baseline was 22.273 points with paired 95% intervals of `[16.818, 27.727]` points for dense and hybrid.
Hybrid changed 80 of the 110 routed packets and selected 48 example slots absent from the saved dense top 20, but six reviewer verdicts canceled to the same aggregate outcome.
Directly against dense, the hybrid recall delta was zero with a paired interval of `[-2.273, 2.273]` points, and the FPR delta and interval were both zero.
This supports the full retrieval pipeline over the no-example reviewer, but it still does not establish an incremental hybrid quality gain.

The retained resource canary is bound to an earlier manifest revision, so its latency and memory values are not attributed to the rebuilt bundle here.
The registered recipe is bound in `artifacts/models/mmbert-lora-full-ctx1024-u17000-s42/serving/promotion-retrieval.json`.
The tracked strict-provider-safe parity record is `reports/retrieval-lineage-hybrid-parity-20260820.json`.

### Co-resident Azure zero-traffic canary

The completed Azure run compared the no-example stable revision with the retrieval-assisted candidate on 15 randomized AB/BA pairs of one fixed public synthetic review-route probe.
Both revisions used the same model and threshold identities on a 2-vCPU, 4-GiB shape, but the stable revision used OpenVINO 2026.2.1 and the candidate used OpenVINO 2026.3.0.

| Measurement | Stable no-example | Retrieval-assisted candidate | Observed change |
|---|---:|---:|---:|
| Local-pass p95 | 30.187 ms | 31.062 ms | +0.875 ms |
| Local-pass throughput | 40.468 QPS | 38.008 QPS | -6.08% |
| Process peak RSS | 2.437 GiB | 2.508 GiB | +73.1 MiB |
| Routed client p95 | 15,140.900 ms | 6,835.620 ms | -8,305.280 ms |
| Routed provider p95 | 14,085.435 ms | 1,923.671 ms | -12,161.764 ms |
| Routed service total p95 | 14,115.295 ms | 1,956.960 ms | -12,158.334 ms |

The candidate retrieval p95 was 146.113 ms, with marginal stage p95 values of 133.325 ms for embedding, 10.667 ms for dense search, 3.284 ms for sparse search, and 1.528 ms for fusion.
Those component percentiles are not additive because they summarize different requests and the dense and sparse branches execute concurrently.
The candidate process peak left 1.492 GiB of headroom against the declared 4-GiB revision limit, but cgroup-v2 current, peak, and limit values were unavailable, so the record uses process `VmHWM` and the declared Azure limit.

The stable arm recorded 16 reviewer calls, 5,160 reviewer input tokens, 120 reviewer output tokens, and zero terminal reviewer failures.
The candidate recorded 15 reviewer calls, 9,195 reviewer input tokens, 105 reviewer output tokens, 165 embedding input tokens, all 15 retrievals as `ok`, four selected examples per request, and zero terminal reviewer failures.
The only response-exposed candidate cost was $0.00000495 for embeddings, while reviewer billed cost was not exposed, so the run does not support a total-cost comparison.
Both local smokes recorded zero errors.

Deployment to `Running` took 174 seconds.
The candidate remained at 0% traffic and was not promoted.
The routed latency point estimates passed the predeclared no-added-second gate, but the large provider-latency difference cannot be attributed to retrieval in this small run.
This is a 15-pair synthetic deployment canary with provider variance and an OpenVINO-version difference, not quality evidence or a production latency distribution.
Machine-readable evidence: [azure-preview-retrieval-canary-20260819T174113Z.json](azure-preview-retrieval-canary-20260819T174113Z.json).

A subsequent promotion run on the same protocol failed the frozen less-than-one-second added-p95 gate and automatically restored the stable revision before traffic moved.
The then-current script exited before persisting that failed summary, so its exact measurements are unavailable; the consumed single-probe comparison path has since been removed.
The opposing pass and fail results show that nearest-rank p95 over 15 repeated requests is a single provider-tail observation, not a reproducible retrieval-latency estimate.
Repeating the same test until it passes would be optional stopping, so the next attempt requires a predeclared larger, multi-probe paired protocol that preserves every run.

### Owner-authorized POC rollout on 2026-08-20

Azure revision `morgott-api--0000019` passed the exact model, policy, retrieval-manifest, packet, prompt, provider, and memory-headroom smoke while receiving 0% traffic.
The same revision then received 100% preview traffic, and a public routed request returned retrieval status `ok`, four selected examples, one successful DeepSeek call, zero DeepSeek failures, and advisory decision `allow`.
Revision `morgott-api--0000016` remains healthy and active at 0% as the rollback point.
This was an explicit owner-authorized POC rollout, not a new latency experiment, quality evaluation, or statistically valid promotion-gate result.

## BM25 and RRF

SQLite FTS5 supplies the Unicode61 and trigram tokenizers, contentless indexes, and built-in BM25 rank used by these experiments.
The trigram tokenizer implements substring matching and is not equivalent to a fuzzy character-gram retriever.
See the official [SQLite FTS5 reference](https://www.sqlite.org/fts5.html).

### Earlier hybrid evidence

The curated-screen equal-weight trigram RRF arm reduced recall from 89.545% for PPLX dense to 87.273%, increased fallbacks from 8 to 24, and was slower.

The redesigned partitioned Unicode index on the lineage bank occupied about 21 MiB.
Sparse retrieval formed packets for 91 of 110 units with an 87.905 ms p95 and 214.170 ms maximum under its approximate 250 ms fail-soft setting.
The dense-plus-sparse replay formed all 110 packets, changed 78 packets, and had a 755.973 ms concurrent component estimate.
Sparse-only reached 86.364% recall and 0.124% FPR, while dense and hybrid both reached 93.636% recall and 0.249% FPR.
Six unit verdicts moved in opposite directions and canceled at the aggregate, so the extra branch showed novelty but no quality gain on this consumed panel.
Machine-readable evidence: `artifacts/retrieval_assisted_reviewer_full_sparse_v2/validation-analysis.json`.

### Full-row ef1024 hybrid at the historical 250 ms cutoff

The later full-row sidecar occupied 58.5 MiB, built in 9.918 seconds, used Unicode tokenization, retained 50 unique lineages per label from 320 raw hits, and fused with dense using RRF `k=60` at weight 2:1.
The nominal 250 ms value was an approximate SQLite virtual-machine execution cutoff checked every 1,000 operations, not a financial budget or a hard request deadline.

| Retrieval measurement | Result |
|---|---:|
| HNSW four-worker p95 | 9.740 ms |
| BM25 four-worker p50 / p95 | 180.085 / 259.463 ms |
| RRF plus selection p95 | 0.802 ms |
| Four-worker BM25 interruptions | 123/330 |
| Hybrid packets changed from dense | 59/110 |
| Selected slots absent from saved dense top 20 | 47 |
| Dense packet failures rescued | 3/6 |
| Sequential materialization sparse fallbacks | 28/110, including 19 timeouts |

The fusion cost was negligible relative to postings traversal.
The sparse deadline behavior was too load-sensitive to call this a stable serving policy.

| Reviewer arm | Recall | FPR | Retrieval fallbacks | Terminal failures |
|---|---:|---:|---:|---:|
| Full-row HNSW `ef1024/top160` | 93.182% | 0.249% | 6 | 0 |
| HNSW plus partitioned BM25 RRF | **93.636%** | **0.124%** | **3** | 0 |

Hybrid minus dense recall was +0.455 point with a paired 95% interval of `[-0.909, +2.273]` points.
Hybrid minus dense FPR was -0.124 point with a paired interval of `[-0.373, 0.000]` points.
The hybrid added one true positive and removed one false positive while leaving the total restriction rate unchanged.
It changed only 59 DeepSeek requests after exact response reuse and cost $0.025870504 in new reviewer calls.
Machine-readable evidence: `artifacts/retrieval_assisted_reviewer_full_rows_hnsw_sparse/validation-hnsw-hybrid-analysis.json`.

### Corrected 1,000 ms full-row diagnostic and IDF diagnostic

A matched provider-free timeout sweep over the same 110 queries produced 37 timeouts at 250 ms, 7 at 500 ms, and none at 1,000 ms.
The first three-repeat 1,000 ms confirmation completed all 330 four-worker searches with 510.972 ms p95, 663.568 ms p99, and 21.220 QPS.
A separate three-repeat determinism run again had zero timeouts and identical hashes for every unit, with 571.593 ms p95, 737.099 ms p99, and 20.614 QPS.
This range is retained as local run variance rather than selecting the better number.

The matched first-eight 1,000 ms replay changed 77 dense packets, rescued five of six dense packet failures, returned nine sparse fallbacks, and left one hybrid packet failure.
The corpus-IDF top-eight selector reduced p95 to 367.529 ms and supplied 73 selected dense-missing slots, but returned 14 sparse fallbacks and rescued no additional dense failure.
It therefore failed the predeclared no-worse packet-availability gate and did not replace first-eight selection.

No reviewer call was made for the corrected full-row 1,000 ms or IDF packets.
Their evidence is operational and deterministic, not a new hybrid quality result.
See [retrieval-token-selection-research-20260819.md](retrieval-token-selection-research-20260819.md).

The 1,000 ms first-eight result corrects the full-row diagnostic but is not the lineage production budget.
The production candidate consequently uses first-eight Unicode BM25 with the lineage bank's approximate 250 ms fail-soft setting.
Its inclusion is the stated owner choice despite unresolved incremental quality, and dense HNSW remains the exact sparse-failure result.

## ColBERT and late interaction

The tested late-interaction component was `lightonai/mLateOn`, a 307M-parameter, 128-dimensional MaxSim checkpoint with an 8,192-token query and document contract.
The first-party model card labels it Apache-2.0 and multilingual.
See the [mLateOn model card](https://huggingface.co/lightonai/mLateOn).

The off-by-default experiment reranked only the frozen HNSW-plus-BM25 candidate union and built no corpus-wide token index.
Stage 0 matched Sentence Transformers MaxSim within `9.54e-7`, matched the published canary scores within `4.20e-5`, and loaded in 6.60 seconds.
The 110-unit replay had zero sparse timeouts, nine sparse fallbacks, and the same one inherited RRF packet failure.

| Measurement | Result |
|---|---:|
| Changed selected packets | 109/110 |
| Candidate document cache | 8,204 documents, 238,525,440 bytes |
| Cache build time | 64.01 s |
| Added latency p50 / p95 / p99 | 12.36 / 235.21 / 423.43 ms |
| Maximum added latency | 995.54 ms |
| Estimated HNSW plus BM25 plus MaxSim p95 | 378.89 ms |
| Peak process RSS | 2,627,984 KiB |
| Peak CUDA reserved memory | 2,814,377,984 bytes |

The reranker materially changed order, but it failed the predeclared 100 ms added-stage p95 target and received no downstream reviewer quality comparison.
It is not part of maintained or Azure inference.
Machine-readable evidence: `artifacts/retrieval_assisted_reviewer_colbert/validation-mlateon-rerank.json`.
See the [late-interaction report](late-interaction-research-20260819.md).

The next late-interaction iteration should use a fresh panel and the same frozen candidate union before considering LateOn 149M, PPLX Late 0.6B, PLAID, Qdrant multivectors, or a complete token index.

## Learned sparse and attention research

No learned-sparse bank or full-cascade arm has been executed.
The first production-eligible challenger selected by primary-source research is local `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte`.
Its asymmetric contract performs neural document expansion offline while live queries use the pinned tokenizer and frozen IDF lookup.
The official card is Apache-2.0 and documents that query contract.
See the [OpenSearch v3-gte model card](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte) and [learned-sparse report](learned-sparse-model-research-20260819.md).

BGE-M3 remains the conditional broader multilingual and long-context sparse challenger.
SPLADE-v3 is research-only unless its noncommercial license receives a separate decision.
Neither should enter the current PR because ordinary BM25 has not yet demonstrated a statistically reliable incremental cascade gain.

Raw attention outliers are not a sound next selector.
The PPLX API returns one embedding per input and exposes no token attention tensor, while transformer attention is a layer-by-head token-to-token matrix rather than one defined importance score.
Primary studies also show that raw attention need not track gradient importance and that BERT heads perform diverse positional, delimiter, syntactic, and coreference functions.
See [Attention is not Explanation](https://aclanthology.org/N19-1357/) and [What Does BERT Look at?](https://aclanthology.org/W19-4828/).

The executed corpus-IDF diagnostic tested the useful core intuition with no second model.
It was faster but less reliable than first-eight BM25, so it was rejected without another outcome-tuned selector.

## Cost record

At the 2026-08-20 audit point, the current mutable ledger reserved or settled $21.540407656 against its $50 cap with a separate $2 reserve.
This sum comes from `artifacts/retrieval_assisted_reviewer-budget.json` and is a research accounting snapshot, not a closed total or projected production cost.

Notable incremental calls were:

- The strict-provider-safe 110-query PPLX HNSW matrix used 21,881 tokens, took 2.829 seconds as one batch, and cost $0.00065643.
- The rebuild reused every retained document vector, spent another $0.00065643 on the exact-dense query pass, and spent $0.09358736 on 330 fresh baseline, dense, and hybrid reviewer calls.
- The HNSW cascade made 119 new reviewer calls after 211 exact response reuses and cost $0.037944432, or $0.038600862 including that embedding batch.
- The full-row hybrid called the reviewer only for 59 changed packets and cost $0.025870504.
- Each eight-call Qwen remote canary cost $0.00000276.
- The local mLateOn and IDF diagnostics had zero provider spend.

Production measurement must report embedding tokens, longer-prompt tokens, reviewer calls, cost per routed request, and cost per total request.
OpenRouter's API permits embedding dimensions and provider preferences, while its routing documentation says provider fallback is enabled unless disabled.
An immutable index therefore requires a pinned compatible embedding route and must never fall back to another model's vector space.
OpenRouter does not expose an immutable PPLX weight revision behind that route.
The maintained policy binds the endpoint slug and request identity, and zero-traffic validation checks a frozen selected-packet hash.
The replacement scheduled retrieval canary is deferred with the new promotion protocol, and neither check cryptographically attests every provider weight.
See the official [OpenRouter embeddings API](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings) and [provider-routing guide](https://openrouter.ai/docs/guides/routing/provider-selection).

The owner does not require zero data retention, but the live query still crosses a provider boundary.
The private bank and indexes remain local or in the private Azure artifact store, while the four selected provider-safe public example texts cross to the pinned DeepSeek provider with the reviewed text.
OpenRouter documents provider-specific logging, retention, and training policies, so the selected endpoint and account policy must be recorded rather than inferred from key availability.
See the official [provider logging and retention documentation](https://openrouter.ai/docs/guides/privacy/provider-logging/).

## Evidence status and caveats

| Evidence | Status now | What it supports |
|---|---|---|
| 1,024-row canonical validation | Consumed | Development comparisons only |
| 12,000-row dev-test | Frozen prospective confirmation when run, now consumed | All-row dense quality confirmation |
| WMT matched-pair source | Prospective source-heldout when run, now consumed | Lineage over all-row on one synthetic transfer suite |
| HNSW, hybrid, IDF, and ColBERT replays | Post-hoc on consumed validation | Mechanics, scale, latency, packet change, and exploratory quality only |
| Qwen public-safe canaries | Contract and resource diagnostics | No Morgott retrieval quality conclusion |
| Strict-provider-safe source-lineage HNSW bundle | Completed locally and uploaded privately | Exact-dense retrieval, immutable payload, and provider-egress contract |
| Strict-provider-safe HNSW plus BM25 packet replay | Completed provider-free replay | Exact fresh reviewed-packet parity for the maintained recipe |
| New source-and-time-heldout selection block | Not collected | Required for an unbiased hybrid or reranker quality claim |
| Untouched confirmation block | Not collected | Required to reproduce the new direction |
| Earlier co-resident Azure end-to-end run | Completed at 0% traffic on the superseded bundle | Historical deployment latency and process-memory evidence only |
| Strict-provider-safe Azure end-to-end run | Not run | Required before any traffic decision |

Provider latency was noisy enough that marginal p95 comparisons sometimes contradicted paired request deltas.
Local component timing must therefore be reported separately from end-to-end routed latency.

The saved full-row HNSW runtime used warm page cache, did not load the maintained detector, and was not an Azure hardware result.
Its 1.17 GiB process RSS cannot be added mechanically to a separately measured mmBERT RSS and called a valid capacity proof.

The WMT route-floor miss shows an architectural ceiling: retrieval cannot improve examples for requests that never reach DeepSeek.
The current work must not retune that floor on consumed evidence.

## Required before-and-after record

The integration PR must add a machine-readable result and a concise report that compare the current no-example pipeline with the exact owner-selected candidate on the same frozen inputs.

The record must include:

- Recall, FPR, paired 95% deltas, macro-domain results, and worst adequately sized domain.
- Results by channel, source, time, language, length, attack subtype, and task-bearing benign category.
- Dense candidate coverage, sparse-only selected examples, packet changes, source concentration, and lineage concentration.
- Embedding, HNSW, BM25, RRF, prompt assembly, DeepSeek, and total p50, p95, and p99 latency.
- Startup time, artifact bytes, image bytes, RSS, cgroup peak, CPU, QPS, timeout rate, and OOM or swap events.
- Embedding failures, sparse fallbacks, dense failures, invalid packets, reviewer failures, and exact fallback parity.
- Embedding tokens, reviewer prompt tokens, provider-native spend, and cost per routed and total request.
- Code commit, image digest, prompt identity, request identity, bank manifest, and index manifest by reference to their machine records.
- A plain-language account of improvements, regressions, limitations, and the final deployment disposition.

The baseline and candidate must run in randomized AB/BA order on the same privacy-approved fixed requests.
Existing Azure pass-path numbers are historical references rather than a valid routed baseline for this feature.

### Current before-and-after summary

| Metric | Before | Integrated candidate | Change or interpretation |
|---|---:|---:|---|
| Strict-provider-safe development recall | 71.818% | 94.091% | +22.273 points, paired 95% interval `[16.818, 27.727]` |
| Strict-provider-safe development FPR | 0.373% | 0.249% | -0.124 point, paired interval `[-0.373, 0.000]` |
| Four-worker dense search p95 | 37.791 ms exact NumPy | 11.481 ms HNSW | About 3.3 times faster |
| Hybrid packet parity after HNSW substitution | Exact-dense reference | 100% | No additional HNSW-specific reviewer calls after the fresh exact-hybrid run |
| Azure routed latency and rollout | One 15-pair run passed; a later identical run failed | Strict bundle is live for owner-authorized POC use | Latency remains inconclusive; rollout is not a benchmark pass |

## Remaining gates

Before the candidate can be described as target-proven, complete these checks without changing the frozen recipe:

1. Run the full lineage hybrid on a new independently adjudicated source-and-time-heldout selection block and preserve a separate untouched confirmation block.
2. Report the hybrid as an owner-selected advisory layer even if its gain remains statistically unresolved, and report any regression without hiding it behind the aggregate.
3. Rerun the resource gate for the strict-provider-safe bundle and preserve cgroup, swap, and process-memory evidence.
4. Treat the favorable routed latency point estimate as provider-noisy deployment evidence rather than a retrieval speedup claim.
5. Treat the current 100% preview rollout as an owner-authorized POC state, preserve the healthy rollback revision, and require a larger predeclared multi-probe protocol before making latency or production-readiness claims.
6. Keep the result advisory and keep every side effect behind the deterministic reference monitor.

No vector database or reranker should be added unless these checks expose a requirement that the current in-process static design cannot meet.
