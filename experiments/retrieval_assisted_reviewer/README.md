# Retrieval-assisted reviewer benchmark

This disposable study tests whether four train-only labeled examples improve the fixed maintained DeepSeek reviewer and full advisory cascade.
It does not change the API, Azure deployment, model registry, thresholds, reviewer model, reviewer provider, or advisory authority.

The runner stores raw example text only in the gitignored local SQLite bank.
Frozen panels, score records, retrieval ledgers, and provider ledgers contain identities, hashes, parsed outputs, timings, token usage, and cost, but no raw prompts or provider responses.

## Qwen Stage 0 contract canary

The Qwen Stage 0 command tests only the remote embedding contract using two hardcoded, public-safe synthetic samples.
It sends documents without a prefix and queries with the frozen `Instruct: ...\nQuery:` transformation, requests exactly 256 dimensions, and rejects any transformed input over 16,000 bytes locally.
It pins the request to the selected `nebius` or `deepinfra` provider with fallbacks disabled and required-parameter support enabled.
It does not request ZDR routing.
The canary checks finite, nonzero, normalized vectors, repeat-input stability, and batch-versus-single parity with a `1e-5` absolute tolerance.
It stores only text and vector hashes plus returned model, timing, token, and available cost metadata, never raw sample text, vectors, or provider responses.
The returned provider field is retained for diagnosis, but an OpenRouter embedding response cannot prove which provider served the request.
Each provider run reserves exactly $0.01 in the shared research ledger and reuses an already-passed artifact without making another paid call.
This command cannot build an embedding bank or call DeepSeek, and Qwen is intentionally unavailable to `embed-bank`.

```bash
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output artifacts/retrieval_assisted_reviewer_qwen_stage0 \
  qwen-stage0-canary --provider nebius
```

Use `--provider deepinfra` to run the same pinned contract against DeepInfra instead.

The local control uses the pinned `Qwen/Qwen3-Embedding-0.6B` revision already present in the Hugging Face cache and refuses network fallback.
It applies left padding without truncation, pools the final token, takes the first 256 dimensions, and then L2-normalizes.
It records only hashes and runtime telemetry and remains unavailable to `embed-bank`.

```bash
uv run --locked --extra cascade --extra encoder \
  python -m experiments.retrieval_assisted_reviewer.run \
  --output artifacts/retrieval_assisted_reviewer_qwen_local_stage0 \
  qwen-local-stage0-canary --device cpu
```

Use `--device cuda` for the same contract with CUDA memory telemetry.

Run the stages in order:

```bash
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run prepare
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run score --split validation
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run retrieve --split validation
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run embed-bank --config voyage-large
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run retrieve --split validation --config voyage-large
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run review --split validation --arms baseline,wrapper,fixed,sparse_unicode,dense_voyage-large,hybrid_voyage-large_unicode
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run analyze --split validation
```

`prepare` creates the deterministic curated screening bank, a 1,024-row validation panel, and a sealed 12,000-row dev-test confirmation panel.
The final panel is not called until validation selects one configuration.
`OPENROUTER_API_KEY` may come from the process environment or the repository's existing targeted `.env` loader; the value is never printed or persisted.
Provider calls across curated and full-bank output directories share one $50 ledger and retain a $2 reserve.

Only a validation winner advances to two larger-bank comparisons.
`full-lineage` keeps one deterministic representative per label, channel, subtype, source, and lineage cell.
`full` keeps every eligible training row, including repeated variants.

```bash
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run --output artifacts/retrieval_assisted_reviewer_full prepare --bank-size full-lineage
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run --output artifacts/retrieval_assisted_reviewer_full score --split validation
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run --output artifacts/retrieval_assisted_reviewer_full embed-bank --config pplx-4b
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run --output artifacts/retrieval_assisted_reviewer_full retrieve --split validation --config pplx-4b
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run --output artifacts/retrieval_assisted_reviewer_full review --split validation --arms baseline,dense_pplx-4b
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run --output artifacts/retrieval_assisted_reviewer_full analyze --split validation
```

Repeat the same `score`, `embed-bank`, `retrieve`, `review`, and `analyze` commands under `artifacts/retrieval_assisted_reviewer_full_rows` after preparing it with:

```bash
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run --output artifacts/retrieval_assisted_reviewer_full_rows prepare --bank-size full
```

After validation freezes a winner, run exactly that arm once on dev-test:

```bash
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run --output artifacts/retrieval_assisted_reviewer_full_rows score --split dev_test
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run --output artifacts/retrieval_assisted_reviewer_full_rows retrieve --split dev_test --config pplx-4b
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run --output artifacts/retrieval_assisted_reviewer_full_rows review --split dev_test --arms baseline,dense_pplx-4b
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run --output artifacts/retrieval_assisted_reviewer_full_rows analyze --split dev_test
```

Bank embedding defaults to one worker because the observed OpenRouter/PPLX quota rejected sustained concurrency of two and four.
Every batch commits independently, so rerunning resumes at the first missing row.

The first dense configurations are `voyage-large`, `voyage-4`, `voyage-lite`, and `pplx-4b` at 256 dimensions.
The post-selection dimension diagnostic also exposes `pplx-4b-512` and `pplx-4b-1024` for a separate curated-screen study.
Voyage query variants share the `voyage-4-large` document index.
Gemini is intentionally absent until the first two families are inconclusive.
Reranking is intentionally absent until no-rerank hybrid retrieval demonstrates an ordering problem.
Exact NumPy search is the dense baseline; HNSW is intentionally absent until full-bank exact search misses the latency gate.

The completed follow-ups used separate artifact directories so they could not overwrite the frozen selection or dev-test evidence.
The dimension arm ran fresh baseline, 256d, 512d, and 1,024d reviewer calls on the byte-identical curated-screen bank.
The 256d arm won on quality, latency, memory, and provider reliability, so no larger full-bank index was built.

The full-bank hybrid diagnostic is generated with one locked setting:

```bash
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output artifacts/retrieval_assisted_reviewer_full_rows_hybrid \
  retrieve-weighted-hybrid --config pplx-4b --tokenizer trigram
```

It over-retrieves 50 candidates per label, removes repeated source-lineage groups, and applies a 2:1 dense-to-sparse RRF weight.
It did not beat dense-only and failed the retrieval latency gate, so no reranker was tested.

The redesigned Unicode diagnostic uses a separate derived output so it cannot alter the frozen lineage selection.
It builds a hash-bound contentless FTS5 sidecar with one table per channel-label partition, uses the first eight normalized terms, and retrieves 50 candidates per label with `ORDER BY rank`.
The replay reuses saved `dense_pplx-4b` top-20 candidates and makes no embedding or reviewer calls.
Create the derived output from immutable local evidence:

```bash
sparse_output=artifacts/retrieval_assisted_reviewer_full_sparse_v2
install -d "$sparse_output"
cp --reflink=auto \
  artifacts/retrieval_assisted_reviewer_full/{manifest.json,validation-panel.jsonl.gz,validation-runtime.json,validation-scores.jsonl.gz,validation-review-units.jsonl.gz,validation-retrieval.jsonl,validation-reviews.jsonl} \
  "$sparse_output"/
ln -s ../retrieval_assisted_reviewer_full/bank.sqlite3 "$sparse_output"/bank.sqlite3
```

Build and replay the local indexes:

```bash
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output "$sparse_output" build-partitioned-sparse-index
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output "$sparse_output" replay-partitioned-hybrid
```

The emitted methods are `sparse_unicode_partitioned8_lineage50` and `hybrid_pplx-4b_unicode_partitioned8_sparse50_dense20_rrf2_replay`.
The hybrid accepts valid partial sparse rankings, but any sparse execution, validation, or fusion failure returns the exact saved dense packet.
Its latency is explicitly a concurrent replay estimate using the historical dense latency, not a new end-to-end concurrency measurement.
The sidecar stores no document text column, but its vocabulary and postings remain sensitive derived corpus data and stay local and gitignored.

The exact full-bank benchmark keeps Faiss outside the lock file and pins the ephemeral package used for the recorded run:

```bash
uv run --locked --extra cascade --with faiss-cpu==1.15.0 \
  python -m experiments.retrieval_assisted_reviewer.run \
  --output artifacts/retrieval_assisted_reviewer_full_rows \
  benchmark-faiss-flat --config pplx-4b
```

FlatIP missed the strict ordered top-20 parity gate and was slower than NumPy at four workers.
NumPy retained operational headroom, so HNSW was not triggered.
That historical command keeps its original result as provenance.
The corrected growth diagnostic embeds the frozen validation queries once, shares that byte-identical temporary matrix, and runs NumPy, FlatIP, and HNSW in separate child processes:

```bash
uv run --locked --extra cascade --with faiss-cpu==1.15.0 \
  python -m experiments.retrieval_assisted_reviewer.run \
  --output artifacts/retrieval_assisted_reviewer_full_rows \
  benchmark-dense-indexes --config pplx-4b
```

The HNSW arm fixes `M=32` and `efConstruction=200`, compares `efSearch` 64, 128, and 256, retrieves 80 candidates, and exactly rescores the final 20.
All native search libraries use one thread, while latency and throughput are measured with one and four Python workers.
The result records tie-aware score parity, ordinary set recall, build and load time, memory, CPU and native-library metadata, and explicitly labels the local host as different from the 2-vCPU/4-GiB Azure target.
It is an infrastructure diagnostic and does not alter maintained inference or authorize a deployment conclusion.

The bounded HNSW extension keeps the earlier result untouched and tests only `efSearch` 512 and 1,024 crossed with over-retrieval depths 160 and 320:

```bash
uv run --locked --extra cascade --with faiss-cpu==1.15.0 \
  python -m experiments.retrieval_assisted_reviewer.run \
  --output artifacts/retrieval_assisted_reviewer_full_rows \
  benchmark-hnsw-extension --config pplx-4b
```

It keeps `M=32` and `efConstruction=200`, exactly rescores each candidate pool in float32 to the final top 20, and compares every variant with NumPy using one fresh byte-identical PPLX query matrix in isolated child processes.
The write-once result retains only candidate IDs, scores, selected four-example IDs, and binding hashes needed for a later cascade comparison, never raw text or query vectors.
Advancement requires mean candidate set Recall@20 of at least 0.98, at least 0.95 in every adequately sized predeclared slice, and at least a 2x local p95 speedup over NumPy.
Selected packet parity is diagnostic, and any passing variant still requires a separate full-cascade gate before promotion.
The local development host is not the target deployment, so this command cannot establish deployment latency by itself.

The selected serving bundle uses the full-lineage bank and the fixed `efSearch=1024/top160` recipe:

```bash
lineage_study=artifacts/retrieval_assisted_reviewer_full
sparse_study=artifacts/retrieval_assisted_reviewer_full_sparse_v2

uv run --locked --extra cascade --with faiss-cpu==1.15.0 \
  python -m experiments.retrieval_assisted_reviewer.run \
  --output "$lineage_study" \
  benchmark-hnsw-extension --config pplx-4b

uv run --locked --extra cascade --with faiss-cpu==1.15.0 \
  python -m experiments.retrieval_assisted_reviewer.run \
  --output artifacts/models/mmbert-lora-full-ctx1024-u17000-s42/serving/retrieval/lineage-hybrid-v3 \
  build-lineage-serving-bundle \
  --source-output "$lineage_study" --sparse-source "$sparse_study"

uv run --locked --extra cascade \
  python -m experiments.retrieval_assisted_reviewer.run \
  --output "$lineage_study" write-lineage-hybrid-parity \
  --sparse-source "$sparse_study" \
  --serving-manifest artifacts/models/mmbert-lora-full-ctx1024-u17000-s42/serving/retrieval/lineage-hybrid-v3/manifest.json \
  --evidence-output reports/retrieval-lineage-hybrid-parity-relaxed-20260820.json
```

The vector-reuse command performs an exact identity join and makes no provider call.
Run the existing prepare, sparse-replay, and reviewer steps before these commands; the parity writer then recomputes the fixed HNSW-plus-BM25 packets without a provider call.
The builder copies the bank and sparse sidecar, excludes the dense source vectors, hashes all 10 payloads, and verifies the result through the maintained retrieval runtime.
The retained quality, cost, latency, memory, and packet-parity results are in `reports/retrieval-assisted-reviewer-findings-20260819.md`.

Materialize the direct cascade continuation into a separate write-once study after the extension artifact passes its retrieval gates:

```bash
hnsw_output=artifacts/retrieval_assisted_reviewer_hnsw_cascade
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output "$hnsw_output" materialize-hnsw-cascade \
  --source-output artifacts/retrieval_assisted_reviewer_full_rows
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output "$hnsw_output" review --split validation --concurrency 4 \
  --arms baseline,dense_pplx-4b_numpy_hnsw_extension,dense_pplx-4b_faiss_hnsw_ef512_top160_hnsw_extension,dense_pplx-4b_faiss_hnsw_ef1024_top160_hnsw_extension
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output "$hnsw_output" analyze-hnsw-cascade
```

The materializer verifies the source manifest, extension artifact, query and unit bindings, every candidate ranking, and every recomputed selected packet before exporting fresh NumPy plus the non-dominated `efSearch=512/top160` and `efSearch=1024/top160` arms.
It imports only latest successful baseline records whose recomputed job identities match, reserves the exact packet-bound request ceiling, and deduplicates byte-identical retrieval-arm requests within this study only.
Reused responses retain their source timing with explicit provenance and zero incremental provider cost.
The dedicated analysis compares each HNSW arm directly with fresh same-matrix NumPy and never treats the local four-worker search p95 as end-to-end latency.
Passing the cascade gates does not select a production arm; target-shaped latency and resource validation remains required.

The full-row HNSW plus partitioned-BM25 diagnostic uses the saved `efSearch=1024/top160` exact-rescored top 20 as its dense branch.
It builds a new bank-hash-bound contentless FTS5 sidecar over the same all-row bank, retrieves 320 raw Unicode matches per label, keeps the first 50 unique source-lineages, and applies the locked 2:1 dense-to-sparse RRF with `k=60`.
The sparse FTS query has an approximate 250 ms SQLite execution budget and returns the exact HNSW packet on interruption, empty output, invalid output, or fusion failure.
SQLite checks the budget through a periodic progress callback, and row metadata lookup occurs outside it, so this is not a hard 250 ms request deadline.
It does not reuse the lineage-bank sparse sidecar or claim that the unretained raw HNSW top 160 entered fusion.

```bash
hnsw_hybrid_output=artifacts/retrieval_assisted_reviewer_full_rows_hnsw_sparse
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output "$hnsw_hybrid_output" materialize-hnsw-hybrid \
  --source-output artifacts/retrieval_assisted_reviewer_full_rows \
  --review-source artifacts/retrieval_assisted_reviewer_hnsw_cascade
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output "$hnsw_hybrid_output" review --split validation --concurrency 4 \
  --arms dense_pplx-4b_faiss_hnsw_ef1024_top160_hnsw_extension,hybrid_pplx4b_hnsw_ef1024_top160_dense20_unicode_partitioned8_fullrows_raw320_lineage50_rrf2
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output "$hnsw_hybrid_output" analyze-hnsw-hybrid
```

The executed sidecar is about 58.5 MiB and built in 9.92 seconds.
The hybrid changed 59 of 110 packets, supplied 47 selected slots absent from the saved dense top 20, and rescued three of the six dense packet failures.
Its conservative local component p95 was 270.0 ms, but BM25 timed out in 123 of 330 searches under the four-worker repeated load test.
All 19 interruptions during the sequential quality materialization were direct-user attacks: 14 of 21 WildJailbreak units and 5 of 61 HackAPrompt units, with none among the 20 untrusted-content units.
The reviewer result therefore describes that one sequentially materialized packet ledger, not a stable four-worker serving policy.
Exact request reuse reduced the new reviewer work to 59 calls costing `$0.025870504`.
Hybrid recall was 93.636% versus 93.182% for HNSW, and FPR was 0.124% versus 0.249%, but both paired confidence intervals included no change.
The added branch therefore failed its material-gain gate and remains the locked challenger for a fresh prospective test, not the production default.
The next quality-selection experiment is that frozen hybrid against dense ef1024 on fresh source-and-time-heldout evidence; dense remains the paired control and fail-soft fallback, not another standalone challenger.

A provider-free execution-budget replay showed that the historical 250 ms SQLite cutoff was too aggressive rather than evidence that BM25 could not complete.
At four workers, timeout counts over one pass were 37/110 at 250 ms, 7/110 at 500 ms, and 0/110 at 1,000 ms.
Every query that completed at all three budgets returned the same candidate hash.
A three-repeat confirmation at 1,000 ms had 0/330 timeouts, 511.0 ms p95, and 663.6 ms p99 at four workers.
The next prospective hybrid should therefore freeze a separately versioned 1,000 ms fail-soft FTS budget, while retaining the 250 ms setting as provenance for the completed post-hoc result.
This fixes branch availability on the local warm-cache replay but does not establish target latency or upgrade the consumed-panel quality evidence.

A retrieval-only lexical-salience diagnostic then considered the first 32 normalized unique terms and selected at most eight with the lowest combined document frequency in the request channel.
At the 1,000 ms budget it had zero timeouts over 110 queries, a 367.5 ms four-worker p95, and a 4.0 ms term-selection p95.
It changed 70 packets from dense, rescued five of six dense packet failures, and supplied 73 selected slots absent from dense top 20, compared with 59 changes, three rescues, and 47 slots for first-eight BM25.
It also produced one terminal packet failure and 49 packets different from the completed first-eight hybrid.
No reviewer call was made because this consumed panel can establish mechanics and candidate novelty but not example usefulness or a quality gain.
A matched retrieval-only replay then gave first-eight BM25 the same 1,000 ms budget.
First-eight BM25 had zero timeouts, changed 77 dense packets, rescued five dense failures, supplied 70 dense-missing selected slots, returned nine sparse fallbacks, and left one hybrid failure.
IDF was about 28% faster at four-worker p95 and supplied three more dense-missing slots, but it increased sparse fallbacks from nine to 14 while rescuing no additional dense failure.
It therefore failed the no-worse-packet-availability gate and does not replace first-eight BM25 for the fresh hybrid experiment.
Learned sparse and late-interaction methods remain separate challengers pending the model research and fresh evidence.
A dedicated three-repeat four-worker determinism run for first-eight BM25 at 1,000 ms produced zero timeouts and identical candidate hashes for all 110 units, with 571.6 ms p95 and 737.1 ms p99.

The optional late-interaction experiment reranks that locked HNSW plus first-eight BM25 candidate union with exact local mLateOn MaxSim.
It never searches the full corpus, changes maintained inference, or sends corpus text to a provider.
Sentence Transformers 6 supplies the checkpoint's `[Q]` and `[D]` role tokens, so callers pass raw text without adding prefixes.
Any model error, overlength input, invalid score, selection failure, or post-execution latency-gate miss preserves the RRF status, failure, selected IDs, and candidate IDs.
The latency check is not a hard wall-clock deadline and this consumed validation replay cannot select a default.

Use Hugging Face's revision-addressed download for the native files, then run the hash-bound local canary and replay:

```bash
mlateon_snapshot=$(hf download lightonai/mLateOn \
  --revision edd378f99593c0ac8a15518b97ad89786b02685e \
  --exclude '*.onnx' --quiet)
colbert_output=artifacts/retrieval_assisted_reviewer_colbert
uv run --locked --extra cascade --extra encoder \
  --with sentence-transformers==6.0.0 \
  python -m experiments.retrieval_assisted_reviewer.colbert \
  --output "$colbert_output" stage0 \
  --snapshot "$mlateon_snapshot" --device cuda
uv run --locked --extra cascade --extra encoder \
  --with sentence-transformers==6.0.0 \
  python -m experiments.retrieval_assisted_reviewer.colbert \
  --output "$colbert_output" rerank \
  --source-output artifacts/retrieval_assisted_reviewer_full_rows_hnsw_sparse \
  --snapshot "$mlateon_snapshot" --device cuda
```

The replay reissues BM25 with the separately versioned 1,000 ms SQLite execution budget, retains the saved ef1024 dense top 20, and exposes at most 70 unique candidates per label to mLateOn.
It pre-encodes the frozen union into an in-memory document-token cache before measuring query encoding and exact MaxSim.
The locked project environment supplies Torch 2.13.0 and Transformers 5.14.1, and the canary refuses any runtime drift.
The runner verifies that the local directory name matches the requested revision and hashes every listed native runtime file, but it cannot attest where a copied directory originated and does not hash unrelated README or ONNX files.
It records IDs, hashes, timings, fallbacks, and aggregate packet changes but no raw text or token embeddings.
The first run deliberately uses no persistent document-token cache; add one only if repeated reranks make re-encoding a measured bottleneck.

The prospective source-heldout bank comparison reuses both frozen PPLX indexes and embeds every external query once:

```bash
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output artifacts/retrieval_assisted_reviewer_wmt_external prepare-wmt
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output artifacts/retrieval_assisted_reviewer_wmt_external score --split external
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output artifacts/retrieval_assisted_reviewer_wmt_external retrieve-bank-comparison
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output artifacts/retrieval_assisted_reviewer_wmt_external review --split external \
  --arms baseline,dense_pplx-4b_lineage,dense_pplx-4b_all_rows
uv run --locked --extra cascade python -m experiments.retrieval_assisted_reviewer.run \
  --output artifacts/retrieval_assisted_reviewer_wmt_external analyze --split external
```

The write-once panel uses the pinned Apache-2.0 WMT 2024 English-to-German prompt-injection suite.
It retains 810 fit-disjoint clean-attack pairs, exactly 162 per attack type, after removing three overlapping pairs and four pairs for exact subtype balance.
No raw WMT text is retained in experiment evidence.

The no-example cascade reached 17.28% recall and 0.123% FPR.
The all-row bank reached 39.14% recall and 0% FPR, while the lineage bank reached 43.33% recall and 0% FPR.
Lineage minus all-row recall was +4.20 percentage points with a paired-group 95% interval from +1.36 to +7.04 points.
The lineage bank also had zero retrieval failures versus three and reduced exact-search p95 from 18.5 ms to 11.4 ms.
All locked bank-comparison gates passed, so the lineage bank is the sole candidate for a future shadow.

This result does not authorize promotion.
Both banks had zero recall on the one-shot task-switch slice because all 162 attacks passed below the local reviewer gate, and no representative live shadow traffic exists in the repository.

The completed full-row confirmation improved recall from 89.07% to 94.81% while increasing FPR from 1.168% to 1.274%.
The quality gates passed, but latency measurements were internally inconsistent, so nothing is promoted into maintained inference.
The full results, limitations, and completed follow-ups are in [`reports/retrieval-model-selection-research-20260817.md`](../../reports/retrieval-model-selection-research-20260817.md).

Verification:

```bash
uv run ruff format --check experiments/retrieval_assisted_reviewer
uv run ruff check experiments/retrieval_assisted_reviewer
uv run --locked --extra cascade python -m unittest discover -s experiments/retrieval_assisted_reviewer -p 'test_*.py'
```
