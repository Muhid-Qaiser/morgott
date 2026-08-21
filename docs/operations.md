# Operations

Operational runbooks for the maintained advisory cascade and the Azure
preview. The normative security and data contracts stay in `AGENTS.md`,
`threat-model.md`, and `data-contract.md`; exact experiment metrics stay in
`../reports/model-experiments.md`.

## Maintained advisory cascade

The maintained cascade defaults to `mmbert-lora-full-ctx1024-u17000-s42` and serves its registered FP32 ONNX graph through OpenVINO on CPU.
The Azure preview requests `auto`: OpenVINO uses BF16 when the assigned CPU exposes it and otherwise uses FP32.
`/v1/status` reports both the requested and selected precision.
It also reports the registry-bound pipeline profile, policy hash, and threshold hash so deployment checks reject stale revisions.
There is one portable model artifact rather than a precision-specific copy.
The maintained `balanced-retrieval-20260819` profile passes direct-user scores below `0.2` and untrusted-content scores below `0.025`, and it restricts local scores at or above `0.9999`.
Inputs on the review route use the registered source-lineage HNSW plus BM25/RRF example retriever before the fixed DeepSeek reviewer.
The private bank enforces the fixed byte limit at build time and startup.
OpenRouter credentials are mandatory at maintained cascade startup, but provider calls occur only for inputs that reach the review route.
Multi-window untrusted content without a local high first sends the complete normalized artifact to DeepSeek V4 Flash.
A full-context flag restricts immediately, while a clear result falls back to the existing middle-zone window reviews in batches of up to 4.
Local routing and full-context sequencing remain unchanged, and the 128-window cap applies to the complete multi-window artifact.
The synthetic full-context review record uses window index `-1`; ordinary window records retain their nonnegative tokenizer-window indexes.
An exhausted review fails safe immediately, while a confirmed reviewer flag completes the advisory restrict without starting later calls.
The selected reviewer is `deepseek/deepseek-v4-flash-0731` through Cloudflare.
DeepSeek receives the trusted input channel and restricts at `p_subversion >= 0.5`; invalid or exhausted reviews fail conservatively.
The request pins Cloudflare, disables fallback, requires strict structured output and logprobs, and disables reasoning.
The maintained parser validates the response schema and decision-token logprobs, but it does not independently attest the returned provider build.
The profile, thresholds, request identities, and exact consumed-development results are bound by `model-artifacts.json` to the serving promotion record.
Production initialization suppresses LiteLLM's unsolicited error banners so single-input `morgott cascade` keeps stdout as one JSON document, and batch mode keeps it as clean JSONL, even when a retry is needed.
Every result remains advisory: `decision` is always `allow`, and `advisory_route` never grants authority.

Install the cascade on Python 3.12 or 3.13:

```bash
uv sync --locked --extra cascade

az storage blob download-batch \
  --account-name vulsightdata \
  --source morgott \
  --destination . \
  --pattern 'artifacts/models/mmbert-lora-full-ctx1024-u17000-s42/serving/retrieval/lineage-hybrid-v3/*' \
  --auth-mode login \
  --overwrite true
```

NOOA 0.0.8 currently declares support for Python 3.12 and 3.13.
The rest of Morgott supports Python 3.12 and newer, and the cascade reports a clear startup error when that pinned NOOA release is unavailable.
The private retrieval bundle comes from the existing Blob source of truth, stays outside Git and LFS, and is verified against the registry before loading.

Export and verify a candidate CPU artifact offline:
The export command verifies the registered artifacts plus the exact model-core and normalization sources that execute during scoring.
Historical trainer, evaluator, and lockfile hashes remain immutable provenance without making ordinary runtime maintenance disable the registered model.

```bash
uv run --extra encoder --extra encoder-export \
  python -m morgott.models.mmbert.export_onnx export \
  --model-key mmbert-lora-full-ctx1024-u17000-s42
uv run --extra cascade --extra encoder-export \
  python -m morgott.models.mmbert.export_onnx verify-panel \
  --model-key mmbert-lora-full-ctx1024-u17000-s42 \
  --deepseek-evidence \
  artifacts/openrouter_downstream_eval/deepseek_0731_runtime_evidence.jsonl.gz
uv run --extra cascade \
  python -m morgott.models.mmbert.export_onnx benchmark \
  --model-key mmbert-lora-full-ctx1024-u17000-s42
```

The benchmark prints deployment measurements to stdout and never overwrites registered evidence.
The verification command also treats its evidence as write-once; use a fresh `--output` directory for a new candidate.
Export compares the retained SDPA path with eager PyTorch and ONNX Runtime on both short and exactly 1,024-token inputs.
The production constructor fails closed until the ONNX model, tokenizer, exact cascade policy, retrieval evidence, and private retrieval manifest are hash-bound under the full-LoRA model in `model-artifacts.json`.
Register a serving runtime only after representative export parity and the frozen 20,000-row serving-equivalence gate pass, with deployment-CPU latency and throughput recorded separately.
The verifier binds every provider record to the current prompt, request, model, provider, panel row, and trusted channel; stale prompt evidence fails closed.
The original serving-verification file remains immutable model/runtime provenance and predates the promoted gates.
The separate promotion record binds the exact all-window balanced-cascade evidence without rewriting that history.
These remain consumed advisory engineering results, not production-quality claims.

DeepSeek V4 Flash 0731 replaces the April reviewer under the owner's aggregate-quality criterion, but the gain is not uniform and all results remain advisory.
The exact replacement evidence is in `reports/deepseek-v4-flash-0731-research.md`; the broader model, workload, robustness, and rejected-candidate findings are in `reports/model-experiments.md`; stateful containment findings are in `reports/agentdojo-integration-research.md` and `reports/agent-security-benchmark-options.md`.
Keep exact experiment metrics in those reports so this operational README does not become a second model ledger.

Run an assessment after setting `OPENROUTER_API_KEY`:

```bash
uv run --extra cascade morgott cascade input.txt \
  --input-channel direct_user
```

Batch mode builds the scanner once and amortizes its startup cost over many inputs:

```bash
uv run --extra cascade morgott cascade --jsonl batch.jsonl
```

Each JSONL record is `{"text": ..., "input_channel": ...}`, `-` reads records from stdin, and stdout gets one result JSON per line.
The first malformed record aborts the batch with its line number and a nonzero exit.

For multi-window untrusted content, eligible text includes the complete normalized artifact before any middle-zone fallback.
Files and stdin are read in bounded chunks, normalized only after the complete artifact arrives, and scanned without a configured maximum input length.
The current whole-artifact normalization is intentionally O(N) memory.

Applications use the same narrow async interface:

```python
scanner = CascadeScanner.from_artifacts(
    manifest_path=Path("model-artifacts.json"),
)
try:
    assessment = await scanner.assess_text(text, input_channel="direct_user")
finally:
    await scanner.aclose()
```

The maintained remote path uses NOOA `CompletionClient` for DeepSeek review and a small direct OpenRouter embeddings client for the provider-pinned PPLX query vector.
The Predict-only agent example is in `examples/nooa_preflight.py`; the rejected measured alternative is retained only as metrics and hashes in the evaluation report.
Neither path uses CodeAct, generated Python, memory, plugins, or tracing.

For this non-production preview, p95 below two seconds and at least 0.5 QPS are recorded targets rather than blocking gates.
The deployment uses one model worker.

## Azure preview deployment

The Waleed subscription deployment creates one Basic ACR, one Container App, one Standard Key Vault, one managed identity, and a 30-day Log Analytics workspace in `morgott-preview-rg`.
The candidate defaults to 2 vCPU and 4 GiB, with an explicit 4-vCPU/8-GiB retry available if its measured memory gate fails.
The replacement scheduled canary is deferred until the reviewed promotion path is implemented.
Until that migration, zero-traffic validation leaves the existing stable scheduled canary, alert, and legacy Service Bus resources unchanged.
API text, canary text, corpus rows, credentials, and provider responses are never logged.
Routing scratch SQLite, Trackio SQLite, and experiment ledgers remain local.

```bash
scripts/deploy-azure.sh                              # validate at zero traffic
scripts/deploy-azure.sh --candidate-size 4cpu-8gi   # explicit larger candidate
scripts/check-azure-milestone.sh
```

Traffic promotion is blocked until the replacement predeclared multi-probe paired latency protocol is implemented.
`scripts/deploy-azure.sh --promote` fails before any Azure call so the consumed 15-pair protocol cannot be retried until it happens to pass.
On 2026-08-20, the owner manually promoted the already-validated advisory candidate for POC use while retaining the previous revision at 0% as the rollback point.
That operational override is not a latency-gate pass or a production-quality claim, and the automated deployment path remains zero-traffic-only.

### Promoting and deploying a cascade profile

The four routing values and profile name in `src/morgott/models/downstream.py` are the runtime source of truth.
The registry-bound promotion JSON is immutable evidence that those constants, the reviewer request, and the benchmark selection agree.

For a future promotion:

1. Freeze the benchmark selection before changing serving code.
2. Update the constants and profile name in `src/morgott/models/downstream.py`.
3. Add a new promotion JSON instead of overwriting prior evidence, then update only the `cascade_policy` path and SHA-256 in `model-artifacts.json`.
4. Run `make check` and `git diff --check`.
5. Run `scripts/deploy-azure.sh` and inspect its zero-traffic validation result.
6. Implement and freeze the replacement multi-probe paired latency protocol before adding a separate reviewed traffic-promotion path.

The deployment script requires a clean Git worktree and builds a content-addressed immutable ACR image.
It reads the profile, threshold hash, and policy hash from the verified Python and registry state, so it needs no threshold edits.
It stages the exact private retrieval payloads listed by the registered manifest from a hash-matching local copy or the existing Blob container, verifies every file size and SHA-256, and bakes those bytes into the immutable ACR image without adding them to Git.
It creates a zero-traffic revision and checks its exact model, policy, threshold, and retrieval identities through the protected API.
One frozen public synthetic request must exercise retrieval and DeepSeek with the expected score range, packet, prompt, embedding request, reviewer provider, and reviewer request identities.
Local preflight binds that probe by SHA-256, score range, and expected review route.
The same candidate smoke then checks auth, bounds, advisory behavior, 30 local-pass requests, and at least 512 MiB of memory headroom.
The script cannot move traffic, and the previous revision remains active while the candidate is retained at zero traffic.
Any validation failure keeps traffic on the previous revision.
The consumed paired comparison remains immutable historical evidence in [`reports/azure-preview-retrieval-canary-20260819T174113Z.json`](../reports/azure-preview-retrieval-canary-20260819T174113Z.json); routine validation does not create another report.

### Using the Azure API

Sign in as `waleed@vulsight.com`, then retrieve the live endpoint and shared API key without printing the credential:

```bash
az account set --subscription 25d0cf2e-a75c-46f5-b26c-f57a48f96967
api_url="https://$(az containerapp show \
  --name morgott-api \
  --resource-group morgott-preview-rg \
  --query properties.configuration.ingress.fqdn \
  --output tsv)"
api_key=$(az keyvault secret show \
  --vault-name morgott-vulsight-kv \
  --name morgott-api-key \
  --query value \
  --output tsv)

curl --fail --silent --show-error "$api_url/v1/assess" \
  -H "Authorization: Bearer $api_key" \
  -H "Content-Type: application/json" \
  -d '{"text":"Please summarize this document","input_channel":"direct_user"}' \
  | jq

unset api_key
```

`GET /healthz` is unauthenticated.
`GET /v1/status` and `POST /v1/assess` require the bearer key.
Assessment input accepts `direct_user` or `untrusted_content`, and every response remains advisory with `decision: allow`.

The deployment copies `OPENROUTER_API_KEY` and a generated shared API credential into Key Vault without expiry metadata.
Additional company-owned provider credentials may be stored under explicit Key Vault names when needed, but this preview has no personal BYOK database, selector, or UI.
ACR pull and Key Vault secret references use the user-assigned managed identity.
The deployment operator downloads the registry-selected retrieval bundle from Blob with Azure CLI login and verifies every registered size and SHA-256 before building the image.
The $100 monthly budget sends alerts at 50, 80, and 100 percent; it does not cap spending or stop resources.
The deployment validates a zero-traffic candidate revision while an existing revision stays live, or keeps ingress disabled during a first deployment.
It verifies the selected BF16 or FP32 runtime plus auth, bounds, advisory behavior, the Blob-sourced bundle, one exact routed OpenRouter probe, memory headroom, and 30 local-pass requests while retaining the candidate at zero traffic.
`scripts/check-azure-milestone.sh` tracks only the four intended workloads and their rolling 61-day posted costs.
The legacy Service Bus workload is not counted and remains deployed only for the current stable canary until a reviewed migration and rollback window complete.
That calculation is an operational proxy; the Microsoft for Startups portal remains the eligibility authority and should be checked weekly.
After the Startup portal also shows four workloads, rerun it with `--portal-confirmed` to save the local day-zero marker.
Keep the preview at its declared one-replica minimum so the daily canary does not depend on cold-start behavior.
