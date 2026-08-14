# morgott

morgott is a research POC for prompt-injection and agent-security work.
The maintained deliverable is a reproducible, provenance-preserving data corpus.
The maintained model deliverable is the exact update-17,000, 1,024-token mmBERT advisory preview documented in `reports/model-experiments.md`.
Its safe and ONNX artifacts use Git LFS and are served through the cascade and Azure API.
No model is approved for blocking.

The security design deliberately separates prediction from authority:

```text
direct user or untrusted content -> advisory classifier -> route/review signal
proposed tool action             -> reference monitor   -> approve or deny
```

A detector may miss an attack or flag a benign prompt. It therefore never grants
tool, data, network, memory, credential, or financial authority. Every side
effect must still pass deterministic policy using trusted runtime context.
User-visible model output is an egress action in the policy simulation, so a trusted sensitive-data label can deny protected output without treating every confidentiality request as prompt injection.
The policy simulator can bind variable action arguments to exact runtime source identities.
The small `morgott.runtime` module now unions trusted source, provenance, and sensitivity labels across explicit transformations and invokes a synchronous effect only after policy authorization.
It still relies on trusted runtime instrumentation to identify every input and is not a live agent, connector, or automatic taint-discovery system.

## Development

Morgott supports Python 3.12 and newer.
The repository defaults to Python 3.12, while the pinned NOOA 0.0.8 cascade dependency currently supports Python 3.12 and 3.13.

Install the locked environment and run the canonical local checks:

```bash
uv sync --locked
make check
```

Offline model research, training, and export require the optional encoder environment:

```bash
git lfs install
git lfs pull
uv sync --locked --extra encoder
```

Run `make hooks` once to opt into automatic Ruff fixes and formatting for staged Python files.
See `CONTRIBUTING.md` for the lightweight branch and pull request workflow.

## Build the corpus

```bash
uv sync --locked
make data
make check
```

After changing only partition logic, use `uv run morgott data --routing-only`.
It verifies every canonical source hash and schema from `data/manifest.json`,
then replaces routing derivatives and publishes the manifest last.

Authenticated Hugging Face access is required for HackAPrompt, WildJailbreak,
and WildGuardMix. Their access gates have been accepted for this research
workspace. A missing source, schema mismatch, or digest mismatch fails the build
instead of publishing a partial manifest.

There is one local data root and one versioned machine source of truth:

```text
data/
  manifest.json          revisions, hashes, counts, roles, and paths
  sources/               canonical standardized source shards
  views/injection/       legacy POC detector views
  views/routing/         train, validation, dev_test, and uncertain
  audits/                derived overlap evidence
  quarantine/            conflicts and leakage excluded from model views
```

`data/sources/` preserves source text and lineage. Everything under `views/`,
`audits/`, and `quarantine/` is a deterministic derivative. Those large files
are ignored by Git; `data/manifest.json` pins their hashes and is the only
versioned machine manifest. Exact current counts belong only in that manifest.

The corpus builder has no row cap. Canonical shards retain every valid,
non-empty detector-text projection and the available lineage needed by morgott.
They are standardized projections, not byte-for-byte source mirrors: unrelated
upstream fields and unusable detector inputs may be omitted, with exclusions
recorded in the manifest. Every exact-unique routing-eligible projection enters
a grouped development partition unless conflict or leakage rules quarantine it.
Sampling and source weighting belong to a future model recipe.

SWE-bench Verified problem statements are retained only as a repository-grouped, dev-test long-benign direct-user slice.
They are not training data, a threshold-selection set, or a matched long-context attack benchmark.
Their first frozen local evaluation rejected the registered high gate and stopped before remote review, so the maintained cascade remains unchanged.

The often-mentioned roughly 7,000 rows were the negative side of an old grouped
validation split. They were never a download cap, corpus cap, or context limit.

## Data sync (Azure)

The cloud source of truth is Azure Blob Storage: account `vulsightdata`,
container `morgott`, mirroring repo paths (`data/sources`, `data/views`,
`data/quarantine`, `data/audits`, `data-archive`, `artifacts/models`, plus
`data/manifest.json` and the data card `data/README.md` at the container root).
Feature caches (`artifacts/combined_generic/`) are rebuildable and not uploaded.

Rebuild with `uv run morgott data` when a change can affect published corpus output; output-neutral structural refactors may use focused equivalence tests instead.
Run `scripts/azsync.sh push` only when rebuilt published data changes.
Commands, auth, and new-machine bootstrap (e.g. RunPod: azcopy + the
`MORGOTT_SAS_URL` line from `.env`, no Azure CLI needed) are in `data/README.md`.

## Labels and roles

The first-stage target is conservative binary routing without throwing away why
a row was routed:

- `routing_label=0`: source-supported benign content.
- `routing_label=1`: injection, jailbreak, harmful intent, toxic content, or an
  unresolved item that must not silently become benign.
- `injection_label`: `0`, `1`, or null when the source does not establish it.
- `security_tags`: independent tags such as direct jailbreak, indirect
  injection, harmful intent, and toxicity.
- `routing_training_eligible`: derived from `source_role`; it does not control
  the separate legacy injection views.

Generated and automated benign labels use the normal `candidate` role when the
source context supports likely ordinary content. They can therefore enter all
three grouped development partitions. Their `label_basis` and complete origins
remain attached so training recipes and metrics can isolate weak supervision
from human or source-supported evidence. An official dev-test origin still wins
when the same exact text appears in both roles.

“Not an injection” is not automatically “benign.” Injection-only negatives
remain available for the legacy detector but are auxiliary for broad routing
unless their source also supports the broader benign claim.

Finance, cybersecurity, and other sensitive-topic vocabulary is not itself a
deny label. Benign discussion stays benign; harmful content without instruction
subversion is kept distinct; exact actions are approved or denied by policy.

| Data role | Meaning |
|---|---|
| `train` | Fits model parameters |
| `validation` | Selects models and operating thresholds |
| `dev_test` | Frozen repeated development comparison, not a pristine final test |
| `auxiliary` | Valid source data excluded from default supervision |
| `uncertain` | Labels are insufficient; never coerced to benign |
| `quarantine` | Conflicts or leakage; never used for fitting or evaluation |

Splits are deterministic and grouped by available lineage such as conversation,
team, attacker, challenge level, base prompt, or document. The target after exact
deduplication is approximately 70% train, 10% validation, and 20% dev-test,
stratified by source and routing label where lineage permits. Official holdouts
stay in dev-test. Exact text is merged once; duplicates across unrelated
lineages become singleton split atoms instead of joining whole lineage networks.
Strict near-overlap is audited and conflicting or leaking rows are quarantined.
Because weak-labelled candidates can enter validation and dev-test, those views
are development comparisons rather than independent evidence. Report weak and
non-weak slices separately; neither weak-label agreement nor an aggregate score
supports a production false-positive claim.

## Model status

`morgott benchmark` still provides a cheap character n-gram/logistic-regression
control over the original injection views. It is useful as a code and evaluation
smoke test, but it is not the intended routing model and its generated report is
ignored. `morgott scan` is shadow-only and always returns `decision: allow`.

`morgott routing-baseline` trains one reproducible unweighted word 1-2 gram linear control on source-supported direct-user rows.
It verifies inputs against the canonical manifest, uses the untouched 0.5 cutoff, and reports confusion counts plus aggregate, origin-membership source, fixed-prevalence, and normalized-character length development metrics.
Historical neural and data-ablation runners are not part of the active tree.
Their metrics and stop decisions remain in the versioned reports and Git history.
Historical frozen-mmBERT and 512-token LoRA artifacts remain under `artifacts/models/` as research provenance only.
Only the exact update-17,000 candidate is registered for maintained inference.
It scans complete normalized artifacts in ordered 1,024-token windows with 128-token overlap.
Its external transfer evidence remains too weak for blocking, and every assessment still returns `decision: allow`.

The later owner-authorized LP-FT comparison added repository-grouped SWE-rebench V2 matched pairs and substantially reduced long-task clean flags, but it collapsed on PromptShield transfer and indirect-document recall.
That candidate was rejected and its weights are not registered.
Its weights, scores, result records, and archived source are retained only for later comparison; the progress checkpoint was deleted after its digest was recorded.
None of these artifacts by themselves justify another training run; exact
findings and limitations are in `reports/model-experiments.md`.

The maintained mmBERT package can prepare the pinned external data, preflight the complete canonical mixture, train a frozen head or rank-8 LoRA, and evaluate a run:

```bash
uv run python -m morgott.models.mmbert.external_data
uv run python -m morgott.models.mmbert.train \
  --preflight-only \
  --max-tokens 1024 \
  --no-gradient-checkpointing
uv run --extra encoder python -m morgott.models.mmbert.train \
  --mode lora \
  --microbatch-size 8 \
  --max-tokens 1024 \
  --no-gradient-checkpointing
uv run --extra encoder python -m morgott.models.mmbert.evaluate \
  artifacts/mmbert/runs/mmbert-lora-full-s42-ctx1024
```

The trainer streams every canonical training row with source-supported injection labels after the external leakage guard, then adds filtered PromptShield training rows and the retained matched pairs.
Future preflights use a separate, stricter audit fingerprint for overlap filtering without changing canonical hashes or registered model input.
Its frozen and LoRA modes share one data contract, loss, checkpoint rule, and artifact format.
Generic additional-pair support remains because the registered 1,024-token LoRA candidate used the pinned matched-pair archive.
The rejected LP-FT implementation is provenance-only in the archived campaign source and is not a maintained training mode.
The registered snapshot retains its archived 1,024-token execution identity; a new training recipe produces a different run and does not replace it.
The single generic output treats direct injection, indirect injection, and jailbreak as positive instruction subversion and does not expose separate subtype scores.
Source-supported harmful content without subversion may remain as a negative counterexample; this is not a harmfulness score.
It keeps the retained SDPA attention contract; an FA2 run requires its own pinned kernel and mmBERT parity record.
Having a maintained runner does not waive the evidence gates in
`docs/roadmap.md` or promote its output.

The sole registered model key and its exact hashes are in `model-artifacts.json`.
Use the ONNX cascade below or the Azure API for maintained assessment.

## Shadow cascade POC

The maintained cascade defaults to `mmbert-lora-full-ctx1024-u17000-s42` and serves its registered FP32 ONNX graph through OpenVINO on CPU.
The Azure preview requests `auto`: OpenVINO uses BF16 when the assigned CPU exposes it and otherwise uses FP32.
`/v1/status` reports both the requested and selected precision.
There is one portable model artifact rather than a precision-specific copy.
It passes direct-user scores below `0.2` and untrusted-content scores below `0.1`, and it restricts local scores at or above `0.99999`.
OpenRouter credentials are mandatory at maintained cascade startup, but provider calls occur only for inputs that reach the existing review route.
Multi-window untrusted content without a local high first sends the complete normalized artifact to DeepSeek V4 Flash.
A full-context flag restricts immediately, while a clear result falls back to the existing middle-zone window reviews in batches of up to 4.
Direct-user and single-window behavior remain unchanged, and the 128-window cap applies to the complete multi-window artifact.
The synthetic full-context review record uses window index `-1`; ordinary window records retain their nonnegative tokenizer-window indexes.
An exhausted review fails safe immediately, while a confirmed reviewer flag completes the advisory restrict without starting later calls.
The selected reviewer is `deepseek/deepseek-v4-flash-0731` through Cloudflare.
DeepSeek receives the trusted input channel and restricts at `p_subversion >= 0.6224593312018547`; invalid or exhausted reviews fail conservatively.
Production initialization suppresses LiteLLM's unsolicited error banners so `morgott cascade` keeps stdout as one JSON document even when a retry is needed.
Every result remains advisory: `decision` is always `allow`, and `advisory_route` never grants authority.

Install the cascade on Python 3.12 or 3.13:

```bash
uv sync --locked --extra cascade
```

NOOA 0.0.8 currently declares support for Python 3.12 and 3.13.
The rest of Morgott supports Python 3.12 and newer, and the cascade reports a clear startup error when that pinned NOOA release is unavailable.

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
The production constructor fails closed until the ONNX model and tokenizer hashes are registered under the full-LoRA model in `model-artifacts.json`.
Register a serving runtime only after representative export parity and the frozen 20,000-row serving-equivalence gate pass, with deployment-CPU latency and throughput recorded separately.
The verifier binds every provider record to the current prompt, request, model, provider, panel row, and trusted channel; stale prompt evidence fails closed.
These are already-open shadow engineering results, not new production-quality claims.
The exact 1,024-token update-17,000 native evaluation is bound to the safe serving package, while later-window document aggregation remains new shadow behavior.

DeepSeek V4 Flash 0731 replaces the April reviewer under the owner's aggregate-quality criterion, but the gain is not uniform and all results remain advisory.
The exact replacement evidence is in `reports/deepseek-v4-flash-0731-research.md`; the broader model, workload, robustness, and rejected-candidate findings are in `reports/model-experiments.md`; stateful containment findings are in `reports/agentdojo-integration-research.md` and `reports/agent-security-benchmark-options.md`.
Keep exact experiment metrics in those reports so this operational README does not become a second model ledger.

Run an assessment after setting `OPENROUTER_API_KEY`:

```bash
uv run --extra cascade morgott cascade input.txt \
  --input-channel direct_user
```

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

The maintained remote path uses NOOA `CompletionClient` only.
The Predict-only agent example is in `examples/nooa_preflight.py`; the rejected measured alternative is retained only as metrics and hashes in the evaluation report.
Neither path uses CodeAct, generated Python, memory, plugins, or tracing.

For this non-production preview, p95 below two seconds and at least 0.5 QPS are recorded targets rather than blocking gates.
The deployment uses one model worker.

## Azure preview deployment

The Waleed subscription deployment creates one Basic ACR, one 2-vCPU/4-GiB Container App, one Standard Service Bus namespace and queue, one Standard Key Vault, one managed identity, and a 30-day Log Analytics workspace in `morgott-preview-rg`.
The scheduled job enqueues one versioned command at 02:00 UTC, and the API consumer verifies the Blob manifest before running fixed synthetic canaries.
API text, canary text, corpus rows, credentials, and provider responses are never logged.
Routing scratch SQLite, Trackio SQLite, and experiment ledgers remain local.

```bash
scripts/deploy-azure.sh
scripts/check-azure-milestone.sh
```

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

The deployment copies only `OPENROUTER_API_KEY`, either matching Hugging Face token alias, `MORGOTT_SAS_URL`, optional `OPENAI_API_KEY`, and a generated shared API credential into Key Vault without expiry metadata.
Only the OpenRouter and shared API secrets are exposed to the app.
Additional company-owned provider credentials may be stored under explicit Key Vault names when needed, but this preview has no personal BYOK database, selector, or UI.
Blob, Service Bus, ACR pull, and Key Vault access use the user-assigned managed identity.
The $100 monthly budget sends alerts at 50, 80, and 100 percent; it does not cap spending or stop resources.
The deployment validates a zero-traffic candidate revision while an existing revision stays live, or keeps ingress disabled during a first deployment.
It verifies the selected BF16 or FP32 runtime plus auth, bounds, advisory behavior, Blob, Service Bus, OpenRouter, memory, and 30-request smoke gates before assigning 100 percent traffic.
`scripts/check-azure-milestone.sh` tracks only the five intended workloads and their rolling 61-day posted costs.
That calculation is an operational proxy; the Microsoft for Startups portal remains the eligibility authority and should be checked weekly.
After the Startup portal also shows five workloads, rerun it with `--portal-confirmed` to save the local day-zero marker.
Keep one replica until all five intended workloads have posted at least $1, then scale the preview to zero with `az containerapp update --name morgott-api --resource-group morgott-preview-rg --min-replicas 0`; the HTTP and Service Bus rules wake it for API calls and the daily canary, while a later deployment intentionally restores one replica for validation.

The first proper routing experiment should stay deliberately small:

1. Train a cheap linear text baseline on the canonical routing train split.
2. Compare one end-to-end encoder on the identical selected grouped rows, and state its weighting explicitly; add masked subtype heads only where labels are known.
3. First report the untouched 0.5 cutoff; tune later thresholds on validation only after application costs and prevalence are known; compare on `dev_test`; report per-source and leave-one-source-out results, not only aggregate accuracy.
4. Freeze a genuinely prospective final test before claiming generalization.

Large-source caps may be tested as model ablations, but the canonical corpus
must remain complete. No existing neural, remote-reviewer, or weak-label pilot
is promoted; their durable conclusions are summarized in
`reports/model-experiments.md`.

## Repository map

- `src/morgott/data.py`: core source adapters and legacy injection views.
- `src/morgott/corpus.py`: canonical corpus build and publication interface.
- `src/morgott/sources/`: source adapters grouped by security, finance, task, and authorization-boundary data.
- `src/morgott/routing.py`: disk-backed canonical routing-view materialization.
- `src/morgott/overlap.py`: conservative near-overlap audit.
- `src/morgott/models/mmbert/`: maintained mmBERT preparation, full-data training, evaluation, and advisory inference.
- `src/morgott/models/`: retained linear controls and model-specific packages.
- `src/morgott/normalization.py`: strict inference-side text normalization.
- `src/morgott/policy.py`: deterministic authorization simulation.
- `src/morgott/runtime.py`: trusted label propagation and policy-gated effect invocation.
- `tests/`: maintained data, detector, and policy invariants.
- `artifacts/models/`: the registered advisory model artifacts plus the retained unregistered LP-FT comparison candidate.
- `experiments/`: rules for disposable or study-specific experiments that do not belong in maintained model code.
- `scripts/package-mmbert-snapshot.py`: one-model operator script that materializes the retained 1,024-token snapshot into safe serving artifacts.
- `data/manifest.json`: sole versioned machine data manifest.
- `reports/dataset-selection.md`: source inclusion and exclusion decisions.
- `reports/label-audit.md`: label interpretation and known ambiguity.
- `reports/corpus-sanity-audit.md`: corpus-wide integrity checks and critical limitations.
- `reports/attention-kernel-audit.md`: measured SDPA versus FlashAttention-2 and context-length constraints.
- `reports/model-experiments.md`: authoritative historical model decision ledger.
- `docs/data-contract.md`: canonical data, label, source, and split contracts.
- `docs/threat-model.md`: trust boundary and security claims.
- `docs/roadmap.md`: evidence-gated next steps.
- `AGENTS.md`: operating brief for coding agents.

Only load model artifacts generated locally by this project. Python model
serialization is unsafe for untrusted files.
