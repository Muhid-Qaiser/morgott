# morgott

morgott is a research POC for prompt-injection and agent-security work.
The maintained deliverable is a reproducible, provenance-preserving data corpus.
The current modelling deliverable is a versioned full-data frozen-mmBERT shadow, the historical reduced-mixture LoRA gate, and one completed full-mixture rank-8 LoRA seed documented in `reports/model-experiments.md`.
Their small inference artifacts use Git LFS and are available through the explicit advisory `morgott shadow-score` command.
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

Model research and shadow scoring require the optional encoder environment:

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

The rule: **any data change → `uv run morgott data` → `scripts/azsync.sh push`.**
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
The selected frozen-mmBERT full-data pair-ranking candidate, reduced-mixture LoRA gate, and completed full-mixture LoRA seed are retained only as advisory first-pass research shadows under `artifacts/models/`.
Their external transfer evidence remains too weak for blocking, and every side effect still requires deterministic policy authorization.
The existing `shadow-score` path strictly normalizes and truncates each input to its first 512 tokens.
The separate cascade path scans complete normalized artifacts using ordered 512-token windows with 128-token overlap.

The later owner-authorized LP-FT comparison added repository-grouped SWE-rebench V2 matched pairs and substantially reduced long-task clean flags, but it collapsed on PromptShield transfer and indirect-document recall.
That candidate was rejected, its weights are not registered, and the maintained LoRA and cascade remain unchanged.
Its weights, scores, result records, and reproduction code are retained only for later comparison; the progress checkpoint was deleted after its digest was recorded.
None of these artifacts authorize another training run, and exact findings and limitations are in `reports/model-experiments.md`.

The maintained mmBERT package can prepare the pinned external data, preflight the complete canonical mixture, train a frozen head or rank-8 LoRA, reproduce the completed LP-FT comparison, and evaluate a run:

```bash
uv run python -m morgott.models.mmbert.external_data
uv run python -m morgott.models.mmbert.train \
  --preflight-only \
  --no-gradient-checkpointing
uv run --extra encoder python -m morgott.models.mmbert.train \
  --mode lora \
  --microbatch-size 8 \
  --no-gradient-checkpointing
uv run --extra encoder python -m morgott.models.mmbert.evaluate \
  artifacts/mmbert/runs/mmbert-lora-full-s42
```

The trainer streams every canonical training row with source-supported injection labels after the external leakage guard, then adds filtered PromptShield training rows and the retained matched pairs.
Future preflights use a separate, stricter audit fingerprint for overlap filtering without changing canonical hashes or registered model input.
Its frozen, LoRA, and LP-FT modes share one data contract, loss, checkpoint rule, and artifact format.
LP-FT and additional-pair support are retained only to reproduce the completed rejected comparison.
The registered full-mixture LoRA identity fails closed unless it uses microbatch 8 with gradient checkpointing disabled; a different execution recipe needs a different run identity.
The single generic output treats direct injection, indirect injection, and jailbreak as positive instruction subversion and does not expose separate subtype scores.
Source-supported harmful content without subversion may remain as a negative counterexample; this is not a harmfulness score.
It keeps the retained SDPA attention contract; an FA2 run requires its own pinned kernel and mmBERT parity record.
Having a maintained runner does not authorize a new experiment or promote its output.
The evidence gates in `docs/roadmap.md` still apply.

The registered model keys and exact hashes are in `model-artifacts.json`.
Score downstream JSONL without producing a decision:

```bash
uv run --extra encoder morgott shadow-score \
  mmbert-frozen-s42 input.jsonl scores.jsonl
```

Each input row must contain unique `id`, non-empty `text`, and trusted `input_channel` set to `direct_user` or `untrusted_content`.
The output contains only the raw score, channel, model revision, and artifact hashes.
Use `mmbert-frozen-s42`, `mmbert-lora-s42`, or `mmbert-lora-full-s42`.
Do not average or OR them without evaluating that new ensemble.

## Shadow cascade POC

The maintained cascade serves the registered FP32 ONNX graph through OpenVINO BF16 on CPU.
OpenVINO performs the BF16 lowering at startup, so there is one portable model artifact rather than a second precision-specific copy.
It passes direct-user scores below `0.2` and untrusted-content scores below `0.1`, and it restricts local scores at or above `0.99999`.
With remote review enabled, multi-window untrusted content without a local high first sends the complete normalized artifact to DeepSeek V4 Flash.
A full-context flag restricts immediately, while a clear result falls back to the existing middle-zone window reviews in batches of up to 4.
Direct-user and single-window behavior remain unchanged, and the 128-window cap applies to the complete multi-window artifact.
The synthetic full-context review record uses window index `-1`; ordinary window records retain their nonnegative tokenizer-window indexes.
An exhausted review fails safe immediately, while a confirmed reviewer flag completes the advisory restrict without starting later calls.
The selected reviewer is `deepseek/deepseek-v4-flash-0731` through Cloudflare.
DeepSeek receives the trusted input channel and restricts at `p_subversion >= 0.6224593312`; invalid or exhausted reviews fail conservatively.
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
  python -m morgott.models.mmbert.export_onnx export
uv run --extra cascade \
  python -m morgott.models.mmbert.export_onnx verify-panel \
  --deepseek-evidence \
  artifacts/openrouter_downstream_eval/deepseek_0731_runtime_evidence.jsonl.gz
uv run --extra cascade \
  python -m morgott.models.mmbert.export_onnx benchmark
```

The benchmark prints deployment measurements to stdout and never overwrites registered evidence.
The verification command also treats its evidence as write-once; use a fresh `--output` directory for a new candidate.
The production constructor fails closed until the ONNX model and tokenizer hashes are registered under the full-LoRA model in `model-artifacts.json`.
Register a serving runtime only after representative export parity, the frozen 20,000-row serving-equivalence gate, and deployment-CPU latency and throughput gates pass.
The verifier binds every provider record to the current prompt, request, model, provider, panel row, and trusted channel; stale prompt evidence fails closed.
These are already-open shadow engineering results, not new production-quality claims.
Later-window document behavior is new shadow evidence and is not covered by the retained 512-token evaluation.

DeepSeek V4 Flash 0731 replaces the April reviewer under the owner's aggregate-quality criterion, but the gain is not uniform and all results remain advisory.
The exact replacement evidence is in `reports/deepseek-v4-flash-0731-research.md`; the broader model, workload, robustness, and rejected-candidate findings are in `reports/model-experiments.md`; stateful containment findings are in `reports/agentdojo-integration-research.md` and `reports/agent-security-benchmark-options.md`.
Keep exact experiment metrics in those reports so this operational README does not become a second model ledger.

Run a local-only assessment:

```bash
uv run --extra cascade morgott cascade input.txt \
  --input-channel direct_user
```

Add `--allow-remote` only when eligible text may leave the process and `OPENROUTER_API_KEY` is set.
For multi-window untrusted content, eligible text includes the complete normalized artifact before any middle-zone fallback.
Files and stdin are read in bounded chunks, normalized only after the complete artifact arrives, and scanned without a configured maximum input length.
The current whole-artifact normalization is intentionally O(N) memory.

Applications use the same narrow async interface:

```python
scanner = CascadeScanner.from_artifacts(
    manifest_path=Path("model-artifacts.json"),
    allow_remote=True,
)
try:
    assessment = await scanner.assess_text(text, input_channel="direct_user")
finally:
    await scanner.aclose()
```

The maintained remote path uses NOOA `CompletionClient` only.
The Predict-only agent example is in `examples/nooa_preflight.py`; the rejected measured alternative is retained only as metrics and hashes in the evaluation report.
Neither path uses CodeAct, generated Python, memory, plugins, or tracing.

Before deployment, benchmark the target CPU and require a warm p95 below 500 ms for one 512-token local request plus sustained 5 QPS with zero errors.
Use two identical worker processes if one process misses 5 QPS before considering schedulers or dynamic batching.

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
- `tests/`: maintained data, detector, and policy invariants.
- `artifacts/models/`: the registered advisory model artifacts plus the retained unregistered LP-FT comparison candidate.
- `experiments/`: rules for disposable or study-specific experiments that do not belong in maintained model code.
- `data/manifest.json`: sole versioned machine data manifest.
- `reports/dataset-selection.md`: source inclusion and exclusion decisions.
- `reports/label-audit.md`: label interpretation and known ambiguity.
- `reports/corpus-sanity-audit.md`: corpus-wide integrity checks and critical limitations.
- `reports/attention-kernel-audit.md`: measured SDPA versus FlashAttention-2 and context-length constraints.
- `reports/model-experiments.md`: concise historical model decision ledger.
- `docs/threat-model.md`: trust boundary and security claims.
- `docs/roadmap.md`: evidence-gated next steps.

Only load model artifacts generated locally by this project. Python model
serialization is unsafe for untrusted files.
