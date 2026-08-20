# morgott

morgott is a research POC for prompt-injection and agent-security work.
The maintained deliverable is a reproducible, provenance-preserving data corpus.
The maintained model deliverable is the exact update-17,000, 1,024-token mmBERT advisory preview documented in `reports/model-experiments.md`.
The registry-bound `balanced-retrieval-20260819` cascade profile is its maintained advisory default.
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
`morgott demo` runs the deterministic action-policy ablation and writes `reports/policy_ablation.md`.
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

## Data sync (Azure)

The cloud source of truth is Azure Blob Storage: account `vulsightdata`,
container `morgott`, mirroring repo paths (`data/sources`, `data/views`,
`data/quarantine`, `data/audits`, `data-archive`, `artifacts/models`, plus
`data/manifest.json` and the data card `data/README.md` at the container root).
Feature caches (`artifacts/combined_generic/`) are rebuildable and not uploaded, apart from the two small lfm25 evidence files tracked in git.

The impact-based rebuild-and-push rule is normative in `AGENTS.md`.
Operational commands, auth, and new-machine bootstrap (e.g. RunPod: azcopy + the
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

## Maintained advisory cascade

The maintained cascade defaults to `mmbert-lora-full-ctx1024-u17000-s42`, serves its registered FP32 ONNX graph through OpenVINO on CPU, and applies the registry-bound `balanced-retrieval-20260819` advisory profile.
Every result remains advisory: `decision` is always `allow`, and `advisory_route` never grants authority.
Install, export, verification, and assessment runbooks are in [docs/operations.md](docs/operations.md); exact experiment metrics stay in `reports/model-experiments.md`.

## Azure preview deployment

The advisory preview runs as one Container App in `morgott-preview-rg` on the Waleed subscription, validated at zero traffic by `scripts/deploy-azure.sh`.
Deployment, promotion rules, milestone tracking, and API usage are in [docs/operations.md](docs/operations.md).

The first proper routing-model phase is complete, and its historical steps remain in `docs/roadmap.md` and `reports/model-experiments.md`.
The next evidence phase stays deliberately narrow:

1. Collect prospective task-bearing long benign untrusted HTML, email, and document traffic with matched attacks and position slices.
2. Keep the promoted profile frozen while that shadow panel is collected and adjudicated.
3. Test at most one task-conditioned reviewer contract on a development role, then transport it unchanged to a sealed evaluation role.
4. Run full-cascade mutation and browser-agent outcome studies before making defense-in-depth efficacy claims.
5. Repeat Azure load testing with matched payloads, a true local-pass arm, exact target lengths, decision-presence checks, and provider-native cost accounting.

Large-source caps may be tested as model ablations, but the canonical corpus must remain complete.
The balanced neural-plus-reviewer cascade is promoted only as maintained advisory behavior; rejected prompt variants, other neural guards, and weak-label pilots remain historical evidence.
Their durable conclusions are summarized in `reports/model-experiments.md`.

## Repository map

- `src/morgott/cli.py`: the `morgott` command line (data, benchmark, routing-baseline, demo, scan, cascade).
- `src/morgott/data.py`: SOURCES pin registry, contract helpers, and legacy injection views.
- `src/morgott/corpus.py`: canonical corpus build and publication interface.
- `src/morgott/sources/`: source adapters grouped by core, security, finance, task, and authorization-boundary data.
- `src/morgott/routing.py`: disk-backed canonical routing-view materialization.
- `src/morgott/overlap.py`: conservative near-overlap audit.
- `src/morgott/models/mmbert/`: maintained mmBERT preparation, full-data training, evaluation, and advisory inference.
- `src/morgott/models/`: retained linear controls and model-specific packages.
- `src/morgott/normalization.py`: strict inference-side text normalization.
- `src/morgott/policy.py`: deterministic authorization simulation.
- `src/morgott/runtime.py`: trusted label propagation and policy-gated effect invocation.
- `src/morgott/azure_app.py`: the deployed Azure preview FastAPI app.
- `tests/`: maintained invariants across the data layer, source adapters, models, serving, deployment scripts, and policy.
- `artifacts/models/`: the registered advisory model artifacts plus the retained unregistered LP-FT comparison candidate.
- `artifacts/README.md`: retention classes for everything else under `artifacts/`.
- `experiments/`: rules for disposable or study-specific experiments that do not belong in maintained model code.
- `scripts/`: operational Azure and RunPod scripts, including `azsync.sh`, `deploy-azure.sh`, and `package-mmbert-snapshot.py` (materializes the retained 1,024-token snapshot into safe serving artifacts).
- `infra/`: Azure infrastructure definitions used by the deployment scripts.
- `examples/`: small runnable integration examples such as `nooa_preflight.py`.
- `data/manifest.json`: sole versioned machine data manifest.
- `data-archive/`: retained non-reproducible generated pairs and the red-team campaign corpus (see its README).
- `reports/README.md`: index of every report, its status, and the evidence JSON consumed by code.
- `reports/dataset-selection.md`: source inclusion and exclusion decisions.
- `reports/label-audit.md`: label interpretation and known ambiguity.
- `reports/corpus-sanity-audit.md`: corpus-wide integrity checks and critical limitations.
- `reports/attention-kernel-audit.md`: measured SDPA versus FlashAttention-2 and context-length constraints.
- `reports/model-experiments.md`: authoritative historical model decision ledger.
- `reports/retrieval-assisted-reviewer-findings-20260819.md`: consolidated retrieval-assisted reviewer evidence and the selected integration recipe.
- `reports/pipeline-benchmark-20260816.md`: complete 1,024-context quality, provider, robustness, and runtime benchmark plus the dated advisory promotion disposition.
- `docs/data-contract.md`: canonical data, label, source, and split contracts.
- `docs/threat-model.md`: trust boundary and security claims.
- `docs/roadmap.md`: evidence-gated next steps.
- `docs/operations.md`: cascade and Azure preview runbooks.
- `AGENTS.md`: operating brief for coding agents.

Only load model artifacts generated locally by this project. Python model
serialization is unsafe for untrusted files.
