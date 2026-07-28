# morgott

morgott is a research POC for prompt-injection and agent-security work.
The maintained deliverable is a reproducible, provenance-preserving data corpus.
The current modelling deliverable is a versioned full-data frozen-mmBERT shadow plus a promising one-seed LoRA gate documented in `reports/model-experiments.md`.
Their small inference artifacts use Git LFS, and neither is wired into the product CLI.
No model is finalized or approved for blocking.

The security design deliberately separates prediction from authority:

```text
direct user or untrusted content -> advisory classifier -> route/review signal
proposed tool action             -> reference monitor   -> approve or deny
```

A detector may miss an attack or flag a benign prompt. It therefore never grants
tool, data, network, memory, credential, or financial authority. Every side
effect must still pass deterministic policy using trusted runtime context.

## Development

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

The often-mentioned roughly 7,000 rows were the negative side of an old grouped
validation split. They were never a download cap, corpus cap, or context limit.

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
It verifies inputs against the canonical manifest, uses the untouched 0.5 cutoff, and reports aggregate and per-source development metrics.
Completed neural and data-ablation runners are not part of the active package because none produced a promotable model.
Their metrics and stop decisions remain in the versioned reports.
The frozen-mmBERT full-data pair-ranking candidate and the one-seed LoRA gate are retained only as advisory first-pass research shadows under `artifacts/combined_generic/`.
Their external transfer evidence remains too weak for blocking, and every side effect still requires deterministic policy authorization.
Both generic heads strictly normalize and truncate each input to its first 512 tokens.
They do not chunk long documents or localize injected spans.

The registered model keys and exact hashes are in `model-artifacts.json`.
Score downstream JSONL without producing a decision:

```bash
PYTHONPATH=src:experiments uv run python \
  experiments/score_shadow_model.py \
  model-artifacts.json full-frozen-s42 input.jsonl scores.jsonl
```

Each input row must contain unique `id`, non-empty `text`, and trusted `input_channel` set to `direct_user` or `untrusted_content`.
The output contains only the raw score, channel, model revision, and artifact hashes.
Use `full-frozen-s42`, `full-frozen-s43`, `full-frozen-s44`, and `lora-s42` independently for downstream comparisons.
Do not average or OR them without evaluating that new ensemble.

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
- `src/morgott/corpus.py`: additional source adapters.
- `src/morgott/routing.py`: disk-backed canonical routing-view materialization.
- `src/morgott/overlap.py`: conservative near-overlap audit.
- `src/morgott/detector.py`: optional cheap shadow-control model.
- `src/morgott/policy.py`: deterministic authorization simulation.
- `tests/`: maintained data, detector, and policy invariants.
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
