# AGENTS.md

Durable operating brief for coding agents in this repository. Read this file,
`README.md`, and `docs/threat-model.md` before changing data, labels, model
claims, or authorization code.

The canonical project, package, and CLI names are `morgott`, `morgott`, and
`morgott`. Old `vulsight-*` seeds that remain in historical Git objects are
provenance, not names to revive.

## Current status

The active deliverable is the canonical data corpus.
One minimal unweighted word n-gram baseline is retained as a cheap broad-routing control.
Completed ModernBERT and data-ablation experiments were not promoted.
Clean WildGuardMix counterexamples and an English ModernBERT plus mmBERT-base ensemble improved the open-development direct frontier, but remain single-seed research shadows.
Their conclusions and exact metrics live in reports, while their runners are not active code.
No model is approved for blocking.

External validation in July 2026 found that ensemble does not transfer: 0.00%
TPR at 1% FPR on the public PromptShield split, and 49.2% single-mutation
evasion on its own dev-test suite. Its in-corpus FPR figure describes text of
64 tokens or fewer. See `reports/model-experiments.md`. Do not quote the
retained FPR/recall pair without those qualifiers, and do not treat it as
evidence a detector works.

The retained character n-gram/logistic-regression detector is only a cheap POC
control over the original injection views. Historical neural, OpenRouter,
PromptShield, Nemotron, and WildChat experiment runners were removed from the
active tree because they used an older corpus or ended in a stop decision. Their
short conclusions remain in `reports/model-experiments.md`; Git history is the
archive.

On 2026-07-28 the repository owner authorized one bounded, artifact-only modelling exception.
The exception fits a frozen mmBERT head on the complete leakage-filtered mixture and runs one update-matched rank-8 mmBERT LoRA engineering gate using PromptShield train.
PromptShield validation selects checkpoints but does not fit parameters or select operating thresholds, and PromptShield test remains already-open, source-held-out benchmark development data.
The retained weights use Git LFS and are listed in `model-artifacts.json`.
They are advisory research shadows, are not wired into `morgott scan`, and are not approved for blocking or authorization.
The compact maintained recipe in `src/morgott/models/mmbert/` supports full-data frozen-head and LoRA retraining without reviving the historical runners.
Code availability is not authorization to execute or promote another encoder run.
No further encoder sweep or model promotion is authorized by this exception.

## Mission and non-goals

morgott studies defense in depth for LLM applications and agents. It should
reduce successful direct jailbreaks, direct prompt injections, and indirect
prompt injections while preserving ordinary conversation, including legitimate
security and finance discussion.

The target is not general content moderation, toxicity detection, or proof that
an agent is jailbreak-proof. A harmful request without instruction subversion
is a separate label. Licensing is not an inclusion filter for the user's current
research, but every source must keep license and provenance metadata.

## Security boundary

```text
trusted runtime provenance
  direct user ----------> advisory direct sensor --------+
  untrusted content ----> advisory indirect sensor -------+--> route/review

untrusted user/model/retrieval/tool output/memory/planner
  --> proposed typed action
  --> deterministic reference monitor
  --> commit or deny
```

Non-negotiable rules:

- Learned output is advisory. `scan` continues to return `decision: allow`.
- A score may reduce privilege or escalate review; it never grants authority.
- Every side effect passes through the reference monitor.
- Provenance and sensitive-data labels come from trusted runtime metadata, not
  attacker-controlled text or model output.
- Schemas fail closed and capabilities name exact tools, argument shapes, and
  constrained values.
- Finance, cybersecurity, or other topic vocabulary is never itself a deny rule.
- Keep credentials outside the planner and prefer short-lived, task-scoped
  capabilities in any future runtime integration.

The current policy code is a deterministic simulation, not a deployed agent,
credential broker, or network boundary.

## Canonical data contract

There is one data root and one machine source of truth:

```text
data/
  manifest.json
  sources/
  views/injection/
  views/routing/
  audits/
  quarantine/
```

`data/sources/` contains canonical standardized source shards. Derived views,
audits, and quarantine records may be rebuilt. Only `data/manifest.json` is
versioned; never recreate `processed/`, `expanded/`, a second manifest, or a
second data root. Exact current counts and hashes belong only in the manifest.

The public build is one command:

```bash
uv run morgott data
```

It must fail if any required source, access gate, schema, or pinned digest is
unavailable. Remove the old manifest before mutating outputs and publish the new
one last; a failed build must leave no apparently valid manifest. Do not add a
partial source-build mode. For partition-logic-only changes, `uv run morgott
data --routing-only` must hash- and schema-verify every canonical source against
the current manifest, rebuild only routing derivatives, and publish the updated
manifest last.

Corpus construction has no row cap. Retain every valid, non-empty detector-text
projection and available lineage needed by morgott. Source shards are
standardized projections, not byte-for-byte mirrors: unrelated upstream fields
and unusable detector inputs may be omitted, with exclusions recorded in the
manifest. Every exact-unique routing-eligible row enters a grouped development
role unless conflict or leakage rules quarantine it. Sampling, source caps, and
weighting belong to model recipes and must never rewrite the canonical corpus.

For every row preserve, when available:

- raw detector text and derived normalized-text hash;
- source, revision, split, source ID, group, and split group;
- conversation, task, attacker, team, scenario, document, or time lineage;
- input channel, label basis, language, and known payload span;
- source-specific annotations in `origins`.

Normalization is Unicode NFKC, case folding, and whitespace collapse. It is a
matching/model view; never replace raw text with normalized text.

## Label ontology

Do not collapse these concepts:

- `direct_jailbreak`
- `direct_prompt_injection`
- `indirect_prompt_injection`
- `harmful_non_injection`
- `benign`
- `uncertain`
- `toxicity` as an independent attribute

Canonical row schema 5 supports a conservative first-stage route:

- `routing_label=0` only for source-supported benign content.
- `routing_label=1` for injection, jailbreak, harmful intent, toxicity, or an
  unresolved row that must not silently pass as benign.
- `injection_label` is `0`, `1`, or null. Null is unknown, never negative.
- `security_tags` may co-occur. Future subtype heads use masked losses.
- `injection_subtype_training_eligible=false` masks unknown or disagreeing
  injection annotations.
- `routing_training_eligible` is derived solely from `source_role`; it governs
  routing views, not the separately curated legacy injection views.
- Exact-text merges retain every source annotation in `origins`; disagreement
  is never resolved by source load order.

`source_role` records intended source use. `data_role` records the actual
derived partition. Auxiliary, uncertain, and quarantined rows never silently
enter ordinary supervised fitting.

“Not injection” is not evidence of broad benignity. Deepset negatives, BIPIA
clean contexts, OASST accepted turns, and ToxicChat rows without jailbreak or
toxicity remain available for legacy/auxiliary use but do not supervise the
broad router. OASST source rows retain its available moderation metadata.

No human labelers are available. Public source labels remain source labels;
model/provider/Codex labels are weak supervision only. Agreement is not
accuracy. Selected weak labels may enter the grouped train, validation, and
dev-test development roles when their `label_basis` remains explicit and
metrics are sliced by evidence strength. They never enter a prospective locked
final test or support production-FPR claims.

## Important source mappings

Exact revisions, digests, counts, and licenses belong in the manifest. Detailed
label authority and caveats live in `reports/label-audit.md`; source selection
decisions live in `reports/dataset-selection.md`.

Stable invariants: LLMail `False`/`Unclear` stay uncertain; failed HackAPrompt
and Tensor Trust attacks remain attempts; their target-specific success is only
outcome metadata. Tensor Trust defenses/model outputs and adversarial-benign
WildJailbreak remain auxiliary. WildGuardMix harmfulness and `adversarial` stay
independent; its model-labelled train rows remain auxiliary and its
human-annotated test rows may enter dev-test. BrowseSafe positive HTML remains
whole-document data because payload spans are unavailable. Harmful content is
never silently relabelled as prompt injection. LMSYS automated positive safety
tags remain uncertain metadata and do not supervise the router.

LLMail raw attempts use phase + team + challenge level as split lineage;
HackAPrompt keeps whole challenge levels together because they share a task/base
prompt and the public data does not expose participant identity. Tensor Trust raw
currently groups by anonymized attacker for the first routing POC; its dev-test is
not task-held-out, and task grouping is deferred to prospective final evaluation.

HarperValleyBank retains meaningful human-corrected caller and agent turns grouped
by complete simulated conversation; only callers enter the direct-user recipe.
TAT-QA questions, paragraphs, and serialized tables share hybrid-context lineage,
with official dev/test held out. FinanceBench is dev-test only and grouped by
document. Mind2Web contributes only confirmed official training tasks after local
secret and PII screening; suspicious raw task text stays in source-level quarantine.
API-Bank remains deferred until tool-output diagnostics show a gap.

Authenticated Hugging Face access has been explicitly authorized after the
user accepted the applicable gates. Never print tokens or `.env` contents.

## Split and leakage contract

Use `train`, `validation`, and `dev_test` consistently:

- train fits parameters;
- validation selects models and thresholds;
- dev-test is a frozen repeated development comparison, not a pristine final
  test after its result influences a decision.

Group related conversation trees, BIPIA contexts and variants, mutation goals,
teams, attackers, challenge levels, base prompts, and source documents. Target
approximately 70% train, 10% validation, and 20% dev-test after exact
deduplication, stratified by source and routing label where lineage permits.
Official test/evaluation-only rows remain dev-test and may cause documented
per-source deviations.

Exact text is materialized once with all origins. A duplicate spanning unrelated
lineages becomes a singleton exact-text split atom; it must never union the
surrounding lineages transitively. An official origin fixes that merged text in
dev-test, while a genuine lineage containing an official row remains grouped in
dev-test.

Block exact train/dev-test overlap. Quarantine strict near matches from new
candidate sources to locked evaluation and from validation to train. Preserve
conflicts and rejected rows in quarantine for auditability. Record target and
actual ratios, source/label distributions, and largest-group shares in the sole
manifest rather than forcing ratios by duplicating or discarding rows.

A future final test must be frozen prospectively. Report per-source and
leave-one-source-out performance; aggregate row accuracy can be dominated by a
single large source.

## Model work after the data pass

Do not revive old experiment runners as the default trainer.
Start with a new, small recipe against the routing views:

1. cheap character or word linear baseline;
2. one frozen encoder comparison on the identical grouped data;
3. source-balanced sampling or weights as an explicit ablation;
4. masked direct, indirect, jailbreak, and harmful-intent heads where annotations are known;
5. validation-selected operating points and source-heldout diagnostics;
6. prospective final evaluation before any blocking claim.

Do not repeat context-length, document-bag, or BIPIA augmentation experiments on the same labels.
Before another encoder run, add realistic matched transaction tasks, paired multilingual transformations, known-span long-document attacks, and stronger same-source controls.
Toxicity remains deferred until a second independent positive source and matched negatives exist.

The completed 2026-07-28 frozen-head and one-seed LoRA experiments are the bounded exception recorded above.
Their already-consumed PromptShield and SEP results are development evidence, not a prospective final test.
The maintained mmBERT trainer exists for reproducibility and a future explicitly authorized run.
The next encoder run remains deferred until the evidence requirements in this section are met.

Before promoting auxiliary rows, follow the diagnostic-first OASST1 and
WildJailbreak procedure in `reports/dataset-selection.md`; those rows must not
silently enter fitting or threshold selection.

Current direct-user source identity explains most of the label entropy, and the completed ModernBERT recipe had no long-benign denominator above its context limit.
Treat source-heldout folds, genuinely matched contrasts, and a genuine long-benign denominator as prerequisites for interpreting another encoder run.
Do not increase context length merely because FlashAttention makes it fit.

The 7,000-ish number in historical discussion was the negative count in an old
validation split, not a corpus cap, training cap, or context limit.

## Privacy and external providers

The active corpus builder makes no OpenRouter calls. The old remote-review and
WildChat weak-label paths were removed. Do not send corpus text to a provider
merely because an API key exists. Any future remote-label experiment must be
explicit, bounded, development-only, and separately reviewed.

Never inspect, print, persist, or commit `.env`, API keys, Hugging Face tokens,
credentials, raw provider responses, or sensitive prompts.

## Maintained files and verification

- `src/morgott/data.py`: original source adapters and injection views.
- `src/morgott/corpus.py`: canonical corpus build and publication interface.
- `src/morgott/sources/`: additional source adapters grouped by data domain.
- `src/morgott/routing.py`: disk-backed routing-view assembly.
- `src/morgott/overlap.py`: conservative near-overlap checks.
- `src/morgott/models/detector.py`: optional shadow-control model.
- `src/morgott/models/routing_baseline.py`: minimal broad-routing word n-gram control.
- `src/morgott/models/mmbert/`: maintained external-data preparation, full-data frozen/LoRA training, evaluation, and inference.
- `src/morgott/normalization.py`: strict inference-side text normalization.
- `src/morgott/policy.py`: deterministic reference-monitor simulation.
- `tests/`: maintained invariants.
- `data/manifest.json`: sole machine data manifest.
- `reports/dataset-selection.md`: source decisions.
- `reports/label-audit.md`: label decisions.
- `reports/corpus-sanity-audit.md`: corpus integrity and critical limitations.
- `reports/model-experiments.md`: concise historical model ledger.
- `experiments/`: placement rule for disposable or study-specific work that is not maintained model behavior.
- Git commit `91e8c829c8b39c8ff37a6ca2479c8fc057168d39`: immutable source provenance for the completed July 2026 model runs.
- `artifacts/models/`: selected frozen and LoRA inference artifacts plus immutable result and evaluation records.
- `model-artifacts.json`: sole registry for the owner-approved LFS research
  weights and their hashes.
- `docs/threat-model.md` and `docs/roadmap.md`: current architecture and plan.

Before handing off a data change:

```bash
make check
git diff --check
```

Then rebuild the corpus, verify every manifest hash/count and split invariant,
and inspect the quarantine summary. Never copy current counts out of the
manifest into narrative documents.
