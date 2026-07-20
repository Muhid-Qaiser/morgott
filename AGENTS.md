# AGENTS.md

Durable operating brief for Codex and other coding agents working in this
repository. Read this file before changing code, data, experiments, reports, or
security claims. Then read `README.md`, `docs/threat-model.md`, and the report for
the experiment you are touching.

The canonical project, Python package, and CLI names are `Morgott`, `morgott`,
and `morgott`. Frozen `vulsight-*` experiment seeds already recorded in reports
are protocol provenance; renaming them would invalidate reproducibility.

This file records stable intent and decisions. Generated JSON/Markdown reports
are the source of truth for exact current metrics, hashes, and timestamps. If a
measurement changes, update the relevant report and this file's conclusion when
the decision changes; do not paste transient caches or secrets here.

## Mission, goal, and non-goals

Morgott is a research POC for defense in depth around LLM
applications and agents. The goal is to reduce successful direct jailbreaks,
direct prompt injections, and indirect prompt injections while preserving noisy
normal conversation—including security discussion and harmful requests that do
not try to subvert instruction hierarchy.

The production-shaped objective is not maximum benchmark accuracy. It is:

1. very good precision with useful recall at an explicit, measured review-cost
   tradeoff rather than a universal false-positive target;
2. robustness to source, language, length, position, style, and time shift;
3. no authority granted by a learned score; and
4. deterministic containment when every learned detector misses.

This is not a general toxicity/moderation classifier, not a claim that an agent
is “jailbreak-proof,” and not yet a production blocking system. The current
datasets and derived models are research-only because some sources are
non-commercial or lack a standard dataset license.

## Security architecture and trust boundary

The retained design separates prediction from authority:

```text
trusted runtime provenance
  direct user ----------> direct sensor -------------------+
  untrusted content ----> indirect sensor + direct fallback+--> shadow signal

untrusted user/model/retrieval/tool output/memory/planner
  --> proposed typed action
  --> deterministic reference monitor
      (caller-supplied static policy + exact schema + argument constraints)
  --> commit or deny
```

The architectural novelty worth preserving is provenance-conditioned prediction
plus deterministic authority, not a larger binary text classifier or an LLM
judge in the hot path.

Non-negotiable invariants:

- Detector output is shadow/advisory unless a separately reviewed product
  change explicitly authorizes blocking. `scan` must continue to return
  `decision: allow`; an elevated score only recommends review.
- A score may reduce privilege or escalate review. It must never grant tool,
  data, network, memory, or credential authority.
- Every side effect goes through the reference monitor. The planner, model,
  user text, retrieved content, memory, tool output, and detector are untrusted.
- Provenance and sensitive-data labels come from trusted runtime metadata, not
  attacker-controlled text, prefixes, or model output.
- Reference-monitor schemas fail closed. A capability names the exact tool,
  exact argument schema, permitted constrained values, and whether sensitive
  data may flow.
- Keep credentials outside the planner and prefer short-lived, task-scoped
  capabilities. Summarization must not erase provenance.
- The current policy ablation is a deterministic simulation, not a claim about
  a live agent or network integration.

Current policy evidence: the reference monitor commits 0/8 simulated
unauthorized actions and 2/2 benign actions; an input-filter-only ablation
commits all eight attacks. Four attack shapes are bound to safe categorical
lineage from the pinned Nemotron agentic IPI source. Preserve this ablation when
changing authorization.

## Label ontology

Do not collapse these into one “unsafe” label:

- `direct_jailbreak`: asks the target model to evade its behavioral rules.
- `direct_prompt_injection`: tries to override instruction hierarchy or expose
  privileged instructions in the direct user channel.
- `indirect_prompt_injection`: instructions embedded in untrusted email, RAG,
  web, documents, tool output, or memory. Meaning depends on provenance.
- `harmful_non_injection`: harmful intent without instruction subversion. This
  is a hard negative for this detector and may be handled by another policy.
- `benign`: ordinary or security-related input with no instruction subversion.
- `uncertain`: insufficient evidence; never silently coerce it to benign.
- `toxicity`: an independent attribute/head, not evidence of injection.

For every row preserve, when available: raw text, derived normalized text,
source, source revision/split/id, conversation/context/group lineage, input
channel, label basis, weak-label provenance, language, and known attack span.

No human labelers are available for this project. Do not create a plan that
quietly depends on human annotation or adjudication. Public benchmark labels
remain source labels; LLM/Codex/provider labels are weak supervision only. Judge
agreement measures consistency, not correctness. Weak-labelled data may train a
development ablation but may never enter a locked test, calibrate a production
threshold, or support a production-FPR/lockout claim.

## Current implementation

The retained cheap control under `src/morgott/` is intentionally small:

- Data build: twelve pinned, ungated public sources; raw text retained; NFKC +
  casefold + whitespace normalization used only as a derived matching/model
  view; exact train/evaluation overlap blocked.
- Direct sensor: character 3–5 gram TF-IDF (maximum 100k features) plus balanced
  logistic regression. Grouped validation selects minimum-precision profiles at
  80%, 85%, 90%, and 95%; the artifact uses the 85% floor as the practical
  high-precision shadow-review knee. The 0.1%, 0.5%, 1%, 2%, and 5% FPR grid is
  retained only as diagnostic evidence.
- Indirect sensor: a separate BIPIA-trained character model with a zero-observed-
  validation-FP threshold and max whole-document/blank-line-paragraph scoring.
- Untrusted-content scan: binary OR of the independently locked indirect signal
  and direct-user fallback, so classic override language is not lost.
- Exact-match and keyword rules are reported only as comparison baselines. They
  are not silently ORed into the saved model bundle.
- Model artifact: locally generated `artifacts/guard_bundle.joblib`, schema 2.
  Python deserialization is unsafe; never load an artifact from an untrusted
  source.
- Policy: a small fail-closed `authorize`/`execute` simulation with a static,
  caller-supplied capability allowlist, exact tool/context schemas, constrained
  destinations, and sensitive-data denial. It does not yet issue capabilities
  against task/user identity, expiry, or credentials. It only validates that
  provenance is a nonempty string list; provenance does not change an
  authorization decision in this POC.

Snapshot from the 2026-07-20 retained baseline (use `reports/baseline.json` for
exact current values):

- Training mixture: 35,912 rows, including 311 positives (109 ToxicChat
  jailbreaks and 202 deepset injections) and 35,601 negatives.
- Recommended shadow-review validation profile: 34/66 positives and 4/7,120
  negatives at the 85% precision floor (51.52% recall, 89.47% source-mixture
  precision, 0.0562% observed FPR). Tightening to 90% removes two false signals
  but loses eleven true signals.
- At assumed attack prevalence 0.1%/1%/5%, its validation-based expected
  precision is 47.86%/90.26%/97.97%; replacing observed FPR with its Wilson
  upper bound gives 26.32%/78.28%/94.94%. These are scenarios, not production
  calibration or a full confidence interval.
- External hard negatives: 0/4,208 alerts (Wilson 95% upper bound 0.0912%).
- ToxicChat test: 44/73 attacks; 18/4,630 source-labelled negatives alert.
- deepset test: 12/60 attacks. Obfuscated multi-turn: 908/4,136; this is the
  principal recall regression of the precision-first profile.
- JailbreaksOverTime: 3,203/3,901 source-labelled attacks; 81/18,195
  source-labelled negatives alert. Those negatives are WildChat-derived and
  audited high-score examples include clear DAN/jailbreak language, so this is
  label noise/source confounding—not a clean product FPR.
- BIPIA indirect sensor: 84/125 payloads and 252/375 poisoned contexts; 2/167
  clean contexts alert. The clean denominator is too small for a production FPR.
- Tensor Trust: direct sensor catches 262/908 unique attack texts; the combined
  untrusted-content shadow signal catches 841/1,346 attack contexts. That
  context partition contains no matched clean control and may expose benchmark
  artifacts, so it is recall-only evidence.
- Nemotron agentic IPI: 1,272 fully synthetic successful source attacks reduce
  to 676 exact-unique injection texts with zero exact/near overlap against the
  active corpus. Both retained signals catch 0/676 at recommended thresholds;
  independently selected 5% component-FPR diagnostics catch 46.01% combined.
  The positive-only source has no benign controls and cannot estimate FPR,
  precision, utility, or production safety.

These counts show why false-positive discipline and impact containment matter:
the detector is useful but bypassable and its indirect FPR estimate is weak.

## Data sources and intended roles

`src/morgott/data.py` pins revisions; `reports/data_manifest.json` pins
download/output digests and exact counts.

| Source | Role | Training status |
|---|---|---|
| ToxicChat 0124 | explicit jailbreak labels and noisy chat controls | direct train + official test |
| deepset prompt-injections | direct injection labels/controls | direct train + official test |
| OpenAssistant OASST1 | multilingual accepted human chat | direct train + validation/hard negative |
| BIPIA | provenance-dependent payloads, poisoned and clean contexts | separate indirect train + official test |
| XSTest | over-refusal/security-trigger hard negatives | evaluation only |
| HarmBench | harmful goals without injection | evaluation hard negative only |
| Do-Not-Answer | harmful requests without injection | evaluation hard negative only |
| NotInject | instruction/security wording hard negatives | evaluation only |
| Multi-turn jailbreak corpus | obfuscation/cipher/source-shift positives | grouped evaluation only |
| JailbreaksOverTime | temporal/source/style shift with noisy labels | evaluation/audit only |
| Tensor Trust | human prompt hijacking/extraction | evaluation only; no standard license |
| Nemotron Agentic IPI | synthetic successful tool-output injections with impact/tool lineage | evaluation and policy-scenario lineage only |

Do not treat source negatives as automatically benign. Do not turn HarmBench,
Do-Not-Answer, AgentHarm, AdvBench, Aegis, BeaverTails, or toxicity labels into
positive injection labels. Skip gated sources under the current scope. The
review of the user's private dataset-research document led to adding the two
small Tensor Trust robustness suites and to the WildChat experiment; the large
Tensor Trust game dump was deliberately not downloaded.

ToxicChat is CC-BY-NC-4.0 and Do-Not-Answer is CC-BY-NC-SA-4.0. Replace them,
and resolve Tensor Trust redistribution terms, before commercial derivation.

### Deferred attack-source decisions

HackAPrompt is not an active source. As observed on 2026-07-20,
`hackaprompt/hackaprompt-dataset` remains an auto-gated/contact-sharing Hugging
Face repository; an anonymous Parquet request returns 401. Do not use the
configured Hugging Face token, accept the gate, or fetch a mirror unless the
user explicitly changes the ungated-data scope. The pinned card/repository
audit is `reports/hackaprompt-yaklang-audit.md`.

If HackAPrompt is explicitly authorized later, start evaluation-only and use
only `user_input` as detector text. A competition submission is an attack
attempt even when `correct=false`; `correct` is target-model-specific success,
never a benign label. Do not train on `prompt`, `completion`, or
`expected_completion`. Verify the gated file hash/schema, locally scan PII and
toxicity, deduplicate against every existing source, aggregate repeated text
across models/levels, and hold out whole challenge levels. Participant-group
isolation cannot be claimed from the published card schema.

The Yaklang `llm-prompt-injection` skill is pinned only as a taxonomy and
stateful-scenario reference. Do not install/vendor it or ingest/generate rows
from its examples. Its direct overrides, prompt extraction, encoding, and basic
indirect examples largely duplicate deepset, Tensor Trust, multi-turn, and
BIPIA. Preserve its useful gaps—tool-description/tool-output injection,
cross-tool/MCP exfiltration, Markdown egress, and multi-turn splitting—as
future typed AgentDojo/reference-monitor scenario families. Direct tool misuse,
SQL injection, and harmful fiction are not automatically prompt injection.

PromptShield is a completed, isolated evaluation-only audit, not an active data
build input. Its public test has 6,486 source-positive and 17,030
source-negative rows but no row-level source/group lineage; the paper discloses
aggregation from public chat, instruction, HackAPrompt, and generated-attack
families. At the locked 85% profile the character control catches 194 positives
and alerts on 240 negatives; at 95% it catches 86 and alerts on 11. Removing 97
test rows overlapping active fit data barely changes the result. Never train on
this release without first reconstructing source/license/group lineage.

## Split, leakage, and evaluation contract

- Pin every source revision and record license, download digest, output digest,
  label basis, channel, group, and source split.
- Preserve raw text locally; normalization is a derived view.
- Block exact train/evaluation overlap. Near-deduplicate new broad web/chat data
  against all training and locked evaluation sets before use.
- Group conversation trees, BIPIA contexts and their clean/poisoned views,
  jailbreak goals, templates, mutation seeds, source documents, and derived
  windows across splits. Never split related rows independently.
- Select thresholds on deterministic grouped validation data before reading
  official/external results. Never retune a threshold, feature, or prompt to fix
  an untouched final-test failure after observing it. A public suite becomes a
  repeated development comparison once its result influences model or data
  selection; keep its rows and threshold frozen, but stop describing it as an
  independent final test or production evidence.
- Compare candidates across the same predeclared grid: 80%, 85%, 90%, and 95%
  minimum validation precision plus the 0.1%, 0.5%, 1%, 2%, and 5% FPR
  diagnostics. Report exact TP/FN/FP/TN counts, recall, precision with its
  evaluation prevalence, expected precision at stated prevalence scenarios,
  FPR, Wilson bounds, PR-AUC where useful, subgroup results, latency, VRAM,
  model/data revision, and known training overlap.
- Stratify user-friction evidence by language, topic, length/window count,
  toxicity, and security-trigger terms. Report expected precision at realistic
  attack prevalence and leave-one-source-out performance before promotion.
- Document-level max/OR aggregation increases FPR with window count. Evaluate
  document FPR by number of windows; do not infer it from single-window FPR.
- For positive long documents, train only windows that contain a known payload
  span. Never label every window in an attacked document positive.
- New generated, mutated, weak-labelled, or exploratory data is development
  only unless frozen prospectively. It never improves a headline by being added
  after the result is known.

## WildChat model-only weak-supervision contract

WildChat-1M is broader and noisier than the current normal-chat corpus, but
`non-toxic` is not synonymous with `benign` for injection. The experiment under
`experiments/wildchat_pseudolabel/` must obey all of the following:

The pinned public sample exposes no source-toxicity variation: all 5,000 pilot
rows are `unavailable` for that stratum. Do not invent a toxicity split or infer
one from injection labels. Keep ToxicChat toxicity evidence separate, and only
add a WildChat toxicity stratum if a future ungated source actually supplies it.

1. Stream a deterministic, conversation-aware 5k-row sampled pilot; do not
   download the full multi-gigabyte dataset merely to sample it. Stratify by
   language, length, available source toxicity, local coarse topic, and
   security-trigger terms. `sampled_rows` and `accepted_negative_rows` are
   different quantities.
2. Use user turns only. Drop metadata, local PII/secret matches, exact and near
   duplicates, and every overlap with training and JailbreaksOverTime/locked
   evaluation. Do not send raw provider responses or secrets anywhere.
3. Label `benign`, `injection_or_jailbreak`, `harmful_non_injection`, and
   `uncertain`; store toxicity separately.
4. Use two independent OpenRouter model families, temperature 0, one-shot strict
   JSON, no tools, fallback, retry, repair, or ReAct. Request ZDR routing and
   deny provider data collection. Count unavailable calls explicitly.
5. Accept a weak benign training row only when both primary judges return
   high-confidence benign. Discard disagreement/uncertainty. Send every
   detector-hard unanimous-benign row and a deterministic random 10% of all
   other unanimous-benign rows to a third, independent model family; retain it
   only if the third judge agrees with high confidence.
6. Store prompt version, input/sample hashes, model/provider IDs, structured
   label/confidence, agreement, redaction flags, token/cost/latency aggregates,
   and lineage. Do not store raw prompts, provider responses, or credentials.
7. First compare zero weak negatives with all accepted negatives from the 5k
   sampled pilot. Only if that predeclared development gate improves recall
   without worsening a normal subgroup may sampling scale until 5k, 20k, then
   50k accepted negatives exist. Use source weighting so weak negatives cannot
   swamp the 311 direct positives.
8. Report judge agreement as agreement, never accuracy. No metric on these
   labels is a production FPR. ToxicChat, deepset, multi-turn, BIPIA, Tensor
   Trust, NotInject, position stress, and JailbreaksOverTime may remain frozen
   repeated comparison suites, but after they select an ablation they are
   development evidence—not untouched final tests. A final production gate
   requires genuinely independent labels that do not currently exist.

The completed 5,000-row sample accepted 2,430 weak negatives after 10,266
one-shot judgments. The weighted ablation failed its scale gate at every
declared precision profile. At the 85% profile it kept grouped validation at
34 TP / 4 FP but reduced multi-turn recall from 908/4,136 to 291/4,136 and
slightly reduced indirect macro recall. The recorded decision is `stop`: do not
collect 20k/50k under this recipe and do not add these rows to the baseline.
See `reports/wildchat-ablation.md` and the machine-readable results.

Network execution must remain opt-in and bounded with `--execute`; offline
sample/preflight/self-tests are the default. No human review queue is required
for completion and no missing human review may be silently treated as approval.

## Model experiment ledger and design decisions

The cheap character model remains the control. Existing neural comparisons used
the stringent 0.1% diagnostic. They did not earn promotion there, but that does
not establish dominance across the broader precision/FPR grid; rerun plausible
candidates on all declared profiles before a future architecture decision:

- Frozen multilingual E5 + FP32 logistic heads: rejected as a promotion; it
  largely missed multi-turn attacks and raised indirect/NotInject false signals.
- Frozen `answerdotai/ModernBERT-base`, masked-mean and CLS probes: rejected as
  frozen features. Mean beat CLS but both lost substantial recall, including
  zero multi-turn recall. This rules out linear separability of frozen features,
  not end-to-end ModernBERT. A longer frozen run cannot repair short-input
  separability, so the 1K/2K probe was not spent.
- `siberiancat/modernbert-prompt-injection`: rejected. Its published threshold
  produced severe false positives; a locally locked threshold collapsed recall.
- Patronus Wolf/mmBERT small: rejected. Its card threshold over-defended, its
  locked direct recall was zero, and a high Tensor-context result lacked clean
  controls and did not transfer to BIPIA. No 2K rerun was justified.
- PIGuard DeBERTa checkpoint: rejected for promotion at the 0.1% diagnostic;
  NotInject false signals were high and BIPIA is a disclosed benchmark family.
- ProtectAI DeBERTa checkpoint: rejected at the 0.1% diagnostic; its required
  threshold saturated above useful scores and removed direct recall.
- One-shot OpenRouter reviewers: advisory research only. They added seconds of
  latency, false/unavailable outcomes, privacy/cost/provider dependencies, and
  no deterministic security boundary. Do not add ReAct, fallback, or a remote
  hot-path gate.

Off-the-shelf checkpoints are not a fair architecture comparison because their
training corpora differ and may overlap evaluations. The controlled one-seed,
one-epoch 512-token screen used identical grouped data, masked-mean head, loss,
seed, FP16 policy, and operating grid for base ModernBERT and DeBERTa-v3:

- ModernBERT underfit: 6/66 validation TP and 1/7,120 FP at the 85% profile,
  with zero deepset and multi-turn detections.
- DeBERTa was competitive but not promotable: 36/66 TP and 6/7,120 FP, plus
  4/4,208 hard-negative alerts and only 5/4,136 multi-turn detections. Its 95%
  profile (25/66 TP, 1/7,120 FP, 0/4,208 hard FP) merits a fuller rerun, while
  its 0.999 threshold warns about saturation/calibration stability.

This screen does not establish intrinsic architecture superiority. “Newer” is
not evidence, and one seed/epoch is not a release result.

Run candidate ablations in this order and stop when evidence fails:

1. Rerun DeBERTa-v3 for three seeds and a predeclared fuller schedule, preserving
   the same split/grid and adding calibration stability checks.
2. Add positive long-position attacks and matched clean documents before any
   512-window versus native-2K comparison. ModernBERT earns no 2K run yet.
3. Train tiny channel-specific heads only with realistic indirect positives and
   matched clean application content, routed by immutable provenance.
4. Test OOD-energy/tokenizer perturbation separately only after the fuller
   DeBERTa control; scalar mixing, LoRA, attention aggregation, or an ensemble
   remain later hypotheses and require a measured error prediction.

ModernBERT Decoder is causal generation machinery and is not appropriate for
this encoder classification test. Longformer adds little before native
long-context ModernBERT is fairly tested. Do not add architectural novelty for
its own sake; the security contribution is the prediction/authority split.

## Hardware, model-loading, privacy, and provider rules

- Local GPU: NVIDIA RTX 4050 Laptop, 6 GB VRAM. Run GPU experiments serially;
  coordinate with other agents and check GPU health before starting.
- Use FP16 encoder inference/training where numerically safe; keep classifier
  heads, thresholds, probabilities, losses when needed, and reported metrics in
  FP32. Use dynamic padding/length bucketing. Add gradient accumulation or
  checkpointing only after measuring need.
- Prefer PyTorch SDPA. FlashAttention 2 or `torch.compile` must earn their added
  dependency/compile cost through a measured bottleneck.
- Pin Hugging Face revisions; require safetensors and
  `trust_remote_code=False`. If custom model code is unavoidable, audit and
  vendor the smallest local implementation before loading weights.
- Never inspect, print, persist, or commit `.env`, API keys, Hugging
  Face/OpenRouter tokens, credentials, raw provider responses, or sensitive
  prompts. Scripts may silently load secrets into their process environment
  without logging their names or values.
- Redact or drop likely PII and secrets locally before any provider call.
  Public-source chat can still contain personal data.
- Remote calls default to dry-run and require an explicit bounded `--execute`.
  Record cost, latency, unavailability, and privacy routing without recording
  raw content. No retry/fallback means one unavailable call stays unavailable.

## Repository map and source-of-truth files

- `src/morgott/data.py`: source pins, consolidation, normalization,
  grouping metadata, overlap blocking, and manifest generation.
- `src/morgott/detector.py`: cheap sensors, grouped threshold selection,
  evaluation, local artifact, and shadow scanner.
- `src/morgott/openrouter.py`: shared no-redirect, one-attempt provider
  transport; experiment-specific prompts and response validation stay separate.
- `src/morgott/policy.py`: deterministic reference monitor and ablation.
- `tests/`: fast standard-library invariants for the retained POC.
- `experiments/gpu_baselines/`: E5, PIGuard, and ProtectAI frozen evaluations.
- `experiments/modernbert/`: frozen ModernBERT mean/CLS linear probe.
- `experiments/modernbert_checkpoints/`: pinned SiberianCat and Wolf audit.
- `experiments/openrouter_review/`: bounded one-shot provider reviewer smoke test.
- `experiments/wildchat_pseudolabel/`: broad-chat sampling, weak labels, and
  training ablations; outputs/data are ignored.
- `experiments/encoder_finetune/`: fair one-seed ModernBERT/DeBERTa base-model
  screen and compact comparison.
- `experiments/nemotron_agentic_ipi/`: positive-only agentic IPI transfer and
  source-bound containment audit.
- `experiments/promptshield_audit/`: isolated public-corpus leakage, length,
  and locked-character evaluation; raw source data is ignored.
- `reports/baseline.{json,md}`: retained sensor metrics and operating points.
- `reports/data_manifest.json`: exact data lineage, hashes, counts, and licenses.
- `reports/label-audit.md`: known source-label noise and interpretation limits.
- `reports/dataset-selection.md`: why candidates were included, deferred, or
  rejected after reviewing the user's dataset-research list.
- `reports/hackaprompt-yaklang-audit.md`: pinned access/schema/label/taxonomy
  decisions for the two deferred sources; no gated source data is bundled.
- `reports/model-experiments.md`: consolidated neural/provider decisions when
  present; experiment READMEs/JSON hold full detail.
- `reports/architecture-research.md`: literature-backed decision map from local
  encoder experiments through provenance and action containment.
- `reports/policy_ablation.{json,md}`: authorization evidence.
- `docs/threat-model.md`: security claim and trusted computing base.
- `docs/roadmap.md`: evidence-gated next work.

Generated `data/`, `artifacts/`, model/embedding caches, and raw/local provider
outputs are ignored. Compact reproducible experiment result JSON/Markdown is
versioned with code, tests, and manifests. Never commit `.env`.

## Reproduction and validation

Core POC:

```bash
python3 -m pip install -e .
make data
make benchmark
make demo
make test
make poc
```

Selected experiment checks:

```bash
python experiments/gpu_baselines/run_embeddings.py --batch-size 128
python experiments/gpu_baselines/run_attention.py --model piguard --device cuda
python experiments/gpu_baselines/run_attention.py --model protectai --device cuda
python experiments/modernbert/run_probe.py --max-length 512 --batch-size 16 --device cuda
PYTHONPATH=src python experiments/modernbert_checkpoints/evaluate.py --self-test
python -m unittest discover -s experiments/openrouter_review -v
PYTHONPATH=src python experiments/encoder_finetune/run.py --self-test
python experiments/encoder_finetune/compare.py
PYTHONPATH=src:. python -m unittest discover -s experiments/nemotron_agentic_ipi -v
PYTHONPATH=src:. python -m unittest discover -s experiments/promptshield_audit -v
```

The core editable install does not include optional Torch/Transformers GPU
dependencies. These commands assume this preconfigured workspace; read each
experiment README before reproducing elsewhere. Its dependency assumptions,
pinned input hashes, and flags are part of the protocol. Run GPU commands
serially. Check broadly when useful, but format only files owned by the active
task, for example:

```bash
ruff format path/to/changed.py path/to/test_changed.py
ruff check src tests experiments
```

Before handoff, run the proportional tests, regenerate any report whose inputs
changed, inspect `git status --ignored`, and confirm no ignored data, artifacts,
caches, provider outputs, or secrets are staged.

## Agent change workflow

1. State assumptions, the smallest useful plan, and measurable success criteria.
   If two interpretations materially change security or data use, surface the
   tradeoff instead of silently choosing.
2. Inspect current code/results and preserve unrelated or user-authored changes.
   Match local style; do not refactor adjacent code merely because it could be
   cleaner.
3. Prefer the cheapest test that can falsify the idea. Add or update an
   invariant before risky behavior changes. Avoid speculative abstractions,
   dependencies, configurability, and model cascades.
4. Make surgical changes. Remove only imports/functions/artifacts made dead by
   your own change. Every changed line should trace to the active goal.
5. Reproduce evidence at the appropriate risk level. Do not promote a candidate
   because it is modern, has a favorable model card, or improves recall by
   silently accepting a worse tradeoff; compare the declared operating curves.
6. Update README, manifest, label/data audit, threat model, roadmap, experiment
   ledger, and this brief whenever their claims or decisions change.

When multiple agents work concurrently, assign bounded, non-overlapping tasks;
only one agent owns a GPU job at a time. Agents share the same worktree, so do
not reset, discard, or overwrite another agent's edits. The primary agent owns
final integration, validation, and Git operations.

## Promotion and completion gates

A learned candidate earns promotion only if it improves on the character
control across the declared precision/FPR grid or at an explicitly justified
precision-first review tradeoff; has no regression on normal/harmful-non-injection language,
topic, length, toxicity, or trigger strata; has acceptable paired/grouped
uncertainty; discloses overlap; and meets batch-1/p95 latency and VRAM budgets.
An OR ensemble must be recalibrated and evaluated as its own detector because
component FPRs accumulate.

No learned detector may block users based on current public or model-judged
evidence. A future blocking experiment requires independently labelled,
production-shaped traffic and narrow user-impact confidence intervals. Because
no human labelers are available now, that gate remains intentionally unmet; do
not weaken or redefine it to declare success.

Near-term priorities:

1. Add positive long-document/position tests with known spans and matched clean
   controls; current negative-only position stress cannot measure buried attack
   recall.
2. Integrate the reference monitor with AgentDojo or an equivalent stateful
   environment, propagate provenance/taint through memory/tool/RAG flows, and
   measure unauthorized side effects and benign task utility. Include the
   pinned tool-description/tool-output, cross-tool/MCP, Markdown-egress, and
   multi-turn-splitting scenario families from the Yaklang audit.
3. Run the fuller three-seed DeBERTa continuation only after its schedule,
   calibration checks, and source-held-out success criteria are frozen.
4. Add realistic benign application/tool-output controls for the Nemotron
   domains. The current positive-only 0/676 transfer result diagnoses a gap but
   cannot safely become blanket positive training data.
5. Once static stateful evaluation is reproducible, add defense-adaptive
   PIArena-style attacks. Keep browser/image injection as a separate threat
   model; text-sensor recall does not establish multimodal agent safety.

Known limitations to keep visible: English-heavy positive labels, noisy source
labels, small indirect clean control, severe PromptShield/Nemotron transfer
failures, no adaptive target-model attack, no live tool credentials or egress,
no stateful-agent benchmark yet, no independently labelled production traffic,
and research-only data licensing.
