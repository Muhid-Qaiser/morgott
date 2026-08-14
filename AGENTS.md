# AGENTS.md

Durable operating brief for coding agents in this repository.

`CLAUDE.md` is a symlink to this file. Edit only `AGENTS.md`; the mirror
follows automatically. Never edit `CLAUDE.md` directly or create divergent
per-agent instruction files.

The canonical project, package, and CLI names are all `morgott`. Old
`vulsight-*` seeds in historical Git objects are provenance, not names to
revive.

## Read before you change things

Always read this file and the relevant section of [README.md](README.md).
Then follow the shortest task-specific route:

- data, labels, partitions, manifests, or model recipes:
  [docs/data-contract.md](docs/data-contract.md) and
  [data/README.md](data/README.md);
- security, runtime authority, or claims:
  [docs/threat-model.md](docs/threat-model.md);
- model training or evaluation:
  [docs/roadmap.md](docs/roadmap.md) and the authoritative historical ledger,
  [reports/model-experiments.md](reports/model-experiments.md);
- branches, PRs, credentials, or release hygiene:
  [CONTRIBUTING.md](CONTRIBUTING.md).

## Mission and non-goals

morgott studies defense in depth for LLM applications and agents. It should
reduce successful direct jailbreaks, direct prompt injections, and indirect
prompt injections while preserving ordinary conversation, including legitimate
security and finance discussion.

The target is not general content moderation, toxicity detection, or proof
that an agent is jailbreak-proof. A harmful request without instruction
subversion is a separate label. Licensing is not an inclusion filter for this
research, but every source must keep license and provenance metadata.

## Current status

The active deliverable is the canonical data corpus. No model is approved for
blocking. The retained character n-gram detector and word n-gram
routing baseline are cheap controls, not intended models.

July 2026 external validation showed the historical ModernBERT ensemble does
not transfer: 0.00% TPR at 1% FPR on the public PromptShield split and 49.2%
single-mutation evasion on its own dev-test suite; its in-corpus FPR figure
describes text of 64 tokens or fewer. Never quote its retained FPR/recall pair
without those qualifiers or treat it as evidence a detector works.

Owner-authorized bounded exceptions (2026-07-28) produced a frozen-mmBERT
head, reduced-mixture LoRA gate, and full-mixture rank-8 LoRA seed.
Only the later 1,024-token update-17,000 candidate remains loadable through
`model-artifacts.json`; older artifacts are retained as provenance and none is
wired into blocking.
A later LP-FT comparison (2026-08-05) reduced long-task clean flags but
collapsed on PromptShield transfer and indirect-document
recall; it was rejected and its retained weights stay outside the registry.
Git commit `91e8c829c8b39c8ff37a6ca2479c8fc057168d39` is immutable provenance
for the July 2026 runs. Agents may autonomously run training and evaluation
when scientifically justified and within the repository's safety, privacy,
data, and evidence gates; no separate owner-approval step is required. Moving
an artifact into maintained inference still requires licensing, privacy, and
reproducibility review plus a `model-artifacts.json` registry entry. Learned
output remains advisory. Details and exact metrics: `reports/model-experiments.md`.
Historical reports may describe bounded owner authorizations; those statements
record scope at the time and are not current approval requirements.

## Non-negotiable security rules

Prediction is separated from authority; the diagram is in `README.md` and the
full claims in `docs/threat-model.md`.

- Learned output is advisory. `scan` and `cascade` always return
  `decision: allow`.
- A score may reduce privilege or escalate review; it never grants authority.
- Every side effect passes through the deterministic reference monitor.
- Provenance and sensitive-data labels come from trusted runtime metadata, not
  attacker-controlled text or model output.
- Schemas fail closed and capabilities name exact tools, argument shapes, and
  constrained values.
- Finance, cybersecurity, or other topic vocabulary is never itself a deny
  rule; harmful content without instruction subversion is a separate label.
- Keep credentials outside the planner and prefer short-lived, task-scoped
  capabilities in any future runtime integration.

The current policy code is a deterministic simulation, not a deployed agent,
credential broker, or network boundary.

## Data rules

One local data root (`data/`) and one versioned machine manifest
(`data/manifest.json`). Never recreate `processed/`, `expanded/`, a second
manifest, or a second data root. Exact counts and hashes belong only in the
manifest — never copy them into narrative documents.

The build fails closed on any missing source, access gate, schema, or pinned
digest, and publishes the manifest last:

```bash
uv run morgott data                 # full rebuild
uv run morgott data --routing-only  # partition-logic-only changes
```

The cloud source of truth is Azure Blob Storage (account `vulsightdata`,
container `morgott`).
Rebuild when a change can affect published corpus bytes, membership, labels,
provenance, partitions, manifest contents, or another data-contract output:

```bash
uv run morgott data
scripts/azsync.sh push  # only when published data changed
```

Pure structural refactors may skip the rebuild when focused tests cover the
unchanged behavior and the handoff explains why generated output is unaffected.
If equivalence is uncertain, rebuild.
Push to Azure only when rebuilt published data changes.

Auth, pull, and new-machine bootstrap are in `data/README.md`. The full
corpus, label, source, and split contracts are in `docs/data-contract.md`.

## Privacy and external providers

The corpus builder makes no remote provider calls. Do not send corpus text to
a provider merely because an API key exists; any remote-label experiment must
be explicit, bounded, development-only, and separately reviewed.

Never inspect, print, persist, or commit `.env`, API keys, Hugging Face
tokens, credentials, raw provider responses, or sensitive prompts.

## Verification

Before handing off any change:

```bash
make check
git diff --check
```

Changes meeting the data-impact criteria above also require the applicable full
or routing-only rebuild, followed by verification of manifest hashes, counts,
split invariants, and the quarantine summary.
Output-neutral refactors use focused equivalence tests instead.

## Maintained files

The file map is the "Repository map" in `README.md`. Two rules live here, not
there:

- `model-artifacts.json` is the sole registry for reviewed LFS research
  weights loadable by maintained inference; the rejected LP-FT comparison
  stays intentionally outside it.
- `experiments/` holds disposable or study-specific work that is never
  maintained model behavior.
