# WildChat model-only weak-label pilot

This experiment broadens the normal-chat training distribution without treating
WildChat source toxicity labels as injection labels. No human labelers are
available. Model judgments are therefore weak supervision only: they may add
conservative training negatives, but they never become threshold-calibration or
production-FPR ground truth.

## Protocol

The first pilot has **5,000 sampled rows**, not 5,000 accepted negatives. Those
counts diverge after PII/secret removal, overlap filtering, unavailable calls,
judge disagreement, uncertainty, and the third-family audit. The first training
comparison is zero weak rows versus every accepted row from the 5k sample. The
sampled pool may grow only if that predeclared development comparison improves
recall without worsening a normal-chat subgroup; 5k/20k/50k are later
**accepted-negative** targets, not assumptions about this pilot.

Sampling:

- Pin `allenai/WildChat-1M` at commit
  `7d6490e462285cf85d91eabea0f9a954fbddcd1f`. Download only exact-revision
  Parquet shards 0, 6, and 13 (756,333,381 bytes rather than the 3.36 GB split),
  and verify each pinned LFS size and SHA-256. Each raw shard exists only in an
  ephemeral temporary file while it is scanned; only a deterministic,
  hash-ranked 10k locally redacted candidate cache per shard persists.
- Select one deterministic user turn per conversation. Discard assistant turns
  and all country, IP hash, header, timestamp, model, state, and moderation
  metadata. Retain only local text plus safe hashes/strata needed for lineage.
- Drop or redact likely local PII/secrets before persistence or provider use.
  Block exact and SimHash-near overlap with core training, JailbreaksOverTime,
  and all frozen evaluation rows. Deduplicate the pilot itself.
- Stratify across language, length, source toxicity, local coarse topic, and
  security-trigger terms. Source toxicity remains an independent attribute;
  this pinned public sample has no source-toxicity variation, so the stratum is
  `unavailable` rather than an invented all-benign/false group.

Weak labeling:

- Primary families: `mistralai/mistral-small-2603` and `qwen/qwen3.5-27b`. Third
  audit family:
  `anthropic/claude-sonnet-4.6`. Model IDs are configurable but the families
  must remain independent.
- Each judgment is one strict-JSON call at temperature 0. Requests require ZDR,
  deny provider data collection and provider fallback, require parameter
  support, disable/exclude reasoning tokens, and enable no tools, plugins,
  retry, repair, or ReAct loop.
- Labels are `benign`, `injection_or_jailbreak`,
  `harmful_non_injection`, or `uncertain`; toxicity is independent.
- A row is provisionally benign only when both primary judges return
  high-confidence benign. Every detector-hard provisional benign plus a
  deterministic 10% of the rest receives a blind third-family judgment. An
  audited row survives only on high-confidence third-family agreement.
- Unavailable, malformed, uncertain, low-confidence, and disagreement rows are
  discarded from training. They remain aggregate diagnostic counts.

Privacy and provenance:

- Ignored local sample files contain the redacted text required for training.
  Versioned reports contain counts/hashes only.
- The provider journal stores sample/request/response hashes, normalized enums,
  confidence, toxicity, requested/returned model and selected provider IDs,
  agreement state, token/cost/latency, and unavailability category. It never
  stores prompts, raw responses, exception messages, credentials, or `.env`.
- Judge agreement and third-model disagreement are consistency measurements,
  not error rates. No result from this experiment is a production FPR.

## Evaluation and scaling

Weak negatives receive a fixed aggregate source-weight budget so 5k/20k/50k
rows cannot dominate the original 311 direct positives. The evaluator validates
the accepted rows, selects each deterministic ablation, applies the fixed source
weights, and trains each available candidate directly.

The user's requested ToxicChat, deepset, multi-turn, BIPIA, Tensor Trust,
NotInject, position-stress, and non-overlapping JailbreaksOverTime comparisons
must use identical threshold selection and fixed rows. Once these results decide
whether to scale, they are repeated development benchmarks—not untouched final
tests. There is no production blocking exit gate without independent labels.

All commands are offline/dry-run unless their explicit network execution flag is
present. Public shard downloads have three bounded attempts for transient
failures and validate content before processing; model judgments still make
exactly one attempt.

```bash
# Print the bounded sampling plan; no network.
PYTHONPATH=src python experiments/wildchat_pseudolabel/sample.py

# Download three pinned shards through ephemeral files, build/reuse bounded
# redacted caches, filter, score, and write the ignored local 5k sample plus
# reports/wildchat-sample.json.
PYTHONPATH=src python experiments/wildchat_pseudolabel/sample.py --execute-fetch

# Validate the sample and print the provider-call plan; no provider call.
python experiments/wildchat_pseudolabel/label.py

# Recommended paid preflight. Use distinct ignored outputs so the full run has
# a different immutable run fingerprint and journal.
python experiments/wildchat_pseudolabel/label.py \
  --offset 40 --limit 20 --workers 8 --execute \
  --journal experiments/wildchat_pseudolabel/outputs/smoke3_journal.jsonl \
  --accepted experiments/wildchat_pseudolabel/outputs/smoke3_accepted.jsonl \
  --report experiments/wildchat_pseudolabel/outputs/smoke3_report.json

# Full pilot after the preflight passes. Completed/unavailable journal entries
# are never retried on resume.
python experiments/wildchat_pseudolabel/label.py --workers 16 --execute

# Validate, select, weight, and train every currently available
# 0/pilot/5k/20k/50k accepted-negative candidate. Weak rows enter fitting only;
# original grouped validation alone selects each precision-first operating point.
PYTHONPATH=src python experiments/wildchat_pseudolabel/evaluate.py

python -m unittest discover -s experiments/wildchat_pseudolabel -v
```

The core editable install does not add OpenRouter or Torch dependencies. These
scripts use the Python standard library for network requests, the existing
scikit-learn/joblib artifact for local scoring, and silently consume
`OPENROUTER_API_KEY` from the process environment or ignored `.env`.

## Pilot result

The 5,000-row run accepted 2,430 conservative weak negatives. At the
minimum-85%-validation-precision profile, adding them left grouped validation
at 34 true and 4 false signals but reduced multi-turn recall from 908/4,136 to
291/4,136 and Tensor Trust embedded recall from 841/1,346 to 825/1,346. Direct
and indirect attack macro recall both decreased. The predeclared scale gate
therefore returned `stop`; do not run 20k/50k collection under this recipe.
See [`reports/wildchat-ablation.md`](../../reports/wildchat-ablation.md).
