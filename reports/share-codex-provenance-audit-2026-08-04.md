# share-codex provenance audit

Date: 2026-08-04.

Dataset: [`nmuendler/share-codex` at `3d8b1397c72dbfbf8b04f518064e2c99dde84ca0`](https://huggingface.co/datasets/nmuendler/share-codex/tree/3d8b1397c72dbfbf8b04f518064e2c99dde84ca0).

## Decision

**Defer.**

Do not add this release to the canonical corpus or use it as benign supervision, an FPR denominator, or population-representative traffic evidence.

It is technically usable only as a quarantined, local-only, artifact-specific alarm-load stress sample after a fresh local privacy scan.
That narrower use could measure what Morgott would route or review on these exact sessions, but it could not establish benignity, false-positive rate, independent-user behavior, or public-repository traffic.

The blockers are missing contributor and consent provenance, no enforced public-repository restriction, no stable participant identity, weak repository lineage, and a pinned manifest whose recorded final secret-scan gate is nonzero.

## Audit scope

This audit used only the pinned dataset card and manifest, Hugging Face repository metadata and commit history, and the uploader's first-party source repository.
It did not download or inspect `train.jsonl`, any conversation row, or any prompt text, and it made no provider call.

## Evidence

### Population and consent

The pinned release contains 4,333 sessions, 16,482 user turns, and 202,056 messages, of which 4,314 sessions are Codex and 19 are Claude Code ([card](https://huggingface.co/datasets/nmuendler/share-codex/blob/3d8b1397c72dbfbf8b04f518064e2c99dde84ca0/README.md), [manifest](https://huggingface.co/datasets/nmuendler/share-codex/blob/3d8b1397c72dbfbf8b04f518064e2c99dde84ca0/export_manifest.json)).

All four Hugging Face commits have the single publishing account `nmuendler` as author, which establishes one publisher but not the number of human contributors ([commit history API](https://huggingface.co/api/datasets/nmuendler/share-codex/commits/main)).

The final export read 423 pre-existing output rows, collected 6,918 new local rows, read no rows from the Hub, and retained 4,333 rows after a working-directory review removed 3,003 rows under 37 excluded prefixes ([manifest](https://huggingface.co/datasets/nmuendler/share-codex/blob/3d8b1397c72dbfbf8b04f518064e2c99dde84ca0/export_manifest.json)).

The first-party tool requires an operator to run collection and an authenticated upload, so publication itself is an uploader opt-in action ([uploader README](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/README.md), [upload command](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/src/upload/__main__.py)).

The uploader's licensing note assumes that sessions contain the operator's own work and that the operator has the rights to publish them, but the collector does not verify that assumption or record contributor consent ([licensing note](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/docs/licensing.md)).

The artifact therefore does not establish whether its prompts came from one human or several, or whether every prompt author, repository owner, coworker, client, or third party represented in tool output consented.

### Public-repository restriction

No public-repository restriction is documented in the card or manifest.

The collector includes local session files by working directory, with optional exact-path and regular-expression filters, while the review tool lets the operator exclude selected working-directory subtrees ([collector](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/src/collector/codex.py), [CWD review](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/src/review_cwds/__main__.py)).

Neither path performs or records a repository-visibility check.
The 37 excluded prefixes are not classified as public, private, sensitive, or irrelevant in the published manifest.

The retained rows therefore cannot be described as public-repository-only traffic.

### Lineage

Session lineage is the strongest available axis.
Codex rows use the source session ID as both row ID and `metadata.session_id`, while Claude rows use `claude:<session-id>` as row ID and retain the source session ID in metadata ([Codex exporter](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/src/collector/codex.py), [Claude exporter](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/src/collector/claude.py)).

Incremental assembly merges by that row ID, so a session can be kept together and duplicate session IDs can be detected ([merge logic](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/src/collector/dataset.py)).

The exporter can retain a redacted working directory, source-file path, timestamp, raw `git` metadata, source product, entry point, and model.
Working directory and raw git metadata are only repository-lineage surrogates because there is no documented canonical repository identity across renamed directories, multiple checkouts, or machines.

There is no stable participant identifier or contributor count.
The single Hub publisher, one `$HOME` layout, and absence of a Hub merge are consistent with one operator's accumulated local sessions, but they do not prove it because the exporter can preserve pre-existing rows from other machines ([incremental workflow](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/README.md)).

### Privacy and redaction

The exporter replaces the local home prefix, applies regular expressions for common credentials and private keys, omits internal prompts, developer messages, reasoning, and turn context by default, then uses gitleaks and trufflehog findings for exact-value replacement ([regex redaction](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/src/collector/redaction.py), [scanner replacement](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/src/scan/redact.py)).

The manifest records final-row redaction, 132 scanner findings, 4,476 replacements, unlimited tool-output length, and a manual CWD review that removed 3,003 rows ([manifest](https://huggingface.co/datasets/nmuendler/share-codex/blob/3d8b1397c72dbfbf8b04f518064e2c99dde84ca0/export_manifest.json)).

The same manifest records `secret_scanning.gate_exit_code: 1`, not a successful final gate.
The upload command scans by default but also supports `--skip-secret-scan`, and it does not publish an attestation proving which path produced a given Hub commit ([upload command](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/src/upload/__main__.py)).

The uploader explicitly calls redaction best effort, warns that secrets and private data can remain in local tool output, and requires manual inspection before publication ([warning](https://github.com/nielstron/share-codex/blob/e952d45055a7864f3d9770010958f2df88c4081b/README.md)).

The public artifact does not attest that a row-level manual privacy review occurred or that the post-CWD-review content passed a fresh scanner gate.

### Licensing metadata

The dataset card declares CC-BY-4.0.
Per-session derivation records 3,622 rows as `not_found` and 56 as `unknown`, or 84.88% combined, while the remaining rows carry detected or declared SPDX-like values ([card](https://huggingface.co/datasets/nmuendler/share-codex/blob/3d8b1397c72dbfbf8b04f518064e2c99dde84ca0/README.md), [manifest](https://huggingface.co/datasets/nmuendler/share-codex/blob/3d8b1397c72dbfbf8b04f518064e2c99dde84ca0/export_manifest.json)).

These values are provenance metadata only.
They do not establish consent, repository visibility, privacy, label correctness, or benignity, and they are not used as the decision gate here.

## Diagnostic fit

The corpus is large enough to exercise message-size, turn-count, tool-traffic, latency, throughput, and advisory alarm-volume paths.
It is also concentrated: 3,792 of 4,333 sessions use `codex:exec`, 98.53% of user-prompt characters are Codex, and `.lean` is the largest edited-path type at 28.24% ([card](https://huggingface.co/datasets/nmuendler/share-codex/blob/3d8b1397c72dbfbf8b04f518064e2c99dde84ca0/README.md)).

Those distributions make it useful as a Codex-heavy stress sample but weak evidence for a broad real-user traffic population.

If a future decision admits it, use the exact pinned revision and the data object's published SHA-256 `37a024018c5f92f789abd7a5d6cc11c95d255ff1580c1266ee06dfcddab2ba97`, quarantine it outside supervised corpus roles, scan secrets and PII locally before parsing, group by session ID, persist only aggregate metrics or hashes, and make no network or provider call ([manifest](https://huggingface.co/datasets/nmuendler/share-codex/blob/3d8b1397c72dbfbf8b04f518064e2c99dde84ca0/export_manifest.json)).

An unflagged row must never become a benign label, and alarm volume must never be reported as FPR.

## Reconsideration gates

Reconsider adoption only after a pinned replacement release provides all of the following:

1. An attested contributor count and explicit consent or ownership basis for every contributed session.
2. An enforced and recorded public-repository or otherwise share-authorized inclusion rule.
3. Stable participant and canonical repository lineage, while preserving the existing session identity.
4. A manifest-bound exporter revision and a successful privacy gate recorded after the final CWD and row filtering.
5. A documented manual privacy-review procedure or an equivalent auditable release check.

Until then, SWE-chat remains the better-governed traffic source, and share-codex remains a deferred fallback rather than canonical data.
