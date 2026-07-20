# HackAPrompt and Yaklang exploratory-source audit

Generated: 2026-07-20

## Decision

HackAPrompt is **not added to the dataset build**. Its Hugging Face repository
remains auto-gated and requires sharing contact information; an unauthenticated
request for the pinned Parquet returned HTTP 401. This audit did not use the
configured Hugging Face token, accept the gate, or fetch a mirror. The MIT
dataset-card metadata does not make the access step ungated.

The Yaklang `llm-prompt-injection` skill is integrated only as a pinned taxonomy
and future scenario-design reference. It is not installed as an agent skill,
vendored, treated as a labelled corpus, or used to generate training rows.

The pinned provenance and the decision not to activate HackAPrompt are recorded
in this audit; no adapter or source data is retained locally.

## HackAPrompt access and provenance

- Repository: [`hackaprompt/hackaprompt-dataset`](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset),
  pinned at `25b87fbedfb86840abaf8cd09af7a029208a971a`.
- Access observed: Hugging Face `gated: auto`, with contact-sharing acceptance;
  the Parquet is not available anonymously.
- Dataset-card license metadata: MIT.
- `README.md`: 5,555 bytes, Git blob
  `63bdb779825fc7b25fa8a4a347b1a0e0dfb2a89b`, SHA-256
  `627ac0ec7ebfcb5baf93bbc9b80c72a73443846debf06196803aee1138486d2a`.
- `hackaprompt.parquet`: 150,419,795 bytes, Git blob
  `82492aaac92010b5383169ea1d8ba9f403368f8c`. Its content SHA-256 is
  intentionally unknown because reading it requires accepting the gate.
- Primary paper: [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.302/).
  The 4,580,639-byte PDF has SHA-256
  `deaebfb272b7544e5d906ddd58435f73626f47743d83d153f41d4af6a2cc12de`.
  It reports 589,331 anonymous playground entries and 58,257 prompts from 7,332
  formal submissions across three target models.

The card lists `level`, `user_input`, `prompt`, `completion`, `model`,
`expected_completion`, `token_count`, `correct`, `error`, `score`, `dataset`,
and `timestamp`. This is a card-level audit only; the gated Parquet schema and
content hashes have not been verified.

## Labels, fields, and leakage

HackAPrompt records competition submissions. Therefore every valid
`user_input` is an attack **attempt by collection context**, even when
`correct=false`. `correct` means that one target model produced the expected
challenge completion. It is target- and template-specific attack-success
metadata—not a benign/injection label. Unsuccessful attempts must never be
converted into training negatives.

Only `user_input` is eligible as detector text. The following are excluded:

- `prompt`: contains the user input embedded in the challenge/system template,
  creating target-template shortcuts and exposing privileged prompt material;
- `completion`: leaks target-model behavior and success cues; and
- `expected_completion`: leaks the fixed challenge target.

Token count, score, timestamp, model, level, and source partition are metadata,
not text features. A future report should show both all-attempt recall and the
`correct=true` subset; neither alone establishes general jailbreak recall.

The paper documents exact repetition, spam, random-character probes, and
uncurated offensive material. A future authorized local audit must:

1. hash-pin the gated file and verify the real schema before projection;
2. scan/redact likely PII and secrets locally despite the source's anonymity
   statement, and retain toxicity as an independent attribute;
3. exact- and near-deduplicate normalized `user_input` against every active
   train/evaluation source, especially deepset and Tensor Trust;
4. aggregate duplicate user inputs across levels, target models, and source
   partitions while retaining per-target success counts; and
5. keep the source evaluation-only initially.

There is no official train/test split. If training is later justified, hold out
whole challenge levels, group all text/template mutations, and remove
cross-level text overlap after splitting. The paper describes participant
linkage in formal submissions, but the public card does not list a participant
identifier; participant-group isolation cannot be claimed until the real schema
is inspected.

## Marginal value versus current sources

| Family | Existing evidence | HackAPrompt/Yaklang decision |
|---|---|---|
| Direct override and context termination | deepset, ToxicChat, Tensor Trust | HackAPrompt may add large-scale human optimization, but is gated and highly challenge-specific. |
| Prompt extraction/hijacking | Tensor Trust | Do not duplicate Yaklang's canonical examples. |
| Encoding and obfuscation | grouped multi-turn corpus; JailbreaksOverTime | HackAPrompt levels may add target-specific variants if later accessed; keep source-held-out first. |
| Indirect email/table/code injection | BIPIA | Yaklang's RAG/web/email examples are taxonomy references, not new labels. |
| Tool-description/tool-output injection | no stateful benchmark | Convert to typed AgentDojo/reference-monitor scenarios. |
| Cross-tool/MCP and Markdown exfiltration | policy simulation only | High-value stateful containment gap; detector text recall is not the release metric. |
| Harmful fiction, direct tool requests, SQL injection | HarmBench/Do-Not-Answer are non-injection controls | Do not label these positive unless they actually subvert instruction hierarchy. |

## Yaklang reference quality

The pinned source is
[`yaklang/hack-skills`](https://github.com/yaklang/hack-skills/tree/c9a4b9ee8645eb60763eb4eef172f1ecb0a5b3e8/skills/llm-prompt-injection)
at `c9a4b9ee8645eb60763eb4eef172f1ecb0a5b3e8`, MIT licensed:

- `SKILL.md`: SHA-256
  `f145e6ba1d394c91f69d5ce0a23cf49617a9b5bdbee29ffde848b05a265abbad`;
- `JAILBREAK_PATTERNS.md`: SHA-256
  `3713405029c3168d8b751ff8508050c1dc4d7b1d892a1031cfc25d520fcc4b11`;
- repository `LICENSE`: SHA-256
  `92b640638bf4ca37756dee0284c84c3600e8d17a7fa61ac8c179d79bdb3ef735`.

Its useful contribution is an attack-surface checklist: direct/indirect
provenance, RAG/web/email ingestion, tool descriptions and outputs, cross-MCP
flows, Markdown exfiltration, encoding, and multi-turn splitting. It is not a
validated taxonomy or generator benchmark: examples are few and canonical,
model-specific claims are unsourced and time-sensitive, no labels/groups/
success measurements are supplied, and some sections conflate prompt injection
with harmful requests, direct tool misuse, or ordinary SQL injection.

Consequently, do not train on the examples or ask an agent to expand them
freely. For future authorized scenario generation, define outcome-first specs
(unauthorized action, secret egress, or utility failure), preserve provenance,
group every mutation by base scenario/template/seed, and keep generated rows in
development only. The reference monitor—not a detector score—must determine
whether a stateful scenario causes an unauthorized side effect.
