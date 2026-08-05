# New data source scan

Date: 2026-08-04.

Research cutoff: 2026-08-04.

## Decision

Adopt no new traffic corpus or primary security benchmark from this scan.
SWE-chat remains the strongest public source for traffic-like coding-agent detector inputs, but its completed local diagnostic rejects the registered cascade and does not establish a production false-positive rate.
Agent-Diff remains the strongest exact-state substrate, while PIArena remains the stronger static clean-versus-attack text pairing.
No newly reviewed release beats those baselines on its relevant evidence target.

The only material new traffic candidate is `nmuendler/share-codex`, but its pinned provenance does not establish multiple contributors, contributor consent, a public-repository restriction, or stable participant and repository lineage, and its manifest records a nonzero secret-scan gate.
Defer it as an ungated Codex-heavy fallback rather than replacing SWE-chat; the [primary-source provenance audit](share-codex-provenance-audit-2026-08-04.md) permits only a quarantined local stress sample after a fresh privacy scan.

AgentDyn is the strongest dynamic security complement, but it is not new to Morgott, its fixed attack-text arm is already consumed development evidence, and its pinned action oracles have known conformance defects.
Keep it as a secondary stress test after an independent oracle preflight rather than adding it as the next primary substrate.

## Acceptance bar

A traffic source must expose real human-to-agent prompts, an explicit consent and privacy contract, stable session and repository lineage, a pinnable release, and enough independent contributors to measure restriction and review load.
Traffic data without instruction-subversion labels may estimate load, but it cannot establish a false-positive rate or supervise the router as benign data.

A security source must preserve the same legitimate task and stable objects across clean and attacked conditions, place provenance and authority outside attacker-controlled text, mediate every side effect, and deterministically check both legitimate completion and every extra state mutation.
A large static row count does not substitute for independent tasks, complete-state oracles, or a runnable pinned environment.

## Coding-agent traffic comparison

| Source and exact pin | Relevant evidence | Provenance and lineage | Decision |
| --- | --- | --- | --- |
| [SWE-chat `f66cca95`](https://huggingface.co/datasets/SALT-NLP/SWE-chat/tree/f66cca95b14caaa4177f7ed5eaa424608dadcffa) | 5,851 sessions, 2,692,480 events, and 205 repositories | Developers opted into Entire CLI tracking on public repositories; stable repository, session, checkpoint, commit, and turn identifiers; Presidio and TruffleHog redaction plus IRB exemption are documented in the [paper](https://arxiv.org/html/2604.20779#S5) | Completed local-only proxy rejects the registered cascade on restriction and review load; consume the panel and do not treat it as benign ground truth |
| [Real Pi `8c593252`](https://huggingface.co/datasets/MaxDevv/real-pi-coding-agent-traces-sessions/tree/8c593252ddad7dca08a0afc07896195fa73f2d6e) | 1,291 sessions aggregated from 21 upstream datasets | Explicit upload and privacy screening, but the four largest sources contribute 1,024 sessions and all manifest licenses are `other` | Retain only as the existing concentrated fallback |
| [share-codex `3d8b1397`](https://huggingface.co/datasets/nmuendler/share-codex/tree/3d8b1397c72dbfbf8b04f518064e2c99dde84ca0) | 4,333 sessions, 16,482 user turns, and 202,056 messages; 98.53% of prompt characters are Codex | Its [export manifest](https://huggingface.co/datasets/nmuendler/share-codex/blob/3d8b1397c72dbfbf8b04f518064e2c99dde84ca0/export_manifest.json) records 132 findings, 4,476 replacements, and 3,003 excluded rows, but also a nonzero secret-scan gate; contributor consent, public-repository restriction, participant identity, and canonical repository lineage remain unproven | Defer; permit only a quarantined local stress sample after a fresh privacy scan, never benign, FPR, representative-traffic, fitting, or evaluation evidence; see the [provenance audit](share-codex-provenance-audit-2026-08-04.md) |
| [TraceLab v0.0.1](https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.1) | 357,161 LLM rounds from 43 developers | The public sanitizer intentionally removes prompts, model text, tool inputs, and tool outputs | Reject as detector-input data; retain only as serving-workload evidence |
| [Zen Agentic Dataset `cdc75caa`](https://huggingface.co/datasets/zenlm/zen-agentic-dataset/tree/cdc75caa622c76b86d040a3423c58c9b4aa335b1) | Its root card claims 5.2 million entries, but the pinned tree contains documentation and no data blobs | The separate [dataset card](https://huggingface.co/datasets/zenlm/zen-agentic-dataset/blob/cdc75caa622c76b86d040a3423c58c9b4aa335b1/DATASET_CARD.md) calls the repository a placeholder, while availability, size, and license claims conflict across its files | Reject because it is not a runnable public dataset |

The bounded official-source search found no public, multi-contributor real coding-chat release that exceeds SWE-chat while preserving its explicit opt-in contract and stable session and repository lineage.
RECAP collected consented classroom interaction data, but its ethics statement says individual student data are not released, so it is a collection design rather than a runnable source ([paper](https://arxiv.org/html/2605.01104)).

## Matched transaction and agent-security comparison

| Source and exact pin | Pairing and scale | Execution, oracle, and authority fit | Decision |
| --- | --- | --- | --- |
| [PIArena data `e9f56791`](https://huggingface.co/datasets/sleeepeer/PIArena/tree/e9f56791974132a803632dc4b5fc18f3de90e91b) and [code `c39fd88e`](https://github.com/sleeepeer/PIArena/tree/c39fd88e733493242a8ea6bdbc824ad30245bcf7) | The same clean context is paired with four official attack constructions | Strong static detector pairing, but no live state, typed authority, or complete mutation oracle | Retain as the static matched benchmark |
| [Agent-Diff `3bb9c407`](https://github.com/agent-diff-bench/agent-diff/tree/3bb9c40707df23d89e5dbc0e40c424ba38c69ff8) | 224 tasks across Box, Calendar, Linear, and Slack | Stable seeded objects, isolated replicas, deterministic assertions, and complete before-and-after database diff machinery; Morgott still must add the one-field attack overlay and trusted authority sidecar | Retain as the primary exact-state substrate |
| [AgentDyn `5353cf76`](https://github.com/SaFo-Lab/AgentDyn/tree/5353cf7615b135cace8d07c8f12dac53a16b6db3) | 60 manually designed dynamic user tasks and 560 injection cases across Shopping, GitHub, and Daily Life | Typed function-call ground truth and utility and security checks improve dynamic planning and over-defense coverage, but released GitHub and Daily Life ground truths disagree with their available tool schemas ([GitHub tasks](https://github.com/SaFo-Lab/AgentDyn/blob/5353cf7615b135cace8d07c8f12dac53a16b6db3/src/agentdojo/default_suites/v1/github/injection_tasks.py), [Daily Life tasks](https://github.com/SaFo-Lab/AgentDyn/blob/5353cf7615b135cace8d07c8f12dac53a16b6db3/src/agentdojo/default_suites/v1/dailylife/injection_tasks.py)) | Continue only as an already-audited secondary stress test after oracle preflight |
| [AgentSecBench v1.0.0 `a125aa44`](https://github.com/zheyuanhu2-sketch/agentsecbench/tree/a125aa446854a12d2741d1c7625bc91887d82955) | 20 normal tasks and 10 attack tasks are not exact clean and attacked twins | Runtime-owned provenance, typed actions, capability checks, and pre-execution mediation are useful design references, but the in-memory evaluator checks named required or forbidden actions rather than a complete state diff ([evaluator](https://github.com/zheyuanhu2-sketch/agentsecbench/blob/a125aa446854a12d2741d1c7625bc91887d82955/src/agentsecbench/evaluator.py)) | Defer as a reference-monitor design source, not empirical replacement |
| [Coding-agent injection benchmark `5c5e392c`](https://github.com/sebastianripa/coding-agent-injection-benchmark/tree/5c5e392c904430f90bb8bdf2884e5bb2a325374c) | Six coding tasks each run clean and under source-comment, rules-file, and tool-metadata poisoning | The matched-condition pattern is useful, but only two tasks have full Docker task-success tests, tool calls mutate the workspace without a trusted monitor, and the grader does not reject every extra mutation ([variants](https://github.com/sebastianripa/coding-agent-injection-benchmark/blob/5c5e392c904430f90bb8bdf2884e5bb2a325374c/bench/variants.py), [grading](https://github.com/sebastianripa/coding-agent-injection-benchmark/blob/5c5e392c904430f90bb8bdf2884e5bb2a325374c/bench/grading.py)) | Defer only as a coding-specific transfer pattern |
| [SafeClawBench `e6c29204`](https://huggingface.co/datasets/sairights/safeclawbench/tree/e6c29204c24a5910600aae854baae57a51586655) | 600 synthetic adversarial tasks across six attack families with no matched clean arm | It includes semantic, audit, and deterministic sandbox endpoints, but cannot measure matched utility or over-defense | Defer as an attack-endpoint disagreement diagnostic only |
| [NetInjectBench v1](https://arxiv.org/abs/2607.10490v1) | The paper describes 130 benign and attacked scenarios | The official record exposes no versioned scenarios, runner, evaluator, or result artifact, and the paper promises release only after acceptance ([reproducibility statement](https://arxiv.org/html/2607.10490v1#S7)) | Reject until an official immutable artifact exists |

GitInject provides realistic ephemeral GitHub and GitLab workflows, but its [pinned release tree](https://github.com/ceferisbarov/GitInject/tree/c74c255b4ed03469f563cec27b68fc253b94cb4c) has only 11 malicious and 8 benign scenarios, needs authenticated external accounts, and sometimes relies on model-judged outcomes.
Defer it as a focused live CI/CD integration smoke test, not detector evaluation or primary matched evidence.

ClawsBench publishes 7,834 traces from 44 tasks, but its pinned [dataset card](https://huggingface.co/datasets/benchflow/ClawsBench/tree/e7c45cc9ff486502176267c1294ac5809cf0700a) says the task definitions and Dockerized environments will be added later.
Reject it as a runnable benchmark at this cutoff.

## Adopt, defer, and reject

**Adopt:** no new source.
The SWE-chat local-only traffic-load diagnostic is complete, and Agent-Diff remains the preferred exact-state clean-versus-attack substrate.

**Defer:** reconsider share-codex only as a separately labelled concentrated stress source after contributor, repository, and consent provenance are clarified.
Keep AgentDyn, AgentSecBench, the coding-agent injection benchmark, SafeClawBench, and GitInject as separately labelled secondary diagnostics for the narrow gaps described above.

**Reject:** do not integrate Zen, TraceLab, RECAP, ClawsBench, or paper-only NetInjectBench as Morgott detector data or as the primary security benchmark.
Do not infer independent realism from synthetic Cartesian row counts or treat static trajectory labels as transaction outcomes.

## Next action

Do not add another corpus adapter, runner, or model-fitting source from this scan.
For traffic, retain the completed SWE-chat result as consumed development evidence and do not tune the registered cascade on it.
Do not add a second traffic corpus until a materially different architecture and independent evaluation question justify one.

For security, continue with a fresh Agent-Diff task whose clean and attacked replicas differ in exactly one untrusted content field, carry identical trusted object and authority metadata, mediate every proposed mutation, and compare the complete database diff against one exact utility and no-extra-mutation oracle.
Do not reuse AgentDyn's already opened fixed attack-text arm as prospective evidence, and do not trust its released action outcomes without an independent oracle preflight.

## Claim boundary

This report establishes public artifact availability, provenance documentation, and contract fit at the cutoff.
It does not validate author-reported security results, prove that no unindexed or private source exists, or authorize another encoder run.
