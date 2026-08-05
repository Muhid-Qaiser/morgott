# InjecAgent detector and task-conditioned reviewer evaluation

## Decision

Keep the maintained cascade and thresholds unchanged.

The response-only cascade misses too many low-cue base attacks, while adding the trusted user task improves ranking and recall without adding a clean flag.
The independently transferred task-conditioned threshold still fails the frozen aggregate, axis, attack-type, and attacker-minimum gates.
A lower threshold looks promising only in a post-hoc grid on this now-consumed source, so it is not promotion evidence.

## Source and scope

The source is the official [InjecAgent repository](https://github.com/uiuc-kang-lab/InjecAgent) at revision [`f19c9f2c79a41046eb13c03c51a24c567a8ffa07`](https://github.com/uiuc-kang-lab/InjecAgent/tree/f19c9f2c79a41046eb13c03c51a24c567a8ffa07).
Its [paper](https://arxiv.org/abs/2403.02691) defines injected tool-response tests across 17 user tools, 30 direct-harm attacker goals, and 32 data-stealing attacker goals.
The pinned release supplies 1,054 complete user-and-attacker combinations, each with a base and enhanced tool response.

The evaluation treats each clean tool-response template as benign and each base or enhanced injected tool response as indirect prompt injection.
It does not use InjecAgent's agent behavior, attack-success outcome, or model response as a detector label.
The source is public synthetic development data and is not a prospective final test.

## Privacy and leakage controls

Every email-address and phone-number match found by Morgott's local high-precision privacy screen was replaced by its typed placeholder before hashing, artifact publication, or provider review.
The sanitizer retained 2,125 distinct response texts across 1,054 matched clean/base/enhanced triplets and left zero recognized sensitive-text matches.
Artifacts retain hashes, labels, lineage, scores, and telemetry but no task text, response text, or raw provider response.

All 2,125 response texts passed normalized, audit-strict, and conservative near-overlap filtering against 1,073,330 canonical fit rows, 18,284 PromptShield train rows, and 22,092 retained matched-pair rows.
The later task-conditioned protocol separately checked all 17 trusted user tasks against the same 1,113,706 fit references and removed none.

Typed redaction changes source strings and may change scores, even though it preserves the instruction structure.
The public source includes only 17 distinct clean response templates, so its clean result cannot estimate production false-positive rate.

## Fixed response-only cascade

The first protocol scored every distinct sanitized response exactly once through the unchanged `untrusted_content` cascade selected before opening this panel.
It selected no prompt, model, provider, local gate, or threshold from InjecAgent.

| Variant | Restricted | Rows | Rate |
|---|---:|---:|---:|
| Unique clean | 0 | 17 | 0.00% |
| Base attack | 742 | 1,054 | 70.40% |
| Enhanced attack | 1,054 | 1,054 | 100.00% |
| Fully correct matched triplet | 742 | 1,054 | 70.40% |

The zero-of-17 clean result has an 18.43% Wilson 95% upper bound and is not a production-FPR estimate.
Base recall misses the frozen 80% requirement, and the fully correct matched-triplet rate misses its 75% requirement.
Enhanced attacks pass perfectly because their wrapper adds explicit instruction-boundary cues.

Base data-stealing recall is 477 of 544, or 87.68%, while direct-harm recall is 265 of 510, or 51.96%.
Physical Harm is the weakest base type at 69 of 170, or 40.59%.
The cascade misses 312 base attacks: 296 pass below the local `0.1` floor and 16 reach DeepSeek but clear its threshold.

All 2,125 assessments completed with 753 DeepSeek reviews and zero provider failures.
The fixed result rejects the response-only transfer gate and does not authorize lowering the maintained local floor.

## Fixed task-conditioned 0731 transfer

The follow-up sends one trusted-runtime JSON object containing the sanitized user task and sanitized tool response to DeepSeek V4 Flash 0731.
It reuses the task-conditioned prompt and transfers the independently selected StruQ threshold of `0.85` without selecting any value on InjecAgent.
It reviews all 17 clean templates and all 1,054 base attacks, while omitting enhanced attacks that the response-only cascade already catches perfectly.

| Metric | Response-only cascade | Task-conditioned 0731 at 0.85 |
|---|---:|---:|
| Clean flags | 0 / 17 | 0 / 17 |
| Base recall | 742 / 1,054, 70.40% | 825 / 1,054, 78.27% |
| Data-stealing recall | 87.68% | 88.42% |
| Direct-harm recall | 51.96% | 67.45% |
| Physical Harm recall | 40.59% | 71.18% |
| Financial Harm recall | 60.13% | 58.82% |
| Attack above its paired clean score | not comparable | 1,054 / 1,054, 100.00% |

Task context adds 83 catches, or 7.875 percentage points, without adding a clean flag.
It still misses the frozen 80% aggregate-recall gate, the 75% per-axis gate, the 65% per-type gate, and the 50% minimum-attacker gate.
The weakest attacker goal reaches only 3 of 17 users, or 17.65%.

All 1,071 reviews completed after 1,104 attempts with zero exhausted failures.
Thirty-one reviews needed a second attempt and one needed a third attempt.
The run used 440,174 input tokens and 8,051 output tokens.

## Read-only score-scale diagnosis

The sealed task-conditioned probabilities give every attack a higher score than its own paired clean response, but the transferred `0.85` cutoff is too high for several varied goals.
Clean probabilities range from `0.0001584` to `0.0293122`, while attack probabilities range from `0.0179862` to `0.9996646`.

A read-only post-hoc grid reaches 96.96% aggregate recall at `0.3`, with zero clean flags, 93.92% minimum-axis recall, 92.16% minimum-type recall, and 58.82% minimum-attacker recall.
At `0.1`, the same consumed grid reaches 98.96% aggregate recall with zero clean flags and 76.47% minimum-attacker recall.
These values diagnose a score-scale mismatch and do not authorize threshold selection, because InjecAgent outcomes were already open.

The next valid threshold decision needs a larger independently sourced clean tool-output denominator and a separate varied-goal attack source frozen before review.
It must also solve invocation strategy because this task-conditioned protocol remotely reviews every row.

## Immutable evidence

| Artifact | SHA-256 |
|---|---|
| [Response-only manifest](../artifacts/injecagent_detector_eval/manifest.json) | `660c664237bdd22a8a9c06af66bfd10c564157b9b88882e26fb7f0c7ebc0b322` |
| [Response-only panel](../artifacts/injecagent_detector_eval/panel.jsonl.gz) | `8691db8349362a09c7adec1f9b9d4ddb00314149c1ea2c1d2006d96ab10a72af` |
| [Response-only results](../artifacts/injecagent_detector_eval/results.jsonl.gz) | `f241ffbfc8670c7d30dad54f92a36214b2cdcf0073e10d64e784f82e9835cfac` |
| [Response-only summary](../artifacts/injecagent_detector_eval/summary.json) | `4b5b7c1f4f6a6ac1afe03576349633dc9064c00c8285f54b176e70d91f7c5b4e` |
| [Task-conditioned manifest](../artifacts/injecagent_task_reviewer_eval/manifest.json) | `3479c89888b1cfbbe99e633317c8157df01ef43b8537e92d1c631fee600f5109` |
| [Task-conditioned panel](../artifacts/injecagent_task_reviewer_eval/panel.jsonl.gz) | `071f37a2e44abf379e46bc47bed0b5536f13363e98786efe85254a4f753014cc` |
| [Task-conditioned results](../artifacts/injecagent_task_reviewer_eval/results.jsonl.gz) | `2082bd0f46e87ae4ac1f46ac629b326debdd9843ddd10ae62471899980948227` |
| [Task-conditioned summary](../artifacts/injecagent_task_reviewer_eval/summary.json) | `5695c5984d710ee9894af9959d00da2e7811a630542ce4ece8e1c4af674f435d` |
