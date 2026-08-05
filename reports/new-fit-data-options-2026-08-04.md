# New fit-data options for the next LP-FT run

Date: 2026-08-04.

Research cutoff: 2026-08-04.

## Decision

Do not split the already-consumed SWE-rebench panel into fitting data by default.
That would add new optimizer rows, but it would erase the cleanest evidence for the long coding-task workload that motivated the change.

The better next mixture uses four new fit sources or constructions:

1. Use SWE-rebench V2 problem statements as repository-grouped legitimate coding-task negatives and construct one deterministic known-span injected twin per retained task.
2. Use ToolSandbox only as an executable substrate for clean and one-field-attacked tool-result twins that pass exact utility, exposure, attack-success, and no-extra-state-delta gates.
3. Use MultiDoc2Dial documents for natural long-document clean and known-span attacked twins, grouped by source document and domain.
4. Use the WMT 2024 prompt-injection test suite for multilingual clean and attack families, grouped by the underlying TruthfulQA question across every language and attack format.

These additions would be genuinely new to the fitted weights.
They should be source-balanced rather than concatenated at raw frequency.

## Candidate sources

| Source and pin | Usable evidence | Scale before Morgott exclusions | Split and leakage rule | Decision |
| --- | --- | ---: | --- | --- |
| [SWE-rebench V2 `475dd5e8`](https://huggingface.co/datasets/nebius/SWE-rebench-V2/tree/475dd5e8703bb5fb22dd3c60b5d038b019eba1e0) | Real issue and pull-request tasks with repository, commit, timestamp, programming language, repository-license, executable-test, and installation lineage | 32,079 train rows, 3,600-plus repositories, and 20 programming languages according to the [official paper](https://arxiv.org/abs/2602.23866) and [dataset card](https://huggingface.co/datasets/nebius/SWE-rebench-V2); CC-BY-4.0 | Exclude every repository and exact or near text already present in SWE-rebench, SWE-bench Verified, SWE-chat, canonical fitting data, or prospective evaluation; then group all clean and injected variants by repository and original instance | Adopt after metadata-only and text-overlap preflight |
| [ToolSandbox `165848b9`](https://github.com/apple/ToolSandbox/tree/165848b9a78cead7ca7fe7c89c688b58e6501219) | Stateful tool execution, complete world-state snapshots at every turn, and ordered intermediate and final milestones | The [paper](https://arxiv.org/abs/2408.04682) reports 1,032 cases and 34 tools; the [official repository](https://github.com/apple/ToolSandbox) describes settings, contacts, messaging, reminders, typed tools, and dynamic milestone evaluation; the [custom Apple license](https://github.com/apple/ToolSandbox/blob/165848b9a78cead7ca7fe7c89c688b58e6501219/LICENSE) permits use, modification, and redistribution | Clone the same initial state, alter exactly one untrusted tool-return field, and retain a pair only if the clean arm completes, the agent consumes the attacked field, the unmonitored arm performs the exact attacker action, and the complete state delta contains nothing else | Adopt as a bounded construction substrate, not as pre-labelled injection data |
| [MultiDoc2Dial `1108a969`](https://huggingface.co/datasets/IBM/multidoc2dial/tree/1108a969d076f04c7367f0c2427d1c5d6d6bdaa0) | Natural government-service documents with document IDs, domains, structural spans, grounded dialogue turns, and existing official partitions | 488 documents and 4,796 dialogues across four domains in the [official paper](https://aclanthology.org/2021.emnlp-main.498/); Apache-2.0 on the [dataset card](https://huggingface.co/datasets/IBM/multidoc2dial) | Insert a locally registered payload into one exact document span, keep its unchanged clean twin, group every placement and payload variant by document, and hold out one complete domain or a document-grouped test partition | Adopt for a modest natural long-document pair set |
| [WMT adversarial MT prompt injection `0d2107ad`](https://github.com/Avmb/adversarial_MT_prompt_injection/tree/0d2107adc2515193a39919b672979223b67dbc7c) | Clean and five attacked subtasks over the same 817 TruthfulQA questions and 11 language pairs, including source-language and English attack variants for Czech-to-Ukrainian and Japanese-to-Chinese | The [official README](https://github.com/Avmb/adversarial_MT_prompt_injection/blob/0d2107adc2515193a39919b672979223b67dbc7c/README.md) reports 4,902 rows per English-to-any language pair and 8,987 rows for each of Czech-to-Ukrainian and Japanese-to-Chinese, under Apache-2.0 | Group the same TruthfulQA question across all language pairs and all six subtasks; never split translations or attack templates independently; exclude overlap with the consumed CyberSecEval multilingual scenarios | Adopt as paired multilingual fit data with an explicit synthetic-template slice |

## Deferred alternatives

[SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith) currently exposes 59,136 rows and 222 repository values, while its narrative describes 50,137 tasks from 128 repositories.
Defer it until a pinned metadata audit resolves that version mismatch, and prefer SWE-rebench V2 because SWE-smith issue text is procedurally generated.

[R2E-Gym V1](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-V1) exposes 8,101 Apache-2.0 training rows with repository, commit, problem-statement, parsed-commit, execution-result, and modified-file lineage.
It is a defensible fallback after repository overlap filtering, but its task text is generated from commits and is weaker than SWE-rebench V2 for realistic coding-task negatives.

[AppWorld](https://github.com/StonyBrookNLP/appworld) has 750 tasks, nine apps, 457 APIs, and state-based tests that can detect collateral changes.
Its task assets are intentionally protected against training contamination and derivatives must remain encrypted, so keep it for evaluation or use its engine only for independently authored tasks.

## Why not fit the current SWE-rebench panel

The existing panel contains 20,762 fit-disjoint tasks from 3,435 repositories, but the complete source has already influenced architecture decisions through its aggregate restriction and review-load results.
It may be reclassified as fitting data if the owner accepts losing it as evaluation evidence, but a random row split would leak repository style and repeated issue families.
If it must be reused, split by repository first, group every clean and injected twin together, and call the held-out portion repeated development data rather than a prospective test.

One clean and one injected twin for every current row would add at most 41,524 raw rows before overlap and privacy exclusions.
Against the prepared 1,109,885-row LP-FT mixture, that gross construction would produce at most 1,151,409 fitted rows, a 3.74% increase.
Its value would come from matched same-format contrast and longer negatives, not from corpus size.

## Recommended new-data mixture

Use the 32,079 SWE-rebench V2 tasks only after repository, time, privacy, exact-overlap, strict-overlap, and near-overlap filtering.
The pinned release contains one shard, `data/train-00000-of-00001.parquet`, with fields `base_commit`, `created_at`, `image_name`, `instance_id`, `interface`, `language`, `license`, `patch`, `pr_description`, `problem_statement`, `repo`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `install_config`, and `meta`.
Project only the non-empty `problem_statement` as detector text and retain `repo`, `instance_id`, `base_commit`, `created_at`, `language`, and `license` as trusted lineage.
For every retained clean problem statement, choose one payload family and one insertion boundary by a stable hash of `instance_id`, insert exactly one registered training-only payload, and record its exact span.
Do not expose patches, tests, installation instructions, or LLM metadata to the detector.
This construction has a gross ceiling of 64,158 new rows before exclusions.

Add a document-balanced MultiDoc2Dial slice and a question-grouped WMT slice.
Add ToolSandbox pairs only after the same causal gate that rejected the tau2 pairs succeeds.
Do not count failed or non-vulnerable ToolSandbox attacks as positive training examples.

The exact new total cannot be stated responsibly until preflight resolves cross-source duplicates, overlapping repositories, empty inputs, privacy findings, and pair eligibility.
The gross SWE-rebench V2 ceiling alone would take the prepared mixture from 1,109,885 to 1,174,043 rows, a 5.78% increase.
The final recipe should cap or weight each new source and report effective examples and optimizer updates, because raw WMT template multiplicity and coding-task volume would otherwise overwhelm the smaller executable and natural-document pair sets.

## Frozen split contract

- Hash the complete source revisions before reading outcome metrics.
- Derive split groups from repository, original task, document, TruthfulQA question, scenario state, and payload family before generating variants.
- Keep every clean twin, attacked twin, language variant, insertion position, and mutation in the same split atom.
- Remove every repository shared with the retained SWE-rebench and SWE-bench evaluation sources from fitting unless those evaluations are explicitly retired.
- Use source-macro validation selection and report per-source and leave-one-source-out results.
- Keep LogInject, WASP, and any other declared source-heldout panels absent from fitting, thresholds, architecture selection, and prompt selection.

## Claim boundary

SWE-rebench V2 tasks are source-supported software-engineering tasks, not adjudicated broad-benign conversation.
Constructed injected twins provide exact insertion labels, but they do not prove that an unmonitored agent would follow the payload.
Only executable ToolSandbox pairs that pass the complete causal and state-delta gate support a transaction-vulnerability claim.
The WMT clean arm supports `injection_label=0` for that task boundary, not broad `routing_label=0`, and the source should not be presented as general multilingual agent traffic.
