# Finance and Web3 prompt-injection benchmark audit

## Scope and conclusion

This audit asks whether a public benchmark measures prompt injection or jailbreak detection specifically in finance, banking, payments, crypto, or Web3 agents.
It uses official papers, repositories, project pages, and dataset cards rather than secondary descriptions.
It distinguishes instruction subversion from financial question answering, fraud detection, harmful financial requests, policy violations, and smart-contract vulnerability detection.

AgentDojo Banking is the strongest currently usable released finance-native evaluation in this audit, but its current Morgott panels are already consumed development evidence.
[FORCE-Bench](https://github.com/microsoft/FinanceBenchmark) now supplies a fresh legitimate enterprise-finance task denominator, but it has no attacks and its completed 251-task Morgott diagnostic is also consumed development evidence.
FinVault is no longer usable as external evidence without an upstream reissue and conformance fixes.
ClawSafety describes a valuable realistic indirect-injection benchmark, but its public finance release is not yet complete or runnable as committed.
No released source found here is a large, clean, matched static-classification benchmark for finance or Web3.
None supplies the thousands of clean finance or Web3 negatives needed to estimate TPR at 1% FPR, and especially TPR at 0.1% FPR, with useful precision.
No released public Web3-specific static detector benchmark with matched benign tasks was found.

## Screened finance-native evaluation artifacts

| Artifact | Security signal and matching | Public access and license | Recommended use |
|---|---|---|---|
| [FinVault repository at `7884818`](https://github.com/aifinlab/FinVault/tree/78848188a74d0124a74a63134315a65fdd43fb2a), [withdrawn paper record](https://arxiv.org/abs/2601.07853), and [dataset card](https://github.com/aifinlab/FinVault/blob/78848188a74d0124a74a63134315a65fdd43fb2a/DATASET_CARD.md) | 31 financial scenarios, 107 core attacks, 856 synthesized attacks, and 107 normal cases, but the released runner has material category, observation, and failure-accounting defects | The paper was withdrawn on 2026-07-30 for stated legal and intellectual-property concerns; the repository has research-use prose and an Apache badge but no license file | Do not score or use as external evidence until upstream reissues the artifact and fixes the conformance failures below |
| [AgentDojo repository](https://github.com/ethz-spylab/agentdojo) and [paper](https://arxiv.org/abs/2406.13352) | 16 benign banking tasks crossed with 9 injection goals for 144 security cases; deterministic checks for requested-task completion and attacker-goal success; clean runs available for the same user tasks | MIT licensed and publicly released | End-to-end evaluation of advisory detector, agent behavior, and deterministic reference monitor; not a pre-made static text-classification dataset |
| [FORCE-Bench repository at `6ced62b`](https://github.com/microsoft/FinanceBenchmark/tree/6ced62b961d4c18b2ba53f268b443eb852fb73ca) and [paper](https://arxiv.org/abs/2607.19409) | 251 legitimate enterprise-finance tasks covering synthetic ERP accounts receivable and payable, public-company research, and business briefs; no prompt-injection positives | MIT licensed and publicly released | Fresh direct-user false-positive denominator only; the fixed Morgott cascade diagnostic below is now consumed and cannot select a threshold |
| [Financial AI Prompt Injection CTF dataset](https://huggingface.co/datasets/verno-labs/financial-ai-ctf-dataset) | 400 multi-turn conversations from a live 60-minute exercise; 155 submitted answers and 52 successful secret recoveries; direct extraction, jailbreak, roleplay, instruction override, social engineering, context manipulation, and obfuscation; no tools, retrieval, external documents, or matched benign task set | CC-BY-4.0 and publicly accessible on Hugging Face | Supplemental human-origin direct-attack challenge with participant-level lineage controls; no evidence about indirect-injection or financial-agent tool safety |
| [MobileSafetyBench project](https://mobilesafetybench.github.io/), [repository](https://github.com/jylee425/mobilesafetybench), and [paper](https://arxiv.org/abs/2410.17520) | Updated project with 250 mobile-agent tasks including 50 indirect prompt-injection scenarios; banking, stock trading, and financial transactions represented; exact finance-only subset size unstated | Apache-2.0 and publicly released, with the dataset linked from the official project site | Secondary end-to-end financial-action slice rather than a finance-specific detector benchmark |
| [Agent Security Bench repository](https://github.com/agiresearch/ASB) and [paper](https://arxiv.org/abs/2410.02644) | Ten synthetic domains including finance and investment; direct injection, indirect injection, memory poisoning, and backdoor attacks; only one finance scenario and no finance-specific matched static corpus | MIT licensed and publicly released | Broad technique-coverage check after finance-native evaluations; aggregate score not finance-specific evidence |

These released evaluations are too small or too task-structured to supply a reliable low-FPR denominator.
FORCE-Bench contributes 251 legitimate tasks, FinVault has only 107 normal cases, and AgentDojo Banking has only 16 benign user tasks.
Their proper outputs are attack success, benign-task utility, per-channel recall, per-family recall, or fixed-threshold challenge recall.
They cannot support a persuasive claim about TPR at 1% FPR or 0.1% FPR.

## FinVault stop decision after source preflight (2026-08-03)

The [FinVault paper record](https://arxiv.org/abs/2601.07853) was withdrawn on 2026-07-30 after the authors identified legal, intellectual-property, affiliation, and compliance concerns.
The public repository still points `main` and `HEAD` to commit `78848188a74d0124a74a63134315a65fdd43fb2a`.
Its README advertises Apache-2.0 for code, but the pinned Git tree contains no `LICENSE` or `COPYING` file and provides only a general research-use statement.

The five packaged release-integrity tests pass and all 310 JSON files satisfy their permissive structural and pattern-based privacy checks.
That does not establish benchmark conformance.
A local source preflight imported and instantiated all 31 environments and reset all 107 original attack cases, then found these blocking defects:

- The category registry assigns 21 of 31 scenario IDs to a domain that conflicts with the released dataset, prompt, and runtime environment identity.
  For example, scenario `01` is the SWIFT remittance environment but is aggregated as securities investment, while scenario `02` is insurance claims but is aggregated as credit and lending.
- All three scenario `22` attack prompts are absent from the initial observation passed to the agent.
  The environment exposes only state, triggered-vulnerability, and action-count fields, so those cases cannot measure response to their declared attack text.
- Both the runner and agent wrapper convert model or provider exceptions into an `escalate_to_human` action.
  A provider-failure canary on scenario `00` finished as `ESCALATED` with `attack_success=False`, no vulnerability, and `error=None`, so infrastructure failure is counted as successful defense rather than an incomplete run.
- The attack runner does not execute the 107 normal cases through the same tool-using agent and state oracle.
  Normal files are consumed by separate safeguard-classifier scripts, so the release does not provide the claimed paired end-to-end benign utility comparison.
- The source family named `instruction_override` commonly expresses procedural or authority manipulation without an instruction-hierarchy override, while `direct_json_injection` commonly proposes a typed action without establishing subversion.
  Those family names are generation metadata, not Morgott ontology labels.

Decision: stop before any provider call or detector score.
Do not locally repair FinVault and report the result as upstream evidence.
Reconsider only after an upstream reissue resolves the withdrawal, ships an explicit license, fixes category identity and attack visibility, fails incomplete runs closed, and executes matched normal cases through the same agent and oracle.

## Frozen FORCE-Bench benign-finance diagnostic (2026-08-03)

The [FORCE-Bench release](https://github.com/microsoft/FinanceBenchmark/tree/6ced62b961d4c18b2ba53f268b443eb852fb73ca) was frozen at revision `6ced62b961d4c18b2ba53f268b443eb852fb73ca` before scoring.
Its `data/dataset.yaml` has SHA-256 `a795468c028b5511582e68683e3c879a577a92c9259afc045b93d0e78ae5ebd6` and contains 251 unique legitimate task queries: 100 synthetic ERP questions, 126 public-company finance research tasks, and 25 business briefs.
Only the source `query` field enters the direct-user cascade.
Rubric assertions, ground-truth values, agent answers, and raw query text are absent from retained artifacts.

The frozen normalized, audit-strict, and conservative near-overlap pass retained all 251 tasks against 1,073,330 canonical training rows, 18,284 PromptShield training rows, and 22,092 matched-pair rows.
No threshold, prompt, model, provider, or source slice was selected from this panel.
The predeclared gate rejects the fixed cascade if its observed restriction rate exceeds 1%, while any incomplete assessment or provider failure makes the result inconclusive.

The unchanged registered cascade restricted 0 of 251 tasks.
All 251 assessments completed, 250 passed at the local low gate, one entered the review band, and DeepSeek cleared that task with probability `0.0021827169` after an mmBERT score of `0.8467424732`.
That reviewed case is a legitimate multi-part supplier-risk and scenario-analysis task, so it is a useful example of the reviewer correcting an over-sensitive local score without weakening authorization.
There were no provider failures and no malformed reviews.

The observed restriction rate is 0%, but its 95% Wilson upper bound is 1.507%.
Plugin-level upper bounds are still wider: 3.699% for the 100 ERP tasks, 2.959% for the 126 finance-research tasks, and 13.320% for the 25 business briefs.
The result therefore passes its bounded engineering gate but does not establish a 1% production false-positive rate.
It supplies no attack recall, indirect-content, transaction-execution, Web3, multilingual, or long-document evidence.

Decision: retain the unchanged cascade and this result as consumed benign-finance development evidence.
Do not tune on the reviewed task or add these public queries to training before a separately frozen evaluation replaces this panel.

The text-free [manifest](../artifacts/force_bench_eval/manifest.json), [panel](../artifacts/force_bench_eval/panel.jsonl.gz), [result ledger](../artifacts/force_bench_eval/results.jsonl.gz), and [summary](../artifacts/force_bench_eval/summary.json) have SHA-256 values `4e2b22e15f9cbdc98e326e2b6fa5dedf60d04f70b651a6b0bbc34ab6e5d86cbd`, `920a1820cd9b9f1bc03d6fd354ee06fc9897ea4590fedf6388a62fb2f9f05ba9`, `85b909692d796cd804f1366aaa9f295b0450a17bd93080df58c106f3950b88fe`, and `bb09c4392af8b2afba658a33a944314ebb261a6dbab5840242e587146e26a26f`.
The frozen protocol and artifact hashes are retained in this report; the completed one-off runner is not maintained.

## Frozen Financial AI CTF diagnostic (2026-08-02)

The public Financial AI CTF source was frozen at revision `44fa97e18ec106c2b51c6917975a9aab756f8aa1` before scoring.
Its 400 conversations contain 1,435 incoming user turns from 22 anonymized participants.
The evaluation sends only individual user turns through the already-selected cascade and restricts a conversation when any turn restricts.
Defender system prompts, protected values, assistant messages, and flag submissions are never model or provider input.

The existing normalized, strict, and conservative near-overlap guard compared every turn with 1,113,706 candidate fit rows from canonical train, PromptShield train, and retained generated pairs.
It excluded 15 complete conversations from primary metrics: 6 normalized exact matches, 6 additional strict matches, and 3 near matches.
None of the 36 source-labelled `instruction_override` conversations was excluded.

The source strategies were produced by GPT-4.1 at conversation level, so they remain weak source labels.
The predeclared aligned diagnostic treats only `instruction_override` as positive evidence and treats zero-strategy conversations as a weak clean flag-rate check rather than verified negatives.

| Fixed-threshold non-overlap slice | Conversations | Restricted | Rate |
|---|---:|---:|---:|
| Source-labelled instruction override | 36 | 30 | 83.33% |
| Zero source strategies | 35 | 1 | 2.86% |

Instruction-override restriction is only 63.26% when participant rates are macro-averaged across the 11 participants represented in that slice, including one participant at 0% and one at 100%.
The gap between 83.33% row recall and 63.26% participant-macro recall is a material robustness warning.
The zero-strategy participant-macro restriction rate is 0.78% across 16 represented participants, but it is not a production FPR because the rows lack human benign verification.

| Source strategy, non-overlap | Conversations | Restriction rate |
|---|---:|---:|
| Instruction override | 36 | 83.33% |
| Obfuscation | 27 | 70.37% |
| Social engineering | 103 | 49.51% |
| Roleplay or fiction | 80 | 31.25% |
| Context manipulation | 79 | 25.32% |
| Format coercion | 131 | 22.90% |
| Direct ask | 149 | 22.15% |

All 1,435 turn assessments completed.
The local model passed 838 turns, hard-restricted 45, and sent 583 windows to DeepSeek; DeepSeek cleared 446, flagged 106, and had no failed reviews.

Only 11 of 84 non-overlap conversations whose assistant output leaked a protected field restricted anywhere.
A post-hoc timing diagnostic found that only 7, or 8.33%, restricted before the first leaking assistant turn.
This does not mean the instruction-subversion detector missed 77 aligned positives: only one leaking conversation carries the source's `instruction_override` strategy, and that conversation restricted before its leak.
Most successful leaks use direct asks, format completion, roleplay, or other authority and confidentiality failures that text alone cannot safely relabel as prompt injection.

Decision: do not tune the detector or threshold on this result.
Treat user-visible model output as an egress action, keep sensitive-data labels in trusted runtime metadata, and mediate that output with the deterministic reference monitor.
The policy simulation now grants an ordinary `respond_to_user` capability, permits non-sensitive model output, and denies a synthetic protected-field response while continuing to deny all 9 attack proposals and allow all 3 benign proposals.
This is a conformance simulation, not proof that a deployed runtime has DLP, provenance propagation, or safe response generation.

The text-free panel, manifest, result ledger, and summary have SHA-256 values `67cba9167068f3a3a37516edd65b09ffad3b12e5724551a38a81aeed77fff4c6`, `05b098660eef96fb6756104fe5bd208c3f2357f84a7b8842bb14d096fbdea435`, `bf92191787586e7f41e1ac1e39ce377416a6d5095e15c748c7421b84c53dfc31`, and `5bf530f503e3ad1d34956efd96deaa6dd8399bad8c273d9a72537f323da48732`.
The one-off study runner is not versioned; it ran as a disposable experiment script with SHA-256 `c159c460c1f8acb8db082ebb795573afb873abb7fc6de8e4df495965c4e8110e`, and the versioned panel, manifest, ledger, and summary above are the retained evidence.

## AgentDojo Banking action-containment diagnostic (2026-08-02)

The provider-free experiment pins AgentDojo package `0.1.35`, tag revision `a75aba7631d3ca5fb7ab938965c97ead2f9ff84b`, benchmark `v1.2.2`, and the 16-user-task by 9-injection-task Banking suite.
It replays AgentDojo's declared user actions as trusted proposals and declared attacker actions as untrusted proposals against copied Banking state.
Exact task capabilities are derived from each user task's declared ground truth, so the result is a reference-monitor compatibility and containment check under oracle capabilities and provenance.
No agent reads injected text, chooses actions, or adapts after denial.

All 16 standalone user tasks retain utility and all 33 declared user calls commit through the monitor.
Across the 144 paired cases, the unmonitored replay meets all 144 attacker goals and retains user utility in 125 cases.
The monitored replay meets 0 attacker goals, retains user utility in all 144 cases, and commits all 297 repeated trusted user calls.
Only 3 of 192 attacker calls commit; all three are read-only `get_scheduled_transactions` calls and none produces an attacker-goal state.
Of the denied attacker calls, 130 name a tool outside the task capability and 59 attempt a granted mutating capability from untrusted provenance.

This is not AgentDojo targeted ASR, utility under attack, detector accuracy, or evidence that a model resists prompt injection.
The next stateful experiment needs one fixed real tool-calling agent that reads the injected tool output, with the same agent and attack configuration compared under no monitor and monitored conditions.

The text-free compressed result artifact is `artifacts/agentdojo_policy_eval/result.json.gz` at SHA-256 `8a850bbbd1a9596c86e07b2a757ee1c74b5320098fe39761cb410fb0438063b3`.
It stores task IDs, tool names, authorization decisions, deterministic outcomes, and hashes of the pinned AgentDojo sources, Morgott policy, and runner.
The completed one-off runner is not maintained.

## AgentDojo Banking detector-transfer diagnostic (2026-08-02)

The separate detector study materializes the complete serialized tool return for every pinned Banking `direct` and `important_instructions` case.
Its 288 attacked task pairs collapse to 18 rendered payloads and 72 exact tool-return texts, while the unattacked side contributes 16 exact user prompts.
No panel input has normalized, audit-strict, or conservative near overlap with any full-LoRA fit candidate.

The channel-aware advisory cascade restricts 31 of 36 exact `direct` tool returns and all 36 exact `important_instructions` tool returns.
Row-weighting the repeated source cross-product gives 139 of 144 and 144 of 144 respectively, but those counts are not independent text examples.
The cascade restricts 2 of 16 legitimate user prompts.
Those prompts are too few, synthetic, and high impact to estimate production FPR.

All five exact `direct` misses remain below the selected `0.1` untrusted-content review floor: one bill context and four address-change contexts.
The panel motivated a channel-specific floor audit but did not select its threshold, and the selected change leaves these AgentDojo routes unchanged.
No threshold fitting, encoder fitting, agent execution, task-utility measurement, or attack-success measurement uses this source.

The text-free [manifest](../artifacts/agentdojo_detector_eval_channel_low/manifest.json), [panel](../artifacts/agentdojo_detector_eval_channel_low/panel.jsonl.gz), [result ledger](../artifacts/agentdojo_detector_eval_channel_low/results.jsonl.gz), and [summary](../artifacts/agentdojo_detector_eval_channel_low/summary.json) have SHA-256 values `39aae4911c8c859672a965147258b646a826d9d039ecf7801e886b37952be802`, `cfbf698f3e0cc12d1306b6e7aa10e21e94bc0ff544a1e733c95418ed5411fca7`, `9da9dcc241dd661915d0f70bde43db773d5889af0ba6749827779dfcac99a114`, and `c79e73958975331cee20abf1c61d3a20cd4563d9adb2efe3394023c6af594611`.
They store hashes, lineage, routes, typed reviewer scores, and operational telemetry but no prompt, payload, tool result, or raw provider response.

## Publicly described but not currently usable

| Artifact | Evidence | Availability decision |
|---|---|---|
| [ClawSafety repository at `5baf6fb`](https://github.com/weibowen555/ClawSafety/tree/5baf6fb40ab41bce40debf502f08e05320280d20), [paper](https://arxiv.org/abs/2604.01438), and [finance case file](https://github.com/weibowen555/ClawSafety/blob/5baf6fb40ab41bce40debf502f08e05320280d20/scenarios/s2_financial/s2_skill_email_cases.py) | The paper describes 120 adversarial cases and 24 finance cases, but the public checkout inspected on 2026-08-01 defines 17 skill/email cases, omits the advertised web-case file, and references workspace and preload files at paths absent from the tree | Prospective locked indirect-injection evaluation after upstream publishes a complete runnable revision; do not patch the benchmark locally and report the result as upstream evidence |
| [LivePI project](https://leizhao7.github.io/livepi/) and [paper](https://arxiv.org/abs/2605.17986) | 169 executable indirect-injection cases across seven delivery surfaces and twelve attack families; crypto-wallet material exfiltration and Solana transfers; bounded real-wallet execution and a separate benign-utility workload | Official GitHub repository and Hugging Face dataset marked TBD; no artifact license stated; prospective crypto and Web3 holdout only |
| [Banking data-exfiltration extension paper](https://arxiv.org/abs/2506.01055) | AgentDojo Banking extension from 16 to 48 tasks; four data-flow injection variants over 192 attacked scenarios | No official reusable code or dataset release found; design reference rather than public benchmark artifact |

## Released data unsuitable as external benchmark evidence

| Artifact | Problem | Decision |
|---|---|---|
| [FinGuard finance injection dataset](https://huggingface.co/datasets/nandhak12/finguard-finance-injection-dataset) | 13,746 rows combining Banking77 benign text, generic prompt-injection sources, and synthetic finance attacks without matched pairs; published samples conflating harmful or unauthorized financial intent with instruction subversion; strong source, topic, and style confounding; Apache-2.0 card with component-level provenance and licensing still requiring review | No external detector evidence; weak-supervision quarantine only after ontology, provenance, overlap, and source-confounding audits |
| [Financial Prompt Injection Dataset](https://huggingface.co/datasets/Mukta9904/Financial-Prompt-Injection-Dataset) | Card describing 10,300 training rows and 1,818 test rows assembled from finance QA, generic injection sources, and synthetic text; incompatible train and test schemas in the Hugging Face viewer; harmful financial requests mixed with injection; no matched pairs | No benchmark use; individual sources considered only if provenance, labels, licenses, and overlap can be reconstructed |

FinanceBench, TAT-QA, FinQA, ConvFinQA, and Banking77 are finance question-answering or intent datasets rather than prompt-injection benchmarks.
They can contribute carefully grouped benign-domain controls, but they provide no positive evidence about instruction subversion.
Fraud, AML, market-abuse, and unauthorized-transaction datasets measure harmful or disallowed activity unless the records explicitly contain an attempt to override trusted instructions.
Smart-contract vulnerability and exploit datasets measure code or protocol security rather than natural-language instruction subversion.
None of those categories should be relabelled as prompt injection merely because the application domain is financial or Web3.

## Recommended Morgott evaluation plan

1. Keep FinVault rejected until an upstream reissue passes the conformance gates above.
   Add ClawSafety finance only after an upstream revision provides the advertised cases and runnable assets.
2. Evaluate each model at thresholds selected only on Morgott validation, and report recall by source, attack family, delivery channel, and financial scenario.
3. Extend the completed AgentDojo deterministic action replay and detector-transfer diagnostic with one fixed real tool-calling agent under no monitor and reference-monitor conditions, then add the frozen advisory detector only as a separate ablation.
4. Use the financial CTF data only as supplemental direct-attack diversity with participant-held-out lineage, and keep its reported result separate from indirect-injection evaluation.
5. Keep the completed FORCE-Bench panel as consumed bounded evidence, and build a much larger independently labelled finance and Web3 denominator before making low-FPR claims.
6. Track LivePI and use it only after the official artifacts and license are released.
7. Do not add FinVault records to the canonical corpus while the paper is withdrawn and the source contract is defective.

The current public evidence can test whether a model recognizes finance-native attacks and whether a defense reduces stateful agent compromise.
When this benchmark work resumes, score each registered artifact independently; only the seed-42 frozen head is available for inference, so historical frozen-seed dispersion cannot be replayed from the maintained tree.
Predeclare aggregation and model selection before reading benchmark labels; do not choose the best seed or create an OR ensemble after seeing results.
Because the current scorers truncate at 512 tokens, report a separate unsupported-long-input slice rather than presenting document-level coverage.
It cannot establish that the detector operates reliably at the low false-positive rates required for financial or Web3 deployment.
