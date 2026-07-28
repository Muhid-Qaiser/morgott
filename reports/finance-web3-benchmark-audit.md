# Finance and Web3 prompt-injection benchmark audit

## Scope and conclusion

This audit asks whether a public benchmark measures prompt injection or jailbreak detection specifically in finance, banking, payments, crypto, or Web3 agents.
It uses official papers, repositories, project pages, and dataset cards rather than secondary descriptions.
It distinguishes instruction subversion from financial question answering, fraud detection, harmful financial requests, policy violations, and smart-contract vulnerability detection.

FinVault, ClawSafety, and AgentDojo Banking are the strongest released finance-native evaluations.
They measure complementary things: FinVault covers broad financial-agent attacks, ClawSafety contains realistic indirect injections in financial workspaces, and AgentDojo Banking measures stateful attack success and benign-task utility.
No released source found here is a large, clean, matched static-classification benchmark for finance or Web3.
None supplies the thousands of clean finance or Web3 negatives needed to estimate TPR at 1% FPR, and especially TPR at 0.1% FPR, with useful precision.
No released public Web3-specific static detector benchmark with matched benign tasks was found.

## Released and usable evaluation sets

| Artifact | Security signal and matching | Public access and license | Recommended use |
|---|---|---|---|
| [FinVault repository](https://github.com/aifinlab/FinVault), [paper](https://arxiv.org/abs/2601.07853), and [dataset card](https://github.com/aifinlab/FinVault/blob/main/DATASET_CARD.md) | 31 financial scenarios and 107 vulnerabilities across six domain groups; 107 core attacks, 856 synthesized attacks, and 107 scenario-linked or vulnerability-linked normal cases; domain-matched rather than minimal text-edit controls | Public repository with a research-use statement but no observed standard license file; redistribution and commercial-use rights unclear | Finance-native system benchmark; detector subset limited to explicitly typed instruction override and direct JSON injection plus manually audited encoding disguise; other cases retained as fraud, identity manipulation, authority abuse, privacy extraction, or policy bypass |
| [ClawSafety repository](https://github.com/weibowen555/ClawSafety), [paper](https://arxiv.org/abs/2604.01438), and [released finance cases](https://github.com/weibowen555/ClawSafety/blob/main/scenarios/s2_financial/s2_skill_email_cases.py) | Full benchmark of 120 adversarial cases across five domains; current v0.1 finance release of 24 attacks across skill files, email, and web content; adversarial content embedded in benign multi-turn financial tasks | MIT code; CC-BY-4.0 scenario narratives; adversarial test cases limited to defensive safety research by the repository security policy | Locked indirect-injection challenge set; attack-bearing content and same-workspace clean content used only for evaluation |
| [AgentDojo repository](https://github.com/ethz-spylab/agentdojo) and [paper](https://arxiv.org/abs/2406.13352) | 16 benign banking tasks crossed with 9 injection goals for 144 security cases; deterministic checks for requested-task completion and attacker-goal success; clean runs available for the same user tasks | MIT licensed and publicly released | End-to-end evaluation of advisory detector, agent behavior, and deterministic reference monitor; not a pre-made static text-classification dataset |
| [Financial AI Prompt Injection CTF dataset](https://huggingface.co/datasets/verno-labs/financial-ai-ctf-dataset) | 400 multi-turn conversations from a live 60-minute exercise; 155 submitted answers and 52 successful secret recoveries; direct extraction, jailbreak, roleplay, instruction override, social engineering, context manipulation, and obfuscation; no tools, retrieval, external documents, or matched benign task set | CC-BY-4.0 and publicly accessible on Hugging Face | Supplemental human-origin direct-attack challenge with participant-level lineage controls; no evidence about indirect-injection or financial-agent tool safety |
| [MobileSafetyBench project](https://mobilesafetybench.github.io/), [repository](https://github.com/jylee425/mobilesafetybench), and [paper](https://arxiv.org/abs/2410.17520) | Updated project with 250 mobile-agent tasks including 50 indirect prompt-injection scenarios; banking, stock trading, and financial transactions represented; exact finance-only subset size unstated | Apache-2.0 and publicly released, with the dataset linked from the official project site | Secondary end-to-end financial-action slice rather than a finance-specific detector benchmark |
| [Agent Security Bench repository](https://github.com/agiresearch/ASB) and [paper](https://arxiv.org/abs/2410.02644) | Ten synthetic domains including finance and investment; direct injection, indirect injection, memory poisoning, and backdoor attacks; only one finance scenario and no finance-specific matched static corpus | MIT licensed and publicly released | Broad technique-coverage check after finance-native evaluations; aggregate score not finance-specific evidence |

These released evaluations are too small or too task-structured to supply a reliable low-FPR denominator.
FinVault has only 107 normal cases, ClawSafety v0.1 has only 24 released finance attacks, and AgentDojo Banking has only 16 benign user tasks.
Their proper outputs are attack success, benign-task utility, per-channel recall, per-family recall, or fixed-threshold challenge recall.
They cannot support a persuasive claim about TPR at 1% FPR or 0.1% FPR.

## Publicly described but not currently usable

| Artifact | Evidence | Availability decision |
|---|---|---|
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

1. Freeze ClawSafety finance and a manually typed FinVault detector subset as external domain tests, with no training or threshold selection on either set.
2. Evaluate each model at thresholds selected only on Morgott validation, and report recall by source, attack family, delivery channel, and financial scenario.
3. Run AgentDojo Banking under no detector, advisory detector, and advisory detector plus deterministic reference-monitor conditions, then report attack success and benign-task utility together.
4. Use the financial CTF data only as supplemental direct-attack diversity with participant-held-out lineage, and keep its reported result separate from indirect-injection evaluation.
5. Build and prospectively freeze a much larger benign denominator from realistic finance and Web3 workflows before making low-FPR claims.
6. Track LivePI and use it only after the official artifacts and license are released.
7. Clarify FinVault's redistribution and commercial-use terms before adding any of its records to the canonical corpus.

The current public evidence can test whether a model recognizes finance-native attacks and whether a defense reduces stateful agent compromise.
When this benchmark work resumes, score all three retained frozen-head seeds and the one retained LoRA checkpoint independently.
Predeclare aggregation and model selection before reading benchmark labels; do not choose the best seed or create an OR ensemble after seeing results.
Because the current scorers truncate at 512 tokens, report a separate unsupported-long-input slice rather than presenting document-level coverage.
It cannot establish that the detector operates reliably at the low false-positive rates required for financial or Web3 deployment.
