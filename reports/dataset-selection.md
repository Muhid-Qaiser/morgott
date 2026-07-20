# Dataset selection and expansion audit

Generated: 2026-07-20

The user's Linear research list was reviewed against the detector's actual
target: attempts to subvert instruction hierarchy, not harmful content in
general. More rows are useful only when they add an attack family, provenance
condition, language, or realistic non-attack distribution that is absent from
the current corpus.

## Included now

The reproducible build uses twelve ungated, pinned public sources. ToxicChat,
deepset, and OASST1 provide the direct training mixture. BIPIA alone trains the
separate untrusted-content channel. XSTest, HarmBench, Do-Not-Answer, NotInject,
the multi-turn cipher corpus, JailbreaksOverTime, and Tensor Trust stay out of
training and threshold selection.

Tensor Trust was an earlier immediate addition from the expanded audit. Its two
small official robustness suites contain 1,346 human-written attack records:
908 unique attack texts after exact deduplication, plus all 1,346 records
embedded between their benchmark defense prompts. They are pinned at
[`747a75e`](https://github.com/HumanCompatibleAI/tensor-trust-data/commit/747a75e096761ebc01bd3970158827326b4add23)
and are evaluation-only because the data repository has no explicit standard
dataset license. The large game dump is deliberately not downloaded.

NVIDIA's [Nemotron Agentic Indirect Prompt Injection
suite](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1/tree/d738d4f361cc38bb4d7a42b9066776dade5332f5)
adds 1,272 fully synthetic agentic attacks across nine domains, four impact
categories, 36 injection vectors, and 40 target tools. The build pins its
CC-BY-4.0 release and retains 676 exact-unique `injection_text` values as an
evaluation-only untrusted-content suite. It stores categorical lineage but not
the source environment and its synthetic identity records, system/user prompts,
tool schemas, attack goal, or target arguments. Because the source is
positive-only and retains only
attacks that succeeded against its defender, it measures transfer recall—not
false positives, benign task utility, or production safety. See the [dedicated
audit](nemotron-agentic-ipi.md).

## Deferred or rejected

| Candidate | Decision | Reason |
|---|---|---|
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | Completed weak-label experiment; stopped | The 5k sampled pilot accepted 2,430 model-agreed weak negatives, but the bounded-weight ablation reduced direct and indirect macro recall, including multi-turn recall from 908/4,136 to 291/4,136 at the 85% profile. The rows stay out of the baseline and collection does not scale. They cannot support a lockout/FPR claim. See [the pilot report](wildchat-ablation.md). |
| [PromptShield](https://huggingface.co/datasets/hendzh/PromptShield) | Completed evaluation-only audit; do not train | Its 43,425 rows lack row-level source/group lineage, the paper discloses aggregation from public corpora with active-family overlap, and 97 test rows overlap the active fit set. At the locked 85% profile the character control catches 194/6,486 source positives and alerts on 240/17,030 source negatives; removing fit overlaps barely changes this. See [the audit](../experiments/promptshield_audit/REPORT.md). |
| [HackAPrompt](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) | Skip while gated; provenance audit only | The MIT-labelled repository still requires contact-sharing acceptance and returns 401 anonymously, so no token or mirror was used. If explicitly accessed later, use only `user_input`, keep `correct` as target-specific success rather than a benign label, and begin evaluation-only. See [the pinned audit](hackaprompt-yaklang-audit.md). |
| [Yaklang llm-prompt-injection](https://github.com/yaklang/hack-skills/tree/c9a4b9ee8645eb60763eb4eef172f1ecb0a5b3e8/skills/llm-prompt-injection) | Taxonomy/scenario reference only | It is a playbook rather than a labelled corpus. Canonical direct examples mostly duplicate current sources, while tool-output, cross-MCP, and Markdown-exfiltration gaps belong in stateful reference-monitor tests. It is not installed, vendored, or used as a payload generator. |
| [AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) | Reference-monitor evaluation later | Agent misuse tasks test authorization and harmful actions, not prompt injection. Treating them as detector attacks would recreate harmful-content conflation. |
| [SafeDialBench](https://huggingface.co/datasets/HongyeCao/SafeDialBench) | Defer | Synthetic multi-turn attacks with no clean controls, unclear redistribution metadata, and attack-template leakage risk; the current grouped multi-turn holdout already tests this gap. |
| [AdvBench](https://github.com/llm-attacks/llm-attacks) | Skip | The released goals are harmful requests, not jailbreak prompts; generated GCG suffixes are separate. HarmBench already supplies this negative control. |
| [Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) and [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Skip | Content-safety labels do not identify instruction subversion, and both add overlap/license risk without a new injection signal. |
| WildGuardMix, Chatbot Arena, LMSYS-Chat-1M, CeSIA Jailbreak, CeSIA BET | Skip for now | Access conditions or manual gating conflict with the requested ungated baseline. BET/technique transformations also require grouping by primitive/base prompt to avoid catastrophic leakage. |

## Anti-overfitting rules

- Training, calibration, and external evaluation sources remain distinct where
  the source permits it.
- Conversation trees, BIPIA contexts, mutation goals, and future synthetic
  seeds are split as groups; exact evaluation duplicates are blocked from
  training.
- Harmful intent and injection intent remain separate labels.
- No model- or Codex-generated row enters frozen public evaluation. With no human labelers,
  the WildChat pilot uses conservative model-only weak supervision: two model
  families must agree on high-confidence benign, every detector-hard candidate
  and a deterministic random audit slice receive a third-family judgment, and
  disagreements or uncertain rows are discarded. Agreement is not accuracy.
- Report leave-one-source-out and production-sampled metrics before any model
  is allowed to block users. Until then, detector output remains shadow-only.
