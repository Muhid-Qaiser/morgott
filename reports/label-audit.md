# Label audit

This document records how source annotations map into the canonical schema. It
is not independent human adjudication. Public labels remain source labels, and
model/Codex judgments are never presented as ground truth.

## Target boundaries

- Direct jailbreak and prompt injection attempt to subvert instruction
  hierarchy.
- Indirect injection embeds such instructions in untrusted email, documents,
  web pages, retrieval, tool output, or memory.
- Harmful non-injection describes harmful intent without instruction subversion.
- Toxicity is independent and may co-occur with any other label.
- Benign means the source supports ordinary/safe content with no known
  instruction-subversion signal.
- Uncertain means the source cannot establish the required distinction. It is
  never coerced to benign.

The binary router is deliberately broader than injection detection:
source-supported benign rows receive `routing_label=0`; injection, harmful,
toxic, and unresolved rows receive `routing_label=1`. Nullable subtype fields
and independent tags preserve why a row is routed.

## Known source limitations

- ToxicChat and JailbreaksOverTime negatives do not establish broad benignity.
  ToxicChat rows without jailbreak or toxicity and JailbreaksOverTime negatives
  remain auxiliary for routing; ToxicChat's human-annotation and moderation
  metadata is preserved so label strength is not flattened.
- Deepset negatives and BIPIA clean contexts establish no injection under their
  source task, not general safety. They remain legacy injection controls and are
  auxiliary for broad routing.
- OASST1 accepted turns remain auxiliary for broad routing because acceptance
  does not establish benign safety. Only the legacy injection control uses its
  historical ordinary-chat view; available labels and Detoxify metadata remain
  in the source shard.
- BIPIA payload meaning depends on provenance. A standalone ordinary-looking
  question may become an attack only when inserted into untrusted context.
- Tensor Trust defense text contains security and instruction language. Attack
  contexts have no matched clean control and can expose benchmark shortcuts.
  Standalone attacks are direct injection while attacks embedded between
  defense prompts are indirect injection.
- Nemotron agentic IPI is synthetic and positive-only; it cannot estimate false
  positives, precision, or benign utility.
- LLMail `False` and `Unclear` annotations mean the challenge audit did not
  establish a confirmed attempt. They do not establish benign intent and remain
  outside ordinary supervision.
- HackAPrompt `correct` is target-model success, not attack intent. Failed
  submissions remain attack attempts.
- Tensor Trust game success is outcome metadata, not attack intent. Failed
  attacks remain attack attempts; defenses and outputs are auxiliary.
- WildJailbreak adversarial-benign and WildGuardMix's adversarial flag do not by
  themselves establish an injection label. The former and
  adversarial-unharmful WildGuardMix rows stay auxiliary. All WildGuard train
  prompt-harmfulness rows are model-labelled weak supervision and remain
  auxiliary; its test prompt harmfulness uses three human annotators, preserves
  agreement metadata, and may enter dev-test.
- BrowseSafe positive documents lack payload spans. They must not create
  positive labels for every chunk or window.
- Taskmaster, Schema-Guided Dialogue, BANKING77, and MASSIVE en-US are
  collection-supported ordinary task data, not independently adjudicated
  safety data. Their bounded source tasks support benign intent; sensitive
  topic words alone do not change that mapping.
- HarperValleyBank, TAT-QA, FinanceBench, and Mind2Web are also task-supported benign data, not human safety annotation.
  HarperValleyBank is simulated and narrow.
  TAT-QA and FinanceBench annotations establish finance-QA correctness and relevance, not arbitrary prompt safety.
  Mind2Web annotations establish executable web tasks; retained tasks pass a local high-precision sensitive-text screen, while suspicious tasks remain outside supervised views in quarantine.
- SWE-bench Verified human review establishes legitimate solvable software tasks, which is the source authority for routing label `0` rather than an absence-of-attack assumption.
  It is not a safety annotation, stays entirely in dev-test, and supports only repository- and length-sliced legitimate-workload diagnostics after the local sensitive-text screen.
- TAT-QA report paragraphs and tables and FinanceBench evidence are clean controls only in their published finance-task context.
  They do not prove that arbitrary retrieved financial text is safe, and their trusted `untrusted_content` channel comes from the adapter rather than text.
- FinanceBench is dev-test only and has already influenced source selection.
  It cannot become a prospective final test or support a production false-positive claim.
- FalseReject generated prompts and CoCoNot preference prompts are explicit
  weak benign development labels. Their respective human-annotated and
  human-verified test prompts remain identifiable dev-test evidence.
- LMSYS Chatbot Arena retains unsafe conversations upstream. morgott treats
  English user prompts for which OpenAI moderation and both published ToxicChat
  taggers are unflagged as weak benign development supervision. Paired model
  outputs are a separate weak-benign slice inferred from that context, with
  toxicity left unknown because the tags do not label the responses. Flagged
  user prompts and paired model outputs remain uncertain. The September 2023
  source does not pin tagger checkpoints or establish performance for morgott's
  routing ontology, and audited positives mix false positives with genuine harm
  and jailbreaks.
- JBB benign behaviors are curated thematic contrasts to misuse requests. They
  are hard-benign dev-test rows, not evidence about arbitrary live traffic.
- HarmBench, Do-Not-Answer, AdvBench, AgentHarm, BeaverTails, and generic
  toxicity annotations must not be converted into injection positives.

## Masked model-target projection

The research encoder target projection is deterministic and leaves the canonical labels unchanged.
A head receives a positive only from explicit evidence for that axis and a negative only from explicit matching-axis evidence.
Missing, weakly incompatible, or conflicting evidence is null and loss-masked.
A positive subtype is never silently used as a negative for another subtype.

`direct_instruction_subversion` covers direct prompt injection and jailbreak attempts.
`jailbreak` remains a separate co-occurring head.
`indirect_instruction_subversion` depends on trusted runtime provenance and must not infer the input channel from attacker-controlled text.
`harmful_intent` may co-occur with instruction subversion.
`harmful_non_injection`, `review_required`, `no_security_signal`, and `uncertain` are derived by advisory routing logic rather than mutually exclusive learned classes.
Toxicity remains outside the active model until a second independent positive source and matched negatives exist.

Boundary Pairs may supervise only eight instruction-subversion families: direct override, indirect content injection, memory poisoning, multi-agent spoofing, obfuscation, RAG poisoning, roleplay jailbreak, and system-prompt extraction.
Approval bypass, authority claims, sensitive-data exfiltration, and tool abuse remain reference-monitor diagnostics with every learned target masked.
Clean Boundary rows are negatives only for their supported paired head and never become broad benign routing examples.
Official split, pair ID, scenario ID, source context, risk domain, and boundary metadata are preserved.
Pairs and scenarios must not cross fitting and evaluation roles.

The global label-support audit found two positive source families, explicit negatives, and a same-source or paired contrast for each active head.
That gate establishes only that fitting is mechanically supportable.
The finance quick fold still had undefined masked per-head PR-AUC because legitimate held-out task rows lacked explicit matching-axis negatives, while its combined advisory route had catastrophic false positives.
Broad source-supported benign route metrics must therefore accompany every masked-head report.

## Merge and conflict policy

- Detector text is preserved in canonical projections; normalization is derived
  only for matching.
- Exact same-label duplicates merge into one view row while every source
  annotation remains in `origins`.
- Exact routing-label conflicts go to quarantine.
- If routing agrees but subtype annotations disagree, disputed top-level fields
  become unknown and subtype training is masked. Source order never decides.
- Auxiliary, uncertain, and quarantined rows never silently enter train,
  validation, or dev-test.
- Weak-labelled rows use `source_role=candidate`; `label_basis` and origins
  retain their evidence strength after exact merging and grouped partitioning.
- `routing_training_eligible` is derived from `source_role`. It is separate from
  injection-label availability and the historical injection-view recipe.

## Weak labels

No project-specific human labelers are available. Public synthetic and
automated labels remain weak even at scale: agreement measures consistency
rather than correctness. Selected weak rows may enter train, validation, and
dev-test as candidates, but must be sliced separately from human/source labels.
They stay out of any prospective locked final evaluation and cannot support
production-FPR claims.

## Decision

Retain nullable injection labels and independent security tags.
Use masked multi-task losses rather than a single mutually exclusive unsafe subtype.
Derive `no_security_signal` and `review_required` through advisory routing logic.
The completed quick encoder failed finance and long-document false-positive gates.
A bounded consolidated repair passed the finance and matched-pair gates but failed the BrowseSafe ranking and recall gates.
Harmful intent remains content-safety metadata and is not OR-ed into the transaction-security route.
The detector stays shadow-only, and no learned output may authorize or block an action.
