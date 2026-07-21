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
- HarmBench, Do-Not-Answer, AdvBench, AgentHarm, Aegis, BeaverTails, and generic
  toxicity annotations must not be converted into injection positives.

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
- `routing_training_eligible` is derived from `source_role`. It is separate from
  injection-label availability and the historical injection-view recipe.

## Weak labels

No human labelers are available. A future provider-assisted experiment may
produce development-only weak labels, but agreement measures consistency rather
than correctness. Weak labels require their own versioned recipe and must stay
out of validation, dev-test, final evaluation, and production-FPR claims.

## Decision

Keep the binary route for the first model while retaining nullable injection
labels and independent security tags. Use masked multi-task losses rather than a
single mutually exclusive “unsafe subtype.” The detector stays shadow-only until
independently labelled product evidence exists.
