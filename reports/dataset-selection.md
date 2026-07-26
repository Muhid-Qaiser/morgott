# Dataset selection

The corpus serves two related targets: the retained channel-specific injection
POC and a future conservative benign/non-benign router. A source is useful only
when its label meaning, detector text, and grouping lineage survive
canonicalization.

Exact revisions, file digests, output counts, and licenses live only in
`data/manifest.json`. This document records durable interpretation decisions.

## Included sources

| Source | Canonical role |
|---|---|
| ToxicChat | Jailbreak/toxicity positives; ordinary rows remain auxiliary for broad routing |
| deepset prompt-injections | Injection positives; negatives remain legacy injection controls |
| OASST1 | Multilingual accepted chat retained as auxiliary; historical view remains a legacy injection control |
| XSTest | Over-refusal and sensitive-topic hard negatives |
| HarmBench | Harmful goals without instruction subversion |
| Do-Not-Answer | Harmful requests without instruction subversion |
| Multi-turn jailbreak corpus | Goal-grouped obfuscated attacks and matched contrast families |
| BIPIA | Indirect payloads/poisoned contexts; clean contexts remain legacy injection controls and routing auxiliary |
| NotInject | Instruction/security trigger-word hard negatives |
| JailbreaksOverTime | Source/style-shift development evaluation with noisy labels |
| Tensor Trust robustness suites | Human attacks and attack-in-defense contexts; evaluation only |
| Nemotron Agentic IPI | Positive-only synthetic agentic injection development evaluation |
| Gandalf | Human direct attack attempts; official train candidate, validation/test dev-test |
| LLMail | Adaptive email submissions with team/scenario/time lineage and official controls |
| BrowseSafe | Long HTML attacks and matched controls; whole documents retained |
| HackAPrompt | Competition `user_input` attack attempts; success is metadata |
| WildJailbreak | Four-way harmful/benign and vanilla/adversarial source construction |
| WildGuardMix | Prompt harmfulness with a separate adversarial flag |
| [HarperValleyBank](https://github.com/cricketclub/gridspace-stanford-harper-valley) | Simulated human-human banking turns grouped by complete conversation |
| [TAT-QA](https://github.com/NExTplusplus/TAT-QA) | Finance questions plus clean report paragraphs and serialized tables grouped by hybrid context |
| [FinanceBench](https://github.com/patronus-ai/financebench) | Public 150-example finance diagnostic held entirely in dev-test and grouped by document |
| [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | Confirmed official training tasks after local secret and PII quarantine |
| Tensor Trust raw | Human game attacks plus auxiliary defenses and model outputs |
| [Taskmaster 1-3](https://github.com/google-research-datasets/Taskmaster) | English task-oriented user and assistant turns grouped by conversation and shared task instructions |
| [Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | English task-dialogue turns grouped by official split and dialogue |
| [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) | English finance-language queries with its official test held out |
| [MASSIVE 1.1 en-US](https://huggingface.co/datasets/AmazonScience/massive) | English voice-assistant intents; multilingual translations remain excluded |
| [FalseReject](https://huggingface.co/datasets/AmazonScience/FalseReject) | Synthetic hard-benign candidates plus human-annotated dev-test |
| [CoCoNot](https://huggingface.co/datasets/allenai/coconot) | Safe-to-comply preference candidates plus human-verified contrast dev-test |
| [JBB benign behaviors](https://github.com/JailbreakBench/jailbreakbench#accessing-the-jbb-behaviors-datasets) | Curated, safety-sensitive benign dev-test goals |
| [LMSYS Chatbot Arena Conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) | English Arena messages with weak-benign candidates and flagged conversations retained as uncertain |
| [Agentic Prompt Injection Boundary Pairs](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-boundary-pairs/tree/a5682e7573e1c7bc4b12e64d49c0dcd90ca776cf) | Auxiliary-only matched instruction-boundary pairs with official split and pair/scenario lineage |

The two Tensor Trust adapters intentionally expose different official artifacts:
the compact robustness suites remain a historical development comparison, while
the raw game dump supplies full attack/defender/time lineage for the routing
corpus.

## High-risk mapping decisions

- LLMail phase-1 `True` is a candidate. Phase-2 `True` and official benign
  controls are dev-test. `False` and `Unclear` annotations are uncertain, not
  benign, because “not confirmed as an attack attempt” does not establish benign
  intent.
- LLMail raw submissions are grouped by phase, team, and challenge level. This
  keeps adaptive attempts against one task together without letting a team that
  participated in phase 2 pull unrelated phase-1 tasks into dev-test.
- Every nonempty HackAPrompt `user_input` is an attack attempt. `correct` records
  success against a particular target and never turns a failed attempt benign.
  Challenge prompts, expected completions, and model completions are not detector
  input.
- Every nonempty Tensor Trust attack record remains an attack attempt regardless
  of game success. Defenses and labelled model outputs are auxiliary. The first
  routing POC groups raw attacks by anonymized attacker, not defense task. Its
  dev-test split is therefore not task-held-out and cannot support an unseen-task
  generalization claim; revisit task grouping for the prospective final test.
- WildJailbreak vanilla harmful is harmful non-injection; adversarial harmful is
  a jailbreak; vanilla benign is benign. Adversarial benign stays auxiliary
  because benign intent does not settle whether jailbreak-like wording is
  instruction subversion.
- For the first proper router, score OASST1 accepted user prompts and
  WildJailbreak adversarial-benign rows as diagnostics only; do not fit on them
  or tune thresholds from them. Audit genuine errors first. If an audited subset
  later enters a training ablation, split it by conversation or base-prompt
  lineage and retain a disjoint diagnostic subset. OASST1 assistant messages
  remain in the `model_output` channel rather than direct-user training.
- WildGuardMix `adversarial` is not an injection label. Its train prompt labels
  are model-produced weak supervision, so all labelled train rows remain
  auxiliary unless a future training-only ablation explicitly selects them.
  Eligible three-human-annotator test rows remain dev-test; unlabelled rows are
  uncertain.
- HarperValleyBank caller and agent segments both use human-corrected transcripts.
  Caller turns enter the `direct_user` channel and agent turns enter `model_output`, while every meaningful turn stays grouped by call ID.
  Empty and marker-only segments are not detector text.
  Speaker IDs, sessions, and task intents are retained, but participant names, task slot values, audio, timing, model dialog acts and emotions, surveys, and machine transcripts are excluded.
  The corpus is simulated, covers only eight banking intents, and was deliberately built with limited vocabulary, so it cannot represent natural financial-agent traffic by itself.
- TAT-QA uses the pinned official GitHub raw JSON because the hosted viewer is not the source of truth for this projection.
  Questions are `direct_user`; report paragraphs and reversible TSV tables are `untrusted_content`.
  Every item in one hybrid context uses the table UID as split lineage because the release does not expose a stable report identifier.
  Official development and test contexts stay in dev-test.
  Answers, derivations, reasoning metadata, and supporting-fact labels never become detector text.
- FinanceBench contributes all 150 public examples to dev-test only.
  Questions and `evidence_text` passages share `doc_name` lineage, and answers, justifications, full-page text, PDFs, results, and vector stores are excluded.
  This is a repeated development diagnostic, not a prospective final test.
- Mind2Web contributes only `confirmed_task` from the 1,009 official training annotations.
  Protected test data, HTML, actions, DOM data, traces, and browser or session artifacts are excluded.
  A local high-precision secret and PII screen runs before routing eligibility; suspicious task text is retained verbatim only in source-level quarantine and is never silently redacted.
  Pattern screening is not proof that every retained task is free of personal information.
- BrowseSafe does not publish payload spans. Positive HTML is retained at
  document granularity and must not be split into all-positive windows.
- Nemotron is positive-only synthetic data. It can measure transfer recall, not
  precision, benign utility, or false-positive rate.
- FalseReject generated prompts, CoCoNot safe-to-comply preference prompts, and
  eligible LMSYS messages use the normal `candidate` role. They are grouped
  across train, validation, and dev-test, while their weak `label_basis` remains
  machine-readable for training ablations and metric slices. LMSYS flagged user
  prompts and their responses remain uncertain. An exact duplicate with an
  official dev-test origin is held out entirely and never also appears in train.
- Taskmaster and Schema-Guided Dialogue retain every non-empty user and
  assistant/system turn. Taskmaster groups shared task-instruction families;
  SGD groups whole conversations. Published task
  instructions, schemas, dialogue frames, and API annotations are not detector
  text. Official Taskmaster-1 and SGD evaluation splits stay in dev-test.
- MASSIVE is limited to the complete en-US release for this first router. Its
  source IDs remain translation-compatible, but adding all 52 parallel
  languages before comparable multilingual attack coverage would make language
  an avoidable label shortcut.
- LMSYS Chatbot Arena is weak benign supervision, not ground truth. English user
  prompts unflagged by OpenAI moderation and both published ToxicChat taggers
  enter as one candidate slice. Paired model outputs enter as a separate weak
  candidate slice inferred from that context; their toxicity remains unknown
  because the published tags do not label responses. Automated positive tags are
  retained only as uncertain metadata: a stratified audit found ordinary
  security discussion, jokes, fiction, and sexual-health questions mixed with
  genuine harmful requests and jailbreaks, including under two-tagger agreement.
  Their paired model outputs remain uncertain because they may refuse or comply.
  The unflagged slice also contains occasional jailbreak-shaped false negatives,
  so the first recipe must report a source-weighted with/without-LMSYS ablation.
  Non-English conversations are omitted. User lineage is grouped by a hash of
  the published anonymized judge ID, never the raw value. User prompts and model
  outputs retain their distinct licenses.
- FalseReject's generated train prompts enter as candidates; its human-annotated
  test prompts enter dev-test. CoCoNot uses the same source-label distinction
  between its candidate preference prompts and human-verified contrast test.
  Only prompt text is detector input. JBB contributes only its curated benign
  goals to dev-test.
Canonical shards retain every valid, non-empty detector-text projection and
available lineage used by morgott. They are standardized projections rather
than byte-for-byte mirrors; exclusions are recorded in the manifest.
“Auxiliary” or “uncertain” means excluded from the default routing supervision,
not deleted.

## Deferred or rejected

| Candidate | Decision |
|---|---|
| WildChat-1M | Stopped weak-label pilot. Accepted model-agreed negatives reduced recall; no rows enter the canonical corpus. |
| PromptShield | Evaluation-only audit. It lacks row-level source/group lineage and aggregates families that overlap active sources. Do not train on this release. |
| Yaklang prompt-injection skill | Taxonomy and scenario reference, not a labelled corpus. Do not vendor or generate rows from it. |
| AgentHarm | Future stateful authorization evaluation; its harmful actions are not prompt-injection labels. |
| SafeDialBench | Deferred pending cleaner grouping and independent controls. |
| AdvBench | Released goals are harmful requests, not prompt-injection positives. |
| Aegis 2.0 | Removed from the active corpus. Its content-safety labels belong to a separate harmfulness task that has no active model recipe. |
| BeaverTails | Content-safety labels do not establish instruction subversion. |
| OR-Bench | Removed after the hard-benign ablation reduced recall without materially improving false positives. Its generated weak labels and missing rewrite-family lineage do not justify a required source. |
| SWE-bench | Deferred until matched same-format attacks and repository-heldout evaluation exist. A large auxiliary-only download should not be required by the canonical build. |
| AgentDojo and AgentDyn | Stateful evaluation only. Their deterministic utility and security outcomes are stronger evidence for banking and tool-use behavior than flattened prompt labels. |
| WASP and LivePI | Browser, messaging, file, and wallet evaluation only. Prefer observable final-state side effects over their model-judged intermediate labels. |
| PIArena | Adaptive cross-benchmark evaluation only after a candidate is frozen; never add its public attack texts to that candidate's training recipe. |
| CyberSecEval prompt injection | Deferred prospective multilingual evaluation after the first recipe; positive-only machine translations are derived lineages, not training rows. |
| JailbreakBench jailbreak artifacts | Deferred method/target-held-out attack evaluation; the separate benign-behavior goals are included. |
| API-Bank | Defer until tool-output diagnostics show a gap. Its generated, context-dependent tool dialogues would add weak supervision where Schema-Guided Dialogue already supplies ordinary API interactions. |
| MInDS-14 plus CyberSecEval multilingual injection | Recommended paired multilingual diagnostic. Keep the benign and positive-only sources separate so language and source cannot become the label. |
| ASPI and Prompt Injection as Role Confusion | Full-system, provenance-aware evaluation only. Flattening context-dependent messages into text labels would create false certainty. |
| tau banking and AgentLAB | Future stateful utility, transaction-invariant, long-horizon, and memory-poisoning evaluation. Prefer deterministic environment outcomes and keep planner, attacker, judge, and search budgets fixed. |
| Prahari Bank Lending | Strong finance-specific matched diagnostic, but gated and explicitly evaluation-only. Do not use for training. |
| PINT | One-shot private external evaluation after model, threshold, and preprocessing are frozen. |
| Mindgard evasion samples | Held-out robustness stress only. Apply deterministic transformations symmetrically to benign and attack controls instead of treating detector-specific evasions as independent training rows. |
| ACL indirect-PIA detection corpus | Reject for ordinary training and headline evaluation. A small payload set is reused across nominal corpora, so payload-family grouping would be mandatory even for a span-localization ablation. |
| FinGuard and Mukta finance injection aggregates | Reject. They combine benign finance data with unrelated attack sources and directly recreate the source-label shortcut. |

## External reference audits

The [Wolf Defender small model card](https://huggingface.co/patronus-studio/wolf-defender-prompt-injection-small) is useful recipe context, not comparable promotion evidence.
Its published strategy supports symmetric transformations across benign and attack rows, hard negatives, counterfactuals, long-context position variation, and aggressive similarity deduplication.
Its binary benign/injection target does not preserve morgott's provenance, harmful-intent, or authorization boundaries, and model-card aggregate metrics do not replace source-heldout and scenario-heldout evaluation on morgott's canonical rows.
Wolf Defender was not added as a training teacher or promoted checkpoint.

The gated [Rogue Security prompt-injections benchmark](https://huggingface.co/datasets/rogue-security/prompt-injections-benchmark) is test-only and is not an independent benchmark for the present corpus.
A local normalized exact-overlap audit matched 2,714 of its 5,000 rows, or 54.28%, against existing morgott canonical sources.
The overlap was dominated by WildJailbreak and WildGuardMix families, with smaller matches to other existing sources.
It must not enter training, threshold selection, or an uncontaminated headline metric.
At most, a frozen future candidate may use the exact-overlap-excluded remainder as a clearly labelled post-hoc diagnostic after near-duplicate and lineage review.

Historical experiment counts are summarized in `reports/model-experiments.md`.
Their runners and generated JSON were removed from the active tree because they
used the old corpus or ended in a stop decision.

## Corpus rules

- There is no row cap. Source sampling or weights are model-recipe decisions.
- Canonical source shards retain repeated valid projections. Exact same-annotation repeats
  project once into views while `origins` retains lineage.
- Conflicting routing annotations and exact/strict-near leakage are quarantined.
- Candidate lineages touching dev-test move to dev-test instead of training.
- Train, validation, and dev-test are development roles. A prospective final
  test does not yet exist.
- Explicit weak labels may enter train, validation, and dev-test, all of which
  are development roles. They must be reported separately and cannot support a
  prospective final-test or production-FPR claim.
- Report source-held-out results before promoting any model.
