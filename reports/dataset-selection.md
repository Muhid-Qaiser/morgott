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
| [SWE-bench Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) | Human-verified software issue statements held entirely in dev-test as a repository-grouped long-benign FPR slice after local secret and PII quarantine |
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
- Agentic Prompt Injection Boundary Pairs remain auxiliary matched development data.
  The official test, validation, and every train-split scenario have now influenced normalization, prompt, threshold, or linear-gate decisions.
  A word model fitted on its first 20 selected train scenarios passed the last 15-scenario block, but that result is within-source synthetic transfer and does not authorize adding the source to a general router recipe.
  Treat all source scenarios as consumed and use a different lineage for the next matched-boundary claim.
- Operant AI prompt-injection false-positive data remains an external hard-benign diagnostic rather than a routing source.
  Its 571 privacy-filtered exact-unique rows have all been consumed by the frozen linear-transfer decision, most rows are mined or generated probes, and only 39 are marked manually reviewed.
  The source-specific linear gate failed its English admission cap, so these rows do not authorize fitting, threshold selection, or a production false-positive claim.
- AgentAbstain remains an external hard-benign direct-user diagnostic rather than a routing source.
  Pair-wide privacy exclusion and exact and normalized deduplication retained 348 unique instructions from 231 complete act and abstain pairs, with zero overlap against the complete retained fit population.
  Its outcomes describe whether an agent should act or abstain under task and environment semantics, not whether the user instruction is an injection or broadly safe.
  The fixed cascade has consumed this panel and exceeds both its aggregate and high-stakes false-positive caps even if the one failed provider review is assumed clear, so do not fit, retune, or claim a production false-positive rate from these rows.
- SafeClawBench remains an external task-relative agent-security diagnostic rather than a routing source.
  Privacy screening and complete-fit overlap filtering retained 89 DPI prompts and 94 ADI controls from its pinned Semantic Core release.
  The text-only projection omitted the source scenario, safe behavior, success predicate, tools, and executable state, so the family labels cannot establish a context-free production detector target.
  Both prompts and every retained row are now consumed development evidence; do not fit, retune, or reopen this panel after its direct-user prompt and threshold rejection.
- BFCL v4 live remains an external direct-user control diagnostic rather than a routing source.
  The pinned six-file projection retains 2,050 privacy-screened, normalized-unique, complete-fit-disjoint last-user messages from official real-world function-calling tasks.
  BFCL does not explicitly adjudicate injection, and the projection omits prior conversation and function schemas, so these rows are non-subversion controls rather than verified benign labels or complete task contexts.
  The unchanged cascade passes its frozen observed-rate and review-load gates, but the 64-word-and-longer slices remain weak and the 95% Wilson upper bound exceeds 1%.
  Treat the complete panel as consumed development evidence and do not fit, retune, or make a production false-positive claim from it.
- The ACL Inj-SQuAD and Inj-TriviaQA release remains an external repeated-payload diagnostic rather than a routing source.
  Its 300 generated payload families each repeat across three documents in each source, so all six repetitions require one lineage group.
  The first 20 payload families have influenced local window design and a task-conditioned reviewer canary, while the remaining 280 families have influenced the frozen local-gate rejection but were not remotely reviewed.
  The canary's post-hoc score scale does not authorize threshold selection, another review pass, training, or an independent-generalization claim from this source.
- SecAlign's StruQ test construction remains an external structured-query diagnostic rather than a routing source.
  After empty-output exclusion, local privacy screening, and whole-pair fit-overlap removal, 197 Alpaca Farm instruction/input pairs were frozen into 40 calibration and 157 evaluation pairs.
  Its ignore and completion attacks all target one fixed synthetic output, so the successful task-conditioned review result does not authorize training, a production false-positive claim, or transfer claims for natural tool and retrieval content.
- InjecAgent remains an external tool-output diagnostic rather than a routing source.
  Its 1,054 complete user-and-attacker pairs, 62 varied goals, and 17 user tools expose a response-only miss and support task-conditioned ranking, but all text is public synthetic development data and only 17 clean templates exist.
  The source is fully consumed for detector and task-conditioned decisions, so neither its post-hoc lower threshold nor its attack text may authorize fitting or a production false-positive claim.
- API-Bank remains an external clean tool-output diagnostic rather than a routing source.
  Its full-block review was invalid as a runtime projection because it mixed trusted API call syntax and arguments with the returned value, while output-only review reduced but did not eliminate threshold-`0.3` false positives.
  Both projections and the post-hoc threshold grid are consumed development evidence, so its generated dialogue must not enter fitting or support a production false-positive claim.
- AgentDyn remains an external stateful benchmark whose fixed important-instructions text arm is now also consumed reviewer evidence.
  The frozen `0.5` candidate flagged all 560 task-and-goal attacks after operational completion, but the explicit attack-only panel supplies no clean denominator, adaptive attack, tool execution, or authorization outcome.
- AgentPIMA remains an external matched progressive-content diagnostic rather than a routing source.
  Its selected defense subset supplied 112 clean tasks and 672 matched attacks across 14 attack families, but the source is synthetic, highly templated, and attached to an anonymized submission with limited independent adoption.
  The prospectively frozen `0.5` transfer and sealed threshold diagnostic are consumed evidence that reject this all-row task-conditioned reviewer; do not fit the source, retune on it, or claim representative false-positive behavior from it.
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
- SWE-bench Verified contributes only official-test `problem_statement` text and retains repository, instance, commit, time, version, and difficulty lineage.
  Every retained task stays in dev-test and is grouped by repository; patches, test patches, test names, hints, and repository contents are excluded.
  Human review establishes that each row is a legitimate solvable software task, which is the source authority for its routing label `0`; this is not merely an absence-of-attack assumption.
  The source is not safety-annotated, so use it only for repository- and length-sliced legitimate-workload diagnostics and do not interpret it as exhaustive proof that every string is harmless in every context.
  The same local secret and PII screen used for Mind2Web quarantines suspicious issue text without redaction.
  The complete retained slice has now influenced the frozen local-gate rejection and is consumed development evidence; do not tune a threshold or run its stopped reviewer branch post hoc.
  Training, threshold selection, aggregate performance claims, and long-context robustness claims remain out of scope until a separately sourced same-format attack arm exists.
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
| PromptShield | Excluded from the canonical corpus because it lacks row-level source/group lineage and aggregates families that overlap active sources. A bounded 2026-07-28 owner-authorized experiment fits only the release's train split outside the corpus; validation selects checkpoints, test remains PromptShield-internal source-disjoint development, and cross-corpus overlap prevents a source-OOD claim for the complete fit. No result supports promotion or a final-test claim. |
| Yaklang prompt-injection skill | Taxonomy and scenario reference, not a labelled corpus. Do not vendor or generate rows from it. |
| AgentHarm | Future stateful authorization evaluation; its harmful actions are not prompt-injection labels. |
| SafeDialBench | Deferred pending cleaner grouping and independent controls. |
| AdvBench | Released goals are harmful requests, not prompt-injection positives. |
| Aegis 2.0 | Removed from the active corpus. Its content-safety labels belong to a separate harmfulness task that has no active model recipe. |
| BeaverTails | Content-safety labels do not establish instruction subversion. |
| OR-Bench | Removed after the hard-benign ablation reduced recall without materially improving false positives. Its generated weak labels and missing rewrite-family lineage do not justify a required source. |
| Broader SWE-bench use | The small Verified problem-statement projection is included only as a dev-test benign denominator. Training, threshold selection, non-Verified tasks, repository artifacts, and any claim that requires matched same-format attacks remain deferred. |
| [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat/tree/f66cca95b14caaa4177f7ed5eaa424608dadcffa) | Completed local-only coding-agent traffic proxy, not benign ground truth or representative production traffic. The privacy-screened, fit-disjoint earliest-prompt projection rejects the registered cascade at 1.133% local-high session load and 52.255% candidate-review load, with zero provider calls and no prompt text retained in artifacts. Treat the panel as consumed development evidence; do not fit it, retune on it, or report its restriction load as a false-positive rate. See the [traffic research report](swe-chat-traffic-research.md). |
| [share-codex](https://huggingface.co/datasets/nmuendler/share-codex/tree/3d8b1397c72dbfbf8b04f518064e2c99dde84ca0) | Deferred ungated coding-agent traffic fallback. Its 4,333 sessions are smaller than SWE-chat, contributor consent and public-repository restriction remain unproven, participant identity is absent, repository lineage is only a surrogate, and its manifest records a nonzero secret-scan gate. Permit only a quarantined local stress sample after a fresh privacy scan, never benign labels, FPR, representative traffic, fitting, or evaluation evidence; see the [provenance audit](share-codex-provenance-audit-2026-08-04.md). |
| AgentDojo and AgentDyn | Stateful evaluation remains preferred. AgentDyn's fixed important-instructions text arm is consumed attack-only reviewer evidence; deterministic utility and security outcomes remain stronger evidence than flattened prompt labels. |
| WASP and LivePI | Browser, messaging, file, and wallet evaluation only. Prefer observable final-state side effects over their model-judged intermediate labels. |
| PIArena | Completed static matched development evaluation only. The balanced 260-task clean-plus-four-attack panel passes every quality gate after a bounded transport retry, but the first run has 37 failed reviews and twelve incomplete artifacts. Treat all selected static rows as consumed, keep them out of training and threshold selection, and reserve adaptive search for a separately frozen future test. |
| Adaptive Adversaries | Completed winning-transcript diagnostic only. The pinned release has 121 winner flags while the paper reports 78 genuine winning turns, and the source does not expose that filtered subset. The full-input projection also mixes trusted outer-task text with the attacker-controlled field. Keep it out of training and threshold selection, and require field-level provenance plus complete action and utility oracles before reuse. |
| CyberSecEval prompt injection | Completed paired multilingual development diagnostic only. The fixed cascade recalled 45.38% of retained English attacks and 43.57% of machine translations, so the source is consumed and must not select another candidate. |
| JailbreakBench jailbreak artifacts | Completed local preflight for the direct-user high-review candidate only. Full-fit overlap removed 88 of 100 behavior families, the retained 46-prompt attack arm put 31 documents in the high zone, and the remote branch stayed unopened because the independent benign arm had no high-zone error. |
| SWE-Lancer individual-contributor tasks | Completed sanitized direct-user high-review preflight only. The retained 197-task benign arm had zero high-zone documents, so it cannot justify extending reviewer admission and is now consumed development evidence. |
| API-Bank | Completed external clean tool-output diagnosis only. Output-only projection is the correct boundary, but both projections and the post-hoc grid are consumed; do not fit its generated dialogue or claim production FPR from it. |
| MInDS-14 plus CyberSecEval multilingual injection | Recommended paired multilingual diagnostic. Keep the benign and positive-only sources separate so language and source cannot become the label. |
| ASPI and Prompt Injection as Role Confusion | Full-system, provenance-aware evaluation only. Flattening context-dependent messages into text labels would create false certainty. |
| tau banking and AgentLAB | Future stateful utility, transaction-invariant, long-horizon, and memory-poisoning evaluation. Prefer deterministic environment outcomes and keep planner, attacker, judge, and search budgets fixed. |
| Prahari Bank Lending | Strong finance-specific matched diagnostic, but gated and explicitly evaluation-only. Do not use for training. |
| PINT | One-shot private external evaluation after model, threshold, and preprocessing are frozen. |
| FORCE-Bench | Completed external benign-finance diagnostic only. Its 251 public tasks are now consumed development evidence and must not enter fitting before a separately frozen replacement exists. |
| [CrackedPDFs](https://github.com/volkthienpreecha/crackedpdfs/releases/tag/v1.0.0-paper) | Its public MIT-licensed release and [Zenodo archive](https://doi.org/10.5281/zenodo.21735803) now include the complete PDFs, code, frozen features, grouped splits, hashes, and evaluator, and the pinned reproduction regenerates its result table byte-for-byte; keep it out of canonical text fitting because every PDF is synthetic and one page, its label depends on structure and visibility that flattening destroys, and its frozen feature table contains no extracted text, so reserve it for a future structure-aware PDF-ingestion evaluation rather than a long-benign or natural-document claim. |
| [OpenRAG-Soc](https://arxiv.org/abs/2601.10923) | The WWW 2026 paper describes paired social-web carriers and a real-web stress set, but the official paper records expose no versioned corpus, source registry, code, license, or evaluator and no author release is publicly discoverable; do not reconstruct the benchmark from prose, and reconsider it only after an official versioned artifact appears. |
| LongBench v2 | Completed matched long-context development preflight only. The fit-disjoint 59-pair panel had nine local-high attacks but no local-high clean artifact, so it stopped before OpenRouter and cannot select the bounded reviewer candidate. Keep it out of fitting and treat it as consumed. |
| LongBench Chinese tasks | Completed matched long-context development evaluation only. The fit-disjoint 100-pair panel passed its local coverage gate, but the full-context plus top-eight candidate restricted 11 clean controls against a maximum of 2 and rejected itself. Keep all selected contexts out of fitting and threshold selection and treat them as consumed. |

The PromptShield exception does not change the canonical source decision.
Its rows stay under ignored experiment artifacts, its train split is leakage-filtered against held-out text, and its test split remains deliberately source-disjoint according to the paper.
The result must be called PromptShield-internal source-disjoint development performance, not IID, complete-fit source-OOD, or independent transfer evidence.
| Mindgard evasion samples | Held-out robustness stress only. Apply deterministic transformations symmetrically to benign and attack controls instead of treating detector-specific evasions as independent training rows. |
| ACL indirect-PIA detection corpus | Completed provider-free span-localization diagnostic only; reject for ordinary training and headline evaluation. All six repetitions of each exact payload stayed grouped, 20 payload families were consumed by window development, and the remaining 280-family frozen panel rejected the local gate. The repeated generated payloads and source-labelled construction do not support a final-test, prevalence, or training claim. |
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

The [Prompt Overflow paper](https://arxiv.org/abs/2605.23196) is architectural evidence rather than a new corpus source for Morgott.
Its released artifact omits the frozen generated panel and result ledger, does not pin the underlying Hugging Face revisions, and loads the same Rogue Security benchmark whose present rows overlap the canonical corpus heavily.
Do not ingest or reconstruct its generated examples from scripts, and do not cite the paper's detector rates as Morgott evaluation evidence.
The artifact audit and the separate fit-disjoint long-context experiment are recorded in [the long-context reviewer report](long-context-reviewer-research.md).

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
