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
| Tensor Trust raw | Human game attacks plus auxiliary defenses and model outputs |

The two Tensor Trust adapters intentionally expose different official artifacts:
the compact robustness suites remain a historical development comparison, while
the raw game dump supplies full attack/defender/time lineage for the routing
corpus.

## High-risk mapping decisions

- LLMail phase-1 `True` is a candidate. Phase-2 `True` and official benign
  controls are dev-test. `False` and `Unclear` annotations are uncertain—not
  benign—because “not confirmed as an attack attempt” does not establish benign
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
- BrowseSafe does not publish payload spans. Positive HTML is retained at
  document granularity and must not be split into all-positive windows.
- Nemotron is positive-only synthetic data. It can measure transfer recall, not
  precision, benign utility, or false-positive rate.

Canonical shards retain every valid, non-empty detector-text projection and
available lineage used by Morgott. They are standardized projections rather
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
| Aegis and BeaverTails | Content-safety labels do not establish instruction subversion. |
| PIArena, MCP/Agent benchmarks, AgentDojo | Future stateful evaluation; do not flatten trajectories into independent text rows. |
| CyberSecEval prompt injection | Deferred prospective multilingual evaluation after the first recipe; positive-only machine translations are derived lineages, not training rows. |
| JailbreakBench artifacts | Deferred method/target-held-out evaluation; preserve behavior/method/target/outcome and known HarmBench/AdvBench lineage overlap. |

## Next benign-source pass

Do not add these during the routing-split repair. Audit and version them as a
separate corpus change:

| Source | Proposed role |
|---|---|
| [Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | Ordinary task-dialogue candidate rows grouped by `dialogue_id`; official test stays dev-test. |
| [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) | Finance-language benign candidates; preserve the official test and note the lack of conversation lineage. |
| [FalseReject](https://huggingface.co/datasets/AmazonScience/FalseReject) | Human-validated hard-benign test rows to dev-test; synthetic train remains auxiliary. |
| [CoCoNot](https://huggingface.co/datasets/allenai/coconot) and [JailbreakBench benign behaviors](https://github.com/JailbreakBench/jailbreakbench#accessing-the-jbb-behaviors-datasets) | Small safety-sensitive benign contrasts for dev-test. |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) | Defer until multilingual coverage is needed; group translations by original ID and prevent its scale from defining the corpus. |

Moderation-safe, non-toxic, accepted-chat, or “not injection” labels alone still
do not establish broad benignity.

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
- No model-generated label enters dev-test or supports a production-FPR claim.
- Report source-held-out results before promoting any model.
