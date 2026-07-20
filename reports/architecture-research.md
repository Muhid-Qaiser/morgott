# Architecture and research direction audit

Research snapshot: July 2026. This is an exploratory decision map, not a claim
that any tested detector is a security boundary. The practical objective is
high precision with useful recall, followed by deterministic containment of
the attacks that every detector will miss.

## Bottom line

The most practical near-term system is a small cascade with only one learned
stage in the request path:

```text
trusted channel/provenance
        |
        +--> cheap high-precision sensor --> shadow/review signal
        |
untrusted data --> structured data/control separation
        |
planner proposes typed action
        |
deterministic capability + data-flow monitor --> allow or deny side effect
```

The detector should answer “does this text appear to attempt instruction
subversion?” It should not decide whether a tool call, network request, memory
write, or data release is authorized. A remote LLM/ReAct reviewer is useful for
offline weak labels and error analysis, but the measured seconds of latency and
provider unavailability make it a poor default hot-path gate.

## What the local experiments establish

- The character 3–5 gram control remains the best current precision-first
  request sensor. At the 85% validation-precision profile it retains 34/66
  validation attacks with 4/7,120 validation false signals and 0/4,208 alerts
  on the external hard-negative aggregate. These are source-mixture metrics,
  not production precision or FPR.
- Frozen E5, frozen ModernBERT mean/CLS, PIGuard, ProtectAI DeBERTa, and two
  community ModernBERT-family checkpoints do not beat that control at the same
  operating-point contract.
- The controlled one-seed end-to-end screen did not promote either base model.
  ModernBERT underfit at 6/66 validation TP and 1/7,120 FP. DeBERTa reached
  36/66 TP and 6/7,120 FP, but caught only 5/4,136 multi-turn attacks and added
  4/4,208 hard-negative alerts. Its stricter 95% profile merits a fuller rerun.
- Off-the-shelf model-card thresholds over-defend this distribution. Retuning
  them to high precision often collapses recall. Adding their alerts with an OR
  would increase review volume without evidence of complementary errors.
- A one-shot OpenRouter reviewer costs seconds rather than milliseconds and can
  be unavailable. It stays out of the request path.
- The reference-monitor simulation is the strongest result: it blocks every
  unauthorized side effect even when the input filter is assumed to miss,
  whereas the input-filter-only ablation commits all eight attacks.

Exact curves, revisions, overlap caveats, latency, and VRAM are in
[`model-experiments.md`](model-experiments.md).

## Encoder candidates

### DeBERTa-v3

DeBERTa-v3 is the sample-efficient incumbent. Its replaced-token-detection
pretraining is explicitly designed to improve pretraining efficiency and NLU
quality. That is attractive with only 245 fit positives in the current grouped
training split. Existing prompt-guard systems also provide evidence that small
DeBERTa-family encoders can be useful when the data, objective, and tokenizer
are purpose-built—not when a generic checkpoint is accepted at its card
threshold. See the [DeBERTa-v3 paper](https://arxiv.org/abs/2111.09543).

### ModernBERT

ModernBERT is the efficiency/long-context candidate, not automatically the more
accurate model. It was pretrained natively to 8,192 tokens and is designed for
fast, memory-efficient classification and retrieval. That makes it the better
platform for attacks buried deep in documents, provided end-to-end fine-tuning
first demonstrates useful 512-token separation. Neither the frozen probe nor
the small end-to-end pilot did. See the
[ModernBERT paper](https://arxiv.org/abs/2412.13663).

### Controlled screen complete

The repository compared base ModernBERT and base DeBERTa end to end at 512
tokens using the same grouped fit rows, untouched validation rows, class-
weighted cross entropy, seed, optimizer budget, effective batch, precision/FPR
grid, FP16 policy, and evaluation suites. It used pinned safetensors and no
remote code. Both fit comfortably on the 6 GB RTX 4050. ModernBERT underfit;
DeBERTa was competitive on validation but did not generalize to multi-turn or
hard-negative controls. Neither is promoted, and one seed/epoch does not rank
the architectures intrinsically.

DeBERTa's 95% profile—25/66 validation TP, 1/7,120 FP, and 0/4,208 hard
FP—earns only a predeclared fuller three-seed rerun. Then test hypotheses one at
a time:

1. Three seeds and paired group bootstrap intervals.
2. Benign out-of-distribution energy penalty or focal-loss ablation.
3. Whitespace, Unicode, token-fragmentation, and encoding perturbations grouped
   by the original example.
4. Native 2K ModernBERT versus 512-token windows with 128-token overlap on
   positive position-stress and length-matched clean documents.
5. Separate tiny direct-user and untrusted-content heads routed by immutable
   provenance.

Meta's Prompt Guard 2 card is useful design evidence here: it attributes its
precision improvements to a custom benign-OOD energy objective, expanded data,
and adversarial tokenization, and uses 22M/86M DeBERTa-family encoders. Its
reported private-benchmark numbers are not comparable to this corpus, and the
weights are gated, so the model remains out of scope. See the
[Prompt Guard 2 model card](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M).

The Sentinel paper provides evidence that a fully fine-tuned ModernBERT-large
can be competitive for this task, but its best evidence depends on private data
and an internal test set, and the current checkpoint is gated. It motivates the
fair local ModernBERT run; it does not settle it. See
[Sentinel](https://arxiv.org/abs/2506.05446).

## Data decisions

### Already useful

- **deepset prompt-injections** is already pinned and used for direct fit plus
  official-test development evidence. It is small (662 rows) and its semantics
  are application-dependent, so it cannot carry the model alone. Preserve its
  official split and label provenance. The card is
  [Apache-2.0](https://huggingface.co/datasets/deepset/prompt-injections).
- **Tensor Trust** is already pinned as evaluation-only. The two official
  robustness suites contribute 908 unique attack strings and 1,346 attacks in
  defense-prompt context. The raw game dump is not a ready-made binary corpus:
  it contains attempts, outcomes, defenses, repeated attackers/defenders, and
  task secrets. Any later raw-data experiment must distinguish attempt from
  success and group by defense/player/task. The repository has no explicit
  standard data license, so it is not used for training. See the
  [Tensor Trust paper and release](https://tensortrust.ai/paper/).
- **WildChat-1M** supplies broader normal-chat candidates only through strict
  model-only weak supervision. The completed 5k pilot accepted 2,430 weak
  negatives but worsened direct and indirect recall at all declared precision
  profiles, so it stopped and none entered the baseline. Source toxicity is not
  an injection label, and agreement is not ground truth.

### Newly assessed

- **HackAPrompt** contains over 600k competition attempts against fixed puzzle
  prompts. The `correct` field means the target model produced the expected
  puzzle output; it is attack success, not a benign/malicious label. Failed
  attempts are still attack attempts. If access is explicitly accepted later,
  use `user_input` only, preserve success as a separate axis, group by
  challenge/submission/user lineage, and exclude full system prompts and model
  completions from the detector input. For now the official Hugging Face
  release requires contact-sharing acceptance, so it is skipped under the
  ungated-only rule. See the [dataset card](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset)
  and [EMNLP paper](https://aclanthology.org/2023.emnlp-main.302/).
- **Yaklang's prompt-injection skill** is an MIT-licensed attack playbook, not a
  labelled dataset. It is useful as a taxonomy checklist for direct override,
  indirect/RAG injection, tool abuse, exfiltration, MCP, encoding, and
  multi-turn stress cases. Training directly on its handful of canonical
  payloads would encourage lexical overfit. Keep it reference-only, pin commit
  `c9a4b9ee8645eb60763eb4eef172f1ecb0a5b3e8`, and group any future generated
  variants by technique and base payload. See the
  [upstream repository](https://github.com/yaklang/hack-skills).
- **NVIDIA Nemotron Agentic IPI** is a pinned, CC-BY-4.0, positive-only
  synthetic evaluation suite. Both retained sensors missed all 676 exact-unique
  injection texts at their default thresholds. It supplies valuable domain,
  injection-vector, impact, and target-tool lineage, but no benign controls;
  blanket training on it would invite synthetic shortcut learning. See the
  [dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1)
  and [local audit](nemotron-agentic-ipi.md).
- **PromptShield** is a useful 23,516-row evaluation-only stress test, not safe
  new training data. Its rows have only `prompt` and `label`, while the paper
  describes an aggregate of public corpora and attack strategies with active
  source overlap. At the locked 85% character profile, recall is 194/6,486 and
  false signals are 240/17,030; no positive longer than 256 whitespace tokens
  is caught. See the [dataset](https://huggingface.co/datasets/hendzh/PromptShield),
  [paper](https://arxiv.org/abs/2501.15145), and
  [local audit](../experiments/promptshield_audit/REPORT.md).

## What current defense research changes

Classifier-only mitigation is insufficient. The stronger system work changes
the architecture around the model:

- [StruQ](https://arxiv.org/abs/2402.06363) separates trusted instructions and
  untrusted data into distinct channels and trains the model not to execute
  instructions from the data channel.
- [CaMeL](https://arxiv.org/abs/2503.18813) extracts control/data flow from the
  trusted query and enforces capabilities around tool calls. It reports 77% task
  utility with its security construction versus 84% undefended on AgentDojo;
  the important lesson is the security layer, not copying its exact stack.
- [IPIGuard](https://aclanthology.org/2025.emnlp-main.53/) plans a tool
  dependency graph before exposure to untrusted tool results, constraining
  injected text from inventing new action paths.
- [AgentDojo](https://arxiv.org/abs/2406.13352) evaluates task utility and
  security outcomes in 97 realistic tasks and 629 security cases. This is a
  better P3 target than classifier recall alone.
- [PIArena](https://aclanthology.org/2026.acl-long.1533/) is the strongest
  current warning against selecting a detector from static text suites alone.
  Its cross-benchmark study finds limited defense generalization and much
  higher attack success when the attacker adapts to the defense; detector and
  sanitization gains often trade against no-attack utility. Its framework is a
  candidate remote-GPU P3 harness, but its target-model/LLM costs make it a poor
  local P0 dependency.
- [Rethinking prompt-injection assessment](https://aclanthology.org/2026.findings-acl.1191/)
  finds that ambiguous in-the-wild target tasks remain harder after defenses,
  and that instruction form—not only content—changes attack success. This
  supports grouping syntactic mutations and measuring task utility rather than
  expanding a keyword-heavy positive corpus.
- [WAInjectBench](https://arxiv.org/abs/2510.01354) shows why the current text
  scope must remain explicit: web-agent detectors can work on overt text or
  visible image attacks yet fail on attacks without explicit instructions and
  imperceptible image perturbations. Multimodal/browser evidence is a separate
  future track; text-detector recall must not be presented as web-agent safety.
- A 2026 [Prompt Control-Flow Integrity](https://arxiv.org/abs/2603.18433)
  preprint also argues for provenance/priority-aware middleware. Its perfect
  result is on a custom synthetic/semi-realistic benchmark and is not treated
  as independent proof, but the design direction agrees with the stronger
  peer-reviewed/system work above.

PIGuard's [NotInject work](https://aclanthology.org/2025.acl-long.1468/) is the
most relevant warning for the product objective: trigger-heavy benign prompts
can cause severe over-defense. That is why this repository retains NotInject,
ordinary security discussions, harmful-but-non-injection prompts, broad chat,
and precision-first operating profiles.

## Practical experiment order

1. Preserve the completed one-epoch ModernBERT/DeBERTa screen. Promote neither;
   only DeBERTa's 95% profile earns a frozen-protocol, three-seed continuation.
2. Keep the completed WildChat pilot stopped. Its 2,430 accepted weak negatives
   reduced multi-turn and indirect recall at the precision-first profile, so
   collecting 20k/50k under the same recipe is not justified.
3. If the fuller DeBERTa control is stable, test OOD-energy and
   tokenizer-perturbation ablations separately. Otherwise invest in matched
   indirect data and structural controls rather than a larger encoder.
4. Add positive long-document position stress before spending on 2K/8K
   training. Compare native long context with overlapping windows at the
   document level because max-window aggregation raises false-positive risk.
5. Implement immutable provenance segments and strengthen the capability
   monitor with task/user binding, expiry, egress labels, and memory-write
   quarantine.
6. Move to AgentDojo-style end-to-end evaluation: utility, attack success,
   unauthorized side effects, secret exfiltration, latency, and review load.
   Add defense-adaptive PIArena-style attacks only after the static end-to-end
   harness is reproducible; keep multimodal web-agent attacks as a distinct
   threat model and benchmark.

On a remote GPU, the first useful upgrade is enough memory to run three-seed
full fine-tuning and long-context ablations without changing the experiment
contract. It is not a reason to jump to decoder judges, ensembles, Longformer,
or ReAct. Those add cost and attack surface before the simpler candidates have
earned continuation.
