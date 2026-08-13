# Semalith review and evaluation protocol

Date: 2026-08-12

Status: research plan only. Semalith access is gated and no Semalith weights have
been downloaded or evaluated. Nothing in this note authorizes model promotion or
changes Morgott's advisory-only trust boundary.

Primary sources:

- [Semalith v1.4 paper](https://arxiv.org/html/2607.22545v1)
- [Semalith v1.5 model card](https://huggingface.co/Tejasvi-addagada/semalith-v1.5)
- [FinProof dataset card](https://huggingface.co/datasets/Zytra/finproof-bench)

## Decision

Evaluate Semalith only after access is granted, and treat the released v1.5
checkpoint as a new artifact rather than assigning it the v1.4 paper's results.
It is a higher-priority scientific baseline than another large generative guard:
it is compact, directly prompt-injection-oriented, and exposes class logits.
Its published metrics do not answer Morgott's low-FPR transfer question, so it
must run on the identical Morgott panel before comparison.

Do not add Semalith or FinProof data to Morgott training. Do not treat Semalith's
BFSI labels as evidence about benign-finance false positives. Morgott detects
instruction subversion; a disallowed, risky, or regulated finance request is not
automatically an injection.

## The two public versions are materially different

| Item | Paper v1.4 | Gated model card v1.5 |
|---|---:|---:|
| Backbone | DeBERTa-v3-base, 184M | 184M checkpoint |
| Training rows | 76,204 from 49 sources | 70,500 |
| Maximum tokens | 256 | 512 |
| Batch size | 32 | 16 |
| Epochs | 6 | 6 |
| Auxiliary weight | grid disclosed; selected value not stated | alpha 0.20 |
| Validation | stratified 5% per class | macro-F1 0.8836 |
| Access | paper and claimed harness | gated, research-access license |

The v1.5 card also repeats the v1.4 RTX 4090 latency numbers even though its
declared sequence cap and training recipe changed. V1.5 512-token throughput is
therefore unverified until a local canary.

The card's HackAPrompt F1 of 0.997 is not false-positive evidence. The reported
1,501-row panel is positive-only: with precision mechanically equal to one,
recall near 0.994 algebraically becomes F1 near 0.997. Morgott's primary
coordinates instead include a benign denominator and TPR at a low FPR.

## Training objective and what it does not solve

The paper uses weighted 22-way cross-entropy plus an auxiliary four-way
super-category cross-entropy:

`loss = fine_label_CE + alpha * super_category_CE`

The fine taxonomy contains BENIGN, nine injection subtypes, general harm, and
eleven BFSI classes. The disclosed class weighting raises BENIGN, agentic,
indirect-injection, and BFSI emphasis; AdamW uses learning rate 2e-5, weight
decay 0.01, gradient clipping at 1.0, 6% warmup, and cosine decay.

This is an interesting architecture lesson, not a checkpoint-selection recipe.
The paper does not specify:

- which epoch/checkpoint was selected or an early-stopping rule;
- a continuous low-FPR operating-threshold protocol;
- probability calibration despite using "calibrated" in the title;
- a sampler or class-balanced batch construction.

Its published binary decision maps the 22-way argmax into broad categories.
Morgott has co-occurring and null labels, so copying a mutually exclusive
22-class head or its weights would violate the current label contract. A future
masked subtype/coarse auxiliary study is plausible, but only as a preregistered
one-variable experiment with explicit Pareto gates.

## Comparison with results already available

These are overlap diagnostics, not same-row head-to-head results. Semalith and
Morgott use different subsets, label mappings, thresholds, and often different
tasks.

| Family | Morgott 512 / 1024 at update 17,000 | Semalith paper/card issue |
|---|---:|---|
| HackAPrompt | 95.04% / 95.08% recall on 55,622 rows | 1,501 positive rows; no FPR |
| Gandalf | 98.21% / 99.10% recall on 223 rows | 1,000 different rows |
| WildJailbreak | 94.25% / 92.82% recall, 0.0417% / 0.0536% FPR on 26,188 mixed rows | 2,000 positives; no matched benign FPR |
| WildGuardMix | 6 / 1 flags on 903 off-target negatives | content-safety F1; different ontology |
| HarmBench | 10.50% / 13.25% off-target flags on 400 harmful non-subversion rows | treated as positive harm recall |
| ToxicChat | 82.35% / 83.53% recall and 6.08% / 7.60% FPR on a 348-row projection | moderation F1 on a different population |
| Benign finance | 0 / 2 flags on 7,043 controls | no equivalent benign-finance claim |

AgentHarm, AdvBench, and BeaverTails are not prompt-injection positives under
Morgott's ontology. They may diagnose harmfulness or authorization policy, but
must not be used to claim instruction-subversion recall.

The paper says evaluation sets were deduplicated from training, but later
reports 169 AgentHarm collisions while the benign panel has only 208 rows. That
requires clarification before treating the zero-false-positive claim as an
independent result. The iterations also explicitly targeted misses on prior
benchmarks, including Mosscap, WildJailbreak, and BFSI panels; those panels are
development evidence rather than prospective final tests.

## FinProof boundary

The live repository describes 11,474 prompts across all access tiers, while the
public Dataset Viewer exposes 2,388 rows: 1,606 attacks and 782 benign rows.
It mixes prompt injection with regulatory and policy-defined BFSI attacks.
Several ordinary user requests can therefore be labelled `attack` without
instruction subversion.

If FinProof is ever scored:

- pin the exact repository revision, subset, tier, and task;
- report it as a separate regulatory-routing diagnostic;
- never merge it into Morgott's 7,043 benign-finance control claim;
- never use its labels to train Morgott's injection head without a new ontology
  review and manifest-verified data rebuild.

## Preregistered evaluation after access

1. Pin the exact v1.5 repository commit, config, tokenizer, file hashes, and
   license. Inspect the state dict and load strictly; do not rely on a
   `strict=False` example without explaining missing or unexpected keys.
2. Use one primary scalar: summed softmax mass over the nine injection classes.
   Treat the auxiliary D-attack probability as diagnostic. Reproduce the
   official attack-plus-harm argmax only as a secondary card check.
3. Score the identical Morgott calibration, canonical dev-test, PromptShield,
   SEP, finance, reserve, and long-code row identities at native first-512
   truncation. Do not tune on external panels.
4. Report AUROC, PR-AUC, descriptive TPR at 1% FPR, the transported component
   threshold, SEP pair ordering, finance flags, reserve attested recall versus
   bare-harmful off-target rate, truncation, throughput, and peak memory.
5. Keep sliding-window maximum scoring as a separate post-primary diagnostic.
6. If the author supplies the exact 1,501 HackAPrompt rows, score both Morgott
   checkpoints and Semalith on those identities for the only valid
   paper-protocol head-to-head.
7. Run exact and near-overlap audits on every supplied evaluation file.
   Training-family overlap remains unknown without the manifest or raw corpus.

No evaluation should be published as a SOTA claim. The currently opened panels
are repeated development evidence, and Semalith's published recall/F1 metrics
are not interchangeable with Morgott's low-FPR operating diagnostics.
