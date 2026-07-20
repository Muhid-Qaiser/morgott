# Model experiment decision ledger

Generated 2026-07-20. Every threshold below is selected only on the same
deterministic grouped validation split. The product preference is precision
first, so the primary comparison uses a minimum 85% observed validation
precision; 80%, 90%, and 95% profiles are retained in the result JSON. The
0.1%, 0.5%, 1%, 2%, and 5% validation-FPR points are diagnostics, not mandatory
gates. Neither constraint transfers automatically to product traffic or proves
a production precision/FPR. Classifier outputs remain shadow signals;
authorization is enforced separately by the reference monitor.

## Precision-first direct-user comparison

Cells are `true positives / positives; false positives / negatives`. “Not
attained” means no observed validation threshold met even the requested
precision floor; it is not silently replaced with a looser threshold. JOT means
JailbreaksOverTime, whose source-negative labels are known to be noisy.

| Candidate at 85% validation precision | Validation | ToxicChat | deepset | Multi-turn | JOT | Hard FP /4,208 |
|---|---:|---:|---:|---:|---:|---:|
| Character n-gram control | 34/66; 4/7,120 | 44/73; 18/4,630 | 12/60; 0/56 | 908/4,136 | 3,203/3,901; 81/18,195 | 0 |
| Frozen E5 + linear head | 18/66; 3/7,120 | 29/73; 10/4,630 | 14/60; 0/56 | 0/4,136 | 1,443/3,901; 45/18,195 | 0 |
| Frozen ModernBERT mean | 8/66; 1/7,120 | 18/73; 7/4,630 | 3/60; 0/56 | 0/4,136 | 1,207/3,901; 69/18,195 | 1 |
| Frozen ModernBERT CLS | 1/66; 0/7,120 | 1/73; 1/4,630 | 0/60; 0/56 | 0/4,136 | 249/3,901; 7/18,195 | 0 |
| PIGuard DeBERTa | 19/66; 3/7,120 | 4/73; 5/4,630 | 10/60; 0/56 | 0/4,136 | 32/3,901; 17/18,195 | 11 |
| ProtectAI DeBERTa | not attained | — | — | — | — | — |
| SiberianCat ModernBERT | not attained | — | — | — | — | — |
| Wolf mmBERT-small shape | not attained | — | — | — | — | — |

The character control retains 51.5% validation recall at the default profile;
the qualifying frozen/checkpoint candidates retain 1.5–28.8%, and none catches
a multi-turn example. The 85% validation target also does not transfer as a
guarantee: ToxicChat precision is 71.0% for the character control, 74.4% for
E5, 72.0% for ModernBERT mean, and 44.4% for PIGuard. The profile is therefore
useful for choosing a shadow-review tradeoff, not for blocking or a product
precision claim.

## Controlled end-to-end base-encoder screen

Base ModernBERT and DeBERTa-v3 were then fine-tuned end to end under one shared
protocol: the same 8,408 ordered fit rows, all 245 fit positives, untouched
7,186-row grouped validation partition, masked-mean 768-to-2 head, class-weighted
cross entropy, seed 42, one epoch, 512 tokens, AdamW at 2e-5, effective batch 32,
gradient checkpointing, and CUDA FP16 with FP32 loss and optimizer state.

| Candidate at 85% validation precision | Validation | ToxicChat | deepset | Multi-turn | JOT | Hard FP /4,208 |
|---|---:|---:|---:|---:|---:|---:|
| Character n-gram control | 34/66; 4/7,120 | 44/73; 18/4,630 | 12/60; 0/56 | 908/4,136 | 3,203/3,901; 81/18,195 | 0 |
| End-to-end ModernBERT-base | 6/66; 1/7,120 | 24/73; 5/4,630 | 0/60; 0/56 | 0/4,136 | 2,358/3,901; 32/18,195 | 0 |
| End-to-end DeBERTa-v3-base | 36/66; 6/7,120 | 56/73; 21/4,630 | 19/60; 0/56 | 5/4,136 | 3,413/3,901; 93/18,195 | 4 |

ModernBERT trained in 330.2 seconds with 2,955 MiB peak allocated VRAM and
9.1/14.2 ms batch-1 p50/p95 latency. It underfit this protocol and is not a 2K
candidate yet. DeBERTa took 364.8 seconds, 3,547 MiB, and 10.8/13.9 ms. Its
small validation gain did not transfer to multi-turn or hard negatives, and its
0.9941 threshold is near score saturation, so it is not promoted.

DeBERTa's 95% profile is the one useful continuation signal: 25/66 validation
TP with 1/7,120 FP, 0/4,208 hard FP, and 48/73 ToxicChat TP with 5/4,630 FP at
threshold 0.9990. This remains a one-seed observation. A predeclared fuller,
three-seed run must test calibration stability before any deployment choice.

## Validation-FPR diagnostic grid

Each cell is `validation TP / validation FP | hard-negative FP`; denominators
are 66 attacks, 7,120 validation negatives, and 4,208 external hard negatives.

| Candidate | 0.1% | 0.5% | 1% | 2% | 5% |
|---|---:|---:|---:|---:|---:|
| Character n-gram control | 35/7 \| 0 | 49/35 \| 17 | 54/71 \| 64 | 61/142 \| 121 | 64/356 \| 305 |
| Frozen E5 + linear head | 24/7 \| 4 | 39/35 \| 22 | 47/71 \| 58 | 55/142 \| 145 | 58/356 \| 379 |
| Frozen ModernBERT mean | 11/7 \| 4 | 35/35 \| 25 | 44/71 \| 50 | 52/142 \| 88 | 62/356 \| 182 |
| Frozen ModernBERT CLS | 6/7 \| 8 | 21/35 \| 31 | 28/71 \| 57 | 36/142 \| 102 | 51/356 \| 215 |
| PIGuard DeBERTa | 34/7 \| 29 | 47/35 \| 90 | 51/71 \| 165 | 56/142 \| 267 | 61/356 \| 522 |
| ProtectAI DeBERTa | 0/1 \| 0 | 4/28 \| 16 | 6/71 \| 48 | 18/142 \| 99 | 24/356 \| 240 |
| SiberianCat ModernBERT | 0/7 \| 6 | 8/34 \| 35 | 8/71 \| 64 | 12/142 \| 117 | 20/356 \| 260 |
| Wolf mmBERT-small shape | 0/7 \| 1 | 0/35 \| 13 | 4/71 \| 47 | 21/142 \| 120 | 32/356 \| 307 |

Relaxing the FPR diagnostic does not reverse the decision. The character model
keeps the best validation-recall/hard-negative tradeoff at most points. Frozen
ModernBERT mean has fewer hard-negative alerts at some broader points, but also
lower validation recall and zero multi-turn recall. ProtectAI and the two
ModernBERT-family checkpoints recover recall only after admitting many
validation and external false signals.

## Historical 0.1% direct-user diagnostic

Cells are `true positives / positives; false positives / negatives`. JOT means
JailbreaksOverTime. The hard-negative aggregate has 4,208 rows.

| Candidate | Calibration | ToxicChat | deepset | Multi-turn | JOT | Tensor Trust | Hard FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Character n-gram control | 35/66; 7/7,120 | 46/73; 18/4,630 | 14/60; 0/56 | 1,934/4,136 | 3,311/3,901; 91/18,195 | 304/908 | 0/4,208 |
| Frozen E5 + linear head | 24/66; 7/7,120 | 38/73; 19/4,630 | 16/60; 0/56 | 9/4,136 | 2,260/3,901; 83/18,195 | 103/908 | 4/4,208 |
| Frozen ModernBERT mean | 11/66; 7/7,120 | 26/73; 21/4,630 | 4/60; 0/56 | 0/4,136 | 2,053/3,901; 189/18,195 | 87/908 | 4/4,208 |
| Frozen ModernBERT CLS | 6/66; 7/7,120 | 17/73; 15/4,630 | 1/60; 0/56 | 0/4,136 | 899/3,901; 92/18,195 | 75/908 | 8/4,208 |
| PIGuard DeBERTa | 34/66; 7/7,120 | 32/73; 14/4,630 | 14/60; 0/56 | 0/4,136 | 1,957/3,901; 81/18,195 | 450/908 | 29/4,208 |
| ProtectAI DeBERTa | 0/66; 1/7,120 | 0/73; 0/4,630 | 0/60; 0/56 | 0/4,136 | 0/3,901; 0/18,195 | 19/908 | 0/4,208 |
| SiberianCat ModernBERT | 0/66; 7/7,120 | 0/73; 5/4,630 | 0/60; 0/56 | 0/4,136 | 8/3,901; 74/18,195 | 25/908 | 6/4,208 |
| Wolf mmBERT-small shape | 0/66; 7/7,120 | 0/73; 0/4,630 | 0/60; 0/56 | 0/4,136 | 0/3,901; 21/18,195 | 0/908 | 1/4,208 |

At this stringent diagnostic the character control had useful but incomplete
transfer and zero observed FPs on the broad hard-negative aggregate. It remains
advisory; the precision profiles above are now the primary review tradeoff, and
neither family is sufficient for a security boundary.

## Untrusted-content channel

Calibration cells are `TP/84; FP/142`. BIPIA clean is false positives; all
other evaluation cells are true positives.

| Candidate | Calibration | BIPIA payload /125 | BIPIA context /375 | BIPIA clean FP /167 | Tensor context /1,346 |
|---|---:|---:|---:|---:|---:|
| Character channel head | 56; 0 | 84 | 252 | 2 | 354; 903 with direct fallback |
| Frozen E5 | 29; 0 | 98 | 297 | 10 | 991 |
| Frozen ModernBERT mean | 24; 0 | 0 | 72 | 1 | 42 |
| Frozen ModernBERT CLS | 18; 0 | 25 | 74 | 7 | 47 |
| PIGuard DeBERTa | 73; 0 | 72 | 344 | 4 | 1,315 |
| ProtectAI DeBERTa | 0; 0 | 3 | 30 | 7 | 1,346 |
| SiberianCat ModernBERT | 0; 0 | 0 | 12 | 0 | 145 |
| Wolf mmBERT-small shape | 16; 0 | 18 | 24 | 0 | 1,333 |

Tensor contexts contain defensive system instructions and have no matched clean
Tensor controls. Their large context-versus-payload jumps—most starkly ProtectAI
19/908 direct versus 1,346/1,346 contextual—therefore indicate shortcut/context
sensitivity, not proven attack understanding. BIPIA clean FPs and payload recall
remain the more useful counter-checks.

## Remote reviewer smoke test

This frozen 100-row sample mixes 50 direct/indirect attacks with 50 controls, so
it is not numerically comparable to the full frozen benchmark tables.

| OpenRouter model | Availability | TP / available attacks | FP / available controls | p50 / p95 | Cost |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash Lite | 100/100 | 35/50 | 1/50 | 1,417 / 2,629 ms | $0.0033793 |
| GPT-5.4 nano | 84/100 | 24/36 | 0/48 | 1,844 / 3,074 ms | $0.0069230 |

GPT's apparent zero FPR excludes 16 unavailable results, disproportionately on
attacks. Neither remote model earns a request-path call, retry loop, or ReAct
stage; keep the harness for bounded offline weak-label/red-team work only.

## Compute and latency

| Candidate | Precision / context | Peak GPU allocation / reservation | Measured latency |
|---|---|---:|---:|
| Character heads | CPU / character TF-IDF | — | about 0.9 ms direct; 1.6 ms indirect |
| Frozen E5 | FP16 encoder, 384 tokens | 686.5 / 928 MiB | 4.80 ms batch 1; 1.69 ms batch 64 |
| Frozen ModernBERT mean + CLS | FP16 encoder, 512 tokens; one shared forward | 502.6 / 536 MiB | 7.62 ms batch 1; 5.33 ms batch 16 |
| PIGuard | FP16, 384 tokens | 1,303.4 / 1,492 MiB | 13.77 ms batch 1; 9.04 ms batch 32 |
| ProtectAI | FP16, 384 tokens | 1,303.3 / 1,492 MiB | 12.26 ms batch 1; 10.54 ms batch 32 |
| SiberianCat | FP16, 256 tokens | 672.7 MiB allocated | 4.90–7.51 ms/text on uncached Tensor sets |
| Wolf Small | FP16, evaluated at 512 tokens | 498.4 MiB allocated | 0.36–8.06 ms/text across length-bucketed sets |
| End-to-end ModernBERT-base | FP16 train, 512 tokens | 2,955 MiB allocated | 9.1 / 14.2 ms batch-1 p50/p95 |
| End-to-end DeBERTa-v3-base | FP16 train, 512 tokens | 3,547 MiB allocated | 10.8 / 13.9 ms batch-1 p50/p95 |
| OpenRouter | Remote | — | 1.42–1.84 s median plus provider failure |

FP16 makes every local encoder feasible on the 6 GB RTX 4050. VRAM is not the
selection bottleneck; operating-curve generalization and user friction are. Flash
Attention, `torch.compile`, an ensemble, and a 2,048-token Wolf rerun are not
justified by the measured errors.

## Revisions and gated skips

| Evaluated candidate | Pinned revision |
|---|---|
| `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| `answerdotai/ModernBERT-base` | `8949b909ec900327062f0ebf497f51aef5e6f0c8` |
| `leolee99/PIGuard` | `dd78b24e330193a22d2293ac66922dd4f982f563` |
| `protectai/deberta-v3-base-prompt-injection-v2` | `90c9989b1a342275dd0d1a95aad283c04e075671` |
| `siberiancat/modernbert-prompt-injection` | `fd9d17421e2e6bbe2eeea1874269fddc64e95e03` |
| `patronus-studio/wolf-defender-prompt-injection-small` | `9cf7dc2febf057238138ec256f16a0dbeda0d806` |

The relevant
[`qualifire/prompt-injection-sentinel`](https://huggingface.co/qualifire/prompt-injection-sentinel)
ModernBERT-large checkpoint was not evaluated: the live Hugging Face API marks
revision `e24131782819a4787c129ed2b9a4a67ae57a584e` as `gated: auto`, and an
anonymous pinned file request returns HTTP 401. The two
[`Llama-Prompt-Guard-2`](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)
sizes are `gated: manual`, and
[`qualifire/prompt-injection-jailbreak-sentinel-v2`](https://huggingface.co/qualifire/prompt-injection-jailbreak-sentinel-v2)
is `gated: auto`. They remain skipped under the ungated-only scope; a locally
available token is not used to bypass that rule.

## Evidence caveats and decisions

- JOT source and time are confounded, and qualitative agent audit found obvious jailbreaks
  among its source-labelled negatives. Its FP counts measure source labels, not
  clean benign-chat FPR.
- ToxicChat/deepset are same-source train/test families. Tensor Trust is held out
  from local fitting but public-checkpoint contamination cannot be excluded.
- PIGuard authored NotInject and reports BIPIA in development/evaluation; those
  results are contaminated. ProtectAI discloses XSTest-derived training.
- SiberianCat's generated corpus and split are unpublished; its 256-token card
  threshold over-defends (1,203/4,630 ToxicChat FPs). Wolf explicitly trains on
  deepset and NotInject and has WildJailbreak family overlap; its 0.5 threshold
  produces 1,073/4,630 ToxicChat FPs. Neither checkpoint is an independent gate.
- The frozen ModernBERT result rejects only linear separation of frozen features.
  The end-to-end pilot then underfit under its deliberately small protocol; this
  still does not reject a future data-richer ModernBERT run. The decoder variant
  is a causal generator and adds no value to this one-pass classification test.

Decision: retain the character and provenance-scoped heads in shadow mode, keep
all neural and remote candidates as reproducible controls, and rely on the
deterministic action/egress monitor for containment. Plausible trainable
candidates need the full declared operating-point comparison before promotion.
Do not OR their alerts together: their high-context errors are correlated and
would increase review load.

## Evidence-gated continuation

The controlled one-seed screen is complete. Only DeBERTa earned continuation:
freeze a fuller schedule and source-held-out success criteria, rerun three
seeds, and preserve the identical precision-floor/FPR grid. Evaluate calibration
and threshold saturation explicitly. Do not add focal loss, OOD energy, LoRA,
layer mixing, attention pooling, or an ensemble until that control identifies a
specific error those changes are expected to fix.

Before any ModernBERT 1,024/2,048-token capability test, add attacks with known
payload spans deliberately buried after token 512 plus length/topic-matched clean
documents. The current position set has negatives only and cannot establish
long-position recall. Promote no model unless it beats the character control
across held-out attack families without exceeding its hard-negative or
clean-context review load.

## Source artifacts

- [`baseline.json`](baseline.json)
- [`embedding_results.json`](../experiments/gpu_baselines/embedding_results.json),
  [`piguard_cuda_results.json`](../experiments/gpu_baselines/piguard_cuda_results.json),
  [`protectai_cuda_results.json`](../experiments/gpu_baselines/protectai_cuda_results.json)
- [`results_512_cuda.json`](../experiments/modernbert/results_512_cuda.json)
- [`siberiancat_cuda_results.json`](../experiments/modernbert_checkpoints/siberiancat_cuda_results.json),
  [`wolf_small_cuda_results.json`](../experiments/modernbert_checkpoints/wolf_small_cuda_results.json)
- [`comparison.json`](../experiments/encoder_finetune/comparison.json),
  [`modernbert_pilot.json`](../experiments/encoder_finetune/modernbert_pilot.json),
  [`deberta_pilot.json`](../experiments/encoder_finetune/deberta_pilot.json)
- [`openrouter-smoke.md`](openrouter-smoke.md)
