# Off-the-shelf ModernBERT checkpoint audit

This experiment compares two ungated Apache-2.0 sequence classifiers in shadow mode.
Both load through stock Transformers from pinned revisions, safetensors only, with
`trust_remote_code=False`. CUDA runs use FP16; CPU runs use FP32. Inputs are length-bucketed
and dynamically padded to avoid wasting the 6 GB GPU on padding.

## Checkpoints

| | `siberiancat` | `wolf_small` |
|---|---|---|
| Pinned revision | [`fd9d1742`](https://huggingface.co/siberiancat/modernbert-prompt-injection/tree/fd9d17421e2e6bbe2eeea1874269fddc64e95e03) | [`9cf7dc2f`](https://huggingface.co/patronus-studio/wolf-defender-prompt-injection-small/tree/9cf7dc2febf057238138ec256f16a0dbeda0d806) |
| Backbone | English ModernBERT-base | Multilingual ModernBERT/mmBERT, small shape |
| Parameters | 149,606,402 | 140,642,306 |
| Safetensors | 598,439,784 bytes; SHA-256 `0e70e2e0…d4d4858f` | 562,583,392 bytes; SHA-256 `0ddb75f2…be638ad0` |
| Config | 22 layers, hidden 768, 12 heads | 22 layers, hidden 384, 6 heads, 256k vocabulary |
| Architectural context | 8,192 | 8,192 |
| Injection fine-tune context | 256 | 2,048 |
| Positive class | index 1 | index 1 |

The Wolf card tags `jhu-clsp/mmBERT-base`, but the released config and parameter count
match the published **mmBERT-small** shape (384 hidden, 6 heads, about 140M parameters),
not mmBERT-base (768 hidden, 12 heads, about 307M). This audit follows the weights and
config. The underlying [ModernBERT paper](https://arxiv.org/abs/2412.13663) and
[mmBERT model card](https://huggingface.co/jhu-clsp/mmBERT-base) support the 8K encoder
architecture; they do not validate either injection head.

Both configs use ModernBERT's built-in mean-pooled sequence-classification head. These are
fine-tuned classifiers, not frozen feature extractors, and the decoder variant is not
relevant to this binary classification comparison.

These are very recent artifacts: Hugging Face reports SiberianCat created on 2026-07-08
and updated on 2026-07-10, while Wolf Small's audited revision was updated on 2026-07-16.
Pinning and local reproduction matter more than popularity or a card leaderboard here.

## Claims and evidence quality

The [SiberianCat model card](https://huggingface.co/siberiancat/modernbert-prompt-injection)
reports about 14k mostly synthetic examples, a 256-token fine-tune, 98% recall at 1% FPR
on its own held-out split, and 7/7 hand-written novel probes. It mixes generated data with
Giskard prompt-injection samples and PayloadsAllTheThings. The exact generated corpus and
split are not published, so those numbers are author-reported in-distribution evidence,
not an independent gate.

The [Wolf Defender Small model card](https://huggingface.co/patronus-studio/wolf-defender-prompt-injection-small)
reports roughly 50k selected training rows, a 2,048-token fine-tune, multilingual and
position/noise augmentation, and many public plus internal sources. Its table reports
4.0% aggregate FPR and 5.5% FPR after excluding the internal Patronus validation set;
both are far above the local experiment's stringent 0.1% diagnostic. It has no independent paper or
independently reproduced checkpoint evaluation yet. Its source list also includes the
access-conditioned WildJailbreak dataset; this experiment evaluates the ungated checkpoint
without downloading that gated training source.

The cards' accuracy/F1 values are not used for promotion. The evaluator calibrates frozen,
channel-specific direct-user and untrusted-content thresholds on the same grouped local
validation splits as the control, then compares recall at the locked FPR, hard-negative
false signals per 10k, BIPIA position categories, and the JailbreaksOverTime source groups.
It reports the cards' operating choices (0.03 for SiberianCat and argmax/0.5 for Wolf)
separately from the locked local thresholds.

## Training-overlap audit

“No disclosed overlap” means only that the checkpoint authors did not name that source;
the exact synthetic/internal corpora are unavailable, so hidden duplicates cannot be
ruled out.

| Local evaluation | SiberianCat checkpoint | Wolf Small checkpoint | Local calibration caveat |
|---|---|---|---|
| ToxicChat | No disclosed overlap | No disclosed exact overlap | Same-source official train/test split |
| deepset prompt-injections | No disclosed overlap. The published Giskard source has zero exact normalized overlap, but the checkpoint pipeline did not pin its Giskard revision | **Same dataset named for training**; the disclosed filename is the train split, so exact test-row overlap is not established | Same-source official train/test split |
| XSTest | No disclosed overlap | Family-informed: its WildJailbreak source generated benign categories inspired by XSTest | Not used for calibration |
| NotInject | No disclosed overlap | **Explicit training source, including named test files** | Not used for calibration |
| BIPIA payload/context/clean | No disclosed overlap | No disclosed overlap | Same-benchmark official train/test split for the untrusted-content threshold |
| OASST position stress | No disclosed overlap | No disclosed overlap | Same source as local calibration negatives |
| JailbreaksOverTime | No disclosed overlap | Source-family risk: Wolf trained on WildJailbreak, while this holdout contains in-the-wild JailbreakChat/JailbreakHub attacks | Not used for calibration |
| Tensor Trust attack/context | No disclosed overlap; public-benchmark contamination remains unknown | No disclosed overlap; public-benchmark contamination remains unknown | Source-heldout; not used for calibration |

[NotInject's official repository](https://github.com/leolee99/PIGuard),
[BIPIA's official repository](https://github.com/microsoft/BIPIA), and the
[JailbreaksOverTime paper/repository](https://github.com/wagner-group/JailbreaksOverTime)
define the relevant holdouts. WildJailbreak is synthetic rather than an exact copy of
JailbreaksOverTime, but its [dataset card](https://huggingface.co/datasets/allenai/wildjailbreak)
documents in-the-wild tactics and XSTest-inspired hard negatives, so Wolf's JOT and XSTest
results must be labeled source-family rather than clean OOD results.

For the SiberianCat/deepset check, the public Giskard source at
[`ce50a549`](https://github.com/Giskard-AI/prompt-injections/tree/ce50a549dadc46b48c931250d2dd71d5f003c0c2)
contains 35 records and has zero exact NFKC/casefold/whitespace-normalized matches against
the 660 local deepset records. That reduces known contamination risk, but cannot prove
absence because the checkpoint's generated corpus and source revisions were not released.

Tensor Trust is human-generated and source-heldout, but its public release has no explicit
standard dataset license. It is used here for evaluation only, never training; derived
commercial use needs a separate license decision. Source: the pinned
[`tensor-trust-data`](https://github.com/HumanCompatibleAI/tensor-trust-data) repository.

## SiberianCat result

The 2026-07-17 FP16 run at its published 256-token fine-tune length is in
[`siberiancat_cuda_results.json`](siberiancat_cuda_results.json). Every metric was
recomputed from the cached scores after the run; the result file SHA-256 is
`7fbdcca643a60e91ec992bf8dd9b620a79aa9636d7f223bc6e69ea840ee00b00`.

| Historical 0.1% diagnostic | Recall | False-positive rate |
|---|---:|---:|
| Direct calibration | 0/66 (0%) | 7/7,120 (0.098%) |
| ToxicChat | 0/73 (0%) | 5/4,630 (0.108%) |
| deepset | 0/60 (0%) | 0/56 (0%) |
| JailbreaksOverTime | 8/3,901 (0.21%) | 74/18,195 (0.41%) |
| Untrusted-content calibration | 0/84 (0%) | 0/142 (0%) |
| BIPIA payload | 0/125 (0%) | — |
| BIPIA poisoned context | 12/375 (3.2%) | — |
| BIPIA clean context | — | 0/167 (0%) |
| Tensor Trust direct attack | 25/908 (2.8%) | — |
| Tensor Trust attack context | 145/1,346 (10.8%) | — |

At the model card's `0.03` threshold, recall rises only by making normal input unusable:

| Published threshold | Recall | False-positive rate |
|---|---:|---:|
| ToxicChat | 66/73 (90.4%) | 1,203/4,630 (26.0%) |
| deepset | 45/60 (75.0%) | 18/56 (32.1%) |
| JailbreaksOverTime | 3,894/3,901 (99.8%) | 12,195/18,195 (67.0%) |
| NotInject | — | 62/339 (18.3%) |
| OASST position stress | — | 439/500 (87.8%) |
| BIPIA payload | 51/125 (40.8%) | — |
| BIPIA poisoned context | 370/375 (98.7%) | — |
| BIPIA clean context | — | 161/167 (96.4%) |
| Tensor Trust direct attack | 753/908 (82.9%) | — |
| Tensor Trust attack context | 1,345/1,346 (99.9%) | — |

The direct and indirect calibration PR-AUC values are 0.058 and 0.446 respectively.
Peak allocation was 672.7 MiB; batched set scoring ranged from 0.38 to 7.16 ms/text.
The checkpoint cannot attain even 80% precision on grouped validation at any
observed threshold. Its five-point FPR diagnostic also fails to justify an
OR-cascade sensor.

## Wolf Small result

The FP16 diagnostic used 512 tokens and batch 32, not the card's full 2,048-token
fine-tune length. The validated result is in
[`wolf_small_cuda_results.json`](wolf_small_cuda_results.json), SHA-256
`d33a937b18c40256ef92f824e6129158e819ec18de556e8d4e4b0841fa61e791`.

| Historical 0.1% diagnostic | Recall | False-positive rate |
|---|---:|---:|
| Direct calibration | 0/66 (0%) | 7/7,120 (0.098%) |
| ToxicChat | 0/73 (0%) | 0/4,630 (0%) |
| deepset | 0/60 (0%) | 0/56 (0%) |
| JailbreaksOverTime | 0/3,901 (0%) | 21/18,195 (0.12%) |
| Tensor Trust direct attack | 0/908 (0%) | — |
| Untrusted-content calibration | 16/84 (19.0%) | 0/142 (0%) |
| BIPIA payload | 18/125 (14.4%) | — |
| BIPIA poisoned context | 24/375 (6.4%) | — |
| BIPIA clean context | — | 0/167 (0%) |
| Tensor Trust attack context | 1,333/1,346 (99.0%) | — |

Wolf's card-style argmax/`0.5` operating point again over-defends:

| Published threshold | Recall | False-positive rate |
|---|---:|---:|
| ToxicChat | 67/73 (91.8%) | 1,073/4,630 (23.2%) |
| deepset | 39/60 (65.0%) | 0/56 (0%) |
| JailbreaksOverTime | 3,773/3,901 (96.7%) | 6,180/18,195 (34.0%) |
| XSTest | — | 23/450 (5.1%) |
| NotInject | — | 68/339 (20.1%) |
| OASST position stress | — | 176/500 (35.2%) |
| BIPIA payload | 30/125 (24.0%) | — |
| BIPIA poisoned context | 137/375 (36.5%) | — |
| BIPIA clean context | — | 28/167 (16.8%) |
| Tensor Trust direct attack | 880/908 (96.9%) | — |
| Tensor Trust attack context | 1,343/1,346 (99.8%) | — |

The Tensor-context result is not sufficient for promotion: that set has no matched clean
contexts, the public-benchmark contamination status is unknown, and Wolf performs poorly
on BIPIA's poisoned contexts under the same untrusted-content threshold. The direct and
indirect calibration PR-AUC values are 0.069 and 0.515. Peak allocation was 498.4 MiB;
batched set scoring ranged from 0.36 to 8.06 ms/text.

Wolf does not earn a 2,048-token run. Its locked direct recall is zero, and its short BIPIA
payload and mostly in-window context results already trail the char-ngram control badly.
Longer context cannot repair the operating-point failure demonstrated on short inputs.

## Run

```bash
PYTHONPATH=src python experiments/modernbert_checkpoints/evaluate.py --self-test

PYTHONPATH=src python experiments/modernbert_checkpoints/evaluate.py \
  --model siberiancat --device cuda

PYTHONPATH=src python experiments/modernbert_checkpoints/evaluate.py \
  --model wolf_small --device cuda --max-length 512 --batch-size 32
```

The result JSON includes validation-selected 80%, 85%, 90%, and 95% precision
profiles and 0.1%, 0.5%, 1%, 2%, and 5% FPR diagnostics. Neither checkpoint can
attain even the 80% profile. The default run includes ToxicChat, deepset, XSTest,
NotInject, OASST chat/position, Do-Not-Answer, HarmBench, multi-turn, BIPIA,
JailbreaksOverTime, and Tensor Trust. Score caches are keyed by checkpoint
revision, precision/device, effective context, and input hash. It deliberately scores one
whole sequence with tail truncation rather than adding a sliding-window ensemble; this
isolates whether the released 256- versus 2,048-token fine-tunes actually help. Run the
models serially on the 6 GB GPU.

## Recommendation

Reject both off-the-shelf checkpoints as gates or OR-cascade sensors. Their card operating
points grossly over-defend, neither reaches 80% grouped-validation precision,
and their FPR curves recover recall only with high false-signal counts. Wolf's
isolated Tensor-context success lacks clean controls and does not reproduce on BIPIA. Keep
the pinned evaluators and results as regression evidence; retain the char-ngram/reference-
monitor baseline while the separately trained frozen-encoder probe is assessed. Do not
count Wolf's deepset, XSTest, NotInject, or JailbreaksOverTime-family results as clean OOD
evidence.
