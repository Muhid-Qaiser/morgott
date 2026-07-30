# LFM2.5 viability for Morgott

Research snapshot: 2026-07-30.
The inspected revisions are `0b649ad0c684378b03d4d8304f7577a662ab89bc` for the 230M base encoder, `8949b909ec900327062f0ebf497f51aef5e6f0c8` for ModernBERT-base, `6413fb38e02ea22e972e481d5a1a5828fc61e755` for the encoder evaluation harness, `35ca4a0469f180f1cf05a630df8842fa17ac18e3` for the 350M Prompt Router, `d5fbaa6022dea4d22841945fcd9a374d90507739` for its Space, and `e568a95ecc44416a773b0152262f591c35135e96` for the classification cookbook.
The initial source review did not download or run the model, and no corpus text was sent outside the repository.
The repository owner subsequently authorized one bounded same-GPU frozen-head comparison, which is now complete.

## Decision

| Proposed alternative | Decision |
|---|---|
| Replace `jhu-clsp/mmBERT-base` with `LiquidAI/LFM2.5-Encoder-230M` | Do not replace it. LFM has materially better PromptShield and SEP ranking, but worse aggregate canonical ranking, badly transferring validation thresholds, no local 512-token speed advantage, narrower advertised language coverage, and a less portable license. |
| Replace the DeepSeek middle-zone classifier with the published Prompt Router | Not viable on current evidence. It is a different 350M full fine-tune with relative forced-choice routing scores and no published injection metrics. |

Keep the retained mmBERT artifacts and the evaluated DeepSeek development route unchanged.
The owner explicitly authorized one bounded 230M frozen-head comparison on 2026-07-29.
That authorization permits the experiment, not model promotion, and the missing matched and prospective evidence still limits interpretation.
The 350M router is worth a separate local inference-only comparison only if its operational savings matter enough to justify selecting another model on the already-open downstream panel.

## Bounded frozen-head result

The controlled run froze the complete LFM backbone and fitted the same three-way `CLS + mean + max` pooling head used for the retained mmBERT comparison.
The 1.19M-parameter LFM head is `LayerNorm(3072) -> Linear(3072, 384) -> GELU -> Dropout(0.1) -> Linear(384, 1)`.
Its objective used equal-domain BCE across 1,069,607 canonical rows, 18,197 PromptShield training rows, and 22,082 matched generated rows forming 11,041 pairs, plus a `0.25` aligned-pair ranking loss.
The run used seed 42, three epochs, a 512-token cap, a 4,096-token batching budget, and selected epoch 2 by the equal-domain mean of Morgott and PromptShield validation BCE.
Freezing the backbone made the head fit compatible with the already-running mmBERT LoRA job, but the initial LFM feature pass over the full corpus still took 7,696 seconds because it had to encode every row once under shared GPU load.

The selected LFM checkpoint reached 0.9949 Morgott-selection AUROC and 0.9988 PromptShield-validation AUROC, versus 0.9953 and 0.9949 for the retained frozen mmBERT run.
Its validation macro BCE was worse at 0.0982 versus 0.0935.
Those validation figures did not decide the replacement question, so the same fail-closed development evaluation was applied to both models.

| Already-open development metric | LFM2.5 frozen | mmBERT frozen | LFM delta |
|---|---:|---:|---:|
| Canonical dev-test AUROC | 0.9853 | 0.9877 | -0.0025 |
| Canonical TPR at descriptive same-test 0.1% row FPR | 43.12% | 24.14% | +18.98 pp |
| Canonical TPR at descriptive same-test 1% row FPR | 63.21% | 73.84% | -10.64 pp |
| PromptShield test AUROC | 0.7820 | 0.7634 | +0.0186 |
| PromptShield TPR at descriptive same-test 1% row FPR | 23.99% | 3.22% | +20.77 pp |
| SEP AUROC | 0.8812 | 0.8368 | +0.0444 |
| SEP TPR at descriptive same-test 1% row FPR | 8.91% | 1.47% | +7.43 pp |
| Finance-negative row FPR at the validation-selected threshold | 0.014% | 0.071% | -0.057 pp |

The aggregate canonical result hides a meaningful split.
LFM reduced direct-user AUROC from 0.9891 to 0.9850, while improving untrusted-content AUROC from 0.9596 to 0.9810.
The external gains are therefore real ranking gains rather than a uniformly better backbone.
The descriptive same-test points are curve diagnostics only and cannot select an operating threshold.

The validation-selected threshold is the blocking failure.
LFM's threshold for the available 1% component false-alarm target had a 0.021% row FPR on canonical calibration, then transferred to a 1.764% row FPR on canonical dev-test and an 8.497% row FPR on PromptShield test.
The corresponding frozen mmBERT row FPRs were 0.217% and 0.164%.
LFM recovered much more recall at its applied threshold, but only by producing 1,447 PromptShield false positives instead of 28.
The 0.1% component target remained unavailable for both models because the untrusted-content calibration channel is underpowered.

This is useful evidence for complementarity, especially for untrusted-content ranking, but not evidence for replacing or promoting mmBERT.
PromptShield and SEP were already open before this run, the canonical dev-test is repeated development data, generated pairs were fitted rather than held out, and no prospective final set was consumed.
The LFM weights, scores, and head remain an unregistered advisory research shadow.
The complete local records are the [training result](../artifacts/combined_generic/lfm25_full_runs/liquidai-lfm2-5-encoder-230m_objective-full-balanced_pair-rank-0p25_s42/result.json) and [evaluation result](../artifacts/combined_generic/lfm25_full_runs/liquidai-lfm2-5-encoder-230m_objective-full-balanced_pair-rank-0p25_s42/evaluation_generic_v3/evaluation.json), whose evaluation JSON has SHA-256 `4675a4c3b97d6f6f88f4e7c8295dfc2d6010af3cd44a736a33a4384ce6f4c9d4`.
The exact one-off runner delta is preserved as [a zero-context patch](../experiments/lfm25-frozen-backbone.patch) against provenance commit `91e8c829c8b39c8ff37a6ca2479c8fc057168d39`; apply it there with `git apply --unidiff-zero lfm25-frozen-backbone.patch`.
This preserves reproducibility without restoring the removed historical runners to the active tree.

## 230M as an mmBERT backbone replacement

LFM2.5-Encoder-230M is a general masked-language encoder, not an injection detector.
Its card reports about 229.7M parameters, a 1,024-wide hidden state, a 65,536-token vocabulary, an 8,192-token trained context, and support for 15 named languages ([model card](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/blob/0b649ad0c684378b03d4d8304f7577a662ab89bc/README.md)).
Its pinned configuration contains 14 layers, comprising eight short-convolution layers and six grouped-query attention layers ([config](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/blob/0b649ad0c684378b03d4d8304f7577a662ab89bc/config.json)).
Liquid AI reports a 79.29 mean across 17 general classification tasks after a full supervised fine-tune per task, versus 78.19 for ModernBERT-base, but that suite contains no prompt-injection task and no mmBERT comparison ([benchmark table and method](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M#17-task-results-avg5-fresh-seeds-std)).
This establishes general classification capacity, not a better low-FPR instruction-subversion representation.

The official cookbook confirms that the backbone can support ordinary document classification by mean-pooling `last_hidden_state` and adding a linear BCE head ([classification implementation](https://github.com/Liquid4All/cookbook/blob/e568a95ecc44416a773b0152262f591c35135e96/examples/lfm-encoder-classification/train.py)).
That example full-fine-tunes the encoder and tunes F1-oriented thresholds on validation, so it does not validate Morgott's frozen multipool head, pair-ranking loss, source-heldout transfer, or component-level false-positive target.

The frozen-head concept is compatible with Morgott because the model exposes `last_hidden_state` and `config.hidden_size`, which are the two backbone properties the current multipool head consumes ([Morgott core](../src/morgott/models/mmbert/core.py), [pinned LFM implementation](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/blob/0b649ad0c684378b03d4d8304f7577a662ab89bc/modeling_lfm2_bidirectional.py)).
It is not a code-level drop-in.
Morgott pins the mmBERT identity throughout training, evaluation, inference, and artifact verification, while LFM loading requires `trust_remote_code=True`.
The live preflight also exposed a Transformers 5.14.1 loader trap: the repository's `AutoModel` mapping accepted the checkpoint but reported every pretrained tensor as unexpected and initialized the backbone from scratch.
Those partial features were discarded.
Loading the declared `AutoModelForMaskedLM` wrapper and using its `.lfm2` member loaded the saved tensor namespace correctly; preflight verified an embedding value against the pinned safetensors checkpoint and completed a finite 1,024-wide forward pass.
Morgott's LoRA regex targets mmBERT `attn.Wqkv` and `attn.Wo` modules, whereas LFM exposes separate `self_attn.q_proj`, `k_proj`, `v_proj`, and `out_proj` modules, so static inspection indicates that the current adapter contract would match no LFM attention modules ([Morgott LoRA contract](../src/morgott/models/mmbert/core.py), [LFM tensor names](https://huggingface.co/spaces/LiquidAI/prompt-routing/blob/d5fbaa6022dea4d22841945fcd9a374d90507739/router.py)).
A future LFM comparison therefore needs a separate pinned backbone and adapter identity rather than weakening the current mmBERT checks.

### Scope of the ModernBERT speed claim

Liquid AI's pinned 230M model card says the encoder is 3.3 times faster than ModernBERT-base at an 8K CPU input, while its release post says about 3.7 times faster and reports roughly 28 seconds versus more than 90 seconds per forward pass at 8,192 tokens ([pinned model card](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/blob/0b649ad0c684378b03d4d8304f7577a662ab89bc/README.md#inference-speed), [release post](https://www.liquid.ai/blog/lfm2-5-encoders#inference)).
Those are inconsistent published ratios, so neither should be quoted as a hardware-independent speedup.

The underlying CPU plot identifies a MacBook M4 Max CPU, FP16, batch size 1, and p50 measurement, and reports both per-forward latency and input-token throughput across lengths from 128 to 8,192 tokens ([CPU plot](https://aypchzzf9pftwuto.public.blob.vercel-storage.com/cpu_inference-cmErVqDt7VY4cFX7Ve3vJTFw3Nhrr6.png)).
At 512 tokens, that plot places ModernBERT-base below LFM2.5-Encoder-230M in latency and above it in throughput.
The Liquid AI post correspondingly says LFM2.5-Encoder-230M becomes the fastest CPU model from 1K tokens, although the Hugging Face mirror instead says it is fastest at every length ([Liquid AI post](https://www.liquid.ai/blog/lfm2-5-encoders#inference), [Hugging Face mirror](https://huggingface.co/blog/LiquidAI/lfm2-5-encoders#inference-speed-on-cpu-and-gpu)).
The plot and the Liquid AI text support the narrower claim, not the mirror's all-length CPU claim.

The GPU plot reports batch size 1 and p50 on two systems: an NVIDIA H100 in BF16 and a MacBook M4 Max MPS device in FP16 ([GPU plot](https://aypchzzf9pftwuto.public.blob.vercel-storage.com/inference-gpu-l0AdQEqUfk4NErSprvUab8ZyAAl0Gc.png)).
It depicts LFM2.5-Encoder-230M slightly ahead of ModernBERT-base at 512 tokens on the H100, but ModernBERT-base ahead below about 1K tokens on Apple MPS.
It contains no RTX 4050 measurement.
Because each published LFM-versus-ModernBERT plot uses batch size 1, its tokens-per-second panel is single-sequence forward throughput rather than evidence about batched throughput, and it does not measure backward passes or optimizer updates.
The separate internal-stack concurrency plot contains only the two LFM encoders, so it cannot establish a ModernBERT speedup ([concurrency plot](https://aypchzzf9pftwuto.public.blob.vercel-storage.com/latency-vs-concurrency-ZXWBhUuPv5hhCxvBsnOOd5sI9VORpS.png)).

This is also not a smaller-model speed comparison.
The pinned cards report about 229.7M parameters for LFM2.5-Encoder-230M and 149M for ModernBERT-base, while their pinned configurations describe a 14-layer, 1,024-wide hybrid and a 22-layer, 768-wide Transformer, respectively ([LFM card](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/blob/0b649ad0c684378b03d4d8304f7577a662ab89bc/README.md#model-details), [LFM config](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/blob/0b649ad0c684378b03d4d8304f7577a662ab89bc/config.json), [ModernBERT card](https://huggingface.co/answerdotai/ModernBERT-base/blob/8949b909ec900327062f0ebf497f51aef5e6f0c8/README.md#model-summary), [ModernBERT config](https://huggingface.co/answerdotai/ModernBERT-base/blob/8949b909ec900327062f0ebf497f51aef5e6f0c8/config.json)).
Both pinned Hub configurations identify float32 checkpoints, whereas the plotted runtimes explicitly use FP16 or BF16.

Both official model cards recommend Flash Attention 2 for their highest GPU efficiency, but the release does not state the attention backend, model entry point, padding distribution, software versions, warm-up count, or repetition count used for the plots ([LFM guidance](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/blob/0b649ad0c684378b03d4d8304f7577a662ab89bc/README.md#how-to-run), [ModernBERT guidance](https://huggingface.co/answerdotai/ModernBERT-base/blob/8949b909ec900327062f0ebf497f51aef5e6f0c8/README.md#usage)).
The linked reproducibility repository at commit `6413fb38e02ea22e972e481d5a1a5828fc61e755` is a supervised fine-tuning evaluation harness and contains no inference-latency script or raw speed results, so those details cannot be recovered from its public source ([pinned repository tree](https://github.com/Liquid4All/encoder_eval/tree/6413fb38e02ea22e972e481d5a1a5828fc61e755)).

Morgott's relevant workload is an RTX 4050 Laptop GPU, BF16 execution, a 512-token cap, and a configured 4,096-token training budget ([Morgott core](../src/morgott/models/mmbert/core.py), [retained run](../artifacts/models/mmbert-frozen-s42/result.json), [recorded device](../artifacts/models/mmbert-frozen-s42/evaluation.json)).
That combination differs from the H100 result in GPU class and batch shape, differs from both M4 results in hardware and runtime, and sits in the short-context region where the vendor plots do not show a consistent cross-device winner.
The published evidence therefore does not establish that LFM2.5-Encoder-230M will run faster than ModernBERT for Morgott.

A local same-GPU spot check measured the exact Morgott frozen-feature path on 2,048 deterministic validation rows.
All three pinned backbones used BF16, SDPA, the 512-token cap, the 4,096-token batching budget, the same 256-record chunks, one warm-up chunk, and three timed repetitions while the existing LoRA process remained active.

| Backbone | Median seconds | Rows per second | Process peak reserved |
|---|---:|---:|---:|
| ModernBERT-base | 9.840 | 208.1 | 440 MiB |
| LFM2.5-Encoder-230M | 10.279 | 199.2 | 604 MiB |
| mmBERT-base | 9.940 | 206.0 | 736 MiB |

On this bounded workload, LFM took 4.46% longer than ModernBERT and 3.41% longer than mmBERT.
The small difference under a concurrent training load is not a general performance benchmark, but it is enough to reject an assumed Morgott speed advantage.
LFM did retain 132 MiB more process-local VRAM headroom than mmBERT, while using 164 MiB more than ModernBERT.

LFM's 8K context also does not solve Morgott's demonstrated quality bottleneck.
The maintained scorer deliberately uses the first 512 normalized tokens, and Morgott's controlled 2,048-token ModernBERT run made BrowseSafe ranking and recall worse ([current experiment ledger](model-experiments.md#consolidated-modernbert-repair)).
Known payload spans, matched clean documents, source-heldout controls, and a long-benign denominator remain prerequisites before context length becomes a useful differentiator.

The advertised language breadth is also a regression from mmBERT.
The current backbone reports 1,800-plus languages, 307M parameters, and the same 8,192-token maximum under an MIT license ([mmBERT model card](https://huggingface.co/jhu-clsp/mmBERT-base#model-architecture)).
Neither advertised language count proves multilingual injection transfer, but replacing broad mmBERT coverage with 15 languages would require explicit per-language evidence rather than an aggregate benchmark.

## 350M Prompt Router as a DeepSeek replacement

The linked prompt-routing demo does not use the 230M encoder.
Its default model is `LiquidAI/LFM2.5-Encoder-350M-Prompt-Router`, which its card describes as a full fine-tune of the 350M encoder with a zero-shot routing head ([router model card](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router/blob/35ca4a0469f180f1cf05a630df8842fa17ac18e3/README.md), [Space code](https://huggingface.co/spaces/LiquidAI/prompt-routing/blob/d5fbaa6022dea4d22841945fcd9a374d90507739/router.py)).
The publisher discloses neither its fine-tuning data nor routing, safety, jailbreak, or prompt-injection metrics.

The router concatenates route descriptions and the prompt, mean-pools their token representations separately, projects them into two normalized 256-dimensional towers, and produces cosine-similarity logits.
Its public `route` method applies a softmax across the supplied routes, and the official Space calls it without a threshold and returns the top route ([model implementation](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router/blob/35ca4a0469f180f1cf05a630df8842fa17ac18e3/modeling_lfm2_bidirectional.py), [Space implementation](https://huggingface.co/spaces/LiquidAI/prompt-routing/blob/d5fbaa6022dea4d22841945fcd9a374d90507739/router.py)).
That default is forced choice: scores sum to one, depend on the complete route list and wording, and provide no calibrated absolute subversion probability or uncertainty band.
The underlying helper can filter routes below a caller-provided threshold, but no abstention threshold or calibration evidence is published, and the Space does not use one.

DeepSeek has weaker locality and privacy properties, but it has task evidence that the router lacks.
Morgott evaluated its fixed subversion-only contract on a grouped 6,000-row calibration split and separate 14,000-row evaluation split, retained raw binary-token log odds, measured transfer and operational failures, and selected explicit clear, flag, and review behavior ([downstream evaluation](openrouter-downstream-evaluation.md#log-probability-three-zone-follow-up-2026-07-29)).
The LFM router has not shown that it can preserve those attack rankings, distinguish harmful non-injection from instruction subversion, or improve the selected cascade.
Running locally could remove provider cost, transport failures, and external prompt disclosure, but the CPU Space is only an implementation demonstration and publishes no target-hardware latency or Morgott-quality result.

If a separate inference comparison is authorized, use the existing frozen 6,000/14,000 downstream split without changing its rows.
Predeclare exactly two route descriptions, record the raw `subversion - non-subversion` logit margin instead of treating the route softmax as calibrated, select two calibration thresholds for clear and flag, and send the interval plus every runtime failure to review.
Apply those thresholds once to evaluation and report the same aggregate, per-source, PromptShield, SEP, finance, pair, latency, and failure metrics as DeepSeek.
Do not replace DeepSeek unless the router improves the measured cascade frontier rather than merely being local.

## License and code-execution gates

Both LFM checkpoints use the LFM Open License v1.0 rather than MIT or Apache 2.0.
The license defines the threshold as annual revenue of at least $10M and states that commercial use by a legal entity at or above that threshold is not licensed under the agreement ([license sections 1 and 5](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/blob/0b649ad0c684378b03d4d8304f7577a662ab89bc/LICENSE)).
That does not exclude Morgott's present research under its repository rules, but it is a material portability and deployment disadvantage versus MIT-licensed mmBERT and requires legal review before commercial use or redistribution.

Both checkpoints also execute repository-authored Python through `trust_remote_code=True`.
At import time, that code globally replaces Transformers' LFM2 causal-mask function and `Lfm2ShortConv` forward methods before constructing the bidirectional model ([pinned implementation](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/blob/0b649ad0c684378b03d4d8304f7577a662ab89bc/modeling_lfm2_bidirectional.py)).
Any future use should pin the exact revision, hash or vendor the reviewed custom code, record it in artifact provenance, and isolate it from unrelated LFM2 models.
The current one-day-old release should not be allowed to turn a floating Hub revision into executable production code.

Apart from the explicitly authorized bounded LFM frozen-head comparison, no further encoder sweep or promotion is authorized.
The next evidence-producing work remains matched transaction data, paired multilingual transformations, known-span long documents, and prospective traffic-like negatives ([repository authorization](../AGENTS.md), [roadmap](../docs/roadmap.md)).
