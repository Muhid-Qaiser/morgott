# DeepSeek V4 Flash 0731 research

Date: 2026-08-03.

Status update, 2026-08-17: this report preserves the reviewer replacement decision and its then-maintained `0.2 / 0.1 / 0.99999 / 0.6224593312018547` profile as historical evidence.
The registry-bound maintained advisory default is now the benchmark-selected balanced profile `0.2 / 0.025 / 0.9999 / 0.5`, with the prompt, Cloudflare request, and cascade flow unchanged.
See [the complete pipeline benchmark](pipeline-benchmark-20260816.md) and [the model decision ledger](model-experiments.md#balanced-advisory-promotion-on-2026-08-17).

## Decision

Replace Morgott's retained April reviewer with `deepseek/deepseek-v4-flash-0731` through the tested Cloudflare route and use its independently selected threshold `0.6224593312018547`.
The original preregistered comparison correctly recorded a stop decision under its slice-robustness and ZDR gates.
The repository owner subsequently made aggregate measured result quality the selection criterion and explicitly removed ZDR, quantization, and logprob availability as decision factors.
Under that criterion, 0731 is better on the frozen 14,000-row evaluation: it catches 58 additional positives with the same 142 false positives and slightly higher precision.
A paired exact comparison finds 128 positive rows caught only by 0731 versus 70 caught only by April, with two-sided exact `p=0.0000454`.
The upgrade is not uniform: PromptShield loses 21 true positives, while canonical gains 48 and SEP gains 31, so the PromptShield regression remains a material limitation rather than being hidden by the aggregate result.
The cascade remains advisory and is not approved for blocking or authorization.
For remote-enabled multi-window `untrusted_content`, the maintained cascade now uses one complete normalized 0731 review before its existing per-window fallback.
That later architecture decision leaves direct-user and single-window behavior unchanged and does not alter the selected model, provider, prompt, or threshold.

The exact published model-artifact revision is the Hugging Face release commit [`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/9e165c30e2704aec5d9d593cce3eebd58bbef1cb).
The current repository head inspected for documentation is [`7872f01b1d1fe23eabc4c98b48bffcef5a386062`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062).
The exact OpenRouter request selector is `deepseek/deepseek-v4-flash-0731`, and OpenRouter currently reports the provider-side canonical dated slug as `deepseek/deepseek-v4-flash-20260731` in its [model API](https://openrouter.ai/api/v1/models).
The maintained route uses the exact selector that produced the frozen evaluation and never uses the mutable `~deepseek/deepseek-v4-flash-latest` redirect.
OpenRouter does not expose a checkpoint hash or immutable provider-build identifier, so even the canonical dated slug is not equivalent to a Hugging Face revision pin.

The primary-source phase made no inference or provider-completion call.
The separate empirical comparison described next then used the authorized OpenRouter key on the already-public development panel.
All external claims below come from DeepSeek's official Hugging Face repository, the official DeepSeek-authored model materials in that repository, or OpenRouter's official API and documentation.

## Completed frozen replacement comparison

Before any full-panel result was observed, the comparison froze the exact model identity, four FP8 provider routes, production prompt and parser, 6,000-row calibration role, 14,000-row evaluation role, retained full-LoRA scores, old reviewer outputs, and replacement gates.
The local direct-user floor remained `0.2`, the untrusted-content floor remained `0.1`, and the high gate remained `0.99999`.
Each reviewer threshold was independently selected on calibration by maximizing cascade true positives under the same 2% aggregate FPR cap, then applied once to evaluation.

The fixed 20-row shape canary produced 20 of 20 valid outputs through Cloudflare.
GMICloud, Novita, and DeepSeek each produced 20 HTTP 404 failures under the exact required-parameter, strict-schema, top-logprob, and FP8 contract.
Cloudflare was therefore the only eligible route in that preregistered provider set.
All 20,000 Cloudflare calls then passed the production parser, including the exact one-field integer schema and both finite decision-token alternatives.

A later same-day operational recheck followed OpenRouter metadata drift that advertised the exact required parameters on more endpoints.
Two synthetic exact-contract canaries still returned HTTP 404 through both DeepSeek and Io Net, while Cloudflare returned two valid production-parser results in one attempt each.
The API key was valid with positive remaining allowance at the time of the check.
This volatile diagnostic supports retaining the tested route, but it is not a labelled quality comparison and does not authorize mixed-provider fallback under the Cloudflare-calibrated threshold.

The old route independently reselected its retained threshold `0.7310585786300049`.
The 0731 route selected `0.6224593312018547`.

| Frozen evaluation slice | April CoreWeave recall / FPR | 0731 Cloudflare recall / FPR | Change |
|---|---:|---:|---:|
| Overall | 70.267% / 1.773% | 71.235% / 1.773% | Recall +0.968 points; FPR unchanged |
| Canonical | 80.153% / 1.960% | 81.618% / 2.148% | Recall +1.466; FPR +0.188 |
| PromptShield | 74.611% / 2.643% | 72.435% / 2.367% | Recall -2.176; FPR -0.276 |
| SEP | 49.371% / 0.114% | 51.143% / 0.114% | Recall +1.771; FPR unchanged |
| Direct user | 75.043% / 1.921% | 75.702% / 1.807% | Recall +0.659; FPR -0.114 |
| Untrusted content | 63.600% / 1.286% | 65.000% / 1.661% | Recall +1.400; FPR +0.375 |

Overall, 0731 found 58 additional positives with the same 142 false positives and raised precision from 96.736% to 96.779%.
Its standalone evaluation AUROC was 0.95315 versus 0.95361 for the April route, while average precision was 0.93835 versus 0.93816.
Its runtime-zone p95 latency was 2.261 seconds versus 2.573 seconds, and its nominal uncached panel cost was `$0.49511` versus `$0.49432` at the observed endpoint prices.

The aggregate gain cleared the material-improvement gate, but the route failed the original predeclared slice gate because PromptShield recall fell by more than 0.5 points and untrusted-content FPR rose by more than 0.2 points.
Cloudflare is also absent from OpenRouter's current ZDR endpoint list, while the retained CoreWeave route is present.
The write-once machine summary therefore remains `replace_current_reviewer=false`; it accurately records the original protocol and was not rewritten after the owner changed the decision criterion.
Under the later quality-only criterion, the aggregate confusion matrix selects 0731 because recall improves significantly while the total false-positive count is unchanged.
Matched Boundary Pairs, AgentDojo, and Fireworks replay were not required to establish that narrower decision and were not run post hoc.

The write-once manifest, canary, result ledger, and summary SHA-256 values are `daa821ece3f03cd286f1d4aff0e38f457993061eb50e7e0eaa0066266aeb3a4d`, `378098bf9cb518eedbb98c08435226b8f30b652d0c901549d98e06f786bdf000`, `d3955b82f264ea4eb40317a0e39a6fa9f27b1e2afecbdea00617286d49cb07b5`, and `da0a75e67e32de08a53ba1cefacf1a8f917d8adb4ace381ed161ea9500bd5b7a`.
They live under `artifacts/openrouter_downstream_eval/` and retain no corpus text or raw provider response.
The frozen runner SHA-256 is `af0f7595ed9d81019c15cc1891260b67236689b86154dc8d0145451264cd079f`.
The runtime-shaped 0731 evidence ledger has compressed SHA-256 `1bb51757292e6bac03bc1575b7ab31608bc6377e8cc1b9d724336153edb54ae8` and decompressed SHA-256 `95380f1468242d5c41c53e4bf9e347deb7653191bb8fff4225fa81e82bebccbb`.
It differs from the frozen comparison ledger only by using the maintained runtime's deterministic job identifier and retains no text or raw response.

### PromptShield regression diagnosis

A read-only post-hoc diagnosis used only the frozen parsed score ledgers and made no provider call.
PromptShield evaluation contains 116 positives below the reviewer floor, 322 positives in the reviewer zone, and 527 positives above the local high gate; its corresponding negative counts are 1,885, 605, and 45.
The final-route disagreement is concentrated in the reviewer zone: 0731 recovers 19 positives missed by April but loses 40 positives caught by April, a two-sided paired exact `p=0.00864`.
It simultaneously repairs nine April false positives and introduces two, which explains the slice's lower recall and lower FPR.
Within that middle zone, April has AUROC `0.95765` and average precision `0.88635`, while 0731 has AUROC `0.94632` and average precision `0.89440`.
Of the 40 lost positives, 17 receive a 0731 probability from `0.5` up to but excluding the selected threshold, 16 receive `0.25` to below `0.5`, five receive `0.1` to below `0.25`, and two receive less than `0.1`.

The selected threshold sits on the calibration limit rather than leaving a free scalar repair.
Lowering it by one representable probability step raises calibration from 68 to 69 false positives and FPR from `1.9808%` to `2.0099%`, already exceeding the frozen 2% cap.
Taking the union of the April and 0731 decisions would recover all 40 lost PromptShield positives, but it raises global calibration false positives from 68 to 77 and FPR to `2.2429%`, adds ten evaluation false positives, and requires two remote reviews per artifact.
An exhaustive calibration-only sweep of a local-score auto-restrict override selected behavior equivalent to the then-current `0.99999` high gate and produced no gain.
These checks reject threshold lowering, a two-review union, and a local-score override on the consumed panel.
The maintained 0731 route remains unchanged, and any repair needs a prospectively frozen, independently sourced direct-user contrast rather than another PromptShield adjustment.

### Low-call gate diagnosis

A separate read-only diagnosis used only the frozen local and 0731 score ledgers and made no provider call.
The analytic calibration gate that preserved every calibration decision raised the direct-user and untrusted-content floors only to the first retained reviewer restriction and lowered the high gate only past the last retained reviewer pass.
It removed two of 1,370 calibration calls and seven of 3,208 evaluation calls, but the evaluation changed one SEP true positive to a pass.
That saving is immaterial and the candidate is rejected.

A bounded simple grid then reused the already-declared aggregate and slice noninferiority tolerances, fixed the 0731 reviewer threshold, and selected only from calibration by minimizing calls.
It selected direct-user floor `0.26`, untrusted-content floor `0.12`, and the unchanged `0.99999` high gate.
The point removed 27 calibration calls while losing four calibration true positives at the same 68 false positives.
On the already-open evaluation it removed 87 of 3,208 calls, lost nine true positives, and removed one false positive, moving recall from `71.235%` to `71.085%` and FPR from `1.773%` to `1.760%`.
The 2.7% relative call reduction is not worth lower measured recall under the owner's quality-first criterion.

Decision: keep the existing `0.2 / 0.1 / 0.99999` local gates and the `0.6224593312018547` reviewer threshold.
The next low-call design must be materially different and prospectively evaluated rather than another scalar gate retune on this consumed panel.

## High-zone scope decision

A separate prospectively frozen experiment tested whether 0731 should also review direct-user windows at or above the local `0.99999` gate.
The fit-disjoint panel retained 197 sanitized SWE-Lancer tasks and 46 successful JailbreakBench prompts from twelve behavior families.
Local scoring placed zero benign tasks and 31 attacks in the high zone.
The predeclared protocol therefore stopped before OpenRouter because the extension had no measured benign error to repair and could only preserve or weaken local attack restrictions.
The maintained high gate remains unchanged, 0731 remains a middle-zone reviewer, and the stopped panel must not be reopened post hoc.

## Prospective PIArena transfer and provider reliability

The next prospectively frozen evaluation pinned the public PIArena dataset and official static attack implementation before reading any score ([paper](https://arxiv.org/abs/2604.08499), [repository](https://github.com/sleeepeer/PIArena), [dataset](https://huggingface.co/datasets/sleeepeer/PIArena)).
It excluded knowledge-corruption rows, removed 154 rows under local sensitive-text screening, deduplicated 28 audit-strict contexts, and found no fit overlap among 9,108 checked context, payload, and attacked components.
The balanced panel retained 260 source tasks and 1,300 exact-unique artifacts across 13 question-answering, extraction, summarization, RAG, and long-context splits.
Each task contributed its clean context plus the official direct, ignore, completion, and combined attacks at the supported end position.

The local gate passed before provider calls.
It admitted 246 of 260 direct attacks and all 780 attacks from the other three families, found zero clean high-zone errors, and bounded the remote candidate set at 2,736 middle windows across 7,179 total windows.

The first complete-cascade attempt restricted 998 of 1,040 attacks, or 95.96%, and 3 of 260 clean contexts, or 1.15%.
It completed only 1,288 of 1,300 assessments because 35 reviewed windows ended in HTTP 500 and two ended in HTTP 429 after the maintained retries.
The frozen zero-failure and complete-assessment gates therefore rejected that run even though every measured quality gate passed.

A separately frozen operational retry selected every and only the twelve incomplete artifacts, kept the model, prompt, request, provider, thresholds, source texts, and routing unchanged, and reduced reviewer concurrency from 32 to 4.
All twelve assessments then completed, with zero failed reviews across 95 reviewed windows and 99 total attempts.
Replacing only those incomplete assessments for quality analysis leaves attack recall at 998 of 1,040, lowers clean restrictions to 2 of 260, or 0.77%, and raises fully correct matched tasks to 218 of 260, or 83.85%.
Every original aggregate, source-class, attack-family, and source-class-by-family quality gate passes after recovery.

The retry does not reverse the original provider-reliability failure or authorize a policy change.
It does isolate a smaller operational setting that preserved quality on the affected rows, so the maintained reviewer concurrency is now capped at 4.
The model selector, Cloudflare-only route, FP8 requirement, strict response schema, decision-token parser, threshold, and middle-zone scope are unchanged.

The base manifest, panel, local results, remote results, and summary SHA-256 values are `0df901642b4369fafb5058faa8047b995530ac57920192ac7fad9982fbc96b05`, `bab00b2c77dcb4fec8794c2816efa749228708b529ea6812ec93ade196d5828f`, `783f5704d219ea827d739a97c710ac40142933335760cd1c4bfd84987ab185b0`, `cc826777f71ce76ec71342bf7eafe9ed6db6ba4e91574b08dca6f4eb12eb00c6`, and `e07106afc703c46737ea24b921b33a56778e9914a413f4301162f5faa698ee2b`.
The retry manifest, panel, results, combined results, and summary SHA-256 values are `68ba4501fb5b77c6bdfe6ff460a9b86562cd2f05c6be99892575ad7e4804e510`, `a4d2851e28de8e4811c3ee143dca89f16e44414b69e538219578360053d5f849`, `e640337f8da147cf99e7182bf07582e5e58c47e761c998509e1ba21dad4ba571`, `3846b63838f72b9f46cbc5758dfa1ef4b675b21dd7415d90315da5a6c8c5ff24`, and `c7891515802af1a50c39fe06a541af120bf859a9a9f45e04292e3f9ed4346a34`.
PIArena is now consumed development evidence, and its April 2026 publication predates the July 2026 remote model, so unknown foundation-training exposure remains a limitation.

## Prospective AgentAbstain hard-benign rejection

A later prospective diagnostic projected only direct-user instructions from [AgentAbstain revision `8422284`](https://huggingface.co/datasets/antiquality/agentabstain/tree/842228426c2a703347396501af61c7890972c7ee).
Pair-wide privacy exclusion, exact and normalized deduplication, and complete-fit overlap filtering retained 348 unique instructions from 231 act and abstain pairs.
The panel tests whether the subversion cascade passes difficult ordinary user tasks; act and abstain are runtime outcomes rather than injection labels.

The local preflight restricted 3 instructions and admitted 27 middle-zone windows.
DeepSeek V4 Flash 0731 flagged 9 of the 26 successfully reviewed windows, cleared 17, and exhausted three HTTP 429 attempts on one.
Even assuming that failed review would clear, at least 12 of 348 instructions restrict, or 3.45%, above the frozen 1% aggregate cap.
The high-stakes slice independently restricts 7 of 50 instructions, or 14%, above its 10% cap.

Decision: preserve 0731 as the aggregate-development winner over the April preview, but reject any claim that the selected cascade has acceptable hard-benign direct-user behavior.
This result does not compare 0731 with April on AgentAbstain and therefore does not reverse the frozen replacement comparison.
It instead narrows the maintained claim and shows that text-only reviewer confidence still conflates high-stakes or abstention semantics with instruction-boundary subversion.
Do not retry, retune, or fit the consumed panel; require a materially different reviewer contract and a new independent matched source before reconsidering this failure mode.

## Prospective SafeClawBench context rejection

The next prospective diagnostic pinned [SafeClawBench revision `e6c2920`](https://huggingface.co/datasets/sairights/safeclawbench/tree/e6c29204c24a5910600aae854baae57a51586655) and retained 89 direct prompt-injection prompts plus 94 ambiguity-driven unsafe-inference controls after privacy and complete-fit overlap filtering.
Only the source `user_prompt` reached the detector, matching the maintained direct-user interface while withholding task-relative `scenario`, `safe_behavior`, and `success_predicate` fields.

The current cascade reached 9 of 89 DPI positives and restricted 4 of 94 ADI controls.
Replacing only its prompt with a direct-user instruction-boundary definition reduced those counts to 1 and 1 and failed the frozen recall gate.
When every row reached 0731, the current prompt reached 48 true positives and 11 false positives, while the candidate reached 3 and 2.
The candidate never flagged a row that the current prompt cleared and removed 45 true positives along with 9 false positives.

Post-hoc ledger analysis finds no scalar rescue under the panel's four-false-positive cap: the current and candidate reviewers can recover at most 13 and 5 positives respectively.
This analysis diagnoses the failure but cannot select a threshold on consumed data.

Decision: keep 0731 because this experiment does not compare it with April and does not overturn its frozen aggregate gain.
Reject the specialized prompt and leave the registered model, provider, threshold, and middle-zone scope unchanged.
The result instead exposes a context-free architecture limit: SafeClawBench's instruction-source family is relative to a trusted task and safe behavior that the text-only reviewer never receives.
Future work needs an independent task-conditioned action-boundary evaluation, not another prompt or threshold sweep on this source.

## Prospective BFCL live control transfer

A subsequent prospective diagnostic pinned six [official BFCL v4 live files](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data) and retained 2,050 privacy-screened, normalized-unique, complete-fit-disjoint last-user messages.
BFCL describes real-world function-calling tasks but does not supply explicit injection adjudication, so this panel measures non-subversion control behavior rather than a verified production false-positive rate.

The registered local model passed 1,745 controls, restricted 11, and admitted 294 controls across 310 windows to DeepSeek V4 Flash 0731.
The final complete ledger records seven reviewer flags, 287 reviewer clears, zero failed reviews, and 18 total restrictions, or 0.878%.
The observed result passes the frozen 1% aggregate gate, while its 1.384% 95% Wilson upper bound does not establish a sub-1% population rate.

The first pass exhausted provider retries on two controls.
The already-frozen resume path retried only those incomplete rows and produced a complete final ledger, so the transient transport failure does not select a different model, provider, threshold, or concurrency.

Decision: retain 0731 and its maintained middle-zone contract unchanged.
This result supplies broader ordinary tool-task evidence but does not reverse the AgentAbstain rejection or the SafeClaw context limitation.
The concentration of 5 restrictions among 41 controls with at least 64 word-like tokens remains a direct-user length warning that requires a new independent matched source rather than tuning on BFCL.

## Full-context long-artifact follow-up

The [Prompt Overflow paper](https://arxiv.org/abs/2605.23196) supplied architectural evidence that a long-context consumer can reconstruct distributed intent that remains weak in each independent detector window.
Its [released artifact](https://anonymous.4open.science/r/Prompt-Overflow-2624/) does not include the frozen generated panel or result ledger, does not pin its Hugging Face inputs, and implements a materially different weak-signal defense from the paper.
Morgott therefore audited but did not reproduce or copy that defense.

A separate prospectively frozen experiment reviewed 575 complete normalized multi-window artifacts once with the selected 0731 contract and retained another 485 local-high restrictions.
The one-review replacement improved most PIArena slices and cleared both existing clean restrictions, but completion restrictions fell from 182 of 183 to 181 of 183.
It rejected itself under its exact non-inferiority gate, so the existing window branch was not removed.

The maintained change uses the full review only as a first pass for remote-enabled multi-window `untrusted_content`.
A full-context flag restricts early, while a clear result falls back to the existing middle-window reviews.
An offline replay raises PIArena multi-window attack restrictions from 697 of 732 to 725 of 732 without adding a matched-clean restriction, and reduces attempts by 34.05% on the attack-heavy provider-eligible panel.
The clean slice instead requires 32.92% more attempts, so representative benign traffic still needs shadow measurement.

This result changes invocation order and visible context, not the selected 0731 model, Cloudflare provider, frozen prompt, parser, or threshold.
It leaves direct-user and single-window routing unchanged and remains advisory.
The exact protocol, hashes, source audit, and claim boundaries are in [the long-context reviewer report](long-context-reviewer-research.md).

## Exact identity and license

The Hugging Face [commit history](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731/commits/main) labels `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` as the 2026-07-31 release and labels the later head commit as a model-card update adding an SGLang cookbook link.
The release and current-head API records have identical blob IDs for `config.json`, the weight index, and the sampled weight shard, so the later documentation commit did not change the inspected model artifacts.
The current head's values are directly inspectable in the [revision-specific Hugging Face API record](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731/revision/7872f01b1d1fe23eabc4c98b48bffcef5a386062?blobs=true), and the release artifacts are directly inspectable in the corresponding [release API record](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731/revision/9e165c30e2704aec5d9d593cce3eebd58bbef1cb?blobs=true).
The repository and model weights use the MIT license, as stated in both the [pinned model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md) and [pinned license file](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/LICENSE).

OpenRouter maps `deepseek/deepseek-v4-flash-0731` to Hugging Face repository `deepseek-ai/DeepSeek-V4-Flash-0731`, labels it a re-post-trained revision, and reports a release date of 2026-07-31.
DeepSeek's [hosted-API changelog](https://api-docs.deepseek.com/updates/) keeps the mutable name `deepseek-v4-flash` and says that name now serves the latest official release, so it is not an immutable model-revision identifier.
Its [public model page](https://openrouter.ai/deepseek/deepseek-v4-flash-0731) currently lists the lowest available price as $0.09 per million input tokens and $0.18 per million output tokens.
That headline price belongs to the cheapest provider route and is not the price of every endpoint that satisfies Morgott's response contract.

## Architecture, context, and checkpoint representation

OpenRouter describes the target model as a sparse mixture-of-experts network with 284 billion total target-model parameters and 13 billion activated parameters.
The pinned DeepSeek configuration identifies `DeepseekV4ForCausalLM` with model type `deepseek_v4`, 43 hidden layers, hidden size 4,096, 64 attention heads, one key-value head, 256 routed experts, one shared expert, and six routed experts selected per token.
The full machine-readable values are in the [pinned `config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json).

The same configuration sets `max_position_embeddings` to 1,048,576 and applies YaRN scaling by a factor of 16 from an original 65,536-position base.
OpenRouter likewise reports a 1,048,576-token model context in the [live model API](https://openrouter.ai/api/v1/models).
Provider-specific context limits can be smaller than the model limit, so the selected endpoint remains part of the effective context contract.

The checkpoint is mixed precision rather than uniformly FP8.
Its configuration sets MoE expert storage to FP4 and describes dynamic E4M3 FP8 block quantization with 128 by 128 weight blocks for the other quantized tensors.
DeepSeek's [local inference instructions](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/README.md) say that converting the experts to FP8 requires removing the `expert_dtype` FP4 setting and requesting FP8 during conversion.
OpenRouter's endpoint `quantization` field describes each hosted route and must not be treated as a byte-for-byte description of the published mixed-precision checkpoint.

The Hugging Face revision API counts 304,180,418,494 typed tensor elements in the complete 0731 checkpoint.
DeepSeek separately says 0731 has the same structure as `DeepSeek-V4-Flash-DSpark` and includes an attached speculative-decoding module.
The 284-billion OpenRouter model figure and the larger Hugging Face checkpoint count are therefore measurements of different published surfaces.
The attached module is a plausible contributor to the difference, but the primary sources do not provide a formal reconciliation, so this report does not add or subtract them as if their counting rules were identical.
Neither source publishes a separate activated-parameter count for the attached DSpark module.

The configuration includes DSpark block size 5, noise token ID 128799, Markov rank 256, and target layer IDs 40, 41, and 42.
DeepSeek's pinned model card recommends seven speculative tokens with greedy DSpark drafting for its vLLM example.
OpenRouter does not disclose whether each hosted endpoint enables DSpark, so no latency or output-equivalence claim should assume that it does.

The published generation defaults are sampling enabled with temperature 1.0 and top-p 1.0 in the [pinned `generation_config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/generation_config.json).
DeepSeek recommends temperature 1.0 with top-p 0.95 for agentic use and top-p 1.0 otherwise.
Those general-generation recommendations do not override Morgott's already-tested deterministic classifier choice of temperature 0.

## Intended prompt encoding, reasoning, tools, and response schemas

The release has no Jinja chat template, and `tokenizer_config.json` contains no `chat_template` value.
DeepSeek instead ships a dedicated [encoding guide](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/encoding/README.md) and [reference encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/encoding/encoding_dsv4.py).
The encoder places the system content after the beginning-of-sequence token, prefixes user turns with `<｜User｜>`, and prefixes assistant turns with `<｜Assistant｜>`.
In non-thinking chat mode it places `</think>` immediately after the assistant prefix, while thinking mode opens a `<think>` block and expects reasoning before the final content.

The 0731 reference encoder recognizes `low`, `high`, and `max` reasoning effort in thinking mode.
The low level adds no prefix, while high and max prepend different natural-language instructions before the system message.
OpenRouter currently marks reasoning as optional but enabled by default for this model, with high as the default effort, in the [live model record](https://openrouter.ai/api/v1/models).
Morgott must therefore disable reasoning explicitly rather than rely on a provider default.

The reference encoder accepts OpenAI-shaped tool definitions but renders tool calls in DeepSeek's DSML markup.
It folds tool results into user messages using `<tool_result>` blocks because the native encoding has no standalone tool role.
The encoding guide says the `developer` role is reserved for DeepSeek's internal search pipeline and is not accepted by the official Chat Completions API, so Morgott's existing system-plus-user Chat Completions shape is appropriate.

The encoder can append a response-format schema instruction to a system or developer message.
This is a prompt-level capability in DeepSeek's reference code.
OpenRouter's separate [structured outputs contract](https://openrouter.ai/docs/guides/features/structured-outputs) defines `response_format.type=json_schema`, a named schema, and `strict=true` for endpoints that advertise `structured_outputs`.
Morgott should require both OpenRouter's `response_format` and `structured_outputs` endpoint capabilities because merely advertising generic response formatting does not prove strict JSON Schema enforcement.

DeepSeek's hosted interfaces expose different structured-output contracts.
The official [Chat Completions reference](https://api-docs.deepseek.com/api/create-chat-completion/) supports only `text` and `json_object` response formats, not a caller-supplied JSON Schema, and it documents content-token logprobs with up to 20 alternatives.
The official [JSON mode guide](https://api-docs.deepseek.com/guides/json_mode/) warns that JSON mode can occasionally return empty content and still requires an explicit JSON instruction in the prompt.
The official [Responses API reference](https://api-docs.deepseek.com/api/create-response/) supports `text.format.type=json_schema` with a name and schema, supports top logprobs up to 20, and accepts tools in the Responses shape.
DeepSeek's [2026-07-31 changelog](https://api-docs.deepseek.com/updates/) calls native Responses API support a 0731 feature, but OpenRouter's Chat Completions endpoint and Morgott's current raw-response parser are different wire contracts.
Native Responses support therefore does not prove that an OpenRouter Chat Completions provider performs the same constrained decoding or returns the same logprob structure.

## Differences from the April preview

OpenRouter maps the current April selector `deepseek/deepseek-v4-flash` to canonical slug `deepseek/deepseek-v4-flash-20260423` and Hugging Face repository `deepseek-ai/DeepSeek-V4-Flash` in the [live model API](https://openrouter.ai/api/v1/models).
The pinned April Hugging Face revision inspected for comparison was [`60d8d70770c6776ff598c94bb586a859a38244f1`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1).
There is no official Hugging Face repository named `deepseek-ai/DeepSeek-V4-Flash-20260423`; that date is OpenRouter's canonical slug for the preview repository.

DeepSeek explicitly calls 0731 the official release that supersedes the preview and says its agentic capabilities were substantially enhanced.
The official DeepSeek changelog says the architecture and size are unchanged and that 0731 was only re-post-trained.
The official model card reports gains over the preview on Terminal Bench 2.1 from 61.8 to 82.7, NL2Repo from 39.4 to 54.2, CyberGym from 38.7 to 76.7, DeepSWE from 7.3 to 54.4, and Toolathlon-Verified from 49.7 to 70.3.
DeepSeek ran the listed public code-agent evaluations with an unreleased minimal harness, max reasoning effort, temperature 1.0, and top-p 0.95.
Those results are relevant evidence of a materially different post-training target, but they are not evidence about instruction-subversion classification with reasoning disabled.

The common target architecture fields remain the same between the pinned preview and 0731 configurations.
The 0731 configuration adds the four DSpark fields and extends the compression-ratio list with two trailing zeros for the attached target layers.
The preview has 46 weight shards, while 0731 has 48 weight shards.
None of the 48 0731 weight-shard SHA-256 values appears among the 46 preview weight-shard SHA-256 values in the two revision-specific Hugging Face API records.
This proves the published weights are not the preview weights with only a renamed repository or changed model card.

The tokenizer blob is identical across the two pinned revisions, and the tokenizer configuration and generation configuration blob IDs are also identical.
The [pinned tokenizer vocabulary](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json) assigns standalone `0` and `1` token IDs 18 and 19, and the reference tokenizer keeps each digit as one token inside Morgott's compact verdict JSON.
That continuity removes one tokenization uncertainty, but it does not prove that a new provider returns the same OpenAI logprob shape or that both alternatives remain in its top-20 list.
The 0731 encoder changes the reasoning-effort implementation by making low the empty default, giving high the former strongest prompt, and adding a stronger max prompt.
The remainder of the reference-encoder diff does not establish a Morgott classifier equivalence guarantee.

## Current OpenRouter availability and prices

This section is a live snapshot retrieved on 2026-08-03 from OpenRouter's [0731 endpoint API](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints).
Prices below are dollars per million input, output, and cache-read tokens, in that order.
OpenRouter's status field is reproduced literally because its public endpoint response does not define the negative status codes used below.

- DeepInfra is tagged `deepinfra/fp4`, reports 1,048,576 context and 65,536 maximum completion tokens, costs $0.09, $0.18, and $0.018, advertises structured outputs but not logprobs, and reports status `0`.
- GMICloud is tagged `gmicloud/fp8`, reports 1,048,575 context and no maximum-completion value, costs $0.14, $0.28, and $0.028, advertises logprobs but not structured outputs, and reports status `0`.
- Fireworks is tagged `fireworks` with unknown quantization, reports 1,048,576 context and no maximum-completion value, costs $0.14, $0.28, and $0.028, advertises both structured outputs and top logprobs, and reports status `0`.
- Novita is tagged `novita/fp8`, reports 1,048,576 context and 393,216 maximum completion tokens, costs $0.14, $0.28, and $0.028, advertises logprobs but not structured outputs, and reports status `0`.
- Cloudflare is tagged `cloudflare/fp8`, reports 384,000 context and 384,000 maximum completion tokens, costs $0.14, $0.28, and $0.028, advertises both structured outputs and top logprobs, and reports status `0`.
- DeepSeek is tagged `deepseek/fp8`, reports 1,048,576 context and 384,000 maximum completion tokens, costs $0.14, $0.28, and $0.0028, advertises logprobs but not structured outputs, and reports status `0`.
- Parasail is tagged `parasail/fp8`, reports 1,048,576 context and 1,048,576 maximum completion tokens, costs $0.14, $0.28, and $0.07, advertises both structured outputs and top logprobs, and reports status `-2`.
- AtlasCloud is tagged `atlas-cloud/fp8`, reports 262,144 context and 131,072 maximum completion tokens, costs $0.14, $0.28, and $0.028, advertises structured outputs but not logprobs, and reports status `0`.
- SiliconFlow is tagged `siliconflow/fp8`, reports 1,048,576 context and 393,216 maximum completion tokens, costs $0.14, $0.28, and $0.028, advertises structured outputs but not logprobs, and reports status `0`.
- Io Net is tagged `io-net/fp8`, reports 262,100 context and 65,536 maximum completion tokens, costs $0.18, $0.34, and $0.08, advertises logprobs but not response formatting or structured outputs, and reports status `0`.
- Mancer 2 is tagged `mancer/fp8`, reports 1,048,576 context and 1,048,576 maximum completion tokens, costs $0.25 and $1.00 with no cache-read price, advertises both structured outputs and top logprobs, and reports status `-2`.

OpenRouter's model-level supported-parameter union includes `response_format`, `structured_outputs`, `logprobs`, `top_logprobs`, tools, tool choice, reasoning, reasoning effort, seed, temperature, and common sampling controls.
That union is not an endpoint guarantee, which is why `provider.require_parameters=true` remains necessary.
The official [provider-routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection) says that without this option a provider may receive a request and ignore unsupported parameters.

OpenRouter's [ZDR endpoint API](https://openrouter.ai/api/v1/endpoints/zdr) currently includes Fireworks, DeepInfra, Novita, Parasail, SiliconFlow, Io Net, and Mancer 2 for 0731.
Fireworks is the only status-`0` ZDR endpoint in that snapshot that advertises both strict structured outputs and top logprobs.
Cloudflare is the only status-`0` endpoint in the snapshot with disclosed FP8 quantization and both required capabilities, but it is not present in the ZDR endpoint list.
The route decision is therefore a real privacy-versus-known-quantization tradeoff rather than a choice that OpenRouter can currently satisfy on all dimensions.

A later same-day refresh showed how quickly that endpoint metadata moves: Fireworks had changed to status `-2`, while the API added tool-capable Ionstream and Venice records and changed several advertised capabilities and prices.
The refreshed raw endpoint document had SHA-256 `3403d2728e399f69875bffd1692f1297cfe31b44db3b4fb79911b48b531ea796`.
This volatility reinforces that endpoint metadata is a routing hint rather than calibration evidence.

Ionstream subsequently passed one synthetic exact-contract request, so a prospective provider-specific gate froze the same 20-row shape canary, middle-zone calibration and evaluation populations, parser, prompt, score gates, and no-fallback routing.
The first frozen runner stopped before a provider call when it detected that the canonical corpus had been repartitioned after the privacy-preserving panel was frozen.
Its successor recovered all 4,592 unique phase rows in memory by stable ID plus exact raw-text SHA-256, using the frozen current LLMail and Tensor Trust source shards for the 68 rows no longer present in a routing role.
No recovered text or raw provider response was persisted.

At four concurrent requests, Ionstream returned 12 valid results and eight HTTP 429 failures after 42 total attempts.
A separately frozen retry changed only concurrency from four to one and returned 18 valid results plus two HTTP 429 failures after 27 attempts.
Every completed result came from Ionstream and passed the production schema and decision-logprob parser, but neither canary met the predeclared 20-of-20 validity requirement.
Calibration and evaluation therefore remained locked, Ionstream was rejected as a classifier route, and no provider-quality claim was made from the canary.
The v2 manifest and canary SHA-256 values are `b6b80e610a8100fd8ead97d0a6a1c380a8bfe16b1559935a80abc1c4b5a291d0` and `a12ce9a45f88fa81193be1892bb38b925f687b7d8dbd978e5c1310728651a487`.
The concurrency-one manifest and canary SHA-256 values are `96ef6ce5c975349a7a2ffb9f68a30f1f5cd720bf86c910b29b4e8fc87910971e` and `a835a5092d5274c8da8767deb70e56229cbfe2b4bf726b79aa0c7d18ecc36125`.

Cloudflare later exhausted the frozen retry budget in a bounded public synthetic Agent-Diff tool trajectory.
A one-request Novita canary then returned HTTP 200 with one valid tool call under the same model selector, seed, temperature, reasoning, and required-parameter shape.
A final Novita-only typed-tool panel completed fifteen provider requests with zero provider failures and exact utility in both clean conditions.
Its fixed injection did not transfer in the no-monitor control, so it supplies no containment result and no comparative provider-quality result.
Novita is therefore a verified operational option for this non-probabilistic agent tool shape, not a drop-in route for Morgott's calibrated classifier.
A later six-source cross-channel Agent-Diff successor again received HTTP 429 from its one-request Cloudflare canary before sending task text, then passed a Novita-only canary and completed 18 task provider requests with no provider failure.
All three conditions reached exact legitimate utility, but the attacked control ignored the injection, so this adds operational typed-tool evidence only and still does not compare provider quality or establish containment.
A later Box source-lineage transfer passed another Novita-only exact-tool canary and completed fifteen task provider requests with no provider failure.
Its clean control reached exact utility, while the attacked control observed the changed source but committed neither the injected comment nor the legitimate comment.
This adds another operational typed-tool completion and a clean non-Slack adapter result, but no provider comparison, attacked-utility success, or containment evidence.
OpenRouter documents provider selection and fallbacks as request-level routing features, but fallback remains disabled for the classifier because the retained threshold was measured only on Cloudflare ([provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)).

On 2026-08-04 a newly frozen Agent-Diff task-49 run ended with a generic `HTTPError` before producing a valid clean result.
Its privacy-minimal failure row preserved neither failure phase nor HTTP status, so the outcome cannot be attributed to OpenRouter, Novita, or the local Agent-Diff replica.
A follow-up operational canary sent one public synthetic exact tool-call request without retry to every status-`0` endpoint that advertised seed, reasoning, tools, and tool choice in the live [endpoint API](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints).
DeepInfra, Novita, Cloudflare, Parasail, AtlasCloud, GMICloud, and Io Net each returned HTTP 200 with the exact tool call and matching provider identity, while Ionstream returned HTTP 429.
This one-request snapshot verifies current shape compatibility only and does not compare model quality or establish route reliability.
Future frozen agent experiments must preserve a safe failure phase and HTTP status class without retaining a response body, and they should run a separate exact-shape route canary before consuming a task.

## Morgott's selected reviewer contract

The frozen evaluation request pinned `deepseek/deepseek-v4-flash-0731`, Cloudflare, FP8, no provider fallback, and required-parameter filtering.
On 2026-08-14 OpenRouter's live Cloudflare endpoint metadata changed its quantization label from FP8 to unknown, causing the otherwise unchanged request to return HTTP 404 because no route matched the FP8 filter.
The maintained runtime removed only that stale quantization filter and still pins the same model, Cloudflare provider, no fallback, and required-parameter filtering.
`EVALUATION_REQUEST_SHA256` preserves the frozen 20,000-row request identity separately from the live operational request.
This restores the same provider route but is not fresh calibration evidence; the retained threshold remains provisional until a new frozen evaluation is justified.
It sends temperature 0, `max_tokens=16`, a strict one-field integer JSON Schema, `logprobs=true`, `top_logprobs=20`, and explicit reasoning disablement.
It also disables NOOA's cache-control injection and refuses tracing when corpus content could be exposed.

The parser requires one normally stopped choice, exact JSON with only integer field `subversion`, exactly one chosen decision token whose bytes equal ASCII `0` or `1`, and both classes exactly once among the chosen token's finite top-logprob alternatives.
It converts `logprob(1) - logprob(0)` through a sigmoid and applies the separately retained reviewer threshold.
An endpoint advertising `logprobs` and `top_logprobs` is necessary but not sufficient to meet that parser contract.

The historical [OpenRouter downstream report](openrouter-downstream-evaluation.md) records the April CoreWeave route that this selection supersedes.
The then-selected 512-token FP32 cascade reached 71.235% evaluation recall, 1.773% FPR, 96.779% precision, and a 22.914% provider call rate.
The then-registered 512-token OpenVINO BF16 runtime reached 71.386% recall, 1.798% FPR, 96.742% precision, and a 22.893% call rate.
These are already-open development results tied to the exact model, provider, prompt, request, threshold, and panel.

## Selected OpenRouter request

Cloudflare is selected because it is the exact route that produced all 20,000 valid candidate scores and the measured quality win.
The request body below shows the frozen contract, with the maintained system prompt and review window substituted into `messages`.

```json
{
  "model": "deepseek/deepseek-v4-flash-0731",
  "messages": [
    {
      "role": "system",
      "content": "<the frozen Morgott channel-aware classifier prompt>"
    },
    {
      "role": "user",
      "content": "<the exact review window>"
    }
  ],
  "temperature": 0,
  "max_tokens": 16,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "instruction_subversion",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "subversion": {
            "type": "integer",
            "enum": [0, 1]
          }
        },
        "required": ["subversion"],
        "additionalProperties": false
      }
    }
  },
  "logprobs": true,
  "top_logprobs": 20,
  "reasoning": {
    "enabled": false,
    "exclude": true
  },
  "provider": {
    "order": ["cloudflare"],
    "allow_fallbacks": false,
    "require_parameters": true,
    "quantizations": ["fp8"]
  }
}
```

OpenRouter's machine-readable [OpenAPI contract](https://openrouter.ai/openapi.yaml) allows at most 20 top alternatives and notes that some providers enforce a minimum completion budget of 16 tokens.
Keeping `max_tokens=16` preserves Morgott's tested minimum while avoiding a reasoning budget that could consume the short response.
Although the API now calls `max_tokens` deprecated in favor of `max_completion_tokens`, the endpoint capability records advertise `max_tokens`, so changing that field simultaneously would add an unnecessary compatibility variable.
The exact prompt, request, and threshold SHA-256 values are `6793cd3df00ea49c6da801692ef94b8200b212056fba27d298830186843b99a1`, `b5df77d444d1c16cce2aca82d35abf5a9d07869ad61fc11a051fc4a792a0619b`, and `89ac994922b6fb014179aa30df40b79001359f7e8df1de62ef6c3ec4a604d6f2`.
Fallbacks remain disabled because mixing provider implementations would invalidate the single calibrated threshold.

## Completed canary, recalibration, and serving verification

The Cloudflare canary produced 20 of 20 valid responses under the exact contract.
The full candidate run then produced 20,000 of 20,000 valid responses with the exact schema, decision token, and both finite class alternatives.
Calibration independently selected `0.6224593312018547` under the same 2% FPR cap used for April.
The threshold was applied once to the frozen evaluation role and no threshold was selected from evaluation labels.

The complete OpenVINO BF16 serving check then replayed all 20,000 typed 0731 records through the optimized local stage.
It found 89 local-zone differences and 27 final-route differences relative to FP32 and passed every established recall, FPR, precision, call-rate, and dataset-slice equivalence check.
The then-registered verification record is `artifacts/models/mmbert-lora-full-s42/serving/verification-0731.json` at SHA-256 `38d6dc33fdf3fd4e84d5cca2ac6d9e25cd187607c23e1f1f73caf2ca7b5ace38`.
The previous April verification records remain preserved as historical evidence.

## Explicit unknowns

OpenRouter does not publish a provider build hash, container revision, exact checkpoint SHA, template revision, or DSpark enablement state for any 0731 endpoint.
The live endpoint API can change after this report, including provider availability, status, supported parameters, context, and price.
At evaluation time Cloudflare advertised the required parameters and FP8, but OpenRouter did not list that endpoint as ZDR.
The Cloudflare route's parser shape, latency, and nominal cost were observed on the frozen public panel, but OpenRouter still does not expose its provider build or checkpoint hash.
No official source reports 0731 performance on prompt injection, jailbreak detection, harmful-intent routing, PromptShield, SEP, Morgott's corpus, or Morgott's outer-intent boundary pairs.
The official agent benchmark gains did not justify reusing the April threshold, which is why the separate project evaluation and recalibration were required.
The matching tokenizer reduces one migration risk but does not make logits, calibration, structured decoding, or provider serialization equivalent.
The evaluation roles are already open development evidence, not a prospective final test or production-prevalence sample.
The selected route is therefore a better advisory research reviewer under the owner's aggregate-quality criterion, not proof of production safety or permission to block.
