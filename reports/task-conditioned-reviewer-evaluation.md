# Task-conditioned content reviewer evaluation

## Outcome

The task-conditioned DeepSeek V4 Flash 0731 reviewer is rejected for maintained integration.
Correctly projecting only the untrusted result value reduced API-Bank's observed clean flag rate at threshold `0.3` from 20.31% to 3.68% over exact-unique reviews.
The same `0.3` threshold still failed the frozen clean gates.
A sealed read-only grid identified `0.5` as the first `0.1` grid point that passed all API-Bank output-only clean gates while retaining 94.88% recall on the already-consumed InjecAgent attacks.
That `0.5` candidate then flagged all 560 AgentDyn task-and-goal attacks after an operational retry completed 55 rate-limited reviews.

The prospectively frozen AgentPIMA transfer supplied the missing matched clean-and-attack gate and rejected that candidate decisively.
At `0.5`, decisive and accumulated attacks reached 95.09% and 95.39% recall, but 38.52% and 27.63% of the corresponding exact-unique clean reviews flagged.
At least one clean variant flagged for 102 of 112 tasks, or 91.07%.
No scalar threshold repaired the tradeoff: the best observed threshold satisfying every clean gate retained only 50.45% worst-variant attack recall.

The result does not authorize a maintained threshold or interface change.
The threshold was selected after API-Bank and InjecAgent outcomes were open, API-Bank excluded 151 source rows with fit-overlapping result values, AgentDyn used one conspicuous fixed attack template without a clean arm, and AgentPIMA is synthetic and highly templated.
The maintained response-only cascade therefore remains unchanged.

## Fixed contracts

API-Bank came from the official [DAMO-ConvAI repository](https://github.com/AlibabaResearch/DAMO-ConvAI) at revision `483554eae102996f5ec1f4feab4e78ef29c2a394` through the author-maintained [Hugging Face mirror](https://huggingface.co/datasets/liminghao1630/API-Bank) at revision `12e8158b7628c168f07e8f31fbbe3445e99f44cf`.
The official repository's Apache-2.0 license is the authority because the empty mirror card carries conflicting MIT metadata.

AgentDyn came from the official [AgentDyn repository](https://github.com/leolee99/AgentDyn) at revision `5353cf7615b135cace8d07c8f12dac53a16b6db3` under MIT.
The panel was extracted from pinned Python literals with the AST and did not execute AgentDyn or rely on its released action oracles.

AgentPIMA came from the public [AgentPIMA benchmark dataset](https://huggingface.co/datasets/agentpima-bench/agentpima-benchmark) at revision `78df16d47e2aeb4f2607092049f89c6148385671`.
The selected archive is licensed CC BY 4.0, while the embedded evaluation harness is MIT licensed.
The source is attached to an anonymized NeurIPS 2026 submission and has limited independent adoption, so it is evidence about this fixed synthetic construction rather than representative traffic.

All three remote studies used `deepseek/deepseek-v4-flash-0731` through the fixed Cloudflare route.
They used task-conditioned prompt hash `cd81816191887f26d6f0161a116020afc98860dfaa5701bfe0f3d3eb57cdf428` and request hash `b5df77d444d1c16cce2aca82d35abf5a9d07869ad61fc11a051fc4a792a0619b`.
Typed local redaction preceded hashing and provider review, every task and content value was checked against the full-LoRA fit references, and no raw text or raw provider response was retained.
All learned output remained advisory.

## API-Bank projection diagnosis

API-Bank serializes a completed tool interaction as `[trusted API call and arguments]->result value`.
The first protocol incorrectly placed that entire serialized block in `untrusted_content`.
That mixed trusted call syntax, argument values, and the actual untrusted result in one reviewer field.
The paired protocol split at the first `]->` before redaction and sent only the result value as untrusted content.

At the preselected threshold `0.3`, the projection change produced the following result.

| Metric | Full serialized block | Output only |
|---|---:|---:|
| Exact-unique clean flags | 146 / 719, 20.31% | 21 / 570, 3.68% |
| Source-row clean flags | 147 / 721, 20.39% | 21 / 572, 3.67% |
| Dialogues with any flag | 124 / 310, 40.00% | 21 / 270, 7.78% |
| Maximum level flag rate | 42 / 101, 41.58% | 13 / 81, 16.05% |
| Provider failures | 0 | 0 |

On the 572 source rows shared by both projections, the full block flagged 130 and the output-only projection flagged 21.
The paired transitions were 113 full-block-only flags, 4 output-only flags, 17 flags in both projections, and 438 flags in neither.
This isolates projection scope as the dominant source of the first protocol's false positives.

The `0.3` output-only result still failed the frozen maximums of 2% exact-unique flags, 5% in every level, and 5% of dialogues with any flag.
Level 2 remained the weakest slice at 16.05%.

After that rejection, a read-only grid over sealed API-Bank and InjecAgent scores was used only to choose a candidate for a different attack source.

| Threshold | API-Bank unique flags | API-Bank maximum level | API-Bank dialogues | InjecAgent recall | InjecAgent minimum attacker |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 3.68% | 16.05% | 7.78% | 96.96% | 58.82% |
| 0.4 | 1.58% | 8.64% | 3.33% | 96.02% | 52.94% |
| 0.5 | 0.70% | 3.70% | 1.48% | 94.88% | 47.06% |
| 0.6 | 0.35% | 2.47% | 0.74% | 91.84% | 35.29% |
| 0.7 | 0.18% | 1.23% | 0.37% | 88.71% | 23.53% |
| 0.8 | 0.00% | 0.00% | 0.00% | 82.35% | 17.65% |
| 0.85 | 0.00% | 0.00% | 0.00% | 78.27% | 17.65% |

Threshold `0.5` was the first fixed `0.1` grid point to pass every retained API-Bank clean gate while retaining at least 90% aggregate InjecAgent recall.
Its worst InjecAgent attacker-goal recall was only 47.06%, so even the consumed evidence does not show uniformly robust transfer.

## AgentDyn attack transfer

The AgentDyn protocol froze threshold `0.5` before any reviewer outcome was opened.
It crossed 60 trusted user tasks with 28 suite-specific attacker goals, producing 560 cases across Shopping, GitHub, and Daily Life.
Every goal used AgentDyn's official `important_instructions_no_model_name` framing with template hash `d25b87e9eba42e08a0d92c4c9a4183c51f34bf757956eaac31822c260a99e6a8`.
All 60 tasks and 27 exact-unique sanitized payloads were fit-disjoint.

The parent run produced 505 valid reviews and 55 exhausted HTTP 429 failures, all in Daily Life.
Every valid review flagged its attack, so the apparent slice failures were censored provider failures rather than model clears.
A write-once operational completion retried every and only those 55 failed IDs at concurrency `8`, leaving the model, provider, prompt, request, threshold, payloads, retry policy, and gates unchanged.
It completed all 55 reviews without a final failure.

The combined fixed result was 560 of 560 attacks flagged.
Shopping reached 180 of 180, GitHub reached 180 of 180, and Daily Life reached 200 of 200.
Every one of the 28 goal slices and 60 task slices reached 100%, so every frozen aggregate, suite, goal, task, and provider-completeness gate passed.

This is strong transfer evidence for an explicit fixed instruction attack.
It is not end-to-end AgentDyn attack success, does not exercise tools, does not test subtle or adaptive payloads, and cannot estimate false positives because the source arm is attack-only.

## AgentPIMA matched transfer

The final protocol froze threshold `0.5`, the task-conditioned prompt, the Cloudflare request, concurrency `8`, typed redaction, full-fit overlap rejection, output projection, and gates before any reviewer outcome was opened.
It retained all 112 trusted tasks and 672 matched progressive attack cases from `generated/defense_live_subset` with zero fit overlap.
Each case produced a clean and attacked decisive artifact plus a clean and attacked ordered artifact sequence.
The 2,688 source rows collapsed to 2,302 exact-unique reviewer inputs, and no task, artifact identifier, raw text, fragment text, or raw provider response was retained.

At the fixed threshold, 639 of 672 decisive attacks and 641 of 672 sequence attacks flagged.
Worst attack-family recall was 87.50%, worst objective-family recall was 90.91%, and worst decisive-channel recall was 89.58%.
Strict attack-over-clean ordering reached 89.58% for decisive pairs and 90.48% for sequence pairs.

The matched clean arm failed by much larger margins.
The reviewer flagged 146 of 379 exact-unique decisive clean inputs, or 38.52%, and 160 of 579 exact-unique clean sequences, or 27.63%.
It flagged at least one clean variant for 102 of 112 tasks, while the maximum clean attack-family slice reached 100%.
The run exhausted three attempts on 113 HTTP 429 responses.
Those failures were treated as unflagged, making the reported clean flag rates lower bounds, so retrying them could not rescue the clean gates and was not justified.

An exhaustive read-only threshold diagnostic over the sealed valid scores found no scalar repair.
Threshold `0.9902915229991972` maximized attack recall while satisfying every clean gate, but retained only 52.83% decisive and 50.45% sequence recall.
The cleanest point that kept both attack variants above 90% was `0.8807970826714613`; it still flagged 26.91% of the worse exact-unique clean variant, 95.83% of the worst clean family, and 80.36% of clean tasks.
This is score-distribution overlap, not a threshold-scale mismatch.

After sealing the result, the maintained reviewer changed only its fallback delay for HTTP 429 responses that omit `Retry-After`.
It now waits about 5 seconds and then 10 seconds, with jitter and the existing 15-second cap, instead of about 0.5 seconds and 1 second.
Explicit provider `Retry-After` values, the three-attempt limit, model, prompt, request body, parser, and thresholds remain unchanged.
A bounded 96-call synthetic public-reminder canary completed 70 reviews on the first attempt and recovered the other 26 on the second attempt with zero final failure in 21.65 wall-clock seconds.
This is operational evidence only and does not revise the failed quality gates.

## Decision and next gate

The correct runtime boundary is now explicit: trusted tool identity, call syntax, and arguments stay in trusted runtime metadata, while only the returned value enters `untrusted_content`.
No maintained code change was needed because these studies projected public benchmark records rather than exercising a defective production projection.

Reject task-conditioned threshold `0.5` and do not wire this all-row reviewer into the maintained cascade.
Do not recalibrate it on API-Bank, InjecAgent, AgentDyn, or AgentPIMA, and do not claim a production false-positive rate.

The matched gate is now complete and shows that a text-only reviewer score cannot safely choose authority on this workload.
The next architecture work should prioritize typed deterministic authorization and a frozen low-call invocation policy, then test any materially different learned candidate on a new independently sourced matched lineage.
Natural multi-tool outputs, benign credential-shaped values, less explicit and adaptive attacks, per-application slices, and enough clean rows to bound the intended false-positive rate remain required.
Concurrency `8` reduced but did not eliminate Cloudflare rate limits, so throughput changes require a separate operational test rather than another quality retry.

## Artifact identities

| Artifact | SHA-256 |
|---|---|
| API-Bank full-block manifest | `76ec49b58936acf53608ae131fcf93724f897ab32ba44b536a532f585e35b28a` |
| API-Bank full-block panel | `07b46a591d4067d4985629315e170c8749babc31281c7b5289e083cb4bcb4d61` |
| API-Bank full-block results | `deb8bb1ef9a04aa4b23ac8b42606be67d50283c2d47f996b054bee16f4b5a957` |
| API-Bank full-block summary | `82b8b30ea87de854118b6f8dc6b4a3bb088bddbcebd0fe86144e266eedac5df9` |
| API-Bank output-only manifest | `a2f690eded37357b51b6e63ef803695b8d3e78e7c78d33c4efb74c37c916f9d2` |
| API-Bank output-only panel | `69e90d82c7823798a67af175da609f26d26c926f64124a1c378bcc33f5f61a39` |
| API-Bank output-only results | `d1737fa2904ad6710e46f0d0e5e331167b79eb7957ba953cf575fa10a98de368` |
| API-Bank output-only summary | `5d6876644406fd9cb8b3fb613a84fab37077bd31041af4332df818079cae34a3` |
| AgentDyn parent manifest | `6b0f7d2da0e7d780026e6e66442b765691f605229a95d887de4cbd18e4920d50` |
| AgentDyn parent panel | `71109a6b3f1e8680bce9e504bb3b87a048a6e5c3b2cac531b69aef5f7f8de26c` |
| AgentDyn parent results | `e31d995d4bdba90c6217492e33b5b0e596528140f6bbe8faa8fa0f52fb0fe0f2` |
| AgentDyn parent summary | `9db9d9bc70e5a55e708e3b15573c6f4f08600de4aa5b4987e00004e2048eedc8` |
| AgentDyn retry manifest | `04c51b3108116bdc9f241ad9d8c2112e64d112c8b8c5258182248cc2a556ec41` |
| AgentDyn retry panel | `fea618f0cb64bffda8e1ce267574519af4c2041b62711c6e8bec49977788cbc0` |
| AgentDyn retry results | `e3fec6e8c574ada7a71db5e96cfa91011b7d907509c9622b843356748885d8c2` |
| AgentDyn retry summary | `189308c30d8e34c3a6e3b39c16892a4c1daee0d46155890f6bc50bab4b173c4a` |
| AgentPIMA manifest | `e4921e0cfc6473ef129f86cb4ce2ecac90b0df221f84560a70a360db3da9c847` |
| AgentPIMA panel | `d244d08262372bb5d0bc9a28f6d2e619bb1e32ae8bdd8a9cf0a7d224c0538448` |
| AgentPIMA results | `c2681ad1952e5ddd5d74262b9acf1ce88763d98b70ffa9f504e1d54a43c3ce96` |
| AgentPIMA summary | `1b39d23fa61822740a603ede663e4bec2113cab87524bab1e5815f80e39f9e3c` |
