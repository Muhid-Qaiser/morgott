# LongBench Pro long-content control research

Date: 2026-08-03.

## Decision

LongBench Pro was used as a prospectively frozen benign-heavy operational control for the full-context-first remote path.
Its local preflight stopped before OpenRouter, so no remote result was opened and the source cannot promote or reject DeepSeek V4 Flash 0731.
The stopped gate exposes two independent deployment problems: the retained local-high branch restricts too many natural long documents, and exhaustive clear-result fallback has an unacceptable worst-case review workload.
Do not bypass either gate or treat the source as verified benign or as a prompt-injection benchmark.

## Source audit

The [LongBench Pro paper](https://arxiv.org/abs/2601.02872) describes 1,500 naturally occurring English and Chinese long-document samples across 11 primary tasks, 25 secondary tasks, and six balanced length buckets from 8k through 256k tokens.
Its task taxonomy covers retrieval, sequencing, evidence-grounded QA, summarization, citation alignment, aggregation, compliance checking, structured reasoning, code differences, rule induction, and dialogue memory.
The paper says source documents came from the public internet across news, medicine, science, literature, law, and education in formats including reports, tables, code, dialogues, lists, and JSON.
It also says human annotators excluded privacy-sensitive, copyrighted, and otherwise non-compliant documents before sample construction.

The public [Hugging Face dataset](https://huggingface.co/datasets/caskcsg/LongBench-Pro) is ungated and Apache-2.0.
The evaluated source is pinned to repository revision [`4996884deae51f5e5d23c88da9d857fc54e5fa15`](https://huggingface.co/datasets/caskcsg/LongBench-Pro/tree/4996884deae51f5e5d23c88da9d857fc54e5fa15).
Its 531,535,940-byte JSON file has SHA-256 `92ff05f6088e212d06c5a731ab86000b69cee6a0900cbbd524a25851e3c30de0`.
The Hugging Face Dataset Viewer independently reports 1,500 rows, 750 per language, 250 per length bucket, 840 full-context tasks, and 660 partial-context tasks.

The source has 1,102 exact-unique contexts because one document can support multiple task questions.
Only the 8k, 16k, and 32k buckets are candidates because longer buckets are likely to exceed Morgott's existing 128-window remote cap.
Among those 750 source origins, Morgott's local privacy screen excludes 137 origins and leaves 613 origins, 479 exact-unique contexts, and 471 canonical-normalized unique contexts.
The privacy hits are conservative pattern matches and do not imply that the public source actually contains live private data.

All 471 normalized-unique candidates survive exact, strict, and conservative near-overlap filtering against 1,073,829 canonical training rows, 18,284 PromptShield training rows, and 22,092 matched-pair rows.
No provider call was made during source, privacy, deduplication, or overlap audit.

## Frozen gate

The frozen protocol, stop rule, measurements, and artifact hashes are retained in this report.
Preparation additionally retains only contexts with 2 through 128 windows under the registered tokenizer.
The local preflight stops before provider calls if local-high restrictions exceed 1%, complete-artifact input exceeds 16 million registered tokenizer tokens, or the clear-result fallback could require more than 8,192 middle-window reviews.
The remote result requires complete assessments with zero provider failure.
It rejects the full-context-first fallback if observed restrictions exceed 1% overall or 2% in either language or a source length bucket with at least 100 retained controls.

## Preflight result

The frozen panel contains 471 normalized-unique contexts: 177 English and 294 Chinese, with 179 from the 8k bucket, 155 from 16k, and 137 from 32k.
They span 10 through 121 model windows, 22,646 windows in total, and 8,625,620 registered tokenizer tokens.

The local cascade restricted 7 of 471 controls before remote review, or 1.486%, above the frozen 1% gate.
The Wilson 95% upper bound is 3.036%.
Six restrictions were Chinese and one was English; two came from 8k contexts, two from 16k, and three from 32k.
Five were full-context tasks and two were partial-context tasks.

The 464 remaining complete-artifact reviews would consume 8,473,938 registered input tokens, below the frozen 16 million-token ceiling.
However, an all-clear result would then require 17,877 middle-window fallback reviews, 2.182 times the frozen 8,192-window budget.
Including the 464 complete-artifact reviews, that worst case is 18,341 typed review records.

The preflight therefore recorded `stop_local_high_restriction_gate` and made no OpenRouter request.
No result ledger or summary exists.
The manifest, panel, and preflight SHA-256 digests are `d1a4b39b4fdf464beadbe6492d7781b00650e477bd92066002372e9170c0d4e6`, `97e5f762b6bc942138a9d9c68de13f9c8e80e46c15c12a932ca4b45ebfa52b03`, and `45d9322035e960074f061427b7cca1a3da13db33ff0e01626bd7e2147ac36a19`.

This is evidence against the current local-high and exhaustive-fallback architecture on benign-heavy long traffic, not evidence against the remote reviewer.
It cannot establish a production false-positive rate because LongBench Pro does not adjudicate prompt injection, and it cannot authorize blocking because the cascade remains advisory.

## Bounded-fallback diagnosis

A sealed post-hoc replay ranked each consumed PIArena artifact's middle windows by its existing local score and retained only the highest-ranked fixed count after a clear complete-artifact review.
Reviewing the top eight preserved every restriction produced by the exhaustive full-context-first union: 725 of 732 attacks and 2 of 183 matched-clean artifacts.
It required 993 provider attempts on that panel rather than 1,061 for exhaustive fallback or 1,618 for the old per-window order.

Applied only as a workload bound, top-eight fallback would cap this LongBench panel at 3,712 fallback reviews plus 464 complete-artifact reviews, or 4,176 total, below the frozen 8,192-window budget.
It would not repair the seven local-high restrictions because the maintained branch currently stops before remote review.
Both the top-eight selection and any proposal to let DeepSeek review local-high long artifacts require a fresh, prospectively frozen matched attack/control source.
The consumed PIArena and LongBench panels must not select or promote that change.
