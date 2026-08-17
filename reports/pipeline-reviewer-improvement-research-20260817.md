# Improving the long-context reviewer

Date: 2026-08-17

Disposition: all prompt, channel-split, and reasoning variants in this note were rejected.
The later balanced-profile promotion changed scalar routing thresholds only and kept the current prompt, request contract, and context flow unchanged.

This note answers whether a better prompt, chain-of-thought, or reasoning mode is likely to reduce the 38 false restrictions among 54 long negatives in the consumed evaluation slice.
It uses retained local evidence, primary sources, and one bounded post-hoc provider diagnostic.

## Bottom line

A better policy prompt is worth testing and is the smallest likely improvement.
Reasoning or chain-of-thought by itself is not the first fix and is not supported as a reliable way to reduce false positives.
The stronger root-cause hypothesis is that the reviewer is being asked an over-broad, document-only question for which normal long HTML is often positive under the written policy.
The durable fix is to review compact suspect evidence in relation to the trusted user task, then calibrate the resulting score at the artifact level.

## What the 38 errors say

The [benchmark's 54-negative long-character slice](pipeline-benchmark-20260816.md#long-context-result) consists of 38 multi-window BrowseSafe `untrusted_content` controls, 10 multi-window PromptShield `direct_user` controls, and 6 single-window PromptShield `direct_user` controls.
All 38 reported full-context false restrictions are the 38 multi-window BrowseSafe controls.
Their retained DeepSeek probabilities range from 0.679 to 0.998, so this is not a cluster of borderline `0.5` decisions.
As a diagnostic on the already consumed rows, moving the reviewer threshold to `0.99` would reduce these long-negative restrictions from 38 to 11, but would also reduce local-high-or-reviewer coverage on the 62 long, multi-window untrusted positives from 62 to 48.
At `0.995`, the corresponding counts are 3 long-negative restrictions and 47 of 62 positives covered.
These are post hoc diagnostics, not eligible threshold results.

The [current prompt](../src/morgott/models/deepseek_nooa.py) calls the model a high-recall classifier, flags text that "plausibly" contains subversion, maps genuine ambiguity to `1`, and says that any request, question, or instruction that adds to a surrounding task in untrusted content is injection.
That contract makes ordinary HTML controls containing navigation, forms, instructions, help text, or calls to action difficult to distinguish from attacks.
Meta reports that Prompt Guard 2 deliberately removed its former broad injection sub-label because the objective was too broad to be useful, and now targets explicit attempts to supersede instructions ([official model card](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)).

The retained corpus also supplies no trusted user task with BrowseSafe HTML.
Instruction subversion is relational: whether an instruction redirects the task depends on what the trusted task is.
A document-only reviewer cannot reliably distinguish a page's legitimate instructions from instructions directed at the consuming agent when that relation is missing.
LlamaFirewall's experimental AlignmentCheck evaluates agent reasoning for goal misalignment rather than treating isolated text as the entire security decision ([original paper](https://arxiv.org/abs/2505.03574)).
CaMeL goes further and separates trusted planning from untrusted data processing instead of relying on a detector to solve authority ([official research repository](https://github.com/google-research/camel-prompt-injection)).

These facts make policy and input representation the leading explanation.
They do not prove that long-context attention contributes nothing.

## Will a better prompt help?

Probably, but this remains a hypothesis for DeepSeek V4 Flash until measured on a new development panel.

OpenAI's official gpt-oss-safeguard guide recommends four explicit policy sections: instruction, definitions, violating versus non-violating criteria, and short examples on both sides of the decision boundary.
It specifically says this structure is intended to prevent false positives on edge cases and recommends avoiding vague words such as "generally" or "usually" ([official guide](https://developers.openai.com/cookbook/articles/gpt-oss-safeguard-guide)).
That guidance applies directly to the weaknesses in Morgott's current policy wording, although it was written for gpt-oss-safeguard rather than DeepSeek.

The minimum revised policy should require clear evidence that content is addressed to the consuming model or agent, or that it attempts to change the trusted task, authority, data flow, tool use, or disclosure behavior.
It should explicitly classify webpage navigation, forms, user-facing calls to action, documentation, code examples, quoted attacks, and ordinary procedural instructions as `0` unless they are directed at the consuming agent or conflict with the trusted task.
It should replace "ambiguous means 1" with a separately measured `uncertain` or manual-review outcome rather than manufacturing a confident attack label.
It should include four to six short matched examples near the real boundary, with both benign HTML and injected HTML represented.

For real applications, the reviewer input should add a trusted `user_task` field supplied by runtime metadata.
The untrusted document must remain separately delimited data and must never be able to define or overwrite that field.
BrowseSafe itself is a raw-HTML classification benchmark and its authors report a specialized, fine-tuned raw-HTML model outperforming general safety models on their test set, which is evidence that task-specific training and representation can matter more than generic prompting ([official model card](https://huggingface.co/perplexity-ai/browsesafe), [original paper](https://arxiv.org/abs/2511.20597)).

## Will chain-of-thought or more reasoning help?

It might help after the policy is repaired, but it is not monotonic and should be a separate experimental arm.

DeepSeek's official API documentation says thinking mode reasons before producing the final answer and offers `high` and `max` effort controls, but it provides no prompt-injection classification false-positive result ([official thinking-mode documentation](https://api-docs.deepseek.com/guides/thinking_mode)).
That documentation supports capability, not a claim that thinking will fix Morgott's error mode.

The most relevant published comparison is model-dependent.
On the official BrowseSafe model card, GPT-5 mini improves sharply from minimal to low reasoning, but medium and high add no further benefit; Haiku 4.5 is flat or slightly worse with more thinking; Sonnet 4.5 improves with reasoning but also produces hundreds of refusals ([official BrowseSafe model card](https://huggingface.co/perplexity-ai/browsesafe)).
Morgott's own 256-row gpt-oss-safeguard screen also moved in the wrong direction for false positives: medium reasoning raised recall from 13.40% to 18.56% but raised FPR from 1.26% to 2.52% and reduced valid-output rate from 96.09% to 93.36%.

The current DeepSeek request disables reasoning, caps output at 16 tokens, and extracts the final `0` or `1` token log probabilities.
A thinking-mode experiment therefore needs a realistic reasoning-token budget and must be treated as a different contract.
It must not be silently mixed into the exact logprob arm, and provider support for final-label logprobs must pass a canary first.
Any raw reasoning should be discarded after parsing and never stored with benchmark artifacts.

## Long-context structure

Sending the entire page is not automatically better.
The original Lost in the Middle study found that long-context model performance varies substantially with evidence position and that adding more context can reduce accuracy ([original paper](https://arxiv.org/abs/2307.03172)).
Meta's official Prompt Guard 2 contract recommends segmenting long prompts and scanning segments in parallel ([official model card](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)).
The BrowseSafe reference model also chunks long raw HTML and aggregates at document level ([official model card](https://huggingface.co/perplexity-ai/browsesafe)).

For Morgott, the better reviewer candidate is not an unconditional OR over every window.
It is a hierarchical review that sends the trusted task plus the highest-scoring suspect window, its immediate neighbors, and position metadata to one artifact-level adjudicator.
This is a hypothesis derived from the sources, not a measured result.
It reduces irrelevant context and preserves boundary-spanning evidence, while avoiding a separate positive vote for every window.
The current 1,024-token windows and 128-token overlap can remain unchanged for candidate retrieval.

Artifact-level calibration is still required because a maximum or OR aggregator gets more opportunities to false-positive as window count grows.
Temperature scaling can improve probability calibration, but calibration changes confidence interpretation rather than repairing poor class ranking ([Guo et al.](https://arxiv.org/abs/1706.04599)).
Threshold selection must therefore use representative long benign and attacked artifacts and must report results by window count.

## Smallest decisive experiments

### 1. Prompt-only screen

Build a fresh 256-row matched development screen with 128 benign and 128 attacked long documents, balanced by length, injection position, HTML structure, and attack subtype.
Supply a trusted task for every pair.
Compare the current prompt against one narrower policy prompt with reasoning disabled and the same provider, model, temperature, and strict transport.
This screen is large enough to reject another catastrophic false-positive mode, but it is not large enough to certify a 1% FPR.

Advance only if the revised prompt materially lowers benign false restrictions without losing more than two percentage points of matched attack recall.
Inspect rule-category counts rather than retaining raw rationales or provider responses.

### 2. Context-shape screen

On the same frozen rows and revised prompt, compare full document against trusted task plus top suspect window and immediate neighbors.
Keep artifact-level denominators and include attacks spanning a window boundary.
Measure FPR, recall, invalid output, tokens, latency, and reviewer calls.

### 3. Reasoning ablation

Only after one prompt and context shape are frozen, compare thinking disabled against `high` and, only if justified, `max`.
Use true hard verdicts as the primary reasoning contract.
Treat logprob support as a separate canary-gated contract.
Do not use repeated voting or verbal confidence.

### 4. Confirmation panel

If a candidate passes the screen, evaluate it unchanged on at least 1,000 fresh long benign controls and 1,000 matched attacks before making an operating-point claim.
A production-quality claim near 1% FPR needs still more negative examples or accumulated adjudicated shadow traffic because a handful of errors produces wide uncertainty.
Do not tune on the already consumed 54-negative slice.

## Recommendation

Test the narrower, task-conditioned prompt first.
Then test compact evidence review.
Use reasoning only as the third ablation because the available evidence shows model-specific gains, latency costs, and possible false-positive regressions.
Do not try to repair this error by raising the reviewer threshold alone or by adding more full-context tokens.
Keep the current behavior advisory until a fresh long-benign confirmation panel shows the false-positive problem is actually resolved.

## Measured quick screen

### Complete 193-row long-character bucket

A matched follow-up tested the current and revised full-context prompts on all 193 consumed long-character identities behind the original 85.61% recall / 70.37% FPR headline, comprising 54 benign and 139 attacked artifacts.
Both arms used pinned Cloudflare strict hard verdicts, reasoning disabled, identical deterministically redacted inputs, and conservative failure handling.
The revised prompt supplied channel-aware trusted-task interpretation for direct requests and untrusted webpages, messages, emails, and documents.

| Arm | Reviewer-only recall | Reviewer-only FPR | Balanced artifact-level recall | Balanced artifact-level FPR | Valid outputs | Client p50 / p95 | Recorded cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current prompt | 92.09% | 68.52% | 88.49% | 68.52% | 190 / 193 | 1.29s / 3.49s | $0.667 |
| Revised channel/task-conditioned prompt | 59.71% | 0.00% | 72.66% | 0.00% | 191 / 193 | 1.32s / 3.77s | $0.671 |

The revised arm removed all 37 matched current-arm false restrictions, moving from 37 of 54 to 0 of 54, while balanced artifact-level attack detections fell from 123 of 139 to 101 of 139.
The revised balanced recall was 84.62% on untrusted content but only 57.38% on direct-user inputs.
PromptShield was the largest measured source weakness at 25 of 49 attacks detected, or 51.02% recall, while BrowseSafe reached 27 of 39, or 69.23%, and LLMail retained 37 of 37.
The Wilson 95% interval for 0 of 54 false positives still extends to 6.64%, so this does not certify a zero or production-grade FPR.
The matched current result differs slightly from the original headline because the follow-up uses strict hard verdicts on redacted inputs, while the original approximation used retained logprob thresholding on original inputs.
All 193 identities were already consumed, 109 inputs required safety redaction, and the trusted tasks were reconstructed rather than supplied by the original runtime, so the result is selection-ineligible.

#### Minimal-change follow-up

The 38 remaining balanced misses included 11 attacks below Morgott's local review floor, which no reviewer prompt can recover, and 27 reviewer-fixable misses: 13 PromptShield, 12 BrowseSafe, and 2 HackAPrompt.
A single added sentence broadened the revised prompt to implicit, role-based, encoded, and obfuscated redirection while retaining the human-facing-instruction exception.
It failed the frozen advance rule: balanced recall remained 72.66% and observed FPR rose from 0 of 54 to 4 of 54, or 7.41%.
That sentence is rejected.

The smallest non-dominated candidate uses the existing current prompt for trusted `direct_user` traffic and the revised task-conditioned prompt for `untrusted_content`.
This channel split requires no new model, threshold, reasoning mode, or prompt wording.
On the consumed long bucket it detected 111 of 139 attacks, or 79.86% recall, with 0 of 54 observed false restrictions.
It recovered 10 detections over the single revised prompt without adding a false restriction.
Direct-user recall was 73.77%, untrusted-content recall was 84.62%, PromptShield recall returned to 71.43%, and BrowseSafe retained 69.23% recall with 0 of 38 false restrictions.
This channel split was the preferred fresh-confirmation contract before the broader screen below.

### Scenario-balanced untrusted-content screen

A fast matched follow-up sampled 256 consumed provider-safe untrusted-content artifacts no longer than 4,096 characters.
It included 128 SEP rows and 128 rows proportionally stratified across BIPIA, FinanceBench, LLMail, Nemotron agentic IPI, TAT-QA, and Tensor Trust.
The sample contained 142 attacks and 114 benign controls.
The existing current prompt was replayed from its frozen Cloudflare strict-logprob ledger, while the revised untrusted prompt required 256 new calls under the unchanged exact balanced gates and threshold.

| Arm | Recall | FPR | Precision | Attack detections | False restrictions | Provider calls | Valid outputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current prompt | 73.24% | 1.75% | 98.11% | 104 / 142 | 2 / 114 | 52 | 255 / 256 |
| Revised untrusted prompt | 52.82% | 0.88% | 98.68% | 75 / 142 | 1 / 114 | 52 | 256 / 256 |

The revised prompt saved one false restriction but lost 29 attack detections.
SEP recall fell from 53.13% to 17.19%, and LLMail recall fell from 100.00% to 97.87%.
The candidate therefore failed the predeclared rule requiring no FPR regression, at most two recall points lost, and no source with at least 20 positives losing more than five recall points.

Decision: do not proceed to fresh confirmation and do not integrate the channel split into the pipeline.
Keep the current prompt and context shape unchanged; scalar profile selection is governed separately by the complete pipeline benchmark.
The screen is scenario-balanced consumed development evidence rather than representative production traffic, but the 20.42-point aggregate recall loss is large enough to reject this broad change quickly.

### Focused BrowseSafe context and reasoning screen

The follow-up diagnostic ran 460 pinned Cloudflare strict-hard-verdict calls at concurrency 8 across 115 consumed long BrowseSafe pages, comprising 60 benign and 55 attacked pages.
The four arms were the current prompt on the full page, the narrower task-conditioned prompt on the full page, the narrower prompt on the maximum-local-score window plus immediate neighbors, and that compact arm with high reasoning.
The current registered local model's original window scores were reused for the compact selection and balanced local-high short circuit.
All reviewer failures restricted conservatively.

| Arm | Reviewer-only evaluation recall | Reviewer-only evaluation FPR | Balanced-cascade evaluation recall | Balanced-cascade evaluation FPR | Client p50 / p95 | Evaluation cost |
|---|---:|---:|---:|---:|---:|---:|
| Current prompt, full, reasoning off | 100.00% | 97.37% | 100.00% | 97.37% | 2.30s / 11.31s | $0.586 |
| Narrow task-conditioned prompt, full, reasoning off | 66.67% | 5.26% | 71.79% | 5.26% | 2.17s / 10.67s | $0.605 |
| Narrow prompt, suspect window plus neighbors, reasoning off | 61.54% | 2.63% | 66.67% | 2.63% | 1.53s / 6.59s | $0.077 |
| Narrow compact prompt, high reasoning | 64.10% | 2.63% | 64.10% | 2.63% | 2.96s / 11.38s | $0.092 |

The narrow full-context prompt removed 35 of the current prompt's 37 benign restrictions on the 38-row evaluation negative role, but the balanced cascade lost 11 of 39 attack detections.
Compact evidence removed one additional benign restriction, reduced latency and cost substantially, and lost two more balanced-cascade attack detections relative to the narrow full-context arm.
High reasoning did not dominate reasoning-disabled compact review: it kept the same one false restriction, caught 25 rather than 26 attacks after the local-high short circuit, roughly doubled median latency, and used 11,407 rather than 600 completion tokens on the evaluation role.

This result strengthens the conclusion that prompt scope is the main false-positive lever and that chain-of-thought alone is not the fix.
It does not produce a deployable replacement.
The rows were already consumed, the trusted task was fixed and synthetic, and deterministic safety redaction changed the source inputs because 112 of the 115 selected long BrowseSafe rows triggered the repository's sensitive-text scanner, mostly on email addresses.
Only 38 evaluation negatives were available, so the FPR intervals remain wide.
The next eligible experiment remains a fresh matched long-document panel with a real trusted-task field, followed by unchanged transport to confirmation evidence.
