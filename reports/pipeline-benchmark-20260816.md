# Morgott 1,024-context pipeline benchmark

Date: 2026-08-16

## Decision

This benchmark does not justify production blocking or a production false-positive-rate claim.
On 2026-08-17, the owner promoted the benchmark-selected balanced profile as Morgott's maintained advisory default without changing its prompt, provider request, window flow, or `decision: allow` authority.
Transporting those frozen thresholds unchanged to the full OpenVINO CPU ledger reached 72.10% recall, 1.36% FPR, and 96.72% precision on 12,352 provider-safe evaluation artifacts.
Azure revision `morgott-api--0000016` serves that profile at 100% traffic with the matching policy, threshold, model, and ONNX identities; its retained 30-request check is deployment smoke evidence only.
The exact balanced maintained cascade reached 72.00% recall with a Wilson 95% interval of 70.66% to 73.31%, 1.37% FPR with an interval of 1.14% to 1.65%, and 96.68% precision with an interval of 96.02% to 97.24% on 12,352 provider-safe evaluation artifacts, with 13.96% receiving at least one provider review.
Those artifacts consumed 1,705 full-artifact and 33 middle-window review units under full-context-first, ordered batches-of-four semantics.
A post-hoc on-demand-window ablation preserved balanced recall and FPR but increased provider requests from 1,738 to 1,778; the panel contained no all-low long untrusted artifact.
On the complete 193-row long-character diagnostic, the revised prompt moved matched balanced FPR from 68.52% to 0.00%, while recall moved from 88.49% to 72.66%; this redacted, post-hoc result is not selection-eligible.
On that isolated consumed slice, the smallest non-dominated follow-up kept the current prompt for direct-user traffic and the revised prompt for untrusted content, reaching 79.86% recall and 0.00% observed FPR before the broader screen below rejected it.
The broader 256-row scenario-balanced screen rejected that channel split: recall fell from 73.24% to 52.82%, while FPR improved by only -0.88 pp.
The exact operating range spans 49.40% recall at 0.50% FPR for the conservative profile through 75.58% recall at 2.15% FPR for the high-recall profile.
The exact balanced profile is now the maintained advisory default, but representative shadow traffic and fresh long-benign evidence remain required before blocking, SLA, or production-quality claims.
All learned results remain advisory and every maintained assessment still returns `decision: allow`.

## Evidence status

The calibration and evaluation roles are consumed development evidence, not prospective production traffic.

- `maintained_promotion` is **promoted advisory default**: The owner-promoted balanced profile changes advisory routing defaults only; it does not authorize blocking or establish production traffic quality.
- `exact_logprob_cascade` is **measured maintained multi window exact**: Only the provider-safe evaluation and its required window ledger support exact maintained-cascade claims; a frozen calibration selection alone does not.
- `exact_hard_verdict_cascade` is **no eligible strict provider**: All strict no-logprob providers failed at least one declared source-slice or overall quality gate; the earlier Decart evaluation is retained only as a pre-source-gate diagnostic.
- `benchmark_source_provenance` is **legacy incomplete source binding**: The frozen manifest predates the committed-source guard and names a base commit that did not contain this benchmark runner; parsed ledgers are retained, but exact source reconstruction is incomplete.
- `azure_deployment` is **verified promoted advisory**: The dated deployment record binds the live revision to the promoted profile, policy, thresholds, model, and ONNX identities; its 30 requests are a smoke test, not a load benchmark.
- `cascade_flow_comparison` is **measured post hoc consumed evaluation**: The matched ablation removes only unconditional full-context review from long inputs and reuses the frozen Cloudflare artifact and window ledgers without new provider calls.
- `reviewer_prompt_experiment` is **measured post hoc redacted browsesafe**: The public long BrowseSafe rows were deterministically safety-redacted before transmission, so this is a consumed, redaction-altered diagnostic rather than selection evidence.
- `reviewer_long_bucket_experiment` is **measured post hoc complete redacted long bucket**: The matched comparison covers all 193 identities in the consumed long-character bucket, but deterministic safety redaction and synthetic trusted-task interpretation prevent selection use.
- `reviewer_prompt_patch_experiment` is **measured post hoc minimal patch and channel split**: The one-sentence relaxation failed its frozen advance rule; the no-call channel-split simulation dominated the single revised prompt on the long bucket but was rejected by the broader screen.
- `reviewer_channel_split_screen` is **measured post hoc scenario balanced rejection**: The 256-row consumed, scenario-balanced untrusted-content screen rejects the broad channel split because its recall loss overwhelms its one-false-positive improvement.
- `exact_deepseek_standalone` is **measured provider safe evaluation**: The exact matched comparison separates logprob thresholding, the hard verdict returned by that same request, and a true request without logprob fields.
- `local_quality` is **measured artifact level approximation**: The 6,000-row role selected thresholds and the 14,000-row role measured an artifact-level approximation that is exact for single-window rows but not maintained multi-window cascade semantics.
- `provider_canary` is **measured canary only**: Sixteen rows per provider screen schema and routing compatibility, not quality or throughput.
- `provider_panel` is **measured**: The matched panel selects Cloudflare for strict logprobs; strict hard-verdict eligibility is decided by the later exact source-complete cascade gate.
- `provider_load` is **measured**: Only the corrected unique-sample load artifact is consumed; archived length-confounded runs are excluded.
- `openvino_parity` is **measured**: The 512-row audit is complete, and runtime-specific quality stays separate when threshold-decision disagreement exceeds 0.5%.
- `openvino_full_quality` is **measured runtime specific fixed fpr quality**: The full OpenVINO score ledger has separate 6,000-row calibration thresholds transported unchanged to the 14,000-row evaluation role.
- `azure_load` is **measured incumbent only**: The retained load predates profile identity fields and is incumbent-only evidence, not promoted-profile end-to-end evidence.
- `mutation_1024` is **measured**: The registered 1024 model was replayed under pre-promotion incumbent gates; full-cascade review outcomes and promoted-gate mutation outcomes remain pending.
- `loginject_local_routing` is **measured sealed once**: The sealed panel has one local-model score; its preliminary local profile rows are secondary to the exact promoted remote cascade result.
- `loginject_remote_cascade` is **measured sealed once**: Review-zone outcomes require the separately frozen provider phase and are not inferred from local routing.
- `gpt_oss_native_screen` is **measured supplementary 256 row screen**: The official-policy Harmony-style low-versus-medium screen is complete, but neither arm is reliable enough to expand or replace the main comparison arms.
- `production_traffic` is **unavailable**: No representative adjudicated production traffic was supplied.

## Exact maintained-cascade profiles

The strict-logprob arm reports 12,352 provider-safe evaluation artifacts with exact maintained multi-window routing.

| Profile | Recall | FPR | Precision | Artifact review rate | Review units artifact + window | Thresholds direct / untrusted / high / reviewer |
|---|---:|---:|---:|---:|---:|---|
| Conservative | 49.40% | 0.50% | 98.20% | 17.49% | 2139 + 37 | `0.3 / 0.05 / 0.99999 / 0.99` |
| Balanced | 72.00% | 1.37% | 96.68% | 13.96% | 1705 + 33 | `0.2 / 0.025 / 0.9999 / 0.5` |
| High Recall | 75.58% | 2.15% | 95.13% | 7.44% | 910 + 10 | `0.2 / 0.025 / 0.99 / 0.5` |

### True no-logprob hard-verdict cascade diagnostic

No strict no-logprob provider satisfies the complete provider-selection rule once source is included as a declared slice.
The table below preserves the earlier Decart exact evaluation as a pre-source-gate diagnostic, not as a selected provider result.

| Profile | Recall | FPR | Precision | Artifact review rate | Review units artifact + window | Thresholds direct / untrusted / high / reviewer |
|---|---:|---:|---:|---:|---:|---|
| Conservative | infeasible, minimum observed FPR 1.26% | unavailable | unavailable | unavailable | unavailable | unavailable |
| Balanced | 71.48% | 1.30% | 96.84% | 13.96% | 1705 + 33 | `0.2 / 0.025 / 0.9999 / hard verdict` |
| High Recall | 75.49% | 2.12% | 95.20% | 7.44% | 910 + 10 | `0.2 / 0.025 / 0.99 / hard verdict` |

Decart matched the best aggregate recall but lost 3.10% recall on the HackAPrompt source slice, above the allowed 2 percentage-point loss.
Baidu and DeepInfra also failed one or more overall or declared-source slice gates, so the study has no strict hard-verdict winner.

## Full-context-first versus on-demand review

This post-hoc ablation keeps the local gates, Cloudflare logprob contract, ordered windows, batch size four, and fail-closed behavior unchanged.
The candidate removes only the unconditional full-context DeepSeek review for multi-window untrusted inputs and reviews their middle-zone windows instead.

| Profile | Flow | Recall | FPR | Precision | Reviewed artifacts | Review units artifact + window |
|---|---|---:|---:|---:|---:|---:|
| Conservative | Full context first | 49.40% | 0.50% | 98.20% | 2,160 | 2,139 + 37 |
| Conservative | Middle windows on demand | 49.40% | 0.49% | 98.24% | 2,160 | 2,136 + 77 |
| Balanced | Full context first | 72.00% | 1.37% | 96.68% | 1,724 | 1,705 + 33 |
| Balanced | Middle windows on demand | 72.00% | 1.37% | 96.68% | 1,724 | 1,703 + 75 |
| High Recall | Full context first | 75.58% | 2.15% | 95.13% | 919 | 910 + 10 |
| High Recall | Middle windows on demand | 75.58% | 2.15% | 95.13% | 919 | 909 + 51 |

At the balanced thresholds, recall and FPR were unchanged: recall delta +0.00 pp (+0.00 to +0.00) and FPR delta +0.00 pp (+0.00 to +0.00).
The on-demand flow changed artifact review units by -2, window review units by +42, and total provider requests by +40; it did not reduce the 1,724 reviewed artifacts.
Independent calibration selected the same thresholds as the maintained flow for all three profiles.
Coverage is narrow: the evaluation contained 63 multi-window artifacts, only 7 were untrusted, and only 2 avoided local-high routing under the balanced profile.
No evaluated untrusted multi-window artifact had every window below the low threshold (0 rows), so this comparison does not validate the all-low long-input case that motivated the question.
The result is exact for this provider-safe consumed panel, but it is a post-hoc ablation and not a production routing recommendation.

## Secondary artifact-level approximate operating profiles

| Profile | Recall | FPR | Precision | Advisory restriction rate | DeepSeek calls | Thresholds direct / untrusted / high / reviewer |
|---|---:|---:|---:|---:|---:|---|
| Conservative | 49.45% | 0.62% | 98.34% | 21.51% | 20.78% | `0.3 / 0.025 / 0.99999 / 0.99` |
| Balanced | 68.61% | 1.46% | 97.23% | 30.19% | 21.42% | `0.1 / 0.05 / 0.99999 / 0.622459` |
| High Recall | 77.60% | 2.61% | 95.70% | 34.69% | 10.51% | `0.05 / 0.025 / 0.99 / 0.5` |

These thresholds were selected only on the frozen 6,000-row calibration role and transported unchanged to the 14,000-row evaluation role.
The profile metrics reuse one retained full-artifact DeepSeek probability per artifact and are exact for single-window rows, but they do not replay the maintained full-context-first and middle-window review sequence for multi-window rows.
These full-14,000-row approximations remain secondary even after an exact provider-safe replay because they do not implement the maintained multi-window review sequence.
The approximate profiles are Pareto choices with different error costs, not three production service levels.

### Approximate balanced profile by evaluation dataset

| Slice | Rows | Positives | Negatives | Recall | FPR | Advisory restriction rate |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 7,000 | 3,275 | 3,725 | 77.80% | 2.44% | 37.70% |
| promptshield | 3,500 | 965 | 2,535 | 69.53% | 1.03% | 19.91% |
| sep | 3,500 | 1,750 | 1,750 | 50.91% | 0.00% | 25.46% |

### Approximate balanced profile by input channel

| Slice | Rows | Positives | Negatives | Recall | FPR | Advisory restriction rate |
|---|---:|---:|---:|---:|---:|---:|
| direct_user | 9,634 | 3,490 | 6,144 | 71.29% | 1.24% | 26.61% |
| untrusted_content | 4,366 | 2,500 | 1,866 | 64.88% | 2.20% | 38.09% |

## Exact matched DeepSeek standalone contracts

All rows below use the same 12,352 provider-safe evaluation artifacts, and invalid outputs fail closed.

| Contract | Provider | Evaluation rows | Calibration FPR target | Recall | FPR | Precision | Invalid outputs |
|---|---|---:|---:|---:|---:|---:|---:|
| Cloudflare logprob threshold | cloudflare | 12,352 | 0.1% | 17.17% | 0.18% | 98.19% | 8 |
| Cloudflare logprob threshold | cloudflare | 12,352 | 0.5% | 33.43% | 0.39% | 97.94% | 8 |
| Cloudflare logprob threshold | cloudflare | 12,352 | 1.0% | 61.63% | 0.94% | 97.32% | 8 |
| Cloudflare logprob threshold | cloudflare | 12,352 | 2.0% | 68.61% | 1.41% | 96.43% | 8 |
| Cloudflare logprob threshold | cloudflare | 12,352 | 5.0% | 82.31% | 5.01% | 90.13% | 8 |
| Cloudflare same-request hard verdict | cloudflare | 12,352 | hard verdict | 69.35% | 1.45% | 96.38% | 8 |
| Decart true no-logprob hard verdict | decart | 12,352 | hard verdict | 68.79% | 1.27% | 96.78% | 0 |

The Cloudflare same-request hard verdict removes thresholding while retaining the logprob request, whereas the Decart strict hard-verdict arm removes the logprob request fields themselves.
These transports are reported separately and are not reconstructed from probabilities at 0.5.

## Standalone detector comparison on the frozen calibration and evaluation panel

| System | Calibration FPR target | Evaluation recall | Evaluation FPR | Precision | AUROC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Morgott 1024 all-window | 0.1% | 34.71% | 0.09% | 99.66% | 0.923 | 0.931 |
| Morgott 1024 all-window | 0.5% | 45.99% | 0.46% | 98.67% | 0.923 | 0.931 |
| Morgott 1024 all-window | 1.0% | 52.04% | 0.80% | 97.99% | 0.923 | 0.931 |
| Morgott 1024 all-window | 2.0% | 68.08% | 1.74% | 96.70% | 0.923 | 0.931 |
| Morgott 1024 all-window | 5.0% | 80.52% | 4.00% | 93.78% | 0.923 | 0.931 |
| Prompt Guard 2 86M segmented | 0.1% | 20.88% | 0.07% | 99.52% | 0.861 | 0.834 |
| Prompt Guard 2 86M segmented | 0.5% | 23.27% | 0.46% | 97.41% | 0.861 | 0.834 |
| Prompt Guard 2 86M segmented | 1.0% | 27.11% | 0.79% | 96.27% | 0.861 | 0.834 |
| Prompt Guard 2 86M segmented | 2.0% | 35.83% | 1.67% | 94.12% | 0.861 | 0.834 |
| Prompt Guard 2 86M segmented | 5.0% | 45.33% | 4.52% | 88.24% | 0.861 | 0.834 |

### Paired Morgott 1,024 versus segmented Prompt Guard 2 deltas

Each system selected its threshold independently on the consumed 6,000-row calibration role at the same target FPR, and both thresholds were transported to the same 14,000 evaluation identities.
Deltas are Morgott minus Prompt Guard percentage points with paired stratified bootstrap 95% intervals from 2,000 resamples and deterministic seed 42.

| Calibration FPR target | Recall delta | FPR delta | Precision delta | Restriction-rate delta |
|---:|---:|---:|---:|---:|
| 0.1% | +13.82 pp (+12.67 to +15.13) | +0.01 pp (-0.06 to +0.09) | +0.14 pp (-0.23 to +0.55) | +5.92 pp (+5.42 to +6.49) |
| 0.5% | +22.72 pp (+21.55 to +23.99) | +0.00 pp (-0.20 to +0.19) | +1.26 pp (+0.44 to +2.18) | +9.72 pp (+9.20 to +10.27) |
| 1.0% | +24.92 pp (+23.77 to +26.19) | +0.01 pp (-0.19 to +0.20) | +1.72 pp (+0.99 to +2.55) | +10.67 pp (+10.18 to +11.24) |
| 2.0% | +32.25 pp (+31.05 to +33.49) | +0.06 pp (-0.19 to +0.30) | +2.58 pp (+1.83 to +3.35) | +13.84 pp (+13.29 to +14.38) |
| 5.0% | +35.19 pp (+33.99 to +36.38) | -0.52 pp (-1.00 to -0.05) | +5.54 pp (+4.52 to +6.54) | +14.76 pp (+14.18 to +15.33) |

The current Morgott score aggregates every ordered 1,024-token window with 128-token overlap at the artifact level.
Prompt Guard 2 aggregates 512-token segments with 64-token overlap, so this comparison does not silently truncate long artifacts.

## Retained external and historical baselines

| System | Canonical TPR at 1% FPR | PromptShield | SEP | Evidence |
|---|---:|---:|---:|---|
| Registered Morgott 1024 native evaluation | 71.9% | 52.9% | 49.8% | retained consumed development |
| Historical mmBERT 512 | 55.2% | 48.0% | 38.8% | retained report level only |
| Llama Prompt Guard 2 86M, retained current panel | 43.1% | 15.7% | 3.3% | retained report level only |
| ModernGuard-1 | 0.0% | 0.1% | 2.4% | retained report level only |
| Qwen3Guard Stream 4B, query head | 1.7% | 5.9% | 0.5% | retained report level only |
| Qwen3Guard Stream 4B, jailbreak head | 37.7% | 9.2% | 8.0% | retained report level only |
| Kanana Safeguard Prompt 2.1B | 37.7% | 1.7% | 3.0% | retained report level only |

The retained guard table comes from versioned report-level evidence because several original row ledgers are not present locally.
Those rows cannot support new paired confidence intervals or post-hoc threshold selection.
The retained GPT-OSS Safeguard 20B shared-binary-prompt result was 12.75% valid-output recall, 0.06% FPR, 99.36% precision, 1.64-second mean latency, 10.58-second p95 latency, and $0.0687 per 1,000 inputs.
That retained GPT-OSS result is a shared-binary-prompt comparison and is separate from the supplementary native screen below.
The retained DeepSeek V4 Flash 0731 standalone logprob evidence reached 0.953 AUROC and 0.938 average precision on the consumed evaluation role.

### Supplementary GPT-OSS Safeguard 20B native screen

This 256-row screen passed its canary and used the official 456-word policy with Harmony-style chat rather than the retained shared-binary prompt.
Invalid and terminally failed outputs are conservatively positive in the quality metrics.

| Reasoning | Recall | FPR | Valid outputs | Client p50 | Client p95 | Cost | Failures |
|---|---:|---:|---:|---:|---:|---:|---|
| Low | 13.402% | 1.258% | 96.094% | 0.313s | 1.812s | $0.02165 | invalid_response: 10 |
| Medium | 18.557% | 2.516% | 93.359% | 0.359s | 3.425s | $0.02146 | http_429: 6, invalid_response: 11 |

Neither low nor medium reasoning is reliable enough to expand to the complete panel or displace the main benchmark arms.

## Real-world defense layers and excluded comparators

An artifact-level guard is one advisory layer, while realistic deployment also requires trusted provenance, least-privilege tool schemas, deterministic reference-monitor authorization, output validation, and audit telemetry.
[Meta's official Prompt Guard 2 model card](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) recommends segmenting longer prompts and scanning segments in parallel; this benchmark adds a declared 64-token overlap and maximum artifact aggregation to that 512-token model contract.
[Meta's official LlamaFirewall repository](https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall) describes a defense-in-depth framework over agent interactions, so LlamaFirewall is treated as a trace-level architecture and not inserted into a static text ROC table.
[Azure's official Prompt Shields quickstart](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-jailbreak) requires an Azure AI Content Safety resource, and none is available in the benchmark subscription, so Azure Prompt Shields has no measured comparison row.
[OpenAI's official GPT-OSS Safeguard guide](https://developers.openai.com/cookbook/articles/gpt-oss-safeguard-guide) defines the native evaluation around a developer-provided policy and Harmony formatting, so the retained shared-binary-prompt result remains distinct from the measured supplementary native-policy screen.
[OpenRouter's official provider-routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection) and [router-metadata documentation](https://openrouter.ai/docs/guides/features/router-metadata) support explicit provider selection and route auditing, so this benchmark pins one provider, disables fallbacks, requires advertised parameters, and validates returned identity against [frozen live 0731 endpoint metadata](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints).
OpenRouter endpoint capabilities and commercial metadata can change, so the endpoint record is a per-run snapshot rather than a permanent provider claim.
[WASP's official repository](https://github.com/facebookresearch/wasp) defines an end-to-end browser-agent benchmark, so WASP remains sealed for a separately frozen outcome study and is not flattened into this static detector table.

## Long-context result

The 20,000-row panel contained 225 multi-window artifacts and a maximum of 68 windows.
In the balanced artifact-level approximation, the long-character bucket reached 85.61% recall but 70.37% FPR over only 54 negatives and 139 positives.
That complete-bucket diagnostic uses the preliminary artifact-level profile, not the later exact promoted profile; it remains a warning because the promoted prompt and full-context-first architecture are unchanged.
Among the 54 evaluation negatives of at least 4,096 characters, local-high caused 0 restrictions, 41 entered the middle zone, and retained first full-context review caused all 38 false restrictions for 70.37% FPR.
This decomposition is exact through the first full-context review, so maximum-window local aggregation is not the direct cause of those balanced-gate false restrictions.
Exploratory second-window and top-two aggregation were non-dominant and are not recommended.
The long-bucket false-positive result is the strongest warning in this study and makes representative long-benign shadow traffic a prerequisite for deployment work.

### Complete long-bucket prompt comparison

A matched follow-up tested the current and revised full-context prompts on all 193 identities behind the long-character headline: 54 benign and 139 attacked artifacts.
Both arms used the pinned Cloudflare strict hard-verdict contract with reasoning disabled, identical redacted inputs, conservative failure handling, and the original balanced local gates.
The revised prompt used channel-aware trusted-task interpretation for direct requests and untrusted webpages, messages, emails, and documents.

| Arm | Reviewer-only recall | Reviewer-only FPR | Balanced artifact-level recall | Balanced artifact-level FPR | Valid outputs | Client p50 / p95 | Recorded cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current prompt | 92.09% | 68.52% | 88.49% | 68.52% | 190 / 193 | 1.29s / 3.49s | $0.667 |
| Revised channel/task-conditioned prompt | 59.71% | 0.00% | 72.66% | 0.00% | 191 / 193 | 1.32s / 3.77s | $0.671 |

The revised arm removed all 37 matched current-arm false restrictions, moving from 37 of 54 to 0 of 54, while attack detections fell from 123 of 139 to 101 of 139.
The revised balanced recall was 84.62% on untrusted content but only 57.38% on direct-user inputs; the largest measured source weakness was PromptShield at 25 of 49 attacks detected, or 51.02% recall.
Zero observed false positives does not establish zero population FPR: the revised arm's Wilson 95% FPR interval is 0.00% to 6.64%.
The matched current arm differs slightly from the original 85.61% recall / 70.37% FPR headline because this follow-up uses strict hard verdicts on deterministically redacted inputs, while the original approximation used retained logprob thresholding on the original inputs.
The complete comparison is still post-hoc and selection-ineligible because all 193 identities were already consumed, 109 inputs required safety redaction, and the trusted tasks were reconstructed rather than supplied by the original runtime.

#### Minimal-change follow-up

One additional sentence covering implicit, role-based, encoded, and obfuscated redirection failed its frozen advance rule: recall remained 72.66% while FPR rose from 0 of 54 to 4 of 54, or 7.41%.
That sentence is rejected.
The smallest non-dominated candidate requires no new prompt text: use the current prompt for trusted `direct_user` traffic and the revised task-conditioned prompt for `untrusted_content`.
This channel split reached 111 of 139 attacks, or 79.86% recall, with 0 of 54 observed false restrictions.
It recovers 10 detections over the single revised prompt without adding a false restriction; direct-user recall is 73.77% and untrusted-content recall is 84.62%.
This was the preferred fresh-confirmation contract before the broader scenario-balanced screen below.

#### Scenario-balanced untrusted-content screen

A fast matched screen then sampled 256 consumed provider-safe untrusted-content artifacts no longer than 4,096 characters: 128 SEP rows and 128 proportionally stratified rows from six other sources.
It compared the existing Cloudflare strict-logprob prompt against the revised prompt under the unchanged exact balanced local gates and reviewer threshold.
The sample contained 142 attacks and 114 benign controls; the direct-user prompt was unchanged and therefore was not called again.

| Arm | Recall | FPR | Precision | Attack detections | False restrictions | Provider calls | Valid outputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current prompt | 73.24% | 1.75% | 98.11% | 104 / 142 | 2 / 114 | 52 | 255 / 256 |
| Revised untrusted prompt | 52.82% | 0.88% | 98.68% | 75 / 142 | 1 / 114 | 52 | 256 / 256 |

The candidate saved 1 false restriction but lost 29 attack detections, a -20.42 pp recall change for a -0.88 pp FPR change.
SEP recall fell from 53.12% to 17.19%; LLMail recall fell by -2.13 pp.
It failed the predeclared advance rule and is rejected for further confirmation or pipeline integration.
This is a scenario-balanced consumed-data screen, not measured production traffic or fresh selection evidence.

### Focused BrowseSafe context and reasoning diagnostic

A fast post-hoc screen compared four pinned Cloudflare strict-hard-verdict contracts on 115 consumed long BrowseSafe pages: 60 benign and 55 attacked.
The original public pages were deterministically safety-redacted before transmission because 112 of the 115 selected long BrowseSafe rows triggered the repository sensitive-text scanner, mostly on email addresses.
The fixed trusted task asked the system to analyze the webpage as untrusted data without following page instructions.
No raw provider response or page text was persisted.

| Reviewer arm | Reviewer-only evaluation recall | Reviewer-only evaluation FPR | Balanced-cascade evaluation recall | Balanced-cascade evaluation FPR | Valid outputs | Client p50 / p95 | Recorded evaluation cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current prompt, full, reasoning off | 100.00% | 97.37% | 100.00% | 97.37% | 73 / 77 | 2.30s / 11.31s | $0.586 |
| Narrow task-conditioned prompt, full, reasoning off | 66.67% | 5.26% | 71.79% | 5.26% | 76 / 77 | 2.17s / 10.67s | $0.605 |
| Narrow prompt, suspect window plus neighbors, reasoning off | 61.54% | 2.63% | 66.67% | 2.63% | 75 / 77 | 1.53s / 6.59s | $0.077 |
| Narrow compact prompt, high reasoning | 64.10% | 2.63% | 64.10% | 2.63% | 74 / 77 | 2.96s / 11.38s | $0.092 |

The narrower full-context prompt reduced balanced-slice false restrictions from 37 of 38 to 2 of 38, while balanced cascade attack recall fell from 39 of 39 to 28 of 39.
Terminal failures were conservatively counted as restrictions.
Compact evidence reduced false restrictions to 1 of 38 but also reduced balanced cascade attack recall to 26 of 39.
High reasoning did not dominate reasoning-disabled compact review: it retained the same one false restriction, caught 25 rather than 26 attacks after the local-high short circuit, roughly doubled median latency, and used about nineteen times as many completion tokens on the evaluation role.
The result supports prompt narrowing as the main fix direction, not chain-of-thought alone, but none of these contracts is eligible to replace the frozen cascade because the panel is already consumed, safety redaction changed its inputs, and 38 benign examples cannot certify a low FPR.

## Incumbent-gate 1,024-context mutation replay

The local-high gate caught 127 of 240 eligible base attacks, for 52.9167% base recall.
This local-only replay used local-high `0.99999`, direct low `0.2`, and untrusted low `0.1`.

| Mutations per caught base attack | Local-high ASR | Effective local-high recall | Local-pass ASR floor |
|---:|---:|---:|---:|
| 1 | 1.8898% | 51.9167% | 0.0000% |
| 2 | 3.5249% | 51.0514% | 0.0000% |
| 4 | 6.1429% | 49.6661% | 0.0000% |
| 8 | 9.4253% | 47.9291% | 0.0000% |
| 16 | 11.8590% | 46.6413% | 0.0000% |
| 25 | 12.5984% | 46.2500% | 0.0000% |

The ASR denominator is the set of base attacks caught by the pre-promotion incumbent local-high gate, and the mutation rows are already-open synthetic development evidence.
The zero local-pass ASR floor means mutations moved caught attacks into review rather than local pass; it does not imply a zero full-cascade ASR.
Full-cascade mutation ASR remains unmeasured until review-zone provider outcomes are frozen.

## Sealed LogInject local routing

[The official public synthetic LogInject-1.0 artifact](https://doi.org/10.5281/zenodo.20436935) supplies 2,569 adversarial log entries that this panel converts into 2,569 matched clean and attacked 50-entry log pairs spanning 3,615 to 4,566 tokens.
The attacked batches inject ordered complete entries or fragments through error message, http referer, json api, ssh username, user agent vectors.
LogInject was absent from fitting, architecture selection, prompt selection, and threshold selection, so this is a source-held-out out-of-distribution generated-log diagnostic rather than natural-document or production-log evidence.
Pair-level overlap filtering checked 1,073,230 canonical training rows, 22,092 matched-pair references, and 18,284 PromptShield training references, and no pair was removed.
The sealed local replay measured 2569 clean and attacked pairs exactly once.
This local-only table uses the preliminary profile grid; its row named Balanced is not the later exact promoted balanced profile, whose complete remote result appears below.

| Profile | Clean local-high rate | Clean review-or-high rate | Attack local-high recall | Attack review-or-high recall |
|---|---:|---:|---:|---:|
| Conservative | 0.0000% | 100.0000% | 3.2308% | 100.0000% |
| Balanced | 0.0000% | 100.0000% | 3.2308% | 100.0000% |
| High Recall | 0.1557% | 100.0000% | 32.1137% | 100.0000% |

At the preliminary balanced gates, only 83 of 2569 attacks reached local-high, or 3.2308%, while every attacked pair reached review-or-high and zero attacks passed locally.
Review-or-high is routing coverage rather than completed detection, and every clean pair also entered review, so remote reviewer outcomes are required to measure cascade recall and false restriction.
Attack insertion changed the local score by a mean of 0.3547 and a median of 0.4301, with 75.5936% of pairs moving upward.

### Sealed post-hoc local threshold diagnostic

These thresholds and operating points were computed after the sealed panel was opened and are diagnostic only.
They cannot tune, select, or revise any threshold in this run.

| Post-hoc target FPR | Exact threshold | Attack recall | Observed clean FPR | Attacks at or above threshold | Clean pairs at or above threshold |
|---:|---:|---:|---:|---:|---:|
| 0.00% | `0.997110419` | 25.6520% | 0.0000% | 659/2569 | 0/2569 |
| 0.10% | `0.994779874` | 28.6882% | 0.0779% | 737/2569 | 2/2569 |
| 1.00% | `0.865224083` | 48.4235% | 0.9731% | 1244/2569 | 25/2569 |
| 2.00% | `0.773216365` | 53.3281% | 1.9852% | 1370/2569 | 51/2569 |
| 6.25% | `0.555194078` | 61.4247% | 6.2281% | 1578/2569 | 160/2569 |
| 32.00% | `0.369315080` | 74.8930% | 31.8023% | 1924/2569 | 817/2569 |

Future work must build a separate long-security-log calibration role and fresh shadow evaluation set, evaluate the finer `0.995` to `0.999` local-high grid only on that calibration role, freeze one threshold, and transport it unchanged to the fresh evaluation evidence.

## Sealed LogInject exact remote cascade

The write-once Cloudflare strict-logprob replay resolved 2,569 matched clean and attacked pairs with 18,081 unique provider calls and 31 terminal failures.
The initial concurrency-8 execution failed the predefined reliability gate, so the write-once replay resumed at concurrency 4 without changing thresholds, provider, prompt, or routing semantics.
All 31 terminal failures fail closed as restricted under the frozen reviewer failure rule.

| Profile | Pairs | Attack recall | Paired clean false restrictions | Attack restricted and clean clear | Both restricted | Both clear | Provider calls | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced | 2,569 | 96.57% (2481/2569) | 0.54% (14/2569) | 2467 | 14 | 88 | 17449 | 31 |
| Incumbent | 2,569 | 93.42% (2400/2569) | 0.35% (9/2569) | 2391 | 9 | 169 | 14792 | 22 |

Paired clean false restrictions and attack recall are measured end-to-end under the frozen exact cascade, while local-only post-hoc thresholds remain ineligible for tuning.

## Provider compatibility canary

| Provider | Contract | Quantization | Valid | Client p50 | Client p95 | Failure codes | Canary cost |
|---|---|---|---:|---:|---:|---|---:|
| akashml | `strict_logprob` | fp8 | 16/16 | 7.75s | 27.06s | none | $0.000530 |
| atlas-cloud | `strict_hard_verdict` | unknown | 4/16 | 5.27s | 7.06s | invalid_response: 12 | $0.000129 |
| baidu | `strict_hard_verdict` | fp8 | 16/16 | 4.42s | 6.04s | none | $0.000243 |
| baseten | `forced_tool` | fp8 | 0/16 | 4.21s | 4.72s | http_error: 16 | $0.000000 |
| cloudflare | `strict_logprob` | unknown | 16/16 | 4.62s | 10.74s | none | $0.000585 |
| coreweave | `forced_tool` | fp8 | 0/16 | 0.03s | 0.41s | http_error: 16 | $0.000000 |
| decart | `strict_hard_verdict` | fp4 | 16/16 | 13.62s | 16.65s | none | $0.000385 |
| deepinfra | `strict_hard_verdict` | fp8 | 16/16 | 4.81s | 7.70s | none | $0.000288 |
| digitalocean | `relaxed_json` | unknown | 15/16 | 5.02s | 6.64s | invalid_response: 1 | $0.000312 |
| fireworks | `strict_logprob` | unknown | 0/16 | 0.53s | 1.13s | http_error: 16 | $0.000000 |
| inceptron | `strict_logprob` | fp4 | 16/16 | 5.58s | 13.66s | none | $0.000865 |
| open-inference | `strict_hard_verdict` | unknown | 16/16 | 7.32s | 14.60s | none | $0.000306 |
| phala | `strict_hard_verdict` | unknown | 16/16 | 5.45s | 7.25s | none | $0.000862 |
| siliconflow | `strict_hard_verdict` | fp8 | 16/16 | 6.46s | 7.76s | none | $0.000929 |
| wafer | `strict_logprob` | unknown | 0/16 | 6.19s | 12.91s | http_error: 16 | $0.000000 |

The canary used 16 unique rows per provider and tests exact transport compatibility only.
It is too small for provider quality selection, sustained throughput, or tail-latency claims.

### Matched 1,024-row provider panel

| Provider | Contract | Quantization | Rows | Valid outputs | Recall | FPR | Client p50 | Client p95 | Cost | Selection |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baidu | strict hard verdict | fp8 | 1,024 | 99.609% | 60.15% | 2.20% | 0.716s | 4.913s | $0.01324 | not selected |
| decart | strict hard verdict | fp4 | 1,024 | 100.000% | 62.21% | 1.42% | 0.925s | 12.957s | $0.02080 | not selected |
| deepinfra | strict hard verdict | fp8 | 1,024 | 100.000% | 56.30% | 1.10% | 0.865s | 5.521s | $0.01856 | not selected |
| cloudflare | strict logprob | unknown | 1,024 | 100.000% | 61.70% | 1.89% | 0.708s | 4.923s | $0.03262 | logprob |

The frozen strict-logprob winner is cloudflare.
No strict hard-verdict provider survived the complete overall and declared-slice quality gate; Decart remains a measured diagnostic only.
Quantization labels come from the endpoint snapshot frozen at run start, and `unknown` means the provider did not declare a precision label in that snapshot.
Provider quality is measured on matched calibration-safe artifacts and does not replace the exact evaluation-stage cascade results.

### Corrected unique-sample provider load

| Provider | Contract | Concurrency | Length mix | Requests | Terminal failures | Requests/s | Input tokens/s | Client p50 | Client p95 | Client p99 | Cost |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cloudflare | strict logprob | 1 | 1024-4095: 11, 4096-15999: 10, <1024: 11 | 32 | 0 (0.000%) | 0.994 | 847.1 | 0.718s | 2.178s | 4.033s | $0.00300 |
| cloudflare | strict logprob | 4 | 1024-4095: 10, 4096-15999: 11, <1024: 11 | 32 | 0 (0.000%) | 4.374 | 3960.4 | 0.776s | 1.127s | 2.248s | $0.00321 |
| cloudflare | strict logprob | 8 | 1024-4095: 10, 4096-15999: 11, <1024: 11 | 32 | 0 (0.000%) | 7.880 | 7164.0 | 0.875s | 1.178s | 1.379s | $0.00337 |
| decart | strict hard verdict | 1 | 1024-4095: 10, 4096-15999: 11, <1024: 11 | 32 | 0 (0.000%) | 1.117 | 1059.3 | 0.820s | 1.367s | 1.507s | $0.00081 |
| decart | strict hard verdict | 4 | 1024-4095: 11, 4096-15999: 10, <1024: 10, >=16000: 1 | 32 | 0 (0.000%) | 2.841 | 3094.4 | 0.772s | 3.176s | 4.991s | $0.00093 |
| decart | strict hard verdict | 8 | 1024-4095: 11, 4096-15999: 10, <1024: 10, >=16000: 1 | 32 | 0 (0.000%) | 5.677 | 11737.7 | 0.879s | 1.829s | 4.453s | $0.00124 |

Every load cell reports its own input-length mix, uses unique provider-safe samples, and excludes archived length-confounded load artifacts.
The Decart load cells are retained as transport-performance diagnostics and do not make Decart an eligible hard-verdict winner.
Input tokens per second and requests per second are the primary throughput measures for these one-token verdict contracts.

## Runtime

| System | Evidence | Artifacts | Throughput | Input-token throughput | Peak reserved VRAM |
|---|---|---:|---:|---:|---:|
| Morgott 1024 CUDA BF16 | measured local CUDA | 20,000 | 76.43 artifacts/s | 15690 tokens/s | 2.45 GiB |
| Prompt Guard 2 CUDA FP16 | measured local CUDA | 20,000 | 90.58 artifacts/s | 19419 tokens/s | 3.57 GiB |
| DeepSeek 0731 Cloudflare retained ledger | retained serial client ledger | 20,000 | 0.857 serial requests/s | 449.2 tokens/s | unavailable |

Local CUDA throughput excludes model load time and cannot be substituted for Azure end-to-end latency.
The retained DeepSeek throughput is a sum-of-client-latencies serial equivalent, not a concurrent load measurement.

### CUDA versus OpenVINO parity

The 512-row OpenVINO BF16 audit measured 9.40 artifacts/s and 1805 input tokens/s.
Its maximum absolute score delta was 0.1004, and the worst threshold-decision disagreement was 0.59% at threshold `0.99999`.
This parity sample measures numerical and threshold agreement only and is not a full OpenVINO quality evaluation.
The disagreement exceeds the 0.5% parity gate, so CUDA and OpenVINO quality results must remain runtime-specific.

### Full runtime-specific CUDA and OpenVINO comparison

The full OpenVINO replay scored 20,000 artifacts in 2640.38 seconds at 7.575 artifacts/s and 1555 input tokens/s.
CUDA and OpenVINO each select a separate numerical threshold on the same frozen 6,000-row calibration role and transport only that runtime's threshold unchanged to the aligned consumed 14,000-row evaluation role.
No CUDA threshold is applied to OpenVINO, and no OpenVINO threshold is presented as a CUDA threshold.

| FPR target | CUDA threshold | OpenVINO threshold | CUDA recall | OpenVINO recall | Recall delta | CUDA FPR | OpenVINO FPR | FPR delta | Precision delta | Restriction delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1% | `0.99999824` | `0.999998174` | 34.71% | 34.79% | +0.08 pp | 0.09% | 0.10% | +0.01 pp | -0.05 pp | +0.04 pp |
| 0.5% | `0.99998431` | `0.999983189` | 45.99% | 46.28% | +0.28 pp | 0.46% | 0.50% | +0.04 pp | -0.10 pp | +0.14 pp |
| 1.0% | `0.999941709` | `0.999943619` | 52.04% | 52.02% | -0.02 pp | 0.80% | 0.79% | -0.01 pp | +0.03 pp | -0.01 pp |
| 2.0% | `0.994939668` | `0.995971626` | 68.08% | 67.31% | -0.77 pp | 1.74% | 1.65% | -0.09 pp | +0.13 pp | -0.38 pp |
| 5.0% | `0.546012031` | `0.533760095` | 80.52% | 80.53% | +0.02 pp | 4.00% | 4.07% | +0.07 pp | -0.11 pp | +0.05 pp |

Evaluation AUROC was 0.923268 on CUDA and 0.923490 on OpenVINO, for an OpenVINO-minus-CUDA delta of +0.000222.
Evaluation average precision was 0.930884 on CUDA and 0.931273 on OpenVINO, for an OpenVINO-minus-CUDA delta of +0.000388.
These are runtime-specific consumed-development results rather than evidence that either runtime threshold transfers to the other.

## Warm Azure end-to-end load

The currently recorded live revision is `morgott-api--0000016` with profile `balanced-20260816`, policy `1c173136f385e8d755e01b73ebcf592acd025d6d94169fb76f1f10abdedd957f`, and threshold `e9b375b079667fd7c82bc5439fcde772ad3d3b04b7fb44277b8f180478558353`.
Its 30-request deployment smoke check passed, but it is not used as latency or throughput evidence for the matrix below.
This load run predates the promoted profile identity fields and remains incumbent-only evidence; it does not measure the balanced-20260816 deployment.
The deployment reported model `mmbert-lora-full-ctx1024-u17000-s42`, context length 1024, overlap 128, and ONNX identity `e58706410c76915b622e43b31dc23d6deb1a0416beb1448cf96b2259cb0cb0c5` before measurement.
The table reports all warm route, input-length, and concurrency cells separately rather than pooling unlike paths or lengths.
Across 4,800 measured requests, 4,714 returned HTTP 200, 86 had transport failures, the cell aggregates recorded zero explicit non-`allow` decisions, and the deployment reported 12,207 DeepSeek calls.
Decision-presence counts were not persisted, so the aggregates do not prove that every HTTP 200 response contained `decision: allow`.
The client timer starts before semaphore acquisition, so p50, p95, and p99 are queueing-inclusive burst latencies rather than pure in-service latencies.
The artifact's planning estimates are $4.924912 for this run and $4.924912 for the prior failed finalization attempt.
They are not strict upper bounds or invoiced spend because the calculation omits the reviewer system prompt and chat/schema overhead and uses Morgott rather than provider-native token counts; the hard $25 cap therefore is not independently proven by this artifact.
Because this completed study predates durable preauthorization, its evidence directory is now closed at the full $24 usable budget; the known recorded-plus-Azure planning amount is $17.972814, while unpriced terminal calls remain the reason no unused balance is claimed.

| Route fixture | Channel | Input tokens min/mean/max | Total input tokens | Input bytes | Concurrency | Successes | Errors | Requests/s | Input tokens/s | p50 | p95 | p99 | Reviewer calls | Observed routes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| allow | direct_user | 48/49.8/50 | 4,980 | 256 | 1 | 100/100 | 0 | 0.920 | 45.8 | 55.429s | 103.827s | 107.794s | 100 | pass: 100 |
| allow | direct_user | 48/49.8/50 | 4,980 | 256 | 4 | 100/100 | 0 | 3.786 | 188.5 | 13.769s | 25.280s | 26.411s | 100 | pass: 100 |
| allow | direct_user | 48/49.8/50 | 4,980 | 256 | 8 | 100/100 | 0 | 5.743 | 286.0 | 8.866s | 16.157s | 16.663s | 100 | pass: 100 |
| allow | direct_user | 50/50.9/51 | 5,090 | 256 | 16 | 100/100 | 0 | 5.746 | 292.5 | 8.493s | 16.685s | 17.402s | 100 | pass: 100 |
| allow | direct_user | 619/619.9/620 | 61,990 | 4,096 | 1 | 100/100 | 0 | 0.598 | 370.6 | 83.895s | 155.981s | 165.497s | 100 | pass: 100 |
| allow | direct_user | 619/619.9/620 | 61,990 | 4,096 | 4 | 100/100 | 0 | 1.317 | 816.3 | 38.984s | 72.412s | 75.377s | 100 | pass: 100 |
| allow | direct_user | 619/619.9/620 | 61,990 | 4,096 | 8 | 100/100 | 0 | 1.318 | 817.2 | 38.830s | 72.117s | 75.161s | 100 | pass: 100 |
| allow | direct_user | 620/620.9/621 | 62,090 | 4,096 | 16 | 100/100 | 0 | 1.306 | 811.1 | 39.394s | 73.012s | 75.776s | 100 | pass: 100 |
| allow | direct_user | 2,440/2440.9/2,441 | 244,090 | 16,384 | 1 | 100/100 | 0 | 0.206 | 502.5 | 248.355s | 461.529s | 481.092s | 300 | pass: 100 |
| allow | direct_user | 2,440/2440.9/2,441 | 244,090 | 16,384 | 4 | 100/100 | 0 | 0.287 | 700.1 | 177.484s | 331.675s | 347.578s | 300 | pass: 100 |
| allow | direct_user | 2,440/2440.9/2,441 | 244,090 | 16,384 | 8 | 100/100 | 0 | 0.285 | 695.1 | 177.619s | 333.955s | 347.542s | 300 | pass: 100 |
| allow | direct_user | 2,441/2441.0/2,441 | 244,100 | 16,384 | 16 | 100/100 | 0 | 0.286 | 699.3 | 177.436s | 331.738s | 345.933s | 300 | pass: 100 |
| allow | direct_user | 9,114/9115.8/9,116 | 911,580 | 61,440 | 1 | 100/100 | 0 | 0.059 | 541.1 | 849.471s | 1604.542s | 1668.754s | 1100 | pass: 100 |
| allow | direct_user | 9,114/9115.8/9,116 | 911,580 | 61,440 | 4 | 100/100 | 0 | 0.073 | 663.8 | 695.291s | 1304.574s | 1359.049s | 1100 | pass: 100 |
| allow | direct_user | 9,114/9115.8/9,116 | 911,580 | 61,440 | 8 | 100/100 | 0 | 0.073 | 665.4 | 693.043s | 1300.900s | 1354.884s | 1104 | pass: 100 |
| allow | direct_user | 9,116/9116.9/9,117 | 911,690 | 61,440 | 16 | 100/100 | 0 | 0.074 | 675.1 | 683.131s | 1283.976s | 1337.037s | 110 | pass: 10, restrict: 90 |
| high | untrusted_content | 52/52.9/53 | 5,290 | 256 | 1 | 100/100 | 0 | 3.215 | 170.1 | 15.798s | 29.479s | 30.798s | 0 | restrict: 100 |
| high | untrusted_content | 52/52.9/53 | 5,290 | 256 | 4 | 100/100 | 0 | 11.111 | 587.8 | 4.697s | 8.593s | 8.896s | 0 | restrict: 100 |
| high | untrusted_content | 52/52.9/53 | 5,290 | 256 | 8 | 100/100 | 0 | 14.168 | 749.5 | 3.885s | 6.754s | 7.056s | 0 | restrict: 100 |
| high | untrusted_content | 53/53.0/53 | 5,300 | 256 | 16 | 100/100 | 0 | 14.378 | 762.0 | 3.574s | 6.537s | 6.796s | 0 | restrict: 100 |
| high | untrusted_content | 655/655.9/656 | 65,590 | 4,096 | 1 | 100/100 | 0 | 0.961 | 630.1 | 52.668s | 99.030s | 103.087s | 0 | restrict: 100 |
| high | untrusted_content | 655/655.9/656 | 65,590 | 4,096 | 4 | 100/100 | 0 | 1.295 | 849.2 | 39.175s | 73.483s | 76.428s | 0 | restrict: 100 |
| high | untrusted_content | 655/655.9/656 | 65,590 | 4,096 | 8 | 100/100 | 0 | 1.291 | 846.5 | 39.441s | 73.596s | 76.772s | 0 | restrict: 100 |
| high | untrusted_content | 656/656.9/657 | 65,690 | 4,096 | 16 | 100/100 | 0 | 1.297 | 852.0 | 39.219s | 73.268s | 76.396s | 0 | restrict: 100 |
| high | untrusted_content | 2,586/2587.8/2,588 | 258,780 | 16,384 | 1 | 100/100 | 0 | 0.237 | 612.7 | 213.691s | 401.793s | 418.219s | 0 | restrict: 100 |
| high | untrusted_content | 2,586/2587.8/2,588 | 258,780 | 16,384 | 4 | 100/100 | 0 | 0.274 | 709.8 | 184.699s | 346.613s | 360.979s | 0 | restrict: 100 |
| high | untrusted_content | 2,586/2587.8/2,588 | 258,780 | 16,384 | 8 | 100/100 | 0 | 0.273 | 707.8 | 185.142s | 347.710s | 361.998s | 0 | restrict: 100 |
| high | untrusted_content | 2,588/2588.9/2,589 | 258,890 | 16,384 | 16 | 100/100 | 0 | 0.273 | 706.0 | 185.233s | 348.386s | 363.174s | 0 | restrict: 100 |
| high | untrusted_content | 9,668/9668.0/9,668 | 966,800 | 61,440 | 1 | 100/100 | 0 | 0.066 | 640.5 | 761.481s | 1433.656s | 1494.331s | 0 | restrict: 100 |
| high | untrusted_content | 9,668/9668.0/9,668 | 966,800 | 61,440 | 4 | 100/100 | 0 | 0.070 | 681.4 | 717.517s | 1348.476s | 1405.051s | 0 | restrict: 100 |
| high | untrusted_content | 9,668/9668.0/9,668 | 966,800 | 61,440 | 8 | 100/100 | 0 | 0.070 | 680.9 | 717.170s | 1349.770s | 1405.793s | 0 | restrict: 100 |
| high | untrusted_content | 9,668/9669.8/9,670 | 966,980 | 61,440 | 16 | 15/100 | 85 | 0.072 | 700.4 | 699.273s | 1316.267s | 1366.451s | 0 | missing: 85, restrict: 15 |
| review | untrusted_content | 49/49.9/50 | 4,990 | 256 | 1 | 100/100 | 0 | 1.118 | 55.8 | 45.844s | 84.591s | 88.554s | 100 | pass: 100 |
| review | untrusted_content | 49/49.9/50 | 4,990 | 256 | 4 | 100/100 | 0 | 3.567 | 178.0 | 14.898s | 26.822s | 27.933s | 100 | pass: 100 |
| review | untrusted_content | 49/49.9/50 | 4,990 | 256 | 8 | 100/100 | 0 | 5.844 | 291.6 | 9.009s | 16.587s | 17.100s | 100 | pass: 100 |
| review | untrusted_content | 50/50.0/50 | 5,000 | 256 | 16 | 100/100 | 0 | 5.614 | 280.7 | 9.100s | 16.788s | 17.809s | 100 | pass: 100 |
| review | untrusted_content | 605/605.9/606 | 60,590 | 4,096 | 1 | 100/100 | 0 | 0.634 | 384.0 | 78.168s | 149.063s | 156.277s | 100 | pass: 100 |
| review | untrusted_content | 605/605.9/606 | 60,590 | 4,096 | 4 | 100/100 | 0 | 1.356 | 821.9 | 37.825s | 70.137s | 73.311s | 100 | pass: 100 |
| review | untrusted_content | 605/605.9/606 | 60,590 | 4,096 | 8 | 100/100 | 0 | 1.343 | 814.0 | 38.441s | 71.116s | 73.827s | 100 | pass: 100 |
| review | untrusted_content | 606/606.9/607 | 60,690 | 4,096 | 16 | 100/100 | 0 | 1.332 | 808.5 | 38.248s | 71.407s | 74.247s | 100 | pass: 100 |
| review | untrusted_content | 2,387/2387.9/2,388 | 238,790 | 16,384 | 1 | 100/100 | 0 | 0.187 | 446.6 | 267.825s | 507.153s | 529.610s | 400 | pass: 100 |
| review | untrusted_content | 2,387/2387.9/2,388 | 238,790 | 16,384 | 4 | 100/100 | 0 | 0.289 | 690.4 | 175.314s | 330.489s | 342.315s | 400 | pass: 100 |
| review | untrusted_content | 2,387/2387.9/2,388 | 238,790 | 16,384 | 8 | 100/100 | 0 | 0.291 | 693.7 | 174.801s | 327.003s | 340.472s | 400 | pass: 100 |
| review | untrusted_content | 2,388/2388.0/2,388 | 238,800 | 16,384 | 16 | 100/100 | 0 | 0.291 | 695.4 | 174.499s | 327.010s | 340.198s | 400 | pass: 100 |
| review | untrusted_content | 8,917/8917.0/8,917 | 891,700 | 61,440 | 1 | 100/100 | 0 | 0.058 | 519.5 | 869.639s | 1632.153s | 1699.873s | 1100 | pass: 100 |
| review | untrusted_content | 8,917/8917.0/8,917 | 891,700 | 61,440 | 4 | 100/100 | 0 | 0.075 | 670.0 | 673.038s | 1265.268s | 1317.736s | 1100 | pass: 100 |
| review | untrusted_content | 8,917/8917.0/8,917 | 891,700 | 61,440 | 8 | 100/100 | 0 | 0.075 | 669.9 | 674.622s | 1264.482s | 1318.596s | 1104 | pass: 100 |
| review | untrusted_content | 8,917/8917.9/8,918 | 891,790 | 61,440 | 16 | 99/100 | 1 | 0.075 | 668.8 | 674.946s | 1266.342s | 1319.834s | 1089 | missing: 1, pass: 99 |

At 60 KiB and concurrency 16, the allow fixture returned 10 pass and 90 restrict routes across 100 HTTP 200 responses while reporting only 110 DeepSeek calls.
The corresponding high fixture returned 15 HTTP 200 responses and 85 transport failures, while the review fixture returned 99 HTTP 200 responses and 1 transport failure.
The fixtures include a cell-specific nonce, so payload bytes differ across concurrency cells and the 10/90 boundary tracks nonce-length changes.
This cell is confounded and cannot support a scaling, saturation, or detector-quality conclusion.

The allow fixtures made 5,414 DeepSeek calls, so this run did not measure a true local-allow path.
The nominal 4,096-byte and 16,384-byte fixtures were only about 605-657 and 2,387-2,589 Morgott tokens, respectively, so they do not satisfy the planned approximately 1,024-token and 4,096-token cells.

### Azure resource observations

| Azure metric | Unit | Samples | Maximum reported maximum | Maximum reported average | Maximum reported total |
|---|---|---:|---:|---:|---:|
| UsageNanoCores | NanoCores | 413 | 2052914740.000 | 2048238245.500 | 4096476491.000 |
| WorkingSetBytes | Bytes | 413 | 2637021184.000 | 2636904448.000 | 5273808896.000 |
| Requests | Count | 413 | 289.000 | 202.000 | 404.000 |
| Replicas | Count | 413 | 1.000 | 1.000 | 2.000 |
| CpuPercentage | Percent | 413 | 102.000 | 102.000 | 204.000 |
| MemoryPercentage | Percent | 413 | 59.000 | 59.000 | 118.000 |
| ResponseTime | MilliSeconds | 413 | 216153.000 | 216148.000 | 432296.000 |

Azure `Total` values accumulate samples within each time bucket and are not concurrency counts.
The Replicas series reported maximum and average values of one; its total of two is not evidence that two replicas ran concurrently.

## Representative traffic projections

These figures are arithmetic sensitivity analyses from the exact provider-safe balanced cascade rates, not observations from production traffic.

| Attack prevalence | Expected advisory precision | Expected advisory review rate | True signals per 10,000 | False signals per 10,000 |
|---:|---:|---:|---:|---:|
| 0.01% | 0.52% | 1.38% | 0.72 | 137.32 |
| 0.10% | 4.99% | 1.44% | 7.20 | 137.19 |
| 1.00% | 34.62% | 2.08% | 72.00 | 135.96 |
| 5.00% | 73.40% | 4.90% | 360.02 | 130.46 |

At 0.1% assumed attack prevalence, the balanced rates project only about 4.99% advisory precision because false positives dominate rare attacks.
This is why aggregate benchmark precision cannot answer real review-volume questions without a representative traffic denominator.

### Direct-versus-untrusted traffic-mix sensitivity

These arithmetic projections hold attack prevalence equal within direct and untrusted channels, weight the exact provider-safe balanced channel recall and FPR by the declared traffic mix, and do not claim measured production traffic.

| Direct / untrusted mix | Attack prevalence | Mixed recall | Mixed FPR | Expected advisory precision | Expected advisory review rate | True signals per 10,000 | False signals per 10,000 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 90/10 | 0.01% | 81.56% | 1.56% | 0.52% | 1.57% | 0.82 | 155.75 |
| 90/10 | 0.10% | 81.56% | 1.56% | 4.98% | 1.64% | 8.16 | 155.61 |
| 90/10 | 1.00% | 81.56% | 1.56% | 34.59% | 2.36% | 81.56 | 154.21 |
| 90/10 | 5.00% | 81.56% | 1.56% | 73.38% | 5.56% | 407.82 | 147.98 |
| 50/50 | 0.01% | 69.75% | 0.99% | 0.70% | 0.99% | 0.70 | 98.74 |
| 50/50 | 0.10% | 69.75% | 0.99% | 6.60% | 1.06% | 6.97 | 98.65 |
| 50/50 | 1.00% | 69.75% | 0.99% | 41.64% | 1.68% | 69.75 | 97.76 |
| 50/50 | 5.00% | 69.75% | 0.99% | 78.80% | 4.43% | 348.73 | 93.81 |
| 20/80 | 0.01% | 60.88% | 0.56% | 1.08% | 0.57% | 0.61 | 55.98 |
| 20/80 | 0.10% | 60.88% | 0.56% | 9.82% | 0.62% | 6.09 | 55.93 |
| 20/80 | 1.00% | 60.88% | 0.56% | 52.35% | 1.16% | 60.88 | 55.42 |
| 20/80 | 5.00% | 60.88% | 0.56% | 85.13% | 3.58% | 304.41 | 53.18 |

## Pending evidence and safe next action

All frozen evidence stages completed, but the Azure local-allow and intended approximately 1,024-token and 4,096-token cells and full-cascade mutation outcomes remain unmeasured.
No missing result is treated as zero and no retained 512 result is relabeled as 1,024-context evidence.
The shortest safe next action is to keep the promoted advisory profile frozen while collecting prospective task-bearing long benign and matched-attack shadow traffic.
WASP remains sealed for a separate browser-agent outcome study.

## Limitations

- The 6,000-row calibration role and 14,000-row evaluation role are already consumed development evidence.
- PromptShield and SEP are public transfer development panels, not production traffic.
- No representative adjudicated production traffic was available.
- The Azure run did not measure a local-allow path, did not hit the intended middle token lengths, and used different nonce payloads across concurrency cells.
- Azure remote-cost values are planning estimates rather than strict upper bounds or independently verified spend.
- Mutation robustness is local-routing evidence only because full-cascade mutation review was not run.
- The cascade-flow ablation contained no long untrusted artifact with every window below the local low gate, so that routing case remains unmeasured.
- The reviewer prompt diagnostic used consumed, deterministically redacted BrowseSafe rows and a fixed synthetic trusted task, so it cannot select a production contract or certify a low FPR.
- The scenario-balanced prompt screen deliberately upweighted non-SEP sources and is consumed development evidence, not representative production traffic.
- Report-level baselines with missing ledgers cannot support paired significance tests.
- The provider canary is a schema and routing test rather than a quality benchmark.
- The maintained runtime requests Cloudflare and disables fallbacks, but its response parser does not independently attest the returned provider build; the benchmark ledgers performed that identity validation.
- Learned scores remain advisory and never grant tool, data, network, credential, or financial authority.
