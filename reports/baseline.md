# Broad jailbreak-sensor baseline

Generated: 2026-07-20T06:36:28+00:00

This is a P0 shadow-mode text sensor for direct jailbreak and prompt-injection attempts. It is not a harmful-content classifier, a block decision, or an authorization boundary.

## Data

| Partition | Rows | Attack | Non-attack |
|---|---:|---:|---:|
| bipia_clean_context | 167 | 0 | 167 |
| bipia_context | 375 | 375 | 0 |
| bipia_payload | 125 | 125 | 0 |
| do_not_answer | 937 | 0 | 937 |
| harmbench | 400 | 0 | 400 |
| indirect_train | 1212 | 500 | 712 |
| jailbreaks_over_time | 22096 | 3901 | 18195 |
| multi_turn | 4136 | 4136 | 0 |
| nemotron_agentic_ipi | 676 | 676 | 0 |
| notinject | 339 | 0 | 339 |
| oasst1_chat | 1582 | 0 | 1582 |
| oasst1_position_stress | 500 | 0 | 500 |
| prompt_injections | 116 | 60 | 56 |
| tensor_trust_attack | 908 | 908 | 0 |
| tensor_trust_context | 1346 | 1346 | 0 |
| toxic_chat | 4703 | 73 | 4630 |
| train | 35912 | 311 | 35601 |
| xstest | 450 | 0 | 450 |

Exact normalized duplicates are removed, evaluation text duplicated in training is blocked, OASST1 conversations are grouped by tree, and the multi-turn corpus is grouped by attacker goal.

## Shadow-review operating point

The character n-gram model threshold is 0.898587. It is the recommended high-precision starting point for shadow review, selected only on deterministic validation groups at an 85% minimum precision floor; the official test partitions were not used for threshold selection.
Training used 28726 rows; 7186 rows across 3036 lineage groups were reserved for threshold selection. The selected profile observes 0.8947 precision, 0.5152 recall, and 0.0006 FPR on that source mixture.
The separate untrusted-content sensor threshold is 0.778259; it requires zero false positives on its 226-row BIPIA training holdout and scores the maximum whole-document/paragraph signal. Both sensors remain shadow-only.

| Detector | Evaluation | Recall | FPR | False signals / 10k | Precision | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| no_guard | bipia_clean_context | — | 0.0000 | 0.0000 | 0.0000 | — |
| no_guard | bipia_context | 0.0000 | — | — | 0.0000 | — |
| no_guard | bipia_payload | 0.0000 | — | — | 0.0000 | — |
| no_guard | do_not_answer | — | 0.0000 | 0.0000 | 0.0000 | — |
| no_guard | harmbench | — | 0.0000 | 0.0000 | 0.0000 | — |
| no_guard | jailbreaks_over_time | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1765 |
| no_guard | multi_turn | 0.0000 | — | — | 0.0000 | — |
| no_guard | nemotron_agentic_ipi | 0.0000 | — | — | 0.0000 | — |
| no_guard | notinject | — | 0.0000 | 0.0000 | 0.0000 | — |
| no_guard | oasst1_chat | — | 0.0000 | 0.0000 | 0.0000 | — |
| no_guard | oasst1_position_stress | — | 0.0000 | 0.0000 | 0.0000 | — |
| no_guard | prompt_injections | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5172 |
| no_guard | tensor_trust_attack | 0.0000 | — | — | 0.0000 | — |
| no_guard | tensor_trust_context | 0.0000 | — | — | 0.0000 | — |
| no_guard | toxic_chat | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0155 |
| no_guard | xstest | — | 0.0000 | 0.0000 | 0.0000 | — |
| no_guard | harmful_nonattack | — | 0.0000 | 0.0000 | 0.0000 | — |
| no_guard | external_hard_negatives | — | 0.0000 | 0.0000 | 0.0000 | — |
| exact_match | bipia_clean_context | — | 0.0000 | 0.0000 | 0.0000 | — |
| exact_match | bipia_context | 0.0000 | — | — | 0.0000 | — |
| exact_match | bipia_payload | 0.0000 | — | — | 0.0000 | — |
| exact_match | do_not_answer | — | 0.0000 | 0.0000 | 0.0000 | — |
| exact_match | harmbench | — | 0.0000 | 0.0000 | 0.0000 | — |
| exact_match | jailbreaks_over_time | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1765 |
| exact_match | multi_turn | 0.0000 | — | — | 0.0000 | — |
| exact_match | nemotron_agentic_ipi | 0.0000 | — | — | 0.0000 | — |
| exact_match | notinject | — | 0.0000 | 0.0000 | 0.0000 | — |
| exact_match | oasst1_chat | — | 0.0000 | 0.0000 | 0.0000 | — |
| exact_match | oasst1_position_stress | — | 0.0000 | 0.0000 | 0.0000 | — |
| exact_match | prompt_injections | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5172 |
| exact_match | tensor_trust_attack | 0.0000 | — | — | 0.0000 | — |
| exact_match | tensor_trust_context | 0.0000 | — | — | 0.0000 | — |
| exact_match | toxic_chat | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0155 |
| exact_match | xstest | — | 0.0000 | 0.0000 | 0.0000 | — |
| exact_match | harmful_nonattack | — | 0.0000 | 0.0000 | 0.0000 | — |
| exact_match | external_hard_negatives | — | 0.0000 | 0.0000 | 0.0000 | — |
| keyword_rules | bipia_clean_context | — | 0.0000 | 0.0000 | 0.0000 | — |
| keyword_rules | bipia_context | 0.0000 | — | — | 0.0000 | — |
| keyword_rules | bipia_payload | 0.0000 | — | — | 0.0000 | — |
| keyword_rules | do_not_answer | — | 0.0000 | 0.0000 | 0.0000 | — |
| keyword_rules | harmbench | — | 0.0000 | 0.0000 | 0.0000 | — |
| keyword_rules | jailbreaks_over_time | 0.2299 | 0.0054 | 53.8610 | 0.9015 | 0.3432 |
| keyword_rules | multi_turn | 0.0000 | — | — | 0.0000 | — |
| keyword_rules | nemotron_agentic_ipi | 0.0000 | — | — | 0.0000 | — |
| keyword_rules | notinject | — | 0.0206 | 206.4897 | 0.0000 | — |
| keyword_rules | oasst1_chat | — | 0.0000 | 0.0000 | 0.0000 | — |
| keyword_rules | oasst1_position_stress | — | 0.0000 | 0.0000 | 0.0000 | — |
| keyword_rules | prompt_injections | 0.0167 | 0.0000 | 0.0000 | 1.0000 | 0.5253 |
| keyword_rules | tensor_trust_attack | 0.0804 | — | — | 1.0000 | — |
| keyword_rules | tensor_trust_context | 0.1144 | — | — | 1.0000 | — |
| keyword_rules | toxic_chat | 0.2192 | 0.0006 | 6.4795 | 0.8421 | 0.1967 |
| keyword_rules | xstest | — | 0.0000 | 0.0000 | 0.0000 | — |
| keyword_rules | harmful_nonattack | — | 0.0000 | 0.0000 | 0.0000 | — |
| keyword_rules | external_hard_negatives | — | 0.0017 | 16.6350 | 0.0000 | — |
| char_ngram_logreg | bipia_clean_context | — | 0.0000 | 0.0000 | 0.0000 | — |
| char_ngram_logreg | bipia_context | 0.0000 | — | — | 0.0000 | — |
| char_ngram_logreg | bipia_payload | 0.0000 | — | — | 0.0000 | — |
| char_ngram_logreg | do_not_answer | — | 0.0000 | 0.0000 | 0.0000 | — |
| char_ngram_logreg | harmbench | — | 0.0000 | 0.0000 | 0.0000 | — |
| char_ngram_logreg | jailbreaks_over_time | 0.8211 | 0.0045 | 44.5177 | 0.9753 | 0.9468 |
| char_ngram_logreg | multi_turn | 0.2195 | — | — | 1.0000 | — |
| char_ngram_logreg | nemotron_agentic_ipi | 0.0000 | — | — | 0.0000 | — |
| char_ngram_logreg | notinject | — | 0.0000 | 0.0000 | 0.0000 | — |
| char_ngram_logreg | oasst1_chat | — | 0.0000 | 0.0000 | 0.0000 | — |
| char_ngram_logreg | oasst1_position_stress | — | 0.0000 | 0.0000 | 0.0000 | — |
| char_ngram_logreg | prompt_injections | 0.2000 | 0.0000 | 0.0000 | 1.0000 | 0.9189 |
| char_ngram_logreg | tensor_trust_attack | 0.2885 | — | — | 1.0000 | — |
| char_ngram_logreg | tensor_trust_context | 0.5609 | — | — | 1.0000 | — |
| char_ngram_logreg | toxic_chat | 0.6027 | 0.0039 | 38.8769 | 0.7097 | 0.7261 |
| char_ngram_logreg | xstest | — | 0.0000 | 0.0000 | 0.0000 | — |
| char_ngram_logreg | harmful_nonattack | — | 0.0000 | 0.0000 | 0.0000 | — |
| char_ngram_logreg | external_hard_negatives | — | 0.0000 | 0.0000 | 0.0000 | — |
| indirect_char_ngram_logreg | bipia_clean_context | — | 0.0120 | 119.7605 | 0.0000 | — |
| indirect_char_ngram_logreg | bipia_payload | 0.6720 | — | — | 1.0000 | — |
| indirect_char_ngram_logreg | bipia_context | 0.6720 | — | — | 1.0000 | — |
| indirect_char_ngram_logreg | tensor_trust_context | 0.2630 | — | — | 1.0000 | — |

## Direct-chat precision profiles

The 85% floor is the practical high-precision knee. Compared with the 80% floor, it gives up 10 validation attacks while removing 7 false signals. Tightening from 85% to 90% removes only 2 more false signals while losing 11 more attacks. All profiles remain advisory and were selected without official test results.

Observed precision reflects the validation source mixture (66 attacks among 7,186 rows), not product traffic. Expected-precision cells are prevalence scenarios calculated from validation recall and FPR. Each cell is `point / FPR-upper stress estimate`; neither value is production calibration, and the stress estimate is not a full confidence interval.

| Role | Minimum validation precision | Threshold | TP / attacks | FP / non-attacks | Recall | Observed precision | Observed FPR | Expected precision @ 0.1% attacks | @ 1% | @ 5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| precision diagnostic | 80% | 0.737031 | 44 / 66 | 11 / 7120 | 0.6667 | 0.8000 | 0.0015 | 0.3017 / 0.1944 | 0.8134 / 0.7089 | 0.9578 / 0.9270 |
| recommended shadow review | 85% | 0.898587 | 34 / 66 | 4 / 7120 | 0.5152 | 0.8947 | 0.0006 | 0.4786 / 0.2632 | 0.9026 / 0.7828 | 0.9797 / 0.9494 |
| precision diagnostic | 90% | 0.932541 | 23 / 66 | 2 / 7120 | 0.3485 | 0.9200 | 0.0003 | 0.5539 / 0.2541 | 0.9261 / 0.7747 | 0.9849 / 0.9471 |
| precision diagnostic | 95% | 0.967666 | 16 / 66 | 0 / 7120 | 0.2424 | 1.0000 | 0.0000 | 1.0000 / 0.3103 | 1.0000 / 0.8195 | 1.0000 / 0.9594 |

Frozen-suite transfer at each precision profile is shown below. It did not choose the recommendation.

| Minimum validation precision | External FPR | ToxicChat recall / precision / FPR | deepset recall / precision / FPR | Obfuscated recall | Temporal-source recall / precision / FPR |
|---:|---:|---:|---:|---:|---:|
| 80% | 0.0005 | 0.6712 / 0.5385 / 0.0091 | 0.3500 / 1.0000 / 0.0000 | 0.7848 | 0.8870 / 0.9351 / 0.0132 |
| 85% | 0.0000 | 0.6027 / 0.7097 / 0.0039 | 0.2000 / 1.0000 / 0.0000 | 0.2195 | 0.8211 / 0.9753 / 0.0045 |
| 90% | 0.0000 | 0.5616 / 0.8039 / 0.0022 | 0.1333 / 1.0000 / 0.0000 | 0.0242 | 0.7757 / 0.9831 / 0.0029 |
| 95% | 0.0000 | 0.4247 / 1.0000 / 0.0000 | 0.0667 / 1.0000 / 0.0000 | 0.0000 | 0.6988 / 0.9880 / 0.0018 |

## FPR-budget diagnostics

There is no universal target FPR. These validation-selected rows preserve the 0.1%, 0.5%, 1%, 2%, and 5% diagnostics for comparison, but none is the default review profile or calibrated for blocking or production traffic.

| Role | Validation FPR budget | Threshold | TP / attacks | FP / non-attacks | Recall | Precision | Observed FPR | False signals / 10k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stringent FPR diagnostic | 0.100% | 0.871110 | 35 / 66 | 7 / 7120 | 0.5303 | 0.8333 | 0.0010 | 9.8315 |
| FPR diagnostic | 0.500% | 0.534376 | 49 / 66 | 35 / 7120 | 0.7424 | 0.5833 | 0.0049 | 49.1573 |
| FPR diagnostic | 1.000% | 0.391502 | 54 / 66 | 71 / 7120 | 0.8182 | 0.4320 | 0.0100 | 99.7191 |
| FPR diagnostic | 2.000% | 0.278024 | 61 / 66 | 142 / 7120 | 0.9242 | 0.3005 | 0.0199 | 199.4382 |
| FPR diagnostic | 5.000% | 0.164020 | 64 / 66 | 356 / 7120 | 0.9697 | 0.1524 | 0.0500 | 500.0000 |

| Validation FPR budget | External FPR | ToxicChat recall / precision / FPR | deepset recall / precision / FPR | Obfuscated recall | Temporal-source recall / precision / FPR |
|---:|---:|---:|---:|---:|---:|
| 0.100% | 0.0000 | 0.6301 / 0.7188 / 0.0039 | 0.2333 / 1.0000 / 0.0000 | 0.4676 | 0.8488 / 0.9733 / 0.0050 |
| 0.500% | 0.0040 | 0.8356 / 0.3836 / 0.0212 | 0.5667 / 1.0000 / 0.0000 | 0.9799 | 0.9382 / 0.8391 / 0.0386 |
| 1.000% | 0.0152 | 0.8767 / 0.2581 / 0.0397 | 0.6167 / 1.0000 / 0.0000 | 0.9995 | 0.9605 / 0.5393 / 0.1759 |
| 2.000% | 0.0288 | 0.9178 / 0.1759 / 0.0678 | 0.6667 / 0.9302 / 0.0536 | 1.0000 | 0.9767 / 0.4897 / 0.2182 |
| 5.000% | 0.0725 | 0.9452 / 0.1066 / 0.1248 | 0.8000 / 0.8136 / 0.1964 | 1.0000 | 0.9962 / 0.4168 / 0.2988 |

## Runtime

Latency uses three warm-process passes over at most 2,048 rows sampled deterministically across the evaluation order.

| Detector | Mean batch latency per sample (µs) |
|---|---:|
| no_guard | 0.0 |
| exact_match | 0.3 |
| keyword_rules | 55.8 |
| char_ngram_logreg | 989.4 |
| indirect_char_ngram_logreg | 1944.1 |

## Interpretation

- Across external direct-user hard negatives, FPR is 0.0000 (0/4208), or 0.0000 signals per 10k prompts. The Wilson 95% upper bound is 0.0009.
- Multilingual OASST1 human-chat FPR is 0.0000 (0/1582); held-out harmful but non-injection FPR is 0.0000 (0/1337).
- Two-prompt OASST1 position-stress FPR is 0.0000 (0/500).
- XSTest hard-negative FPR: 0.0000 (0/450); Wilson 95% upper bound 0.0085.
- XSTest safe/unsafe hard-negative FPR: 0.0000 (0/250) / 0.0000 (0/200).
- NotInject trigger-word hard-negative FPR: 0.0000 (0/339); Wilson 95% upper bound 0.0112.
- Out-of-source obfuscated-jailbreak recall: 0.2195 (908/4136; cluster-weighted 0.2195).
- JailbreaksOverTime source-shift recall/FPR is 0.8211/0.0045 (3203/3901 attacks; 81/18195 source-labeled negatives). Source and time are confounded, so this is not a clean temporal causal estimate.
- Human-authored Tensor Trust attack-only recall is 0.2885 (262/908); the provenance-scoped sensor gets 0.2630 (354/1346) on the same attacks embedded between benchmark defenses. Running the direct-override fallback as designed raises combined recall to 0.6248 (841/1346). This source is evaluation only and has no explicit standard dataset license.
- The direct-chat model gets 0.0000 BIPIA payload recall and 0.0000 poisoned-context recall. Lowering its threshold does not solve this without conflating ordinary questions with attacks.
- The provenance-scoped indirect model gets 0.6720 payload recall and 0.6720 poisoned-context recall (cluster-weighted 0.6720), with clean-context FPR 0.0120 (2/167).
- Every classifier result is advisory. Trusted input provenance selects the sensor set; untrusted content also runs the direct-override fallback. The CLI always returns allow, while the reference monitor independently constrains actions and data flow.
- Same-source scores measure a cheap known-distribution sensor. The obfuscated and indirect holdouts test transfer but remain benchmark attack families.

## Obfuscated holdout by variant

| Variant | Rows | Recall |
|---|---:|---:|
| word_mapping_perp_filter | 517 | 0.0000 |
| word_mapping_perp_filter/Base64 | 517 | 0.2669 |
| word_mapping_perp_filter/Caesar | 517 | 0.3211 |
| word_mapping_perp_filter/Leetspeak | 517 | 0.3985 |
| word_mapping_random | 517 | 0.0000 |
| word_mapping_random/Base64 | 517 | 0.1934 |
| word_mapping_random/Caesar | 517 | 0.2456 |
| word_mapping_random/Leetspeak | 517 | 0.3308 |

The 0.2195 aggregate hides a wide range: 0 encoded variants reach 100% recall. The two plain word-mapping variants reach only 0.0000 and 0.0000, which is the actionable weakness.

## Label audit

A qualitative Codex-agent inspection of the highest-confidence errors found obvious jailbreak-style ToxicChat rows labeled non-jailbreak, harmful framing labeled jailbreak without an explicit override, context-dependent deepset positives that look benign as standalone text, and jailbreak/DAN-style WildChat rows labeled negative in JailbreaksOverTime. This was not independent human annotation. These metrics measure the source labels; they are not clean ground truth. See [label-audit.md](label-audit.md).

## Limits

- ToxicChat contributes only 204 jailbreak positives across its official splits.
- OASST1 has no injection label; accepted human prompts are weak negatives and may contain attacks.
- Held-out data estimates benchmark FPR, not the production traffic base rate or user impact.
- Source, language, family, and time are confounded; this is a baseline, not a robustness claim.
- ToxicChat and deepset expose no template lineage here, so their groups are row-level after exact deduplication.
- The cipher holdout scores only its single Prompt field; it is not a multi-turn session evaluation.
- BIPIA context rows are a deterministic three-position slice, not its full task/target-model evaluation.
- The indirect sensor has only 167 held-out clean BIPIA contexts; its FPR confidence interval is wide.
- No adaptive attacker, guard transformer, LLM judge, or live target model is included in headline metrics.
- ToxicChat and Do-Not-Answer have non-commercial licenses, so this model is research-only.
- Classifier misses must be contained by deterministic action and egress policy.

Source details and pinned revisions are in the generated data manifest.
