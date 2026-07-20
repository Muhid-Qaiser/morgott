# PromptShield evaluation-only audit

Generated: 2026-07-20T06:59:10+00:00

**Decision: keep PromptShield evaluation-only and never train on it.** The files lack row-level source/group lineage, while the paper says the corpus aggregates public sources and attack strategies that can overlap this project's active corpora. The metrics below are exploratory development evidence, not a final test or production false-positive claim.

## Pinned source

The requested `NVIDIA/PromptShield` name resolves publicly as [hendzh/PromptShield](https://huggingface.co/datasets/hendzh/PromptShield) at `a5234cb1f5cdb256600cab64b8c961195b5e8404`. The card declares Apache-2.0; component-source license compatibility was not independently audited. Paper: [PromptShield: Deployable Detection for Prompt Injection Attacks](https://arxiv.org/abs/2501.15145).

| File | Bytes | SHA-256 | Rows | Positive | Negative |
|---|---:|---|---:|---:|---:|
| `train.json` | 12,239,651 | `aa33c3ffcc27bd07c0a233b52f1b8c3cbdb30606ce2412da06a88b5290cdc7b6` | 18,909 | 9,452 | 9,457 |
| `validation.json` | 645,951 | `1d93d90d57d3ef44ed0c546fbc04d66324436c5fcd32e7fcb940ceed270fbe77` | 1,000 | 503 | 497 |
| `test.json` | 18,288,615 | `526207c2485829d9961407011d7f4cd929569e7f285dc8396b3f385e0608bc70` | 23,516 | 6,486 | 17,030 |

The pinned dataset card is 1,792 bytes with SHA-256 `d5f36dce4f27d40ae8fda54335d382c74e650485cb4a92c6837602ef84a1a662`. Actual data rows contain exactly `prompt` and `label`; only `prompt` was scored. Label 1 means a source-labelled injection attempt and label 0 means source-labelled benign/no injection—not harmlessness. All data is described as English.

The paper reports benign inputs from UltraChat, LMSYS Chatbot Arena, Alpaca, databricks-dolly, IFEval, Natural Instructions, and Synthetic Python Problems; attacks come from FourAttacks, HackAPrompt, and OpenPromptInject. The released rows do not say which source, conversation, task, template, or mutation produced each prompt. This audit does not guess.

## Leakage and duplicate audit

Raw exact means byte-identical UTF-8. Normalized exact uses this repository's NFKC, casefold, and whitespace view. Near matches use 128-bit SimHash over normalized word unigrams/bigrams, eight 16-bit bands, Hamming distance <= 6, and at least five words. Near matching excludes identical normalized hashes and is a strict, non-exhaustive signal.

### Against every active processed fit/evaluation file

| PromptShield split | Rows | Raw exact | Normalized exact | Near non-exact | Near with no exact anywhere | Any fit overlap |
|---|---:|---:|---:|---:|---:|---:|
| train | 18,909 | 0 | 2 | 3 | 3 | 2 |
| validation | 1,000 | 0 | 0 | 0 | 0 | 0 |
| test | 23,516 | 0 | 116 | 32 | 15 | 97 |

Reference: 75,980 rows across 18 processed files, with 75,973 unique normalized texts. Fit files are `indirect_train, train`; all remaining files are evaluation corpora.

| Active file | Rows | Raw exact source rows | Normalized exact source rows | Near source rows |
|---|---:|---:|---:|---:|
| indirect_train | 1,212 | 0 | 0 | 1 |
| jailbreaks_over_time | 22,096 | 0 | 10 | 5 |
| oasst1_chat | 1,582 | 0 | 7 | 0 |
| toxic_chat | 4,703 | 0 | 19 | 12 |
| train | 35,912 | 0 | 88 | 21 |

### Within PromptShield

| Split | Rows | Raw duplicate rows | Normalized duplicate rows | Conflicting normalized texts |
|---|---:|---:|---:|---:|
| train | 18,909 | 0 | 123 | 0 |
| validation | 1,000 | 0 | 2 | 0 |
| test | 23,516 | 157 | 171 | 0 |

| Split pair | Raw exact unique | Normalized exact unique | Label-conflict exact | Near non-exact pairs | Left/right near rows |
|---|---:|---:|---:|---:|---:|
| train / validation | 0 | 14 | 0 | 1,270 | 526 / 98 |
| train / test | 0 | 0 | 0 | 1 | 1 / 1 |
| validation / test | 0 | 0 | 0 | 0 | 0 / 0 |

## Length and truncation risk

The locked character model scores complete strings and performs no input-length truncation. The following is only a whitespace-token proxy for neural tokenizers; it is not tokenizer-exact.

| Split/label | Rows | Token p50 | p90 | p95 | p99 | Max | >256 | >512 | >1024 | >2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train/all | 18,909 | 51 | 210 | 388 | 704 | 4,778 | 1,415 | 723 | 28 | 1 |
| train/benign_source_label | 9,457 | 20 | 167 | 384 | 771 | 1,761 | 700 | 282 | 13 | 0 |
| train/injection_source_label | 9,452 | 68 | 222 | 392 | 662 | 4,778 | 715 | 441 | 15 | 1 |
| validation/all | 1,000 | 50 | 216 | 383 | 729 | 1,676 | 71 | 37 | 2 | 0 |
| validation/benign_source_label | 497 | 19 | 154 | 382 | 760 | 1,013 | 34 | 15 | 0 | 0 |
| validation/injection_source_label | 503 | 69 | 232 | 410 | 640 | 1,676 | 37 | 22 | 2 | 0 |
| test/all | 23,516 | 67 | 275 | 410 | 794 | 3,803 | 2,666 | 741 | 129 | 20 |
| test/benign_source_label | 17,030 | 42 | 206 | 286 | 615 | 3,803 | 1,142 | 220 | 28 | 4 |
| test/injection_source_label | 6,486 | 112 | 454 | 627 | 1,179 | 3,306 | 1,524 | 521 | 101 | 16 |

## Existing locked character control on PromptShield test

Thresholds come unchanged from `reports/baseline.json`; PromptShield was not used to select or recalibrate them. Counts use PromptShield's public source labels.

| Validation precision floor | Threshold | TP | FN | FP | TN | Recall | FPR | Precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.737031 | 595 | 5,891 | 454 | 16,576 | 0.0917 | 0.0267 | 0.5672 |
| 0.85 | 0.898587 | 194 | 6,292 | 240 | 16,790 | 0.0299 | 0.0141 | 0.4470 |
| 0.90 | 0.932541 | 145 | 6,341 | 110 | 16,920 | 0.0224 | 0.0065 | 0.5686 |
| 0.95 | 0.967666 | 86 | 6,400 | 11 | 17,019 | 0.0133 | 0.0006 | 0.8866 |

A conservative companion view removes every test row with a raw, normalized, or strict near match to either active fit file (`train` or `indirect_train`). This exclusion is based only on hashes/fingerprints, not source inference.

| Validation precision floor | Rows | Positive | Negative | TP | FN | FP | TN | Recall | FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 23,419 | 6,486 | 16,933 | 595 | 5,891 | 451 | 16,482 | 0.0917 | 0.0266 |
| 0.85 | 23,419 | 6,486 | 16,933 | 194 | 6,292 | 238 | 16,695 | 0.0299 | 0.0141 |
| 0.90 | 23,419 | 6,486 | 16,933 | 145 | 6,341 | 108 | 16,825 | 0.0224 | 0.0064 |
| 0.95 | 23,419 | 6,486 | 16,933 | 86 | 6,400 | 10 | 16,923 | 0.0133 | 0.0006 |

The retained 0.85 profile catches only 194/6,486 source-positive tests (2.99%) while alerting on 240/17,030 source negatives (1.41%). Even the 0.95 profile catches only 86/6,486 (1.33%), with 11/17,030 source-negative alerts (0.065%; Wilson 95% upper 0.116%). Removing the 97 fit-overlap test rows barely changes this. The result is a transfer failure for the current control, not a reason to train on an untraceable aggregate or a production lockout estimate.

### Full test metrics by whitespace-token length

Precision floor 0.80, threshold 0.737031

| Bucket | Rows | TP | FN | FP | TN | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-64 | 11,395 | 41 | 335 | 250 | 10,769 | 0.1090 | 0.0227 |
| 65-128 | 6,396 | 451 | 3,099 | 115 | 2,731 | 0.1270 | 0.0404 |
| 129-256 | 3,059 | 84 | 952 | 41 | 1,982 | 0.0811 | 0.0203 |
| 257-512 | 1,925 | 9 | 994 | 43 | 879 | 0.0090 | 0.0466 |
| 513+ | 741 | 10 | 511 | 5 | 215 | 0.0192 | 0.0227 |

Precision floor 0.85, threshold 0.898587

| Bucket | Rows | TP | FN | FP | TN | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-64 | 11,395 | 24 | 352 | 155 | 10,864 | 0.0638 | 0.0141 |
| 65-128 | 6,396 | 165 | 3,385 | 71 | 2,775 | 0.0465 | 0.0249 |
| 129-256 | 3,059 | 5 | 1,031 | 7 | 2,016 | 0.0048 | 0.0035 |
| 257-512 | 1,925 | 0 | 1,003 | 7 | 915 | 0.0000 | 0.0076 |
| 513+ | 741 | 0 | 521 | 0 | 220 | 0.0000 | 0.0000 |

Precision floor 0.90, threshold 0.932541

| Bucket | Rows | TP | FN | FP | TN | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-64 | 11,395 | 24 | 352 | 88 | 10,931 | 0.0638 | 0.0080 |
| 65-128 | 6,396 | 121 | 3,429 | 9 | 2,837 | 0.0341 | 0.0032 |
| 129-256 | 3,059 | 0 | 1,036 | 7 | 2,016 | 0.0000 | 0.0035 |
| 257-512 | 1,925 | 0 | 1,003 | 6 | 916 | 0.0000 | 0.0065 |
| 513+ | 741 | 0 | 521 | 0 | 220 | 0.0000 | 0.0000 |

Precision floor 0.95, threshold 0.967666

| Bucket | Rows | TP | FN | FP | TN | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-64 | 11,395 | 22 | 354 | 1 | 11,018 | 0.0585 | 0.0001 |
| 65-128 | 6,396 | 64 | 3,486 | 1 | 2,845 | 0.0180 | 0.0004 |
| 129-256 | 3,059 | 0 | 1,036 | 3 | 2,020 | 0.0000 | 0.0015 |
| 257-512 | 1,925 | 0 | 1,003 | 6 | 916 | 0.0000 | 0.0065 |
| 513+ | 741 | 0 | 521 | 0 | 220 | 0.0000 | 0.0000 |

No neural result is included: no already-generated PromptShield neural scores existed, and this audit downloaded no weights and used no GPU.

## Interpretation limits

- PromptShield provides no row-level source or grouping lineage, so source-held-out and group-held-out metrics cannot be reconstructed.
- The paper reports aggregation from public chat, instruction, and attack datasets; exact and fuzzy overlap can therefore inflate apparent transfer.
- Only the prompt field is evaluated. No source is inferred from prompt content.
- Source label 0 means no prompt injection, not independently established harmlessness; this audit does not treat it as training data.
- Near-overlap is a strict deterministic heuristic and can miss paraphrases or wrapper changes.
- PromptShield was observed before this experiment, so its metrics are exploratory development evidence, not an untouched final test or production FPR claim.
- No human adjudication is assumed; labels remain public source labels.
