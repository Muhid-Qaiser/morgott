# Routing word n-gram baseline

This is a cheap research control, not a blocking model or a production estimate.
It uses source-supported direct-user rows, unweighted training, and the untouched 0.5 cutoff.

| Split | Recall | FPR | Macro-source recall | Macro-source FPR | Precision | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.9317 | 0.0172 | 0.8636 | 0.0421 | 0.9862 | 0.9936 | 0.9903 |
| dev_test | 0.9246 | 0.0318 | 0.8002 | 0.1524 | 0.9649 | 0.9854 | 0.9827 |

Confusion counts use the same untouched 0.5 cutoff.

| Split | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| validation | 69576 | 974 | 55758 | 5099 |
| dev_test | 119699 | 4352 | 132428 | 9765 |

The following values substitute the measured development recall and FPR into fixed attack-prevalence scenarios; they are not production estimates.

| Split | Expected precision at 0.1% | At 1% | At 5% |
|---|---:|---:|---:|
| validation | 0.0515 | 0.3541 | 0.7407 |
| dev_test | 0.0283 | 0.2269 | 0.6046 |

Selected training support by normalized character length:

| Normalized characters | Benign | Positive |
|---:|---:|---:|
| 0-256 | 92844 | 34977 |
| 257-1024 | 64 | 17196 |
| 1025-4096 | 0 | 8927 |
| 4097+ | 0 | 41 |

Length slices use normalized Unicode character counts and expose their class denominators.

| Split | Normalized characters | Positive | Benign | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| validation | 0-256 | 60442 | 56665 | 0.9204 | 0.0172 |
| validation | 257-1024 | 11361 | 67 | 0.9805 | 0.0000 |
| validation | 1025-4096 | 2851 | 0 | 0.9776 | n/a |
| validation | 4097+ | 21 | 0 | 0.8095 | n/a |
| dev_test | 0-256 | 89113 | 136233 | 0.9000 | 0.0296 |
| dev_test | 257-1024 | 27599 | 283 | 0.9718 | 0.4700 |
| dev_test | 1025-4096 | 12205 | 237 | 0.9938 | 0.7089 |
| dev_test | 4097+ | 547 | 27 | 0.9982 | 0.6296 |

Per-source metrics count exact-merged rows in every origin source membership; aggregate metrics count each row once.
Exact recipe metadata is in `routing-baseline.json`.
