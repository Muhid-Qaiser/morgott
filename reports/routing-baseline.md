# Routing word n-gram baseline

This is a cheap research control, not a blocking model or a production estimate.
It uses source-supported direct-user rows, unweighted training, and the untouched 0.5 cutoff.

| Split | Recall | FPR | Precision | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| validation | 0.9326 | 0.0175 | 0.9859 | 0.9936 | 0.9905 |
| dev_test | 0.9235 | 0.0294 | 0.9676 | 0.9860 | 0.9834 |

Per-source metrics and exact recipe metadata are in `routing-baseline.json`.
