# Frozen ModernBERT probe

This is the cheap falsification test for ModernBERT as a feature extractor, not a
production sensor or an architecture novelty claim. It freezes
`answerdotai/ModernBERT-base`, extracts masked-mean and CLS representations from
the same forward pass, L2-normalizes and caches both 768-dimensional features,
then fits separate balanced logistic heads for direct-user and untrusted-content
channels. Masked mean is the predeclared primary result; CLS is a cheap pooling
ablation.

The checkpoint is pinned to `8949b909ec900327062f0ebf497f51aef5e6f0c8`
(Apache-2.0). Its repository contains standard tokenizer/config/weight assets and
no Python code or `auto_map`; loading forces safetensors and
`trust_remote_code=False`. CUDA uses FP16 and PyTorch SDPA. FlashAttention 2 is
intentionally not added as a dependency.

```bash
python experiments/modernbert/run_probe.py --max-length 512 --batch-size 16 --device cuda
python experiments/modernbert/run_probe.py --max-length 1024 --batch-size 8 --device cuda
pytest -q experiments/modernbert/test_probe.py
```

This probe selects thresholds only on the existing deterministic grouped
validation split. Direct results retain 80%, 85%, 90%, and 95% precision profiles
plus 0.1%, 0.5%, 1%, 2%, and 5% FPR diagnostics; the 85% profile is the primary
advisory comparison, not production calibration. Indirect input retains its
zero-observed-validation-FP diagnostic. Indirect documents use the maximum score
over the whole document and blank-line paragraphs, including threshold
calibration. JOT remains evaluation
only; its source, attack style, label process, and collection time are confounded.
Two SHA-locked Tensor Trust partitions are also evaluation-only: 908 standalone
human attacks go only through the direct-user head, while 1,346 poisoned task
contexts go only through the untrusted-content head and max-paragraph scoring.
Neither partition participates in fitting or threshold selection.

The result JSON embeds compact references to the locked character model and
frozen multilingual-E5 result. A weak probe rules out only linear separability
of frozen ModernBERT features; it does not rule out end-to-end fine-tuning.

## 512-token CUDA result

Run: RTX 4050 6 GB, FP16 encoder, FP32 cached features/head, batch 16. The cold
initial extraction took 685.0 s, dominated by JOT (352.6 s), with 148.3 s for
the training set. The cached Tensor Trust refresh took 115.9 s: 11.4 s for 908
direct attacks and 94.0 s for paragraph-expanded contexts. Its measured latency
was 7.62 ms/text at batch 1 and 5.33 ms/text at batch 16; PyTorch reported 502.6
MiB peak allocated and 536 MiB peak reserved.

The table below preserves the historical 0.1% diagnostic. All direct thresholds
and the newer precision/FPR profiles use the same grouped validation split.
“External FPR” is the 4,208-row deduplicated hard-negative
aggregate.

| Model | Validation recall | ToxicChat recall / FPR | deepset recall | Multi-turn recall | JOT recall / FPR | External FPR |
|---|---:|---:|---:|---:|---:|---:|
| Character TF-IDF | 53.03% | 63.01% / 0.389% | 23.33% | 46.76% | 84.88% / 0.500% | 0.000% |
| Frozen multilingual E5 | 36.36% | 52.05% / 0.410% | 26.67% | 0.22% | 57.93% / 0.456% | 0.095% |
| Frozen ModernBERT, masked mean | 16.67% | 35.62% / 0.454% | 6.67% | 0.00% | 52.63% / 1.039% | 0.095% |
| Frozen ModernBERT, CLS | 9.09% | 23.29% / 0.324% | 1.67% | 0.00% | 23.05% / 0.506% | 0.190% |

Masked mean also beat CLS on validation PR-AUC (0.501 vs 0.282). It produced
zero alerts on the 500 negative-only position-stress chats, which checks
over-defense but does not measure recall for attacks buried after token 512.
JOT is source/time confounded and is not a pure temporal-robustness measure.

For the indirect channel, masked mean recalled 0/125 standalone BIPIA payloads
and 72/375 poisoned contexts, with 1/167 clean-context false alerts. CLS recalled
25/125 and 74/375 but false-alerted on 7/167 clean contexts. The character
control recalled 84/125 and 252/375 with 2/167 false alerts.

Tensor Trust further rejects the frozen-feature hypothesis. These partitions
contain only positives, so the table reports recall rather than FPR.

| Model | Direct human attacks | Poisoned task contexts |
|---|---:|---:|
| Character TF-IDF | 304/908 (33.48%) | 354/1,346 (26.30%) |
| Frozen ModernBERT, masked mean | 87/908 (9.58%) | 42/1,346 (3.12%) |
| Frozen ModernBERT, CLS | 75/908 (8.26%) | 47/1,346 (3.49%) |

**Decision: do not promote either frozen ModernBERT pooling.** Keep the
character model as the control. At the primary 85% validation-precision profile,
masked mean retains 8/66 attacks with 1/7,120 false signals and CLS retains only
1/66 with none; both still catch 0/4,136 multi-turn attacks. A future ModernBERT
test should be end-to-end
fine-tuning (or a small adapter) with a positive long-position stress set; the
frozen result does not justify a full 1,024-token rerun by itself.
