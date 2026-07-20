# GPU model experiments

These runners score frozen processed JSONL files only after their SHA-256
digests match. They do not change the POC model. Direct-user results include
validation-selected 80%, 85%, 90%, and 95% precision profiles plus 0.1%, 0.5%,
1%, 2%, and 5% validation-FPR diagnostics on frozen lineage groups. The primary
advisory comparison uses the 85% profile; no point is production calibration.
The untrusted-content threshold targets zero observed validation FPs. See
[`reports/model-experiments.md`](../../reports/model-experiments.md) for the
common comparison.

Tensor Trust is locked and scored by all three GPU runners:

- `tensor_trust_attack`: 908 direct-user positives, SHA-256
  `e648fb3aef4fa3d583ae1364df3e7396037b9b1fc3d66c2551fc50cc170901c0`.
- `tensor_trust_context`: 1,346 untrusted-content positives, SHA-256
  `4c4d98adeba64312e4c2643d5727963deb22217b2e9852a0c24d8338d855326a`.

## Historical 0.1% diagnostic results

| Direct-user evaluation | Character POC | E5 linear probe | PIGuard | ProtectAI |
|---|---:|---:|---:|---:|
| Validation recall / FPR | 53.03% / 0.098% | 36.36% / 0.098% | 51.52% / 0.098% | 0% / 0.014% |
| ToxicChat recall / FPR | 63.01% / 0.389% | 52.05% / 0.410% | 43.84% / 0.302% | 0% / 0% |
| deepset recall / FPR | 23.33% / 0% | 26.67% / 0% | 23.33% / 0% | 0% / 0% |
| Multi-turn recall | 46.76% | 0.22% | 0% | 0% |
| External hard-negative false positives | 0/4,208 | 4/4,208 (0.095%) | 29/4,208 (0.689%) | 0/4,208 |
| NotInject false positives | 0/339 | 2/339 (0.590%) | 15/339 (4.425%) | 0/339 |
| JailbreaksOverTime recall / source-labelled FPR | 84.88% / 0.500% | 57.93% / 0.456% | 50.17% / 0.445% | 0% / 0% |
| OASST1 long-position false positives | 0/500 | 0/500 | 5/500 (1.00%) | 0/500 |
| Tensor Trust attack recall | 304/908 (33.48%) | 103/908 (11.34%) | 450/908 (49.56%) | 19/908 (2.09%) |

| Untrusted-content evaluation | Character POC | E5 linear probe | PIGuard | ProtectAI |
|---|---:|---:|---:|---:|
| BIPIA payload recall | 67.20% | 78.40% | 57.60% | 2.40% |
| BIPIA poisoned-context recall | 67.20% | 79.20% | 91.73% | 8.00% |
| BIPIA clean-context FPR | 2/167 (1.20%) | 10/167 (5.99%) | 4/167 (2.40%) | 7/167 (4.19%) |
| Tensor Trust context recall | 354/1,346 (26.30%); 903/1,346 (67.09%) with direct fallback | 991/1,346 (73.63%) | 1,315/1,346 (97.70%) | 1,346/1,346 (100%) |

No GPU candidate is promoted. At the 85% validation-precision profile, E5 and
PIGuard retain only 18/66 and 19/66 validation attacks respectively and neither
catches a multi-turn example; ProtectAI cannot attain even 80% validation
precision at an observed threshold. At the historical 0.1% diagnostic, E5 misses
nearly all multi-turn attacks, detects
only 11.3% of direct Tensor Trust attacks, and raises clean-context FPR. PIGuard's
strong BIPIA context result is not independent: BIPIA was used in its reported
development/evaluation, while its locked threshold produces 4.4% false positives
on NotInject, 1.0% on the long-position stress set, and 0.69% across all 4,208
hard negatives. ProtectAI's score distribution saturates below the threshold
needed to meet the 0.1% diagnostic, yielding zero historical-point recall
on the direct benchmark sets other than 2.1% on Tensor Trust.

All three models score Tensor Trust's combined contexts far higher than the exact
same attacks in isolation. Those contexts contain defensive system instructions,
and the clean BIPIA context FPR is 2.4%--6.0%, so the context numbers are evidence
of context sensitivity, not reliable evidence of better attack understanding.
The character POC context cell reports its provenance-scoped indirect sensor
first, then the advisory union with its direct-override fallback; it is not a
single-model recall figure.

For context, the checkpoints' default 0.5 threshold trades substantially more
false positives for recall:

| Default-threshold evaluation | PIGuard | ProtectAI |
|---|---:|---:|
| ToxicChat recall / FPR | 86.30% / 5.16% | 69.86% / 2.77% |
| deepset recall | 66.67% | 36.67% |
| Multi-turn recall | 79.91% | 23.45% |
| JailbreaksOverTime recall / source-labelled FPR | 96.33% / 7.69% | 89.34% / 5.30% |

## Models and performance

- E5: `intfloat/multilingual-e5-small` at
  `614241f622f53c4eeff9890bdc4f31cfecc418b3` (MIT), frozen FP16 encoder,
  attention-mask mean pooling, L2 normalization, and a float32 balanced logistic
  head. It uses 384 tokens and a separate untrusted-content head.
- PIGuard: `leolee99/PIGuard` at
  `dd78b24e330193a22d2293ac66922dd4f982f563` (MIT). The runner contains an
  audited local equivalent of its 18-line DeBERTa subclass and never enables
  `trust_remote_code`.
- ProtectAI: `protectai/deberta-v3-base-prompt-injection-v2` at
  `90c9989b1a342275dd0d1a95aad283c04e075671` (Apache-2.0). Its disclosed
  training sources include XSTest-derived data, so XSTest is not independent.

| Model | Peak allocated / reserved | Batch 1 | Batched | Full measured run |
|---|---:|---:|---:|---:|
| E5 | 686.5 / 928 MiB | 4.80 ms/text | 1.69 ms/text (64) | 29.4 s cached rerun |
| PIGuard | 1,303.4 / 1,492 MiB | 13.77 ms/text | 9.04 ms/text (32) | 513.2 s cached rerun |
| ProtectAI | 1,303.3 / 1,492 MiB | 12.26 ms/text | 10.54 ms/text (32) | 514.1 s cached rerun |

Measurements used an NVIDIA RTX 4050 Laptop GPU. Exact metrics, confidence
bounds, versions, thresholds, timings, and input hashes are in
`embedding_results.json`, `piguard_cuda_results.json`, and
`protectai_cuda_results.json`.

## Reproduce

```bash
python experiments/gpu_baselines/run_embeddings.py --batch-size 128
python experiments/gpu_baselines/run_attention.py --model piguard --device cuda
python experiments/gpu_baselines/run_attention.py --model protectai --device cuda
```

The scripts let the Hugging Face client use process-environment credentials;
they never read or print `.env`.
