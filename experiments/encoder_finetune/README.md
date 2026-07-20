# End-to-end encoder fine-tuning pilot

This is the controlled architecture comparison missing from the frozen-feature
and off-the-shelf-checkpoint experiments. It fine-tunes
`answerdotai/ModernBERT-base` and `microsoft/deberta-v3-base` end to end on the
same direct-user rows. It is a one-epoch, one-seed, negative-subsampled pilot,
not a final model ranking.

## Fixed protocol

- One shared 768-to-2 linear head with masked-mean pooling.
- All 245 fit positives, all 4,160 ToxicChat/deepset fit negatives, and 4,003
  OASST1 rows from 1,286 complete fit groups selected by a seeded SHA-256 rank:
  8,408 rows total.
  Ordered row-ID digest:
  `36ef7c54a385790a4fb946f9aaaa3a876b56000bfcec2f6513853e6628969408`.
- The full 7,186-row, 3,036-group validation partition remains untouched.
- Seed 42, one epoch, 512 tokens, AdamW at 2e-5 with 0.01 weight decay,
  class-weighted cross entropy, effective batch 32, dynamic padding, gradient
  checkpointing, and gradient clipping at 1.0.
- CUDA FP16 autocast with FP32 parameters, classifier, loss, and optimizer.
  A worst-case forward/backward/optimizer memory preflight may lower only the
  physical batch; the effective batch and example order remain fixed.
- ModernBERT uses PyTorch SDPA. Transformers reports that DeBERTa-v2 does not
  support its SDPA integration, so DeBERTa uses eager attention. This backend
  difference is disclosed rather than hidden as an architecture result.
- Thresholds use grouped validation only: minimum observed precision floors of
  80%, 85%, 90%, and 95%, plus diagnostic FPR budgets of 0.1%, 0.5%, 1%, 2%,
  and 5%. Every output remains shadow-only.

The input JSONL files and model weights are SHA-256 checked. Both models load
with `trust_remote_code=False` and `use_safetensors=True`. ModernBERT is pinned
to commit `8949b909ec900327062f0ebf497f51aef5e6f0c8`. DeBERTa's main revision
does not publish safetensors, so this experiment pins conversion commit
`de19fe7db5162df5f3d8f0b41321c0267288fd74`. Its config remains a 12-layer,
768-hidden DeBERTa-v2 implementation with a 512-token limit; loading must report
no missing or mismatched encoder weights. The only allowed unused tensors are
the masked-language-model pretraining head.

## Run

Run serially on an idle 6 GB CUDA device:

```bash
PYTHONPATH=src python experiments/encoder_finetune/run.py --self-test
PYTHONPATH=src python experiments/encoder_finetune/run.py --describe-subset
PYTHONPATH=src python experiments/encoder_finetune/run.py --model modernbert
PYTHONPATH=src python experiments/encoder_finetune/run.py --model deberta
python experiments/encoder_finetune/compare.py
python -m unittest discover -s experiments/encoder_finetune -p 'test_*.py' -v
```

No checkpoint is saved. Hugging Face caches and any future checkpoints are
ignored; result JSON contains only aggregate metrics and provenance.

## Results

Both models fit at physical batch 2 and effective batch 32. ModernBERT trained
in 330.2 seconds with 2,955 MiB peak allocated VRAM; DeBERTa took 364.8 seconds
and 3,547 MiB. Measured batch-1 GPU latency was 9.1/14.2 ms p50/p95 for
ModernBERT and 10.8/13.9 ms for DeBERTa. These timings are local pilot
measurements, not deployment benchmarks.

At the shared 85%-minimum-validation-precision profile:

| model | validation TP / 66 | validation FP / 7,120 | precision | recall | threshold |
|---|---:|---:|---:|---:|---:|
| character control | 34 | 4 | 89.5% | 51.5% | 0.8986 |
| ModernBERT | 6 | 1 | 85.7% | 9.1% | 0.7730 |
| DeBERTa-v3 | 36 | 6 | 85.7% | 54.5% | 0.9941 |

ModernBERT underfit this screen: it caught no deepset, multi-turn, or BIPIA
positives at that profile. DeBERTa was competitive on the validation ranking,
but the small gain over the character control came with 4/4,208 external hard
false positives (all four are NotInject), only 5/4,136 multi-turn detections
versus the control's 908, and a threshold close to score saturation. Its 85%
profile caught 56/73 ToxicChat attacks with 21/4,630 false positives and
3,413/3,901 JailbreaksOverTime source positives with 93/18,195 source-negative
alerts.

The DeBERTa 95% profile is worth preserving as an observation: 25/66 validation
positives with 1/7,120 false positive (96.2% observed precision), 0/4,208 hard
false positives, and 48/73 ToxicChat positives with 5/4,630 false positives.
Its threshold is 0.9990, so calibration and stability need scrutiny before this
can be operationally meaningful.

No backbone is promoted. This is one seed, one epoch, and a deliberately
negative-subsampled fit, so it does not establish that DeBERTa is intrinsically
better than ModernBERT. DeBERTa is the only result interesting enough to justify
a predeclared fuller controlled rerun; ModernBERT's result is evidence of
underfitting in this protocol, not a general rejection. BIPIA and Tensor-context
results remain direct-sensor stress checks, not an indirect-head comparison.

Full precision profiles, FPR diagnostics, frozen-suite counts, hashes, and
provenance are in `modernbert_pilot.json`, `deberta_pilot.json`, and the compact
`comparison.json`. OOD-energy penalties, focal loss, tokenizer perturbations,
layer mixing, LoRA, and ensembles remain follow-up hypotheses only; mixing them
into this first backbone screen would destroy the controlled comparison.
