# ModernBERT attention and context-length audit

This audit separates kernel feasibility from model-quality evidence.
No longer-context model was trained because the selected direct-user recipe could not evaluate long benign inputs.

## Data blocker

Every selected direct-user training row above 256 tokens has routing label 1.
The same defect appears in validation.

| Cohort | Above 256 | Above 512 | Above 1024 | Benign in any column |
|---|---:|---:|---:|---:|
| Selected train | 8,355 | 1,984 | 881 | 0 |
| Validation | 2,323 | 227 | 19 | 0 |

Increasing the token limit could improve positive recall by learning that long inputs require review.
There is no long-benign validation denominator that could reveal the resulting false positives.
Moving from 256 to 512 recovers 4.14% of selected training rows, while moving from 512 to 1024 recovers only another 0.72%.
All recovered rows are positive and mostly come from HackAPrompt and Tensor Trust.

No suitable matched long-benign source is retained for this experiment.
Add one only with same-format attacks, held-out lineage, and a separately declared model ablation.

## Runtime pins

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU with 6 GiB VRAM.
- PyTorch: `2.13.0+cu130`.
- Transformers: `5.14.1`.
- ModernBERT: `answerdotai/ModernBERT-base@8949b909ec900327062f0ebf497f51aef5e6f0c8`.
- Kernel loader: `kernels==0.15.2`.
- FlashAttention 2 kernel: `kernels-community/flash-attn2@239bb21bd566f598d7e2228eab9788b0a9239b2d`.
- Loaded kernel metadata version: 3.
- CUDA 13 x86-64 binary SHA256: `1433d3fe1187211c5ce622a7373d8c0487227384a2b8c74064e5ed6dfc820727`.

PyTorch profiling confirmed that the SDPA control dispatched to memory-efficient scaled-dot-product attention rather than PyTorch flash attention.
The Hugging Face kernel loader and ModernBERT attention options are documented in the [Transformers kernel-loading guide](https://huggingface.co/docs/transformers/main/kernel_doc/loading_kernels) and [ModernBERT documentation](https://huggingface.co/docs/transformers/model_doc/modernbert).
PyTorch documents explicit SDPA backend control and its reproducibility limitations in its [attention API](https://docs.pytorch.org/docs/stable/nn.attention.html) and [reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Representative padded benchmark

The benchmark used FP32 parameters and AdamW state, BF16 autocast, gradient clipping, three warmups, and eleven timed full steps.
The representative 512-token batch had 19.85% non-padding utilization.
The representative 1024-token batch had 26.81% non-padding utilization.

| Length / batch | Checkpointing | Backend | Full step | Peak allocated | Peak reserved |
|---|---|---|---:|---:|---:|
| 512 / 8 | No | SDPA | 364.7 ms | 4,496.9 MiB | 5,222 MiB |
| 512 / 8 | No | FA2 | 334.2 ms | 3,901.8 MiB | 4,520 MiB |
| 512 / 8 | Yes | SDPA | 465.7 ms | 2,654.3 MiB | 2,944 MiB |
| 512 / 8 | Yes | FA2 | 433.6 ms | 2,659.0 MiB | 2,906 MiB |
| 1024 / 4 | No | SDPA | 409.9 ms | 4,641.2 MiB | 5,430 MiB |
| 1024 / 4 | No | FA2 | 346.3 ms | 3,939.4 MiB | 4,576 MiB |
| 1024 / 4 | Yes | SDPA | 520.9 ms | 2,656.3 MiB | 2,944 MiB |
| 1024 / 4 | Yes | FA2 | 438.7 ms | 2,659.5 MiB | 2,908 MiB |

With checkpointing, FA2 reduced a full step by 6.9% at 512 and 15.8% at 1024.
Dense no-padding controls showed reductions of 4.2% and 10.7%, so part of the representative gain came from unpadding.
Without checkpointing, FA2 reduced allocated memory by roughly 13% to 15%, but reserved memory left unsafe headroom on this 6 GiB GPU.
With checkpointing, both backends had safe memory headroom and FA2 had almost no allocated-memory advantage.

## Numerical comparison

The frozen classifier produced identical losses and predictions with SDPA and FA2 on the representative 512-token and 1024-token batches.
The maximum BF16 logit difference was `0.0625`.
The final CLS hidden-state relative L2 difference was about 1.1%.
The first-layer Wqkv gradient relative L2 difference was 2.06% at 512 and 8.51% at 1024, with cosine similarities of 0.99979 and 0.99802.
The forward results are close, but the backward passes are not bitwise interchangeable and may follow different training trajectories.

## Decision

- Keep direct-user experiments at 256 tokens until matched long-benign data and held-out controls exist.
- Test 512 first in that declared ablation and use a padded-token budget or length bucketing.
- Use checkpointing and pinned FA2 for a justified 1024-token experiment.
- Do not add the executable kernel as a default dependency while the selected model recipe remains at 256 tokens.
- Prefetch and hash-verify the binary before any future run because the loader still attempted publisher verification over the network with cached files.
