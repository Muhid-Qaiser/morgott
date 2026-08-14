# Frozen mmBERT LP-FT comparison

This is the one owner-authorized encoder comparison after the fresh evidence pass.

The candidate starts from pinned `jhu-clsp/mmBERT-base` and the retained full-data frozen head with SHA-256 `a3007323edb6da1d671b80ea1f7f46451384ddb2f45abeb2cae9d645404f5813`.
It trains only encoder layers 20 and 21, the final encoder normalization, and the existing head.
It uses seed 42, three epochs, the rebuilt full mixture plus 22,716 retained SWE-rebench V2 training pairs, the unchanged equal-domain BCE plus pair-ranking objective, 25,083 updates, and the unchanged validation checkpoint rule.
The head and encoder learning rates are fixed at `3e-5` and `1e-5`.
No architecture, loss, seed, learning-rate, context-length, or threshold sweep is authorized.

The incumbent is the registered full-mixture rank-8 LoRA seed 42 artifact.
Comparison reports AUROC, PR-AUC, source-macro and worst-source results, precision-recall and ROC curves, matched-pair ordering, finance errors, latency, and memory.
The frozen protocol requested the completed mutation ASR curve, the SWE-rebench workload panel, and the sealed LogInject panel beside the core comparison.
The handoff stopped after the decisive core result, so those three follow-ups were not run for this candidate and are not part of its evidence.

Neither 1% nor 2% FPR is a selection target.
Those coordinates may be shown only as descriptive curve points.
The candidate is retained only if it is Pareto-competitive across source-heldout detection, worst-source transfer, legitimate workload, and mutation robustness.
A gain obtained solely by increasing restriction or review load is not an improvement.

The candidate cascade reviews only the highest-scoring eligible window per artifact.
Direct-user high-zone windows may be reviewed, while untrusted-content high-zone windows remain locally restricted.
The DeepSeek V4 Flash 0731 reviewer, Cloudflare provider, prompt, and reviewer threshold remain unchanged.
Candidate local operating points must be selected from validation only and then frozen before either sealed panel is scored.
If no candidate point dominates or offers a materially better explicit tradeoff than the incumbent, the candidate is rejected and the maintained cascade remains unchanged.

Download the pinned source and rebuild the exact matched-pair inputs with:

```bash
uv run --locked hf download nebius/SWE-rebench-V2 data/train-00000-of-00001.parquet \
  --repo-type dataset \
  --revision 475dd5e8703bb5fb22dd3c60b5d038b019eba1e0 \
  --local-dir artifacts/upstream/swe-rebench-v2

uv run --locked --with pyarrow python experiments/mmbert_lpft_comparison/prepare_new_pairs.py \
  artifacts/upstream/swe-rebench-v2/data/train-00000-of-00001.parquet \
  artifacts/mmbert_lpft_new_data_rebuilt

diff -u artifacts/mmbert_lpft_new_data/manifest.json \
  artifacts/mmbert_lpft_new_data_rebuilt/manifest.json
```

The versioned exclusion snapshot preserves the repository-disjoint boundary from the SWE-rebench, SWE-chat, and SWE-bench Verified evaluation populations without requiring their ignored local panels.
The final `diff` must be empty before training.
The recorded `attack_span_start` values are one character late; see `artifacts/mmbert_lpft_new_data/metadata-correction.json`.
Repository grouping holds out task contexts, not attack templates.
Of the 2,590 unique dev-test attack spans, 2,267 also occur in training, so attack recall, AUROC, and pair ordering are in-family template evidence rather than attack-side generalization.

The retained held-out comparison record is `artifacts/mmbert_lpft_comparison/heldout_summary.json`.
Its ignored validation and dev-test input archives are not required by maintained inference.

This experiment is provenance-only.
The maintained trainer and evaluator no longer implement the rejected LP-FT adaptation or load its weights.
The exact historical implementation is preserved in Git commit `5269fe4cff6db37e802623aa91afbef054e5a6b1`; later context-campaign source is also archived at `reports/provenance/mmbert-context-campaign-source-20260812.tar.gz`.
Use those immutable sources only for a deliberate historical reproduction, not the current training CLI.

The pinned preflight retains 33,757 matched pairs after overlap filtering,
including 22,716 of the 22,719 new SWE-rebench V2 training pairs.

This remains an advisory research shadow and cannot grant authority or become a blocking control from this experiment.

The run completed on 2026-08-05 and was rejected because PromptShield ranking and indirect-document recall regressed severely despite the long-task clean-side improvement.
The candidate remains intentionally unregistered for maintained inference.
Its encoder, head, and scores are retained through Git LFS for later comparison, alongside the small result, evaluation, new-data manifest, and same-row held-out comparison records.
Their SHA-256 digests are `271df253cd4fc807c6060059e9bb62dc85e0c317aed15a892be5c7186cf3d515`, `2b5dbd647484e8753441118bf45bccd7c3836982474978f653c2de606bef98b5`, and `1cf04f41f76c2048d0170880a4794dae1d97da1efb38d1523a8cfd4c6c415aa5` respectively.
The owner approved deleting the resumable progress checkpoint after its SHA-256 `0e46cad31c0a92284ee85a6011c7ba723b78e4fcaac76f8669a4a69929585c2e` was recorded; the retained weights, scores, and evaluation records remain available for later comparison.
