# Experiments

This directory is for disposable or study-specific work.
The maintained mmBERT data recipe, trainer, evaluator, and inference path live in `src/morgott/models/mmbert/`.

The exact source that produced the retained July 2026 artifacts remains at Git commit `91e8c829c8b39c8ff37a6ca2479c8fc057168d39`.
The compact maintained recipe is a clean successor, so new runs record new source and input hashes rather than pretending to reproduce the historical files byte for byte.
The durable historical conclusions and limitations remain in `reports/model-experiments.md`.

Create `experiments/<study>/` only for an explicitly authorized study that does not belong in maintained model behavior.
Promote only code that has a continuing caller and a stable contract.
