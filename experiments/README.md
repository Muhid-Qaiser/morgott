# Experiments

This directory is for disposable or study-specific work.
The maintained mmBERT data recipe, trainer, evaluator, and inference path live in `src/morgott/models/mmbert/`.

The exact source that produced the retained July 2026 artifacts remains at Git commit `91e8c829c8b39c8ff37a6ca2479c8fc057168d39`.
The compact maintained recipe is a clean successor, so new runs record new source and input hashes rather than pretending to reproduce the historical files byte for byte.
The durable historical conclusions and limitations remain in `reports/model-experiments.md`.

Create `experiments/<study>/` only for a bounded study with a written question,
entry point, stop rule, and evidence destination that does not belong in
maintained model behavior.
Promote only code that has a continuing caller and a stable contract.

Forward-only conventions for new studies (existing directories keep their
frozen layout because recorded provenance hashes bind their paths and bytes):

- one directory per study, holding a README that states the question and the
  stop rule;
- no bare `_vN` suffix directories; a genuinely new question gets a new
  descriptive study name;
- split the runner from metrics/reporting code so a study's evidence can be
  recomputed without re-invoking providers.

`experiments/mmbert_evaluation_contract.py` is the one retained shared helper:
frozen snapshot evaluations import it, and `tests/` pins its contract.

Three loose files predate this directory contract and remain in place because
historical reports bind their paths: `benchmark_mmbert_full_lora.py`,
`evaluate_prompt_guard_2_full_mixture.py`, and `lfm25-frozen-backbone.patch`.
Treat them as historical exceptions; new studies use subdirectories with a
README.

A few tests in `tests/` exercise code in this directory
(`test_guard_baseline_*.py`, `test_mmbert_longcode_snapshot_eval.py`,
`test_mmbert_redteam_snapshot_eval.py`). They stay in the default suite as a
deliberate exception: they gate maintained provenance contracts, so a change
that breaks them must fail `make check` even though the code under test lives
in experiments.
