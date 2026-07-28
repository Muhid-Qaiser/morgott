# experiments/

Research scripts. Nothing here is wired into the `morgott` CLI, and per
`AGENTS.md` the archived runner must not become the default trainer.

Conclusions live in `reports/model-experiments.md` and
`reports/corpus-sanity-audit.md`. This file is only how to re-run things.

## Prerequisite: the archived runner is not in git

`experiments/_archived/routing_encoder.py` is generated, gitignored, and rebuilt
from the 2026-07-26 research source archive:

```bash
python3 experiments/_archived/build_shim.py          # write
python3 experiments/_archived/build_shim.py --check  # verify only
```

The archive it reads from is gitignored and absent from git history, so it is
committed compressed in `data-archive/` along with the generated matched pairs.
Restore both before running anything:

```bash
sha256sum -c data-archive/SHA256SUMS
tar xzf data-archive/research-source-archive-2026-07-26.tar.gz
mkdir -p artifacts/matched_pairs
gzip -dc data-archive/matched_pairs_20260726.jsonl.gz \
  > artifacts/matched_pairs/pairs_20260726T105034Z.jsonl
python3 experiments/_archived/build_shim.py
```

Rebuild the pinned PromptShield and SEP projections from their public releases:

```bash
PYTHONPATH=src:experiments uv run python \
  experiments/prepare_external_eval.py
```

The converter verifies raw and projected SHA-256 digests and refuses to replace an existing output.
Historical Phase 3 heads remain local research artifacts; their durable metrics and decisions are in `reports/model-experiments.md`.

**A fresh clone has no corpus.** Only `data/manifest.json` is versioned;
`data/sources/` and `data/views/` are empty until you run `uv run morgott data`.
Every script here needs those views, so build the corpus first. Verified against
a clean clone: the restore above works, but nothing that reads `data/` will run
before the corpus build.

## Environment

Install the exact optional model environment:

```bash
uv sync --locked --extra encoder
```

Python 3.12, `torch` 2.13, `transformers` 5.14, scikit-learn, and `datasets` are pinned by the lockfile.
A 6 GB GPU is enough for evaluation and every retained run.

Archived-run scripts use:

```bash
PYTHONPATH=src:experiments:experiments/_archived uv run python \
  experiments/<script>.py
```

## Run this first after any change

```bash
PYTHONPATH=src:experiments:experiments/_archived uv run python \
  experiments/reproduce_check.py
```

Re-scores the frozen dev-test suite with the clean forward pass in
`encoder_infer.py` and compares against the runner's own recorded metrics.
Expect ROC AUC 0.996491 against 0.996500 recorded, tolerance 0.0001, over 29,173 route-labelled rows.
The earlier approximately 0.9969 figure selected rows by head-target availability rather than the route-label contract and is superseded.
If this check fails, no downstream number is trustworthy.

## Scripts

| script | what it does |
|---|---|
| `common.py` | shared `threshold_at_fpr` / `fpr_and_recall` / `load_records`. One definition on purpose: at a 0.1% operating point ~20 rows set the threshold, so a one-index difference between copies is not rounding |
| `encoder_infer.py` | inference-only reconstruction of the retained members; imports no training code |
| `reproduce_check.py` | harness validation, above |
| `promptshield_eval.py` | external evaluation with a contamination mask |
| `attempt_scaling.py` | ASR@k against `mutations.py`'s seven families |
| `mutations.py` | gradient-free surface mutations; `verify_intent_preserved` guards against counting broken payloads as evasions |
| `strict_normalize.py` | the stricter normaliser; run directly for the 13-technique comparison |
| `run_archived_recipe.py` | **the training entrypoint**; drives the archived recipe with only preprocessing and data varied |
| `eval_phase3.py` | scores every completed run on dev-test, PromptShield and SEP, grouped by condition with seed spreads |
| `prepare_external_eval.py` | fetches and projects PromptShield and SEP at pinned revisions and hashes |
| `matched_pairs/` | generation: `specs.py` categories, `diversity.py` uniqueness and refusal handling, `generate.py` the budgeted runner |
| `prepare_promptshield_training.py` | filters PromptShield train and validation against held-out rows without changing the canonical corpus |
| `prepare_combined_generic.py` | builds the update-matched Morgott and PromptShield causal comparison |
| `prepare_full_combined_generic.py` | builds the full canonical, PromptShield, and generated-pair training recipe |
| `train_combined_generic_head.py` | trains the update-matched frozen-mmBERT control used by the LoRA gate |
| `train_full_combined_generic_head.py` | trains the frozen-mmBERT full-data objective controls and pair-ranking ablation |
| `train_combined_generic_lora.py` | trains the update-matched rank-8 mmBERT LoRA engineering gate |
| `eval_combined_generic_head.py` | applies canonical calibration and separately reports canonical dev-test, PromptShield-internal source-disjoint development with complete-fit overlap caveats, SEP, finance, and pair diagnostics |
| `score_shadow_model.py` | loads one registered frozen or LoRA artifact and emits raw JSONL scores plus provenance only |

## Current generic first-pass shadow

The retained first-pass research shadow is frozen mmBERT with the `full_balanced` objective and pair-ranking weight 0.25.
Its three heads and the LoRA gate artifacts are registered in `model-artifacts.json` and stored through Git LFS.
They are advisory and are not wired into `morgott scan`.
The exact measured results and limitations are in `reports/model-experiments.md`.
The seed-42 rank-8 LoRA gate passed its preliminary matched comparison, but it used the smaller Morgott plus PromptShield mixture and remains a one-seed research result.
The next possible modelling run is one complete-mixture LoRA run with pair ranking, not a seed or hyperparameter sweep.
At the measured LoRA throughput and the full schedule's 25,071 updates, it is approximately a 36 GPU-hour run on the current RTX 4050 and is deferred.

Run these commands only after building the canonical corpus and restoring the pinned PromptShield and SEP files described above.
Use a clean artifact tree because the combined/full preparation, training, and evaluation commands refuse to replace an existing result.

```bash
PYTHONPATH=src:experiments uv run python \
  experiments/prepare_promptshield_training.py
PYTHONPATH=src:experiments uv run python \
  experiments/prepare_combined_generic.py --seed 42
PYTHONPATH=src:experiments uv run python \
  experiments/prepare_full_combined_generic.py --seed 42
for seed in 42 43 44; do
  PYTHONPATH=src:experiments uv run python \
    experiments/train_full_combined_generic_head.py \
    --objective full_balanced --pair-ranking-weight 0 --seed "$seed"
  PYTHONPATH=src:experiments uv run python \
    experiments/train_full_combined_generic_head.py \
    --objective full_balanced --pair-ranking-weight 0.25 --seed "$seed"
  for rank in 0p0 0p25; do
    PYTHONPATH=src:experiments uv run python \
      experiments/eval_combined_generic_head.py \
      "artifacts/combined_generic/full_runs/jhu-clsp-mmbert-base_objective-full-balanced_pair-rank-${rank}_s${seed}"
  done
done
```

Frozen evaluations populate the ignored `artifacts/combined_generic/evaluation_feature_cache_v1/` with content-addressed, SHA-256-verified pooled features.
Later frozen heads reuse those exact BF16 features, while LoRA bypasses the cache because its encoder representation differs.
Deleting the cache only forces recomputation.

The preliminary LoRA gate intentionally uses only the update-matched Morgott and PromptShield populations so that encoder adaptation is the isolated change.
It does not use the generated pairs.
Both gate runs use the seed-42 selection prepared above so their fitted rows are identical.

```bash
PYTHONPATH=src:experiments uv run python \
  experiments/train_combined_generic_head.py \
  --condition combined \
  --seed 42 \
  --output-root artifacts/combined_generic/lora_gate/frozen_runs
PYTHONPATH=src:experiments uv run python \
  experiments/train_combined_generic_lora.py --seed 42
PYTHONPATH=src:experiments uv run python \
  experiments/eval_combined_generic_head.py \
  artifacts/combined_generic/lora_gate/frozen_runs/jhu-clsp-mmbert-base_combined_s42
PYTHONPATH=src:experiments uv run python \
  experiments/eval_combined_generic_head.py \
  artifacts/combined_generic/lora_gate/lora_runs/jhu-clsp-mmbert-base_combined_lora-r8_s42
```

Both generic recipes strictly normalize and truncate each row to its first 512 tokens.
They do not chunk long documents or localize an injected span.

Score trusted-channel JSONL records with either retained shadow:

```bash
PYTHONPATH=src:experiments uv run python \
  experiments/score_shadow_model.py \
  model-artifacts.json \
  full-frozen-s42 \
  input.jsonl \
  scores.jsonl
```

Each input record must contain `id`, `text`, and a trusted runtime-supplied `input_channel` of `direct_user` or `untrusted_content`.
The output contains raw scores and artifact provenance, never an authorization decision.

## Training

Use `run_archived_recipe.py`, never a reimplementation. An earlier reimplementation
reached 38.02% dev recall where the recipe reaches 63.04%, because it batched at
a fixed 256 rows against the runner's 4,096-token budget: about 720 optimizer
updates versus 10,896. Check `training.updates` in two `result.json` files before
explaining any metric gap.

```bash
# baseline reproduction
python3 experiments/run_archived_recipe.py --only-weak-transfer --normalize raw

# the full condition
python3 experiments/run_archived_recipe.py --only-weak-transfer \
    --normalize strict --matched-pairs --seed 42
```

Flags: `--normalize raw|strict`, `--matched-pairs`, `--pairs-fraction`,
`--pair-ranking`, `--seed`, `--model-id`, `--only-weak-transfer` (trains only
`wildguard_weak_transfer`, the recipe that feeds the retained ensemble; roughly
halves wall-clock).

A single run is **~5.5 minutes**. Nine runs took 53 minutes.

## Data generation

```bash
python3 experiments/matched_pairs/generate.py --pilot          # measure cost first
python3 experiments/matched_pairs/generate.py --budget 17.50
```

Constraints that must not be relaxed:

- **Specification-only.** No corpus text is ever sent to a provider. `AGENTS.md`
  is explicit that an API key existing is not a reason.
- Both halves of a pair come from one model in one call, so generator style is
  constant within a pair and cannot become the signal a classifier learns.
- Output goes to `artifacts/matched_pairs/`, never `data/`, as
  `label_basis: model_generated`.
- The budget reserves atomically before each call. An earlier version compared a
  projection against a stale total and overspent its cap by $0.65 across 24
  workers.

## Traps that have already cost reruns

- `pgrep -f foo.py` matches its own watcher's command line. Use `[f]oo.py`, and
  check `ps` **TIME** (accumulated CPU) rather than process existence — a
  wrapper waiting on itself shows 0:00 against 50 minutes elapsed.
- Build synthetic training rows by reading a real record first. Two crashes came
  from assumed shapes: a missing `data_role`, then `pair_head` passed as a name
  where the runner does `HEADS[example["pair_head"]]`.
- `eval_phase3.py` now fails when a requested or default run directory is incomplete and validates exactly one head, report, and result.
  Keep the explicit run list in the command so the expected queue remains auditable.
- mmBERT and ModernBERT both have `hidden_size` 768, so an mmBERT head loads
  into a ModernBERT encoder without error and produces meaningless scores. The
  encoder is chosen from the run directory; keep it that way.
- Compare a single head against a single head. The retained 81.14% is the
  two-model ensemble; its best single member is 70.65%.
