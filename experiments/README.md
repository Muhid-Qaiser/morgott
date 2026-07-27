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

Not archived because they are reproducible: `artifacts/external_eval_data/`
(PromptShield and SEP, re-downloadable at the revisions pinned in each
`_meta.json`) and `artifacts/phase3_archived/` (19 trained heads, about 55
minutes of GPU).

**A fresh clone has no corpus.** Only `data/manifest.json` is versioned;
`data/sources/` and `data/views/` are empty until you run `uv run morgott data`.
Every script here needs those views, so build the corpus first. Verified against
a clean clone: the restore above works, but nothing that reads `data/` will run
before the corpus build.

## Environment

Python 3.12, `torch` 2.13 and `transformers` 5.14 present in the environment
(optional dependency group `encoder`), scikit-learn, `datasets`. A 6 GB GPU is
enough: every run below fits. `.env` supplies `OPENROUTER_API_KEY` and
`HF_TOKEN`; neither is ever printed or committed.

All scripts run as:

```bash
PYTHONPATH=src:experiments/_archived python3 experiments/<script>.py
```

## Run this first after any change

```bash
python3 experiments/reproduce_check.py
```

Re-scores the frozen dev-test suite with the clean forward pass in
`encoder_infer.py` and compares against the runner's own recorded metrics.
Expect ROC AUC 0.996906 against 0.996500 recorded, tolerance 0.002. **If this
fails, no downstream number is trustworthy.**

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
| `matched_pairs/` | generation: `specs.py` categories, `diversity.py` uniqueness and refusal handling, `generate.py` the budgeted runner |

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
- `eval_phase3.py` discovers runs by globbing for `head.safetensors`, so a
  crashed run is skipped rather than counted. Compare the discovered count
  against what was queued, and watch for its `WARNING: unparsed run directory`.
- mmBERT and ModernBERT both have `hidden_size` 768, so an mmBERT head loads
  into a ModernBERT encoder without error and produces meaningless scores. The
  encoder is chosen from the run directory; keep it that way.
- Compare a single head against a single head. The retained 81.14% is the
  two-model ensemble; its best single member is 70.65%.
