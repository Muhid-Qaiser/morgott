# Cascade comparison showcase

This folder preserves the shadow-only management demo and its aggregate July 2026 comparison findings.
It does not include the source red-team corpus, provider responses, prompts, credentials, or resumable SQLite ledgers.

## Saved findings

The positive-only red-team set contains 5,132 non-empty attempts.
It measures attempt recall, not false-positive rate or precision.

| Pipeline | Detected | Valid-output recall | Coverage |
| --- | ---: | ---: | ---: |
| DeepSeek V4 Flash direct | 2,783 / 5,132 | 54.23% | 100.00% |
| Morgott BF16 cascade | 2,024 / 5,132 | 39.44% | 100.00% |
| Meta Prompt Guard 2 86M | 1,012 / 5,132 | 19.72% | 100.00% |
| GPT-OSS Safeguard 20B | 265 / 5,126 valid outputs | 5.17% | 99.88% |

The retained 6,000-row calibration and 14,000-row evaluation panel supports the copied `0.2`, `0.99999`, and `0.9` cascade thresholds under the approximate 2% calibration false-positive ceiling.
On the 14,000-row evaluation split, that policy produced 66.79% recall, 1.81% FPR, 96.50% precision, and a 22.17% provider call rate.
These panels and results are already-open development evidence, not production claims.

The two threshold files answer different questions.
`threshold-pass-results.json` reports positive-only sensitivity over the 5,132 red-team attempts.
`threshold-panel-crosscheck.json` checks the same threshold grid against the retained benign denominator.
Higher positive-only recall alone is not sufficient to change the operating thresholds.

## Run the saved showcase

Fetch the registered Git LFS model artifacts, then install the three optional dependency groups:

```bash
git lfs pull
uv sync --locked --extra encoder --extra cascade --extra showcase
```

If `.env` contains `OPENROUTER_API_KEY`, load it into the process without printing it:

```bash
set -a
. ./.env
set +a
uv run --extra encoder --extra cascade --extra showcase python experiments/cascade_showcase/app.py
```

Open [http://127.0.0.1:7860/](http://127.0.0.1:7860/).
The saved aggregate comparison renders without the source red-team dataset.
The live input comparison requires the registered mmBERT artifacts.
Prompt Guard uses the local Hugging Face cache by default.
Set `MORGOTT_ALLOW_MODEL_DOWNLOAD=1` only when downloading its pinned revision is intended.

## Resume the full comparison

Set the path to the same normalized source parquet before starting a full remote run:

```bash
export MORGOTT_RED_TEAM_DATA=/absolute/path/to/normalized_dataset_dedup.parquet
uv run --extra encoder --extra cascade --extra showcase python experiments/cascade_showcase/app.py
```

The app stores its resumable SQLite ledgers and regenerated summaries under `~/.cache/morgott/cascade-showcase`.
Override that location with `MORGOTT_SHOWCASE_STATE_DIR`.
Remote execution requires an explicit checkbox confirmation in the UI and reads only `OPENROUTER_API_KEY`.

Run the threshold sensitivity pass separately:

```bash
uv run --extra encoder --extra cascade --extra showcase python experiments/cascade_showcase/threshold_pass.py
```

All learned decisions remain advisory.
The showcase does not grant or deny authority.
