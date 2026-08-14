# mmBERT checkpoint red-team reserve evaluation

Scores one explicit completed mmBERT checkpoint on the frozen 5,112-row
first-party red-team reserve. This is consumed development evidence. The panel
contains source-attested subversion rows and bare-harmful off-target controls,
but no benign denominator, so its pooled flag rate is neither recall nor FPR.

The runner does **not** recalibrate. It transports the canonical component
threshold from a completed full-panel evaluation of the exact same run,
checkpoint, and evaluation context cap. Current artifacts bind all of the
following before model loading:

- the run-result, snapshot, full-evaluation, full-score, and reserve hashes;
- training and evaluation caps (`512` or `1024`) and whether the cell is native;
- the full evaluator's model, scoring, and evaluation-identity hashes;
- the head contract, score columns, threshold evidence, inputs, and batch size.

Scoring and truncation both use the bound evaluation cap. The cap is also part
of the score-journal model/scoring identities, the final evaluation identity,
and the default output and journal names. A 1024-token cell therefore cannot
silently execute with the historical 512-token scorer or resume a 512-token
journal.

Only aggregate JSON and numeric score-journal shards are written. Prompt text
and row IDs are not copied into either artifact. The maintained single-output
head produces one instruction-subversion sigmoid score column.

## Required native decision cells

The decision comparison requires both native cells at the retained update
17,000 snapshot:

1. the 512-trained control scored at 512; and
2. the 1024-trained arm scored at 1024.

Run the full-panel evaluations first. These commands deliberately give even
the native 512 cell an explicit context-qualified output name so the new
evidence cannot collide with its historical implicit-512 artifact:

```bash
cd /workspace/code/morgott
export HF_HOME=/workspace/hf_cache
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PAIRS=data-archive/matched_pairs_20260726.jsonl.gz
ADDITIONAL_PAIRS=artifacts/mmbert_lpft_new_data_rebuilt/train/pairs.jsonl.gz

RUN512=artifacts/mmbert/runs/mmbert-lora-full-s42-mb24-nolengthgroup-noharm-current-control
SNAP512=artifacts/mmbert/runs/.mmbert-lora-full-s42-mb24-nolengthgroup-noharm-current-control.snapshots/update-017000.pt
FULL512="$RUN512/evaluation-update-17000-trainctx512-evalctx512"

uv run --locked --extra encoder python -m morgott.models.mmbert.evaluate \
  "$RUN512" \
  --snapshot "$SNAP512" \
  --additional-pairs "$ADDITIONAL_PAIRS" \
  --batch-size 24 \
  --tokenizer-workers 17 \
  --evaluation-max-tokens 512 \
  --output "$FULL512" \
  --score-journal "$RUN512/.evaluation-update-17000-trainctx512-evalctx512.score-journal"

RUN1024=artifacts/mmbert/runs/mmbert-lora-full-s42-mb24-nolengthgroup-noharm-ctx1024
SNAP1024=artifacts/mmbert/runs/.mmbert-lora-full-s42-mb24-nolengthgroup-noharm-ctx1024.snapshots/update-017000.pt
FULL1024="$RUN1024/evaluation-update-17000-trainctx1024-evalctx1024"

uv run --locked --extra encoder python -m morgott.models.mmbert.evaluate \
  "$RUN1024" \
  --snapshot "$SNAP1024" \
  --additional-pairs "$ADDITIONAL_PAIRS" \
  --batch-size 24 \
  --tokenizer-workers 17 \
  --evaluation-max-tokens 1024 \
  --output "$FULL1024" \
  --score-journal "$RUN1024/.evaluation-update-17000-trainctx1024-evalctx1024.score-journal"
```

After each matching full-panel artifact exists, run the reserve cells:

```bash
uv run --locked --extra encoder \
  python -m experiments.mmbert_redteam_snapshot_eval.run \
  "$RUN512" \
  --snapshot "$SNAP512" \
  --full-evaluation "$FULL512/evaluation.json" \
  --evaluation-max-tokens 512 \
  --additional-pairs "$ADDITIONAL_PAIRS" \
  --batch-size 24

uv run --locked --extra encoder \
  python -m experiments.mmbert_redteam_snapshot_eval.run \
  "$RUN1024" \
  --snapshot "$SNAP1024" \
  --full-evaluation "$FULL1024/evaluation.json" \
  --evaluation-max-tokens 1024 \
  --additional-pairs "$ADDITIONAL_PAIRS" \
  --batch-size 24
```

The default reserve outputs are, respectively:

- `$RUN512/redteam-reserve-evaluation-update-17000-trainctx512-evalctx512/`
- `$RUN1024/redteam-reserve-evaluation-update-17000-trainctx1024-evalctx1024/`

Their numeric journals use the same names with a leading dot and the suffix
`.score-journal`.

## Optional cross-cap diagnostics

The off-diagonals can distinguish learned long-context effects from inference
cap effects, but they do not replace either native decision cell. Produce the
matching full panels first, using the same explicit naming convention, then run:

```bash
uv run --locked --extra encoder \
  python -m experiments.mmbert_redteam_snapshot_eval.run \
  "$RUN1024" \
  --snapshot "$SNAP1024" \
  --full-evaluation "$RUN1024/evaluation-update-17000-trainctx1024-evalctx512/evaluation.json" \
  --evaluation-max-tokens 512 \
  --additional-pairs "$ADDITIONAL_PAIRS" \
  --batch-size 24

uv run --locked --extra encoder \
  python -m experiments.mmbert_redteam_snapshot_eval.run \
  "$RUN512" \
  --snapshot "$SNAP512" \
  --full-evaluation "$RUN512/evaluation-update-17000-trainctx512-evalctx1024/evaluation.json" \
  --evaluation-max-tokens 1024 \
  --additional-pairs "$ADDITIONAL_PAIRS" \
  --batch-size 24
```

## Historical evidence

The runner accepts a wholly legacy implicit-512 full evaluation only when it
uses the maintained single-output head: all context, scoring-identity, and
evaluation-identity fields must be absent, and `--evaluation-max-tokens` must
be omitted.
The rejected two-output Arm 6 runner is provenance-only in Git history and the
archived campaign source at
`reports/provenance/mmbert-context-campaign-source-20260812.tar.gz`.

The legacy single-output exception exists only so already-published 512
evidence remains reproducible. It is not accepted as a new 512-vs-1024 decision
cell. Supplying any subset of the current cap metadata fails closed, omitting
the cap for a current full evaluation fails closed, and supplying an explicit
cap against a legacy full evaluation fails closed. Packaged-selected checkpoint
support is retained for that one-output evidence, but the context comparison
uses the fixed retained snapshot.

Outputs are immutable and publish atomically. If scoring is interrupted, rerun
the exact command. Read `instruction_subversion.subversion_attested` as reserve
recall and `instruction_subversion.bare_harmful_control` as an off-target
harmful-control flag rate. Never quote the pooled aggregate as attack recall or
the harmful control as a benign FPR.
