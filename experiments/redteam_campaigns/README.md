# Red-team campaign projection

Projects the first-party red-team campaign parquets into the archive JSONL under
`data-archive/redteam/`. Archive-class work: the output is not a canonical source and must
never be copied into `data/`.

```bash
uv run --extra showcase python -m experiments.redteam_campaigns.normalize \
  --input data-archive/redteam/raw/normalized_dataset_dedup.parquet \
  --out data-archive/redteam/redteam_campaigns_20260806 --check
```

`--input` takes one or more parquets, so later campaign batches fold in without a code
change. Output rows are sorted by id and gzipped with `mtime=0`, so repeated runs are
byte-identical; `--check` asserts unique ids, non-empty text, row accounting, the
`input_channel` and label vocabularies, and hash agreement.

The projection drops `raw_record`, `target_response`, and the three `source_*` path columns.
`proposed_security_label` is a derived heuristic, deliberately named so it is never mistaken
for a contract label. Reading rules and the decision record are in `data-archive/README.md`
and `reports/redteam-campaign-data.md`.
