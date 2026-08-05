# Retained generated data

This directory contains the generated matched pairs used by the retained full-data mmBERT training runs.
They are not canonical corpus sources and must not be copied into `data/`.

`matched_pairs_20260726.jsonl.gz` contains 11,046 generated pairs, or 22,092 rows.
`matched_pairs_20260726.summary.json` records the generation budget, model distribution, refusal rates, and deduplication summary.
`SHA256SUMS` verifies both retained files.

The generation specifications and runner are preserved by Git commit `91e8c829c8b39c8ff37a6ca2479c8fc057168d39`.
The prompts were specification-only and did not contain corpus rows.
Rows remain model-generated weak supervision with their original provenance.
The retained `domain` and `technique` fields are generator-authored free text, not controlled taxonomies; casing, whitespace, and wording vary.
Do not use them for slices or claims without a versioned normalization audit.
The maintained training loader uses only the paired texts, channel, and pair lineage.

Restore the uncompressed training input only when auditing the historical run:

```bash
sha256sum -c data-archive/SHA256SUMS
mkdir -p artifacts/matched_pairs
gzip -dc data-archive/matched_pairs_20260726.jsonl.gz \
  > artifacts/matched_pairs/pairs_20260726T105034Z.jsonl
```
