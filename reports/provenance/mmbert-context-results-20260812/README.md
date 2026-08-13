# ModernBERT context campaign results

This directory is the byte-for-byte result archive indexed by
`reports/mmbert-context-comparison.json`. Each exact copy is named by its unique
manifest key so same-named evaluation files cannot collide.

Contents:

- `records/` contains the 21 manifest-bound aggregate JSON artifacts.
- `SOURCE_PATHS.tsv` retains every original source path and maps it to the copy.
- `SHA256SUMS` covers every archived JSON file.

Verify the archive from this directory:

```bash
cd reports/provenance/mmbert-context-results-20260812
sha256sum -c SHA256SUMS
```

The JSON files contain aggregate training, evaluation, reserve, long-code,
parity, and tail-audit results; they do not contain corpus rows or prompts.
Per-row scoring journals and unused intermediate checkpoints remain reproducible
compute caches. The three snapshots that generated the reported comparisons are
preserved in `../mmbert-context-checkpoints-20260812/`.

These results are advisory research evidence. No archived checkpoint is in the
maintained inference registry or approved for blocking.
