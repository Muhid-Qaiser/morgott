# Trackio experiment history

This directory preserves the Morgott-only Trackio history in Git LFS before
the disposable RunPod volume is removed.

- `morgott.curated.db.zst` is a consistent SQLite online backup of the current
  curated database, including the compact comparison summaries and retained
  historical curves.
- `morgott.pre-curation.db.zst` is an exact copy of the recovery snapshot made
  before dashboard curation.

Both databases passed SQLite `PRAGMA quick_check` before compression and again
after a clean decompression. They use `zstd -19`; compression changes only the
container bytes, while `DATABASE_SHA256SUMS` binds the restored databases.

Integrity files:

- `SOURCE_PATHS.tsv` records origin, snapshot method, raw hash, and archive hash.
- `SHA256SUMS` covers the Git LFS `.zst` objects.
- `DATABASE_SHA256SUMS` covers the decompressed `.db` files.

Restore and verify from the repository root:

```bash
zstd -d reports/provenance/trackio-20260812/morgott.curated.db.zst -o /tmp/morgott.curated.db
zstd -d reports/provenance/trackio-20260812/morgott.pre-curation.db.zst -o /tmp/morgott.pre-curation.db
(cd /tmp && sha256sum -c "$OLDPWD/reports/provenance/trackio-20260812/DATABASE_SHA256SUMS")
```

After restoration, SQLite `PRAGMA quick_check` must return `ok`. A bounded scan
of every preserved artifact found no recognized credential or private-key
patterns. The databases contain experiment metrics, run configuration, and
system telemetry; they do not contain corpus rows or prompts.

The curated database is the normal dashboard source. The pre-curation snapshot
is recovery evidence if a hidden historical metric is needed later; it should
not replace the curated default merely to expose every old metric again.
