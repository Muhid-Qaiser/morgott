# data-archive/

Irreplaceable artifacts, committed because they cannot be regenerated.

`artifacts/` is gitignored, which is correct for the several hundred megabytes
of trained heads and re-downloadable evaluation sets it holds. Two things in
there are not replaceable, and were one lost disk away from taking the July 2026
results with them.

This directory is deliberately small (4.9 MB) and deliberately outside
`data/`, which `AGENTS.md` reserves for the canonical corpus. Nothing here is a
corpus source, and none of it may enter `data/sources/` or a training view
without going through the normal source-review process.

## Contents

| file | what | why committed |
|---|---|---|
| `matched_pairs_20260726.jsonl.gz` | 11,046 generated matched pairs (22,092 rows) | ~$18 of OpenRouter spend, generated at temperature 1.0 across ten models — rerunning would produce different data, not the same data |
| `matched_pairs_20260726.summary.json` | generation run summary | spend, dedup counts, refusal rates, per-model breakdown |
| `research-source-archive-2026-07-26.tar.gz` | the archived `routing_encoder.py` runner | **absent from git history entirely** — `git log --diff-filter=D -- src/morgott/routing_encoder.py` returns nothing. Every Phase 3 result depends on it |
| `SHA256SUMS` | digests | verify before use |

## Restore

```bash
cd /path/to/morgott
sha256sum -c data-archive/SHA256SUMS

mkdir -p artifacts/matched_pairs
gzip -dc data-archive/matched_pairs_20260726.jsonl.gz \
  > artifacts/matched_pairs/pairs_20260726T105034Z.jsonl
cp data-archive/matched_pairs_20260726.summary.json \
   artifacts/matched_pairs/pairs_20260726T105034Z.summary.json

tar xzf data-archive/research-source-archive-2026-07-26.tar.gz   # into artifacts/
python3 experiments/_archived/build_shim.py
```

`build_shim.py` pins the archive's sha256 and refuses to build from anything
else, so a corrupted restore fails loudly rather than producing subtly different
results.

## Deliberately NOT here

- `artifacts/external_eval_data/` (45 MB) — PromptShield and SEP are
  re-downloadable at the revisions pinned in each `_meta.json`.
- `artifacts/phase3_archived/` (670 MB) — 19 trained heads, reproducible in
  about 55 minutes of GPU from the two archives above plus the corpus.

## Provenance of the generated pairs

Specification-only generation: the prompts in `experiments/matched_pairs/specs.py`
describe what to write and never carry a row from `data/`. No corpus text was
sent to any provider. Rows are `label_basis: model_generated` weak supervision.

Measured, not assumed: zero exact and zero near-duplicate collisions against the
PromptShield test split across all 22,092 rows, and zero against the corpus
during generation. Ten models across seven labs, fifteen languages, no single
lab above 36% of output, distinct-3-gram 0.437.
