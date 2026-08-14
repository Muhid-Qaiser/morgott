# Retained generated data

This directory contains generated data that is retained but is **not** part of the canonical corpus.
Nothing here is a canonical corpus source and nothing here may be copied into `data/`.

Two independent sets live here: the matched pairs used by the retained full-data mmBERT training runs, and the first-party red-team campaign corpus under `redteam/`.

## Matched pairs

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

## Red-team campaigns (`redteam/`)

First-party automated red-team output from 96 campaigns run between 2026-07-20 and 2026-07-28.
Attacker models (`mistralai/mixtral-8x22b-instruct`, four DeepSeek variants, `xiaomi/mimo-v2.5-pro`, `microsoft/wizardlm-2-8x22b`, `z-ai/glm-5.2`, others) generated attack prompts against target `z-ai/glm-5.2` (5,099 of 5,112 retained rows), judged by `openai/gpt-5.4-mini` and `openai/gpt-4o`.

`raw/` holds the three source files verbatim and hash-pinned; the derived files are regenerated from them.

The payload files are **not in git** — they live in Azure like the rest of the corpus, and `data-archive/SHA256SUMS` is the git-tracked integrity anchor. Fetch them with `scripts/azsync.sh pull`, then verify with `sha256sum -c data-archive/SHA256SUMS`.

| File | What it is |
| --- | --- |
| `raw/normalized_dataset_dedup.parquet` | 5,133 rows, 31 columns. The campaign corpus. |
| `raw/agentdojo_workspace.parquet` | 6,330 rows from the AgentDojo workspace and travel suites. Only 693 distinct payloads and 63 distinct user prompts; the carrier documents were not materialized. |
| `raw/danger_worked_cannabilism.jsonl` | 10 rows. A prefill + danger-gate smoke record over JailbreakBench behaviors. Not a dataset and never a denominator. |
| `redteam_campaigns_20260806.jsonl.gz` | Derived projection: 5,112 retained rows, 26 of them recovered from the generator envelope. |
| `redteam_campaigns_20260806.quarantine.jsonl.gz` | 21 rows excluded as unusable detector input. |
| `redteam_campaigns_20260806.summary.json` | Counts, input hashes, and caveats. |
| `subversion_pairs_20260806.jsonl.gz` | 864 constructed matched pairs; see below. |

Regenerate the derived files deterministically (repeated runs are byte-identical):

```bash
sha256sum -c data-archive/SHA256SUMS
uv run --extra showcase python -m experiments.redteam_campaigns.normalize \
  --input data-archive/redteam/raw/normalized_dataset_dedup.parquet \
  --out data-archive/redteam/redteam_campaigns_20260806 --check
uv run --extra showcase python -m experiments.redteam_subversion_control.run
```

### What the projection drops, and why

`raw_record` (full judge reasoning and provider metadata), `target_response` (model-generated harmful completions), and `source_root` / `source_run_folder` / `source_file` (which embed the generating machine's user directory) are all omitted.
The raw parquet remains the audit trail for anything the projection drops.

### Reading rules

The corpus is positive-only: there is no benign denominator, so it supports no false-positive rate, precision, or production claim.

`verdict` and `breached` record whether one target model was breached on one day.
They are outcome metadata, never detector labels — the same rule `docs/data-contract.md` already applies to failed HackAPrompt and Tensor Trust attempts.

`proposed_security_label` is a derived heuristic over model-generated text, named `proposed_*` precisely so it is never mistaken for a contract label.
Roughly two thirds of the corpus is a bare harmful request with no instruction subversion, which `docs/data-contract.md` treats as `harmful_non_injection`, a different label from injection.

`category` is confounded with `attack_mode`: whole categories were run under a single strategy, so per-category slices measure the campaign, not the topic.

`agentdojo_derived` flags the 13 retained AgentDojo-derived rows. Repo policy keeps AgentDojo text out of the training corpus.

The leakage atom is `split_group_id` (`redteam:<run_id>:<goal_id>`), not the row: the TAP and PAIR campaigns are attack trees whose children are mutations of their parents.

### Subversion pairs

864 rows use one fixed jailbreak wrapper ending in a literal `Request: <inner>` tail.
Splitting the wrapper off constructs 864 pairs carrying identical harmful content and differing only by the presence of instruction subversion.
The `inner` strings are **derived text not attested in the source** — archive-class diagnostic material, never canonical.
The measured result is in `reports/redteam-campaign-data.md`.
