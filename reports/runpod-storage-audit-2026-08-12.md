# RunPod network-storage audit

Date: 2026-08-12

Status: read-only inventory. No corpus, weights, result, checkpoint, cache, or
Trackio database was deleted by this audit.

## Finding

The volume is not close to a 200 GB limit. The detailed pass measured about
106.62 GB of apparent file content and 116.51 GB of allocated space. MooseFS
small-file allocation accounts for roughly 9.9 GB of the difference, so the
RunPod UI may move by the logical number rather than the larger allocated-space
estimate.

The largest roots were approximately:

| Root | Apparent purpose |
|---|---|
| `/workspace/hf_cache` | 55--58 GB of model and Trackio caches |
| `/workspace/code` | 40--43 GB, chiefly 21.6 GB corpus and 19.2 GB artifacts |
| `/workspace/.cache` | 9.5--10.1 GB of uv, Triton, and pip caches |
| `/workspace/home` | about 6 GB of editor and Codex state |

Trackio is not the storage problem. Its active project plus preserved database
backups occupy well under 0.5 GB. Deleting experiment history would save little
and would weaken provenance.

## Recoverable space

Roughly 44.0 GB apparent / 51.6 GB allocated can be reclaimed without touching
the corpus, result JSONs, selected weights, active mmBERT base model, or
evaluation journals:

- about 38.0 GB of completed external guard-model caches, excluding gated
  AprielGuard;
- about 5.6 GB apparent / 10.1 GB allocated from uv, Triton, and pip caches;
- about 0.1 GB apparent / 3.25 GB allocated from stale Codex plugin clones after
  the current task is closed; and
- a stale approximately 0.22 GB partial LFS ONNX download.

Do not count this entire number as an immediate RunPod-UI reduction because of
the filesystem's allocation overhead.

## Retain for now

- `data/` and its one manifest: canonical evidence, not a cache;
- all context/checkpoint aggregate JSONs and score journals until the tracked
  comparison manifest is committed;
- the mmBERT base model and selected 17,000/18,500 snapshots;
- AprielGuard's gated cache until access/redownload behavior is known;
- Granite Guardian 3.2 and ProtectAI v2 caches because their full identical-panel
  comparisons are explicitly deferred future work;
- Trackio's live `morgott` database and dated raw backups.
