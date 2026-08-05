# morgott data — source of truth

This is the canonical copy of the morgott corpus (prompt-injection / routing guard training data). It lives in Azure Blob Storage — account `vulsightdata`, container `morgott` — and is mirrored into a repo checkout under the same paths. **Azure is the source of truth**: every add/remove/correction on any machine ends with `scripts/azsync.sh push`.

## Layout

| Path | Size | Class | What it is |
|------|------|-------|------------|
| `README.md` | — | — | This data card (repo path: `data/README.md`) |
| `data/manifest.json` | 92K | precious | Integrity anchor: SHA-256 hashes, row counts, roles, and provenance for every file below. Also tracked in git. |
| `data/sources/` | 12G | **precious** | 33 canonical standardized source shards (jsonl). Rebuildable only via authenticated HuggingFace re-downloads (three are gated) plus the exact build pipeline — treat as irreplaceable. |
| `data/views/` | 7.8G | derived | Deterministic training/eval views: `routing/` (train/validation/dev_test/uncertain) and `injection/` (per-benchmark eval slices + direct/indirect train). Ready to train on directly. |
| `data/quarantine/` | 739M | derived | Rows excluded for conflict/leakage/sensitivity (routing, injection, mind2web + swebench sensitive). |
| `data/audits/` | 212K | derived | Overlap/near-duplicate evidence backing the quarantine decisions. |
| `data-archive/` | 4.7M | **precious** | Retained model-generated matched pairs (11,046 pairs; generation was stochastic, not reproducible). See its own README + SHA256SUMS. |
| `artifacts/models/` | 1.3G | precious | Trained model weights (safetensors/onnx/adapters) for the mmbert LoRA/frozen/LP-FT runs. Also in git-LFS. |

Everything marked *derived* is rebuilt deterministically from `data/sources/` + the repo pipeline via `uv run morgott data`; hashes are pinned in `data/manifest.json`, and the build fails closed on any mismatch.

**Deliberately not uploaded:** `artifacts/combined_generic/` feature caches (~19G) — pure recompute speed-ups, regenerated on demand.

## Source shards (`data/sources/`)

Licenses and roles below come from `data/manifest.json`. Row counts and per-shard SHA-256 hashes live only in the manifest — they change with rebuilds, so they are deliberately not copied here.

| Shard | License | Use |
|-------|---------|-----|
| `agentic_boundary_pairs` | CC-BY-4.0 | auxiliary paired instruction-subversion training and authorization diagnostics |
| `banking77` | CC-BY-4.0 | English online-banking intent queries as finance hard negatives |
| `bipia` | MIT attacks; mixed benchmark context licenses | channel-specific indirect-injection train/test with clean-context controls |
| `browsesafe` | MIT | whole-document browser injection train and official test |
| `coconot` | ODC-BY-1.0 + component licenses | safe-to-comply prompts for weak development and hard-benign evaluation |
| `do_not_answer` | CC-BY-NC-SA-4.0 | held-out harmful-goal non-injection negatives |
| `false_reject` | CC-BY-NC-4.0 | hard-benign prompts; generated candidates and human test held out |
| `financebench` | no explicit dataset license declared; public open-source sample | 150 public finance-QA examples as development-only hard-benign diagnostics |
| `gandalf` | MIT | human direct-injection data; official train only for fitting |
| `hackaprompt` | MIT | **gated** human direct attack attempts; user_input only |
| `harmbench` | MIT | held-out harmful-goal non-injection negatives |
| `harper_valley_bank` | CC-BY-4.0 | simulated human-human banking calls; caller and agent channels retained separately |
| `jailbreaks_over_time` | MIT | source-held-out temporal distribution-shift evaluation only |
| `jbb_benign` | MIT | curated benign behaviors thematically matched to misuse requests |
| `llmail` | MIT | human adaptive email injection; phase 1 fit, phase 2 evaluation |
| `lmsys_arena` | CC-BY-4.0 prompts; CC-BY-NC-4.0 model outputs | English Arena messages with weak-benign candidates and flagged conversations retained as uncertain |
| `massive_en` | CC-BY-4.0 | English voice-assistant utterances for benign intent coverage |
| `mind2web` | CC-BY-4.0 | confirmed official training tasks only, after local secret and PII quarantine |
| `multi_turn` | MIT | out-of-source obfuscated-jailbreak test grouped by goal |
| `nemotron_agentic_ipi` | CC-BY-4.0 | synthetic successful agentic indirect-injection evaluation only; never direct-user training |
| `notinject` | MIT | locked trigger-word hard negatives for measuring over-defense |
| `oasst1` | Apache-2.0 | multilingual weak injection controls; auxiliary for broad routing |
| `prompt_injections` | Apache-2.0 | train and same-source test; direct prompt-injection label |
| `schema_guided_dialogue` | CC-BY-SA-4.0 | English crowdworker task-dialogue turns for benign routing balance |
| `swebench_verified` | no dataset-level license declared; upstream repositories vary | human-verified software issue statements as a dev-test-only long-benign FPR slice |
| `taskmaster` | CC-BY-4.0 | English task-oriented user and assistant turns for benign routing balance |
| `tatqa` | CC-BY-4.0 | finance-QA questions and report contexts; task construction is not safety adjudication |
| `tensor_trust` | public research release; no explicit standard dataset license | human prompt-injection robustness evaluation only; never training |
| `tensor_trust_raw` | no standard dataset license declared | human game attack attempts; grouped development data |
| `toxic_chat` | CC-BY-NC-4.0 | train and same-source test; explicit jailbreak label |
| `wildguardmix` | ODC-BY | **gated** prompt harmfulness data for the routing target |
| `wildjailbreak` | ODC-BY | **gated** four-way harmful/benign and adversarial contrast data |
| `xstest` | CC-BY-4.0 | hard-negative test; safe and unsafe requests are not attacks |

Gated shards (`hackaprompt`, `wildguardmix`, `wildjailbreak`) require an authenticated HF token to re-download — another reason this container, not HuggingFace, is the recovery path.

## Verifying integrity

Every local corpus file's SHA-256 is pinned in `data/manifest.json` — sources, views, quarantine, and audits alike. After any pull, verify all of them:

```bash
jq -r '(.source_outputs, .routing_views, .injection_views, .quarantines, .audits)[]
       | select(has("path") and has("sha256"))
       | "\(.sha256)  data/\(.path)"' data/manifest.json | sha256sum -c
```

(Don't walk the whole manifest recursively — `.sources` also records *upstream* HuggingFace file hashes under bare paths like `train.jsonl` that don't exist locally.)

## Sync (any machine)

```bash
# one-time: install azcopy, then either `az login` or export MORGOTT_SAS_URL
scripts/azsync.sh pull            # Azure -> local (add/update only)
scripts/azsync.sh pull --mirror   # exact mirror for fresh machines (deletes local strays)
scripts/azsync.sh push            # local -> Azure after any data change
```

The rule: **change data → `uv run morgott data` (republish manifest) → `scripts/azsync.sh push`.**
