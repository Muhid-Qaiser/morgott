# Ponytail whole-repository audit

`delete:` The 57,405-line experiment tree is more than four times the maintained package and is dominated by completed write-once runners and version chains. After immutable provenance is archived, keep only reusable harnesses and live studies in the active tree; Git history and compact reports replace completed runners. [experiments/]

`done:` Five ignored Phase 3 and Phase 4 GPU queue scripts that pointed at the absent `experiments/_archived` tree were moved to recoverable desktop trash. Nothing replaces them. [artifacts/phase3_queue.sh, artifacts/phase3_extra_seeds.sh, artifacts/phase3_pairs_queue.sh, artifacts/phase4_queue.sh, artifacts/phase4_rerank.sh]

`done:` On 2026-08-05 the owner approved discarding the remaining completed one-shot runners without Git history, keeping their conclusions and hashes in the versioned reports; the discarded copies wait in the ignored `artifacts/session-scratch-2026-08-05/` until deleted. Restored exceptions stay in the tree: the report-referenced `force_bench_eval`, `agentdojo_detector_eval`, and `agentdiff_security_eval` runners, the load-bearing `agentdiff_linear_incident_eval` runner with its evidence now tracked, the sealed-panel generator `loginject_long_span_panel`, and the standing `swerebench_long_task_eval` and `cascade_mutation_asr` benchmark harnesses with their `swebench_long_benign_eval` dependency. The rejected LP-FT candidate gate freezer was never run and was removed. [experiments/, artifacts/session-scratch-2026-08-05/]

`stdlib:` The remaining custom chunked whole-file hash loops repeat functionality in Python 3.12. Use `hashlib.file_digest` when the bound frozen sources can next change. [src/morgott/data.py, src/morgott/models/mmbert/core.py, experiments/]

`done:` The unused `OverlapGuard.add_exact`, `checkpoint_rows`, and `is_checkpoint_group` helpers are absent from the maintained module. Nothing replaces them. [src/morgott/models/mmbert/data.py]

`yagni:` The root `showcase` extra adds five dependencies used by one experiment and no package code. Move those dependencies to an experiment-local environment or PEP 723 script metadata when the showcase is next revised. [pyproject.toml, experiments/cascade_showcase/]

## 2026-08-12 mmBERT campaign review

`done:` The completed campaign's exact source was preserved before surgery. `reports/provenance/mmbert-context-campaign-source-20260812.tar.gz` has SHA-256 `7326148fd92f2486afb908ae73f90c2ecb212d0c6bd68f8ef06fd0d6494dca11`; its internal manifest binds the trainer, evaluator, data loader, head contract, score journal, reusable reserve/long-code/guard harnesses, lock file, and launchers used for the 1,024-token campaign. The context-comparison manifest and model decision ledger retain result identities and conclusions. [reports/provenance/, reports/mmbert-context-comparison.json, reports/model-experiments.md]

`done:` The reusable guard-baseline adapter/journal harnesses, snapshot-aware reserve and long-code evaluators, focused tests, `TrainingData` refactor, explicit head contract, snapshot/cap-aware evaluation, and score journal remain in the maintained tree. [experiments/guard_baselines/, experiments/mmbert_redteam_snapshot_eval/, experiments/mmbert_longcode_snapshot_eval/, src/morgott/models/mmbert/, tests/]

`done:` The completed capacity-ladder canaries, one-shot Trackio importer, rejected hierarchical sampler, cold-prep prototype, and one-shot tests left the maintained tree after their findings were preserved. Recoverable copies and a checksum manifest are in `artifacts/session-scratch-2026-08-12/reviewed-yagni/`; its `README.txt` explicitly marks them unmaintained. [reports/model-experiments.md, artifacts/session-scratch-2026-08-12/reviewed-yagni/]

The cold-prep prototype's retained aggregate finding is narrower than its implementation: on a 10,000-row sample, six spawned workers matched the serial bytes and improved 5.02 seconds to 3.41 seconds (1.47x), while a packed retained-ID representation was 31.2% of a conservative owner-dictionary lower bound. This was not an end-to-end cold-build result; complete physical verification, process-tree RSS, and whole-build parity remain unmeasured. The production change is therefore only the fail-closed physical-input check, not the roughly 1,900-line multiprocessing/cache-v2 design. [artifacts/session-scratch-2026-08-12/reviewed-yagni/experiments/cold_prep/]

The hierarchical sampler also carries a non-obvious objective change: unit-weight draws from its label/source/group hierarchy optimize that sampling distribution rather than the current row-population objective. Preserving the masked harmful auxiliary under draw probability `q_i` requires the explicit `class_weight[y_i] / (N * q_i)` correction. That complexity is not justified before the simpler, preregistered group-DRO experiment. [artifacts/session-scratch-2026-08-12/reviewed-yagni/src/morgott/models/mmbert/hierarchical_sampler.py]

`done:` The incomplete checkpoint-selector prototype and its one-shot test moved to the reviewed-YAGNI quarantine. The replacement design, statistical unit, global multiplicity correction, and first group-DRO experiment are preserved in `reports/checkpoint-selection-design.md`. [reports/checkpoint-selection-design.md, artifacts/session-scratch-2026-08-12/reviewed-yagni/]

`done:` The expanded runtime benchmark moved to the reviewed-YAGNI quarantine rather than remaining active. Its representative stream always length-sorts 512-token batches, so it does not reproduce the selected 1,024-token no-length-grouping workload. The RTX 6000 Ada artifact (`artifacts/mmbert/full-lora-runtime-benchmark-nvidia-rtx-6000-ada-generation.json`, SHA-256 `80f57d34bfddb26b3873d1996444243b7a56b60f472404a92f3d61a9aea182f9`) remains only a superseded 512/grouped diagnostic; its selected compile result must not be applied to the 1,024 recipe without a frozen-identity replay. [artifacts/session-scratch-2026-08-12/reviewed-yagni/]

`done:` Trackio run-name uniqueness and the canonical `morgott` project now fail closed. New runs log selector-specific active keys, directional macros, three finance-negative BCEs, diagnostics, and compact core training metrics; complete per-source-label values remain in the versioned curve artifact instead of becoming dashboard noise. Legacy curves remain untouched as history. [src/morgott/models/mmbert/train.py, tests/test_mmbert_training.py]

`done:` Prep-cache acceptance now binds the physical routing/external inputs and the exact preparation-source contract. The roughly 1,900-line cold-prep design was not retained for that bounded fix. [src/morgott/models/mmbert/train.py, tests/test_mmbert_training.py]

`done:` Score-journal publication is serialized across processes, implicit resume offsets use optimistic compare-and-swap, and full-evaluation identities bind ordered calibration, canonical dev-test, PromptShield, and SEP panels while preserving completed schema-1 compatibility. [src/morgott/models/mmbert/score_journal.py, src/morgott/models/mmbert/evaluate.py]

`done:` The Claude logbook hook and stale raw handoff were neither activated nor committed; recoverable copies are isolated in the reviewed-YAGNI quarantine. Trackio receives only scalar metrics, configuration, and curve documentation, never prompts, corpus rows, credentials, conversation traces, or workspace state. [artifacts/session-scratch-2026-08-12/reviewed-yagni/, src/morgott/models/mmbert/train.py]

`policy:` Training and evaluation need no separate owner-approval ceremony. They still must satisfy the repository's privacy, data, resource, and evidence gates; registry changes require the documented reviews and `model-artifacts.json` update. [AGENTS.md]

The 2026-08-12 pass removed several thousand unmaintained lines while retaining every result, hash, scientific limitation, and reusable evaluator needed to reproduce the decisions.

The apparent 95,523-line addition was not 95,523 lines of implementation.
It included 8,971 lines of patch backups/rejects, now removed, and 66,559
lines of checksum-bound machine evidence. One exact context-parity JSON alone
is 47,156 lines (1.95 MB). The final candidate contains 85,580 added/new text
lines: 66,559 machine evidence and 19,021 lines of implementation, tests,
documentation, scripts, and metadata. The final Ponytail pass removed 975
more lines by pruning unused training branches, genericizing bootstrap,
sharing exact evaluator-contract primitives, and deduplicating tests and
commands. Git attributes mark 65,132 exact result-record JSON lines generated
and non-diffable, leaving about 20,448 review-facing lines without weakening
Git durability or checksum coverage. The three checkpoint and two Trackio
binaries use Git LFS and become pointer files after staging.

## Architecture/KISS outcome

`done:` The bounded correctness work is in place: atomic run publication,
complete training-source identity, exact guard/evaluation identities, shared
path and evaluation-contract primitives, compact Trackio metrics, and clearer
documentation routing. These changes reuse the standard library and existing
modules; no new dependency was warranted.

`defer:` Prep extraction, deeper binding-state-machine consolidation, a public
metrics module, adapter-family splitting, result-builder extraction, and
roadmap splitting are structural refactors rather than current correctness
blockers. A dedicated follow-up can change those boundaries coherently without
churning cache keys, provenance identities, and review evidence during this
campaign cleanup.
