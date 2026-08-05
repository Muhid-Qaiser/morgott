# Ponytail whole-repository audit

`delete:` The 57,405-line experiment tree is more than four times the maintained package and is dominated by completed write-once runners and version chains. After one immutable provenance commit, keep only reusable harnesses and the live showcase in the active tree; Git history and compact reports replace completed runners. [experiments/]

`done:` Five ignored Phase 3 and Phase 4 GPU queue scripts that pointed at the absent `experiments/_archived` tree were moved to recoverable desktop trash. Nothing replaces them. [artifacts/phase3_queue.sh, artifacts/phase3_extra_seeds.sh, artifacts/phase3_pairs_queue.sh, artifacts/phase4_queue.sh, artifacts/phase4_rerank.sh]

`done:` On 2026-08-05 the owner approved discarding the remaining completed one-shot runners without Git history, keeping their conclusions and hashes in the versioned reports; the discarded copies wait in the ignored `artifacts/session-scratch-2026-08-05/` until deleted. Restored exceptions stay in the tree: the report-referenced `force_bench_eval`, `agentdojo_detector_eval`, and `agentdiff_security_eval` runners, the load-bearing `agentdiff_linear_incident_eval` runner with its evidence now tracked, the sealed-panel generator `loginject_long_span_panel`, and the standing `swerebench_long_task_eval` and `cascade_mutation_asr` benchmark harnesses with their `swebench_long_benign_eval` dependency. The rejected LP-FT candidate gate freezer was never run and was removed. [experiments/, artifacts/session-scratch-2026-08-05/]

`stdlib:` Eight custom chunked file-hash loops repeat functionality in Python 3.12. Use `hashlib.file_digest` when the bound frozen sources can next change. [src/morgott/data.py, src/morgott/models/mmbert/core.py, experiments/]

`done:` The unused `OverlapGuard.add_exact`, `checkpoint_rows`, and `is_checkpoint_group` helpers are absent from the maintained module. Nothing replaces them. [src/morgott/models/mmbert/data.py]

`yagni:` The root `showcase` extra adds five dependencies used by one experiment and no package code. Move those dependencies to an experiment-local environment or PEP 723 script metadata when the showcase is next revised. [pyproject.toml, experiments/cascade_showcase/]

net: up to -56,000 lines and -5 root dependencies possible after frozen provenance is archived.
