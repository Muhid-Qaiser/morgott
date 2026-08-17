# Morgott 1,024-context pipeline benchmark

This report-only experiment compares the registered 1,024-token detector, the advisory cascade, DeepSeek transports and providers, Prompt Guard 2, retained baselines, local CUDA, and deployed Azure.
It does not modify registered models, maintained thresholds, Azure configuration, or advisory authority.

## Post-study disposition

The benchmark itself remained report-only while evidence was collected and frozen.
On 2026-08-17, the owner separately promoted its exact balanced profile as the maintained advisory default through `model-artifacts.json`.
That promotion does not rewrite this study's ledgers, authorize blocking, or make the pre-promotion Azure and mutation runs promoted-profile evidence.

## Frozen roles

`prepare` binds the existing 20,000-row panel, its 6,000-row calibration role, its 14,000-row evaluation role, the registered model hashes, and a public provider-safe subset.
It refuses uncommitted benchmark or runtime source, so the recorded Git commit contains the runner that created a new study.
Provider prompts are sent only from that public subset.
Credentials, raw provider responses, and raw prompts are never written to benchmark artifacts.
New remote stages durably reserve a conservative retry-inclusive ceiling before reading a provider key or making a billable call.
The usable ceiling is US$24 from a US$25 ledger with US$1 reserved against overrun.
This completed study predates that durable mechanism, so its directory is closed at the usable ceiling rather than claiming that its old planning estimates prove invoiced spend.

## Run order

Run benchmark modules with both optional dependency groups because the shared runner imports cascade and encoder code.

```bash
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.run prepare
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.run score-local --model morgott
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.run score-local --model prompt-guard
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.run parity
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.run snapshot-providers
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.run run-providers --stage canary --concurrency 4
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.run run-providers --stage panel --concurrency 4
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.run provider-summary
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.provider_windows plan
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.provider_windows execute --concurrency 4
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.hard_verdict
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.logprob_exact
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.run provider-load --requests-per-cell 32
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.run run-providers --stage evaluation --concurrency 4
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.hard_verdict
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.logprob_exact
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.cascade_flow_comparison
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.run analyze
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.report
```

The source-complete hard-verdict selection has no winner: Baidu, Decart, and DeepInfra each exceeded at least one overall or declared-source recall-loss gate.
The earlier Decart evaluation remains a diagnostic and is not silently substituted into the maintained pipeline.

The retained Azure matrix was collected before durable cost reservation.
Its retry-inclusive conservative ceiling now exceeds the study budget, so do not rerun it inside this output directory.
A future Azure study must use a fresh output directory and a smaller frozen matrix that passes preflight.

Concurrency eight is accepted only when the concurrency-four terminal-failure gate passes and measured sustained throughput improves.
Provider-load samples are unique across cells and round-robin across the actual <1,024, 1,024-4,095, 4,096-15,999, and >=16,000 character bands.
The report retains the per-cell band denominators because the longest safe band is sparse and is never repeated or synthesized.
The provider panel, evaluation, load, and canary use separate append-only ledgers so each stage hash remains reproducible.
Interrupted remote stages resume from unique job IDs without repeating completed calls.
The cascade-flow comparison is offline and reuses the frozen artifact and window ledgers.
It is post-hoc on consumed evaluation evidence and does not claim coverage for routing cohorts absent from that panel.

The OpenVINO parity trigger exceeded 0.5% in this run, so the full runtime-specific replay is produced with:

```bash
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.openvino_score
```

The current mutation population is replayed without provider calls with:

```bash
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.mutations
```

The sealed LogInject panel is opened once after profile selection by supplying the hash-verified official archive root:

```bash
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.loginject /path/to/loginject/artifact
```

After the local replay, calibration-only `logprob_exact_selection.json`, and Cloudflare logprob winner are frozen, inspect and execute the text-free remote cascade replay with:

```bash
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.loginject_remote /path/to/loginject/artifact plan
uv run --locked --extra cascade --extra encoder python -m experiments.pipeline_benchmark.loginject_remote /path/to/loginject/artifact execute --concurrency 4
```

The remote stage uses only the exact maintained-semantics balanced selection and the unchanged incumbent profile.
It shares full-context and window reviews across those profiles while retaining each profile's ordered batches of four.
Concurrency 8 is only a probe: return to 4 if its terminal-failure rate exceeds 0.5%.
It writes a separate resumable ledger and never modifies the sealed source panel, local scores, or local summary.

The final report is `reports/pipeline-benchmark-20260816.md` and machine-readable tables are written under `artifacts/pipeline_benchmark/20260816/`.
The text-free parsed evidence is tracked in Git, with large ledgers, the manifest, and machine tables stored through Git LFS.
This keeps failures, timings, identities, and result denominators auditable without committing prompts, raw responses, or credentials.
`source-provenance.json` records the original manifest limitation and binds the recovered byte-identical pre-source-gate hard-verdict helper used by the frozen logprob analysis.

## File map

- Core benchmark flow: `run.py`, `local.py`, `providers.py`, `provider_windows.py`, and `metrics.py`.
- Frozen exact analysis: `logprob_exact.py`, `hard_verdict.py`, and `deepseek_standalone.py`.
- Robustness studies: `mutations.py`, `loginject.py`, and `loginject_remote.py`.
- Consumed post-hoc diagnostics: `cascade_flow_comparison.py` and the `reviewer_*` modules.
- Publication: `report.py` and its generated report and machine tables.
- Evidence controls: `budget_reservations.json` closes legacy spend ambiguity, while `source-provenance.json` binds retained source and explains the original manifest limitation.

The consumed diagnostics, failed or confounded run records, and quarantine ledgers are provenance, not maintained runtime code.
Moving these files into more subfolders would break recorded module commands and code hashes without simplifying the runtime, so the study remains one self-contained directory.

## Verification

```bash
uv run ruff format --check experiments/pipeline_benchmark
uv run ruff check experiments/pipeline_benchmark
uv run --locked --extra cascade --extra encoder python -m unittest discover -s experiments/pipeline_benchmark -p 'test_*.py'
make check
git diff --check
```
