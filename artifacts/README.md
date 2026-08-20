# Artifacts retention policy

Everything under `artifacts/` falls into one of four classes:

1. **Registered models** (`models/`): reviewed research weights in git and
   LFS, mirrored to Azure Blob by `scripts/azsync.sh` (its `PREFIXES` sync
   only `artifacts/models`). `model-artifacts.json` is the sole registry for
   what maintained inference may load; everything else in `models/` is
   retained provenance.
2. **Tracked evaluation evidence**: the directories in the table below.
   Evidence is committed once with `git add -f` (contents of `artifacts/` are
   gitignored by default) and never rewritten afterwards; corrections get new
   dated files. These directories exist only in git and on this disk, not in
   Azure.
3. **Superseded or invalidated archives** (`superseded-experiments/`,
   `invalidated-experiments/`, `phase3_archived/`, `research-source-archive/`,
   session-scratch quarantines): local, untracked, kept because reports or
   audits cite them; delete only with an owner decision recorded in a
   versioned report.
4. **Rebuildable local caches** (`combined_generic/` feature caches,
   `mmbert/data*`, `mmbert/runs`, raw `results.jsonl` beside tracked
   `results.jsonl.gz`): pure recompute speed-ups or raw duplicates, safe to
   regenerate, never uploaded.

## Tracked evidence directories

Origin notes bind each evidence directory to what produced it. "Committed
2026-08-05" entries came in the evidence-consolidation commit of that date;
runners marked hash-pinned were discarded under the recorded 2026-08-05 owner
decision with their SHA-256 kept in the directory manifest.

| Directory | Owning report | Origin |
|---|---|---|
| `agentdiff_linear_incident_eval` | `ponytail-audit-2026-08-04.md` | Produced by experiments/agentdiff_linear_incident_eval/run.py as a prospectively frozen coupled-prefix Agent-Diff Linear incident containment case (committed 2026-08-05, 5269fe4); all frozen gates passed with decision retain_bounded_coupled_linear_incident_containment, also discussed narratively in reports/agent-security-benchmark-options.md. |
| `agentdiff_reaction_eval` | none (see origin) | Produced by experiments/agentdiff_security_eval/reaction.py, a prospectively frozen exact-argument Agent-Diff Slack reaction overlay run with openai/gpt-4.1-mini via OpenRouter (committed 2026-08-05); the monitor prevented the attack effect and all unauthorized mutations but broke exact authorized state, so the summary decision is reject_exact_reaction_pattern. |
| `agentdiff_reaction_recovery_eval` | none (see origin) | Produced by experiments/agentdiff_security_eval/reaction_recovery.py as the trusted denial-recovery follow-up to the reaction overlay (committed 2026-08-05); with a denied-tool-result recovery message the monitor run reached exact authorized state with zero unauthorized mutations, decision retain_trusted_denial_recovery_as_bounded_evidence. |
| `agentdojo_detector_eval_channel_low` | `agentdojo-integration-research.md` | Produced by experiments/agentdojo_detector_eval/run.py as a frozen external AgentDojo Banking (v1.2.2) advisory-cascade diagnostic in its channel-low variant (committed 2026-08-05), pinning the cascade, threshold, and analysis contracts by hash. |
| `agentdojo_live_eval` | none (see origin) | Already-public AgentDojo v1.2.2 banking-suite live agent run (openai/gpt-4.1-mini via OpenRouter) from an uncommitted one-shot runner pinned only as runner_sha256 in the manifest (committed 2026-08-05); with the reference monitor it shows 0 strict attack effects and 0 unauthorized-mutation cases versus 72 unauthorized-mutation cases without it. |
| `agentdojo_live_eval_safe_reads` | none (see origin) | Variant of the AgentDojo banking live eval (same hash-pinned uncommitted runner, committed 2026-08-05) that adds a scoped safe-reads policy against the pinned baseline; the monitor still yields 0 strict attack effects and 0 unauthorized mutations while recovering user utility (54 vs 49 exact-authorized-state attack runs). |
| `agentdojo_live_eval_sensor_warning` | none (see origin) | Banking-suite live-eval variant (same hash-pinned uncommitted runner, committed 2026-08-05) testing a sensor-warning condition; the no-monitor arm records 1 strict attack effect and 73 unauthorized-mutation cases while the monitor arm records 0 of each. |
| `agentdojo_live_eval_slack_exact` | none (see origin) | AgentDojo v1.2.2 Slack-suite live eval with exact pinned Slack user-task call-multiset authority (hash-pinned uncommitted runner, committed 2026-08-05); only manifest.json and results.jsonl.gz are tracked, with no summary. |
| `agentdojo_live_eval_slack_safe_reads` | none (see origin) | Slack-suite live-eval variant (hash-pinned uncommitted runner, committed 2026-08-05) using exact pinned calls plus scoped reads; monitor arm shows 0 strict attack effects and 0 unauthorized-mutation cases versus 49 unauthorized-mutation cases without the monitor over 67 attack runs. |
| `agentdojo_live_eval_slack_stable` | none (see origin) | Post-preflight stable-oracle Slack development diagnostic (hash-pinned uncommitted runner, committed 2026-08-05) restricted to cases where injection leaves the clean user and injection call multisets byte-identical; monitor arm shows 0 strict attack effects versus 2 (and 46 unauthorized-mutation cases) without it. |
| `agentdojo_policy_eval` | `agentdojo-integration-research.md` | Deterministic compromised-planner action replay with oracle task capabilities over AgentDojo declared ground truth (single result.json.gz; runner pinned by sha256, not retained in experiments/), committed 2026-08-05. |
| `aspi_clarification_local_diagnostic` | none (see origin) | Post-hoc local ASPI clarification diagnostic over scaleapi/aspi data reusing the hash-pinned experiments/injecagent_detector_eval/run.py implementation contract (committed 2026-08-05); overall strict ordering rate 0.816 with decision reject_response_only_clarification_routing. |
| `cascade_mutation_asr` | `ponytail-audit-2026-08-04.md` | Produced by the standing experiments/cascade_mutation_asr/run.py harness as the development-only multi-attempt mutation ASR for the selected advisory cascade (committed 2026-08-05); ASR rises from 0.87% at 1 attempt to 2.79% at 25 attempts on the retained full-LoRA fixed mutation population. |
| `combined_generic` | `lfm25-viability-research.md` | LFM2.5 encoder 230M frozen-backbone full-mixture run recorded 2026-07-30 (commit 3dacdba, Record LFM2.5 frozen-head evaluation); the tracked evaluation_generic_v3/evaluation.json is a fail-closed evaluation of a generic instruction-subversion head over the morgott/PromptShield/matched-pairs mixture (its runner, experiments/encoder_finetune/run.py, was later deleted). |
| `comparisons` | `model-experiments.md` | Baseline and guard comparison evaluations written by experiments/guard_baselines/run.py and experiments/evaluate_prompt_guard_2_full_mixture.py (first recorded 2026-07-30, extended 2026-08-05), holding the Prompt Guard 2 86M full-mixture comparison, the post-hoc channel-low review-floor ablation, and DeepSeek V4 Flash encoder-refresh and finance-full-LoRA comparisons. |
| `financial_ctf_eval` | none (see origin) | Frozen external financial CTF advisory-cascade diagnostic over verno-labs/financial-ai-ctf-dataset (400 conversations, 27.5% restricted overall; committed 2026-08-05); its one-shot runner was discarded under the 2026-08-05 owner decision, with only analysis/cascade contract hashes pinned in the manifest, and the substantive decision about this data is discussed in reports/finance-web3-benchmark-audit.md without naming the directory. |
| `force_bench_eval` | `finance-web3-benchmark-audit.md` | Produced by experiments/force_bench_eval/run.py as a prospectively frozen external benign-finance cascade diagnostic on the FORCE benchmark (committed 2026-08-05). |
| `injecagent_detector_eval` | `injecagent-detector-evaluation.md` | Fixed-cascade transfer diagnostic on sanitized InjecAgent tool outputs produced by experiments/injecagent_detector_eval/run.py (hash-pinned in the manifest; runner since removed from the tree), committed 2026-08-05. |
| `injecagent_task_reviewer_eval` | `injecagent-detector-evaluation.md` | Fixed task-conditioned 0731 reviewer transfer on InjecAgent base tool responses produced by experiments/injecagent_task_reviewer_eval/run.py (hash-pinned; runner since removed), committed 2026-08-05. |
| `mmbert` | `model-experiments.md` | Single tracked file full-lora-runtime-benchmark.json, the machine-readable runtime preflight for the full-mixture mmBERT LoRA recipe recorded 2026-07-30 (commit bca1b34, Record full-mixture mmBERT LoRA results #9) and linked by SHA-256 from reports/model-experiments.md. |
| `mmbert_lpft_comparison` | none (see origin) | Held-out summary from the experiments/mmbert_lpft_comparison study (committed 2026-08-05) comparing the mmbert-base-full-lpft-s42 candidate on the frozen new-data validation and dev-test pair splits; the LP-FT candidate was rejected, and the ledger notes 2,267 of 2,590 dev-test attack spans also occur in training so results are in-family template evidence. |
| `mmbert_lpft_new_data` | none (see origin) | Frozen new-data pair archives built by experiments/mmbert_lpft_comparison/prepare_new_pairs.py (problem_statement plus one deterministic known-span injected twin per row, committed 2026-08-05), with metadata-correction.json recording an off-by-one fix to the recorded attack_span_start that leaves the frozen archives unrewritten. |
| `openrouter_downstream_eval` | `openrouter-downstream-evaluation.md` | Produced by experiments/openrouter_downstream_eval/run.py (with followup.py and analyze.py) as the bounded text-only OpenRouter downstream development evaluation, first recorded 2026-07-29 and extended through the DeepSeek 0731 follow-up. |
| `statebench_coupled_exchange_eval` | none (see origin) | Produced by experiments/statebench_coupled_exchange_eval/run.py as a prospectively frozen coupled-prefix STATE-Bench exchange containment case (committed 2026-08-05); the tracked summary records decision incomplete_coupled_run with a 3-row result ledger and no gates evaluated. |
| `swechat_traffic_eval` | none (see origin) | Prospectively frozen traffic-like direct-user local cascade diagnostic over SWE-chat first prompts (5,794 source sessions) produced by experiments/swechat_traffic_eval/run.py (hash-pinned in the manifest; runner since removed), committed 2026-08-05; per-agent local-high flag rates are about 1.2% for Claude Code traffic. |

`models/` and `pipeline_benchmark/` are governed by `model-artifacts.json`
and the pipeline benchmark report respectively and are not listed above.
