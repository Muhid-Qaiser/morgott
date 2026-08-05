# Stateful benchmark release update

Date: 2026-08-04.

Research cutoff: 2026-08-04.

## Decision

NetInjectBench still has no public, author-linked, versioned release of its scenarios, runner, evaluator, or result artifacts.
The official arXiv record remains v1 from 2026-07-11, its code and data panel contains no artifact link, and the paper still says that the complete release will occur only after acceptance ([arXiv record](https://arxiv.org/abs/2607.10490v1), [reproducibility statement](https://arxiv.org/html/2607.10490v1#S7)).
The current arXiv source archive has SHA-256 `07813af29193a46c16bddca5b6473a6b4eae1657efeec60a0ff41cb9263420ec` and contains only manuscript sources and figures, not benchmark code, scenario JSON, evaluators, or result records ([source archive](https://export.arxiv.org/e-print/2607.10490v1)).
An exact-name GitHub repository search also returns no NetInjectBench repository, although that discovery check cannot rule out a private or differently named artifact ([GitHub repository search](https://github.com/search?q=NetInjectBench&type=repositories)).

No newly reviewed public release satisfies all nine conditions in [Morgott's exact-state evaluation gate](agent-security-benchmark-options.md#what-the-next-evaluation-must-measure).
The strongest newly versioned component is the independent [AgentSecBench v1.0.0 release](https://github.com/zheyuanhu2-sketch/agentsecbench/releases/tag/v1.0.0), which supplies runtime-owned provenance, capability and approval metadata, typed actions, and pre-execution policy mediation, but it does not supply clean and attacked matched fixtures or a complete-state-diff oracle.
The strongest newly found matched-condition pattern is the [coding-agent injection benchmark at `5c5e392c`](https://github.com/sebastianripa/coding-agent-injection-benchmark/tree/5c5e392c904430f90bb8bdf2884e5bb2a325374c), but it has no trusted authorization sidecar or reference monitor and does not reject every state mutation outside an exact allowlist.

Keep [Agent-Diff at `3bb9c407`](https://github.com/agent-diff-bench/agent-diff/tree/3bb9c40707df23d89e5dbc0e40c424ba38c69ff8) as the exact-state substrate.
Borrow the narrow action-boundary ideas from AgentSecBench and the matched-condition discipline from the coding-agent benchmark, but do not replace Agent-Diff with either release.
Do not run a provider, score Morgott's detector, or locally reconstruct NetInjectBench from prose.

## Scope and method

This update checked the official NetInjectBench arXiv record and source archive, exact-name public repository and dataset surfaces, and public security-benchmark repositories appearing after the prior 2026-07-14 availability check.
Candidate claims were checked against pinned source trees rather than project descriptions alone.
The AgentSecBench v1.0.0 tag resolves to commit `a125aa446854a12d2741d1c7625bc91887d82955`, and a local lockfile install followed by its complete test command passed 148 tests at that commit ([release](https://github.com/zheyuanhu2-sketch/agentsecbench/releases/tag/v1.0.0), [release source](https://github.com/zheyuanhu2-sketch/agentsecbench/tree/a125aa446854a12d2741d1c7625bc91887d82955)).
No provider request, secret inspection, corpus-text upload, or maintained-code change was part of this research.

## NetInjectBench release status

The official record exposes only arXiv v1 and lists no later submission version as of the cutoff ([submission history](https://arxiv.org/abs/2607.10490v1)).
Section 7 says the authors prepared code, 130 scenarios, prompts, result files, evaluation scripts, tables, and documentation, but it provides no repository URL and promises publication only after acceptance ([Section 7](https://arxiv.org/html/2607.10490v1#S7)).
The manuscript therefore describes a planned artifact rather than an independently runnable or pinnable release.
The availability stop in the prior report remains correct.

## NetInjectBench against the nine conditions

Because no executable artifact is public, none of the following design properties can be independently verified in released code.
The statuses below are a paper-only projection, not release validation.

1. **Exact clean and attacked fixture identity is not met.**
The paper defines separate benign, weak-attack, strong-attack, and approved-change datasets, and it says only that weak attacks reuse similar operational settings rather than declaring exact pairs with identical IDs, ownership, permissions, relationships, prompts, and legitimate outcomes ([dataset schema](https://arxiv.org/html/2607.10490v1#S3.SS2)).

2. **A one-field attack overlay is only partially aligned.**
The paper requires attack text to reside in the artifact rather than the operator task, but it does not specify that an attacked case differs from its clean twin only in one allowlisted field on the same identified record ([scenario-construction rules](https://arxiv.org/html/2607.10490v1#S3.SS2)).

3. **Trusted runtime metadata is only partially aligned.**
The proposed schema separates approval status, maintenance window, approved tool, device, patch, and change-request identifier from artifact text, but it does not provide Morgott's complete trusted record-origin, sensitivity, allowed-sink, and approval sidecar ([field groups](https://arxiv.org/html/2607.10490v1#S3.SS2)).

4. **Complete pre-mutation HTTP mediation is not met.**
The paper describes one structured proposed call whose arguments are checked by a deterministic metadata gate, but all six tools are mocks and there is no replica, HTTP mutation surface, or released proof that every state-changing request crosses one interception point ([gate checks](https://arxiv.org/html/2607.10490v1#S3.SS3), [experimental protocol](https://arxiv.org/html/2607.10490v1#S3.SS5)).

5. **A complete-state-diff oracle is not met.**
The evaluator labels the selected tool, arguments, defense decision, final action, safety, and usefulness, while the mock environment performs no real state transition whose complete insert, update, and delete diff can be compared with exact allowed and attacker-goal diffs ([metrics](https://arxiv.org/html/2607.10490v1#S3.SS4), [mock-tool protocol](https://arxiv.org/html/2607.10490v1#S3.SS5)).

6. **Identical deterministic utility assertions are not met.**
Benign and attack cases are separate scenario groups, utility deliberately permits several safe first actions, and invalid outputs are excluded from completed-case safety and utility denominators rather than evaluated by one exact paired-state assertion ([utility definition](https://arxiv.org/html/2607.10490v1#S3.SS4)).

7. **Advisory learned output aligns in the paper but is implementation-unverified.**
The model proposes a call and a deterministic layer decides whether trusted metadata authorizes execution, which matches Morgott's authority direction, but no released code proves the boundary ([threat model](https://arxiv.org/html/2607.10490v1#S3.SS1), [policy gate](https://arxiv.org/html/2607.10490v1#S3.SS3)).

8. **A fixed first-stage attack aligns in the paper but is not runnable.**
The study uses authored weak and strong attacks and temperature-zero decoding, but the scenario records and runner are absent ([attack sets](https://arxiv.org/html/2607.10490v1#S3.SS2), [limitations](https://arxiv.org/html/2607.10490v1#S5.SS6)).

9. **A separately frozen adaptive stage is not met.**
The paper has no adaptive attacker stage and explicitly leaves multi-step tool-output poisoning for future work ([single-step scope](https://arxiv.org/html/2607.10490v1#S3.SS1)).

## Newly available public candidates

### Independent AgentSecBench v1.0.0

This project is an independent software release by Zheyuan Hu and should not be confused with the separate multi-author arXiv paper that uses the same name ([software citation](https://github.com/zheyuanhu2-sketch/agentsecbench/blob/a125aa446854a12d2741d1c7625bc91887d82955/CITATION.cff), [separate arXiv paper](https://arxiv.org/abs/2605.26269v1)).
Its stable release contains 20 normal mail and file tasks and 10 attack tasks in an offline in-memory environment ([release notes](https://github.com/zheyuanhu2-sketch/agentsecbench/blob/a125aa446854a12d2741d1c7625bc91887d82955/README.md#what-v01-measures), [catalog](https://github.com/zheyuanhu2-sketch/agentsecbench/blob/a125aa446854a12d2741d1c7625bc91887d82955/src/agentsecbench/catalog.py)).
Message and file records carry runtime trust flags, task policy carries tool, recipient-domain, writable-path, approval, and approved-action capabilities, and benchmark ground-truth labels are removed from the policy view ([models](https://github.com/zheyuanhu2-sketch/agentsecbench/blob/a125aa446854a12d2741d1c7625bc91887d82955/src/agentsecbench/models.py), [architecture](https://github.com/zheyuanhu2-sketch/agentsecbench/blob/a125aa446854a12d2741d1c7625bc91887d82955/docs/ARCHITECTURE.md#security-invariants)).
The model runner converts strict JSON decisions into one of four typed mail or file calls, binds runtime-owned action and provenance identifiers, asks the policy before execution, and mutates the in-memory environment only after an allow decision ([typed tool contract](https://github.com/zheyuanhu2-sketch/agentsecbench/blob/a125aa446854a12d2741d1c7625bc91887d82955/src/agentsecbench/model_runner.py#L30-L51), [mediation loop](https://github.com/zheyuanhu2-sketch/agentsecbench/blob/a125aa446854a12d2741d1c7625bc91887d82955/src/agentsecbench/model_runner.py#L300-L456)).

Those are useful reference-monitor contract tests, but they stop short of Morgott's evaluation target.
Normal and attack tasks have different goals and resources rather than exact clean and one-field-attacked twins ([catalog](https://github.com/zheyuanhu2-sketch/agentsecbench/blob/a125aa446854a12d2741d1c7625bc91887d82955/src/agentsecbench/catalog.py#L18-L210)).
The evaluator declares utility when every required action ID executed and attack success when any forbidden action ID executed, so unrelated extra mutations are not globally compared against a complete-state allowlist ([evaluator](https://github.com/zheyuanhu2-sketch/agentsecbench/blob/a125aa446854a12d2741d1c7625bc91887d82955/src/agentsecbench/evaluator.py#L27-L55)).
The environment is an in-memory mail and file simulation rather than an HTTP service replica with stable cross-condition database records ([tools](https://github.com/zheyuanhu2-sketch/agentsecbench/blob/a125aa446854a12d2741d1c7625bc91887d82955/src/agentsecbench/tools.py#L36-L92)).
The fixed catalog has no separately frozen adaptive second stage ([catalog](https://github.com/zheyuanhu2-sketch/agentsecbench/blob/a125aa446854a12d2741d1c7625bc91887d82955/src/agentsecbench/catalog.py#L201-L211)).
Decision: retain it as a compact implementation reference for runtime-owned provenance and pre-execution mediation, not as Morgott's next empirical substrate.

### Coding-agent injection benchmark

The pinned one-commit release runs the same coding task under a clean condition and three poisoning conditions that place a benign canary in source comments, project rules, or simulated MCP tool metadata ([README](https://github.com/sebastianripa/coding-agent-injection-benchmark/blob/5c5e392c904430f90bb8bdf2884e5bb2a325374c/README.md#the-four-conditions-variants), [variant implementation](https://github.com/sebastianripa/coding-agent-injection-benchmark/blob/5c5e392c904430f90bb8bdf2884e5bb2a325374c/bench/variants.py#L138-L162)).
This is the closest new matched-condition pattern because the pinned target, legitimate task, and grading logic stay fixed while the poisoning vector changes.
It is not an exact one-field record-pair contract because one variant prepends source text, another adds a rules file, and another changes tool metadata ([variant implementation](https://github.com/sebastianripa/coding-agent-injection-benchmark/blob/5c5e392c904430f90bb8bdf2884e5bb2a325374c/bench/variants.py#L126-L160)).
The agent loop dispatches model tool calls directly to workspace file operations without a trusted authorization policy between proposal and mutation ([agent loop](https://github.com/sebastianripa/coding-agent-injection-benchmark/blob/5c5e392c904430f90bb8bdf2884e5bb2a325374c/bench/agent.py#L79-L115), [tool dispatcher](https://github.com/sebastianripa/coding-agent-injection-benchmark/blob/5c5e392c904430f90bb8bdf2884e5bb2a325374c/bench/tools.py#L166-L186)).
Task success uses heuristic checks or hidden Docker tests and attack success scans for a planted canary outside seeded lines, but no oracle rejects every other workspace insertion, modification, or deletion ([grading](https://github.com/sebastianripa/coding-agent-injection-benchmark/blob/5c5e392c904430f90bb8bdf2884e5bb2a325374c/bench/grading.py#L59-L78), [documented metrics](https://github.com/sebastianripa/coding-agent-injection-benchmark/blob/5c5e392c904430f90bb8bdf2884e5bb2a325374c/README.md#metrics)).
Decision: use its condition-construction pattern as design input, not its runner or outcome oracle as authority evidence.

### Interbolt on AgentDojo

The pinned integration attaches a deterministic provenance policy to AgentDojo's tool executor and reports four run roles covering clean utility, attacked baseline, clean policy utility, and attacked policy behavior ([README](https://github.com/deconvolute-labs/interbolt-agentdojo/blob/e12b48e1ee2eeffe0bf687f6737e0dc34bf88fe8/README.md#the-four-runs)).
Its executor calls the Interbolt check and enforcement functions before invoking an AgentDojo function, then taints the allowed result before returning it to the model ([executor](https://github.com/deconvolute-labs/interbolt-agentdojo/blob/e12b48e1ee2eeffe0bf687f6737e0dc34bf88fe8/src/interbolt_agentdojo/executor.py#L49-L76)).
Its source trust classes are manually assigned at tool granularity from AgentDojo implementations rather than attached to stable individual records, and its outcomes remain AgentDojo utility and attack oracles plus gated-sink records ([source-classification rule](https://github.com/deconvolute-labs/interbolt-agentdojo/blob/e12b48e1ee2eeffe0bf687f6737e0dc34bf88fe8/README.md#source-trust-classification), [published protocol](https://github.com/deconvolute-labs/interbolt-agentdojo/blob/e12b48e1ee2eeffe0bf687f6737e0dc34bf88fe8/README.md#what-this-measures)).
It therefore supplies useful enforcement-seam evidence but no exact paired fixture or complete-state-diff evaluator, and it reuses AgentDojo tasks that Morgott has already investigated ([prior AgentDojo findings](agent-security-benchmark-options.md)).
Decision: do not treat this integration as a fresh exact-state substrate.

### Other close releases and scaffolds

HolyTrinity Bench is conceptually close to model-proposes and system-authorizes evaluation, but its system under test is not public, its released harness cannot compile standalone, and end-to-end reproduction must wait for that system ([reproducibility boundary](https://github.com/ScriptKittyOS/HolyTrinity-Benchmark/blob/0af6fd65e6c3e94eb77e2d436474f5e4242216e0/README.md#reproducibility--partial-stated-honestly)).
Its published authorization references contain capture timestamps rather than raw approval snapshots, so individual oracle verdicts cannot be re-adjudicated from first principles outside the unavailable system ([artifact limitation](https://github.com/ScriptKittyOS/HolyTrinity-Benchmark/blob/0af6fd65e6c3e94eb77e2d436474f5e4242216e0/artifacts/README.md#what-these-artifacts-can-and-cannot-verify)).
Decision: its architecture is relevant, but its current release cannot support an independent Morgott result.

Contract Held-out IPI defines a promising episode schema with trusted and untrusted channels, attacker goals, predicates, budgets, and train and held-out splits, but the pinned repository explicitly identifies itself as a phase-one scaffold with only two seed episodes and referee stubs ([README](https://github.com/peichengzhao/contract_heldout_ipi/blob/831e458bcf8732de8af80ab6063f4def27e764dd/README.md), [roadmap](https://github.com/peichengzhao/contract_heldout_ipi/blob/831e458bcf8732de8af80ab6063f4def27e764dd/docs/ROADMAP.md)).
Its email environment says LLM execution will be added later and currently exposes only a placeholder tool surface ([environment](https://github.com/peichengzhao/contract_heldout_ipi/blob/831e458bcf8732de8af80ab6063f4def27e764dd/src/contract_heldout_ipi/env/email_agent.py#L1-L5)).
Decision: it is not yet an executable benchmark.

[Context-to-Execution Integrity](https://arxiv.org/abs/2607.06000v1) describes typed releases, exact-effect authorization, invocation capabilities, and a deterministic pre-sink gate that are highly aligned with Morgott's boundary, but the official arXiv record exposes no code or data artifact and its source archive contains manuscript files rather than the claimed 400-episode code-agent benchmark ([paper](https://arxiv.org/html/2607.06000v1), [source archive](https://export.arxiv.org/e-print/2607.06000v1)).
Decision: treat it as paper-only design evidence, not as a released evaluation substrate.

## Strongest next substrate

Agent-Diff remains the strongest reviewed base because it combines stable seeded service objects, isolated replicas, deterministic task assertions, and complete before-and-after database diff machinery ([task extension guide](https://github.com/agent-diff-bench/agent-diff/blob/3bb9c40707df23d89e5dbc0e40c424ba38c69ff8/docs/test-suite-extension-guide.md), [diff implementation](https://github.com/agent-diff-bench/agent-diff/blob/3bb9c40707df23d89e5dbc0e40c424ba38c69ff8/backend/src/platform/evaluationEngine/differ.py)).
Its stock assertions still need Morgott's global no-extra-mutation check, trusted provenance and authority sidecar, one-field injection overlay, and typed pre-request reference-monitor adapter, as already specified in the [existing benchmark decision](agent-security-benchmark-options.md#recommended-prospective-overlay).

The best composition is therefore narrow.

1. Select a fresh, unconsumed Agent-Diff task whose clean trace must read one already identified untrusted record.
2. Freeze clean and attacked replicas that differ only in one allowlisted content field and have identical trusted IDs, ownership, memberships, permissions, relationships, user task, legitimate expected state, and assertions.
3. Keep origin, sensitivity, allowed sinks, approvals, exact permitted actions, and exact forbidden attacker actions in trusted runtime metadata keyed by stable record identity.
4. Convert every state-changing request into a typed proposed action and check it immediately before the Agent-Diff replica call.
5. Compare the complete database diff with the exact allowed and forbidden diffs in both conditions, using one deterministic assertion set.
6. Run the fixed attack before a separately frozen adaptive stage, and never let either stage tune the monitor after results are opened.

NetInjectBench should be reconsidered only when an official immutable release exposes scenario records, runners, mock tools, gate code, evaluator code, and result records that can be pinned and independently checked.
Even then, Morgott should first verify exact pairing and complete-state accounting because those two properties are absent from the current paper's declared contract.

## Claim boundary

This update establishes public artifact availability and contract fit as of the cutoff date.
It does not validate any benchmark author's empirical security claim, prove that no unindexed artifact exists, or show that any reviewed defense transfers to Morgott.
