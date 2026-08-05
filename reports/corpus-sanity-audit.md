# Corpus sanity audit

This audit covers the canonical source shards, routing views, quarantine outputs, labels, grouping, leakage controls, and current model evidence.
Exact current counts, hashes, and split distributions remain in `data/manifest.json` as the sole machine source of truth.

## Checks that passed

- Every manifest-tracked file was hash-verified and its recorded row count was recomputed.
- Canonical schemas, source revisions, downloaded-file digests, and required source contracts fail closed.
- Exact normalized text does not cross train, validation, and dev-test.
- Split groups do not cross train, validation, and dev-test.
- Exact label conflicts, strict near-overlaps, and source-level sensitive-text matches remain visible in quarantine instead of silently entering fitting.
- Official evaluation lineages remain in dev-test, and exact duplicates touching those lineages are held out with them.
- Boundary Pairs preserve official splits, pair IDs, scenario IDs, and source context.
  Every pair and scenario stays within one official role, and authorization-only families remain auxiliary diagnostics.

## Critical red flags

- Source identity predicts most of the training label.
  Most direct-user sources contain only benign rows or only review-required rows, so a model can score well by recognizing dataset style instead of intent.
- The labels do not represent production truth or production prevalence.
  They combine task construction, challenge participation, human safety annotation, model labels, and generated weak supervision.
- `review_required` deliberately mixes instruction subversion, harmful requests, toxicity, and unresolved cases.
  It is a conservative routing target, not a claim that every positive prompt is malicious or an injection.
- Text alone cannot determine authority or provenance.
  The same short instruction can be legitimate from an authorized user and hostile when embedded in retrieved content, tool output, email, or memory.
- SWE-bench Verified now supplies one official-test, dev-test-only long-benign direct-user denominator, but it has no same-format attack arm and is concentrated in a small set of repositories.
  Human solvability review is not a safety annotation, so this slice can measure repository- and length-specific benign review load but cannot establish balanced long-context robustness or production FPR.
  Its first frozen local evaluation rejects the registered high gate: 1.42% of legitimate tasks were hard-restricted before review and 69.11% more required review.
- Exact deduplication removes repeated text, but many source families still contain templated, paraphrased, adaptive, or derived examples.
  SimHash quarantine is conservative and cannot prove semantic independence.
- Development evaluation has already influenced source and model decisions.
  The current dev-test is useful for repeated diagnosis but is not a prospective final test.
- Multilingual, multi-turn, memory, tool-output, clarification-state, and action-level coverage remains too thin for broad product claims.
- The completed multilingual Agentic Boundary Pairs sequence consumes every scenario in that synthetic source.
  Its source-specific word model and broad reviewer pass the last frozen block, but cannot establish cross-source transfer or a representative false-positive rate.
- The same word model fails its first independent hard-benign transfer gate on the public Operant AI false-positive controls.
  It admits 7 of 49 English rows at the frozen threshold, so its apparent zero false positives on Agentic Boundary Pairs do not transfer.
- The registered text-only local gate fails the frozen ACL known-span long-document diagnostic.
  It admits only 65.10% of attacked documents and 61.80% of exact payload spans, while 52.14% of clean TriviaQA documents enter review.
  Lower thresholds and 128-token windows cannot repair the attack miss without overwhelming the benign path.
- Adding the trusted source question gives the DeepSeek reviewer perfect attack-over-clean ordering on a 160-row consumed-family canary, but its registered `0.85` operating point still misses the SQuAD head and TriviaQA head and midpoint recall gates.
  The post-hoc score scale is not transferable evidence, all rows required remote review, and the remaining 280 ACL families were deliberately left unopened.
- Independent StruQ calibration and evaluation then pass at `0.85`, with all 471 untouched attacks above threshold and above their paired clean input while 1 of 157 clean inputs is flagged.
  This is fixed-target synthetic evidence with a 100% remote-call rate, not proof of transfer to varied attack goals, natural documents, or representative traffic.
- The fixed InjecAgent transfer varies 62 attacker goals across 17 tool tasks and exposes a response-only gap.
  The unchanged cascade restricts 70.40% of base attacks, while task-conditioned 0731 review at the transferred `0.85` threshold reaches 78.27% with zero of 17 clean flags and perfect paired ordering.
  It still fails aggregate and worst-slice gates, and the promising lower post-hoc threshold cannot be selected from this consumed source or validated against only 17 clean templates.
- API-Bank demonstrates that a benchmark projection can dominate reviewer false positives.
  Treating trusted API call syntax and arguments as untrusted content flagged 20.31% of exact-unique clean reviews at `0.3`, while output-only projection reduced the rate to 3.68% but still failed the frozen gates.
  A post-hoc `0.5` point passes the consumed API-Bank clean gates and retains 94.88% InjecAgent recall, but its weakest InjecAgent attacker goal reaches only 47.06%.
- The frozen AgentDyn transfer flags all 560 explicit task-and-goal attacks at `0.5` after operationally completing rate-limited calls.
  Its fixed attack-only template has no clean or adaptive arm and does not execute tools, so it cannot validate the post-hoc clean selection or authorize integration.
- Regex-based privacy screening is only a quarantine aid.
  Passing it does not prove that public issue or agent-task text contains no personal or sensitive information.
- Corrected source-heldout experiments did not find an architecture fix for the data problem.
  Full tuning overfit source artifacts, the frozen multipool head retained double-digit macro-source false-positive rates, OR-Bench reduced recall, and paired splices did not improve ranking materially.
- The masked multitask frozen encoder also failed the quick promotion gates.
  Its winning auxiliary-BCE recipe escalated 59.80% of legitimate held-out finance rows and every benign held-out BrowseSafe document at validation-selected operating points.
  Pair ranking improved matched Boundary ordering without repairing either false-positive failure.
- Global label-support counts overstate evaluability when a held-out source lacks explicit matching-axis negatives.
  Masked per-head metrics can be undefined even while the derived route imposes a severe benign review load.
- The Rogue Security benchmark is materially contaminated by current public-source families.
  An exact normalized audit matched 54.28% of its rows to the canonical corpus, so it cannot serve as an independent headline evaluation.

## Normalisation and leakage defects (measured 2026-07-27)

`normalize_text` is NFKC, case folding and whitespace collapse. Tested against
thirteen known evasion techniques as a pure string transform, it collapses
**four**; a stricter normaliser adding invisible-codepoint stripping, homoglyph
folding, combining-mark removal and repeat capping collapses **thirteen**.

The leakage check inherits the gap. Over a 360,000-row sample drawn evenly from
routing train, validation and dev-test:

- 721 texts (0.200%) collapse under the stricter normaliser but not the current
  one.
- **Twelve groups span more than one split** and are invisible to the current
  exact-hash check. One HackAPrompt payload appears in train, validation and
  dev-test simultaneously, in three different obfuscations.

Small in aggregate, but these are obfuscated adversarial payloads: precisely the
population whose held-out status the robustness numbers depend on. Reported
recall on obfuscated attacks is therefore optimistic by an unknown amount.

A follow-up audit on 2026-08-01 found that the registered strict normalizer still preserves U+034F and supplementary variation selectors U+E0100 through U+E01EF.
The retained artifacts bind that normalizer by hash, so this needs a prospectively evaluated model-input revision rather than an inference-only patch.

On 2026-08-02 the future model-preparation overlap guard added a separate audit-only fingerprint that also removes those code points.
The full pinned preflight then exposed one same-label intra-training HackAPrompt duplicate and removed it from the prospective canonical fit population.
A targeted metadata-only scan found no new match from the affected eligible training rows into canonical validation, canonical dev-test, PromptShield, or SEP.
The exact populations for the historical retained weights and prospective preparation remain in their immutable model and preflight artifacts.
The maintained preflight now reuses reference overlap values instead of rebuilding the same guard.
On the same warm-cache host, replacing the audit-only invisible and homoglyph Python loops with an equivalent `str.translate` fold reduced full-preflight wall time from 16:10.33 to 14:14.14, or 12.0%, while peak RSS remained 2.53 GiB and the complete report stayed identical.

On 2026-08-04 a full profile of `uv run morgott data --routing-only` found candidate Hamming checks in the eight-band SimHash index to be the dominant remaining routing-publication cost.
An exact lower bound now skips candidates sharing fewer than two complete bands because a fingerprint within Hamming distance six across eight bands must share at least two unchanged bands.
The uninstrumented routing-only rebuild fell from 1,012.06 to 881.98 seconds on the same host, a further 12.9% reduction, while the manifest and every tracked output remained hash-identical.
A post-pruning traced rebuild then localized 317.83 cumulative seconds to SimHash construction and 95.23 seconds to near-index queries, confirming that the exact band-count bound had removed query comparison as the dominant cost.
A 10,000-entry feature-digest cache improved a 28,739-row all-source fingerprint sample by 20.7% with exact outputs, but the complete isolated routing-only build changed only from 881.98 to 876.60 seconds, or 0.61%, despite warm caches.
The candidate manifest remained byte-identical and the cache was reverted as whole-pipeline timing noise.
Exact-unique supervised rows now reuse their already-compressed canonical payload instead of repeating one decode, JSON serialization, and compression cycle; duplicate and conflict groups retain the existing merge path.
The corrected isolated full rebuild fell from 881.98 to 754.29 seconds, or 14.48%, while peak RSS fell from 2,026,340 to 2,006,728 KiB.
The manifest and every published routing and quarantine output remained byte-identical.
A post-fast-path profile confirmed that exact-group preparation fell from 291.54 to 144.61 cumulative seconds and that final supervised publication now dominates at 604.56 seconds, including 307.10 seconds in SimHash construction.
Encoding word features once as bytes preserved every sampled fingerprint but improved the 28,739-row microbenchmark by only about 1% to 2%, so it was rejected before a full build.
The same profile found 12.59 million JSON serializations used only to compare seven schema-validated annotation fields during exact deduplication.
Direct structural comparison was exact on 50,000 real merged groups and preserved linear behavior for the corpus group containing 13,221 origins.
The full isolated build changed from 754.29 to 747.30 seconds, or 0.93%, which remains within observed timing variance; the change is retained only as simpler code, not as a performance claim.
The manifest and every routing and quarantine output remained byte-identical.
A stateless duplicate-feature aggregation preserved sampled and full outputs but changed the full rebuild by only 1.11 seconds, so it was reverted as timing noise.
Removing level-1 compression from temporary SQLite payloads was also rejected after the build reached 19:06 without completing and used 27 GiB of scratch space; interruption cleanup preserved the published manifest and removed the isolated build directory.
The canonical source writer now hashes each atomically published JSONL shard with the standard streaming file-digest primitive instead of allocating a second in-memory byte buffer as large as the shard; output bytes and digest semantics are unchanged.

On 2026-08-04 a complete read-only audit of the then-current routing views confirmed that the audit-strict hash still crossed development roles outside the future-model preflight.
It found collisions from train and validation into dev-test and from validation into train.
The routing writer now checks the existing audit-only `leakage_text_hash` before SimHash, gives dev-test priority over train and train priority over validation, and records `strict_dev_test_overlap` or `strict_train_overlap` in quarantine.
The rebuilt corpus quarantined the affected train and validation rows; the strict pass removed nothing from dev-test and left the uncertain view unchanged, though the same rebuild also grew the published dev-test view through the new SWE-bench Verified rows and lineage repartitioning.
Some strict matches had already been caught by SimHash, so the strict-reason population rose while the two near-overlap reason populations fell.
The strict quarantine is predominantly review-required and is dominated by Tensor Trust raw and HackAPrompt, which is consistent with the known adversarial-obfuscation failure rather than broad ordinary-text removal.
Some benign rows are also excluded, and every affected row lacks usable language metadata, so this result cannot establish that the conservative fold has no multilingual cost.
An independent full readback verified every routing and quarantine digest and row count and found zero normalized-text, audit-strict-text, or lineage-group crossings between train, validation, and dev-test.
The full routing-only rebuild took 16:05.63 and 2,306,660 KiB peak RSS, compared with the preceding 12:27.30 and 2,006,728 KiB build on the same host.
That sequential measurement made the strict boundary 29.2% slower and raised peak RSS by 14.9%, which justified optimizing only its pure hash computation while leaving ordering and publication serial.
An ASCII canonical-hash shortcut was exact on 100,000 real rows but improved that long-text workload by only 1.3%, and a C-backed combining-mark and repeat pass was exact but 12.4% slower, so both were rejected.
A provider-free benchmark over 20,000 real rows and 519 million characters then computed the existing strict hash and SimHash together in ordered 512-row batches.
Six forked workers preserved every output and reduced that pure-compute slice from 104.30 to 26.29 seconds, but a forced integration test exposed Python's warning that forking the multithreaded test process could deadlock.
The retained implementation therefore uses the safe `spawn` context, caps itself at six standard-library workers, and remains sequential for a routing population smaller than one batch.
All SQLite access, near-index mutation, quarantine decisions, and JSONL writes remain serial in the parent.
The complete isolated safe-spawn rebuild took 13:05.75, an 18.6% reduction from the sequential strict build and 5.1% above the earlier build without strict cross-role checking.
Peak reported RSS was 2,458,052 KiB, 6.6% above the sequential strict run, and user CPU time rose because this is a wall-time optimization rather than a compute reduction.
The isolated manifest and every routing and quarantine output were byte-identical to the published strict-clean corpus.
Raising the cap from six workers to all 12 logical CPUs reduced the exact 10,000-row hash microbenchmark from 9.647 to 8.467 seconds, or 12.2%, with an identical output digest.
The complete isolated rebuild changed only from 13:05.75 to 12:47.39, or 2.34%, while the manifest and every routing and quarantine output remained byte-identical, so the doubled worker count was rejected as end-to-end timing noise.
A later cache targeted only common unigram digests, leaving high-cardinality bigrams uncached so they could not evict the reusable entries as they did in the rejected all-feature cache.
It improved the exact 10,000-row real-data fingerprint sample by about 20% and reduced the isolated six-worker rebuild from 13:05.75 to 12:16.98, or 6.21%, while peak reported RSS fell from 2,458,052 to 2,427,544 KiB.
The entire copied data tree remained byte-identical to the published manifest, so the bounded unigram cache is retained.
Reusing one standard-library JSON encoder preserved every byte but improved two 50,000-row real-data serialization trials by only 0.8% to 1.8%, so it was rejected before another full rebuild.
A separate 10,000-entry bigram cache preserved every fingerprint and improved a 20,000-row real-data microbenchmark by about 10.5%, but the complete isolated rebuild slowed from 12:16.98 to 12:19.32 and peak RSS rose from 2,427,544 to 2,455,560 KiB, so it was reverted.
Six-worker process-map chunk sizes from 1 through 128 produced the same 10,000-row digest, but the best candidate improved the repeated size-16 baseline by only 0.8%, so scheduling stayed unchanged without another full build.
Routing ingestion previously read all 12,417,820,684 source bytes once to verify their digests and then opened every shard again to parse the same bytes.
Hashing those 33 files alone took 8.56 seconds on the same host.
The retained path now updates each source digest while parsing its single binary stream and verifies that digest before committing the source transaction.
The complete six-worker rebuild fell from 12:16.98 to 11:30.07, or 6.37%, while peak reported RSS changed from 2,427,544 to 2,455,760 KiB.
The manifest and every routing and quarantine output remained byte-identical.

Do **not** repair this by editing `normalize_text` or the registered runtime normalizer.
The former changes every `normalized_text_sha256` in the manifest, while the latter silently changes retained-model behavior.
The separate fingerprint is for leakage filtering and audit only; a model-input revision still requires prospective evaluation and a new artifact identity.

## Consequence

No current classifier should block users, approve transactions, or grant tool authority.
The August channel-aware reviewer improves the open mixed panel and matched boundary controls, but it remains prompt-selected development evidence and loses PromptShield recall.
The frozen Financial AI CTF diagnostic reaches 83.33% on weak source-labelled instruction overrides but only 63.26% participant-macro recall, while most successful protected-field leaks fall outside the instruction-subversion label.
That result supports output-egress mediation and stronger participant-heldout evidence, not relabelling every confidentiality failure as prompt injection.
The prospectively frozen FORCE-Bench panel adds 251 zero-overlap legitimate finance tasks with no observed restriction, but its 1.507% 95% upper bound and templated public source are still too weak for a production-FPR claim.
The fit-disjoint SWE-bench Verified panel then rejects the local high gate on legitimate software tasks and stops before DeepSeek review, so its rows cannot be used for post-hoc high-zone calibration.
The final multilingual Boundary Pairs block passes every frozen quality gate, but does so with a source-specific linear fit and an 88.29% remote-review rate after all source scenarios have been consumed.
Its linear gate subsequently failed the frozen independent Operant AI hard-benign check and remains excluded from the maintained cascade.
The known-span ACL long-document diagnostic also rejects the registered local gate, shorter windows, and scalar threshold repair.
Its consumed-family task-conditioned canary improves paired ranking but fails the frozen reviewer threshold and cannot authorize post-hoc calibration on the remaining source families.
Independent StruQ evidence confirms task-conditioned separation for three explicit fixed-target attack constructions, but every row requires remote review and the source is too narrow for integration.
InjecAgent adds varied task-conditioned tool-output goals and improves recall without a clean flag, but the fixed operating point fails and its tiny clean denominator cannot authorize the lower post-hoc score scale.
API-Bank establishes the output-only runtime boundary and supplies consumed clean evidence for a `0.5` candidate, while AgentDyn supplies strong but explicit attack-only transfer at that value.
AgentPIMA supplies the prospectively frozen matched gate and rejects that candidate: roughly 95% attack recall coincides with 27.63% to 38.52% exact-unique clean flags, and no scalar threshold repairs the tradeoff.
The next work is deterministic authorization, a frozen low-call invocation policy, matched finance and agent-boundary tasks, and representative traffic, not additional tuning on open panels.
A future candidate remains a shadow-only advisory sensor behind the deterministic reference monitor.
