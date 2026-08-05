# Long-context reviewer research

## Decision

Do not replace the existing per-window branch with one full-context DeepSeek decision.
That prospectively frozen replacement improved most aggregate metrics but missed its exact PIArena completion non-inferiority gate.

Retain the existing branch as a fallback and add one full-context first pass only for remote-enabled multi-window `untrusted_content`.
A full-context flag can restrict early, while a clear result continues through the existing middle-window reviews.
This union cannot remove an existing catch, leaves direct-user and single-window routing unchanged, and remains advisory only.

## Prompt Overflow audit

The [Prompt Overflow paper](https://arxiv.org/abs/2605.23196) directly targets Morgott's prior architecture: independent bounded windows can each look harmless even when a downstream long-context model can reconstruct distributed malicious intent.
The paper evaluates both disjoint and overlapping windows and attributes the failure to sparse local evidence rather than boundary placement alone.

The paper's [anonymous artifact](https://anonymous.4open.science/r/Prompt-Overflow-2624/) had a disconnected web view during this audit, but its public file API and [ZIP endpoint](https://anonymous.4open.science/api/repo/Prompt-Overflow-2624/zip) remained accessible.
The retrieved 25,795,226-byte ZIP has SHA-256 `d26f4d631bc69619d44273d72d756294400bc162a616502b69f17c8e2a5f11b4`.
The arXiv source archive has SHA-256 `14aa16fdd9d05a7755ab1e1511f91799151cbb6a0a0231d9fba2331cc82a5371`.

The release contains construction, detector, downstream-execution, and weak-signal-defense scripts, but no frozen source data, generated true-positive set, attack panel, or result ledger.
Its attacker scripts load the current [`rogue-security/prompt-injections-benchmark`](https://huggingface.co/datasets/rogue-security/prompt-injections-benchmark) through the retired `qualifire` alias without pinning a dataset revision.
Morgott already found normalized exact overlap for 2,714 of that source's 5,000 rows, so it cannot provide independent headline evidence.

The released defense also differs materially from the paper's description.
The paper describes 2,000 benign calibration prompts, 1,000 held-out benign prompts, and a contiguous-run gate.
The script defaults to 1,500 benign prompts, estimates the 99th percentile from those same windows, does not evaluate a held-out benign set, and computes a top-five excess sum over windows regardless of adjacency.
Its multiple-window condition filters exported examples after `defense_hit` is calculated rather than defining the final decision.

Morgott therefore did not copy that heuristic or claim to reproduce the paper's defense result.
The paper remains strong architectural evidence for aligning the reviewer's view with the downstream model's view.

## Prospectively frozen replacement test

The frozen study retained 1,060 multi-window artifacts before opening any new DeepSeek result.
It retained 485 local-high restrictions and sent the other 575 complete normalized artifacts through the unchanged DeepSeek V4 Flash 0731 contract once each.
The source panels were the existing fit-disjoint PIArena, SWE-bench Verified, and BFCL v4 live artifacts.

| Cohort | Existing branch | One full-context replacement |
|---|---:|---:|
| PIArena clean restrictions | 2 / 183 | 0 / 183 |
| PIArena direct restrictions | 150 / 183 | 170 / 183 |
| PIArena ignore restrictions | 182 / 183 | 183 / 183 |
| PIArena completion restrictions | 182 / 183 | 181 / 183 |
| PIArena combined restrictions | 183 / 183 | 183 / 183 |
| SWE-bench multi-window controls | incomplete old remote phase | 1 / 138, entirely from the retained local high |
| BFCL multi-window controls | 2 / 7 | 1 / 7, entirely from the retained local high |

The replacement completed 575 reviews with zero terminal provider failure, 605 attempts, 1,515,136 input tokens, and 4,601 output tokens.
It rejected itself because completion restrictions fell from 182 to 181, below the frozen non-inferiority gate.
Pairwise PIArena comparison found 28 new attack catches and eight lost existing catches, confirming that full-context and local-window views are complementary rather than interchangeable.

The frozen manifest, panel, results, and summary hashes are `3d5c50ea85f89f4741ee5b8b764b13cc5f63543acb66b864d2315df0fa4faef1`, `e46761d82529a38024c6271c536673e4da61973328123d533df3f1c40571c355`, `60c954c5033f5212f175c3115101ca5da6a1cad9ea9b9071e0218e8877c15dcf`, and `d163125956379bd3c6894a71cd81c773db3241a98a66ce6457487e80bf41a79b`.

## Full-context-first fallback

The maintained repair takes the union in an operationally cheaper order for remote-enabled multi-window untrusted content.
It keeps any local-high restriction, reviews the complete normalized artifact once, stops on a full-context flag, and otherwise runs the existing middle-window branch.
Provider failure remains conservative and incomplete.

Offline replay of the two frozen ledgers changes PIArena multi-window attack restrictions from 697 of 732, or 95.22%, to 725 of 732, or 99.04%.
Matched-clean restrictions remain 2 of 183, or 1.09%.
The full-context pass added no restriction among its 326 selected PIArena-clean, SWE-bench, and BFCL control inputs, whose zero-event Wilson 95% upper bound is 1.165%.

For the same 432 PIArena provider-eligible artifacts, the old branch used 1,618 provider attempts.
The full-context-first order would use 436 full-context attempts plus 631 fallback attempts, or 1,067 total, a 34.05% reduction on this attack-heavy panel.
The clean PIArena slice moves in the opposite direction, from 559 attempts to 743, so benign-heavy deployments may pay more latency and cost.

The result does not establish a production false-positive rate, a universal Prompt Overflow defense, or a final-test claim.
It does establish a simpler advisory architecture that preserves every existing path while adding a bounded global view when the operator explicitly enables remote review.

## Independent long-control gate

The prospectively frozen [LongBench Pro control](longbench-pro-control-research.md) then tested the operational assumptions on 471 normalized-unique English and Chinese natural long documents.
Its local preflight stopped before OpenRouter because 7 controls, or 1.486%, were already local-high restrictions, above the frozen 1% ceiling.
The 464 remaining complete reviews fit within the 16 million-token budget, but an all-clear outcome would require 17,877 fallback window reviews, above the frozen 8,192-window ceiling.

No remote result was opened, so this is not evidence against DeepSeek V4 Flash 0731.
It is evidence that the retained local-high decision and exhaustive fallback are not yet supported for benign-heavy long-document traffic.

A sealed post-hoc PIArena replay found that reviewing only the eight highest-scoring middle windows after a clear complete review preserved all 725 attack restrictions and both clean restrictions produced by exhaustive fallback on that consumed panel.
That bound would reduce the LongBench worst case to 3,712 fallback reviews plus 464 complete reviews.
The top-eight rule is a diagnosed candidate, not maintained behavior, because PIArena is consumed and it does not test whether remote review can safely clear local-high natural documents.
A fresh matched long-context attack/control panel is required before either change.

## Fresh matched LongBench v2 stop

The next [prospectively frozen matched panel](longbench-v2-matched-research.md) paired 59 fit-disjoint LongBench v2 contexts with deterministic known-span attacks across five payload families and three positions.
Its local preflight found nine local-high attacks but zero local-high clean artifacts, so it stopped before OpenRouter under the registered coverage gate.
The bounded full-context plus top-eight candidate fit within its call and token budgets, but this source could not test whether remote review safely repairs the local-high clean errors observed on LongBench Pro.

No reviewer result was opened and no cascade change followed.
Because LongBench v2 is English-only while six of the seven LongBench Pro local-high controls were Chinese, the next fresh source should add Chinese or genuinely multilingual matched long documents without assuming that language alone caused the earlier errors.

## Chinese matched LongBench rejection

The next [prospectively frozen Chinese panel](longbench-zh-matched-research.md) used 100 fit-disjoint clean-attack pairs balanced across the five official Chinese LongBench tasks, five deterministic Chinese payload families, and three insertion positions.
Unlike the LongBench v2 preflight, it contained both local-high clean and attack artifacts and therefore authorized the exact remote successor.

The bounded full-context plus top-eight candidate restricted 99 of 100 attacks and lost none of the 44 local-high attack catches.
It nevertheless restricted 11 of 100 clean controls against the frozen maximum of 2 and produced only 88 fully correct pairs against the minimum of 90.
It repaired one of three local-high clean restrictions, but that benefit did not offset the new clean flags elsewhere.

The candidate rejected itself, and the maintained exhaustive fallback remains unchanged.
The result rules out promoting this bounded union from the consumed panels and shifts the next architecture work toward trusted task or policy context and deterministic action outcomes.
