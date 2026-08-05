# Real coding-agent traffic research

## Decision

The completed local-only SWE-chat diagnostic rejects the registered cascade for coding-agent traffic-like use.
The unchanged local gate hard-restricted 58 of 5,121 retained sessions, or 1.133%, with a 95% Wilson upper bound of 1.461%, which exceeds the prospectively frozen 1% restriction-load cap.
Another 2,676 sessions, or 52.255%, would require remote review, which exceeds the 25% review-load cap and produces 6,063 review windows against a 4,096-window cap.
One prompt needed 131 windows against the frozen maximum of 128.
No threshold, model, prompt, or runtime setting was selected from this result.
SWE-chat is not benign ground truth, so these figures are restriction and review load rather than a false-positive rate.

## Completed population and results

The pinned source contains 5,851 sessions and 2,692,480 events.
The projection found 61,473 non-continuation user-prompt rows and an earliest qualifying prompt for 5,794 sessions.
The release contains 39 duplicated `(session_id, turn_number)` groups despite documenting that turn IDs are globally unique.
All 39 were two-row byte-identical duplicates across the selected prompt and lineage fields, so preparation collapses only those exact duplicates and fails closed on any conflicting duplicate.

Morgott's local sensitive-text screen excluded 659 sessions, including 562 email-address triggers, 73 government or vehicle identifiers, and 23 credential values.
Complete-fit overlap checks then removed 14 sessions across four normalized prompts.
The retained population is 5,121 sessions represented by 4,301 unique normalized prompts after 821 exact duplicate prompt rows and nine normalization-only duplicates were collapsed, with session weights preserved.
Missing optional annotations did not discard otherwise valid prompts: 1,764 selected source rows had no language annotation and 1,061 had no timestamp.

| Prompt length | Sessions | Local-high rate | Candidate-review rate | Review windows |
| --- | ---: | ---: | ---: | ---: |
| Under 16 words | 1,585 | 0.379% | 50.221% | 796 |
| 16 to 63 words | 1,686 | 0.890% | 26.987% | 455 |
| 64 to 255 words | 594 | 2.694% | 42.256% | 281 |
| 256 words or more | 1,256 | 1.672% | 93.471% | 4,531 |

Long prompts dominate review cost: the 1,256 sessions with at least 256 words account for 4,531 of 6,063 candidate-review windows.
The largest prompt contained 50,168 model tokens and required 131 windows.
The local OpenVINO pass completed in 769.83 seconds.

Claude Code supplies 4,182 retained sessions, of which 1.196% were local-high and 51.124% required candidate review.
Codex supplies 209 sessions, of which 3.349% were local-high and 21.531% required candidate review.
OpenCode supplies 622 sessions with no local-high route, but 69.453% required candidate review.
Smaller agent and language slices are too sparse for stable comparative conclusions.

## Candidate comparison

[Trace Commons revision `112ebd4d`](https://huggingface.co/datasets/trace-commons/agent-traces/tree/112ebd4d03ce852b00e935d523107c3d0c9a65bf) contains complete voluntarily donated coding-agent sessions and documents local scrubbing, contributor review, and a takedown path.
Its pinned table has only 30 sessions, its card explicitly says the opt-in sample is not representative, and its anonymization is best effort.
It is too small and concentrated for the missing traffic denominator.

[Claude Code Community Conversations revision `fe11da9a`](https://huggingface.co/datasets/lelouch0110/claudeset-community/tree/fe11da9ac006d5592378a3d284ee2ed81ffb7578) documents contributor consent and community-upload redaction for common secrets, paths, addresses, and user-supplied terms.
The pinned Dataset Viewer exposes only 114 sessions, so even a multi-contributor sample cannot materially tighten the missing denominator.
No conversation row was fetched during this screening.

[OpenCode Agentic Mini revision `43c03fe3`](https://huggingface.co/datasets/Petrouil/opencode-agentic-mini/tree/43c03fe3bbf56294f262df2b3423ae35154ad1ea) claims 507 real CLI sessions across more than 45 projects, but its card documents no consent, secret scan, privacy review, or redaction gate.
The card also reports 19,550 examples while the pinned Hub table contains 7,264 rows, and 414 sessions were split into overlapping training chunks rather than retained as one session row.
It was rejected before downloading conversation content because neither its privacy contract nor its published population is sufficiently stable for this diagnostic.

[TraceLab release `v0.0.1`](https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.1) contains 357,161 LLM rounds from 43 developers and is a much stronger workload sample.
Its public sanitizer deliberately removes tool inputs, model text, and tool outputs and preserves only structural, timing, token, and command-shape metadata.
It cannot provide detector inputs and is therefore unsuitable for Morgott's routing evaluation, although it remains useful serving-workload evidence.

[The Real Pi Coding Agent Traces aggregation at revision `8c593252`](https://huggingface.co/datasets/MaxDevv/real-pi-coding-agent-traces-sessions/tree/8c593252ddad7dca08a0afc07896195fa73f2d6e) is an ungated fallback with 1,291 opt-in public sessions and 777,376,372 source bytes across 21 upstream datasets.
Its source uploader applies exact-secret redaction, blocks every TruffleHog finding, uses an LLM privacy review, and requires the contributor to upload the session explicitly.
The aggregation is too weak to replace SWE-chat: one upstream source contributes 626 sessions, the four largest sources contribute 1,024, all 1,291 manifest rows use the non-specific license value `other`, and its dataset viewer currently fails on malformed JSON.
Its claim that synthetic sessions were excluded also conflicts with 38 retained rows attributed to `aaaaliou/pi-synthetic`.
No session text was downloaded during this audit.
Do not add a second traffic runner for this source after the stronger SWE-chat diagnostic has already answered the current traffic-load question.
Reconsider it only as a clearly labelled concentrated stress source for a materially different future architecture and evaluation question.

[SWE-chat revision `f66cca95`](https://huggingface.co/datasets/SALT-NLP/SWE-chat/tree/f66cca95b14caaa4177f7ed5eaa424608dadcffa) publishes 5,851 sessions, 2,692,480 conversation events, 205 repositories, and stable repository, session, checkpoint, commit, and turn identifiers.
The source comes from developers who opted into Entire CLI tracking on public repositories, and the authors report Microsoft Presidio and TruffleHog redaction plus Stanford IRB exemption in the [paper's ethics statement](https://arxiv.org/html/2604.20779#S5).
The paper also states that the population selects for early adopters, omits proprietary repositories, overrepresents Entire.io's own repository, and misses abandoned sessions whose logs were never committed in its [limitations](https://arxiv.org/html/2604.20779#A1).

## Label and privacy boundary

Public opt-in coding work does not establish that every prompt is benign, non-adversarial, or free of residual personal data.
Commit survival, source redaction, and model annotations such as prompt intent or session success are not instruction-subversion labels.
The source can measure restriction and review load, but not a false-positive rate, calibration, or representative production precision.

The completed run kept source text local, applied Morgott's own sensitive-text screen, removed complete-fit overlap, retained no prompt text in artifacts, and made no OpenRouter call from the gated source.
The earliest non-continuation user prompt per session is the narrowest useful traffic projection because it excludes automatic continuation messages, model text, tool data, code, diffs, and later interaction feedback.

## Access state

The repository owner accepted the standard Hugging Face gate and explicitly authorized authenticated access.
The 1.31 GB source Parquet was hash-verified at revision `f66cca95b14caaa4177f7ed5eaa424608dadcffa`, used only for local preparation, and removed from temporary storage after the run.
OpenRouter provider calls remained zero.

The manifest, panel, local results, preflight, and summary SHA-256 digests are `434e60cb72891bf2a6df4ba45fa8ca2af4f9ec22c0091e87c2375db5fde0ece1`, `2d28fd3c23da331d1e7e08133859e3d30c8502da45e6c9bb4dfc6ed4e108cb3e`, `51b69c19ac7cd5fea62f1cb40ad78b442bb12824d4e1c3c3369d318f19c58df4`, `be511c37f0e12bade8b527626d33efda04f1f6da82436f9cb0e5c5714f0797d9`, and `8caaea06890a712f96e17d3fc751c935a790acfd95593e2818128782c2dd71d7`.
