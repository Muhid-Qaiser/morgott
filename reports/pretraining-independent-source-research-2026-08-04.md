# Independent evidence sources before another encoder run

Date: 2026-08-04.

Research cutoff: 2026-08-04.

## Decision

No single public release satisfies Morgott's matched transaction, known-span long-document, genuine long-benign, and source-heldout requirements.

Adopt a four-source evidence stack instead of merging these roles into one corpus claim.

Use tau2-Bench-Verified only as the executable substrate for newly constructed clean and one-field-attacked transaction pairs.

Use SWE-rebench only as a repository-grouped long legitimate-task denominator after local privacy, overlap, and tokenizer-length preflight.

Reserve LogInject-1.0 and WASP in full as independent source-heldout evaluations, with LogInject measuring long untrusted log artifacts and WASP measuring end-to-end browser behavior.

Do not use the General Analysis Long Context Benchmark because its `injection_text` field mixes prompt security with unrelated harmful-content insertion and does not expose matched clean lineage.

## Evidence matrix

| Source and exact pin | Published scale and artifact size | What it establishes | Limitation | Recommendation |
| --- | --- | --- | --- | --- |
| [tau2-Bench-Verified `864350a8`](https://github.com/amazon-agi/tau2-bench-verified/tree/864350a8971a8f8ee9e7b8472e2edc380a806b0c) | The official tau2 paper reports 115 Retail, 50 Airline, and 114 Telecom tasks, while the pinned task files are 329,895, 150,905, and 13,977,063 bytes respectively ([paper](https://arxiv.org/pdf/2506.07982), [verified release](https://github.com/amazon-agi/tau2-bench-verified)) | Domain policy, typed tools, seeded state, expected actions, and state assertions provide stable task lineage and executable clean utility oracles | It contains no native prompt-injection arm, and the source authors document that the original task definitions and evaluation criteria required correction | Adopt as a substrate, not as ready-made injection-labelled data |
| [SWE-rebench `89cdfbab`](https://huggingface.co/datasets/nebius/SWE-rebench/tree/89cdfbab4ab1bd8f5a658bb212d1b63624f4f881) | The pinned release has 21,336 `test` rows and 6,542 `filtered` rows, 27,878 rows total, a 245,166,035-byte download, and 964,645,964 decoded bytes; problem statements reach 52,500 characters and hints reach 273,000 characters ([dataset card](https://huggingface.co/datasets/nebius/SWE-rebench), [paper](https://arxiv.org/abs/2505.20411)) | Real GitHub issue and pull-request provenance, repository IDs, base commits, timestamps, test patches, and executable tests support repository-heldout, time-aware long legitimate-task analysis | Automated collection and LLM-assisted quality scoring do not adjudicate broad benignity, and the release is limited to Python repositories | Adopt only as a dev-test denominator and call its result restriction or review load, not FPR, until row-level audit supports a benign claim |
| [LogInject-1.0 Zenodo v1](https://doi.org/10.5281/zenodo.20436935) | The paper reports 12,847 log entries, including 10,278 benign and 2,569 adversarial entries; the immutable archive is 952,080 bytes with MD5 `bf56698a2ab2dd2280189620e7654a6d` ([artifact](https://zenodo.org/records/20436935), [paper](https://arxiv.org/abs/2607.14493v1)) | Every adversarial sample records payload text, level, vector, objective, expected behavior, and an attack-success pattern; atomic, fragmented, and obfuscated attacks cover Apache, SSH, JSON, and error-message fields | Its long unit is a constructed log batch rather than a natural document, all released benign and adversarial rows are deterministically generated, attack payloads come from a small template system, and context stitching needs a set of fragment spans rather than one contiguous span | Adopt as a wholly held-out known-span long-untrusted-artifact diagnostic, not as the strict long-document prerequisite or fitting data |
| [WASP `ffee6f41`](https://github.com/facebookresearch/wasp/tree/ffee6f41fde76acd14bd792db442479c506260c2) | The official paper reports 42 attacker-goal and user-goal scenarios, two injection forms, 84 attacked tasks, and 37 utility prompts across self-hosted GitLab and Reddit ([paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/1c9818387f5dd0a0bc151214660f059d-Paper-Datasets_and_Benchmarks_Track.pdf), [repository](https://github.com/facebookresearch/wasp)) | Stable local web state, constrained attacker-editable fields, clean utility checks, and end-to-end attacker outcomes provide a distinct source-heldout browser evaluation | It has only two websites and two hand-written injection forms, requires a costly browser environment, and is not a transaction benchmark | Reserve the complete source unopened until the candidate and operating point are frozen |
| [General Analysis Long Context Benchmark `a9eaff0f`](https://huggingface.co/datasets/GeneralAnalysis/GA_Long_Context_Jailbreak_Benchmark/tree/a9eaff0f0d4a6eb77b92295da0390c758ac25d05) | The gated 72,017,818-byte CSV has 1,500 conversations, 750 rows with `injection_text`, a median 8,200 tokens, and only 144 rows tagged `prompt_security` ([dataset card](https://huggingface.co/datasets/GeneralAnalysis/GA_Long_Context_Jailbreak_Benchmark)) | It exposes an inserted string and turn index over long serialized conversations | Its own example treats harmful chemical guidance as an injection while `prompt_security=0`, the card calls the release synthetic, and no clean-pair identifier or underlying trace provenance is documented | Reject for Morgott injection training and evaluation |

## Required construction and split contract

For tau2-Bench-Verified, create one unchanged clean task and one attacked twin by modifying exactly one untrusted tool-return field while keeping the task, database seed, trusted authority, and expected legitimate final state identical.

Record the inserted payload and exact character span, and keep each clean-attacked pair plus all variants of its task in one split group.

Do not label tau2 user requests as attacks, do not derive authority from the injected field, and do not use model success as the detector label.

Hold out complete task families or domains from fitting so that a row-random split cannot reward memorized policy, entity, or payload templates.

For SWE-rebench, project only the original `problem_statement`, preserve repository, instance, base-commit, and timestamp lineage, and exclude patches, gold tests, generated installation recipes, and LLM quality scores from detector input.

Before using it as a denominator, measure registered-tokenizer lengths, privacy-screen locally, remove all fit overlap, quarantine suspicious instruction-subversion candidates, and require enough rows above 512 tokens to support a narrow long-task claim.

Do not send SWE-rebench text to OpenRouter during source preparation or label uncertain rows benign by model agreement.

For LogInject, retain the clean log entry or batch and attacked counterpart when reconstructable, represent fragmented injections as an ordered span set, and group every instantiation of the same 104-template family together.

Because LogInject is the proposed independent transfer source, do not use any of its rows for fitting, threshold selection, prompt selection, or architecture selection.

For WASP, preserve the browser state and official final-state evaluators and never flatten screenshots, accessibility trees, or traces into standalone text labels.

## Gate before one authorized encoder run

The data gate passes only when tau2 pairs have exact clean utility, vulnerable unmonitored attacks, and complete no-extra-mutation checks; otherwise they are useful clean tasks but not matched attack evidence.

The completed bounded construction did not pass this gate.
Four matched retail pairs completed every golden action in both arms, and six attacked arms consumed the poisoned field without making the injected cancellation, but no arm demonstrated the required vulnerable attacked state.
The construction therefore contributes zero training pairs and does not authorize the pending LP-FT run.

The long-benign gate passes only when the frozen SWE-rebench slice has adequate independent repositories and a substantial above-512-token denominator after privacy and overlap exclusions.

The known-span gate passes narrowly with LogInject only for long untrusted log batches, so a strict natural-document claim still requires a different source.

The source-heldout gate passes only if LogInject and WASP remain completely absent from training, validation, threshold selection, prompt selection, and architecture selection until one candidate is frozen.

If these conditions hold, one controlled encoder comparison is justified; if any condition fails, another training run would convert a known evidence gap into another confounded score.

## Claim boundary

This review establishes public artifact availability and a defensible division of evidence roles.

It does not validate upstream row labels, authorize flattening stateful benchmarks into detector rows, or establish a production false-positive rate.

No source dataset file or gated row text was downloaded during this review.
