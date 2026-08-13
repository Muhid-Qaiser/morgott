# Current handoff (2026-08-12, 2x RTX 4090)

This file supersedes every earlier resume instruction. Training and all queued
update-18,500 evaluations are complete. There is no trainer, evaluator, or GPU
queue to resume. Future training follows the repository's scientific, privacy,
resource, and evidence gates without a separate owner-approval step.

## Current decision

The 1,024-token no-harm run completed 25,083 updates and three epochs. Continue
1,024-token research, but do not promote either checkpoint or call the packaged
update 18,500 checkpoint best overall.

The frozen update-17,000 native comparison remains the primary context result:

| Checkpoint | Canonical TPR@1% | PromptShield TPR@1% | SEP TPR@1% | SEP pair order | Finance flags | Reserve attested | Bare-harmful off-target | Long-code clean flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512, update 17,000 | 60.15% | 57.00% | 44.81% | 82.53% | 0 | 85.12% | 10.35% | 15 |
| 1,024, update 17,000 | 71.86% | 52.85% | 49.83% | 76.78% | 2 | 86.10% | 10.38% | 2 |
| 1,024, update 18,500 | 66.91% | 58.53% | 49.93% | 75.38% | 2 | 87.52% | 13.21% | 3 |

Update 18,500 is a useful historical-selector diagnostic: it improves
PromptShield and reserve recall, but regresses canonical recall, pair ordering,
reserve off-target behavior, and long-code clean flags. The old equal-domain
BCE selector cannot choose the best low-FPR operating trade.

Neither 1,024-context checkpoint is registered for maintained inference or
approved for blocking. Other historical advisory models remain registered as
documented in `model-artifacts.json`.

## Authoritative evidence

Read these first:

- reports/mmbert-context-comparison.json
  SHA-256 18feaf7e1b3a1f9808757363715af6c8ef778120dcd36d133e364ab2543f33fa
- reports/model-experiments.md
- reports/checkpoint-selection-design.md
- reports/semalith-evaluation-plan.md
- reports/runpod-storage-audit-2026-08-12.md
- reports/provenance/mmbert-context-campaign-source-20260812.tar.gz
  SHA-256 7326148fd92f2486afb908ae73f90c2ecb212d0c6bd68f8ef06fd0d6494dca11
- reports/provenance/mmbert-context-results-20260812/
  21 exact aggregate JSONs; SHA-256 manifest
  441c7b901cc80d2476eea71cb2e1429075d449012d33b3e5269a1916d22c488c
- reports/provenance/mmbert-context-checkpoints-20260812/
  three exact evaluated snapshots through Git LFS; SHA-256 manifest
  b70b29e14f196fdd6b4386104bea4c1ce8bbd974c7e90c6af6e0180d2ce2505d

Completed training records:

- artifacts/mmbert/runs/mmbert-lora-full-s42-mb24-nolengthgroup-noharm-current-control/result.json
  SHA-256 49fd6d41dff0638f1044b1e6446ef7d1b1f8f4789e16d0b7010b87eccf76f529
- artifacts/mmbert/runs/mmbert-lora-full-s42-mb24-nolengthgroup-noharm-ctx1024/result.json
  SHA-256 f9f683fbf2aa8c5d0ab0490eebdd9707349eccce4b8b69b815bcf02b56957df6
- 1,024 update-17,000 snapshot
  SHA-256 6de8784ecdb3f954f372f3411f9553889a9cfb8d369b72db20597d4924281774
- 1,024 update-18,500 snapshot
  SHA-256 e33f9deeec260c20d3d60f86158a5d1c13f05d55e50288b58b337bc002cd08af

The machine manifest binds every full, reserve, and long-code result. Exact
aggregate JSONs and the three evaluated checkpoints now have Git-visible,
checksum-verified copies; the checkpoints use Git LFS and remain advisory and
unregistered. Per-row score journals remain reproducible caches on the volume,
not the only copy of any finding. Do not infer evidence from an incomplete
journal or silently substitute an implicit-cap legacy result.

## Trackio

The sole live project is morgott:

    /workspace/hf_cache/trackio/morgott.db

The dashboard is running on port 7860. On this pod:

    https://vr2taetyy2phqh-7860.proxy.runpod.net/

Compact decision summary:

    https://vr2taetyy2phqh-7860.proxy.runpod.net/?project=morgott&runs=summary-context-u17000-native-20260812%2Csummary-checkpoints-train1024-native-20260812&smoothing=0&sidebar=collapsed&hide_empty_tabs=true&metric_filter=%5Esummary%2F

The live project has all 19 historical run identities plus two compact
decision-summary runs. It surfaces 68 useful user metrics, including all eight
historical positive-row val_bce_missed_attacks curves. Those curves are BCE on
positive rows, not missed-true counts or recall. Future training additionally
logs label-aware positive/negative BCE and a descriptive positive-recall
diagnostic without changing the selection rule.

Raw pre-curation backup:

    /workspace/hf_cache/trackio-archive/20260812T105500Z/morgott.pre-curation.db
    SHA-256 08d2aa9df33dbbabbaa340cd1dc779840a6405b354bd1be1fb639edf15d7a11e

Post-summary rollback copy:

    /workspace/hf_cache/trackio-archive/20260812T105500Z/morgott.post-summary.db
    SHA-256 fa9ac33d25765782ab1e0f1819d6a111eb01007265a1a87cc45e37f3b9b6ef52

The archive manifest is in the same directory. Both the curated and
pre-curation databases also have verified compressed Git LFS copies under
`reports/provenance/trackio-20260812/`; its compressed-file SHA-256 manifest is
`d5b5d1457aab3596825f272bd909b76f6af6e8030c670a82528c2ddd066f96f0`.
Dashboard URLs intentionally omit the write token.

## Code and bootstrap state

The code review and Ponytail cleanup retained reusable evaluator/journal/model
contracts and quarantined completed one-shot prototypes. Exact campaign source
was archived before surgery.

Verification at the reviewed PR head:

    make check
    373 tests passed
    Ruff format and lint passed
    ShellCheck passed for all three RunPod scripts
    git diff --check passed

The handoff does not depend on uncommitted pod state.
All code changes described here are tracked in Git.

The reviewed bootstrap bundle is:

    scripts/runpod-bootstrap.sh
    SHA-256 9e7d030be48249f1fd5c6f699ea71243c1f8caf64f353056449a09b1473d71f5
    scripts/runpod-setup-remote-user.sh
    SHA-256 67d42b8109ecfbacce095edd29ae2d5b808672595afb3b056571062577dfb401
    scripts/runpod-editor-cache.sh
    SHA-256 6969c800cc2034c334c1449e613b0610bdd8e9adb867dd0d817e5fbbd3c0c459

The last recorded `/workspace/bootstrap.sh` deployment was made on 2026-08-12 with SHA-256 `c4a32517fe1df566d3444d24e2074acb5df838691c6814353b4975ece39ebdf5`.
It predates the fixes in this revision and must not be used as the replacement-pod source of truth.
This environment cannot access that RunPod volume to refresh or verify the deployed copy.
Before replacing a pod, copy `scripts/runpod-bootstrap.sh` from the reviewed revision to `/workspace/bootstrap.sh` and verify it against the tracked digest above.

It passes bash syntax validation, restores the locked encoder/fa2/tracking environment, validates CUDA and the Trackio database without binding a completed campaign, starts only the morgott dashboard with the compact future-metric order, and never launches GPU work.
The MooseFS mount leaves bootstrap and persistent helpers group/other-writable; manually verify the bootstrap hash on a replacement pod.
Several optional remote CLI installers remain mutable upstream and are a residual supply-chain risk.

## Deferred work (not queued)

Start these only when their stated scientific and resource gates are met:

- Granite Guardian 3.2: rerun through the artifact-writing canary, establish a
  valid projection, then run a full panel only if it meets the runtime gate.
- ProtectAI v2: the canary projected about 117 minutes and showed weak transfer;
  exact aggregate record is reports/provenance/protectai-v2-canary-20260812.json,
  SHA-256 e43bae32ad230804477e3881214d19aebb2f28973301089cd44d41344cd3a585.
- Semalith: wait for gated access, pin the exact v1.5 revision/license, and
  follow reports/semalith-evaluation-plan.md. Do not import its harm/BFSI
  ontology into Morgott prompt-injection training.
- Checkpoint selection: collect the disjoint screen, threshold-calibration, and
  once-open selector panels before using the new constrained protocol. The
  first justified loss candidate is a matched 1,024 BCE versus regularized
  group-DRO study after its disjoint selector panels are ready.
- 2,048 tokens: separate future campaign requiring code support, tail and
  correctness gates, a memory/throughput canary, and the frozen selector
  protocol. It is not launchable by changing one flag.

## Storage and shutdown

The volume measured about 106.62 GB apparent and 116.51 GB allocated. Roughly
44.0 GB apparent / 51.6 GB allocated is later recoverable from completed model
and package caches, but retain Granite, ProtectAI, AprielGuard, active mmBERT,
results, journals, and Trackio backups until the Git commit and LFS push are
verified. After that, the recovery plan in the storage audit can reclaim the
completed baseline and package caches without deleting corpus or evidence.

Both RTX 4090s are idle. It is safe to stop the pod after committing and pushing
the worktree and its LFS objects. No training or evaluation process needs a
graceful checkpoint.
