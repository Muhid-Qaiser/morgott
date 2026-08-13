# Fixed-checkpoint long-code comparison

This experiment scores one completed mmBERT retained snapshot on the exact
repository-held-out SWE-rebench V2 validation and dev-test pair archives. It
uses ordered windows whose size is the explicit evaluation context cap and
takes the maximum primary-head score per document.

The runner does not select a checkpoint or threshold from SWE-rebench. It
requires a completed full-panel evaluation of the same run, snapshot, and
evaluation cap, then transports that evaluation's canonical-calibration
`1.0000%` component threshold. The full 512/1024 two-by-two panel must therefore
run first. Both SWE-rebench splits remain already-open development evidence.

Outputs contain aggregate metrics only. Resumable journals contain numeric
scores and window counts only; neither artifact contains prompt text, row IDs,
or repository identities.

For the fixed update-17,000 comparison, use the exact frozen training-pair
identity and name every cell by both caps:

```bash
export HF_HOME=/workspace/hf_cache
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PAIR_SHA=84a3b1e185755739afca5165ef9aaadb55ce248695bb4c426351f94126ebbbba
RUN512=artifacts/mmbert/runs/mmbert-lora-full-s42-mb24-nolengthgroup-noharm-current-control
SNAP512=artifacts/mmbert/runs/.mmbert-lora-full-s42-mb24-nolengthgroup-noharm-current-control.snapshots/update-017000.pt
RUN1024=artifacts/mmbert/runs/mmbert-lora-full-s42-mb24-nolengthgroup-noharm-ctx1024
SNAP1024=artifacts/mmbert/runs/.mmbert-lora-full-s42-mb24-nolengthgroup-noharm-ctx1024.snapshots/update-017000.pt

uv run --locked --extra encoder python -m experiments.mmbert_longcode_snapshot_eval.run \
  "$RUN512" --snapshot "$SNAP512" \
  --full-evaluation "$RUN512/evaluation-update-17000-trainctx512-evalctx512/evaluation.json" \
  --evaluation-max-tokens 512 --batch-size 24 --require-update 17000 \
  --require-additional-pairs-sha256 "$PAIR_SHA"

uv run --locked --extra encoder python -m experiments.mmbert_longcode_snapshot_eval.run \
  "$RUN512" --snapshot "$SNAP512" \
  --full-evaluation "$RUN512/evaluation-update-17000-trainctx512-evalctx1024/evaluation.json" \
  --evaluation-max-tokens 1024 --batch-size 24 --require-update 17000 \
  --require-additional-pairs-sha256 "$PAIR_SHA"

uv run --locked --extra encoder python -m experiments.mmbert_longcode_snapshot_eval.run \
  "$RUN1024" --snapshot "$SNAP1024" \
  --full-evaluation "$RUN1024/evaluation-update-17000-trainctx1024-evalctx512/evaluation.json" \
  --evaluation-max-tokens 512 --batch-size 24 --require-update 17000 \
  --require-additional-pairs-sha256 "$PAIR_SHA"

uv run --locked --extra encoder python -m experiments.mmbert_longcode_snapshot_eval.run \
  "$RUN1024" --snapshot "$SNAP1024" \
  --full-evaluation "$RUN1024/evaluation-update-17000-trainctx1024-evalctx1024/evaluation.json" \
  --evaluation-max-tokens 1024 --batch-size 24 --require-update 17000 \
  --require-additional-pairs-sha256 "$PAIR_SHA"
```

The two native cells answer whether the 1,024-token candidate is better as
trained and deployed. The off-diagonal cells separate a training-context effect
from merely exposing either model to longer text at evaluation. Compare all
four on identical archive hashes and report threshold-free AUROC/pair ordering
beside the transported-threshold rates.
