"""Shared helpers for the experiment scripts.

`threshold_at_fpr` previously existed in four copies with four different names.
They happened to be semantically identical, so the published numbers are
comparable -- but a single edit to one copy would have silently made them not,
and at a 0.1% FPR operating point the threshold is set by roughly twenty rows,
so a one-index difference is not a rounding error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def threshold_at_fpr(
    scores: np.ndarray, labels: np.ndarray, target: float = 0.001
) -> float:
    """Highest threshold whose FPR on the negatives does not exceed `target`.

    Negatives are sorted descending; index floor(target * n) is the first score
    that may be admitted as a false positive. Clamped so a target below one
    row's worth still returns a real score rather than indexing past the end.
    """
    negatives = np.sort(scores[labels == 0])[::-1]
    if not len(negatives):
        raise ValueError("no negatives to calibrate against")
    index = int(np.floor(target * len(negatives)))
    return float(negatives[min(index, len(negatives) - 1)])


def fpr_and_recall(
    scores: np.ndarray, labels: np.ndarray, target: float = 0.001
) -> tuple[float, float, float]:
    """Return (achieved_fpr, recall, threshold) at a `target` FPR operating point."""
    threshold = threshold_at_fpr(scores, labels, target)
    return (
        float((scores[labels == 0] >= threshold).mean()),
        float((scores[labels == 1] >= threshold).mean()),
        threshold,
    )


def load_records(split: str) -> list[dict]:
    """Rebuild one split exactly as `run_direct_failure_repair` does.

    The train split is NOT just the base routing train view: the retained
    `wildguard_weak_transfer` recipe is
    `base_train + multi_turn.train + wildguard.train`. Dropping the last two is
    not a smaller version of the recipe -- it removes the clean WildGuardMix
    counterexamples that `docs/roadmap.md` credits with fixing the
    false-positive frontier, and costs 41 points of recall.

    Mirrors routing_encoder.py:4855-4925. Requires the shim on PYTHONPATH; see
    experiments/_archived/build_shim.py.
    """
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "_archived"))
    import routing_encoder as archived

    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    lineage = archived._lineage_roles(DATA_DIR, manifest)
    maximum = archived.CONSOLIDATED_MAX_PER_STRATUM

    records, _ = archived.load_capped_fold_records(
        DATA_DIR, manifest, split,
        fold_name=None, training=split == "train", include_boundary=False,
        maximum=maximum, lineage_roles=lineage,
    )
    records = [r for r in records if r["channel"] == "direct_user"]

    if split == "train":
        locked = archived._locked_evaluation_hashes(DATA_DIR, manifest)
        records = [r for r in records if r["hash"] not in locked]
        multi_turn = archived._multi_turn_repair_records(
            DATA_DIR, manifest, maximum_training_rows_per_label=maximum
        )
        wildguard = archived._wildguard_vanilla_repair_records(
            DATA_DIR, manifest,
            maximum_training_rows_per_label=maximum,
            maximum_validation_rows_per_label=max(1, maximum // 4),
            locked_hashes=locked,
            existing_training_hashes={r["hash"] for r in records},
        )
        records = [*records, *multi_turn["train"], *wildguard["train"]]
    elif split == "dev_test":
        # The retained headline excluded multi-turn rows from the direct suite.
        records = [r for r in records if "multi_turn" not in r["sources"]]

    return [
        r for r in records
        if r["targets"]["direct_instruction_subversion"] is not None
    ]
