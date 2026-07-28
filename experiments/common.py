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
from functools import cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def threshold_at_fpr(
    scores: np.ndarray, labels: np.ndarray, target: float = 0.001
) -> float:
    """Greatest-recall threshold that does not exceed `target` FPR."""
    from morgott.detector import choose_threshold

    return choose_threshold(labels, scores, target)


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


@cache
def _recipe_records() -> dict[str, tuple[dict, ...]]:
    """Rebuild the retained weak-transfer recipe's three exact populations."""
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "_archived"))
    import routing_encoder as archived

    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    lineage = archived._lineage_roles(DATA_DIR, manifest)
    maximum = archived.CONSOLIDATED_MAX_PER_STRATUM
    locked = archived._locked_evaluation_hashes(DATA_DIR, manifest)
    base_train, _ = archived.load_capped_fold_records(
        DATA_DIR,
        manifest,
        "train",
        fold_name=None,
        training=True,
        include_boundary=True,
        maximum=maximum,
        lineage_roles=lineage,
    )
    base_validation, _ = archived.load_capped_fold_records(
        DATA_DIR,
        manifest,
        "validation",
        fold_name=None,
        training=True,
        include_boundary=False,
        maximum=maximum,
        lineage_roles=lineage,
    )
    direct_suite, _ = archived.load_capped_fold_records(
        DATA_DIR,
        manifest,
        "dev_test",
        fold_name=None,
        training=False,
        include_boundary=False,
        maximum=maximum,
        lineage_roles=lineage,
    )
    base_train = [
        record
        for record in base_train
        if record["channel"] == "direct_user" and record["hash"] not in locked
    ]
    base_validation = [
        record for record in base_validation if record["channel"] == "direct_user"
    ]
    direct_suite = [
        record
        for record in direct_suite
        if record["channel"] == "direct_user" and "multi_turn" not in record["sources"]
    ]
    multi_turn = archived._multi_turn_repair_records(
        DATA_DIR, manifest, maximum_training_rows_per_label=maximum
    )
    wildguard = archived._wildguard_vanilla_repair_records(
        DATA_DIR,
        manifest,
        maximum_training_rows_per_label=maximum,
        maximum_validation_rows_per_label=max(1, maximum // 4),
        locked_hashes=locked,
        existing_training_hashes={record["hash"] for record in base_train},
    )
    return {
        "train": tuple([*base_train, *multi_turn["train"], *wildguard["train"]]),
        "validation": tuple(
            [*base_validation, *multi_turn["validation"], *wildguard["validation"]]
        ),
        "dev_test": tuple(direct_suite),
    }


def load_records(split: str, *, require_known_direct_target: bool = True) -> list[dict]:
    """Return a direct-user split exactly as the retained recipe constructed it."""
    if split not in {"train", "validation", "dev_test"}:
        raise ValueError(f"unsupported split: {split}")
    return [
        record
        for record in _recipe_records()[split]
        if not require_known_direct_target
        or record["targets"]["direct_instruction_subversion"] is not None
    ]
