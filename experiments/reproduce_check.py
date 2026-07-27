"""Validate `encoder_infer` by reproducing the retained ensemble's own numbers.

The retained result lives in
`artifacts/direct_failure_repair_ensemble/ensemble-audit.json`. This script
rebuilds the exact dev-test direct suite the original run used, scores it with
the reconstruction in `encoder_infer.py`, and compares threshold-free metrics
(ROC AUC and PR AUC) against what the runner recorded.

Threshold-free metrics are the right check here: they do not depend on
reproducing the validation calibration, so a match isolates the question that
matters -- does this forward pass produce the same scores as the original one.

Run:  PYTHONPATH=src:experiments/_archived python3 experiments/reproduce_check.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import routing_encoder as archived  # noqa: E402
from encoder_infer import (  # noqa: E402
    direct_route_probability,
    load_audit_members,
    load_member,
    score_texts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "artifacts" / "phase0_external_eval"


def build_direct_suite() -> list[dict]:
    """Rebuild the dev-test direct suite exactly as run_direct_failure_repair did.

    Mirrors routing_encoder.py:4869-4891.
    """
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    lineage_roles = archived._lineage_roles(DATA_DIR, manifest)
    records, _crossing = archived.load_capped_fold_records(
        DATA_DIR,
        manifest,
        "dev_test",
        fold_name=None,
        training=False,
        include_boundary=False,
        maximum=archived.CONSOLIDATED_MAX_PER_STRATUM,
        lineage_roles=lineage_roles,
    )
    return [
        record
        for record in records
        if record["channel"] == "direct_user" and "multi_turn" not in record["sources"]
    ]


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    audit = json.loads(
        (
            REPO_ROOT / "artifacts/direct_failure_repair_ensemble/ensemble-audit.json"
        ).read_text()
    )
    expected = audit["evaluation"]["full_direct_suite"][
        "ensemble:mean_direct_probability"
    ]["0.100%"]["all"]

    print("rebuilding dev-test direct suite ...", flush=True)
    suite = build_direct_suite()
    texts = [record["text"] for record in suite]
    labels = np.array(
        [
            1 if record["targets"]["direct_instruction_subversion"] == 1 else 0
            for record in suite
        ],
        dtype=np.int64,
    )
    known = np.array(
        [
            record["targets"]["direct_instruction_subversion"] is not None
            for record in suite
        ]
    )

    print(f"  rows={len(suite)}  known-label rows={int(known.sum())}")
    print(f"  expected rows={expected['rows']} "
          f"neg={expected['negative']} pos={expected['positive']}")

    member_probabilities = {}
    for member in load_audit_members():
        print(f"scoring with {member.name} ({member.model_id}) ...", flush=True)
        loaded = load_member(member)
        logits = score_texts(loaded, texts, channel="direct_user")
        member_probabilities[member.name] = direct_route_probability(logits)
        del loaded
        import torch

        torch.cuda.empty_cache()

    fused = np.mean(list(member_probabilities.values()), axis=0)

    from sklearn.metrics import average_precision_score, roc_auc_score

    mask = known
    results = {
        "rows_scored": int(mask.sum()),
        "positive": int(labels[mask].sum()),
        "negative": int((1 - labels[mask]).sum()),
        "reproduced": {
            "roc_auc": float(roc_auc_score(labels[mask], fused[mask])),
            "pr_auc": float(average_precision_score(labels[mask], fused[mask])),
        },
        "recorded": {
            "roc_auc": expected["roc_auc"],
            "pr_auc": expected["pr_auc"],
            "rows": expected["rows"],
            "positive": expected["positive"],
            "negative": expected["negative"],
        },
        "per_member": {},
    }
    for name, probabilities in member_probabilities.items():
        results["per_member"][name] = {
            "roc_auc": float(roc_auc_score(labels[mask], probabilities[mask])),
            "pr_auc": float(average_precision_score(labels[mask], probabilities[mask])),
        }

    delta_roc = results["reproduced"]["roc_auc"] - expected["roc_auc"]
    delta_pr = results["reproduced"]["pr_auc"] - expected["pr_auc"]
    results["delta"] = {"roc_auc": delta_roc, "pr_auc": delta_pr}

    print("\n=== reproduction check ===")
    print(f"rows scored     : {results['rows_scored']} "
          f"(recorded {expected['rows']})")
    print(f"ROC AUC         : {results['reproduced']['roc_auc']:.6f} "
          f"(recorded {expected['roc_auc']:.6f})  delta={delta_roc:+.6f}")
    print(f"PR  AUC         : {results['reproduced']['pr_auc']:.6f} "
          f"(recorded {expected['pr_auc']:.6f})  delta={delta_pr:+.6f}")

    tolerance = 0.002
    ok = abs(delta_roc) < tolerance and abs(delta_pr) < tolerance
    results["passed"] = bool(ok)
    results["tolerance"] = tolerance
    verdict = "PASS" if ok else "FAIL"
    print(f"\nverdict: {verdict} (tolerance {tolerance})")
    if not ok:
        print("The reconstruction does not match. Do not trust downstream numbers.")

    np.save(OUT_DIR / "dev_test_fused_scores.npy", fused)
    np.save(OUT_DIR / "dev_test_labels.npy", labels)
    (OUT_DIR / "reproduce_check.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT_DIR/'reproduce_check.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
