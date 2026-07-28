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

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoder_infer import (  # noqa: E402
    direct_route_probability,
    load_audit_members,
    load_member,
    score_texts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "artifacts" / "phase0_external_eval"
ROUTE_POSITIVE_LABELS = {
    "direct_jailbreak",
    "direct_prompt_injection",
    "indirect_prompt_injection",
}


def _load_archived_runner():
    import routing_encoder

    return routing_encoder


def build_direct_suite() -> list[dict]:
    """Rebuild the dev-test direct suite exactly as run_direct_failure_repair did.

    Mirrors routing_encoder.py:4869-4891.
    """
    archived = _load_archived_runner()
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


def direct_route_rows(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return selected row indices and labels used by direct-route evaluation."""
    selected = []
    labels = []
    for index, record in enumerate(records):
        security_label = record["security_label"]
        if security_label in {"benign", "harmful_non_injection"}:
            label = 0
        elif security_label in ROUTE_POSITIVE_LABELS:
            label = 1
        else:
            continue
        selected.append(index)
        labels.append(label)
    return np.asarray(selected, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def main() -> int:
    archived = _load_archived_runner()
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
    selected, labels = direct_route_rows(suite)
    texts = [record["text"] for record in suite]

    manifest_sha256 = hashlib.sha256(
        (DATA_DIR / "manifest.json").read_bytes()
    ).hexdigest()
    expected_manifest_sha256 = audit["data_manifest_sha256"]
    suite_sha256 = archived.selected_rows_sha256(suite)
    expected_suite_sha256 = audit["selection"]["selected_rows_sha256"][
        "full_direct_suite"
    ]
    provenance_matches = (
        manifest_sha256 == expected_manifest_sha256
        and suite_sha256 == expected_suite_sha256
    )

    print(f"  suite rows={len(suite)}  route-labelled rows={len(selected)}")
    print(
        f"  expected rows={expected['rows']} "
        f"neg={expected['negative']} pos={expected['positive']}"
    )
    print(
        f"  manifest match={manifest_sha256 == expected_manifest_sha256} "
        f"suite hash match={suite_sha256 == expected_suite_sha256}"
    )
    if not provenance_matches:
        print("provenance mismatch: refusing to score a different evaluation suite")
        return 1

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
    selected_fused = fused[selected]

    from sklearn.metrics import average_precision_score, roc_auc_score

    results = {
        "suite_rows": len(suite),
        "data_manifest_sha256": manifest_sha256,
        "expected_data_manifest_sha256": expected_manifest_sha256,
        "suite_sha256": suite_sha256,
        "expected_suite_sha256": expected_suite_sha256,
        "provenance_matches": provenance_matches,
        "rows_scored": len(labels),
        "positive": int(labels.sum()),
        "negative": int((1 - labels).sum()),
        "reproduced": {
            "roc_auc": float(roc_auc_score(labels, selected_fused)),
            "pr_auc": float(average_precision_score(labels, selected_fused)),
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
        selected_probabilities = probabilities[selected]
        recorded_member = audit["evaluation"]["full_direct_suite"][
            f"member:{name}:direct"
        ]["0.100%"]["all"]
        results["per_member"][name] = {
            "roc_auc": float(roc_auc_score(labels, selected_probabilities)),
            "pr_auc": float(average_precision_score(labels, selected_probabilities)),
            "recorded_roc_auc": recorded_member["roc_auc"],
            "recorded_pr_auc": recorded_member["pr_auc"],
        }

    delta_roc = results["reproduced"]["roc_auc"] - expected["roc_auc"]
    delta_pr = results["reproduced"]["pr_auc"] - expected["pr_auc"]
    results["delta"] = {"roc_auc": delta_roc, "pr_auc": delta_pr}

    print("\n=== reproduction check ===")
    print(f"rows scored     : {results['rows_scored']} (recorded {expected['rows']})")
    print(
        f"ROC AUC         : {results['reproduced']['roc_auc']:.6f} "
        f"(recorded {expected['roc_auc']:.6f})  delta={delta_roc:+.6f}"
    )
    print(
        f"PR  AUC         : {results['reproduced']['pr_auc']:.6f} "
        f"(recorded {expected['pr_auc']:.6f})  delta={delta_pr:+.6f}"
    )

    tolerance = 0.0001
    counts_match = (
        results["rows_scored"] == expected["rows"]
        and results["positive"] == expected["positive"]
        and results["negative"] == expected["negative"]
    )
    results["counts_match"] = counts_match
    members_match = all(
        abs(metrics["roc_auc"] - metrics["recorded_roc_auc"]) < tolerance
        and abs(metrics["pr_auc"] - metrics["recorded_pr_auc"]) < tolerance
        for metrics in results["per_member"].values()
    )
    results["members_match"] = members_match
    ok = (
        counts_match
        and members_match
        and abs(delta_roc) < tolerance
        and abs(delta_pr) < tolerance
    )
    results["passed"] = bool(ok)
    results["tolerance"] = tolerance
    verdict = "PASS" if ok else "FAIL"
    print(f"\nverdict: {verdict} (tolerance {tolerance})")
    if not ok:
        print("The reconstruction does not match. Do not trust downstream numbers.")

    np.save(OUT_DIR / "dev_test_fused_scores.npy", selected_fused)
    np.save(OUT_DIR / "dev_test_labels.npy", labels)
    np.save(OUT_DIR / "dev_test_selected_indices.npy", selected)
    (OUT_DIR / "reproduce_check.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT_DIR / 'reproduce_check.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
