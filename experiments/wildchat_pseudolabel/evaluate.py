#!/usr/bin/env python3
"""Train available WildChat-negative ablations on fixed base validation groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import vstack

from vulsight_guard.data import deduplicate, normalize_text, read_jsonl
from vulsight_guard.detector import (
    DIRECT_OPERATING_FPR_BUDGETS,
    DIRECT_PRECISION_FLOORS,
    DIRECT_REVIEW_PRECISION_FLOOR,
    _char_pipeline,
    _rates,
    _score_paragraphs,
    choose_threshold,
    choose_threshold_for_precision,
    split_fit_validation,
)

from ablate import TARGETS, select, validate_accepted, weights
from label import load_sample


def sha256(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def candidate_sets(accepted: list[dict]) -> list[tuple[str, list[dict]]]:
    output = [("zero", [])]
    if accepted:
        output.append((f"pilot_all_{len(accepted)}", select(accepted, len(accepted))))
    for target in TARGETS:
        if target <= len(accepted) and target != len(accepted):
            output.append((f"accepted_{target}", select(accepted, target)))
    return output


def train_candidate(
    base: list[dict], weak: list[dict]
) -> tuple[object, dict, list[dict]]:
    fit, validation = split_fit_validation(base)
    positive = sum(row["label"] == 1 for row in fit)
    negative = sum(row["label"] == 0 for row in fit)
    policy = weights(positive, negative, len(weak))
    model = _char_pipeline()
    model.set_params(classifier__class_weight=None)
    sample_weight = [
        policy["base_positive_per_row"]
        if row["label"] == 1
        else policy["base_negative_per_row"]
        for row in fit
    ] + [policy["weak_negative_per_row"]] * len(weak)
    fit_texts = [normalize_text(row["text"]) for row in fit]
    weak_texts = [normalize_text(row["text"]) for row in weak]
    started = time.perf_counter()
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    fit_matrix = vectorizer.fit_transform(fit_texts)
    matrix = (
        vstack([fit_matrix, vectorizer.transform(weak_texts)], format="csr")
        if weak
        else fit_matrix
    )
    classifier.fit(
        matrix,
        [row["label"] for row in fit] + [row["label"] for row in weak],
        sample_weight=np.asarray(sample_weight, dtype=np.float64),
    )
    fit_seconds = time.perf_counter() - started
    validation_scores = model.predict_proba(
        [normalize_text(row["text"]) for row in validation]
    )[:, 1]
    labels = [row["label"] for row in validation]
    precision_profiles = []
    for floor in DIRECT_PRECISION_FLOORS:
        threshold = choose_threshold_for_precision(labels, validation_scores, floor)
        precision_profiles.append(
            {
                "min_validation_precision": floor,
                "threshold": threshold,
                "validation": _rates(validation, validation_scores, threshold),
                "role": (
                    "shadow_review_candidate"
                    if floor == DIRECT_REVIEW_PRECISION_FLOOR
                    else "diagnostic"
                ),
            }
        )
    fpr_operating_points = []
    for budget in DIRECT_OPERATING_FPR_BUDGETS:
        threshold = choose_threshold(labels, validation_scores, budget)
        fpr_operating_points.append(
            {
                "validation_fpr_budget": budget,
                "threshold": threshold,
                "validation": _rates(validation, validation_scores, threshold),
                "role": "diagnostic",
            }
        )
    review = next(
        point
        for point in precision_profiles
        if point["min_validation_precision"] == DIRECT_REVIEW_PRECISION_FLOOR
    )
    training = {
        "base_fit_rows": len(fit),
        "weak_fit_rows": len(weak),
        "validation_rows": len(validation),
        "validation_weak_rows": 0,
        "tfidf_fit_rows": len(fit),
        "weak_rows_in_tfidf_fit": 0,
        "threshold": review["threshold"],
        "threshold_role": "shadow_review_candidate",
        "precision_profiles": precision_profiles,
        "fpr_operating_points": fpr_operating_points,
        "weights": policy,
        "fit_seconds": fit_seconds,
    }
    return model, training, validation


def evaluation_sets(data_dir: Path) -> dict[str, list[dict]]:
    names = (
        "toxic_chat",
        "prompt_injections",
        "multi_turn",
        "notinject",
        "oasst1_chat",
        "oasst1_position_stress",
        "xstest",
        "harmbench",
        "do_not_answer",
        "jailbreaks_over_time",
        "bipia_clean_context",
        "bipia_payload",
        "bipia_context",
        "tensor_trust_attack",
        "tensor_trust_context",
    )
    sets = {name: read_jsonl(data_dir / f"{name}.jsonl") for name in names}
    sets["external_hard_negatives"], _ = deduplicate(
        sets["oasst1_chat"]
        + sets["oasst1_position_stress"]
        + sets["xstest"]
        + sets["harmbench"]
        + sets["do_not_answer"]
        + sets["notinject"]
    )
    return sets


def subgroup_alerts(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    alerts = scores >= threshold
    output = {}
    for field in (
        "language",
        "length_bucket",
        "source_toxic",
        "topic",
        "security_trigger",
    ):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[str(row[field])].append(index)
        output[field] = {
            value: {
                "rows": len(indices),
                "alerts": int(alerts[indices].sum()),
                "alert_rate": float(alerts[indices].mean()),
            }
            for value, indices in sorted(groups.items())
        }
    return output


def run_candidate(
    name: str,
    model: object,
    training: dict,
    sets: dict[str, list[dict]],
    sample_holdout_rows: list[dict],
    indirect_artifact: dict,
) -> dict:
    direct_scores = {
        set_name: model.predict_proba([normalize_text(row["text"]) for row in rows])[
            :, 1
        ]
        for set_name, rows in sets.items()
    }
    direct_names = (
        "toxic_chat",
        "prompt_injections",
        "multi_turn",
        "external_hard_negatives",
        "notinject",
        "oasst1_position_stress",
        "jailbreaks_over_time",
        "tensor_trust_attack",
    )
    indirect_sensor = indirect_artifact["channels"]["untrusted_content"]
    indirect_threshold = float(indirect_sensor["threshold"])
    indirect_scores = {}
    for set_name in (
        "bipia_clean_context",
        "bipia_payload",
        "bipia_context",
        "tensor_trust_context",
    ):
        rows = sets[set_name]
        indirect_scores[set_name] = _score_paragraphs(
            indirect_sensor["model"], [row["text"] for row in rows]
        )
    sample_scores = model.predict_proba(
        [normalize_text(row["text"]) for row in sample_holdout_rows]
    )[:, 1]
    profiles = [
        {
            **point,
            "selection_kind": "min_validation_precision",
            "selection_value": point["min_validation_precision"],
        }
        for point in training["precision_profiles"]
    ] + [
        {
            **point,
            "selection_kind": "validation_fpr_budget",
            "selection_value": point["validation_fpr_budget"],
        }
        for point in training["fpr_operating_points"]
    ]
    evaluations = []
    for operating_point in profiles:
        threshold = operating_point["threshold"]
        combined = {}
        for set_name, scores in indirect_scores.items():
            elevated = np.logical_or(
                direct_scores[set_name] >= threshold,
                scores >= indirect_threshold,
            )
            combined[set_name] = _rates(sets[set_name], elevated.astype(float), 0.5)
        evaluations.append(
            {
                "selection_kind": operating_point["selection_kind"],
                "selection_value": operating_point["selection_value"],
                "threshold": threshold,
                "role": operating_point["role"],
                "direct_sets": {
                    set_name: _rates(sets[set_name], direct_scores[set_name], threshold)
                    for set_name in direct_names
                },
                "untrusted_content_combined": combined,
                "wildchat_nonaccepted_holdout_alerts": {
                    "rows": len(sample_holdout_rows),
                    "alerts": int((sample_scores >= threshold).sum()),
                    "alert_rate": float((sample_scores >= threshold).mean()),
                    "by_stratum": subgroup_alerts(
                        sample_holdout_rows, sample_scores, threshold
                    ),
                    "metric_note": (
                        "excludes every accepted weak row used by any candidate; "
                        "rejected or unavailable judgments are not benign ground truth, "
                        "so this is alert rate rather than FPR"
                    ),
                },
            }
        )
    return {
        "name": name,
        "training": training,
        "profiles": evaluations,
    }


DIRECT_ATTACK_SETS = (
    "toxic_chat",
    "prompt_injections",
    "multi_turn",
    "jailbreaks_over_time",
    "tensor_trust_attack",
)
INDIRECT_ATTACK_SETS = ("bipia_payload", "bipia_context", "tensor_trust_context")
NORMAL_SETS = ("external_hard_negatives", "notinject", "oasst1_position_stress")


def _point(result: dict, kind: str, value: float) -> dict:
    return next(
        point
        for point in result["profiles"]
        if point["selection_kind"] == kind and point["selection_value"] == value
    )


def _macro_recall(metrics: dict, names: tuple[str, ...]) -> float:
    values = [metrics[name]["recall"] for name in names]
    if any(value is None for value in values):
        raise ValueError("attack assessment set has no positives")
    return float(np.mean(values))


def scale_assessment(results: list[dict]) -> dict:
    """Apply the predeclared conservative pilot gate at every operating point."""
    if len(results) < 2:
        return {"decision": "not_available", "reason": "no weak-data candidate"}
    baseline = results[0]
    comparisons = []
    for candidate in results[1:]:
        for floor in DIRECT_PRECISION_FLOORS:
            base_point = _point(baseline, "min_validation_precision", floor)
            candidate_point = _point(candidate, "min_validation_precision", floor)
            direct_delta = _macro_recall(
                candidate_point["direct_sets"], DIRECT_ATTACK_SETS
            ) - _macro_recall(base_point["direct_sets"], DIRECT_ATTACK_SETS)
            indirect_delta = _macro_recall(
                candidate_point["untrusted_content_combined"], INDIRECT_ATTACK_SETS
            ) - _macro_recall(
                base_point["untrusted_content_combined"], INDIRECT_ATTACK_SETS
            )
            false_positive_deltas = {
                name: (
                    candidate_point["direct_sets"][name]["false_positive"]
                    - base_point["direct_sets"][name]["false_positive"]
                )
                for name in NORMAL_SETS
            }
            passes = (
                direct_delta > 0
                and indirect_delta >= 0
                and all(delta <= 0 for delta in false_positive_deltas.values())
            )
            comparisons.append(
                {
                    "candidate": candidate["name"],
                    "min_validation_precision": floor,
                    "direct_attack_macro_recall_delta": direct_delta,
                    "indirect_attack_macro_recall_delta": indirect_delta,
                    "normal_set_false_positive_deltas": false_positive_deltas,
                    "passes": passes,
                }
            )
    review_passes = [
        row
        for row in comparisons
        if row["min_validation_precision"] == DIRECT_REVIEW_PRECISION_FLOOR
        and row["passes"]
    ]
    return {
        "decision": "scale" if review_passes else "stop",
        "review_min_validation_precision": DIRECT_REVIEW_PRECISION_FLOOR,
        "criteria": {
            "direct_attack_macro_recall_delta": "> 0",
            "indirect_attack_macro_recall_delta": ">= 0",
            "false_positive_delta_on_each_predeclared_normal_set": "<= 0",
            "uncertainty_note": (
                "deterministic paired development comparison; small deltas are not "
                "claimed statistically significant"
            ),
        },
        "attack_macro_sets": {
            "direct": list(DIRECT_ATTACK_SETS),
            "indirect": list(INDIRECT_ATTACK_SETS),
        },
        "normal_sets": list(NORMAL_SETS),
        "comparisons": comparisons,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = Path(__file__).with_name("outputs")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--accepted", type=Path, default=output_dir / "accepted.jsonl")
    parser.add_argument("--sample", type=Path, default=output_dir / "pilot_5k.jsonl")
    parser.add_argument(
        "--artifact", type=Path, default=root / "artifacts/guard_bundle.joblib"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "reports/wildchat-ablation-results.json"
    )
    args = parser.parse_args()
    base = read_jsonl(args.data_dir / "train.jsonl")
    accepted = read_jsonl(args.accepted)
    validate_accepted(accepted)
    sample_rows, sample_hash = load_sample(args.sample)
    accepted_ids = {row["sample_id"] for row in accepted}
    sample_holdout_rows = [
        row for row in sample_rows if row["sample_id"] not in accepted_ids
    ]
    if not sample_holdout_rows:
        raise ValueError(
            "no nonaccepted WildChat rows remain for a fit-independent diagnostic"
        )
    sets = evaluation_sets(args.data_dir)
    artifact = joblib.load(args.artifact)
    results = []
    for name, weak in candidate_sets(accepted):
        model, training, _ = train_candidate(base, weak)
        results.append(
            run_candidate(name, model, training, sets, sample_holdout_rows, artifact)
        )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sample_sha256": sample_hash,
        "sample_rows": len(sample_rows),
        "sample_nonaccepted_holdout_rows": len(sample_holdout_rows),
        "accepted_rows_available": len(accepted),
        "accepted_sha256": sha256(args.accepted.read_bytes()),
        "base_sha256": sha256((args.data_dir / "train.jsonl").read_bytes()),
        "detector_artifact_sha256": sha256(args.artifact.read_bytes()),
        "protocol": {
            "weak_rows_fit_only": True,
            "weak_rows_in_threshold_calibration": 0,
            "weak_rows_in_tfidf_fit": 0,
            "all_accepted_rows_excluded_from_wildchat_diagnostic": True,
            "validation_precision_floors": list(DIRECT_PRECISION_FLOORS),
            "review_min_validation_precision": DIRECT_REVIEW_PRECISION_FLOOR,
            "precision_note": (
                "validation source-mixture precision is not expected production precision"
            ),
            "validation_fpr_budgets": list(DIRECT_OPERATING_FPR_BUDGETS),
            "fpr_budget_note": "diagnostic development points, not production targets",
            "weak_share_of_total_negative_weight": 0.1,
            "class_weight": None,
            "public_suites": "frozen repeated development comparisons",
        },
        "candidates": results,
        "scale_assessment": scale_assessment(results),
        "metric_status": "weak-label development ablation; never production ground truth",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "candidates": [item["name"] for item in results],
                "accepted_rows_available": len(accepted),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
