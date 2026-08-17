#!/usr/bin/env python3
"""Analyze matched standalone DeepSeek contracts from immutable ledgers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.pipeline_benchmark import hard_verdict as hard
from experiments.pipeline_benchmark import metrics

DEFAULT_OUTPUT = hard.DEFAULT_OUTPUT
SELECTION_NAME = "deepseek_standalone_selection.json"
EVALUATION_NAME = "deepseek_standalone_evaluation.json"
LOGPROB_KEY = ("cloudflare", "strict_logprob")
CLOUD_HARD_KEY = ("cloudflare", "strict_logprob")
TRUE_HARD_KEY = ("decart", "strict_hard_verdict")
SLICE_FIELDS = ("input_channel", "dataset")


def _inputs(output: Path) -> tuple[dict, dict, dict[str, Any]]:
    manifest, panel, _, _, identity = hard._selection_inputs(output)
    identity = dict(identity)
    identity["hard_helper_code_sha256"] = identity.pop("analysis_code_sha256")
    identity["analysis_code_sha256"] = hard._sha256(Path(__file__))
    return manifest, panel, identity


def _records(
    records: list[dict[str, Any]], key: tuple[str, str]
) -> list[dict[str, Any]]:
    provider, transport = key
    return [
        row
        for row in records
        if row["requested_provider"] == provider and row["transport"] == transport
    ]


def _logprob_values(
    ids: list[str], records: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    mapped = {row["row_id"]: row for row in records}
    scores = np.zeros(len(ids), dtype=np.float64)
    invalid = np.zeros(len(ids), dtype=bool)
    for index, row_id in enumerate(ids):
        row = mapped[row_id]
        value = row.get("probability") if row.get("status") == "ok" else None
        valid = (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0 <= float(value) <= 1
        )
        if valid:
            scores[index] = float(value)
        else:
            invalid[index] = True
    return scores, invalid


def _hard_values(
    ids: list[str], records: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    mapped = {row["row_id"]: row for row in records}
    predictions = np.ones(len(ids), dtype=bool)
    invalid = np.zeros(len(ids), dtype=bool)
    for index, row_id in enumerate(ids):
        row = mapped[row_id]
        value = row.get("verdict") if row.get("status") == "ok" else None
        valid = type(value) is int and value in (0, 1)
        if valid:
            predictions[index] = bool(value)
        else:
            invalid[index] = True
    return predictions, invalid


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    invalid: np.ndarray,
    target_fpr: float,
) -> float | None:
    """Select the lowest calibration threshold with invalids always positive."""
    if labels.shape != scores.shape or labels.shape != invalid.shape:
        raise ValueError("labels, scores, and invalid mask must align")
    valid_scores = scores[~invalid]
    candidates = np.unique(valid_scores)
    maximum = float(valid_scores.max()) if len(valid_scores) else 1.0
    candidates = np.append(candidates, np.nextafter(maximum, np.inf))
    negatives = labels == 0
    if not np.any(negatives):
        raise ValueError("fixed-FPR selection requires calibration negatives")
    for threshold in candidates:
        predictions = invalid | (scores >= threshold)
        if float(np.mean(predictions[negatives])) <= target_fpr:
            return float(threshold)
    return None


def _summary(
    rows: list[dict[str, Any]], predictions: np.ndarray, invalid: np.ndarray
) -> dict[str, Any]:
    result = metrics.summarize_slices(rows, predictions, slice_fields=SLICE_FIELDS)
    result["invalid_outputs"] = int(np.sum(invalid))
    result["invalid_output_rate"] = float(np.mean(invalid))
    result["invalids_fail_closed"] = True
    return result


def _calibration(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    invalid: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    fixed = {}
    for target in metrics.FPR_TARGETS:
        threshold = select_threshold(labels, scores, invalid, target)
        fixed[f"{target:g}"] = {
            "target_fpr": target,
            "threshold": threshold,
            "calibration": (
                None
                if threshold is None
                else _summary(rows, invalid | (scores >= threshold), invalid)
            ),
        }
    ranking = (
        metrics.score_metrics(labels, scores, 0.5)
        if not np.any(invalid)
        else {"auroc": None, "average_precision": None}
    )
    return {
        "fixed_fpr": fixed,
        "ranking": {
            "auroc": ranking["auroc"],
            "average_precision": ranking["average_precision"],
            "invalid_outputs": int(np.sum(invalid)),
        },
    }


def _paired(
    labels: np.ndarray,
    cloud_hard: np.ndarray,
    true_hard: np.ndarray,
    fixed: dict[str, np.ndarray],
) -> dict[str, Any]:
    result = {
        "decart_true_hard_minus_cloudflare_logprob_hard": metrics.paired_stratified_bootstrap_delta(
            labels, cloud_hard, true_hard
        )
    }
    for coordinate, predictions in fixed.items():
        result[f"cloudflare_fixed_{coordinate}_minus_cloudflare_hard"] = (
            metrics.paired_stratified_bootstrap_delta(labels, cloud_hard, predictions)
        )
        result[f"cloudflare_fixed_{coordinate}_minus_decart_true_hard"] = (
            metrics.paired_stratified_bootstrap_delta(labels, true_hard, predictions)
        )
    return result


def _select(output: Path) -> tuple[dict[str, Any] | None, str]:
    manifest, panel, identity = _inputs(output)
    ids = manifest["roles"]["provider_panel_ids"]
    if not set(ids) <= set(manifest["roles"]["provider_safe_calibration_panel_ids"]):
        raise ValueError("standalone selection is not calibration-only")
    complete = hard._complete_ledger(
        output, stage="panel", expected_ids=ids, panel=panel
    )
    if complete is None:
        return None, "provider panel ledger is incomplete"
    records, ledger_identity = complete
    cloud = _records(records, LOGPROB_KEY)
    decart = _records(records, TRUE_HARD_KEY)
    if len(cloud) != len(ids) or len(decart) != len(ids):
        return None, "standalone comparison contracts are incomplete"
    rows = hard._analysis_rows(panel, ids)
    scores, score_invalid = _logprob_values(ids, cloud)
    cloud_hard, cloud_hard_invalid = _hard_values(ids, cloud)
    true_hard, true_hard_invalid = _hard_values(ids, decart)
    return {
        "schema_version": 1,
        "advisory_only": True,
        "frozen_from": "provider-safe 1024-row calibration panel only",
        "contracts": {
            "cloudflare_logprob": {
                "provider": LOGPROB_KEY[0],
                "transport": LOGPROB_KEY[1],
                "scoring": "decision-token logprob probability",
            },
            "cloudflare_logprob_hard_verdict": {
                "provider": CLOUD_HARD_KEY[0],
                "transport": CLOUD_HARD_KEY[1],
                "scoring": "returned hard verdict from the logprob request",
            },
            "decart_true_no_logprob_hard_verdict": {
                "provider": TRUE_HARD_KEY[0],
                "transport": TRUE_HARD_KEY[1],
                "scoring": "strict hard verdict with no logprob request",
            },
        },
        "cloudflare_logprob": _calibration(rows, scores, score_invalid),
        "cloudflare_logprob_hard_verdict": _summary(
            rows, cloud_hard, cloud_hard_invalid
        ),
        "decart_true_no_logprob_hard_verdict": _summary(
            rows, true_hard, true_hard_invalid
        ),
        "inputs": identity | {"provider_panel": ledger_identity},
    }, "selection complete"


def _evaluate(output: Path, selection: dict[str, Any]) -> tuple[dict | None, str]:
    manifest, panel, identity = _inputs(output)
    if any(
        selection.get("inputs", {}).get(key) != value for key, value in identity.items()
    ):
        raise ValueError("standalone selection inputs changed")
    ids = manifest["roles"]["provider_safe_evaluation_panel_ids"]
    complete = hard._complete_ledger(
        output, stage="evaluation", expected_ids=ids, panel=panel
    )
    if complete is None:
        return None, "provider-safe evaluation ledger is incomplete"
    records, ledger_identity = complete
    cloud = _records(records, LOGPROB_KEY)
    decart = _records(records, TRUE_HARD_KEY)
    if len(cloud) != len(ids) or len(decart) != len(ids):
        return None, "standalone evaluation contracts are incomplete"
    rows = hard._analysis_rows(panel, ids)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    scores, score_invalid = _logprob_values(ids, cloud)
    cloud_hard, cloud_hard_invalid = _hard_values(ids, cloud)
    true_hard, true_hard_invalid = _hard_values(ids, decart)
    fixed_predictions = {}
    fixed_results = {}
    for coordinate, selected in selection["cloudflare_logprob"]["fixed_fpr"].items():
        threshold = selected["threshold"]
        if threshold is None:
            fixed_results[coordinate] = None
            continue
        predictions = score_invalid | (scores >= threshold)
        fixed_predictions[coordinate] = predictions
        fixed_results[coordinate] = {
            "target_fpr": selected["target_fpr"],
            "threshold": threshold,
            "evaluation": _summary(rows, predictions, score_invalid),
        }
    ranking = (
        metrics.score_metrics(labels, scores, 0.5)
        if not np.any(score_invalid)
        else {"auroc": None, "average_precision": None}
    )
    return {
        "schema_version": 1,
        "advisory_only": True,
        "frozen_selection_sha256": hard._sha256(output / SELECTION_NAME),
        "rows": len(rows),
        "contracts": selection["contracts"],
        "cloudflare_logprob": {
            "fixed_fpr": fixed_results,
            "ranking": {
                "auroc": ranking["auroc"],
                "average_precision": ranking["average_precision"],
                "invalid_outputs": int(np.sum(score_invalid)),
            },
        },
        "cloudflare_logprob_hard_verdict": _summary(
            rows, cloud_hard, cloud_hard_invalid
        ),
        "decart_true_no_logprob_hard_verdict": _summary(
            rows, true_hard, true_hard_invalid
        ),
        "paired_bootstrap_deltas": _paired(
            labels, cloud_hard, true_hard, fixed_predictions
        ),
        "inputs": identity | {"provider_evaluation": ledger_identity},
    }, "evaluation complete"


def analyze(output: Path) -> str:
    selection_path = output / SELECTION_NAME
    if selection_path.exists():
        selection = hard._json(selection_path)
    else:
        selection, status = _select(output)
        if selection is None:
            return f"pending: {status}"
        hard._write_once(selection_path, selection)
    evaluation_path = output / EVALUATION_NAME
    if evaluation_path.exists():
        return "complete: standalone selection and evaluation already exist"
    evaluation, status = _evaluate(output, selection)
    if evaluation is None:
        return f"pending: selection frozen; {status}"
    hard._write_once(evaluation_path, evaluation)
    return "complete: standalone selection and evaluation frozen"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(analyze(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
