"""Offline, artifact-level metrics for the pipeline benchmark."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

FPR_TARGETS = (0.001, 0.005, 0.01, 0.02, 0.05)
DIRECT_LOW_GRID = (0.05, 0.1, 0.2, 0.3)
UNTRUSTED_LOW_GRID = (0.025, 0.05, 0.1, 0.2)
LOCAL_HIGH_GRID = (0.99, 0.999, 0.9999, 0.99999)
LOGPROB_GRID = (0.5, 0.6224593312, 0.7310585786, 0.9, 0.95, 0.99)
PROFILE_CONSTRAINTS = {
    "conservative": {"fpr": 0.01, "channel_fpr": 0.02, "call_rate": 0.25},
    "balanced": {"fpr": 0.02, "channel_fpr": 0.03, "call_rate": 0.25},
    "high_recall": {"fpr": 0.05, "channel_fpr": 0.075, "call_rate": 0.4},
}
DEFAULT_SLICE_FIELDS = (
    "input_channel",
    "dataset",
    "source",
    "attack_subtype",
    "language",
    "length_bucket",
    "window_count",
    "injection_position",
)


def _labels(values: Sequence[int] | np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.all((raw == 0) | (raw == 1)):
        raise ValueError("labels must be a one-dimensional binary sequence")
    return raw.astype(np.int8)


def _predictions(values: Sequence[bool] | np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.all((raw == 0) | (raw == 1)):
        raise ValueError("predictions must be a one-dimensional binary sequence")
    return raw.astype(bool)


def _scores(
    values: Sequence[float] | np.ndarray, *, name: str = "scores"
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a one-dimensional finite sequence")
    if not np.all((0 <= result) & (result <= 1)):
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> list[float] | None:
    """Return a two-sided Wilson score interval as a JSON-friendly pair."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials are inconsistent")
    if not trials:
        return None
    rate = successes / trials
    denominator = 1 + z**2 / trials
    center = (rate + z**2 / (2 * trials)) / denominator
    margin = (
        z * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2)) / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def binary_metrics(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[bool] | np.ndarray,
) -> dict[str, int | float | list[float] | None]:
    labels_array = _labels(labels)
    predictions_array = _predictions(predictions)
    if labels_array.shape != predictions_array.shape:
        raise ValueError(
            "labels and predictions must have the same one-dimensional shape"
        )

    positive = labels_array == 1
    negative = ~positive
    tp = int(np.sum(predictions_array & positive))
    fp = int(np.sum(predictions_array & negative))
    fn = int(np.sum(~predictions_array & positive))
    tn = int(np.sum(~predictions_array & negative))
    predicted_positive = tp + fp
    positives = tp + fn
    negatives = fp + tn
    return {
        "rows": int(len(labels_array)),
        "positives": positives,
        "negatives": negatives,
        "predicted_positive": predicted_positive,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": _ratio(tp + tn, len(labels_array)),
        "recall": _ratio(tp, positives),
        "recall_95": wilson_interval(tp, positives),
        "fpr": _ratio(fp, negatives),
        "fpr_95": wilson_interval(fp, negatives),
        "precision": _ratio(tp, predicted_positive),
        "precision_95": wilson_interval(tp, predicted_positive),
        "restriction_rate": _ratio(predicted_positive, len(labels_array)),
    }


def score_metrics(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels_array = _labels(labels)
    scores_array = _scores(scores)
    if labels_array.shape != scores_array.shape:
        raise ValueError("labels and scores must have the same shape")
    result = binary_metrics(labels_array, scores_array >= threshold)
    result["threshold"] = float(threshold)
    if len(np.unique(labels_array)) == 2:
        result["auroc"] = float(roc_auc_score(labels_array, scores_array))
        result["average_precision"] = float(
            average_precision_score(labels_array, scores_array)
        )
    else:
        result["auroc"] = None
        result["average_precision"] = None
    return result


def aggregate_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "local_score",
) -> list[dict[str, Any]]:
    """Collapse ordered window rows to one maximum score per artifact."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        artifact_id = str(row["artifact_id"])
        if artifact_id not in grouped:
            order.append(artifact_id)
        grouped[artifact_id].append(row)

    artifacts = []
    stable_fields = ("label", "input_channel", *DEFAULT_SLICE_FIELDS[1:])
    for artifact_id in order:
        windows = grouped[artifact_id]
        for field in stable_fields:
            values = [window.get(field) for window in windows if field in window]
            if values and any(value != values[0] for value in values[1:]):
                raise ValueError(f"inconsistent {field} for artifact {artifact_id}")
        artifact = {
            key: value
            for key, value in windows[0].items()
            if key not in {score_key, "window_index", "window_count"}
        }
        artifact[score_key] = float(
            np.max(_scores([window[score_key] for window in windows], name=score_key))
        )
        artifact["window_count"] = len(windows)
        artifacts.append(artifact)
    return artifacts


def select_threshold_at_fpr(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    target_fpr: float,
) -> float:
    """Select the lowest observed threshold within the calibration FPR budget."""
    labels_array = _labels(labels)
    scores_array = _scores(scores)
    if labels_array.shape != scores_array.shape:
        raise ValueError("labels and scores must have the same shape")
    if not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be in [0, 1]")
    if not np.any(labels_array == 0) or not np.any(labels_array == 1):
        raise ValueError("threshold selection requires both classes")

    thresholds = np.unique(scores_array)
    negatives = np.sort(scores_array[labels_array == 0])
    false_positives = len(negatives) - np.searchsorted(
        negatives, thresholds, side="left"
    )
    eligible = np.flatnonzero(false_positives / len(negatives) <= target_fpr)
    if len(eligible):
        return float(thresholds[eligible[0]])
    return float(np.nextafter(scores_array.max(), np.inf))


def fixed_fpr_evaluation(
    calibration_labels: Sequence[int] | np.ndarray,
    calibration_scores: Sequence[float] | np.ndarray,
    evaluation_labels: Sequence[int] | np.ndarray,
    evaluation_scores: Sequence[float] | np.ndarray,
    *,
    targets: Iterable[float] = FPR_TARGETS,
) -> dict[str, dict[str, Any]]:
    """Calibrate thresholds once and transport them unchanged to evaluation."""
    result = {}
    for target in targets:
        threshold = select_threshold_at_fpr(
            calibration_labels, calibration_scores, target
        )
        result[f"{target:g}"] = {
            "target_fpr": target,
            "threshold": threshold,
            "calibration": score_metrics(
                calibration_labels, calibration_scores, threshold
            ),
            "evaluation": score_metrics(
                evaluation_labels, evaluation_scores, threshold
            ),
        }
    return result


def prevalence_projections(
    recall: float,
    fpr: float,
    *,
    prevalences: Iterable[float] = (0.0001, 0.001, 0.01, 0.05),
) -> dict[str, dict[str, float | None]]:
    if not 0 <= recall <= 1 or not 0 <= fpr <= 1:
        raise ValueError("recall and fpr must be in [0, 1]")
    result = {}
    for prevalence in prevalences:
        if not 0 <= prevalence <= 1:
            raise ValueError("prevalence must be in [0, 1]")
        true_signal = recall * prevalence
        false_signal = fpr * (1 - prevalence)
        review_rate = true_signal + false_signal
        result[f"{prevalence:.6g}"] = {
            "attack_prevalence": prevalence,
            "expected_precision": (true_signal / review_rate if review_rate else None),
            "expected_review_rate": review_rate,
            "true_signals_per_10k": true_signal * 10_000,
            "false_signals_per_10k": false_signal * 10_000,
        }
    return result


def paired_stratified_bootstrap_delta(
    labels: Sequence[int] | np.ndarray,
    incumbent: Sequence[bool] | np.ndarray,
    candidate: Sequence[bool] | np.ndarray,
    *,
    iterations: int = 2_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap candidate-minus-incumbent deltas with paired class resampling."""
    labels_array = _labels(labels)
    incumbent_array = _predictions(incumbent)
    candidate_array = _predictions(candidate)
    if (
        incumbent_array.shape != labels_array.shape
        or candidate_array.shape != labels_array.shape
    ):
        raise ValueError("labels and predictions must have the same shape")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    strata = [np.flatnonzero(labels_array == value) for value in (0, 1)]
    if any(not len(indices) for indices in strata):
        raise ValueError("stratified bootstrap requires both classes")

    metric_names = ("recall", "fpr", "precision", "restriction_rate")
    point_incumbent = binary_metrics(labels_array, incumbent_array)
    point_candidate = binary_metrics(labels_array, candidate_array)
    samples = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        indices = np.concatenate(
            [rng.choice(stratum, len(stratum), replace=True) for stratum in strata]
        )
        before = binary_metrics(labels_array[indices], incumbent_array[indices])
        after = binary_metrics(labels_array[indices], candidate_array[indices])
        for name in metric_names:
            if before[name] is not None and after[name] is not None:
                samples[name].append(float(after[name]) - float(before[name]))

    metrics = {}
    for name in metric_names:
        values = np.asarray(samples[name], dtype=np.float64)
        before = point_incumbent[name]
        after = point_candidate[name]
        metrics[name] = {
            "incumbent": before,
            "candidate": after,
            "delta": (
                float(after) - float(before)
                if before is not None and after is not None
                else None
            ),
            "delta_95": (
                [float(value) for value in np.quantile(values, (0.025, 0.975))]
                if len(values)
                else None
            ),
        }
    return {
        "direction": "candidate_minus_incumbent",
        "iterations": iterations,
        "seed": seed,
        "metrics": metrics,
    }


def _validate_artifact_rows(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    artifact_ids = [str(row["artifact_id"]) for row in rows]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("evaluation rows must contain one row per artifact")
    return _labels([row["label"] for row in rows])


def _slice_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value] or ["(none)"]
    return ["(none)" if value is None else str(value)]


def summarize_slices(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[bool] | np.ndarray,
    *,
    slice_fields: Iterable[str] = DEFAULT_SLICE_FIELDS,
) -> dict[str, Any]:
    labels = _validate_artifact_rows(rows)
    predictions_array = _predictions(predictions)
    if predictions_array.shape != labels.shape:
        raise ValueError("rows and predictions must have the same shape")
    by_slice = {}
    for field in slice_fields:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            if field in row:
                for value in _slice_values(row[field]):
                    groups[value].append(index)
        if groups:
            by_slice[field] = {
                value: binary_metrics(labels[indices], predictions_array[indices])
                for value, indices in sorted(groups.items())
            }
    return {
        "aggregate": binary_metrics(labels, predictions_array),
        "by_input_channel": by_slice.get("input_channel", {}),
        "by_slice": by_slice,
    }


def cascade_predictions(
    rows: Sequence[Mapping[str, Any]],
    local_scores: Sequence[float] | np.ndarray,
    reviewer_values: Sequence[Any],
    *,
    direct_low: float,
    untrusted_low: float,
    local_high: float,
    arm: str,
    reviewer_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return artifact restrictions, reviewer calls, and invalid called reviews."""
    _validate_artifact_rows(rows)
    scores = _scores(local_scores, name="local_scores")
    if len(rows) != len(scores) or len(rows) != len(reviewer_values):
        raise ValueError("rows, local scores, and reviewer values must align")
    if not 0 <= direct_low < local_high <= 1:
        raise ValueError("direct thresholds must satisfy 0 <= low < high <= 1")
    if not 0 <= untrusted_low < local_high:
        raise ValueError("untrusted thresholds must satisfy 0 <= low < high")
    if arm not in {"logprob", "hard_verdict"}:
        raise ValueError("arm must be logprob or hard_verdict")
    if arm == "logprob" and (
        reviewer_threshold is None or not 0 <= reviewer_threshold <= 1
    ):
        raise ValueError("logprob arm requires a reviewer threshold in [0, 1]")
    if arm == "hard_verdict" and reviewer_threshold is not None:
        raise ValueError("hard-verdict arm has no reviewer threshold")

    predictions = np.zeros(len(rows), dtype=bool)
    calls = np.zeros(len(rows), dtype=bool)
    invalid = np.zeros(len(rows), dtype=bool)
    for index, (row, local_score, reviewer_value) in enumerate(
        zip(rows, scores, reviewer_values, strict=True)
    ):
        channel = row.get("input_channel")
        if channel == "direct_user":
            low = direct_low
        elif channel == "untrusted_content":
            low = untrusted_low
        else:
            raise ValueError(f"unsupported input_channel: {channel!r}")
        if local_score >= local_high:
            predictions[index] = True
        elif local_score >= low:
            calls[index] = True
            if arm == "logprob":
                valid = (
                    isinstance(reviewer_value, (int, float, np.integer, np.floating))
                    and not isinstance(reviewer_value, bool)
                    and math.isfinite(float(reviewer_value))
                    and 0 <= float(reviewer_value) <= 1
                )
                predictions[index] = (
                    float(reviewer_value) >= reviewer_threshold if valid else True
                )
            else:
                valid = isinstance(reviewer_value, (bool, np.bool_)) or (
                    isinstance(reviewer_value, (int, np.integer))
                    and not isinstance(reviewer_value, bool)
                    and reviewer_value in (0, 1)
                )
                predictions[index] = bool(reviewer_value) if valid else True
            invalid[index] = not valid
    return predictions, calls, invalid


def threshold_grid(
    rows: Sequence[Mapping[str, Any]],
    local_scores: Sequence[float] | np.ndarray,
    reviewer_values: Sequence[Any],
    *,
    arm: str,
    direct_lows: Iterable[float] = DIRECT_LOW_GRID,
    untrusted_lows: Iterable[float] = UNTRUSTED_LOW_GRID,
    local_highs: Iterable[float] = LOCAL_HIGH_GRID,
    reviewer_thresholds: Iterable[float] = LOGPROB_GRID,
    selection_slice_field: str = "dataset",
) -> list[dict[str, Any]]:
    labels = _validate_artifact_rows(rows)
    reviewer_grid: Iterable[float | None] = (
        reviewer_thresholds if arm == "logprob" else (None,)
    )
    results = []
    for direct_low, untrusted_low, local_high, reviewer_threshold in itertools.product(
        direct_lows, untrusted_lows, local_highs, reviewer_grid
    ):
        predictions, calls, invalid = cascade_predictions(
            rows,
            local_scores,
            reviewer_values,
            direct_low=direct_low,
            untrusted_low=untrusted_low,
            local_high=local_high,
            arm=arm,
            reviewer_threshold=reviewer_threshold,
        )
        metrics = binary_metrics(labels, predictions)
        summaries = summarize_slices(
            rows,
            predictions,
            slice_fields=("input_channel", selection_slice_field),
        )["by_slice"]
        channel_fprs = [
            value["fpr"]
            for value in summaries.get("input_channel", {}).values()
            if value["fpr"] is not None
        ]
        slice_recalls = [
            value["recall"]
            for value in summaries.get(selection_slice_field, {}).values()
            if value["recall"] is not None
        ]
        thresholds = {
            "direct_low": direct_low,
            "untrusted_low": untrusted_low,
            "local_high": local_high,
            "reviewer": reviewer_threshold,
        }
        threshold_id = ":".join(
            "none" if value is None else f"{value:.12g}"
            for value in thresholds.values()
        )
        call_count = int(np.sum(calls))
        invalid_count = int(np.sum(invalid & calls))
        results.append(
            {
                "configuration_id": f"{arm}:{threshold_id}",
                "arm": arm,
                "thresholds": thresholds,
                "metrics": metrics,
                "call_count": call_count,
                "call_rate": _ratio(call_count, len(rows)),
                "invalid_called_reviews": invalid_count,
                "invalid_called_review_rate": _ratio(invalid_count, call_count),
                "max_channel_fpr": max(channel_fprs) if channel_fprs else None,
                "worst_slice_recall": min(slice_recalls) if slice_recalls else None,
            }
        )
    return results


def select_profiles(
    candidates: Sequence[Mapping[str, Any]],
    *,
    constraints: Mapping[str, Mapping[str, float]] = PROFILE_CONSTRAINTS,
) -> dict[str, Mapping[str, Any] | None]:
    """Choose constrained profiles with stable, documented tie-breaking."""
    selected = {}
    for profile, limits in constraints.items():
        eligible = [
            candidate
            for candidate in candidates
            if candidate["metrics"]["fpr"] is not None
            and candidate["max_channel_fpr"] is not None
            and candidate["call_rate"] is not None
            and candidate["worst_slice_recall"] is not None
            and candidate["metrics"]["fpr"] <= limits["fpr"]
            and candidate["max_channel_fpr"] <= limits["channel_fpr"]
            and candidate["call_rate"] <= limits["call_rate"]
        ]
        selected[profile] = (
            min(
                eligible,
                key=lambda candidate: (
                    -candidate["worst_slice_recall"],
                    -candidate["metrics"]["recall"],
                    candidate["call_rate"],
                    candidate["configuration_id"],
                ),
            )
            if eligible
            else None
        )
    return selected
