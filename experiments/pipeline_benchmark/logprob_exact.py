#!/usr/bin/env python3
"""Select and evaluate the exact maintained logprob cascade offline."""

from __future__ import annotations

import argparse
import itertools
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from experiments.pipeline_benchmark import hard_verdict as hard
from experiments.pipeline_benchmark import metrics

DEFAULT_OUTPUT = hard.DEFAULT_OUTPUT
SELECTION_NAME = "logprob_exact_selection.json"
EVALUATION_NAME = "logprob_exact_evaluation.json"
PROVIDER = "cloudflare"
TRANSPORT = "strict_logprob"


def _probability(value: float | None, threshold: float) -> tuple[bool, bool]:
    valid = (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )
    return (float(value) >= threshold, False) if valid else (True, True)


def _probability_values(
    ids: list[str], records: list[dict[str, Any]]
) -> tuple[dict[str, float | None], int]:
    mapped = {row["row_id"]: row for row in records}
    values = {}
    failures = 0
    for row_id in ids:
        row = mapped[row_id]
        value = row.get("probability") if row.get("status") == "ok" else None
        _, invalid = _probability(value, 0.5)
        values[row_id] = None if invalid else float(value)
        failures += invalid
    return values, failures


def _window_probability_values(
    records: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int], float | None], int]:
    values = {}
    failures = 0
    for row in records:
        key = (row["row_id"], row["window_index"])
        value = row.get("probability") if row.get("status") == "ok" else None
        _, invalid = _probability(value, 0.5)
        values[key] = None if invalid else float(value)
        failures += invalid
    return values, failures


def exact_predictions(
    rows: list[dict[str, Any]],
    score_records: dict[str, dict[str, Any]],
    artifact_probabilities: dict[str, float | None],
    window_probabilities: dict[tuple[str, int], float | None],
    selection: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Replay full-context-first and ordered batch-of-four routing."""
    thresholds = selection["thresholds"]
    predictions = np.zeros(len(rows), dtype=bool)
    artifact_calls = np.zeros(len(rows), dtype=np.int64)
    window_calls = np.zeros(len(rows), dtype=np.int64)
    invalid_reviews = np.zeros(len(rows), dtype=np.int64)
    for row_index, row in enumerate(rows):
        artifact_id = row["artifact_id"]
        scores = score_records[artifact_id]["window_scores"]
        low = (
            thresholds["direct_low"]
            if row["input_channel"] == "direct_user"
            else thresholds["untrusted_low"]
        )
        if any(score >= thresholds["local_high"] for score in scores):
            predictions[row_index] = True
            continue
        if len(scores) == 1:
            if scores[0] >= low:
                artifact_calls[row_index] = 1
                predictions[row_index], invalid = _probability(
                    artifact_probabilities[artifact_id], thresholds["reviewer"]
                )
                invalid_reviews[row_index] += invalid
            continue

        pending = [index for index, score in enumerate(scores) if score >= low]
        if row["input_channel"] == "untrusted_content":
            artifact_calls[row_index] = 1
            restricted, invalid = _probability(
                artifact_probabilities[artifact_id], thresholds["reviewer"]
            )
            invalid_reviews[row_index] += invalid
            if restricted:
                predictions[row_index] = True
                continue
        for offset in range(0, len(pending), 4):
            batch = pending[offset : offset + 4]
            outcomes = [
                _probability(
                    window_probabilities[(artifact_id, index)],
                    thresholds["reviewer"],
                )
                for index in batch
            ]
            window_calls[row_index] += len(batch)
            invalid_reviews[row_index] += sum(invalid for _, invalid in outcomes)
            if any(restricted for restricted, _ in outcomes):
                predictions[row_index] = True
                break
    return {
        "predictions": predictions,
        "artifact_calls": artifact_calls,
        "window_calls": window_calls,
        "invalid_reviews": invalid_reviews,
    }


def required_window_keys(
    rows: list[dict[str, Any]],
    score_records: dict[str, dict[str, Any]],
    artifact_probabilities: dict[str, float | None],
    profiles: dict[str, dict[str, Any] | None],
) -> set[tuple[str, int]]:
    required = set()
    for selected in profiles.values():
        if selected is None:
            continue
        thresholds = selected["thresholds"]
        for row in rows:
            artifact_id = row["artifact_id"]
            scores = score_records[artifact_id]["window_scores"]
            if len(scores) == 1 or any(
                score >= thresholds["local_high"] for score in scores
            ):
                continue
            if row["input_channel"] == "untrusted_content":
                restricted, _ = _probability(
                    artifact_probabilities[artifact_id], thresholds["reviewer"]
                )
                if restricted:
                    continue
            low = (
                thresholds["direct_low"]
                if row["input_channel"] == "direct_user"
                else thresholds["untrusted_low"]
            )
            required.update(
                (artifact_id, index)
                for index, score in enumerate(scores)
                if score >= low
            )
    return required


def _grid(
    rows: list[dict[str, Any]],
    score_records: dict[str, dict[str, Any]],
    artifact_probabilities: dict[str, float | None],
    window_probabilities: dict[tuple[str, int], float | None],
) -> list[dict[str, Any]]:
    labels = [row["label"] for row in rows]
    result = []
    for direct_low, untrusted_low, local_high, reviewer in itertools.product(
        metrics.DIRECT_LOW_GRID,
        metrics.UNTRUSTED_LOW_GRID,
        metrics.LOCAL_HIGH_GRID,
        metrics.LOGPROB_GRID,
    ):
        thresholds = {
            "direct_low": direct_low,
            "untrusted_low": untrusted_low,
            "local_high": local_high,
            "reviewer": reviewer,
        }
        replay = exact_predictions(
            rows,
            score_records,
            artifact_probabilities,
            window_probabilities,
            {"thresholds": thresholds},
        )
        summary = metrics.summarize_slices(
            rows, replay["predictions"], slice_fields=hard.SLICE_FIELDS
        )
        channel_fprs = [
            value["fpr"]
            for value in summary["by_slice"]["input_channel"].values()
            if value["fpr"] is not None
        ]
        dataset_recalls = [
            value["recall"]
            for value in summary["by_slice"]["dataset"].values()
            if value["recall"] is not None
        ]
        called = replay["artifact_calls"] + replay["window_calls"] > 0
        threshold_id = ":".join(
            f"{value:.12g}"
            for value in (direct_low, untrusted_low, local_high, reviewer)
        )
        result.append(
            {
                "configuration_id": f"logprob:{threshold_id}",
                "arm": "logprob",
                "thresholds": thresholds,
                "metrics": metrics.binary_metrics(labels, replay["predictions"]),
                "call_count": int(np.sum(called)),
                "call_rate": float(np.mean(called)),
                "review_units": int(
                    np.sum(replay["artifact_calls"] + replay["window_calls"])
                ),
                "invalid_called_reviews": int(np.sum(replay["invalid_reviews"])),
                "max_channel_fpr": max(channel_fprs) if channel_fprs else None,
                "worst_slice_recall": (
                    min(dataset_recalls) if dataset_recalls else None
                ),
                "semantics": "maintained_multi_window_exact",
            }
        )
    return result


def _metrics(
    rows: list[dict[str, Any]],
    score_records: dict[str, dict[str, Any]],
    artifact_probabilities: dict[str, float | None],
    window_probabilities: dict[tuple[str, int], float | None],
    selected: dict[str, Any],
) -> dict[str, Any]:
    replay = exact_predictions(
        rows,
        score_records,
        artifact_probabilities,
        window_probabilities,
        selected,
    )
    result = metrics.summarize_slices(rows, replay["predictions"])
    artifact_calls = int(np.sum(replay["artifact_calls"]))
    window_calls = int(np.sum(replay["window_calls"]))
    result.update(
        {
            "artifact_review_units": artifact_calls,
            "window_review_units": window_calls,
            "provider_review_units": artifact_calls + window_calls,
            "artifacts_with_provider_review": int(
                np.sum((replay["artifact_calls"] + replay["window_calls"]) > 0)
            ),
            "invalid_called_reviews": int(np.sum(replay["invalid_reviews"])),
        }
    )
    result["prevalence_projections"] = metrics.prevalence_projections(
        result["aggregate"]["recall"], result["aggregate"]["fpr"]
    )
    return result


def _inputs(output: Path) -> tuple[dict, dict, dict, dict[str, Any]]:
    manifest, panel, scores, _, identity = hard._selection_inputs(output)
    identity = dict(identity)
    identity["hard_helper_code_sha256"] = identity.pop("analysis_code_sha256")
    identity["analysis_code_sha256"] = hard._sha256(Path(__file__))
    return manifest, panel, scores, identity


def _provider_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if row["requested_provider"] == PROVIDER and row["transport"] == TRANSPORT
    ]


def _window_records(
    output: Path, panel: dict, scores: dict
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    complete = hard._complete_window_ledger(output, panel=panel, scores=scores)
    if complete is None:
        return None
    _, grouped, identity = complete
    records = grouped.get((PROVIDER, TRANSPORT))
    return (records, identity) if records is not None else None


def _select(output: Path) -> tuple[dict[str, Any] | None, str]:
    manifest, panel, scores, identity = _inputs(output)
    panel_ids = manifest["roles"]["provider_panel_ids"]
    if not set(panel_ids) <= set(
        manifest["roles"]["provider_safe_calibration_panel_ids"]
    ):
        raise ValueError("provider panel is not calibration-only")
    complete = hard._complete_ledger(
        output, stage="panel", expected_ids=panel_ids, panel=panel
    )
    if complete is None:
        return None, "provider panel ledger is incomplete"
    records, panel_identity = complete
    provider_records = _provider_records(records)
    if len(provider_records) != len(panel_ids):
        return None, "Cloudflare logprob panel is incomplete"
    artifact_probabilities, artifact_failures = _probability_values(
        panel_ids, provider_records
    )
    rows = hard._analysis_rows(panel, panel_ids)
    score_records = {row_id: scores[row_id] for row_id in panel_ids}
    window_complete = _window_records(output, panel, scores)
    if window_complete is None:
        return None, "Cloudflare cascade-window ledger is incomplete"
    window_records, window_identity = window_complete
    window_probabilities, _ = _window_probability_values(window_records)
    expected_windows = hard._grid_window_keys(rows, score_records)
    if not set(expected_windows) <= set(window_probabilities):
        return None, "Cloudflare calibration cascade windows are incomplete"
    calibration_windows = {key: window_probabilities[key] for key in expected_windows}
    window_failures = sum(value is None for value in calibration_windows.values())
    grid = _grid(rows, score_records, artifact_probabilities, calibration_windows)
    profiles = metrics.select_profiles(grid)
    if profiles["balanced"] is None or profiles["high_recall"] is None:
        raise RuntimeError("logprob arm lacks a feasible balanced or high-recall point")
    minimum_fpr = min(
        candidate["metrics"]["fpr"]
        for candidate in grid
        if candidate["metrics"]["fpr"] is not None
    )
    infeasible = {
        profile: {
            "reason": (
                "no exact candidate satisfies aggregate FPR, per-channel FPR, "
                "and provider-call constraints"
            ),
            "minimum_observed_fpr": minimum_fpr,
            "constraints": dict(metrics.PROFILE_CONSTRAINTS[profile]),
        }
        for profile, selected in profiles.items()
        if selected is None
    }
    combined_records = provider_records + [
        row
        for row in window_records
        if (row["row_id"], row["window_index"]) in expected_windows
    ]
    valid_output_rate = sum(row["status"] == "ok" for row in combined_records) / len(
        combined_records
    )
    if valid_output_rate < hard.MIN_VALID_OUTPUT_RATE:
        raise RuntimeError("Cloudflare exact logprob output reliability is below 99.5%")
    latencies = np.asarray(
        [float(row["client_seconds"]) for row in combined_records], dtype=np.float64
    )
    cost = sum(
        (
            Decimal(str(row["cost_usd"]))
            for row in combined_records
            if row.get("cost_usd") is not None
        ),
        Decimal("0"),
    )
    return {
        "schema_version": 1,
        "advisory_only": True,
        "frozen_from": "provider-safe 1024-row calibration panel and windows only",
        "profile_semantics": "maintained_multi_window_exact",
        "provider": {"name": PROVIDER, "transport": TRANSPORT},
        "profiles": profiles,
        "profile_infeasibility": infeasible,
        "reliability": {
            "valid_output_rate": valid_output_rate,
            "artifact_invalid_outputs": artifact_failures,
            "window_invalid_outputs": window_failures,
        },
        "latency_seconds": {"p95": float(np.quantile(latencies, 0.95))},
        "cost_usd": str(cost),
        "inputs": identity
        | {
            "provider_panel": panel_identity,
            "provider_cascade_windows": window_identity,
        },
    }, "selection complete"


def _evaluate(output: Path, selection: dict[str, Any]) -> tuple[dict | None, str]:
    manifest, panel, scores, identity = _inputs(output)
    if any(
        selection.get("inputs", {}).get(key) != value for key, value in identity.items()
    ):
        raise ValueError("exact logprob selection inputs changed")
    evaluation_ids = manifest["roles"]["provider_safe_evaluation_panel_ids"]
    complete = hard._complete_ledger(
        output, stage="evaluation", expected_ids=evaluation_ids, panel=panel
    )
    if complete is None:
        return None, "provider-safe evaluation ledger is incomplete"
    records, evaluation_identity = complete
    provider_records = _provider_records(records)
    if len(provider_records) != len(evaluation_ids):
        return None, "Cloudflare logprob evaluation is incomplete"
    artifact_probabilities, artifact_failures = _probability_values(
        evaluation_ids, provider_records
    )
    rows = hard._analysis_rows(panel, evaluation_ids)
    score_records = {row_id: scores[row_id] for row_id in evaluation_ids}
    window_complete = _window_records(output, panel, scores)
    if window_complete is None:
        return None, "Cloudflare cascade-window ledger is incomplete"
    window_records, window_identity = window_complete
    if selection["inputs"]["provider_cascade_windows"] != window_identity:
        raise ValueError("exact logprob cascade-window input changed")
    window_probabilities, _ = _window_probability_values(window_records)
    expected = required_window_keys(
        rows, score_records, artifact_probabilities, selection["profiles"]
    )
    if not expected <= set(window_probabilities):
        return None, "Cloudflare evaluation cascade windows are incomplete"
    selected_windows = {key: window_probabilities[key] for key in expected}
    window_failures = sum(value is None for value in selected_windows.values())
    return {
        "schema_version": 1,
        "advisory_only": True,
        "evaluation_semantics": "maintained_multi_window_exact",
        "frozen_selection_sha256": hard._sha256(output / SELECTION_NAME),
        "provider": selection["provider"],
        "rows": len(rows),
        "artifact_invalid_outputs": artifact_failures,
        "window_invalid_outputs": window_failures,
        "profiles": {
            name: (
                None
                if selected is None
                else _metrics(
                    rows,
                    score_records,
                    artifact_probabilities,
                    selected_windows,
                    selected,
                )
            )
            for name, selected in selection["profiles"].items()
        },
        "inputs": identity
        | {
            "provider_evaluation": evaluation_identity,
            "provider_cascade_windows": window_identity,
        },
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
        return "complete: exact logprob selection and evaluation already exist"
    evaluation, status = _evaluate(output, selection)
    if evaluation is None:
        return f"pending: selection frozen; {status}"
    hard._write_once(evaluation_path, evaluation)
    return "complete: exact logprob selection and evaluation frozen"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(analyze(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
