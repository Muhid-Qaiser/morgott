#!/usr/bin/env python3
"""Compare full-context-first with middle-window-only DeepSeek routing."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np

from experiments.pipeline_benchmark import hard_verdict as hard
from experiments.pipeline_benchmark import logprob_exact, metrics

DEFAULT_OUTPUT = hard.DEFAULT_OUTPUT
OUTPUT_NAME = "cascade_flow_comparison.json"
SEMANTICS = "middle_windows_on_demand"


def on_demand_predictions(
    rows: list[dict[str, Any]],
    score_records: dict[str, dict[str, Any]],
    artifact_probabilities: dict[str, float | None],
    window_probabilities: dict[tuple[str, int], float | None],
    selection: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Review single artifacts or middle-zone windows, never full long inputs."""
    thresholds = selection["thresholds"]
    predictions = np.zeros(len(rows), dtype=bool)
    artifact_calls = np.zeros(len(rows), dtype=np.int64)
    window_calls = np.zeros(len(rows), dtype=np.int64)
    invalid_reviews = np.zeros(len(rows), dtype=np.int64)
    for row_index, row in enumerate(rows):
        artifact_id = row["artifact_id"]
        scores = score_records[artifact_id]["window_scores"]
        low = thresholds[
            "direct_low" if row["input_channel"] == "direct_user" else "untrusted_low"
        ]
        if any(score >= thresholds["local_high"] for score in scores):
            predictions[row_index] = True
            continue
        pending = [index for index, score in enumerate(scores) if score >= low]
        if len(scores) == 1:
            if pending:
                artifact_calls[row_index] = 1
                predictions[row_index], invalid = logprob_exact._probability(
                    artifact_probabilities[artifact_id], thresholds["reviewer"]
                )
                invalid_reviews[row_index] += invalid
            continue
        for offset in range(0, len(pending), 4):
            batch = pending[offset : offset + 4]
            outcomes = [
                logprob_exact._probability(
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


def _summary(
    rows: list[dict[str, Any]], replay: dict[str, np.ndarray]
) -> dict[str, Any]:
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
    return result


def _coverage(
    rows: list[dict[str, Any]],
    score_records: dict[str, dict[str, Any]],
    selected: dict[str, Any],
) -> dict[str, int]:
    thresholds = selected["thresholds"]
    result = {
        "multi_window_artifacts": 0,
        "multi_window_untrusted_artifacts": 0,
        "untrusted_without_local_high": 0,
        "untrusted_all_windows_below_low": 0,
        "untrusted_with_middle_windows": 0,
    }
    for row in rows:
        scores = score_records[row["artifact_id"]]["window_scores"]
        if len(scores) <= 1:
            continue
        result["multi_window_artifacts"] += 1
        if row["input_channel"] != "untrusted_content":
            continue
        result["multi_window_untrusted_artifacts"] += 1
        if any(score >= thresholds["local_high"] for score in scores):
            continue
        result["untrusted_without_local_high"] += 1
        if any(score >= thresholds["untrusted_low"] for score in scores):
            result["untrusted_with_middle_windows"] += 1
        else:
            result["untrusted_all_windows_below_low"] += 1
    return result


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
        replay = on_demand_predictions(
            rows,
            score_records,
            artifact_probabilities,
            window_probabilities,
            {"thresholds": thresholds},
        )
        sliced = metrics.summarize_slices(
            rows, replay["predictions"], slice_fields=hard.SLICE_FIELDS
        )
        channel_fprs = [
            value["fpr"]
            for value in sliced["by_slice"]["input_channel"].values()
            if value["fpr"] is not None
        ]
        dataset_recalls = [
            value["recall"]
            for value in sliced["by_slice"]["dataset"].values()
            if value["recall"] is not None
        ]
        called = replay["artifact_calls"] + replay["window_calls"] > 0
        result.append(
            {
                "configuration_id": "on_demand:"
                + ":".join(
                    f"{value:.12g}"
                    for value in (direct_low, untrusted_low, local_high, reviewer)
                ),
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
                "worst_slice_recall": min(dataset_recalls) if dataset_recalls else None,
                "semantics": SEMANTICS,
            }
        )
    return result


def _records(
    output: Path,
    *,
    stage: str,
    ids: list[str],
    panel: dict[str, dict[str, Any]],
) -> tuple[dict[str, float | None], dict[str, Any]]:
    complete = hard._complete_ledger(output, stage=stage, expected_ids=ids, panel=panel)
    if complete is None:
        raise RuntimeError(f"{stage} provider ledger is incomplete")
    records, identity = complete
    selected = logprob_exact._provider_records(records)
    if len(selected) != len(ids):
        raise RuntimeError(f"Cloudflare {stage} provider ledger is incomplete")
    values, _ = logprob_exact._probability_values(ids, selected)
    return values, identity


def analyze(output: Path) -> dict[str, Any]:
    manifest, panel, scores, identity = logprob_exact._inputs(output)
    identity = dict(identity)
    identity["logprob_exact_code_sha256"] = identity.pop("analysis_code_sha256")
    exact_selection_path = output / logprob_exact.SELECTION_NAME
    exact_evaluation_path = output / logprob_exact.EVALUATION_NAME
    if not exact_selection_path.is_file() or not exact_evaluation_path.is_file():
        raise RuntimeError("exact maintained logprob evidence is incomplete")
    incumbent_selection = hard._json(exact_selection_path)

    window_complete = logprob_exact._window_records(output, panel, scores)
    if window_complete is None:
        raise RuntimeError("Cloudflare cascade-window ledger is incomplete")
    window_records, window_identity = window_complete
    window_probabilities, _ = logprob_exact._window_probability_values(window_records)

    calibration_ids = manifest["roles"]["provider_panel_ids"]
    calibration_rows = hard._analysis_rows(panel, calibration_ids)
    calibration_scores = {row_id: scores[row_id] for row_id in calibration_ids}
    calibration_artifacts, panel_identity = _records(
        output, stage="panel", ids=calibration_ids, panel=panel
    )
    calibration_required = hard._grid_window_keys(calibration_rows, calibration_scores)
    if not set(calibration_required) <= set(window_probabilities):
        raise RuntimeError("on-demand calibration windows are incomplete")
    grid = _grid(
        calibration_rows,
        calibration_scores,
        calibration_artifacts,
        {key: window_probabilities[key] for key in calibration_required},
    )
    candidate_profiles = metrics.select_profiles(grid)

    evaluation_ids = manifest["roles"]["provider_safe_evaluation_panel_ids"]
    evaluation_rows = hard._analysis_rows(panel, evaluation_ids)
    evaluation_scores = {row_id: scores[row_id] for row_id in evaluation_ids}
    evaluation_artifacts, evaluation_identity = _records(
        output, stage="evaluation", ids=evaluation_ids, panel=panel
    )
    labels = [row["label"] for row in evaluation_rows]
    profiles = {}
    for name in ("conservative", "balanced", "high_recall"):
        incumbent_selected = incumbent_selection["profiles"].get(name)
        candidate_selected = candidate_profiles.get(name)
        if incumbent_selected is None:
            continue
        incumbent = logprob_exact.exact_predictions(
            evaluation_rows,
            evaluation_scores,
            evaluation_artifacts,
            window_probabilities,
            incumbent_selected,
        )
        same_threshold = on_demand_predictions(
            evaluation_rows,
            evaluation_scores,
            evaluation_artifacts,
            window_probabilities,
            incumbent_selected,
        )
        reselected = (
            on_demand_predictions(
                evaluation_rows,
                evaluation_scores,
                evaluation_artifacts,
                window_probabilities,
                candidate_selected,
            )
            if candidate_selected is not None
            else None
        )
        profiles[name] = {
            "incumbent_thresholds": incumbent_selected["thresholds"],
            "candidate_thresholds": (
                candidate_selected["thresholds"]
                if candidate_selected is not None
                else None
            ),
            "evaluation_coverage": _coverage(
                evaluation_rows, evaluation_scores, incumbent_selected
            ),
            "maintained_full_context_first": _summary(evaluation_rows, incumbent),
            "on_demand_same_thresholds": _summary(evaluation_rows, same_threshold),
            "on_demand_reselected": (
                _summary(evaluation_rows, reselected)
                if reselected is not None
                else None
            ),
            "paired_delta_same_thresholds": metrics.paired_stratified_bootstrap_delta(
                labels,
                incumbent["predictions"],
                same_threshold["predictions"],
            ),
            "paired_delta_reselected": (
                metrics.paired_stratified_bootstrap_delta(
                    labels,
                    incumbent["predictions"],
                    reselected["predictions"],
                )
                if reselected is not None
                else None
            ),
        }

    return {
        "schema_version": 1,
        "advisory_only": True,
        "status": "post_hoc_comparison_on_consumed_evaluation",
        "provider": incumbent_selection["provider"],
        "incumbent_semantics": "maintained_multi_window_exact",
        "candidate_semantics": SEMANTICS,
        "candidate_description": (
            "Local-high still restricts; single-window middle-zone artifacts and "
            "only middle-zone windows of long artifacts receive DeepSeek review."
        ),
        "calibration_rows": len(calibration_rows),
        "evaluation_rows": len(evaluation_rows),
        "profiles": profiles,
        "inputs": identity
        | {
            "exact_selection_sha256": hard._sha256(exact_selection_path),
            "exact_evaluation_sha256": hard._sha256(exact_evaluation_path),
            "provider_panel": panel_identity,
            "provider_evaluation": evaluation_identity,
            "provider_cascade_windows": window_identity,
            "analysis_code_sha256": hard._sha256(Path(__file__)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    hard._write_once(output / OUTPUT_NAME, analyze(output))
    print(f"wrote {output / OUTPUT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
