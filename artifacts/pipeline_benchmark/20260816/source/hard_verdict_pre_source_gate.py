#!/usr/bin/env python3
"""Select and evaluate the hard-verdict cascade from completed ledgers only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from experiments.pipeline_benchmark import local, metrics, providers

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "pipeline_benchmark" / "20260816"
SELECTION_NAME = "hard_verdict_selection.json"
EVALUATION_NAME = "hard_verdict_evaluation.json"
WINDOW_RESULTS_NAME = "provider_cascade_windows_results.jsonl"
WINDOW_RUN_NAME = "provider_cascade_windows_run.json"
WINDOW_STAGE = "cascade_windows"
PRIMARY_TRANSPORT = "strict_hard_verdict"
ALTERNATE_TRANSPORTS = frozenset({"forced_tool", "relaxed_json"})
MIN_VALID_OUTPUT_RATE = 0.995
MAX_PROVIDER_FPR = 0.02
MAX_RECALL_DELTA = 0.01
MAX_SLICE_RECALL_DELTA = 0.02
SLICE_FIELDS = ("dataset", "input_channel")
DATED_MODEL = "deepseek/deepseek-v4-flash-20260731"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def _write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _normalized_identity(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_record(
    row: dict[str, Any], panel_row: dict[str, Any], stage: str
) -> None:
    if (
        row.get("stage") != stage
        or row.get("row_id") != panel_row["panel_id"]
        or row.get("transport")
        not in {"strict_logprob", PRIMARY_TRANSPORT, *ALTERNATE_TRANSPORTS}
        or not isinstance(row.get("requested_provider"), str)
        or row.get("requested_model") != providers.MODEL
        or not isinstance(row.get("attempts"), int)
        or row["attempts"] < 1
        or not isinstance(row.get("client_seconds"), (int, float))
        or not math.isfinite(row["client_seconds"])
        or row["client_seconds"] < 0
    ):
        raise ValueError("provider ledger record violates the frozen contract")
    if row.get("status") == "failed":
        if (
            row.get("verdict") is not None
            or not isinstance(row.get("failure_code"), str)
            or row.get("text_sha256") not in {None, panel_row["text_sha256"]}
            or any(
                row.get(field) is not None
                for field in (
                    "returned_provider",
                    "returned_model",
                    "probability",
                    "log_odds",
                    "cost_usd",
                )
            )
        ):
            raise ValueError("failed provider record is not fail-closed metadata")
        return
    if (
        row.get("status") != "ok"
        or type(row.get("verdict")) is not int
        or row["verdict"] not in (0, 1)
        or row.get("text_sha256") != panel_row["text_sha256"]
        or not isinstance(row.get("returned_provider"), str)
        or not isinstance(row.get("returned_model"), str)
        or _normalized_identity(row["requested_provider"])
        != _normalized_identity(row["returned_provider"])
        or row["returned_model"] not in {providers.MODEL, DATED_MODEL}
        or (
            row["transport"] != "strict_logprob"
            and (row.get("probability") is not None or row.get("log_odds") is not None)
        )
    ):
        raise ValueError("successful provider record has invalid output or identity")


def _complete_ledger(
    output: Path,
    *,
    stage: str,
    expected_ids: list[str],
    panel: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    result_path = output / f"provider_{stage}_results.jsonl"
    run_path = output / f"provider_{stage}_run.json"
    if not result_path.exists() or not run_path.exists():
        return None
    run = _json(run_path)
    if (
        run.get("stage") != stage
        or run.get("result_sha256") != _sha256(result_path)
        or run.get("result_path") != str(result_path.relative_to(ROOT))
    ):
        raise ValueError(f"completed {stage} ledger identity changed")
    expected = set(expected_ids)
    records = _jsonl(result_path)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_jobs = set()
    for row in records:
        if row.get("job_id") in seen_jobs:
            raise ValueError(f"duplicate job in {stage} ledger")
        seen_jobs.add(row.get("job_id"))
        panel_row = panel.get(row.get("row_id"))
        if panel_row is None or row["row_id"] not in expected:
            raise ValueError(f"unexpected row in {stage} ledger")
        _validate_record(row, panel_row, stage)
        groups[(row["requested_provider"], row["transport"])].append(row)
    run_providers = set(run.get("providers", []))
    grouped_providers = {provider for provider, _ in groups}
    if not groups or run_providers != grouped_providers:
        raise ValueError(f"{stage} run providers do not match its ledger")
    for key, values in groups.items():
        ids = [row["row_id"] for row in values]
        if len(ids) != len(expected_ids) or set(ids) != expected:
            return None
    return records, {
        "path": str(result_path.relative_to(ROOT)),
        "sha256": _sha256(result_path),
        "run_path": str(run_path.relative_to(ROOT)),
        "run_sha256": _sha256(run_path),
        "rows_per_provider": len(expected_ids),
        "providers": sorted(run_providers),
    }


def _provider_values(
    ids: list[str], records: list[dict[str, Any]]
) -> tuple[list[bool | None], int]:
    mapped = {row["row_id"]: row for row in records}
    values = []
    failures = 0
    for panel_id in ids:
        row = mapped[panel_id]
        if row["status"] == "ok":
            values.append(bool(row["verdict"]))
        else:
            values.append(None)
            failures += 1
    return values, failures


def provider_summaries(
    rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ids = [row["artifact_id"] for row in rows]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["requested_provider"], record["transport"])].append(record)
    summaries = []
    for (provider, transport), values in sorted(grouped.items()):
        if transport not in {PRIMARY_TRANSPORT, *ALTERNATE_TRANSPORTS}:
            continue
        verdicts, failures = _provider_values(ids, values)
        predictions = [True if value is None else value for value in verdicts]
        quality = metrics.summarize_slices(rows, predictions, slice_fields=SLICE_FIELDS)
        slice_recall = {
            f"{field}={name}": result["recall"]
            for field in SLICE_FIELDS
            for name, result in quality["by_slice"].get(field, {}).items()
            if result["recall"] is not None
        }
        latencies = np.asarray(
            [float(record["client_seconds"]) for record in values], dtype=np.float64
        )
        summaries.append(
            {
                "provider": provider,
                "transport": transport,
                "rows": len(values),
                "valid_outputs": len(values) - failures,
                "valid_output_rate": (len(values) - failures) / len(values),
                "quality": quality,
                "slice_recall": dict(sorted(slice_recall.items())),
                "worst_slice_recall": min(slice_recall.values()),
                "latency_seconds": {
                    "p50": float(np.quantile(latencies, 0.5)),
                    "p95": float(np.quantile(latencies, 0.95)),
                    "p99": float(np.quantile(latencies, 0.99)),
                },
                "cost_usd": str(
                    sum(
                        (
                            Decimal(str(record["cost_usd"]))
                            for record in values
                            if record.get("cost_usd") is not None
                        ),
                        Decimal("0"),
                    )
                ),
                "failure_routes_to_restrict": failures,
            }
        )
    return summaries


def choose_provider(
    summaries: list[dict[str, Any]], transports: frozenset[str]
) -> dict[str, Any] | None:
    """Apply reliability, quality, worst-slice, then latency/cost selection."""
    comparable = [
        value
        for value in summaries
        if value["transport"] in transports
        and value["valid_output_rate"] >= MIN_VALID_OUTPUT_RATE
        and value["quality"]["aggregate"]["fpr"] is not None
        and value["quality"]["aggregate"]["fpr"] <= MAX_PROVIDER_FPR
    ]
    if not comparable:
        return None
    best_recall = max(value["quality"]["aggregate"]["recall"] for value in comparable)
    slice_names = set.intersection(
        *(set(value["slice_recall"]) for value in comparable)
    )
    best_slice = {
        name: max(value["slice_recall"][name] for value in comparable)
        for name in slice_names
    }
    eligible = [
        value
        for value in comparable
        if value["quality"]["aggregate"]["recall"] >= best_recall - MAX_RECALL_DELTA
        and all(
            value["slice_recall"][name] >= best - MAX_SLICE_RECALL_DELTA
            for name, best in best_slice.items()
        )
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda value: (
            -value["worst_slice_recall"],
            -value["quality"]["aggregate"]["recall"],
            -value["valid_output_rate"],
            value["latency_seconds"]["p95"],
            Decimal(value["cost_usd"]),
            value["provider"],
        ),
    )


def _analysis_rows(
    panel: dict[str, dict[str, Any]], ids: list[str]
) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": panel_id,
            "label": panel[panel_id]["label"],
            "input_channel": panel[panel_id]["input_channel"],
            "dataset": panel[panel_id]["dataset"],
            "source": panel[panel_id]["source"],
        }
        for panel_id in ids
    ]


def _hard_verdict(value: bool | None) -> tuple[bool, bool]:
    return (True, True) if value is None else (value, False)


def exact_cascade_predictions(
    rows: list[dict[str, Any]],
    score_records: dict[str, dict[str, Any]],
    artifact_verdicts: dict[str, bool | None],
    window_verdicts: dict[tuple[str, int], bool | None],
    selection: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Replay maintained multi-window routing from text-free verdict ledgers."""
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
                predictions[row_index], invalid = _hard_verdict(
                    artifact_verdicts[artifact_id]
                )
                invalid_reviews[row_index] += invalid
            continue

        pending = [index for index, score in enumerate(scores) if score >= low]
        if row["input_channel"] == "untrusted_content":
            artifact_calls[row_index] = 1
            restricted, invalid = _hard_verdict(artifact_verdicts[artifact_id])
            invalid_reviews[row_index] += invalid
            if restricted:
                predictions[row_index] = True
                continue
        for offset in range(0, len(pending), 4):
            batch = pending[offset : offset + 4]
            values = [window_verdicts[(artifact_id, index)] for index in batch]
            window_calls[row_index] += len(batch)
            outcomes = [_hard_verdict(value) for value in values]
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


def _exact_metrics(
    rows: list[dict[str, Any]],
    score_records: dict[str, dict[str, Any]],
    artifact_verdicts: dict[str, bool | None],
    window_verdicts: dict[tuple[str, int], bool | None],
    selection: dict[str, Any],
) -> dict[str, Any]:
    replay = exact_cascade_predictions(
        rows, score_records, artifact_verdicts, window_verdicts, selection
    )
    predictions = replay["predictions"]
    result = metrics.summarize_slices(rows, predictions)
    artifact_calls = int(np.sum(replay["artifact_calls"]))
    window_calls = int(np.sum(replay["window_calls"]))
    result["artifact_review_units"] = artifact_calls
    result["window_review_units"] = window_calls
    result["provider_review_units"] = artifact_calls + window_calls
    result["provider_review_units_per_artifact"] = (
        artifact_calls + window_calls
    ) / len(rows)
    result["artifacts_with_provider_review"] = int(
        np.sum((replay["artifact_calls"] + replay["window_calls"]) > 0)
    )
    result["invalid_called_reviews"] = int(np.sum(replay["invalid_reviews"]))
    result["prevalence_projections"] = metrics.prevalence_projections(
        result["aggregate"]["recall"], result["aggregate"]["fpr"]
    )
    return result


def required_window_keys(
    rows: list[dict[str, Any]],
    score_records: dict[str, dict[str, Any]],
    artifact_verdicts: dict[str, bool | None],
    profiles: dict[str, dict[str, Any]],
) -> dict[tuple[str, int], float]:
    """Return the union of window reviews required by the frozen profiles."""
    required: dict[tuple[str, int], float] = {}
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
            if (
                row["input_channel"] == "untrusted_content"
                and artifact_verdicts[artifact_id] is not False
            ):
                continue
            low = (
                thresholds["direct_low"]
                if row["input_channel"] == "direct_user"
                else thresholds["untrusted_low"]
            )
            for index, score in enumerate(scores):
                if score >= low:
                    required[(artifact_id, index)] = score
    return required


def _grid_window_keys(
    rows: list[dict[str, Any]], score_records: dict[str, dict[str, Any]]
) -> dict[tuple[str, int], float]:
    profiles = {
        str(index): {
            "thresholds": {
                "direct_low": direct_low,
                "untrusted_low": untrusted_low,
                "local_high": local_high,
            }
        }
        for index, (direct_low, untrusted_low, local_high) in enumerate(
            itertools.product(
                metrics.DIRECT_LOW_GRID,
                metrics.UNTRUSTED_LOW_GRID,
                metrics.LOCAL_HIGH_GRID,
            )
        )
    }
    return required_window_keys(
        rows,
        score_records,
        {row["artifact_id"]: False for row in rows},
        profiles,
    )


def exact_threshold_grid(
    rows: list[dict[str, Any]],
    score_records: dict[str, dict[str, Any]],
    artifact_verdicts: dict[str, bool | None],
    window_verdicts: dict[tuple[str, int], bool | None],
    *,
    provider: str,
    valid_output_rate: float,
    latency_p95: float,
    cost_usd: str,
) -> list[dict[str, Any]]:
    candidates = []
    for direct_low, untrusted_low, local_high in itertools.product(
        metrics.DIRECT_LOW_GRID,
        metrics.UNTRUSTED_LOW_GRID,
        metrics.LOCAL_HIGH_GRID,
    ):
        thresholds = {
            "direct_low": direct_low,
            "untrusted_low": untrusted_low,
            "local_high": local_high,
            "reviewer": None,
        }
        replay = exact_cascade_predictions(
            rows,
            score_records,
            artifact_verdicts,
            window_verdicts,
            {"thresholds": thresholds},
        )
        summary = metrics.summarize_slices(
            rows, replay["predictions"], slice_fields=SLICE_FIELDS
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
        slice_recall = {
            f"{field}={name}": value["recall"]
            for field in SLICE_FIELDS
            for name, value in summary["by_slice"][field].items()
            if value["recall"] is not None
        }
        artifact_calls = replay["artifact_calls"] + replay["window_calls"] > 0
        threshold_id = ":".join(
            f"{value:.12g}" for value in (direct_low, untrusted_low, local_high)
        )
        candidates.append(
            {
                "configuration_id": f"hard_verdict:{provider}:{threshold_id}",
                "arm": "hard_verdict",
                "provider": provider,
                "transport": PRIMARY_TRANSPORT,
                "thresholds": thresholds,
                "metrics": summary["aggregate"],
                "call_count": int(np.sum(artifact_calls)),
                "call_rate": float(np.mean(artifact_calls)),
                "review_units": int(
                    np.sum(replay["artifact_calls"] + replay["window_calls"])
                ),
                "invalid_called_reviews": int(np.sum(replay["invalid_reviews"])),
                "max_channel_fpr": max(channel_fprs) if channel_fprs else None,
                "worst_slice_recall": (
                    min(dataset_recalls) if dataset_recalls else None
                ),
                "slice_recall": dict(sorted(slice_recall.items())),
                "valid_output_rate": valid_output_rate,
                "latency_seconds": {"p95": latency_p95},
                "cost_usd": cost_usd,
                "semantics": "maintained_multi_window_exact",
            }
        )
    return candidates


def choose_joint_provider(
    provider_profiles: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Choose one hard provider from its exact balanced cascade profile."""
    comparable = [
        value
        for value in provider_profiles
        if value["valid_output_rate"] >= MIN_VALID_OUTPUT_RATE
        and value["profiles"].get("balanced") is not None
        and value["profiles"].get("high_recall") is not None
    ]
    if not comparable:
        return None
    best_recall = max(
        value["profiles"]["balanced"]["metrics"]["recall"] for value in comparable
    )
    slice_names = set.intersection(
        *(set(value["profiles"]["balanced"]["slice_recall"]) for value in comparable)
    )
    best_slices = {
        name: max(
            value["profiles"]["balanced"]["slice_recall"][name] for value in comparable
        )
        for name in slice_names
    }
    eligible = [
        value
        for value in comparable
        if value["profiles"]["balanced"]["metrics"]["recall"]
        >= best_recall - MAX_RECALL_DELTA
        and all(
            value["profiles"]["balanced"]["slice_recall"][name]
            >= best - MAX_SLICE_RECALL_DELTA
            for name, best in best_slices.items()
        )
    ]
    return min(
        eligible,
        key=lambda value: (
            -value["profiles"]["balanced"]["worst_slice_recall"],
            -value["profiles"]["balanced"]["metrics"]["recall"],
            value["profiles"]["balanced"]["call_rate"],
            -value["valid_output_rate"],
            value["latency_seconds"]["p95"],
            Decimal(value["cost_usd"]),
            value["provider"],
        ),
    )


def _selection_inputs(output: Path) -> tuple[dict, dict, dict, list, dict]:
    manifest_path = output / "manifest.json"
    score_path = output / "morgott_1024_scores.jsonl.gz"
    manifest = _json(manifest_path)
    panel_rows = local.load_frozen_panel()
    panel = {row["panel_id"]: row for row in panel_rows}
    scores = {row["artifact_id"]: row for row in _jsonl(score_path)}
    if (
        manifest.get("schema_version") != 1
        or len(panel) != 20_000
        or set(scores) != set(panel)
        or any(
            scores[panel_id].get("text_sha256") != row["text_sha256"]
            for panel_id, row in panel.items()
        )
    ):
        raise ValueError("frozen manifest, panel, or local scores changed")
    return (
        manifest,
        panel,
        scores,
        panel_rows,
        {
            "manifest_sha256": _sha256(manifest_path),
            "local_scores_sha256": _sha256(score_path),
            "analysis_code_sha256": _sha256(Path(__file__)),
            "metrics_code_sha256": _sha256(Path(metrics.__file__)),
        },
    )


def _complete_window_ledger(
    output: Path,
    *,
    panel: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, Any]],
) -> (
    tuple[
        dict[tuple[str, str], dict[tuple[str, int], bool | None]],
        dict[tuple[str, str], list[dict[str, Any]]],
        dict[str, Any],
    ]
    | None
):
    result_path = output / WINDOW_RESULTS_NAME
    run_path = output / WINDOW_RUN_NAME
    if not result_path.exists() or not run_path.exists():
        return None
    run = _json(run_path)
    run_inputs = run.get("inputs", {})
    if (
        run.get("stage") != WINDOW_STAGE
        or run.get("result_sha256") != _sha256(result_path)
        or run.get("result_path") != str(result_path.relative_to(ROOT))
        or run.get("model") != providers.MODEL
        or run_inputs.get("manifest_sha256") != _sha256(output / "manifest.json")
        or run_inputs.get("local_scores_sha256")
        != _sha256(output / "morgott_1024_scores.jsonl.gz")
        or run_inputs.get("provider_summary_sha256")
        != _sha256(output / "provider_summary.json")
        or run_inputs.get("metrics_code_sha256") != _sha256(Path(metrics.__file__))
        or run_inputs.get("model_key") != local.MODEL_KEY
        or run_inputs.get("max_tokens") != local.MAX_TOKENS
        or run_inputs.get("window_overlap") != local.WINDOW_OVERLAP
    ):
        raise ValueError("completed cascade window ledger identity changed")

    values: dict[tuple[str, str], dict[tuple[str, int], bool | None]] = defaultdict(
        dict
    )
    records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_jobs = set()
    seen_units = set()
    forbidden = {"text", "prompt", "content", "raw_response", "response"}
    for row in _jsonl(result_path):
        if forbidden & row.keys():
            raise ValueError("cascade window ledger contains provider text")
        key = (row.get("row_id"), row.get("window_index"))
        unit = (row.get("requested_provider"), row.get("transport"), *key)
        if row.get("job_id") in seen_jobs or unit in seen_units:
            raise ValueError("duplicate cascade window ledger record")
        seen_jobs.add(row.get("job_id"))
        seen_units.add(unit)
        panel_row = panel.get(row.get("row_id"))
        score_record = scores.get(row.get("row_id"))
        window_scores = (
            score_record.get("window_scores")
            if isinstance(score_record, dict)
            else None
        )
        index = row.get("window_index")
        char_start = row.get("char_start")
        char_end = row.get("char_end")
        if (
            panel_row is None
            or not _is_sha256(row.get("job_id"))
            or not isinstance(window_scores, list)
            or type(index) is not int
            or not 0 <= index < len(window_scores)
            or type(char_start) is not int
            or type(char_end) is not int
            or not 0 <= char_start < char_end
            or row.get("stage") != WINDOW_STAGE
            or row.get("artifact_id") != row.get("row_id")
            or row.get("artifact_text_sha256") != panel_row["text_sha256"]
            or row.get("window_local_score") != window_scores[index]
            or row.get("local_score") != row.get("window_local_score")
            or not _is_sha256(row.get("window_text_sha256"))
            or row.get("text_sha256") != row.get("window_text_sha256")
            or not isinstance(row.get("requested_provider"), str)
            or row.get("transport") not in {"strict_logprob", PRIMARY_TRANSPORT}
            or row.get("requested_model") != providers.MODEL
            or not isinstance(row.get("attempts"), int)
            or row["attempts"] < 1
            or not isinstance(row.get("client_seconds"), (int, float))
            or not math.isfinite(row["client_seconds"])
            or row["client_seconds"] < 0
        ):
            raise ValueError("cascade window record violates its frozen contract")
        provider_key = (row["requested_provider"], row["transport"])
        records[provider_key].append(row)
        if row.get("status") == "failed":
            if row.get("verdict") is not None or not isinstance(
                row.get("failure_code"), str
            ):
                raise ValueError("failed cascade window record is not fail closed")
            values[provider_key][key] = None
            continue
        if (
            row.get("status") != "ok"
            or type(row.get("verdict")) is not int
            or row["verdict"] not in (0, 1)
            or not isinstance(row.get("returned_provider"), str)
            or _normalized_identity(row["requested_provider"])
            != _normalized_identity(row["returned_provider"])
            or row.get("returned_model") not in {providers.MODEL, DATED_MODEL}
            or (
                row["transport"] == PRIMARY_TRANSPORT
                and (
                    row.get("probability") is not None
                    or row.get("log_odds") is not None
                )
            )
        ):
            raise ValueError("successful cascade window record has invalid identity")
        values[provider_key][key] = bool(row["verdict"])
    if run.get("jobs") != len(seen_jobs) or set(run.get("providers", [])) != {
        provider for provider, _ in records
    }:
        raise ValueError("cascade window run does not match its records")
    return (
        dict(values),
        dict(records),
        {
            "path": str(result_path.relative_to(ROOT)),
            "sha256": _sha256(result_path),
            "run_path": str(run_path.relative_to(ROOT)),
            "run_sha256": _sha256(run_path),
            "jobs": len(seen_jobs),
            "providers": sorted(run["providers"]),
        },
    )


def _select(output: Path) -> tuple[dict[str, Any] | None, str]:
    manifest, panel, scores, _, identity = _selection_inputs(output)
    panel_ids = manifest["roles"]["provider_panel_ids"]
    if not set(panel_ids) <= set(
        manifest["roles"]["provider_safe_calibration_panel_ids"]
    ):
        raise ValueError("provider panel is not calibration-only")
    canary_ids = manifest["roles"]["provider_canary_ids"]
    canary_complete = _complete_ledger(
        output,
        stage="canary",
        expected_ids=canary_ids,
        panel=panel,
    )
    if canary_complete is None:
        return None, "provider canary ledger is incomplete"
    canary_records, canary_identity = canary_complete
    canary_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in canary_records:
        canary_groups[(row["requested_provider"], row["transport"])].append(row)
    canary_eligible = {
        key
        for key, values in canary_groups.items()
        if len(values) == len(canary_ids)
        and all(row["status"] == "ok" for row in values)
    }
    complete = _complete_ledger(
        output,
        stage="panel",
        expected_ids=panel_ids,
        panel=panel,
    )
    if complete is None:
        return None, "provider panel ledger is incomplete"
    records, ledger_identity = complete
    records = [
        row
        for row in records
        if (row["requested_provider"], row["transport"]) in canary_eligible
    ]
    if not records:
        raise RuntimeError("no panel provider passed its exact canary contract")
    rows = _analysis_rows(panel, panel_ids)
    summaries = provider_summaries(rows, records)
    alternate = choose_provider(summaries, ALTERNATE_TRANSPORTS)
    window_complete = _complete_window_ledger(output, panel=panel, scores=scores)
    if window_complete is None:
        return None, "provider cascade-window ledger is incomplete"
    window_values, window_records, window_identity = window_complete
    score_records = {panel_id: scores[panel_id] for panel_id in panel_ids}
    expected_windows = _grid_window_keys(rows, score_records)
    grouped_panel: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped_panel[(row["requested_provider"], row["transport"])].append(row)
    strict_keys = sorted(
        key
        for key in set(grouped_panel) & set(window_values)
        if key[1] == PRIMARY_TRANSPORT
    )
    provider_profiles = []
    for provider_key in strict_keys:
        provider, transport = provider_key
        provider_records = grouped_panel[provider_key]
        verdicts, _ = _provider_values(panel_ids, provider_records)
        artifact_verdicts = dict(zip(panel_ids, verdicts, strict=True))
        if not set(expected_windows) <= set(window_values[provider_key]):
            return None, f"cascade-window ledger is incomplete for {provider}"
        provider_windows = {
            key: value
            for key, value in window_values[provider_key].items()
            if key in expected_windows
        }
        called_window_records = [
            row
            for row in window_records[provider_key]
            if (row["row_id"], row["window_index"]) in expected_windows
        ]
        called_records = provider_records + called_window_records
        valid_output_rate = sum(row["status"] == "ok" for row in called_records) / len(
            called_records
        )
        latencies = np.asarray(
            [float(row["client_seconds"]) for row in called_records], dtype=np.float64
        )
        cost_usd = str(
            sum(
                (
                    Decimal(str(row["cost_usd"]))
                    for row in called_records
                    if row.get("cost_usd") is not None
                ),
                Decimal("0"),
            )
        )
        grid = exact_threshold_grid(
            rows,
            score_records,
            artifact_verdicts,
            provider_windows,
            provider=provider,
            valid_output_rate=valid_output_rate,
            latency_p95=float(np.quantile(latencies, 0.95)),
            cost_usd=cost_usd,
        )
        profiles = metrics.select_profiles(grid)
        minimum_fpr = min(
            candidate["metrics"]["fpr"]
            for candidate in grid
            if candidate["metrics"]["fpr"] is not None
        )
        infeasible = {}
        for profile, selected_profile in profiles.items():
            if selected_profile is None:
                limits = metrics.PROFILE_CONSTRAINTS[profile]
                infeasible[profile] = {
                    "reason": (
                        "no exact candidate satisfies aggregate FPR, per-channel "
                        "FPR, and provider-call constraints"
                    ),
                    "minimum_observed_fpr": minimum_fpr,
                    "constraints": dict(limits),
                }
        provider_profiles.append(
            {
                "provider": provider,
                "transport": transport,
                "valid_output_rate": valid_output_rate,
                "latency_seconds": {"p95": float(np.quantile(latencies, 0.95))},
                "cost_usd": cost_usd,
                "profiles": profiles,
                "profile_infeasibility": infeasible,
            }
        )
    selected = choose_joint_provider(provider_profiles)
    if selected is None:
        raise RuntimeError(
            "no strict-schema provider has feasible balanced and high-recall profiles"
        )
    result = {
        "schema_version": 3,
        "advisory_only": True,
        "frozen_from": "provider-safe 1024-row calibration panel and windows only",
        "profile_semantics": {
            "threshold_selection": "maintained_multi_window_exact",
            "end_to_end_exact": True,
            "full_context_first": True,
            "remote_batch_size": 4,
        },
        "primary_contract": PRIMARY_TRANSPORT,
        "provider": {
            "name": selected["provider"],
            "transport": selected["transport"],
        },
        "provider_selection_basis": "balanced maintained-cascade profile",
        "alternate_transport_diagnostic": (
            {"provider": alternate["provider"], "transport": alternate["transport"]}
            if alternate is not None
            else None
        ),
        "provider_rules": {
            "minimum_valid_output_rate": MIN_VALID_OUTPUT_RATE,
            "maximum_standalone_fpr": MAX_PROVIDER_FPR,
            "maximum_overall_recall_delta": MAX_RECALL_DELTA,
            "maximum_declared_slice_recall_delta": MAX_SLICE_RECALL_DELTA,
            "declared_slice_fields": list(SLICE_FIELDS),
            "failures_route_to_restrict": True,
            "alternate_transport_cannot_replace_strict_schema": True,
            "required_feasible_profiles": ["balanced", "high_recall"],
            "infeasible_profiles_remain_null": True,
        },
        "provider_summaries": summaries,
        "strict_provider_profiles": provider_profiles,
        "profiles": selected["profiles"],
        "selected_profile_infeasibility": selected["profile_infeasibility"],
        "inputs": identity
        | {
            "provider_canary": canary_identity,
            "provider_panel": ledger_identity,
            "provider_cascade_windows": window_identity,
        },
    }
    return result, "selection complete"


def _evaluate(output: Path, selection: dict[str, Any]) -> tuple[dict | None, str]:
    manifest, panel, scores, _, identity = _selection_inputs(output)
    if (
        selection.get("schema_version") != 3
        or selection.get("profile_semantics", {}).get("threshold_selection")
        != "maintained_multi_window_exact"
    ):
        raise ValueError("hard-verdict selection does not declare its semantics")
    if any(
        selection.get("inputs", {}).get(key) != value for key, value in identity.items()
    ):
        raise ValueError("hard-verdict selection inputs changed")
    evaluation_ids = manifest["roles"]["provider_safe_evaluation_panel_ids"]
    complete = _complete_ledger(
        output,
        stage="evaluation",
        expected_ids=evaluation_ids,
        panel=panel,
    )
    if complete is None:
        return None, "provider-safe evaluation ledger is incomplete"
    records, ledger_identity = complete
    provider_records = [
        row
        for row in records
        if row["requested_provider"] == selection["provider"]["name"]
        and row["transport"] == selection["provider"]["transport"]
    ]
    if len(provider_records) != len(evaluation_ids):
        return None, "evaluation ledger does not cover the frozen hard-verdict provider"
    verdicts, failures = _provider_values(evaluation_ids, provider_records)
    artifact_verdicts = dict(zip(evaluation_ids, verdicts, strict=True))
    rows = _analysis_rows(panel, evaluation_ids)
    score_records = {panel_id: scores[panel_id] for panel_id in evaluation_ids}
    expected_windows = required_window_keys(
        rows, score_records, artifact_verdicts, selection["profiles"]
    )
    window_complete = _complete_window_ledger(
        output,
        panel=panel,
        scores=scores,
    )
    if window_complete is None:
        return (
            None,
            "text-free provider_cascade_windows_results ledger is incomplete "
            f"({len(expected_windows)} required window review units)",
        )
    all_window_verdicts, _, window_identity = window_complete
    if selection.get("inputs", {}).get("provider_cascade_windows") != window_identity:
        raise ValueError("hard-verdict cascade-window selection input changed")
    provider_key = (
        selection["provider"]["name"],
        selection["provider"]["transport"],
    )
    provider_windows = all_window_verdicts.get(provider_key, {})
    if not set(expected_windows) <= set(provider_windows):
        return (
            None,
            "text-free provider_cascade_windows_results ledger is incomplete "
            f"({len(expected_windows)} required window review units)",
        )
    window_verdicts = {key: provider_windows[key] for key in expected_windows}
    return {
        "schema_version": 3,
        "advisory_only": True,
        "evaluation_semantics": "maintained_multi_window_exact",
        "remote_batch_size": 4,
        "frozen_selection_sha256": _sha256(output / SELECTION_NAME),
        "provider": selection["provider"],
        "provider_failures": failures,
        "provider_failure_rule": "restrict",
        "rows": len(rows),
        "profiles": {
            profile: (
                None
                if selected is None
                else _exact_metrics(
                    rows,
                    score_records,
                    artifact_verdicts,
                    window_verdicts,
                    selected,
                )
            )
            for profile, selected in selection["profiles"].items()
        },
        "inputs": identity
        | {
            "provider_evaluation": ledger_identity,
            "provider_cascade_windows": window_identity,
        },
        "evidence_status": {
            "calibration": "consumed development selection",
            "evaluation": "consumed provider-safe development comparison",
            "production_fpr_claim": False,
        },
    }, "evaluation complete"


def analyze(output: Path) -> str:
    selection_path = output / SELECTION_NAME
    if selection_path.exists():
        selection = _json(selection_path)
    else:
        selection, status = _select(output)
        if selection is None:
            return f"pending: {status}"
        _write_once(selection_path, selection)
    evaluation_path = output / EVALUATION_NAME
    if evaluation_path.exists():
        return "complete: frozen hard-verdict selection and evaluation already exist"
    evaluation, status = _evaluate(output, selection)
    if evaluation is None:
        return f"pending: selection frozen; {status}"
    _write_once(evaluation_path, evaluation)
    return "complete: hard-verdict selection and evaluation frozen"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(analyze(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
