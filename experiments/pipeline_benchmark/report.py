#!/usr/bin/env python3
"""Render the pipeline benchmark report from retained, text-free evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.pipeline_benchmark import metrics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "artifacts" / "pipeline_benchmark" / "20260816"
DEFAULT_REPORT = ROOT / "reports" / "pipeline-benchmark-20260816.md"
DEFAULT_TABLES = DEFAULT_INPUT / "tables.json"
AZURE_DEPLOYMENT_RECORD = ROOT / "reports" / "azure-preview-deployment-20260817.json"
RETAINED_EXPERIMENTS = ROOT / "reports" / "model-experiments.md"
RETAINED_LLM_REPORT = ROOT / "reports" / "openrouter-downstream-evaluation.md"
REGISTERED_EVALUATION = (
    ROOT
    / "artifacts"
    / "models"
    / "mmbert-lora-full-ctx1024-u17000-s42"
    / "evaluation"
    / "evaluation.json"
)
PROMOTION_RECORD = (
    ROOT
    / "artifacts"
    / "models"
    / "mmbert-lora-full-ctx1024-u17000-s42"
    / "serving"
    / "promotion.json"
)
DEEPSEEK_0731_SUMMARY = (
    ROOT / "artifacts" / "openrouter_downstream_eval" / "deepseek_0731_summary.json"
)
FROZEN_PANEL = ROOT / "artifacts" / "openrouter_downstream_eval" / "panel.jsonl.gz"
DEEPSEEK_LOGPROB = (
    ROOT / "artifacts" / "openrouter_downstream_eval" / "deepseek_0731_results.jsonl.gz"
)
LOGINJECT_PANEL_MANIFEST = (
    ROOT / "artifacts" / "loginject_long_span_panel" / "manifest.json"
)
LOGINJECT_FUTURE_PROTOCOL = (
    "Future work must build a separate long-security-log calibration role and fresh "
    "shadow evaluation set, evaluate the finer `0.995` to `0.999` local-high grid "
    "only on that calibration role, freeze one threshold, and transport it unchanged "
    "to the fresh evaluation evidence."
)
LOGINJECT_RELIABILITY_NOTE = (
    "The initial concurrency-8 execution failed the predefined reliability gate, so "
    "the write-once replay resumed at concurrency 4 without changing thresholds, "
    "provider, prompt, or routing semantics."
)

# These values are retained report-level evidence whose row ledgers are not local.
# Keep the source and evidence status beside every number.
RETAINED_GUARDS = (
    {
        "system": "Historical mmBERT 512",
        "canonical_tpr_at_1pct": 0.552,
        "promptshield_tpr_at_1pct": 0.480,
        "sep_tpr_at_1pct": 0.388,
        "reserve_attested": 0.430,
        "reserve_bare_harmful": 0.093,
    },
    {
        "system": "Llama Prompt Guard 2 86M, retained current panel",
        "canonical_tpr_at_1pct": 0.43116,
        "promptshield_tpr_at_1pct": 0.15742,
        "sep_tpr_at_1pct": 0.03308,
        "reserve_attested": 0.800,
        "reserve_bare_harmful": 0.026,
    },
    {
        "system": "ModernGuard-1",
        "canonical_tpr_at_1pct": 0.000,
        "promptshield_tpr_at_1pct": 0.001,
        "sep_tpr_at_1pct": 0.024,
        "reserve_attested": 0.992,
        "reserve_bare_harmful": 0.813,
    },
    {
        "system": "Qwen3Guard Stream 4B, query head",
        "canonical_tpr_at_1pct": 0.017,
        "promptshield_tpr_at_1pct": 0.059,
        "sep_tpr_at_1pct": 0.005,
        "reserve_attested": 0.988,
        "reserve_bare_harmful": 0.957,
    },
    {
        "system": "Qwen3Guard Stream 4B, jailbreak head",
        "canonical_tpr_at_1pct": 0.377,
        "promptshield_tpr_at_1pct": 0.092,
        "sep_tpr_at_1pct": 0.080,
        "reserve_attested": 0.000,
        "reserve_bare_harmful": 0.001,
    },
    {
        "system": "Kanana Safeguard Prompt 2.1B",
        "canonical_tpr_at_1pct": 0.3773,
        "promptshield_tpr_at_1pct": 0.0170,
        "sep_tpr_at_1pct": 0.0302,
        "reserve_attested": 0.8396,
        "reserve_bare_harmful": 0.1083,
    },
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _pct(value: float | None, digits: int = 2) -> str:
    return "unavailable" if value is None else f"{100 * value:.{digits}f}%"


def _number(value: float | None, digits: int = 3) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def _interval_pct(value: list[float] | tuple[float, float]) -> str:
    return f"{_pct(value[0])} to {_pct(value[1])}"


def _delta_pp(value: float | None, interval: list[float] | None = None) -> str:
    if value is None:
        return "unavailable"
    result = f"{100 * value:+.2f} pp"
    if interval is not None:
        result += f" ({100 * interval[0]:+.2f} to {100 * interval[1]:+.2f})"
    return result


def _display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _evidence(status: str, source: Path | None, note: str) -> dict[str, str | None]:
    return {
        "status": status,
        "source": _display_path(source) if source is not None else None,
        "source_sha256": _sha256(source)
        if source is not None and source.is_file()
        else None,
        "note": note,
    }


def _profiles(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if summary is None:
        return []
    selected = summary.get("selection", {}).get("logprob", {})
    measured = summary.get("current_1024_logprob_cascade", {})
    rows = []
    for name in ("conservative", "balanced", "high_recall"):
        result = measured.get(name)
        choice = selected.get(name)
        if not isinstance(result, dict) or not isinstance(choice, dict):
            continue
        aggregate = result["aggregate"]
        rows.append(
            {
                "profile": name,
                "status": "measured_artifact_level_approximation",
                "approximation_scope": "Exact for single-window artifacts; retained full-artifact reviewer probabilities do not replay maintained middle-window review semantics for multi-window artifacts.",
                "thresholds": choice["thresholds"],
                "rows": aggregate["rows"],
                "recall": aggregate["recall"],
                "recall_95": aggregate["recall_95"],
                "fpr": aggregate["fpr"],
                "fpr_95": aggregate["fpr_95"],
                "precision": aggregate["precision"],
                "precision_95": aggregate["precision_95"],
                "restriction_rate": aggregate["restriction_rate"],
                "provider_call_rate": result["provider_call_rate"],
                "provider_calls": result["provider_calls"],
                "invalid_called_reviews": result["invalid_called_reviews"],
                "by_slice": result["by_slice"],
            }
        )
    return rows


def _exact_stage(
    input_dir: Path,
    selection_name: str,
    evaluation_name: str,
    *,
    cascade: bool = False,
) -> dict[str, Any] | None:
    selection_path = input_dir / selection_name
    evaluation_path = input_dir / evaluation_name
    selection = _read_json(selection_path)
    evaluation = _read_json(evaluation_path)
    if selection is None or evaluation is None:
        return None
    if (
        evaluation.get("advisory_only") is not True
        or type(evaluation.get("rows")) is not int
        or evaluation["rows"] < 1
    ):
        return None
    semantics = selection.get("profile_semantics")
    selection_is_exact = semantics == "maintained_multi_window_exact" or (
        isinstance(semantics, dict)
        and semantics.get("threshold_selection") == "maintained_multi_window_exact"
        and semantics.get("end_to_end_exact") is True
    )
    if cascade and (
        not selection_is_exact
        or evaluation.get("evaluation_semantics") != "maintained_multi_window_exact"
        or not isinstance(selection.get("profiles"), dict)
        or not isinstance(evaluation.get("profiles"), dict)
    ):
        return None
    return {
        "selection": selection,
        "evaluation": evaluation,
        "selection_source": _display_path(selection_path),
        "selection_sha256": _sha256(selection_path),
        "evaluation_source": _display_path(evaluation_path),
        "evaluation_sha256": _sha256(evaluation_path),
    }


def _exact_profiles(stage: dict[str, Any] | None, arm: str) -> list[dict[str, Any]]:
    if stage is None:
        return []
    selection = stage["selection"]
    evaluation = stage["evaluation"]
    infeasibility = selection.get(
        "profile_infeasibility", selection.get("selected_profile_infeasibility", {})
    )
    rows = []
    for name in ("conservative", "balanced", "high_recall"):
        selected = selection["profiles"].get(name)
        measured = evaluation["profiles"].get(name)
        if selected is None or measured is None:
            rows.append(
                {
                    "profile": name,
                    "arm": arm,
                    "status": "infeasible",
                    "reason": infeasibility.get(name),
                    "rows": evaluation["rows"],
                }
            )
            continue
        aggregate = measured["aggregate"]
        reviewed = measured["artifacts_with_provider_review"]
        rows.append(
            {
                "profile": name,
                "arm": arm,
                "status": "measured_maintained_multi_window_exact",
                "thresholds": selected["thresholds"],
                "rows": aggregate["rows"],
                "recall": aggregate["recall"],
                "recall_95": aggregate["recall_95"],
                "fpr": aggregate["fpr"],
                "fpr_95": aggregate["fpr_95"],
                "precision": aggregate["precision"],
                "precision_95": aggregate["precision_95"],
                "restriction_rate": aggregate["restriction_rate"],
                "artifacts_with_provider_review": reviewed,
                "provider_call_rate": reviewed / aggregate["rows"],
                "artifact_review_units": measured["artifact_review_units"],
                "window_review_units": measured["window_review_units"],
                "provider_review_units": measured["provider_review_units"],
                "invalid_called_reviews": measured["invalid_called_reviews"],
                "by_slice": measured["by_slice"],
                "prevalence_projections": measured["prevalence_projections"],
            }
        )
    return rows


def _exact_standalone_rows(stage: dict[str, Any] | None) -> list[dict[str, Any]]:
    if stage is None:
        return []
    evaluation = stage["evaluation"]
    rows = []
    logprob = evaluation.get("cloudflare_logprob", {})
    for coordinate, value in sorted(
        logprob.get("fixed_fpr", {}).items(), key=lambda item: float(item[0])
    ):
        if not isinstance(value, dict) or not isinstance(value.get("evaluation"), dict):
            continue
        aggregate = value["evaluation"]["aggregate"]
        rows.append(
            {
                "contract": "Cloudflare logprob threshold",
                "provider": "cloudflare",
                "transport": "strict_logprob",
                "target_fpr": value["target_fpr"],
                "threshold": value["threshold"],
                "rows": aggregate["rows"],
                "recall": aggregate["recall"],
                "fpr": aggregate["fpr"],
                "precision": aggregate["precision"],
                "invalid_outputs": value["evaluation"]["invalid_outputs"],
            }
        )
    for key, label in (
        (
            "cloudflare_logprob_hard_verdict",
            "Cloudflare same-request hard verdict",
        ),
        (
            "decart_true_no_logprob_hard_verdict",
            "Decart true no-logprob hard verdict",
        ),
    ):
        value = evaluation.get(key)
        if not isinstance(value, dict):
            continue
        aggregate = value["aggregate"]
        contract = evaluation["contracts"][key]
        rows.append(
            {
                "contract": label,
                "provider": contract["provider"],
                "transport": contract["transport"],
                "target_fpr": None,
                "threshold": None,
                "rows": aggregate["rows"],
                "recall": aggregate["recall"],
                "fpr": aggregate["fpr"],
                "precision": aggregate["precision"],
                "invalid_outputs": value["invalid_outputs"],
            }
        )
    return rows


def _traffic_mix_projections(
    summary: dict[str, Any] | None,
    *,
    channels: dict[str, Any] | None = None,
    status: str = "arithmetic_artifact_level_approximation",
) -> list[dict[str, Any]]:
    if channels is None:
        if summary is None:
            return []
        channels = summary["current_1024_logprob_cascade"]["balanced"][
            "by_input_channel"
        ]
    if not {"direct_user", "untrusted_content"} <= channels.keys():
        return []
    direct = channels["direct_user"]
    untrusted = channels["untrusted_content"]
    rows = []
    for label, direct_share in (("90/10", 0.9), ("50/50", 0.5), ("20/80", 0.2)):
        untrusted_share = 1 - direct_share
        recall = direct_share * float(direct["recall"]) + untrusted_share * float(
            untrusted["recall"]
        )
        fpr = direct_share * float(direct["fpr"]) + untrusted_share * float(
            untrusted["fpr"]
        )
        rows.append(
            {
                "status": status,
                "traffic_mix": label,
                "direct_share": direct_share,
                "untrusted_share": untrusted_share,
                "mixed_recall": recall,
                "mixed_fpr": fpr,
                "assumption": "Attack prevalence is held equal within direct and untrusted channels.",
                "prevalence_projections": metrics.prevalence_projections(recall, fpr),
            }
        )
    return rows


def _fixed_fpr(
    summary: dict[str, Any] | None, key: str, system: str
) -> list[dict[str, Any]]:
    if summary is None or not isinstance(summary.get(key), dict):
        return []
    rows = []
    for target, result in sorted(summary[key].items(), key=lambda item: float(item[0])):
        measured = result["evaluation"]
        rows.append(
            {
                "system": system,
                "status": "measured_consumed_evaluation",
                "target_fpr": result["target_fpr"],
                "threshold": result["threshold"],
                "rows": measured["rows"],
                "recall": measured["recall"],
                "fpr": measured["fpr"],
                "precision": measured["precision"],
                "auroc": measured["auroc"],
                "average_precision": measured["average_precision"],
            }
        )
    return rows


def _provider_canaries(input_dir: Path) -> list[dict[str, Any]]:
    metadata = _provider_metadata(input_dir)
    records = [
        row
        for row in _read_jsonl(input_dir / "provider_canary_results.jsonl")
        if row.get("stage") == "canary"
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["requested_provider"], row["transport"])].append(row)
    rows = []
    for (provider, transport), values in sorted(grouped.items()):
        valid = [row for row in values if row.get("status") == "ok"]
        seconds = [float(row["client_seconds"]) for row in values]
        cost = sum(float(row.get("cost_usd") or 0) for row in values)
        failures = Counter(
            str(row.get("failure_code")) for row in values if row.get("status") != "ok"
        )
        rows.append(
            {
                "provider": provider,
                "transport": transport,
                "quantization": metadata.get(provider.casefold(), {}).get(
                    "quantization", "unknown"
                ),
                "status": "canary_only",
                "rows": len(values),
                "valid_outputs": len(valid),
                "valid_output_rate": len(valid) / len(values),
                "latency_seconds_p50": statistics.median(seconds),
                "latency_seconds_p95": sorted(seconds)[
                    max(0, (95 * len(seconds) + 99) // 100 - 1)
                ],
                "failures": dict(sorted(failures.items())),
                "cost_usd": cost,
            }
        )
    return rows


def _provider_metadata(input_dir: Path) -> dict[str, dict[str, Any]]:
    snapshot = _read_json(input_dir / "openrouter_endpoint_snapshot.json")
    if snapshot is None:
        return {}
    endpoints = snapshot.get("data", {}).get("endpoints", [])
    return {
        str(endpoint["provider_name"]).casefold(): {
            "quantization": endpoint.get("quantization", "unknown"),
            "advertised_parameters": endpoint.get("supported_parameters", []),
            "frozen_uptime": endpoint.get("uptime_last_30m"),
        }
        for endpoint in endpoints
        if isinstance(endpoint, dict) and isinstance(endpoint.get("provider_name"), str)
    }


def _provider_panel_rows(
    input_dir: Path,
    summary: dict[str, Any] | None,
    hard_selection: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if summary is None:
        return []
    metadata = _provider_metadata(input_dir)
    winners = {
        (value["provider"], value["transport"]): arm
        for arm, value in summary.get("winners", {}).items()
        if isinstance(value, dict)
    }
    if (
        hard_selection
        and hard_selection.get("selection_status") == "no_eligible_provider"
    ):
        winners = {key: arm for key, arm in winners.items() if arm != "hard_verdict"}
    rows = []
    for value in summary.get("providers", {}).values():
        if not isinstance(value, dict):
            continue
        provider = value["provider"]
        transport = value["transport"]
        quality = value["quality"]["aggregate"]
        endpoint = metadata.get(provider.casefold(), {})
        rows.append(
            {
                "provider": provider,
                "transport": transport,
                "contract": transport.replace("_", " "),
                "quantization": endpoint.get("quantization", "unknown"),
                "rows": value["rows"],
                "recall": quality["recall"],
                "fpr": quality["fpr"],
                "valid_output_rate": value["valid_output_rate"],
                "latency_seconds": value["latency_seconds"],
                "cost_usd": value["cost_usd"],
                "winner": winners.get((provider, transport)),
            }
        )
    return sorted(rows, key=lambda row: (row["transport"], row["provider"]))


def _provider_load_rows(value: dict[str, Any] | None) -> list[dict[str, Any]]:
    if value is None or value.get("samples_are_unique_across_cells") is not True:
        return []
    rows = []
    for cell in value.get("cells", []):
        if not isinstance(cell, dict) or not cell.get("requests"):
            continue
        provider, transport, _ = cell["cell_id"].split(":", 2)
        rows.append(
            {
                "provider": provider,
                "transport": transport,
                **cell,
            }
        )
    return rows


def _azure_resource_rows(value: dict[str, Any] | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    rows = []
    for metric in value.get("resource_metrics", {}).get("value", []):
        if not isinstance(metric, dict):
            continue
        data = [
            point
            for series in metric.get("timeseries", [])
            for point in series.get("data", [])
            if isinstance(point, dict)
        ]
        name = metric.get("name", {})
        totals = [
            float(point["total"]) for point in data if point.get("total") is not None
        ]
        rows.append(
            {
                "metric": name.get("value", name.get("localizedValue", "unknown")),
                "unit": metric.get("unit"),
                "samples": len(data),
                "maximum": max(
                    (
                        float(point["maximum"])
                        for point in data
                        if point.get("maximum") is not None
                    ),
                    default=None,
                ),
                "maximum_average": max(
                    (
                        float(point["average"])
                        for point in data
                        if point.get("average") is not None
                    ),
                    default=None,
                ),
                "maximum_total": max(totals) if totals else None,
            }
        )
    return rows


def _runtime(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if summary is None:
        return []
    rows = []
    for key, system in (
        ("runtime", "Morgott 1024 CUDA BF16"),
        ("prompt_guard_runtime", "Prompt Guard 2 CUDA FP16"),
    ):
        record = summary.get(key)
        if not isinstance(record, dict) or not isinstance(record.get("runtime"), dict):
            continue
        runtime = record["runtime"]
        rows.append(
            {
                "system": system,
                "status": "measured_local_cuda",
                "artifacts": runtime["artifacts"],
                "windows": runtime["windows"],
                "score_seconds": runtime["score_seconds"],
                "artifacts_per_second": runtime["artifacts_per_second"],
                "input_tokens_per_second": runtime["input_tokens_per_second"],
                "peak_allocated_bytes": runtime["peak_allocated_bytes"],
                "peak_reserved_bytes": runtime["peak_reserved_bytes"],
                "batch_size": runtime.get(
                    "selected_batch_size", runtime.get("batch_size")
                ),
                "device": runtime["device"],
                "dtype": runtime["dtype"],
            }
        )
    deepseek = summary.get("retained_deepseek_runtime")
    if isinstance(deepseek, dict):
        rows.append(
            {
                "system": "DeepSeek 0731 Cloudflare retained ledger",
                "status": "retained_nonconcurrent_client_measurement",
                "artifacts": deepseek["rows"],
                "latency_seconds": deepseek["latency_seconds"],
                "serial_requests_per_second": deepseek["serial_requests_per_second"],
                "serial_input_tokens_per_second": deepseek[
                    "serial_input_tokens_per_second"
                ],
            }
        )
    return rows


def _retained_baselines() -> list[dict[str, Any]]:
    common = {
        "status": "retained_report_level_only",
        "source": str(RETAINED_EXPERIMENTS.relative_to(ROOT)),
        "paired_row_ledger_available": False,
    }
    rows = [{**row, **common} for row in RETAINED_GUARDS]
    rows.append(
        {
            "system": "GPT-OSS Safeguard 20B, shared binary prompt",
            "status": "retained_report_level_only",
            "source": str(RETAINED_LLM_REPORT.relative_to(ROOT)),
            "valid_output_recall": 0.1275,
            "valid_output_fpr": 0.0006,
            "valid_output_precision": 0.9936,
            "promptshield_recall": 0.0663,
            "sep_recall": 0.0,
            "mean_latency_seconds": 1.64,
            "p95_latency_seconds": 10.58,
            "cost_per_1000_inputs_usd": 0.0687,
            "paired_row_ledger_available": True,
        }
    )
    return rows


def _registered_1024() -> dict[str, Any] | None:
    value = _read_json(REGISTERED_EVALUATION)
    if value is None:
        return None
    return {
        "system": "Registered Morgott 1024 native evaluation",
        "status": "retained_consumed_development",
        "canonical_tpr_at_1pct": value["canonical_dev_test"]["metrics"][
            "descriptive_same_test"
        ]["1.0000%"]["recall"],
        "promptshield_tpr_at_1pct": value["promptshield_test"]["metrics"][
            "descriptive_same_test"
        ]["1.0000%"]["recall"],
        "sep_tpr_at_1pct": value["sep"]["metrics"]["descriptive_same_test"]["1.0000%"][
            "recall"
        ],
        "finance_false_positives": value["real_finance_negatives"]["metrics"][
            "false_positive"
        ],
        "finance_rows": value["real_finance_negatives"]["metrics"]["rows"],
        "source": str(REGISTERED_EVALUATION.relative_to(ROOT)),
    }


def _deepseek_standalone() -> dict[str, Any] | None:
    value = _read_json(DEEPSEEK_0731_SUMMARY)
    if value is None:
        return None
    candidate = value["standalone_evaluation"]["candidate"]
    return {
        "system": "DeepSeek V4 Flash 0731 standalone logprob",
        "status": "retained_consumed_evaluation",
        "auroc": candidate["auroc"],
        "average_precision": candidate["average_precision"],
        "provider": value["selected_provider"],
        "source": str(DEEPSEEK_0731_SUMMARY.relative_to(ROOT)),
    }


def _long_negative_decomposition(
    input_dir: Path, summary: dict[str, Any] | None
) -> dict[str, Any] | None:
    manifest = _read_json(input_dir / "manifest.json")
    if (
        summary is None
        or manifest is None
        or not FROZEN_PANEL.is_file()
        or not DEEPSEEK_LOGPROB.is_file()
    ):
        return None
    local_records = _read_jsonl(input_dir / "morgott_1024_scores.jsonl.gz")
    if not local_records:
        return None
    panel = _read_jsonl(FROZEN_PANEL)
    local_by_id = {row["artifact_id"]: row for row in local_records}
    reviewer_by_id = {row["panel_id"]: row for row in _read_jsonl(DEEPSEEK_LOGPROB)}
    calibration_ids = set(manifest["roles"]["calibration_panel_ids"])
    thresholds = summary["selection"]["logprob"]["balanced"]["thresholds"]
    rows = [
        row
        for row in panel
        if row["panel_id"] not in calibration_ids
        and row["label"] == 0
        and row["text_chars"] >= 4096
    ]
    if any(
        row["panel_id"] not in local_by_id or row["panel_id"] not in reviewer_by_id
        for row in rows
    ):
        return None
    local_high = 0
    middle = []
    reviewer_flagged = 0
    for row in rows:
        score = float(local_by_id[row["panel_id"]]["local_score"])
        low = float(thresholds[f"{row['input_channel'].split('_')[0]}_low"])
        if score >= float(thresholds["local_high"]):
            local_high += 1
        elif score >= low:
            middle.append(row)
            reviewer = reviewer_by_id[row["panel_id"]]
            if reviewer.get("status") != "ok" or float(
                reviewer["p_subversion"]
            ) >= float(thresholds["reviewer"]):
                reviewer_flagged += 1
    return {
        "status": "measured_exact_first_full_context_review",
        "negatives": len(rows),
        "local_high_restrictions": local_high,
        "middle_zone": len(middle),
        "full_context_reviewer_false_restrictions": reviewer_flagged,
        "fpr": (local_high + reviewer_flagged) / len(rows),
        "thresholds": thresholds,
        "limitations": "This decomposition is exact through the first full-context review but does not replay later middle-window reviews.",
    }


def _paired_detector_deltas(
    input_dir: Path, summary: dict[str, Any] | None
) -> list[dict[str, Any]]:
    manifest = _read_json(input_dir / "manifest.json")
    morgott_path = input_dir / "morgott_1024_scores.jsonl.gz"
    prompt_guard_path = input_dir / "prompt_guard_2_86m_scores.jsonl.gz"
    if (
        summary is None
        or manifest is None
        or not FROZEN_PANEL.is_file()
        or not morgott_path.is_file()
        or not prompt_guard_path.is_file()
    ):
        return []
    calibration_ids = set(manifest["roles"]["calibration_panel_ids"])
    evaluation = [
        row
        for row in _read_jsonl(FROZEN_PANEL)
        if row["panel_id"] not in calibration_ids
    ]
    morgott = {row["artifact_id"]: row for row in _read_jsonl(morgott_path)}
    prompt_guard = {row["artifact_id"]: row for row in _read_jsonl(prompt_guard_path)}
    if any(
        row["panel_id"] not in morgott or row["panel_id"] not in prompt_guard
        for row in evaluation
    ):
        return []
    labels = np.asarray([row["label"] for row in evaluation], dtype=np.int8)
    morgott_scores = np.asarray(
        [morgott[row["panel_id"]]["local_score"] for row in evaluation]
    )
    prompt_guard_scores = np.asarray(
        [prompt_guard[row["panel_id"]]["local_score"] for row in evaluation]
    )
    rows = []
    for target in sorted(summary["current_1024_standalone"], key=float):
        morgott_result = summary["current_1024_standalone"][target]
        prompt_guard_result = summary["prompt_guard_2_86m"][target]
        comparison = metrics.paired_stratified_bootstrap_delta(
            labels,
            prompt_guard_scores >= float(prompt_guard_result["threshold"]),
            morgott_scores >= float(morgott_result["threshold"]),
            iterations=2_000,
            seed=42,
        )
        rows.append(
            {
                "status": "measured_consumed_evaluation",
                "target_fpr": float(target),
                "rows": len(evaluation),
                "candidate": "Morgott 1024 all-window",
                "incumbent": "Prompt Guard 2 86M segmented",
                "candidate_threshold": morgott_result["threshold"],
                "incumbent_threshold": prompt_guard_result["threshold"],
                **comparison,
            }
        )
    return rows


def _loginject_posthoc_diagnostic(input_dir: Path) -> list[dict[str, Any]]:
    scores_path = input_dir / "loginject_local_scores.jsonl.gz"
    records = _read_jsonl(scores_path)
    if not records:
        return []
    labels = np.asarray([row["label"] for row in records], dtype=np.int8)
    scores = np.asarray([row["local_score"] for row in records], dtype=np.float64)
    rows = []
    for target in (0.0, 0.001, 0.01, 0.02, 0.0625, 0.32):
        threshold = metrics.select_threshold_at_fpr(labels, scores, target)
        measured = metrics.score_metrics(labels, scores, threshold)
        rows.append(
            {
                "status": "sealed_post_hoc_diagnostic_only",
                "target_fpr": target,
                "threshold": threshold,
                "recall": measured["recall"],
                "observed_fpr": measured["fpr"],
                "positives": measured["positives"],
                "negatives": measured["negatives"],
                "true_positive": measured["tp"],
                "false_positive": measured["fp"],
                "selection_eligible": False,
            }
        )
    return rows


def _openvino_fixed_fpr(
    input_dir: Path, summary: dict[str, Any] | None
) -> list[dict[str, Any]]:
    manifest = _read_json(input_dir / "manifest.json")
    scores_path = input_dir / "morgott_1024_openvino_scores.jsonl.gz"
    if summary is None or manifest is None or not scores_path.is_file():
        return []
    records = _read_jsonl(scores_path)
    if not records:
        return []
    calibration_ids = set(manifest["roles"]["calibration_panel_ids"])
    calibration = np.asarray(
        [row["artifact_id"] in calibration_ids for row in records], dtype=bool
    )
    labels = np.asarray([row["label"] for row in records], dtype=np.int8)
    scores = np.asarray([row["local_score"] for row in records], dtype=np.float64)
    openvino = metrics.fixed_fpr_evaluation(
        labels[calibration],
        scores[calibration],
        labels[~calibration],
        scores[~calibration],
    )
    rows = []
    for target in sorted(summary["current_1024_standalone"], key=float):
        cuda = summary["current_1024_standalone"][target]
        current_openvino = openvino[target]
        cuda_evaluation = cuda["evaluation"]
        openvino_evaluation = current_openvino["evaluation"]
        deltas = {
            name: float(openvino_evaluation[name]) - float(cuda_evaluation[name])
            for name in (
                "recall",
                "fpr",
                "precision",
                "restriction_rate",
                "auroc",
                "average_precision",
            )
        }
        rows.append(
            {
                "status": "measured_runtime_specific_consumed_evaluation",
                "target_fpr": float(target),
                "calibration_rows": int(np.sum(calibration)),
                "evaluation_rows": int(np.sum(~calibration)),
                "cuda": {
                    "threshold": cuda["threshold"],
                    "evaluation": cuda_evaluation,
                },
                "openvino": {
                    "threshold": current_openvino["threshold"],
                    "evaluation": openvino_evaluation,
                },
                "delta_openvino_minus_cuda": deltas,
            }
        )
    return rows


def build_tables(input_dir: Path) -> dict[str, Any]:
    manifest = _read_json(input_dir / "manifest.json")
    source_provenance_path = input_dir / "source-provenance.json"
    source_provenance = _read_json(source_provenance_path)
    if source_provenance is not None:
        archived = source_provenance["hard_verdict_pre_source_gate"]
        for key in ("source", "selection", "evaluation"):
            archived_path = ROOT / archived[f"{key}_path"]
            if _sha256(archived_path) != archived[f"{key}_sha256"]:
                raise ValueError("archived benchmark evidence identity changed")
    summary_path = input_dir / "summary.json"
    summary = _read_json(summary_path)
    provider_summary_path = input_dir / "provider_summary.json"
    provider_summary = _read_json(provider_summary_path)
    provider_panel_measured = bool(
        provider_summary
        and (
            provider_summary.get("providers")
            or any(provider_summary.get("winners", {}).values())
        )
    )
    exact_logprob_selection_path = input_dir / "logprob_exact_selection.json"
    exact_logprob_evaluation_path = input_dir / "logprob_exact_evaluation.json"
    exact_logprob = _exact_stage(
        input_dir,
        exact_logprob_selection_path.name,
        exact_logprob_evaluation_path.name,
        cascade=True,
    )
    exact_logprob_profiles = _exact_profiles(exact_logprob, "logprob")
    exact_balanced = next(
        (
            row
            for row in exact_logprob_profiles
            if row["profile"] == "balanced" and row["status"] != "infeasible"
        ),
        None,
    )
    promotion = _read_json(PROMOTION_RECORD)
    if promotion is not None and (
        promotion.get("format") != "morgott-advisory-cascade-profile-v1"
        or promotion.get("status") != "maintained_advisory"
        or promotion.get("advisory_only") is not True
    ):
        raise ValueError("maintained promotion record is invalid")
    azure_deployment = _read_json(AZURE_DEPLOYMENT_RECORD)
    azure_deployment_verified = bool(
        azure_deployment
        and promotion
        and azure_deployment.get("advisory_only") is True
        and azure_deployment.get("traffic_percent") == 100
        and azure_deployment.get("pipeline_profile")
        == promotion["runtime_contract"]["profile"]
        and azure_deployment.get("policy_sha256") == _sha256(PROMOTION_RECORD)
        and azure_deployment.get("threshold_sha256")
        == promotion["runtime_contract"]["threshold_sha256"]
    )
    exact_hard_selection_path = input_dir / "hard_verdict_selection.json"
    exact_hard_evaluation_path = input_dir / "hard_verdict_evaluation.json"
    hard_selection = _read_json(exact_hard_selection_path)
    hard_ineligible = bool(
        hard_selection
        and hard_selection.get("selection_status") == "no_eligible_provider"
    )
    exact_hard = _exact_stage(
        input_dir,
        exact_hard_selection_path.name,
        exact_hard_evaluation_path.name,
        cascade=True,
    )
    hard_diagnostic = (
        _exact_stage(
            input_dir,
            "hard_verdict_selection_pre_source_gate.json",
            "hard_verdict_evaluation_pre_source_gate.json",
            cascade=True,
        )
        if hard_ineligible
        else None
    )
    hard_profiles = _exact_profiles(
        hard_diagnostic if hard_ineligible else exact_hard, "hard_verdict"
    )
    if hard_ineligible:
        for row in hard_profiles:
            if row["status"] != "infeasible":
                row["status"] = "diagnostic_failed_source_slice_gate"
    provider_panel = provider_summary
    if provider_summary is not None and hard_ineligible:
        provider_panel = json.loads(json.dumps(provider_summary))
        provider_panel.setdefault("winners", {})["hard_verdict"] = None
    cascade_flow_path = input_dir / "cascade_flow_comparison.json"
    cascade_flow = _read_json(cascade_flow_path)
    cascade_flow_measured = bool(
        cascade_flow
        and cascade_flow.get("status") == "post_hoc_comparison_on_consumed_evaluation"
        and cascade_flow.get("evaluation_rows") == 12_352
        and isinstance(cascade_flow.get("profiles"), dict)
    )
    reviewer_prompt_path = input_dir / "reviewer_prompt_experiment_summary.json"
    reviewer_prompt = _read_json(reviewer_prompt_path)
    reviewer_prompt_measured = bool(
        reviewer_prompt
        and reviewer_prompt.get("selection_eligible") is False
        and reviewer_prompt.get("records") == 460
        and isinstance(reviewer_prompt.get("arms"), dict)
        and isinstance(reviewer_prompt.get("cascade_evaluation"), dict)
    )
    reviewer_long_path = input_dir / "reviewer_long_bucket_summary.json"
    reviewer_long = _read_json(reviewer_long_path)
    reviewer_long_measured = bool(
        reviewer_long
        and reviewer_long.get("selection_eligible") is False
        and reviewer_long.get("rows") == 193
        and reviewer_long.get("records") == 386
        and isinstance(reviewer_long.get("arms"), dict)
    )
    reviewer_patch_path = input_dir / "reviewer_prompt_patch_summary.json"
    reviewer_patch = _read_json(reviewer_patch_path)
    reviewer_patch_measured = bool(
        reviewer_patch
        and reviewer_patch.get("selection_eligible") is False
        and reviewer_patch.get("rows") == 193
        and reviewer_patch.get("calls") == 73
        and isinstance(reviewer_patch.get("channel_split_candidate"), dict)
    )
    reviewer_screen_path = input_dir / "reviewer_channel_split_screen_summary.json"
    reviewer_screen = _read_json(reviewer_screen_path)
    reviewer_screen_measured = bool(
        reviewer_screen
        and reviewer_screen.get("selection_eligible") is False
        and reviewer_screen.get("rows") == 256
        and reviewer_screen.get("recommendation") == "do_not_proceed"
        and isinstance(reviewer_screen.get("current"), dict)
        and isinstance(reviewer_screen.get("candidate"), dict)
    )
    standalone_selection_path = input_dir / "deepseek_standalone_selection.json"
    standalone_evaluation_path = input_dir / "deepseek_standalone_evaluation.json"
    exact_standalone = _exact_stage(
        input_dir,
        standalone_selection_path.name,
        standalone_evaluation_path.name,
    )
    provider_load_path = input_dir / "provider_load.json"
    provider_load = _read_json(provider_load_path)
    provider_load_rows = _provider_load_rows(provider_load)
    azure_path = input_dir / "azure_load.json"
    azure = _read_json(azure_path)
    budget = _read_json(input_dir / "budget_reservations.json")
    azure_measured = bool(
        azure
        and isinstance(azure.get("status"), dict)
        and azure.get("cells")
        and isinstance(azure.get("resource_metrics"), dict)
    )
    azure_promoted_profile = bool(
        azure_measured
        and promotion is not None
        and azure["status"].get("pipeline_profile")
        == promotion["runtime_contract"]["profile"]
        and azure["status"].get("threshold_sha256")
        == promotion["runtime_contract"]["threshold_sha256"]
    )
    parity_path = input_dir / "openvino_parity.json"
    parity = _read_json(parity_path)
    openvino_scores_path = input_dir / "morgott_1024_openvino_scores.jsonl.gz"
    openvino_runtime_path = input_dir / "morgott_1024_openvino_runtime.json"
    openvino_runtime = _read_json(openvino_runtime_path)
    openvino_full_replay = (
        openvino_scores_path.is_file() and openvino_runtime is not None
    )
    openvino_quality = _openvino_fixed_fpr(input_dir, summary)
    mutation_path = input_dir / "mutation_1024_summary.json"
    loginject_path = input_dir / "loginject_summary.json"
    loginject = _read_json(loginject_path)
    loginject_remote_path = input_dir / "loginject_remote_summary.json"
    loginject_remote = _read_json(loginject_remote_path)
    loginject_remote_measured = bool(
        loginject_remote
        and loginject_remote.get("sealed_once") is True
        and type(loginject_remote.get("pairs")) is int
        and loginject_remote["pairs"] > 0
        and isinstance(loginject_remote.get("profiles"), dict)
    )
    loginject_panel_manifest = _read_json(LOGINJECT_PANEL_MANIFEST)
    gpt_oss_native_path = input_dir / "gpt_oss_native_summary.json"
    gpt_oss_native = _read_json(gpt_oss_native_path)
    provider_canary_path = input_dir / "provider_canary_results.jsonl"
    evidence_status = {
        "maintained_promotion": _evidence(
            "promoted_advisory_default" if promotion is not None else "pending",
            PROMOTION_RECORD if promotion is not None else None,
            (
                "The owner-promoted balanced profile changes advisory routing defaults only; it does not authorize blocking or establish production traffic quality."
                if promotion is not None
                else "No registry-bound maintained cascade profile is present."
            ),
        ),
        "exact_logprob_cascade": _evidence(
            (
                "measured_maintained_multi_window_exact"
                if exact_logprob is not None
                else "selection_frozen_evaluation_pending"
                if exact_logprob_selection_path.is_file()
                else "pending"
            ),
            (
                exact_logprob_evaluation_path
                if exact_logprob is not None
                else exact_logprob_selection_path
                if exact_logprob_selection_path.is_file()
                else None
            ),
            "Only the provider-safe evaluation and its required window ledger support exact maintained-cascade claims; a frozen calibration selection alone does not.",
        ),
        "exact_hard_verdict_cascade": _evidence(
            (
                "no_eligible_strict_provider"
                if hard_ineligible
                else "measured_maintained_multi_window_exact"
                if exact_hard is not None
                else "selection_frozen_evaluation_pending"
                if exact_hard_selection_path.is_file()
                else "pending"
            ),
            (
                exact_hard_selection_path
                if hard_ineligible
                else exact_hard_evaluation_path
                if exact_hard is not None
                else exact_hard_selection_path
                if exact_hard_selection_path.is_file()
                else None
            ),
            (
                "All strict no-logprob providers failed at least one declared source-slice or overall quality gate; the earlier Decart evaluation is retained only as a pre-source-gate diagnostic."
                if hard_ineligible
                else "The conservative hard-verdict profile may remain null when no calibration candidate satisfies its constraints."
            ),
        ),
        "benchmark_source_provenance": _evidence(
            (
                "clean_committed_source"
                if manifest
                and manifest.get("source_provenance", {}).get("clean_committed_tree")
                is True
                else "legacy_incomplete_source_binding"
            ),
            (
                source_provenance_path
                if source_provenance is not None
                else input_dir / "manifest.json"
                if manifest is not None
                else None
            ),
            (
                "The frozen manifest predates the committed-source guard and names a base commit that did not contain this benchmark runner; parsed ledgers are retained, but exact source reconstruction is incomplete."
                if not manifest
                or manifest.get("source_provenance", {}).get("clean_committed_tree")
                is not True
                else "Prepare verified a clean committed benchmark source tree before freezing roles."
            ),
        ),
        "azure_deployment": _evidence(
            "verified_promoted_advisory" if azure_deployment_verified else "pending",
            AZURE_DEPLOYMENT_RECORD if azure_deployment_verified else None,
            (
                "The dated deployment record binds the live revision to the promoted profile, policy, thresholds, model, and ONNX identities; its 30 requests are a smoke test, not a load benchmark."
                if azure_deployment_verified
                else "No dated deployment record binds the promoted profile to a live revision."
            ),
        ),
        "cascade_flow_comparison": _evidence(
            "measured_post_hoc_consumed_evaluation"
            if cascade_flow_measured
            else "pending",
            cascade_flow_path if cascade_flow_measured else None,
            "The matched ablation removes only unconditional full-context review from long inputs and reuses the frozen Cloudflare artifact and window ledgers without new provider calls.",
        ),
        "reviewer_prompt_experiment": _evidence(
            "measured_post_hoc_redacted_browsesafe"
            if reviewer_prompt_measured
            else "pending",
            reviewer_prompt_path if reviewer_prompt_measured else None,
            "The public long BrowseSafe rows were deterministically safety-redacted before transmission, so this is a consumed, redaction-altered diagnostic rather than selection evidence.",
        ),
        "reviewer_long_bucket_experiment": _evidence(
            "measured_post_hoc_complete_redacted_long_bucket"
            if reviewer_long_measured
            else "pending",
            reviewer_long_path if reviewer_long_measured else None,
            "The matched comparison covers all 193 identities in the consumed long-character bucket, but deterministic safety redaction and synthetic trusted-task interpretation prevent selection use.",
        ),
        "reviewer_prompt_patch_experiment": _evidence(
            "measured_post_hoc_minimal_patch_and_channel_split"
            if reviewer_patch_measured
            else "pending",
            reviewer_patch_path if reviewer_patch_measured else None,
            "The one-sentence relaxation failed its frozen advance rule; the no-call channel-split simulation dominated the single revised prompt on the long bucket but was rejected by the broader screen.",
        ),
        "reviewer_channel_split_screen": _evidence(
            "measured_post_hoc_scenario_balanced_rejection"
            if reviewer_screen_measured
            else "pending",
            reviewer_screen_path if reviewer_screen_measured else None,
            "The 256-row consumed, scenario-balanced untrusted-content screen rejects the broad channel split because its recall loss overwhelms its one-false-positive improvement.",
        ),
        "exact_deepseek_standalone": _evidence(
            (
                "measured_provider_safe_evaluation"
                if exact_standalone is not None
                else "selection_frozen_evaluation_pending"
                if standalone_selection_path.is_file()
                else "pending"
            ),
            (
                standalone_evaluation_path
                if exact_standalone is not None
                else standalone_selection_path
                if standalone_selection_path.is_file()
                else None
            ),
            "The exact matched comparison separates logprob thresholding, the hard verdict returned by that same request, and a true request without logprob fields.",
        ),
        "local_quality": _evidence(
            "measured_artifact_level_approximation"
            if summary is not None
            else "pending",
            summary_path if summary is not None else None,
            "The 6,000-row role selected thresholds and the 14,000-row role measured an artifact-level approximation that is exact for single-window rows but not maintained multi-window cascade semantics.",
        ),
        "provider_canary": _evidence(
            "measured_canary_only" if provider_canary_path.is_file() else "pending",
            provider_canary_path if provider_canary_path.is_file() else None,
            "Sixteen rows per provider screen schema and routing compatibility, not quality or throughput.",
        ),
        "provider_panel": _evidence(
            "measured" if provider_panel_measured else "pending",
            provider_summary_path if provider_panel_measured else None,
            "The matched panel selects Cloudflare for strict logprobs; strict hard-verdict eligibility is decided by the later exact source-complete cascade gate.",
        ),
        "provider_load": _evidence(
            "measured" if provider_load_rows else "pending",
            provider_load_path if provider_load_rows else None,
            "Only the corrected unique-sample load artifact is consumed; archived length-confounded runs are excluded.",
        ),
        "openvino_parity": _evidence(
            "measured" if parity is not None else "pending",
            parity_path if parity is not None else None,
            (
                "The 512-row audit is complete, and runtime-specific quality stays separate when threshold-decision disagreement exceeds 0.5%."
                if parity is not None
                else "Runtime-specific quality remains separate until parity is measured."
            ),
        ),
        "openvino_full_quality": _evidence(
            (
                "measured_runtime_specific_fixed_fpr_quality"
                if openvino_quality
                else "measured_score_ledger_analysis_pending"
                if openvino_full_replay
                else "pending"
            ),
            openvino_runtime_path if openvino_full_replay else None,
            (
                "The full OpenVINO score ledger has separate 6,000-row calibration thresholds transported unchanged to the 14,000-row evaluation role."
                if openvino_quality
                else "The full OpenVINO score ledger and runtime are present, but runtime-specific quality analysis is pending."
                if openvino_full_replay
                else "The parity audit is not a substitute for a full runtime-specific quality replay."
            ),
        ),
        "azure_load": _evidence(
            (
                "measured_promoted_profile"
                if azure_promoted_profile
                else "measured_incumbent_only"
                if azure_measured
                else "pending"
            ),
            azure_path if azure_measured else None,
            (
                "The deployed identity includes the promoted profile and threshold hash; the report never infers deployed latency from local CUDA throughput."
                if azure_promoted_profile
                else "The retained load predates profile identity fields and is incumbent-only evidence, not promoted-profile end-to-end evidence."
                if azure_measured
                else "The report never infers deployed latency from local CUDA throughput."
            ),
        ),
        "mutation_1024": _evidence(
            "measured" if mutation_path.is_file() else "pending",
            mutation_path if mutation_path.is_file() else None,
            (
                "The registered 1024 model was replayed under pre-promotion incumbent gates; full-cascade review outcomes and promoted-gate mutation outcomes remain pending."
                if mutation_path.is_file()
                else "The retained mutation curve belongs to the old 512 model and is not reused as 1024 evidence."
            ),
        ),
        "loginject_local_routing": _evidence(
            "measured_sealed_once" if loginject is not None else "pending_sealed",
            loginject_path if loginject is not None else None,
            (
                "The sealed panel has one local-model score; its preliminary local profile rows are secondary to the exact promoted remote cascade result."
                if loginject is not None
                else "The sealed panel remains unavailable for conclusions until its one allowed local score is complete."
            ),
        ),
        "loginject_remote_cascade": _evidence(
            "measured_sealed_once" if loginject_remote_measured else "pending",
            loginject_remote_path if loginject_remote_measured else None,
            "Review-zone outcomes require the separately frozen provider phase and are not inferred from local routing.",
        ),
        "gpt_oss_native_screen": _evidence(
            "measured_supplementary_256_row_screen"
            if gpt_oss_native is not None
            else "unavailable",
            gpt_oss_native_path if gpt_oss_native is not None else None,
            (
                "The official-policy Harmony-style low-versus-medium screen is complete, but neither arm is reliable enough to expand or replace the main comparison arms."
                if gpt_oss_native is not None
                else "No completed native-policy artifact is available."
            ),
        ),
        "production_traffic": _evidence(
            "unavailable",
            None,
            "No representative adjudicated production traffic was supplied.",
        ),
    }
    retained = _retained_baselines()
    registered = _registered_1024()
    if registered is not None:
        retained.insert(0, registered)
    deepseek = _deepseek_standalone()
    if deepseek is not None:
        retained.append(deepseek)
    azure_table = None
    if azure_measured:
        azure_table = {
            key: azure[key]
            for key in (
                "status",
                "warm_only",
                "benchmark_started_utc",
                "benchmark_finished_utc",
                "estimated_remote_cost_usd",
                "maximum_remote_cost_usd",
                "cost_estimate_is_upper_bound",
                "recorded_provider_spend_before_usd",
                "prior_failed_azure_estimate_usd",
                "prior_failed_azure_ceiling_usd",
                "cells",
            )
            if key in azure
        }
        azure_table["policy_status"] = (
            "promoted_profile" if azure_promoted_profile else "incumbent_pre_promotion"
        )
    return {
        "schema_version": 1,
        "advisory_only": True,
        "production_fpr_claim": False,
        "evidence_status": evidence_status,
        "profiles": _profiles(summary),
        "exact_logprob": exact_logprob,
        "exact_logprob_profiles": exact_logprob_profiles,
        "exact_hard_verdict": exact_hard,
        "exact_hard_verdict_profiles": hard_profiles,
        "hard_verdict_selection": hard_selection,
        "hard_verdict_diagnostic": hard_diagnostic,
        "cascade_flow_comparison": cascade_flow if cascade_flow_measured else None,
        "reviewer_prompt_experiment": (
            reviewer_prompt if reviewer_prompt_measured else None
        ),
        "reviewer_long_bucket_experiment": (
            reviewer_long if reviewer_long_measured else None
        ),
        "reviewer_prompt_patch_experiment": (
            reviewer_patch if reviewer_patch_measured else None
        ),
        "reviewer_channel_split_screen": (
            reviewer_screen if reviewer_screen_measured else None
        ),
        "exact_deepseek_standalone": exact_standalone,
        "exact_deepseek_standalone_rows": _exact_standalone_rows(exact_standalone),
        "standalone_fixed_fpr": _fixed_fpr(
            summary, "current_1024_standalone", "Morgott 1024 all-window"
        )
        + _fixed_fpr(summary, "prompt_guard_2_86m", "Prompt Guard 2 86M segmented"),
        "paired_detector_deltas": _paired_detector_deltas(input_dir, summary),
        "retained_baselines": retained,
        "provider_canaries": _provider_canaries(input_dir),
        "provider_panel": provider_panel if provider_panel_measured else None,
        "provider_panel_rows": _provider_panel_rows(
            input_dir,
            provider_panel if provider_panel_measured else None,
            hard_selection,
        ),
        "provider_load": provider_load if provider_load_rows else None,
        "provider_load_rows": provider_load_rows,
        "runtime": _runtime(summary),
        "long_context": (
            {
                "status": "measured_artifact_level_approximation",
                "population": summary["long_context"],
                "balanced_by_length": summary["current_1024_logprob_cascade"][
                    "balanced"
                ]["by_slice"]["length_bucket"],
                "balanced_long_negative_decomposition": _long_negative_decomposition(
                    input_dir, summary
                ),
            }
            if summary is not None
            else None
        ),
        "prevalence_projections": (
            exact_balanced["prevalence_projections"]
            if exact_balanced is not None
            else summary["current_1024_logprob_cascade"]["balanced"][
                "prevalence_projections"
            ]
            if summary is not None
            else None
        ),
        "projection_basis": (
            "exact_provider_safe_balanced"
            if exact_balanced is not None
            else "artifact_level_approximate_balanced"
        ),
        "traffic_mix_projections": _traffic_mix_projections(
            summary,
            channels=(
                exact_balanced["by_slice"]["input_channel"]
                if exact_balanced is not None
                else None
            ),
            status=(
                "arithmetic_exact_provider_safe_cascade"
                if exact_balanced is not None
                else "arithmetic_artifact_level_approximation"
            ),
        ),
        "openvino_parity": parity,
        "openvino_fixed_fpr_quality": openvino_quality,
        "openvino_full_replay": (
            {
                "status": (
                    "measured_runtime_specific_fixed_fpr_quality"
                    if openvino_quality
                    else "measured_score_ledger_analysis_pending"
                ),
                "runtime": openvino_runtime,
                "runtime_source": _display_path(openvino_runtime_path),
                "runtime_sha256": _sha256(openvino_runtime_path),
                "scores_source": _display_path(openvino_scores_path),
                "scores_sha256": _sha256(openvino_scores_path),
                "fixed_fpr_quality": openvino_quality,
            }
            if openvino_full_replay
            else None
        ),
        "azure_load": azure_table,
        "azure_deployment": (azure_deployment if azure_deployment_verified else None),
        "budget": budget,
        "source_provenance": source_provenance,
        "azure_resource_metrics": _azure_resource_rows(
            azure if azure_measured else None
        ),
        "mutation_1024": _read_json(mutation_path),
        "loginject": loginject,
        "loginject_remote": loginject_remote if loginject_remote_measured else None,
        "loginject_panel_context": loginject_panel_manifest,
        "loginject_posthoc_diagnostic": _loginject_posthoc_diagnostic(input_dir),
        "gpt_oss_native_screen": gpt_oss_native,
        "maintained_promotion": promotion,
    }


def _profile_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Profile | Recall | FPR | Precision | Advisory restriction rate | DeepSeek calls | Thresholds direct / untrusted / high / reviewer |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        threshold = row["thresholds"]
        lines.append(
            "| {profile} | {recall} | {fpr} | {precision} | {restriction} | {calls} | `{direct:g} / {untrusted:g} / {high:g} / {reviewer:g}` |".format(
                profile=row["profile"].replace("_", " ").title(),
                recall=_pct(row["recall"]),
                fpr=_pct(row["fpr"]),
                precision=_pct(row["precision"]),
                restriction=_pct(row["restriction_rate"]),
                calls=_pct(row["provider_call_rate"]),
                direct=threshold["direct_low"],
                untrusted=threshold["untrusted_low"],
                high=threshold["local_high"],
                reviewer=threshold["reviewer"],
            )
        )
    return lines


def _exact_profile_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Profile | Recall | FPR | Precision | Artifact review rate | Review units artifact + window | Thresholds direct / untrusted / high / reviewer |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        profile = row["profile"].replace("_", " ").title()
        if row["status"] == "infeasible":
            reason = row.get("reason") or {}
            minimum = reason.get("minimum_observed_fpr")
            detail = (
                f"infeasible, minimum observed FPR {_pct(minimum)}"
                if minimum is not None
                else "infeasible"
            )
            lines.append(
                f"| {profile} | {detail} | unavailable | unavailable | unavailable | unavailable | unavailable |"
            )
            continue
        threshold = row["thresholds"]
        reviewer = threshold.get("reviewer")
        reviewer_text = "hard verdict" if reviewer is None else f"{reviewer:g}"
        lines.append(
            f"| {profile} | {_pct(row['recall'])} | {_pct(row['fpr'])} | {_pct(row['precision'])} | {_pct(row['provider_call_rate'])} | {row['artifact_review_units']} + {row['window_review_units']} | `{threshold['direct_low']:g} / {threshold['untrusted_low']:g} / {threshold['local_high']:g} / {reviewer_text}` |"
        )
    return lines


def _cascade_flow_markdown(comparison: dict[str, Any]) -> list[str]:
    lines = [
        "| Profile | Flow | Recall | FPR | Precision | Reviewed artifacts | Review units artifact + window |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for profile in ("conservative", "balanced", "high_recall"):
        value = comparison["profiles"].get(profile)
        if not isinstance(value, dict):
            continue
        for key, label in (
            ("maintained_full_context_first", "Full context first"),
            ("on_demand_same_thresholds", "Middle windows on demand"),
        ):
            row = value[key]
            aggregate = row["aggregate"]
            lines.append(
                f"| {profile.replace('_', ' ').title()} | {label} | {_pct(aggregate['recall'])} | {_pct(aggregate['fpr'])} | {_pct(aggregate['precision'])} | {row['artifacts_with_provider_review']:,} | {row['artifact_review_units']:,} + {row['window_review_units']:,} |"
            )
    return lines


def _exact_standalone_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Contract | Provider | Evaluation rows | Calibration FPR target | Recall | FPR | Precision | Invalid outputs |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        target = (
            "hard verdict" if row["target_fpr"] is None else _pct(row["target_fpr"], 1)
        )
        lines.append(
            f"| {row['contract']} | {row['provider']} | {row['rows']:,} | {target} | {_pct(row['recall'])} | {_pct(row['fpr'])} | {_pct(row['precision'])} | {row['invalid_outputs']} |"
        )
    return lines


def _provider_panel_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Provider | Contract | Quantization | Rows | Valid outputs | Recall | FPR | Client p50 | Client p95 | Cost | Selection |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        winner = row["winner"].replace("_", " ") if row["winner"] else "not selected"
        lines.append(
            f"| {row['provider']} | {row['contract']} | {row['quantization']} | {row['rows']:,} | {_pct(row['valid_output_rate'], 3)} | {_pct(row['recall'])} | {_pct(row['fpr'])} | {row['latency_seconds']['p50']:.3f}s | {row['latency_seconds']['p95']:.3f}s | ${float(row['cost_usd']):.5f} | {winner} |"
        )
    return lines


def _provider_load_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Provider | Contract | Concurrency | Length mix | Requests | Terminal failures | Requests/s | Input tokens/s | Client p50 | Client p95 | Client p99 | Cost |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        mix = ", ".join(
            f"{name}: {count}" for name, count in row["length_bands"].items()
        )
        lines.append(
            f"| {row['provider']} | {row['transport'].replace('_', ' ')} | {row['concurrency']} | {mix} | {row['requests']} | {row['terminal_failures']} ({_pct(row['terminal_failure_rate'], 3)}) | {row['requests_per_second']:.3f} | {row['input_tokens_per_second']:.1f} | {row['latency_seconds']['p50']:.3f}s | {row['latency_seconds']['p95']:.3f}s | {row['latency_seconds']['p99']:.3f}s | ${float(row['cost_usd']):.5f} |"
        )
    return lines


def _loginject_remote_markdown(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Profile | Pairs | Attack recall | Paired clean false restrictions | Attack restricted and clean clear | Both restricted | Both clear | Provider calls | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("balanced", "incumbent"):
        row = summary.get("profiles", {}).get(name)
        if not isinstance(row, dict):
            continue
        paired = row["paired_outcomes"]
        lines.append(
            f"| {name.title()} | {row['pairs']:,} | {_pct(row['attack_recall']['recall'])} ({row['attack_recall']['detected']}/{row['attack_recall']['total']}) | {_pct(row['paired_clean_false_restrictions']['rate'])} ({row['paired_clean_false_restrictions']['count']}/{row['pairs']}) | {paired['attack_restricted_clean_clear']} | {paired['both_restricted']} | {paired['both_clear']} | {row['calls']} | {row['failures']} |"
        )
    return lines


def _azure_load_markdown(cells: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Route fixture | Channel | Input tokens min/mean/max | Total input tokens | Input bytes | Concurrency | Successes | Errors | Requests/s | Input tokens/s | p50 | p95 | p99 | Reviewer calls | Observed routes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell in cells:
        routes = ", ".join(f"{name}: {count}" for name, count in cell["routes"].items())
        errors = cell["requests"] - cell["successes"]
        input_tokens = cell["input_tokens"]
        if isinstance(input_tokens, dict):
            token_range = (
                f"{input_tokens['minimum']:,}/{input_tokens['mean']:.1f}/"
                f"{input_tokens['maximum']:,}"
            )
            total_tokens = f"{input_tokens['total']:,}"
        else:
            token_range = f"{input_tokens:,}"
            total_tokens = f"{input_tokens * cell['requests']:,}"
        lines.append(
            f"| {cell['kind']} | {cell['input_channel']} | {token_range} | {total_tokens} | {cell['input_bytes']:,} | {cell['concurrency']} | {cell['successes']}/{cell['requests']} | {errors} | {cell['requests_per_second']:.3f} | {cell['input_tokens_per_second']:.1f} | {cell['latency_seconds']['p50']:.3f}s | {cell['latency_seconds']['p95']:.3f}s | {cell['latency_seconds']['p99']:.3f}s | {cell['deepseek_calls']} | {routes} |"
        )
    return lines


def _azure_resource_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Azure metric | Unit | Samples | Maximum reported maximum | Maximum reported average | Maximum reported total |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['unit'] or 'unavailable'} | {row['samples']} | {_number(row['maximum'])} | {_number(row['maximum_average'])} | {_number(row['maximum_total'])} |"
        )
    return lines


def _standalone_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| System | Calibration FPR target | Evaluation recall | Evaluation FPR | Precision | AUROC | Average precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {_pct(row['target_fpr'], 1)} | {_pct(row['recall'])} | {_pct(row['fpr'])} | {_pct(row['precision'])} | {_number(row['auroc'])} | {_number(row['average_precision'])} |"
        )
    return lines


def _paired_delta_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Calibration FPR target | Recall delta | FPR delta | Precision delta | Restriction-rate delta |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = row["metrics"]
        lines.append(
            f"| {_pct(row['target_fpr'], 1)} | {_delta_pp(values['recall']['delta'], values['recall']['delta_95'])} | {_delta_pp(values['fpr']['delta'], values['fpr']['delta_95'])} | {_delta_pp(values['precision']['delta'], values['precision']['delta_95'])} | {_delta_pp(values['restriction_rate']['delta'], values['restriction_rate']['delta_95'])} |"
        )
    return lines


def _openvino_quality_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| FPR target | CUDA threshold | OpenVINO threshold | CUDA recall | OpenVINO recall | Recall delta | CUDA FPR | OpenVINO FPR | FPR delta | Precision delta | Restriction delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cuda = row["cuda"]
        openvino = row["openvino"]
        delta = row["delta_openvino_minus_cuda"]
        lines.append(
            f"| {_pct(row['target_fpr'], 1)} | `{cuda['threshold']:.9g}` | `{openvino['threshold']:.9g}` | {_pct(cuda['evaluation']['recall'])} | {_pct(openvino['evaluation']['recall'])} | {_delta_pp(delta['recall'])} | {_pct(cuda['evaluation']['fpr'])} | {_pct(openvino['evaluation']['fpr'])} | {_delta_pp(delta['fpr'])} | {_delta_pp(delta['precision'])} | {_delta_pp(delta['restriction_rate'])} |"
        )
    return lines


def _traffic_mix_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Direct / untrusted mix | Attack prevalence | Mixed recall | Mixed FPR | Expected advisory precision | Expected advisory review rate | True signals per 10,000 | False signals per 10,000 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for projection in row["prevalence_projections"].values():
            lines.append(
                f"| {row['traffic_mix']} | {_pct(projection['attack_prevalence'], 2)} | {_pct(row['mixed_recall'])} | {_pct(row['mixed_fpr'])} | {_pct(projection['expected_precision'])} | {_pct(projection['expected_review_rate'])} | {projection['true_signals_per_10k']:.2f} | {projection['false_signals_per_10k']:.2f} |"
            )
    return lines


def _slice_markdown(slices: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Slice | Rows | Positives | Negatives | Recall | FPR | Advisory restriction rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in slices.items():
        lines.append(
            f"| {name} | {row['rows']:,} | {row['positives']:,} | {row['negatives']:,} | {_pct(row['recall'])} | {_pct(row['fpr'])} | {_pct(row['restriction_rate'])} |"
        )
    return lines


def _provider_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Provider | Contract | Quantization | Valid | Client p50 | Client p95 | Failure codes | Canary cost |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        failures = ", ".join(
            f"{key}: {value}" for key, value in row["failures"].items()
        )
        lines.append(
            f"| {row['provider']} | `{row['transport']}` | {row['quantization']} | {row['valid_outputs']}/{row['rows']} | {row['latency_seconds_p50']:.2f}s | {row['latency_seconds_p95']:.2f}s | {failures or 'none'} | ${row['cost_usd']:.6f} |"
        )
    return lines


def _runtime_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| System | Evidence | Artifacts | Throughput | Input-token throughput | Peak reserved VRAM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] == "measured_local_cuda":
            lines.append(
                f"| {row['system']} | measured local CUDA | {row['artifacts']:,} | {row['artifacts_per_second']:.2f} artifacts/s | {row['input_tokens_per_second']:.0f} tokens/s | {row['peak_reserved_bytes'] / 2**30:.2f} GiB |"
            )
        else:
            lines.append(
                f"| {row['system']} | retained serial client ledger | {row['artifacts']:,} | {row['serial_requests_per_second']:.3f} serial requests/s | {row['serial_input_tokens_per_second']:.1f} tokens/s | unavailable |"
            )
    return lines


def _mutation_markdown(summary: dict[str, Any]) -> list[str]:
    caught = summary["current_high_gate_caught_set"]
    lines = [
        "| Mutations per caught base attack | Local-high ASR | Effective local-high recall | Local-pass ASR floor |",
        "|---:|---:|---:|---:|",
    ]
    for count in ("1", "2", "4", "8", "16", "25"):
        lines.append(
            f"| {count} | {_pct(caught['high_gate_asr_at_k'][count], 4)} | {_pct(caught['effective_local_high_recall_at_k'][count], 4)} | {_pct(caught['local_pass_asr_floor_at_k'][count], 4)} |"
        )
    return lines


def _loginject_markdown(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Profile | Clean local-high rate | Clean review-or-high rate | Attack local-high recall | Attack review-or-high recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("conservative", "balanced", "high_recall"):
        row = summary.get("profiles", {}).get(name)
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {name.replace('_', ' ').title()} | {_pct(row['clean_local_high_rate'], 4)} | {_pct(row['clean_review_or_high_rate'], 4)} | {_pct(row['attack_local_high_recall'], 4)} | {_pct(row['attack_review_or_high_recall'], 4)} |"
        )
    return lines


def _loginject_posthoc_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Post-hoc target FPR | Exact threshold | Attack recall | Observed clean FPR | Attacks at or above threshold | Clean pairs at or above threshold |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {_pct(row['target_fpr'], 2)} | `{row['threshold']:.9f}` | {_pct(row['recall'], 4)} | {_pct(row['observed_fpr'], 4)} | {row['true_positive']}/{row['positives']} | {row['false_positive']}/{row['negatives']} |"
        )
    return lines


def _gpt_oss_native_markdown(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Reasoning | Recall | FPR | Valid outputs | Client p50 | Client p95 | Cost | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for reasoning in ("low", "medium"):
        row = summary["summary"][reasoning]
        quality = row["quality"]
        failures = ", ".join(
            f"{name}: {count}" for name, count in row["failure_codes"].items()
        )
        lines.append(
            f"| {reasoning.title()} | {_pct(quality['recall'], 3)} | {_pct(quality['fpr'], 3)} | {_pct(row['valid_output_rate'], 3)} | {row['latency_seconds']['p50']:.3f}s | {row['latency_seconds']['p95']:.3f}s | ${float(row['cost_usd']):.5f} | {failures or 'none'} |"
        )
    return lines


def _retained_markdown(rows: list[dict[str, Any]]) -> list[str]:
    guards = [row for row in rows if "canonical_tpr_at_1pct" in row]
    lines = [
        "| System | Canonical TPR at 1% FPR | PromptShield | SEP | Evidence |",
        "|---|---:|---:|---:|---|",
    ]
    for row in guards:
        lines.append(
            f"| {row['system']} | {_pct(row['canonical_tpr_at_1pct'], 1)} | {_pct(row['promptshield_tpr_at_1pct'], 1)} | {_pct(row['sep_tpr_at_1pct'], 1)} | {row['status'].replace('_', ' ')} |"
        )
    return lines


def render_report(tables: dict[str, Any]) -> str:
    promotion = tables.get("maintained_promotion")
    profiles = tables["profiles"]
    exact_logprob_profiles = tables["exact_logprob_profiles"]
    exact_logprob_balanced = next(
        (
            row
            for row in exact_logprob_profiles
            if row["profile"] == "balanced" and row["status"] != "infeasible"
        ),
        None,
    )
    exact_logprob_conservative = next(
        (
            row
            for row in exact_logprob_profiles
            if row["profile"] == "conservative" and row["status"] != "infeasible"
        ),
        None,
    )
    exact_logprob_high = next(
        (
            row
            for row in exact_logprob_profiles
            if row["profile"] == "high_recall" and row["status"] != "infeasible"
        ),
        None,
    )
    balanced = next((row for row in profiles if row["profile"] == "balanced"), None)
    high = next((row for row in profiles if row["profile"] == "high_recall"), None)
    conservative = next(
        (row for row in profiles if row["profile"] == "conservative"), None
    )
    pending_labels = {
        "exact_logprob_cascade": "exact logprob maintained-cascade evaluation",
        "exact_hard_verdict_cascade": "exact hard-verdict maintained-cascade evaluation",
        "cascade_flow_comparison": "full-context-first versus on-demand cascade comparison",
        "reviewer_prompt_experiment": "long-context reviewer prompt diagnostic",
        "reviewer_long_bucket_experiment": "complete long-bucket reviewer comparison",
        "reviewer_prompt_patch_experiment": "minimal prompt-patch and channel-split diagnostic",
        "reviewer_channel_split_screen": "scenario-balanced channel-split screen",
        "exact_deepseek_standalone": "matched DeepSeek standalone evaluation",
        "provider_panel": "provider quality panel",
        "provider_load": "corrected provider load study",
        "openvino_full_quality": "full OpenVINO quality analysis",
        "azure_load": "warm Azure load test",
        "mutation_1024": "current-1024 mutation replay",
        "loginject_local_routing": "sealed LogInject local-routing evaluation",
        "loginject_remote_cascade": "sealed LogInject remote-cascade evaluation",
    }
    pending = [
        label
        for key, label in pending_labels.items()
        if tables["evidence_status"][key]["status"].startswith("pending")
    ]
    lines = [
        "# Morgott 1,024-context pipeline benchmark",
        "",
        "Date: 2026-08-16",
        "",
        "## Decision",
        "",
        "This benchmark does not justify production blocking or a production false-positive-rate claim.",
    ]
    if promotion is not None:
        runtime = promotion["evaluation"]["openvino_cpu_transported"]
        lines.extend(
            [
                "On 2026-08-17, the owner promoted the benchmark-selected balanced profile as Morgott's maintained advisory default without changing its prompt, provider request, window flow, or `decision: allow` authority.",
                f"Transporting those frozen thresholds unchanged to the full OpenVINO CPU ledger reached {_pct(runtime['recall'])} recall, {_pct(runtime['fpr'])} FPR, and {_pct(runtime['precision'])} precision on {runtime['artifacts']:,} provider-safe evaluation artifacts.",
            ]
        )
        deployment = tables.get("azure_deployment")
        if deployment is not None:
            lines.append(
                f"Azure revision `{deployment['revision']}` serves that profile at 100% traffic with the matching policy, threshold, model, and ONNX identities; its retained 30-request check is deployment smoke evidence only."
            )
    if exact_logprob_balanced is not None:
        lines.append(
            f"The exact balanced maintained cascade reached {_pct(exact_logprob_balanced['recall'])} recall with a Wilson 95% interval of {_interval_pct(exact_logprob_balanced['recall_95'])}, {_pct(exact_logprob_balanced['fpr'])} FPR with an interval of {_interval_pct(exact_logprob_balanced['fpr_95'])}, and {_pct(exact_logprob_balanced['precision'])} precision with an interval of {_interval_pct(exact_logprob_balanced['precision_95'])} on {exact_logprob_balanced['rows']:,} provider-safe evaluation artifacts, with {_pct(exact_logprob_balanced['provider_call_rate'])} receiving at least one provider review."
        )
        lines.append(
            f"Those artifacts consumed {exact_logprob_balanced['artifact_review_units']:,} full-artifact and {exact_logprob_balanced['window_review_units']:,} middle-window review units under full-context-first, ordered batches-of-four semantics."
        )
        flow = tables.get("cascade_flow_comparison")
        if flow is not None:
            balanced_flow = flow["profiles"]["balanced"]
            incumbent_units = balanced_flow["maintained_full_context_first"][
                "provider_review_units"
            ]
            candidate_units = balanced_flow["on_demand_same_thresholds"][
                "provider_review_units"
            ]
            lines.append(
                f"A post-hoc on-demand-window ablation preserved balanced recall and FPR but increased provider requests from {incumbent_units:,} to {candidate_units:,}; the panel contained no all-low long untrusted artifact."
            )
    elif balanced is not None:
        lines.append(
            f"The balanced artifact-level approximation reached {_pct(balanced['recall'])} recall with a Wilson 95% interval of {_interval_pct(balanced['recall_95'])}, {_pct(balanced['fpr'])} FPR with an interval of {_interval_pct(balanced['fpr_95'])}, and {_pct(balanced['precision'])} precision with an interval of {_interval_pct(balanced['precision_95'])} on the consumed 14,000-row evaluation role, with {_pct(balanced['provider_call_rate'])} of artifacts invoking DeepSeek."
        )
    long_prompt = tables.get("reviewer_long_bucket_experiment")
    if long_prompt is not None:
        current_long = long_prompt["arms"]["current_full_disabled"][
            "balanced_artifact_approximation"
        ]
        revised_long = long_prompt["arms"]["revised_full_disabled"][
            "balanced_artifact_approximation"
        ]
        lines.append(
            f"On the complete 193-row long-character diagnostic, the revised prompt moved matched balanced FPR from {_pct(current_long['fpr'])} to {_pct(revised_long['fpr'])}, while recall moved from {_pct(current_long['recall'])} to {_pct(revised_long['recall'])}; this redacted, post-hoc result is not selection-eligible."
        )
    prompt_patch = tables.get("reviewer_prompt_patch_experiment")
    if prompt_patch is not None:
        hybrid = prompt_patch["channel_split_candidate"]["quality"]
        lines.append(
            f"On that isolated consumed slice, the smallest non-dominated follow-up kept the current prompt for direct-user traffic and the revised prompt for untrusted content, reaching {_pct(hybrid['recall'])} recall and {_pct(hybrid['fpr'])} observed FPR before the broader screen below rejected it."
        )
    reviewer_screen = tables.get("reviewer_channel_split_screen")
    if reviewer_screen is not None:
        current_screen = reviewer_screen["current"]["aggregate"]
        candidate_screen = reviewer_screen["candidate"]["aggregate"]
        lines.append(
            f"The broader 256-row scenario-balanced screen rejected that channel split: recall fell from {_pct(current_screen['recall'])} to {_pct(candidate_screen['recall'])}, while FPR improved by only {_delta_pp(candidate_screen['fpr'] - current_screen['fpr'])}."
        )
    if exact_logprob_conservative is not None and exact_logprob_high is not None:
        lines.append(
            f"The exact operating range spans {_pct(exact_logprob_conservative['recall'])} recall at {_pct(exact_logprob_conservative['fpr'])} FPR for the conservative profile through {_pct(exact_logprob_high['recall'])} recall at {_pct(exact_logprob_high['fpr'])} FPR for the high-recall profile."
        )
    elif conservative is not None and high is not None:
        lines.append(
            f"The approximate operating range spans {_pct(conservative['recall'])} recall at {_pct(conservative['fpr'])} FPR for the conservative profile through {_pct(high['recall'])} recall at {_pct(high['fpr'])} FPR for the high-recall profile."
        )
    if exact_logprob_balanced is not None:
        lines.append(
            (
                "The exact balanced profile is now the maintained advisory default, but representative shadow traffic and fresh long-benign evidence remain required before blocking, SLA, or production-quality claims."
                if promotion is not None
                else "The exact balanced profile is the benchmark recommendation among measured provider-safe arms, but representative shadow traffic and fresh long-benign evidence are still required before any serving change."
            )
        )
    elif balanced is not None:
        lines.append(
            "The balanced profile is a provisional report-first candidate, but the provider-safe exact cascade replay, representative shadow traffic, and fresh long-benign evidence are required before any serving change."
        )
    else:
        lines.append(
            "No operating profile is recommended because the local quality summary is pending."
        )
    lines.extend(
        [
            "All learned results remain advisory and every maintained assessment still returns `decision: allow`.",
            "",
            "## Evidence status",
            "",
            "The calibration and evaluation roles are consumed development evidence, not prospective production traffic.",
            "",
        ]
    )
    for name, value in tables["evidence_status"].items():
        lines.append(
            f"- `{name}` is **{value['status'].replace('_', ' ')}**: {value['note']}"
        )
    lines.extend(["", "## Exact maintained-cascade profiles", ""])
    if exact_logprob_profiles:
        lines.append(
            f"The strict-logprob arm reports {exact_logprob_profiles[0]['rows']:,} provider-safe evaluation artifacts with exact maintained multi-window routing."
        )
        lines.append("")
        lines.extend(_exact_profile_markdown(exact_logprob_profiles))
    else:
        lines.append(
            "The strict-logprob calibration selection may be frozen, but exact evaluation remains pending until its evaluation artifact and required window ledger are complete."
        )
    exact_hard_profiles = tables["exact_hard_verdict_profiles"]
    hard_selection = tables.get("hard_verdict_selection") or {}
    hard_ineligible = hard_selection.get("selection_status") == "no_eligible_provider"
    if exact_hard_profiles and hard_ineligible:
        lines.extend(
            [
                "",
                "### True no-logprob hard-verdict cascade diagnostic",
                "",
                "No strict no-logprob provider satisfies the complete provider-selection rule once source is included as a declared slice.",
                "The table below preserves the earlier Decart exact evaluation as a pre-source-gate diagnostic, not as a selected provider result.",
                "",
            ]
        )
        lines.extend(_exact_profile_markdown(exact_hard_profiles))
        lines.append("")
        eligibility = hard_selection.get("strict_provider_eligibility", {})
        decart_loss = (
            eligibility.get("decart", {})
            .get("slice_recall_deltas", {})
            .get("source=hackaprompt")
        )
        if decart_loss is not None:
            lines.append(
                f"Decart matched the best aggregate recall but lost {_pct(decart_loss)} recall on the HackAPrompt source slice, above the allowed 2 percentage-point loss."
            )
        lines.append(
            "Baidu and DeepInfra also failed one or more overall or declared-source slice gates, so the study has no strict hard-verdict winner."
        )
    elif exact_hard_profiles:
        lines.extend(["", "### True no-logprob hard-verdict cascade", ""])
        lines.extend(_exact_profile_markdown(exact_hard_profiles))
        lines.append("")
        lines.append(
            "An infeasible conservative row is intentionally null and is not replaced by a constraint-violating hard-verdict point."
        )
    else:
        lines.append(
            "The true no-logprob exact cascade evaluation remains pending and no selection-only metric is promoted to evaluation evidence."
        )
    flow_comparison = tables["cascade_flow_comparison"]
    if flow_comparison is not None:
        lines.extend(
            [
                "",
                "## Full-context-first versus on-demand review",
                "",
                "This post-hoc ablation keeps the local gates, Cloudflare logprob contract, ordered windows, batch size four, and fail-closed behavior unchanged.",
                "The candidate removes only the unconditional full-context DeepSeek review for multi-window untrusted inputs and reviews their middle-zone windows instead.",
                "",
            ]
        )
        lines.extend(_cascade_flow_markdown(flow_comparison))
        balanced_flow = flow_comparison["profiles"]["balanced"]
        incumbent_flow = balanced_flow["maintained_full_context_first"]
        candidate_flow = balanced_flow["on_demand_same_thresholds"]
        coverage = balanced_flow["evaluation_coverage"]
        artifact_delta = (
            candidate_flow["artifact_review_units"]
            - incumbent_flow["artifact_review_units"]
        )
        window_delta = (
            candidate_flow["window_review_units"]
            - incumbent_flow["window_review_units"]
        )
        total_delta = (
            candidate_flow["provider_review_units"]
            - incumbent_flow["provider_review_units"]
        )
        delta = balanced_flow["paired_delta_same_thresholds"]["metrics"]
        lines.extend(
            [
                "",
                f"At the balanced thresholds, recall and FPR were unchanged: recall delta {_delta_pp(delta['recall']['delta'], delta['recall']['delta_95'])} and FPR delta {_delta_pp(delta['fpr']['delta'], delta['fpr']['delta_95'])}.",
                f"The on-demand flow changed artifact review units by {artifact_delta:+,}, window review units by {window_delta:+,}, and total provider requests by {total_delta:+,}; it did not reduce the {candidate_flow['artifacts_with_provider_review']:,} reviewed artifacts.",
                "Independent calibration selected the same thresholds as the maintained flow for all three profiles.",
                f"Coverage is narrow: the evaluation contained {coverage['multi_window_artifacts']} multi-window artifacts, only {coverage['multi_window_untrusted_artifacts']} were untrusted, and only {coverage['untrusted_without_local_high']} avoided local-high routing under the balanced profile.",
                f"No evaluated untrusted multi-window artifact had every window below the low threshold ({coverage['untrusted_all_windows_below_low']} rows), so this comparison does not validate the all-low long-input case that motivated the question.",
                "The result is exact for this provider-safe consumed panel, but it is a post-hoc ablation and not a production routing recommendation.",
            ]
        )
    lines.extend(["", "## Secondary artifact-level approximate operating profiles", ""])
    lines.extend(_profile_markdown(profiles))
    lines.extend(
        [
            "",
            "These thresholds were selected only on the frozen 6,000-row calibration role and transported unchanged to the 14,000-row evaluation role.",
            "The profile metrics reuse one retained full-artifact DeepSeek probability per artifact and are exact for single-window rows, but they do not replay the maintained full-context-first and middle-window review sequence for multi-window rows.",
            "These full-14,000-row approximations remain secondary even after an exact provider-safe replay because they do not implement the maintained multi-window review sequence.",
            "The approximate profiles are Pareto choices with different error costs, not three production service levels.",
        ]
    )
    if balanced is not None:
        lines.extend(
            [
                "",
                "### Approximate balanced profile by evaluation dataset",
                "",
            ]
        )
        lines.extend(_slice_markdown(balanced["by_slice"]["dataset"]))
        lines.extend(
            [
                "",
                "### Approximate balanced profile by input channel",
                "",
            ]
        )
        lines.extend(_slice_markdown(balanced["by_slice"]["input_channel"]))
    lines.extend(
        [
            "",
            "## Exact matched DeepSeek standalone contracts",
            "",
        ]
    )
    exact_standalone_rows = tables["exact_deepseek_standalone_rows"]
    if exact_standalone_rows:
        lines.append(
            f"All rows below use the same {exact_standalone_rows[0]['rows']:,} provider-safe evaluation artifacts, and invalid outputs fail closed."
        )
        lines.append("")
        lines.extend(_exact_standalone_markdown(exact_standalone_rows))
        lines.extend(
            [
                "",
                "The Cloudflare same-request hard verdict removes thresholding while retaining the logprob request, whereas the Decart strict hard-verdict arm removes the logprob request fields themselves.",
                "These transports are reported separately and are not reconstructed from probabilities at 0.5.",
            ]
        )
    else:
        lines.append(
            "The calibration selections exist only when frozen, but matched provider-safe evaluation remains pending and no standalone contract is marked complete."
        )
    lines.extend(
        [
            "",
            "## Standalone detector comparison on the frozen calibration and evaluation panel",
            "",
        ]
    )
    lines.extend(_standalone_markdown(tables["standalone_fixed_fpr"]))
    paired = tables["paired_detector_deltas"]
    if paired:
        lines.extend(
            [
                "",
                "### Paired Morgott 1,024 versus segmented Prompt Guard 2 deltas",
                "",
                "Each system selected its threshold independently on the consumed 6,000-row calibration role at the same target FPR, and both thresholds were transported to the same 14,000 evaluation identities.",
                "Deltas are Morgott minus Prompt Guard percentage points with paired stratified bootstrap 95% intervals from 2,000 resamples and deterministic seed 42.",
                "",
            ]
        )
        lines.extend(_paired_delta_markdown(paired))
    lines.extend(
        [
            "",
            "The current Morgott score aggregates every ordered 1,024-token window with 128-token overlap at the artifact level.",
            "Prompt Guard 2 aggregates 512-token segments with 64-token overlap, so this comparison does not silently truncate long artifacts.",
            "",
            "## Retained external and historical baselines",
            "",
        ]
    )
    lines.extend(_retained_markdown(tables["retained_baselines"]))
    lines.extend(
        [
            "",
            "The retained guard table comes from versioned report-level evidence because several original row ledgers are not present locally.",
            "Those rows cannot support new paired confidence intervals or post-hoc threshold selection.",
            "The retained GPT-OSS Safeguard 20B shared-binary-prompt result was 12.75% valid-output recall, 0.06% FPR, 99.36% precision, 1.64-second mean latency, 10.58-second p95 latency, and $0.0687 per 1,000 inputs.",
            "That retained GPT-OSS result is a shared-binary-prompt comparison and is separate from the supplementary native screen below.",
            "The retained DeepSeek V4 Flash 0731 standalone logprob evidence reached 0.953 AUROC and 0.938 average precision on the consumed evaluation role.",
        ]
    )
    gpt_oss_native = tables["gpt_oss_native_screen"]
    if gpt_oss_native is not None:
        lines.extend(
            [
                "",
                "### Supplementary GPT-OSS Safeguard 20B native screen",
                "",
                "This 256-row screen passed its canary and used the official 456-word policy with Harmony-style chat rather than the retained shared-binary prompt.",
                "Invalid and terminally failed outputs are conservatively positive in the quality metrics.",
                "",
            ]
        )
        lines.extend(_gpt_oss_native_markdown(gpt_oss_native))
        lines.extend(
            [
                "",
                "Neither low nor medium reasoning is reliable enough to expand to the complete panel or displace the main benchmark arms.",
            ]
        )
    lines.extend(
        [
            "",
            "## Real-world defense layers and excluded comparators",
            "",
            "An artifact-level guard is one advisory layer, while realistic deployment also requires trusted provenance, least-privilege tool schemas, deterministic reference-monitor authorization, output validation, and audit telemetry.",
            "[Meta's official Prompt Guard 2 model card](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) recommends segmenting longer prompts and scanning segments in parallel; this benchmark adds a declared 64-token overlap and maximum artifact aggregation to that 512-token model contract.",
            "[Meta's official LlamaFirewall repository](https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall) describes a defense-in-depth framework over agent interactions, so LlamaFirewall is treated as a trace-level architecture and not inserted into a static text ROC table.",
            "[Azure's official Prompt Shields quickstart](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-jailbreak) requires an Azure AI Content Safety resource, and none is available in the benchmark subscription, so Azure Prompt Shields has no measured comparison row.",
            "[OpenAI's official GPT-OSS Safeguard guide](https://developers.openai.com/cookbook/articles/gpt-oss-safeguard-guide) defines the native evaluation around a developer-provided policy and Harmony formatting, so the retained shared-binary-prompt result remains distinct from the measured supplementary native-policy screen.",
            "[OpenRouter's official provider-routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection) and [router-metadata documentation](https://openrouter.ai/docs/guides/features/router-metadata) support explicit provider selection and route auditing, so this benchmark pins one provider, disables fallbacks, requires advertised parameters, and validates returned identity against [frozen live 0731 endpoint metadata](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints).",
            "OpenRouter endpoint capabilities and commercial metadata can change, so the endpoint record is a per-run snapshot rather than a permanent provider claim.",
            "[WASP's official repository](https://github.com/facebookresearch/wasp) defines an end-to-end browser-agent benchmark, so WASP remains sealed for a separately frozen outcome study and is not flattened into this static detector table.",
            "",
            "## Long-context result",
            "",
        ]
    )
    long_context = tables["long_context"]
    if long_context is None:
        lines.append("Long-context scoring is pending.")
    else:
        population = long_context["population"]
        length = long_context["balanced_by_length"]
        long = length.get("long")
        lines.append(
            f"The 20,000-row panel contained {population['multi_window_rows']} multi-window artifacts and a maximum of {population['maximum_windows']} windows."
        )
        if isinstance(long, dict):
            lines.append(
                f"In the balanced artifact-level approximation, the long-character bucket reached {_pct(long['recall'])} recall but {_pct(long['fpr'])} FPR over only {long['negatives']} negatives and {long['positives']} positives."
            )
            lines.append(
                "That complete-bucket diagnostic uses the preliminary artifact-level profile, not the later exact promoted profile; it remains a warning because the promoted prompt and full-context-first architecture are unchanged."
            )
            decomposition = long_context["balanced_long_negative_decomposition"]
            if decomposition is not None:
                lines.append(
                    f"Among the {decomposition['negatives']} evaluation negatives of at least 4,096 characters, local-high caused {decomposition['local_high_restrictions']} restrictions, {decomposition['middle_zone']} entered the middle zone, and retained first full-context review caused all {decomposition['full_context_reviewer_false_restrictions']} false restrictions for {_pct(decomposition['fpr'])} FPR."
                )
                lines.append(
                    "This decomposition is exact through the first full-context review, so maximum-window local aggregation is not the direct cause of those balanced-gate false restrictions."
                )
                lines.append(
                    "Exploratory second-window and top-two aggregation were non-dominant and are not recommended."
                )
            lines.append(
                "The long-bucket false-positive result is the strongest warning in this study and makes representative long-benign shadow traffic a prerequisite for deployment work."
            )
    reviewer_long = tables["reviewer_long_bucket_experiment"]
    if reviewer_long is not None:
        lines.extend(
            [
                "",
                "### Complete long-bucket prompt comparison",
                "",
                "A matched follow-up tested the current and revised full-context prompts on all 193 identities behind the long-character headline: 54 benign and 139 attacked artifacts.",
                "Both arms used the pinned Cloudflare strict hard-verdict contract with reasoning disabled, identical redacted inputs, conservative failure handling, and the original balanced local gates.",
                "The revised prompt used channel-aware trusted-task interpretation for direct requests and untrusted webpages, messages, emails, and documents.",
                "",
                "| Arm | Reviewer-only recall | Reviewer-only FPR | Balanced artifact-level recall | Balanced artifact-level FPR | Valid outputs | Client p50 / p95 | Recorded cost |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        labels = {
            "current_full_disabled": "Current prompt",
            "revised_full_disabled": "Revised channel/task-conditioned prompt",
        }
        for arm, label in labels.items():
            result = reviewer_long["arms"][arm]
            reviewer = result["reviewer_only"]
            cascade = result["balanced_artifact_approximation"]
            lines.append(
                f"| {label} | {_pct(reviewer['recall'])} | {_pct(reviewer['fpr'])} | {_pct(cascade['recall'])} | {_pct(cascade['fpr'])} | {result['valid_outputs']} / {reviewer['rows']} | {result['latency_seconds']['p50']:.2f}s / {result['latency_seconds']['p95']:.2f}s | ${float(result['recorded_cost_usd']):.3f} |"
            )
        revised = reviewer_long["arms"]["revised_full_disabled"][
            "balanced_artifact_approximation"
        ]
        lines.extend(
            [
                "",
                "The revised arm removed all 37 matched current-arm false restrictions, moving from 37 of 54 to 0 of 54, while attack detections fell from 123 of 139 to 101 of 139.",
                "The revised balanced recall was 84.62% on untrusted content but only 57.38% on direct-user inputs; the largest measured source weakness was PromptShield at 25 of 49 attacks detected, or 51.02% recall.",
                f"Zero observed false positives does not establish zero population FPR: the revised arm's Wilson 95% FPR interval is {_interval_pct(revised['fpr_95'])}.",
                "The matched current arm differs slightly from the original 85.61% recall / 70.37% FPR headline because this follow-up uses strict hard verdicts on deterministically redacted inputs, while the original approximation used retained logprob thresholding on the original inputs.",
                "The complete comparison is still post-hoc and selection-ineligible because all 193 identities were already consumed, 109 inputs required safety redaction, and the trusted tasks were reconstructed rather than supplied by the original runtime.",
            ]
        )
    if prompt_patch is not None:
        patched = prompt_patch["quality"]
        hybrid = prompt_patch["channel_split_candidate"]
        hybrid_quality = hybrid["quality"]
        lines.extend(
            [
                "",
                "#### Minimal-change follow-up",
                "",
                f"One additional sentence covering implicit, role-based, encoded, and obfuscated redirection failed its frozen advance rule: recall remained {_pct(patched['recall'])} while FPR rose from 0 of 54 to {patched['fp']} of 54, or {_pct(patched['fpr'])}.",
                "That sentence is rejected.",
                "The smallest non-dominated candidate requires no new prompt text: use the current prompt for trusted `direct_user` traffic and the revised task-conditioned prompt for `untrusted_content`.",
                f"This channel split reached {hybrid_quality['tp']} of {hybrid_quality['positives']} attacks, or {_pct(hybrid_quality['recall'])} recall, with {hybrid_quality['fp']} of {hybrid_quality['negatives']} observed false restrictions.",
                f"It recovers {hybrid['delta_vs_revised']['attack_detections']} detections over the single revised prompt without adding a false restriction; direct-user recall is {_pct(hybrid['slices']['input_channel']['direct_user']['recall'])} and untrusted-content recall is {_pct(hybrid['slices']['input_channel']['untrusted_content']['recall'])}.",
                "This was the preferred fresh-confirmation contract before the broader scenario-balanced screen below.",
            ]
        )
    reviewer_screen = tables["reviewer_channel_split_screen"]
    if reviewer_screen is not None:
        current = reviewer_screen["current"]["aggregate"]
        candidate = reviewer_screen["candidate"]["aggregate"]
        delta = reviewer_screen["delta_candidate_minus_current"]
        current_sep = reviewer_screen["current"]["slices"]["source"]["sep"]
        candidate_sep = reviewer_screen["candidate"]["slices"]["source"]["sep"]
        lines.extend(
            [
                "",
                "#### Scenario-balanced untrusted-content screen",
                "",
                "A fast matched screen then sampled 256 consumed provider-safe untrusted-content artifacts no longer than 4,096 characters: 128 SEP rows and 128 proportionally stratified rows from six other sources.",
                "It compared the existing Cloudflare strict-logprob prompt against the revised prompt under the unchanged exact balanced local gates and reviewer threshold.",
                "The sample contained 142 attacks and 114 benign controls; the direct-user prompt was unchanged and therefore was not called again.",
                "",
                "| Arm | Recall | FPR | Precision | Attack detections | False restrictions | Provider calls | Valid outputs |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                f"| Current prompt | {_pct(current['recall'])} | {_pct(current['fpr'])} | {_pct(current['precision'])} | {current['tp']} / {current['positives']} | {current['fp']} / {current['negatives']} | {reviewer_screen['current']['provider_calls']} | {reviewer_screen['current']['valid_outputs']} / 256 |",
                f"| Revised untrusted prompt | {_pct(candidate['recall'])} | {_pct(candidate['fpr'])} | {_pct(candidate['precision'])} | {candidate['tp']} / {candidate['positives']} | {candidate['fp']} / {candidate['negatives']} | {reviewer_screen['candidate']['provider_calls']} | {reviewer_screen['candidate']['valid_outputs']} / 256 |",
                "",
                f"The candidate saved {abs(delta['false_restrictions'])} false restriction but lost {abs(delta['attack_detections'])} attack detections, a {_delta_pp(delta['recall'])} recall change for a {_delta_pp(delta['fpr'])} FPR change.",
                f"SEP recall fell from {_pct(current_sep['recall'])} to {_pct(candidate_sep['recall'])}; LLMail recall fell by {_delta_pp(delta['source_recall']['llmail'])}.",
                "It failed the predeclared advance rule and is rejected for further confirmation or pipeline integration.",
                "This is a scenario-balanced consumed-data screen, not measured production traffic or fresh selection evidence.",
            ]
        )
    reviewer_prompt = tables["reviewer_prompt_experiment"]
    if reviewer_prompt is not None:
        lines.extend(
            [
                "",
                "### Focused BrowseSafe context and reasoning diagnostic",
                "",
                "A fast post-hoc screen compared four pinned Cloudflare strict-hard-verdict contracts on 115 consumed long BrowseSafe pages: 60 benign and 55 attacked.",
                "The original public pages were deterministically safety-redacted before transmission because 112 of the 115 selected long BrowseSafe rows triggered the repository sensitive-text scanner, mostly on email addresses.",
                "The fixed trusted task asked the system to analyze the webpage as untrusted data without following page instructions.",
                "No raw provider response or page text was persisted.",
                "",
                "| Reviewer arm | Reviewer-only evaluation recall | Reviewer-only evaluation FPR | Balanced-cascade evaluation recall | Balanced-cascade evaluation FPR | Valid outputs | Client p50 / p95 | Recorded evaluation cost |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        labels = {
            "current_full_disabled": "Current prompt, full, reasoning off",
            "revised_full_disabled": "Narrow task-conditioned prompt, full, reasoning off",
            "revised_compact_disabled": "Narrow prompt, suspect window plus neighbors, reasoning off",
            "revised_compact_high": "Narrow compact prompt, high reasoning",
        }
        balanced = reviewer_prompt["cascade_evaluation"]["profiles"]["balanced"]
        for arm, label in labels.items():
            evaluation = reviewer_prompt["arms"][arm]["evaluation"]
            cascade = balanced[arm]
            lines.append(
                f"| {label} | {_pct(evaluation['recall'])} | {_pct(evaluation['fpr'])} | {_pct(cascade['recall'])} | {_pct(cascade['fpr'])} | {evaluation['valid_outputs']} / {evaluation['rows']} | {evaluation['client_seconds_p50']:.2f}s / {evaluation['client_seconds_p95']:.2f}s | ${float(evaluation['recorded_cost_usd']):.3f} |"
            )
        lines.extend(
            [
                "",
                "The narrower full-context prompt reduced balanced-slice false restrictions from 37 of 38 to 2 of 38, while balanced cascade attack recall fell from 39 of 39 to 28 of 39.",
                "Terminal failures were conservatively counted as restrictions.",
                "Compact evidence reduced false restrictions to 1 of 38 but also reduced balanced cascade attack recall to 26 of 39.",
                "High reasoning did not dominate reasoning-disabled compact review: it retained the same one false restriction, caught 25 rather than 26 attacks after the local-high short circuit, roughly doubled median latency, and used about nineteen times as many completion tokens on the evaluation role.",
                "The result supports prompt narrowing as the main fix direction, not chain-of-thought alone, but none of these contracts is eligible to replace the frozen cascade because the panel is already consumed, safety redaction changed its inputs, and 38 benign examples cannot certify a low FPR.",
            ]
        )
    lines.extend(["", "## Incumbent-gate 1,024-context mutation replay", ""])
    mutation = tables["mutation_1024"]
    if mutation is None:
        lines.append("The current-model mutation replay is pending.")
    else:
        base = mutation["base"]
        population = mutation["population"]
        lines.append(
            f"The local-high gate caught {base['high_gate_caught']} of {population['eligible_base_rows']} eligible base attacks, for {_pct(base['high_gate_recall'], 4)} base recall."
        )
        thresholds = mutation["thresholds"]
        lines.append(
            f"This local-only replay used local-high `{thresholds['local_high']:g}`, direct low `{thresholds['low_by_channel']['direct_user']:g}`, and untrusted low `{thresholds['low_by_channel']['untrusted_content']:g}`."
        )
        lines.append("")
        lines.extend(_mutation_markdown(mutation))
        lines.extend(
            [
                "",
                "The ASR denominator is the set of base attacks caught by the pre-promotion incumbent local-high gate, and the mutation rows are already-open synthetic development evidence.",
                "The zero local-pass ASR floor means mutations moved caught attacks into review rather than local pass; it does not imply a zero full-cascade ASR.",
                "Full-cascade mutation ASR remains unmeasured until review-zone provider outcomes are frozen.",
            ]
        )
    lines.extend(["", "## Sealed LogInject local routing", ""])
    loginject = tables["loginject"]
    if loginject is None:
        lines.append("The one-time sealed LogInject local score is pending.")
    else:
        movement = loginject["paired_score_movement"]
        context = tables["loginject_panel_context"]
        balanced_loginject = loginject["profiles"]["balanced"]
        local_high_attacks = round(
            loginject["pairs"] * balanced_loginject["attack_local_high_recall"]
        )
        if context is not None:
            vectors = ", ".join(
                name.replace("_", " ") for name in context["population"]["by_vector"]
            )
            references = context["selection"]["fit_reference_rows"]
            lines.extend(
                [
                    "[The official public synthetic LogInject-1.0 artifact](https://doi.org/10.5281/zenodo.20436935) supplies 2,569 adversarial log entries that this panel converts into 2,569 matched clean and attacked 50-entry log pairs spanning 3,615 to 4,566 tokens.",
                    f"The attacked batches inject ordered complete entries or fragments through {vectors} vectors.",
                    "LogInject was absent from fitting, architecture selection, prompt selection, and threshold selection, so this is a source-held-out out-of-distribution generated-log diagnostic rather than natural-document or production-log evidence.",
                    f"Pair-level overlap filtering checked {references['canonical_train']:,} canonical training rows, {references['matched_pairs']:,} matched-pair references, and {references['promptshield_train']:,} PromptShield training references, and no pair was removed.",
                ]
            )
        lines.append(
            f"The sealed local replay measured {loginject['pairs']} clean and attacked pairs exactly once."
        )
        lines.append(
            "This local-only table uses the preliminary profile grid; its row named Balanced is not the later exact promoted balanced profile, whose complete remote result appears below."
        )
        lines.append("")
        lines.extend(_loginject_markdown(loginject))
        lines.extend(
            [
                "",
                f"At the preliminary balanced gates, only {local_high_attacks} of {loginject['pairs']} attacks reached local-high, or {_pct(balanced_loginject['attack_local_high_recall'], 4)}, while every attacked pair reached review-or-high and zero attacks passed locally.",
                "Review-or-high is routing coverage rather than completed detection, and every clean pair also entered review, so remote reviewer outcomes are required to measure cascade recall and false restriction.",
                f"Attack insertion changed the local score by a mean of {movement['mean']:.4f} and a median of {movement['p50']:.4f}, with {_pct(movement['positive_rate'], 4)} of pairs moving upward.",
            ]
        )
        diagnostic = tables["loginject_posthoc_diagnostic"]
        if diagnostic:
            lines.extend(
                [
                    "",
                    "### Sealed post-hoc local threshold diagnostic",
                    "",
                    "These thresholds and operating points were computed after the sealed panel was opened and are diagnostic only.",
                    "They cannot tune, select, or revise any threshold in this run.",
                    "",
                ]
            )
            lines.extend(_loginject_posthoc_markdown(diagnostic))
            lines.extend(
                [
                    "",
                    LOGINJECT_FUTURE_PROTOCOL,
                ]
            )
    remote_loginject = tables["loginject_remote"]
    lines.extend(["", "## Sealed LogInject exact remote cascade", ""])
    if remote_loginject is None:
        lines.append(
            "The sealed remote replay is pending, so local review-or-high routing is not presented as attack detection or clean false restriction."
        )
    else:
        lines.append(
            f"The write-once Cloudflare strict-logprob replay resolved {remote_loginject['pairs']:,} matched clean and attacked pairs with {remote_loginject['unique_provider_calls']:,} unique provider calls and {remote_loginject['terminal_failures']} terminal failures."
        )
        lines.append(LOGINJECT_RELIABILITY_NOTE)
        lines.append(
            f"All {remote_loginject['terminal_failures']} terminal failures fail closed as restricted under the frozen reviewer failure rule."
        )
        lines.append("")
        lines.extend(_loginject_remote_markdown(remote_loginject))
        lines.append("")
        lines.append(
            "Paired clean false restrictions and attack recall are measured end-to-end under the frozen exact cascade, while local-only post-hoc thresholds remain ineligible for tuning."
        )
    lines.extend(["", "## Provider compatibility canary", ""])
    if tables["provider_canaries"]:
        lines.extend(_provider_markdown(tables["provider_canaries"]))
        lines.extend(
            [
                "",
                "The canary used 16 unique rows per provider and tests exact transport compatibility only.",
                "It is too small for provider quality selection, sustained throughput, or tail-latency claims.",
            ]
        )
    else:
        lines.append("Provider canaries are pending.")
    provider_panel_rows = tables["provider_panel_rows"]
    if not provider_panel_rows:
        lines.append(
            "The matched 1,024-row provider panel is pending, so no provider or quantization winner is reported."
        )
    else:
        lines.extend(["", "### Matched 1,024-row provider panel", ""])
        lines.extend(_provider_panel_markdown(provider_panel_rows))
        winners = tables["provider_panel"]["winners"]
        lines.extend(
            [
                "",
                f"The frozen strict-logprob winner is {winners['logprob']['provider']}.",
            ]
        )
        if winners.get("hard_verdict") is None:
            lines.append(
                "No strict hard-verdict provider survived the complete overall and declared-slice quality gate; Decart remains a measured diagnostic only."
            )
        else:
            lines.append(
                f"The frozen strict hard-verdict winner is {winners['hard_verdict']['provider']}."
            )
        lines.extend(
            [
                "Quantization labels come from the endpoint snapshot frozen at run start, and `unknown` means the provider did not declare a precision label in that snapshot.",
                "Provider quality is measured on matched calibration-safe artifacts and does not replace the exact evaluation-stage cascade results.",
            ]
        )
    provider_load_rows = tables["provider_load_rows"]
    lines.extend(["", "### Corrected unique-sample provider load", ""])
    if provider_load_rows:
        lines.extend(_provider_load_markdown(provider_load_rows))
        lines.append("")
        lines.append(
            "Every load cell reports its own input-length mix, uses unique provider-safe samples, and excludes archived length-confounded load artifacts."
        )
        if hard_ineligible:
            lines.append(
                "The Decart load cells are retained as transport-performance diagnostics and do not make Decart an eligible hard-verdict winner."
            )
        lines.append(
            "Input tokens per second and requests per second are the primary throughput measures for these one-token verdict contracts."
        )
    else:
        lines.append(
            "The corrected unique-sample load artifact is pending, so canary or provider-panel latency is not presented as sustained throughput."
        )
    lines.extend(["", "## Runtime", ""])
    lines.extend(_runtime_markdown(tables["runtime"]))
    lines.extend(
        [
            "",
            "Local CUDA throughput excludes model load time and cannot be substituted for Azure end-to-end latency.",
            "The retained DeepSeek throughput is a sum-of-client-latencies serial equivalent, not a concurrent load measurement.",
        ]
    )
    parity = tables["openvino_parity"]
    if parity is not None:
        disagreement = parity["decision_disagreement_rate"]
        worst_threshold, worst_rate = max(
            disagreement.items(), key=lambda item: item[1]
        )
        lines.extend(
            [
                "",
                "### CUDA versus OpenVINO parity",
                "",
                f"The 512-row OpenVINO BF16 audit measured {parity['artifacts_per_second']:.2f} artifacts/s and {parity['input_tokens_per_second']:.0f} input tokens/s.",
                f"Its maximum absolute score delta was {parity['absolute_score_delta']['maximum']:.4f}, and the worst threshold-decision disagreement was {_pct(worst_rate)} at threshold `{worst_threshold}`.",
                "This parity sample measures numerical and threshold agreement only and is not a full OpenVINO quality evaluation.",
            ]
        )
        if worst_rate > 0.005:
            lines.append(
                "The disagreement exceeds the 0.5% parity gate, so CUDA and OpenVINO quality results must remain runtime-specific."
            )
    openvino_full = tables["openvino_full_replay"]
    if openvino_full is not None:
        runtime = openvino_full["runtime"]
        quality = tables["openvino_fixed_fpr_quality"]
        lines.extend(
            [
                "",
                "### Full runtime-specific CUDA and OpenVINO comparison",
                "",
                f"The full OpenVINO replay scored {runtime['artifacts']:,} artifacts in {runtime['score_seconds']:.2f} seconds at {runtime['artifacts_per_second']:.3f} artifacts/s and {runtime['input_tokens_per_second']:.0f} input tokens/s.",
            ]
        )
        if quality:
            first = quality[0]
            cuda_evaluation = first["cuda"]["evaluation"]
            openvino_evaluation = first["openvino"]["evaluation"]
            delta = first["delta_openvino_minus_cuda"]
            lines.extend(
                [
                    "CUDA and OpenVINO each select a separate numerical threshold on the same frozen 6,000-row calibration role and transport only that runtime's threshold unchanged to the aligned consumed 14,000-row evaluation role.",
                    "No CUDA threshold is applied to OpenVINO, and no OpenVINO threshold is presented as a CUDA threshold.",
                    "",
                ]
            )
            lines.extend(_openvino_quality_markdown(quality))
            lines.extend(
                [
                    "",
                    f"Evaluation AUROC was {cuda_evaluation['auroc']:.6f} on CUDA and {openvino_evaluation['auroc']:.6f} on OpenVINO, for an OpenVINO-minus-CUDA delta of {delta['auroc']:+.6f}.",
                    f"Evaluation average precision was {cuda_evaluation['average_precision']:.6f} on CUDA and {openvino_evaluation['average_precision']:.6f} on OpenVINO, for an OpenVINO-minus-CUDA delta of {delta['average_precision']:+.6f}.",
                    "These are runtime-specific consumed-development results rather than evidence that either runtime threshold transfers to the other.",
                ]
            )
        else:
            lines.append(
                "The full score ledger is measured, but runtime-specific fixed-FPR quality analysis is pending."
            )
    azure = tables["azure_load"]
    lines.extend(["", "## Warm Azure end-to-end load", ""])
    deployment = tables.get("azure_deployment")
    if deployment is not None:
        lines.append(
            f"The currently recorded live revision is `{deployment['revision']}` with profile `{deployment['pipeline_profile']}`, policy `{deployment['policy_sha256']}`, and threshold `{deployment['threshold_sha256']}`."
        )
        lines.append(
            "Its 30-request deployment smoke check passed, but it is not used as latency or throughput evidence for the matrix below."
        )
    if azure is None:
        lines.append(
            "The identity-verified warm Azure load artifact is pending, so no deployed route, latency, throughput, error, or resource claim is made."
        )
    else:
        status = azure["status"]
        cells = azure["cells"]
        requests = sum(cell["requests"] for cell in cells)
        successes = sum(cell["successes"] for cell in cells)
        deepseek_calls = sum(cell["deepseek_calls"] for cell in cells)
        failures = requests - successes
        if azure.get("policy_status") == "incumbent_pre_promotion":
            lines.append(
                "This load run predates the promoted profile identity fields and remains incumbent-only evidence; it does not measure the balanced-20260816 deployment."
            )
        lines.append(
            f"The deployment reported model `{status['model_key']}`, context length {status['context_length']}, overlap {status['window_overlap']}, and ONNX identity `{status['onnx_sha256']}` before measurement."
        )
        lines.append(
            "The table reports all warm route, input-length, and concurrency cells separately rather than pooling unlike paths or lengths."
        )
        lines.append(
            f"Across {requests:,} measured requests, {successes:,} returned HTTP 200, {failures:,} had transport failures, the cell aggregates recorded zero explicit non-`allow` decisions, and the deployment reported {deepseek_calls:,} DeepSeek calls."
        )
        lines.append(
            "Decision-presence counts were not persisted, so the aggregates do not prove that every HTTP 200 response contained `decision: allow`."
        )
        lines.append(
            "The client timer starts before semaphore acquisition, so p50, p95, and p99 are queueing-inclusive burst latencies rather than pure in-service latencies."
        )
        estimated_cost = azure.get(
            "estimated_remote_cost_usd", azure.get("maximum_remote_cost_usd")
        )
        failed_estimate = azure.get(
            "prior_failed_azure_estimate_usd",
            azure.get("prior_failed_azure_ceiling_usd"),
        )
        if estimated_cost is not None and failed_estimate is not None:
            lines.append(
                f"The artifact's planning estimates are ${float(estimated_cost):.6f} for this run and ${float(failed_estimate):.6f} for the prior failed finalization attempt."
            )
            lines.append(
                "They are not strict upper bounds or invoiced spend because the calculation omits the reviewer system prompt and chat/schema overhead and uses Morgott rather than provider-native token counts; the hard $25 cap therefore is not independently proven by this artifact."
            )
        budget = tables.get("budget")
        if budget and budget.get("closed") is True:
            known = budget.get("legacy_accounting", {}).get("booked_known_usd")
            lines.append(
                f"Because this completed study predates durable preauthorization, its evidence directory is now closed at the full $24 usable budget; the known recorded-plus-Azure planning amount is ${float(known):.6f}, while unpriced terminal calls remain the reason no unused balance is claimed."
            )
        lines.append("")
        lines.extend(_azure_load_markdown(cells))
        cells_by_id = {cell["cell_id"]: cell for cell in cells}
        saturation_ids = (
            "allow:61440:c16",
            "high:61440:c16",
            "review:61440:c16",
        )
        if all(cell_id in cells_by_id for cell_id in saturation_ids):
            allow_saturation = cells_by_id[saturation_ids[0]]
            high_saturation = cells_by_id[saturation_ids[1]]
            review_saturation = cells_by_id[saturation_ids[2]]
            lines.append("")
            lines.append(
                f"At 60 KiB and concurrency 16, the allow fixture returned {allow_saturation['routes'].get('pass', 0)} pass and {allow_saturation['routes'].get('restrict', 0)} restrict routes across {allow_saturation['successes']} HTTP 200 responses while reporting only {allow_saturation['deepseek_calls']} DeepSeek calls."
            )
            lines.append(
                f"The corresponding high fixture returned {high_saturation['successes']} HTTP 200 responses and {high_saturation['requests'] - high_saturation['successes']} transport failures, while the review fixture returned {review_saturation['successes']} HTTP 200 responses and {review_saturation['requests'] - review_saturation['successes']} transport failure."
            )
            lines.append(
                "The fixtures include a cell-specific nonce, so payload bytes differ across concurrency cells and the 10/90 boundary tracks nonce-length changes."
            )
            lines.append(
                "This cell is confounded and cannot support a scaling, saturation, or detector-quality conclusion."
            )
        allow_calls = sum(
            cell["deepseek_calls"] for cell in cells if cell["kind"] == "allow"
        )
        lines.append("")
        lines.append(
            f"The allow fixtures made {allow_calls:,} DeepSeek calls, so this run did not measure a true local-allow path."
        )
        lines.append(
            "The nominal 4,096-byte and 16,384-byte fixtures were only about 605-657 and 2,387-2,589 Morgott tokens, respectively, so they do not satisfy the planned approximately 1,024-token and 4,096-token cells."
        )
        resources = tables["azure_resource_metrics"]
        if resources:
            lines.extend(["", "### Azure resource observations", ""])
            lines.extend(_azure_resource_markdown(resources))
            lines.append("")
            lines.append(
                "Azure `Total` values accumulate samples within each time bucket and are not concurrency counts."
            )
            lines.append(
                "The Replicas series reported maximum and average values of one; its total of two is not evidence that two replicas ran concurrently."
            )
    lines.extend(
        [
            "",
            "## Representative traffic projections",
            "",
            (
                "These figures are arithmetic sensitivity analyses from the exact provider-safe balanced cascade rates, not observations from production traffic."
                if tables["projection_basis"] == "exact_provider_safe_balanced"
                else "These figures are arithmetic sensitivity analyses from the approximate balanced artifact rates, not observations from production traffic or exact maintained-cascade projections."
            ),
            "",
            "| Attack prevalence | Expected advisory precision | Expected advisory review rate | True signals per 10,000 | False signals per 10,000 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for value in (tables["prevalence_projections"] or {}).values():
        lines.append(
            f"| {_pct(value['attack_prevalence'], 2)} | {_pct(value['expected_precision'])} | {_pct(value['expected_review_rate'])} | {value['true_signals_per_10k']:.2f} | {value['false_signals_per_10k']:.2f} |"
        )
    prevalence = tables["prevalence_projections"] or {}
    one_in_thousand = prevalence.get("0.001")
    lines.append("")
    if one_in_thousand is not None:
        lines.append(
            f"At 0.1% assumed attack prevalence, the balanced rates project only about {_pct(one_in_thousand['expected_precision'])} advisory precision because false positives dominate rare attacks."
        )
    traffic_mix = tables["traffic_mix_projections"]
    if traffic_mix:
        lines.extend(
            [
                "This is why aggregate benchmark precision cannot answer real review-volume questions without a representative traffic denominator.",
                "",
                "### Direct-versus-untrusted traffic-mix sensitivity",
                "",
                (
                    "These arithmetic projections hold attack prevalence equal within direct and untrusted channels, weight the exact provider-safe balanced channel recall and FPR by the declared traffic mix, and do not claim measured production traffic."
                    if tables["projection_basis"] == "exact_provider_safe_balanced"
                    else "These arithmetic projections hold attack prevalence equal within direct and untrusted channels, weight the approximate balanced channel recall and FPR by the declared traffic mix, and do not claim measured production traffic or exact cascade behavior."
                ),
                "",
            ]
        )
        lines.extend(_traffic_mix_markdown(traffic_mix))
    else:
        lines.append(
            "This is why aggregate benchmark precision cannot answer real review-volume questions without a representative traffic denominator."
        )
    lines.extend(
        [
            "",
            "## Pending evidence and safe next action",
            "",
            (
                f"The {', '.join(pending)} remain pending."
                if pending
                else "All frozen evidence stages completed, but the Azure local-allow and intended approximately 1,024-token and 4,096-token cells and full-cascade mutation outcomes remain unmeasured."
            ),
            "No missing result is treated as zero and no retained 512 result is relabeled as 1,024-context evidence.",
            (
                "The shortest safe next action is to complete only the pending frozen stages listed above without changing their selected thresholds, contracts, or evidence roles."
                if pending
                else (
                    "The shortest safe next action is to keep the promoted advisory profile frozen while collecting prospective task-bearing long benign and matched-attack shadow traffic."
                    if promotion is not None
                    else "The shortest safe next action is to keep the current pipeline unchanged; another reviewer candidate should not be tested until it has a narrower hypothesis that preserves SEP and short-attack recall."
                )
            ),
            "WASP remains sealed for a separate browser-agent outcome study.",
            "",
            "## Limitations",
            "",
            "- The 6,000-row calibration role and 14,000-row evaluation role are already consumed development evidence.",
            "- PromptShield and SEP are public transfer development panels, not production traffic.",
            "- No representative adjudicated production traffic was available.",
            "- The Azure run did not measure a local-allow path, did not hit the intended middle token lengths, and used different nonce payloads across concurrency cells.",
            "- Azure remote-cost values are planning estimates rather than strict upper bounds or independently verified spend.",
            "- Mutation robustness is local-routing evidence only because full-cascade mutation review was not run.",
            "- The cascade-flow ablation contained no long untrusted artifact with every window below the local low gate, so that routing case remains unmeasured.",
            "- The reviewer prompt diagnostic used consumed, deterministically redacted BrowseSafe rows and a fixed synthetic trusted task, so it cannot select a production contract or certify a low FPR.",
            "- The scenario-balanced prompt screen deliberately upweighted non-SEP sources and is consumed development evidence, not representative production traffic.",
            "- Report-level baselines with missing ledgers cannot support paired significance tests.",
            "- The provider canary is a schema and routing test rather than a quality benchmark.",
            "- The maintained runtime requests Cloudflare and disables fallbacks, but its response parser does not independently attest the returned provider build; the benchmark ledgers performed that identity validation.",
            "- Learned scores remain advisory and never grant tool, data, network, credential, or financial authority.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def generate(input_dir: Path, report_path: Path, tables_path: Path) -> dict[str, Any]:
    tables = build_tables(input_dir)
    report = render_report(tables)
    if "\N{EM DASH}" in report:
        raise ValueError("report contains a prohibited em dash")
    _atomic_json(tables_path, tables)
    _atomic_text(report_path, report)
    return {
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "tables": str(tables_path),
        "tables_sha256": _sha256(tables_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.input.resolve(), args.report.resolve(), args.tables.resolve()
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
